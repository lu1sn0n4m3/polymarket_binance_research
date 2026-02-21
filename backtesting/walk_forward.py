"""Walk-forward backtest: recalibrate model at each fold, trade OOS only.

Expanding window: train on days [1..d], backtest on day d+1.
At each fold, the model params AND seasonal vol are re-estimated from
training data only — no look-ahead bias.

Usage:
    venv/bin/python3 backtesting/walk_forward.py
"""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

from collections import defaultdict
from datetime import date, datetime, time as dt_time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from backtesting.core import BacktestResult, MarketResult
from backtesting.simulator import _run_on_labels, _run_on_labels_with_bankroll, _build_pm_lookup, _config
from backtesting.strategies import Strategy

ET = ZoneInfo("America/New_York")

MIN_TRAIN_DAYS = 7


def walk_forward_backtest(
    asset: str,
    start_date: date,
    end_date: date,
    strategy: Strategy,
    ewma_half_life: float = 600.0,
    initial_bankroll: float = 100.0,
) -> BacktestResult:
    """Walk-forward backtest with per-fold recalibration.

    For each test day d (after MIN_TRAIN_DAYS warm-up):
      1. Build calibration dataset on days [start..d-1]
      2. Calibrate model params via QLIKE
      3. Build Pricer from those params + training seasonal vol
      4. Backtest on day d using that Pricer

    EWMA sigma_rv is computed once over the full range (it's purely
    backward-looking, no look-ahead).
    """
    from marketdata.data import load_binance, load_binance_labels
    from pricing.features.realized_vol import compute_rv_ewma
    from pricing.features.seasonal_vol import compute_seasonal_vol_split
    from pricing.models.gaussian import GaussianModel
    from pricing.calibrate import calibrate_vol
    from pricing.pricer import Pricer

    # ------------------------------------------------------------------
    # 1. Pre-load data that spans the full range (no look-ahead)
    # ------------------------------------------------------------------
    full_start = datetime.combine(start_date, dt_time(0), tzinfo=timezone.utc)
    full_end = datetime.combine(end_date + timedelta(days=1), dt_time(0), tzinfo=timezone.utc)

    print(f"Loading Binance BBO {start_date} to {end_date} ({asset}, 1s) ...")
    bbo = load_binance(start=full_start, end=full_end, asset=asset, interval="1s")
    print(f"  {len(bbo):,} BBO rows")

    if bbo.empty:
        return BacktestResult(markets=[], config={})

    print(f"Computing EWMA sigma_rv (H={ewma_half_life}s) ...")
    ts_rv, sigma_rv_full = compute_rv_ewma(bbo, half_life_sec=ewma_half_life)
    print(f"  {len(ts_rv):,} tick-time values")

    bbo_ts = bbo["ts_recv"].values
    bbo_mid = bbo["mid_px"].values

    print("Loading hourly labels ...")
    labels = load_binance_labels(start=full_start, end=full_end, asset=asset)
    print(f"  {len(labels)} market hours")

    print("Loading Polymarket data ...")
    pm_lookup = _build_pm_lookup(full_start, full_end, asset, labels)
    print(f"  {len(pm_lookup)} markets with PM data")

    # ------------------------------------------------------------------
    # 2. Load pre-built calibration dataset (reuse for training splits)
    # ------------------------------------------------------------------
    from pathlib import Path
    dataset_path = Path("pricing/output/calibration_dataset.parquet")
    if not dataset_path.exists():
        print(f"ERROR: {dataset_path} not found. Run pricing/run_calibration.py first.")
        return BacktestResult(markets=[], config={})

    print(f"\nLoading calibration dataset from {dataset_path} ...")
    cal_dataset = pd.read_parquet(dataset_path)
    cal_dataset["day"] = cal_dataset["market_id"].str[4:12]
    print(f"  {len(cal_dataset):,} rows, {cal_dataset['market_id'].nunique()} markets")
    days = sorted(cal_dataset["day"].unique())
    n_days = len(days)

    # Map labels to days for easy slicing
    labels["day"] = labels.apply(
        lambda r: datetime.fromtimestamp(
            int(r["hour_start_ms"]) / 1000, tz=timezone.utc
        ).strftime("%Y%m%d"),
        axis=1,
    )

    print(f"\n{n_days} days, {n_days - MIN_TRAIN_DAYS} test folds (min train = {MIN_TRAIN_DAYS}d)")
    print(f"{'='*70}")

    # ------------------------------------------------------------------
    # 3. Walk-forward loop
    # ------------------------------------------------------------------
    all_results: list[MarketResult] = []
    param_history = defaultdict(list)
    current_bankroll = initial_bankroll
    bankroll_history = [current_bankroll]

    for fold_idx in range(MIN_TRAIN_DAYS, n_days):
        train_days = days[:fold_idx]
        test_day = days[fold_idx]

        train_data = cal_dataset[cal_dataset["day"].isin(train_days)]
        test_labels = labels[labels["day"] == test_day]

        if train_data.empty or test_labels.empty:
            continue

        # --- Calibrate on training data ---
        model = GaussianModel()
        result = calibrate_vol(model, train_data, objective="qlike",
                               verbose=False, output_dir=None)
        vp = result.params
        for k, v in vp.items():
            param_history[k].append(v)

        # --- Build Pricer from training-period seasonal vol ---
        # Extract seasonal vol from training BBO
        train_end_ms = int(test_labels.iloc[0]["hour_start_ms"])
        train_bbo_mask = bbo_ts < train_end_ms
        train_bbo = pd.DataFrame({
            "ts_recv": bbo_ts[train_bbo_mask],
            "mid_px": bbo_mid[train_bbo_mask],
        })
        seasonal = compute_seasonal_vol_split(train_bbo, bucket_minutes=5)

        pricer = Pricer(
            vol_params=vp,
            sigma_tod_weekday=seasonal.weekday.sigma_tod,
            sigma_tod_weekend=seasonal.weekend.sigma_tod,
            bucket_minutes=5,
        )

        # --- Backtest on test day with bankroll ---
        fold_results, fold_bankroll = _run_on_labels_with_bankroll(
            asset, test_labels, strategy, pricer,
            bbo_ts, bbo_mid, ts_rv, sigma_rv_full, pm_lookup,
            bankroll=current_bankroll,
        )

        traded = sum(1 for r in fold_results if r.trades)
        pnl = sum(r.pnl for r in fold_results)
        n_markets = len(fold_results)
        all_results.extend(fold_results)

        # Update bankroll from fold results
        current_bankroll += pnl
        current_bankroll = max(current_bankroll, 0.0)
        bankroll_history.append(current_bankroll)

        print(f"  Fold {fold_idx - MIN_TRAIN_DAYS:2d}  test={test_day}  "
              f"train={len(train_days):2d}d  "
              f"markets={n_markets:2d}  traded={traded:2d}  "
              f"PnL={pnl:+.3f}  bankroll=${current_bankroll:.2f}  "
              f"c={vp['c']:.3f}  α={vp['alpha']:.3f}")

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"WALK-FORWARD SUMMARY ({n_days - MIN_TRAIN_DAYS} folds)")
    print(f"{'='*70}")

    if param_history:
        print(f"\n  Parameter stability:")
        for k in ["c", "alpha", "k0", "k1"]:
            if k in param_history:
                vals = np.array(param_history[k])
                print(f"    {k:6s}: mean={np.mean(vals):+.4f}  std={np.std(vals):.4f}  "
                      f"CV={np.std(vals)/abs(np.mean(vals))*100:.1f}%")

    return BacktestResult(
        markets=all_results,
        config={
            **_config(asset, start_date, end_date, strategy, ewma_half_life),
            "mode": "walk_forward",
            "min_train_days": MIN_TRAIN_DAYS,
            "n_folds": n_days - MIN_TRAIN_DAYS,
            "initial_bankroll": initial_bankroll,
        },
        bankroll_curve=np.array(bankroll_history),
    )


if __name__ == "__main__":
    from backtesting.strategies.threshold import EdgeThresholdStrategy
    from backtesting.analytics import print_summary, plot_backtest

    for min_edge in [0.03, 0.05, 0.08]:
        strategy = EdgeThresholdStrategy(
            min_edge=min_edge, min_tau=60.0, sizing="fixed",
        )
        result = walk_forward_backtest(
            "BTC", date(2026, 1, 19), date(2026, 2, 18), strategy,
        )
        print_summary(result)
        print()

    # Detailed plot for edge=0.05
    strategy = EdgeThresholdStrategy(min_edge=0.05, min_tau=60.0, sizing="fixed")
    result = walk_forward_backtest(
        "BTC", date(2026, 1, 19), date(2026, 2, 18), strategy,
    )
    plot_backtest(result, save_path="backtesting/output/walk_forward.png")
