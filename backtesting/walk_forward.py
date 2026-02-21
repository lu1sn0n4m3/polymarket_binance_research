"""Walk-forward backtest: recalibrate model at each fold, trade OOS only.

Expanding window: train on days [1..d], backtest on day d+1.
At each fold, the model params AND seasonal vol are re-estimated from
training data only — no look-ahead bias.

Calibration results are cached to backtesting/output/wf_calibration_cache.json
so subsequent runs skip the expensive recalibration step.

Usage:
    venv/bin/python3 backtesting/walk_forward.py
"""

from __future__ import annotations

import json
import sys
sys.path.insert(0, ".")

from collections import defaultdict
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from backtesting.core import BacktestResult, MarketResult
from backtesting.simulator import _run_on_labels_with_bankroll, _build_pm_lookup, _config
from backtesting.strategies import Strategy

ET = ZoneInfo("America/New_York")

MIN_TRAIN_DAYS = 7
CACHE_PATH = Path("backtesting/output/wf_calibration_cache.json")


def _load_calibration_cache() -> dict | None:
    """Load cached per-fold calibration results."""
    if not CACHE_PATH.exists():
        return None
    with open(CACHE_PATH) as f:
        return json.load(f)


def _save_calibration_cache(cache: dict) -> None:
    """Save per-fold calibration results to JSON."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"  Saved calibration cache to {CACHE_PATH}")


def _calibrate_all_folds(
    cal_dataset: pd.DataFrame,
    days: list[str],
    bbo_ts: np.ndarray,
    bbo_mid: np.ndarray,
    labels: pd.DataFrame,
) -> dict:
    """Run calibration for every fold and return cache dict.

    Also computes seasonal vol from training BBO and stores it.
    """
    from pricing.models.gaussian import GaussianModel
    from pricing.calibrate import calibrate_vol
    from pricing.features.seasonal_vol import compute_seasonal_vol_split

    n_days = len(days)
    folds = {}

    for fold_idx in range(MIN_TRAIN_DAYS, n_days):
        train_days = days[:fold_idx]
        test_day = days[fold_idx]

        train_data = cal_dataset[cal_dataset["day"].isin(train_days)]
        test_labels = labels[labels["day"] == test_day]

        if train_data.empty or test_labels.empty:
            continue

        model = GaussianModel()
        result = calibrate_vol(model, train_data, objective="qlike",
                               verbose=False, output_dir=None)
        vp = result.params

        # Compute seasonal vol from training BBO only
        train_end_ms = int(test_labels.iloc[0]["hour_start_ms"])
        train_bbo_mask = bbo_ts < train_end_ms
        train_bbo = pd.DataFrame({
            "ts_recv": bbo_ts[train_bbo_mask],
            "mid_px": bbo_mid[train_bbo_mask],
        })
        seasonal = compute_seasonal_vol_split(train_bbo, bucket_minutes=5)

        folds[test_day] = {
            "vol_params": vp,
            "n_train_days": len(train_days),
            "sigma_tod_weekday": seasonal.weekday.sigma_tod.tolist(),
            "sigma_tod_weekend": seasonal.weekend.sigma_tod.tolist(),
        }

        print(f"  Fold {fold_idx - MIN_TRAIN_DAYS:2d}  test={test_day}  "
              f"train={len(train_days):2d}d  "
              f"c={vp['c']:.3f}  α={vp['alpha']:.3f}")

    cache = {
        "n_days": n_days,
        "days": days,
        "min_train_days": MIN_TRAIN_DAYS,
        "folds": folds,
    }
    return cache


def walk_forward_backtest(
    asset: str,
    start_date: date,
    end_date: date,
    strategy: Strategy,
    ewma_half_life: float = 600.0,
    initial_bankroll: float = 100.0,
    recalibrate: bool = False,
    grid_ms: int = 1000,
) -> BacktestResult:
    """Walk-forward backtest with per-fold recalibration.

    Calibration results are cached. Pass recalibrate=True to force
    re-running the calibration step.
    """
    from marketdata.data import load_binance, load_binance_labels
    from pricing.features.realized_vol import compute_rv_ewma
    from pricing.pricer import Pricer

    # ------------------------------------------------------------------
    # 1. Pre-load data that spans the full range (no look-ahead)
    # ------------------------------------------------------------------
    full_start = datetime.combine(start_date, dt_time(0), tzinfo=timezone.utc)
    full_end = datetime.combine(end_date + timedelta(days=1), dt_time(0), tzinfo=timezone.utc)

    _interval_str = {100: "100ms", 500: "500ms", 1000: "1s", 5000: "5s"}.get(grid_ms, "1s")
    print(f"Loading Binance BBO {start_date} to {end_date} ({asset}, {_interval_str}) ...")
    bbo = load_binance(start=full_start, end=full_end, asset=asset, interval=_interval_str)
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
    pm_lookup = _build_pm_lookup(full_start, full_end, asset, labels, interval_ms=grid_ms)
    print(f"  {len(pm_lookup)} markets with PM data")

    # ------------------------------------------------------------------
    # 2. Load calibration dataset + get/build calibration cache
    # ------------------------------------------------------------------
    dataset_path = Path("pricing/output/calibration_dataset.parquet")
    if not dataset_path.exists():
        print(f"ERROR: {dataset_path} not found. Run pricing/run_calibration.py first.")
        return BacktestResult(markets=[], config={})

    cal_dataset = pd.read_parquet(dataset_path)
    cal_dataset["day"] = cal_dataset["market_id"].str[4:12]
    days = sorted(cal_dataset["day"].unique())
    n_days = len(days)

    # Map labels to days for easy slicing
    labels["day"] = labels.apply(
        lambda r: datetime.fromtimestamp(
            int(r["hour_start_ms"]) / 1000, tz=timezone.utc
        ).strftime("%Y%m%d"),
        axis=1,
    )

    # Try to load cached calibration
    cache = None if recalibrate else _load_calibration_cache()
    if cache is not None and cache.get("days") == days:
        print(f"\nLoaded cached calibration ({len(cache['folds'])} folds)")
    else:
        print(f"\nCalibrating {n_days - MIN_TRAIN_DAYS} folds ...")
        cache = _calibrate_all_folds(cal_dataset, days, bbo_ts, bbo_mid, labels)
        _save_calibration_cache(cache)

    folds_cache = cache["folds"]
    print(f"\n{n_days} days, {len(folds_cache)} test folds (min train = {MIN_TRAIN_DAYS}d)")
    print(f"{'='*70}")

    # ------------------------------------------------------------------
    # 3. Walk-forward loop (fast — no calibration, just backtest)
    # ------------------------------------------------------------------
    all_results: list[MarketResult] = []
    param_history = defaultdict(list)
    current_bankroll = initial_bankroll
    bankroll_history = [current_bankroll]  # per-market granularity

    for fold_idx in range(MIN_TRAIN_DAYS, n_days):
        test_day = days[fold_idx]

        if test_day not in folds_cache:
            continue

        fold_data = folds_cache[test_day]
        vp = fold_data["vol_params"]
        test_labels = labels[labels["day"] == test_day]

        if test_labels.empty:
            continue

        for k, v in vp.items():
            param_history[k].append(v)

        # Build Pricer from cached params + seasonal vol
        pricer = Pricer(
            vol_params=vp,
            sigma_tod_weekday=np.array(fold_data["sigma_tod_weekday"]),
            sigma_tod_weekend=np.array(fold_data["sigma_tod_weekend"]),
            bucket_minutes=5,
        )

        # ---- Walk-forward hour scores for ConditionalStrategy ----
        # Compute PnL-by-hour from ALL PAST results only (no lookahead)
        if hasattr(strategy, '_hour_scores'):
            hour_pnl: dict[int, float] = defaultdict(float)
            hour_count: dict[int, int] = defaultdict(int)
            for mr in all_results:  # only past folds' results
                for t in mr.trades:
                    if t.hour_et >= 0:
                        pnl_t = t.settle(mr.outcome)
                        hour_pnl[t.hour_et] += pnl_t
                        hour_count[t.hour_et] += 1
            # Score = total PnL for that hour (positive = good, negative = skip)
            hour_scores = {}
            for h in range(24):
                if hour_count[h] >= 5:  # need at least 5 trades to judge
                    hour_scores[h] = hour_pnl[h]
                else:
                    hour_scores[h] = 0.0  # unknown → neutral
            strategy._hour_scores = hour_scores

        # Backtest on test day with bankroll
        fold_results, fold_br_curve = _run_on_labels_with_bankroll(
            asset, test_labels, strategy, pricer,
            bbo_ts, bbo_mid, ts_rv, sigma_rv_full, pm_lookup,
            bankroll=current_bankroll,
            grid_ms=grid_ms,
        )

        traded = sum(1 for r in fold_results if r.trades)
        pnl = sum(r.pnl for r in fold_results)
        n_markets = len(fold_results)
        all_results.extend(fold_results)

        # Append per-market bankroll points from this fold (skip initial
        # which is same as last point of previous fold)
        if len(fold_br_curve) > 1:
            bankroll_history.extend(fold_br_curve[1:].tolist())

        current_bankroll = bankroll_history[-1]

        # Log hour filter stats for conditional strategy
        hour_info = ""
        if hasattr(strategy, '_hour_scores') and strategy._hour_scores:
            n_good = sum(1 for v in strategy._hour_scores.values() if v > 0)
            n_bad = sum(1 for v in strategy._hour_scores.values() if v < 0)
            hour_info = f"  hrs:+{n_good}/-{n_bad}"

        print(f"  Fold {fold_idx - MIN_TRAIN_DAYS:2d}  test={test_day}  "
              f"train={fold_data['n_train_days']:2d}d  "
              f"markets={n_markets:2d}  traded={traded:2d}  "
              f"PnL={pnl:+.3f}  bankroll=${current_bankroll:.2f}  "
              f"c={vp['c']:.3f}  α={vp['alpha']:.3f}{hour_info}")

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"WALK-FORWARD SUMMARY ({len(folds_cache)} folds)")
    print(f"{'='*70}")

    if param_history:
        print(f"\n  Parameter stability:")
        for k in ["c", "alpha", "k0", "k1"]:
            if k in param_history:
                vals = np.array(param_history[k])
                print(f"    {k:6s}: mean={np.mean(vals):+.4f}  std={np.std(vals):.4f}  "
                      f"CV={np.std(vals)/abs(np.mean(vals))*100:.1f}%")

    print(f"\n  Bankroll: ${initial_bankroll:.2f} → ${current_bankroll:.2f}  "
          f"({(current_bankroll/initial_bankroll - 1)*100:+.1f}%)")

    return BacktestResult(
        markets=all_results,
        config={
            **_config(asset, start_date, end_date, strategy, ewma_half_life),
            "mode": "walk_forward",
            "min_train_days": MIN_TRAIN_DAYS,
            "n_folds": len(folds_cache),
            "initial_bankroll": initial_bankroll,
        },
        bankroll_curve=np.array(bankroll_history),
    )


if __name__ == "__main__":
    from backtesting.strategies.threshold import EdgeThresholdStrategy
    from backtesting.analytics import print_summary, plot_backtest

    # Kelly sizing, starting with $100
    strategy = EdgeThresholdStrategy(
        min_edge=0.03, min_tau=60.0,
        sizing="kelly", kelly_fraction=0.5, max_kelly_fraction=0.25,
    )
    result = walk_forward_backtest(
        "BTC", date(2026, 1, 19), date(2026, 2, 18), strategy,
        initial_bankroll=100.0,
    )
    print_summary(result)
    plot_backtest(result, save_path="backtesting/output/walk_forward_kelly.png")
