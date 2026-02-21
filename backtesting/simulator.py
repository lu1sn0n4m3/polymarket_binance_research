"""Backtest simulator for market-taker strategies."""

from __future__ import annotations

from datetime import date, datetime, time as dt_time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from backtesting.core import BacktestResult, MarketResult
from backtesting.strategies import Strategy

ET = ZoneInfo("America/New_York")


def run_backtest(
    asset: str,
    start_date: date,
    end_date: date,
    strategy: Strategy,
    pricer,
    ewma_half_life: float = 600.0,
    initial_bankroll: float = 100.0,
    verbose: bool = True,
    grid_ms: int = 1000,
) -> BacktestResult:
    """Run a backtest over a date range with bankroll tracking.

    Args:
        asset: "BTC" or "ETH"
        start_date: First date (inclusive).
        end_date: Last date (inclusive).
        strategy: Strategy instance.
        pricer: Pricer instance (from pricing.pricer).
        ewma_half_life: EWMA half-life in seconds for sigma_rv.
        initial_bankroll: Starting bankroll in dollars.
        verbose: Print progress.

    Returns:
        BacktestResult with per-market results and bankroll curve.
    """
    from marketdata.data import load_binance, load_binance_labels
    from marketdata.data.resampled_polymarket import load_resampled_polymarket
    from pricing.features.realized_vol import compute_rv_ewma

    start_dt = datetime.combine(start_date, dt_time(0), tzinfo=timezone.utc)
    end_dt = datetime.combine(end_date + timedelta(days=1), dt_time(0), tzinfo=timezone.utc)

    # Map grid_ms to interval string for data loading
    _interval_str = {100: "100ms", 500: "500ms", 1000: "1s", 5000: "5s"}.get(grid_ms, "1s")
    if verbose:
        print(f"Loading Binance BBO {start_date} to {end_date} ({asset}, {_interval_str}) ...")
    bbo = load_binance(start=start_dt, end=end_dt, asset=asset, interval=_interval_str)
    if verbose:
        print(f"  {len(bbo):,} BBO rows")

    if bbo.empty:
        if verbose:
            print("No BBO data.")
        return BacktestResult(markets=[], config=_config(asset, start_date, end_date, strategy))

    if verbose:
        print(f"Computing EWMA sigma_rv (H={ewma_half_life}s) ...")
    ts_rv, sigma_rv_full = compute_rv_ewma(bbo, half_life_sec=ewma_half_life)
    if verbose:
        print(f"  {len(ts_rv):,} tick-time values")

    bbo_ts = bbo["ts_recv"].values
    bbo_mid = bbo["mid_px"].values
    del bbo

    if len(ts_rv) == 0:
        if verbose:
            print("No sigma_rv computed.")
        return BacktestResult(markets=[], config=_config(asset, start_date, end_date, strategy))

    if verbose:
        print("Loading hourly labels ...")
    labels = load_binance_labels(start=start_dt, end=end_dt, asset=asset)
    if verbose:
        print(f"  {len(labels)} market hours")

    if labels.empty:
        if verbose:
            print("No labels.")
        return BacktestResult(markets=[], config=_config(asset, start_date, end_date, strategy))

    if verbose:
        print("Loading Polymarket data ...")
    pm_lookup = _build_pm_lookup(start_dt, end_dt, asset, labels, interval_ms=grid_ms)
    if verbose:
        print(f"  {len(pm_lookup)} markets with PM data")

    results, bankroll_curve = _run_on_labels_with_bankroll(
        asset, labels, strategy, pricer, bbo_ts, bbo_mid,
        ts_rv, sigma_rv_full, pm_lookup, initial_bankroll,
        grid_ms=grid_ms,
    )

    traded = sum(1 for r in results if r.trades)
    total_trades = sum(len(r.trades) for r in results)
    if verbose:
        print(f"\n  {len(results)} markets processed")
        print(f"  {traded} markets traded, {total_trades} total trades")
        print(f"  Final bankroll: ${bankroll_curve[-1]:.2f}" if len(bankroll_curve) > 0 else "")

    return BacktestResult(
        markets=results,
        config={
            **_config(asset, start_date, end_date, strategy, ewma_half_life),
            "initial_bankroll": initial_bankroll,
        },
        bankroll_curve=bankroll_curve,
    )


def _build_pm_lookup(
    start_dt, end_dt, asset, labels, interval_ms: int = 1000,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Pre-load and normalize Polymarket data for all market hours."""
    from marketdata.data.resampled_polymarket import load_resampled_polymarket

    pm_lookup: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    try:
        pm_all = load_resampled_polymarket(
            start_dt=start_dt, end_dt=end_dt, interval_ms=interval_ms, asset=asset,
        )
    except Exception:
        return pm_lookup

    if pm_all.empty:
        return pm_lookup

    pm_all_ts = pm_all["ts_recv"].values
    pm_all_bid = pm_all["bid"].values
    pm_all_ask = pm_all["ask"].values
    pm_all_mid = pm_all["mid"].values

    for _, lbl in labels.iterrows():
        hs = int(lbl["hour_start_ms"])
        he = int(lbl["hour_end_ms"])
        Y_lbl = int(lbl["Y"])

        hour_mask = (pm_all_ts >= hs) & (pm_all_ts < he)
        if not hour_mask.any():
            continue

        h_bid = pm_all_bid[hour_mask]
        h_ask = pm_all_ask[hour_mask]
        h_mid = pm_all_mid[hour_mask]

        terminal_mid = (h_bid[-1] + h_ask[-1]) / 2
        token_is_up = (terminal_mid > 0.5) if Y_lbl == 1 else (terminal_mid <= 0.5)

        if token_is_up:
            pm_lookup[hs] = (pm_all_ts[hour_mask], h_bid, h_ask, h_mid)
        else:
            pm_lookup[hs] = (
                pm_all_ts[hour_mask],
                (1.0 - h_ask).astype(np.float64),
                (1.0 - h_bid).astype(np.float64),
                (1.0 - h_mid).astype(np.float64),
            )

    return pm_lookup


def _compute_mfe_mae_tpsl(
    trade_ts: int,
    trade_side: str,
    trade_price: float,
    trade_size: float,
    grid_ts: np.ndarray,
    pm_bid_grid: np.ndarray,
    pm_ask_grid: np.ndarray,
    tp_frac: float = 0.0,
    sl_frac: float = 0.0,
) -> tuple[float, float, float | None, str]:
    """Compute MFE/MAE and check TP/SL for a trade.

    Returns (mfe, mae, exit_pnl, exit_reason).
    exit_pnl is None if held to expiry, otherwise the realized PnL from early exit.
    exit_reason is "tp", "sl", or "".
    """
    post_mask = grid_ts > trade_ts
    if not post_mask.any():
        return 0.0, 0.0, None, ""

    if trade_side == "BUY":
        unrealized = trade_size * (pm_bid_grid[post_mask] - trade_price)
    else:
        unrealized = trade_size * (trade_price - pm_ask_grid[post_mask])

    # Filter NaN
    valid = ~np.isnan(unrealized)
    unrealized_clean = unrealized[valid]
    if len(unrealized_clean) == 0:
        return 0.0, 0.0, None, ""

    mfe = float(max(np.max(unrealized_clean), 0.0))
    mae = float(min(np.min(unrealized_clean), 0.0))

    # Check TP/SL (scan chronologically, first trigger wins)
    # Use ACTUAL unrealized at trigger tick, not the threshold.
    # This is realistic: you exit at whatever the book shows when
    # the threshold is crossed. On 1s data this is very close to
    # the threshold, but for SL it can be worse (gap through stop).
    exit_pnl = None
    exit_reason = ""
    tp_threshold = trade_size * tp_frac if tp_frac > 0 else float("inf")
    sl_threshold = -trade_size * sl_frac if sl_frac > 0 else float("-inf")

    if tp_frac > 0 or sl_frac > 0:
        for u in unrealized:
            if np.isnan(u):
                continue
            if u >= tp_threshold:
                exit_pnl = float(u)  # actual fill, not threshold
                exit_reason = "tp"
                break
            if u <= sl_threshold:
                exit_pnl = float(u)  # actual fill — may be worse than stop
                exit_reason = "sl"
                break

    return mfe, mae, exit_pnl, exit_reason


def _run_on_labels(
    asset: str,
    labels: pd.DataFrame,
    strategy: Strategy,
    pricer,
    bbo_ts: np.ndarray,
    bbo_mid: np.ndarray,
    ts_rv: np.ndarray,
    sigma_rv_full: np.ndarray,
    pm_lookup: dict,
    grid_ms: int = 1000,
) -> list[MarketResult]:
    """Run strategy on a set of market-hour labels. Core inner loop."""
    results, _ = _run_on_labels_with_bankroll(
        asset, labels, strategy, pricer, bbo_ts, bbo_mid,
        ts_rv, sigma_rv_full, pm_lookup, bankroll=None,
        grid_ms=grid_ms,
    )
    return results


def _run_on_labels_with_bankroll(
    asset: str,
    labels: pd.DataFrame,
    strategy: Strategy,
    pricer,
    bbo_ts: np.ndarray,
    bbo_mid: np.ndarray,
    ts_rv: np.ndarray,
    sigma_rv_full: np.ndarray,
    pm_lookup: dict,
    bankroll: float | None = None,
    grid_ms: int = 1000,
) -> tuple[list[MarketResult], np.ndarray]:
    """Run strategy on labels with optional bankroll tracking.

    If bankroll is None, no bankroll tracking (backward compat).
    Returns (results, bankroll_curve).
    """
    results: list[MarketResult] = []
    current_bankroll = bankroll if bankroll is not None else 0.0
    bankroll_history = [current_bankroll]

    for _, lbl in labels.iterrows():
        hour_start_ms = int(lbl["hour_start_ms"])
        hour_end_ms = int(lbl["hour_end_ms"])
        K = float(lbl["K"])
        Y = int(lbl["Y"])

        dt_start = datetime.fromtimestamp(hour_start_ms / 1000, tz=timezone.utc)
        dt_et = dt_start.astimezone(ET)
        market_id = f"{asset}_{dt_start.strftime('%Y%m%d')}_{dt_et.hour:02d}"

        if hour_start_ms not in pm_lookup:
            results.append(MarketResult(market_id=market_id, outcome=Y,
                                        trades=[], pnl=0.0, gross_edge=0.0))
            continue

        pm_ts, pm_bid, pm_ask, pm_mid_arr = pm_lookup[hour_start_ms]

        grid_ts = np.arange(hour_start_ms, hour_end_ms, grid_ms, dtype=np.int64)

        # LOCF Binance mid onto grid
        idx_bnc = np.searchsorted(bbo_ts, grid_ts, side="right") - 1
        idx_bnc = np.clip(idx_bnc, 0, len(bbo_ts) - 1)
        S_grid = bbo_mid[idx_bnc]

        # LOCF PM bid/ask/mid onto grid
        idx_pm = np.searchsorted(pm_ts, grid_ts, side="right") - 1
        idx_pm = np.clip(idx_pm, 0, len(pm_ts) - 1)
        pm_bid_grid = pm_bid[idx_pm]
        pm_ask_grid = pm_ask[idx_pm]
        pm_mid_grid = pm_mid_arr[idx_pm]

        tau_grid = (hour_end_ms - grid_ts) / 1000.0

        valid = ~np.isnan(S_grid) & (tau_grid > 0)
        if not valid.any():
            results.append(MarketResult(market_id=market_id, outcome=Y,
                                        trades=[], pnl=0.0, gross_edge=0.0))
            continue

        grid_ts = grid_ts[valid]
        S_grid = S_grid[valid]
        tau_grid = tau_grid[valid]
        pm_bid_grid = pm_bid_grid[valid]
        pm_ask_grid = pm_ask_grid[valid]
        pm_mid_grid = pm_mid_grid[valid]

        # Nearest-neighbor sigma_rv lookup
        idx = np.searchsorted(ts_rv, grid_ts)
        idx = np.clip(idx, 0, len(ts_rv) - 1)
        idx_prev = np.clip(idx - 1, 0, len(ts_rv) - 1)
        dist_right = np.abs(ts_rv[idx] - grid_ts)
        dist_left = np.abs(ts_rv[idx_prev] - grid_ts)
        best_idx = np.where(dist_left < dist_right, idx_prev, idx)
        sigma_rv_grid = sigma_rv_full[best_idx]

        model_p = pricer.price(
            S=S_grid, K=np.full_like(S_grid, K),
            tau=tau_grid, sigma_rv=sigma_rv_grid, t_ms=grid_ts,
        )

        # Compute additional features for conditional strategies
        sigma_tod_grid = pricer.sigma_tod(t_ms=grid_ts, tau=tau_grid)
        sigma_rel_grid = sigma_rv_grid / np.maximum(sigma_tod_grid, 1e-12)
        pm_spread_grid = pm_ask_grid - pm_bid_grid

        snapshots = pd.DataFrame({
            "ts": grid_ts,
            "tau": tau_grid,
            "S": S_grid,
            "K": K,
            "model_p": model_p,
            "pm_bid": pm_bid_grid,
            "pm_ask": pm_ask_grid,
            "pm_mid": pm_mid_grid,
            "sigma_rv": sigma_rv_grid,
            "sigma_rel": sigma_rel_grid,
            "pm_spread": pm_spread_grid,
        })

        # Inject bankroll into strategy for Kelly sizing
        if bankroll is not None and hasattr(strategy, '_bankroll'):
            strategy._bankroll = current_bankroll

        trades = strategy.generate_trades(snapshots)

        # Populate hour_et on trades
        for t in trades:
            t.hour_et = dt_et.hour

        # Compute MFE/MAE + apply TP/SL
        tp_frac = getattr(strategy, 'tp_frac', 0.0)
        sl_frac = getattr(strategy, 'sl_frac', 0.0)
        for t in trades:
            t.mfe, t.mae, t.exit_pnl, t.exit_reason = _compute_mfe_mae_tpsl(
                t.ts, t.side, t.price, t.size,
                grid_ts, pm_bid_grid, pm_ask_grid,
                tp_frac=tp_frac, sl_frac=sl_frac,
            )

        mr = MarketResult.from_trades(market_id, Y, trades)
        results.append(mr)

        # Update bankroll
        if bankroll is not None:
            current_bankroll += mr.pnl
            current_bankroll = max(current_bankroll, 0.0)  # can't go negative
            bankroll_history.append(current_bankroll)

    return results, np.array(bankroll_history)


def _config(asset, start_date, end_date, strategy, ewma_half_life=600.0) -> dict:
    return {
        "asset": asset,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "strategy": strategy.name,
        "ewma_half_life": ewma_half_life,
    }
