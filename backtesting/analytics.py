"""Backtest analytics: comprehensive summary metrics and plots."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np

from backtesting.core import BacktestResult, Trade

ET = ZoneInfo("America/New_York")


def _all_trade_data(result: BacktestResult) -> dict[str, np.ndarray]:
    """Extract flat arrays of trade-level data for sliced analysis."""
    trades = []
    for m in result.markets:
        for t in m.trades:
            trades.append({
                "ts": t.ts,
                "side": 1 if t.side == "BUY" else -1,
                "side_str": t.side,
                "size": t.size,
                "price": t.price,
                "model_p": t.model_price,
                "edge": t.edge,
                "tau": t.tau,
                "sigma_rv": t.sigma_rv,
                "sigma_rel": t.sigma_rel,
                "pm_spread": t.pm_spread,
                "hour_et": t.hour_et,
                "bankroll": t.bankroll_at_entry,
                "kelly_f": t.kelly_fraction,
                "mfe": t.mfe,
                "mae": t.mae,
                "pnl": t.settle(m.outcome),
                "outcome": m.outcome,
                "market_id": m.market_id,
            })
    if not trades:
        return {}
    return {k: np.array([t[k] for t in trades]) for k in trades[0]}


def _print_section(title: str, width: int = 60) -> None:
    print(f"\n{'─'*width}")
    print(f"  {title}")
    print(f"{'─'*width}")


def _bucket_analysis(
    values: np.ndarray,
    pnls: np.ndarray,
    sizes: np.ndarray,
    bins: list[float],
    label: str,
    format_fn=None,
) -> None:
    """Print PnL breakdown by bucket."""
    if format_fn is None:
        format_fn = lambda lo, hi: f"[{lo:.3f}, {hi:.3f})"

    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if i == len(bins) - 2:
            mask = (values >= lo) & (values <= hi)
        else:
            mask = (values >= lo) & (values < hi)
        n = mask.sum()
        if n == 0:
            continue
        total_pnl = np.sum(pnls[mask])
        avg_pnl = np.mean(pnls[mask])
        hr = np.mean(pnls[mask] > 0)
        avg_size = np.mean(sizes[mask])
        print(f"  {format_fn(lo, hi):18s}  n={n:3d}  PnL={total_pnl:+8.2f}  "
              f"avg={avg_pnl:+.4f}  hit={hr:.0%}  avg_size=${avg_size:.2f}")


def print_summary(result: BacktestResult) -> None:
    """Print comprehensive backtest summary to stdout."""
    cfg = result.config
    print(f"\n{'='*70}")
    print(f"  BACKTEST REPORT")
    print(f"{'='*70}")
    print(f"  Strategy:   {cfg.get('strategy', '?')}")
    print(f"  Asset:      {cfg.get('asset', '?')}")
    print(f"  Period:     {cfg.get('start_date')} to {cfg.get('end_date')}")
    print(f"  Mode:       {cfg.get('mode', 'in-sample')}")
    initial_br = cfg.get("initial_bankroll", None)
    if initial_br is not None:
        print(f"  Bankroll:   ${initial_br:.2f} initial")

    td = _all_trade_data(result)
    if not td:
        print("\n  No trades to analyze.")
        return

    pnls = td["pnl"]
    sizes = td["size"]
    edges = td["edge"]
    taus = td["tau"]
    sigma_rvs = td["sigma_rv"]
    hours_et = td["hour_et"]
    mfes = td["mfe"]
    maes = td["mae"]
    bankrolls = td["bankroll"]

    n_trades = len(pnls)
    n_win = np.sum(pnls > 0)
    n_lose = np.sum(pnls <= 0)

    # =================================================================
    # HEADLINE NUMBERS
    # =================================================================
    _print_section("HEADLINE NUMBERS")

    print(f"  Markets:         {result.n_markets} total, {result.n_markets_traded} traded")
    print(f"  Trades:          {n_trades}")
    print(f"  Total PnL:       ${np.sum(pnls):+.2f}")
    print(f"  PnL / trade:     ${np.mean(pnls):+.4f}")
    print(f"  Hit rate:        {n_win}/{n_trades} = {n_win/n_trades:.1%}")
    print(f"  Avg win:         ${np.mean(pnls[pnls > 0]):+.4f}" if n_win > 0 else "")
    print(f"  Avg loss:        ${np.mean(pnls[pnls <= 0]):+.4f}" if n_lose > 0 else "")

    # Win/loss ratio
    if n_win > 0 and n_lose > 0:
        avg_w = np.mean(pnls[pnls > 0])
        avg_l = abs(np.mean(pnls[pnls <= 0]))
        print(f"  Win/loss ratio:  {avg_w/avg_l:.2f}")

    # Sharpe (per-market)
    market_pnls = result.pnl_per_market
    if len(market_pnls) > 1:
        daily_sharpe = np.mean(market_pnls) / np.std(market_pnls, ddof=1) * np.sqrt(24)
        print(f"  Daily Sharpe:    {daily_sharpe:.2f}")
        print(f"  Ann. Sharpe:     {daily_sharpe * np.sqrt(365):.2f}")

    print(f"  Max drawdown:    ${result.max_drawdown:.2f}")

    # Bankroll
    if initial_br is not None and len(result.bankroll_curve) > 0:
        final_br = result.bankroll_curve[-1]
        ret = (final_br / initial_br - 1) * 100
        # Running peak drawdown (correct method)
        curve = result.bankroll_curve
        running_peak = np.maximum.accumulate(curve)
        drawdowns = (running_peak - curve) / np.where(running_peak > 0, running_peak, 1)
        br_dd = float(np.max(drawdowns)) * 100
        print(f"\n  Bankroll:        ${initial_br:.2f} → ${final_br:.2f}  ({ret:+.1f}%)")
        print(f"  Peak bankroll:   ${np.max(curve):.2f}")
        print(f"  Max BR drawdown: {br_dd:.1f}%")

    # =================================================================
    # EDGE ANALYSIS
    # =================================================================
    _print_section("PnL BY EDGE BUCKET")

    edge_bins = [0.0, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 1.0]
    _bucket_analysis(edges, pnls, sizes, edge_bins, "Edge")

    # =================================================================
    # PnL BY TIME OF DAY (ET)
    # =================================================================
    _print_section("PnL BY HOUR (ET)")

    for h in sorted(np.unique(hours_et)):
        if h < 0:
            continue
        mask = hours_et == h
        n = mask.sum()
        total = np.sum(pnls[mask])
        hr = np.mean(pnls[mask] > 0)
        avg_size = np.mean(sizes[mask])
        avg_edge = np.mean(edges[mask])
        print(f"  {int(h):02d}:00 ET  n={n:3d}  PnL=${total:+8.2f}  "
              f"hit={hr:.0%}  avg_edge={avg_edge:.3f}  avg_size=${avg_size:.2f}")

    # =================================================================
    # PnL BY VOLATILITY REGIME
    # =================================================================
    _print_section("PnL BY VOLATILITY REGIME (sigma_rv quantiles)")

    valid_srv = sigma_rvs > 0
    if valid_srv.any():
        srv_valid = sigma_rvs[valid_srv]
        # Use quantiles for adaptive bins
        q_vals = np.percentile(srv_valid, [0, 25, 50, 75, 100])
        q_vals = np.unique(q_vals)
        if len(q_vals) > 1:
            labels_q = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
            for i in range(len(q_vals) - 1):
                lo, hi = q_vals[i], q_vals[i + 1]
                if i == len(q_vals) - 2:
                    mask = (sigma_rvs >= lo) & (sigma_rvs <= hi)
                else:
                    mask = (sigma_rvs >= lo) & (sigma_rvs < hi)
                n = mask.sum()
                if n == 0:
                    continue
                total = np.sum(pnls[mask])
                hr = np.mean(pnls[mask] > 0)
                avg_edge = np.mean(edges[mask])
                lbl = labels_q[i] if i < len(labels_q) else f"Q{i+1}"
                print(f"  {lbl:12s} σ_rv=[{lo:.2e},{hi:.2e})  n={n:3d}  "
                      f"PnL=${total:+8.2f}  hit={hr:.0%}  avg_edge={avg_edge:.3f}")

    # =================================================================
    # PnL BY TAU (time to expiry)
    # =================================================================
    _print_section("PnL BY TAU (seconds to expiry)")

    tau_bins = [60, 300, 600, 900, 1800, 2700, 3600]
    _bucket_analysis(taus, pnls, sizes, tau_bins, "Tau",
                     format_fn=lambda lo, hi: f"[{lo:.0f}s, {hi:.0f}s)")

    # =================================================================
    # BUY vs SELL
    # =================================================================
    _print_section("BUY vs SELL BREAKDOWN")

    for side_str in ["BUY", "SELL"]:
        mask = td["side_str"] == side_str
        n = mask.sum()
        if n == 0:
            continue
        total = np.sum(pnls[mask])
        hr = np.mean(pnls[mask] > 0)
        avg_edge = np.mean(edges[mask])
        avg_size = np.mean(sizes[mask])
        print(f"  {side_str:5s}  n={n:3d}  PnL=${total:+8.2f}  "
              f"hit={hr:.0%}  avg_edge={avg_edge:.3f}  avg_size=${avg_size:.2f}")

    # =================================================================
    # MFE / MAE ANALYSIS (early exit opportunities)
    # =================================================================
    _print_section("MFE / MAE ANALYSIS (unrealized PnL path)")

    has_mfe = np.any(mfes != 0) or np.any(maes != 0)
    # Filter out NaN
    mfe_valid = ~np.isnan(mfes)
    mae_valid = ~np.isnan(maes)
    if has_mfe:
        print(f"  Avg MFE (best unrealized):   ${np.nanmean(mfes):+.4f}")
        print(f"  Avg MAE (worst unrealized):  ${np.nanmean(maes):+.4f}")
        safe_sizes = np.where(sizes > 0, sizes, 1)
        print(f"  Avg MFE/size:                {np.nanmean(mfes/safe_sizes):.4f}")
        print(f"  Avg MAE/size:                {np.nanmean(maes/safe_sizes):.4f}")

        # How many winners had significant drawdown first?
        winners = pnls > 0
        losers = pnls <= 0
        if winners.any():
            avg_mae_winners = np.nanmean(maes[winners])
            pct_winners_with_dd = np.nanmean(maes[winners] < -0.01)
            print(f"\n  Winners ({winners.sum()}):")
            print(f"    Avg MAE before winning:    ${avg_mae_winners:+.4f}")
            print(f"    % that dipped > $0.01:     {pct_winners_with_dd:.0%}")
            print(f"    Avg MFE (peak unrealized): ${np.nanmean(mfes[winners]):+.4f}")
        if losers.any():
            avg_mfe_losers = np.nanmean(mfes[losers])
            pct_losers_with_profit = np.nanmean(mfes[losers] > 0.01)
            print(f"  Losers ({losers.sum()}):")
            print(f"    Avg MFE before losing:     ${avg_mfe_losers:+.4f}")
            print(f"    % that were profitable:    {pct_losers_with_profit:.0%}")
            print(f"    Avg MAE (worst point):     ${np.nanmean(maes[losers]):+.4f}")

        # Early exit simulation
        print(f"\n  Early exit simulation:")
        # What if we took profit at X% of size?
        for tp_frac in [0.3, 0.5, 0.8]:
            tp_threshold = sizes * tp_frac  # take profit threshold
            # For each trade: PnL = min(MFE, tp_threshold) if MFE > tp, else actual PnL
            simulated_pnl = np.where(mfes >= tp_threshold, tp_threshold, pnls)
            sim_total = np.sum(simulated_pnl)
            sim_hr = np.mean(simulated_pnl > 0)
            print(f"    TP at {tp_frac:.0%} of size: PnL=${sim_total:+.2f}  "
                  f"hit={sim_hr:.0%}  "
                  f"vs actual ${np.sum(pnls):+.2f}")

        # What if we cut losses at X% of size?
        for sl_frac in [0.3, 0.5, 0.8]:
            sl_threshold = -sizes * sl_frac
            simulated_pnl = np.where(maes <= sl_threshold, sl_threshold, pnls)
            sim_total = np.sum(simulated_pnl)
            sim_hr = np.mean(simulated_pnl > 0)
            print(f"    SL at {sl_frac:.0%} of size: PnL=${sim_total:+.2f}  "
                  f"hit={sim_hr:.0%}  "
                  f"vs actual ${np.sum(pnls):+.2f}")

    # =================================================================
    # KELLY SIZING DIAGNOSTICS
    # =================================================================
    kelly_fs = td["kelly_f"]
    if np.any(kelly_fs > 0):
        _print_section("KELLY SIZING DIAGNOSTICS")
        print(f"  Avg Kelly fraction:  {np.mean(kelly_fs):.3f}")
        print(f"  Max Kelly fraction:  {np.max(kelly_fs):.3f}")
        print(f"  Avg size ($):        ${np.mean(sizes):.2f}")
        print(f"  Max size ($):        ${np.max(sizes):.2f}")
        print(f"  Avg size / bankroll: {np.mean(sizes/np.where(bankrolls>0,bankrolls,1)):.1%}")

    # =================================================================
    # ROBUSTNESS: REMOVE TOP N% OF TRADES
    # =================================================================
    _print_section("ROBUSTNESS: PnL WITHOUT TOP TRADES")

    for pct in [1, 5, 10]:
        n_remove = max(1, int(np.ceil(n_trades * pct / 100)))
        # Sort by PnL descending, remove top N
        sorted_idx = np.argsort(pnls)[::-1]
        kept_idx = sorted_idx[n_remove:]
        kept_pnl = pnls[kept_idx]
        removed_pnl = pnls[sorted_idx[:n_remove]]
        kept_total = np.sum(kept_pnl)
        kept_hr = np.mean(kept_pnl > 0) if len(kept_pnl) > 0 else 0
        print(f"  Remove top {pct:2d}% ({n_remove:3d} trades, "
              f"avg PnL=${np.mean(removed_pnl):+.2f}):  "
              f"remaining PnL=${kept_total:+.2f}  "
              f"hit={kept_hr:.0%}  "
              f"avg=${np.mean(kept_pnl):+.4f}" if len(kept_pnl) > 0 else "")

    # Also show what happens removing top AND bottom
    _print_section("ROBUSTNESS: REMOVE TOP & BOTTOM 5% (WINSORIZED)")
    n_remove = max(1, int(np.ceil(n_trades * 5 / 100)))
    sorted_idx = np.argsort(pnls)
    kept_idx = sorted_idx[n_remove:-n_remove] if n_remove * 2 < n_trades else sorted_idx
    kept_pnl = pnls[kept_idx]
    if len(kept_pnl) > 0:
        print(f"  Kept {len(kept_pnl)} of {n_trades} trades")
        print(f"  Winsorized PnL:  ${np.sum(kept_pnl):+.2f}")
        print(f"  Winsorized avg:  ${np.mean(kept_pnl):+.4f}")
        print(f"  Winsorized hit:  {np.mean(kept_pnl > 0):.0%}")

    print(f"\n{'='*70}\n")


def plot_backtest(result: BacktestResult, save_path: str = "backtesting/output/backtest.png") -> None:
    """Generate comprehensive diagnostic plots (3x2 grid)."""
    import matplotlib.pyplot as plt
    from pathlib import Path

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    td = _all_trade_data(result)
    if not td:
        print("No trades to plot.")
        return

    pnls = td["pnl"]
    edges = td["edge"]
    hours_et = td["hour_et"]
    sigma_rvs = td["sigma_rv"]
    mfes = td["mfe"]
    maes = td["mae"]
    sizes = td["size"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    fig.suptitle(f"Backtest: {result.config.get('strategy', '?')}", fontsize=13, fontweight="bold")

    c_green = "#27ae60"
    c_red = "#e74c3c"
    c_blue = "#3498db"
    c_purple = "#9b59b6"

    # (0,0) Cumulative PnL / Bankroll
    ax = axes[0, 0]
    if len(result.bankroll_curve) > 1:
        ax.plot(result.bankroll_curve, color=c_blue, lw=1.5)
        ax.set_ylabel("Bankroll ($)")
        ax.set_title("Bankroll Over Time")
        ax.axhline(result.bankroll_curve[0], color="k", ls="--", lw=0.8, alpha=0.5)
    else:
        pnl_series = np.array([m.pnl for m in result.markets if m.trades])
        if len(pnl_series) > 0:
            cum_pnl = np.cumsum(pnl_series)
            ax.plot(cum_pnl, color=c_green, lw=1.5)
            ax.axhline(0, color="k", ls="--", lw=0.8)
            ax.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                            where=cum_pnl >= 0, alpha=0.15, color=c_green)
            ax.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                            where=cum_pnl < 0, alpha=0.15, color=c_red)
        ax.set_ylabel("Cumulative PnL")
        ax.set_title("Cumulative PnL")
    ax.set_xlabel("Market #")
    ax.grid(True, alpha=0.3)

    # (0,1) PnL distribution per trade
    ax = axes[0, 1]
    ax.hist(pnls, bins=40, color=c_blue, alpha=0.7, edgecolor="white")
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.axvline(np.mean(pnls), color=c_red, ls="-", lw=1.5,
               label=f"mean=${np.mean(pnls):.4f}")
    ax.legend(fontsize=8)
    ax.set_xlabel("PnL per trade ($)")
    ax.set_ylabel("Count")
    ax.set_title("PnL Distribution")
    ax.grid(True, alpha=0.3)

    # (1,0) PnL by hour (ET)
    ax = axes[1, 0]
    hour_pnl: dict[int, float] = {}
    hour_count: dict[int, int] = {}
    for h in np.unique(hours_et):
        if h < 0:
            continue
        h = int(h)
        mask = hours_et == h
        hour_pnl[h] = float(np.sum(pnls[mask]))
        hour_count[h] = int(mask.sum())
    if hour_pnl:
        hours = sorted(hour_pnl)
        vals = [hour_pnl[h] for h in hours]
        colors = [c_green if p >= 0 else c_red for p in vals]
        ax.bar(hours, vals, color=colors, alpha=0.7)
        ax.axhline(0, color="k", ls="--", lw=0.8)
        # Add trade count labels
        for h, v in zip(hours, vals):
            ax.text(h, v, f"n={hour_count[h]}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=6)
    ax.set_xlabel("Hour (ET)")
    ax.set_ylabel("Total PnL ($)")
    ax.set_title("PnL by Hour (ET)")
    ax.grid(True, alpha=0.3)

    # (1,1) Hit rate + avg PnL by edge bucket
    ax = axes[1, 1]
    edge_bins = np.percentile(edges, np.linspace(0, 100, 8))
    edge_bins = np.unique(edge_bins)
    if len(edge_bins) > 1:
        centers, hit_rates, avg_pnls = [], [], []
        for i in range(len(edge_bins) - 1):
            lo, hi = edge_bins[i], edge_bins[i + 1]
            mask = (edges >= lo) & (edges < hi) if i < len(edge_bins) - 2 else (edges >= lo) & (edges <= hi)
            if mask.sum() > 0:
                centers.append(f"{lo:.3f}")
                hit_rates.append(np.mean(pnls[mask] > 0))
                avg_pnls.append(np.mean(pnls[mask]))
        x = range(len(centers))
        ax.bar(x, hit_rates, color=c_purple, alpha=0.7)
        ax.set_xticks(list(x))
        ax.set_xticklabels(centers, rotation=45, fontsize=7)
        ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Edge at entry")
    ax.set_ylabel("Hit rate")
    ax.set_title("Hit Rate by Edge Bucket")
    ax.grid(True, alpha=0.3)

    # (2,0) PnL by sigma_rv regime
    ax = axes[2, 0]
    valid_srv = sigma_rvs > 0
    if valid_srv.any():
        srv_q = np.percentile(sigma_rvs[valid_srv], [0, 20, 40, 60, 80, 100])
        srv_q = np.unique(srv_q)
        if len(srv_q) > 1:
            centers, totals = [], []
            for i in range(len(srv_q) - 1):
                lo, hi = srv_q[i], srv_q[i + 1]
                mask = (sigma_rvs >= lo) & (sigma_rvs < hi) if i < len(srv_q) - 2 else (sigma_rvs >= lo) & (sigma_rvs <= hi)
                n = mask.sum()
                if n > 0:
                    centers.append(f"{lo:.1e}")
                    totals.append(float(np.sum(pnls[mask])))
            colors = [c_green if t >= 0 else c_red for t in totals]
            ax.bar(range(len(centers)), totals, color=colors, alpha=0.7)
            ax.set_xticks(range(len(centers)))
            ax.set_xticklabels(centers, rotation=45, fontsize=7)
            ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("sigma_rv quintile")
    ax.set_ylabel("Total PnL ($)")
    ax.set_title("PnL by Volatility Regime")
    ax.grid(True, alpha=0.3)

    # (2,1) MFE vs MAE scatter
    ax = axes[2, 1]
    has_excursion = np.any(mfes != 0) or np.any(maes != 0)
    if has_excursion:
        colors_scatter = np.where(pnls > 0, c_green, c_red)
        ax.scatter(maes, mfes, c=colors_scatter, s=8, alpha=0.5)
        ax.axhline(0, color="k", ls="--", lw=0.5)
        ax.axvline(0, color="k", ls="--", lw=0.5)
        # Reference line: perfect trade (MFE = max win)
        ax.set_xlabel("MAE (worst unrealized, $)")
        ax.set_ylabel("MFE (best unrealized, $)")
        ax.set_title("MFE vs MAE (green=winner, red=loser)")
    else:
        ax.text(0.5, 0.5, "No MFE/MAE data", transform=ax.transAxes,
                ha="center", va="center")
        ax.set_title("MFE vs MAE")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {save_path}")
    plt.close()
