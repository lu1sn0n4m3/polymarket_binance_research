"""Backtest analytics: summary metrics and plots."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np

from backtesting.core import BacktestResult, Trade

ET = ZoneInfo("America/New_York")


def print_summary(result: BacktestResult) -> None:
    """Print backtest summary to stdout."""
    cfg = result.config
    print(f"\n{'='*60}")
    print(f"BACKTEST SUMMARY")
    print(f"{'='*60}")
    print(f"  Strategy:   {cfg.get('strategy', '?')}")
    print(f"  Asset:      {cfg.get('asset', '?')}")
    print(f"  Period:     {cfg.get('start_date')} to {cfg.get('end_date')}")
    print(f"  EWMA H:     {cfg.get('ewma_half_life', 600)}s")

    print(f"\n  Markets:    {result.n_markets} total, {result.n_markets_traded} traded")
    print(f"  Trades:     {result.n_trades}")

    if result.n_trades == 0:
        print("  No trades to analyze.")
        return

    pnl = result.pnl_per_market
    all_trades = [t for m in result.markets for t in m.trades]

    print(f"\n--- PnL ---")
    print(f"  Total PnL:       {result.total_pnl:+.4f}")
    print(f"  PnL / trade:     {result.total_pnl / result.n_trades:+.4f}")
    print(f"  PnL / market:    {result.total_pnl / result.n_markets_traded:+.4f}")
    print(f"  Hit rate:        {result.hit_rate:.1%}")
    print(f"  Sharpe (ann.):   {result.sharpe:.2f}")
    print(f"  Max drawdown:    {result.max_drawdown:.4f}")

    edges = np.array([t.edge for t in all_trades])
    print(f"\n--- Edge ---")
    print(f"  Avg edge:        {np.mean(edges):.4f}")
    print(f"  Median edge:     {np.median(edges):.4f}")
    print(f"  Max edge:        {np.max(edges):.4f}")

    # Buy vs Sell breakdown
    buys = [t for t in all_trades if t.side == "BUY"]
    sells = [t for t in all_trades if t.side == "SELL"]
    print(f"\n--- Side breakdown ---")
    print(f"  BUY:  {len(buys)} trades")
    print(f"  SELL: {len(sells)} trades")

    # PnL by hour (ET)
    print(f"\n--- PnL by hour (ET) ---")
    hour_pnl: dict[int, list[float]] = {}
    for m in result.markets:
        if not m.trades:
            continue
        ts_ms = m.trades[0].ts
        dt_et = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(ET)
        h = dt_et.hour
        hour_pnl.setdefault(h, []).append(m.pnl)

    for h in sorted(hour_pnl):
        pnls = hour_pnl[h]
        total = sum(pnls)
        n = len(pnls)
        hr = sum(1 for p in pnls if p > 0) / n if n > 0 else 0
        print(f"  {h:02d}:00 ET  n={n:3d}  PnL={total:+.4f}  hit={hr:.0%}")

    # PnL by edge bucket
    print(f"\n--- PnL by edge bucket ---")
    edge_bins = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 1.0]
    trade_pnls = []
    trade_edges = []
    for m in result.markets:
        for t in m.trades:
            trade_pnls.append(t.settle(m.outcome))
            trade_edges.append(t.edge)
    trade_pnls = np.array(trade_pnls)
    trade_edges = np.array(trade_edges)

    for i in range(len(edge_bins) - 1):
        lo, hi = edge_bins[i], edge_bins[i + 1]
        mask = (trade_edges >= lo) & (trade_edges < hi)
        n = mask.sum()
        if n == 0:
            continue
        avg_pnl = np.mean(trade_pnls[mask])
        hr = np.mean(trade_pnls[mask] > 0)
        print(f"  [{lo:.2f}, {hi:.2f})  n={n:3d}  avg_pnl={avg_pnl:+.4f}  hit={hr:.0%}")


def plot_backtest(result: BacktestResult, save_path: str = "backtesting/output/backtest.png") -> None:
    """Generate 2x2 diagnostic plot."""
    import matplotlib.pyplot as plt
    from pathlib import Path

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Backtest: {result.config.get('strategy', '?')}", fontsize=13, fontweight="bold")

    # (0,0) Cumulative PnL
    ax = axes[0, 0]
    pnl_series = np.array([m.pnl for m in result.markets if m.trades])
    if len(pnl_series) > 0:
        cum_pnl = np.cumsum(pnl_series)
        ax.plot(cum_pnl, color="#27ae60", lw=1.5)
        ax.axhline(0, color="k", ls="--", lw=0.8)
        ax.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                        where=cum_pnl >= 0, alpha=0.15, color="#27ae60")
        ax.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                        where=cum_pnl < 0, alpha=0.15, color="#e74c3c")
    ax.set_xlabel("Traded market #")
    ax.set_ylabel("Cumulative PnL")
    ax.set_title("Cumulative PnL")

    # (0,1) PnL distribution per trade
    ax = axes[0, 1]
    trade_pnls = []
    for m in result.markets:
        for t in m.trades:
            trade_pnls.append(t.settle(m.outcome))
    if trade_pnls:
        trade_pnls = np.array(trade_pnls)
        ax.hist(trade_pnls, bins=40, color="#3498db", alpha=0.7, edgecolor="white")
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.axvline(np.mean(trade_pnls), color="#e74c3c", ls="-", lw=1.5,
                   label=f"mean={np.mean(trade_pnls):.4f}")
        ax.legend(fontsize=8)
    ax.set_xlabel("PnL per trade")
    ax.set_ylabel("Count")
    ax.set_title("PnL Distribution")

    # (1,0) Hit rate by edge bucket
    ax = axes[1, 0]
    if trade_pnls is not None and len(trade_pnls) > 0:
        edges = np.array([t.edge for m in result.markets for t in m.trades])
        bins = np.percentile(edges, np.linspace(0, 100, 8))
        bins = np.unique(bins)
        if len(bins) > 1:
            centers, hit_rates = [], []
            for i in range(len(bins) - 1):
                mask = (edges >= bins[i]) & (edges < bins[i + 1])
                if i == len(bins) - 2:
                    mask = (edges >= bins[i]) & (edges <= bins[i + 1])
                if mask.sum() > 0:
                    centers.append((bins[i] + bins[i + 1]) / 2)
                    hit_rates.append(np.mean(trade_pnls[mask] > 0))
            ax.bar(range(len(centers)), hit_rates, color="#9b59b6", alpha=0.7)
            ax.set_xticks(range(len(centers)))
            ax.set_xticklabels([f"{c:.3f}" for c in centers], rotation=45, fontsize=7)
            ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Edge at entry")
    ax.set_ylabel("Hit rate")
    ax.set_title("Hit Rate by Edge")

    # (1,1) PnL by hour (ET)
    ax = axes[1, 1]
    hour_pnl: dict[int, float] = {}
    for m in result.markets:
        if not m.trades:
            continue
        ts_ms = m.trades[0].ts
        dt_et = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(ET)
        h = dt_et.hour
        hour_pnl[h] = hour_pnl.get(h, 0) + m.pnl
    if hour_pnl:
        hours = sorted(hour_pnl)
        pnls = [hour_pnl[h] for h in hours]
        colors = ["#27ae60" if p >= 0 else "#e74c3c" for p in pnls]
        ax.bar(hours, pnls, color=colors, alpha=0.7)
        ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Hour (ET)")
    ax.set_ylabel("Total PnL")
    ax.set_title("PnL by Hour (ET)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {save_path}")
    plt.close()
