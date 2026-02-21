"""Strategy comparison at realistic 250ms latency with 100ms grid resolution.

Usage:
    venv/bin/python3 backtesting/run_best.py
"""

import sys
sys.path.insert(0, ".")

from datetime import date
import numpy as np

from backtesting.strategies.threshold import EdgeThresholdStrategy
from backtesting.strategies.conditional import ConditionalStrategy
from backtesting.walk_forward import walk_forward_backtest
from backtesting.analytics import print_summary, plot_backtest

ASSET = "BTC"
START = date(2026, 1, 19)
END = date(2026, 2, 18)
BANKROLL = 100.0
LATENCY = 250  # realistic: max 200ms, use 250ms to be safe
GRID_MS = 100  # 100ms grid for sub-second latency resolution

results = {}

# =================================================================
# A: Flat threshold at 250ms
# =================================================================
print("\n" + "=" * 70)
print(f"  A: FLAT THRESHOLD — {LATENCY}ms latency ({GRID_MS}ms grid)")
print("=" * 70)

strat_a = EdgeThresholdStrategy(
    min_edge=0.05, min_tau=60.0, max_entries=1,
    sizing="kelly", kelly_fraction=0.5, max_kelly_fraction=0.15,
    tp_frac=0.5, sl_frac=0.2,
    latency_ms=LATENCY,
)
results["flat_250ms"] = walk_forward_backtest(
    ASSET, START, END, strat_a, initial_bankroll=BANKROLL, grid_ms=GRID_MS,
)
print_summary(results["flat_250ms"])
plot_backtest(results["flat_250ms"], save_path="backtesting/output/flat_250ms.png")


# =================================================================
# B: Walk-forward conditional at 250ms
# =================================================================
print("\n" + "=" * 70)
print(f"  B: CONDITIONAL (walk-forward hours) — {LATENCY}ms latency ({GRID_MS}ms grid)")
print("=" * 70)

strat_b = ConditionalStrategy(
    min_edge=0.05, min_tau=60.0, max_entries=1,
    kelly_fraction=0.5, max_kelly_fraction=0.15,
    tp_frac=0.5, sl_frac=0.2,
    latency_ms=LATENCY,
    skip_unknown_mult=0.5,
)
results["cond_250ms"] = walk_forward_backtest(
    ASSET, START, END, strat_b, initial_bankroll=BANKROLL, grid_ms=GRID_MS,
)
print_summary(results["cond_250ms"])
plot_backtest(results["cond_250ms"], save_path="backtesting/output/conditional_250ms.png")


# =================================================================
# C: Flat threshold at 0ms (ceiling)
# =================================================================
print("\n" + "=" * 70)
print(f"  C: FLAT THRESHOLD — 0ms (theoretical ceiling, {GRID_MS}ms grid)")
print("=" * 70)

strat_c = EdgeThresholdStrategy(
    min_edge=0.05, min_tau=60.0, max_entries=1,
    sizing="kelly", kelly_fraction=0.5, max_kelly_fraction=0.15,
    tp_frac=0.5, sl_frac=0.2,
    latency_ms=0,
)
results["flat_0ms"] = walk_forward_backtest(
    ASSET, START, END, strat_c, initial_bankroll=BANKROLL, grid_ms=GRID_MS,
)
print_summary(results["flat_0ms"])
plot_backtest(results["flat_0ms"], save_path="backtesting/output/flat_0ms.png")


# =================================================================
# COMPARISON
# =================================================================
print("\n" + "=" * 70)
print(f"  STRATEGY COMPARISON (all walk-forward, {GRID_MS}ms grid, no lookahead)")
print("=" * 70)

print(f"\n  {'Strategy':30s} {'trades':>6s} {'PnL':>10s} {'ret%':>8s} {'DD%':>6s} "
      f"{'hit':>5s} {'PnL/trade':>10s}")
print(f"  {'-'*30} {'-'*6} {'-'*10} {'-'*8} {'-'*6} {'-'*5} {'-'*10}")

for label, r in results.items():
    curve = r.bankroll_curve
    final = curve[-1] if len(curve) > 0 else BANKROLL
    ret = (final / BANKROLL - 1) * 100
    if len(curve) > 1:
        running_peak = np.maximum.accumulate(curve)
        dd = float(np.max(
            (running_peak - curve) / np.where(running_peak > 0, running_peak, 1)
        )) * 100
    else:
        dd = 0
    avg_pnl = r.total_pnl / max(r.n_trades, 1)
    print(f"  {label:30s} {r.n_trades:6d} ${r.total_pnl:+9.2f} {ret:+7.1f}% "
          f"{dd:5.1f}% {r.hit_rate:4.0%} ${avg_pnl:+9.4f}")
