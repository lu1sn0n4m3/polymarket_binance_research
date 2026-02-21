"""Parameter sweep for walk-forward backtest.

Post-audit sweep: SL/TP exits now use actual fill price (not threshold),
drawdown uses running peak. No PM taker fees on hourly markets.

Usage:
    venv/bin/python3 backtesting/sweep.py
"""

import sys
sys.path.insert(0, ".")

from datetime import date
from backtesting.strategies.threshold import EdgeThresholdStrategy
from backtesting.walk_forward import walk_forward_backtest

ASSET = "BTC"
START = date(2026, 1, 19)
END = date(2026, 2, 18)
BANKROLL = 100.0

configs = [
    # Baselines: no TP/SL (unchanged by the fix)
    dict(min_edge=0.03, kelly_fraction=0.5, max_kelly_fraction=0.15, tp_frac=0.0, sl_frac=0.0),
    dict(min_edge=0.05, kelly_fraction=0.5, max_kelly_fraction=0.15, tp_frac=0.0, sl_frac=0.0),

    # Previous "winners" — now with realistic fills
    dict(min_edge=0.03, kelly_fraction=0.5, max_kelly_fraction=0.15, tp_frac=0.5, sl_frac=0.3),
    dict(min_edge=0.03, kelly_fraction=0.5, max_kelly_fraction=0.20, tp_frac=0.5, sl_frac=0.3),
    dict(min_edge=0.05, kelly_fraction=0.5, max_kelly_fraction=0.15, tp_frac=0.5, sl_frac=0.3),
    dict(min_edge=0.05, kelly_fraction=0.5, max_kelly_fraction=0.20, tp_frac=0.5, sl_frac=0.3),
    dict(min_edge=0.05, kelly_fraction=0.5, max_kelly_fraction=0.15, tp_frac=0.5, sl_frac=0.2),

    # Wider net around best combos
    dict(min_edge=0.04, kelly_fraction=0.5, max_kelly_fraction=0.15, tp_frac=0.5, sl_frac=0.3),
    dict(min_edge=0.04, kelly_fraction=0.5, max_kelly_fraction=0.20, tp_frac=0.5, sl_frac=0.3),

    # More conservative sizing
    dict(min_edge=0.05, kelly_fraction=0.25, max_kelly_fraction=0.10, tp_frac=0.5, sl_frac=0.3),
    dict(min_edge=0.05, kelly_fraction=0.25, max_kelly_fraction=0.15, tp_frac=0.5, sl_frac=0.3),

    # Aggressive: no TP/SL, big Kelly, high edge
    dict(min_edge=0.05, kelly_fraction=0.5, max_kelly_fraction=0.25, tp_frac=0.0, sl_frac=0.0),
]

print(f"POST-AUDIT SWEEP ({len(configs)} configs, realistic SL fills, running-peak DD)\n")
print(f"{'#':>2}  {'edge':>5}  {'kf':>4}  {'max':>4}  {'tp':>4}  {'sl':>4}  "
      f"{'trades':>6}  {'PnL':>9}  {'hit':>5}  {'final$':>8}  {'maxDD%':>6}  {'ret%':>7}")
print("-" * 95)

results = []

for i, cfg in enumerate(configs):
    strategy = EdgeThresholdStrategy(
        min_edge=cfg["min_edge"],
        min_tau=60.0,
        max_entries=1,
        sizing="kelly",
        kelly_fraction=cfg["kelly_fraction"],
        max_kelly_fraction=cfg["max_kelly_fraction"],
        tp_frac=cfg["tp_frac"],
        sl_frac=cfg["sl_frac"],
    )

    import io, contextlib
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        result = walk_forward_backtest(
            ASSET, START, END, strategy,
            initial_bankroll=BANKROLL,
        )

    final = result.bankroll_curve[-1] if len(result.bankroll_curve) > 0 else BANKROLL
    ret = (final / BANKROLL - 1) * 100
    n = result.n_trades
    pnl = result.total_pnl
    hr = result.hit_rate

    # Running peak drawdown (correct)
    if len(result.bankroll_curve) > 1:
        import numpy as np
        curve = np.array(result.bankroll_curve)
        running_peak = np.maximum.accumulate(curve)
        drawdowns = (running_peak - curve) / np.where(running_peak > 0, running_peak, 1)
        max_dd_pct = float(np.max(drawdowns)) * 100
    else:
        max_dd_pct = 0.0

    # Count TP/SL exits
    n_tp = sum(1 for m in result.markets for t in m.trades if t.exit_reason == "tp")
    n_sl = sum(1 for m in result.markets for t in m.trades if t.exit_reason == "sl")
    n_hold = n - n_tp - n_sl

    print(f"{i:2d}  {cfg['min_edge']:5.2f}  {cfg['kelly_fraction']:4.2f}  "
          f"{cfg['max_kelly_fraction']:4.2f}  {cfg['tp_frac']:4.2f}  {cfg['sl_frac']:4.2f}  "
          f"{n:6d}  {pnl:+9.2f}  {hr:5.1%}  {final:8.2f}  {max_dd_pct:5.1f}%  {ret:+6.1f}%  "
          f"[tp={n_tp} sl={n_sl} hold={n_hold}]")

    results.append({
        "idx": i, **cfg,
        "n_trades": n, "pnl": pnl, "hit_rate": hr,
        "final_bankroll": final, "max_dd_pct": max_dd_pct, "return_pct": ret,
        "n_tp": n_tp, "n_sl": n_sl, "n_hold": n_hold,
    })

print(f"\n{'='*95}")
best_ret = max(results, key=lambda r: r["return_pct"])
best_risk = max(results, key=lambda r: r["return_pct"] / max(r["max_dd_pct"], 1))
print(f"Best return:     #{best_ret['idx']} — {best_ret['return_pct']:+.1f}% "
      f"(DD={best_ret['max_dd_pct']:.1f}%)")
print(f"Best ret/DD:     #{best_risk['idx']} — {best_risk['return_pct']:+.1f}% "
      f"(DD={best_risk['max_dd_pct']:.1f}%, "
      f"ratio={best_risk['return_pct']/max(best_risk['max_dd_pct'],1):.2f})")
