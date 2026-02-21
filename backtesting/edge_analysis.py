"""Edge durability analysis: where does the model have real statistical advantage?

Runs walk-forward backtest at 1s latency, then cross-tabs every available
condition against hit rate and PnL to find where edge survives execution delay.

Usage:
    venv/bin/python3 backtesting/edge_analysis.py
"""

import sys
sys.path.insert(0, ".")

from datetime import date

import numpy as np

from backtesting.strategies.threshold import EdgeThresholdStrategy
from backtesting.walk_forward import walk_forward_backtest
from backtesting.core import BacktestResult


ASSET = "BTC"
START = date(2026, 1, 19)
END = date(2026, 2, 18)
BANKROLL = 100.0


def extract_trades(result: BacktestResult) -> dict[str, np.ndarray]:
    """Extract flat arrays of trade-level features + outcomes."""
    trades = []
    for m in result.markets:
        for t in m.trades:
            pnl = t.settle(m.outcome)
            trades.append({
                "pnl": pnl,
                "win": 1 if pnl > 0 else 0,
                "edge": t.edge,
                "tau": t.tau,
                "sigma_rv": t.sigma_rv,
                "sigma_rel": t.sigma_rel,
                "pm_spread": t.pm_spread,
                "hour_et": t.hour_et,
                "side": t.side,
                "size": t.size,
                "price": t.price,
                "model_p": t.model_price,
            })
    if not trades:
        return {}
    return {k: np.array([t[k] for t in trades]) for k in trades[0]}


def print_section(title: str):
    print(f"\n{'='*75}")
    print(f"  {title}")
    print(f"{'='*75}")


def analyze_dimension(
    name: str,
    values: np.ndarray,
    pnls: np.ndarray,
    wins: np.ndarray,
    sizes: np.ndarray,
    bins: list[float] | None = None,
    labels: list[str] | None = None,
    quantiles: int = 0,
):
    """Analyze hit rate and PnL by bucketed dimension."""
    if quantiles > 0:
        valid = ~np.isnan(values) & (values > 0)
        if valid.sum() < 10:
            print(f"  Insufficient data for {name}")
            return {}
        q_vals = np.percentile(values[valid], np.linspace(0, 100, quantiles + 1))
        q_vals = np.unique(q_vals)
        bins = q_vals.tolist()
        labels = [f"Q{i+1}" for i in range(len(bins) - 1)]

    if bins is None:
        return {}

    results = {}
    print(f"\n  {'Bucket':20s} {'n':>5s} {'hit%':>6s} {'avgPnL':>9s} {'totPnL':>10s} {'avgEdge':>8s} {'avgSize':>9s}")
    print(f"  {'-'*20} {'-'*5} {'-'*6} {'-'*9} {'-'*10} {'-'*8} {'-'*9}")

    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if i == len(bins) - 2:
            mask = (values >= lo) & (values <= hi)
        else:
            mask = (values >= lo) & (values < hi)
        n = mask.sum()
        if n < 3:
            continue

        hit = np.mean(wins[mask])
        avg_pnl = np.mean(pnls[mask])
        tot_pnl = np.sum(pnls[mask])
        avg_size = np.mean(sizes[mask])

        lbl = labels[i] if labels and i < len(labels) else f"[{lo:.3f},{hi:.3f})"
        # Highlight good buckets
        marker = " **" if hit > 0.40 and avg_pnl > 0 else ""
        print(f"  {lbl:20s} {n:5d} {hit:5.0%}  ${avg_pnl:+8.4f} ${tot_pnl:+9.2f} "
              f"  {np.mean(pnls[mask]/np.where(sizes[mask]>0,sizes[mask],1)):+7.3f} "
              f"${avg_size:8.2f}{marker}")
        results[lbl] = {"n": int(n), "hit": float(hit), "avg_pnl": float(avg_pnl),
                        "tot_pnl": float(tot_pnl)}

    return results


def cross_tab(
    name1: str, vals1: np.ndarray, bins1: list[float], labels1: list[str],
    name2: str, vals2: np.ndarray, bins2: list[float], labels2: list[str],
    pnls: np.ndarray, wins: np.ndarray,
):
    """2D cross-tab of hit rate."""
    print(f"\n  Cross-tab: {name1} x {name2} (hit rate, n trades)")
    header = f"  {'':15s}"
    for lbl in labels2:
        header += f" {lbl:>12s}"
    print(header)
    print(f"  {'-'*15}" + f" {'-'*12}" * len(labels2))

    for i in range(len(bins1) - 1):
        lo1, hi1 = bins1[i], bins1[i + 1]
        m1 = (vals1 >= lo1) & (vals1 < hi1) if i < len(bins1) - 2 else (vals1 >= lo1) & (vals1 <= hi1)

        row = f"  {labels1[i]:15s}"
        for j in range(len(bins2) - 1):
            lo2, hi2 = bins2[j], bins2[j + 1]
            m2 = (vals2 >= lo2) & (vals2 < hi2) if j < len(bins2) - 2 else (vals2 >= lo2) & (vals2 <= hi2)
            mask = m1 & m2
            n = mask.sum()
            if n < 3:
                row += f" {'--':>12s}"
            else:
                hit = np.mean(wins[mask])
                row += f" {hit:5.0%} n={n:3d} "

        print(row)


def main():
    # Run backtest at 1s latency
    print_section("EDGE DURABILITY ANALYSIS (1s latency)")

    strategy = EdgeThresholdStrategy(
        min_edge=0.05, min_tau=60.0, max_entries=1,
        sizing="kelly", kelly_fraction=0.5, max_kelly_fraction=0.15,
        tp_frac=0.5, sl_frac=0.2,
        latency_ms=1000,
    )

    result = walk_forward_backtest(
        ASSET, START, END, strategy, initial_bankroll=BANKROLL,
    )

    td = extract_trades(result)
    if not td:
        print("No trades!")
        return

    pnls = td["pnl"]
    wins = td["win"]
    edges = td["edge"]
    taus = td["tau"]
    sigma_rvs = td["sigma_rv"]
    sigma_rels = td["sigma_rel"]
    pm_spreads = td["pm_spread"]
    hours = td["hour_et"]
    sizes = td["size"]

    n = len(pnls)
    print(f"\n  Total trades: {n}")
    print(f"  Hit rate: {np.mean(wins):.1%}")
    print(f"  Total PnL: ${np.sum(pnls):+.2f}")
    print(f"  Avg PnL/trade: ${np.mean(pnls):+.4f}")

    # ================================================================
    # 1. BY TAU (time to expiry)
    # ================================================================
    print_section("1. EDGE BY TAU (time to expiry at entry)")
    tau_bins = [60, 300, 600, 900, 1500, 2100, 2700, 3600]
    tau_labels = ["1-5min", "5-10min", "10-15min", "15-25min", "25-35min", "35-45min", "45-60min"]
    analyze_dimension("tau", taus, pnls, wins, sizes, bins=tau_bins, labels=tau_labels)

    # ================================================================
    # 2. BY SIGMA_RV (realized vol regime)
    # ================================================================
    print_section("2. EDGE BY SIGMA_RV (realized vol quartiles)")
    analyze_dimension("sigma_rv", sigma_rvs, pnls, wins, sizes, quantiles=4)

    # ================================================================
    # 3. BY SIGMA_REL (relative vol = sigma_rv / sigma_tod)
    # ================================================================
    print_section("3. EDGE BY SIGMA_REL (relative vol = rv/seasonal)")
    srel_bins = [0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 10.0]
    srel_labels = ["<0.5 (calm)", "0.5-0.8", "0.8-1.0 (normal)", "1.0-1.2",
                   "1.2-1.5", "1.5-2.0 (elevated)", ">2.0 (spike)"]
    analyze_dimension("sigma_rel", sigma_rels, pnls, wins, sizes, bins=srel_bins, labels=srel_labels)

    # ================================================================
    # 4. BY PM SPREAD (Polymarket bid-ask spread)
    # ================================================================
    print_section("4. EDGE BY PM SPREAD")
    analyze_dimension("pm_spread", pm_spreads, pnls, wins, sizes, quantiles=4)

    # ================================================================
    # 5. BY HOUR (ET)
    # ================================================================
    print_section("5. EDGE BY HOUR (ET)")
    hour_bins = list(range(25))
    hour_labels = [f"{h:02d}:00" for h in range(24)]
    analyze_dimension("hour_et", hours, pnls, wins, sizes, bins=hour_bins, labels=hour_labels)

    # ================================================================
    # 6. BY EDGE SIZE (signal strength)
    # ================================================================
    print_section("6. EDGE BY SIGNAL STRENGTH (edge at fill)")
    edge_bins = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.15, 1.0]
    edge_labels = ["0-1%", "1-2%", "2-3%", "3-4%", "4-5%", "5-8%", "8-10%", "10-15%", ">15%"]
    analyze_dimension("edge", edges, pnls, wins, sizes, bins=edge_bins, labels=edge_labels)

    # ================================================================
    # CROSS-TABS
    # ================================================================
    print_section("CROSS-TABS")

    # Tau x Vol regime
    srv_q = np.percentile(sigma_rvs[sigma_rvs > 0], [0, 25, 50, 75, 100])
    srv_q = np.unique(srv_q).tolist()
    srv_labels = [f"σ_Q{i+1}" for i in range(len(srv_q) - 1)]

    tau_ct_bins = [60, 900, 1800, 3600]
    tau_ct_labels = ["<15min", "15-30min", "30-60min"]
    cross_tab("tau", taus, tau_ct_bins, tau_ct_labels,
              "sigma_rv", sigma_rvs, srv_q, srv_labels, pnls, wins)

    # Tau x Spread
    spr_q = np.percentile(pm_spreads[pm_spreads > 0], [0, 33, 67, 100])
    spr_q = np.unique(spr_q).tolist()
    spr_labels = ["tight", "medium", "wide"]
    cross_tab("tau", taus, tau_ct_bins, tau_ct_labels,
              "pm_spread", pm_spreads, spr_q, spr_labels, pnls, wins)

    # Hour group x Vol regime
    hour_group_bins = [0, 6, 12, 18, 24]
    hour_group_labels = ["night(0-6)", "morning(6-12)", "afternoon(12-18)", "evening(18-24)"]
    cross_tab("hour_group", hours, hour_group_bins, hour_group_labels,
              "sigma_rv", sigma_rvs, srv_q, srv_labels, pnls, wins)

    # ================================================================
    # OPTIMAL FILTER RECOMMENDATION
    # ================================================================
    print_section("FILTER RECOMMENDATIONS")

    # Find best tau cutoff
    print("\n  Scanning tau minimum cutoffs:")
    for tau_min in [300, 600, 900, 1200, 1800]:
        mask = taus >= tau_min
        n_m = mask.sum()
        if n_m < 10:
            continue
        hit = np.mean(wins[mask])
        avg = np.mean(pnls[mask])
        tot = np.sum(pnls[mask])
        print(f"    tau >= {tau_min:5.0f}s: n={n_m:4d}  hit={hit:5.1%}  "
              f"avg=${avg:+.4f}  total=${tot:+.2f}")

    # Find best spread cutoff
    print("\n  Scanning PM spread maximum cutoffs:")
    for spr_max in [0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
        mask = pm_spreads <= spr_max
        n_m = mask.sum()
        if n_m < 10:
            continue
        hit = np.mean(wins[mask])
        avg = np.mean(pnls[mask])
        tot = np.sum(pnls[mask])
        print(f"    spread <= {spr_max:.2f}: n={n_m:4d}  hit={hit:5.1%}  "
              f"avg=${avg:+.4f}  total=${tot:+.2f}")

    # Find best sigma_rel range
    print("\n  Scanning sigma_rel range:")
    for lo, hi in [(0, 0.8), (0, 1.0), (0, 1.2), (0.5, 1.5), (0.8, 2.0), (1.0, 10.0)]:
        mask = (sigma_rels >= lo) & (sigma_rels <= hi)
        n_m = mask.sum()
        if n_m < 10:
            continue
        hit = np.mean(wins[mask])
        avg = np.mean(pnls[mask])
        tot = np.sum(pnls[mask])
        print(f"    σ_rel ∈ [{lo:.1f}, {hi:.1f}]: n={n_m:4d}  hit={hit:5.1%}  "
              f"avg=${avg:+.4f}  total=${tot:+.2f}")

    # Combined filter scan
    print("\n  Combined filter scan (best combos):")
    best_combo = None
    best_ratio = -999
    for tau_min in [600, 900, 1200]:
        for spr_max in [0.04, 0.06, 0.08]:
            for srel_max in [1.2, 1.5, 2.0, 10.0]:
                mask = (taus >= tau_min) & (pm_spreads <= spr_max) & (sigma_rels <= srel_max)
                n_m = mask.sum()
                if n_m < 20:
                    continue
                hit = np.mean(wins[mask])
                avg = np.mean(pnls[mask])
                tot = np.sum(pnls[mask])
                if n_m > 0 and np.std(pnls[mask]) > 0:
                    sharpe = avg / np.std(pnls[mask])
                else:
                    sharpe = 0
                # Score: Sharpe-like but penalize small n
                score = sharpe * np.sqrt(n_m)
                if score > best_ratio:
                    best_ratio = score
                    best_combo = (tau_min, spr_max, srel_max, n_m, hit, avg, tot, sharpe)

    if best_combo:
        tau_min, spr_max, srel_max, n_m, hit, avg, tot, sharpe = best_combo
        print(f"\n  ** BEST COMBO: tau>={tau_min}s, spread<={spr_max}, sigma_rel<={srel_max}")
        print(f"     n={n_m}, hit={hit:.1%}, avg=${avg:+.4f}, total=${tot:+.2f}, sharpe={sharpe:.3f}")

    # All combos with positive total PnL
    print("\n  All profitable combos (total PnL > 0, n >= 30):")
    combos = []
    for tau_min in [300, 600, 900, 1200]:
        for spr_max in [0.04, 0.06, 0.08, 0.10]:
            for srel_max in [1.0, 1.2, 1.5, 2.0, 10.0]:
                mask = (taus >= tau_min) & (pm_spreads <= spr_max) & (sigma_rels <= srel_max)
                n_m = mask.sum()
                if n_m < 30:
                    continue
                tot = np.sum(pnls[mask])
                if tot <= 0:
                    continue
                hit = np.mean(wins[mask])
                avg = np.mean(pnls[mask])
                combos.append((tau_min, spr_max, srel_max, n_m, hit, avg, tot))

    combos.sort(key=lambda x: x[-1], reverse=True)
    for c in combos[:15]:
        tau_min, spr_max, srel_max, n_m, hit, avg, tot = c
        print(f"    tau>={tau_min:5.0f}  spread<={spr_max:.2f}  σ_rel<={srel_max:4.1f}  "
              f"n={n_m:4d}  hit={hit:5.1%}  avg=${avg:+.4f}  total=${tot:+.2f}")


if __name__ == "__main__":
    main()
