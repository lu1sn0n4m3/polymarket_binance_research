#!/usr/bin/env python3
"""
Polymarket vs Pricer Comparison

Uses the EXACT same calibration dataset (60s samples) as training.
Matches Polymarket midprices to those timestamps via load_session().
"""

from datetime import date, timedelta
from pathlib import Path

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

from src.data import load_session

# =============================================================================
# CONFIGURATION
# =============================================================================

ASSET = "BTC"
PRICER_OUTPUT = Path("pricer_calibration/output")
SCRIPT_OUTPUT = Path("outputs")


# =============================================================================
# Model
# =============================================================================

def apply_model(S, K, tau, sigma_tod, sigma_rv, params):
    """Apply fitted model to get P(Up)."""
    a0, a1, beta = params["a0"], params["a1"], params["beta"]
    probit = params["probit_layer"]

    sigma_rel = sigma_rv / np.maximum(sigma_tod, 1e-12)
    tau_min = tau / 60.0
    a_tau = a0 + a1 * np.sqrt(np.maximum(tau_min, 0))
    sigma_eff = a_tau * sigma_tod * np.power(np.maximum(sigma_rel, 1e-12), beta)

    sqrt_tau = np.sqrt(np.maximum(tau, 1e-6))
    z = (np.log(K / S) + 0.5 * sigma_eff**2 * tau) / (sigma_eff * sqrt_tau)

    score = -z
    log_tau = np.log(np.maximum(tau, 1.0))
    log_sr = np.log(np.maximum(sigma_rel, 1e-6))
    zp = (probit["b0"] + probit["b1"] * score + probit["b2"] * log_tau
          + probit["b3"] * log_sr + probit["b4"] * log_sr**2)
    return np.clip(norm.cdf(zp), 1e-9, 1 - 1e-9)


# =============================================================================
# Infrastructure
# =============================================================================

def load_seasonal():
    cache_path = PRICER_OUTPUT / "seasonal_curve.json"
    with open(cache_path) as f:
        data = json.load(f)
    return np.array(data["sigma_tod"]), data["bucket_minutes"]


def load_ewma_rv():
    rv_df = pd.read_parquet(PRICER_OUTPUT / "sigma_rv_cache.parquet")
    return rv_df["ts"].values, rv_df["sigma_rv"].values


def sigma_tod_at_ms(ts_ms, sigma_tod_curve, bucket_minutes=5):
    total_sec = (ts_ms // 1000) % 86400
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    buckets = ((hours * 60 + minutes) // bucket_minutes).astype(int)
    buckets = np.clip(buckets, 0, len(sigma_tod_curve) - 1)
    return sigma_tod_curve[buckets]


def lookup_sigma_rv_vec(grid_ts, ts_rv, sigma_rv_full):
    idx = np.searchsorted(ts_rv, grid_ts)
    idx = np.clip(idx, 0, len(ts_rv) - 1)
    idx_prev = np.clip(idx - 1, 0, len(ts_rv) - 1)
    dist_r = np.abs(ts_rv[idx] - grid_ts)
    dist_l = np.abs(ts_rv[idx_prev] - grid_ts)
    best = np.where(dist_l < dist_r, idx_prev, idx)
    return sigma_rv_full[best]


# =============================================================================
# Metrics
# =============================================================================

def _log_loss(y, p):
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def brier_score(y, p):
    return np.mean((p - y) ** 2)


def clustered_se(residuals, cluster_ids):
    unique_ids = np.unique(cluster_ids)
    n_clusters = len(unique_ids)
    if n_clusters < 2:
        return np.std(residuals) / np.sqrt(len(residuals))
    cluster_means = np.array([np.mean(residuals[cluster_ids == cid]) for cid in unique_ids])
    return np.std(cluster_means, ddof=1) / np.sqrt(n_clusters)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Polymarket vs Pricer — exact training samples")
    print("=" * 70)

    # Load calibration dataset (exact training samples)
    print("\nLoading calibration dataset...")
    ds = pd.read_parquet(PRICER_OUTPUT / "calibration_dataset.parquet")
    print(f"  {len(ds):,} samples, {ds['market_id'].nunique()} sessions")

    # Load model params
    with open(PRICER_OUTPUT / "params_final.json") as f:
        params = json.load(f)
    print(f"  Model: {params['model']}")
    print(f"  Train LL: {params['train_log_loss_final']:.4f}")

    # Load infrastructure
    print("\nLoading σ_tod and EWMA σ_rv...")
    sigma_tod_curve, tod_bucket_min = load_seasonal()
    ts_rv, sigma_rv_full = load_ewma_rv()

    # Map σ_tod and σ_rv to calibration timestamps
    t_calib = ds["t"].values
    sigma_tod = sigma_tod_at_ms(t_calib, sigma_tod_curve, tod_bucket_min)
    sigma_rv = lookup_sigma_rv_vec(t_calib, ts_rv, sigma_rv_full)

    # Compute model probabilities
    model_p = apply_model(
        ds["S"].values, ds["K"].values, ds["tau"].values,
        sigma_tod, sigma_rv, params,
    )
    ds["model_p"] = model_p

    # Sanity check: model LL should match training
    y = ds["y"].values.astype(float)
    print(f"  Model LL on this data: {_log_loss(y, model_p):.4f} (train was {params['train_log_loss_final']:.4f})")

    # -------------------------------------------------------------------------
    # Load Polymarket midprices for each session
    # -------------------------------------------------------------------------
    print("\nLoading Polymarket data per session...")

    market_ids = ds["market_id"].unique()
    pm_lookup = {}  # market_id -> (ts_recv array, mid array)

    loaded, skipped = 0, 0
    for mid in market_ids:
        # market_id format: "BTC_20260119_00" where 00 is UTC hour
        parts = mid.split("_")
        d_utc = date(int(parts[1][:4]), int(parts[1][4:6]), int(parts[1][6:8]))
        hour_utc = int(parts[2])

        # Convert UTC hour to ET (ET = UTC - 5 in winter)
        hour_et = (hour_utc - 5) % 24
        d = d_utc
        if hour_utc < 5:
            d = d_utc - timedelta(days=1)

        try:
            session = load_session(ASSET, d, hour_et=hour_et, lookback_hours=0)
            if session.outcome is None:
                skipped += 1
                continue

            pm_bbo = session.polymarket_bbo
            if pm_bbo is None or pm_bbo.empty:
                skipped += 1
                continue

            pm_bbo = pm_bbo.copy()
            pm_bbo["mid"] = (pm_bbo["bid_px"] + pm_bbo["ask_px"]) / 2
            if session.token_is_up is False:
                pm_bbo["mid"] = 1.0 - pm_bbo["mid"]

            pm_lookup[mid] = (pm_bbo["ts_recv"].values, pm_bbo["mid"].values)
            loaded += 1
        except Exception:
            skipped += 1

    print(f"  Loaded PM data for {loaded} sessions, skipped {skipped}")

    # -------------------------------------------------------------------------
    # Match PM midprices to calibration timestamps
    # -------------------------------------------------------------------------
    print("\nMatching PM midprices to calibration samples...")

    pm_mid_arr = np.full(len(ds), np.nan)

    for mid, (pm_ts, pm_mids) in pm_lookup.items():
        mask = ds["market_id"] == mid
        t_samples = ds.loc[mask, "t"].values

        # Vectorized nearest-neighbor
        idx = np.searchsorted(pm_ts, t_samples)
        idx = np.clip(idx, 0, len(pm_ts) - 1)
        idx_prev = np.clip(idx - 1, 0, len(pm_ts) - 1)
        dist_r = np.abs(pm_ts[idx] - t_samples)
        dist_l = np.abs(pm_ts[idx_prev] - t_samples)
        best = np.where(dist_l < dist_r, idx_prev, idx)

        # Only keep matches within 30s
        dist_best = np.abs(pm_ts[best] - t_samples)
        matched = pm_mids[best].copy()
        matched[dist_best > 30000] = np.nan

        pm_mid_arr[mask.values] = matched

    ds["pm_mid"] = pm_mid_arr

    n_before = len(ds)
    ds = ds.dropna(subset=["pm_mid"]).copy()
    print(f"  Matched {len(ds)} / {n_before} samples ({len(ds)/n_before:.0%})")

    if ds.empty:
        print("No matched data!")
        return

    # -------------------------------------------------------------------------
    # Comparison
    # -------------------------------------------------------------------------
    y = ds["y"].values.astype(float)
    pm_p = ds["pm_mid"].values
    mod_p = ds["model_p"].values
    sids = ds["market_id"].values
    tau_min = ds["tau"].values / 60.0

    ll_pm = _log_loss(y, pm_p)
    ll_mod = _log_loss(y, mod_p)
    ll_const = _log_loss(y, np.full_like(y, np.mean(y)))
    bs_pm = brier_score(y, pm_p)
    bs_mod = brier_score(y, mod_p)
    imp_pm = (ll_const - ll_pm) / ll_const * 100
    imp_mod = (ll_const - ll_mod) / ll_const * 100
    n_sessions = ds["market_id"].nunique()

    print(f"\n{'='*70}")
    print("OVERALL COMPARISON")
    print(f"{'='*70}")
    print(f"  {n_sessions} sessions, {len(ds):,} observations (60s training samples)")
    print()
    print(f"  {'Metric':<25s} {'Polymarket':>12s} {'Model':>12s} {'Winner':>10s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'Log Loss':<25s} {ll_pm:>12.4f} {ll_mod:>12.4f} {'Model' if ll_mod < ll_pm else 'PM':>10s}")
    print(f"  {'Brier Score':<25s} {bs_pm:>12.4f} {bs_mod:>12.4f} {'Model' if bs_mod < bs_pm else 'PM':>10s}")
    print(f"  {'vs Constant (%)':<25s} {imp_pm:>+12.1f}% {imp_mod:>+12.1f}%")
    print()

    e_pm = y - pm_p
    e_mod = y - mod_p
    se_pm = clustered_se(e_pm, sids)
    se_mod = clustered_se(e_mod, sids)
    print(f"  Mean residual (PM):    {np.mean(e_pm):+.4f} ± {se_pm:.4f} (clustered SE)")
    print(f"  Mean residual (Model): {np.mean(e_mod):+.4f} ± {se_mod:.4f} (clustered SE)")
    print()

    # By τ-bucket
    tau_edges = [0, 5, 15, 30, 60]
    tau_labels = ["0-5", "5-15", "15-30", "30-60"]

    print("BY TIME-TO-EXPIRY (minutes):")
    print(f"  {'τ':<8s} {'LL(PM)':>8s} {'LL(Mod)':>8s} {'Winner':>8s} {'n':>8s}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    tau_data = []
    for i in range(len(tau_edges) - 1):
        mask = (tau_min >= tau_edges[i]) & (tau_min < tau_edges[i + 1])
        n = mask.sum()
        if n > 0:
            ll_pm_b = _log_loss(y[mask], pm_p[mask])
            ll_mod_b = _log_loss(y[mask], mod_p[mask])
            winner = "Model" if ll_mod_b < ll_pm_b else "PM"
            print(f"  {tau_labels[i]:<8s} {ll_pm_b:>8.4f} {ll_mod_b:>8.4f} {winner:>8s} {n:>8d}")
            tau_data.append({"label": tau_labels[i], "ll_pm": ll_pm_b, "ll_mod": ll_mod_b, "n": n})
    print()

    # Calibration buckets
    prob_buckets = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pm_calib = _bucket_calibration(pm_p, y, prob_buckets)
    mod_calib = _bucket_calibration(mod_p, y, prob_buckets)

    print("CALIBRATION BY PROBABILITY BUCKET:")
    print(f"  {'Bucket':<10s} {'PM pred':>8s} {'PM hit':>8s} {'PM err':>8s}   "
          f"{'Mod pred':>8s} {'Mod hit':>8s} {'Mod err':>8s}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8}   {'-'*8} {'-'*8} {'-'*8}")

    pm_map = {r["label"]: r for r in pm_calib}
    mod_map = {r["label"]: r for r in mod_calib}
    all_bucket_labels = sorted(set(pm_map) | set(mod_map))
    for label in all_bucket_labels:
        pm_r = pm_map.get(label)
        mod_r = mod_map.get(label)
        pm_str = f"{pm_r['pred']:.1%}  {pm_r['hit']:.1%} {pm_r['err']:+.1%}" if pm_r else "   -      -      -  "
        mod_str = f"{mod_r['pred']:.1%}  {mod_r['hit']:.1%} {mod_r['err']:+.1%}" if mod_r else "   -      -      -  "
        print(f"  {label:<10s} {pm_str}   {mod_str}")
    print()

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    print("Creating visualizations...")
    SCRIPT_OUTPUT.mkdir(exist_ok=True)

    fig = create_comparison_chart(ds, pm_calib, mod_calib, tau_data, params)
    html_path = SCRIPT_OUTPUT / f"polymarket_vs_pricer_{ASSET}.html"
    fig.write_html(str(html_path))
    print(f"  Saved to: {html_path}")

    try:
        import webbrowser
        webbrowser.open(f"file://{html_path.absolute()}")
    except Exception:
        pass

    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}")
    return ds


def _bucket_calibration(p, y, buckets):
    results = []
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i + 1]
        mask = (p >= lo) & (p <= hi) if i == 0 else (p > lo) & (p <= hi)
        n = mask.sum()
        if n == 0:
            continue
        results.append({
            "label": f"{lo:.0%}-{hi:.0%}",
            "lo": lo, "hi": hi,
            "pred": p[mask].mean(),
            "hit": y[mask].mean(),
            "err": y[mask].mean() - p[mask].mean(),
            "n": n,
        })
    return results


def create_comparison_chart(df, pm_calib, mod_calib, tau_data, params):
    y = df["y"].astype(float).values
    pm_p = df["pm_mid"].values
    mod_p = df["model_p"].values

    ll_pm = _log_loss(y, pm_p)
    ll_mod = _log_loss(y, mod_p)
    ll_const = _log_loss(y, np.full_like(y, np.mean(y)))
    bs_pm = brier_score(y, pm_p)
    bs_mod = brier_score(y, mod_p)
    n_sessions = df["market_id"].nunique()

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "Calibration: Polymarket",
            "Calibration: Pricer Model",
            "Log Loss by Time-to-Expiry",
            "Prediction Distribution",
            "PM vs Model Scatter",
            "Calibration Error by Bucket",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # Plot 1: PM calibration
    if pm_calib:
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
            line=dict(color="gray", dash="dash"), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[r["pred"] for r in pm_calib], y=[r["hit"] for r in pm_calib],
            mode="markers+lines", marker=dict(size=8, color="#2196F3"),
            line=dict(color="#2196F3", width=2),
            name=f"PM (LL={ll_pm:.4f})",
        ), row=1, col=1)
    fig.update_xaxes(title_text="Predicted P(Up)", range=[0,1], row=1, col=1)
    fig.update_yaxes(title_text="Actual P(Up)", range=[0,1], row=1, col=1)

    # Plot 2: Model calibration
    if mod_calib:
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
            line=dict(color="gray", dash="dash"), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=[r["pred"] for r in mod_calib], y=[r["hit"] for r in mod_calib],
            mode="markers+lines", marker=dict(size=8, color="#FF5722"),
            line=dict(color="#FF5722", width=2),
            name=f"Model (LL={ll_mod:.4f})",
        ), row=1, col=2)
    fig.update_xaxes(title_text="Predicted P(Up)", range=[0,1], row=1, col=2)
    fig.update_yaxes(title_text="Actual P(Up)", range=[0,1], row=1, col=2)

    # Plot 3: LL by τ
    if tau_data:
        labels = [d["label"] for d in tau_data]
        fig.add_trace(go.Bar(x=labels, y=[d["ll_pm"] for d in tau_data],
            name="PM", marker_color="#2196F3", opacity=0.7), row=1, col=3)
        fig.add_trace(go.Bar(x=labels, y=[d["ll_mod"] for d in tau_data],
            name="Model", marker_color="#FF5722", opacity=0.7), row=1, col=3)
        fig.add_hline(y=ll_const, line_dash="dot", line_color="gray",
            annotation_text=f"Constant={ll_const:.3f}", row=1, col=3)
    fig.update_xaxes(title_text="τ (minutes)", row=1, col=3)
    fig.update_yaxes(title_text="Log Loss", row=1, col=3)
    fig.update_layout(barmode="group")

    # Plot 4: Distribution
    fig.add_trace(go.Histogram(x=pm_p[y==1], name="PM|y=1", marker_color="#2196F3",
        opacity=0.5, nbinsx=30, histnorm="probability density"), row=2, col=1)
    fig.add_trace(go.Histogram(x=pm_p[y==0], name="PM|y=0", marker_color="#90CAF9",
        opacity=0.5, nbinsx=30, histnorm="probability density"), row=2, col=1)
    fig.add_trace(go.Histogram(x=mod_p[y==1], name="Mod|y=1", marker_color="#FF5722",
        opacity=0.5, nbinsx=30, histnorm="probability density"), row=2, col=1)
    fig.add_trace(go.Histogram(x=mod_p[y==0], name="Mod|y=0", marker_color="#FFAB91",
        opacity=0.5, nbinsx=30, histnorm="probability density"), row=2, col=1)
    fig.update_xaxes(title_text="Predicted Probability", row=2, col=1)
    fig.update_yaxes(title_text="Density", row=2, col=1)

    # Plot 5: Scatter
    fig.add_trace(go.Scatter(
        x=pm_p, y=mod_p, mode="markers",
        marker=dict(size=3, opacity=0.3,
            color=y, colorscale=[[0,"#F44336"],[1,"#4CAF50"]]),
        showlegend=False,
        hovertemplate="PM: %{x:.2f}<br>Model: %{y:.2f}<extra></extra>",
    ), row=2, col=2)
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
        line=dict(color="gray", dash="dash"), showlegend=False), row=2, col=2)
    fig.update_xaxes(title_text="Polymarket P(Up)", range=[0,1], row=2, col=2)
    fig.update_yaxes(title_text="Model P(Up)", range=[0,1], row=2, col=2)

    # Plot 6: Calibration error
    if pm_calib and mod_calib:
        pm_err_map = {r["label"]: r["err"] for r in pm_calib}
        mod_err_map = {r["label"]: r["err"] for r in mod_calib}
        all_labels = sorted(set(pm_err_map) | set(mod_err_map))
        fig.add_trace(go.Bar(x=all_labels,
            y=[pm_err_map.get(l,0)*100 for l in all_labels],
            name="PM err", marker_color="#2196F3", opacity=0.7), row=2, col=3)
        fig.add_trace(go.Bar(x=all_labels,
            y=[mod_err_map.get(l,0)*100 for l in all_labels],
            name="Mod err", marker_color="#FF5722", opacity=0.7), row=2, col=3)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=3)
    fig.update_xaxes(title_text="Probability Bucket", row=2, col=3)
    fig.update_yaxes(title_text="Calibration Error (pp)", row=2, col=3)

    fig.update_layout(
        title=dict(
            text=(f"Polymarket vs Pricer | {n_sessions} sessions | "
                  f"LL: PM={ll_pm:.4f} Model={ll_mod:.4f} | "
                  f"Brier: PM={bs_pm:.4f} Model={bs_mod:.4f}"),
            font=dict(size=14)),
        height=800, width=1400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


if __name__ == "__main__":
    main()
