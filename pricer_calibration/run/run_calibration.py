#!/usr/bin/env python
"""One-click calibration: rebuild dataset + fit model.

Usage:
    python -m pricer_calibration.run.run_calibration [--config path/to/config.yaml]

This script:
1. Builds calibration dataset (loads BBO, builds grid, samples labels)
2. Computes σ_rv (EWMA) features
3. Fits model parameters (a0, a1, β)
4. Generates diagnostic plots
5. Saves results to output/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

from pricer_calibration.config import load_config
from pricer_calibration.run.build_dataset import build_dataset
from pricer_calibration.features.seasonal_vol import compute_seasonal_vol_ticktime


# ============================================================================
# Compute σ_rv (EWMA)
# ============================================================================

def compute_rv_ewma(bbo: pd.DataFrame, half_life_sec: float = 300.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute EWMA realized volatility using sum-of-squares / sum-of-time.

    Uses "sum(dx^2) / sum(dt)" estimator with EWMA weighting:
        ewma_sum_sq = decay * ewma_sum_sq_prev + dx^2
        ewma_sum_dt = decay * ewma_sum_dt_prev + dt
        var_per_sec = ewma_sum_sq / ewma_sum_dt
        sigma = sqrt(var_per_sec)

    This is robust to microstructure noise - we do NOT normalize individual
    returns by sqrt(dt), which can blow up when dt is tiny.

    Args:
        bbo: DataFrame with columns [ts_event, mid].
        half_life_sec: EWMA half-life in seconds.

    Returns:
        (ts_tick, sigma_rv): timestamps and σ_rv values in per-√sec units.
    """
    ts = bbo["ts_event"].values
    mid = bbo["mid"].values

    changed = np.zeros(len(mid), dtype=bool)
    changed[0] = True
    changed[1:] = mid[1:] != mid[:-1]

    ts_changed = ts[changed]
    mid_changed = mid[changed]

    if len(mid_changed) < 2:
        return np.array([]), np.array([])

    log_mid = np.log(mid_changed)
    dx = np.diff(log_mid)  # log return (NOT normalized)
    dt_ms = np.diff(ts_changed)
    dt_sec = dt_ms / 1000.0
    ts_tick = ts_changed[1:]

    # Small dt floor to avoid issues with duplicate timestamps
    dt_floor = 0.0001  # 0.1ms
    valid = dt_sec > dt_floor
    dx = dx[valid]
    dt_sec = dt_sec[valid]
    ts_tick = ts_tick[valid]

    n = len(dx)
    if n == 0:
        return np.array([]), np.array([])

    sigma_rv = np.zeros(n)
    dx_sq = dx ** 2

    # Initialize with first few ticks
    init_window = min(100, n // 10)
    if init_window > 10:
        ewma_sum_sq = np.sum(dx_sq[:init_window])
        ewma_sum_dt = np.sum(dt_sec[:init_window])
    else:
        # Reasonable default: assume ~1e-8 variance per second (approx 0.3% daily vol)
        ewma_sum_sq = 1e-8 * half_life_sec
        ewma_sum_dt = half_life_sec

    for i in range(n):
        # Decay factor based on time elapsed
        decay = np.exp(-np.log(2) * dt_sec[i] / half_life_sec)

        # Update EWMA of sum(dx^2) and sum(dt)
        ewma_sum_sq = decay * ewma_sum_sq + dx_sq[i]
        ewma_sum_dt = decay * ewma_sum_dt + dt_sec[i]

        # Variance per second = sum(dx^2) / sum(dt)
        var_per_sec = ewma_sum_sq / max(ewma_sum_dt, 1e-6)
        sigma_rv[i] = np.sqrt(max(var_per_sec, 1e-12))

    return ts_tick, sigma_rv


# ============================================================================
# Calibrate Model
# ============================================================================

def _log_loss(y, p):
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def _neg_ll(params, S, K, tau, sigma_tod, sigma_rel, y):
    a0, a1, beta = params
    tau_min = tau / 60.0
    a_tau = a0 + a1 * np.sqrt(tau_min)
    sigma_eff = a_tau * sigma_tod * np.power(sigma_rel, beta)
    sqrt_tau = np.sqrt(np.maximum(tau, 1e-6))
    x = (np.log(K / S) + 0.5 * sigma_eff**2 * tau) / (sigma_eff * sqrt_tau)
    p = 1.0 - norm.cdf(x)
    return _log_loss(y, p)


def _compute_z_score(S, K, tau, sigma_eff):
    """Compute raw z-score: z = (ln(K/S) + ½σ²τ) / (σ√τ)."""
    sqrt_tau = np.sqrt(np.maximum(tau, 1e-6))
    return (np.log(K / S) + 0.5 * sigma_eff**2 * tau) / (sigma_eff * sqrt_tau)


def _smooth_probit_neg_ll(params, score, log_tau, log_sigma_rel, y, lambda_l2=0.001):
    """Negative log-likelihood for smooth probit with quadratic vol + L2 ridge.

    z' = b0 + b1·score + b2·log(τ) + b3·log(σ_rel) + b4·(log σ_rel)²
    p  = Φ(z')
    """
    b0, b1, b2, b3, b4 = params
    zp = (b0 + b1 * score + b2 * log_tau
          + b3 * log_sigma_rel + b4 * log_sigma_rel**2)
    p = norm.cdf(zp)
    ll = _log_loss(y, p)
    # L2 ridge on auxiliary coefficients
    penalty = lambda_l2 * (b0**2 + b2**2 + b3**2 + b4**2)
    return ll + penalty


# τ-bucket definitions (kept for diagnostics only)
TAU_BUCKET_EDGES = [0, 5, 15, 30, 60]  # in minutes
TAU_BUCKET_LABELS = ["0-5", "5-15", "15-30", "30-60"]
VOL_QUARTILE_LABELS = ["Q1", "Q2", "Q3", "Q4"]


def _clustered_se(residuals, cluster_ids):
    """Compute cluster-robust SE for mean(residuals).

    All samples within a cluster (hourly candle) share the same outcome y,
    so naive SE underestimates uncertainty by ~√(cluster_size).

    SE = std(cluster_means) / √(n_clusters)
    """
    unique_ids = np.unique(cluster_ids)
    n_clusters = len(unique_ids)
    if n_clusters < 2:
        return np.std(residuals) / np.sqrt(len(residuals))
    cluster_means = np.array([np.mean(residuals[cluster_ids == cid]) for cid in unique_ids])
    return np.std(cluster_means, ddof=1) / np.sqrt(n_clusters)


def _fit_probit_layer(z, tau, y, sigma_rel=None, lambda_l2=0.001):
    """Fit smooth probit with quadratic vol + L2 ridge.

    z' = b0 + b1·score + b2·log(τ) + b3·log(σ_rel) + b4·(log σ_rel)²
    p  = Φ(z')

    score = -z so that Φ(score) ≈ p_raw.

    Returns:
        dict with keys: "b0", "b1", "b2", "b3", "b4"
    """
    score = -z
    tau_sec = np.maximum(tau, 1.0)
    log_tau = np.log(tau_sec)

    if sigma_rel is not None:
        log_sr = np.log(np.maximum(sigma_rel, 1e-6))
    else:
        log_sr = np.zeros_like(z)

    res = minimize(
        _smooth_probit_neg_ll,
        x0=[0.0, 1.0, 0.0, 0.0, 0.0],
        args=(score, log_tau, log_sr, y, lambda_l2),
        method="L-BFGS-B",
        bounds=[(-2.0, 2.0), (0.3, 2.0), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)],
    )
    b0, b1, b2, b3, b4 = res.x

    return {
        "b0": float(b0),
        "b1": float(b1),
        "b2": float(b2),
        "b3": float(b3),
        "b4": float(b4),
    }


def _apply_probit_layer(z, tau, probit_params, sigma_rel=None):
    """Apply probit calibration to get p_cal.

    z' = b0 + b1·score + b2·log(τ) + b3·log(σ_rel) + b4·(log σ_rel)²
    p  = Φ(z')
    """
    score = -z
    tau_sec = np.maximum(tau, 1.0)
    log_tau = np.log(tau_sec)

    b0 = probit_params["b0"]
    b1 = probit_params["b1"]
    b2 = probit_params["b2"]
    b3 = probit_params["b3"]
    b4 = probit_params.get("b4", 0.0)

    zp = b0 + b1 * score + b2 * log_tau
    if sigma_rel is not None:
        log_sr = np.log(np.maximum(sigma_rel, 1e-6))
        zp = zp + b3 * log_sr + b4 * log_sr**2

    p_cal = norm.cdf(zp)
    return np.clip(p_cal, 1e-9, 1.0 - 1e-9)


def calibrate_model(dataset, bbo, seasonal, cfg):
    """Fit model parameters via maximum likelihood.

    Two-stage calibration:
      Stage 1: Fit (a0, a1, β) for σ_eff = a(τ) * σ_tod^(1-β) * σ_rv^β
      Stage 2: Per-τ probit layer: p_cal = Φ(α(τ) + γ(τ) * score)
    """
    output_dir = Path(cfg.output_dir)

    # Compute σ_rv
    print("Computing σ_rv (EWMA H=5min)...")
    ts_rv, sigma_rv_full = compute_rv_ewma(bbo, half_life_sec=300.0)

    # Map σ_tod and σ_rv to calibration samples (vectorized)
    t_calib = dataset["t"].values

    # σ_tod: map via time-of-day bucket
    total_seconds = (t_calib // 1000) % 86400
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    buckets = ((hours * 60 + minutes) // cfg.tod_bucket_minutes).astype(int)
    sigma_tod_mapped = seasonal.sigma_tod[buckets]

    # σ_rv: nearest-neighbor lookup
    idx_rv = np.searchsorted(ts_rv, t_calib)
    idx_rv = np.clip(idx_rv, 0, len(ts_rv) - 1)
    # Check if previous index is closer
    idx_prev = np.clip(idx_rv - 1, 0, len(ts_rv) - 1)
    dist_right = np.abs(ts_rv[idx_rv] - t_calib)
    dist_left = np.abs(ts_rv[idx_prev] - t_calib)
    use_prev = dist_left < dist_right
    best_idx = np.where(use_prev, idx_prev, idx_rv)
    sigma_rv_mapped = sigma_rv_full[best_idx]

    # Prepare data
    S = dataset["S"].values
    K = dataset["K"].values
    tau = dataset["tau"].values
    y = dataset["y"].values.astype(float)
    sigma_rel = sigma_rv_mapped / sigma_tod_mapped

    # Baselines
    ll_const = _log_loss(y, np.full_like(y, np.mean(y), dtype=float))

    # ---- Stage 1: Fit base vol model ----
    print("Fitting base model parameters (stage 1)...")
    res = minimize(
        _neg_ll,
        x0=[0.5, 0.05, 0.5],
        args=(S, K, tau, sigma_tod_mapped, sigma_rel, y),
        method="L-BFGS-B",
        bounds=[(0.1, 3.0), (-0.5, 0.5), (0.0, 2.0)],
    )

    a0, a1, beta = res.x
    ll_stage1 = res.fun
    improvement_stage1 = (ll_const - ll_stage1) / ll_const * 100

    print(f"\n{'='*60}")
    print("STAGE 1: BASE VOL MODEL")
    print('='*60)
    print(f"\nModel: σ_eff = ({a0:.3f} + {a1:.3f}√τ_min) × σ_tod^{1-beta:.2f} × σ_rv^{beta:.2f}")
    print(f"\nParameters:")
    print(f"  a0 = {a0:.6f}")
    print(f"  a1 = {a1:.6f}")
    print(f"  β  = {beta:.6f}")
    print(f"\nPerformance:")
    print(f"  Log loss: {ll_stage1:.6f}")
    print(f"  vs Constant rate: {improvement_stage1:+.2f}%")
    print(f"  Samples: {len(y):,}")

    # Compute z-scores from stage 1
    tau_min = tau / 60.0
    a_tau = a0 + a1 * np.sqrt(tau_min)
    sigma_eff = a_tau * sigma_tod_mapped * np.power(sigma_rel, beta)
    z = _compute_z_score(S, K, tau, sigma_eff)

    # ---- Stage 2: Smooth probit calibration ----
    print(f"\n{'='*60}")
    print("STAGE 2: SMOOTH PROBIT CALIBRATION")
    print('='*60)
    print("\n  z' = b0 + b1·score + b2·log(τ) + b3·log(σ_rel) + b4·(log σ_rel)²")
    print("  p  = Φ(z')")
    print("  score = -z,  L2 ridge λ=0.001\n")

    probit_params = _fit_probit_layer(z, tau, y, sigma_rel=sigma_rel, lambda_l2=0.001)

    b0, b1, b2, b3, b4 = (probit_params["b0"], probit_params["b1"],
                            probit_params["b2"], probit_params["b3"],
                            probit_params["b4"])
    print(f"  b0 = {b0:+.6f}  (intercept)")
    print(f"  b1 = {b1:+.6f}  (score = -z)")
    print(f"  b2 = {b2:+.6f}  (log τ)")
    print(f"  b3 = {b3:+.6f}  (log σ_rel)")
    print(f"  b4 = {b4:+.6f}  ((log σ_rel)²)")

    # Final calibrated predictions
    p_cal = _apply_probit_layer(z, tau, probit_params, sigma_rel=sigma_rel)
    ll_final = _log_loss(y, p_cal)
    improvement_final = (ll_const - ll_final) / ll_const * 100
    e = y - p_cal

    # Cluster IDs for clustered SE (all samples within same hour share same y)
    market_ids = dataset["market_id"].values if "market_id" in dataset.columns else None

    # Diagnostics by τ-bucket
    print(f"\n  Per-τ-bucket diagnostics (clustered SE by hour):")
    for i in range(len(TAU_BUCKET_EDGES) - 1):
        lo, hi = TAU_BUCKET_EDGES[i], TAU_BUCKET_EDGES[i + 1]
        label = TAU_BUCKET_LABELS[i]
        mask = (tau_min >= lo) & (tau_min < hi)
        n = mask.sum()
        if n > 0:
            p_raw_bucket = np.clip(1.0 - norm.cdf(z[mask]), 1e-9, 1.0 - 1e-9)
            ll_raw = _log_loss(y[mask], p_raw_bucket)
            ll_cal = _log_loss(y[mask], p_cal[mask])
            bias = np.mean(e[mask])
            se_naive = np.std(e[mask]) / np.sqrt(n)
            se_clust = _clustered_se(e[mask], market_ids[mask]) if market_ids is not None else se_naive
            sig = "*" if abs(bias) > 2 * se_clust else ""
            print(f"    τ ∈ [{label:5s}] min: LL {ll_raw:.4f} → {ll_cal:.4f}  "
                  f"e={bias:+.4f}±{se_clust:.4f} (naive {se_naive:.4f}) {sig}  n={n}")

    # Diagnostics by σ_rel quartile
    vol_edges = list(np.quantile(sigma_rel, [0.0, 0.25, 0.5, 0.75, 1.0]))
    print(f"\n  Per-σ_rel-quartile diagnostics (clustered SE by hour):")
    for qi, ql in enumerate(VOL_QUARTILE_LABELS):
        lo, hi = vol_edges[qi], vol_edges[qi + 1]
        if qi == 3:
            mask_q = sigma_rel >= lo
        else:
            mask_q = (sigma_rel >= lo) & (sigma_rel < hi)
        if mask_q.sum() > 0:
            bias = np.mean(e[mask_q])
            se_naive = np.std(e[mask_q]) / np.sqrt(mask_q.sum())
            se_clust = _clustered_se(e[mask_q], market_ids[mask_q]) if market_ids is not None else se_naive
            sig = "*" if abs(bias) > 2 * se_clust else ""
            print(f"    {ql} [{lo:.4f}, {hi:.4f}]: e={bias:+.4f}±{se_clust:.4f} (naive {se_naive:.4f}) {sig}")

    print(f"\n{'='*60}")
    print("FINAL CALIBRATED MODEL")
    print('='*60)
    print(f"\n  Log loss: {ll_stage1:.6f} → {ll_final:.6f} (Δ={ll_final-ll_stage1:+.6f})")
    print(f"  vs Constant rate: {improvement_final:+.2f}% (was {improvement_stage1:+.2f}%)")
    print(f"  Global mean(e): {np.mean(e):+.4f}")
    print(f"  Parameters: 3 (stage 1) + 5 (stage 2) = 8 total")

    # Save parameters
    params = {
        "model": "ewma_smooth_probit_quad",
        "formula": "z' = b0+b1*score+b2*log(tau)+b3*log(sigma_rel)+b4*(log(sigma_rel))^2; p = Phi(z'), score=-z",
        "a0": float(a0),
        "a1": float(a1),
        "beta": float(beta),
        "ewma_half_life_sec": 300,
        "probit_layer": probit_params,
        "train_log_loss_stage1": float(ll_stage1),
        "train_log_loss_final": float(ll_final),
        "n_samples": len(y),
        "improvement_vs_constant_pct": float(improvement_final),
        "date_range": f"{cfg.start_date} to {cfg.end_date}",
    }

    with open(output_dir / "params_final.json", "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nSaved parameters to {output_dir / 'params_final.json'}")

    # Generate diagnostic plots: pure model (no probit) and model+probit
    generate_plots(S, K, tau, sigma_tod_mapped, sigma_rel, y, a0, a1, beta,
                   ll_stage1, ll_const, output_dir, probit_params=None,
                   sigma_rel_for_probit=None, filename_prefix="pure_model")

    generate_plots(S, K, tau, sigma_tod_mapped, sigma_rel, y, a0, a1, beta,
                   ll_final, ll_const, output_dir, probit_params=probit_params,
                   sigma_rel_for_probit=sigma_rel, filename_prefix="calibration")

    return params


# ============================================================================
# Generate Plots
# ============================================================================

def _equal_mass_reliability(p_pred, y, n_bins=20):
    """Compute reliability curve using equal-mass (quantile) bins.

    Returns (bin_pred, bin_actual, bin_se, bin_n) arrays.
    """
    order = np.argsort(p_pred)
    p_sorted = p_pred[order]
    y_sorted = y[order]
    bin_size = len(p_pred) // n_bins

    bin_pred, bin_actual, bin_se, bin_n = [], [], [], []
    for b in range(n_bins):
        lo = b * bin_size
        hi = (b + 1) * bin_size if b < n_bins - 1 else len(p_pred)
        n = hi - lo
        if n == 0:
            continue
        p_mean = np.mean(p_sorted[lo:hi])
        y_mean = np.mean(y_sorted[lo:hi])
        se = np.sqrt(y_mean * (1 - y_mean) / n) if n > 1 else 0
        bin_pred.append(p_mean)
        bin_actual.append(y_mean)
        bin_se.append(se)
        bin_n.append(n)

    return np.array(bin_pred), np.array(bin_actual), np.array(bin_se), np.array(bin_n)


def generate_plots(S, K, tau, sigma_tod, sigma_rel, y, a0, a1, beta, ll, ll_const, output_dir, probit_params=None, sigma_rel_for_probit=None, filename_prefix="calibration"):
    """Generate calibration diagnostic plots.

    Two figures:
      1. Main 2x3 diagnostics (equal-mass reliability, LL by τ, LL by |z'|,
         LL by σ_rel, prediction distribution, residuals vs calibrated score z')
      2. Conditional reliability: 4x4 grid of (τ-bucket × σ_rel quartile)
    """
    print("\nGenerating diagnostic plots...")

    # Compute z-scores
    tau_min = tau / 60.0
    a_tau = a0 + a1 * np.sqrt(tau_min)
    sigma_eff = a_tau * sigma_tod * np.power(sigma_rel, beta)
    z_raw = _compute_z_score(S, K, tau, sigma_eff)

    # Apply probit calibration layer if available
    if probit_params is not None:
        p_pred = _apply_probit_layer(z_raw, tau, probit_params, sigma_rel=sigma_rel_for_probit)
    else:
        p_pred = np.clip(1.0 - norm.cdf(z_raw), 1e-9, 1.0 - 1e-9)

    improvement = (ll_const - ll) / ll_const * 100

    # Compute calibrated score z' (the argument to Φ)
    score = -z_raw
    z_cal = np.copy(score)  # fallback: just score
    if probit_params is not None and isinstance(probit_params, dict) and "b0" in probit_params:
        b0 = probit_params["b0"]
        b1 = probit_params["b1"]
        b2 = probit_params["b2"]
        b3 = probit_params["b3"]
        b4 = probit_params.get("b4", 0.0)
        tau_sec = np.maximum(tau, 1.0)
        log_tau = np.log(tau_sec)
        zp = b0 + b1 * score + b2 * log_tau
        if sigma_rel_for_probit is not None:
            log_sr = np.log(np.maximum(sigma_rel_for_probit, 1e-6))
            zp = zp + b3 * log_sr + b4 * log_sr**2
        z_cal = zp

    # ======================================================================
    # Figure 1: Main diagnostics (2x3)
    # ======================================================================
    fig = plt.figure(figsize=(16, 12))

    # 1. Calibration plot (equal-mass bins)
    ax1 = fig.add_subplot(2, 3, 1)
    bp, ba, bse, bn = _equal_mass_reliability(p_pred, y, n_bins=20)
    ax1.errorbar(bp, ba, yerr=1.96*bse, fmt='o', capsize=3, alpha=0.8, markersize=5)
    ax1.plot([0, 1], [0, 1], 'k--', lw=2)
    ax1.set_xlabel('Predicted probability')
    ax1.set_ylabel('Actual frequency')
    ax1.set_title(f'Calibration (equal-mass bins)\nLL = {ll:.4f} ({improvement:+.1f}% vs const)')
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # 2. Performance by τ
    ax2 = fig.add_subplot(2, 3, 2)
    tau_buckets = [0, 5, 15, 30, 60, 120, np.inf]
    tau_labels_plot = ['0-5', '5-15', '15-30', '30-60', '60-120', '120+']
    tau_ll, tau_n = [], []

    for i in range(len(tau_buckets) - 1):
        mask = (tau_min >= tau_buckets[i]) & (tau_min < tau_buckets[i+1])
        if mask.sum() > 10:
            tau_ll.append(_log_loss(y[mask], p_pred[mask]))
            tau_n.append(mask.sum())
        else:
            tau_ll.append(np.nan)
            tau_n.append(0)

    bars = ax2.bar(tau_labels_plot, tau_ll, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(tau_labels_plot))), edgecolor='black')
    ax2.axhline(ll, color='red', linestyle='--', label=f'Overall: {ll:.4f}')
    ax2.axhline(ll_const, color='gray', linestyle=':', label=f'Constant: {ll_const:.4f}')
    ax2.set_xlabel('τ (minutes)')
    ax2.set_ylabel('Log Loss')
    ax2.set_title('Performance by Time to Expiry')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, n in zip(bars, tau_n):
        if n > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'n={n}', ha='center', va='bottom', fontsize=7)

    # 3. Performance by |z'| (calibrated score)
    ax3 = fig.add_subplot(2, 3, 3)
    z_abs = np.abs(z_cal)
    z_buckets = [0, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf]
    z_labels = ['0-0.5', '0.5-1', '1-1.5', '1.5-2', '2-3', '3+']
    z_ll, z_n = [], []

    for i in range(len(z_buckets) - 1):
        mask = (z_abs >= z_buckets[i]) & (z_abs < z_buckets[i+1])
        if mask.sum() > 10:
            z_ll.append(_log_loss(y[mask], p_pred[mask]))
            z_n.append(mask.sum())
        else:
            z_ll.append(np.nan)
            z_n.append(0)

    bars = ax3.bar(z_labels, z_ll, color=plt.cm.plasma(np.linspace(0.2, 0.8, len(z_labels))), edgecolor='black')
    ax3.axhline(ll, color='red', linestyle='--', label=f'Overall: {ll:.4f}')
    ax3.axhline(np.log(2), color='orange', linestyle=':', label='Random (ln2)')
    ax3.set_xlabel("|z'| (calibrated score)")
    ax3.set_ylabel('Log Loss')
    ax3.set_title("Performance by |z'|\n(calibrated score magnitude)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, n in zip(bars, z_n):
        if n > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'n={n}', ha='center', va='bottom', fontsize=7)

    # 4. Performance by σ_rel
    ax4 = fig.add_subplot(2, 3, 4)
    rel_buckets = [0, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf]
    rel_labels = ['0-0.5', '0.5-1', '1-1.5', '1.5-2', '2-3', '3+']
    rel_ll, rel_n = [], []

    for i in range(len(rel_buckets) - 1):
        mask = (sigma_rel >= rel_buckets[i]) & (sigma_rel < rel_buckets[i+1])
        if mask.sum() > 10:
            rel_ll.append(_log_loss(y[mask], p_pred[mask]))
            rel_n.append(mask.sum())
        else:
            rel_ll.append(np.nan)
            rel_n.append(0)

    bars = ax4.bar(rel_labels, rel_ll, color=plt.cm.coolwarm(np.linspace(0.1, 0.9, len(rel_labels))), edgecolor='black')
    ax4.axhline(ll, color='red', linestyle='--', label=f'Overall: {ll:.4f}')
    ax4.set_xlabel('σ_rel = σ_rv / σ_tod')
    ax4.set_ylabel('Log Loss')
    ax4.set_title('Performance by Volatility Regime')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, n in zip(bars, rel_n):
        if n > 0:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'n={n}', ha='center', va='bottom', fontsize=7)

    # 5. Prediction distribution
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(p_pred[y == 1], bins=30, alpha=0.6, label='y=1 (above strike)', density=True)
    ax5.hist(p_pred[y == 0], bins=30, alpha=0.6, label='y=0 (below strike)', density=True)
    ax5.axvline(0.5, color='black', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Predicted probability')
    ax5.set_ylabel('Density')
    ax5.set_title('Prediction Distribution by Outcome')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. Residuals vs calibrated score z'
    ax6 = fig.add_subplot(2, 3, 6)
    residuals = y - p_pred
    z_bins = np.linspace(-4, 4, 41)
    z_centers = (z_bins[:-1] + z_bins[1:]) / 2
    res_means, res_stds = [], []

    for i in range(len(z_bins) - 1):
        mask = (z_cal >= z_bins[i]) & (z_cal < z_bins[i+1])
        if mask.sum() > 5:
            res_means.append(np.mean(residuals[mask]))
            res_stds.append(np.std(residuals[mask]) / np.sqrt(mask.sum()))
        else:
            res_means.append(np.nan)
            res_stds.append(np.nan)

    res_means = np.array(res_means)
    res_stds = np.array(res_stds)
    valid = ~np.isnan(res_means)
    ax6.errorbar(z_centers[valid], res_means[valid], yerr=1.96*res_stds[valid],
                 fmt='o', markersize=4, capsize=2, alpha=0.7)
    ax6.axhline(0, color='black', linestyle='-', lw=1)
    ax6.fill_between([-4, 4], [-0.05, -0.05], [0.05, 0.05], alpha=0.1, color='green')
    ax6.set_xlabel("z' = α_τ + α_vol + γ_τ·score (calibrated)")
    ax6.set_ylabel('Mean residual (y - p)')
    ax6.set_title("Residual vs Calibrated Score z'")
    ax6.set_xlim(-4, 4)
    ax6.set_ylim(-0.3, 0.3)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    diag_path = output_dir / f"{filename_prefix}_diagnostics.png"
    plt.savefig(diag_path, dpi=150, bbox_inches='tight')
    print(f"  Saved {diag_path}")

    # ======================================================================
    # Figure 2: Conditional reliability (τ-bucket × σ_rel quartile)
    # ======================================================================
    vol_edges = None
    if sigma_rel_for_probit is not None:
        vol_edges = list(np.quantile(sigma_rel_for_probit, [0.0, 0.25, 0.5, 0.75, 1.0]))

    if vol_edges is not None and sigma_rel_for_probit is not None:
        fig2, axes = plt.subplots(4, 4, figsize=(18, 16))
        fig2.suptitle('Conditional Reliability: τ-bucket × σ_rel quartile\n'
                       f'(equal-mass bins, LL={ll:.4f}, {improvement:+.1f}% vs const)',
                       fontsize=14, y=0.98)

        for ti, tl in enumerate(TAU_BUCKET_LABELS):
            t_lo, t_hi = TAU_BUCKET_EDGES[ti], TAU_BUCKET_EDGES[ti + 1]
            tau_mask = (tau_min >= t_lo) & (tau_min < t_hi)

            for qi, ql in enumerate(VOL_QUARTILE_LABELS):
                ax = axes[ti, qi]
                v_lo, v_hi = vol_edges[qi], vol_edges[qi + 1]
                if qi == 3:
                    vol_mask = sigma_rel_for_probit >= v_lo
                else:
                    vol_mask = (sigma_rel_for_probit >= v_lo) & (sigma_rel_for_probit < v_hi)

                cell_mask = tau_mask & vol_mask
                n_cell = cell_mask.sum()

                if n_cell < 30:
                    ax.text(0.5, 0.5, f'n={n_cell}\n(too few)', ha='center', va='center',
                            transform=ax.transAxes, fontsize=10)
                    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.3)
                else:
                    n_bins_cell = max(8, min(15, n_cell // 50))
                    bp_c, ba_c, bse_c, _ = _equal_mass_reliability(
                        p_pred[cell_mask], y[cell_mask], n_bins=n_bins_cell)
                    ax.errorbar(bp_c, ba_c, yerr=1.96*bse_c, fmt='o', capsize=2,
                                markersize=4, alpha=0.8)
                    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)

                    cell_ll = _log_loss(y[cell_mask], p_pred[cell_mask])
                    cell_e = np.mean(y[cell_mask] - p_pred[cell_mask])
                    ax.set_title(f'LL={cell_ll:.3f}, e={cell_e:+.3f}, n={n_cell}',
                                 fontsize=8)

                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.2)

                if ti == 0:
                    ax.set_title(f'{ql} [{v_lo:.2f},{v_hi:.2f}]\n' + ax.get_title(),
                                 fontsize=8)
                if qi == 0:
                    ax.set_ylabel(f'τ∈[{tl}]min\nActual', fontsize=8)
                else:
                    ax.set_ylabel('')
                if ti == 3:
                    ax.set_xlabel('Predicted', fontsize=8)
                else:
                    ax.set_xlabel('')

                ax.tick_params(labelsize=7)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        cond_path = output_dir / f"{filename_prefix}_conditional.png"
        plt.savefig(cond_path, dpi=150, bbox_inches='tight')
        print(f"  Saved {cond_path}")

    plt.close('all')


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="One-click calibration")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild dataset from scratch")
    args = parser.parse_args()

    print("=" * 60)
    print("BINARY OPTION PRICER CALIBRATION")
    print("=" * 60)

    cfg = load_config(args.config)
    print(f"\nConfig: {cfg.start_date} to {cfg.end_date}, asset={cfg.asset}")

    output_dir = Path(cfg.output_dir)
    dataset_path = output_dir / "calibration_dataset.parquet"

    # Step 1: Build or load dataset
    print("\n" + "-" * 60)
    print("STEP 1: Loading/building dataset")
    print("-" * 60)

    if dataset_path.exists() and not args.rebuild:
        print(f"Loading existing dataset from {dataset_path}")
        dataset = pd.read_parquet(dataset_path)
        print(f"  {len(dataset):,} calibration samples")
    else:
        print("Building dataset from scratch...")
        dataset = build_dataset(cfg)

    if dataset.empty:
        print("No data to calibrate. Exiting.")
        return

    # Load BBO and seasonal vol for σ_rv computation
    print("\nLoading BBO for σ_rv computation...")
    bbo = pd.read_parquet(output_dir / "bbo_cache.parquet")

    print("Loading seasonal volatility...")
    seasonal = compute_seasonal_vol_ticktime(
        bbo,
        bucket_minutes=cfg.tod_bucket_minutes,
        smoothing_window=cfg.tod_smoothing_window,
        floor=cfg.sigma_tod_floor,
        target_interval_sec=1.0,
    )

    # Step 2: Calibrate
    print("\n" + "-" * 60)
    print("STEP 2: Calibrating model")
    print("-" * 60)
    params = calibrate_model(dataset, bbo, seasonal, cfg)

    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved to: {cfg.output_dir}/")
    print("  - params_final.json (fitted parameters)")
    print("  - calibration_diagnostics.png (plots)")
    print("  - calibration_dataset.parquet (data)")


if __name__ == "__main__":
    main()
