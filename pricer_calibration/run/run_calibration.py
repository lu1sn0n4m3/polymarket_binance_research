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
    """Compute EWMA realized volatility on tick-time data."""
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
    r_tick = np.diff(log_mid)
    dt_ms = np.diff(ts_changed)
    dt_sec = dt_ms / 1000.0
    ts_tick = ts_changed[1:]

    valid = dt_sec > 0.001
    r_tick = r_tick[valid]
    dt_sec = dt_sec[valid]
    ts_tick = ts_tick[valid]

    r_per_sqrt_sec = r_tick / np.sqrt(dt_sec)

    n = len(r_per_sqrt_sec)
    sigma_rv = np.zeros(n)

    init_window = min(100, n // 10)
    v = np.mean(r_per_sqrt_sec[:init_window] ** 2) if init_window > 10 else 1.0

    for i in range(n):
        alpha = 1 - np.exp(-np.log(2) * dt_sec[i] / half_life_sec)
        alpha = np.clip(alpha, 0.001, 0.999)
        v = (1 - alpha) * v + alpha * r_per_sqrt_sec[i] ** 2
        sigma_rv[i] = np.sqrt(max(v, 1e-12))

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
    x = np.log(K / S) / (sigma_eff * sqrt_tau)
    p = 1.0 - norm.cdf(x)
    return _log_loss(y, p)


def calibrate_model(dataset, bbo, seasonal, cfg):
    """Fit model parameters via maximum likelihood."""
    output_dir = Path(cfg.output_dir)

    # Compute σ_rv
    print("Computing σ_rv (EWMA H=5min)...")
    ts_rv, sigma_rv_full = compute_rv_ewma(bbo, half_life_sec=300.0)

    # Map σ_tod to calibration samples
    t_calib = dataset["t"].values
    sigma_tod_mapped = np.zeros(len(dataset))
    sigma_rv_mapped = np.zeros(len(dataset))

    for i, t in enumerate(t_calib):
        # σ_tod
        total_seconds = (t // 1000) % 86400
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        bucket = (hours * 60 + minutes) // cfg.tod_bucket_minutes
        sigma_tod_mapped[i] = seasonal.sigma_tod[int(bucket)]

        # σ_rv (nearest)
        idx = np.searchsorted(ts_rv, t)
        if idx == 0:
            sigma_rv_mapped[i] = sigma_rv_full[0]
        elif idx >= len(ts_rv):
            sigma_rv_mapped[i] = sigma_rv_full[-1]
        else:
            if abs(ts_rv[idx] - t) < abs(ts_rv[idx-1] - t):
                sigma_rv_mapped[i] = sigma_rv_full[idx]
            else:
                sigma_rv_mapped[i] = sigma_rv_full[idx-1]

    # Prepare data
    S = dataset["S"].values
    K = dataset["K"].values
    tau = dataset["tau"].values
    y = dataset["y"].values.astype(float)
    sigma_rel = sigma_rv_mapped / sigma_tod_mapped

    # Baselines
    ll_const = _log_loss(y, np.full_like(y, np.mean(y), dtype=float))

    # Fit model
    print("Fitting model parameters...")
    res = minimize(
        _neg_ll,
        x0=[0.5, 0.05, 0.5],
        args=(S, K, tau, sigma_tod_mapped, sigma_rel, y),
        method="L-BFGS-B",
        bounds=[(0.1, 3.0), (-0.5, 0.5), (0.0, 2.0)],
    )

    a0, a1, beta = res.x
    ll = res.fun
    improvement = (ll_const - ll) / ll_const * 100

    print(f"\n{'='*60}")
    print("CALIBRATION RESULTS")
    print('='*60)
    print(f"\nModel: σ_eff = ({a0:.3f} + {a1:.3f}√τ_min) × σ_tod^{1-beta:.2f} × σ_rv^{beta:.2f}")
    print(f"\nParameters:")
    print(f"  a0 = {a0:.6f}")
    print(f"  a1 = {a1:.6f}")
    print(f"  β  = {beta:.6f}")
    print(f"\nPerformance:")
    print(f"  Log loss: {ll:.6f}")
    print(f"  vs Constant rate: {improvement:+.2f}%")
    print(f"  Samples: {len(y):,}")

    # Save parameters
    params = {
        "model": "ewma_combined",
        "formula": "sigma_eff = a(tau) * sigma_tod * (sigma_rv/sigma_tod)^beta",
        "a0": float(a0),
        "a1": float(a1),
        "beta": float(beta),
        "ewma_half_life_sec": 300,
        "dist": "normal",
        "train_log_loss": float(ll),
        "n_samples": len(y),
        "improvement_vs_constant_pct": float(improvement),
        "date_range": f"{cfg.start_date} to {cfg.end_date}",
    }

    with open(output_dir / "params_final.json", "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nSaved parameters to {output_dir / 'params_final.json'}")

    # Generate diagnostic plots
    generate_plots(S, K, tau, sigma_tod_mapped, sigma_rel, y, a0, a1, beta, ll, ll_const, output_dir)

    return params


# ============================================================================
# Generate Plots
# ============================================================================

def generate_plots(S, K, tau, sigma_tod, sigma_rel, y, a0, a1, beta, ll, ll_const, output_dir):
    """Generate calibration diagnostic plots."""
    print("\nGenerating diagnostic plots...")

    # Compute predictions
    tau_min = tau / 60.0
    a_tau = a0 + a1 * np.sqrt(tau_min)
    sigma_eff = a_tau * sigma_tod * np.power(sigma_rel, beta)
    sqrt_tau = np.sqrt(np.maximum(tau, 1e-6))
    x = np.log(K / S) / (sigma_eff * sqrt_tau)
    p_pred = np.clip(1.0 - norm.cdf(x), 1e-9, 1.0 - 1e-9)

    improvement = (ll_const - ll) / ll_const * 100

    fig = plt.figure(figsize=(16, 12))

    # 1. Calibration plot
    ax1 = fig.add_subplot(2, 3, 1)
    n_bins = 20
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.clip(np.digitize(p_pred, bin_edges) - 1, 0, n_bins - 1)

    bin_pred, bin_actual, bin_std = [], [], []
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() > 0:
            bin_pred.append(np.mean(p_pred[mask]))
            bin_actual.append(np.mean(y[mask]))
            p_act = np.mean(y[mask])
            bin_std.append(np.sqrt(p_act * (1 - p_act) / mask.sum()) if mask.sum() > 1 else 0)

    ax1.errorbar(bin_pred, bin_actual, yerr=1.96*np.array(bin_std), fmt='o', capsize=3, alpha=0.8)
    ax1.plot([0, 1], [0, 1], 'k--', lw=2)
    ax1.set_xlabel('Predicted probability')
    ax1.set_ylabel('Actual frequency')
    ax1.set_title(f'Calibration Plot\nLog Loss = {ll:.4f} ({improvement:+.1f}% vs const)')
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # 2. Performance by τ
    ax2 = fig.add_subplot(2, 3, 2)
    tau_buckets = [0, 5, 15, 30, 60, 120, np.inf]
    tau_labels = ['0-5', '5-15', '15-30', '30-60', '60-120', '120+']
    tau_ll, tau_n = [], []

    for i in range(len(tau_buckets) - 1):
        mask = (tau_min >= tau_buckets[i]) & (tau_min < tau_buckets[i+1])
        if mask.sum() > 10:
            tau_ll.append(_log_loss(y[mask], p_pred[mask]))
            tau_n.append(mask.sum())
        else:
            tau_ll.append(np.nan)
            tau_n.append(0)

    bars = ax2.bar(tau_labels, tau_ll, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(tau_labels))), edgecolor='black')
    ax2.axhline(ll, color='red', linestyle='--', label=f'Overall: {ll:.4f}')
    ax2.axhline(ll_const, color='gray', linestyle=':', label=f'Constant: {ll_const:.4f}')
    ax2.set_xlabel('τ (minutes)')
    ax2.set_ylabel('Log Loss')
    ax2.set_title('Performance by Time to Expiry')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, n in zip(bars, tau_n):
        if n > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'n={n}', ha='center', va='bottom', fontsize=8)

    # 3. Performance by |x|
    ax3 = fig.add_subplot(2, 3, 3)
    x_abs = np.abs(x)
    x_buckets = [0, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf]
    x_labels = ['0-0.5', '0.5-1', '1-1.5', '1.5-2', '2-3', '3+']
    x_ll, x_n = [], []

    for i in range(len(x_buckets) - 1):
        mask = (x_abs >= x_buckets[i]) & (x_abs < x_buckets[i+1])
        if mask.sum() > 10:
            x_ll.append(_log_loss(y[mask], p_pred[mask]))
            x_n.append(mask.sum())
        else:
            x_ll.append(np.nan)
            x_n.append(0)

    bars = ax3.bar(x_labels, x_ll, color=plt.cm.plasma(np.linspace(0.2, 0.8, len(x_labels))), edgecolor='black')
    ax3.axhline(ll, color='red', linestyle='--', label=f'Overall: {ll:.4f}')
    ax3.axhline(np.log(2), color='orange', linestyle=':', label='Random (ln2)')
    ax3.set_xlabel('|x| = |ln(K/S)/(σ√τ)|')
    ax3.set_ylabel('Log Loss')
    ax3.set_title('Performance by Distance from Strike\n(|x|<1 is "dead zone")')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, n in zip(bars, x_n):
        if n > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'n={n}', ha='center', va='bottom', fontsize=8)

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
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, n in zip(bars, rel_n):
        if n > 0:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'n={n}', ha='center', va='bottom', fontsize=8)

    # 5. Prediction distribution
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(p_pred[y == 1], bins=30, alpha=0.6, label='y=1 (above strike)', density=True)
    ax5.hist(p_pred[y == 0], bins=30, alpha=0.6, label='y=0 (below strike)', density=True)
    ax5.axvline(0.5, color='black', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Predicted probability')
    ax5.set_ylabel('Density')
    ax5.set_title('Prediction Distribution by Outcome')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Residuals
    ax6 = fig.add_subplot(2, 3, 6)
    residuals = y - p_pred
    x_bins = np.linspace(-4, 4, 41)
    x_centers = (x_bins[:-1] + x_bins[1:]) / 2
    res_means, res_stds = [], []

    for i in range(len(x_bins) - 1):
        mask = (x >= x_bins[i]) & (x < x_bins[i+1])
        if mask.sum() > 5:
            res_means.append(np.mean(residuals[mask]))
            res_stds.append(np.std(residuals[mask]) / np.sqrt(mask.sum()))
        else:
            res_means.append(np.nan)
            res_stds.append(np.nan)

    valid = ~np.isnan(res_means)
    ax6.errorbar(x_centers[valid], np.array(res_means)[valid], yerr=1.96*np.array(res_stds)[valid], fmt='o', markersize=4, capsize=2, alpha=0.7)
    ax6.axhline(0, color='black', linestyle='-', lw=1)
    ax6.fill_between([-4, 4], [-0.05, -0.05], [0.05, 0.05], alpha=0.1, color='green')
    ax6.set_xlabel('x = ln(K/S)/(σ√τ)')
    ax6.set_ylabel('Mean residual (y - p)')
    ax6.set_title('Residual Analysis')
    ax6.set_xlim(-4, 4)
    ax6.set_ylim(-0.3, 0.3)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "calibration_diagnostics.png", dpi=150, bbox_inches='tight')
    print(f"Saved plots to {output_dir / 'calibration_diagnostics.png'}")


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
