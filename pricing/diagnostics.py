"""Calibration diagnostic plots.

Generates two figures:
    1. Main diagnostics (2x3): calibration curve, LL by tau, LL by score,
       LL by sigma_rel, prediction distribution, residuals vs score.
    2. Conditional reliability (4x4): tau-bucket x sigma_rel quartile.

All plots are model-agnostic: they take predictions and labels as input.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from pricing.models.base import Model, CalibrationResult
from pricing.calibrate import log_loss


# ============================================================================
# Metrics
# ============================================================================

def equal_mass_reliability(p_pred: np.ndarray, y: np.ndarray, n_bins: int = 20):
    """Reliability curve using equal-mass (quantile) bins.

    Returns (bin_pred, bin_actual, bin_se, bin_n).
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


# ============================================================================
# Main API
# ============================================================================

TAU_BUCKET_EDGES = [0, 5, 15, 30, 60]
TAU_BUCKET_LABELS = ["0-5", "5-15", "15-30", "30-60"]
VOL_QUARTILE_LABELS = ["Q1", "Q2", "Q3", "Q4"]


def generate_diagnostics(
    model: Model,
    dataset: pd.DataFrame,
    result: CalibrationResult,
    output_dir: str | Path,
    prefix: str | None = None,
) -> None:
    """Generate all diagnostic plots for a calibrated model.

    Saves:
        {prefix}_diagnostics.png  -- 2x3 main diagnostics
        {prefix}_conditional.png  -- 4x4 conditional reliability

    Args:
        model: Model instance (uses predict + required_features).
        dataset: Calibration dataset from build_dataset().
        result: CalibrationResult from calibrate().
        output_dir: Directory to save plots.
        prefix: Filename prefix (default: model.name).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or model.name

    # Compute predictions
    S = dataset["S"].values.astype(np.float64)
    K = dataset["K"].values.astype(np.float64)
    tau = dataset["tau"].values.astype(np.float64)
    y = dataset["y"].values.astype(np.float64)
    features = {f: dataset[f].values.astype(np.float64) for f in model.required_features()}

    p_pred = model.predict(result.params, S, K, tau, features)
    p_pred = np.clip(p_pred, 1e-9, 1.0 - 1e-9)

    tau_min = tau / 60.0
    ll = result.log_loss
    ll_const = result.log_loss_baseline
    improvement = result.improvement_pct

    # Compute a "score" for the x-axis of residual plots.
    # Use inverse CDF of predictions as a generic score.
    score = norm.ppf(np.clip(p_pred, 1e-6, 1 - 1e-6))

    # sigma_rel is a standard dataset column (even if the model doesn't use it)
    sigma_rel = dataset["sigma_rel"].values.astype(np.float64) if "sigma_rel" in dataset.columns else None

    _plot_main_diagnostics(
        p_pred, y, tau_min, score, sigma_rel,
        ll, ll_const, improvement, output_dir, prefix,
    )

    if sigma_rel is not None:
        _plot_conditional_reliability(
            p_pred, y, tau_min, sigma_rel,
            ll, improvement, output_dir, prefix,
        )

    plt.close("all")
    print(f"  Diagnostic plots saved to {output_dir}/")


# ============================================================================
# Figure 1: Main diagnostics (2x3)
# ============================================================================

def _plot_main_diagnostics(p_pred, y, tau_min, score, sigma_rel,
                           ll, ll_const, improvement, output_dir, prefix):
    fig = plt.figure(figsize=(16, 12))

    # 1. Calibration curve
    ax1 = fig.add_subplot(2, 3, 1)
    bp, ba, bse, bn = equal_mass_reliability(p_pred, y, n_bins=20)
    ax1.errorbar(bp, ba, yerr=1.96 * bse, fmt="o", capsize=3, alpha=0.8, markersize=5)
    ax1.plot([0, 1], [0, 1], "k--", lw=2)
    ax1.set_xlabel("Predicted probability")
    ax1.set_ylabel("Actual frequency")
    ax1.set_title(f"Calibration (equal-mass bins)\nLL = {ll:.4f} ({improvement:+.1f}% vs const)")
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # 2. Performance by tau
    ax2 = fig.add_subplot(2, 3, 2)
    tau_buckets = [0, 5, 15, 30, 60, 120, np.inf]
    tau_labels = ["0-5", "5-15", "15-30", "30-60", "60-120", "120+"]
    tau_ll, tau_n = [], []
    for i in range(len(tau_buckets) - 1):
        mask = (tau_min >= tau_buckets[i]) & (tau_min < tau_buckets[i + 1])
        if mask.sum() > 10:
            tau_ll.append(log_loss(y[mask], p_pred[mask]))
            tau_n.append(mask.sum())
        else:
            tau_ll.append(np.nan)
            tau_n.append(0)
    bars = ax2.bar(tau_labels, tau_ll,
                   color=plt.cm.viridis(np.linspace(0.2, 0.8, len(tau_labels))),
                   edgecolor="black")
    ax2.axhline(ll, color="red", linestyle="--", label=f"Overall: {ll:.4f}")
    ax2.axhline(ll_const, color="gray", linestyle=":", label=f"Constant: {ll_const:.4f}")
    ax2.set_xlabel("τ (minutes)")
    ax2.set_ylabel("Log Loss")
    ax2.set_title("Performance by Time to Expiry")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, n in zip(bars, tau_n):
        if n > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"n={n}", ha="center", va="bottom", fontsize=7)

    # 3. Performance by |score|
    ax3 = fig.add_subplot(2, 3, 3)
    s_abs = np.abs(score)
    s_buckets = [0, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf]
    s_labels = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3+"]
    s_ll, s_n = [], []
    for i in range(len(s_buckets) - 1):
        mask = (s_abs >= s_buckets[i]) & (s_abs < s_buckets[i + 1])
        if mask.sum() > 10:
            s_ll.append(log_loss(y[mask], p_pred[mask]))
            s_n.append(mask.sum())
        else:
            s_ll.append(np.nan)
            s_n.append(0)
    bars = ax3.bar(s_labels, s_ll,
                   color=plt.cm.plasma(np.linspace(0.2, 0.8, len(s_labels))),
                   edgecolor="black")
    ax3.axhline(ll, color="red", linestyle="--", label=f"Overall: {ll:.4f}")
    ax3.axhline(np.log(2), color="orange", linestyle=":", label="Random (ln2)")
    ax3.set_xlabel("|score| (Φ⁻¹(p))")
    ax3.set_ylabel("Log Loss")
    ax3.set_title("Performance by Score Magnitude")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis="y")
    for bar, n in zip(bars, s_n):
        if n > 0:
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"n={n}", ha="center", va="bottom", fontsize=7)

    # 4. Performance by sigma_rel
    ax4 = fig.add_subplot(2, 3, 4)
    if sigma_rel is not None:
        rel_buckets = [0, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf]
        rel_labels = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3+"]
        rel_ll, rel_n = [], []
        for i in range(len(rel_buckets) - 1):
            mask = (sigma_rel >= rel_buckets[i]) & (sigma_rel < rel_buckets[i + 1])
            if mask.sum() > 10:
                rel_ll.append(log_loss(y[mask], p_pred[mask]))
                rel_n.append(mask.sum())
            else:
                rel_ll.append(np.nan)
                rel_n.append(0)
        bars = ax4.bar(rel_labels, rel_ll,
                       color=plt.cm.coolwarm(np.linspace(0.1, 0.9, len(rel_labels))),
                       edgecolor="black")
        ax4.axhline(ll, color="red", linestyle="--", label=f"Overall: {ll:.4f}")
        ax4.set_xlabel("σ_rel = σ_rv / σ_tod")
        ax4.set_ylabel("Log Loss")
        ax4.set_title("Performance by Volatility Regime")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis="y")
        for bar, n in zip(bars, rel_n):
            if n > 0:
                ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f"n={n}", ha="center", va="bottom", fontsize=7)
    else:
        ax4.text(0.5, 0.5, "sigma_rel not available", ha="center", va="center",
                 transform=ax4.transAxes)

    # 5. Prediction distribution
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(p_pred[y == 1], bins=30, alpha=0.6, label="y=1 (Up)", density=True)
    ax5.hist(p_pred[y == 0], bins=30, alpha=0.6, label="y=0 (Down)", density=True)
    ax5.axvline(0.5, color="black", linestyle="--", alpha=0.5)
    ax5.set_xlabel("Predicted probability")
    ax5.set_ylabel("Density")
    ax5.set_title("Prediction Distribution by Outcome")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. Residuals vs score
    ax6 = fig.add_subplot(2, 3, 6)
    residuals = y - p_pred
    s_bins = np.linspace(-4, 4, 41)
    s_centers = (s_bins[:-1] + s_bins[1:]) / 2
    res_means, res_stds = [], []
    for i in range(len(s_bins) - 1):
        mask = (score >= s_bins[i]) & (score < s_bins[i + 1])
        if mask.sum() > 5:
            res_means.append(np.mean(residuals[mask]))
            res_stds.append(np.std(residuals[mask]) / np.sqrt(mask.sum()))
        else:
            res_means.append(np.nan)
            res_stds.append(np.nan)
    res_means = np.array(res_means)
    res_stds = np.array(res_stds)
    valid = ~np.isnan(res_means)
    ax6.errorbar(s_centers[valid], res_means[valid], yerr=1.96 * res_stds[valid],
                 fmt="o", markersize=4, capsize=2, alpha=0.7)
    ax6.axhline(0, color="black", linestyle="-", lw=1)
    ax6.fill_between([-4, 4], [-0.05, -0.05], [0.05, 0.05], alpha=0.1, color="green")
    ax6.set_xlabel("Score (Φ⁻¹(p))")
    ax6.set_ylabel("Mean residual (y - p)")
    ax6.set_title("Residual vs Score")
    ax6.set_xlim(-4, 4)
    ax6.set_ylim(-0.3, 0.3)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / f"{prefix}_diagnostics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")


# ============================================================================
# Figure 2: Conditional reliability (4x4)
# ============================================================================

def _plot_conditional_reliability(p_pred, y, tau_min, sigma_rel,
                                  ll, improvement, output_dir, prefix):
    vol_edges = list(np.quantile(sigma_rel, [0.0, 0.25, 0.5, 0.75, 1.0]))

    fig, axes = plt.subplots(4, 4, figsize=(18, 16))
    fig.suptitle(
        f"Conditional Reliability: τ-bucket × σ_rel quartile\n"
        f"(equal-mass bins, LL={ll:.4f}, {improvement:+.1f}% vs const)",
        fontsize=14, y=0.98,
    )

    for ti, tl in enumerate(TAU_BUCKET_LABELS):
        t_lo, t_hi = TAU_BUCKET_EDGES[ti], TAU_BUCKET_EDGES[ti + 1]
        tau_mask = (tau_min >= t_lo) & (tau_min < t_hi)

        for qi, ql in enumerate(VOL_QUARTILE_LABELS):
            ax = axes[ti, qi]
            v_lo, v_hi = vol_edges[qi], vol_edges[qi + 1]
            vol_mask = (sigma_rel >= v_lo) if qi == 3 else (sigma_rel >= v_lo) & (sigma_rel < v_hi)

            cell_mask = tau_mask & vol_mask
            n_cell = cell_mask.sum()

            if n_cell < 30:
                ax.text(0.5, 0.5, f"n={n_cell}\n(too few)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10)
                ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.3)
            else:
                n_bins_cell = max(8, min(15, n_cell // 50))
                bp_c, ba_c, bse_c, _ = equal_mass_reliability(
                    p_pred[cell_mask], y[cell_mask], n_bins=n_bins_cell)
                ax.errorbar(bp_c, ba_c, yerr=1.96 * bse_c, fmt="o", capsize=2,
                            markersize=4, alpha=0.8)
                ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)

                cell_ll = log_loss(y[cell_mask], p_pred[cell_mask])
                cell_e = np.mean(y[cell_mask] - p_pred[cell_mask])
                ax.set_title(f"LL={cell_ll:.3f}, e={cell_e:+.3f}, n={n_cell}", fontsize=8)

            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.2)

            if ti == 0:
                ax.set_title(f"{ql} [{v_lo:.2f},{v_hi:.2f}]\n" + ax.get_title(), fontsize=8)
            if qi == 0:
                ax.set_ylabel(f"τ∈[{tl}]min\nActual", fontsize=8)
            else:
                ax.set_ylabel("")
            if ti == 3:
                ax.set_xlabel("Predicted", fontsize=8)
            else:
                ax.set_xlabel("")
            ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = output_dir / f"{prefix}_conditional.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
