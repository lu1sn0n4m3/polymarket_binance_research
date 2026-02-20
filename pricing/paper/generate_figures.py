"""Generate publication-quality figures for the Gaussian pricer paper.

Usage:
    python pricing/paper/generate_figures.py

Outputs PNGs to pricing/paper/figures/.
"""

import sys
sys.path.insert(0, ".")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import norm

from pricing.models import get_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("pricing/paper/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("pricing/output")
DPI = 300

# Colors
C_MODEL = "#2563eb"     # model (blue)
C_GAUSS = "#94a3b8"     # Gaussian reference (gray)
C_EMP = "#dc2626"       # empirical (red)
C_ACCENT = "#f59e0b"    # accent (amber)
C_GREEN = "#16a34a"

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 100,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data...")
dataset = pd.read_parquet(DATA_DIR / "calibration_dataset.parquet")
seasonal_df = pd.read_parquet(DATA_DIR / "seasonal_vol_weekday.parquet")

with open(DATA_DIR / "gaussian_vol_params.json") as f:
    vol_params = json.load(f)

# Model
model_gauss = get_model("gaussian")

# Arrays
S = dataset["S"].values.astype(np.float64)
K = dataset["K"].values.astype(np.float64)
tau = dataset["tau"].values.astype(np.float64)
S_T = dataset["S_T"].values.astype(np.float64)
y = dataset["y"].values.astype(np.float64)
features = {f: dataset[f].values.astype(np.float64) for f in model_gauss.required_features()}
tau_min = tau / 60.0

# Params
gauss_params = {k: vol_params[k] for k in model_gauss.param_names()}

# Predictions
p_gauss = np.clip(model_gauss.predict(gauss_params, S, K, tau, features), 1e-9, 1 - 1e-9)

# Residuals
v_pred = model_gauss.predict_variance(gauss_params, S, K, tau, features)
z = np.log(S_T / S) / np.sqrt(np.maximum(v_pred, 1e-20))

# Log loss
def log_loss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

ll_baseline = log_loss(y, np.full_like(y, y.mean()))
ll_gauss = log_loss(y, p_gauss)

print(f"  Baseline LL: {ll_baseline:.4f}")
print(f"  Gaussian LL: {ll_gauss:.4f}")
print(f"  {len(dataset):,} rows")


# ---------------------------------------------------------------------------
# Figure 1: Seasonal Volatility
# ---------------------------------------------------------------------------
print("Figure 1: Seasonal volatility...")
fig, ax = plt.subplots(figsize=(6.5, 3))
buckets = seasonal_df["bucket"].values
sigma_tod = seasonal_df["sigma_tod"].values
hours = buckets * 5 / 60  # 5-minute buckets -> hours

ax.plot(hours, sigma_tod * 1e4, color=C_MODEL, lw=1.5)
ax.set_xlabel("Hour (UTC)")
ax.set_ylabel(r"$\sigma_{\mathrm{tod}}$ ($\times 10^{-4}$ / $\sqrt{\mathrm{s}}$)")
ax.set_title("Seasonal Volatility Curve")
ax.set_xlim(0, 24)
ax.set_xticks(range(0, 25, 3))

# Session markers (ET hours -> UTC: ET+5 in winter)
for h_et, label in [(9.5, "US Open"), (16, "US Close")]:
    h_utc = h_et + 5  # EST -> UTC
    ax.axvline(h_utc, color=C_ACCENT, ls="--", lw=0.8, alpha=0.7)
    ax.text(h_utc + 0.2, ax.get_ylim()[1] * 0.95, label, fontsize=8,
            color=C_ACCENT, va="top")

ax.grid(True, alpha=0.2)
fig.savefig(OUTPUT_DIR / "fig_seasonal_vol.png")
plt.close()


# ---------------------------------------------------------------------------
# Figure 2: Model Comparison
# ---------------------------------------------------------------------------
print("Figure 2: Model comparison...")
fig, ax = plt.subplots(figsize=(5, 3.5))

models = ["Baseline\n(constant)", "Gaussian\n(QLIKE)"]
lls = [ll_baseline, ll_gauss]
improvements = [0, (ll_baseline - ll_gauss) / ll_baseline * 100]
colors = [C_GAUSS, C_MODEL]

bars = ax.bar(models, lls, color=colors, width=0.55, edgecolor="white", linewidth=1.5)
ax.set_ylabel("Binary Log-Loss")
ax.set_title("Model Comparison")

for bar, imp in zip(bars, improvements):
    if imp > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.015,
                f"$-${imp:.1f}%", ha="center", va="top", fontsize=9,
                fontweight="bold", color="white")

ax.set_ylim(0.4, 0.75)
ax.grid(True, alpha=0.2, axis="y")
fig.savefig(OUTPUT_DIR / "fig_model_comparison.png")
plt.close()


# ---------------------------------------------------------------------------
# Figure 3: Variance Ratios
# ---------------------------------------------------------------------------
print("Figure 3: Variance ratios...")
fig, ax = plt.subplots(figsize=(5, 3.5))

log_return_sq = np.log(S_T / S) ** 2

tau_edges = [0, 5, 15, 30, 60]
tau_labels = ["0-5", "5-15", "15-30", "30-60"]
ratios = []
for i in range(len(tau_edges) - 1):
    mask = (tau_min >= tau_edges[i]) & (tau_min < tau_edges[i + 1])
    ratios.append(np.mean(log_return_sq[mask]) / np.mean(v_pred[mask]))

bars = ax.bar(tau_labels, ratios, color=C_MODEL, width=0.55, alpha=0.85)
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.set_xlabel(r"$\tau$ bucket (minutes)")
ax.set_ylabel("Variance Ratio (realized / predicted)")
ax.set_title("QLIKE Variance Calibration")
ax.set_ylim(0.7, 1.3)

for bar, r in zip(bars, ratios):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{r:.3f}", ha="center", va="bottom", fontsize=9)

ax.grid(True, alpha=0.2, axis="y")
fig.savefig(OUTPUT_DIR / "fig_variance_ratios.png")
plt.close()


# ---------------------------------------------------------------------------
# Figure 4: QQ Plot (Gaussian only)
# ---------------------------------------------------------------------------
print("Figure 4: QQ plot...")
fig, ax = plt.subplots(figsize=(4.5, 4.5))

z_sorted = np.sort(z)
n = len(z_sorted)
probs = np.linspace(0.5 / n, 1 - 0.5 / n, n)

# Gaussian quantiles
gauss_q = norm.ppf(probs)

step = max(1, n // 1500)
ax.scatter(gauss_q[::step], z_sorted[::step], s=3, alpha=0.3, color=C_MODEL,
           label="Gaussian", rasterized=True)

lim = 5.5
ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
ax.set_xlabel("Theoretical quantiles (Gaussian)")
ax.set_ylabel("Empirical quantiles ($z$)")
ax.set_title("QQ Plot")
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
ax.legend(loc="upper left", framealpha=0.9)
ax.grid(True, alpha=0.15)
fig.savefig(OUTPUT_DIR / "fig_qq_plot.png")
plt.close()


# ---------------------------------------------------------------------------
# Figure 5: Tail Coverage (Gaussian only)
# ---------------------------------------------------------------------------
print("Figure 5: Tail coverage...")
fig, ax = plt.subplots(figsize=(5.5, 3.5))

thresholds = np.linspace(0.5, 4.5, 80)
tail_empirical = [np.mean(np.abs(z) > c) * 100 for c in thresholds]
tail_gauss = [(1 - norm.cdf(c) + norm.cdf(-c)) * 100 for c in thresholds]

ax.semilogy(thresholds, tail_empirical, color=C_EMP, lw=2, label="Empirical")
ax.semilogy(thresholds, tail_gauss, color=C_GAUSS, lw=2, ls="--", label="Gaussian")

ax.set_xlabel("Threshold $c$")
ax.set_ylabel(r"$P(|z| > c)$  [%]")
ax.set_title("Tail Exceedance Probability")
ax.legend(framealpha=0.9)
ax.grid(True, alpha=0.2)
ax.set_xlim(0.5, 4.5)
fig.savefig(OUTPUT_DIR / "fig_tail_coverage.png")
plt.close()


# ---------------------------------------------------------------------------
# Figure 6: Calibration Curve (Gaussian only)
# ---------------------------------------------------------------------------
print("Figure 6: Calibration curve...")
fig, ax = plt.subplots(figsize=(5, 5))

ax.plot([0, 1], [0, 1], "k--", lw=0.8)

# Equal-mass bins
n_bins = 20
order = np.argsort(p_gauss)
p_sorted = p_gauss[order]
y_sorted = y[order]
bin_size = len(p_gauss) // n_bins

bp, ba, bse = [], [], []
for b in range(n_bins):
    lo = b * bin_size
    hi = (b + 1) * bin_size if b < n_bins - 1 else len(p_gauss)
    n = hi - lo
    p_mean = np.mean(p_sorted[lo:hi])
    y_mean = np.mean(y_sorted[lo:hi])
    se = np.sqrt(y_mean * (1 - y_mean) / n) if n > 1 else 0
    bp.append(p_mean)
    ba.append(y_mean)
    bse.append(se)

bp, ba, bse = np.array(bp), np.array(ba), np.array(bse)

ax.fill_between(bp, ba - 1.96 * bse, ba + 1.96 * bse, alpha=0.2, color=C_MODEL)
ax.plot(bp, ba, "o-", color=C_MODEL, markersize=5, lw=1.5)

imp = (ll_baseline - ll_gauss) / ll_baseline * 100

ax.set_xlabel("Predicted $P(\\mathrm{Up})$")
ax.set_ylabel("Observed frequency")
ax.set_title(f"Gaussian (QLIKE)\nLL = {ll_gauss:.4f} ($-${imp:.1f}\\%)")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect("equal")
ax.grid(True, alpha=0.15)

fig.savefig(OUTPUT_DIR / "fig_calibration_curve.png")
plt.close()

print(f"\nAll figures saved to {OUTPUT_DIR}/")
