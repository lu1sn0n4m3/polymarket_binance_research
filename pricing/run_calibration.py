"""Two-stage calibration: volatility (QLIKE) + tails (Student-t LL).

Usage:
    python pricing/run_calibration.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from scipy.stats import norm, t as student_t

from pricing.dataset import DatasetConfig, build_dataset
from pricing.calibrate import calibrate_vol, calibrate_tail, log_loss, TAU_BUCKET_EDGES_MIN, TAU_BUCKET_LABELS
from pricing.models import get_model

# =====================================================================
# Build dataset
# =====================================================================
cfg = DatasetConfig(start_date=date(2026, 1, 19), end_date=date(2026, 1, 30))
dataset = build_dataset(cfg)

if dataset.empty:
    print("No data!")
    sys.exit(1)

# =====================================================================
# Stage 1: Volatility calibration (QLIKE)
# =====================================================================
model_gauss = get_model("gaussian")
result_vol = calibrate_vol(model_gauss, dataset, objective="qlike", verbose=True)

# =====================================================================
# Stage 2: Tail calibration (Student-t LL)
# =====================================================================
model_t = get_model("gaussian_t")
result_tail = calibrate_tail(model_t, dataset, verbose=True)

# =====================================================================
# Summary comparison
# =====================================================================
S = dataset["S"].values.astype(np.float64)
K = dataset["K"].values.astype(np.float64)
tau = dataset["tau"].values.astype(np.float64)
S_T = dataset["S_T"].values.astype(np.float64)
y = dataset["y"].values.astype(np.float64)
features = {f: dataset[f].values.astype(np.float64) for f in model_t.required_features()}

p_gauss = model_gauss.predict(result_vol.params, S, K, tau, features)
p_t = model_t.predict(result_tail.params, S, K, tau, features)

ll_baseline = log_loss(y, np.full_like(y, y.mean()))
ll_gauss = log_loss(y, p_gauss)
ll_t = log_loss(y, p_t)

print(f"\n{'='*60}")
print(f"BINARY LOG-LOSS COMPARISON")
print(f"{'='*60}")
print(f"  Baseline (constant):  {ll_baseline:.6f}")
print(f"  Gaussian (QLIKE vol): {ll_gauss:.6f}  ({(ll_baseline - ll_gauss)/ll_baseline*100:+.1f}%)")
print(f"  Adaptive-t:           {ll_t:.6f}  ({(ll_baseline - ll_t)/ll_baseline*100:+.1f}%)")

# =====================================================================
# Diagnostic plot
# =====================================================================
sigma_eff = model_t._sigma_eff(tau, features)
sqrt_tau = np.sqrt(np.maximum(tau, 1e-6))
z = np.log(S_T / S) / (sigma_eff * sqrt_tau)
nu = model_t._nu(result_tail.params, tau, features)
scale = np.sqrt((nu - 2.0) / nu)
tau_min = tau / 60.0

fig, axes = plt.subplots(2, 3, figsize=(17, 11))
fig.suptitle("Adaptive-t Calibration Diagnostics", fontsize=14, fontweight="bold")

c_t = "#2980b9"
c_gauss = "#95a5a6"
c_accent = "#e74c3c"

# Panel 1: nu histogram
ax = axes[0, 0]
ax.hist(nu, bins=50, color=c_t, alpha=0.7, edgecolor="white", density=True)
ax.axvline(np.median(nu), color=c_accent, ls="-", lw=2, label=f"Median = {np.median(nu):.1f}")
ax.axvline(np.mean(nu), color="k", ls="--", lw=1.5, label=f"Mean = {np.mean(nu):.1f}")
ax.set_xlabel("Degrees of freedom (nu)")
ax.set_ylabel("Density")
ax.set_title("Distribution of nu")
ax.legend(fontsize=9)

# Panel 2: nu vs staleness
ax = axes[0, 1]
tsm = features["time_since_move"]
tsm_bins = np.percentile(tsm, np.linspace(0, 100, 30))
nu_binned, tsm_binned = [], []
for i in range(len(tsm_bins) - 1):
    mask = (tsm >= tsm_bins[i]) & (tsm < tsm_bins[i+1] + 1e-6)
    if mask.sum() > 10:
        tsm_binned.append(np.median(tsm[mask]))
        nu_binned.append(np.median(nu[mask]))
ax.plot(tsm_binned, nu_binned, "o-", color=c_t, markersize=5)
ax.set_xlabel("time_since_move (seconds)")
ax.set_ylabel("Median nu")
ax.set_title("nu vs Staleness")
ax.set_xscale("symlog", linthresh=10)

# Panel 3: QQ plot
ax = axes[0, 2]
z_sorted = np.sort(z)
n = len(z_sorted)
theoretical = norm.ppf(np.linspace(0.5/n, 1 - 0.5/n, n))
step = max(1, n // 2000)
ax.scatter(theoretical[::step], z_sorted[::step], s=4, alpha=0.3, color=c_gauss, label="Gaussian z")
nu_med = np.median(nu)
scale_med = np.sqrt((nu_med - 2.0) / nu_med)
t_theoretical = student_t.ppf(np.linspace(0.5/n, 1 - 0.5/n, n), df=nu_med) * scale_med
ax.scatter(t_theoretical[::step], z_sorted[::step], s=4, alpha=0.3, color=c_t, label=f"t(nu={nu_med:.0f}) z")
lim = 5
ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
ax.set_xlabel("Theoretical quantiles")
ax.set_ylabel("Empirical z")
ax.set_title("QQ Plot")
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.legend(fontsize=8)

# Panel 4: Tail coverage
ax = axes[1, 0]
thresholds = np.linspace(0.5, 4.5, 50)
tail_gauss_th = [(1 - norm.cdf(c) + norm.cdf(-c)) * 100 for c in thresholds]
tail_empirical = [np.mean(np.abs(z) > c) * 100 for c in thresholds]
tail_adaptive_t = [np.mean(2.0 * student_t.sf(c / scale, df=nu)) * 100 for c in thresholds]

ax.semilogy(thresholds, tail_empirical, color=c_accent, lw=2, label="Empirical")
ax.semilogy(thresholds, tail_adaptive_t, color=c_t, lw=2, label="Adaptive-t (model)")
ax.semilogy(thresholds, tail_gauss_th, color=c_gauss, lw=2, ls="--", label="Gaussian")
ax.set_xlabel("Threshold c")
ax.set_ylabel("P(|z| > c)  [%]")
ax.set_title("Tail Coverage")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 5: Variance ratios
ax = axes[1, 1]
log_return_sq = np.log(S_T / S) ** 2
var_pred = sigma_eff ** 2 * tau
bucket_labels_plot, ratios = [], []
for i in range(len(TAU_BUCKET_EDGES_MIN) - 1):
    lo, hi = TAU_BUCKET_EDGES_MIN[i], TAU_BUCKET_EDGES_MIN[i + 1]
    mask = (tau_min >= lo) & (tau_min < hi)
    if mask.sum() > 0:
        bucket_labels_plot.append(TAU_BUCKET_LABELS[i])
        ratios.append(np.mean(log_return_sq[mask]) / np.mean(var_pred[mask]))
ax.bar(range(len(bucket_labels_plot)), ratios, color=c_t, alpha=0.8)
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.set_xticks(range(len(bucket_labels_plot)))
ax.set_xticklabels([f"{l} min" for l in bucket_labels_plot])
ax.set_ylabel("Variance Ratio")
ax.set_title("Variance Ratio by tau")
ax.set_ylim(0.5, 1.5)

# Panel 6: Summary table
ax = axes[1, 2]
ax.axis("off")
from scipy.stats import kurtosis as sp_kurtosis
stats = [
    ["", "Gauss (QLIKE)", "Adaptive-t"],
    ["Binary LL", f"{ll_gauss:.4f}", f"{ll_t:.4f}"],
    ["vs baseline", f"{(ll_baseline-ll_gauss)/ll_baseline*100:+.1f}%",
     f"{(ll_baseline-ll_t)/ll_baseline*100:+.1f}%"],
    ["", "", ""],
    ["Vol params", "a0, a1, beta", "(frozen)"],
    ["Tail params", "(none)", "b0..b_tau, nu_max"],
    ["", "", ""],
    ["std(z)", f"{np.std(z):.4f}", f"{np.std(z):.4f}"],
    ["kurtosis", f"{sp_kurtosis(z, fisher=True):.2f}", f"{sp_kurtosis(z, fisher=True):.2f}"],
    ["P(|z|>2)", f"{np.mean(np.abs(z)>2)*100:.2f}%", f"{np.mean(np.abs(z)>2)*100:.2f}%"],
    ["P(|z|>3)", f"{np.mean(np.abs(z)>3)*100:.2f}%", f"{np.mean(np.abs(z)>3)*100:.2f}%"],
    ["", "", ""],
    ["nu median", "", f"{np.median(nu):.1f}"],
    ["nu range", "", f"[{np.min(nu):.1f}, {np.max(nu):.1f}]"],
]
table = ax.table(cellText=stats, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.3)
for j in range(3):
    table[0, j].set_text_props(fontweight="bold")
    table[0, j].set_facecolor("#ecf0f1")
for i in range(1, len(stats)):
    table[i, 1].set_facecolor("#fadbd8")
    table[i, 2].set_facecolor("#d4e6f1")
ax.set_title("Summary", pad=20)

plt.tight_layout()
out_path = "pricing/output/calibration_diagnostics.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved plot to {out_path}")
plt.close()
