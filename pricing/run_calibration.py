"""Two-stage calibration: volatility (QLIKE) + tails (Student-t LL).

Paper workflow:
  Stage 1:  QLIKE -> (c, beta, alpha, lam)
  Stage 2a: Fixed-nu MLE -> nu  (paper base model, Section 6)
  Stage 2b: Adaptive-nu MLE -> (b0, b_stale, b_sess, b_tau, nu_max)  (extension, Section 10.1)

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
from pricing.calibrate import calibrate_vol, calibrate_tail, calibrate_tail_fixed, log_loss
from pricing.models import get_model
from pricing.diagnostics import variance_ratio_diagnostics

# =====================================================================
# Build dataset
# =====================================================================
cfg = DatasetConfig(start_date=date(2026, 1, 19), end_date=date(2026, 2, 15))
dataset = build_dataset(cfg)

if dataset.empty:
    print("No data!")
    sys.exit(1)

# =====================================================================
# Stage 1: Volatility calibration (QLIKE with analytic gradients)
# =====================================================================
model_gauss = get_model("gaussian")
result_vol = calibrate_vol(model_gauss, dataset, objective="qlike", verbose=True)

# =====================================================================
# Stage 2a: Fixed-nu tail calibration (paper base model, Section 6)
# =====================================================================
model_fixed_t = get_model("fixed_t")
result_fixed = calibrate_tail_fixed(model_fixed_t, dataset, verbose=True)

# =====================================================================
# Stage 2b: Adaptive-nu tail calibration (extension, Section 10.1)
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
p_fixed = model_fixed_t.predict(result_fixed.params, S, K, tau, features)
p_t = model_t.predict(result_tail.params, S, K, tau, features)

ll_baseline = log_loss(y, np.full_like(y, y.mean()))
ll_gauss = log_loss(y, p_gauss)
ll_fixed = log_loss(y, p_fixed)
ll_t = log_loss(y, p_t)

print(f"\n{'='*60}")
print(f"BINARY LOG-LOSS COMPARISON")
print(f"{'='*60}")
print(f"  Baseline (constant):  {ll_baseline:.6f}")
print(f"  Gaussian (QLIKE vol): {ll_gauss:.6f}  ({(ll_baseline - ll_gauss)/ll_baseline*100:+.1f}%)")
print(f"  Fixed-t (nu={result_fixed.params['nu']:.1f}):    {ll_fixed:.6f}  ({(ll_baseline - ll_fixed)/ll_baseline*100:+.1f}%)")
print(f"  Adaptive-t:           {ll_t:.6f}  ({(ll_baseline - ll_t)/ll_baseline*100:+.1f}%)")

# =====================================================================
# Diagnostic plot
# =====================================================================
v_pred = model_t._variance(tau, features)
z = np.log(S_T / S) / np.sqrt(np.maximum(v_pred, 1e-20))
nu_adaptive = model_t._nu(result_tail.params, tau, features)
nu_fixed = result_fixed.params["nu"]
scale_adaptive = np.sqrt((nu_adaptive - 2.0) / nu_adaptive)
scale_fixed = np.sqrt((nu_fixed - 2.0) / nu_fixed)
u = np.log(S_T / S) ** 2 / np.maximum(v_pred, 1e-20)

fig, axes = plt.subplots(2, 3, figsize=(17, 11))
fig.suptitle("Variance-First Calibration Diagnostics", fontsize=14, fontweight="bold")

c_t = "#2980b9"
c_fixed = "#27ae60"
c_gauss = "#95a5a6"
c_accent = "#e74c3c"

# Panel 1: E[u|tau] — variance ratio by horizon
ax = axes[0, 0]
eu_results = variance_ratio_diagnostics(dataset, v_pred, n_bins=10, verbose=False)
if "tau" in eu_results:
    bins = eu_results["tau"]
    x_vals = [b["mean_x"] for b in bins]
    y_vals = [b["mean_u"] for b in bins]
    se_vals = [1.96 * b["se_u"] for b in bins]
    ax.errorbar(x_vals, y_vals, yerr=se_vals, fmt="o-", color=c_t, markersize=5, capsize=3)
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.set_xlabel("tau (minutes)")
ax.set_ylabel("E[u | tau]")
ax.set_title("E[u | tau]  (target: 1.0)")
ax.set_ylim(0.5, 1.5)

# Panel 2: E[u|tsm] — variance ratio by staleness
ax = axes[0, 1]
if "tsm" in eu_results:
    bins = eu_results["tsm"]
    x_vals = [b["mean_x"] for b in bins]
    y_vals = [b["mean_u"] for b in bins]
    se_vals = [1.96 * b["se_u"] for b in bins]
    ax.errorbar(x_vals, y_vals, yerr=se_vals, fmt="o-", color=c_t, markersize=5, capsize=3)
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.set_xlabel("time_since_move (seconds)")
ax.set_ylabel("E[u | tsm]")
ax.set_title("E[u | tsm]  (target: 1.0)")
ax.set_xscale("symlog", linthresh=10)
ax.set_ylim(0.5, 1.5)

# Panel 3: QQ plot (with both fixed and adaptive t overlays)
ax = axes[0, 2]
z_sorted = np.sort(z)
n = len(z_sorted)
theoretical = norm.ppf(np.linspace(0.5/n, 1 - 0.5/n, n))
step = max(1, n // 2000)
ax.scatter(theoretical[::step], z_sorted[::step], s=4, alpha=0.3, color=c_gauss, label="Gaussian z")
t_theoretical_fixed = student_t.ppf(np.linspace(0.5/n, 1 - 0.5/n, n), df=nu_fixed) * scale_fixed
ax.scatter(t_theoretical_fixed[::step], z_sorted[::step], s=4, alpha=0.3, color=c_fixed,
           label=f"Fixed-t(nu={nu_fixed:.0f})")
nu_med = np.median(nu_adaptive)
scale_med = np.sqrt((nu_med - 2.0) / nu_med)
t_theoretical_adapt = student_t.ppf(np.linspace(0.5/n, 1 - 0.5/n, n), df=nu_med) * scale_med
ax.scatter(t_theoretical_adapt[::step], z_sorted[::step], s=4, alpha=0.3, color=c_t,
           label=f"Adaptive-t(med nu={nu_med:.0f})")
lim = 5
ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
ax.set_xlabel("Theoretical quantiles")
ax.set_ylabel("Empirical z")
ax.set_title("QQ Plot")
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.legend(fontsize=7)

# Panel 4: Tail coverage (with fixed-t added)
ax = axes[1, 0]
thresholds = np.linspace(0.5, 4.5, 50)
tail_gauss_th = [(1 - norm.cdf(c) + norm.cdf(-c)) * 100 for c in thresholds]
tail_empirical = [np.mean(np.abs(z) > c) * 100 for c in thresholds]
tail_fixed_t = [2.0 * student_t.sf(c / scale_fixed, df=nu_fixed) * 100 for c in thresholds]
tail_adaptive_t = [np.mean(2.0 * student_t.sf(c / scale_adaptive, df=nu_adaptive)) * 100 for c in thresholds]

ax.semilogy(thresholds, tail_empirical, color=c_accent, lw=2, label="Empirical")
ax.semilogy(thresholds, tail_fixed_t, color=c_fixed, lw=2, label=f"Fixed-t (nu={nu_fixed:.0f})")
ax.semilogy(thresholds, tail_adaptive_t, color=c_t, lw=2, label="Adaptive-t")
ax.semilogy(thresholds, tail_gauss_th, color=c_gauss, lw=2, ls="--", label="Gaussian")
ax.set_xlabel("Threshold c")
ax.set_ylabel("P(|z| > c)  [%]")
ax.set_title("Tail Coverage")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 5: E[u|sigma_rel]
ax = axes[1, 1]
if "sigma_rel" in eu_results:
    bins = eu_results["sigma_rel"]
    x_vals = [b["mean_x"] for b in bins]
    y_vals = [b["mean_u"] for b in bins]
    se_vals = [1.96 * b["se_u"] for b in bins]
    ax.errorbar(x_vals, y_vals, yerr=se_vals, fmt="o-", color=c_t, markersize=5, capsize=3)
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.set_xlabel("sigma_rel")
ax.set_ylabel("E[u | sigma_rel]")
ax.set_title("E[u | sigma_rel]  (target: 1.0)")
ax.set_ylim(0.5, 1.5)

# Panel 6: Summary table
ax = axes[1, 2]
ax.axis("off")
from scipy.stats import kurtosis as sp_kurtosis
stats = [
    ["", "Gaussian", f"Fixed-t", "Adaptive-t"],
    ["Binary LL", f"{ll_gauss:.4f}", f"{ll_fixed:.4f}", f"{ll_t:.4f}"],
    ["vs baseline",
     f"{(ll_baseline-ll_gauss)/ll_baseline*100:+.1f}%",
     f"{(ll_baseline-ll_fixed)/ll_baseline*100:+.1f}%",
     f"{(ll_baseline-ll_t)/ll_baseline*100:+.1f}%"],
    ["", "", "", ""],
    ["Vol params", "c,beta,alpha,lam", "(frozen)", "(frozen)"],
    ["Tail params", "(none)", f"nu={nu_fixed:.1f}", "b0..b_tau,nu_max"],
    ["", "", "", ""],
    ["std(z)", f"{np.std(z):.4f}", "", ""],
    ["kurtosis", f"{sp_kurtosis(z, fisher=True):.2f}", "", ""],
    ["P(|z|>2)", f"{np.mean(np.abs(z)>2)*100:.2f}%", "", ""],
    ["P(|z|>3)", f"{np.mean(np.abs(z)>3)*100:.2f}%", "", ""],
    ["", "", "", ""],
    ["E[u]", f"{np.mean(u):.4f}", "", ""],
    ["nu", "", f"{nu_fixed:.1f}", f"med={nu_med:.1f}"],
]
table = ax.table(cellText=stats, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 1.3)
for j in range(4):
    table[0, j].set_text_props(fontweight="bold")
    table[0, j].set_facecolor("#ecf0f1")
for i in range(1, len(stats)):
    table[i, 1].set_facecolor("#fadbd8")
    table[i, 2].set_facecolor("#d5f5e3")
    table[i, 3].set_facecolor("#d4e6f1")
ax.set_title("Summary", pad=20)

plt.tight_layout()
out_path = "pricing/output/calibration_diagnostics.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved plot to {out_path}")
plt.close()
