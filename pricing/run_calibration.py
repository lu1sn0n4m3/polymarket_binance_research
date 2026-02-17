"""Two-stage calibration: volatility (QLIKE) + tails (fixed-nu Student-t LL).

Paper workflow:
  Stage 1:  QLIKE -> (c, beta, alpha, lam)
  Stage 2:  Fixed-nu MLE -> nu  (paper Section 6)

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
from pricing.calibrate import calibrate_vol, calibrate_tail_fixed, log_loss
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
# Stage 2: Fixed-nu tail calibration (paper Section 6)
# =====================================================================
model_fixed_t = get_model("fixed_t")
result_fixed = calibrate_tail_fixed(model_fixed_t, dataset, verbose=True)

# =====================================================================
# Summary comparison
# =====================================================================
S = dataset["S"].values.astype(np.float64)
K = dataset["K"].values.astype(np.float64)
tau = dataset["tau"].values.astype(np.float64)
S_T = dataset["S_T"].values.astype(np.float64)
y = dataset["y"].values.astype(np.float64)
features = {f: dataset[f].values.astype(np.float64) for f in model_fixed_t.required_features()}

p_gauss = model_gauss.predict(result_vol.params, S, K, tau, features)
p_fixed = model_fixed_t.predict(result_fixed.params, S, K, tau, features)

ll_baseline = log_loss(y, np.full_like(y, y.mean()))
ll_gauss = log_loss(y, p_gauss)
ll_fixed = log_loss(y, p_fixed)

nu = result_fixed.params["nu"]

print(f"\n{'='*60}")
print(f"BINARY LOG-LOSS COMPARISON")
print(f"{'='*60}")
print(f"  Baseline (constant):  {ll_baseline:.6f}")
print(f"  Gaussian (QLIKE vol): {ll_gauss:.6f}  ({(ll_baseline - ll_gauss)/ll_baseline*100:+.1f}%)")
print(f"  Fixed-t (nu={nu:.1f}):    {ll_fixed:.6f}  ({(ll_baseline - ll_fixed)/ll_baseline*100:+.1f}%)")

# =====================================================================
# Diagnostic plot
# =====================================================================
v_pred = model_fixed_t._variance(tau, features)
z = np.log(S_T / S) / np.sqrt(np.maximum(v_pred, 1e-20))
scale = np.sqrt((nu - 2.0) / nu)
u = np.log(S_T / S) ** 2 / np.maximum(v_pred, 1e-20)

fig, axes = plt.subplots(2, 3, figsize=(17, 11))
fig.suptitle("Variance-First Calibration Diagnostics", fontsize=14, fontweight="bold")

c_t = "#27ae60"
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

# Panel 3: QQ plot
ax = axes[0, 2]
z_sorted = np.sort(z)
n = len(z_sorted)
theoretical = norm.ppf(np.linspace(0.5/n, 1 - 0.5/n, n))
step = max(1, n // 2000)
ax.scatter(theoretical[::step], z_sorted[::step], s=4, alpha=0.3, color=c_gauss, label="Gaussian z")
t_theoretical = student_t.ppf(np.linspace(0.5/n, 1 - 0.5/n, n), df=nu) * scale
ax.scatter(t_theoretical[::step], z_sorted[::step], s=4, alpha=0.3, color=c_t,
           label=f"Fixed-t(nu={nu:.0f})")
lim = 5
ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
ax.set_xlabel("Theoretical quantiles")
ax.set_ylabel("Empirical z")
ax.set_title("QQ Plot")
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.legend(fontsize=7)

# Panel 4: Tail coverage
ax = axes[1, 0]
thresholds = np.linspace(0.5, 4.5, 50)
tail_gauss_th = [(1 - norm.cdf(c) + norm.cdf(-c)) * 100 for c in thresholds]
tail_empirical = [np.mean(np.abs(z) > c) * 100 for c in thresholds]
tail_fixed_t = [2.0 * student_t.sf(c / scale, df=nu) * 100 for c in thresholds]

ax.semilogy(thresholds, tail_empirical, color=c_accent, lw=2, label="Empirical")
ax.semilogy(thresholds, tail_fixed_t, color=c_t, lw=2, label=f"Fixed-t (nu={nu:.0f})")
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
    ["", "Gaussian", f"Fixed-t"],
    ["Binary LL", f"{ll_gauss:.4f}", f"{ll_fixed:.4f}"],
    ["vs baseline",
     f"{(ll_baseline-ll_gauss)/ll_baseline*100:+.1f}%",
     f"{(ll_baseline-ll_fixed)/ll_baseline*100:+.1f}%"],
    ["", "", ""],
    ["Vol params", "c,beta,alpha,lam", "(frozen)"],
    ["Tail params", "(none)", f"nu={nu:.1f}"],
    ["", "", ""],
    ["std(z)", f"{np.std(z):.4f}", ""],
    ["kurtosis", f"{sp_kurtosis(z, fisher=True):.2f}", ""],
    ["P(|z|>2)", f"{np.mean(np.abs(z)>2)*100:.2f}%", ""],
    ["P(|z|>3)", f"{np.mean(np.abs(z)>3)*100:.2f}%", ""],
    ["", "", ""],
    ["E[u]", f"{np.mean(u):.4f}", ""],
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
    table[i, 2].set_facecolor("#d5f5e3")
ax.set_title("Summary", pad=20)

plt.tight_layout()
out_path = "pricing/output/calibration_diagnostics.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved plot to {out_path}")
plt.close()
