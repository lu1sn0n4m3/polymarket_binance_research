"""Two-stage calibration: volatility (QLIKE) + tails (fixed-nu Student-t LL).

Paper workflow:
  Stage 1: QLIKE -> (c, alpha, k0, k1) — shrinkage diffusion
  Stage 2: Fixed-nu MLE -> nu  (paper Section 6)

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
from pricing.calibrate import (
    calibrate_vol, calibrate_tail_fixed,
    calibrate_tail_truncated, calibrate_tail_exceedance, log_loss,
)
from pricing.models import get_model
from pricing.diagnostics import variance_ratio_diagnostics

# =====================================================================
# Build dataset
# =====================================================================
cfg = DatasetConfig(start_date=date(2026, 1, 19), end_date=date(2026, 2, 18))
dataset = build_dataset(cfg)

if dataset.empty:
    print("No data!")
    sys.exit(1)

# =====================================================================
# Stage 1: Shrinkage diffusion (QLIKE)
# =====================================================================
model_gauss = get_model("gaussian")

print("\n" + "="*60)
print("STAGE 1: Shrinkage diffusion")
print("="*60)
result_vol = calibrate_vol(
    model_gauss, dataset, objective="qlike",
    verbose=True,
)

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
# Stage 2 variants: tail-targeted nu estimation
# =====================================================================
print(f"\n{'='*60}")
print(f"TAIL-TARGETED NU ESTIMATION")
print(f"{'='*60}")

# Collect all nu variants for comparison
nu_variants = [("Full MLE", nu)]

# Truncated MLE at various cutoffs
for z0 in [1.0, 1.5, 2.0]:
    model_t = get_model("fixed_t")
    res_trunc = calibrate_tail_truncated(model_t, dataset, z_cutoff=z0, verbose=True)
    if not np.isnan(res_trunc["nu"]):
        nu_variants.append((f"Trunc z>{z0:.1f}", res_trunc["nu"]))

# Exceedance matching at various thresholds
for q in [1.5, 2.0, 2.5]:
    model_t = get_model("fixed_t")
    res_exc = calibrate_tail_exceedance(model_t, dataset, threshold=q, verbose=True)
    nu_variants.append((f"Exc q={q:.1f}", res_exc["nu"]))

# Compare all nu variants by |k| bins
k_log = np.log(K / S)
abs_k = np.abs(k_log)
v_pred = model_fixed_t._variance(tau, features)
sqrt_v = np.sqrt(np.maximum(v_pred, 1e-20))

edges = [0, 0.001, 0.003, 0.005, 0.01, 0.02, 0.05, np.inf]
labels_bin = ['0-0.1%', '0.1-0.3%', '0.3-0.5%', '0.5-1%', '1-2%', '2-5%', '>5%']

def compute_ll_by_bins(nu_val):
    """Compute per-bin log-loss for a given nu."""
    s_nu = np.sqrt((nu_val - 2.0) / nu_val)
    p = student_t.cdf(-k_log / (s_nu * sqrt_v), df=nu_val)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    ll_row = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    bin_lls = []
    for i in range(len(labels_bin)):
        mask = (abs_k >= edges[i]) & (abs_k < edges[i+1])
        bin_lls.append(ll_row[mask].mean() if mask.sum() > 0 else np.nan)
    return bin_lls, ll_row.mean()

# Gaussian baseline
p_g = norm.cdf(-k_log / sqrt_v)
p_g = np.clip(p_g, 1e-9, 1 - 1e-9)
ll_g_row = -(y * np.log(p_g) + (1 - y) * np.log(1 - p_g))
gauss_bins = []
for i in range(len(labels_bin)):
    mask = (abs_k >= edges[i]) & (abs_k < edges[i+1])
    gauss_bins.append(ll_g_row[mask].mean() if mask.sum() > 0 else np.nan)

print(f"\n{'='*60}")
print(f"NU VARIANT COMPARISON BY |k| BINS")
print(f"{'='*60}")

# Header
header = f"{'Method':>18s}  {'nu':>6s}  {'Total':>8s}"
for lb in labels_bin:
    header += f"  {lb:>8s}"
print(header)
print("-" * len(header))

# Gaussian row
row = f"{'Gaussian':>18s}  {'inf':>6s}  {ll_g_row.mean():>8.4f}"
for v_b in gauss_bins:
    row += f"  {v_b:>8.4f}" if not np.isnan(v_b) else f"  {'N/A':>8s}"
print(row)

# Each nu variant
for name, nu_v in nu_variants:
    bins_ll, total_ll = compute_ll_by_bins(nu_v)
    row = f"{name:>18s}  {nu_v:>6.2f}  {total_ll:>8.4f}"
    for v_b in bins_ll:
        row += f"  {v_b:>8.4f}" if not np.isnan(v_b) else f"  {'N/A':>8s}"
    print(row)

# Delta vs Gaussian (positive = t is better)
print(f"\nDelta vs Gaussian (positive = t helps):")
header_d = f"{'Method':>18s}  {'nu':>6s}  {'Total':>8s}"
for lb in labels_bin:
    header_d += f"  {lb:>8s}"
print(header_d)
print("-" * len(header_d))

for name, nu_v in nu_variants:
    bins_ll, total_ll = compute_ll_by_bins(nu_v)
    row = f"{name:>18s}  {nu_v:>6.2f}  {ll_g_row.mean() - total_ll:>+8.4f}"
    for gb, tb in zip(gauss_bins, bins_ll):
        d = gb - tb if not (np.isnan(gb) or np.isnan(tb)) else np.nan
        row += f"  {d:>+8.4f}" if not np.isnan(d) else f"  {'N/A':>8s}"
    print(row)

# Bin counts
row_n = f"{'N':>18s}  {'':>6s}  {len(y):>8,}"
for i in range(len(labels_bin)):
    mask = (abs_k >= edges[i]) & (abs_k < edges[i+1])
    row_n += f"  {mask.sum():>8,}"
print(row_n)

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

# Panel 2: E[u|sigma_rel] — variance ratio by relative vol
ax = axes[0, 1]
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

# Panel 5: E[u|hour_et]
ax = axes[1, 1]
if "hour_et" in eu_results:
    bins = eu_results["hour_et"]
    x_vals = [b["mean_x"] for b in bins]
    y_vals = [b["mean_u"] for b in bins]
    se_vals = [1.96 * b["se_u"] for b in bins]
    ax.errorbar(x_vals, y_vals, yerr=se_vals, fmt="o-", color=c_t, markersize=5, capsize=3)
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.set_xlabel("Hour (ET)")
ax.set_ylabel("E[u | hour]")
ax.set_title("E[u | hour_et]  (target: 1.0)")
ax.set_ylim(0.0, 2.0)

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
    ["Vol params", "c,alpha", "(frozen)"],
    ["Shrinkage", "k0,k1", "(frozen)"],
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
