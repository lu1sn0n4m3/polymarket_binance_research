"""Calibration: volatility (QLIKE) for the Gaussian binary option pricer.

Usage:
    python pricing/run_calibration.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from scipy.stats import norm

from pricing.dataset import DatasetConfig, build_dataset
from pricing.calibrate import calibrate_vol, log_loss
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
# Calibrate: Shrinkage diffusion (QLIKE)
# =====================================================================
model_gauss = get_model("gaussian")

print("\n" + "="*60)
print("CALIBRATION: Shrinkage diffusion")
print("="*60)
result_vol = calibrate_vol(
    model_gauss, dataset, objective="qlike",
    verbose=True,
)

# =====================================================================
# Summary
# =====================================================================
S = dataset["S"].values.astype(np.float64)
K = dataset["K"].values.astype(np.float64)
tau = dataset["tau"].values.astype(np.float64)
S_T = dataset["S_T"].values.astype(np.float64)
y = dataset["y"].values.astype(np.float64)
features = {f: dataset[f].values.astype(np.float64) for f in model_gauss.required_features()}

p_gauss = model_gauss.predict(result_vol.params, S, K, tau, features)

ll_baseline = log_loss(y, np.full_like(y, y.mean()))
ll_gauss = log_loss(y, p_gauss)

print(f"\n{'='*60}")
print(f"BINARY LOG-LOSS")
print(f"{'='*60}")
print(f"  Baseline (constant):  {ll_baseline:.6f}")
print(f"  Gaussian (QLIKE vol): {ll_gauss:.6f}  ({(ll_baseline - ll_gauss)/ll_baseline*100:+.1f}%)")

if result_vol.metadata.get("param_cov") is not None:
    cov = np.array(result_vol.metadata["param_cov"])
    names = result_vol.metadata.get("param_names", list(result_vol.params.keys()))
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    print(f"\n{'='*60}")
    print(f"PARAMETER STANDARD ERRORS (Hessian, N={result_vol.n_samples:,})")
    print(f"{'='*60}")
    for n, s in zip(names, se):
        val = result_vol.params[n]
        cv = s / abs(val) * 100 if abs(val) > 1e-8 else float("nan")
        print(f"  {n:10s} = {val:+.6f}  +/-{s:.6f}  (CV = {cv:.1f}%)")

# =====================================================================
# Diagnostic plot
# =====================================================================
v_pred = model_gauss.predict_variance(result_vol.params, S, K, tau, features)
z = np.log(S_T / S) / np.sqrt(np.maximum(v_pred, 1e-20))
u = np.log(S_T / S) ** 2 / np.maximum(v_pred, 1e-20)

fig, axes = plt.subplots(2, 3, figsize=(17, 11))
fig.suptitle("Variance-First Calibration Diagnostics", fontsize=14, fontweight="bold")

c_main = "#27ae60"
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
    ax.errorbar(x_vals, y_vals, yerr=se_vals, fmt="o-", color=c_main, markersize=5, capsize=3)
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
    ax.errorbar(x_vals, y_vals, yerr=se_vals, fmt="o-", color=c_main, markersize=5, capsize=3)
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.set_xlabel("sigma_rel")
ax.set_ylabel("E[u | sigma_rel]")
ax.set_title("E[u | sigma_rel]  (target: 1.0)")
ax.set_ylim(0.5, 1.5)

# Panel 3: QQ plot (Gaussian)
ax = axes[0, 2]
z_sorted = np.sort(z)
n = len(z_sorted)
theoretical = norm.ppf(np.linspace(0.5/n, 1 - 0.5/n, n))
step = max(1, n // 2000)
ax.scatter(theoretical[::step], z_sorted[::step], s=4, alpha=0.3, color=c_gauss, label="Gaussian z")
lim = 5
ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
ax.set_xlabel("Theoretical quantiles")
ax.set_ylabel("Empirical z")
ax.set_title("QQ Plot (Gaussian)")
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.legend(fontsize=7)

# Panel 4: Tail coverage
ax = axes[1, 0]
thresholds = np.linspace(0.5, 4.5, 50)
tail_gauss_th = [(1 - norm.cdf(c) + norm.cdf(-c)) * 100 for c in thresholds]
tail_empirical = [np.mean(np.abs(z) > c) * 100 for c in thresholds]

ax.semilogy(thresholds, tail_empirical, color=c_accent, lw=2, label="Empirical")
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
    ax.errorbar(x_vals, y_vals, yerr=se_vals, fmt="o-", color=c_main, markersize=5, capsize=3)
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
    ["", "Gaussian"],
    ["Binary LL", f"{ll_gauss:.4f}"],
    ["vs baseline",
     f"{(ll_baseline-ll_gauss)/ll_baseline*100:+.1f}%"],
    ["", ""],
    ["Vol params", "c,alpha"],
    ["Shrinkage", "k0,k1"],
    ["", ""],
    ["std(z)", f"{np.std(z):.4f}"],
    ["kurtosis", f"{sp_kurtosis(z, fisher=True):.2f}"],
    ["P(|z|>2)", f"{np.mean(np.abs(z)>2)*100:.2f}%"],
    ["P(|z|>3)", f"{np.mean(np.abs(z)>3)*100:.2f}%"],
    ["", ""],
    ["E[u]", f"{np.mean(u):.4f}"],
]
table = ax.table(cellText=stats, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.3)
for j in range(2):
    table[0, j].set_text_props(fontweight="bold")
    table[0, j].set_facecolor("#ecf0f1")
for i in range(1, len(stats)):
    table[i, 1].set_facecolor("#d5f5e3")
ax.set_title("Summary", pad=20)

plt.tight_layout()
out_path = "pricing/output/calibration_diagnostics.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved plot to {out_path}")
plt.close()
