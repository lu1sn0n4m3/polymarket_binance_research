"""Compare binary-LL vs QLIKE calibration for the Gaussian model."""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from scipy.stats import kurtosis as sp_kurtosis, norm

from pricing.dataset import DatasetConfig, build_dataset
from pricing.calibrate import calibrate, calibrate_vol, TAU_BUCKET_EDGES_MIN, TAU_BUCKET_LABELS
from pricing.models import get_model

# Build dataset (now includes S_T)
cfg = DatasetConfig(start_date=date(2026, 1, 19), end_date=date(2026, 1, 30))
dataset = build_dataset(cfg)

if dataset.empty:
    print("No data!")
    sys.exit(1)

# --- Calibrate both ---
model_ll = get_model("gaussian")
result_ll = calibrate(model_ll, dataset, verbose=False)

model_qlike = get_model("gaussian")
result_qlike = calibrate_vol(model_qlike, dataset, objective="qlike", verbose=False)

print(f"Binary LL params: {result_ll.params}")
print(f"QLIKE params:     {result_qlike.params}")

# --- Extract arrays ---
S = dataset["S"].values.astype(np.float64)
K = dataset["K"].values.astype(np.float64)
tau = dataset["tau"].values.astype(np.float64)
S_T = dataset["S_T"].values.astype(np.float64)
features = {f: dataset[f].values.astype(np.float64) for f in model_ll.required_features()}

log_return = np.log(S_T / S)
log_return_sq = log_return ** 2
tau_min = tau / 60.0

# Predicted variance for each calibration
var_pred_ll = model_ll.predict_variance(result_ll.params, S, K, tau, features)
var_pred_qlike = model_qlike.predict_variance(result_qlike.params, S, K, tau, features)

z_ll = log_return / np.sqrt(np.maximum(var_pred_ll, 1e-20))
z_qlike = log_return / np.sqrt(np.maximum(var_pred_qlike, 1e-20))


# =====================================================================
# PLOTS
# =====================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Binary LL vs QLIKE Calibration — Gaussian Model", fontsize=14, fontweight="bold")

colors = {"ll": "#e74c3c", "qlike": "#2980b9", "gaussian": "#95a5a6"}

# --- Panel 1: Standardized return histograms ---
ax = axes[0, 0]
bins = np.linspace(-5, 5, 80)
ax.hist(z_ll, bins=bins, density=True, alpha=0.5, color=colors["ll"], label="Binary LL")
ax.hist(z_qlike, bins=bins, density=True, alpha=0.5, color=colors["qlike"], label="QLIKE")
x_gauss = np.linspace(-5, 5, 200)
ax.plot(x_gauss, norm.pdf(x_gauss), "k--", lw=1.5, label="N(0,1)")
ax.set_xlabel("z = log(S_T/S) / sqrt(var_pred)")
ax.set_ylabel("Density")
ax.set_title("Standardized Returns")
ax.legend(fontsize=8)
ax.set_xlim(-5, 5)

# --- Panel 2: Variance ratio by tau bucket ---
ax = axes[0, 1]
bucket_labels = []
ratios_ll = []
ratios_qlike = []
for i in range(len(TAU_BUCKET_EDGES_MIN) - 1):
    lo, hi = TAU_BUCKET_EDGES_MIN[i], TAU_BUCKET_EDGES_MIN[i + 1]
    mask = (tau_min >= lo) & (tau_min < hi)
    if mask.sum() > 0:
        bucket_labels.append(TAU_BUCKET_LABELS[i])
        ratios_ll.append(np.mean(log_return_sq[mask]) / np.mean(var_pred_ll[mask]))
        ratios_qlike.append(np.mean(log_return_sq[mask]) / np.mean(var_pred_qlike[mask]))

x_pos = np.arange(len(bucket_labels))
w = 0.35
ax.bar(x_pos - w/2, ratios_ll, w, color=colors["ll"], alpha=0.8, label="Binary LL")
ax.bar(x_pos + w/2, ratios_qlike, w, color=colors["qlike"], alpha=0.8, label="QLIKE")
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{l} min" for l in bucket_labels])
ax.set_ylabel("Variance Ratio (realized / predicted)")
ax.set_title("Variance Ratio by Time-to-Expiry")
ax.legend(fontsize=8)
ax.set_ylim(0.5, 1.8)

# --- Panel 3: Variance ratio by sigma_rel quartile ---
ax = axes[0, 2]
sigma_rel = features["sigma_rel"]
edges = np.quantile(sigma_rel, [0.0, 0.25, 0.5, 0.75, 1.0])
q_labels = []
q_ratios_ll = []
q_ratios_qlike = []
for qi, ql in enumerate(["Q1", "Q2", "Q3", "Q4"]):
    lo, hi = edges[qi], edges[qi + 1]
    mask = (sigma_rel >= lo) & (sigma_rel < hi) if qi < 3 else (sigma_rel >= lo)
    if mask.sum() > 0:
        q_labels.append(ql)
        q_ratios_ll.append(np.mean(log_return_sq[mask]) / np.mean(var_pred_ll[mask]))
        q_ratios_qlike.append(np.mean(log_return_sq[mask]) / np.mean(var_pred_qlike[mask]))

x_pos = np.arange(len(q_labels))
ax.bar(x_pos - w/2, q_ratios_ll, w, color=colors["ll"], alpha=0.8, label="Binary LL")
ax.bar(x_pos + w/2, q_ratios_qlike, w, color=colors["qlike"], alpha=0.8, label="QLIKE")
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.set_xticks(x_pos)
ax.set_xticklabels(q_labels)
ax.set_ylabel("Variance Ratio")
ax.set_title("Variance Ratio by sigma_rel Quartile")
ax.legend(fontsize=8)
ax.set_ylim(0.5, 1.8)

# --- Panel 4: QQ plot ---
ax = axes[1, 0]
z_sorted_ll = np.sort(z_ll)
z_sorted_qlike = np.sort(z_qlike)
n = len(z_sorted_ll)
theoretical = norm.ppf(np.linspace(0.5/n, 1 - 0.5/n, n))
# Subsample for speed
step = max(1, n // 2000)
ax.scatter(theoretical[::step], z_sorted_ll[::step], s=4, alpha=0.4, color=colors["ll"], label="Binary LL")
ax.scatter(theoretical[::step], z_sorted_qlike[::step], s=4, alpha=0.4, color=colors["qlike"], label="QLIKE")
lim = 5
ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
ax.set_xlabel("Theoretical (N(0,1))")
ax.set_ylabel("Empirical z")
ax.set_title("QQ Plot")
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.legend(fontsize=8)

# --- Panel 5: Tail coverage P(|z| > c) ---
ax = axes[1, 1]
thresholds = np.linspace(0.5, 4.5, 50)
tail_ll = [np.mean(np.abs(z_ll) > c) * 100 for c in thresholds]
tail_qlike = [np.mean(np.abs(z_qlike) > c) * 100 for c in thresholds]
tail_gauss = [(1 - norm.cdf(c) + norm.cdf(-c)) * 100 for c in thresholds]

ax.semilogy(thresholds, tail_ll, color=colors["ll"], lw=2, label="Binary LL")
ax.semilogy(thresholds, tail_qlike, color=colors["qlike"], lw=2, label="QLIKE")
ax.semilogy(thresholds, tail_gauss, color=colors["gaussian"], lw=2, ls="--", label="Gaussian")
ax.set_xlabel("Threshold c")
ax.set_ylabel("P(|z| > c)  [%]")
ax.set_title("Tail Coverage")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 6: Summary stats table ---
ax = axes[1, 2]
ax.axis("off")

stats = [
    ["", "Binary LL", "QLIKE", "Target"],
    ["a0", f"{result_ll.params['a0']:.4f}", f"{result_qlike.params['a0']:.4f}", ""],
    ["a1", f"{result_ll.params['a1']:.4f}", f"{result_qlike.params['a1']:.4f}", ""],
    ["beta", f"{result_ll.params['beta']:.4f}", f"{result_qlike.params['beta']:.4f}", ""],
    ["", "", "", ""],
    ["std(z)", f"{np.std(z_ll):.4f}", f"{np.std(z_qlike):.4f}", "1.0"],
    ["kurtosis", f"{sp_kurtosis(z_ll, fisher=True):.2f}", f"{sp_kurtosis(z_qlike, fisher=True):.2f}", "0.0"],
    ["P(|z|>2)", f"{np.mean(np.abs(z_ll)>2)*100:.2f}%", f"{np.mean(np.abs(z_qlike)>2)*100:.2f}%", "4.55%"],
    ["P(|z|>3)", f"{np.mean(np.abs(z_ll)>3)*100:.2f}%", f"{np.mean(np.abs(z_qlike)>3)*100:.2f}%", "0.27%"],
    ["", "", "", ""],
    ["Binary LL", f"{result_ll.log_loss:.4f}", f"{result_qlike.metadata['obj_value']:.4f}*", ""],
]

table = ax.table(cellText=stats, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.4)

# Style header row
for j in range(4):
    table[0, j].set_text_props(fontweight="bold")
    table[0, j].set_facecolor("#ecf0f1")
# Color the calibration columns
for i in range(1, len(stats)):
    table[i, 1].set_facecolor("#fadbd8")  # light red
    table[i, 2].set_facecolor("#d4e6f1")  # light blue

ax.set_title("Calibration Summary", pad=20)

plt.tight_layout()
out_path = "pricing/output/vol_calibration_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved plot to {out_path}")
plt.close()
