"""Plot predicted vs realized variance for QLIKE-calibrated Gaussian model."""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from scipy.stats import norm

from pricing.dataset import DatasetConfig, build_dataset
from pricing.calibrate import calibrate_vol
from pricing.models import get_model

# Build dataset + calibrate
cfg = DatasetConfig(start_date=date(2026, 1, 19), end_date=date(2026, 1, 30))
dataset = build_dataset(cfg)

model = get_model("gaussian")
result = calibrate_vol(model, dataset, objective="qlike", verbose=False)
print(f"QLIKE params: {result.params}")

# --- Extract arrays ---
S = dataset["S"].values.astype(np.float64)
K = dataset["K"].values.astype(np.float64)
tau = dataset["tau"].values.astype(np.float64)
S_T = dataset["S_T"].values.astype(np.float64)
features = {f: dataset[f].values.astype(np.float64) for f in model.required_features()}
market_ids = dataset["market_id"].values

log_return = np.log(S_T / S)
var_realized = log_return ** 2
var_predicted = model.predict_variance(result.params, S, K, tau, features)

# Compute sigma (vol) versions for more intuitive display
sigma_predicted = np.sqrt(var_predicted / np.maximum(tau, 1e-6))  # per-sqrt(sec)
sigma_realized_mkt = {}  # per-market realized sigma
for m in np.unique(market_ids):
    mask = market_ids == m
    # Use the earliest observation (largest tau) for best single-sample estimate
    idx_max_tau = np.argmax(tau[mask])
    r = log_return[mask][idx_max_tau]
    t = tau[mask][idx_max_tau]
    sigma_realized_mkt[m] = abs(r) / np.sqrt(t)

# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
fig.suptitle("Predicted vs Realized Variance — QLIKE Gaussian",
             fontsize=14, fontweight="bold", y=0.98)

c_main = "#2980b9"
c_accent = "#e74c3c"

# --- Panel 1: Calibration curve on LOG-LOG axes ---
ax = axes[0, 0]
n_bins = 15
# Bin by predicted variance (log-spaced)
log_vp = np.log(var_predicted)
bin_edges = np.percentile(log_vp, np.linspace(0, 100, n_bins + 1))

bin_mean_pred = []
bin_mean_real = []
bin_ci_lo = []
bin_ci_hi = []

for i in range(n_bins):
    mask = (log_vp >= bin_edges[i]) & (log_vp < bin_edges[i + 1] + 1e-10)
    n = mask.sum()
    if n > 10:
        vp = np.mean(var_predicted[mask])
        vr = np.mean(var_realized[mask])
        # Bootstrap CI for mean realized variance
        boot = np.array([np.mean(np.random.choice(var_realized[mask], n, replace=True))
                         for _ in range(500)])
        bin_mean_pred.append(vp)
        bin_mean_real.append(vr)
        bin_ci_lo.append(np.percentile(boot, 2.5))
        bin_ci_hi.append(np.percentile(boot, 97.5))

bin_mean_pred = np.array(bin_mean_pred)
bin_mean_real = np.array(bin_mean_real)
bin_ci_lo = np.array(bin_ci_lo)
bin_ci_hi = np.array(bin_ci_hi)

ax.errorbar(bin_mean_pred, bin_mean_real,
            yerr=[bin_mean_real - bin_ci_lo, bin_ci_hi - bin_mean_real],
            fmt="o", color=c_main, markersize=6, capsize=3, zorder=3)
rng = [min(bin_mean_pred.min(), bin_mean_real.min()) * 0.5,
       max(bin_mean_pred.max(), bin_mean_real.max()) * 2]
ax.plot(rng, rng, "k--", lw=1, alpha=0.5)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Predicted variance")
ax.set_ylabel("Realized variance (binned mean)")
ax.set_title("Calibration Curve (log-log)")
ax.set_aspect("equal")

# --- Panel 2: Market-level scatter colored by sigma_rel ---
ax = axes[0, 1]
unique_markets = np.unique(market_ids)
mkt_pred_var = []
mkt_real_var = []
mkt_sigma_rel = []

for m in unique_markets:
    mask = market_ids == m
    mkt_pred_var.append(np.mean(var_predicted[mask]))
    mkt_real_var.append(np.mean(var_realized[mask]))
    mkt_sigma_rel.append(np.median(features["sigma_rel"][mask]))

mkt_pred_var = np.array(mkt_pred_var)
mkt_real_var = np.array(mkt_real_var)
mkt_sigma_rel = np.array(mkt_sigma_rel)

sc = ax.scatter(mkt_pred_var, mkt_real_var, c=mkt_sigma_rel, cmap="coolwarm",
                s=25, alpha=0.8, edgecolors="k", linewidths=0.3, vmin=0.5, vmax=2.0)
rng2 = [min(mkt_pred_var.min(), mkt_real_var[mkt_real_var > 0].min()) * 0.5,
        max(mkt_pred_var.max(), mkt_real_var.max()) * 2]
ax.plot(rng2, rng2, "k--", lw=1, alpha=0.5)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Mean predicted variance per market")
ax.set_ylabel("Mean realized variance per market")
ax.set_title(f"Market-level ({len(unique_markets)} hours)")
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Median sigma_rel")

# --- Panel 3: Predicted vs realized sigma (annualized bps) through time ---
# Show rolling 24h averages
ax = axes[1, 0]
t_ms = dataset["t"].values
sort_idx = np.argsort(t_ms)
t_sorted = t_ms[sort_idx]
vp_sorted = var_predicted[sort_idx]
vr_sorted = var_realized[sort_idx]
tau_sorted = tau[sort_idx]

# Convert to annualized vol in bps for readability: sigma * sqrt(365*24*3600) * 1e4
ANN = np.sqrt(365.25 * 24 * 3600) * 1e4

# Rolling window (by number of observations, ~500 = ~8 hours)
win = 500
if len(vp_sorted) > win:
    # Cumulative sums for efficient rolling mean
    cs_vp = np.cumsum(vp_sorted / np.maximum(tau_sorted, 1))
    cs_vr = np.cumsum(vr_sorted / np.maximum(tau_sorted, 1))
    cs_vp = np.insert(cs_vp, 0, 0)
    cs_vr = np.insert(cs_vr, 0, 0)

    roll_vp = (cs_vp[win:] - cs_vp[:-win]) / win
    roll_vr = (cs_vr[win:] - cs_vr[:-win]) / win
    roll_t = t_sorted[win-1:]

    # Convert variance-per-sec to annualized vol in bps
    roll_sigma_pred = np.sqrt(np.maximum(roll_vp, 0)) * ANN
    roll_sigma_real = np.sqrt(np.maximum(roll_vr, 0)) * ANN

    # Convert timestamps to days from start
    t_days = (roll_t - roll_t[0]) / (86400 * 1000)

    ax.plot(t_days, roll_sigma_pred, color=c_main, lw=1.5, label="Predicted", alpha=0.9)
    ax.plot(t_days, roll_sigma_real, color=c_accent, lw=1.5, label="Realized", alpha=0.7)
    ax.set_xlabel("Days from start")
    ax.set_ylabel("Annualized vol (bps)")
    ax.set_title(f"Rolling sigma (window={win} obs)")
    ax.legend(fontsize=9)

# --- Panel 4: Variance ratio distribution ---
ax = axes[1, 1]
# Per-market variance ratio
mkt_ratio = mkt_real_var / np.maximum(mkt_pred_var, 1e-20)
log_ratio = np.log(mkt_ratio)

bins = np.linspace(-3, 3, 40)
ax.hist(log_ratio, bins=bins, color=c_main, alpha=0.7, edgecolor="white", density=True)
# Overlay: if perfectly calibrated, log(chi2_1 / 1) distribution
# But with market averaging, closer to normal
ax.axvline(0, color="k", ls="--", lw=1.5)
ax.axvline(np.median(log_ratio), color=c_accent, ls="-", lw=1.5,
           label=f"Median = {np.median(log_ratio):+.2f}")

# Stats annotation
n_over = np.sum(mkt_ratio > 1)
n_under = np.sum(mkt_ratio < 1)
ax.set_xlabel("log(realized var / predicted var)")
ax.set_ylabel("Density")
ax.set_title(f"Market-level ratio: {n_over} over, {n_under} under")
ax.legend(fontsize=9)

plt.tight_layout()
out_path = "pricing/output/variance_pred_vs_realized.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved to {out_path}")
plt.close()
