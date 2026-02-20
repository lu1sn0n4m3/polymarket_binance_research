"""Comprehensive comparison: model prices vs Polymarket prices.

Analyzes:
  1. Overall accuracy (log-loss) — model vs PM vs baseline
  2. Conditional accuracy by tau, moneyness, hour, staleness
  3. Bias analysis: E[p - y] for model and PM
  4. When the model prices better / worse than PM
  5. Price disagreement analysis

Usage:
    python pricing/analyze_vs_polymarket.py
"""

import sys
sys.path.insert(0, ".")

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm

from pricing.models import get_model

# =====================================================================
# Load data & compute predictions
# =====================================================================
DATA_DIR = Path("pricing/output")
dataset = pd.read_parquet(DATA_DIR / "calibration_dataset.parquet")

# Filter to rows with PM data
df = dataset[dataset["pm_mid"].notna()].copy()
print(f"Dataset: {len(df):,} rows with Polymarket data ({df['market_id'].nunique()} markets)")

# Load model and params
model = get_model("gaussian")

with open(DATA_DIR / "gaussian_vol_params.json") as f:
    vol_params = json.load(f)

S = df["S"].values.astype(np.float64)
K = df["K"].values.astype(np.float64)
tau = df["tau"].values.astype(np.float64)
y = df["y"].values.astype(np.float64)
pm = df["pm_mid"].values.astype(np.float64)
features = {f: df[f].values.astype(np.float64) for f in model.required_features()}

model_params = {k: vol_params[k] for k in model.param_names()}

p_model = model.predict(model_params, S, K, tau, features)

# Clip PM too (it can be 0 or 1 on thin markets)
pm_clipped = np.clip(pm, 1e-3, 1 - 1e-3)

# Moneyness: k = ln(K/S), positive = OTM call (S < K)
k = np.log(K / S)
tau_min = tau / 60.0

# =====================================================================
# Helper
# =====================================================================
def log_loss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def log_loss_per_row(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

# =====================================================================
# 1. Overall comparison
# =====================================================================
ll_baseline = log_loss(y, np.full_like(y, y.mean()))
ll_model = log_loss(y, p_model)
ll_pm = log_loss(y, pm_clipped)

print(f"\n{'='*65}")
print(f"OVERALL LOG-LOSS COMPARISON")
print(f"{'='*65}")
print(f"  Baseline (constant):   {ll_baseline:.6f}")
print(f"  Model (Gaussian):      {ll_model:.6f}  ({(ll_baseline-ll_model)/ll_baseline*100:+.1f}%)")
print(f"  Polymarket:            {ll_pm:.6f}  ({(ll_baseline-ll_pm)/ll_baseline*100:+.1f}%)")
print(f"  Model vs PM:           {ll_model - ll_pm:+.6f} {'(model better)' if ll_model < ll_pm else '(PM better)'}")

# =====================================================================
# 2. Per-row: when does model beat PM?
# =====================================================================
ll_model_row = log_loss_per_row(y, p_model)
ll_pm_row = log_loss_per_row(y, pm_clipped)
model_better = ll_model_row < ll_pm_row

print(f"\n{'='*65}")
print(f"ROW-LEVEL: MODEL vs POLYMARKET")
print(f"{'='*65}")
print(f"  Model better: {model_better.sum():,} / {len(model_better):,} ({model_better.mean()*100:.1f}%)")
print(f"  PM better:    {(~model_better).sum():,} / {len(model_better):,} ({(~model_better).mean()*100:.1f}%)")
print(f"  Mean advantage when model wins: {(ll_pm_row[model_better] - ll_model_row[model_better]).mean():.4f}")
print(f"  Mean advantage when PM wins:    {(ll_model_row[~model_better] - ll_pm_row[~model_better]).mean():.4f}")

# =====================================================================
# 3. Market-level: who wins more markets?
# =====================================================================
df["ll_model"] = ll_model_row
df["ll_pm"] = ll_pm_row
market_ll = df.groupby("market_id").agg(
    ll_model=("ll_model", "mean"),
    ll_pm=("ll_pm", "mean"),
    y=("y", "first"),
    n=("y", "count"),
).reset_index()
market_ll["model_wins"] = market_ll["ll_model"] < market_ll["ll_pm"]

print(f"\n{'='*65}")
print(f"MARKET-LEVEL: MODEL vs POLYMARKET")
print(f"{'='*65}")
print(f"  Model wins: {market_ll['model_wins'].sum()} / {len(market_ll)} markets ({market_ll['model_wins'].mean()*100:.1f}%)")
print(f"  PM wins:    {(~market_ll['model_wins']).sum()} / {len(market_ll)} markets ({(~market_ll['model_wins']).mean()*100:.1f}%)")

# =====================================================================
# 4. Conditional log-loss by tau
# =====================================================================
tau_edges = [0, 5, 10, 20, 30, 45, 60]
tau_labels = [f"{tau_edges[i]}-{tau_edges[i+1]}" for i in range(len(tau_edges)-1)]

print(f"\n{'='*65}")
print(f"LOG-LOSS BY TAU (minutes)")
print(f"{'='*65}")
print(f"  {'tau':>8s}  {'Model':>8s}  {'PM':>8s}  {'Delta':>8s}  {'Winner':>8s}  {'n':>6s}")
print(f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}")

ll_by_tau_model = []
ll_by_tau_pm = []
for i in range(len(tau_edges)-1):
    mask = (tau_min >= tau_edges[i]) & (tau_min < tau_edges[i+1])
    if mask.sum() == 0:
        continue
    ll_m = log_loss(y[mask], p_model[mask])
    ll_p = log_loss(y[mask], pm_clipped[mask])
    delta = ll_m - ll_p
    winner = "Model" if delta < 0 else "PM"
    print(f"  {tau_labels[i]:>8s}  {ll_m:8.4f}  {ll_p:8.4f}  {delta:+8.4f}  {winner:>8s}  {mask.sum():6d}")
    ll_by_tau_model.append(ll_m)
    ll_by_tau_pm.append(ll_p)

# =====================================================================
# 5. Conditional log-loss by |moneyness|
# =====================================================================
abs_k_pct = np.abs(k) * 100  # in percent
money_edges = [0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
money_labels = [f"{money_edges[i]}-{money_edges[i+1]}%" for i in range(len(money_edges)-1)]

print(f"\n{'='*65}")
print(f"LOG-LOSS BY |MONEYNESS| (|ln(K/S)| %)")
print(f"{'='*65}")
print(f"  {'|k|':>10s}  {'Model':>8s}  {'PM':>8s}  {'Delta':>8s}  {'Winner':>8s}  {'n':>6s}")
print(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}")
for i in range(len(money_edges)-1):
    mask = (abs_k_pct >= money_edges[i]) & (abs_k_pct < money_edges[i+1])
    if mask.sum() < 50:
        continue
    ll_m = log_loss(y[mask], p_model[mask])
    ll_p = log_loss(y[mask], pm_clipped[mask])
    delta = ll_m - ll_p
    winner = "Model" if delta < 0 else "PM"
    print(f"  {money_labels[i]:>10s}  {ll_m:8.4f}  {ll_p:8.4f}  {delta:+8.4f}  {winner:>8s}  {mask.sum():6d}")

# =====================================================================
# 6. Conditional log-loss by hour (ET)
# =====================================================================
hour_et = df["hour_et"].values
print(f"\n{'='*65}")
print(f"LOG-LOSS BY HOUR (ET)")
print(f"{'='*65}")
print(f"  {'Hour':>6s}  {'Model':>8s}  {'PM':>8s}  {'Delta':>8s}  {'Winner':>8s}  {'n':>6s}")
print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}")
for h in range(24):
    mask = hour_et == h
    if mask.sum() < 50:
        continue
    ll_m = log_loss(y[mask], p_model[mask])
    ll_p = log_loss(y[mask], pm_clipped[mask])
    delta = ll_m - ll_p
    winner = "Model" if delta < 0 else "PM"
    print(f"  {h:>4d}h  {ll_m:8.4f}  {ll_p:8.4f}  {delta:+8.4f}  {winner:>8s}  {mask.sum():6d}")

# =====================================================================
# 7. Bias analysis: E[p - y | X]
# =====================================================================
bias_model = p_model - y
bias_pm = pm_clipped - y

print(f"\n{'='*65}")
print(f"BIAS ANALYSIS: E[p - y]  (positive = overestimates P(up))")
print(f"{'='*65}")
print(f"  Overall model bias:  {np.mean(bias_model):+.4f}")
print(f"  Overall PM bias:     {np.mean(bias_pm):+.4f}")

# Bias by predicted probability bin
print(f"\n  Bias by predicted probability (model):")
print(f"  {'p_model':>10s}  {'E[p-y]':>8s}  {'E[pm-y]':>8s}  {'n':>6s}")
print(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*6}")
prob_edges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for i in range(len(prob_edges)-1):
    mask = (p_model >= prob_edges[i]) & (p_model < prob_edges[i+1])
    if mask.sum() < 50:
        continue
    label = f"{prob_edges[i]:.1f}-{prob_edges[i+1]:.1f}"
    print(f"  {label:>10s}  {np.mean(bias_model[mask]):+8.4f}  {np.mean(bias_pm[mask]):+8.4f}  {mask.sum():6d}")

# =====================================================================
# 8. Price disagreement analysis
# =====================================================================
disagreement = p_model - pm_clipped

print(f"\n{'='*65}")
print(f"PRICE DISAGREEMENT: model - PM")
print(f"{'='*65}")
print(f"  Mean:    {np.mean(disagreement):+.4f}")
print(f"  Median:  {np.median(disagreement):+.4f}")
print(f"  Std:     {np.std(disagreement):.4f}")
print(f"  |diff| > 5pp:  {np.mean(np.abs(disagreement) > 0.05)*100:.1f}%")
print(f"  |diff| > 10pp: {np.mean(np.abs(disagreement) > 0.10)*100:.1f}%")
print(f"  |diff| > 20pp: {np.mean(np.abs(disagreement) > 0.20)*100:.1f}%")

# When model and PM disagree a lot, who is right?
big_disagree = np.abs(disagreement) > 0.10
if big_disagree.sum() > 0:
    print(f"\n  When |model - PM| > 10pp ({big_disagree.sum():,} rows):")
    ll_m_big = log_loss(y[big_disagree], p_model[big_disagree])
    ll_p_big = log_loss(y[big_disagree], pm_clipped[big_disagree])
    print(f"    Model LL: {ll_m_big:.4f}")
    print(f"    PM LL:    {ll_p_big:.4f}")
    print(f"    Winner:   {'Model' if ll_m_big < ll_p_big else 'PM'} (delta: {ll_m_big - ll_p_big:+.4f})")

# =====================================================================
# 9. Directional signal in disagreement
# =====================================================================
print(f"\n{'='*65}")
print(f"DIRECTIONAL VALUE OF DISAGREEMENT")
print(f"{'='*65}")
# If model > PM, does it tend to be right (y=1 more often)?
model_higher = disagreement > 0.05
model_lower = disagreement < -0.05
if model_higher.sum() > 0:
    print(f"  Model > PM by >5pp:  y_mean={y[model_higher].mean():.3f}  (n={model_higher.sum():,})")
    print(f"    → Model says higher, actual up rate = {y[model_higher].mean():.3f}")
if model_lower.sum() > 0:
    print(f"  Model < PM by >5pp:  y_mean={y[model_lower].mean():.3f}  (n={model_lower.sum():,})")
    print(f"    → Model says lower, actual up rate = {y[model_lower].mean():.3f}")

# =====================================================================
# PLOTS
# =====================================================================
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle("Model vs Polymarket: Comprehensive Comparison", fontsize=14, fontweight="bold")

C_MODEL = "#2563eb"
C_PM = "#dc2626"
C_GAUSS = "#94a3b8"

# Panel 1: Overall LL bar chart
ax = axes[0, 0]
names = ["Baseline", "Model", "Polymarket"]
lls = [ll_baseline, ll_model, ll_pm]
colors = [C_GAUSS, C_MODEL, C_PM]
bars = ax.bar(names, lls, color=colors, width=0.6, edgecolor="white", linewidth=1.5)
for bar, ll in zip(bars, lls):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{ll:.4f}", ha="center", va="bottom", fontsize=8)
ax.set_ylabel("Binary Log-Loss")
ax.set_title("Overall Accuracy")
ax.set_ylim(0.35, 0.75)
ax.grid(True, alpha=0.2, axis="y")

# Panel 2: LL by tau
ax = axes[0, 1]
x = np.arange(len(tau_labels))
w = 0.35
ax.bar(x - w/2, ll_by_tau_model, w, color=C_MODEL, label="Model", alpha=0.85)
ax.bar(x + w/2, ll_by_tau_pm, w, color=C_PM, label="PM", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(tau_labels, fontsize=8)
ax.set_xlabel("tau (minutes)")
ax.set_ylabel("Binary Log-Loss")
ax.set_title("Accuracy by Time-to-Expiry")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2, axis="y")

# Panel 3: LL by hour
ax = axes[0, 2]
hours = np.arange(24)
ll_hour_model = []
ll_hour_pm = []
valid_hours = []
for h in hours:
    mask = hour_et == h
    if mask.sum() < 50:
        continue
    valid_hours.append(h)
    ll_hour_model.append(log_loss(y[mask], p_model[mask]))
    ll_hour_pm.append(log_loss(y[mask], pm_clipped[mask]))
ax.plot(valid_hours, ll_hour_model, "o-", color=C_MODEL, markersize=4, label="Model")
ax.plot(valid_hours, ll_hour_pm, "o-", color=C_PM, markersize=4, label="PM")
ax.set_xlabel("Hour (ET)")
ax.set_ylabel("Binary Log-Loss")
ax.set_title("Accuracy by Hour")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# Panel 4: Calibration curve — model
ax = axes[1, 0]
n_bins = 20
for p_pred, label, color in [(p_model, "Model", C_MODEL), (pm_clipped, "PM", C_PM)]:
    order = np.argsort(p_pred)
    p_sorted = p_pred[order]
    y_sorted = y[order]
    bin_size = len(p_pred) // n_bins
    bp, ba = [], []
    for b in range(n_bins):
        lo = b * bin_size
        hi = (b + 1) * bin_size if b < n_bins - 1 else len(p_pred)
        bp.append(np.mean(p_sorted[lo:hi]))
        ba.append(np.mean(y_sorted[lo:hi]))
    ax.plot(bp, ba, "o-", color=color, markersize=4, lw=1.5, label=label)
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("Predicted P(Up)")
ax.set_ylabel("Observed Frequency")
ax.set_title("Calibration Curves")
ax.legend(fontsize=8)
ax.set_aspect("equal")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.grid(True, alpha=0.15)

# Panel 5: Bias E[p - y] by probability bin
ax = axes[1, 1]
bp_model, bp_pm, bp_x = [], [], []
for i in range(len(prob_edges)-1):
    mask = (p_model >= prob_edges[i]) & (p_model < prob_edges[i+1])
    if mask.sum() < 50:
        continue
    bp_x.append((prob_edges[i] + prob_edges[i+1]) / 2)
    bp_model.append(np.mean(bias_model[mask]))
    bp_pm.append(np.mean(bias_pm[mask]))
ax.bar(np.array(bp_x) - 0.025, bp_model, 0.045, color=C_MODEL, alpha=0.85, label="Model")
ax.bar(np.array(bp_x) + 0.025, bp_pm, 0.045, color=C_PM, alpha=0.85, label="PM")
ax.axhline(0, color="k", ls="--", lw=0.8)
ax.set_xlabel("Predicted P(Up) bin")
ax.set_ylabel("E[p - y]  (bias)")
ax.set_title("Bias by Probability Level")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# Panel 6: Disagreement histogram
ax = axes[1, 2]
ax.hist(disagreement * 100, bins=100, color=C_MODEL, alpha=0.7, edgecolor="white", density=True)
ax.axvline(0, color="k", ls="--", lw=1)
ax.axvline(np.mean(disagreement) * 100, color=C_PM, ls="-", lw=2,
           label=f"Mean = {np.mean(disagreement)*100:+.1f}pp")
ax.set_xlabel("Model - PM (percentage points)")
ax.set_ylabel("Density")
ax.set_title("Price Disagreement Distribution")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# Panel 7: LL by moneyness
ax = axes[2, 0]
ll_money_model, ll_money_pm, money_x = [], [], []
for i in range(len(money_edges)-1):
    mask = (abs_k_pct >= money_edges[i]) & (abs_k_pct < money_edges[i+1])
    if mask.sum() < 50:
        continue
    money_x.append(f"{money_edges[i]}-{money_edges[i+1]}%")
    ll_money_model.append(log_loss(y[mask], p_model[mask]))
    ll_money_pm.append(log_loss(y[mask], pm_clipped[mask]))
x = np.arange(len(money_x))
w = 0.35
ax.bar(x - w/2, ll_money_model, w, color=C_MODEL, label="Model", alpha=0.85)
ax.bar(x + w/2, ll_money_pm, w, color=C_PM, label="PM", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(money_x, fontsize=7, rotation=30)
ax.set_xlabel("|ln(K/S)|")
ax.set_ylabel("Binary Log-Loss")
ax.set_title("Accuracy by Moneyness")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2, axis="y")

# Panel 8: Scatter model vs PM
ax = axes[2, 1]
step = max(1, len(p_model) // 3000)
sc = ax.scatter(pm_clipped[::step], p_model[::step], s=2, alpha=0.15,
                c=tau_min[::step], cmap="viridis", rasterized=True)
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("Polymarket Price")
ax.set_ylabel("Model Price")
ax.set_title("Model vs PM Prices")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect("equal")
plt.colorbar(sc, ax=ax, label="tau (min)", shrink=0.8)

# Panel 9: Who wins by |disagreement| bucket
ax = axes[2, 2]
disagree_edges = [0, 0.02, 0.05, 0.10, 0.20, 1.0]
disagree_labels = ["<2pp", "2-5pp", "5-10pp", "10-20pp", ">20pp"]
model_win_rate = []
n_per_bin = []
for i in range(len(disagree_edges)-1):
    mask = (np.abs(disagreement) >= disagree_edges[i]) & (np.abs(disagreement) < disagree_edges[i+1])
    if mask.sum() < 20:
        model_win_rate.append(np.nan)
        n_per_bin.append(0)
        continue
    model_win_rate.append(np.mean(ll_model_row[mask] < ll_pm_row[mask]))
    n_per_bin.append(mask.sum())
bars = ax.bar(disagree_labels, model_win_rate, color=C_MODEL, alpha=0.85)
ax.axhline(0.5, color="k", ls="--", lw=0.8)
for bar, n in zip(bars, n_per_bin):
    if not np.isnan(bar.get_height()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"n={n:,}", ha="center", va="bottom", fontsize=7)
ax.set_xlabel("|Model - PM| bucket")
ax.set_ylabel("Fraction model wins")
ax.set_title("Who Wins by Disagreement Size")
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.2, axis="y")

plt.tight_layout()
out_path = "pricing/output/model_vs_polymarket.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved plot to {out_path}")
plt.close()
