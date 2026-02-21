"""Calibrate the Gaussian model to match Polymarket prices.

Fits (c, alpha, k0, k1) to minimize cross-entropy between model P(Up)
and Polymarket mid P(Up). Finds the implied vol parameters from
Polymarket's pricing.

Usage:
    python pricing/calibrate_to_pm.py
"""

import sys
sys.path.insert(0, ".")

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

from pricing.models import get_model
from pricing.calibrate import log_loss


DATA_DIR = Path("pricing/output")


def _market_weighted_mean(values, market_ids):
    """Average within each market, then average across markets."""
    unique = np.unique(market_ids)
    return float(np.mean([np.mean(values[market_ids == m]) for m in unique]))


def cross_entropy(target, pred):
    """Cross-entropy with soft labels (not binary)."""
    pred = np.clip(pred, 1e-9, 1 - 1e-9)
    return -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))


def main():
    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    dataset = pd.read_parquet(DATA_DIR / "calibration_dataset.parquet")
    df = dataset[dataset["pm_mid"].notna()].copy()
    print(f"Dataset: {len(df):,} rows with PM data ({df['market_id'].nunique()} markets)")

    S = df["S"].values.astype(np.float64)
    K = df["K"].values.astype(np.float64)
    tau = df["tau"].values.astype(np.float64)
    y = df["y"].values.astype(np.float64)
    pm = np.clip(df["pm_mid"].values.astype(np.float64), 1e-3, 1 - 1e-3)
    market_ids = df["market_id"].values

    model = get_model("gaussian")
    features = {f: df[f].values.astype(np.float64) for f in model.required_features()}

    # ------------------------------------------------------------------
    # Optimize: minimize cross-entropy(pm, model.predict)
    # ------------------------------------------------------------------
    x0 = [model.initial_params()[n] for n in model.param_names()]
    bounds = [model.param_bounds()[n] for n in model.param_names()]

    def obj_fn(x):
        params = dict(zip(model.param_names(), x))
        p = model.predict(params, S, K, tau, features)
        scores = -(pm * np.log(np.maximum(p, 1e-9)) + (1 - pm) * np.log(np.maximum(1 - p, 1e-9)))
        return _market_weighted_mean(scores, market_ids)

    print(f"\nFitting {model.name} to Polymarket prices ({len(model.param_names())} params) ...")
    res = minimize(obj_fn, x0, method="L-BFGS-B", bounds=bounds)
    pm_params = dict(zip(model.param_names(), [float(v) for v in res.x]))

    # ------------------------------------------------------------------
    # Load QLIKE params for comparison
    # ------------------------------------------------------------------
    with open(DATA_DIR / "gaussian_vol_params.json") as f:
        qlike_raw = json.load(f)
    qlike_params = {k: qlike_raw[k] for k in model.param_names()}

    # ------------------------------------------------------------------
    # Evaluate both
    # ------------------------------------------------------------------
    p_pm = model.predict(pm_params, S, K, tau, features)
    p_qlike = model.predict(qlike_params, S, K, tau, features)

    ce_pm = cross_entropy(pm, p_pm)
    ce_qlike = cross_entropy(pm, p_qlike)

    ll_pm_cal = log_loss(y, p_pm)
    ll_qlike_cal = log_loss(y, p_qlike)
    ll_baseline = log_loss(y, np.full_like(y, y.mean()))

    mse_pm = float(np.mean((p_pm - pm) ** 2))
    mse_qlike = float(np.mean((p_qlike - pm) ** 2))

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"PM-IMPLIED vs QLIKE PARAMETERS")
    print(f"{'='*65}")
    print(f"  {'Param':>8s}  {'PM-implied':>12s}  {'QLIKE':>12s}  {'Delta':>12s}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*12}")
    for n in model.param_names():
        delta = pm_params[n] - qlike_params[n]
        print(f"  {n:>8s}  {pm_params[n]:+12.6f}  {qlike_params[n]:+12.6f}  {delta:+12.6f}")

    print(f"\n{'='*65}")
    print(f"ACCURACY COMPARISON")
    print(f"{'='*65}")
    print(f"\n  Cross-entropy vs PM prices (lower = closer to PM):")
    print(f"    PM-calibrated:  {ce_pm:.6f}")
    print(f"    QLIKE-calibrated: {ce_qlike:.6f}")

    print(f"\n  MSE vs PM prices:")
    print(f"    PM-calibrated:  {mse_pm:.6f}")
    print(f"    QLIKE-calibrated: {mse_qlike:.6f}")

    print(f"\n  Binary log-loss vs outcomes Y (lower = better predictor):")
    print(f"    Baseline:         {ll_baseline:.6f}")
    imp_pm = (ll_baseline - ll_pm_cal) / ll_baseline * 100
    imp_qlike = (ll_baseline - ll_qlike_cal) / ll_baseline * 100
    print(f"    PM-calibrated:    {ll_pm_cal:.6f}  ({imp_pm:+.1f}%)")
    print(f"    QLIKE-calibrated: {ll_qlike_cal:.6f}  ({imp_qlike:+.1f}%)")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out = {
        "model": model.name,
        "objective": "cross_entropy_vs_pm",
        **pm_params,
        "cross_entropy_vs_pm": ce_pm,
        "binary_log_loss": ll_pm_cal,
        "n_samples": len(df),
        "n_markets": df["market_id"].nunique(),
    }
    out_path = DATA_DIR / "gaussian_pm_params.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved PM-implied params to {out_path}")


if __name__ == "__main__":
    main()
