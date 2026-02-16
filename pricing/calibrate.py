"""Generic MLE calibration for any Model.

Minimizes binary log loss via scipy.optimize.minimize (L-BFGS-B).
Includes cluster-robust standard errors for hourly markets.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pricing.models.base import Model, CalibrationResult


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Binary cross-entropy log loss."""
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def clustered_se(residuals: np.ndarray, cluster_ids: np.ndarray) -> float:
    """Cluster-robust standard error for mean(residuals).

    All samples within an hourly market share the same outcome y,
    so naive SE underestimates uncertainty by ~sqrt(cluster_size).
    """
    unique_ids = np.unique(cluster_ids)
    n_clusters = len(unique_ids)
    if n_clusters < 2:
        return np.std(residuals) / np.sqrt(len(residuals))
    cluster_means = np.array([
        np.mean(residuals[cluster_ids == cid]) for cid in unique_ids
    ])
    return np.std(cluster_means, ddof=1) / np.sqrt(n_clusters)


def calibrate(
    model: Model,
    dataset: pd.DataFrame,
    method: str = "L-BFGS-B",
    l2_lambda: float = 0.0,
    output_dir: str | Path | None = "pricing/output",
    verbose: bool = True,
) -> CalibrationResult:
    """Fit model parameters by minimizing log loss.

    Args:
        model: Model instance.
        dataset: DataFrame from build_dataset() with columns
                 [S, K, tau, y, market_id] + model.required_features().
        method: scipy.optimize.minimize method.
        l2_lambda: L2 regularization strength (0 = none).
        output_dir: Where to save params JSON (None = don't save).
        verbose: Print progress and diagnostics.

    Returns:
        CalibrationResult with fitted parameters and metrics.
    """
    # Extract arrays
    S = dataset["S"].values.astype(np.float64)
    K = dataset["K"].values.astype(np.float64)
    tau = dataset["tau"].values.astype(np.float64)
    y = dataset["y"].values.astype(np.float64)
    features = {f: dataset[f].values.astype(np.float64) for f in model.required_features()}

    market_ids = dataset["market_id"].values if "market_id" in dataset.columns else None

    # Baseline: constant rate prediction
    ll_baseline = log_loss(y, np.full_like(y, y.mean()))

    # Build objective
    names = model.param_names()
    x0 = [model.initial_params()[n] for n in names]
    bounds = [model.param_bounds()[n] for n in names]

    def neg_ll(x):
        params = dict(zip(names, x))
        p = model.predict(params, S, K, tau, features)
        p = np.clip(p, 1e-9, 1.0 - 1e-9)
        ll = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        if l2_lambda > 0:
            ll += l2_lambda * sum(v ** 2 for v in params.values())
        return ll

    if verbose:
        print(f"\nCalibrating {model.name} ({len(names)} params, {len(y):,} samples) ...")

    res = minimize(neg_ll, x0, method=method, bounds=bounds)
    fitted = dict(zip(names, [float(v) for v in res.x]))
    model.set_params(fitted)

    ll_final = res.fun
    improvement = (ll_baseline - ll_final) / ll_baseline * 100

    n_markets = dataset["market_id"].nunique() if "market_id" in dataset.columns else 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"CALIBRATION RESULT: {model.name}")
        print(f"{'='*60}")
        print(f"\n  Parameters:")
        for n in names:
            print(f"    {n:10s} = {fitted[n]:+.6f}")
        print(f"\n  Log loss:   {ll_final:.6f}")
        print(f"  Baseline:   {ll_baseline:.6f}")
        print(f"  Improvement: {improvement:+.2f}%")
        print(f"  Samples:    {len(y):,}")
        print(f"  Markets:    {n_markets}")

        # Per-tau-bucket diagnostics
        _print_bucket_diagnostics(model, fitted, S, K, tau, y, features, market_ids)

    result = CalibrationResult(
        model_name=model.name,
        params=fitted,
        log_loss=ll_final,
        log_loss_baseline=ll_baseline,
        improvement_pct=improvement,
        n_samples=len(y),
        n_markets=n_markets,
    )

    # Save params
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        params_path = output_dir / f"{model.name}_params.json"
        params_out = {
            "model": model.name,
            **fitted,
            "log_loss": ll_final,
            "log_loss_baseline": ll_baseline,
            "improvement_pct": improvement,
            "n_samples": len(y),
            "n_markets": n_markets,
        }
        with open(params_path, "w") as f:
            json.dump(params_out, f, indent=2)
        if verbose:
            print(f"\n  Saved params to {params_path}")

    return result


TAU_BUCKET_EDGES_MIN = [0, 5, 15, 30, 60]
TAU_BUCKET_LABELS = ["0-5", "5-15", "15-30", "30-60"]


def _print_bucket_diagnostics(model, params, S, K, tau, y, features, market_ids):
    """Print per-tau and per-vol diagnostics with cluster-robust SE."""
    p_pred = model.predict(params, S, K, tau, features)
    p_pred = np.clip(p_pred, 1e-9, 1.0 - 1e-9)
    e = y - p_pred
    tau_min = tau / 60.0

    print(f"\n  Per-τ-bucket diagnostics (clustered SE):")
    for i in range(len(TAU_BUCKET_EDGES_MIN) - 1):
        lo, hi = TAU_BUCKET_EDGES_MIN[i], TAU_BUCKET_EDGES_MIN[i + 1]
        label = TAU_BUCKET_LABELS[i]
        mask = (tau_min >= lo) & (tau_min < hi)
        n = mask.sum()
        if n > 0:
            ll_bucket = log_loss(y[mask], p_pred[mask])
            bias = np.mean(e[mask])
            se = clustered_se(e[mask], market_ids[mask]) if market_ids is not None else np.std(e[mask]) / np.sqrt(n)
            sig = "*" if abs(bias) > 2 * se else ""
            print(f"    τ ∈ [{label:5s}] min: LL={ll_bucket:.4f}  e={bias:+.4f}±{se:.4f} {sig}  n={n}")

    # Per-sigma_rel quartile
    if "sigma_rel" in features:
        sigma_rel = features["sigma_rel"]
        vol_edges = list(np.quantile(sigma_rel, [0.0, 0.25, 0.5, 0.75, 1.0]))
        print(f"\n  Per-σ_rel quartile diagnostics:")
        for qi, ql in enumerate(["Q1", "Q2", "Q3", "Q4"]):
            lo, hi = vol_edges[qi], vol_edges[qi + 1]
            mask = (sigma_rel >= lo) & (sigma_rel < hi) if qi < 3 else (sigma_rel >= lo)
            if mask.sum() > 0:
                bias = np.mean(e[mask])
                se = clustered_se(e[mask], market_ids[mask]) if market_ids is not None else np.std(e[mask]) / np.sqrt(mask.sum())
                sig = "*" if abs(bias) > 2 * se else ""
                print(f"    {ql} [{lo:.4f}, {hi:.4f}]: e={bias:+.4f}±{se:.4f} {sig}")
