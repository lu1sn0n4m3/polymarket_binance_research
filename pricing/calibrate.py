"""Calibration pipeline for the variance-first binary option pricer.

calibrate_vol() fits variance parameters (c, alpha, k0, k1) by minimizing
the QLIKE scoring rule on realized variance. Uses the Gaussian model.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pricing.models.base import Model, CalibrationResult
from pricing.diagnostics import variance_ratio_diagnostics, tail_diagnostics


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Binary cross-entropy log loss."""
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def clustered_se(residuals: np.ndarray, cluster_ids: np.ndarray) -> float:
    """Cluster-robust standard error for mean(residuals)."""
    unique_ids = np.unique(cluster_ids)
    n_clusters = len(unique_ids)
    if n_clusters < 2:
        return np.std(residuals) / np.sqrt(len(residuals))
    cluster_means = np.array([
        np.mean(residuals[cluster_ids == cid]) for cid in unique_ids
    ])
    return np.std(cluster_means, ddof=1) / np.sqrt(n_clusters)



# ---------------------------------------------------------------------------
# Stage 1: Variance-targeted calibration (QLIKE)
# ---------------------------------------------------------------------------

def _market_weighted_mean(values: np.ndarray, market_ids: np.ndarray) -> float:
    """Average within each market, then average across markets."""
    unique = np.unique(market_ids)
    market_means = np.array([np.mean(values[market_ids == m]) for m in unique])
    return float(np.mean(market_means))


def calibrate_vol(
    model: Model,
    dataset: pd.DataFrame,
    objective: str = "qlike",
    market_weighted: bool = True,
    method: str = "L-BFGS-B",
    output_dir: str | Path | None = "pricing/output",
    verbose: bool = True,
) -> CalibrationResult:
    """Stage 1: Fit volatility parameters by minimizing variance forecast error.

    QLIKE:  mean( log(var_pred) + var_realized / var_pred )
            Minimized when var_pred = E[var_realized].

    Args:
        model: Model instance (must implement predict_variance()).
        dataset: DataFrame with columns [S, K, tau, S_T, market_id] + features.
        objective: "qlike".
        market_weighted: If True, each market hour gets equal weight.
        method: scipy.optimize.minimize method.
        output_dir: Where to save params JSON (None = don't save).
        verbose: Print progress and diagnostics.
    """
    S = dataset["S"].values.astype(np.float64)
    K = dataset["K"].values.astype(np.float64)
    tau = dataset["tau"].values.astype(np.float64)
    S_T = dataset["S_T"].values.astype(np.float64)
    features = {f: dataset[f].values.astype(np.float64) for f in model.required_features()}
    market_ids = dataset["market_id"].values if "market_id" in dataset.columns else None

    log_return = np.log(S_T / S)
    log_return_sq = log_return ** 2

    valid = log_return_sq > 1e-20
    if not valid.all() and verbose:
        print(f"  Filtering {(~valid).sum()} zero-return rows for objective")

    all_names = model.param_names()
    free_names = all_names

    init = model.initial_params()
    x0 = [init[n] for n in free_names]
    bounds = [model.param_bounds()[n] for n in free_names]

    def _full_params(x):
        return dict(zip(free_names, x))

    def obj_fn(x):
        params = _full_params(x)
        var_pred = model.predict_variance(params, S, K, tau, features)
        var_pred = np.maximum(var_pred, 1e-20)

        if objective == "qlike":
            scores = np.log(var_pred) + log_return_sq / var_pred
            if market_weighted and market_ids is not None:
                return _market_weighted_mean(scores, market_ids)
            return float(np.mean(scores))

        raise ValueError(f"Unknown objective: {objective}")

    # Analytic gradient (paper Section 9.1) if model supports it
    jac = None
    if objective == "qlike":
        test_grad = model.qlike_gradient(
            _full_params(x0), S, K, tau, features, log_return_sq
        )
        if test_grad is not None:
            def jac_fn(x):
                params = _full_params(x)
                grad_dict = model.qlike_gradient(params, S, K, tau, features, log_return_sq)
                grad_vec = []
                for n in free_names:
                    g = grad_dict[n]
                    if market_weighted and market_ids is not None:
                        grad_vec.append(_market_weighted_mean(g, market_ids))
                    else:
                        grad_vec.append(float(np.mean(g)))
                return np.array(grad_vec)
            jac = jac_fn

    if verbose:
        grad_str = "analytic" if jac is not None else "numeric"
        print(f"\n[Stage 1] Calibrating {model.name} [vol/{objective}, {grad_str} grad] "
              f"({len(free_names)} params, {len(S):,} samples) ...")

    res = minimize(obj_fn, x0, method=method, bounds=bounds, jac=jac)
    fitted = dict(zip(free_names, [float(v) for v in res.x]))
    model.set_params(fitted)

    obj_final = res.fun
    n_markets = dataset["market_id"].nunique() if "market_id" in dataset.columns else 0

    mean_var = float(np.mean(log_return_sq))
    obj_baseline = float(np.log(mean_var) + 1.0)

    if verbose:
        print(f"\n{'='*60}")
        print(f"STAGE 1 — VOL CALIBRATION: {model.name} [{objective}]")
        print(f"{'='*60}")
        print(f"\n  Parameters:")
        for n in all_names:
            print(f"    {n:10s} = {fitted[n]:+.6f}")
        print(f"\n  {objective}: {obj_final:.6f}  (baseline: {obj_baseline:.6f})")
        print(f"  Samples: {len(S):,}  Markets: {n_markets}")

        # E(u|X) conditional bias diagnostics (paper Section 3)
        var_pred = model.predict_variance(fitted, S, K, tau, features)
        variance_ratio_diagnostics(dataset, var_pred, n_bins=10, verbose=True)

        # Tail exceedance summary (Gaussian only at this stage)
        z = log_return / np.sqrt(np.maximum(var_pred, 1e-20))
        tail_diagnostics(z, verbose=True)

    y = dataset["y"].values.astype(np.float64) if "y" in dataset.columns else None
    ll_final = None
    if y is not None:
        p_pred = model.predict(fitted, S, K, tau, features)
        ll_final = log_loss(y, p_pred)
        if verbose:
            ll_baseline = log_loss(y, np.full_like(y, y.mean()))
            imp = (ll_baseline - ll_final) / ll_baseline * 100
            print(f"\n  Binary LL (for reference): {ll_final:.6f}  (baseline: {ll_baseline:.6f}, {imp:+.1f}%)")

    result = CalibrationResult(
        model_name=model.name,
        params=fitted,
        log_loss=ll_final if ll_final is not None else obj_final,
        log_loss_baseline=obj_baseline,
        improvement_pct=0.0,
        n_samples=len(S),
        n_markets=n_markets,
        metadata={"objective": objective, "obj_value": obj_final, "obj_baseline": obj_baseline},
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        params_path = output_dir / f"{model.name}_vol_params.json"
        params_out = {
            "model": model.name,
            "objective": objective,
            **fitted,
            f"{objective}": obj_final,
            f"{objective}_baseline": obj_baseline,
            "n_samples": len(S),
            "n_markets": n_markets,
        }
        if ll_final is not None:
            params_out["binary_log_loss"] = ll_final
        with open(params_path, "w") as f:
            json.dump(params_out, f, indent=2)
        if verbose:
            print(f"\n  Saved params to {params_path}")

    return result
