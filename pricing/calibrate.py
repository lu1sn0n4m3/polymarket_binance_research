"""Two-stage calibration pipeline for the adaptive-t binary option pricer.

Stage 1 — Volatility (QLIKE):
    calibrate_vol() fits sigma_eff parameters (a0, a1, beta) by minimizing
    the QLIKE scoring rule on realized variance. Uses the Gaussian model.

Stage 2 — Tails (Student-t LL):
    calibrate_tail() fits the tail parameters (b0, b_stale, b_sess, b_tau,
    nu_max) by maximizing Student-t log-likelihood on the z-residuals from
    stage 1. Uses the GaussianT model with frozen vol params.
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
    """Cluster-robust standard error for mean(residuals)."""
    unique_ids = np.unique(cluster_ids)
    n_clusters = len(unique_ids)
    if n_clusters < 2:
        return np.std(residuals) / np.sqrt(len(residuals))
    cluster_means = np.array([
        np.mean(residuals[cluster_ids == cid]) for cid in unique_ids
    ])
    return np.std(cluster_means, ddof=1) / np.sqrt(n_clusters)


TAU_BUCKET_EDGES_MIN = [0, 5, 15, 30, 60]
TAU_BUCKET_LABELS = ["0-5", "5-15", "15-30", "30-60"]


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
        objective: "qlike" or "log_ratio_mse".
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

    names = model.param_names()
    x0 = [model.initial_params()[n] for n in names]
    bounds = [model.param_bounds()[n] for n in names]

    def obj_fn(x):
        params = dict(zip(names, x))
        var_pred = model.predict_variance(params, S, K, tau, features)
        var_pred = np.maximum(var_pred, 1e-20)

        if objective == "qlike":
            scores = np.log(var_pred) + log_return_sq / var_pred
            if market_weighted and market_ids is not None:
                return _market_weighted_mean(scores, market_ids)
            return float(np.mean(scores))

        elif objective == "log_ratio_mse":
            lr = np.log(log_return_sq[valid] / var_pred[valid])
            if market_weighted and market_ids is not None:
                return _market_weighted_mean(lr ** 2, market_ids[valid])
            return float(np.mean(lr ** 2))

        raise ValueError(f"Unknown objective: {objective}")

    if verbose:
        print(f"\n[Stage 1] Calibrating {model.name} [vol/{objective}] "
              f"({len(names)} params, {len(S):,} samples) ...")

    res = minimize(obj_fn, x0, method=method, bounds=bounds)
    fitted = dict(zip(names, [float(v) for v in res.x]))
    model.set_params(fitted)

    obj_final = res.fun
    n_markets = dataset["market_id"].nunique() if "market_id" in dataset.columns else 0

    mean_var = float(np.mean(log_return_sq))
    if objective == "qlike":
        obj_baseline = float(np.log(mean_var) + 1.0)
    else:
        obj_baseline = float(np.mean(np.log(log_return_sq[valid] / mean_var) ** 2))

    if verbose:
        print(f"\n{'='*60}")
        print(f"STAGE 1 — VOL CALIBRATION: {model.name} [{objective}]")
        print(f"{'='*60}")
        print(f"\n  Parameters:")
        for n in names:
            print(f"    {n:10s} = {fitted[n]:+.6f}")
        print(f"\n  {objective}: {obj_final:.6f}  (baseline: {obj_baseline:.6f})")
        print(f"  Samples: {len(S):,}  Markets: {n_markets}")

        _print_vol_diagnostics(model, fitted, S, K, tau, S_T, features, market_ids)

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


def _print_vol_diagnostics(model, params, S, K, tau, S_T, features, market_ids):
    """Print variance ratio and standardized-return diagnostics."""
    from scipy.stats import kurtosis as sp_kurtosis

    var_pred = model.predict_variance(params, S, K, tau, features)
    var_pred = np.maximum(var_pred, 1e-20)

    log_return = np.log(S_T / S)
    log_return_sq = log_return ** 2
    z = log_return / np.sqrt(var_pred)

    z_std = np.std(z)
    z_kurt = float(sp_kurtosis(z, fisher=True))
    pct_gt2 = np.mean(np.abs(z) > 2) * 100
    pct_gt3 = np.mean(np.abs(z) > 3) * 100

    print(f"\n  Standardized returns z = log(S_T/S) / sqrt(var_pred):")
    print(f"    std(z) = {z_std:.4f}  (target: 1.0)")
    print(f"    kurtosis = {z_kurt:.2f}  (target: 0.0)")
    print(f"    P(|z|>2) = {pct_gt2:.2f}%  (Gaussian: 4.55%)")
    print(f"    P(|z|>3) = {pct_gt3:.2f}%  (Gaussian: 0.27%)")

    tau_min = tau / 60.0

    print(f"\n  Per-tau variance ratios (realized/predicted):")
    for i in range(len(TAU_BUCKET_EDGES_MIN) - 1):
        lo, hi = TAU_BUCKET_EDGES_MIN[i], TAU_BUCKET_EDGES_MIN[i + 1]
        label = TAU_BUCKET_LABELS[i]
        mask = (tau_min >= lo) & (tau_min < hi)
        n = mask.sum()
        if n > 0:
            ratio = np.mean(log_return_sq[mask]) / np.mean(var_pred[mask])
            z_std_bucket = np.std(z[mask])
            print(f"    tau [{label:5s}] min: ratio={ratio:.4f}  std(z)={z_std_bucket:.4f}  n={n}")

    if "sigma_rel" in features:
        sigma_rel = features["sigma_rel"]
        edges = np.quantile(sigma_rel, [0.0, 0.25, 0.5, 0.75, 1.0])
        print(f"\n  Per-sigma_rel quartile variance ratios:")
        for qi, ql in enumerate(["Q1", "Q2", "Q3", "Q4"]):
            lo, hi = edges[qi], edges[qi + 1]
            mask = (sigma_rel >= lo) & (sigma_rel < hi) if qi < 3 else (sigma_rel >= lo)
            n = mask.sum()
            if n > 0:
                ratio = np.mean(log_return_sq[mask]) / np.mean(var_pred[mask])
                print(f"    {ql} [{lo:.3f}, {hi:.3f}]: ratio={ratio:.4f}  n={n}")


# ---------------------------------------------------------------------------
# Stage 2: Tail calibration (Student-t log-likelihood on z-residuals)
# ---------------------------------------------------------------------------

def calibrate_tail(
    model,
    dataset: pd.DataFrame,
    market_weighted: bool = True,
    method: str = "L-BFGS-B",
    output_dir: str | Path | None = "pricing/output",
    verbose: bool = True,
) -> CalibrationResult:
    """Stage 2: Fit tail parameters by maximizing Student-t log-likelihood.

    sigma_eff is frozen from stage 1. Only the b-params controlling
    nu(state) and nu_max are fitted.

    Args:
        model: GaussianTModel instance (with frozen vol params).
        dataset: DataFrame with S, K, tau, S_T, market_id + features.
        market_weighted: If True, each market hour gets equal weight.
        method: scipy.optimize.minimize method.
        output_dir: Where to save params JSON.
        verbose: Print diagnostics.
    """
    from scipy.special import gammaln

    S = dataset["S"].values.astype(np.float64)
    K = dataset["K"].values.astype(np.float64)
    tau = dataset["tau"].values.astype(np.float64)
    S_T = dataset["S_T"].values.astype(np.float64)
    features = {f: dataset[f].values.astype(np.float64) for f in model.required_features()}
    market_ids = dataset["market_id"].values if "market_id" in dataset.columns else None

    sigma_eff = model._sigma_eff(tau, features)
    sqrt_tau = np.sqrt(np.maximum(tau, 1e-6))
    z = np.log(S_T / S) / (sigma_eff * sqrt_tau)

    names = model.param_names()
    x0 = [model.initial_params()[n] for n in names]
    bounds = [model.param_bounds()[n] for n in names]

    def neg_t_ll(x):
        params = dict(zip(names, x))
        nu = model._nu(params, tau, features)
        nu = np.maximum(nu, 3.0 + 1e-6)

        scale = np.sqrt((nu - 2.0) / nu)
        w = z / scale

        ll_obs = (gammaln(0.5 * (nu + 1)) - gammaln(0.5 * nu)
                  - 0.5 * np.log(nu * np.pi)
                  - 0.5 * (nu + 1) * np.log(1.0 + w ** 2 / nu)
                  - np.log(scale))

        if market_weighted and market_ids is not None:
            return -_market_weighted_mean(ll_obs, market_ids)
        return -float(np.mean(ll_obs))

    if verbose:
        print(f"\n[Stage 2] Calibrating {model.name} [tail/student-t LL] "
              f"({len(names)} params, {len(S):,} samples) ...")

    res = minimize(neg_t_ll, x0, method=method, bounds=bounds)
    fitted = dict(zip(names, [float(v) for v in res.x]))
    model.set_params(fitted)

    obj_final = res.fun
    n_markets = dataset["market_id"].nunique() if "market_id" in dataset.columns else 0

    nu = model._nu(fitted, tau, features)

    if verbose:
        print(f"\n{'='*60}")
        print(f"STAGE 2 — TAIL CALIBRATION: {model.name}")
        print(f"{'='*60}")
        print(f"\n  Parameters:")
        for n in names:
            print(f"    {n:10s} = {fitted[n]:+.6f}")
        print(f"\n  Neg t-LL: {obj_final:.6f}")
        print(f"  Samples: {len(S):,}  Markets: {n_markets}")

        print(f"\n  nu distribution:")
        print(f"    mean={np.mean(nu):.2f}  median={np.median(nu):.2f}  "
              f"min={np.min(nu):.2f}  max={np.max(nu):.2f}")

        scale = np.sqrt((nu - 2.0) / nu)
        z_std = np.std(z)
        pct_z_gt2 = np.mean(np.abs(z) > 2) * 100
        pct_z_gt3 = np.mean(np.abs(z) > 3) * 100
        print(f"\n  Standardized returns:")
        print(f"    std(z) = {z_std:.4f}")
        print(f"    P(|z|>2) = {pct_z_gt2:.2f}%  (Gaussian: 4.55%)")
        print(f"    P(|z|>3) = {pct_z_gt3:.2f}%  (Gaussian: 0.27%)")

        log_return_sq = np.log(S_T / S) ** 2
        var_pred = model.predict_variance(fitted, S, K, tau, features)
        var_pred = np.maximum(var_pred, 1e-20)
        tau_min = tau / 60.0
        print(f"\n  Per-tau variance ratios (should be ~1.0):")
        for i in range(len(TAU_BUCKET_EDGES_MIN) - 1):
            lo, hi = TAU_BUCKET_EDGES_MIN[i], TAU_BUCKET_EDGES_MIN[i + 1]
            label = TAU_BUCKET_LABELS[i]
            mask = (tau_min >= lo) & (tau_min < hi)
            n = mask.sum()
            if n > 0:
                ratio = np.mean(log_return_sq[mask]) / np.mean(var_pred[mask])
                print(f"    tau [{label:5s}] min: ratio={ratio:.4f}  n={n}")

        y = dataset["y"].values.astype(np.float64) if "y" in dataset.columns else None
        if y is not None:
            p_pred = model.predict(fitted, S, K, tau, features)
            ll_t = log_loss(y, p_pred)
            ll_baseline = log_loss(y, np.full_like(y, y.mean()))
            imp = (ll_baseline - ll_t) / ll_baseline * 100
            print(f"\n  Binary LL: {ll_t:.6f}  (baseline: {ll_baseline:.6f}, {imp:+.1f}%)")

        if market_ids is not None:
            _print_clustered_se_tail(fitted, names, z, tau, features, market_ids, model)

    result = CalibrationResult(
        model_name=model.name,
        params=fitted,
        log_loss=obj_final,
        log_loss_baseline=0.0,
        improvement_pct=0.0,
        n_samples=len(S),
        n_markets=n_markets,
        metadata={"objective": "student_t_ll", "neg_t_ll": obj_final},
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        params_path = output_dir / f"{model.name}_params.json"
        params_out = {
            "model": model.name,
            "objective": "student_t_ll",
            **fitted,
            "neg_t_ll": obj_final,
            "n_samples": len(S),
            "n_markets": n_markets,
            "nu_mean": float(np.mean(nu)),
            "nu_median": float(np.median(nu)),
        }
        with open(params_path, "w") as f:
            json.dump(params_out, f, indent=2)
        if verbose:
            print(f"\n  Saved params to {params_path}")

    return result


def _print_clustered_se_tail(fitted, names, z, tau, features, market_ids, model):
    """Print clustered standard errors for tail params via finite differences."""
    from scipy.special import gammaln

    unique = np.unique(market_ids)
    n_clusters = len(unique)

    eps = 1e-4
    grad_per_market = np.zeros((n_clusters, len(names)))

    for j, name in enumerate(names):
        params_plus = fitted.copy()
        params_minus = fitted.copy()
        params_plus[name] = fitted[name] + eps
        params_minus[name] = fitted[name] - eps

        nu_p = model._nu(params_plus, tau, features)
        nu_m = model._nu(params_minus, tau, features)

        for which, nu_val in [(0, nu_p), (1, nu_m)]:
            nu_val = np.maximum(nu_val, 3.0 + 1e-6)
            scale = np.sqrt((nu_val - 2.0) / nu_val)
            w = z / scale
            ll_obs = (gammaln(0.5 * (nu_val + 1)) - gammaln(0.5 * nu_val)
                      - 0.5 * np.log(nu_val * np.pi)
                      - 0.5 * (nu_val + 1) * np.log(1.0 + w ** 2 / nu_val)
                      - np.log(scale))
            for ci, m in enumerate(unique):
                mask = market_ids == m
                val = np.mean(ll_obs[mask])
                if which == 0:
                    grad_per_market[ci, j] = val
                else:
                    grad_per_market[ci, j] = (grad_per_market[ci, j] - val) / (2 * eps)

    se = np.std(grad_per_market, axis=0, ddof=1) / np.sqrt(n_clusters)

    print(f"\n  Clustered SE (n_clusters={n_clusters}):")
    for j, name in enumerate(names):
        t_stat = fitted[name] / max(se[j], 1e-12)
        sig = "***" if abs(t_stat) > 2.58 else "**" if abs(t_stat) > 1.96 else "*" if abs(t_stat) > 1.64 else ""
        print(f"    {name:10s} = {fitted[name]:+.4f} +/- {se[j]:.4f}  t={t_stat:+.2f} {sig}")
