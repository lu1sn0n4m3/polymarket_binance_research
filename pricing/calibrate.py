"""Two-stage calibration pipeline for the variance-first binary option pricer.

Stage 1 — Volatility (QLIKE):
    calibrate_vol() fits variance parameters (c, alpha, k0, k1) by minimizing
    the QLIKE scoring rule on realized variance. Uses the Gaussian model.

Stage 2 — Tails (Fixed-nu Student-t LL):
    calibrate_tail_fixed() fits a single scalar nu by maximizing Student-t
    log-likelihood on z-residuals from stage 1. Uses the FixedT model.
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

        elif objective == "log_ratio_mse":
            lr = np.log(log_return_sq[valid] / var_pred[valid])
            if market_weighted and market_ids is not None:
                return _market_weighted_mean(lr ** 2, market_ids[valid])
            return float(np.mean(lr ** 2))

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
    if objective == "qlike":
        obj_baseline = float(np.log(mean_var) + 1.0)
    else:
        obj_baseline = float(np.mean(np.log(log_return_sq[valid] / mean_var) ** 2))

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


# ---------------------------------------------------------------------------
# Stage 2: Fixed-nu tail calibration via Brent's method
# ---------------------------------------------------------------------------

def calibrate_tail_fixed(
    model,
    dataset: pd.DataFrame,
    market_weighted: bool = True,
    output_dir: str | Path | None = "pricing/output",
    verbose: bool = True,
) -> CalibrationResult:
    """Stage 2 (fixed-nu): Fit scalar nu by maximizing Student-t log-likelihood.

    Uses Brent's method (scipy.optimize.minimize_scalar) for the 1D problem.
    Paper Section 6, eq 24.

    Args:
        model: FixedTModel instance (with frozen vol params).
        dataset: DataFrame with S, K, tau, S_T, market_id + features.
        market_weighted: If True, each market hour gets equal weight.
        output_dir: Where to save params JSON.
        verbose: Print diagnostics.
    """
    from scipy.optimize import minimize_scalar
    from scipy.special import gammaln

    S = dataset["S"].values.astype(np.float64)
    K = dataset["K"].values.astype(np.float64)
    tau = dataset["tau"].values.astype(np.float64)
    S_T = dataset["S_T"].values.astype(np.float64)
    features = {f: dataset[f].values.astype(np.float64) for f in model.required_features()}
    market_ids = dataset["market_id"].values if "market_id" in dataset.columns else None

    v = model._variance(tau, features)
    z = np.log(S_T / S) / np.sqrt(np.maximum(v, 1e-20))

    def neg_t_ll(nu):
        scale = np.sqrt((nu - 2.0) / nu)
        w = z / scale
        ll_obs = (gammaln(0.5 * (nu + 1)) - gammaln(0.5 * nu)
                  - 0.5 * np.log(nu * np.pi)
                  - 0.5 * (nu + 1) * np.log(1.0 + w ** 2 / nu)
                  - np.log(scale))
        if market_weighted and market_ids is not None:
            return -_market_weighted_mean(ll_obs, market_ids)
        return -float(np.mean(ll_obs))

    bounds = model.param_bounds()["nu"]

    if verbose:
        print(f"\n[Stage 2a] Calibrating {model.name} [tail/fixed-nu, Brent's method] "
              f"({len(S):,} samples) ...")

    res = minimize_scalar(neg_t_ll, bounds=bounds, method="bounded")
    nu_hat = float(res.x)
    fitted = {"nu": nu_hat}
    model.set_params(fitted)

    obj_final = float(res.fun)
    n_markets = dataset["market_id"].nunique() if "market_id" in dataset.columns else 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"STAGE 2a — FIXED-NU CALIBRATION: {model.name}")
        print(f"{'='*60}")
        print(f"\n  nu = {nu_hat:.4f}")
        print(f"  Neg t-LL: {obj_final:.6f}")
        print(f"  Samples: {len(S):,}  Markets: {n_markets}")

        # Tail exceedance diagnostics
        tail_diagnostics(z, nu=np.full_like(z, nu_hat), verbose=True)

        y = dataset["y"].values.astype(np.float64) if "y" in dataset.columns else None
        if y is not None:
            p_pred = model.predict(fitted, S, K, tau, features)
            ll_t = log_loss(y, p_pred)
            ll_baseline = log_loss(y, np.full_like(y, y.mean()))
            imp = (ll_baseline - ll_t) / ll_baseline * 100
            print(f"\n  Binary LL: {ll_t:.6f}  (baseline: {ll_baseline:.6f}, {imp:+.1f}%)")

    result = CalibrationResult(
        model_name=model.name,
        params=fitted,
        log_loss=obj_final,
        log_loss_baseline=0.0,
        improvement_pct=0.0,
        n_samples=len(S),
        n_markets=n_markets,
        metadata={"objective": "student_t_ll_fixed", "neg_t_ll": obj_final},
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        params_path = output_dir / f"{model.name}_params.json"
        params_out = {
            "model": model.name,
            "objective": "student_t_ll_fixed",
            "nu": nu_hat,
            "neg_t_ll": obj_final,
            "n_samples": len(S),
            "n_markets": n_markets,
        }
        with open(params_path, "w") as f:
            json.dump(params_out, f, indent=2)
        if verbose:
            print(f"\n  Saved params to {params_path}")

    return result


# ---------------------------------------------------------------------------
# Stage 2 variants: tail-targeted nu estimation
# ---------------------------------------------------------------------------

def calibrate_tail_truncated(
    model,
    dataset: pd.DataFrame,
    z_cutoff: float = 1.5,
    verbose: bool = True,
) -> dict:
    """Fit nu by MLE on truncated residuals |z| > z_cutoff.

    The truncated log-likelihood conditions on |z| > z_cutoff:
        l_trunc(nu) = sum_{|z_i|>z0} [log f_nu(z_i/s) - log s - log P(|Z|>z0; nu)]

    Returns dict with 'nu', 'n_tail', 'neg_t_ll_truncated'.
    """
    from scipy.optimize import minimize_scalar
    from scipy.special import gammaln
    from scipy.stats import t as student_t

    S = dataset["S"].values.astype(np.float64)
    S_T = dataset["S_T"].values.astype(np.float64)
    tau = dataset["tau"].values.astype(np.float64)
    features = {f: dataset[f].values.astype(np.float64) for f in model.required_features()}

    v = model._variance(tau, features)
    z = np.log(S_T / S) / np.sqrt(np.maximum(v, 1e-20))

    tail_mask = np.abs(z) > z_cutoff
    z_tail = z[tail_mask]
    n_tail = len(z_tail)

    if n_tail < 10:
        if verbose:
            print(f"  [truncated MLE] Only {n_tail} residuals with |z|>{z_cutoff}, skipping.")
        return {"nu": np.nan, "n_tail": n_tail, "neg_t_ll_truncated": np.nan}

    def neg_trunc_ll(nu):
        scale = np.sqrt((nu - 2.0) / nu)
        w = z_tail / scale
        # Log-density at each tail observation
        ll_obs = (gammaln(0.5 * (nu + 1)) - gammaln(0.5 * nu)
                  - 0.5 * np.log(nu * np.pi)
                  - 0.5 * (nu + 1) * np.log(1.0 + w ** 2 / nu)
                  - np.log(scale))
        # Subtract log of tail probability (truncation normalization)
        tail_prob = 2.0 * student_t.sf(z_cutoff / scale, df=nu)
        tail_prob = max(tail_prob, 1e-20)
        ll_obs = ll_obs - np.log(tail_prob)
        return -float(np.mean(ll_obs))

    bounds = model.param_bounds()["nu"]

    if verbose:
        print(f"\n[Stage 2 — truncated MLE] z_cutoff={z_cutoff:.1f}, "
              f"{n_tail:,} tail obs ({n_tail/len(z)*100:.1f}%) ...")

    res = minimize_scalar(neg_trunc_ll, bounds=bounds, method="bounded")
    nu_hat = float(res.x)

    if verbose:
        print(f"  nu = {nu_hat:.4f}  (neg trunc LL = {res.fun:.6f})")

    return {"nu": nu_hat, "n_tail": n_tail, "neg_t_ll_truncated": float(res.fun)}


def calibrate_tail_exceedance(
    model,
    dataset: pd.DataFrame,
    threshold: float = 2.0,
    verbose: bool = True,
) -> dict:
    """Fit nu by minimizing binary log-loss on the exceedance indicator 1{|z|>q}.

    For each observation, the model predicts P(|z|>q) = 2*(1 - T_nu(q/s(nu))).
    nu is chosen to minimize log-loss on Y_i = 1{|z_i| > q}.

    Returns dict with 'nu', 'exceedance_rate', 'exceedance_ll'.
    """
    from scipy.optimize import minimize_scalar
    from scipy.stats import t as student_t

    S = dataset["S"].values.astype(np.float64)
    S_T = dataset["S_T"].values.astype(np.float64)
    tau = dataset["tau"].values.astype(np.float64)
    features = {f: dataset[f].values.astype(np.float64) for f in model.required_features()}

    v = model._variance(tau, features)
    z = np.log(S_T / S) / np.sqrt(np.maximum(v, 1e-20))

    Y = (np.abs(z) > threshold).astype(np.float64)
    exc_rate = float(Y.mean())

    def neg_exc_ll(nu):
        scale = np.sqrt((nu - 2.0) / nu)
        p_exc = 2.0 * student_t.sf(threshold / scale, df=nu)
        p_exc = np.clip(p_exc, 1e-9, 1.0 - 1e-9)
        # Binary log-loss on exceedance indicator
        ll = -(Y * np.log(p_exc) + (1.0 - Y) * np.log(1.0 - p_exc))
        return float(np.mean(ll))

    bounds = model.param_bounds()["nu"]

    if verbose:
        print(f"\n[Stage 2 — exceedance matching] threshold={threshold:.1f}, "
              f"exceedance rate={exc_rate:.4f} ({exc_rate*100:.2f}%) ...")

    res = minimize_scalar(neg_exc_ll, bounds=bounds, method="bounded")
    nu_hat = float(res.x)

    if verbose:
        scale = np.sqrt((nu_hat - 2.0) / nu_hat)
        model_rate = 2.0 * student_t.sf(threshold / scale, df=nu_hat)
        print(f"  nu = {nu_hat:.4f}  (exc LL = {res.fun:.6f})")
        print(f"  Empirical rate: {exc_rate:.4f}  Model rate: {model_rate:.4f}")

    return {"nu": nu_hat, "exceedance_rate": exc_rate, "exceedance_ll": float(res.fun)}
