"""Conditional bias diagnostics for the variance-first pricer (paper Section 3).

Core idea: if Stage 1 is correctly calibrated then u_i = r_i^2 / v_hat_i
satisfies E[u | X] = 1 for any conditioning variable X observable at quote time.

We check this for X in {tau, sigma_rel, time_since_move, hour_et}.
"""

import numpy as np
import pandas as pd


def _equal_mass_bins(x, n_bins=10):
    """Return bin edges for equal-mass (quantile) bins."""
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(x, percentiles)
    # Ensure unique edges
    edges = np.unique(edges)
    return edges


def _clustered_mean_se(values, cluster_ids):
    """Compute mean and cluster-robust SE."""
    unique = np.unique(cluster_ids)
    n_clusters = len(unique)
    cluster_means = np.array([np.mean(values[cluster_ids == c]) for c in unique])
    overall_mean = float(np.mean(cluster_means))
    if n_clusters < 2:
        return overall_mean, 0.0
    se = float(np.std(cluster_means, ddof=1) / np.sqrt(n_clusters))
    return overall_mean, se


def variance_ratio_diagnostics(
    dataset: pd.DataFrame,
    var_pred: np.ndarray,
    n_bins: int = 10,
    verbose: bool = True,
) -> dict:
    """Compute E[u|X] conditional bias diagnostics (paper Section 3).

    u_i = r_i^2 / v_hat_i should have E[u|X] = 1 for all conditioning vars.

    Args:
        dataset: Calibration DataFrame with S, S_T, tau, sigma_rel, time_since_move, hour_et, market_id.
        var_pred: Model's variance forecast v_hat for each row.
        n_bins: Number of equal-mass bins per conditioning variable.
        verbose: Print formatted table.

    Returns:
        Dict mapping variable name -> list of dicts with keys
        {bin_lo, bin_hi, mean_u, se_u, n, mean_x}.
    """
    S = dataset["S"].values.astype(np.float64)
    S_T = dataset["S_T"].values.astype(np.float64)
    market_ids = dataset["market_id"].values if "market_id" in dataset.columns else None

    r = np.log(S_T / S)
    r_sq = r ** 2
    u = r_sq / np.maximum(var_pred, 1e-20)

    # Overall
    u_mean = float(np.mean(u))
    u_std = float(np.std(u))

    if verbose:
        print(f"\n  Variance ratio u = r^2 / v_hat:")
        print(f"    E[u] = {u_mean:.4f}  (target: 1.0)")
        print(f"    std(u) = {u_std:.2f}")

    # State variables to diagnose
    state_vars = {}
    if "tau" in dataset.columns:
        state_vars["tau"] = dataset["tau"].values.astype(np.float64) / 60.0  # minutes
    if "sigma_rel" in dataset.columns:
        state_vars["sigma_rel"] = dataset["sigma_rel"].values.astype(np.float64)
    if "time_since_move" in dataset.columns:
        state_vars["tsm"] = dataset["time_since_move"].values.astype(np.float64)
    if "hour_et" in dataset.columns:
        state_vars["hour_et"] = dataset["hour_et"].values.astype(np.float64)

    results = {}
    for var_name, x in state_vars.items():
        if var_name == "hour_et":
            # Use natural bins (0-23)
            bins_list = []
            for h in range(24):
                mask = (x >= h) & (x < h + 1)
                n = mask.sum()
                if n < 20:
                    continue
                if market_ids is not None:
                    m_u, se_u = _clustered_mean_se(u[mask], market_ids[mask])
                else:
                    m_u = float(np.mean(u[mask]))
                    se_u = float(np.std(u[mask]) / np.sqrt(n))
                bins_list.append({
                    "bin_lo": h, "bin_hi": h + 1,
                    "mean_u": m_u, "se_u": se_u,
                    "n": int(n), "mean_x": float(h + 0.5),
                })
        else:
            edges = _equal_mass_bins(x, n_bins)
            bins_list = []
            for i in range(len(edges) - 1):
                lo, hi = edges[i], edges[i + 1]
                if i < len(edges) - 2:
                    mask = (x >= lo) & (x < hi)
                else:
                    mask = (x >= lo) & (x <= hi)
                n = mask.sum()
                if n < 20:
                    continue
                if market_ids is not None:
                    m_u, se_u = _clustered_mean_se(u[mask], market_ids[mask])
                else:
                    m_u = float(np.mean(u[mask]))
                    se_u = float(np.std(u[mask]) / np.sqrt(n))
                bins_list.append({
                    "bin_lo": float(lo), "bin_hi": float(hi),
                    "mean_u": m_u, "se_u": se_u,
                    "n": int(n), "mean_x": float(np.mean(x[mask])),
                })

        results[var_name] = bins_list

        if verbose:
            _print_eu_table(var_name, bins_list)

    return results


def _print_eu_table(var_name, bins_list):
    """Print E[u|X] table for a single state variable."""
    unit = " min" if var_name == "tau" else ("s" if var_name == "tsm" else "")
    print(f"\n  E[u | {var_name}]:")
    print(f"    {'Bin':>12s}  {'E[u]':>7s}  {'SE':>6s}  {'n':>6s}")
    print(f"    {'─'*12}  {'─'*7}  {'─'*6}  {'─'*6}")
    for b in bins_list:
        if var_name == "hour_et":
            label = f"{int(b['bin_lo']):02d}h ET"
        else:
            label = f"[{b['bin_lo']:.1f}, {b['bin_hi']:.1f}]{unit}"
        flag = " *" if abs(b["mean_u"] - 1.0) > 2 * b["se_u"] and b["se_u"] > 0 else ""
        print(f"    {label:>12s}  {b['mean_u']:7.4f}  {b['se_u']:6.4f}  {b['n']:6d}{flag}")


def tail_diagnostics(
    z: np.ndarray,
    nu: np.ndarray | None = None,
    thresholds: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """Tail exceedance diagnostics: empirical P(|z|>c) vs Gaussian and Student-t.

    Args:
        z: Standardized residuals r / sqrt(v_hat).
        nu: Degrees-of-freedom array (same length as z). None = skip t comparison.
        thresholds: Array of thresholds c. Default: [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0].
        verbose: Print table.

    Returns:
        Dict with keys {thresholds, empirical, gaussian, student_t (if nu given)}.
    """
    from scipy.stats import norm, t as student_t_dist

    if thresholds is None:
        thresholds = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

    abs_z = np.abs(z)
    empirical = np.array([float(np.mean(abs_z > c)) for c in thresholds])
    gaussian = np.array([float(2 * norm.sf(c)) for c in thresholds])

    result = {"thresholds": thresholds, "empirical": empirical, "gaussian": gaussian}

    if nu is not None:
        scale = np.sqrt((nu - 2.0) / nu)
        adaptive_t = np.array([
            float(np.mean(2.0 * student_t_dist.sf(c / scale, df=nu)))
            for c in thresholds
        ])
        result["student_t"] = adaptive_t

    if verbose:
        print(f"\n  Tail exceedance P(|z| > c):")
        header = f"    {'c':>5s}  {'Empir':>8s}  {'Gauss':>8s}"
        sep = f"    {'─'*5}  {'─'*8}  {'─'*8}"
        if nu is not None:
            header += f"  {'t-model':>8s}"
            sep += f"  {'─'*8}"
        print(header)
        print(sep)
        for i, c in enumerate(thresholds):
            line = f"    {c:5.1f}  {empirical[i]:8.4%}  {gaussian[i]:8.4%}"
            if nu is not None:
                line += f"  {result['student_t'][i]:8.4%}"
            print(line)

    return result
