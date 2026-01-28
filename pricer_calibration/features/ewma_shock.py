"""EWMA volatility and shock statistic on the 100ms grid.

Computes:
- u_k = r_k / σ_step              (normalized returns, where σ_step is per-100ms)
- v_k = EWMA of u_k²               (relative variance)
- σ_rel_k = sqrt(v_k)              (relative vol multiplier, floored at 1.0)
- σ_Δ_k = σ_step * σ_rel_k        (live 100ms vol)
- σ_base_k = σ_per_sec * σ_rel_k  (per-√sec vol for pricer)
- z_k = rolling_max(|u_k|, M)      (shock statistic)

Scaling from seasonal vol (computed on sub_interval_sec returns) to grid:
- σ_per_sec = σ_tod / √(sub_interval_sec)  (convert to per-√sec)
- σ_step = σ_per_sec * √(delta_sec)         (convert to per-100ms)
"""

import numpy as np
import pandas as pd

from pricer_calibration.features.seasonal_vol import SeasonalVolCurve


def _ewma_shock_loop(
    r: np.ndarray,
    sigma_tod_arr: np.ndarray,
    alpha: float,
    u_sq_cap: float,
    shock_M: int,
    delta_sec: float,
    sub_interval_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Core loop: compute u, v, sigma_rel, sigma_delta, sigma_base, z."""
    n = len(r)
    u = np.empty(n)
    v = np.empty(n)
    sigma_rel = np.empty(n)
    sigma_delta = np.empty(n)
    sigma_base = np.empty(n)
    z = np.empty(n)

    sqrt_delta = np.sqrt(delta_sec)

    # Initialize
    v_k = 1.0

    sqrt_sub = np.sqrt(sub_interval_sec)

    for k in range(n):
        # σ_tod is MAD-scaled stdev of sub_interval returns (default 1s).
        # Convert to per-√sec: σ_per_sec = σ_tod / √(sub_interval)
        # Per-step σ for 100ms grid: σ_step = σ_per_sec * √Δ
        s_tod = sigma_tod_arr[k]
        if s_tod < 1e-15:
            s_tod = 1e-15

        s_per_sec = s_tod / sqrt_sub
        s_step = s_per_sec * sqrt_delta  # per-step σ
        u_k = r[k] / s_step
        u[k] = u_k

        u_sq = u_k * u_k
        if u_sq > u_sq_cap:
            u_sq = u_sq_cap

        if k == 0:
            v_k = 1.0
        else:
            v_k = (1.0 - alpha) * v_k + alpha * u_sq

        v[k] = v_k
        # Floor σ_rel at 1.0: EWMA can only inflate vol, never deflate
        # below seasonal baseline. Zero 100ms returns decay v_k → 0.
        sr = max(np.sqrt(v_k), 1.0)
        sigma_rel[k] = sr
        sd = s_step * sr  # per-step adjusted σ
        sigma_delta[k] = sd
        # σ_base in per-√sec units
        sigma_base[k] = s_per_sec * sr

        # Rolling max of |u| over window M (causal)
        z_k = abs(u_k)
        start = max(0, k - shock_M + 1)
        for j in range(start, k):
            a = abs(u[j])
            if a > z_k:
                z_k = a
        z[k] = z_k

    return u, v, sigma_rel, sigma_delta, sigma_base, z


def compute_ewma_shock(
    grid: pd.DataFrame,
    seasonal: SeasonalVolCurve,
    ewma_half_life_sec: float = 20.0,
    delta_ms: int = 100,
    u_sq_cap: float = 100.0,
    shock_M: int = 5,
) -> pd.DataFrame:
    """Add EWMA volatility and shock columns to the grid.

    Args:
        grid: DataFrame with columns [t, S, logS, r].
        seasonal: SeasonalVolCurve for σ_tod lookup.
        ewma_half_life_sec: EWMA half-life H in seconds.
        delta_ms: Grid interval in ms.
        u_sq_cap: Winsorization cap for u_k².
        shock_M: Rolling max window in grid steps.

    Returns:
        Copy of grid with added columns:
        [bucket, sigma_tod, u, v, sigma_rel, sigma_delta, sigma_base, z]
    """
    delta_sec = delta_ms / 1000.0
    alpha = 1.0 - np.exp(-np.log(2) * delta_sec / ewma_half_life_sec)

    t_ms = grid["t"].values
    r = grid["r"].values.astype(np.float64)

    # Lookup σ_tod for each row
    total_seconds = (t_ms // 1000) % 86400
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    bucket_idx = (hours * 60 + minutes) // seasonal.bucket_minutes
    bucket_idx = bucket_idx.astype(np.int64) % seasonal.n_buckets

    sigma_tod_arr = seasonal.sigma_tod[bucket_idx].astype(np.float64)

    u, v, sigma_rel, sigma_delta, sigma_base, z = _ewma_shock_loop(
        r, sigma_tod_arr, alpha, u_sq_cap, shock_M, delta_sec, seasonal.sub_interval_sec
    )

    result = grid.copy()
    result["bucket"] = bucket_idx
    result["sigma_tod"] = sigma_tod_arr
    result["u"] = u
    result["v"] = v
    result["sigma_rel"] = sigma_rel
    result["sigma_delta"] = sigma_delta
    result["sigma_base"] = sigma_base
    result["z"] = z

    return result
