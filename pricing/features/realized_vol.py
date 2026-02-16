"""EWMA realized volatility using tick-time sum(dx^2)/sum(dt) estimator.

Computes sigma_rv at each price change using exponentially-weighted moving
average of the realized variance per second. This is robust to microstructure
noise -- we do NOT normalize individual returns by sqrt(dt).

Ported from pricer_calibration/run/run_calibration.py:compute_rv_ewma.
"""

import numpy as np
import pandas as pd


def compute_rv_ewma(
    bbo: pd.DataFrame,
    half_life_sec: float = 300.0,
    ts_col: str = "ts_recv",
    mid_col: str = "mid_px",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute EWMA realized volatility on tick-time data.

    Uses "sum(dx^2) / sum(dt)" estimator with EWMA weighting:
        ewma_sum_sq = decay * ewma_sum_sq_prev + dx^2
        ewma_sum_dt = decay * ewma_sum_dt_prev + dt
        sigma_rv = sqrt(ewma_sum_sq / ewma_sum_dt)

    Args:
        bbo: DataFrame with timestamp and mid price columns.
        half_life_sec: EWMA half-life in seconds.
        ts_col: Name of timestamp column (epoch ms).
        mid_col: Name of mid price column.

    Returns:
        (ts_tick, sigma_rv): timestamps (epoch ms) and sigma_rv values
        in per-sqrt(sec) units. Only at times where mid price changed.
    """
    ts = bbo[ts_col].values
    mid = bbo[mid_col].values

    # Find where mid actually changed
    changed = np.zeros(len(mid), dtype=bool)
    changed[0] = True
    changed[1:] = mid[1:] != mid[:-1]

    ts_changed = ts[changed]
    mid_changed = mid[changed]

    if len(mid_changed) < 2:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    log_mid = np.log(mid_changed)
    dx = np.diff(log_mid)
    dt_ms = np.diff(ts_changed)
    dt_sec = dt_ms / 1000.0
    ts_tick = ts_changed[1:]

    # Filter zero/tiny dt
    valid = dt_sec > 0.0001
    dx = dx[valid]
    dt_sec = dt_sec[valid]
    ts_tick = ts_tick[valid]

    n = len(dx)
    if n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    sigma_rv = np.zeros(n)
    dx_sq = dx ** 2

    # Initialize with first few ticks
    init_window = min(100, n // 10)
    if init_window > 10:
        ewma_sum_sq = np.sum(dx_sq[:init_window])
        ewma_sum_dt = np.sum(dt_sec[:init_window])
    else:
        ewma_sum_sq = 1e-8 * half_life_sec
        ewma_sum_dt = half_life_sec

    for i in range(n):
        decay = np.exp(-np.log(2) * dt_sec[i] / half_life_sec)
        ewma_sum_sq = decay * ewma_sum_sq + dx_sq[i]
        ewma_sum_dt = decay * ewma_sum_dt + dt_sec[i]
        var_per_sec = ewma_sum_sq / max(ewma_sum_dt, 1e-6)
        sigma_rv[i] = np.sqrt(max(var_per_sec, 1e-12))

    return ts_tick, sigma_rv
