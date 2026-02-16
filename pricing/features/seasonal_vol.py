"""Seasonal (time-of-day) volatility estimation.

Computes sigma_tod per TOD bucket using tick-time realized variance:
    sigma_tod(b) = median across days of sqrt(sum(dx^2) / sum(dt))

This uses the "sum-of-squares / sum-of-time" estimator, which is robust to
microstructure noise and irregular tick arrivals.

The curve is computed once on the full dataset (intentional look-ahead for
calibration; for live use, compute on historical data only).

Ported from pricer_calibration/features/seasonal_vol.py.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SeasonalVolCurve:
    """Stores sigma_tod per TOD bucket and provides lookup.

    sigma_tod values are in per-sqrt(sec) units.
    """

    bucket_minutes: int
    sigma_tod: np.ndarray  # shape (n_buckets,), per-sqrt(sec)

    @property
    def n_buckets(self) -> int:
        return len(self.sigma_tod)

    def bucket_index(self, hour: int, minute: int) -> int:
        return (hour * 60 + minute) // self.bucket_minutes

    def bucket_index_from_ms(self, t_ms: int) -> int:
        total_seconds = (t_ms // 1000) % 86400
        hour = total_seconds // 3600
        minute = (total_seconds % 3600) // 60
        return self.bucket_index(hour, minute)

    def sigma_at_bucket(self, b: int) -> float:
        return float(self.sigma_tod[b % self.n_buckets])

    def sigma_at_ms(self, t_ms: int) -> float:
        return self.sigma_at_bucket(self.bucket_index_from_ms(t_ms))

    def lookup_array(self, t_ms: np.ndarray) -> np.ndarray:
        """Vectorized lookup: return sigma_tod for an array of epoch-ms timestamps."""
        total_seconds = (t_ms // 1000) % 86400
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        buckets = ((hours * 60 + minutes) // self.bucket_minutes).astype(int)
        return self.sigma_tod[buckets]


def compute_seasonal_vol(
    bbo: pd.DataFrame,
    bucket_minutes: int = 5,
    smoothing_window: int = 3,
    floor: float = 1e-10,
    winsorize_quantile: float = 0.01,
    ts_col: str = "ts_recv",
    mid_col: str = "mid_px",
) -> SeasonalVolCurve:
    """Compute seasonal volatility from BBO data using tick-time realized variance.

    Uses "sum-of-squares / sum-of-time" estimator:
        var_per_sec = sum(dx_i^2) / sum(dt_i)
        sigma = sqrt(var_per_sec)

    For each TOD bucket:
        1. Group ticks by (day, bucket)
        2. For each day's bucket: sigma_day = sqrt(sum(dx^2) / sum(dt))
        3. Aggregate across days: sigma_tod = median(sigma_day)

    Args:
        bbo: DataFrame with timestamp and mid price columns.
        bucket_minutes: TOD bucket width in minutes (default 5 -> 288 buckets/day).
        smoothing_window: Circular moving average window (0 = no smoothing).
        floor: Minimum sigma_tod value.
        winsorize_quantile: Winsorize squared returns at this quantile (0 = disable).
        ts_col: Name of timestamp column (epoch ms).
        mid_col: Name of mid price column.

    Returns:
        SeasonalVolCurve with sigma_tod per bucket in per-sqrt(sec) units.
    """
    n_buckets = 24 * 60 // bucket_minutes

    ts = bbo[ts_col].values
    mid = bbo[mid_col].values

    # Find where mid actually changed (filter duplicate ticks)
    changed = np.zeros(len(mid), dtype=bool)
    changed[0] = True
    changed[1:] = mid[1:] != mid[:-1]

    ts_changed = ts[changed]
    mid_changed = mid[changed]

    if len(mid_changed) < 2:
        return SeasonalVolCurve(
            bucket_minutes=bucket_minutes,
            sigma_tod=np.full(n_buckets, floor),
        )

    # Tick-time log returns
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

    # Day and bucket indices
    day_idx = ts_tick // (86400 * 1000)
    total_seconds = (ts_tick // 1000) % 86400
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    bucket_idx = (hours * 60 + minutes) // bucket_minutes

    dx_sq = dx ** 2

    # Optional winsorization
    if winsorize_quantile > 0:
        cap = np.quantile(dx_sq, 1 - winsorize_quantile)
        dx_sq = np.minimum(dx_sq, cap)

    # For each bucket: compute per-day sigma, then median across days
    sigma_tod = np.full(n_buckets, floor)
    unique_days = np.unique(day_idx)

    for b in range(n_buckets):
        bucket_mask = bucket_idx == b
        if bucket_mask.sum() < 10:
            continue

        day_sigmas = []
        for d in unique_days:
            day_bucket_mask = bucket_mask & (day_idx == d)
            if day_bucket_mask.sum() < 2:
                continue
            sum_dx_sq = np.sum(dx_sq[day_bucket_mask])
            sum_dt = np.sum(dt_sec[day_bucket_mask])
            if sum_dt > 0:
                day_sigmas.append(np.sqrt(sum_dx_sq / sum_dt))

        if len(day_sigmas) >= 1:
            sigma_tod[b] = max(np.median(day_sigmas), floor)

    # Optional smoothing (circular moving average)
    if smoothing_window > 0 and smoothing_window < n_buckets:
        padded = np.concatenate([
            sigma_tod[-smoothing_window:], sigma_tod, sigma_tod[:smoothing_window]
        ])
        kernel = np.ones(2 * smoothing_window + 1) / (2 * smoothing_window + 1)
        smoothed = np.convolve(padded, kernel, mode="valid")
        sigma_tod = smoothed[:n_buckets]
        sigma_tod = np.maximum(sigma_tod, floor)

    return SeasonalVolCurve(bucket_minutes=bucket_minutes, sigma_tod=sigma_tod)
