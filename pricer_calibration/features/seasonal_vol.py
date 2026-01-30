"""Seasonal (time-of-day) volatility estimation.

σ_tod(b) = median across days of σ_bucket_day(b)

where σ_bucket_day = sqrt(sum(dx_i^2) / sum(dt_i)) for ticks in that bucket.

This uses the "sum-of-squares / sum-of-time" realized variance estimator,
which is robust to microstructure noise and irregular tick arrivals.
We do NOT normalize individual returns by sqrt(dt) - that can blow up when
dt is tiny and injects quote-intensity artifacts.

The curve is computed once on the full dataset (intentional look-ahead).
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class SeasonalVolCurve:
    """Stores σ_tod per TOD bucket and provides lookup.

    σ_tod values are in per-√sec units, computed via realized variance:
    σ = sqrt(sum(dx^2) / sum(dt)) where dx = log returns, dt = time deltas.
    """

    bucket_minutes: int
    sigma_tod: np.ndarray  # shape (n_buckets,), per-√sec units
    sub_interval_sec: float = 1.0  # kept for compatibility

    @property
    def n_buckets(self) -> int:
        return len(self.sigma_tod)

    def bucket_index(self, hour: int, minute: int) -> int:
        """Return bucket index for a given (hour, minute)."""
        return (hour * 60 + minute) // self.bucket_minutes

    def bucket_index_from_ms(self, t_ms: int) -> int:
        """Return bucket index from epoch milliseconds (UTC)."""
        # Extract hour and minute from epoch ms
        total_seconds = (t_ms // 1000) % 86400
        hour = total_seconds // 3600
        minute = (total_seconds % 3600) // 60
        return self.bucket_index(hour, minute)

    def sigma_at_bucket(self, b: int) -> float:
        return float(self.sigma_tod[b % self.n_buckets])

    def sigma_at_ms(self, t_ms: int) -> float:
        """Return σ_tod for a given epoch-ms timestamp."""
        return self.sigma_at_bucket(self.bucket_index_from_ms(t_ms))


def compute_seasonal_vol(
    grid: pd.DataFrame,
    bucket_minutes: int = 5,
    smoothing_window: int = 0,
    floor: float = 1e-10,
    sub_interval_ms: int = 1_000,
) -> SeasonalVolCurve:
    """Compute seasonal volatility curve from grid data using MAD.

    Uses non-overlapping sub-intervals (default 1 second) to compute
    MAD-based volatility per TOD bucket, avoiding the zero-inflation problem
    of raw 100ms returns. The output σ_tod is in per-√(sub_interval)
    units, i.e. the MAD-scaled stdev of log returns over sub_interval_ms.

    Formula: σ_tod(b) = 1.4826 * MAD(r_b) where MAD = median(|r - median(r)|)

    Args:
        grid: DataFrame with columns [t, S, logS] where t is epoch ms.
        bucket_minutes: TOD bucket width in minutes.
        smoothing_window: Circular moving average window (0 = no smoothing).
        floor: Minimum σ_tod value.
        sub_interval_ms: Sub-interval for computing returns (default 1000ms = 1s).
            Using 1s avoids zero-inflation on 100ms returns.

    Returns:
        SeasonalVolCurve with σ_tod per bucket. σ_tod is the MAD-scaled
        stdev of log returns over sub_interval_ms. The EWMA module handles
        conversion to the 100ms grid step.
    """
    n_buckets = 24 * 60 // bucket_minutes
    t_ms = grid["t"].values
    logS = grid["logS"].values

    # Subsample the grid at sub_interval_ms spacing to get 1s returns
    step = sub_interval_ms // (t_ms[1] - t_ms[0]) if len(t_ms) > 1 else 1
    step = max(step, 1)

    t_sub = t_ms[::step]
    logS_sub = logS[::step]
    r_sub = np.diff(logS_sub)
    t_sub_r = t_sub[1:]  # timestamps for returns

    # Compute bucket index for each sub-interval return
    total_seconds = (t_sub_r // 1000) % 86400
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    bucket_idx = (hours * 60 + minutes) // bucket_minutes

    sigma_tod = np.full(n_buckets, floor)

    for b in range(n_buckets):
        mask = bucket_idx == b
        if mask.sum() < 10:
            continue
        rb = r_sub[mask]

        # Use MAD (Median Absolute Deviation) estimator as specified in task.md:
        # σ_tod(b) = 1.4826 * MAD where MAD = median(|r - median(r)|)
        # The 1.4826 factor makes it consistent with Gaussian σ.
        # Computing on coarser sub-intervals (e.g., 1s or 10s) avoids the
        # zero-inflation problem of raw 100ms returns.
        med = np.median(rb)
        mad = np.median(np.abs(rb - med))
        sigma_mad = 1.4826 * mad
        sigma_tod[b] = max(sigma_mad, floor)

    # Optional smoothing (circular moving average)
    if smoothing_window > 0 and smoothing_window < n_buckets:
        padded = np.concatenate([sigma_tod[-smoothing_window:], sigma_tod, sigma_tod[:smoothing_window]])
        kernel = np.ones(2 * smoothing_window + 1) / (2 * smoothing_window + 1)
        smoothed = np.convolve(padded, kernel, mode="valid")
        # Trim to original length
        sigma_tod = smoothed[:n_buckets]
        sigma_tod = np.maximum(sigma_tod, floor)

    return SeasonalVolCurve(
        bucket_minutes=bucket_minutes,
        sigma_tod=sigma_tod,
        sub_interval_sec=sub_interval_ms / 1000.0,
    )


def compute_seasonal_vol_ticktime(
    bbo: pd.DataFrame,
    bucket_minutes: int = 5,
    smoothing_window: int = 0,
    floor: float = 1e-10,
    target_interval_sec: float = 1.0,
    winsorize_quantile: float = 0.01,
) -> SeasonalVolCurve:
    """Compute seasonal volatility from raw BBO using realized variance.

    Uses "sum-of-squares / sum-of-time" estimator:
        var_per_sec = sum(dx_i^2) / sum(dt_i)
        sigma = sqrt(var_per_sec)

    This is robust to irregular tick arrivals and doesn't blow up when dt is tiny.
    We do NOT normalize individual returns by sqrt(dt).

    For each TOD bucket, we:
    1. Group ticks by (day, bucket)
    2. For each day's bucket: sigma_day = sqrt(sum(dx^2) / sum(dt))
    3. Aggregate across days: sigma_tod = median(sigma_day)

    Args:
        bbo: DataFrame with columns [ts_event, mid] where ts_event is epoch ms.
        bucket_minutes: TOD bucket width in minutes.
        smoothing_window: Circular moving average window (0 = no smoothing).
        floor: Minimum σ_tod value.
        target_interval_sec: Unused, kept for API compatibility.
        winsorize_quantile: Winsorize squared returns at this quantile to reduce
            outlier influence. Set to 0 to disable.

    Returns:
        SeasonalVolCurve with σ_tod per bucket in per-√sec units.
    """
    n_buckets = 24 * 60 // bucket_minutes

    ts = bbo["ts_event"].values
    mid = bbo["mid"].values

    # Find where mid actually changed
    changed = np.zeros(len(mid), dtype=bool)
    changed[0] = True
    changed[1:] = mid[1:] != mid[:-1]

    ts_changed = ts[changed]
    mid_changed = mid[changed]

    if len(mid_changed) < 2:
        return SeasonalVolCurve(
            bucket_minutes=bucket_minutes,
            sigma_tod=np.full(n_buckets, floor),
            sub_interval_sec=1.0,
        )

    # Compute tick-time returns (NOT normalized by sqrt(dt))
    log_mid = np.log(mid_changed)
    dx = np.diff(log_mid)  # log return between consecutive changes
    dt_ms = np.diff(ts_changed)
    dt_sec = dt_ms / 1000.0
    ts_tick = ts_changed[1:]

    # Filter out zero/tiny dt (duplicate timestamps)
    dt_floor = 0.0001  # 0.1ms floor just to avoid division issues
    valid = dt_sec > dt_floor
    dx = dx[valid]
    dt_sec = dt_sec[valid]
    ts_tick = ts_tick[valid]

    # Extract day and bucket for each tick
    # Day: integer days since epoch
    day_idx = ts_tick // (86400 * 1000)
    # Bucket: TOD bucket index
    total_seconds = (ts_tick // 1000) % 86400
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    bucket_idx = (hours * 60 + minutes) // bucket_minutes

    # Squared returns
    dx_sq = dx ** 2

    # Optional winsorization of squared returns (robust to outliers)
    if winsorize_quantile > 0:
        cap = np.quantile(dx_sq, 1 - winsorize_quantile)
        dx_sq = np.minimum(dx_sq, cap)

    # For each bucket, compute sigma for each day, then take median across days
    sigma_tod = np.full(n_buckets, floor)

    unique_days = np.unique(day_idx)

    for b in range(n_buckets):
        bucket_mask = bucket_idx == b

        if bucket_mask.sum() < 10:
            continue

        # Compute sigma for each day in this bucket
        day_sigmas = []
        for d in unique_days:
            day_bucket_mask = bucket_mask & (day_idx == d)
            if day_bucket_mask.sum() < 2:
                continue

            sum_dx_sq = np.sum(dx_sq[day_bucket_mask])
            sum_dt = np.sum(dt_sec[day_bucket_mask])

            if sum_dt > 0:
                var_per_sec = sum_dx_sq / sum_dt
                sigma_day = np.sqrt(var_per_sec)
                day_sigmas.append(sigma_day)

        if len(day_sigmas) >= 1:
            # Median across days (robust aggregation)
            sigma_tod[b] = max(np.median(day_sigmas), floor)

    # Optional smoothing (circular moving average)
    if smoothing_window > 0 and smoothing_window < n_buckets:
        padded = np.concatenate([sigma_tod[-smoothing_window:], sigma_tod, sigma_tod[:smoothing_window]])
        kernel = np.ones(2 * smoothing_window + 1) / (2 * smoothing_window + 1)
        smoothed = np.convolve(padded, kernel, mode="valid")
        sigma_tod = smoothed[:n_buckets]
        sigma_tod = np.maximum(sigma_tod, floor)

    return SeasonalVolCurve(
        bucket_minutes=bucket_minutes,
        sigma_tod=sigma_tod,
        sub_interval_sec=1.0,  # per-√sec units
    )
