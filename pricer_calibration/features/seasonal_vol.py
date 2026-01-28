"""Seasonal (time-of-day) volatility estimation via MAD.

σ_tod(b) = 1.4826 * MAD(r_k for k in bucket b)

where MAD = median(|r - median(r)|) and the 1.4826 factor makes it
consistent with Gaussian standard deviation.

Two modes:
1. Grid-based (sub-interval): Computes on calendar-time grid returns.
   Works when grid captures most price changes.
2. Tick-time: Computes on returns between consecutive mid changes in raw BBO.
   Required when grid is too coarse and misses most changes.

The curve is computed once on the full dataset (intentional look-ahead).
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class SeasonalVolCurve:
    """Stores σ_tod per TOD bucket and provides lookup.

    σ_tod values are in per-√(sub_interval) units, i.e., the MAD-scaled
    stdev of log returns over sub_interval_sec seconds.
    """

    bucket_minutes: int
    sigma_tod: np.ndarray  # shape (n_buckets,)
    sub_interval_sec: float = 1.0  # timescale of sigma_tod (default 1s)

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
) -> SeasonalVolCurve:
    """Compute seasonal volatility from raw BBO using tick-time returns.

    This is the preferred method when the calendar-time grid misses most
    price changes (e.g., when grid only captures 8% of mid changes).

    Instead of using fixed calendar intervals, we:
    1. Find consecutive mid changes (where mid[i] != mid[i-1])
    2. Compute log returns between consecutive changes
    3. Scale returns by sqrt(dt) to normalize to per-√sec units
    4. Bucket by time-of-day and compute MAD

    Args:
        bbo: DataFrame with columns [ts_event, mid] where ts_event is epoch ms.
        bucket_minutes: TOD bucket width in minutes.
        smoothing_window: Circular moving average window (0 = no smoothing).
        floor: Minimum σ_tod value.
        target_interval_sec: Target interval for σ_tod units (default 1s).
            The output σ_tod will be in per-√(target_interval) units.

    Returns:
        SeasonalVolCurve with σ_tod per bucket in per-√(target_interval) units.
    """
    n_buckets = 24 * 60 // bucket_minutes

    ts = bbo["ts_event"].values
    mid = bbo["mid"].values

    # Find where mid actually changed
    changed = np.zeros(len(mid), dtype=bool)
    changed[0] = True  # Include first point
    changed[1:] = mid[1:] != mid[:-1]

    # Extract only the changed points
    ts_changed = ts[changed]
    mid_changed = mid[changed]

    if len(mid_changed) < 2:
        return SeasonalVolCurve(
            bucket_minutes=bucket_minutes,
            sigma_tod=np.full(n_buckets, floor),
            sub_interval_sec=target_interval_sec,
        )

    # Compute tick-time returns
    log_mid = np.log(mid_changed)
    r_tick = np.diff(log_mid)  # log return between consecutive changes
    dt_ms = np.diff(ts_changed)  # time delta in ms
    dt_sec = dt_ms / 1000.0

    # Normalize returns to per-√sec: r_normalized = r / sqrt(dt)
    # This gives us a "per-sqrt-second" return that's comparable across different dt
    # Filter out very small dt (< 1ms) to avoid numerical issues
    valid = dt_sec > 0.001
    r_tick = r_tick[valid]
    dt_sec = dt_sec[valid]
    ts_tick = ts_changed[1:][valid]  # timestamp of each return

    r_per_sqrt_sec = r_tick / np.sqrt(dt_sec)

    # Compute bucket index for each tick-time return
    total_seconds = (ts_tick // 1000) % 86400
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    bucket_idx = (hours * 60 + minutes) // bucket_minutes

    sigma_tod = np.full(n_buckets, floor)

    for b in range(n_buckets):
        mask = bucket_idx == b
        if mask.sum() < 10:
            continue
        rb = r_per_sqrt_sec[mask]

        # MAD estimator: σ = 1.4826 * median(|r - median(r)|)
        med = np.median(rb)
        mad = np.median(np.abs(rb - med))
        sigma_mad = 1.4826 * mad

        # Convert from per-√sec to per-√(target_interval)
        # σ(target) = σ(1sec) * √(target_interval)
        sigma_target = sigma_mad * np.sqrt(target_interval_sec)
        sigma_tod[b] = max(sigma_target, floor)

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
        sub_interval_sec=target_interval_sec,
    )
