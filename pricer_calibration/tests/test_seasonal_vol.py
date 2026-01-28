"""Tests for seasonal volatility estimation via MAD on sub-intervals."""

import numpy as np
import pandas as pd
import pytest

from pricer_calibration.features.seasonal_vol import compute_seasonal_vol, SeasonalVolCurve


class TestSeasonalVol:
    def test_known_mad(self):
        """Known returns should produce expected MAD-scaled σ = 1.4826 * MAD."""
        # Create grid with 100ms spacing, enough data for 1s sub-intervals
        # Need at least 1s of data = 10 grid points per sub-interval
        n = 100000  # 10,000 seconds of data
        t = np.arange(0, n * 100, 100, dtype=np.int64)  # 100ms spacing

        # Generate S with known volatility
        rng = np.random.RandomState(42)
        sigma_per_1s = 0.001  # target σ_tod (per-√1s)

        # Build price path: sub-interval returns have stdev = sigma_per_1s
        # Sub-interval = 1s = 10 grid points
        step = 10
        n_subs = n // step
        sub_returns = rng.normal(0, sigma_per_1s, n_subs)

        # Build logS from sub-interval returns
        logS = np.zeros(n)
        for i in range(n_subs):
            start = i * step
            end = (i + 1) * step
            # Spread the sub-interval return across the 100ms grid
            logS[start:end] = np.log(100) + sub_returns[:i].sum()
        logS[-1] = np.log(100) + sub_returns.sum()

        S = np.exp(logS)
        grid = pd.DataFrame({"t": t, "S": S, "logS": logS})

        curve = compute_seasonal_vol(grid, bucket_minutes=5, smoothing_window=0, sub_interval_ms=1_000)

        # σ_tod should be close to sigma_per_1s for bucket 0
        # MAD for normal ≈ 0.6745 * σ, so MAD-scaled σ ≈ 1.4826 * 0.6745 * σ ≈ σ
        actual = curve.sigma_at_bucket(0)
        # Allow 50% tolerance due to finite sample and MAD estimator variance
        assert abs(actual - sigma_per_1s) / sigma_per_1s < 0.5

    def test_sigma_tod_positive(self):
        """All σ_tod values should be positive."""
        n = 50000
        t = np.arange(0, n * 100, 100, dtype=np.int64)
        rng = np.random.RandomState(123)

        # Build logS from cumulative returns
        r = rng.normal(0, 0.0005, n)
        r[0] = 0.0
        logS = np.log(100) + np.cumsum(r)
        S = np.exp(logS)

        grid = pd.DataFrame({"t": t, "S": S, "logS": logS})
        curve = compute_seasonal_vol(grid, bucket_minutes=5, smoothing_window=0)

        assert (curve.sigma_tod > 0).all()

    def test_floor_applied(self):
        """Buckets with zero returns should get the floor value."""
        # Need enough data for at least one 1s sub-interval
        n = 200  # 20 seconds
        t = np.arange(0, n * 100, 100, dtype=np.int64)
        logS = np.full(n, np.log(100))  # constant price → zero returns
        S = np.exp(logS)

        grid = pd.DataFrame({"t": t, "S": S, "logS": logS})
        floor = 1e-8
        curve = compute_seasonal_vol(grid, bucket_minutes=5, smoothing_window=0, floor=floor)

        # Bucket 0 has all-zero returns → MAD = 0 → should use floor
        assert curve.sigma_at_bucket(0) == floor

    def test_n_buckets(self):
        """Should have 288 buckets for 5-minute intervals."""
        n = 1000
        t = np.arange(0, n * 100, 100, dtype=np.int64)
        logS = np.full(n, np.log(100))
        S = np.exp(logS)

        grid = pd.DataFrame({"t": t, "S": S, "logS": logS})
        curve = compute_seasonal_vol(grid, bucket_minutes=5)
        assert curve.n_buckets == 288

    def test_bucket_index(self):
        """Bucket index calculation should be correct."""
        curve = SeasonalVolCurve(bucket_minutes=5, sigma_tod=np.ones(288), sub_interval_sec=10.0)
        assert curve.bucket_index(0, 0) == 0
        assert curve.bucket_index(0, 4) == 0
        assert curve.bucket_index(0, 5) == 1
        assert curve.bucket_index(12, 30) == 150
        assert curve.bucket_index(23, 55) == 287

    def test_sub_interval_sec_default(self):
        """SeasonalVolCurve should have sub_interval_sec field."""
        curve = SeasonalVolCurve(bucket_minutes=5, sigma_tod=np.ones(288))
        assert curve.sub_interval_sec == 1.0

    def test_sub_interval_custom(self):
        """SeasonalVolCurve should accept custom sub_interval_sec."""
        curve = SeasonalVolCurve(bucket_minutes=5, sigma_tod=np.ones(288), sub_interval_sec=1.0)
        assert curve.sub_interval_sec == 1.0
