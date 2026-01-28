"""Tests for EWMA volatility and shock statistic."""

import numpy as np
import pandas as pd
import pytest

from pricer_calibration.features.seasonal_vol import SeasonalVolCurve
from pricer_calibration.features.ewma_shock import compute_ewma_shock


def _make_grid(n, delta_ms=100, price=100.0, returns=None):
    """Helper to create a simple grid DataFrame."""
    t = np.arange(0, n * delta_ms, delta_ms, dtype=np.int64)
    S = np.full(n, price)
    logS = np.log(S)
    if returns is None:
        r = np.zeros(n)
    else:
        r = returns
    return pd.DataFrame({"t": t, "S": S, "logS": logS, "r": r})


def _constant_seasonal(sigma=1e-4, bucket_minutes=5, sub_interval_sec=10.0):
    """Create a constant seasonal vol curve.

    Note: sigma is in per-√(sub_interval_sec) units, matching the actual
    seasonal vol computation which uses 10s sub-intervals by default.
    """
    n_buckets = 24 * 60 // bucket_minutes
    return SeasonalVolCurve(
        bucket_minutes=bucket_minutes,
        sigma_tod=np.full(n_buckets, sigma),
        sub_interval_sec=sub_interval_sec,
    )


class TestAlpha:
    def test_alpha_from_halflife(self):
        """α = 1 - exp(-ln2 * Δ / H) should match expected values."""
        delta = 0.1
        H = 10.0
        alpha = 1.0 - np.exp(-np.log(2) * delta / H)
        assert abs(alpha - 0.00693) < 0.0001

        H = 20.0
        alpha = 1.0 - np.exp(-np.log(2) * delta / H)
        assert abs(alpha - 0.00347) < 0.0001


class TestEWMAShock:
    def test_calm_sigma_rel_near_one(self):
        """In calm conditions (u ~ N(0,1)), σ_rel should hover near 1 (floored)."""
        n = 50000
        # sigma_tod is in per-√(sub_interval_sec) units (default 10s)
        # For 100ms grid step: σ_step = σ_per_sec * √Δ = (σ_tod/√10) * √0.1
        sigma_tod = 1e-3  # per-√10s
        sigma_per_sec = sigma_tod / np.sqrt(10.0)
        sigma_step = sigma_per_sec * np.sqrt(0.1)

        rng = np.random.RandomState(42)
        returns = rng.normal(0, sigma_step, n)
        returns[0] = 0.0
        grid = _make_grid(n, returns=returns)
        seasonal = _constant_seasonal(sigma=sigma_tod, sub_interval_sec=10.0)

        result = compute_ewma_shock(grid, seasonal, ewma_half_life_sec=20.0)

        # σ_rel is floored at 1.0, so mean should be >= 1.0
        tail = result["sigma_rel"].values[-10000:]
        assert np.mean(tail) >= 1.0
        # Should be close to 1.0 in calm conditions (not inflated much)
        assert np.mean(tail) < 1.5

    def test_shock_rises_on_spike(self):
        """Shock z should spike when a large return occurs."""
        n = 1000
        # Use per-√10s sigma_tod, convert to step sigma
        sigma_tod = 1e-3  # per-√10s
        sigma_per_sec = sigma_tod / np.sqrt(10.0)
        sigma_step = sigma_per_sec * np.sqrt(0.1)

        returns = np.zeros(n)
        returns[500] = 10 * sigma_step  # 10-sigma move in step units
        grid = _make_grid(n, returns=returns)
        seasonal = _constant_seasonal(sigma=sigma_tod, sub_interval_sec=10.0)

        result = compute_ewma_shock(grid, seasonal, shock_M=5)

        z_before = result["z"].values[490]
        z_at_spike = result["z"].values[500]
        assert z_at_spike > z_before * 5  # z should jump significantly

    def test_shock_no_lookahead(self):
        """Shock at time k should only use data up to k."""
        n = 100
        sigma_tod = 1e-3  # per-√10s
        sigma_per_sec = sigma_tod / np.sqrt(10.0)
        sigma_step = sigma_per_sec * np.sqrt(0.1)

        returns = np.zeros(n)
        returns[50] = 20 * sigma_step  # big spike at t=50 in step units
        grid = _make_grid(n, returns=returns)
        seasonal = _constant_seasonal(sigma=sigma_tod, sub_interval_sec=10.0)

        result = compute_ewma_shock(grid, seasonal, shock_M=5)

        # z at t=49 should not see the spike at t=50
        assert result["z"].values[49] < 5  # no spike visible yet
        assert result["z"].values[50] > 10  # spike visible

    def test_no_nans(self):
        """Output should have no NaNs."""
        n = 500
        grid = _make_grid(n)
        seasonal = _constant_seasonal()

        result = compute_ewma_shock(grid, seasonal)

        for col in ["u", "v", "sigma_rel", "sigma_delta", "sigma_base", "z"]:
            assert np.isfinite(result[col].values).all(), f"NaN in {col}"

    def test_winsorization(self):
        """u_k² capping should prevent EWMA explosion."""
        n = 1000
        sigma_tod = 1e-3  # per-√10s
        sigma_per_sec = sigma_tod / np.sqrt(10.0)
        sigma_step = sigma_per_sec * np.sqrt(0.1)

        returns = np.zeros(n)
        returns[100] = 100 * sigma_step  # extreme outlier in step units
        grid = _make_grid(n, returns=returns)
        seasonal = _constant_seasonal(sigma=sigma_tod, sub_interval_sec=10.0)

        result = compute_ewma_shock(grid, seasonal, u_sq_cap=100.0)

        # v should not explode beyond reasonable values
        assert result["v"].values[101] <= 110  # capped at 100 + some decay

    def test_sigma_rel_floored_at_one(self):
        """σ_rel should be floored at 1.0, never deflate below seasonal baseline."""
        n = 1000
        sigma_tod = 1e-3  # per-√10s

        # All zero returns → EWMA v_k would decay to 0 without floor
        returns = np.zeros(n)
        grid = _make_grid(n, returns=returns)
        seasonal = _constant_seasonal(sigma=sigma_tod, sub_interval_sec=10.0)

        result = compute_ewma_shock(grid, seasonal, ewma_half_life_sec=20.0)

        # σ_rel should never go below 1.0
        assert (result["sigma_rel"].values >= 1.0).all()
