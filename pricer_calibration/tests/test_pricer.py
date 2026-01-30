"""Tests for the pricer module."""

import numpy as np
import pytest

from pricer_calibration.model.pricer import price_probability, kappa
from pricer_calibration.model.sensitivities import dp_dS, dp_dlogS, delta_p_one_sided


class TestKappa:
    def test_below_threshold(self):
        assert kappa(2.0, gamma=1.0, c=3.5) == 1.0

    def test_above_threshold(self):
        assert kappa(5.0, gamma=1.0, c=3.5) == pytest.approx(2.5)

    def test_gamma_zero(self):
        assert kappa(10.0, gamma=0.0, c=3.5) == 1.0

    def test_vectorized(self):
        z = np.array([1.0, 3.5, 5.0])
        result = kappa(z, gamma=2.0, c=3.5)
        np.testing.assert_allclose(result, [1.0, 1.0, 4.0])


class TestPricer:
    def test_monotone_in_S(self):
        """Higher S → higher P(S_T > K)."""
        S_low = 100.0
        S_high = 101.0
        K = 100.0
        tau = 1800.0
        sigma_base = 0.001
        z = 0.0

        p_low = price_probability(S_low, K, tau, sigma_base, z)
        p_high = price_probability(S_high, K, tau, sigma_base, z)
        assert p_high > p_low

    def test_atm_near_half(self):
        """At-the-money with symmetric distribution should be ~0.5."""
        p = price_probability(S=100.0, K=100.0, tau=1800.0, sigma_base=0.001, z=0.0)
        assert abs(p - 0.5) < 0.01

    def test_deep_itm(self):
        """Deep in-the-money should be close to 1."""
        p = price_probability(S=110.0, K=100.0, tau=60.0, sigma_base=0.0001, z=0.0)
        assert p > 0.99

    def test_deep_otm(self):
        """Deep out-of-the-money should be close to 0."""
        p = price_probability(S=90.0, K=100.0, tau=60.0, sigma_base=0.0001, z=0.0)
        assert p < 0.01

    def test_clipping(self):
        """p should be clipped to [1e-9, 1-1e-9]."""
        p = price_probability(S=50.0, K=100.0, tau=1.0, sigma_base=0.0001, z=0.0)
        assert p >= 1e-9
        assert p <= 1.0 - 1e-9

    def test_correct_tail_direction(self):
        """p = 1 - Φ(x), so S > K → x < 0 → p > 0.5."""
        p = price_probability(S=101.0, K=100.0, tau=3600.0, sigma_base=0.001, z=0.0)
        assert p > 0.5

    def test_vectorized(self):
        """Should work with numpy arrays."""
        S = np.array([99.0, 100.0, 101.0])
        K = np.full(3, 100.0)
        tau = np.full(3, 1800.0)
        sigma = np.full(3, 0.001)
        z = np.zeros(3)

        p = price_probability(S, K, tau, sigma, z)
        assert p.shape == (3,)
        assert p[0] < p[1] < p[2]


class TestSensitivities:
    def test_dp_dS_positive(self):
        """dp/dS must be positive."""
        d = dp_dS(S=100.0, K=100.0, tau=1800.0, sigma_base=0.001, z=0.0)
        assert d > 0

    def test_dp_dlogS_equals_S_times_dp_dS(self):
        """dp/dlnS = S * dp/dS."""
        S = 100.0
        kwargs = dict(S=S, K=100.0, tau=1800.0, sigma_base=0.001, z=0.0)
        d_S = dp_dS(**kwargs)
        d_logS = dp_dlogS(**kwargs)
        assert abs(d_logS - S * d_S) < 1e-10

    def test_delta_p_signs(self):
        """Δp+ ≥ 0, Δp- ≤ 0 (typically)."""
        dp_plus, dp_minus = delta_p_one_sided(
            S=100.0, K=100.0, tau=1800.0, sigma_base=0.001, z=0.0,
            q_plus=0.001, q_minus=0.001,
        )
        assert dp_plus >= 0
        assert dp_minus <= 0
