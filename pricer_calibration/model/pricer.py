"""Pricer: p = 1 - F(x) with shock inflation.

Supports Student-t and Normal distributions via an abstract CDF interface.
"""

from typing import Protocol

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Abstract CDF interface
# ---------------------------------------------------------------------------

class DistributionCDF(Protocol):
    """Protocol for a symmetric location-scale CDF."""

    def cdf(self, x: np.ndarray | float) -> np.ndarray | float: ...
    def pdf(self, x: np.ndarray | float) -> np.ndarray | float: ...


class StudentT:
    """Student-t CDF wrapper."""

    def __init__(self, nu: float):
        self.nu = nu
        self._dist = stats.t(df=nu)

    def cdf(self, x):
        return self._dist.cdf(x)

    def pdf(self, x):
        return self._dist.pdf(x)


class Normal:
    """Standard Normal CDF wrapper."""

    def __init__(self):
        self._dist = stats.norm

    def cdf(self, x):
        return self._dist.cdf(x)

    def pdf(self, x):
        return self._dist.pdf(x)


def get_distribution(dist: str = "student_t", nu: float = 6.0) -> DistributionCDF:
    """Factory for distribution objects."""
    if dist == "normal":
        return Normal()
    return StudentT(nu)


# ---------------------------------------------------------------------------
# Shock inflation
# ---------------------------------------------------------------------------

def kappa(z: np.ndarray | float, gamma: float, c: float) -> np.ndarray | float:
    """Shock inflation hinge: κ(z) = 1 + γ * max(0, z - c)."""
    return 1.0 + gamma * np.maximum(0.0, z - c)


# ---------------------------------------------------------------------------
# Core pricer
# ---------------------------------------------------------------------------

def price_probability(
    S: np.ndarray | float,
    K: np.ndarray | float,
    tau: np.ndarray | float,
    sigma_base: np.ndarray | float,
    z: np.ndarray | float,
    gamma: float = 0.0,
    c: float = 3.5,
    nu: float = 6.0,
    dist: str = "student_t",
    a: float = 1.0,
    b: float = 0.0,
) -> np.ndarray | float:
    """Compute fair probability p = P(S_T > K).

    Args:
        S: Current price.
        K: Strike price.
        tau: Time to expiry in seconds.
        sigma_base: Base volatility (per √sec).
        z: Shock statistic.
        gamma: Shock inflation slope.
        c: Shock threshold.
        nu: Student-t degrees of freedom.
        dist: "student_t" or "normal".
        a: Global scale multiplier for sigma_base.
        b: Variance floor (per √sec). Effective σ = sqrt((a*σ_base)² + b²).

    Returns:
        p in (0, 1), clipped to [1e-9, 1-1e-9].
    """
    distribution = get_distribution(dist, nu)

    # Scale sigma_base by a, then apply shock inflation
    sigma_scaled = a * sigma_base * kappa(z, gamma, c)

    # Add variance floor: σ_eff = sqrt(σ_scaled² + b²)
    if b > 0:
        sigma_eff = np.sqrt(sigma_scaled**2 + b**2)
    else:
        sigma_eff = sigma_scaled

    sqrt_tau = np.sqrt(np.maximum(tau, 1e-6))
    x = np.log(K / S) / (sigma_eff * sqrt_tau)

    p = 1.0 - distribution.cdf(x)
    return np.clip(p, 1e-9, 1.0 - 1e-9)
