"""Pricer: p = 1 - Φ(x) with Normal distribution.

Core pricing formula (log-normal diffusion with convexity correction):
    x = (ln(K/S) + ½σ²τ) / (σ_eff * √τ)
    p = 1 - Φ(x)
"""

import numpy as np
from scipy.stats import norm


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
        a: Global scale multiplier for sigma_base.
        b: Variance floor (per √sec). Effective σ = sqrt((a*σ_base)² + b²).

    Returns:
        p in (0, 1), clipped to [1e-9, 1-1e-9].
    """
    # Scale sigma_base by a, then apply shock inflation
    sigma_scaled = a * sigma_base * kappa(z, gamma, c)

    # Add variance floor: σ_eff = sqrt(σ_scaled² + b²)
    if b > 0:
        sigma_eff = np.sqrt(sigma_scaled**2 + b**2)
    else:
        sigma_eff = sigma_scaled

    sqrt_tau = np.sqrt(np.maximum(tau, 1e-6))
    x = (np.log(K / S) + 0.5 * sigma_eff**2 * tau) / (sigma_eff * sqrt_tau)

    p = 1.0 - norm.cdf(x)
    return np.clip(p, 1e-9, 1.0 - 1e-9)
