"""Sensitivities: dp/dS, dp/dlnS, and finite-horizon shocks Δp±(h)."""

import numpy as np

from pricer_calibration.model.pricer import get_distribution, kappa


def dp_dS(
    S: float,
    K: float,
    tau: float,
    sigma_base: float,
    z: float,
    gamma: float = 0.0,
    c: float = 3.5,
    nu: float = 6.0,
    dist: str = "student_t",
) -> float:
    """Local sensitivity dp/dS (positive)."""
    distribution = get_distribution(dist, nu)
    sigma_eff = sigma_base * kappa(z, gamma, c)
    sqrt_tau = np.sqrt(max(tau, 1e-6))
    x = np.log(K / S) / (sigma_eff * sqrt_tau)
    return float(distribution.pdf(x) / (S * sigma_eff * sqrt_tau))


def dp_dlogS(
    S: float,
    K: float,
    tau: float,
    sigma_base: float,
    z: float,
    gamma: float = 0.0,
    c: float = 3.5,
    nu: float = 6.0,
    dist: str = "student_t",
) -> float:
    """Sensitivity dp/d(lnS) = S * dp/dS."""
    distribution = get_distribution(dist, nu)
    sigma_eff = sigma_base * kappa(z, gamma, c)
    sqrt_tau = np.sqrt(max(tau, 1e-6))
    x = np.log(K / S) / (sigma_eff * sqrt_tau)
    return float(distribution.pdf(x) / (sigma_eff * sqrt_tau))


def delta_p_one_sided(
    S: float,
    K: float,
    tau: float,
    sigma_base: float,
    z: float,
    q_plus: float,
    q_minus: float,
    gamma: float = 0.0,
    c: float = 3.5,
    nu: float = 6.0,
    dist: str = "student_t",
) -> tuple[float, float]:
    """Finite-horizon one-sided probability shocks.

    Args:
        q_plus: Positive log-move for upside (> 0).
        q_minus: Positive log-move for downside (> 0).

    Returns:
        (Δp_plus, Δp_minus) where Δp_plus = p(S*exp(q+)) - p(S),
        Δp_minus = p(S*exp(-q-)) - p(S).
    """
    from pricer_calibration.model.pricer import price_probability

    kwargs = dict(K=K, tau=tau, sigma_base=sigma_base, z=z, gamma=gamma, c=c, nu=nu, dist=dist)
    p0 = float(price_probability(S=S, **kwargs))
    p_up = float(price_probability(S=S * np.exp(q_plus), **kwargs))
    p_down = float(price_probability(S=S * np.exp(-q_minus), **kwargs))

    return (p_up - p0, p_down - p0)
