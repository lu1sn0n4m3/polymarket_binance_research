"""Gaussian model with additive jump variance.

    sigma_eff = (a0 + a1 * sqrt(tau_min)) * sigma_tod * sigma_rel^beta
    jump_var_rate = j0 * sigma_tod^2 * (1 + j1 * stale_factor) * session_bump
    sigma_total = sqrt(sigma_eff^2 + jump_var_rate)
    z = (ln(K/S) + 0.5 * sigma_total^2 * tau) / (sigma_total * sqrt(tau))
    p = Phi(-z)

The diffusive part (sigma_eff) is identical to the Gaussian model. The additive
jump variance captures sudden repricings that the diffusion proxy can't see:
  - Baseline proportional to sigma_tod^2 (never zero in calm regimes)
  - Staleness ramp: sqrt(time_since_move / 60) increases risk after long quiet spells
  - Session bumps at 06:00 and 20:00 ET (pre-market open + evening transitions)

6 parameters: a0, a1, beta, j0, j1, j_session.
"""

import numpy as np
from scipy.stats import norm

from pricing.models.base import Model


class GaussianJumpModel(Model):

    name = "gaussian_jump"

    def predict(self, params, S, K, tau, features):
        a0 = params["a0"]
        a1 = params["a1"]
        beta = params["beta"]
        j0 = params["j0"]
        j1 = params["j1"]
        j_session = params["j_session"]

        sigma_tod = features["sigma_tod"]
        sigma_rel = features["sigma_rel"]
        time_since_move = features["time_since_move"]
        hour_et = features["hour_et"]

        # Diffusive component (same as gaussian model)
        tau_min = tau / 60.0
        a_tau = a0 + a1 * np.sqrt(np.maximum(tau_min, 0))
        sigma_eff = a_tau * sigma_tod * np.power(np.maximum(sigma_rel, 1e-12), beta)

        # Jump variance component
        stale_factor = np.sqrt(np.maximum(time_since_move, 0) / 60.0)

        hour = np.asarray(hour_et, dtype=np.float64)
        bump_06 = np.exp(-((hour - 6.0) ** 2) / (2.0 * 1.5 ** 2))
        bump_20 = np.exp(-((hour - 20.0) ** 2) / (2.0 * 1.5 ** 2))
        session_bump = 1.0 + j_session * (bump_06 + bump_20)

        jump_var_rate = j0 * sigma_tod ** 2 * (1.0 + j1 * stale_factor) * session_bump

        # Total variance per second
        sigma_total_sq = sigma_eff ** 2 + jump_var_rate
        sigma_total = np.sqrt(np.maximum(sigma_total_sq, 1e-20))

        sqrt_tau = np.sqrt(np.maximum(tau, 1e-6))
        z = (np.log(K / S) + 0.5 * sigma_total_sq * tau) / (sigma_total * sqrt_tau)

        p = norm.cdf(-z)
        return np.clip(p, 1e-9, 1.0 - 1e-9)

    def predict_variance(self, params, S, K, tau, features):
        a0 = params["a0"]
        a1 = params["a1"]
        beta = params["beta"]
        j0 = params["j0"]
        j1 = params["j1"]
        j_session = params["j_session"]

        sigma_tod = features["sigma_tod"]
        sigma_rel = features["sigma_rel"]
        time_since_move = features["time_since_move"]
        hour_et = features["hour_et"]

        tau_min = tau / 60.0
        a_tau = a0 + a1 * np.sqrt(np.maximum(tau_min, 0))
        sigma_eff = a_tau * sigma_tod * np.power(np.maximum(sigma_rel, 1e-12), beta)

        stale_factor = np.sqrt(np.maximum(time_since_move, 0) / 60.0)
        hour = np.asarray(hour_et, dtype=np.float64)
        bump_06 = np.exp(-((hour - 6.0) ** 2) / (2.0 * 1.5 ** 2))
        bump_20 = np.exp(-((hour - 20.0) ** 2) / (2.0 * 1.5 ** 2))
        session_bump = 1.0 + j_session * (bump_06 + bump_20)

        jump_var_rate = j0 * sigma_tod ** 2 * (1.0 + j1 * stale_factor) * session_bump
        sigma_total_sq = sigma_eff ** 2 + jump_var_rate
        return sigma_total_sq * tau

    def param_names(self):
        return ["a0", "a1", "beta", "j0", "j1", "j_session"]

    def initial_params(self):
        return {"a0": 1.0, "a1": 0.03, "beta": 0.7, "j0": 0.1, "j1": 0.5, "j_session": 0.5}

    def param_bounds(self):
        return {
            "a0": (0.1, 3.0),
            "a1": (-0.5, 0.5),
            "beta": (0.0, 2.0),
            "j0": (0.0, 2.0),
            "j1": (0.0, 5.0),
            "j_session": (0.0, 5.0),
        }

    def required_features(self):
        return ["sigma_tod", "sigma_rv", "sigma_rel", "time_since_move", "hour_et"]
