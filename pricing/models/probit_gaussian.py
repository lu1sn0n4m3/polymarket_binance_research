"""Two-stage probit-Gaussian model.

Stage 1: Effective volatility
    sigma_eff = (a0 + a1 * sqrt(tau_min)) * sigma_tod^(1-beta) * sigma_rv^beta
    z = (ln(K/S) + 0.5 * sigma_eff^2 * tau) / (sigma_eff * sqrt(tau))

Stage 2: Probit calibration
    z' = b0 + b1 * score + b2 * log(tau) + b3 * log(sigma_rel) + b4 * (log sigma_rel)^2
    p = Phi(z')
    where score = -z

8 parameters total. Ported from pricer_calibration/run/run_calibration.py.
"""

import numpy as np
from scipy.stats import norm

from pricing.models.base import Model


class ProbitGaussianModel(Model):

    name = "probit_gaussian"

    def predict(self, params, S, K, tau, features):
        a0 = params["a0"]
        a1 = params["a1"]
        beta = params["beta"]
        sigma_tod = features["sigma_tod"]
        sigma_rv = features["sigma_rv"]
        sigma_rel = features["sigma_rel"]

        # Stage 1: effective volatility
        tau_min = tau / 60.0
        a_tau = a0 + a1 * np.sqrt(np.maximum(tau_min, 0))
        sigma_eff = a_tau * sigma_tod * np.power(np.maximum(sigma_rel, 1e-12), beta)

        sqrt_tau = np.sqrt(np.maximum(tau, 1e-6))
        z = (np.log(K / S) + 0.5 * sigma_eff ** 2 * tau) / (sigma_eff * sqrt_tau)

        # Stage 2: probit calibration
        score = -z
        log_tau = np.log(np.maximum(tau, 1.0))
        log_sr = np.log(np.maximum(sigma_rel, 1e-6))

        zp = (params["b0"]
              + params["b1"] * score
              + params["b2"] * log_tau
              + params["b3"] * log_sr
              + params["b4"] * log_sr ** 2)

        p = norm.cdf(zp)
        return np.clip(p, 1e-9, 1.0 - 1e-9)

    def param_names(self):
        return ["a0", "a1", "beta", "b0", "b1", "b2", "b3", "b4"]

    def initial_params(self):
        return {
            "a0": 0.5, "a1": 0.05, "beta": 0.5,
            "b0": 0.0, "b1": 1.0, "b2": 0.0, "b3": 0.0, "b4": 0.0,
        }

    def param_bounds(self):
        return {
            "a0": (0.1, 3.0),
            "a1": (-0.5, 0.5),
            "beta": (0.0, 2.0),
            "b0": (-2.0, 2.0),
            "b1": (0.3, 2.0),
            "b2": (-0.5, 0.5),
            "b3": (-0.5, 0.5),
            "b4": (-0.5, 0.5),
        }

    def required_features(self):
        return ["sigma_tod", "sigma_rv", "sigma_rel"]
