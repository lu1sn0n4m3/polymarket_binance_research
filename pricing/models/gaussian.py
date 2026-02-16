"""Gaussian model with effective volatility.

    sigma_eff = (a0 + a1 * sqrt(tau_min)) * sigma_tod * sigma_rel^beta
    z = (ln(K/S) + 0.5 * sigma_eff^2 * tau) / (sigma_eff * sqrt(tau))
    p = Phi(-z)   [= P(S_T > K) under GBM with zero drift]

3 parameters: a0, a1, beta.
"""

import numpy as np
from scipy.stats import norm

from pricing.models.base import Model


class GaussianModel(Model):

    name = "gaussian"

    def predict(self, params, S, K, tau, features):
        a0 = params["a0"]
        a1 = params["a1"]
        beta = params["beta"]
        sigma_tod = features["sigma_tod"]
        sigma_rv = features["sigma_rv"]
        sigma_rel = features["sigma_rel"]

        tau_min = tau / 60.0
        a_tau = a0 + a1 * np.sqrt(np.maximum(tau_min, 0))
        sigma_eff = a_tau * sigma_tod * np.power(np.maximum(sigma_rel, 1e-12), beta)

        sqrt_tau = np.sqrt(np.maximum(tau, 1e-6))
        z = (np.log(K / S) + 0.5 * sigma_eff ** 2 * tau) / (sigma_eff * sqrt_tau)

        p = norm.cdf(-z)
        return np.clip(p, 1e-9, 1.0 - 1e-9)

    def predict_variance(self, params, S, K, tau, features):
        a0 = params["a0"]
        a1 = params["a1"]
        beta = params["beta"]
        tau_min = tau / 60.0
        a_tau = a0 + a1 * np.sqrt(np.maximum(tau_min, 0))
        sigma_eff = a_tau * features["sigma_tod"] * np.power(
            np.maximum(features["sigma_rel"], 1e-12), beta)
        return sigma_eff ** 2 * tau

    def param_names(self):
        return ["a0", "a1", "beta"]

    def initial_params(self):
        return {"a0": 0.5, "a1": 0.05, "beta": 0.5}

    def param_bounds(self):
        return {
            "a0": (0.1, 3.0),
            "a1": (-0.5, 0.5),
            "beta": (0.0, 2.0),
        }

    def required_features(self):
        return ["sigma_tod", "sigma_rv", "sigma_rel"]
