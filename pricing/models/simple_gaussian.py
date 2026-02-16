"""Simplest possible model: Gaussian CDF with a single scale parameter.

    p = Phi(a * ln(S/K) / (sigma_rv * sqrt(tau)))

One parameter (a), requires only sigma_rv.
Use this as a template for new models.
"""

import numpy as np
from scipy.stats import norm

from pricing.models.base import Model


class SimpleGaussianModel(Model):

    name = "simple_gaussian"

    def predict(self, params, S, K, tau, features):
        a = params["a"]
        sigma_rv = features["sigma_rv"]
        sqrt_tau = np.sqrt(np.maximum(tau, 1e-6))
        z = a * np.log(S / K) / (sigma_rv * sqrt_tau)
        return np.clip(norm.cdf(z), 1e-9, 1.0 - 1e-9)

    def predict_variance(self, params, S, K, tau, features):
        a = params["a"]
        sigma_eff = features["sigma_rv"] / a
        return sigma_eff ** 2 * tau

    def param_names(self):
        return ["a"]

    def initial_params(self):
        return {"a": 1.0}

    def param_bounds(self):
        return {"a": (0.1, 10.0)}

    def required_features(self):
        return ["sigma_rv"]
