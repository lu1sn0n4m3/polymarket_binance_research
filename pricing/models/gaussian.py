"""Gaussian model with variance-first decomposition and horizon-dependent shrinkage.

    v = c^2 * sigma_tod^2 * tau^alpha * [ m + (1-m) * sigma_rel^2 ]

    m = sigmoid(k0 + k1*log(tau))

    p = Phi(-k / sqrt(v))   where k = ln(K/S)

The shrinkage weight m pulls the variance forecast toward the seasonal baseline
(sigma_rel=1) at longer horizons. At short horizons m -> 0, giving full regime
pass-through. At long horizons m -> 1, reverting toward the seasonal curve.

4 parameters: c, alpha, k0, k1.
"""

import numpy as np
from scipy.stats import norm

from pricing.models.base import Model


def _expit(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


class GaussianModel(Model):

    name = "gaussian"

    def predict(self, params, S, K, tau, features):
        v = self.predict_variance(params, S, K, tau, features)
        k = np.log(K / S)
        p = norm.cdf(-k / np.sqrt(np.maximum(v, 1e-20)))
        return np.clip(p, 1e-9, 1.0 - 1e-9)

    def _shrinkage_components(self, params, tau, features):
        """Compute shrinkage weight m, squared regime term, and factor f.

        Returns (m, sr_sq, f, log_tau)
        """
        k0, k1 = params["k0"], params["k1"]

        sigma_rel = np.maximum(features["sigma_rel"], 1e-12)
        log_tau = np.log(np.maximum(tau, 1e-6))

        z = k0 + k1 * log_tau
        m = _expit(z)

        sr_sq = sigma_rel ** 2
        f = m + (1.0 - m) * sr_sq

        return m, sr_sq, f, log_tau

    def predict_variance(self, params, S, K, tau, features):
        c = params["c"]
        alpha = params["alpha"]
        sigma_tod = features["sigma_tod"]

        _, _, f, _ = self._shrinkage_components(params, tau, features)

        base = sigma_tod ** 2 * np.power(np.maximum(tau, 1e-6), alpha)
        return c ** 2 * base * f

    def qlike_gradient(self, params, S, K, tau, features, log_return_sq):
        """Analytic QLIKE gradient for shrinkage model.

        L_Q = log(v) + r^2/v
        dL_Q/d(theta) = (1 - u) / v * dv/d(theta)   where u = r^2/v
        """
        c = params["c"]
        alpha = params["alpha"]
        sigma_tod = features["sigma_tod"]

        m, sr_sq, f, log_tau = self._shrinkage_components(params, tau, features)

        base = sigma_tod ** 2 * np.power(np.maximum(tau, 1e-6), alpha)
        v = np.maximum(c ** 2 * base * f, 1e-20)

        common = (1.0 - log_return_sq / v) / v

        # dv/dc = 2c * base * f = 2*v/c
        grad_c = common * (2.0 * v / c)

        # dv/dalpha = v * log(tau)
        grad_alpha = common * (v * log_tau)

        # dv/dk_j = c^2 * base * (1 - sr_sq) * m*(1-m) * dz/dk_j
        dm_factor = (1.0 - sr_sq) * m * (1.0 - m)
        base_k = common * (c ** 2 * base * dm_factor)

        grad_k0 = base_k * 1.0
        grad_k1 = base_k * log_tau

        return {
            "c": grad_c,
            "alpha": grad_alpha,
            "k0": grad_k0,
            "k1": grad_k1,
        }

    def param_names(self):
        return ["c", "alpha", "k0", "k1"]

    def initial_params(self):
        return {
            "c": 1.0,
            "alpha": 1.0,
            "k0": -3.0,
            "k1": 0.0,
        }

    def param_bounds(self):
        return {
            "c": (0.1, 3.0),
            "alpha": (0.5, 1.5),
            "k0": (-10.0, 5.0),
            "k1": (-3.0, 3.0),
        }

    def required_features(self):
        return ["sigma_tod", "sigma_rel"]
