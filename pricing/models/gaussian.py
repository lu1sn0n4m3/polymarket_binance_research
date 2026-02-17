"""Gaussian model with variance-first decomposition (paper eq 4).

    v_t(tau; theta) = c^2 * sigma_tod^2 * sigma_rel^{2*beta} * tau^alpha * Gamma(tsm)
    Gamma(tsm) = 1 + lam * (1 - exp(-tsm / kappa))

    p = Phi(-k / sqrt(v))   where k = ln(K/S)

4 parameters: c, beta, alpha, lam.  kappa is fixed.
"""

import numpy as np
from scipy.stats import norm

from pricing.models.base import Model

KAPPA = 600.0  # staleness saturation time (seconds), fixed


class GaussianModel(Model):

    name = "gaussian"

    def _gamma(self, tsm, lam):
        """Staleness adjustment Gamma(tsm) = 1 + lam * (1 - exp(-tsm/kappa))."""
        return 1.0 + lam * (1.0 - np.exp(-tsm / KAPPA))

    def predict(self, params, S, K, tau, features):
        v = self.predict_variance(params, S, K, tau, features)
        k = np.log(K / S)
        p = norm.cdf(-k / np.sqrt(np.maximum(v, 1e-20)))
        return np.clip(p, 1e-9, 1.0 - 1e-9)

    def predict_variance(self, params, S, K, tau, features):
        c = params["c"]
        beta = params["beta"]
        alpha = params["alpha"]
        lam = params["lam"]

        sigma_tod = features["sigma_tod"]
        sigma_rel = features["sigma_rel"]
        tsm = features["time_since_move"]

        gamma = self._gamma(tsm, lam)
        v = (c ** 2
             * sigma_tod ** 2
             * np.power(np.maximum(sigma_rel, 1e-12), 2 * beta)
             * np.power(np.maximum(tau, 1e-6), alpha)
             * gamma)
        return v

    def qlike_gradient(self, params, S, K, tau, features, log_return_sq):
        """Analytic QLIKE gradient (paper eqs 33-36).

        dL_Q/d(theta_j) = (1 - r^2/v) / v * dv/d(theta_j)

        where:
            dv/dc   = 2v/c
            dv/dbeta = 2v * ln(sigma_rel)
            dv/dalpha = v * ln(tau)
            dv/dlam  = v / Gamma(tsm) * (1 - exp(-tsm/kappa))
        """
        v = self.predict_variance(params, S, K, tau, features)
        v = np.maximum(v, 1e-20)

        common = (1.0 - log_return_sq / v) / v

        c = params["c"]
        lam = params["lam"]
        sigma_rel = features["sigma_rel"]
        tsm = features["time_since_move"]
        gamma = self._gamma(tsm, lam)

        return {
            "c": common * (2.0 * v / c),
            "beta": common * (2.0 * v * np.log(np.maximum(sigma_rel, 1e-12))),
            "alpha": common * (v * np.log(np.maximum(tau, 1e-6))),
            "lam": common * (v / gamma * (1.0 - np.exp(-tsm / KAPPA))),
        }

    def param_names(self):
        return ["c", "beta", "alpha", "lam"]

    def initial_params(self):
        return {"c": 1.0, "beta": 0.5, "alpha": 1.0, "lam": 0.0}

    def param_bounds(self):
        return {
            "c": (0.1, 3.0),
            "beta": (0.0, 2.0),
            "alpha": (0.5, 1.5),
            "lam": (0.0, 2.0),
        }

    def required_features(self):
        return ["sigma_tod", "sigma_rel", "time_since_move"]
