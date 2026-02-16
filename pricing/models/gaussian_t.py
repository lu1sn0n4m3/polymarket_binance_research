"""Gaussian volatility + adaptive Student-t tails.

Two-stage model:
  Stage 1 (frozen): sigma_eff from QLIKE calibration
    sigma_eff = (a0 + a1 * sqrt(tau_min)) * sigma_tod * sigma_rel^beta

  Stage 2 (fitted): state-dependent degrees-of-freedom
    eta = b0 + b_stale * log1p(time_since_move) + b_sess * session_bump + b_tau * log1p(tau)
    nu  = nu_min + (nu_max - nu_min) * sigmoid(eta)

  Variance-preserving scale so changing nu doesn't distort the vol forecast:
    scale = sqrt((nu - 2) / nu)
    p = T_nu(-z / scale)

5 free parameters: b0, b_stale, b_sess, b_tau, nu_max.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from pricing.models.base import Model

NU_MIN = 3.0  # hard floor — ensures variance exists


class GaussianTModel(Model):

    name = "gaussian_t"

    def __init__(self, vol_params_path: str | Path = "pricing/output/gaussian_vol_params.json"):
        """Load frozen sigma_eff parameters from QLIKE calibration."""
        path = Path(vol_params_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Frozen vol params not found at {path}. "
                "Run QLIKE calibration first (calibrate_vol with gaussian model)."
            )
        with open(path) as f:
            vp = json.load(f)
        self.a0 = vp["a0"]
        self.a1 = vp["a1"]
        self.beta = vp["beta"]

    def _sigma_eff(self, tau, features):
        """Compute frozen sigma_eff."""
        tau_min = tau / 60.0
        a_tau = self.a0 + self.a1 * np.sqrt(np.maximum(tau_min, 0))
        return a_tau * features["sigma_tod"] * np.power(
            np.maximum(features["sigma_rel"], 1e-12), self.beta)

    def _nu(self, params, tau, features):
        """Compute state-dependent degrees-of-freedom."""
        b0 = params["b0"]
        b_stale = params["b_stale"]
        b_sess = params["b_sess"]
        b_tau = params["b_tau"]
        nu_max = params["nu_max"]

        hour = np.asarray(features["hour_et"], dtype=np.float64)
        bump_06 = np.exp(-((hour - 6.0) ** 2) / (2.0 * 1.5 ** 2))
        bump_20 = np.exp(-((hour - 20.0) ** 2) / (2.0 * 1.5 ** 2))
        sess = bump_06 + bump_20

        eta = (b0
               + b_stale * np.log1p(features["time_since_move"])
               + b_sess * sess
               + b_tau * np.log1p(tau))
        nu = NU_MIN + (nu_max - NU_MIN) / (1.0 + np.exp(-eta))
        return nu

    def predict(self, params, S, K, tau, features):
        sigma_eff = self._sigma_eff(tau, features)
        nu = self._nu(params, tau, features)

        # Variance-preserving scale: Var(scale * T_nu) = 1.0
        scale = np.sqrt((nu - 2.0) / nu)

        sqrt_tau = np.sqrt(np.maximum(tau, 1e-6))
        z = np.log(K / S) / (sigma_eff * sqrt_tau)

        p = student_t.cdf(-z / scale, df=nu)
        return np.clip(p, 1e-9, 1.0 - 1e-9)

    def predict_variance(self, params, S, K, tau, features):
        sigma_eff = self._sigma_eff(tau, features)
        return sigma_eff ** 2 * tau

    def param_names(self):
        return ["b0", "b_stale", "b_sess", "b_tau", "nu_max"]

    def initial_params(self):
        return {"b0": 2.0, "b_stale": 0.0, "b_sess": 0.0, "b_tau": 0.0, "nu_max": 30.0}

    def param_bounds(self):
        return {
            "b0": (-5.0, 10.0),
            "b_stale": (-2.0, 2.0),
            "b_sess": (-5.0, 5.0),
            "b_tau": (-2.0, 2.0),
            "nu_max": (5.0, 100.0),
        }

    def required_features(self):
        return ["sigma_tod", "sigma_rv", "sigma_rel", "time_since_move", "hour_et"]
