"""Fixed-nu Student-t model (paper Section 6).

Two-stage model:
  Stage 1 (frozen): variance from QLIKE calibration (paper eq 9)
    v = c^2 * sigma_tod^2 * sigma_rel^{2*beta} * tau^alpha * Gamma(tsm)

  Stage 2 (fitted): single scalar nu, calibrated via MLE on z-residuals
    p = T_nu(-k / (s(nu) * sqrt(v)))

1 free parameter: nu.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from pricing.models.base import Model
from pricing.models.gaussian import KAPPA

NU_MIN = 3.0


class FixedTModel(Model):

    name = "fixed_t"

    def __init__(self, vol_params_path: str | Path = "pricing/output/gaussian_vol_params.json"):
        """Load frozen variance parameters from QLIKE calibration."""
        path = Path(vol_params_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Frozen vol params not found at {path}. "
                "Run QLIKE calibration first (calibrate_vol with gaussian model)."
            )
        with open(path) as f:
            vp = json.load(f)
        self.c = vp["c"]
        self.beta = vp["beta"]
        self.alpha = vp["alpha"]
        self.lam = vp["lam"]

    def _gamma(self, tsm):
        """Staleness adjustment Gamma(tsm) = 1 + lam * (1 - exp(-tsm/kappa))."""
        return 1.0 + self.lam * (1.0 - np.exp(-tsm / KAPPA))

    def _variance(self, tau, features):
        """Compute frozen variance v_t(tau) from Stage 1 params."""
        sigma_tod = features["sigma_tod"]
        sigma_rel = features["sigma_rel"]
        tsm = features["time_since_move"]
        gamma = self._gamma(tsm)
        return (self.c ** 2
                * sigma_tod ** 2
                * np.power(np.maximum(sigma_rel, 1e-12), 2 * self.beta)
                * np.power(np.maximum(tau, 1e-6), self.alpha)
                * gamma)

    def predict(self, params, S, K, tau, features):
        v = self._variance(tau, features)
        nu = params["nu"]
        scale = np.sqrt((nu - 2.0) / nu)
        k = np.log(K / S)
        p = student_t.cdf(-k / (scale * np.sqrt(np.maximum(v, 1e-20))), df=nu)
        return np.clip(p, 1e-9, 1.0 - 1e-9)

    def predict_variance(self, params, S, K, tau, features):
        return self._variance(tau, features)

    def param_names(self):
        return ["nu"]

    def initial_params(self):
        return {"nu": 8.0}

    def param_bounds(self):
        return {"nu": (3.01, 100.0)}

    def required_features(self):
        return ["sigma_tod", "sigma_rel", "time_since_move"]
