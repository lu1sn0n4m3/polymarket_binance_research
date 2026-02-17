"""Online pricer (paper Section 9.2).

Loads frozen calibration parameters and computes binary option prices
from live market features. No dataset or pandas dependency at runtime.

Usage:
    pricer = Pricer.from_calibration("pricing/output")
    p_gauss, p_t = pricer.price(
        S=104000, K=103500, tau=1800,
        sigma_rv=3.5e-5, tsm=12.0, t_ms=1706745600000,
    )
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import norm, t as student_t

KAPPA = 600.0


class Pricer:
    """Lightweight online pricer with frozen calibration parameters.

    Implements the 6-step procedure from paper Section 9.2:
      1. Look up sigma_tod from seasonal curve using timestamp
      2. Compute sigma_rel = sigma_rv / sigma_tod
      3. Compute variance: v = c^2 * sigma_tod^2 * sigma_rel^{2*beta} * tau^alpha * Gamma(tsm)
      4. Compute log-strike: k = ln(K/S)
      5. Gaussian price: p = Phi(-k / sqrt(v))
      6. Student-t price: p = T_nu(-k / (s(nu) * sqrt(v)))
    """

    def __init__(
        self,
        vol_params: dict,
        nu: float,
        seasonal_sigma_tod: np.ndarray,
        bucket_minutes: int = 5,
    ):
        self.c = vol_params["c"]
        self.beta = vol_params["beta"]
        self.alpha = vol_params["alpha"]
        self.lam = vol_params["lam"]
        self.nu = nu
        self.s_nu = np.sqrt((nu - 2.0) / nu)
        self.sigma_tod = seasonal_sigma_tod
        self.bucket_minutes = bucket_minutes

    @classmethod
    def from_calibration(cls, output_dir: str | Path = "pricing/output") -> "Pricer":
        """Load from standard calibration output directory."""
        output_dir = Path(output_dir)

        # Vol params
        with open(output_dir / "gaussian_vol_params.json") as f:
            vp = json.load(f)
        vol_params = {k: vp[k] for k in ["c", "beta", "alpha", "lam"]}

        # Tail params
        fixed_path = output_dir / "fixed_t_params.json"
        if fixed_path.exists():
            with open(fixed_path) as f:
                tp = json.load(f)
            nu = tp["nu"]
        else:
            nu = 30.0  # near-Gaussian default

        # Seasonal vol curve
        import pandas as pd
        sv = pd.read_parquet(output_dir / "seasonal_vol.parquet")
        sigma_tod = sv["sigma_tod"].values
        bucket_minutes = 5

        return cls(vol_params, nu, sigma_tod, bucket_minutes)

    def _sigma_tod_at(self, t_ms):
        """Look up seasonal vol from epoch-ms timestamp(s)."""
        t_ms = np.asarray(t_ms, dtype=np.int64)
        total_seconds = (t_ms // 1000) % 86400
        minutes_of_day = total_seconds // 60
        bucket_idx = (minutes_of_day // self.bucket_minutes).astype(int)
        bucket_idx = np.clip(bucket_idx, 0, len(self.sigma_tod) - 1)
        return self.sigma_tod[bucket_idx]

    def _gamma(self, tsm):
        return 1.0 + self.lam * (1.0 - np.exp(-np.asarray(tsm, dtype=np.float64) / KAPPA))

    def price(self, S, K, tau, sigma_rv, tsm, t_ms):
        """Compute Gaussian and Student-t binary option prices.

        Args:
            S: Current mid price(s).
            K: Strike price(s).
            tau: Time to expiry in seconds.
            sigma_rv: EWMA realized volatility (per-sqrt-sec).
            tsm: Time since last mid-price change (seconds).
            t_ms: Current timestamp (epoch milliseconds, for sigma_tod lookup).

        Returns:
            (p_gauss, p_t): Gaussian and Student-t prices as numpy arrays.
        """
        S = np.asarray(S, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        tau = np.asarray(tau, dtype=np.float64)
        sigma_rv = np.asarray(sigma_rv, dtype=np.float64)

        # Step 1-2: sigma_tod and sigma_rel
        sigma_tod = self._sigma_tod_at(t_ms)
        sigma_rel = sigma_rv / np.maximum(sigma_tod, 1e-12)

        # Step 3: Variance
        gamma = self._gamma(tsm)
        v = (self.c ** 2
             * sigma_tod ** 2
             * np.power(np.maximum(sigma_rel, 1e-12), 2 * self.beta)
             * np.power(np.maximum(tau, 1e-6), self.alpha)
             * gamma)

        # Step 4: Log-strike
        k = np.log(K / S)

        # Step 5: Gaussian price
        sqrt_v = np.sqrt(np.maximum(v, 1e-20))
        p_gauss = norm.cdf(-k / sqrt_v)

        # Step 6: Student-t price
        p_t = student_t.cdf(-k / (self.s_nu * sqrt_v), df=self.nu)

        return (
            np.clip(p_gauss, 1e-9, 1.0 - 1e-9),
            np.clip(p_t, 1e-9, 1.0 - 1e-9),
        )
