"""Online pricer.

Loads frozen calibration parameters and computes binary option prices
from live market features. No dataset or pandas dependency at runtime.

Usage:
    pricer = Pricer.from_calibration("pricing/output")
    p_gauss, p_t = pricer.price(
        S=104000, K=103500, tau=1800,
        sigma_rv=3.5e-5, t_ms=1706745600000,
    )
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import norm, t as student_t


def _is_weekend_ms(t_ms: np.ndarray) -> np.ndarray:
    """Vectorized weekend detection from epoch ms (UTC)."""
    days = np.asarray(t_ms, dtype=np.int64) // 86_400_000
    dow = (days + 3) % 7  # 0=Mon ... 6=Sun
    return dow >= 5


def _expit(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


class Pricer:
    """Lightweight online pricer with frozen calibration parameters.

    v = c^2 * sigma_tod^2 * tau^alpha * [m + (1-m)*sigma_rel^2]
    m = sigmoid(k0 + k1*log(tau))
    """

    def __init__(
        self,
        vol_params: dict,
        nu: float,
        sigma_tod_weekday: np.ndarray,
        sigma_tod_weekend: np.ndarray,
        bucket_minutes: int = 5,
    ):
        self.c = vol_params["c"]
        self.alpha = vol_params["alpha"]
        self.k0 = vol_params.get("k0", -100.0)
        self.k1 = vol_params.get("k1", 0.0)
        self.nu = nu
        self.s_nu = np.sqrt((nu - 2.0) / nu)
        self.sigma_tod_weekday = sigma_tod_weekday
        self.sigma_tod_weekend = sigma_tod_weekend
        self.bucket_minutes = bucket_minutes

    @classmethod
    def from_calibration(cls, output_dir: str | Path = "pricing/output") -> "Pricer":
        """Load from standard calibration output directory."""
        output_dir = Path(output_dir)

        with open(output_dir / "gaussian_vol_params.json") as f:
            vp = json.load(f)
        vol_params = {k: vp[k] for k in ["c", "alpha"]}
        for k in ["k0", "k1"]:
            if k in vp:
                vol_params[k] = vp[k]

        fixed_path = output_dir / "fixed_t_params.json"
        if fixed_path.exists():
            with open(fixed_path) as f:
                tp = json.load(f)
            nu = tp["nu"]
        else:
            nu = 30.0

        import pandas as pd
        sv_wd = pd.read_parquet(output_dir / "seasonal_vol_weekday.parquet")
        sv_we = pd.read_parquet(output_dir / "seasonal_vol_weekend.parquet")
        bucket_minutes = 5

        return cls(vol_params, nu, sv_wd["sigma_tod"].values, sv_we["sigma_tod"].values, bucket_minutes)

    def _sigma_tod_integrated(self, t_ms, tau):
        """Integrated remaining seasonal vol: RMS average of sigma_tod over [t, t+tau]."""
        t_ms = np.asarray(t_ms, dtype=np.int64)
        tau = np.asarray(tau, dtype=np.float64)
        wknd = _is_weekend_ms(t_ms)

        result = self._integrate_curve(self.sigma_tod_weekday, t_ms, tau)
        if wknd.any():
            result[wknd] = self._integrate_curve(self.sigma_tod_weekend, t_ms[wknd], tau[wknd])
        return result

    def _integrate_curve(self, sigma_tod, t_ms, tau):
        """Integrate a single sigma_tod curve over [t, t+tau]."""
        bucket_sec = self.bucket_minutes * 60
        n_bkt = len(sigma_tod)
        sigma_sq = sigma_tod ** 2

        start_sec = ((t_ms // 1000) % 86400).astype(np.float64)
        first_bucket = np.floor(start_sec / bucket_sec).astype(np.int64)
        first_bucket_end = (first_bucket + 1) * bucket_sec

        integrated = np.zeros_like(tau)
        remaining = tau.copy()

        overlap = np.minimum(first_bucket_end - start_sec, remaining)
        overlap = np.maximum(overlap, 0.0)
        integrated += overlap * sigma_sq[first_bucket % n_bkt]
        remaining -= overlap

        for offset in range(1, 14):
            mask = remaining > 0
            if not mask.any():
                break
            b_idx = (first_bucket + offset) % n_bkt
            contrib = np.where(mask, np.minimum(remaining, bucket_sec), 0.0)
            integrated += contrib * sigma_sq[b_idx]
            remaining -= contrib

        avg_var = integrated / np.maximum(tau, 1e-6)
        return np.sqrt(avg_var)

    def price(self, S, K, tau, sigma_rv, t_ms):
        """Compute Gaussian and Student-t binary option prices.

        Args:
            S: Current mid price(s).
            K: Strike price(s).
            tau: Time to expiry in seconds.
            sigma_rv: EWMA realized volatility (per-sqrt-sec).
            t_ms: Current timestamp (epoch milliseconds, for sigma_tod lookup).

        Returns:
            (p_gauss, p_t): Gaussian and Student-t prices as numpy arrays.
        """
        S = np.asarray(S, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        tau = np.asarray(tau, dtype=np.float64)
        sigma_rv = np.asarray(sigma_rv, dtype=np.float64)

        sigma_tod = self._sigma_tod_integrated(t_ms, tau)
        sigma_rel = sigma_rv / np.maximum(sigma_tod, 1e-12)
        sigma_rel = np.maximum(sigma_rel, 1e-12)

        log_tau = np.log(np.maximum(tau, 1e-6))

        # Shrinkage
        z = self.k0 + self.k1 * log_tau
        m = _expit(z)
        sr_sq = sigma_rel ** 2
        f = m + (1.0 - m) * sr_sq

        base = sigma_tod ** 2 * np.power(np.maximum(tau, 1e-6), self.alpha)
        v = self.c ** 2 * base * f

        k = np.log(K / S)
        sqrt_v = np.sqrt(np.maximum(v, 1e-20))

        p_gauss = norm.cdf(-k / sqrt_v)
        p_t = student_t.cdf(-k / (self.s_nu * sqrt_v), df=self.nu)

        return (
            np.clip(p_gauss, 1e-9, 1.0 - 1e-9),
            np.clip(p_t, 1e-9, 1.0 - 1e-9),
        )
