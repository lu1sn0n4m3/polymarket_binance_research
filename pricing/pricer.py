"""Online pricer.

Loads frozen calibration parameters and computes binary option prices
from live market features. No dataset or pandas dependency at runtime.

Usage:
    pricer = Pricer.from_calibration("pricing/output")
    p = pricer.price(
        S=104000, K=103500, tau=1800,
        sigma_rv=3.5e-5, t_ms=1706745600000,
    )
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import norm


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
    p = Phi(-k / sqrt(v))
    """

    def __init__(
        self,
        vol_params: dict,
        sigma_tod_weekday: np.ndarray,
        sigma_tod_weekend: np.ndarray,
        bucket_minutes: int = 5,
    ):
        self.c = vol_params["c"]
        self.alpha = vol_params["alpha"]
        self.k0 = vol_params.get("k0", -100.0)
        self.k1 = vol_params.get("k1", 0.0)
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

        import pandas as pd
        sv_wd = pd.read_parquet(output_dir / "seasonal_vol_weekday.parquet")
        sv_we = pd.read_parquet(output_dir / "seasonal_vol_weekend.parquet")
        bucket_minutes = 5

        return cls(vol_params, sv_wd["sigma_tod"].values, sv_we["sigma_tod"].values, bucket_minutes)

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

    def sigma_tod(self, t_ms, tau):
        """Expose integrated sigma_tod for external use."""
        return self._sigma_tod_integrated(t_ms, tau)

    def price(self, S, K, tau, sigma_rv, t_ms):
        """Compute Gaussian binary option price P(S_T > K).

        Args:
            S: Current mid price(s).
            K: Strike price(s).
            tau: Time to expiry in seconds.
            sigma_rv: EWMA realized volatility (per-sqrt-sec).
            t_ms: Current timestamp (epoch milliseconds, for sigma_tod lookup).

        Returns:
            Gaussian price as numpy array.
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

        p = norm.cdf(-k / sqrt_v)

        return np.clip(p, 1e-9, 1.0 - 1e-9)

    def price_with_ci(self, S, K, tau, sigma_rv, t_ms, ci=0.95,
                      ewma_half_life=600.0, tick_dt=1.0, n_draws=200):
        """Compute price with EWMA variance-uncertainty confidence interval.

        Samples sigma_rv from its chi-squared sampling distribution
        (n_eff = ewma_half_life / (2 * tick_dt) effective observations)
        and propagates through the pricing formula.

        Returns (p_mid, p_lo, p_hi).
        """
        S = np.atleast_1d(np.asarray(S, dtype=np.float64))
        K = np.atleast_1d(np.asarray(K, dtype=np.float64))
        tau = np.atleast_1d(np.asarray(tau, dtype=np.float64))
        sigma_rv = np.atleast_1d(np.asarray(sigma_rv, dtype=np.float64))

        sigma_tod = self._sigma_tod_integrated(t_ms, tau)
        log_tau = np.log(np.maximum(tau, 1e-6))
        k = np.log(K / S)

        # Shrinkage (fixed params, shared across draws)
        z = self.k0 + self.k1 * log_tau
        m = _expit(z)  # (N,)

        # Chi-squared draws for EWMA variance uncertainty
        n_eff = ewma_half_life / (2.0 * tick_dt)
        rng = np.random.default_rng(42)
        chi2_scale = rng.chisquare(n_eff, size=n_draws) / n_eff  # (D,)

        # sigma_rv_d: (D, N) — perturbed realized vol for each draw
        sigma_rv_d = sigma_rv[None, :] * np.sqrt(chi2_scale[:, None])
        sigma_rel_d = sigma_rv_d / np.maximum(sigma_tod[None, :], 1e-12)
        sigma_rel_d = np.maximum(sigma_rel_d, 1e-12)

        sr_sq = sigma_rel_d ** 2  # (D, N)
        f = m[None, :] + (1.0 - m[None, :]) * sr_sq

        base = sigma_tod ** 2 * np.power(np.maximum(tau, 1e-6), self.alpha)  # (N,)
        v = self.c ** 2 * base[None, :] * f  # (D, N)

        sqrt_v = np.sqrt(np.maximum(v, 1e-20))
        p = norm.cdf(-k[None, :] / sqrt_v)
        p = np.clip(p, 1e-9, 1.0 - 1e-9)

        alpha_tail = (1.0 - ci) / 2.0
        p_mid = self.price(S.ravel(), K.ravel(), tau.ravel(), sigma_rv.ravel(), t_ms)

        # Recenter MC quantiles on point estimate (Jensen's inequality correction)
        mc_mean = np.mean(p, axis=0)
        shift = p_mid - mc_mean
        p_lo = np.quantile(p, alpha_tail, axis=0) + shift
        p_hi = np.quantile(p, 1.0 - alpha_tail, axis=0) + shift

        return p_mid, p_lo, p_hi
