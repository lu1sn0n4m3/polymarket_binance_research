"""Fundamental pricing model for hourly Up/Down binary options.

Based on the whitepaper: "A Fundamental Pricing Framework for Hourly Up/Down Prediction Markets"

The model estimates P(Up) = P(Close ≥ Open | F_t) using:
1. Student-t distribution for remaining returns (heavy tails)
2. Realized variance from intrahour Binance data
3. Blending of multiple volatility estimates
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy import stats

from src.pricing.base import Pricer, PricerOutput

if TYPE_CHECKING:
    from src.data.session import HourlyMarketSession


# =============================================================================
# Volatility Blending
# =============================================================================

class VolatilityBlender(ABC):
    """Abstract interface for blending volatility estimates.
    
    Combines multiple variance estimates into a single σ_t² for pricing.
    """
    
    @abstractmethod
    def blend(
        self,
        tau_sec: float,
        rv_since_open: float,
        rv_recent: float,
        rv_historical: float | None = None,
        T: float = 3600.0,
    ) -> float:
        """Blend variance estimates into remaining variance σ_t².
        
        Args:
            tau_sec: Time remaining until expiry (seconds)
            rv_since_open: Realized variance from open to now
            rv_recent: Realized variance in recent window
            rv_historical: Historical average variance (optional)
            T: Total market duration in seconds
            
        Returns:
            Estimated variance of remaining return
        """
        pass


class LinearBlender(VolatilityBlender):
    """Linear blending weights as functions of τ.
    
    - Weight on realized variance increases as hour progresses
    - Weight on historical prior decreases as hour progresses
    - Recent window provides regime adaptation
    """
    
    def __init__(
        self,
        w_recent: float = 0.3,
        use_historical: bool = True,
    ):
        """
        Args:
            w_recent: Fixed weight on recent window variance
            use_historical: Whether to use historical prior when available
        """
        self.w_recent = w_recent
        self.use_historical = use_historical
    
    def blend(
        self,
        tau_sec: float,
        rv_since_open: float,
        rv_recent: float,
        rv_historical: float | None = None,
        T: float = 3600.0,
    ) -> float:
        """Blend with linear time-varying weights."""
        # Fraction of hour elapsed
        elapsed_frac = (T - tau_sec) / T
        elapsed_frac = np.clip(elapsed_frac, 0.0, 1.0)
        
        # Scale realized variance to remaining time
        # If we've observed RV over t seconds, scale to τ remaining seconds
        elapsed_sec = T - tau_sec
        if elapsed_sec > 0:
            # Assume variance scales linearly with time
            rv_since_open_scaled = rv_since_open * (tau_sec / elapsed_sec)
        else:
            rv_since_open_scaled = 0.0
        
        # Scale recent window variance
        # This is already a short-window estimate, use as proxy for current regime
        rv_recent_scaled = rv_recent * (tau_sec / 60.0) if rv_recent > 0 else 0.0
        
        # Weights
        w_recent = self.w_recent
        
        if rv_historical is not None and self.use_historical:
            # Blend three sources
            # Historical weight decreases, realized weight increases
            w_historical = (1 - w_recent) * (tau_sec / T)
            w_realized = (1 - w_recent) * (1 - tau_sec / T)
            
            sigma_sq = (
                w_realized * rv_since_open_scaled +
                w_recent * rv_recent_scaled +
                w_historical * rv_historical
            )
        else:
            # Blend two sources
            w_realized = 1 - w_recent
            sigma_sq = (
                w_realized * rv_since_open_scaled +
                w_recent * rv_recent_scaled
            )
        
        # Floor to prevent zero variance
        return max(sigma_sq, 1e-10)


# =============================================================================
# Historical Volatility Profile (Interface + Stub)
# =============================================================================

class HistoricalVolProfile(ABC):
    """Abstract interface for historical volatility profiles.
    
    Provides σ̄²(h, τ) - the historical average remaining variance
    for a given hour-of-day and time remaining.
    """
    
    @abstractmethod
    def get(self, hour_et: int, tau_sec: float) -> float | None:
        """Get historical average remaining variance.
        
        Args:
            hour_et: Hour of day in Eastern Time (0-23)
            tau_sec: Time remaining until expiry (seconds)
            
        Returns:
            Historical average variance, or None if not available
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Whether the profile has enough data to be used."""
        pass


class EmptyHistoricalProfile(HistoricalVolProfile):
    """Stub profile that returns None (no historical data)."""
    
    def get(self, hour_et: int, tau_sec: float) -> float | None:
        return None
    
    def is_available(self) -> bool:
        return False


class SimpleHistoricalProfile(HistoricalVolProfile):
    """Simple historical profile built from session data.
    
    Computes average realized variance by hour-of-day.
    """
    
    def __init__(self, min_sessions: int = 10):
        """
        Args:
            min_sessions: Minimum sessions per hour required
        """
        self.min_sessions = min_sessions
        self._profile: dict[int, float] = {}  # hour_et -> avg hourly variance
        self._counts: dict[int, int] = {}
    
    def fit(self, sessions: list["HourlyMarketSession"]) -> "SimpleHistoricalProfile":
        """Fit the profile from session data.
        
        Args:
            sessions: List of HourlyMarketSession instances
            
        Returns:
            self for chaining
        """
        from collections import defaultdict
        
        variances_by_hour: dict[int, list[float]] = defaultdict(list)
        
        for session in sessions:
            try:
                # Compute total hourly realized variance
                end_ms = int(session.utc_end.timestamp() * 1000)
                rv = session.compute_rv_since_open(end_ms, sample_ms=1000)
                variances_by_hour[session.hour_et].append(rv)
            except Exception:
                continue
        
        for hour, variances in variances_by_hour.items():
            if len(variances) >= 1:  # Accept any data for now
                self._profile[hour] = float(np.mean(variances))
                self._counts[hour] = len(variances)
        
        return self
    
    def get(self, hour_et: int, tau_sec: float) -> float | None:
        if hour_et not in self._profile:
            return None
        
        # Scale by remaining time fraction
        hourly_var = self._profile[hour_et]
        tau_frac = tau_sec / 3600.0
        
        return hourly_var * tau_frac
    
    def is_available(self) -> bool:
        # Check if we have enough data for most hours
        return sum(self._counts.values()) >= self.min_sessions


# =============================================================================
# Degrees of Freedom Estimation
# =============================================================================

def estimate_degrees_of_freedom(
    returns: np.ndarray,
    method: Literal["mle", "kurtosis"] = "mle",
    min_nu: float = 2.5,
    max_nu: float = 30.0,
) -> float:
    """Estimate Student-t degrees of freedom from return data.
    
    Args:
        returns: Array of log returns
        method: Estimation method
            - "kurtosis": Method of moments using excess kurtosis
            - "mle": Maximum likelihood (slower, more accurate)
        min_nu: Minimum degrees of freedom
        max_nu: Maximum degrees of freedom
        
    Returns:
        Estimated degrees of freedom
    """
    returns = returns[~np.isnan(returns)]
    
    if len(returns) < 30:
        return 5.0  # Default for small samples
    
    if method == "kurtosis":
        # Method of moments: excess kurtosis = 6 / (nu - 4) for nu > 4
        # So nu = 6 / kurtosis + 4
        excess_kurt = stats.kurtosis(returns, fisher=True)
        
        if excess_kurt <= 0:
            return max_nu  # Near-Gaussian
        
        nu = 6.0 / excess_kurt + 4.0
        
    elif method == "mle":
        # Maximum likelihood estimation
        # Fit t-distribution and extract nu
        try:
            params = stats.t.fit(returns)
            nu = params[0]  # degrees of freedom
        except Exception:
            nu = 5.0
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return float(np.clip(nu, min_nu, max_nu))


# =============================================================================
# Fundamental Pricer
# =============================================================================

@dataclass
class FundamentalPricerConfig:
    """Configuration for the fundamental pricer."""
    
    nu: float = 5.0                          # Degrees of freedom (fallback if not estimating)
    sample_ms: int = 1000                    # Sample interval for RV computation
    recent_window_ms: int = 300_000          # Window for RV_recent (5 minutes - longer is more stable)
    estimate_nu_from_data: bool = False      # Whether to estimate nu from session data
    nu_estimation_method: str = "mle"        # "mle" (more accurate) or "kurtosis" (faster)
    mu: float = 0.0                          # Drift (0 for baseline model)
    
    # Lookback-only mode: use pre-market volatility, ignore in-market vol
    use_lookback_vol_only: bool = False      # If True, use ONLY lookback period vol (no in-market updates)
    lookback_sample_ms: int = 100            # Sample interval for lookback RV (100ms = microprice level)


class FundamentalPricer(Pricer):
    """Fundamental pricer based on remaining return distribution.
    
    Estimates P(Up) = P(r_{t→T} ≥ -r_{0→t}) using Student-t distribution.
    
    Attributes:
        config: Pricer configuration
        blender: Volatility blending strategy
        historical_profile: Historical variance lookup (optional)
    """
    
    def __init__(
        self,
        config: FundamentalPricerConfig | None = None,
        blender: VolatilityBlender | None = None,
        historical_profile: HistoricalVolProfile | None = None,
    ):
        """
        Args:
            config: Pricer configuration
            blender: Volatility blending strategy (default: LinearBlender)
            historical_profile: Historical variance profile (default: empty)
        """
        self.config = config or FundamentalPricerConfig()
        self.blender = blender or LinearBlender()
        self.historical_profile = historical_profile or EmptyHistoricalProfile()
        
        # Cached estimates per session
        self._session_nu: dict[str, float] = {}
        self._session_lookback_vol: dict[str, float] = {}  # Hourly vol from lookback
    
    def _get_session_key(self, session: "HourlyMarketSession") -> str:
        """Get cache key for a session."""
        return f"{session.asset}_{session.market_date}_{session.hour_et}"
    
    def _get_lookback_returns(self, session: "HourlyMarketSession") -> np.ndarray:
        """Get log returns from the lookback period (before market open).
        
        Uses high-frequency sampling (lookback_sample_ms, default 100ms) for
        better estimation of distribution parameters.
        """
        # Get lookback BBO data
        lookback_bbo = session.binance_lookback_bbo
        market_bbo = session.binance_bbo
        
        if lookback_bbo.empty:
            return np.array([])
        
        # Filter to ONLY lookback period (before market start)
        start_ms = int(session.lookback_start.timestamp() * 1000)
        end_ms = int(session.utc_start.timestamp() * 1000)  # Market start, not end
        
        mask = (lookback_bbo["ts_recv"] >= start_ms) & (lookback_bbo["ts_recv"] < end_ms)
        pre_market = lookback_bbo[mask].copy()
        
        if len(pre_market) < 100:
            return np.array([])
        
        # Compute mid prices
        pre_market["mid"] = (pre_market["bid_px"] + pre_market["ask_px"]) / 2
        
        # Resample to regular grid at lookback_sample_ms
        sample_ms = self.config.lookback_sample_ms
        grid_ts = np.arange(start_ms, end_ms, sample_ms)
        
        sorted_data = pre_market.sort_values("ts_recv")
        indices = np.searchsorted(sorted_data["ts_recv"].values, grid_ts, side="right") - 1
        indices = np.clip(indices, 0, len(sorted_data) - 1)
        
        mids = sorted_data["mid"].values[indices]
        
        # Compute log returns
        log_returns = np.diff(np.log(mids))
        
        return log_returns
    
    def _get_nu(self, session: "HourlyMarketSession") -> float:
        """Get degrees of freedom, estimating from lookback data if configured."""
        if not self.config.estimate_nu_from_data:
            return self.config.nu
        
        key = self._get_session_key(session)
        
        if key in self._session_nu:
            return self._session_nu[key]
        
        # Estimate from lookback data at high frequency
        log_returns = self._get_lookback_returns(session)
        
        if len(log_returns) < 100:
            nu = self.config.nu
        else:
            nu = estimate_degrees_of_freedom(
                log_returns, 
                method=self.config.nu_estimation_method
            )
        
        self._session_nu[key] = nu
        return nu
    
    def _get_lookback_hourly_vol(self, session: "HourlyMarketSession") -> float:
        """Get hourly volatility estimated from the lookback period.
        
        Computes realized variance from high-frequency returns in the lookback
        period, then scales to 1-hour variance.
        
        Returns:
            Hourly standard deviation (sigma for 1 hour)
        """
        key = self._get_session_key(session)
        
        if key in self._session_lookback_vol:
            return self._session_lookback_vol[key]
        
        log_returns = self._get_lookback_returns(session)
        
        if len(log_returns) < 100:
            # Fallback: typical BTC hourly vol ~0.5%
            sigma_hourly = 0.005
        else:
            # Realized variance over the lookback period
            rv_lookback = np.sum(log_returns ** 2)
            
            # Scale to 1 hour
            # We have N samples over lookback_hours, scale RV to 1 hour
            lookback_hours = session.lookback_hours
            rv_hourly = rv_lookback / lookback_hours
            
            sigma_hourly = np.sqrt(rv_hourly)
        
        self._session_lookback_vol[key] = sigma_hourly
        return sigma_hourly
    
    def price_at(
        self,
        session: "HourlyMarketSession",
        ts_ms: int,
    ) -> PricerOutput:
        """Compute fundamental price at a specific timestamp.
        
        Args:
            session: The market session
            ts_ms: Timestamp in milliseconds
            
        Returns:
            PricerOutput with probability and metadata
        """
        # Get realized return
        r_0_to_t = session.get_realized_return(ts_ms)
        
        if r_0_to_t is None:
            return PricerOutput.from_up_prob(0.5, metadata={"error": "no_price_data"})
        
        # Get time remaining
        tau_sec = session.time_to_expiry(ts_ms)
        
        if tau_sec <= 0:
            # At or past expiry - use deterministic
            return PricerOutput.from_up_prob(
                1.0 if r_0_to_t >= 0 else 0.0,
                metadata={"expired": True},
            )
        
        # Compute realized variances
        rv_since_open = session.compute_rv_since_open(ts_ms, self.config.sample_ms)
        rv_recent = session.compute_rv_recent(ts_ms, self.config.recent_window_ms, self.config.sample_ms)
        
        # Get historical variance if available
        rv_historical = None
        if self.historical_profile.is_available():
            rv_historical = self.historical_profile.get(session.hour_et, tau_sec)
        
        # Blend to get σ_t²
        sigma_sq = self.blender.blend(
            tau_sec=tau_sec,
            rv_since_open=rv_since_open,
            rv_recent=rv_recent,
            rv_historical=rv_historical,
        )
        sigma_t = np.sqrt(sigma_sq)
        
        # Get degrees of freedom
        nu = self._get_nu(session)
        
        # Compute probability: p_t = 1 - F_{t_ν}((-r_{0→t} - μ) / σ_t)
        mu = self.config.mu
        z = (-r_0_to_t - mu) / sigma_t
        p_up = 1.0 - stats.t.cdf(z, df=nu)
        
        # Clamp to reasonable range
        p_up = float(np.clip(p_up, 0.001, 0.999))
        
        return PricerOutput.from_up_prob(
            p_up,
            metadata={
                "r_0_to_t": r_0_to_t,
                "tau_sec": tau_sec,
                "sigma_t": sigma_t,
                "sigma_sq": sigma_sq,
                "nu": nu,
                "z": z,
                "rv_since_open": rv_since_open,
                "rv_recent": rv_recent,
                "rv_historical": rv_historical,
            },
        )
    
    def price_session(
        self,
        session: "HourlyMarketSession",
        sample_ms: int = 1000,
    ) -> pd.DataFrame:
        """Price an entire session on a regular grid (vectorized, fast).
        
        Args:
            session: The market session
            sample_ms: Sample interval in milliseconds
            
        Returns:
            DataFrame with columns:
            - ts_ms, tau_sec, elapsed_sec
            - bnc_mid, r_0_to_t, pm_mid
            - fundamental_prob: Our estimated P(Up)
            - edge: pm_mid - fundamental_prob
        """
        # Get pricing grid from session
        df = session.get_pricing_grid(sample_ms=sample_ms)
        
        if df.empty:
            return df
        
        # Vectorized pricing - compute everything at once
        df = self._price_vectorized(df, session, sample_ms)
        
        return df
    
    def _price_vectorized(
        self,
        df: pd.DataFrame,
        session: "HourlyMarketSession",
        sample_ms: int,
    ) -> pd.DataFrame:
        """Vectorized pricing - O(n) instead of O(n²)."""
        
        T = 3600.0  # 1 hour in seconds
        
        # Get degrees of freedom (computed once from lookback)
        nu = self._get_nu(session)
        
        # Already have: ts_ms, tau_sec, elapsed_sec, bnc_mid, r_0_to_t, pm_mid
        tau_sec = df["tau_sec"].values
        elapsed_sec = df["elapsed_sec"].values
        r_0_to_t = df["r_0_to_t"].values
        
        if self.config.use_lookback_vol_only:
            # =========================================================
            # LOOKBACK-ONLY MODE: Use pre-market vol, no in-market updates
            # =========================================================
            # Get hourly vol from the lookback period
            sigma_hourly = self._get_lookback_hourly_vol(session)
            
            # Scale to remaining time: sigma_t = sigma_hourly * sqrt(tau / T)
            # This assumes vol scales with sqrt(time)
            sigma_t = sigma_hourly * np.sqrt(np.maximum(tau_sec, 0) / T)
            
            # Store placeholders for compatibility
            rv_cumsum = np.zeros(len(df))
            rv_recent = np.zeros(len(df))
            
        else:
            # =========================================================
            # STANDARD MODE: Use in-market vol estimation
            # =========================================================
            # Compute log returns between consecutive samples
            log_prices = np.log(df["bnc_mid"].values)
            log_returns = np.diff(log_prices, prepend=log_prices[0])
            log_returns[0] = 0  # First return is 0
            
            squared_returns = log_returns ** 2
            
            # Cumulative RV since open
            rv_cumsum = np.cumsum(squared_returns)
            
            # Rolling RV for recent window
            recent_window_samples = self.config.recent_window_ms // sample_ms
            rv_series = pd.Series(squared_returns)
            rv_recent = rv_series.rolling(window=recent_window_samples, min_periods=1).sum().values
            
            # Scale variances to remaining time
            elapsed_sec_safe = np.maximum(elapsed_sec, sample_ms / 1000)
            
            # Scale RV since open to remaining time
            rv_since_open_scaled = rv_cumsum * (tau_sec / elapsed_sec_safe)
            
            # Scale recent RV to remaining time
            recent_window_sec = self.config.recent_window_ms / 1000
            rv_recent_scaled = rv_recent * (tau_sec / recent_window_sec)
            
            # Blend variances
            w_recent = self.blender.w_recent if hasattr(self.blender, 'w_recent') else 0.3
            w_realized = 1 - w_recent
            
            sigma_sq = w_realized * rv_since_open_scaled + w_recent * rv_recent_scaled
            sigma_sq = np.maximum(sigma_sq, 1e-10)
            sigma_t = np.sqrt(sigma_sq)
        
        # Compute z-scores
        mu = self.config.mu
        sigma_t_safe = np.maximum(sigma_t, 1e-10)
        z = (-r_0_to_t - mu) / sigma_t_safe
        
        # Compute probabilities using Student-t CDF
        p_up = 1.0 - stats.t.cdf(z, df=nu)
        
        # Clamp
        p_up = np.clip(p_up, 0.001, 0.999)
        
        # Handle edge cases: at expiry, use deterministic
        at_expiry = tau_sec <= 0
        p_up = np.where(at_expiry, np.where(r_0_to_t >= 0, 1.0, 0.0), p_up)
        
        # Add to DataFrame
        df["fundamental_prob"] = p_up
        df["sigma_t"] = sigma_t
        df["rv_cumsum"] = rv_cumsum
        df["rv_recent"] = rv_recent
        
        # Compute edge
        df["edge"] = df["pm_mid"] - df["fundamental_prob"]
        df["abs_edge"] = df["edge"].abs()
        
        return df
    
    # Implement abstract method from base Pricer
    def price(
        self,
        time_to_expiry_sec: float,
        realized_vol: float,
        current_price: float | None = None,
        strike_price: float | None = None,
        **features,
    ) -> PricerOutput:
        """Generic pricing interface (less precise than price_at).
        
        For full functionality, use price_at() with a session.
        """
        if current_price is None or strike_price is None:
            return PricerOutput.from_up_prob(0.5)
        
        r_0_to_t = np.log(current_price / strike_price)
        
        # Use realized_vol as sigma_t directly
        sigma_t = realized_vol * np.sqrt(time_to_expiry_sec / 3600)
        
        if sigma_t <= 0:
            return PricerOutput.from_up_prob(1.0 if r_0_to_t >= 0 else 0.0)
        
        nu = self.config.nu
        z = -r_0_to_t / sigma_t
        p_up = 1.0 - stats.t.cdf(z, df=nu)
        
        return PricerOutput.from_up_prob(float(np.clip(p_up, 0.001, 0.999)))


# =============================================================================
# Incremental Pricer for Live Trading
# =============================================================================

class IncrementalPricer:
    """Fast incremental pricer for live trading.
    
    Maintains rolling state and can price in microseconds.
    
    Usage:
        pricer = IncrementalPricer(open_price=100000.0, nu=5.0)
        
        # Feed prices as they come in
        pricer.update(price=100050.0, ts_ms=1234567890)
        
        # Get current probability (fast!)
        prob = pricer.get_probability(tau_sec=1800)
    """
    
    def __init__(
        self,
        open_price: float,
        open_ts_ms: int,
        expiry_ts_ms: int,
        nu: float = 5.0,
        sample_ms: int = 1000,
        recent_window_ms: int = 60_000,
        w_recent: float = 0.3,
    ):
        """
        Args:
            open_price: Opening price (strike)
            open_ts_ms: Market open timestamp (ms)
            expiry_ts_ms: Market expiry timestamp (ms)
            nu: Degrees of freedom
            sample_ms: Expected sample interval
            recent_window_ms: Window for recent RV
            w_recent: Weight on recent RV
        """
        self.open_price = open_price
        self.log_open = np.log(open_price)
        self.open_ts_ms = open_ts_ms
        self.expiry_ts_ms = expiry_ts_ms
        self.nu = nu
        self.sample_ms = sample_ms
        self.recent_window_ms = recent_window_ms
        self.w_recent = w_recent
        
        # State
        self.last_log_price: float | None = None
        self.last_ts_ms: int | None = None
        self.current_price: float | None = None
        self.current_ts_ms: int | None = None
        
        # Rolling RV (LAGGED - excludes the most recent return)
        # This prevents the jump that moved the price from also spiking volatility
        self.rv_cumsum: float = 0.0
        self.rv_cumsum_lagged: float = 0.0  # Excludes last return
        self.n_samples: int = 0
        self.last_sq_return: float = 0.0  # Track last return for lagging
        
        # Recent window (circular buffer) - also lagged
        self.recent_window_size = max(1, recent_window_ms // sample_ms)
        self.recent_buffer: list[float] = []
        self.rv_recent: float = 0.0
        self.rv_recent_lagged: float = 0.0  # Excludes last return
    
    def update(self, price: float, ts_ms: int) -> None:
        """Update with a new price observation.
        
        Args:
            price: Current price
            ts_ms: Current timestamp (ms)
        """
        log_price = np.log(price)
        
        if self.last_log_price is not None:
            # Save previous "lagged" state (what was current becomes lagged)
            self.rv_cumsum_lagged = self.rv_cumsum
            self.rv_recent_lagged = self.rv_recent
            
            # Compute squared return
            log_return = log_price - self.last_log_price
            sq_return = log_return ** 2
            
            # Update cumulative RV (current includes this return)
            self.rv_cumsum += sq_return
            self.n_samples += 1
            self.last_sq_return = sq_return
            
            # Update recent window
            self.recent_buffer.append(sq_return)
            if len(self.recent_buffer) > self.recent_window_size:
                removed = self.recent_buffer.pop(0)
                self.rv_recent = self.rv_recent + sq_return - removed
            else:
                self.rv_recent += sq_return
        
        self.last_log_price = log_price
        self.last_ts_ms = ts_ms
        self.current_price = price
        self.current_ts_ms = ts_ms
    
    def get_probability(self, ts_ms: int | None = None) -> float:
        """Get current P(Up) estimate.
        
        Args:
            ts_ms: Timestamp (default: last update time)
            
        Returns:
            Probability of Up outcome
        """
        if self.current_price is None:
            return 0.5
        
        if ts_ms is None:
            ts_ms = self.current_ts_ms
        
        # Time remaining
        tau_ms = self.expiry_ts_ms - ts_ms
        tau_sec = max(0, tau_ms / 1000)
        
        if tau_sec <= 0:
            # At expiry - deterministic
            r = np.log(self.current_price / self.open_price)
            return 1.0 if r >= 0 else 0.0
        
        # Current log return
        r_0_to_t = np.log(self.current_price / self.open_price)
        
        # Elapsed time
        elapsed_ms = ts_ms - self.open_ts_ms
        elapsed_sec = max(self.sample_ms / 1000, elapsed_ms / 1000)
        
        # Scale RVs to remaining time
        # Use LAGGED volatility (excludes the most recent return) to prevent
        # the price jump from also spiking vol and dampening the probability move
        rv_since_open_scaled = self.rv_cumsum_lagged * (tau_sec / elapsed_sec)
        
        recent_window_sec = max(1, len(self.recent_buffer) - 1) * self.sample_ms / 1000
        if recent_window_sec > 0 and self.rv_recent_lagged > 0:
            rv_recent_scaled = self.rv_recent_lagged * (tau_sec / recent_window_sec)
        else:
            rv_recent_scaled = rv_since_open_scaled  # Fall back to cumulative
        
        # Blend
        sigma_sq = (1 - self.w_recent) * rv_since_open_scaled + self.w_recent * rv_recent_scaled
        sigma_sq = max(sigma_sq, 1e-10)
        sigma_t = np.sqrt(sigma_sq)
        
        # Compute probability
        z = -r_0_to_t / sigma_t
        p_up = 1.0 - stats.t.cdf(z, df=self.nu)
        
        return float(np.clip(p_up, 0.001, 0.999))
    
    def get_state(self) -> dict:
        """Get current state for debugging/logging."""
        tau_sec = 0.0
        if self.current_ts_ms:
            tau_sec = max(0, (self.expiry_ts_ms - self.current_ts_ms) / 1000)
        
        return {
            "current_price": self.current_price,
            "open_price": self.open_price,
            "r_0_to_t": np.log(self.current_price / self.open_price) if self.current_price else None,
            "tau_sec": tau_sec,
            "rv_cumsum": self.rv_cumsum,
            "rv_cumsum_lagged": self.rv_cumsum_lagged,
            "rv_recent": self.rv_recent,
            "rv_recent_lagged": self.rv_recent_lagged,
            "n_samples": self.n_samples,
            "p_up": self.get_probability(),
        }
    
    @classmethod
    def from_session(
        cls,
        session: "HourlyMarketSession",
        nu: float = 5.0,
        **kwargs,
    ) -> "IncrementalPricer":
        """Create an IncrementalPricer from a session.
        
        Args:
            session: The market session
            nu: Degrees of freedom
            **kwargs: Additional arguments to IncrementalPricer
            
        Returns:
            IncrementalPricer initialized with session parameters
        """
        open_price = session.get_open_price()
        if open_price is None:
            raise ValueError("Session has no open price")
        
        return cls(
            open_price=open_price,
            open_ts_ms=int(session.utc_start.timestamp() * 1000),
            expiry_ts_ms=int(session.utc_end.timestamp() * 1000),
            nu=nu,
            **kwargs,
        )


# =============================================================================
# Comparison Utilities
# =============================================================================

def compare_prices(
    session: "HourlyMarketSession",
    pricer: FundamentalPricer,
    sample_ms: int = 1000,
) -> pd.DataFrame:
    """Compare Polymarket prices vs fundamental prices.
    
    Args:
        session: The market session
        pricer: The fundamental pricer
        sample_ms: Sample interval
        
    Returns:
        DataFrame with both prices and edge analysis
    """
    return pricer.price_session(session, sample_ms)


def analyze_edge(df: pd.DataFrame) -> dict:
    """Analyze edge statistics from comparison DataFrame.
    
    Args:
        df: Output from compare_prices()
        
    Returns:
        Dictionary with edge statistics
    """
    edge = df["edge"].dropna()
    
    if len(edge) == 0:
        return {"error": "no_data"}
    
    return {
        "mean_edge": float(edge.mean()),
        "std_edge": float(edge.std()),
        "mean_abs_edge": float(edge.abs().mean()),
        "max_edge": float(edge.max()),
        "min_edge": float(edge.min()),
        "median_edge": float(edge.median()),
        "n_points": len(edge),
        "pct_pm_overpriced": float((edge < 0).mean()),  # PM price > fundamental
        "pct_pm_underpriced": float((edge > 0).mean()),  # PM price < fundamental
    }
