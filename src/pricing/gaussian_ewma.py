"""Gaussian EWMA Pricer with capped returns and TOD prior.

Based on the specification in docs/pricer_guide.md.
Simple, robust, designed for online pricing of hourly binary markets.

Key Features:
- Dual EWMA (fast/slow) for robust volatility estimation
- Asymmetric EWMA: reacts fast to vol increases, slow to decreases
- Capped squared returns to prevent jump-induced vol explosion
- Time-of-day prior for early-hour stability
- Optional drift adjustment to anticipate PM repricing
- Gaussian and Student-t distribution support
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from src.data.session import HourlyMarketSession


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GaussianEWMAConfig:
    """Configuration for the Gaussian EWMA Pricer."""
    
    # Sampling
    update_interval_sec: float = 1.0          # Volatility update grid (1 second)
    
    # EWMA half-lives (in seconds)
    fast_halflife_sec: float = 300.0          # Fast EWMA (~5 min)
    slow_halflife_sec: float = 1800.0         # Slow EWMA (~30 min)
    
    # Fast/slow blending
    alpha: float = 0.3                        # Weight on fast EWMA (vs slow)
    
    # Asymmetric EWMA (react fast to high vol, slow to low vol)
    asymmetric_ewma: bool = True              # Enable asymmetric updates
    downside_lambda_mult: float = 0.3         # Multiply lambda by this when vol decreasing
    
    # Jump capping
    cap_multiplier: float = 8.0               # Cap = c² * v_slow * dt
    enable_capping: bool = True               # Can disable for comparison
    
    # TOD prior blending
    tod_ramp_sec: float = 60.0                # Ramp to full trust in live estimate
    use_tod_prior: bool = True                # Can disable TOD prior
    
    # Variance floor (near expiry)
    floor_fraction: float = 0.1               # Floor at this fraction of blended variance
    min_variance_floor: float = 1e-12         # Absolute minimum variance
    
    # Student-t degrees of freedom (for comparison)
    student_t_nu: float = 5.0                 # Degrees of freedom (must be > 2)
    
    # Drift adjustment (anticipate PM repricing when Binance is trending)
    use_drift: bool = False                   # Enable drift-adjusted pricing
    drift_halflife_sec: float = 30.0          # EWMA halflife for drift estimate
    drift_weight: float = 0.5                 # How much drift shifts z-score
    drift_horizon_sec: float = 60.0           # Max time horizon for drift extrapolation
    
    def lambda_from_halflife(self, halflife_sec: float) -> float:
        """Convert half-life to EWMA lambda parameter."""
        return 1.0 - 2.0 ** (-self.update_interval_sec / halflife_sec)
    
    @property
    def lambda_fast(self) -> float:
        return self.lambda_from_halflife(self.fast_halflife_sec)
    
    @property
    def lambda_slow(self) -> float:
        return self.lambda_from_halflife(self.slow_halflife_sec)
    
    @property
    def lambda_drift(self) -> float:
        return self.lambda_from_halflife(self.drift_halflife_sec)


# =============================================================================
# Time-of-Day Profile
# =============================================================================

@dataclass
class TODProfile:
    """Time-of-day volatility baseline.
    
    Stores variance-per-second for each hour of day (0-23 in ET).
    """
    
    hourly_var_rate: dict[int, float] = field(default_factory=dict)
    default_var_rate: float = 1e-10  # ~0.3% hourly vol
    
    def get(self, hour: int) -> float:
        """Get variance rate for a given hour of day."""
        return self.hourly_var_rate.get(hour, self.default_var_rate)
    
    @classmethod
    def from_sessions(
        cls,
        sessions: list["HourlyMarketSession"],
        sample_ms: int = 1000,
        use_median: bool = True,
    ) -> "TODProfile":
        """Build TOD profile from historical sessions."""
        from collections import defaultdict
        
        hourly_vars: dict[int, list[float]] = defaultdict(list)
        
        for session in sessions:
            try:
                mid_series = session.get_binance_mid_series(
                    sample_ms=sample_ms,
                    include_lookback=False,
                )
                
                if len(mid_series) < 100:
                    continue
                
                log_prices = np.log(mid_series["mid"].values)
                log_returns = np.diff(log_prices)
                
                rv_hour = np.sum(log_returns ** 2)
                n_seconds = len(log_returns) * sample_ms / 1000
                var_rate = rv_hour / n_seconds
                
                hourly_vars[session.hour_et].append(var_rate)
                
            except Exception:
                continue
        
        hourly_var_rate = {}
        all_vars = []
        
        for hour, vars_list in hourly_vars.items():
            if vars_list:
                hourly_var_rate[hour] = float(np.median(vars_list) if use_median else np.mean(vars_list))
                all_vars.extend(vars_list)
        
        default_var_rate = float(np.median(all_vars)) if all_vars else 1e-10
        
        return cls(hourly_var_rate=hourly_var_rate, default_var_rate=default_var_rate)
    
    @classmethod
    def constant(cls, hourly_vol_pct: float = 0.3) -> "TODProfile":
        """Create a constant TOD profile (all hours same vol)."""
        sigma_hourly = hourly_vol_pct / 100
        var_per_sec = (sigma_hourly ** 2) / 3600
        
        return cls(
            hourly_var_rate={h: var_per_sec for h in range(24)},
            default_var_rate=var_per_sec,
        )


# =============================================================================
# Volatility State (for incremental updates)
# =============================================================================

@dataclass
class VolatilityState:
    """State for incremental volatility tracking."""
    
    v_fast: float = 0.0
    v_slow: float = 0.0
    drift: float = 0.0                # EWMA of signed returns (momentum)
    last_log_price: float | None = None
    last_update_sec: float = 0.0
    n_updates: int = 0
    total_capped: int = 0
    
    def reset(self, initial_var_rate: float):
        """Reset state with initial variance rate."""
        self.v_fast = initial_var_rate
        self.v_slow = initial_var_rate
        self.drift = 0.0
        self.last_log_price = None
        self.last_update_sec = 0.0
        self.n_updates = 0
        self.total_capped = 0


# =============================================================================
# Gaussian EWMA Pricer
# =============================================================================

class GaussianEWMAPricer:
    """Gaussian EWMA Pricer with capped returns and TOD prior.
    
    Usage:
        # Batch mode (research/backtesting)
        pricer = GaussianEWMAPricer(config, tod_profile)
        df, diagnostics = pricer.price_session(session)
        
        # Incremental mode (live trading)
        pricer.initialize(open_price, hour_et)
        pricer.update(price, elapsed_sec)
        prob = pricer.get_probability(price, elapsed_sec, tau_sec)
    """
    
    def __init__(
        self,
        config: GaussianEWMAConfig | None = None,
        tod_profile: TODProfile | None = None,
    ):
        self.config = config or GaussianEWMAConfig()
        self.tod_profile = tod_profile or TODProfile.constant(0.3)
        self.state = VolatilityState()
        self._open_price: float | None = None
        self._log_open: float | None = None
        self._hour_et: int | None = None
        self._T: float = 3600.0
    
    # -------------------------------------------------------------------------
    # Incremental Interface
    # -------------------------------------------------------------------------
    
    def initialize(self, open_price: float, hour_et: int, initial_var_rate: float | None = None):
        """Initialize for a new market hour."""
        self._open_price = open_price
        self._log_open = np.log(open_price)
        self._hour_et = hour_et
        
        v_init = initial_var_rate if initial_var_rate is not None else self.tod_profile.get(hour_et)
        self.state.reset(v_init)
    
    def estimate_lookback_variance(
        self, 
        session: "HourlyMarketSession",
        sample_ms: int = 1000,
    ) -> float:
        """Estimate variance rate from the pre-market lookback period."""
        bnc_trades = session.binance_lookback_trades
        if bnc_trades is None or bnc_trades.empty:
            return self.tod_profile.get(session.hour_et)
        
        market_start_ms = int(session.utc_start.timestamp() * 1000)
        lookback_trades = bnc_trades[bnc_trades["ts_recv"] < market_start_ms]
        
        if len(lookback_trades) < 10:
            return self.tod_profile.get(session.hour_et)
        
        lookback_trades = lookback_trades.copy()
        lookback_trades["bucket"] = (lookback_trades["ts_recv"] // sample_ms) * sample_ms
        
        grid = lookback_trades.groupby("bucket").agg({"price": "last"}).reset_index()
        grid = grid.sort_values("bucket")
        
        if len(grid) < 2:
            return self.tod_profile.get(session.hour_et)
        
        log_prices = np.log(grid["price"].values)
        log_returns = np.diff(log_prices)
        
        sq_returns = log_returns ** 2
        dt = sample_ms / 1000.0
        var_rate = np.mean(sq_returns) / dt
        
        return var_rate
    
    def update(self, price: float, elapsed_sec: float):
        """Update volatility state with new price observation."""
        log_price = np.log(price)
        dt = self.config.update_interval_sec
        cfg = self.config
        
        if self.state.last_log_price is not None:
            log_return = log_price - self.state.last_log_price
            sq_return = log_return ** 2
            
            # Capping
            if cfg.enable_capping:
                v_slow_safe = max(self.state.v_slow, cfg.min_variance_floor)
                cap_var = (cfg.cap_multiplier ** 2) * v_slow_safe * dt
                if sq_return > cap_var:
                    sq_return = cap_var
                    self.state.total_capped += 1
            
            var_rate = sq_return / dt
            
            # Asymmetric EWMA update
            lf = cfg.lambda_fast
            ls = cfg.lambda_slow
            
            if cfg.asymmetric_ewma:
                lf_use = lf if var_rate >= self.state.v_fast else lf * cfg.downside_lambda_mult
                ls_use = ls if var_rate >= self.state.v_slow else ls * cfg.downside_lambda_mult
                self.state.v_fast = (1 - lf_use) * self.state.v_fast + lf_use * var_rate
                self.state.v_slow = (1 - ls_use) * self.state.v_slow + ls_use * var_rate
            else:
                self.state.v_fast = (1 - lf) * self.state.v_fast + lf * var_rate
                self.state.v_slow = (1 - ls) * self.state.v_slow + ls * var_rate
            
            # Drift update (signed returns for momentum)
            if cfg.use_drift:
                ld = cfg.lambda_drift
                self.state.drift = (1 - ld) * self.state.drift + ld * log_return
            
            self.state.n_updates += 1
        
        self.state.last_log_price = log_price
        self.state.last_update_sec = elapsed_sec
    
    def get_variance_rate(self, elapsed_sec: float) -> float:
        """Get blended variance rate at current time."""
        v_ewma = self.config.alpha * self.state.v_fast + (1 - self.config.alpha) * self.state.v_slow
        
        if self.config.use_tod_prior and self._hour_et is not None:
            v_tod = self.tod_profile.get(self._hour_et)
            w = min(1.0, elapsed_sec / self.config.tod_ramp_sec)
            return w * v_ewma + (1 - w) * v_tod
        
        return v_ewma
    
    def get_probability(self, price: float, elapsed_sec: float, tau_sec: float) -> float:
        """Get probability of Up outcome."""
        if self._log_open is None:
            return 0.5
        
        if tau_sec <= 0:
            return 1.0 if np.log(price / self._open_price) >= 0 else 0.0
        
        r = np.log(price) - self._log_open
        v_blend = self.get_variance_rate(elapsed_sec)
        V_rem = max(v_blend * tau_sec, self.config.min_variance_floor)
        sigma_rem = np.sqrt(V_rem)
        
        z = r / sigma_rem
        
        # Drift adjustment: if price is trending, shift z
        if self.config.use_drift:
            # drift is in log-return units per second
            # Only extrapolate for a short horizon (PM will reprice quickly)
            drift_horizon = min(tau_sec, self.config.drift_horizon_sec)
            drift_contribution = self.state.drift * drift_horizon / sigma_rem
            z += self.config.drift_weight * drift_contribution
        
        return float(np.clip(stats.norm.cdf(z), 0.001, 0.999))
    
    # -------------------------------------------------------------------------
    # Batch Interface
    # -------------------------------------------------------------------------
    
    def price_session(
        self,
        session: "HourlyMarketSession",
        output_sample_ms: int = 1000,
    ) -> tuple[pd.DataFrame, dict]:
        """Price an entire session. Returns (DataFrame, diagnostics)."""
        diagnostics = {}
        
        grid = session.get_pricing_grid(sample_ms=output_sample_ms)
        if grid.empty:
            return grid, diagnostics
        
        open_price = session.get_open_price()
        if open_price is None:
            return grid, diagnostics
        
        cfg = self.config
        
        # Initialize variance estimates
        v_tod = self.tod_profile.get(session.hour_et)
        v_lookback = self.estimate_lookback_variance(session, sample_ms=output_sample_ms)
        v_init = v_lookback
        
        diagnostics["v_tod"] = v_tod
        diagnostics["v_lookback"] = v_lookback
        diagnostics["sigma_hourly_tod"] = np.sqrt(v_tod * 3600) * 100
        diagnostics["sigma_hourly_lookback"] = np.sqrt(v_lookback * 3600) * 100
        
        # Extract arrays
        prices = grid["bnc_mid"].values
        elapsed = grid["elapsed_sec"].values
        tau = grid["tau_sec"].values
        n = len(grid)
        dt = cfg.update_interval_sec
        
        # Compute log returns
        log_prices = np.log(prices)
        log_returns = np.diff(log_prices, prepend=log_prices[0])
        log_returns[0] = 0
        sq_returns = log_returns ** 2
        
        # EWMA parameters
        lf = cfg.lambda_fast
        ls = cfg.lambda_slow
        cap_mult_sq = cfg.cap_multiplier ** 2
        min_var = cfg.min_variance_floor
        
        # Arrays for output
        v_fast_arr = np.zeros(n)
        v_slow_arr = np.zeros(n)
        drift_arr = np.zeros(n)
        cap_applied_arr = np.zeros(n, dtype=bool)
        
        v_fast = v_init
        v_slow = v_init
        drift = 0.0
        total_capped = 0
        
        # Main EWMA loop
        for i in range(n):
            if i > 0:
                sq_ret = sq_returns[i]
                
                # Capping
                if cfg.enable_capping:
                    cap_var = cap_mult_sq * max(v_slow, min_var) * dt
                    if sq_ret > cap_var:
                        sq_ret = cap_var
                        cap_applied_arr[i] = True
                        total_capped += 1
                
                var_rate = sq_ret / dt
                
                # Asymmetric EWMA
                if cfg.asymmetric_ewma:
                    lf_use = lf if var_rate >= v_fast else lf * cfg.downside_lambda_mult
                    ls_use = ls if var_rate >= v_slow else ls * cfg.downside_lambda_mult
                    v_fast = (1 - lf_use) * v_fast + lf_use * var_rate
                    v_slow = (1 - ls_use) * v_slow + ls_use * var_rate
                else:
                    v_fast = (1 - lf) * v_fast + lf * var_rate
                    v_slow = (1 - ls) * v_slow + ls * var_rate
                
                # Drift (momentum)
                if cfg.use_drift:
                    ld = cfg.lambda_drift
                    drift = (1 - ld) * drift + ld * log_returns[i]
            
            v_fast_arr[i] = v_fast
            v_slow_arr[i] = v_slow
            drift_arr[i] = drift
        
        diagnostics["total_capped"] = total_capped
        diagnostics["capping_pct"] = 100.0 * total_capped / n if n > 0 else 0
        
        # Blend fast and slow
        v_ewma_arr = cfg.alpha * v_fast_arr + (1 - cfg.alpha) * v_slow_arr
        
        # TOD blending with time ramp
        if cfg.use_tod_prior:
            w = np.minimum(1.0, elapsed / cfg.tod_ramp_sec)
            v_blend_arr = w * v_ewma_arr + (1 - w) * v_tod
        else:
            v_blend_arr = v_ewma_arr
        
        # Remaining variance
        V_rem_arr = v_blend_arr * tau
        V_floor = np.maximum(cfg.floor_fraction * v_blend_arr * tau, cfg.min_variance_floor)
        V_rem_arr = np.maximum(V_rem_arr, V_floor)
        sigma_rem_arr = np.sqrt(V_rem_arr)
        
        # Log return from open
        r = np.log(prices / open_price)
        
        # Gaussian z-score
        z_gaussian = r / sigma_rem_arr
        
        # Drift adjustment (only extrapolate for short horizon)
        if cfg.use_drift:
            drift_horizon = np.minimum(tau, cfg.drift_horizon_sec)
            drift_contribution = drift_arr * drift_horizon / sigma_rem_arr
            z_gaussian = z_gaussian + cfg.drift_weight * drift_contribution
            diagnostics["drift_enabled"] = True
            diagnostics["drift_horizon_sec"] = cfg.drift_horizon_sec
        else:
            drift_contribution = np.zeros(n)
            diagnostics["drift_enabled"] = False
        
        prob_gaussian = np.clip(stats.norm.cdf(z_gaussian), 0.001, 0.999)
        
        # Student-t (variance-matched)
        nu = cfg.student_t_nu
        if nu > 2:
            scale_factor = np.sqrt((nu - 2) / nu)
            z_student = r / (sigma_rem_arr * scale_factor)
            if cfg.use_drift:
                # Same drift adjustment, scaled for Student-t
                z_student = z_student + cfg.drift_weight * drift_contribution / scale_factor
            prob_student_t = np.clip(stats.t.cdf(z_student, df=nu), 0.001, 0.999)
            diagnostics["student_t_nu"] = nu
        else:
            prob_student_t = prob_gaussian.copy()
            diagnostics["student_t_nu"] = None
        
        # Handle expiry
        at_expiry = tau <= 0
        prob_gaussian[at_expiry] = np.where(r[at_expiry] >= 0, 1.0, 0.0)
        prob_student_t[at_expiry] = np.where(r[at_expiry] >= 0, 1.0, 0.0)
        
        # Add to DataFrame
        grid["v_fast"] = v_fast_arr
        grid["v_slow"] = v_slow_arr
        grid["v_ewma"] = v_ewma_arr
        grid["v_blend"] = v_blend_arr
        grid["V_rem"] = V_rem_arr
        grid["sigma_rem"] = sigma_rem_arr
        grid["drift"] = drift_arr
        grid["prob_gaussian"] = prob_gaussian
        grid["prob_student_t"] = prob_student_t
        grid["cap_applied"] = cap_applied_arr
        
        grid["edge_gaussian"] = grid["pm_mid"] - grid["prob_gaussian"]
        grid["edge_student_t"] = grid["pm_mid"] - grid["prob_student_t"]
        
        return grid, diagnostics


# =============================================================================
# Helper Functions
# =============================================================================

def analyze_edge(df: pd.DataFrame, prob_col: str = "prob_gaussian") -> dict:
    """Analyze edge between Polymarket and model prices."""
    if prob_col not in df.columns:
        prob_col = "fundamental_prob" if "fundamental_prob" in df.columns else prob_col
    
    valid = df.dropna(subset=["pm_mid", prob_col])
    if valid.empty:
        return {}
    
    edge = valid["pm_mid"] - valid[prob_col]
    
    return {
        "mean_edge": float(edge.mean()),
        "std_edge": float(edge.std()),
        "mean_abs_edge": float(edge.abs().mean()),
        "max_edge": float(edge.max()),
        "min_edge": float(edge.min()),
        "pct_pm_overpriced": float((edge > 0).mean() * 100),
        "n_capped": int(df["cap_applied"].sum()) if "cap_applied" in df.columns else 0,
    }
