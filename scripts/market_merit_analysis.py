#!/usr/bin/env python3
"""
Market Merit Analysis - Statistically Rigorous Evaluation

This script properly evaluates whether Polymarket has predictive merit by:
1. Using ONE independent sample per hour per τ (time-to-expiry)
2. Using proper scoring rules (log loss, Brier)
3. Comparing to baselines (coin flip, simple Gaussian model)
4. Block bootstrap by hour for confidence intervals
5. Testing incremental information (does PM add info beyond Binance state?)

Based on proper forecaster evaluation methodology.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import multiprocessing as mp

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import minimize

# =============================================================================
# CONFIGURATION
# =============================================================================

# Date range (UTC)
START_DATE = datetime(2026, 1, 18, 13, tzinfo=timezone.utc)
END_DATE = datetime(2026, 1, 25, 15, tzinfo=timezone.utc)

# Fixed τ values (time remaining in seconds) - ONE sample per hour per τ
TAU_VALUES = [3000, 1800, 900, 300, 120]  # 50min, 30min, 15min, 5min, 2min

# Asset
ASSET = "BTC"

# Parallel workers
MAX_WORKERS = None

# Bootstrap iterations
N_BOOTSTRAP = 1000

# Pricer config - clean config with asymmetric EWMA and optional drift
PRICER_CONFIG = {
    "fast_halflife_sec": 300.0,
    "slow_halflife_sec": 1800.0,
    "alpha": 0.3,
    "cap_multiplier": 8.0,
    "enable_capping": True,
    "tod_ramp_sec": 60.0,
    "use_tod_prior": True,
    "student_t_nu": 5.0,
    # Asymmetric EWMA - react fast to high vol, slow to low vol
    "asymmetric_ewma": True,
    "downside_lambda_mult": 0.3,
    # Drift adjustment - anticipate PM repricing when Binance trends
    "use_drift": False,           # Tested: hurts performance, PM already prices momentum
    "drift_halflife_sec": 30.0,
    "drift_weight": 0.5,
    "drift_horizon_sec": 60.0,    # Only extrapolate 60s of drift
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class HourResult:
    """One independent observation per hour per τ."""
    utc_time: datetime
    hour_et: int
    outcome_up: bool  # y ∈ {0, 1}
    return_pct: float  # final return
    tau_sec: int  # time to expiry when sampled
    
    # Polymarket forecast
    pm_bid: float
    pm_ask: float
    pm_mid: float
    pm_spread: float
    
    # Model forecast
    model_prob: float
    
    # Binance state at snapshot
    binance_price: float
    open_price: float
    r_at_snapshot: float  # log(price/open)
    
    # Realized volatility (properly computed)
    hourly_realized_vol: float  # σ per hour, NOT annualized
    
    error: str | None = None


# =============================================================================
# WORKER FUNCTION
# =============================================================================

def process_hour(args: tuple) -> list[HourResult]:
    """Process a single hour - return one sample per τ."""
    utc_time, asset, pricer_config, tau_values, tod_lookback_days = args
    
    from src.data import load_session
    from src.pricing import GaussianEWMAPricer, GaussianEWMAConfig, TODProfile
    
    market_date = utc_time.date()
    hour_utc = utc_time.hour
    
    hour_et = (hour_utc - 5) % 24
    if hour_utc < 5:
        market_date = (utc_time - timedelta(days=1)).date()
    
    results = []
    
    try:
        session = load_session(asset, market_date, hour_et=hour_et, lookback_hours=3)
        
        if session.outcome is None:
            return [HourResult(
                utc_time=utc_time, hour_et=hour_et, outcome_up=False, return_pct=0,
                tau_sec=0, pm_bid=0, pm_ask=0, pm_mid=0, pm_spread=0,
                model_prob=0, binance_price=0, open_price=0, r_at_snapshot=0,
                hourly_realized_vol=0, error="No outcome"
            )]
        
        # Build TOD profile
        tod_sessions = []
        for days_back in range(1, tod_lookback_days + 1):
            past_date = market_date - timedelta(days=days_back)
            try:
                tod_sess = load_session(asset, past_date, hour_et=hour_et, lookback_hours=0)
                tod_sessions.append(tod_sess)
            except Exception:
                pass
        
        tod_profile = TODProfile.from_sessions(tod_sessions, sample_ms=1000) if tod_sessions else TODProfile.constant(0.25)
        
        config = GaussianEWMAConfig(**pricer_config)
        pricer = GaussianEWMAPricer(config=config, tod_profile=tod_profile)
        
        model_df, _ = pricer.price_session(session, output_sample_ms=1000)
        
        if model_df.empty:
            return [HourResult(
                utc_time=utc_time, hour_et=hour_et, outcome_up=False, return_pct=0,
                tau_sec=0, pm_bid=0, pm_ask=0, pm_mid=0, pm_spread=0,
                model_prob=0, binance_price=0, open_price=0, r_at_snapshot=0,
                hourly_realized_vol=0, error="Empty model grid"
            )]
        
        # Get PM data
        pm_bbo = session.polymarket_bbo
        if pm_bbo is None or pm_bbo.empty:
            return [HourResult(
                utc_time=utc_time, hour_et=hour_et, outcome_up=False, return_pct=0,
                tau_sec=0, pm_bid=0, pm_ask=0, pm_mid=0, pm_spread=0,
                model_prob=0, binance_price=0, open_price=0, r_at_snapshot=0,
                hourly_realized_vol=0, error="No PM data"
            )]
        
        pm_bbo = pm_bbo.copy()
        
        # Normalize to UP probability
        if session.token_is_up is False:
            pm_bbo["bid_px_up"] = 1.0 - pm_bbo["ask_px"]
            pm_bbo["ask_px_up"] = 1.0 - pm_bbo["bid_px"]
        else:
            pm_bbo["bid_px_up"] = pm_bbo["bid_px"]
            pm_bbo["ask_px_up"] = pm_bbo["ask_px"]
        
        pm_bbo["mid_up"] = (pm_bbo["bid_px_up"] + pm_bbo["ask_px_up"]) / 2
        pm_bbo["spread"] = pm_bbo["ask_px_up"] - pm_bbo["bid_px_up"]
        
        start_ms = int(session.utc_start.timestamp() * 1000)
        pm_bbo["elapsed_sec"] = (pm_bbo["ts_recv"] - start_ms) / 1000
        
        outcome_is_up = session.outcome.outcome.lower() == "up"
        open_price = session.outcome.open_price
        
        # Compute PROPER hourly realized volatility
        # IMPORTANT: Use resampled data to avoid microstructure noise bias
        # Per-trade returns underestimate vol due to bid-ask bounce cancellation
        trades = session.binance_trades
        hourly_realized_vol = 0.0
        if trades is not None and len(trades) > 100:
            prices = trades["price"].values
            ts = trades["ts_recv"].values
            
            # Method: Resample to 1-second grid, then compute returns
            # This removes microstructure noise
            ts_sec = (ts - ts[0]) / 1000  # seconds from start
            log_prices = np.log(prices)
            
            # Linear interpolation to 1-second grid
            from scipy import interpolate
            try:
                f = interpolate.interp1d(ts_sec, log_prices, kind='linear', 
                                         bounds_error=False, fill_value='extrapolate')
                # Sample every second for the duration we have
                max_sec = min(ts_sec[-1], 3600)
                t_grid = np.arange(0, max_sec, 1)
                if len(t_grid) > 10:
                    log_prices_grid = f(t_grid)
                    returns_1sec = np.diff(log_prices_grid)
                    # Hourly vol = std(1-sec returns) * sqrt(3600)
                    hourly_realized_vol = np.std(returns_1sec) * np.sqrt(3600)
            except Exception:
                # Fallback: Parkinson high-low estimator
                high, low = prices.max(), prices.min()
                if low > 0:
                    hourly_realized_vol = np.log(high / low) / (2 * np.sqrt(np.log(2)))
        
        # Sample at each τ (time remaining)
        for tau_sec in tau_values:
            elapsed_sec = 3600 - tau_sec  # elapsed = 1h - remaining
            
            # Get PM snapshot at this elapsed time
            pm_at = pm_bbo[pm_bbo["elapsed_sec"] <= elapsed_sec + 2]
            if pm_at.empty:
                continue
            pm_row = pm_at.iloc[-1]
            
            # Get model snapshot
            model_at = model_df[model_df["elapsed_sec"] <= elapsed_sec + 2]
            if model_at.empty:
                continue
            model_row = model_at.iloc[-1]
            
            # Get Binance price (column name is 'bnc_mid')
            bnc_price = float(model_row["bnc_mid"])
            r_snapshot = np.log(bnc_price / open_price) if open_price > 0 else 0.0
            
            results.append(HourResult(
                utc_time=utc_time,
                hour_et=hour_et,
                outcome_up=outcome_is_up,
                return_pct=session.outcome.return_pct,
                tau_sec=tau_sec,
                pm_bid=float(pm_row["bid_px_up"]),
                pm_ask=float(pm_row["ask_px_up"]),
                pm_mid=float(pm_row["mid_up"]),
                pm_spread=float(pm_row["spread"]),
                model_prob=float(model_row["prob_gaussian"]),
                binance_price=bnc_price,
                open_price=open_price,
                r_at_snapshot=r_snapshot,
                hourly_realized_vol=hourly_realized_vol,
            ))
        
        return results if results else [HourResult(
            utc_time=utc_time, hour_et=hour_et, outcome_up=False, return_pct=0,
            tau_sec=0, pm_bid=0, pm_ask=0, pm_mid=0, pm_spread=0,
            model_prob=0, binance_price=0, open_price=0, r_at_snapshot=0,
            hourly_realized_vol=0, error="No valid samples"
        )]
        
    except Exception as e:
        return [HourResult(
            utc_time=utc_time, hour_et=hour_et, outcome_up=False, return_pct=0,
            tau_sec=0, pm_bid=0, pm_ask=0, pm_mid=0, pm_spread=0,
            model_prob=0, binance_price=0, open_price=0, r_at_snapshot=0,
            hourly_realized_vol=0, error=str(e)
        )]


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    """Compute log loss (cross-entropy). Lower is better."""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    """Compute Brier score. Lower is better."""
    return np.mean((p - y) ** 2)


def information_gain(p: np.ndarray, y: np.ndarray) -> float:
    """
    Information gain vs coin flip, in bits per forecast.
    
    IG > 0: forecaster has skill
    IG ≈ 0: forecaster has no skill
    IG < 0: forecaster is worse than random
    """
    p = np.clip(p, 1e-10, 1 - 1e-10)
    # Probability assigned to realized outcome
    pi = np.where(y == 1, p, 1 - p)
    # Bits gained vs 0.5
    return np.mean(np.log2(pi / 0.5))


def calibration_slope(p: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """
    Fit logistic calibration: logit(E[y|p]) = a + b*logit(p)
    
    Returns: (intercept, slope, slope_se)
    
    Interpretation:
    - b > 1: underconfident (too close to 50%)
    - b < 1: overconfident (too extreme)
    - b ≈ 1: well-calibrated
    """
    from scipy.special import logit, expit
    
    p = np.clip(p, 0.01, 0.99)
    X = logit(p)
    
    # Fit logistic regression
    def neg_log_lik(params):
        a, b = params
        logit_pred = a + b * X
        pred = expit(logit_pred)
        pred = np.clip(pred, 1e-10, 1 - 1e-10)
        return -np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred))
    
    result = minimize(neg_log_lik, [0, 1], method='BFGS')
    a, b = result.x
    
    # Approximate standard error via Hessian
    try:
        from scipy.optimize import approx_fprime
        hess = np.zeros((2, 2))
        eps = 1e-5
        for i in range(2):
            for j in range(2):
                def f(x):
                    return neg_log_lik(x)
                e_i = np.zeros(2); e_i[i] = eps
                e_j = np.zeros(2); e_j[j] = eps
                hess[i,j] = (f(result.x + e_i + e_j) - f(result.x + e_i - e_j) 
                           - f(result.x - e_i + e_j) + f(result.x - e_i - e_j)) / (4 * eps**2)
        cov = np.linalg.inv(hess)
        b_se = np.sqrt(cov[1, 1])
    except:
        b_se = np.nan
    
    return a, b, b_se


# =============================================================================
# BOOTSTRAP
# =============================================================================

def block_bootstrap(df: pd.DataFrame, metric_fn, n_iter: int = 1000, block_col: str = "utc_time") -> tuple[float, float, float]:
    """
    Block bootstrap by hour to get confidence intervals.
    
    Returns: (mean, ci_low, ci_high)
    """
    blocks = df[block_col].unique()
    n_blocks = len(blocks)
    
    estimates = []
    for _ in range(n_iter):
        # Sample blocks with replacement
        sampled_blocks = np.random.choice(blocks, size=n_blocks, replace=True)
        boot_df = pd.concat([df[df[block_col] == b] for b in sampled_blocks], ignore_index=True)
        if len(boot_df) > 0:
            estimates.append(metric_fn(boot_df))
    
    estimates = np.array(estimates)
    return np.mean(estimates), np.percentile(estimates, 2.5), np.percentile(estimates, 97.5)


# =============================================================================
# BASELINE MODELS
# =============================================================================

def coin_flip_forecast(n: int) -> np.ndarray:
    """Always predict 50%."""
    return np.full(n, 0.5)


def simple_gaussian_forecast(r: np.ndarray, tau_sec: np.ndarray, sigma_hourly: float) -> np.ndarray:
    """
    Simple Gaussian model: P(UP) = Φ(r / σ_remaining)
    
    Uses a FIXED hourly volatility (not the actual realized vol which we don't know yet).
    """
    # σ_remaining = σ_hourly * sqrt(τ / 3600)
    sigma_remaining = sigma_hourly * np.sqrt(tau_sec / 3600)
    sigma_remaining = np.maximum(sigma_remaining, 1e-10)
    z = r / sigma_remaining
    return stats.norm.cdf(z)


# =============================================================================
# INCREMENTAL INFO TEST
# =============================================================================

def test_incremental_info(df: pd.DataFrame, sigma_hourly: float) -> dict:
    """
    Test if PM adds information beyond obvious Binance state.
    
    Model A (baseline only): logit(p̂) = α₀ + α₁*logit(p₀)
    Model B (baseline + PM): logit(p̂) = β₀ + β₁*logit(p₀) + β₂*logit(q)
    
    Compare out-of-sample log loss.
    """
    from scipy.special import logit, expit
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    
    y = df["outcome_up"].astype(float).values
    r = df["r_at_snapshot"].values
    tau = df["tau_sec"].values
    q = np.clip(df["pm_mid"].values, 0.01, 0.99)
    
    # Baseline forecast
    p0 = simple_gaussian_forecast(r, tau, sigma_hourly)
    p0 = np.clip(p0, 0.01, 0.99)
    
    # Features
    X_baseline = logit(p0).reshape(-1, 1)
    X_with_pm = np.column_stack([logit(p0), logit(q)])
    
    # Fit and score via cross-validation
    model_A = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    model_B = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    
    # Use negative log loss as score (sklearn convention)
    scores_A = cross_val_score(model_A, X_baseline, y, cv=5, scoring='neg_log_loss')
    scores_B = cross_val_score(model_B, X_with_pm, y, cv=5, scoring='neg_log_loss')
    
    # Convert back to positive log loss
    ll_A = -np.mean(scores_A)
    ll_B = -np.mean(scores_B)
    
    # Also fit full models for coefficients
    model_A.fit(X_baseline, y)
    model_B.fit(X_with_pm, y)
    
    return {
        "baseline_log_loss": ll_A,
        "baseline_plus_pm_log_loss": ll_B,
        "pm_reduces_loss": ll_B < ll_A,
        "pm_reduction_pct": (ll_A - ll_B) / ll_A * 100 if ll_A > 0 else 0,
        "baseline_coef": model_A.coef_[0][0],
        "pm_coef_in_combined": model_B.coef_[0][1] if len(model_B.coef_[0]) > 1 else 0,
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("=" * 80)
    print("MARKET MERIT ANALYSIS - Statistically Rigorous Evaluation")
    print("=" * 80)
    print()
    print("Methodology:")
    print("  - ONE independent sample per hour per τ (no correlated samples)")
    print("  - Proper scoring rules: log loss, Brier, information gain")
    print("  - Baselines: coin flip (0.693 log loss, 0.25 Brier)")
    print("  - Block bootstrap by hour for confidence intervals")
    print("  - Incremental info test: does PM add info beyond Binance state?")
    print()
    print(f"Date range: {START_DATE} to {END_DATE} (UTC)")
    print(f"τ values: {TAU_VALUES} seconds")
    print(f"Bootstrap iterations: {N_BOOTSTRAP}")
    print()
    
    # Build hour list
    hours_to_process = []
    current = START_DATE
    while current <= END_DATE:
        hours_to_process.append((current, ASSET, PRICER_CONFIG, TAU_VALUES, 2))
        current += timedelta(hours=1)
    
    print(f"Processing {len(hours_to_process)} hours...")
    
    # Process in parallel
    all_results: list[HourResult] = []
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_hour, args): args[0] for args in hours_to_process}
        
        for i, future in enumerate(as_completed(futures)):
            hour_results = future.result()
            all_results.extend(hour_results)
            if (i + 1) % 10 == 0 or i == len(futures) - 1:
                print(f"  Processed {i+1}/{len(futures)} hours...")
    
    print()
    
    # Filter successful results
    successful = [r for r in all_results if r.error is None]
    failed = [r for r in all_results if r.error is not None]
    
    print(f"Successful samples: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print(f"Sample errors: {[f.error for f in failed[:5]]}")
    
    if not successful:
        print("No data to analyze!")
        return
    
    # Build DataFrame
    df = pd.DataFrame([{
        "utc_time": r.utc_time,
        "hour_et": r.hour_et,
        "outcome_up": r.outcome_up,
        "return_pct": r.return_pct,
        "tau_sec": r.tau_sec,
        "pm_bid": r.pm_bid,
        "pm_ask": r.pm_ask,
        "pm_mid": r.pm_mid,
        "pm_spread": r.pm_spread,
        "model_prob": r.model_prob,
        "r_at_snapshot": r.r_at_snapshot,
        "hourly_realized_vol": r.hourly_realized_vol,
    } for r in successful])
    
    y = df["outcome_up"].astype(float).values
    
    # Verify sample counts
    print()
    print("=" * 80)
    print("SAMPLE COUNTS (should be ~1 per hour per τ)")
    print("=" * 80)
    for tau in TAU_VALUES:
        n = len(df[df["tau_sec"] == tau])
        n_hours = df[df["tau_sec"] == tau]["utc_time"].nunique()
        print(f"  τ = {tau:4}s ({tau//60:2}min): {n} samples across {n_hours} hours")
    print()
    
    # Sanity check: volatility
    print("=" * 80)
    print("VOLATILITY SANITY CHECK")
    print("=" * 80)
    vol_hourly = df["hourly_realized_vol"].dropna()
    if len(vol_hourly) > 0:
        print(f"  Hourly σ (realized): mean={vol_hourly.mean()*100:.3f}%, median={vol_hourly.median()*100:.3f}%")
        print(f"  Hourly σ range: [{vol_hourly.min()*100:.3f}%, {vol_hourly.max()*100:.3f}%]")
        annualized = vol_hourly.mean() * np.sqrt(24 * 365)
        print(f"  Implied annual σ: {annualized*100:.1f}%")
        if annualized < 0.20:
            print("  ⚠️  WARNING: Annualized vol < 20% seems too low for BTC!")
        elif annualized > 1.5:
            print("  ⚠️  WARNING: Annualized vol > 150% seems too high!")
        else:
            print("  ✓ Volatility looks reasonable for BTC")
    print()
    
    # Estimate typical hourly vol for baseline model
    sigma_hourly = vol_hourly.median() if len(vol_hourly) > 0 else 0.005
    
    # ==========================================================================
    # SCORING BY τ
    # ==========================================================================
    
    print("=" * 80)
    print("SCORING BY TIME-TO-EXPIRY (τ)")
    print("=" * 80)
    print()
    print("Baselines: Coin flip log_loss=0.693, Brier=0.250")
    print()
    
    results_by_tau = []
    
    for tau in TAU_VALUES:
        tau_df = df[df["tau_sec"] == tau].copy()
        if len(tau_df) < 10:
            continue
        
        y_tau = tau_df["outcome_up"].astype(float).values
        pm = tau_df["pm_mid"].values
        model = tau_df["model_prob"].values
        
        # Baselines
        coin = coin_flip_forecast(len(tau_df))
        gaussian_baseline = simple_gaussian_forecast(
            tau_df["r_at_snapshot"].values,
            tau_df["tau_sec"].values,
            sigma_hourly
        )
        
        # Scores
        pm_ll = log_loss(pm, y_tau)
        pm_brier = brier_score(pm, y_tau)
        pm_ig = information_gain(pm, y_tau)
        
        model_ll = log_loss(model, y_tau)
        model_brier = brier_score(model, y_tau)
        model_ig = information_gain(model, y_tau)
        
        coin_ll = log_loss(coin, y_tau)
        coin_brier = brier_score(coin, y_tau)
        
        gauss_ll = log_loss(gaussian_baseline, y_tau)
        gauss_brier = brier_score(gaussian_baseline, y_tau)
        gauss_ig = information_gain(gaussian_baseline, y_tau)
        
        # Calibration slope
        pm_a, pm_b, pm_b_se = calibration_slope(pm, y_tau)
        model_a, model_b, model_b_se = calibration_slope(model, y_tau)
        
        results_by_tau.append({
            "tau_sec": tau,
            "tau_min": tau // 60,
            "n": len(tau_df),
            "n_hours": tau_df["utc_time"].nunique(),
            "pm_ll": pm_ll,
            "pm_brier": pm_brier,
            "pm_ig": pm_ig,
            "pm_cal_slope": pm_b,
            "model_ll": model_ll,
            "model_brier": model_brier,
            "model_ig": model_ig,
            "model_cal_slope": model_b,
            "coin_ll": coin_ll,
            "gauss_ll": gauss_ll,
            "gauss_brier": gauss_brier,
            "gauss_ig": gauss_ig,
        })
        
        print(f"τ = {tau}s ({tau//60}min) | n={len(tau_df)}")
        print(f"  {'':20} {'Log Loss':>10} {'Brier':>10} {'Info Gain':>10} {'Cal Slope':>10}")
        print(f"  {'Coin Flip':20} {coin_ll:>10.4f} {coin_brier:>10.4f} {'0.000':>10} {'-':>10}")
        print(f"  {'Gaussian Baseline':20} {gauss_ll:>10.4f} {gauss_brier:>10.4f} {gauss_ig:>10.4f} {'-':>10}")
        print(f"  {'Polymarket':20} {pm_ll:>10.4f} {pm_brier:>10.4f} {pm_ig:>10.4f} {pm_b:>10.2f}")
        print(f"  {'Model':20} {model_ll:>10.4f} {model_brier:>10.4f} {model_ig:>10.4f} {model_b:>10.2f}")
        
        # Interpretation
        pm_beats_coin = pm_ll < coin_ll
        pm_beats_gauss = pm_ll < gauss_ll
        model_beats_pm = model_ll < pm_ll
        
        print(f"  → PM beats coin flip: {'YES' if pm_beats_coin else 'NO'} (Δ={coin_ll - pm_ll:.4f})")
        print(f"  → PM beats Gaussian baseline: {'YES' if pm_beats_gauss else 'NO'} (Δ={gauss_ll - pm_ll:.4f})")
        print(f"  → Model beats PM: {'YES' if model_beats_pm else 'NO'} (Δ={pm_ll - model_ll:.4f})")
        print()
    
    # ==========================================================================
    # BOOTSTRAP CONFIDENCE INTERVALS
    # ==========================================================================
    
    print("=" * 80)
    print("BOOTSTRAP CONFIDENCE INTERVALS (by hour)")
    print("=" * 80)
    print()
    
    for tau in TAU_VALUES:
        tau_df = df[df["tau_sec"] == tau].copy()
        if len(tau_df) < 20:
            continue
        
        def pm_ll_metric(d):
            return log_loss(d["pm_mid"].values, d["outcome_up"].astype(float).values)
        
        def model_ll_metric(d):
            return log_loss(d["model_prob"].values, d["outcome_up"].astype(float).values)
        
        pm_mean, pm_lo, pm_hi = block_bootstrap(tau_df, pm_ll_metric, N_BOOTSTRAP)
        model_mean, model_lo, model_hi = block_bootstrap(tau_df, model_ll_metric, N_BOOTSTRAP)
        
        print(f"τ = {tau}s: PM log_loss = {pm_mean:.4f} [{pm_lo:.4f}, {pm_hi:.4f}]")
        print(f"        Model log_loss = {model_mean:.4f} [{model_lo:.4f}, {model_hi:.4f}]")
        
        # Check if CIs overlap
        if pm_hi < model_lo:
            print(f"        → PM significantly better (CIs don't overlap)")
        elif model_hi < pm_lo:
            print(f"        → Model significantly better (CIs don't overlap)")
        else:
            print(f"        → No significant difference (CIs overlap)")
        print()
    
    # ==========================================================================
    # INCREMENTAL INFORMATION TEST
    # ==========================================================================
    
    print("=" * 80)
    print("INCREMENTAL INFORMATION TEST")
    print("=" * 80)
    print()
    print("Question: Does PM add information beyond what's obvious from Binance state?")
    print()
    
    try:
        inc_info = test_incremental_info(df, sigma_hourly)
        print(f"  Baseline (Gaussian only) log loss: {inc_info['baseline_log_loss']:.4f}")
        print(f"  Baseline + PM log loss:            {inc_info['baseline_plus_pm_log_loss']:.4f}")
        print()
        if inc_info['pm_reduces_loss']:
            print(f"  ✓ PM ADDS INFORMATION beyond Binance state")
            print(f"    Reduction: {inc_info['pm_reduction_pct']:.2f}%")
        else:
            print(f"  ✗ PM does NOT add significant information")
        print()
        print(f"  PM coefficient in combined model: {inc_info['pm_coef_in_combined']:.3f}")
        print(f"  (Higher = PM has more unique info)")
    except ImportError:
        print("  ⚠️  sklearn not installed, skipping incremental info test")
    except Exception as e:
        print(f"  ⚠️  Error in incremental info test: {e}")
    
    print()
    
    # ==========================================================================
    # CALIBRATION ANALYSIS
    # ==========================================================================
    
    print("=" * 80)
    print("CALIBRATION BY BUCKET (all τ combined)")
    print("=" * 80)
    print()
    
    pm_all = df["pm_mid"].values
    model_all = df["model_prob"].values
    y_all = df["outcome_up"].astype(float).values
    
    buckets = np.linspace(0, 1, 11)
    
    print(f"{'Bucket':>12} | {'PM Actual':>10} {'PM n':>6} {'PM Error':>10} | {'Model Actual':>12} {'Model n':>8} {'Model Error':>12}")
    print("-" * 90)
    
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i+1]
        mid = (lo + hi) / 2
        
        pm_mask = (pm_all >= lo) & (pm_all < hi)
        model_mask = (model_all >= lo) & (model_all < hi)
        
        pm_n = pm_mask.sum()
        model_n = model_mask.sum()
        
        pm_actual = y_all[pm_mask].mean() if pm_n > 0 else np.nan
        model_actual = y_all[model_mask].mean() if model_n > 0 else np.nan
        
        pm_err = (pm_actual - mid) * 100 if not np.isnan(pm_actual) else np.nan
        model_err = (model_actual - mid) * 100 if not np.isnan(model_actual) else np.nan
        
        pm_str = f"{pm_actual*100:.1f}%" if not np.isnan(pm_actual) else "n/a"
        model_str = f"{model_actual*100:.1f}%" if not np.isnan(model_actual) else "n/a"
        pm_err_str = f"{pm_err:+.1f}pp" if not np.isnan(pm_err) else "n/a"
        model_err_str = f"{model_err:+.1f}pp" if not np.isnan(model_err) else "n/a"
        
        print(f"{lo*100:.0f}-{hi*100:.0f}%".rjust(12) + f" | {pm_str:>10} {pm_n:>6} {pm_err_str:>10} | {model_str:>12} {model_n:>8} {model_err_str:>12}")
    
    print()
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    
    print("=" * 80)
    print("SUMMARY: DOES POLYMARKET HAVE MERIT?")
    print("=" * 80)
    print()
    
    # Aggregate metrics
    pm_ll_total = log_loss(pm_all, y_all)
    pm_ig_total = information_gain(pm_all, y_all)
    model_ll_total = log_loss(model_all, y_all)
    model_ig_total = information_gain(model_all, y_all)
    coin_ll_total = np.log(2)  # 0.693
    
    print(f"Overall log loss (lower is better):")
    print(f"  Coin flip:   {coin_ll_total:.4f}")
    print(f"  Polymarket:  {pm_ll_total:.4f} (Δ vs coin: {coin_ll_total - pm_ll_total:+.4f})")
    print(f"  Model:       {model_ll_total:.4f} (Δ vs coin: {coin_ll_total - model_ll_total:+.4f})")
    print()
    print(f"Information Gain (bits per forecast, >0 = skill):")
    print(f"  Polymarket:  {pm_ig_total:+.4f} bits")
    print(f"  Model:       {model_ig_total:+.4f} bits")
    print()
    
    # Final verdict
    pm_has_skill = pm_ll_total < coin_ll_total - 0.01  # meaningful margin
    model_has_skill = model_ll_total < coin_ll_total - 0.01
    pm_beats_model = pm_ll_total < model_ll_total
    
    print("VERDICT:")
    if pm_has_skill:
        print(f"  ✓ Polymarket HAS predictive skill (beats coin flip by {(coin_ll_total - pm_ll_total):.4f})")
    else:
        print(f"  ✗ Polymarket has NO meaningful skill vs coin flip")
    
    if model_has_skill:
        print(f"  ✓ Model HAS predictive skill (beats coin flip by {(coin_ll_total - model_ll_total):.4f})")
    else:
        print(f"  ✗ Model has NO meaningful skill vs coin flip")
    
    if pm_beats_model:
        print(f"  → Polymarket beats Model by {(model_ll_total - pm_ll_total):.4f} log loss")
    else:
        print(f"  → Model beats Polymarket by {(pm_ll_total - model_ll_total):.4f} log loss")
    
    print()
    print("=" * 80)
    
    # Save results
    output_path = Path(__file__).parent.parent / "outputs"
    output_path.mkdir(exist_ok=True)
    
    results_df = pd.DataFrame(results_by_tau)
    results_df.to_csv(output_path / "market_merit_results.csv", index=False)
    print(f"Results saved to: {output_path / 'market_merit_results.csv'}")
    
    # ==========================================================================
    # VISUALIZATION
    # ==========================================================================
    
    print()
    print("Generating visualizations...")
    
    fig = create_merit_dashboard(df, results_by_tau, TAU_VALUES)
    
    html_path = output_path / "market_merit_dashboard.html"
    fig.write_html(str(html_path))
    print(f"Dashboard saved to: {html_path}")
    
    # Open in browser
    import webbrowser
    webbrowser.open(f"file://{html_path.resolve()}")
    
    return df, results_by_tau


def create_merit_dashboard(df: pd.DataFrame, results_by_tau: list, tau_values: list):
    """Create comprehensive visualization dashboard."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Log Loss by Time-to-Expiry (τ)",
            "Information Gain by τ",
            "Calibration: Polymarket",
            "Calibration: Model",
            "Performance Gap (PM - Model)",
            "Calibration Slope by τ"
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )
    
    # Colors
    pm_color = "#2ecc71"     # green
    model_color = "#e74c3c"  # red
    baseline_color = "#95a5a6"  # gray
    
    # Extract data from results
    taus = [r["tau_sec"] for r in results_by_tau]
    tau_labels = [f"{r['tau_min']}m" for r in results_by_tau]
    pm_ll = [r["pm_ll"] for r in results_by_tau]
    model_ll = [r["model_ll"] for r in results_by_tau]
    gauss_ll = [r["gauss_ll"] for r in results_by_tau]
    pm_ig = [r["pm_ig"] for r in results_by_tau]
    model_ig = [r["model_ig"] for r in results_by_tau]
    pm_slope = [r["pm_cal_slope"] for r in results_by_tau]
    model_slope = [r["model_cal_slope"] for r in results_by_tau]
    
    # 1. Log Loss by τ (bar chart)
    fig.add_trace(
        go.Bar(name="Polymarket", x=tau_labels, y=pm_ll, marker_color=pm_color, legendgroup="pm"),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name="Model", x=tau_labels, y=model_ll, marker_color=model_color, legendgroup="model"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(name="Gaussian Baseline", x=tau_labels, y=gauss_ll, 
                   mode="lines+markers", line=dict(color=baseline_color, dash="dash"), legendgroup="baseline"),
        row=1, col=1
    )
    # Coin flip reference
    fig.add_hline(y=0.693, line_dash="dot", line_color="black", annotation_text="Coin Flip", row=1, col=1)
    
    # 2. Information Gain by τ
    fig.add_trace(
        go.Bar(name="Polymarket", x=tau_labels, y=pm_ig, marker_color=pm_color, showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(name="Model", x=tau_labels, y=model_ig, marker_color=model_color, showlegend=False),
        row=1, col=2
    )
    fig.add_hline(y=0, line_dash="dot", line_color="black", row=1, col=2)
    
    # 3 & 4. Calibration curves
    buckets = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
               (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    
    pm_cal = []
    model_cal = []
    bucket_mids = []
    pm_counts = []
    model_counts = []
    
    for lo, hi in buckets:
        mid = (lo + hi) / 2
        bucket_mids.append(mid)
        
        # PM calibration
        pm_mask = (df["pm_mid"] >= lo) & (df["pm_mid"] < hi)
        if pm_mask.sum() > 0:
            pm_cal.append(df.loc[pm_mask, "outcome_up"].astype(float).mean())
            pm_counts.append(pm_mask.sum())
        else:
            pm_cal.append(np.nan)
            pm_counts.append(0)
        
        # Model calibration
        model_mask = (df["model_prob"] >= lo) & (df["model_prob"] < hi)
        if model_mask.sum() > 0:
            model_cal.append(df.loc[model_mask, "outcome_up"].astype(float).mean())
            model_counts.append(model_mask.sum())
        else:
            model_cal.append(np.nan)
            model_counts.append(0)
    
    # Perfect calibration line
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash="dash"),
                   name="Perfect", showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash="dash"),
                   showlegend=False),
        row=2, col=2
    )
    
    # PM calibration
    fig.add_trace(
        go.Scatter(x=bucket_mids, y=pm_cal, mode="markers+lines",
                   marker=dict(size=[max(5, min(20, n/5)) for n in pm_counts], color=pm_color),
                   line=dict(color=pm_color),
                   text=[f"n={n}" for n in pm_counts],
                   hovertemplate="Predicted: %{x:.0%}<br>Actual: %{y:.0%}<br>%{text}",
                   name="PM", showlegend=False),
        row=2, col=1
    )
    
    # Model calibration
    fig.add_trace(
        go.Scatter(x=bucket_mids, y=model_cal, mode="markers+lines",
                   marker=dict(size=[max(5, min(20, n/5)) for n in model_counts], color=model_color),
                   line=dict(color=model_color),
                   text=[f"n={n}" for n in model_counts],
                   hovertemplate="Predicted: %{x:.0%}<br>Actual: %{y:.0%}<br>%{text}",
                   name="Model", showlegend=False),
        row=2, col=2
    )
    
    # 5. Performance Gap (PM - Model log loss)
    gap = [m - p for p, m in zip(pm_ll, model_ll)]
    colors = [pm_color if g > 0 else model_color for g in gap]
    
    fig.add_trace(
        go.Bar(x=tau_labels, y=gap, marker_color=colors, showlegend=False,
               text=[f"{g:+.3f}" for g in gap], textposition="outside"),
        row=3, col=1
    )
    fig.add_hline(y=0, line_dash="solid", line_color="black", row=3, col=1)
    fig.add_annotation(
        text="↑ PM wins", x=0.05, y=0.95, xref="x5 domain", yref="y5 domain",
        showarrow=False, font=dict(color=pm_color, size=10)
    )
    fig.add_annotation(
        text="↓ Model wins", x=0.05, y=0.05, xref="x5 domain", yref="y5 domain",
        showarrow=False, font=dict(color=model_color, size=10)
    )
    
    # 6. Calibration Slope by τ
    fig.add_trace(
        go.Scatter(x=tau_labels, y=pm_slope, mode="lines+markers", name="PM Slope",
                   line=dict(color=pm_color), marker=dict(size=10), showlegend=False),
        row=3, col=2
    )
    fig.add_trace(
        go.Scatter(x=tau_labels, y=model_slope, mode="lines+markers", name="Model Slope",
                   line=dict(color=model_color), marker=dict(size=10), showlegend=False),
        row=3, col=2
    )
    fig.add_hline(y=1.0, line_dash="dot", line_color="black", 
                  annotation_text="Perfect=1.0", row=3, col=2)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="<b>Market Merit Analysis: Polymarket vs Model</b><br>" +
                 f"<sup>73 independent samples across 5 time horizons | Asymmetric EWMA</sup>",
            x=0.5, xanchor="center"
        ),
        height=900,
        width=1200,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        template="plotly_white"
    )
    
    # Axis labels
    fig.update_yaxes(title_text="Log Loss ↓", row=1, col=1)
    fig.update_yaxes(title_text="Bits ↑", row=1, col=2)
    fig.update_xaxes(title_text="Predicted Prob", row=2, col=1)
    fig.update_yaxes(title_text="Actual Freq", row=2, col=1)
    fig.update_xaxes(title_text="Predicted Prob", row=2, col=2)
    fig.update_yaxes(title_text="Actual Freq", row=2, col=2)
    fig.update_xaxes(title_text="Time to Expiry", row=3, col=1)
    fig.update_yaxes(title_text="Δ Log Loss (Model - PM)", row=3, col=1)
    fig.update_xaxes(title_text="Time to Expiry", row=3, col=2)
    fig.update_yaxes(title_text="Cal Slope (1=perfect)", row=3, col=2)
    
    return fig


if __name__ == "__main__":
    main()
