#!/usr/bin/env python3
"""
Pricer Calibration Test Framework

Comprehensive analysis comparing pricer predictions against Polymarket and outcomes.
Uses parallel processing for speed since markets are independent.

Usage:
    python pricer_calibration_test.py
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
import multiprocessing as mp

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# =============================================================================
# CONFIGURATION
# =============================================================================

# Date range (UTC)
START_DATE = datetime(2026, 1, 18, 13, tzinfo=timezone.utc)
END_DATE = datetime(2026, 1, 21, 17, tzinfo=timezone.utc)

# Sample at every N seconds into the hour (more granular = more data points)
SAMPLE_SECONDS = [60, 120, 180, 300, 600, 900, 1200, 1800, 2400, 3000, 3300, 3540]
# Translates to: 1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 55, 59 minutes

# Asset
ASSET = "BTC"

# Parallel workers (None = use all CPUs)
MAX_WORKERS = None

# Pricer config (for Gaussian EWMA)
PRICER_CONFIG = {
    "fast_halflife_sec": 300.0,
    "slow_halflife_sec": 1800.0,
    "alpha": 0.3,
    "cap_multiplier": 8.0,
    "enable_capping": True,
    "tod_ramp_sec": 60.0,
    "use_tod_prior": True,
    "tod_blend_beta": 0.0,
    "vol_multiplier": 1.0,
    "student_t_nu": 8.0,
    
    # FIX 1: Volatility floor (prevents overconfidence in calm periods)
    # Set to annualized volatility, e.g., 0.036 = 3.6% min vol
    "vol_floor_annual": 0.036,            # Try: 0.0 (off), 0.03, 0.036, 0.04
    
    # FIX 2: ITM volatility boost (extra uncertainty when price moved up)
    # Adds vol when price is above open to account for reversal risk
    "itm_vol_boost": 0.3,                 # Try: 0.0 (off), 0.2, 0.3, 0.5
    "itm_vol_time_scaling": True,         # Scale boost by time elapsed
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SessionResult:
    """Result from processing one session."""
    utc_time: datetime
    market_date: str
    hour_et: int
    outcome_up: bool
    return_pct: float
    samples: list[dict]
    lookback_vol: float = 0.0  # 3h pre-market volatility (annualized)
    realized_vol: float = 0.0  # In-market realized volatility (annualized)
    error: str | None = None


# =============================================================================
# WORKER FUNCTION (runs in separate process)
# =============================================================================

def process_session(args: tuple) -> SessionResult | None:
    """Process a single session - runs in parallel."""
    utc_time, asset, pricer_config, sample_seconds, tod_lookback_days = args
    
    from src.data import load_session
    from src.pricing import GaussianEWMAPricer, GaussianEWMAConfig, TODProfile
    
    market_date = utc_time.date()
    hour_utc = utc_time.hour
    
    hour_et = (hour_utc - 5) % 24
    if hour_utc < 5:
        market_date = (utc_time - timedelta(days=1)).date()
    
    try:
        session = load_session(asset, market_date, hour_et=hour_et, lookback_hours=3)
        
        if session.outcome is None:
            return SessionResult(
                utc_time=utc_time, market_date=str(market_date), hour_et=hour_et,
                outcome_up=False, return_pct=0, samples=[], error="No outcome"
            )
        
        # Build TOD profile
        tod_sessions = []
        for days_back in range(1, tod_lookback_days + 1):
            past_date = market_date - timedelta(days=days_back)
            try:
                tod_sess = load_session(asset, past_date, hour_et=hour_et, lookback_hours=0)
                tod_sessions.append(tod_sess)
            except Exception:
                pass
        
        if tod_sessions:
            tod_profile = TODProfile.from_sessions(tod_sessions, sample_ms=1000)
        else:
            tod_profile = TODProfile.constant(0.25)
        
        config = GaussianEWMAConfig(**pricer_config)
        pricer = GaussianEWMAPricer(config=config, tod_profile=tod_profile)
        
        df, _ = pricer.price_session(session, output_sample_ms=1000)
        
        if df.empty:
            return SessionResult(
                utc_time=utc_time, market_date=str(market_date), hour_et=hour_et,
                outcome_up=False, return_pct=0, samples=[], error="Empty pricing grid"
            )
        
        # Get PM data
        pm_bbo = session.polymarket_bbo
        if pm_bbo is None or pm_bbo.empty:
            return SessionResult(
                utc_time=utc_time, market_date=str(market_date), hour_et=hour_et,
                outcome_up=False, return_pct=0, samples=[], error="No PM data"
            )
        
        pm_bbo = pm_bbo.copy()
        pm_bbo["mid"] = (pm_bbo["bid_px"] + pm_bbo["ask_px"]) / 2
        if session.token_is_up is False:
            pm_bbo["mid"] = 1.0 - pm_bbo["mid"]
        
        start_ms = int(session.utc_start.timestamp() * 1000)
        pm_bbo["elapsed_sec"] = (pm_bbo["ts_recv"] - start_ms) / 1000
        
        outcome_is_up = session.outcome.outcome.lower() == "up"
        
        # Compute volatility measures
        lookback_vol = 0.0
        realized_vol = 0.0
        try:
            # Lookback vol from 3h pre-market
            lookback_trades = session.binance_lookback_trades
            if lookback_trades is not None and len(lookback_trades) > 100:
                lb_prices = lookback_trades["price"].values
                lb_returns = np.diff(np.log(lb_prices))
                # Annualize: assume ~1 trade per 100ms avg, so ~36000 per hour, ~3600*3 for 3h
                lookback_vol = np.std(lb_returns) * np.sqrt(365 * 24 * 3600 / 0.1)  # rough annualization
            
            # Realized vol from market hour
            market_trades = session.binance_trades
            if market_trades is not None and len(market_trades) > 100:
                m_prices = market_trades["price"].values
                m_returns = np.diff(np.log(m_prices))
                realized_vol = np.std(m_returns) * np.sqrt(365 * 24 * 3600 / 0.1)
        except Exception:
            pass
        
        samples = []
        
        for sec in sample_seconds:
            # PM mid
            pm_at = pm_bbo[pm_bbo["elapsed_sec"] <= sec + 5]
            if pm_at.empty:
                continue
            pm_mid = float(pm_at.iloc[-1]["mid"])
            
            # Model predictions
            model_at = df[df["elapsed_sec"] <= sec + 5]
            if model_at.empty:
                continue
            
            # Also get current return at this point
            current_return = float(model_at.iloc[-1].get("r_0_to_t", 0)) * 100  # as percentage
            
            samples.append({
                "elapsed_sec": sec,
                "pm_mid": pm_mid,
                "model_gaussian": float(model_at.iloc[-1]["prob_gaussian"]),
                "model_student_t": float(model_at.iloc[-1]["prob_student_t"]),
                "current_return_pct": current_return,
            })
        
        return SessionResult(
            utc_time=utc_time,
            market_date=str(market_date),
            hour_et=hour_et,
            outcome_up=outcome_is_up,
            return_pct=session.outcome.return_pct,
            samples=samples,
            lookback_vol=lookback_vol,
            realized_vol=realized_vol,
        )
        
    except Exception as e:
        return SessionResult(
            utc_time=utc_time, market_date=str(market_date), hour_et=hour_et,
            outcome_up=False, return_pct=0, samples=[], error=str(e)
        )


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def brier_decomposition(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> dict:
    """
    Decompose Brier score into reliability, resolution, and uncertainty.
    
    - Reliability (lower is better): measures calibration error
    - Resolution (higher is better): measures ability to separate outcomes
    - Uncertainty: base rate uncertainty (constant for dataset)
    
    Brier = Reliability - Resolution + Uncertainty
    """
    o_bar = outcomes.mean()
    uncertainty = o_bar * (1 - o_bar)
    
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(probs, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    
    reliability = 0.0
    resolution = 0.0
    
    for b in range(n_bins):
        mask = bin_idx == b
        n_b = mask.sum()
        if n_b == 0:
            continue
        
        f_b = probs[mask].mean()  # forecast probability in bin
        o_b = outcomes[mask].mean()  # observed frequency in bin
        
        reliability += n_b * (f_b - o_b) ** 2
        resolution += n_b * (o_b - o_bar) ** 2
    
    n = len(probs)
    reliability /= n
    resolution /= n
    
    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier": reliability - resolution + uncertainty,
    }


def compute_sharpness(probs: np.ndarray) -> dict:
    """
    Measure sharpness (spread) of predictions.
    
    - High sharpness = predictions are confident (close to 0 or 1)
    - Low sharpness = predictions hedge toward 0.5
    """
    # Distance from 0.5
    dist_from_half = np.abs(probs - 0.5)
    
    return {
        "mean_confidence": dist_from_half.mean(),  # avg distance from 0.5
        "std_confidence": dist_from_half.std(),
        "pct_confident": (dist_from_half > 0.25).mean(),  # % predictions >0.75 or <0.25
        "pct_extreme": (dist_from_half > 0.4).mean(),  # % predictions >0.9 or <0.1
    }


def compute_discrimination(probs: np.ndarray, outcomes: np.ndarray) -> dict:
    """
    Measure discrimination ability (can model separate UP from DOWN?).
    
    Returns AUC and mean probabilities for UP vs DOWN outcomes.
    """
    # Manual AUC calculation (Wilcoxon-Mann-Whitney)
    try:
        pos_probs = probs[outcomes == 1]
        neg_probs = probs[outcomes == 0]
        if len(pos_probs) == 0 or len(neg_probs) == 0:
            auc = 0.5
        else:
            # Count concordant pairs
            auc = np.mean([np.mean(pos_probs > n) + 0.5 * np.mean(pos_probs == n) for n in neg_probs])
    except Exception:
        auc = 0.5
    
    up_probs = probs[outcomes == 1]
    down_probs = probs[outcomes == 0]
    
    return {
        "auc": auc,
        "mean_prob_when_up": up_probs.mean() if len(up_probs) > 0 else 0.5,
        "mean_prob_when_down": down_probs.mean() if len(down_probs) > 0 else 0.5,
        "separation": (up_probs.mean() - down_probs.mean()) if len(up_probs) > 0 and len(down_probs) > 0 else 0,
    }


def analyze_edge_vs_pm(df: pd.DataFrame) -> dict:
    """
    Analyze where model differs from PM and whether those edges are profitable.
    """
    df = df.copy()
    
    for model in ["model_gaussian", "model_student_t"]:
        df[f"{model}_edge"] = df[model] - df["pm_mid"]  # positive = model more bullish
    
    results = {}
    
    for model in ["model_gaussian", "model_student_t"]:
        edge_col = f"{model}_edge"
        
        # When model is more bullish than PM
        bullish = df[df[edge_col] > 0.05]
        bearish = df[df[edge_col] < -0.05]
        
        # Does being more bullish when outcome is UP pay off?
        bullish_correct = bullish["outcome_up"].mean() if len(bullish) > 0 else 0.5
        bearish_correct = (1 - bearish["outcome_up"]).mean() if len(bearish) > 0 else 0.5
        
        # Edge magnitude when correct vs incorrect
        correct_mask = (df[model] > 0.5) == df["outcome_up"]
        
        results[model] = {
            "mean_edge": df[edge_col].mean(),
            "std_edge": df[edge_col].std(),
            "n_bullish": len(bullish),
            "n_bearish": len(bearish),
            "bullish_accuracy": bullish_correct,
            "bearish_accuracy": bearish_correct,
            "edge_when_correct": df.loc[correct_mask, edge_col].mean() if correct_mask.sum() > 0 else 0,
            "edge_when_wrong": df.loc[~correct_mask, edge_col].mean() if (~correct_mask).sum() > 0 else 0,
        }
    
    return results


def analyze_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance by time into the hour."""
    df = df.copy()
    df["minute"] = df["elapsed_sec"] / 60
    
    rows = []
    for model in ["pm_mid", "model_gaussian", "model_student_t"]:
        for minute, group in df.groupby(df["minute"].round()):
            brier = ((group[model] - group["outcome_up"].astype(float)) ** 2).mean()
            accuracy = ((group[model] > 0.5) == group["outcome_up"]).mean()
            sharpness = np.abs(group[model] - 0.5).mean()
            
            rows.append({
                "model": model,
                "minute": minute,
                "brier": brier,
                "accuracy": accuracy,
                "sharpness": sharpness,
                "n": len(group),
            })
    
    return pd.DataFrame(rows)


def analyze_by_volatility(df: pd.DataFrame, vol_col: str = "lookback_vol") -> pd.DataFrame:
    """
    Analyze performance by volatility regime.
    
    Uses percentile-based buckets:
    - Low vol: bottom 33%
    - Medium vol: middle 33%
    - High vol: top 33%
    """
    df = df.copy()
    
    # Get unique session volatilities (one per session, not per sample)
    session_vols = df.groupby("utc_time")[vol_col].first()
    
    # Compute percentiles
    p33 = session_vols.quantile(0.33)
    p66 = session_vols.quantile(0.66)
    
    def vol_bucket(v):
        if v <= p33:
            return f"Low (<{p33:.1%})"
        elif v <= p66:
            return f"Medium ({p33:.1%}-{p66:.1%})"
        else:
            return f"High (>{p66:.1%})"
    
    df["vol_regime"] = df[vol_col].apply(vol_bucket)
    
    rows = []
    models = ["pm_mid", "model_gaussian", "model_student_t"]
    model_names = {"pm_mid": "Polymarket", "model_gaussian": "Gaussian", "model_student_t": "Student-t"}
    
    for regime in [f"Low (<{p33:.1%})", f"Medium ({p33:.1%}-{p66:.1%})", f"High (>{p66:.1%})"]:
        regime_df = df[df["vol_regime"] == regime]
        if len(regime_df) == 0:
            continue
            
        for model in models:
            brier = ((regime_df[model] - regime_df["outcome_up"].astype(float)) ** 2).mean()
            accuracy = ((regime_df[model] > 0.5) == regime_df["outcome_up"]).mean()
            sharpness = np.abs(regime_df[model] - 0.5).mean()
            
            # Edge vs PM (only for models)
            if model != "pm_mid":
                edge = (regime_df[model] - regime_df["pm_mid"]).mean()
            else:
                edge = 0.0
            
            rows.append({
                "vol_regime": regime,
                "model": model_names[model],
                "brier": brier,
                "accuracy": accuracy,
                "sharpness": sharpness,
                "edge_vs_pm": edge,
                "n": len(regime_df),
            })
    
    return pd.DataFrame(rows), p33, p66


def analyze_by_moneyness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze performance by 'moneyness' (how far current price is from open).
    
    Buckets:
    - Near ATM: |return| < 0.1%
    - Slightly ITM/OTM: 0.1% <= |return| < 0.3%
    - Deep ITM/OTM: |return| >= 0.3%
    """
    df = df.copy()
    
    def moneyness_bucket(r):
        abs_r = abs(r)
        if abs_r < 0.1:
            return "ATM (|r|<0.1%)"
        elif abs_r < 0.3:
            return "Near (0.1-0.3%)"
        else:
            return "Deep (>0.3%)"
    
    df["moneyness"] = df["current_return_pct"].apply(moneyness_bucket)
    
    rows = []
    for model in ["pm_mid", "model_gaussian", "model_student_t"]:
        for bucket, group in df.groupby("moneyness"):
            brier = ((group[model] - group["outcome_up"].astype(float)) ** 2).mean()
            accuracy = ((group[model] > 0.5) == group["outcome_up"]).mean()
            sharpness = np.abs(group[model] - 0.5).mean()
            
            rows.append({
                "model": model,
                "moneyness": bucket,
                "brier": brier,
                "accuracy": accuracy,
                "sharpness": sharpness,
                "n": len(group),
            })
    
    return pd.DataFrame(rows)


def compute_calibration(df: pd.DataFrame, prob_col: str, n_bins: int = 10) -> list[dict]:
    """Compute calibration stats for a probability column."""
    df = df.copy()
    buckets = np.linspace(0, 1, n_bins + 1)
    df["bucket"] = pd.cut(df[prob_col], bins=buckets, labels=False, include_lowest=True)
    
    results = []
    for i in range(n_bins):
        bucket_df = df[df["bucket"] == i]
        n = len(bucket_df)
        if n == 0:
            continue
        
        hits = bucket_df["outcome_up"].sum()
        hit_rate = hits / n
        bucket_mid = (buckets[i] + buckets[i+1]) / 2
        
        # Wilson confidence interval
        ci_low, ci_high = wilson_ci(int(hits), n)
        
        results.append({
            "bucket_low": buckets[i],
            "bucket_high": buckets[i+1],
            "bucket_mid": bucket_mid,
            "label": f"{buckets[i]*100:.0f}-{buckets[i+1]*100:.0f}%",
            "n": n,
            "hits": hits,
            "hit_rate": hit_rate,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "error": hit_rate - bucket_mid,
        })
    
    return results


def wilson_ci(hits: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score confidence interval."""
    if n == 0:
        return 0.0, 1.0
    p = hits / n
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return max(0, (centre - spread) / denom), min(1, (centre + spread) / denom)


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_analysis_dashboard(df: pd.DataFrame) -> go.Figure:
    """Create comprehensive analysis dashboard."""
    
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[
            "Calibration Curves",
            "Brier Score Decomposition",
            "Sharpness Distribution",
            "Performance by Time",
            "Performance by Moneyness",
            "Edge vs Polymarket",
            "Discrimination (ROC-like)",
            "Model vs PM Scatter",
            "Prediction Distributions",
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
        specs=[
            [{"type": "scatter"}, {"type": "bar"}, {"type": "histogram"}],
            [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "histogram"}],
        ]
    )
    
    colors = {"pm_mid": "#2196F3", "model_gaussian": "#FF9800", "model_student_t": "#4CAF50"}
    names = {"pm_mid": "Polymarket", "model_gaussian": "Gaussian", "model_student_t": "Student-t"}
    
    # 1. Calibration curves
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash", color="gray"), showlegend=False), row=1, col=1)
    
    for model, color in colors.items():
        calib = compute_calibration(df, model)
        x = [c["bucket_mid"] for c in calib]
        y = [c["hit_rate"] for c in calib]
        ci_low = [c["ci_low"] for c in calib]
        ci_high = [c["ci_high"] for c in calib]
        
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers+lines", name=names[model],
            marker=dict(size=8, color=color),
            error_y=dict(type="data", array=[h-m for h, m in zip(ci_high, y)], arrayminus=[m-l for m, l in zip(y, ci_low)], visible=True),
        ), row=1, col=1)
    
    fig.update_xaxes(title_text="Predicted", range=[0,1], row=1, col=1)
    fig.update_yaxes(title_text="Actual", range=[0,1], row=1, col=1)
    
    # 2. Brier decomposition
    brier_data = []
    for model in colors.keys():
        probs = df[model].values
        outcomes = df["outcome_up"].astype(float).values
        decomp = brier_decomposition(probs, outcomes)
        brier_data.append({
            "model": names[model],
            "Reliability": decomp["reliability"],
            "Resolution": decomp["resolution"],
            "Uncertainty": decomp["uncertainty"],
            "Brier": decomp["brier"],
        })
    
    brier_df = pd.DataFrame(brier_data)
    
    for i, metric in enumerate(["Reliability", "Resolution"]):
        fig.add_trace(go.Bar(
            x=brier_df["model"], y=brier_df[metric], name=metric,
            marker_color=["#E91E63", "#9C27B0"][i],
            showlegend=(i == 0),
        ), row=1, col=2)
    
    fig.update_xaxes(title_text="", row=1, col=2)
    fig.update_yaxes(title_text="Score", row=1, col=2)
    
    # 3. Sharpness distribution (histogram of predictions)
    for model, color in colors.items():
        fig.add_trace(go.Histogram(
            x=df[model], nbinsx=30, name=f"{names[model]} dist",
            marker_color=color, opacity=0.5, showlegend=False,
        ), row=1, col=3)
    
    fig.update_xaxes(title_text="Predicted P(Up)", row=1, col=3)
    fig.update_yaxes(title_text="Count", row=1, col=3)
    
    # 4. Performance by time
    time_df = analyze_by_time(df)
    for model, color in colors.items():
        model_data = time_df[time_df["model"] == model]
        fig.add_trace(go.Scatter(
            x=model_data["minute"], y=model_data["brier"],
            mode="markers+lines", name=f"{names[model]} by time",
            marker=dict(color=color), showlegend=False,
        ), row=2, col=1)
    
    fig.update_xaxes(title_text="Minute", row=2, col=1)
    fig.update_yaxes(title_text="Brier Score", row=2, col=1)
    
    # 5. Performance by moneyness
    money_df = analyze_by_moneyness(df)
    bucket_order = ["ATM (|r|<0.1%)", "Near (0.1-0.3%)", "Deep (>0.3%)"]
    
    for model, color in colors.items():
        model_data = money_df[money_df["model"] == model].set_index("moneyness").reindex(bucket_order)
        fig.add_trace(go.Bar(
            x=bucket_order, y=model_data["brier"].values,
            name=f"{names[model]} by money", marker_color=color, showlegend=False,
        ), row=2, col=2)
    
    fig.update_xaxes(title_text="Moneyness", row=2, col=2)
    fig.update_yaxes(title_text="Brier Score", row=2, col=2)
    
    # 6. Edge vs PM
    df_copy = df.copy()
    df_copy["gauss_edge"] = df_copy["model_gaussian"] - df_copy["pm_mid"]
    
    up_df = df_copy[df_copy["outcome_up"]]
    down_df = df_copy[~df_copy["outcome_up"]]
    
    fig.add_trace(go.Scatter(
        x=up_df["pm_mid"], y=up_df["gauss_edge"],
        mode="markers", name="UP outcomes",
        marker=dict(color="#4CAF50", size=5, opacity=0.6), showlegend=False,
    ), row=2, col=3)
    
    fig.add_trace(go.Scatter(
        x=down_df["pm_mid"], y=down_df["gauss_edge"],
        mode="markers", name="DOWN outcomes",
        marker=dict(color="#F44336", size=5, opacity=0.6), showlegend=False,
    ), row=2, col=3)
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=3)
    
    fig.update_xaxes(title_text="PM Probability", row=2, col=3)
    fig.update_yaxes(title_text="Model - PM Edge", row=2, col=3)
    
    # 7. Discrimination - probabilities when UP vs DOWN
    for model, color in colors.items():
        up_probs = df[df["outcome_up"]][model]
        down_probs = df[~df["outcome_up"]][model]
        
        # Box plot style using scatter
        fig.add_trace(go.Box(
            y=up_probs, name=f"{names[model][:4]} UP",
            marker_color=color, showlegend=False,
        ), row=3, col=1)
        fig.add_trace(go.Box(
            y=down_probs, name=f"{names[model][:4]} DN",
            marker_color=color, opacity=0.5, showlegend=False,
        ), row=3, col=1)
    
    fig.update_yaxes(title_text="Predicted P(Up)", row=3, col=1)
    
    # 8. Model vs PM scatter
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash", color="gray"), showlegend=False), row=3, col=2)
    
    fig.add_trace(go.Scatter(
        x=df["pm_mid"], y=df["model_gaussian"],
        mode="markers", marker=dict(color="#FF9800", size=4, opacity=0.5),
        name="Gauss vs PM", showlegend=False,
    ), row=3, col=2)
    
    fig.update_xaxes(title_text="Polymarket P(Up)", row=3, col=2)
    fig.update_yaxes(title_text="Model P(Up)", row=3, col=2)
    
    # 9. Edge distribution
    df_copy["edge"] = df_copy["model_gaussian"] - df_copy["pm_mid"]
    
    fig.add_trace(go.Histogram(
        x=df_copy[df_copy["outcome_up"]]["edge"], nbinsx=40,
        name="Edge when UP", marker_color="#4CAF50", opacity=0.6, showlegend=False,
    ), row=3, col=3)
    fig.add_trace(go.Histogram(
        x=df_copy[~df_copy["outcome_up"]]["edge"], nbinsx=40,
        name="Edge when DOWN", marker_color="#F44336", opacity=0.6, showlegend=False,
    ), row=3, col=3)
    
    fig.add_vline(x=0, line_dash="dash", line_color="gray", row=3, col=3)
    
    fig.update_xaxes(title_text="Model - PM Edge", row=3, col=3)
    fig.update_yaxes(title_text="Count", row=3, col=3)
    
    # Layout
    fig.update_layout(
        title="Pricer Calibration Analysis Dashboard",
        height=1000,
        showlegend=True,
        barmode="group",
    )
    
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("PRICER CALIBRATION ANALYSIS")
    print("=" * 80)
    print()
    print(f"Date range: {START_DATE} to {END_DATE} (UTC)")
    print(f"Asset: {ASSET}")
    print(f"Sample points per hour: {len(SAMPLE_SECONDS)}")
    print(f"Workers: {MAX_WORKERS or mp.cpu_count()}")
    print()
    
    # Build session list
    sessions_to_process = []
    current = START_DATE
    while current <= END_DATE:
        sessions_to_process.append((
            current, ASSET, PRICER_CONFIG, SAMPLE_SECONDS, 2
        ))
        current += timedelta(hours=1)
    
    print(f"Processing {len(sessions_to_process)} sessions...")
    print()
    
    # Process in parallel
    results: list[SessionResult] = []
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_session, args): args[0] for args in sessions_to_process}
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                results.append(result)
                if (i + 1) % 10 == 0 or i == len(futures) - 1:
                    print(f"  Processed {i+1}/{len(futures)} sessions...")
    
    print()
    
    successful = [r for r in results if r.error is None and len(r.samples) > 0]
    failed = [r for r in results if r.error is not None]
    
    print(f"Successful: {len(successful)}, Failed: {len(failed)}")
    print()
    
    # Build DataFrame
    rows = []
    for result in successful:
        for sample in result.samples:
            rows.append({
                "utc_time": result.utc_time,
                "hour_et": result.hour_et,
                "outcome_up": result.outcome_up,
                "return_pct": result.return_pct,
                "elapsed_sec": sample["elapsed_sec"],
                "pm_mid": sample["pm_mid"],
                "model_gaussian": sample["model_gaussian"],
                "model_student_t": sample["model_student_t"],
                "current_return_pct": sample["current_return_pct"],
                "lookback_vol": result.lookback_vol,
                "realized_vol": result.realized_vol,
            })
    
    df = pd.DataFrame(rows)
    print(f"Total data points: {len(df)}")
    print()
    
    # ==========================================================================
    # DETAILED ANALYSIS
    # ==========================================================================
    
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()
    
    models = ["pm_mid", "model_gaussian", "model_student_t"]
    model_names = ["Polymarket", "Gaussian", "Student-t"]
    
    # Basic metrics table
    metrics = []
    for model, name in zip(models, model_names):
        probs = df[model].values
        outcomes = df["outcome_up"].astype(float).values
        
        brier = ((probs - outcomes) ** 2).mean()
        accuracy = ((probs > 0.5) == outcomes.astype(bool)).mean()
        log_loss = -np.mean(outcomes * np.log(np.clip(probs, 1e-10, 1)) + (1 - outcomes) * np.log(np.clip(1 - probs, 1e-10, 1)))
        
        decomp = brier_decomposition(probs, outcomes)
        sharp = compute_sharpness(probs)
        disc = compute_discrimination(probs, outcomes)
        
        metrics.append({
            "Model": name,
            "Brier": brier,
            "Accuracy": accuracy,
            "Log Loss": log_loss,
            "AUC": disc["auc"],
            "Reliability": decomp["reliability"],
            "Resolution": decomp["resolution"],
            "Sharpness": sharp["mean_confidence"],
            "Separation": disc["separation"],
        })
    
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    
    # Calibration details
    print("=" * 80)
    print("CALIBRATION BY BUCKET")
    print("=" * 80)
    print()
    
    for model, name in zip(models, model_names):
        print(f"{name}:")
        print("-" * 60)
        calib = compute_calibration(df, model)
        for c in calib:
            ci_str = f"[{c['ci_low']:.1%}, {c['ci_high']:.1%}]"
            print(f"  {c['label']:>10}: {c['hit_rate']:6.1%} actual (n={c['n']:3}) | Error: {c['error']*100:+5.1f}pp | 95% CI: {ci_str}")
        print()
    
    # Edge analysis
    print("=" * 80)
    print("EDGE VS POLYMARKET")
    print("=" * 80)
    print()
    
    edge_analysis = analyze_edge_vs_pm(df)
    for model in ["model_gaussian", "model_student_t"]:
        name = "Gaussian" if "gaussian" in model else "Student-t"
        ea = edge_analysis[model]
        print(f"{name} vs PM:")
        print(f"  Mean edge: {ea['mean_edge']*100:+.2f}pp (std: {ea['std_edge']*100:.2f}pp)")
        print(f"  When model more bullish (>{'+5pp'}): {ea['n_bullish']} samples, {ea['bullish_accuracy']:.1%} accuracy")
        print(f"  When model more bearish (<{'-5pp'}): {ea['n_bearish']} samples, {ea['bearish_accuracy']:.1%} accuracy")
        print(f"  Edge when correct: {ea['edge_when_correct']*100:+.2f}pp | Edge when wrong: {ea['edge_when_wrong']*100:+.2f}pp")
        print()
    
    # By moneyness
    print("=" * 80)
    print("PERFORMANCE BY MONEYNESS")
    print("=" * 80)
    print()
    
    money_df = analyze_by_moneyness(df)
    for bucket in ["ATM (|r|<0.1%)", "Near (0.1-0.3%)", "Deep (>0.3%)"]:
        print(f"{bucket}:")
        for model, name in zip(models, model_names):
            row = money_df[(money_df["model"] == model) & (money_df["moneyness"] == bucket)]
            if len(row) > 0:
                r = row.iloc[0]
                print(f"  {name:12}: Brier={r['brier']:.4f} | Acc={r['accuracy']:.1%} | Sharp={r['sharpness']:.3f} | n={r['n']}")
        print()
    
    # By volatility regime
    print("=" * 80)
    print("PERFORMANCE BY VOLATILITY REGIME (Lookback 3h)")
    print("=" * 80)
    print()
    
    vol_df, p33, p66 = analyze_by_volatility(df, "lookback_vol")
    vol_regimes = [f"Low (<{p33:.1%})", f"Medium ({p33:.1%}-{p66:.1%})", f"High (>{p66:.1%})"]
    
    for regime in vol_regimes:
        print(f"{regime}:")
        for name in ["Polymarket", "Gaussian", "Student-t"]:
            row = vol_df[(vol_df["model"] == name) & (vol_df["vol_regime"] == regime)]
            if len(row) > 0:
                r = row.iloc[0]
                edge_str = f"Edge={r['edge_vs_pm']*100:+.1f}pp" if name != "Polymarket" else ""
                print(f"  {name:12}: Brier={r['brier']:.4f} | Acc={r['accuracy']:.1%} | Sharp={r['sharpness']:.3f} | n={int(r['n'])} {edge_str}")
        print()
    
    # Realized vol analysis
    print("=" * 80)
    print("PERFORMANCE BY VOLATILITY REGIME (Realized In-Hour)")
    print("=" * 80)
    print()
    
    rvol_df, rp33, rp66 = analyze_by_volatility(df, "realized_vol")
    rvol_regimes = [f"Low (<{rp33:.1%})", f"Medium ({rp33:.1%}-{rp66:.1%})", f"High (>{rp66:.1%})"]
    
    for regime in rvol_regimes:
        print(f"{regime}:")
        for name in ["Polymarket", "Gaussian", "Student-t"]:
            row = rvol_df[(rvol_df["model"] == name) & (rvol_df["vol_regime"] == regime)]
            if len(row) > 0:
                r = row.iloc[0]
                edge_str = f"Edge={r['edge_vs_pm']*100:+.1f}pp" if name != "Polymarket" else ""
                print(f"  {name:12}: Brier={r['brier']:.4f} | Acc={r['accuracy']:.1%} | Sharp={r['sharpness']:.3f} | n={int(r['n'])} {edge_str}")
        print()
    
    # Frequency and win rate analysis
    print("=" * 80)
    print("REGIME FREQUENCY & WIN RATE ANALYSIS")
    print("=" * 80)
    print()
    
    # Add squared errors to df for comparison
    df_analysis = df.copy()
    df_analysis["pm_sq_err"] = (df_analysis["pm_mid"] - df_analysis["outcome_up"].astype(float)) ** 2
    df_analysis["gauss_sq_err"] = (df_analysis["model_gaussian"] - df_analysis["outcome_up"].astype(float)) ** 2
    df_analysis["model_wins"] = df_analysis["gauss_sq_err"] < df_analysis["pm_sq_err"]
    df_analysis["pm_wins"] = df_analysis["pm_sq_err"] < df_analysis["gauss_sq_err"]
    df_analysis["tie"] = df_analysis["gauss_sq_err"] == df_analysis["pm_sq_err"]
    
    total_n = len(df_analysis)
    model_win_rate = df_analysis["model_wins"].mean()
    pm_win_rate = df_analysis["pm_wins"].mean()
    
    print(f"OVERALL WIN RATE (per sample, by squared error):")
    print(f"  Model wins: {df_analysis['model_wins'].sum()} ({model_win_rate:.1%})")
    print(f"  PM wins:    {df_analysis['pm_wins'].sum()} ({pm_win_rate:.1%})")
    print(f"  Ties:       {df_analysis['tie'].sum()} ({df_analysis['tie'].mean():.1%})")
    print()
    
    # By moneyness
    print("WIN RATE BY MONEYNESS:")
    for bucket in ["ATM (|r|<0.1%)", "Near (0.1-0.3%)", "Deep (>0.3%)"]:
        bucket_df = money_df[money_df["moneyness"] == bucket]
        if len(bucket_df) == 0:
            continue
        n = bucket_df[bucket_df["model"] == "pm_mid"]["n"].values[0] if len(bucket_df) > 0 else 0
        pct = n / total_n * 100
        
        # Get actual win rate for this bucket
        def money_bucket(r):
            abs_r = abs(r)
            if abs_r < 0.1:
                return "ATM (|r|<0.1%)"
            elif abs_r < 0.3:
                return "Near (0.1-0.3%)"
            else:
                return "Deep (>0.3%)"
        
        df_analysis["money_bucket"] = df_analysis["current_return_pct"].apply(money_bucket)
        bucket_samples = df_analysis[df_analysis["money_bucket"] == bucket]
        win_rate = bucket_samples["model_wins"].mean() if len(bucket_samples) > 0 else 0
        
        pm_brier = money_df[(money_df["model"] == "pm_mid") & (money_df["moneyness"] == bucket)]["brier"].values[0]
        gauss_brier = money_df[(money_df["model"] == "model_gaussian") & (money_df["moneyness"] == bucket)]["brier"].values[0]
        better = "MODEL" if gauss_brier < pm_brier else "PM"
        
        print(f"  {bucket:20}: {n:4} samples ({pct:5.1f}%) | Model wins {win_rate:.1%} | Brier: {better}")
    print()
    
    # By volatility regime
    print("WIN RATE BY LOOKBACK VOLATILITY REGIME:")
    
    # Re-bucket for win rate
    session_vols = df_analysis.groupby("utc_time")["lookback_vol"].first()
    vol_p33 = session_vols.quantile(0.33)
    vol_p66 = session_vols.quantile(0.66)
    
    def vol_bucket_fn(v):
        if v <= vol_p33:
            return "Low"
        elif v <= vol_p66:
            return "Medium"
        else:
            return "High"
    
    df_analysis["vol_bucket"] = df_analysis["lookback_vol"].apply(vol_bucket_fn)
    
    for regime, regime_label in [("Low", vol_regimes[0]), ("Medium", vol_regimes[1]), ("High", vol_regimes[2])]:
        regime_samples = df_analysis[df_analysis["vol_bucket"] == regime]
        n = len(regime_samples)
        pct = n / total_n * 100
        win_rate = regime_samples["model_wins"].mean() if n > 0 else 0
        
        pm_brier = vol_df[(vol_df["model"] == "Polymarket") & (vol_df["vol_regime"] == regime_label)]["brier"].values
        gauss_brier = vol_df[(vol_df["model"] == "Gaussian") & (vol_df["vol_regime"] == regime_label)]["brier"].values
        
        if len(pm_brier) > 0 and len(gauss_brier) > 0:
            better = "MODEL" if gauss_brier[0] < pm_brier[0] else "PM"
            diff = abs(gauss_brier[0] - pm_brier[0]) / pm_brier[0] * 100
            print(f"  {regime:8} ({regime_label}): {n:4} samples ({pct:5.1f}%) | Model wins {win_rate:.1%} | Brier: {better} (+{diff:.1f}%)")
    print()
    
    # Combined analysis: where should you trade?
    print("=" * 80)
    print("WHERE SHOULD YOU TRADE? (Combined Analysis)")
    print("=" * 80)
    print()
    
    # Cross-tabulate moneyness x volatility
    print("BRIER SCORE BY MONEYNESS x VOLATILITY:")
    print()
    print(f"{'':25} | {'Low Vol':^20} | {'Med Vol':^20} | {'High Vol':^20}")
    print("-" * 90)
    
    for bucket in ["ATM (|r|<0.1%)", "Near (0.1-0.3%)", "Deep (>0.3%)"]:
        row = f"{bucket:25} |"
        for regime in ["Low", "Medium", "High"]:
            cross_df = df_analysis[(df_analysis["money_bucket"] == bucket) & (df_analysis["vol_bucket"] == regime)]
            if len(cross_df) < 5:
                row += f" {'n/a':^18} |"
                continue
            pm_brier = ((cross_df["pm_mid"] - cross_df["outcome_up"].astype(float))**2).mean()
            gauss_brier = ((cross_df["model_gaussian"] - cross_df["outcome_up"].astype(float))**2).mean()
            win_rate = cross_df["model_wins"].mean()
            winner = "M" if gauss_brier < pm_brier else "P"
            row += f" {winner} {gauss_brier:.3f} (n={len(cross_df):3}) |"
        print(row)
    print()
    print("Legend: M=Model better, P=PM better")
    print()
    
    # Key insights
    print("=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()
    
    pm_brier = metrics_df[metrics_df["Model"] == "Polymarket"]["Brier"].values[0]
    gauss_brier = metrics_df[metrics_df["Model"] == "Gaussian"]["Brier"].values[0]
    
    pm_sharp = metrics_df[metrics_df["Model"] == "Polymarket"]["Sharpness"].values[0]
    gauss_sharp = metrics_df[metrics_df["Model"] == "Gaussian"]["Sharpness"].values[0]
    
    print(f"1. BRIER SCORE: PM ({pm_brier:.4f}) vs Gaussian ({gauss_brier:.4f})")
    if gauss_brier > pm_brier:
        print(f"   → Model is {(gauss_brier/pm_brier - 1)*100:.1f}% worse than Polymarket")
    else:
        print(f"   → Model is {(1 - gauss_brier/pm_brier)*100:.1f}% better than Polymarket")
    
    print()
    print(f"2. SHARPNESS: PM ({pm_sharp:.3f}) vs Gaussian ({gauss_sharp:.3f})")
    if gauss_sharp < pm_sharp:
        print(f"   → Model hedges toward 50% more than PM ({(1-gauss_sharp/pm_sharp)*100:.1f}% less confident)")
        print(f"   → This suggests model MAY OVERESTIMATE volatility")
    else:
        print(f"   → Model is more confident than PM ({(gauss_sharp/pm_sharp - 1)*100:.1f}% sharper)")
    
    print()
    pm_res = metrics_df[metrics_df["Model"] == "Polymarket"]["Resolution"].values[0]
    gauss_res = metrics_df[metrics_df["Model"] == "Gaussian"]["Resolution"].values[0]
    print(f"3. RESOLUTION: PM ({pm_res:.4f}) vs Gaussian ({gauss_res:.4f})")
    print(f"   → {'PM' if pm_res > gauss_res else 'Model'} better separates UP from DOWN outcomes")
    
    print()
    atm_pm = money_df[(money_df["model"] == "pm_mid") & (money_df["moneyness"] == "ATM (|r|<0.1%)")]["brier"].values
    atm_gauss = money_df[(money_df["model"] == "model_gaussian") & (money_df["moneyness"] == "ATM (|r|<0.1%)")]["brier"].values
    if len(atm_pm) > 0 and len(atm_gauss) > 0:
        print(f"4. ATM PERFORMANCE: PM Brier ({atm_pm[0]:.4f}) vs Gaussian ({atm_gauss[0]:.4f})")
        if atm_gauss[0] > atm_pm[0]:
            print(f"   → Model struggles most when price is near open (high uncertainty)")
        else:
            print(f"   → Model handles ATM situations well")
    
    print()
    # Volatility regime insights
    low_vol_pm = vol_df[(vol_df["model"] == "Polymarket") & (vol_df["vol_regime"] == vol_regimes[0])]["brier"].values
    low_vol_gauss = vol_df[(vol_df["model"] == "Gaussian") & (vol_df["vol_regime"] == vol_regimes[0])]["brier"].values
    high_vol_pm = vol_df[(vol_df["model"] == "Polymarket") & (vol_df["vol_regime"] == vol_regimes[2])]["brier"].values
    high_vol_gauss = vol_df[(vol_df["model"] == "Gaussian") & (vol_df["vol_regime"] == vol_regimes[2])]["brier"].values
    
    if len(low_vol_pm) > 0 and len(low_vol_gauss) > 0:
        print(f"5. LOW VOL REGIME: PM Brier ({low_vol_pm[0]:.4f}) vs Gaussian ({low_vol_gauss[0]:.4f})")
        if low_vol_gauss[0] > low_vol_pm[0]:
            print(f"   → Model struggles in low volatility environments")
        else:
            print(f"   → Model handles low vol well")
    
    if len(high_vol_pm) > 0 and len(high_vol_gauss) > 0:
        print()
        print(f"6. HIGH VOL REGIME: PM Brier ({high_vol_pm[0]:.4f}) vs Gaussian ({high_vol_gauss[0]:.4f})")
        if high_vol_gauss[0] < high_vol_pm[0]:
            print(f"   → Model BEATS PM in high volatility environments! (+{(1 - high_vol_gauss[0]/high_vol_pm[0])*100:.1f}%)")
        else:
            print(f"   → Model underperforms PM even in high vol")
    
    # Suggestions for low vol fix
    print()
    print("=" * 80)
    print("SUGGESTIONS FOR LOW-VOL FIX")
    print("=" * 80)
    print()
    print("Your model underperforms in low/medium vol because it UNDERESTIMATES volatility")
    print("in calm periods, making predictions TOO confident (sharper than they should be).")
    print()
    print("Potential fixes:")
    print()
    print("1. VOLATILITY FLOOR: V_used = max(V_estimated, V_floor)")
    print(f"   Suggested floor: ~{vol_p33:.1%} (33rd percentile of lookback vol)")
    print()
    print("2. LONG-RUN BLEND: V_used = alpha * V_estimated + (1-alpha) * V_longrun")
    print("   This pulls low estimates toward historical average")
    print()
    print("3. REGIME-DEPENDENT MULTIPLIER:")
    print(f"   - Low vol regime:  vol_multiplier = 1.3-1.5")
    print(f"   - Med vol regime:  vol_multiplier = 1.1-1.2")  
    print(f"   - High vol regime: vol_multiplier = 1.0 (current)")
    print()
    print("4. USE PM AS SIGNAL IN LOW VOL:")
    print("   When lookback_vol < threshold, blend toward PM or skip trading")
    
    # Create visualization
    print()
    print("Creating dashboard...")
    
    fig = create_analysis_dashboard(df)
    
    output_path = Path(__file__).parent.parent / "outputs"
    output_path.mkdir(exist_ok=True)
    
    html_path = output_path / f"pricer_analysis_{ASSET}.html"
    fig.write_html(str(html_path))
    print(f"  Saved to: {html_path}")
    
    try:
        import webbrowser
        webbrowser.open(f"file://{html_path.absolute()}")
    except Exception:
        pass
    
    print()
    print("=" * 80)
    print("Done!")
    print("=" * 80)
    
    return df, metrics_df


if __name__ == "__main__":
    main()
