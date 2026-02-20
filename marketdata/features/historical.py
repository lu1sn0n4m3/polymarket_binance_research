"""Historical cross-session feature computation.

Utilities for analyzing patterns across multiple hourly market sessions.
"""

from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd

from marketdata.data.session import HourlyMarketSession, load_sessions_range


def compute_hourly_returns(
    sessions: list[HourlyMarketSession],
) -> pd.DataFrame:
    """Compute hourly returns for a list of market sessions.
    
    Args:
        sessions: List of HourlyMarketSession instances
        
    Returns:
        DataFrame with columns:
        - date: Market date
        - hour_et: Hour in ET
        - asset: BTC or ETH
        - open_price: First trade price
        - close_price: Last trade price
        - return_pct: Percentage return
        - outcome: "up", "down", or "flat"
        - utc_start: Market start in UTC
    """
    rows = []
    
    for session in sessions:
        outcome = session.outcome
        
        if outcome is None:
            continue
        
        rows.append({
            "date": session.market_date,
            "hour_et": session.hour_et,
            "asset": session.asset,
            "open_price": outcome.open_price,
            "close_price": outcome.close_price,
            "return_pct": outcome.return_pct,
            "outcome": outcome.outcome,
            "utc_start": session.utc_start,
        })
    
    return pd.DataFrame(rows)


def get_historical_hourly_stats(
    asset: Literal["BTC", "ETH"],
    hour_et: int,
    start_date: date,
    end_date: date,
    conn=None,
) -> dict:
    """Get historical statistics for a specific hour.
    
    Useful for understanding typical behavior at a given hour.
    
    Args:
        asset: "BTC" or "ETH"
        hour_et: Hour in Eastern Time
        start_date: Start date
        end_date: End date
        conn: DuckDB connection
        
    Returns:
        Dictionary with stats:
        - n_sessions: Number of sessions
        - up_rate: Fraction of "up" outcomes
        - mean_return: Mean return
        - std_return: Standard deviation of returns
        - max_return: Maximum return
        - min_return: Minimum return
        - sessions: List of session data
    """
    sessions = load_sessions_range(
        asset=asset,
        start_date=start_date,
        end_date=end_date,
        hours_et=[hour_et],
        preload=False,
        conn=conn,
    )
    
    returns_df = compute_hourly_returns(sessions)
    
    if returns_df.empty:
        return {
            "n_sessions": 0,
            "up_rate": np.nan,
            "mean_return": np.nan,
            "std_return": np.nan,
            "max_return": np.nan,
            "min_return": np.nan,
            "sessions": [],
        }
    
    returns = returns_df["return_pct"]
    outcomes = returns_df["outcome"]
    
    return {
        "n_sessions": len(returns_df),
        "up_rate": (outcomes == "up").mean(),
        "mean_return": returns.mean(),
        "std_return": returns.std(),
        "max_return": returns.max(),
        "min_return": returns.min(),
        "sessions": returns_df.to_dict("records"),
    }


def compute_same_hour_features(
    session: HourlyMarketSession,
    lookback_days: int = 7,
    conn=None,
) -> dict:
    """Compute features from the same hour on previous days.
    
    Useful for capturing hour-of-day effects.
    
    Args:
        session: Current session
        lookback_days: Number of days to look back
        conn: DuckDB connection
        
    Returns:
        Dictionary with features:
        - prev_up_rate: Up rate in previous sessions at this hour
        - prev_mean_return: Mean return in previous sessions
        - prev_vol: Standard deviation of returns
        - n_prev_sessions: Number of previous sessions used
    """
    start_date = session.market_date - timedelta(days=lookback_days)
    end_date = session.market_date - timedelta(days=1)
    
    if start_date > end_date:
        return {
            "prev_up_rate": np.nan,
            "prev_mean_return": np.nan,
            "prev_vol": np.nan,
            "n_prev_sessions": 0,
        }
    
    prev_sessions = load_sessions_range(
        asset=session.asset,
        start_date=start_date,
        end_date=end_date,
        hours_et=[session.hour_et],
        preload=False,
        conn=conn,
    )
    
    returns_df = compute_hourly_returns(prev_sessions)
    
    if returns_df.empty:
        return {
            "prev_up_rate": np.nan,
            "prev_mean_return": np.nan,
            "prev_vol": np.nan,
            "n_prev_sessions": 0,
        }
    
    return {
        "prev_up_rate": (returns_df["outcome"] == "up").mean(),
        "prev_mean_return": returns_df["return_pct"].mean(),
        "prev_vol": returns_df["return_pct"].std(),
        "n_prev_sessions": len(returns_df),
    }


def compute_prior_hours_features(
    session: HourlyMarketSession,
    prior_hours: int = 3,
    conn=None,
) -> dict:
    """Compute features from prior hours on the same day.
    
    Useful for capturing intraday momentum/mean-reversion.
    
    Args:
        session: Current session
        prior_hours: Number of prior hours to consider
        conn: DuckDB connection
        
    Returns:
        Dictionary with features:
        - prior_up_count: Number of up outcomes in prior hours
        - prior_down_count: Number of down outcomes
        - prior_cumulative_return: Total return in prior hours
        - prior_sessions: List of prior session outcomes
    """
    prior_session_hours = [
        session.hour_et - i - 1
        for i in range(prior_hours)
        if session.hour_et - i - 1 >= 0
    ]
    
    if not prior_session_hours:
        return {
            "prior_up_count": 0,
            "prior_down_count": 0,
            "prior_cumulative_return": 0.0,
            "prior_sessions": [],
        }
    
    # Load prior sessions on same day
    prior_sessions = load_sessions_range(
        asset=session.asset,
        start_date=session.market_date,
        end_date=session.market_date,
        hours_et=prior_session_hours,
        preload=False,
        conn=conn,
    )
    
    returns_df = compute_hourly_returns(prior_sessions)
    
    if returns_df.empty:
        return {
            "prior_up_count": 0,
            "prior_down_count": 0,
            "prior_cumulative_return": 0.0,
            "prior_sessions": [],
        }
    
    return {
        "prior_up_count": int((returns_df["outcome"] == "up").sum()),
        "prior_down_count": int((returns_df["outcome"] == "down").sum()),
        "prior_cumulative_return": returns_df["return_pct"].sum(),
        "prior_sessions": returns_df.to_dict("records"),
    }


def build_session_feature_matrix(
    sessions: list[HourlyMarketSession],
    include_same_hour: bool = True,
    include_prior_hours: bool = True,
    same_hour_lookback: int = 7,
    prior_hours: int = 3,
    conn=None,
) -> pd.DataFrame:
    """Build a feature matrix for a list of sessions.
    
    Creates a DataFrame with one row per session and columns for
    various features that can be used for modeling.
    
    Args:
        sessions: List of sessions
        include_same_hour: Include same-hour historical features
        include_prior_hours: Include prior-hours-today features
        same_hour_lookback: Days to look back for same-hour features
        prior_hours: Number of prior hours for prior-hours features
        conn: DuckDB connection
        
    Returns:
        DataFrame with features and outcomes
    """
    rows = []
    
    for session in sessions:
        outcome = session.outcome
        if outcome is None:
            continue
        
        row = {
            "date": session.market_date,
            "hour_et": session.hour_et,
            "asset": session.asset,
            "open_price": outcome.open_price,
            "close_price": outcome.close_price,
            "return_pct": outcome.return_pct,
            "outcome": outcome.outcome,
            "is_up": 1 if outcome.is_up else 0,
        }
        
        if include_same_hour:
            same_hour = compute_same_hour_features(
                session, same_hour_lookback, conn
            )
            for k, v in same_hour.items():
                row[f"sh_{k}"] = v
        
        if include_prior_hours:
            prior = compute_prior_hours_features(session, prior_hours, conn)
            row["ph_up_count"] = prior["prior_up_count"]
            row["ph_down_count"] = prior["prior_down_count"]
            row["ph_cum_return"] = prior["prior_cumulative_return"]
        
        rows.append(row)
    
    return pd.DataFrame(rows)
