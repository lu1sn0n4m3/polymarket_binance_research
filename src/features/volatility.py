"""Volatility estimation from high-frequency data.

This module provides interfaces and basic implementations for volatility estimation.
You can extend these with more sophisticated estimators.
"""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import pandas as pd


def compute_returns(
    prices: pd.Series,
    method: Literal["log", "simple"] = "log",
) -> pd.Series:
    """Compute price returns.
    
    Args:
        prices: Price series
        method: "log" for log returns, "simple" for percentage returns
        
    Returns:
        Return series
    """
    if method == "log":
        return np.log(prices / prices.shift(1))
    else:
        return prices.pct_change()


def compute_realized_vol(
    prices: pd.Series,
    window: int | None = None,
    annualize: bool = True,
    periods_per_year: float = 365 * 24,  # Hourly data
) -> float | pd.Series:
    """Compute realized volatility from price series.
    
    Simple close-to-close volatility estimator.
    
    Args:
        prices: Price series
        window: Rolling window size (None = full period)
        annualize: If True, annualize the volatility
        periods_per_year: Number of periods per year for annualization
        
    Returns:
        Volatility (scalar if window=None, else Series)
    """
    returns = compute_returns(prices, method="log")
    
    if window is None:
        vol = returns.std()
    else:
        vol = returns.rolling(window).std()
    
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    
    return vol


def compute_yang_zhang_vol(
    open_prices: pd.Series,
    high_prices: pd.Series,
    low_prices: pd.Series,
    close_prices: pd.Series,
    window: int = 20,
    annualize: bool = True,
    periods_per_year: float = 365 * 24,
) -> pd.Series:
    """Compute Yang-Zhang volatility estimator.
    
    Combines overnight, open-to-close, and Rogers-Satchell components.
    More efficient than close-to-close for OHLC data.
    
    Args:
        open_prices: Open prices
        high_prices: High prices
        low_prices: Low prices
        close_prices: Close prices
        window: Rolling window
        annualize: If True, annualize
        periods_per_year: Periods per year
        
    Returns:
        Yang-Zhang volatility series
    """
    # Overnight component
    log_oc = np.log(open_prices / close_prices.shift(1))
    overnight_var = log_oc.rolling(window).var()
    
    # Open-to-close component
    log_co = np.log(close_prices / open_prices)
    open_close_var = log_co.rolling(window).var()
    
    # Rogers-Satchell component
    log_ho = np.log(high_prices / open_prices)
    log_lo = np.log(low_prices / open_prices)
    log_hc = np.log(high_prices / close_prices)
    log_lc = np.log(low_prices / close_prices)
    
    rs_var = (log_ho * log_hc + log_lo * log_lc).rolling(window).mean()
    
    # Yang-Zhang estimator
    k = 0.34 / (1 + (window + 1) / (window - 1))
    vol_squared = overnight_var + k * open_close_var + (1 - k) * rs_var
    vol = np.sqrt(vol_squared.clip(lower=0))
    
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    
    return vol


def compute_parkinson_vol(
    high_prices: pd.Series,
    low_prices: pd.Series,
    window: int = 20,
    annualize: bool = True,
    periods_per_year: float = 365 * 24,
) -> pd.Series:
    """Compute Parkinson volatility from high-low range.
    
    More efficient than close-to-close when high/low data is available.
    
    Args:
        high_prices: High prices
        low_prices: Low prices
        window: Rolling window
        annualize: If True, annualize
        periods_per_year: Periods per year
        
    Returns:
        Parkinson volatility series
    """
    log_hl = np.log(high_prices / low_prices)
    
    # Parkinson factor: 1 / (4 * ln(2))
    factor = 1.0 / (4 * np.log(2))
    
    vol_squared = factor * (log_hl ** 2).rolling(window).mean()
    vol = np.sqrt(vol_squared.clip(lower=0))
    
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    
    return vol


class RealizedVolEstimator(ABC):
    """Abstract base class for realized volatility estimators.
    
    Subclass this to implement custom volatility estimators.
    """
    
    @abstractmethod
    def compute(
        self,
        df: pd.DataFrame,
        ts_col: str = "ts_recv",
        price_col: str = "price",
    ) -> float:
        """Compute realized volatility from a DataFrame.
        
        Args:
            df: DataFrame with timestamp and price columns
            ts_col: Timestamp column name
            price_col: Price column name
            
        Returns:
            Annualized volatility estimate
        """
        pass
    
    @abstractmethod
    def compute_rolling(
        self,
        df: pd.DataFrame,
        window_seconds: int,
        ts_col: str = "ts_recv",
        price_col: str = "price",
    ) -> pd.Series:
        """Compute rolling realized volatility.
        
        Args:
            df: DataFrame with timestamp and price columns
            window_seconds: Window size in seconds
            ts_col: Timestamp column name
            price_col: Price column name
            
        Returns:
            Series of volatility estimates
        """
        pass


class SimpleRealizedVol(RealizedVolEstimator):
    """Simple close-to-close realized volatility estimator."""
    
    def __init__(
        self,
        sample_interval_ms: int = 1000,
        annualize: bool = True,
        periods_per_year: float = 365 * 24 * 3600,  # Seconds
    ):
        """
        Args:
            sample_interval_ms: Resample to this interval before computing
            annualize: Whether to annualize
            periods_per_year: Seconds per year for annualization
        """
        self.sample_interval_ms = sample_interval_ms
        self.annualize = annualize
        self.periods_per_year = periods_per_year
    
    def _resample(
        self,
        df: pd.DataFrame,
        ts_col: str,
        price_col: str,
    ) -> pd.Series:
        """Resample to regular intervals."""
        df_sorted = df.sort_values(ts_col).copy()
        
        # Bucket timestamps
        df_sorted["_bucket"] = (
            df_sorted[ts_col] // self.sample_interval_ms
        ) * self.sample_interval_ms
        
        # Take last price in each bucket
        resampled = df_sorted.groupby("_bucket")[price_col].last()
        
        return resampled
    
    def compute(
        self,
        df: pd.DataFrame,
        ts_col: str = "ts_recv",
        price_col: str = "price",
    ) -> float:
        """Compute realized volatility for the entire DataFrame."""
        if df.empty:
            return np.nan
        
        prices = self._resample(df, ts_col, price_col)
        
        if len(prices) < 2:
            return np.nan
        
        returns = np.log(prices / prices.shift(1)).dropna()
        vol = returns.std()
        
        if self.annualize:
            # Annualize based on sample interval
            samples_per_second = 1000 / self.sample_interval_ms
            vol = vol * np.sqrt(samples_per_second * self.periods_per_year)
        
        return float(vol)
    
    def compute_rolling(
        self,
        df: pd.DataFrame,
        window_seconds: int,
        ts_col: str = "ts_recv",
        price_col: str = "price",
    ) -> pd.Series:
        """Compute rolling realized volatility."""
        if df.empty:
            return pd.Series(dtype=float)
        
        prices = self._resample(df, ts_col, price_col)
        
        # Window size in samples
        samples_per_second = 1000 / self.sample_interval_ms
        window_samples = int(window_seconds * samples_per_second)
        
        returns = np.log(prices / prices.shift(1))
        vol = returns.rolling(window_samples).std()
        
        if self.annualize:
            vol = vol * np.sqrt(samples_per_second * self.periods_per_year)
        
        return vol


class TradeBasedVol(RealizedVolEstimator):
    """Volatility estimator based on trade-by-trade returns."""
    
    def __init__(
        self,
        annualize: bool = True,
        min_trades: int = 10,
    ):
        """
        Args:
            annualize: Whether to annualize
            min_trades: Minimum trades required for estimate
        """
        self.annualize = annualize
        self.min_trades = min_trades
    
    def compute(
        self,
        df: pd.DataFrame,
        ts_col: str = "ts_recv",
        price_col: str = "price",
    ) -> float:
        """Compute trade-based realized volatility."""
        if len(df) < self.min_trades:
            return np.nan
        
        df_sorted = df.sort_values(ts_col)
        prices = df_sorted[price_col]
        timestamps = df_sorted[ts_col]
        
        returns = np.log(prices / prices.shift(1)).dropna()
        
        # Time between trades in seconds
        time_diffs = (timestamps - timestamps.shift(1)).dropna() / 1000
        
        if len(returns) == 0 or len(time_diffs) == 0:
            return np.nan
        
        # Variance per second
        total_time_sec = (timestamps.iloc[-1] - timestamps.iloc[0]) / 1000
        total_variance = (returns ** 2).sum()
        
        if total_time_sec <= 0:
            return np.nan
        
        variance_per_sec = total_variance / total_time_sec
        vol_per_sec = np.sqrt(variance_per_sec)
        
        if self.annualize:
            vol_per_sec = vol_per_sec * np.sqrt(365 * 24 * 3600)
        
        return float(vol_per_sec)
    
    def compute_rolling(
        self,
        df: pd.DataFrame,
        window_seconds: int,
        ts_col: str = "ts_recv",
        price_col: str = "price",
    ) -> pd.Series:
        """Compute rolling trade-based volatility."""
        if df.empty:
            return pd.Series(dtype=float)
        
        df_sorted = df.sort_values(ts_col).reset_index(drop=True)
        
        vols = []
        timestamps_out = []
        
        window_ms = window_seconds * 1000
        
        for idx in range(len(df_sorted)):
            current_ts = df_sorted.loc[idx, ts_col]
            window_start = current_ts - window_ms
            
            # Get window data
            mask = (df_sorted[ts_col] >= window_start) & (df_sorted[ts_col] <= current_ts)
            window_df = df_sorted[mask]
            
            if len(window_df) >= self.min_trades:
                vol = self.compute(window_df, ts_col, price_col)
            else:
                vol = np.nan
            
            vols.append(vol)
            timestamps_out.append(current_ts)
        
        return pd.Series(vols, index=timestamps_out)
