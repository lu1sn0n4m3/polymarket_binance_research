"""HourlyMarketSession - the core abstraction for analyzing hourly markets."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, date
from typing import Literal
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

from src.config import get_config
from src.data.connection import get_connection
from src.data.loaders import (
    load_binance_bbo,
    load_binance_trades,
    load_polymarket_bbo,
    load_polymarket_trades,
    load_polymarket_book,
    get_unique_token_ids,
)
from src.data.alignment import align_asof, compute_derived_fields


ET = ZoneInfo("America/New_York")
UTC = timezone.utc


def et_to_utc(dt: datetime) -> datetime:
    """Convert Eastern Time datetime to UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ET)
    return dt.astimezone(UTC)


def utc_to_et(dt: datetime) -> datetime:
    """Convert UTC datetime to Eastern Time."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(ET)


@dataclass
class MarketOutcome:
    """Outcome of an hourly market based on Binance trades."""
    
    open_price: float
    close_price: float
    open_ts: int  # ms
    close_ts: int  # ms
    
    @property
    def is_up(self) -> bool:
        """True if close > open."""
        return self.close_price > self.open_price
    
    @property
    def is_down(self) -> bool:
        """True if close < open."""
        return self.close_price < self.open_price
    
    @property
    def return_pct(self) -> float:
        """Percentage return from open to close."""
        return (self.close_price - self.open_price) / self.open_price * 100
    
    @property
    def outcome(self) -> Literal["up", "down", "flat"]:
        """String outcome."""
        if self.is_up:
            return "up"
        elif self.is_down:
            return "down"
        return "flat"


@dataclass
class HourlyMarketSession:
    """A single hourly Polymarket binary option market with aligned Binance data.
    
    This is the core abstraction for analyzing hourly markets. It bundles:
    - Polymarket BBO, trades, and order book data
    - Binance BBO and trades data
    - Aligned time series for analysis
    - Derived quantities (mid, spread, microprice)
    - Market outcome based on Binance open/close
    
    Attributes:
        asset: "BTC" or "ETH"
        market_date: UTC date of the market
        hour_et: Hour in Eastern Time (0-23)
        lookback_hours: Hours of Binance data before market open for volatility
        
    Time Properties:
        utc_start: Market start in UTC
        utc_end: Market end in UTC
        lookback_start: Start of lookback window in UTC
    """
    
    asset: Literal["BTC", "ETH"]
    market_date: date
    hour_et: int
    lookback_hours: int = 3
    
    # Internal state
    _conn: duckdb.DuckDBPyConnection | None = field(default=None, repr=False)
    _primary_token_id: str | None = field(default=None, repr=False)
    _token_is_up: bool | None = field(default=None, repr=False)  # Inferred from resolution
    
    # Cached data
    _polymarket_bbo: pd.DataFrame | None = field(default=None, repr=False)
    _polymarket_trades: pd.DataFrame | None = field(default=None, repr=False)
    _polymarket_book: pd.DataFrame | None = field(default=None, repr=False)
    _binance_bbo: pd.DataFrame | None = field(default=None, repr=False)
    _binance_trades: pd.DataFrame | None = field(default=None, repr=False)
    _binance_lookback_bbo: pd.DataFrame | None = field(default=None, repr=False)
    _binance_lookback_trades: pd.DataFrame | None = field(default=None, repr=False)
    _aligned: pd.DataFrame | None = field(default=None, repr=False)
    _outcome: MarketOutcome | None = field(default=None, repr=False)
    
    def __post_init__(self):
        if self.hour_et < 0 or self.hour_et > 23:
            raise ValueError(f"hour_et must be 0-23, got {self.hour_et}")
    
    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Get DuckDB connection."""
        if self._conn is None:
            self._conn = get_connection()
        return self._conn
    
    @property
    def utc_start(self) -> datetime:
        """Market start time in UTC."""
        # Construct ET datetime, then convert to UTC
        et_dt = datetime(
            self.market_date.year,
            self.market_date.month,
            self.market_date.day,
            self.hour_et,
            0,
            0,
            tzinfo=ET,
        )
        return et_to_utc(et_dt)
    
    @property
    def utc_end(self) -> datetime:
        """Market end time in UTC (exclusive)."""
        return self.utc_start + timedelta(hours=1)
    
    @property
    def lookback_start(self) -> datetime:
        """Start of lookback window in UTC."""
        return self.utc_start - timedelta(hours=self.lookback_hours)
    
    @property
    def primary_token_id(self) -> str | None:
        """Primary token ID we're analyzing (first alphabetically, may be Up or Down)."""
        if self._primary_token_id is None:
            tokens = get_unique_token_ids(
                self.utc_start,
                self.utc_end,
                self.asset,
                self.conn,
            )
            if tokens:
                self._primary_token_id = tokens[0]
        return self._primary_token_id
    
    @property
    def token_is_up(self) -> bool | None:
        """Whether the primary token represents 'Up' outcome (inferred from resolution).
        
        Returns None if we can't determine (e.g., no outcome data or flat market).
        """
        if self._token_is_up is not None:
            return self._token_is_up
        
        # Need outcome and polymarket data to infer
        outcome = self.outcome
        pm_bbo = self.polymarket_bbo
        
        if outcome is None or pm_bbo.empty:
            return None
        
        # Get final Polymarket price (last mid price)
        final_mid = (pm_bbo.iloc[-1]["bid_px"] + pm_bbo.iloc[-1]["ask_px"]) / 2
        
        # Infer based on resolution:
        # - If BTC went UP and PM price → 1, token is "Up"
        # - If BTC went UP and PM price → 0, token is "Down"
        # - If BTC went DOWN and PM price → 1, token is "Down"
        # - If BTC went DOWN and PM price → 0, token is "Up"
        price_went_high = final_mid > 0.5
        
        if outcome.outcome == "up":
            self._token_is_up = price_went_high
        elif outcome.outcome == "down":
            self._token_is_up = not price_went_high
        else:
            # Flat market - can't determine
            return None
        
        return self._token_is_up
    
    # Backward compatibility alias
    @property
    def up_token_id(self) -> str | None:
        """Deprecated: Use primary_token_id instead."""
        return self.primary_token_id
    
    # -------------------------------------------------------------------------
    # Data Loading Properties (lazy)
    # -------------------------------------------------------------------------
    
    @property
    def polymarket_bbo(self) -> pd.DataFrame:
        """Polymarket BBO data for the market hour (primary token)."""
        if self._polymarket_bbo is None:
            self._polymarket_bbo = load_polymarket_bbo(
                self.utc_start,
                self.utc_end,
                self.asset,
                token_id_prefix=self.primary_token_id[:6] if self.primary_token_id else None,
                conn=self.conn,
            )
        return self._polymarket_bbo
    
    @property
    def polymarket_trades(self) -> pd.DataFrame:
        """Polymarket trades for the market hour (primary token)."""
        if self._polymarket_trades is None:
            self._polymarket_trades = load_polymarket_trades(
                self.utc_start,
                self.utc_end,
                self.asset,
                token_id_prefix=self.primary_token_id[:6] if self.primary_token_id else None,
                conn=self.conn,
            )
        return self._polymarket_trades
    
    @property
    def polymarket_book(self) -> pd.DataFrame:
        """Polymarket L2 order book snapshots for the market hour (primary token)."""
        if self._polymarket_book is None:
            self._polymarket_book = load_polymarket_book(
                self.utc_start,
                self.utc_end,
                self.asset,
                token_id_prefix=self.primary_token_id[:6] if self.primary_token_id else None,
                conn=self.conn,
            )
        return self._polymarket_book
    
    @property
    def binance_bbo(self) -> pd.DataFrame:
        """Binance BBO data for the market hour."""
        if self._binance_bbo is None:
            self._binance_bbo = load_binance_bbo(
                self.utc_start,
                self.utc_end,
                self.asset,
                self.conn,
            )
        return self._binance_bbo
    
    @property
    def binance_trades(self) -> pd.DataFrame:
        """Binance trades for the market hour."""
        if self._binance_trades is None:
            self._binance_trades = load_binance_trades(
                self.utc_start,
                self.utc_end,
                self.asset,
                self.conn,
            )
        return self._binance_trades
    
    @property
    def binance_lookback_bbo(self) -> pd.DataFrame:
        """Binance BBO data including lookback period (for volatility estimation)."""
        if self._binance_lookback_bbo is None:
            self._binance_lookback_bbo = load_binance_bbo(
                self.lookback_start,
                self.utc_end,
                self.asset,
                self.conn,
            )
        return self._binance_lookback_bbo
    
    @property
    def binance_lookback_trades(self) -> pd.DataFrame:
        """Binance trades including lookback period (for volatility estimation)."""
        if self._binance_lookback_trades is None:
            self._binance_lookback_trades = load_binance_trades(
                self.lookback_start,
                self.utc_end,
                self.asset,
                self.conn,
            )
        return self._binance_lookback_trades
    
    # -------------------------------------------------------------------------
    # Derived Data
    # -------------------------------------------------------------------------
    
    @property
    def aligned(self) -> pd.DataFrame:
        """Aligned DataFrame with Polymarket and Binance data joined on ts_recv.
        
        **Important:** Polymarket prices are normalized to always represent the
        "Up" probability. If the primary token was "Down", prices are flipped (1 - price).
        
        Columns:
        - ts_recv: Receive timestamp (ms)
        - pm_bid, pm_ask, pm_bid_sz, pm_ask_sz: Polymarket BBO (normalized to Up)
        - pm_mid, pm_spread, pm_microprice: Derived Polymarket fields
        - bnc_bid, bnc_ask, bnc_bid_sz, bnc_ask_sz: Binance BBO (ASOF matched)
        - bnc_mid, bnc_spread: Derived Binance fields
        """
        if self._aligned is None:
            self._aligned = self._build_aligned()
        return self._aligned
    
    def _build_aligned(self) -> pd.DataFrame:
        """Build the aligned DataFrame."""
        pm = self.polymarket_bbo.copy()
        bnc = self.binance_bbo.copy()
        
        if pm.empty:
            return pd.DataFrame()
        
        # Normalize to "Up" probability if we loaded the "Down" token
        # This ensures pm_bid/ask always represent P(Up)
        token_is_up = self.token_is_up
        if token_is_up is False:
            # Flip prices: Up_bid = 1 - Down_ask, Up_ask = 1 - Down_bid
            pm_bid_new = 1.0 - pm["ask_px"]
            pm_ask_new = 1.0 - pm["bid_px"]
            # Sizes swap too
            pm_bid_sz_new = pm["ask_sz"]
            pm_ask_sz_new = pm["bid_sz"]
            
            pm["bid_px"] = pm_bid_new
            pm["ask_px"] = pm_ask_new
            pm["bid_sz"] = pm_bid_sz_new
            pm["ask_sz"] = pm_ask_sz_new
        
        # Rename Polymarket columns
        pm = pm.rename(columns={
            "bid_px": "pm_bid",
            "ask_px": "pm_ask",
            "bid_sz": "pm_bid_sz",
            "ask_sz": "pm_ask_sz",
        })
        
        # Add derived fields
        pm = compute_derived_fields(pm, prefix="pm_")
        
        if bnc.empty:
            return pm
        
        # Rename Binance columns
        bnc = bnc.rename(columns={
            "bid_px": "bnc_bid",
            "ask_px": "bnc_ask",
            "bid_sz": "bnc_bid_sz",
            "ask_sz": "bnc_ask_sz",
        })
        
        # Add derived fields
        bnc = compute_derived_fields(bnc, prefix="bnc_")
        
        # Keep only relevant columns for join
        bnc_cols = ["ts_recv", "bnc_bid", "bnc_ask", "bnc_bid_sz", "bnc_ask_sz", 
                    "bnc_mid", "bnc_spread"]
        bnc_subset = bnc[[c for c in bnc_cols if c in bnc.columns]]
        
        # ASOF join: for each Polymarket update, get latest Binance state
        aligned = align_asof(
            left=pm,
            right=bnc_subset,
            left_ts_col="ts_recv",
            right_ts_col="ts_recv",
            direction="backward",
        )
        
        # Clean up duplicate ts_recv columns
        if "ts_recv_left" in aligned.columns:
            aligned = aligned.drop(columns=["ts_recv_right"], errors="ignore")
            aligned = aligned.rename(columns={"ts_recv_left": "ts_recv"})
        
        return aligned
    
    @property
    def outcome(self) -> MarketOutcome | None:
        """Market outcome based on Binance first/last trade in the hour."""
        if self._outcome is None:
            self._outcome = self._compute_outcome()
        return self._outcome
    
    def _compute_outcome(self) -> MarketOutcome | None:
        """Compute market outcome from Binance trades."""
        trades = self.binance_trades
        
        if trades.empty:
            return None
        
        # First and last trade in the hour
        first_trade = trades.iloc[0]
        last_trade = trades.iloc[-1]
        
        return MarketOutcome(
            open_price=float(first_trade["price"]),
            close_price=float(last_trade["price"]),
            open_ts=int(first_trade["ts_recv"]),
            close_ts=int(last_trade["ts_recv"]),
        )
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    def get_binance_at(self, ts_ms: int) -> pd.Series | None:
        """Get Binance BBO state at a specific timestamp.
        
        Args:
            ts_ms: Timestamp in milliseconds
            
        Returns:
            Series with Binance BBO at that time, or None if no data
        """
        bbo = self.binance_bbo
        if bbo.empty:
            return None
        
        # Find latest BBO before or at ts_ms
        mask = bbo["ts_recv"] <= ts_ms
        if not mask.any():
            return None
        
        return bbo[mask].iloc[-1]
    
    def get_polymarket_at(self, ts_ms: int) -> pd.Series | None:
        """Get Polymarket BBO state at a specific timestamp.
        
        Args:
            ts_ms: Timestamp in milliseconds
            
        Returns:
            Series with Polymarket BBO at that time, or None if no data
        """
        bbo = self.polymarket_bbo
        if bbo.empty:
            return None
        
        # Find latest BBO before or at ts_ms
        mask = bbo["ts_recv"] <= ts_ms
        if not mask.any():
            return None
        
        return bbo[mask].iloc[-1]
    
    def time_to_expiry(self, ts_ms: int) -> float:
        """Get time to market close in seconds.
        
        Args:
            ts_ms: Current timestamp in milliseconds
            
        Returns:
            Seconds until market close
        """
        end_ms = int(self.utc_end.timestamp() * 1000)
        return max(0, (end_ms - ts_ms) / 1000)
    
    def elapsed_time(self, ts_ms: int) -> float:
        """Get time since market open in seconds.
        
        Args:
            ts_ms: Current timestamp in milliseconds
            
        Returns:
            Seconds since market open
        """
        start_ms = int(self.utc_start.timestamp() * 1000)
        return max(0, (ts_ms - start_ms) / 1000)
    
    def clear_cache(self) -> None:
        """Clear all cached data to free memory."""
        self._polymarket_bbo = None
        self._polymarket_trades = None
        self._polymarket_bbo = None
        self._polymarket_trades = None
        self._polymarket_book = None
        self._binance_bbo = None
        self._binance_trades = None
        self._binance_lookback_bbo = None
        self._binance_lookback_trades = None
        self._aligned = None
        self._outcome = None
        self._primary_token_id = None
        self._token_is_up = None
    
    def __repr__(self) -> str:
        return (
            f"HourlyMarketSession(asset={self.asset!r}, "
            f"date={self.market_date}, hour_et={self.hour_et}, "
            f"utc={self.utc_start.strftime('%Y-%m-%d %H:%M')} UTC)"
        )


def load_session(
    asset: Literal["BTC", "ETH"],
    market_date: date,
    hour_et: int,
    lookback_hours: int = 3,
    preload: bool = True,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> HourlyMarketSession:
    """Load a HourlyMarketSession with optional preloading.
    
    Args:
        asset: "BTC" or "ETH"
        market_date: Date of the market (UTC)
        hour_et: Hour in Eastern Time (0-23)
        lookback_hours: Hours of Binance data before market for volatility
        preload: If True, load aligned data immediately
        conn: DuckDB connection (uses global if not provided)
        
    Returns:
        HourlyMarketSession instance
        
    Example:
        >>> session = load_session("BTC", date(2026, 1, 18), hour_et=9)
        >>> df = session.aligned
        >>> outcome = session.outcome
    """
    session = HourlyMarketSession(
        asset=asset,
        market_date=market_date,
        hour_et=hour_et,
        lookback_hours=lookback_hours,
        _conn=conn,
    )
    
    if preload:
        # Trigger loading of aligned data
        _ = session.aligned
    
    return session


def load_sessions_range(
    asset: Literal["BTC", "ETH"],
    start_date: date,
    end_date: date,
    hours_et: list[int] | None = None,
    lookback_hours: int = 3,
    preload: bool = False,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> list[HourlyMarketSession]:
    """Load multiple sessions for a date range.
    
    Args:
        asset: "BTC" or "ETH"
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        hours_et: List of hours to load (default: all 24 hours)
        lookback_hours: Hours of Binance data before each market
        preload: If True, load aligned data for all sessions
        conn: DuckDB connection
        
    Returns:
        List of HourlyMarketSession instances
    """
    if hours_et is None:
        hours_et = list(range(24))
    
    if conn is None:
        conn = get_connection()
    
    sessions = []
    current_date = start_date
    
    while current_date <= end_date:
        for hour in hours_et:
            session = HourlyMarketSession(
                asset=asset,
                market_date=current_date,
                hour_et=hour,
                lookback_hours=lookback_hours,
                _conn=conn,
            )
            if preload:
                _ = session.aligned
            sessions.append(session)
        
        current_date += timedelta(days=1)
    
    return sessions
