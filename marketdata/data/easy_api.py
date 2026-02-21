"""
Easy-to-use API for loading resampled market data.

This module provides separate, clean interfaces for Binance (continuous time series)
and Polymarket (individual hourly markets).

Design Philosophy:
    - Binance: Flexible time ranges, column selection for efficiency
    - Polymarket: One market at a time (markets are independent)
    - Separate loading, then combine with provided utilities

Quick Examples:
    >>> from datetime import date
    >>> from marketdata.data.easy_api import load_binance, load_polymarket_market, align_timestamps
    >>>
    >>> # Load Binance data for any time range
    >>> bnc = load_binance(
    ...     start="2026-01-18 17:45:32",
    ...     end="2026-02-01 12:34:12",
    ...     asset="BTC",
    ...     interval="1s",
    ...     columns=["ts_recv", "mid", "spread"],  # Optional: select columns
    ... )
    >>>
    >>> # Load Polymarket data for ONE specific hourly market
    >>> pm = load_polymarket_market(
    ...     asset="BTC",
    ...     date=date(2026, 1, 19),
    ...     hour_et=9,  # 9am-10am ET market
    ...     interval="1s",
    ... )
    >>>
    >>> # Combine them by timestamp
    >>> combined = align_timestamps(
    ...     left=pm,
    ...     right=bnc,
    ...     method="asof_backward",  # Match PM updates to latest BNC state
    ... )

LLM Agent Usage Notes:
    1. Always load Binance and Polymarket separately
    2. Polymarket markets are hourly and independent - load one at a time
    3. Use align_timestamps() to combine by ts_recv column
    4. All Polymarket prices represent "Up" probability (automatic normalization)
    5. All functions accept string or datetime for dates/times
"""

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd

from marketdata.data.resampled_bbo import load_resampled_bbo
from marketdata.data.resampled_polymarket import load_resampled_polymarket
from marketdata.data.resampled_labels import load_resampled_labels
from marketdata.data.cache_manager import get_cache_info as _get_cache_info, clear_cache as _clear_cache
from marketdata.data.alignment import align_asof


# ============================================================================
# BINANCE DATA LOADING (Continuous Time Series)
# ============================================================================

def load_binance(
    start: str | datetime,
    end: str | datetime,
    asset: Literal["BTC", "ETH"] = "BTC",
    interval: Literal["100ms", "500ms", "1s", "5s"] = "1s",
    columns: list[str] | None = None,
    cache_dir: Path | str | None = None,
    force_reload: bool = False,
) -> pd.DataFrame:
    """Load Binance BBO data for arbitrary time range with optional column selection.

    Use this for:
    - Continuous price history across multiple days
    - Volatility analysis
    - Bitcoin/Ethereum price movements
    - Any analysis NOT specific to Polymarket markets

    Args:
        start: Start time as "YYYY-MM-DD HH:MM:SS" or datetime object (UTC)
        end: End time as string or datetime (UTC)
        asset: "BTC" or "ETH"
        interval: Resampling interval - determines granularity
            - "500ms": High frequency (120 rows/min)
            - "1s": Standard (60 rows/min)
            - "5s": Low frequency (12 rows/min)
        columns: Optional list of columns to return. Available columns:
            - "ts_recv": Timestamp (always included)
            - "bid_px", "ask_px": Best bid/ask prices (USD)
            - "bid_sz", "ask_sz": Best bid/ask sizes
            - "mid_px": Mid price (bid + ask) / 2
            - "spread": Spread (ask - bid)
            If None, returns all columns.
        cache_dir: Override default cache directory
        force_reload: If True, ignore cache and reload from S3

    Returns:
        DataFrame with selected columns. Always includes ts_recv.
        Sorted by ts_recv (ascending).

    Example - Load price history:
        >>> bnc = load_binance(
        ...     start="2026-01-15 09:00:00",
        ...     end="2026-01-20 17:00:00",
        ...     asset="BTC",
        ...     interval="1s",
        ...     columns=["ts_recv", "mid_px", "spread"],
        ... )
        >>> print(f"Loaded {len(bnc)} rows")
        >>> print(f"Price range: ${bnc['mid_px'].min():.2f} - ${bnc['mid_px'].max():.2f}")

    Example - Compute volatility:
        >>> bnc = load_binance(
        ...     start="2026-01-19 00:00:00",
        ...     end="2026-01-20 00:00:00",
        ...     asset="BTC",
        ...     interval="1s",
        ...     columns=["ts_recv", "mid_px"],
        ... )
        >>> returns = bnc["mid_px"].pct_change()
        >>> import numpy as np
        >>> volatility = returns.std() * np.sqrt(86400)  # Annualized

    LLM Agent Notes:
        - Accepts arbitrary time ranges (not limited to hour boundaries)
        - Use column selection to reduce memory when only need specific fields
        - All times in UTC
        - Returns empty DataFrame if no data available (no error raised)
    """
    # Parse times
    start_dt = _parse_datetime(start)
    end_dt = _parse_datetime(end)

    # Convert interval to milliseconds
    interval_ms = _parse_interval(interval)

    # Load resampled data
    df = load_resampled_bbo(
        start_dt=start_dt,
        end_dt=end_dt,
        interval_ms=interval_ms,
        asset=asset,
        cache_dir=Path(cache_dir) if cache_dir else None,
        force_reload=force_reload,
    )

    # Apply column selection
    if columns is not None and not df.empty:
        # Ensure ts_recv is always included
        if "ts_recv" not in columns:
            columns = ["ts_recv"] + columns

        # Filter to requested columns
        available_cols = [col for col in columns if col in df.columns]
        df = df[available_cols]

    return df


# ============================================================================
# BINANCE HOURLY LABELS (Opening/Closing Prices + Outcome)
# ============================================================================

def load_binance_labels(
    start: str | datetime,
    end: str | datetime,
    asset: Literal["BTC", "ETH"] = "BTC",
    cache_dir: Path | str | None = None,
    force_reload: bool = False,
) -> pd.DataFrame:
    """Load hourly market labels (open/close/outcome) from Binance trades.

    Returns one row per UTC hour with the first and last Binance trade prices.
    Data is cached as lightweight daily parquet files (~24 rows per day).

    Args:
        start: Start time as "YYYY-MM-DD HH:MM:SS" or datetime (UTC)
        end: End time (UTC)
        asset: "BTC" or "ETH"
        cache_dir: Override default cache directory
        force_reload: If True, ignore cache and reload from S3

    Returns:
        DataFrame with columns:
            - hour_start_ms: Hour start (epoch ms)
            - hour_end_ms: Hour end (epoch ms)
            - K: Opening price (first trade of the hour)
            - S_T: Closing price (last trade of the hour)
            - Y: 1 if S_T > K, else 0

    Example:
        >>> labels = load_binance_labels("2026-01-19", "2026-01-20", "BTC")
        >>> print(f"{len(labels)} hours labeled")
        >>> print(f"Up hours: {labels['Y'].sum()}, Down hours: {(1-labels['Y']).sum()}")
    """
    start_dt = _parse_datetime(start)
    end_dt = _parse_datetime(end)

    return load_resampled_labels(
        start_dt=start_dt,
        end_dt=end_dt,
        asset=asset,
        cache_dir=Path(cache_dir) if cache_dir else None,
        force_reload=force_reload,
    )


# ============================================================================
# POLYMARKET DATA LOADING (Individual Hourly Markets)
# ============================================================================

def load_polymarket_market(
    asset: Literal["BTC", "ETH"],
    date: str | date | datetime,
    hour_et: int,
    interval: Literal["100ms", "500ms", "1s", "5s"] = "1s",
    cache_dir: Path | str | None = None,
    force_reload: bool = False,
) -> pd.DataFrame:
    """Load Polymarket data for ONE specific hourly market.

    IMPORTANT: Polymarket markets are hourly and independent.
    This function loads exactly ONE market (one hour). To analyze multiple
    markets, call this function multiple times.

    Each market asks: "Will {asset} close this hour higher than it opened?"
    - Pays $1 if yes (Up)
    - Pays $0 if no (Down)

    Args:
        asset: "BTC" or "ETH"
        date: Market date as "YYYY-MM-DD", date object, or datetime
            Examples: "2026-01-19", date(2026, 1, 19), datetime(2026, 1, 19)
        hour_et: Hour in Eastern Time (0-23)
            Example: hour_et=9 means 9:00am-10:00am ET market
        interval: Resampling interval (same as Binance)
        cache_dir: Override default cache directory
        force_reload: If True, ignore cache and reload from S3

    Returns:
        DataFrame with columns:
            - ts_recv: Timestamp (epoch milliseconds)
            - bid, ask: Best bid/ask prices (0-1 probability scale)
            - bid_sz, ask_sz: Best bid/ask sizes
            - mid: Mid price (Up probability)
            - spread: Bid-ask spread
            - microprice: Size-weighted mid price

        IMPORTANT: All prices represent "Up" probability (automatic normalization).
        If market resolves Up, mid price → 1.0 at expiry.
        If market resolves Down, mid price → 0.0 at expiry.

    Example - Load single market:
        >>> pm = load_polymarket_market(
        ...     asset="BTC",
        ...     date="2026-01-19",
        ...     hour_et=9,  # 9am-10am ET
        ...     interval="1s",
        ... )
        >>> print(f"Market duration: {len(pm)} seconds")
        >>> print(f"Opening Up probability: {pm['mid'].iloc[0]:.3f}")
        >>> print(f"Closing Up probability: {pm['mid'].iloc[-1]:.3f}")

    Example - Analyze multiple markets:
        >>> markets = []
        >>> for hour in [9, 10, 11, 14, 15]:
        ...     pm = load_polymarket_market("BTC", "2026-01-19", hour, "1s")
        ...     if not pm.empty:
        ...         markets.append(pm)
        >>> # Each market is independent, analyze separately

    LLM Agent Notes:
        - Load ONE market at a time (hour_et specifies which hour)
        - Markets are independent - don't concatenate across hours
        - Eastern Time (ET) used for market hours (NYSE hours)
        - Returns empty DataFrame if market doesn't exist for that hour
        - Prices always represent "Up" probability (framework handles normalization)
    """
    # Parse date
    if isinstance(date, str):
        market_date = datetime.strptime(date, "%Y-%m-%d").date()
    elif isinstance(date, datetime):
        market_date = date.date()
    else:
        market_date = date

    # Convert to datetime for market hour boundaries
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")

    # Create market hour boundaries in ET, then convert to UTC
    et_start = datetime(market_date.year, market_date.month, market_date.day, hour_et, 0, 0, tzinfo=ET)
    et_end = et_start.replace(hour=hour_et + 1 if hour_et < 23 else 0)
    if hour_et == 23:
        from datetime import timedelta
        et_end = et_end + timedelta(days=1)

    # Convert to UTC
    utc_start = et_start.astimezone(timezone.utc)
    utc_end = et_end.astimezone(timezone.utc)

    # Convert interval to milliseconds
    interval_ms = _parse_interval(interval)

    # Load resampled data for this specific hour
    df = load_resampled_polymarket(
        start_dt=utc_start,
        end_dt=utc_end,
        interval_ms=interval_ms,
        asset=asset,
        cache_dir=Path(cache_dir) if cache_dir else None,
        force_reload=force_reload,
    )

    if df.empty:
        return df

    # --- Normalize to "Up" probability ---
    # The cache stores the "primary" token (alphabetically first), which may
    # be the Up or Down token.  We use the labels cache (Binance open/close)
    # to determine the market outcome, then compare with the terminal PM
    # price to decide whether a flip is needed.
    df = _normalize_to_up(df, utc_start, utc_end, asset, cache_dir)

    return df


# ============================================================================
# ALIGNMENT UTILITIES (Combine Binance + Polymarket)
# ============================================================================

def align_timestamps(
    left: pd.DataFrame,
    right: pd.DataFrame,
    method: Literal["asof_backward", "asof_forward", "inner", "outer"] = "asof_backward",
    left_suffix: str = "_left",
    right_suffix: str = "_right",
) -> pd.DataFrame:
    """Align two DataFrames by ts_recv timestamp column.

    Use this to combine Binance and Polymarket data after loading separately.

    Args:
        left: First DataFrame (must have ts_recv column)
        right: Second DataFrame (must have ts_recv column)
        method: Alignment strategy
            - "asof_backward": For each left timestamp, find latest right timestamp ≤ left
              Use when: Right is reference (e.g., left=Polymarket, right=Binance)
            - "asof_forward": For each left timestamp, find earliest right timestamp ≥ left
            - "inner": Only timestamps present in BOTH datasets
            - "outer": All timestamps from BOTH datasets (fills NaN where missing)
        left_suffix: Suffix for overlapping column names from left DataFrame
        right_suffix: Suffix for overlapping column names from right DataFrame

    Returns:
        DataFrame with aligned data, sorted by ts_recv

    Example - Typical use case (Polymarket + Binance):
        >>> # Load Polymarket market
        >>> pm = load_polymarket_market("BTC", "2026-01-19", 9, "1s")
        >>>
        >>> # Load Binance data for wider range (includes market hour + context)
        >>> bnc = load_binance(
        ...     start="2026-01-19 13:00:00",  # 9am ET = 2pm UTC (winter)
        ...     end="2026-01-19 15:00:00",    # Wider range for context
        ...     asset="BTC",
        ...     interval="1s",
        ...     columns=["ts_recv", "mid_px", "spread"],
        ... )
        >>>
        >>> # Combine: For each PM update, get latest Binance state
        >>> combined = align_timestamps(
        ...     left=pm,
        ...     right=bnc,
        ...     method="asof_backward",
        ...     left_suffix="_pm",
        ...     right_suffix="_bnc",
        ... )
        >>> # Result: PM updates with corresponding BNC prices
        >>> print(combined.columns)
        Index(['ts_recv', 'bid_pm', 'ask_pm', 'mid_pm', 'mid_px_bnc', 'spread_bnc'], dtype='object')

    Example - Inner join (only overlapping timestamps):
        >>> combined = align_timestamps(
        ...     left=pm,
        ...     right=bnc,
        ...     method="inner",
        ... )
        >>> # Only rows where both PM and BNC have data at exact timestamp

    LLM Agent Notes:
        - Both DataFrames MUST have ts_recv column
        - Method "asof_backward" is most common for PM+BNC combination
        - Returns empty DataFrame if no overlapping time range
        - All timestamps in epoch milliseconds (int64)
    """
    if left.empty or right.empty:
        return pd.DataFrame()

    if "ts_recv" not in left.columns or "ts_recv" not in right.columns:
        raise ValueError("Both DataFrames must have 'ts_recv' column")

    if method == "asof_backward":
        result = align_asof(
            left=left,
            right=right,
            left_ts_col="ts_recv",
            right_ts_col="ts_recv",
            direction="backward",
            left_suffix=left_suffix,
            right_suffix=right_suffix,
        )
    elif method == "asof_forward":
        result = align_asof(
            left=left,
            right=right,
            left_ts_col="ts_recv",
            right_ts_col="ts_recv",
            direction="forward",
            left_suffix=left_suffix,
            right_suffix=right_suffix,
        )
    elif method == "inner":
        # Inner join on exact timestamps
        result = pd.merge(
            left,
            right,
            on="ts_recv",
            how="inner",
            suffixes=(left_suffix, right_suffix),
        )
    elif method == "outer":
        # Outer join - all timestamps from both
        result = pd.merge(
            left,
            right,
            on="ts_recv",
            how="outer",
            suffixes=(left_suffix, right_suffix),
        )
        result = result.sort_values("ts_recv").reset_index(drop=True)
    else:
        raise ValueError(f"Invalid method: {method}")

    return result


# ============================================================================
# CACHE MANAGEMENT
# ============================================================================

def get_cache_status(
    venue: Literal["binance", "polymarket"],
    asset: Literal["BTC", "ETH"] = "BTC",
    interval: Literal["100ms", "500ms", "1s", "5s"] = "1s",
) -> dict:
    """Check what data is cached locally for a venue.

    Args:
        venue: "binance" or "polymarket"
        asset: "BTC" or "ETH"
        interval: Resampling interval

    Returns:
        dict with keys:
            - dates_cached: List of dates (YYYY-MM-DD strings)
            - date_range: Tuple of (first_date, last_date)
            - total_rows: Total cached rows
            - size_mb: Total cache size in megabytes

    Example:
        >>> status = get_cache_status("binance", "BTC", "1s")
        >>> print(f"Cached: {len(status['dates_cached'])} days")
        >>> print(f"Range: {status['date_range']}")
        >>> print(f"Size: {status['size_mb']:.1f} MB")
    """
    interval_ms = _parse_interval(interval)
    cache_dir = Path("data/resampled_data")

    return _get_cache_info(
        asset=asset,
        interval_ms=interval_ms,
        cache_dir=cache_dir,
        venue=venue,
    )


def clear_cache(
    venue: Literal["binance", "polymarket"],
    asset: Literal["BTC", "ETH"] = "BTC",
    interval: Literal["100ms", "500ms", "1s", "5s"] = "1s",
    before_date: str | date | None = None,
) -> int:
    """Clear cached data for a venue.

    Args:
        venue: "binance" or "polymarket"
        asset: "BTC" or "ETH"
        interval: Resampling interval
        before_date: If specified, only clear dates before this (keep recent data)

    Returns:
        Number of files deleted

    Example - Clear old data:
        >>> # Keep last 30 days, delete older
        >>> from datetime import date, timedelta
        >>> cutoff = date.today() - timedelta(days=30)
        >>> deleted = clear_cache("binance", "BTC", "1s", before_date=cutoff)
        >>> print(f"Deleted {deleted} files")
    """
    interval_ms = _parse_interval(interval)
    cache_dir = Path("data/resampled_data")

    # Convert before_date to datetime if needed
    if before_date is not None:
        if isinstance(before_date, str):
            before_dt = datetime.strptime(before_date, "%Y-%m-%d")
        elif isinstance(before_date, date):
            before_dt = datetime(before_date.year, before_date.month, before_date.day)
        else:
            before_dt = before_date
    else:
        before_dt = None

    return _clear_cache(
        asset=asset,
        interval_ms=interval_ms,
        cache_dir=cache_dir,
        venue=venue,
        before_date=before_dt,
    )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _parse_datetime(dt: str | datetime) -> datetime:
    """Parse string or datetime to timezone-aware datetime."""
    if isinstance(dt, str):
        # Try to parse string
        try:
            parsed = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except ValueError:
            # Try without time
            parsed = datetime.strptime(dt, "%Y-%m-%d")

        # Ensure UTC timezone
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    elif isinstance(dt, datetime):
        # Ensure timezone aware
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    else:
        raise ValueError(f"Invalid datetime format: {dt}")


def _parse_interval(interval: str) -> int:
    """Parse interval string to milliseconds."""
    mapping = {
        "100ms": 100,
        "500ms": 500,
        "1s": 1000,
        "5s": 5000,
    }
    if interval not in mapping:
        raise ValueError(f"Invalid interval: {interval}. Must be one of {list(mapping.keys())}")
    return mapping[interval]


def _normalize_to_up(
    df: pd.DataFrame,
    utc_start: datetime,
    utc_end: datetime,
    asset: str,
    cache_dir: "Path | str | None",
) -> pd.DataFrame:
    """Normalize Polymarket prices so they always represent Up probability.

    The resampled cache stores whichever token is alphabetically first (the
    "primary" token).  That may be Up or Down.  We detect the orientation by
    comparing the terminal PM mid price with the Binance-based market outcome
    (from the labels cache).

    Logic (mirrors HourlyMarketSession.token_is_up):
        outcome=Up  & terminal_mid > 0.5  →  token is Up   (no flip)
        outcome=Up  & terminal_mid < 0.5  →  token is Down  (flip)
        outcome=Down & terminal_mid > 0.5  →  token is Down  (flip)
        outcome=Down & terminal_mid < 0.5  →  token is Up   (no flip)
    """
    # Load the label for this hour
    try:
        labels = load_resampled_labels(
            start_dt=utc_start,
            end_dt=utc_end,
            asset=asset,
            cache_dir=Path(cache_dir) if cache_dir else None,
        )
    except Exception:
        return df  # Can't normalize without labels — return raw

    utc_start_ms = int(utc_start.timestamp() * 1000)
    if labels.empty:
        return df

    match = labels[labels["hour_start_ms"] == utc_start_ms]
    if match.empty:
        return df

    Y = int(match.iloc[0]["Y"])  # 1 = Up, 0 = Down

    # Terminal mid price of the PM data
    terminal_mid = (df["bid"].iloc[-1] + df["ask"].iloc[-1]) / 2
    price_went_high = terminal_mid > 0.5

    # Determine if token is Up
    if Y == 1:  # outcome = Up
        token_is_up = price_went_high
    else:       # outcome = Down
        token_is_up = not price_went_high

    if token_is_up:
        return df  # Already represents Up probability

    # Flip: Up_bid = 1 - Down_ask,  Up_ask = 1 - Down_bid
    df = df.copy()
    new_bid = 1.0 - df["ask"]
    new_ask = 1.0 - df["bid"]
    new_bid_sz = df["ask_sz"].copy()
    new_ask_sz = df["bid_sz"].copy()

    df["bid"] = new_bid
    df["ask"] = new_ask
    df["bid_sz"] = new_bid_sz
    df["ask_sz"] = new_ask_sz
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread"] = df["ask"] - df["bid"]
    if "microprice" in df.columns:
        total_sz = df["bid_sz"] + df["ask_sz"]
        df["microprice"] = df["bid"] + (df["ask"] - df["bid"]) * df["bid_sz"] / total_sz.clip(lower=1e-12)

    return df
