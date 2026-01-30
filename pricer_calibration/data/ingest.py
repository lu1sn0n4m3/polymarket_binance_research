"""Load and clean Binance BBO data from S3 with incremental daily caching.

Caches data per day so only missing days are fetched on subsequent runs.
Includes retry logic for transient S3 connection errors.
"""

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.data.loaders import load_binance_bbo


def _load_single_day(
    date: datetime,
    asset: str,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> pd.DataFrame:
    """Load a single day of BBO data with retry logic."""
    start_dt = datetime.combine(date.date(), datetime.min.time(), tzinfo=timezone.utc)
    end_dt = start_dt + timedelta(days=1)

    for attempt in range(max_retries):
        try:
            df = load_binance_bbo(start_dt, end_dt, asset=asset)
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Retry {attempt + 1}/{max_retries} after error: {e}")
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            else:
                raise


def _clean_bbo(df: pd.DataFrame) -> pd.DataFrame:
    """Clean BBO data: rename columns, drop invalid, compute mid."""
    if df.empty:
        return pd.DataFrame(columns=["ts_event", "bid", "ask", "mid"])

    df = df.rename(columns={"bid_px": "bid", "ask_px": "ask"})
    df = df[["ts_event", "bid", "ask"]].copy()

    # Drop invalid
    df = df.dropna(subset=["bid", "ask"])
    df = df[(df["bid"] > 0) & (df["ask"] > 0) & (df["bid"] <= df["ask"])]

    # Compute mid
    df["mid"] = (df["bid"] + df["ask"]) / 2.0

    return df


def load_binance_bbo_cached(
    start_dt: datetime,
    end_dt: datetime,
    asset: str = "BTC",
    cache_dir: Path | str = "pricer_calibration/output/bbo_daily",
) -> pd.DataFrame:
    """Load Binance BBO with incremental daily caching.

    Caches each day separately. On subsequent runs, only fetches missing days.

    Args:
        start_dt: Start datetime (UTC).
        end_dt: End datetime (UTC).
        asset: "BTC" or "ETH".
        cache_dir: Directory for daily cache files.

    Returns:
        DataFrame with columns [ts_event, bid, ask, mid], sorted by ts_event.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Generate list of dates to load
    dates = []
    current = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    while current < end_dt:
        dates.append(current)
        current += timedelta(days=1)

    dfs = []
    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        cache_file = cache_dir / f"{asset}_{date_str}.parquet"

        if cache_file.exists():
            # Load from cache
            df = pd.read_parquet(cache_file)
            print(f"  {date_str}: loaded from cache ({len(df):,} rows)")
        else:
            # Fetch from S3
            print(f"  {date_str}: fetching from S3...", end=" ", flush=True)
            try:
                df_raw = _load_single_day(date, asset)
                df = _clean_bbo(df_raw)
                # Save to cache
                df.to_parquet(cache_file, index=False)
                print(f"({len(df):,} rows, cached)")
            except Exception as e:
                print(f"FAILED: {e}")
                continue  # Skip this day, continue with others

        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=["ts_event", "bid", "ask", "mid"])

    # Combine and sort
    result = pd.concat(dfs, ignore_index=True)
    result = result.sort_values("ts_event").reset_index(drop=True)

    return result


def load_binance_bbo_clean(
    start_dt: datetime,
    end_dt: datetime,
    asset: str = "BTC",
) -> pd.DataFrame:
    """Load Binance BBO, compute mid, clean, sort by ts_event.

    This is the legacy interface. Now uses incremental daily caching.

    Args:
        start_dt: Start datetime (UTC, timezone-aware).
        end_dt: End datetime (UTC, timezone-aware).
        asset: "BTC" or "ETH".

    Returns:
        DataFrame with columns [ts_event, bid, ask, mid],
        ts_event in milliseconds, sorted ascending.
    """
    return load_binance_bbo_cached(start_dt, end_dt, asset=asset)
