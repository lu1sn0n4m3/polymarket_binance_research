"""Build hourly contract labels from Binance trades.

K = price of first trade with ts_event >= hour_start
S_T = price of last trade with ts_event < hour_end
Y = 1{S_T > K}

Includes incremental daily caching and retry logic for trades data.
"""

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.data.loaders import load_binance_trades


def _load_trades_single_day(
    date: datetime,
    asset: str,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> pd.DataFrame:
    """Load a single day of trades data with retry logic."""
    start_dt = datetime.combine(date.date(), datetime.min.time(), tzinfo=timezone.utc)
    end_dt = start_dt + timedelta(days=1)

    for attempt in range(max_retries):
        try:
            df = load_binance_trades(start_dt, end_dt, asset=asset)
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Retry {attempt + 1}/{max_retries} after error: {e}")
                time.sleep(retry_delay * (attempt + 1))
            else:
                raise


def _load_trades_cached(
    start_dt: datetime,
    end_dt: datetime,
    asset: str = "BTC",
    cache_dir: Path | str = "pricer_calibration/output/trades_daily",
) -> pd.DataFrame:
    """Load trades with incremental daily caching."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Generate list of dates
    dates = []
    current = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    while current < end_dt:
        dates.append(current)
        current += timedelta(days=1)

    dfs = []
    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        cache_file = cache_dir / f"{asset}_trades_{date_str}.parquet"

        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            print(f"    {date_str}: loaded from cache ({len(df):,} trades)")
        else:
            print(f"    {date_str}: fetching from S3...", end=" ", flush=True)
            try:
                df = _load_trades_single_day(date, asset)
                if not df.empty:
                    df = df[["ts_event", "price"]].copy()
                df.to_parquet(cache_file, index=False)
                print(f"({len(df):,} trades, cached)")
            except Exception as e:
                print(f"FAILED: {e}")
                continue

        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=["ts_event", "price"])

    result = pd.concat(dfs, ignore_index=True)
    result = result.sort_values("ts_event").reset_index(drop=True)
    return result


def build_hourly_labels(
    start_dt: datetime,
    end_dt: datetime,
    asset: str = "BTC",
) -> pd.DataFrame:
    """Build contract labels for each complete hour in the range.

    Uses incremental daily caching for trades data.

    Args:
        start_dt: Start datetime (UTC, timezone-aware).
        end_dt: End datetime (UTC, timezone-aware).
        asset: "BTC".

    Returns:
        DataFrame with columns:
        [market_id, hour_start_ms, hour_end_ms, K, S_T, Y]
        where hour_start/end are in epoch ms.
    """
    print("  Loading trades with daily caching...")
    trades = _load_trades_cached(start_dt, end_dt, asset=asset)

    if trades.empty:
        return pd.DataFrame(
            columns=["market_id", "hour_start_ms", "hour_end_ms", "K", "S_T", "Y"]
        )

    trades = trades.sort_values("ts_event").reset_index(drop=True)

    # Iterate over each complete hour
    current = start_dt.replace(minute=0, second=0, microsecond=0)
    if current < start_dt:
        current += timedelta(hours=1)

    rows = []
    while current + timedelta(hours=1) <= end_dt:
        hour_start_ms = int(current.timestamp() * 1000)
        hour_end_ms = int((current + timedelta(hours=1)).timestamp() * 1000)

        # First trade at or after hour_start
        open_mask = trades["ts_event"] >= hour_start_ms
        # Last trade strictly before hour_end
        close_mask = trades["ts_event"] < hour_end_ms

        open_trades = trades[open_mask & close_mask]

        if len(open_trades) >= 2:
            K = float(open_trades.iloc[0]["price"])
            S_T = float(open_trades.iloc[-1]["price"])
            Y = 1 if S_T > K else 0

            market_id = f"{asset}_{current.strftime('%Y%m%d_%H')}"
            rows.append({
                "market_id": market_id,
                "hour_start_ms": hour_start_ms,
                "hour_end_ms": hour_end_ms,
                "K": K,
                "S_T": S_T,
                "Y": Y,
            })

        current += timedelta(hours=1)

    return pd.DataFrame(rows)
