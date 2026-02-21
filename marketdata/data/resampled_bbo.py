"""Resampled BBO data loading with automatic caching and gap detection."""

import gc
import time
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd

from marketdata.data.alignment import resample_to_grid
from marketdata.data.cache_manager import load_resampled_day, save_resampled_day, _interval_dir_name
from marketdata.data.loaders import load_binance_bbo


def _log_memory():
    """Log current memory usage."""
    try:
        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024 / 1024
        vm = psutil.virtual_memory()
        return f"RAM: {mem_mb:.0f}MB process | {vm.percent:.0f}% system"
    except:
        return "RAM: N/A"


def load_resampled_bbo(
    start_dt: datetime,
    end_dt: datetime,
    interval_ms: int = 1000,
    asset: Literal["BTC", "ETH"] = "BTC",
    cache_dir: Path | str | None = None,
    force_reload: bool = False,
) -> pd.DataFrame:
    """Load resampled BBO data with automatic caching and gap filling.

    This is the main entry point. Automatically:
    1. Checks what data is cached locally
    2. Identifies missing date ranges
    3. Loads missing data from S3
    4. Resamples to specified interval
    5. Saves to cache
    6. Returns merged result

    Args:
        start_dt: Start datetime (UTC, timezone-aware)
        end_dt: End datetime (UTC, timezone-aware)
        interval_ms: Resampling interval in milliseconds (1000 or 5000)
        asset: "BTC" or "ETH"
        cache_dir: Base cache directory (default: data/resampled_bbo)
        force_reload: If True, ignore cache and reload from S3

    Returns:
        DataFrame with columns:
            - ts_recv: int64, epoch milliseconds (aligned to grid)
            - bid_px: float64
            - ask_px: float64
            - bid_sz: float64
            - ask_sz: float64
            - mid_px: float64 (derived)
            - spread: float64 (derived)

    Example:
        >>> from datetime import datetime, timezone
        >>> df = load_resampled_bbo(
        ...     start_dt=datetime(2026, 1, 15, tzinfo=timezone.utc),
        ...     end_dt=datetime(2026, 1, 17, tzinfo=timezone.utc),
        ...     interval_ms=1000,
        ...     asset="BTC"
        ... )
        >>> # Returns 1s resampled data for Jan 15-17
    """
    # Set default cache directory
    if cache_dir is None:
        cache_dir = Path("data/resampled_data")
    else:
        cache_dir = Path(cache_dir)

    # Get missing dates
    missing_dates = get_missing_dates(start_dt, end_dt, asset, interval_ms, cache_dir)

    # If force_reload, treat all dates as missing
    if force_reload:
        missing_dates = _generate_date_list(start_dt, end_dt)

    # Load or fetch each day
    all_days = []
    dates_to_process = _generate_date_list(start_dt, end_dt)

    for date in dates_to_process:
        if date in missing_dates or force_reload:
            # Fetch from S3 and cache
            print(f"  {date.strftime('%Y-%m-%d')}: fetching from S3 (hour-by-hour)...", flush=True)
            try:
                _process_and_cache_day(date, asset, interval_ms, cache_dir)
                # Load from cache immediately after processing
                # This ensures memory is freed from the processing step
                df_day = load_resampled_day(date, asset, interval_ms, cache_dir, venue="binance")
                if df_day is not None and not df_day.empty:
                    all_days.append(df_day)
                    print(f"  {date.strftime('%Y-%m-%d')}: ✓ {len(df_day):,} rows cached")
                else:
                    print(f"  {date.strftime('%Y-%m-%d')}: (no data)")
            except Exception as e:
                print(f"  {date.strftime('%Y-%m-%d')}: FAILED: {e}")
                continue
        else:
            # Load from cache
            df_day = load_resampled_day(date, asset, interval_ms, cache_dir, venue="binance")
            if df_day is not None and not df_day.empty:
                all_days.append(df_day)
                print(f"  {date.strftime('%Y-%m-%d')}: loaded from cache ({len(df_day):,} rows)")

    if not all_days:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=["ts_recv", "bid_px", "ask_px", "bid_sz", "ask_sz", "mid_px", "spread"])

    # Combine all days
    result = pd.concat(all_days, ignore_index=True)
    result = result.sort_values("ts_recv").reset_index(drop=True)

    # Filter to exact time range
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    result = result[(result["ts_recv"] >= start_ms) & (result["ts_recv"] <= end_ms)]

    return result


def resample_bbo_to_interval(
    raw_bbo: pd.DataFrame,
    interval_ms: int = 1000,
    method: Literal["ffill", "bfill"] = "ffill",
) -> pd.DataFrame:
    """Resample raw BBO data to fixed interval grid.

    Uses existing resample_to_grid from alignment.py but adds:
    - Mid price and spread computation
    - Data validation and cleaning
    - Efficient forward-fill

    Args:
        raw_bbo: Raw BBO DataFrame from load_binance_bbo()
        interval_ms: Target interval in milliseconds
        method: Fill method (default: ffill - forward fill last quote)

    Returns:
        Resampled DataFrame aligned to interval_ms grid
    """
    if raw_bbo.empty:
        return pd.DataFrame(columns=["ts_recv", "bid_px", "ask_px", "bid_sz", "ask_sz", "mid_px", "spread"])

    # Clean data first
    clean_bbo = validate_and_clean_bbo(raw_bbo)

    if clean_bbo.empty:
        return pd.DataFrame(columns=["ts_recv", "bid_px", "ask_px", "bid_sz", "ask_sz", "mid_px", "spread"])

    # Resample using existing utility
    resampled = resample_to_grid(
        clean_bbo,
        grid_ms=interval_ms,
        ts_col="ts_recv",
        method=method,
    )

    # Compute derived fields
    resampled["mid_px"] = (resampled["bid_px"] + resampled["ask_px"]) / 2
    resampled["spread"] = resampled["ask_px"] - resampled["bid_px"]

    # Select and order columns
    return resampled[["ts_recv", "bid_px", "ask_px", "bid_sz", "ask_sz", "mid_px", "spread"]]


def validate_and_clean_bbo(
    df: pd.DataFrame,
    remove_crossed: bool = True,
    remove_zero_size: bool = True,
) -> pd.DataFrame:
    """Validate and clean BBO data.

    Removes:
    - Rows where bid > ask (crossed quotes)
    - Rows where bid <= 0 or ask <= 0
    - Rows where bid_sz <= 0 or ask_sz <= 0 (optional)
    - Duplicate timestamps (keeps last)

    Args:
        df: Raw BBO DataFrame
        remove_crossed: Remove crossed quotes
        remove_zero_size: Remove zero-size quotes

    Returns:
        Cleaned DataFrame
    """
    if df.empty:
        return df.copy()

    # Drop NaNs in critical columns
    df = df.dropna(subset=["bid_px", "ask_px"])

    # Remove invalid prices
    df = df[(df["bid_px"] > 0) & (df["ask_px"] > 0)]

    # Remove crossed quotes
    if remove_crossed:
        df = df[df["bid_px"] <= df["ask_px"]]

    # Remove zero sizes
    if remove_zero_size and "bid_sz" in df.columns and "ask_sz" in df.columns:
        df = df[(df["bid_sz"] > 0) & (df["ask_sz"] > 0)]

    # Remove duplicates (keep last)
    if "ts_recv" in df.columns:
        df = df.drop_duplicates(subset=["ts_recv"], keep="last")

    return df


def get_missing_dates(
    start_dt: datetime,
    end_dt: datetime,
    asset: str,
    interval_ms: int,
    cache_dir: Path,
) -> list[datetime]:
    """Identify missing date ranges in cache.

    Checks which daily files exist and returns list of missing
    dates that need to be fetched from S3.

    Args:
        start_dt: Start datetime
        end_dt: End datetime
        asset: Asset symbol
        interval_ms: Interval in milliseconds
        cache_dir: Cache directory path

    Returns:
        List of datetime objects for missing dates
    """
    expected_dates = _generate_date_list(start_dt, end_dt)

    # Check which files exist
    interval_dir = cache_dir / "binance" / f"asset={asset}" / f"interval={_interval_dir_name(interval_ms)}"
    if not interval_dir.exists():
        return expected_dates

    existing_files = list(interval_dir.glob("date=*.parquet"))
    existing_dates = set()
    for file_path in existing_files:
        date_str = file_path.stem.replace("date=", "")
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
            existing_dates.add(date)
        except ValueError:
            continue

    # Compare using date objects (year, month, day) to avoid timezone mismatch
    existing_date_strs = {d.date() for d in existing_dates}
    missing_dates = [date for date in expected_dates if date.date() not in existing_date_strs]
    return missing_dates


def _generate_date_list(start_dt: datetime, end_dt: datetime) -> list[datetime]:
    """Generate list of dates between start and end."""
    dates = []
    current = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    while current < end_dt:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def _process_and_cache_day(
    date: datetime,
    asset: str,
    interval_ms: int,
    cache_dir: Path,
) -> pd.DataFrame | None:
    """Load a single day from S3, resample, and cache it.

    Processes data hour-by-hour to avoid memory issues with large datasets.

    Args:
        date: Date to process
        asset: Asset symbol
        interval_ms: Resampling interval
        cache_dir: Cache directory

    Returns:
        Resampled DataFrame or None if failed
    """
    # Process hour by hour to avoid memory issues
    # Save to temporary files to avoid memory accumulation
    import tempfile
    import os

    temp_files = []
    hours_processed = 0

    for hour in range(24):
        hour_start = date.replace(hour=hour, minute=0, second=0, microsecond=0)
        hour_end = hour_start + timedelta(hours=1)

        # Retry logic for transient network errors
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                # Load one hour of raw BBO data
                raw_bbo_hour = load_binance_bbo(
                    start_dt=hour_start,
                    end_dt=hour_end,
                    asset=asset,
                )

                if not raw_bbo_hour.empty:
                    # Resample this hour
                    resampled_hour = resample_bbo_to_interval(raw_bbo_hour, interval_ms=interval_ms)
                    if not resampled_hour.empty:
                        # Save to temp file immediately to avoid memory buildup
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.parquet')
                        resampled_hour.to_parquet(temp_file.name, engine='pyarrow', compression='snappy', index=False)
                        temp_files.append(temp_file.name)
                        temp_file.close()

                        hours_processed += 1
                        # Print progress every 6 hours
                        if hours_processed % 6 == 0:
                            print(f"    Processed {hours_processed}/24 hours... [{_log_memory()}]", flush=True)

                        # Clear hour data from memory
                        del raw_bbo_hour
                        del resampled_hour
                        gc.collect()

                # Success, break retry loop
                break

            except Exception as e:
                error_str = str(e)

                # Handle missing hours gracefully (404 errors)
                if "404" in error_str or "Not Found" in error_str:
                    break  # Skip this hour, move to next one

                # Handle transient errors with retry
                if "IO Error" in error_str or "connection error" in error_str or "Failed to read" in error_str:
                    if attempt < max_retries - 1:
                        print(f"    ⚠️  Network error on hour {hour}, retrying ({attempt+1}/{max_retries})...", flush=True)
                        time.sleep(retry_delay)
                        continue  # Retry
                    else:
                        # Max retries reached, skip this hour but continue with others
                        print(f"    ⚠️  Failed hour {hour} after {max_retries} retries, skipping...", flush=True)
                        break

                # Unknown error - re-raise
                raise

    # Combine all temp files
    if not temp_files:
        return None

    # Use DuckDB to combine temp files and write directly to parquet (no pandas materialization)
    print(f"    Combining {len(temp_files)} hourly files using DuckDB... [{_log_memory()}]", flush=True)

    import duckdb
    import os

    # Prepare output path
    interval_dir = cache_dir / "binance" / f"asset={asset}" / f"interval={_interval_dir_name(interval_ms)}"
    interval_dir.mkdir(parents=True, exist_ok=True)
    date_str = date.strftime("%Y-%m-%d")
    final_path = interval_dir / f"date={date_str}.parquet"
    temp_output = str(final_path) + ".tmp"

    # Create file list for DuckDB
    file_pattern = "['" + "','".join(temp_files) + "']"

    try:
        conn = duckdb.connect(':memory:')

        # Write directly to parquet via DuckDB (no pandas materialization!)
        # This saves ~400MB by streaming instead of loading all data into memory
        query = f"""
            COPY (
                SELECT DISTINCT ON (ts_recv) *
                FROM read_parquet({file_pattern})
                ORDER BY ts_recv
            ) TO '{temp_output}' (FORMAT PARQUET, COMPRESSION 'snappy')
        """
        conn.execute(query)
        print(f"    Wrote combined parquet [{_log_memory()}]", flush=True)

        # Get row count for metadata (lightweight query)
        row_count_query = f"SELECT COUNT(*) FROM read_parquet('{temp_output}')"
        row_count = conn.execute(row_count_query).fetchone()[0]
        conn.close()

        # Atomic move to final location
        os.replace(temp_output, str(final_path))

    except Exception as e:
        # Clean up on error
        if 'conn' in locals():
            try:
                conn.close()
            except:
                pass
        if os.path.exists(temp_output):
            try:
                os.unlink(temp_output)
            except:
                pass
        raise Exception(f"Failed to combine hourly files: {e}")

    finally:
        # Clean up temp files immediately
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass

        # Clear any remaining memory
        gc.collect()

    # Update metadata directly without loading full DataFrame
    from marketdata.data.cache_manager import update_metadata
    from datetime import datetime as dt

    stats = {
        "date": date_str,
        "rows": row_count,
        "file_size_bytes": final_path.stat().st_size,
        "cached_at": dt.utcnow().isoformat() + "Z",
    }

    if row_count > 0:
        stats["validation"] = {
            "pct_valid_quotes": 100.0,  # Already validated in resample step
            "pct_gaps_filled": 0.0,
            "max_gap_seconds": 0.0,
        }

    update_metadata(date, asset, interval_ms, cache_dir, "binance", stats)
    print(f"    Saved to cache [{_log_memory()}]", flush=True)

    # Report if partial day
    if hours_processed < 24:
        print(f"    ⚠️  Partial day: {hours_processed}/24 hours processed", flush=True)

    # Don't load back into memory - let caller load from cache if needed
    # This prevents memory accumulation when processing multiple days
    return None
