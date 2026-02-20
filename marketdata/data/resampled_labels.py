"""Hourly market labels (open/close/outcome) with automatic daily caching.

Each daily parquet stores up to 24 rows — one per hour — with the first and
last Binance trade prices.  This eliminates the need to reload millions of
trade rows on subsequent runs.

Cache layout:
    data/resampled_data/binance_labels/asset=BTC/date=2026-01-19.parquet
"""

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from marketdata.data.loaders import load_binance_trades


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_resampled_labels(
    start_dt: datetime,
    end_dt: datetime,
    asset: str = "BTC",
    cache_dir: Path | str | None = None,
    force_reload: bool = False,
) -> pd.DataFrame:
    """Load hourly open/close labels with automatic daily caching.

    For each complete UTC hour in [start_dt, end_dt) the result contains:
        hour_start_ms, hour_end_ms  – epoch ms boundaries
        K        – price of first trade in the hour  (opening price)
        S_T      – price of last trade in the hour   (closing price)
        Y        – 1 if S_T > K, else 0              (market outcome)

    Only hours that contain >= 2 trades are included.
    """
    if cache_dir is None:
        cache_dir = Path("data/resampled_data")
    else:
        cache_dir = Path(cache_dir)

    dates = _generate_date_list(start_dt, end_dt)
    missing = _get_missing_dates(dates, asset, cache_dir) if not force_reload else dates

    # Fetch and cache missing days
    for date in missing:
        date_str = date.strftime("%Y-%m-%d")
        print(f"  {date_str}: fetching labels from S3...", end=" ", flush=True)
        try:
            _fetch_and_cache_day(date, asset, cache_dir)
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")

    # Load all days from cache
    dfs = []
    for date in dates:
        df = _load_cached_day(date, asset, cache_dir)
        if df is not None and not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=["hour_start_ms", "hour_end_ms", "K", "S_T", "Y"])

    result = pd.concat(dfs, ignore_index=True)
    result = result.sort_values("hour_start_ms").reset_index(drop=True)

    # Filter to requested range
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    result = result[(result["hour_start_ms"] >= start_ms) & (result["hour_end_ms"] <= end_ms)]

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _cache_dir_for(asset: str, cache_dir: Path) -> Path:
    return cache_dir / "binance_labels" / f"asset={asset}"


def _cache_path(date: datetime, asset: str, cache_dir: Path) -> Path:
    return _cache_dir_for(asset, cache_dir) / f"date={date.strftime('%Y-%m-%d')}.parquet"


def _load_cached_day(date: datetime, asset: str, cache_dir: Path) -> pd.DataFrame | None:
    path = _cache_path(date, asset, cache_dir)
    if path.exists():
        return pd.read_parquet(path)
    return None


def _get_missing_dates(dates: list[datetime], asset: str, cache_dir: Path) -> list[datetime]:
    return [d for d in dates if not _cache_path(d, asset, cache_dir).exists()]


# ---------------------------------------------------------------------------
# Fetch one day of trades → hourly labels → save
# ---------------------------------------------------------------------------

def _fetch_and_cache_day(
    date: datetime,
    asset: str,
    cache_dir: Path,
    max_retries: int = 3,
) -> None:
    day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    if day_start.tzinfo is None:
        day_start = day_start.replace(tzinfo=timezone.utc)

    # Load hour-by-hour to avoid OOM on heavy trade days
    rows = []
    for hour in range(24):
        hour_start = day_start + timedelta(hours=hour)
        hour_end = hour_start + timedelta(hours=1)
        hour_start_ms = int(hour_start.timestamp() * 1000)
        hour_end_ms = int(hour_end.timestamp() * 1000)

        for attempt in range(max_retries):
            try:
                trades = load_binance_trades(hour_start, hour_end, asset=asset)
                break
            except Exception as e:
                if "404" in str(e) or "Not Found" in str(e):
                    trades = pd.DataFrame(columns=["ts_event", "price"])
                    break
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    trades = pd.DataFrame(columns=["ts_event", "price"])
                    break

        if not trades.empty:
            trades = trades[["ts_event", "price"]].sort_values("ts_event")
            mask = (trades["ts_event"] >= hour_start_ms) & (trades["ts_event"] < hour_end_ms)
            hour_trades = trades[mask]
            if len(hour_trades) >= 2:
                K = float(hour_trades.iloc[0]["price"])
                S_T = float(hour_trades.iloc[-1]["price"])
                rows.append({
                    "hour_start_ms": hour_start_ms,
                    "hour_end_ms": hour_end_ms,
                    "K": K,
                    "S_T": S_T,
                    "Y": 1 if S_T > K else 0,
                })
            del trades

    labels = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["hour_start_ms", "hour_end_ms", "K", "S_T", "Y"]
    )

    # Save to cache
    out_dir = _cache_dir_for(asset, cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = _cache_path(date, asset, cache_dir)
    labels.to_parquet(path, engine="pyarrow", compression="snappy", index=False)

    # Update metadata
    _update_metadata(date, asset, cache_dir, len(labels), path.stat().st_size)


# ---------------------------------------------------------------------------
# Metadata (mirrors cache_manager pattern)
# ---------------------------------------------------------------------------

def _update_metadata(
    date: datetime,
    asset: str,
    cache_dir: Path,
    rows: int,
    file_size_bytes: int,
) -> None:
    meta_path = _cache_dir_for(asset, cache_dir) / ".metadata.json"

    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
    else:
        metadata = {"venue": "binance_labels", "asset": asset, "cached_dates": []}

    date_str = date.strftime("%Y-%m-%d")
    entry = {
        "date": date_str,
        "rows": rows,
        "file_size_bytes": file_size_bytes,
        "cached_at": datetime.utcnow().isoformat() + "Z",
    }

    existing = {e["date"]: i for i, e in enumerate(metadata["cached_dates"])}
    if date_str in existing:
        metadata["cached_dates"][existing[date_str]] = entry
    else:
        metadata["cached_dates"].append(entry)

    metadata["cached_dates"].sort(key=lambda x: x["date"])
    metadata["total_rows"] = sum(e["rows"] for e in metadata["cached_dates"])
    if metadata["cached_dates"]:
        metadata["date_range"] = [
            metadata["cached_dates"][0]["date"],
            metadata["cached_dates"][-1]["date"],
        ]
    metadata["last_updated"] = datetime.utcnow().isoformat() + "Z"

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_date_list(start_dt: datetime, end_dt: datetime) -> list[datetime]:
    dates = []
    current = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    while current < end_dt:
        dates.append(current)
        current += timedelta(days=1)
    return dates
