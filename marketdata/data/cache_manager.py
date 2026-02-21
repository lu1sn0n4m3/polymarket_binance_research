"""Cache management utilities for resampled BBO data."""

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd


def _interval_dir_name(interval_ms: int) -> str:
    """Convert interval_ms to a directory name that is unique per interval.

    100  -> '100ms'
    500  -> '500ms'
    1000 -> '1s'
    5000 -> '5s'
    """
    if interval_ms >= 1000 and interval_ms % 1000 == 0:
        return f"{interval_ms // 1000}s"
    return f"{interval_ms}ms"


def save_resampled_day(
    df: pd.DataFrame,
    date: datetime,
    asset: str,
    interval_ms: int,
    cache_dir: Path,
    venue: Literal["binance", "polymarket"] = "binance",
) -> None:
    """Save a single day of resampled data to cache.

    Args:
        df: Resampled DataFrame for one day
        date: Date being saved
        asset: Asset symbol ("BTC" or "ETH")
        interval_ms: Resampling interval in milliseconds
        cache_dir: Base cache directory
        venue: Data venue ("binance" or "polymarket")
    """
    # Create directory structure: venue/asset={asset}/interval={interval}s/
    interval_dir = cache_dir / venue / f"asset={asset}" / f"interval={_interval_dir_name(interval_ms)}"
    interval_dir.mkdir(parents=True, exist_ok=True)

    # Save as date=YYYY-MM-DD.parquet
    date_str = date.strftime("%Y-%m-%d")
    file_path = interval_dir / f"date={date_str}.parquet"

    df.to_parquet(
        file_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    # Update metadata
    stats = {
        "date": date_str,
        "rows": len(df),
        "file_size_bytes": file_path.stat().st_size,
        "cached_at": datetime.utcnow().isoformat() + "Z",
    }

    # Add data quality stats
    if not df.empty:
        stats["validation"] = {
            "pct_valid_quotes": (df["spread"] >= 0).mean() * 100,
            "pct_gaps_filled": 0.0,  # Placeholder, could compute from raw data
            "max_gap_seconds": 0.0,  # Placeholder
        }

    update_metadata(date, asset, interval_ms, cache_dir, venue, stats)


def load_resampled_day(
    date: datetime,
    asset: str,
    interval_ms: int,
    cache_dir: Path,
    venue: Literal["binance", "polymarket"] = "binance",
) -> pd.DataFrame | None:
    """Load a single day of resampled data from cache.

    Args:
        date: Date to load
        asset: Asset symbol ("BTC" or "ETH")
        interval_ms: Resampling interval in milliseconds
        cache_dir: Base cache directory
        venue: Data venue ("binance" or "polymarket")

    Returns:
        DataFrame if file exists, None otherwise
    """
    date_str = date.strftime("%Y-%m-%d")
    file_path = cache_dir / venue / f"asset={asset}" / f"interval={_interval_dir_name(interval_ms)}" / f"date={date_str}.parquet"

    if not file_path.exists():
        return None

    return pd.read_parquet(file_path)


def update_metadata(
    date: datetime,
    asset: str,
    interval_ms: int,
    cache_dir: Path,
    venue: Literal["binance", "polymarket"],
    stats: dict,
) -> None:
    """Update metadata file with cache statistics.

    Metadata tracks:
    - Last update timestamp
    - Number of rows
    - Date range coverage
    - Data quality metrics (% valid quotes, % gaps, etc.)

    Args:
        date: Date being updated
        asset: Asset symbol
        interval_ms: Resampling interval
        cache_dir: Base cache directory
        venue: Data venue ("binance" or "polymarket")
        stats: Statistics dictionary
    """
    interval_dir = cache_dir / venue / f"asset={asset}" / f"interval={_interval_dir_name(interval_ms)}"
    metadata_path = interval_dir / ".metadata.json"

    # Load existing metadata or create new
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {
            "venue": venue,
            "asset": asset,
            "interval_ms": interval_ms,
            "cached_dates": [],
        }

    # Update or add date entry
    date_str = stats["date"]
    existing_dates = {entry["date"]: i for i, entry in enumerate(metadata["cached_dates"])}

    if date_str in existing_dates:
        # Update existing entry
        metadata["cached_dates"][existing_dates[date_str]] = stats
    else:
        # Add new entry
        metadata["cached_dates"].append(stats)

    # Sort by date
    metadata["cached_dates"].sort(key=lambda x: x["date"])

    # Update aggregates
    metadata["total_rows"] = sum(entry["rows"] for entry in metadata["cached_dates"])
    if metadata["cached_dates"]:
        metadata["date_range"] = [
            metadata["cached_dates"][0]["date"],
            metadata["cached_dates"][-1]["date"],
        ]
    else:
        metadata["date_range"] = []

    metadata["last_updated"] = datetime.utcnow().isoformat() + "Z"

    # Save metadata
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def get_cache_info(
    asset: str,
    interval_ms: int,
    cache_dir: Path,
    venue: Literal["binance", "polymarket"] = "binance",
) -> dict:
    """Get cache coverage information.

    Args:
        asset: Asset symbol
        interval_ms: Resampling interval
        cache_dir: Base cache directory
        venue: Data venue ("binance" or "polymarket")

    Returns:
        Dictionary with:
        - available_dates: List of dates in cache
        - total_rows: Total cached rows
        - date_range: (min_date, max_date)
        - total_size_mb: Total cache size
    """
    interval_dir = cache_dir / venue / f"asset={asset}" / f"interval={_interval_dir_name(interval_ms)}"
    metadata_path = interval_dir / ".metadata.json"

    if not metadata_path.exists():
        return {
            "available_dates": [],
            "total_rows": 0,
            "date_range": (None, None),
            "total_size_mb": 0.0,
        }

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Calculate total size
    total_size_bytes = sum(entry["file_size_bytes"] for entry in metadata.get("cached_dates", []))

    return {
        "available_dates": [entry["date"] for entry in metadata.get("cached_dates", [])],
        "total_rows": metadata.get("total_rows", 0),
        "date_range": (
            metadata["date_range"][0] if metadata.get("date_range") else None,
            metadata["date_range"][1] if metadata.get("date_range") else None,
        ),
        "total_size_mb": total_size_bytes / (1024 * 1024),
    }


def clear_cache(
    asset: str | None = None,
    interval_ms: int | None = None,
    cache_dir: Path | None = None,
    venue: Literal["binance", "polymarket"] | None = None,
    before_date: datetime | None = None,
) -> int:
    """Clear cache files based on filters.

    Args:
        asset: If specified, only clear this asset
        interval_ms: If specified, only clear this interval
        cache_dir: Cache directory
        venue: If specified, only clear this venue
        before_date: If specified, only clear files before this date

    Returns:
        Number of files deleted
    """
    if cache_dir is None:
        return 0

    deleted_count = 0

    # Determine which venue directories to scan
    if venue is not None:
        venue_dirs = [cache_dir / venue]
    else:
        venue_dirs = [cache_dir / "binance", cache_dir / "polymarket"]

    for venue_dir in venue_dirs:
        if not venue_dir.exists():
            continue

        # Determine which asset directories to scan
        if asset is not None:
            asset_dirs = [venue_dir / f"asset={asset}"]
        else:
            asset_dirs = list(venue_dir.glob("asset=*"))

        for asset_dir in asset_dirs:
            if not asset_dir.exists():
                continue

            if interval_ms is not None:
                interval_dirs = [asset_dir / f"interval={_interval_dir_name(interval_ms)}"]
            else:
                interval_dirs = list(asset_dir.glob("interval=*"))

            for interval_dir in interval_dirs:
                if not interval_dir.exists():
                    continue

                # Get all parquet files
                for parquet_file in interval_dir.glob("date=*.parquet"):
                    # Extract date from filename
                    date_str = parquet_file.stem.replace("date=", "")
                    try:
                        file_date = datetime.strptime(date_str, "%Y-%m-%d")
                    except ValueError:
                        continue

                    # Check if should delete
                    if before_date is None or file_date < before_date:
                        parquet_file.unlink()
                        deleted_count += 1

                # If metadata exists, update it
                metadata_path = interval_dir / ".metadata.json"
                if metadata_path.exists() and before_date is not None:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)

                    # Filter out deleted dates
                    before_date_str = before_date.strftime("%Y-%m-%d")
                    metadata["cached_dates"] = [
                        entry for entry in metadata["cached_dates"]
                        if entry["date"] >= before_date_str
                    ]

                    # Update aggregates
                    if metadata["cached_dates"]:
                        metadata["total_rows"] = sum(entry["rows"] for entry in metadata["cached_dates"])
                        metadata["date_range"] = [
                            metadata["cached_dates"][0]["date"],
                            metadata["cached_dates"][-1]["date"],
                        ]
                    else:
                        metadata["total_rows"] = 0
                        metadata["date_range"] = []

                    metadata["last_updated"] = datetime.utcnow().isoformat() + "Z"

                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f, indent=2)

    return deleted_count
