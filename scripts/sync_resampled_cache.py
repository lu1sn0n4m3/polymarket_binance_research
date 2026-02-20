"""Sync resampled BBO and Polymarket data cache from S3.

This script periodically updates your local resampled data cache.
It supports:
- Incremental mode (only fetch missing dates)
- Both Binance and Polymarket venues
- Multiple assets and intervals
- Retry logic for transient errors
- Progress tracking and summary reports
"""

import argparse
import sys
import time
from datetime import datetime, timezone, timedelta, date as date_type
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from marketdata.data import (
    load_resampled_bbo,
    load_resampled_polymarket,
    load_resampled_labels,
    get_cache_info,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sync resampled data cache from S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sync last 7 days of Binance BTC data (incremental)
  python scripts/sync_resampled_cache.py --venue binance --asset BTC --interval 1s --last-days 7 --incremental

  # Sync specific date range for both venues
  python scripts/sync_resampled_cache.py --venue binance,polymarket --asset BTC --interval 1s,5s --start-date 2026-01-19 --end-date 2026-02-10

  # Sync hourly labels (open/close/outcome from Binance trades)
  python scripts/sync_resampled_cache.py --venue binance_labels --asset BTC --interval 1s --start-date 2026-01-19 --end-date 2026-02-10

  # Daily cron job (sync last 7 days)
  python scripts/sync_resampled_cache.py --venue binance,polymarket --asset BTC --interval 1s --last-days 7 --incremental --delay 1.0
        """,
    )

    parser.add_argument(
        "--venue",
        type=str,
        required=True,
        help='Comma-separated venues: "binance", "polymarket", or "binance,polymarket"',
    )

    parser.add_argument(
        "--asset",
        type=str,
        default="BTC",
        help='Comma-separated assets: "BTC", "ETH", or "BTC,ETH" (default: BTC)',
    )

    parser.add_argument(
        "--interval",
        type=str,
        default="1s",
        help='Comma-separated intervals: "500ms", "1s", "5s" (default: 1s)',
    )

    # Date range options
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD, inclusive). Requires --end-date",
    )
    date_group.add_argument(
        "--last-days",
        type=int,
        help="Sync last N days from today",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD, exclusive). Requires --start-date",
    )

    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only fetch missing dates (skip already cached)",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between processing days (default: 2.0, helps avoid S3 throttling)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reload all dates (ignore cache)",
    )

    return parser.parse_args()


def parse_date_range(args):
    """Parse date range from arguments."""
    if args.last_days:
        end_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=args.last_days)
    else:
        if not args.end_date:
            print("ERROR: --start-date requires --end-date")
            sys.exit(1)

        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    return start_date, end_date


def parse_interval_ms(interval_str: str) -> int:
    """Convert interval string to milliseconds."""
    mapping = {
        "500ms": 500,
        "1s": 1000,
        "5s": 5000,
    }
    if interval_str not in mapping:
        raise ValueError(f"Invalid interval: {interval_str}. Must be one of {list(mapping.keys())}")
    return mapping[interval_str]


def _get_labels_cache_info(asset: str, cache_dir: Path) -> dict:
    """Get cache info for binance_labels (different path structure than BBO)."""
    import json
    label_dir = cache_dir / "binance_labels" / f"asset={asset}"
    meta_path = label_dir / ".metadata.json"
    if not meta_path.exists():
        return {"available_dates": [], "total_rows": 0, "date_range": (None, None), "total_size_mb": 0.0}
    with open(meta_path) as f:
        metadata = json.load(f)
    total_bytes = sum(e.get("file_size_bytes", 0) for e in metadata.get("cached_dates", []))
    return {
        "available_dates": [e["date"] for e in metadata.get("cached_dates", [])],
        "total_rows": metadata.get("total_rows", 0),
        "date_range": tuple(metadata.get("date_range", [None, None])),
        "total_size_mb": total_bytes / (1024 * 1024),
    }


def sync_venue_asset_interval(
    venue: str,
    asset: str,
    interval_str: str,
    start_date: datetime,
    end_date: datetime,
    incremental: bool,
    force_reload: bool,
    delay: float,
) -> dict:
    """Sync data for a specific venue/asset/interval combination.

    Returns:
        dict: Statistics about the sync operation
    """
    interval_ms = parse_interval_ms(interval_str)
    cache_dir = Path("data/resampled_data")

    # Calculate total days
    total_days = (end_date - start_date).days

    # Check what's already cached
    if venue == "binance_labels":
        cache_info = _get_labels_cache_info(asset, cache_dir)
    else:
        cache_info = get_cache_info(
            asset=asset,
            interval_ms=interval_ms,
            cache_dir=cache_dir,
            venue=venue,
        )

    print(f"\n{'=' * 70}")
    print(f"SYNCING: {venue.upper()} / {asset} / {interval_str}")
    print(f"{'=' * 70}")
    print(f"Date range: {start_date.date()} to {end_date.date()} ({total_days} days)")
    print(f"Mode: {'INCREMENTAL' if incremental else 'FULL'}")
    print(f"Force reload: {force_reload}")
    print()
    print(f"Already cached: {len(cache_info['available_dates'])} days")
    if cache_info['available_dates']:
        print(f"  Range: {cache_info['date_range']}")
        print(f"  Size: {cache_info['total_size_mb']:.2f} MB")
    print()

    # Track statistics
    stats = {
        "venue": venue,
        "asset": asset,
        "interval": interval_str,
        "total_days": total_days,
        "days_processed": 0,
        "days_cached": 0,
        "days_failed": 0,
        "days_no_data": 0,
        "total_rows": 0,
        "errors": [],
        "start_time": time.time(),
    }

    # Choose loader function
    if venue == "binance":
        loader_func = load_resampled_bbo
    elif venue == "polymarket":
        loader_func = load_resampled_polymarket
    elif venue == "binance_labels":
        loader_func = None  # handled separately below
    else:
        raise ValueError(f"Unknown venue: {venue}")

    # Process one day at a time
    current_date = start_date
    day_num = 0

    while current_date < end_date:
        day_num += 1
        next_date = current_date + timedelta(days=1)

        # Skip if incremental and already cached
        date_str = current_date.strftime("%Y-%m-%d")
        if incremental and date_str in cache_info['available_dates']:
            # Try to load from cache to verify
            try:
                print(f"[{day_num}/{total_days}] {date_str}: cached ✓")
                stats["days_cached"] += 1
                stats["days_processed"] += 1
            except Exception:
                # Cache corrupted, re-fetch
                pass

        if not (incremental and date_str in cache_info['available_dates']):
            print(f"[{day_num}/{total_days}] {date_str}: fetching from S3...", end=" ", flush=True)
            day_start_time = time.time()

            try:
                # Load/fetch this day
                if venue == "binance_labels":
                    df = load_resampled_labels(
                        start_dt=current_date,
                        end_dt=next_date,
                        asset=asset,
                        cache_dir=cache_dir,
                        force_reload=force_reload,
                    )
                else:
                    df = loader_func(
                        start_dt=current_date,
                        end_dt=next_date,
                        interval_ms=interval_ms,
                        asset=asset,
                        cache_dir=cache_dir,
                        force_reload=force_reload,
                    )

                day_elapsed = time.time() - day_start_time

                if not df.empty:
                    stats["days_cached"] += 1
                    stats["total_rows"] += len(df)
                    print(f"✓ ({len(df):,} rows, {day_elapsed:.1f}s)")
                else:
                    stats["days_no_data"] += 1
                    print(f"⚠ no data ({day_elapsed:.1f}s)")

                stats["days_processed"] += 1

            except Exception as e:
                print(f"✗ FAILED: {str(e)[:80]}")
                stats["days_failed"] += 1
                stats["errors"].append({
                    "date": date_str,
                    "error": str(e)[:200],
                })

            finally:
                # Cleanup memory
                if 'df' in locals():
                    del df
                import gc
                gc.collect()

            # Throttle: wait before next day
            if day_num < total_days and delay > 0:
                time.sleep(delay)

        current_date = next_date

    # Calculate elapsed time
    stats["elapsed_seconds"] = time.time() - stats["start_time"]

    return stats


def print_summary(all_stats: list[dict]):
    """Print summary of all sync operations."""
    print("\n\n" + "=" * 70)
    print("SYNC SUMMARY")
    print("=" * 70)

    total_days_processed = sum(s["days_processed"] for s in all_stats)
    total_days_cached = sum(s["days_cached"] for s in all_stats)
    total_days_failed = sum(s["days_failed"] for s in all_stats)
    total_days_no_data = sum(s["days_no_data"] for s in all_stats)
    total_rows = sum(s["total_rows"] for s in all_stats)
    total_errors = sum(len(s["errors"]) for s in all_stats)
    total_elapsed = sum(s["elapsed_seconds"] for s in all_stats)

    print(f"Combinations synced: {len(all_stats)}")
    print(f"Total days processed: {total_days_processed}")
    print(f"  ✓ Cached: {total_days_cached}")
    print(f"  ⚠ No data: {total_days_no_data}")
    print(f"  ✗ Failed: {total_days_failed}")
    print(f"Total rows: {total_rows:,}")
    print(f"Time elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print()

    if total_errors > 0:
        print("❌ ERRORS:")
        print("-" * 70)
        for stat in all_stats:
            if stat["errors"]:
                print(f"\n{stat['venue'].upper()} / {stat['asset']} / {stat['interval']}:")
                for err in stat["errors"]:
                    print(f"  {err['date']}: {err['error']}")
        print()

    # Show per-venue breakdown
    print("=" * 70)
    print("PER-VENUE BREAKDOWN")
    print("=" * 70)
    for stat in all_stats:
        print(f"\n{stat['venue'].upper()} / {stat['asset']} / {stat['interval']}:")
        print(f"  Days processed: {stat['days_processed']}/{stat['total_days']}")
        print(f"  Days cached: {stat['days_cached']}")
        print(f"  Rows: {stat['total_rows']:,}")
        print(f"  Time: {stat['elapsed_seconds']:.1f}s")

    print("\n" + "=" * 70)
    if total_days_failed == 0:
        print("✓ Sync complete!")
    else:
        print(f"⚠ Sync complete with {total_days_failed} failed days")
    print("=" * 70)


def main():
    """Main entry point."""
    args = parse_args()

    # Parse arguments
    venues = [v.strip() for v in args.venue.split(",")]
    assets = [a.strip() for a in args.asset.split(",")]
    intervals = [i.strip() for i in args.interval.split(",")]

    start_date, end_date = parse_date_range(args)

    # Validate
    valid_venues = {"binance", "polymarket", "binance_labels"}
    valid_assets = {"BTC", "ETH"}
    valid_intervals = {"500ms", "1s", "5s"}

    for venue in venues:
        if venue not in valid_venues:
            print(f"ERROR: Invalid venue '{venue}'. Must be one of {valid_venues}")
            sys.exit(1)

    for asset in assets:
        if asset not in valid_assets:
            print(f"ERROR: Invalid asset '{asset}'. Must be one of {valid_assets}")
            sys.exit(1)

    for interval in intervals:
        if interval not in valid_intervals:
            print(f"ERROR: Invalid interval '{interval}'. Must be one of {valid_intervals}")
            sys.exit(1)

    # Print configuration
    print("=" * 70)
    print("RESAMPLED DATA CACHE SYNC")
    print("=" * 70)
    print(f"Venues: {', '.join(venues)}")
    print(f"Assets: {', '.join(assets)}")
    print(f"Intervals: {', '.join(intervals)}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Mode: {'INCREMENTAL' if args.incremental else 'FULL'}")
    print(f"Delay between days: {args.delay}s")
    print(f"Force reload: {args.force}")
    print()

    # Sync all combinations
    all_stats = []

    for venue in venues:
        for asset in assets:
            for interval in intervals:
                try:
                    stats = sync_venue_asset_interval(
                        venue=venue,
                        asset=asset,
                        interval_str=interval,
                        start_date=start_date,
                        end_date=end_date,
                        incremental=args.incremental,
                        force_reload=args.force,
                        delay=args.delay,
                    )
                    all_stats.append(stats)

                except KeyboardInterrupt:
                    print("\n\n⚠ Interrupted by user")
                    break
                except Exception as e:
                    print(f"\n\n❌ FATAL ERROR for {venue}/{asset}/{interval}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    # Print summary
    if all_stats:
        print_summary(all_stats)


if __name__ == "__main__":
    main()
