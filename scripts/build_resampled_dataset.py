"""Build complete resampled BBO dataset for a date range."""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import load_resampled_bbo, get_cache_info


def build_dataset(
    start_date: datetime,
    end_date: datetime,
    interval_ms: int = 1000,
    asset: str = "BTC",
    delay_between_days: float = 2.0,
):
    """Build resampled dataset for date range.

    Args:
        start_date: Start date (inclusive)
        end_date: End date (exclusive)
        interval_ms: Resampling interval in milliseconds
        asset: Asset symbol
        delay_between_days: Seconds to wait between processing days (to avoid CPU spikes)
    """
    print("=" * 70)
    print("BUILDING RESAMPLED BBO DATASET")
    print("=" * 70)
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Interval: {interval_ms}ms ({interval_ms/1000}s)")
    print(f"Asset: {asset}")
    print(f"CPU throttle: {delay_between_days}s delay between days")
    print()

    # Calculate total days
    total_days = (end_date - start_date).days

    # Check what's already cached
    cache_dir = Path("data/resampled_bbo")
    cache_info = get_cache_info(asset=asset, interval_ms=interval_ms, cache_dir=cache_dir)

    print(f"Total days to process: {total_days}")
    print(f"Already cached: {len(cache_info['available_dates'])} days")
    print(f"  Dates: {cache_info['date_range'] if cache_info['available_dates'] else 'none'}")
    print(f"  Size: {cache_info['total_size_mb']:.2f} MB")
    print()
    print("Days to fetch from S3: {0}".format(total_days - len(cache_info['available_dates'])))
    print("=" * 70)
    print()

    # Track statistics
    stats = {
        "days_processed": 0,
        "days_with_data": 0,
        "days_without_data": 0,
        "total_rows": 0,
        "missing_hours": [],
        "errors": [],
    }

    start_time = time.time()

    # Process one day at a time
    current_date = start_date
    day_num = 1

    while current_date < end_date:
        next_date = current_date + timedelta(days=1)

        print(f"[{day_num}/{total_days}] Processing {current_date.date()}...")
        day_start_time = time.time()

        try:
            # Load this single day
            df = load_resampled_bbo(
                start_dt=current_date,
                end_dt=next_date,
                interval_ms=interval_ms,
                asset=asset,
                cache_dir=cache_dir,
            )

            day_elapsed = time.time() - day_start_time

            if not df.empty:
                stats["days_with_data"] += 1
                stats["total_rows"] += len(df)

                # Calculate expected rows for the interval
                expected_rows = (24 * 3600 * 1000) // interval_ms  # 86,400 for 1s
                coverage = (len(df) / expected_rows) * 100

                print(f"  ✓ {len(df):,} rows cached ({coverage:.1f}% coverage)")
                print(f"  ⏱️  Processed in {day_elapsed:.1f}s")

                # Check for missing hours (if coverage is less than 95%)
                if coverage < 95:
                    missing_pct = 100 - coverage
                    stats["missing_hours"].append({
                        "date": current_date.date(),
                        "rows": len(df),
                        "expected": expected_rows,
                        "missing_pct": missing_pct
                    })
                    print(f"  ⚠️  Warning: {missing_pct:.1f}% data missing")
            else:
                stats["days_without_data"] += 1
                print(f"  ✗ No data available")

            stats["days_processed"] += 1

        except Exception as e:
            print(f"  ✗ ERROR: {str(e)[:100]}")
            stats["errors"].append({
                "date": current_date.date(),
                "error": str(e)[:200]
            })

        # Explicit memory cleanup: delete DataFrame and force garbage collection
        # This prevents memory accumulation between days
        if 'df' in locals():
            del df
        import gc
        gc.collect()

        # CPU throttle: wait before processing next day
        if day_num < total_days:  # Don't wait after last day
            print(f"  💤 Waiting {delay_between_days}s (CPU throttle)...")
            time.sleep(delay_between_days)

        print()

        current_date = next_date
        day_num += 1

    # Print summary
    elapsed_total = time.time() - start_time

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} minutes)")
    print(f"Days processed: {stats['days_processed']}/{total_days}")
    print(f"Days with data: {stats['days_with_data']}")
    print(f"Days without data: {stats['days_without_data']}")
    print(f"Total rows: {stats['total_rows']:,}")
    print()

    if stats["missing_hours"]:
        print("⚠️  DAYS WITH MISSING HOURS:")
        print("-" * 70)
        for item in stats["missing_hours"]:
            print(f"  {item['date']}: {item['rows']:,}/{item['expected']:,} rows ({item['missing_pct']:.1f}% missing)")
        print()

    if stats["errors"]:
        print("❌ ERRORS:")
        print("-" * 70)
        for item in stats["errors"]:
            print(f"  {item['date']}: {item['error']}")
        print()

    # Show cache info
    print("=" * 70)
    print("CACHE INFO")
    print("=" * 70)
    cache_dir = Path("data/resampled_bbo")
    info = get_cache_info(asset=asset, interval_ms=interval_ms, cache_dir=cache_dir)

    print(f"Available dates: {len(info['available_dates'])}")
    print(f"Date range: {info['date_range']}")
    print(f"Total rows in cache: {info['total_rows']:,}")
    print(f"Total cache size: {info['total_size_mb']:.2f} MB")

    if info['total_rows'] > 0:
        bytes_per_row = (info['total_size_mb'] * 1024 * 1024) / info['total_rows']
        print(f"Compression: {bytes_per_row:.2f} bytes/row")

    print()
    print("=" * 70)
    print("✓ Dataset build complete!")
    print("=" * 70)


def main():
    # Build dataset from 2026-01-19 to 2026-02-09
    start_date = datetime(2026, 1, 19, tzinfo=timezone.utc)
    end_date = datetime(2026, 2, 10, tzinfo=timezone.utc)  # Exclusive, so will process up to 2026-02-08

    # Build 1s interval dataset
    print("\n" + "=" * 70)
    print("BUILDING 1s INTERVAL DATASET")
    print("=" * 70)
    build_dataset(
        start_date=start_date,
        end_date=end_date,
        interval_ms=1000,
        asset="BTC",
        delay_between_days=2.0,  # 2 second delay between days to avoid CPU spikes
    )

    # Optional: Build 5s interval dataset
    print("\n\n")
    user_input = input("Do you want to build 5s interval dataset as well? (y/n): ")
    if user_input.lower() in ['y', 'yes']:
        print("\n" + "=" * 70)
        print("BUILDING 5s INTERVAL DATASET")
        print("=" * 70)
        build_dataset(
            start_date=start_date,
            end_date=end_date,
            interval_ms=5000,
            asset="BTC",
            delay_between_days=2.0,
        )


if __name__ == "__main__":
    main()
