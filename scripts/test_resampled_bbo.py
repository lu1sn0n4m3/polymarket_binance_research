"""Test script for resampled BBO data loading."""

import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from marketdata.data import load_resampled_bbo, get_cache_info


def main():
    print("Testing resampled BBO data loading...")
    print("=" * 60)

    # Test with a small date range
    # Note: Using 2026-01-20 as a test date - adjust if needed based on your data availability
    start_dt = datetime(2026, 1, 20, tzinfo=timezone.utc)
    end_dt = datetime(2026, 1, 21, tzinfo=timezone.utc)

    print(f"\nLoading BTC BBO data from {start_dt} to {end_dt}")
    print(f"Interval: 1s (1000ms)")
    print()

    # Load 1s resampled data
    df = load_resampled_bbo(
        start_dt=start_dt,
        end_dt=end_dt,
        interval_ms=1000,
        asset="BTC",
    )

    print()
    print("=" * 60)
    print("Results:")
    print(f"  Total rows: {len(df):,}")
    if not df.empty:
        print(f"  Columns: {list(df.columns)}")
        print(f"  Time range: {df['ts_recv'].min()} to {df['ts_recv'].max()}")
        print(f"  Mid price range: {df['mid_px'].min():.2f} to {df['mid_px'].max():.2f}")
        print(f"  Avg spread: {df['spread'].mean():.2f}")
        print()
        print("Sample (first 5 rows):")
        print(df.head())

    # Check cache info
    print()
    print("=" * 60)
    print("Cache Info:")
    cache_dir = Path("data/resampled_bbo")
    cache_info = get_cache_info(asset="BTC", interval_ms=1000, cache_dir=cache_dir)
    print(f"  Available dates: {len(cache_info['available_dates'])}")
    print(f"  Date range: {cache_info['date_range']}")
    print(f"  Total rows: {cache_info['total_rows']:,}")
    print(f"  Total size: {cache_info['total_size_mb']:.2f} MB")

    # Test 5s interval
    print()
    print("=" * 60)
    print("\nTesting 5s interval...")
    df_5s = load_resampled_bbo(
        start_dt=start_dt,
        end_dt=end_dt,
        interval_ms=5000,
        asset="BTC",
    )
    print(f"  Total rows (5s): {len(df_5s):,}")

    print()
    print("Test completed successfully!")


if __name__ == "__main__":
    main()
