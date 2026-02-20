"""Check what BBO data is available in S3."""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from marketdata.data import load_binance_bbo


def check_date_range(start_date: datetime, end_date: datetime, asset: str = "BTC"):
    """Check which dates have data available."""
    print(f"Checking {asset} BBO data availability from {start_date.date()} to {end_date.date()}")
    print("=" * 60)

    current = start_date
    available_dates = []

    while current < end_date:
        try:
            # Try to load first hour of the day
            df = load_binance_bbo(
                start_dt=current,
                end_dt=current + timedelta(hours=1),
                asset=asset,
            )
            if not df.empty:
                available_dates.append(current)
                print(f"✓ {current.date()} - {len(df):,} rows in first hour")
            else:
                print(f"✗ {current.date()} - no data")
        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e):
                print(f"✗ {current.date()} - not found")
            else:
                print(f"✗ {current.date()} - error: {str(e)[:50]}")

        current += timedelta(days=1)

    print()
    print("=" * 60)
    print(f"Found {len(available_dates)} days with data")
    if available_dates:
        print(f"First available: {available_dates[0].date()}")
        print(f"Last available: {available_dates[-1].date()}")

    return available_dates


if __name__ == "__main__":
    # Check recent dates
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=14)  # Check last 14 days

    print(f"Checking data availability for last 14 days...")
    print()

    available = check_date_range(start, end, asset="BTC")

    if available:
        print()
        print("Suggested test range:")
        print(f"  start_dt = datetime({available[0].year}, {available[0].month}, {available[0].day}, tzinfo=timezone.utc)")
        print(f"  end_dt = datetime({available[-1].year}, {available[-1].month}, {available[-1].day}, tzinfo=timezone.utc)")
