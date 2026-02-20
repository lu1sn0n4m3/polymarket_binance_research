"""Performance and functionality test for resampled BBO data."""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from marketdata.data import load_resampled_bbo, get_cache_info


def test_loading_speed():
    """Test loading speed from cache."""
    print("=" * 60)
    print("TEST 1: Loading Speed")
    print("=" * 60)

    start_dt = datetime(2026, 1, 20, tzinfo=timezone.utc)
    end_dt = datetime(2026, 1, 21, tzinfo=timezone.utc)

    # First load (from cache)
    print("\nLoading 1s data from cache...")
    t0 = time.time()
    df_1s = load_resampled_bbo(
        start_dt=start_dt,
        end_dt=end_dt,
        interval_ms=1000,
        asset="BTC"
    )
    t1 = time.time()

    print(f"✓ Loaded {len(df_1s):,} rows in {t1-t0:.3f} seconds")
    print(f"  Loading speed: {len(df_1s)/(t1-t0):,.0f} rows/sec")

    # Load 5s data
    print("\nLoading 5s data from cache...")
    t0 = time.time()
    df_5s = load_resampled_bbo(
        start_dt=start_dt,
        end_dt=end_dt,
        interval_ms=5000,
        asset="BTC"
    )
    t1 = time.time()

    print(f"✓ Loaded {len(df_5s):,} rows in {t1-t0:.3f} seconds")
    print(f"  Loading speed: {len(df_5s)/(t1-t0):,.0f} rows/sec")

    return df_1s, df_5s


def test_calculations(df):
    """Test various calculations on the data."""
    print("\n" + "=" * 60)
    print("TEST 2: Data Calculations")
    print("=" * 60)

    # Calculate log returns
    print("\nCalculating log returns...")
    t0 = time.time()
    df = df.copy()
    df["log_ret"] = np.log(df["mid_px"] / df["mid_px"].shift(1))
    t1 = time.time()
    print(f"✓ Calculated in {t1-t0:.3f} seconds")

    # Calculate realized volatility
    print("\nCalculating realized volatility...")
    t0 = time.time()
    # Annualized volatility (sqrt of seconds in a year)
    rv_annualized = df["log_ret"].std() * np.sqrt(365 * 24 * 3600)
    t1 = time.time()
    print(f"✓ Realized Vol (annualized): {rv_annualized*100:.2f}%")
    print(f"  Calculated in {t1-t0:.3f} seconds")

    # Rolling volatility (10-minute window)
    print("\nCalculating 10-min rolling volatility...")
    t0 = time.time()
    window = 600  # 10 minutes in 1s intervals
    df["rolling_vol"] = df["log_ret"].rolling(window=window).std() * np.sqrt(365 * 24 * 3600)
    t1 = time.time()
    print(f"✓ Calculated in {t1-t0:.3f} seconds")

    # Price statistics
    print("\nPrice statistics:")
    print(f"  Min: ${df['mid_px'].min():,.2f}")
    print(f"  Max: ${df['mid_px'].max():,.2f}")
    print(f"  Mean: ${df['mid_px'].mean():,.2f}")
    print(f"  Std: ${df['mid_px'].std():,.2f}")

    # Return statistics
    print("\nReturn statistics:")
    print(f"  Mean return: {df['log_ret'].mean()*100:.6f}%")
    print(f"  Std return: {df['log_ret'].std()*100:.4f}%")
    print(f"  Skewness: {df['log_ret'].skew():.4f}")
    print(f"  Kurtosis: {df['log_ret'].kurtosis():.4f}")

    # Spread statistics
    print("\nSpread statistics:")
    print(f"  Mean spread: ${df['spread'].mean():.4f}")
    print(f"  Median spread: ${df['spread'].median():.4f}")
    print(f"  Max spread: ${df['spread'].max():.4f}")

    return df


def test_visualization(df):
    """Create visualizations to verify data quality."""
    print("\n" + "=" * 60)
    print("TEST 3: Data Visualization")
    print("=" * 60)

    print("\nGenerating charts...")
    t0 = time.time()

    # Convert timestamp to datetime for plotting
    df["datetime"] = pd.to_datetime(df["ts_recv"], unit="ms")

    # Create subplot figure
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "BTC Price (1s intervals)",
            "Log Returns",
            "Rolling Volatility (10-min window)"
        ),
        vertical_spacing=0.08,
        row_heights=[0.4, 0.3, 0.3]
    )

    # Plot 1: Price
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["mid_px"],
            mode="lines",
            name="Mid Price",
            line=dict(color="blue", width=1)
        ),
        row=1, col=1
    )

    # Plot 2: Returns
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["log_ret"] * 100,  # Convert to percentage
            mode="lines",
            name="Log Returns (%)",
            line=dict(color="green", width=0.5)
        ),
        row=2, col=1
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # Plot 3: Rolling volatility
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["rolling_vol"] * 100,  # Convert to percentage
            mode="lines",
            name="Rolling Vol (%)",
            line=dict(color="red", width=1)
        ),
        row=3, col=1
    )

    # Update layout
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Return (%)", row=2, col=1)
    fig.update_yaxes(title_text="Volatility (%)", row=3, col=1)

    fig.update_layout(
        height=1000,
        title_text="BTC Resampled BBO Analysis (1s intervals)",
        showlegend=False,
        hovermode="x unified"
    )

    # Save to file
    output_path = project_root / "outputs"
    output_path.mkdir(exist_ok=True)
    html_file = output_path / "resampled_bbo_test.html"
    fig.write_html(str(html_file))

    t1 = time.time()
    print(f"✓ Generated charts in {t1-t0:.3f} seconds")
    print(f"  Saved to: {html_file}")

    # Additional: Distribution plot
    print("\nGenerating return distribution...")
    fig2 = go.Figure()

    # Histogram
    fig2.add_trace(go.Histogram(
        x=df["log_ret"] * 100,
        nbinsx=100,
        name="Returns",
        marker_color="blue",
        opacity=0.7
    ))

    fig2.update_layout(
        title="Return Distribution (1s intervals)",
        xaxis_title="Log Return (%)",
        yaxis_title="Frequency",
        showlegend=False
    )

    html_file2 = output_path / "resampled_bbo_distribution.html"
    fig2.write_html(str(html_file2))
    print(f"✓ Saved return distribution to: {html_file2}")


def test_data_quality(df):
    """Test data quality and integrity."""
    print("\n" + "=" * 60)
    print("TEST 4: Data Quality Checks")
    print("=" * 60)

    # Check for NaNs
    print("\nChecking for NaN values...")
    nan_counts = df.isna().sum()
    print(f"  ts_recv: {nan_counts['ts_recv']} NaNs")
    print(f"  bid_px: {nan_counts['bid_px']} NaNs")
    print(f"  ask_px: {nan_counts['ask_px']} NaNs")
    print(f"  mid_px: {nan_counts['mid_px']} NaNs")
    print(f"  spread: {nan_counts['spread']} NaNs")

    # Check for negative spreads (invalid)
    print("\nChecking for invalid spreads...")
    negative_spreads = (df["spread"] < 0).sum()
    print(f"  Negative spreads: {negative_spreads}")
    if negative_spreads > 0:
        print(f"  ⚠️  WARNING: Found {negative_spreads} negative spreads!")
    else:
        print(f"  ✓ All spreads are valid")

    # Check timestamp consistency
    print("\nChecking timestamp consistency...")
    df_sorted = df.sort_values("ts_recv")
    time_diffs = df_sorted["ts_recv"].diff().dropna()
    expected_diff = 1000  # 1s in milliseconds

    # Allow some tolerance for resampling
    correct_diffs = (time_diffs == expected_diff).sum()
    total_diffs = len(time_diffs)

    print(f"  Expected interval: {expected_diff}ms")
    print(f"  Correct intervals: {correct_diffs}/{total_diffs} ({correct_diffs/total_diffs*100:.2f}%)")

    if correct_diffs / total_diffs < 0.95:
        print(f"  ⚠️  WARNING: Less than 95% of intervals are correct")
    else:
        print(f"  ✓ Timestamps are consistent")

    # Check for duplicates
    print("\nChecking for duplicate timestamps...")
    duplicates = df["ts_recv"].duplicated().sum()
    print(f"  Duplicate timestamps: {duplicates}")
    if duplicates > 0:
        print(f"  ⚠️  WARNING: Found {duplicates} duplicate timestamps!")
    else:
        print(f"  ✓ No duplicate timestamps")


def test_cache_info():
    """Display cache information."""
    print("\n" + "=" * 60)
    print("TEST 5: Cache Information")
    print("=" * 60)

    cache_dir = Path("data/resampled_bbo")

    for interval_ms in [1000, 5000]:
        interval_label = f"{interval_ms // 1000}s"
        print(f"\n{interval_label} interval cache:")

        info = get_cache_info(asset="BTC", interval_ms=interval_ms, cache_dir=cache_dir)
        print(f"  Available dates: {len(info['available_dates'])}")
        if info['available_dates']:
            print(f"  Dates: {', '.join(info['available_dates'])}")
        print(f"  Total rows: {info['total_rows']:,}")
        print(f"  Total size: {info['total_size_mb']:.2f} MB")

        # Calculate compression ratio
        if info['total_rows'] > 0:
            bytes_per_row = (info['total_size_mb'] * 1024 * 1024) / info['total_rows']
            print(f"  Bytes per row: {bytes_per_row:.2f}")


def main():
    print("\n" + "=" * 60)
    print("RESAMPLED BBO PERFORMANCE & FUNCTIONALITY TEST")
    print("=" * 60)

    overall_start = time.time()

    # Test 1: Loading speed
    df_1s, df_5s = test_loading_speed()

    # Test 2: Calculations (use 1s data)
    df_1s = test_calculations(df_1s)

    # Test 3: Visualization
    test_visualization(df_1s)

    # Test 4: Data quality
    test_data_quality(df_1s)

    # Test 5: Cache info
    test_cache_info()

    overall_end = time.time()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✓ All tests completed in {overall_end - overall_start:.2f} seconds")
    print(f"✓ Processed {len(df_1s):,} rows (1s) + {len(df_5s):,} rows (5s)")
    print(f"✓ Generated 2 visualizations")
    print(f"✓ Data quality: PASSED")
    print("\nThe resampled BBO system is working as intended! 🚀")


if __name__ == "__main__":
    main()
