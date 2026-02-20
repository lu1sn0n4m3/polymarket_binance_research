"""Visualize Binance bid/ask/mid prices using the resampled data framework.

Quick visualization to verify the cache migration and test the new API.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from marketdata.data import load_binance
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualize_prices(
    start_date: str,
    end_date: str,
    asset: str = "BTC",
    interval: str = "1s",
):
    """Visualize bid/ask/mid prices for a date range.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        asset: "BTC" or "ETH"
        interval: "500ms", "1s", or "5s"
    """
    print("=" * 70)
    print(f"VISUALIZING {asset} PRICES")
    print("=" * 70)
    print(f"Date range: {start_date} to {end_date}")
    print(f"Interval: {interval}")
    print()

    # Load data using new framework
    print("Loading data from cache...")
    bnc = load_binance(
        start=start_date,
        end=end_date,
        asset=asset,
        interval=interval,
        columns=["ts_recv", "bid_px", "ask_px", "mid_px", "spread"],
    )

    if bnc.empty:
        print("❌ No data loaded. Check cache status with:")
        print(f'   python -c "from marketdata.data import get_cache_status; print(get_cache_status(\\"binance\\", \\"{asset}\\", \\"{interval}\\"))"')
        return

    print(f"✓ Loaded {len(bnc):,} rows")
    print(f"  Date range: {bnc['ts_recv'].min()} to {bnc['ts_recv'].max()} (epoch ms)")
    print(f"  Price range: ${bnc['mid_px'].min():.2f} - ${bnc['mid_px'].max():.2f}")
    print()

    # Convert timestamps to datetime for plotting
    bnc["timestamp"] = bnc["ts_recv"].apply(lambda x: datetime.fromtimestamp(x / 1000, tz=timezone.utc))

    # Downsample for visualization if too many points
    max_points = 100_000
    if len(bnc) > max_points:
        print(f"⚠️  Dataset has {len(bnc):,} rows, downsampling to {max_points:,} for visualization...")
        step = len(bnc) // max_points
        bnc_plot = bnc.iloc[::step].copy()
        print(f"✓ Downsampled to {len(bnc_plot):,} rows (every {step} rows)")
    else:
        bnc_plot = bnc

    # Create figure with subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            f"{asset}/USD Prices (Bid/Ask/Mid)",
            "Spread (Ask - Bid)"
        ),
        row_heights=[0.7, 0.3],
    )

    # Top subplot: Bid/Ask/Mid prices
    fig.add_trace(
        go.Scatter(
            x=bnc_plot["timestamp"],
            y=bnc_plot["bid_px"],
            mode="lines",
            name="Bid",
            line=dict(color="green", width=1),
            hovertemplate="Bid: $%{y:.2f}<br>%{x}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=bnc_plot["timestamp"],
            y=bnc_plot["ask_px"],
            mode="lines",
            name="Ask",
            line=dict(color="red", width=1),
            hovertemplate="Ask: $%{y:.2f}<br>%{x}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=bnc_plot["timestamp"],
            y=bnc_plot["mid_px"],
            mode="lines",
            name="Mid",
            line=dict(color="blue", width=1.5),
            hovertemplate="Mid: $%{y:.2f}<br>%{x}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Bottom subplot: Spread
    fig.add_trace(
        go.Scatter(
            x=bnc_plot["timestamp"],
            y=bnc_plot["spread"],
            mode="lines",
            name="Spread",
            line=dict(color="orange", width=1),
            fill="tozeroy",
            hovertemplate="Spread: $%{y:.2f}<br>%{x}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{asset}/USD Binance BBO - {start_date} to {end_date} ({interval} interval)",
            x=0.5,
            xanchor="center",
        ),
        height=800,
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    # Update axes
    fig.update_xaxes(title_text="Time (UTC)", row=2, col=1)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Spread (USD)", row=2, col=1)

    # Add range slider
    fig.update_xaxes(rangeslider_visible=False)

    # Show figure
    print("Opening interactive plot in browser...")
    fig.show()

    # Print statistics
    print()
    print("=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print(f"Rows: {len(bnc):,}")
    print(f"Time span: {(bnc['ts_recv'].max() - bnc['ts_recv'].min()) / 1000 / 3600:.1f} hours")
    print()
    print("Prices:")
    print(f"  Min mid: ${bnc['mid_px'].min():.2f}")
    print(f"  Max mid: ${bnc['mid_px'].max():.2f}")
    print(f"  Mean mid: ${bnc['mid_px'].mean():.2f}")
    print(f"  Std dev: ${bnc['mid_px'].std():.2f}")
    print()
    print("Spread:")
    print(f"  Min: ${bnc['spread'].min():.2f}")
    print(f"  Max: ${bnc['spread'].max():.2f}")
    print(f"  Mean: ${bnc['spread'].mean():.2f}")
    print(f"  Median: ${bnc['spread'].median():.2f}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize Binance prices using resampled cache"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2026-01-23",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2026-01-26",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--asset",
        type=str,
        default="BTC",
        choices=["BTC", "ETH"],
        help="Asset to visualize",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1s",
        choices=["500ms", "1s", "5s"],
        help="Resampling interval",
    )

    args = parser.parse_args()

    visualize_prices(
        start_date=args.start,
        end_date=args.end,
        asset=args.asset,
        interval=args.interval,
    )


if __name__ == "__main__":
    main()
