"""Order book visualization utilities."""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from src.data.session import HourlyMarketSession


# Colors
COLORS = {
    "bid": "#22c55e",
    "ask": "#ef4444",
    "bid_fill": "rgba(34, 197, 94, 0.3)",
    "ask_fill": "rgba(239, 68, 68, 0.3)",
    "mid": "#3b82f6",
}


def plot_book_snapshot(
    bid_prices: list[float],
    bid_sizes: list[float],
    ask_prices: list[float],
    ask_sizes: list[float],
    title: str = "Order Book Snapshot",
    max_levels: int = 20,
    height: int = 400,
) -> go.Figure:
    """Plot a single order book snapshot.
    
    Args:
        bid_prices: Bid prices (best first)
        bid_sizes: Bid sizes
        ask_prices: Ask prices (best first)
        ask_sizes: Ask sizes
        title: Chart title
        max_levels: Maximum levels to show
        height: Chart height
        
    Returns:
        Plotly Figure with horizontal bar chart
    """
    # Limit levels
    bid_prices = bid_prices[:max_levels]
    bid_sizes = bid_sizes[:max_levels]
    ask_prices = ask_prices[:max_levels]
    ask_sizes = ask_sizes[:max_levels]
    
    fig = go.Figure()
    
    # Bids (positive size, lower prices)
    fig.add_trace(
        go.Bar(
            y=[f"{p:.4f}" for p in bid_prices],
            x=bid_sizes,
            orientation="h",
            name="Bids",
            marker_color=COLORS["bid"],
            hovertemplate="Price: %{y}<br>Size: %{x:,.2f}<extra>Bid</extra>",
        )
    )
    
    # Asks (negative size for visual separation, higher prices)
    fig.add_trace(
        go.Bar(
            y=[f"{p:.4f}" for p in ask_prices],
            x=[-s for s in ask_sizes],
            orientation="h",
            name="Asks",
            marker_color=COLORS["ask"],
            hovertemplate="Price: %{y}<br>Size: %{customdata:,.2f}<extra>Ask</extra>",
            customdata=ask_sizes,
        )
    )
    
    fig.update_layout(
        title=title,
        height=height,
        barmode="overlay",
        xaxis=dict(title="Size (negative = asks)", zeroline=True),
        yaxis=dict(title="Price", categoryorder="array", categoryarray=[
            f"{p:.4f}" for p in sorted(set(bid_prices + ask_prices))
        ]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    return fig


def plot_book_depth(
    bid_prices: list[float],
    bid_sizes: list[float],
    ask_prices: list[float],
    ask_sizes: list[float],
    title: str = "Order Book Depth",
    height: int = 400,
) -> go.Figure:
    """Plot cumulative order book depth (market depth chart).
    
    Args:
        bid_prices: Bid prices (best first)
        bid_sizes: Bid sizes
        ask_prices: Ask prices (best first)
        ask_sizes: Ask sizes
        title: Chart title
        height: Chart height
        
    Returns:
        Plotly Figure with depth chart
    """
    # Cumulative sizes
    bid_cumsum = np.cumsum(bid_sizes)
    ask_cumsum = np.cumsum(ask_sizes)
    
    fig = go.Figure()
    
    # Bids - cumulative from best bid downward
    fig.add_trace(
        go.Scatter(
            x=bid_prices,
            y=bid_cumsum,
            name="Bid Depth",
            fill="tozeroy",
            fillcolor=COLORS["bid_fill"],
            line=dict(color=COLORS["bid"], width=2),
            hovertemplate="Price: %{x:.4f}<br>Cumulative: %{y:,.2f}<extra>Bids</extra>",
        )
    )
    
    # Asks - cumulative from best ask upward
    fig.add_trace(
        go.Scatter(
            x=ask_prices,
            y=ask_cumsum,
            name="Ask Depth",
            fill="tozeroy",
            fillcolor=COLORS["ask_fill"],
            line=dict(color=COLORS["ask"], width=2),
            hovertemplate="Price: %{x:.4f}<br>Cumulative: %{y:,.2f}<extra>Asks</extra>",
        )
    )
    
    # Mid price line
    if len(bid_prices) > 0 and len(ask_prices) > 0:
        mid = (bid_prices[0] + ask_prices[0]) / 2
        max_depth = max(max(bid_cumsum) if bid_cumsum.size > 0 else 0,
                        max(ask_cumsum) if ask_cumsum.size > 0 else 0)
        fig.add_vline(
            x=mid,
            line_dash="dash",
            line_color=COLORS["mid"],
            annotation_text=f"Mid: {mid:.4f}",
        )
    
    fig.update_layout(
        title=title,
        height=height,
        xaxis=dict(title="Price"),
        yaxis=dict(title="Cumulative Size"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    return fig


def plot_book_depth_over_time(
    book_df: pd.DataFrame,
    depth: int = 5,
    ts_col: str = "ts_recv",
    sample_interval: int = 10,
    title: str = "Book Depth Over Time",
    height: int = 500,
) -> go.Figure:
    """Plot book depth metrics over time.
    
    Args:
        book_df: DataFrame with bid_prices, bid_sizes, ask_prices, ask_sizes columns
        depth: Number of levels to sum for depth
        ts_col: Timestamp column
        sample_interval: Sample every N rows (for performance)
        title: Chart title
        height: Chart height
        
    Returns:
        Plotly Figure
    """
    df = book_df.iloc[::sample_interval].copy()
    df["datetime"] = pd.to_datetime(df[ts_col], unit="ms", utc=True)
    
    # Compute depth metrics
    bid_depths = []
    ask_depths = []
    spreads = []
    imbalances = []
    
    for _, row in df.iterrows():
        bid_prices = row.get("bid_prices", [])
        bid_sizes = row.get("bid_sizes", [])
        ask_prices = row.get("ask_prices", [])
        ask_sizes = row.get("ask_sizes", [])
        
        bid_depth = sum(bid_sizes[:depth]) if len(bid_sizes) > 0 else 0
        ask_depth = sum(ask_sizes[:depth]) if len(ask_sizes) > 0 else 0
        
        bid_depths.append(bid_depth)
        ask_depths.append(ask_depth)
        
        if len(bid_prices) > 0 and len(ask_prices) > 0:
            spreads.append(ask_prices[0] - bid_prices[0])
        else:
            spreads.append(np.nan)
        
        total = bid_depth + ask_depth
        if total > 0:
            imbalances.append((bid_depth - ask_depth) / total)
        else:
            imbalances.append(0)
    
    df["bid_depth"] = bid_depths
    df["ask_depth"] = ask_depths
    df["spread"] = spreads
    df["imbalance"] = imbalances
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.4, 0.3, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=["Depth (Top 5 Levels)", "Spread", "Imbalance"],
    )
    
    # Depth
    fig.add_trace(
        go.Scatter(
            x=df["datetime"], y=df["bid_depth"],
            name="Bid Depth", line=dict(color=COLORS["bid"]),
            fill="tozeroy", fillcolor=COLORS["bid_fill"],
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["datetime"], y=df["ask_depth"],
            name="Ask Depth", line=dict(color=COLORS["ask"]),
            fill="tozeroy", fillcolor=COLORS["ask_fill"],
        ),
        row=1, col=1,
    )
    
    # Spread
    fig.add_trace(
        go.Scatter(
            x=df["datetime"], y=df["spread"],
            name="Spread", line=dict(color="#6b7280"),
        ),
        row=2, col=1,
    )
    
    # Imbalance
    fig.add_trace(
        go.Scatter(
            x=df["datetime"], y=df["imbalance"],
            name="Imbalance", line=dict(color="#8b5cf6"),
            fill="tozeroy",
        ),
        row=3, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
    
    fig.update_layout(
        title=title,
        height=height,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    return fig


def animate_book(
    book_df: pd.DataFrame,
    frame_interval: int = 10,
    max_levels: int = 15,
    title: str = "Order Book Animation",
    height: int = 500,
) -> go.Figure:
    """Create an animated order book visualization.
    
    Args:
        book_df: DataFrame with book snapshots
        frame_interval: Use every Nth snapshot as a frame
        max_levels: Maximum levels to display
        title: Chart title
        height: Chart height
        
    Returns:
        Plotly Figure with animation
    """
    df = book_df.iloc[::frame_interval].reset_index(drop=True)
    
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)
        return fig
    
    frames = []
    
    # Get price range for consistent axis
    all_prices = []
    for _, row in df.iterrows():
        all_prices.extend(row.get("bid_prices", [])[:max_levels])
        all_prices.extend(row.get("ask_prices", [])[:max_levels])
    
    if not all_prices:
        fig = go.Figure()
        fig.add_annotation(text="No price data", x=0.5, y=0.5, showarrow=False)
        return fig
    
    price_min, price_max = min(all_prices), max(all_prices)
    
    # Create frames
    for idx, row in df.iterrows():
        bid_prices = row.get("bid_prices", [])[:max_levels]
        bid_sizes = row.get("bid_sizes", [])[:max_levels]
        ask_prices = row.get("ask_prices", [])[:max_levels]
        ask_sizes = row.get("ask_sizes", [])[:max_levels]
        
        ts = row.get("ts_recv", idx)
        dt = pd.to_datetime(ts, unit="ms", utc=True)
        
        frame = go.Frame(
            data=[
                go.Bar(
                    x=bid_prices,
                    y=bid_sizes,
                    name="Bids",
                    marker_color=COLORS["bid"],
                ),
                go.Bar(
                    x=ask_prices,
                    y=ask_sizes,
                    name="Asks",
                    marker_color=COLORS["ask"],
                ),
            ],
            name=str(idx),
            layout=go.Layout(title=f"{title} - {dt.strftime('%H:%M:%S')}"),
        )
        frames.append(frame)
    
    # Initial frame
    first_row = df.iloc[0]
    
    fig = go.Figure(
        data=[
            go.Bar(
                x=first_row.get("bid_prices", [])[:max_levels],
                y=first_row.get("bid_sizes", [])[:max_levels],
                name="Bids",
                marker_color=COLORS["bid"],
            ),
            go.Bar(
                x=first_row.get("ask_prices", [])[:max_levels],
                y=first_row.get("ask_sizes", [])[:max_levels],
                name="Asks",
                marker_color=COLORS["ask"],
            ),
        ],
        frames=frames,
    )
    
    # Animation controls
    fig.update_layout(
        title=title,
        height=height,
        xaxis=dict(title="Price", range=[price_min - 0.01, price_max + 0.01]),
        yaxis=dict(title="Size"),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.15,
                x=0.5,
                xanchor="center",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=200, redraw=True),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                            ),
                        ],
                    ),
                ],
            ),
        ],
        sliders=[
            dict(
                active=0,
                steps=[
                    dict(
                        args=[
                            [str(i)],
                            dict(
                                frame=dict(duration=0, redraw=True),
                                mode="immediate",
                            ),
                        ],
                        label=str(i),
                        method="animate",
                    )
                    for i in range(len(frames))
                ],
                x=0.1,
                len=0.8,
                xanchor="left",
                y=-0.1,
                yanchor="top",
            ),
        ],
    )
    
    return fig
