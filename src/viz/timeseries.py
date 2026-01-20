"""Time series visualization for market data."""

from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from src.data.session import HourlyMarketSession


# Color palette
COLORS = {
    "pm_bid": "#22c55e",       # Green
    "pm_ask": "#ef4444",       # Red
    "pm_mid": "#3b82f6",       # Blue
    "pm_microprice": "#8b5cf6", # Purple
    "bnc_mid": "#f59e0b",      # Amber
    "bnc_bid": "#10b981",      # Emerald
    "bnc_ask": "#f43f5e",      # Rose
    "spread": "#6b7280",       # Gray
}


def _ts_to_datetime(ts_ms: pd.Series) -> pd.Series:
    """Convert millisecond timestamps to datetime."""
    return pd.to_datetime(ts_ms, unit="ms", utc=True)


def plot_session(
    session: "HourlyMarketSession",
    pm_fields: list[str] | None = None,
    bnc_fields: list[str] | None = None,
    show_outcome: bool = True,
    title: str | None = None,
    height: int = 600,
) -> go.Figure:
    """Plot a complete session with dual-axis (Polymarket + Binance).
    
    Args:
        session: HourlyMarketSession to plot
        pm_fields: Polymarket fields to plot (default: bid, ask, mid, microprice)
        bnc_fields: Binance fields to plot (default: mid)
        show_outcome: Show vertical line and outcome at market close
        title: Chart title (auto-generated if None)
        height: Chart height in pixels
        
    Returns:
        Plotly Figure
    """
    if pm_fields is None:
        pm_fields = ["pm_bid", "pm_ask", "pm_mid", "pm_microprice"]
    if bnc_fields is None:
        bnc_fields = ["bnc_mid"]
    
    df = session.aligned
    
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Convert timestamps
    df = df.copy()
    df["datetime"] = _ts_to_datetime(df["ts_recv"])
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]],
    )
    
    # Plot Polymarket fields (left y-axis)
    for field in pm_fields:
        if field in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["datetime"],
                    y=df[field],
                    name=field.replace("pm_", "PM ").replace("_", " ").title(),
                    line=dict(color=COLORS.get(field, "#888888"), width=1.5),
                    hovertemplate=f"{field}: %{{y:.4f}}<br>%{{x}}<extra></extra>",
                ),
                secondary_y=False,
            )
    
    # Plot Binance fields (right y-axis)
    for field in bnc_fields:
        if field in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["datetime"],
                    y=df[field],
                    name=field.replace("bnc_", "BNC ").replace("_", " ").title(),
                    line=dict(color=COLORS.get(field, "#888888"), width=2, dash="dot"),
                    hovertemplate=f"{field}: $%{{y:,.2f}}<br>%{{x}}<extra></extra>",
                ),
                secondary_y=True,
            )
    
    # Add outcome annotation
    if show_outcome and session.outcome:
        outcome = session.outcome
        close_dt = pd.to_datetime(outcome.close_ts, unit="ms", utc=True)
        
        # Vertical line at close (use shape instead of vline to avoid Timestamp issues)
        fig.add_shape(
            type="line",
            x0=close_dt,
            x1=close_dt,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(dash="dash", color="gray", width=1),
        )
        
        # Add annotation separately
        fig.add_annotation(
            x=close_dt,
            y=1,
            yref="paper",
            text=f"Close: {outcome.outcome.upper()} ({outcome.return_pct:+.2f}%)",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(size=10, color="gray"),
        )
    
    # Update layout
    if title is None:
        token_info = ""
        if session.token_is_up is not None:
            token_info = " (normalized to Up)" if not session.token_is_up else ""
        title = f"{session.asset} {session.market_date} {session.hour_et}:00 ET{token_info}"
    
    fig.update_layout(
        title=title,
        height=height,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(title="Time (UTC)"),
    )
    
    fig.update_yaxes(
        title_text="Polymarket Probability",
        secondary_y=False,
        range=[0, 1],
        tickformat=".0%",
    )
    fig.update_yaxes(
        title_text="Binance Price ($)",
        secondary_y=True,
    )
    
    return fig


def plot_aligned_prices(
    df: pd.DataFrame,
    ts_col: str = "ts_recv",
    pm_cols: list[str] | None = None,
    bnc_cols: list[str] | None = None,
    title: str = "Aligned Prices",
    height: int = 500,
) -> go.Figure:
    """Plot aligned price data from a DataFrame.
    
    More flexible than plot_session - works with any aligned DataFrame.
    
    Args:
        df: Aligned DataFrame with timestamp and price columns
        ts_col: Timestamp column name
        pm_cols: Polymarket columns to plot
        bnc_cols: Binance columns to plot
        title: Chart title
        height: Chart height
        
    Returns:
        Plotly Figure
    """
    if pm_cols is None:
        pm_cols = [c for c in df.columns if c.startswith("pm_") and "sz" not in c]
    if bnc_cols is None:
        bnc_cols = [c for c in df.columns if c.startswith("bnc_") and "sz" not in c]
    
    df = df.copy()
    df["datetime"] = _ts_to_datetime(df[ts_col])
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    for col in pm_cols:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["datetime"],
                    y=df[col],
                    name=col,
                    line=dict(color=COLORS.get(col, "#888"), width=1.5),
                ),
                secondary_y=False,
            )
    
    for col in bnc_cols:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["datetime"],
                    y=df[col],
                    name=col,
                    line=dict(color=COLORS.get(col, "#888"), width=2, dash="dot"),
                ),
                secondary_y=True,
            )
    
    fig.update_layout(
        title=title,
        height=height,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Probability", secondary_y=False)
    fig.update_yaxes(title_text="Price ($)", secondary_y=True)
    
    return fig


def plot_polymarket_bbo(
    df: pd.DataFrame,
    ts_col: str = "ts_recv",
    show_spread: bool = True,
    title: str = "Polymarket BBO",
    height: int = 400,
) -> go.Figure:
    """Plot Polymarket BBO data.
    
    Args:
        df: DataFrame with bid_px, ask_px columns
        ts_col: Timestamp column name
        show_spread: Show spread as a secondary trace
        title: Chart title
        height: Chart height
        
    Returns:
        Plotly Figure
    """
    df = df.copy()
    df["datetime"] = _ts_to_datetime(df[ts_col])
    
    if show_spread:
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True,
            vertical_spacing=0.05,
        )
    else:
        fig = go.Figure()
    
    # Bid/Ask
    bid_col = "pm_bid" if "pm_bid" in df.columns else "bid_px"
    ask_col = "pm_ask" if "pm_ask" in df.columns else "ask_px"
    
    row = 1 if show_spread else None
    
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df[bid_col],
            name="Bid",
            line=dict(color=COLORS["pm_bid"], width=1.5),
            fill=None,
        ),
        row=row, col=1 if show_spread else None,
    )
    
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df[ask_col],
            name="Ask",
            line=dict(color=COLORS["pm_ask"], width=1.5),
            fill="tonexty",
            fillcolor="rgba(128, 128, 128, 0.2)",
        ),
        row=row, col=1 if show_spread else None,
    )
    
    # Mid
    mid = (df[bid_col] + df[ask_col]) / 2
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=mid,
            name="Mid",
            line=dict(color=COLORS["pm_mid"], width=2),
        ),
        row=row, col=1 if show_spread else None,
    )
    
    # Spread
    if show_spread:
        spread = df[ask_col] - df[bid_col]
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=spread,
                name="Spread",
                line=dict(color=COLORS["spread"], width=1),
                fill="tozeroy",
                fillcolor="rgba(107, 114, 128, 0.3)",
            ),
            row=2, col=1,
        )
        fig.update_yaxes(title_text="Spread", row=2, col=1)
    
    fig.update_layout(
        title=title,
        height=height,
        hovermode="x unified",
    )
    
    if show_spread:
        fig.update_yaxes(title_text="Probability", row=1, col=1)
    else:
        fig.update_yaxes(title_text="Probability")
    
    return fig


def plot_binance_bbo(
    df: pd.DataFrame,
    ts_col: str = "ts_recv",
    show_spread: bool = True,
    title: str = "Binance BBO",
    height: int = 400,
) -> go.Figure:
    """Plot Binance BBO data.
    
    Args:
        df: DataFrame with bid_px, ask_px columns
        ts_col: Timestamp column name
        show_spread: Show spread as a secondary trace
        title: Chart title
        height: Chart height
        
    Returns:
        Plotly Figure
    """
    df = df.copy()
    df["datetime"] = _ts_to_datetime(df[ts_col])
    
    if show_spread:
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True,
            vertical_spacing=0.05,
        )
    else:
        fig = go.Figure()
    
    bid_col = "bnc_bid" if "bnc_bid" in df.columns else "bid_px"
    ask_col = "bnc_ask" if "bnc_ask" in df.columns else "ask_px"
    
    row = 1 if show_spread else None
    
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df[bid_col],
            name="Bid",
            line=dict(color=COLORS["bnc_bid"], width=1),
        ),
        row=row, col=1 if show_spread else None,
    )
    
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df[ask_col],
            name="Ask",
            line=dict(color=COLORS["bnc_ask"], width=1),
        ),
        row=row, col=1 if show_spread else None,
    )
    
    # Mid
    mid = (df[bid_col] + df[ask_col]) / 2
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=mid,
            name="Mid",
            line=dict(color=COLORS["bnc_mid"], width=2),
        ),
        row=row, col=1 if show_spread else None,
    )
    
    if show_spread:
        spread = df[ask_col] - df[bid_col]
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=spread,
                name="Spread",
                line=dict(color=COLORS["spread"], width=1),
                fill="tozeroy",
            ),
            row=2, col=1,
        )
        fig.update_yaxes(title_text="Spread ($)", row=2, col=1)
    
    fig.update_layout(
        title=title,
        height=height,
        hovermode="x unified",
    )
    
    if show_spread:
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    else:
        fig.update_yaxes(title_text="Price ($)")
    
    return fig
