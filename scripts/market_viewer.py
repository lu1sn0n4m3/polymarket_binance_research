"""Interactive market viewer — Binance + Polymarket side-by-side.

Launch:
    streamlit run scripts/market_viewer.py

Requires cached resampled data in data/resampled_data/.
"""

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data import load_binance, load_binance_labels, load_polymarket_market

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ET = ZoneInfo("America/New_York")
CACHE_DIR = project_root / "data" / "resampled_data"

COLORS = {
    "pm_bid": "#22c55e",
    "pm_ask": "#ef4444",
    "pm_mid": "#3b82f6",
    "pm_microprice": "#8b5cf6",
    "bnc_mid": "#f59e0b",
    "bnc_bid": "#10b981",
    "bnc_ask": "#f43f5e",
    "K": "#ff6b00",
    "S_T": "#a855f7",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _available_dates(asset: str = "BTC") -> list[date]:
    """Return sorted list of dates that have polymarket cache files."""
    pm_dir = CACHE_DIR / "polymarket" / f"asset={asset}" / "interval=1s"
    if not pm_dir.exists():
        return []
    dates = []
    for p in sorted(pm_dir.glob("date=*.parquet")):
        try:
            d = datetime.strptime(p.stem, "date=%Y-%m-%d").date()
            dates.append(d)
        except ValueError:
            continue
    return dates


def _ts_to_dt(ts_ms: pd.Series) -> pd.Series:
    return pd.to_datetime(ts_ms, unit="ms", utc=True)


def _et_hour_to_utc(market_date: date, hour_et: int) -> tuple[datetime, datetime]:
    """Convert ET market hour to UTC start/end datetimes."""
    et_start = datetime(market_date.year, market_date.month, market_date.day,
                        hour_et, 0, 0, tzinfo=ET)
    et_end = et_start + timedelta(hours=1)
    return et_start.astimezone(timezone.utc), et_end.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def build_binance_chart(
    bnc: pd.DataFrame,
    K: float | None = None,
    S_T: float | None = None,
    Y: int | None = None,
) -> go.Figure:
    """Binance mid price chart with optional K / S_T markers."""
    df = bnc.copy()
    df["datetime"] = _ts_to_dt(df["ts_recv"])

    fig = go.Figure()

    # Mid price
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["mid_px"],
        name="Binance Mid",
        line=dict(color=COLORS["bnc_mid"], width=2),
        hovertemplate="$%{y:,.2f}<br>%{x}<extra></extra>",
    ))

    # Opening price K
    if K is not None:
        fig.add_hline(
            y=K, line_dash="dash", line_color=COLORS["K"], line_width=1.5,
            annotation_text=f"K (open) = ${K:,.2f}",
            annotation_position="top left",
            annotation_font_color=COLORS["K"],
        )

    # Closing price S_T
    if S_T is not None:
        fig.add_hline(
            y=S_T, line_dash="dot", line_color=COLORS["S_T"], line_width=1.5,
            annotation_text=f"S_T (close) = ${S_T:,.2f}",
            annotation_position="bottom left",
            annotation_font_color=COLORS["S_T"],
        )

    resolution_str = ""
    if Y is not None:
        resolution_str = " | Up" if Y == 1 else " | Down"

    fig.update_layout(
        title=f"Binance BTC Price{resolution_str}",
        height=400,
        hovermode="x unified",
        yaxis_title="Price (USD)",
        xaxis_title="Time (UTC)",
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


def build_polymarket_chart(pm: pd.DataFrame) -> go.Figure:
    """Polymarket bid / ask / mid chart."""
    df = pm.copy()
    df["datetime"] = _ts_to_dt(df["ts_recv"])

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.06,
    )

    # Bid
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["bid"],
        name="Bid", line=dict(color=COLORS["pm_bid"], width=1.2),
        hovertemplate="Bid: %{y:.3f}<extra></extra>",
    ), row=1, col=1)

    # Ask
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["ask"],
        name="Ask", line=dict(color=COLORS["pm_ask"], width=1.2),
        fill="tonexty", fillcolor="rgba(128,128,128,0.15)",
        hovertemplate="Ask: %{y:.3f}<extra></extra>",
    ), row=1, col=1)

    # Mid
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["mid"],
        name="Mid", line=dict(color=COLORS["pm_mid"], width=2),
        hovertemplate="Mid: %{y:.3f}<extra></extra>",
    ), row=1, col=1)

    # Microprice
    if "microprice" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=df["microprice"],
            name="Microprice", line=dict(color=COLORS["pm_microprice"], width=1, dash="dot"),
            hovertemplate="Microprice: %{y:.3f}<extra></extra>",
        ), row=1, col=1)

    # Spread subplot
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["spread"],
        name="Spread", line=dict(color="#6b7280", width=1),
        fill="tozeroy", fillcolor="rgba(107,114,128,0.3)",
        hovertemplate="Spread: %{y:.3f}<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        title="Polymarket Up Probability",
        height=450,
        hovermode="x unified",
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Probability", range=[0, 1], tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="Spread", row=2, col=1)
    fig.update_xaxes(title_text="Time (UTC)", row=2, col=1)

    return fig


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Market Viewer", layout="wide")
st.title("Polymarket + Binance Market Viewer")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Market Selector")

    asset = st.selectbox("Asset", ["BTC", "ETH"], index=0)

    available = _available_dates(asset)
    if not available:
        st.error(f"No cached Polymarket data found for {asset}. Run sync_resampled_cache.py first.")
        st.stop()

    selected_date = st.date_input(
        "Date",
        value=available[-1],
        min_value=available[0],
        max_value=available[-1],
    )

    hour_et = st.slider("Hour (ET)", min_value=0, max_value=23, value=10)

    interval = st.selectbox("Interval", ["1s", "500ms", "5s"], index=0)

    st.divider()
    st.caption(f"Available dates: {available[0]} to {available[-1]}")
    st.caption(f"Total cached days: {len(available)}")

# --- Load data ---
utc_start, utc_end = _et_hour_to_utc(selected_date, hour_et)

with st.spinner("Loading data..."):
    # Polymarket
    try:
        pm = load_polymarket_market(
            asset=asset,
            date=selected_date,
            hour_et=hour_et,
            interval=interval,
        )
    except Exception as e:
        pm = pd.DataFrame()
        st.warning(f"Polymarket data not available: {e}")

    # Binance (same UTC hour)
    try:
        bnc = load_binance(
            start=utc_start,
            end=utc_end,
            asset=asset,
            interval=interval,
        )
    except Exception as e:
        bnc = pd.DataFrame()
        st.warning(f"Binance data not available: {e}")

    # Labels
    try:
        labels_df = load_binance_labels(
            start=utc_start,
            end=utc_end,
            asset=asset,
        )
    except Exception as e:
        labels_df = pd.DataFrame()

# --- Find the matching label row ---
K, S_T, Y = None, None, None
utc_start_ms = int(utc_start.timestamp() * 1000)

if not labels_df.empty:
    match = labels_df[labels_df["hour_start_ms"] == utc_start_ms]
    if not match.empty:
        row = match.iloc[0]
        K = float(row["K"])
        S_T = float(row["S_T"])
        Y = int(row["Y"])

# --- Info metrics ---
cols = st.columns(5)
with cols[0]:
    st.metric("Market", f"{asset} {selected_date} {hour_et:02d}:00 ET")
with cols[1]:
    st.metric("Opening (K)", f"${K:,.2f}" if K is not None else "N/A")
with cols[2]:
    st.metric("Closing (S_T)", f"${S_T:,.2f}" if S_T is not None else "N/A")
with cols[3]:
    if K is not None and S_T is not None:
        ret = (S_T - K) / K * 100
        st.metric("Return", f"{ret:+.4f}%")
    else:
        st.metric("Return", "N/A")
with cols[4]:
    if Y is not None:
        label = "Up" if Y == 1 else "Down"
        color = "green" if Y == 1 else "red"
        st.metric("Resolution", label)
    else:
        st.metric("Resolution", "N/A")

st.divider()

# --- Charts ---
left, right = st.columns(2)

with left:
    if not bnc.empty:
        fig_bnc = build_binance_chart(bnc, K=K, S_T=S_T, Y=Y)
        st.plotly_chart(fig_bnc, use_container_width=True)
    else:
        st.info("No Binance data for this hour.")

with right:
    if not pm.empty:
        fig_pm = build_polymarket_chart(pm)
        st.plotly_chart(fig_pm, use_container_width=True)
    else:
        st.info("No Polymarket data for this hour.")

# --- Raw data preview ---
with st.expander("Raw data preview"):
    tab1, tab2, tab3 = st.tabs(["Binance BBO", "Polymarket BBO", "Labels"])
    with tab1:
        if not bnc.empty:
            st.dataframe(bnc.head(20), use_container_width=True)
        else:
            st.write("No data")
    with tab2:
        if not pm.empty:
            st.dataframe(pm.head(20), use_container_width=True)
        else:
            st.write("No data")
    with tab3:
        if not labels_df.empty:
            st.dataframe(labels_df, use_container_width=True)
        else:
            st.write("No labels")
