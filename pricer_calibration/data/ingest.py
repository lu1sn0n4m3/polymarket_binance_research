"""Load and clean Binance BBO data from S3.

Outputs a DataFrame with columns: ts_event, bid, ask, mid
sorted by ts_event, with invalid rows removed.
"""

from datetime import datetime, timezone

import pandas as pd

from src.data.loaders import load_binance_bbo


def load_binance_bbo_clean(
    start_dt: datetime,
    end_dt: datetime,
    asset: str = "BTC",
) -> pd.DataFrame:
    """Load Binance BBO, compute mid, clean, sort by ts_event.

    Args:
        start_dt: Start datetime (UTC, timezone-aware).
        end_dt: End datetime (UTC, timezone-aware).
        asset: "BTC" or "ETH".

    Returns:
        DataFrame with columns [ts_event, bid, ask, mid],
        ts_event in milliseconds, sorted ascending.
    """
    df = load_binance_bbo(start_dt, end_dt, asset=asset)

    if df.empty:
        return pd.DataFrame(columns=["ts_event", "bid", "ask", "mid"])

    df = df.rename(columns={"bid_px": "bid", "ask_px": "ask"})
    df = df[["ts_event", "bid", "ask"]].copy()

    # Drop invalid
    df = df.dropna(subset=["bid", "ask"])
    df = df[(df["bid"] > 0) & (df["ask"] > 0) & (df["bid"] <= df["ask"])]

    # Compute mid
    df["mid"] = (df["bid"] + df["ask"]) / 2.0

    # Sort
    df = df.sort_values("ts_event").reset_index(drop=True)

    return df
