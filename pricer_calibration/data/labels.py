"""Build hourly contract labels from Binance trades.

K = price of first trade with ts_event >= hour_start
S_T = price of last trade with ts_event < hour_end
Y = 1{S_T > K}
"""

from datetime import datetime, timedelta

import pandas as pd

from src.data.loaders import load_binance_trades


def build_hourly_labels(
    start_dt: datetime,
    end_dt: datetime,
    asset: str = "BTC",
) -> pd.DataFrame:
    """Build contract labels for each complete hour in the range.

    Args:
        start_dt: Start datetime (UTC, timezone-aware).
        end_dt: End datetime (UTC, timezone-aware).
        asset: "BTC" or "ETH".

    Returns:
        DataFrame with columns:
        [market_id, hour_start_ms, hour_end_ms, K, S_T, Y]
        where hour_start/end are in epoch ms.
    """
    trades = load_binance_trades(start_dt, end_dt, asset=asset)

    if trades.empty:
        return pd.DataFrame(
            columns=["market_id", "hour_start_ms", "hour_end_ms", "K", "S_T", "Y"]
        )

    trades = trades[["ts_event", "price"]].sort_values("ts_event").reset_index(drop=True)

    # Iterate over each complete hour
    current = start_dt.replace(minute=0, second=0, microsecond=0)
    if current < start_dt:
        current += timedelta(hours=1)

    rows = []
    while current + timedelta(hours=1) <= end_dt:
        hour_start_ms = int(current.timestamp() * 1000)
        hour_end_ms = int((current + timedelta(hours=1)).timestamp() * 1000)

        # Trades within [hour_start, hour_end)
        mask = (trades["ts_event"] >= hour_start_ms) & (trades["ts_event"] < hour_end_ms)
        hour_trades = trades[mask]

        if len(hour_trades) >= 2:
            K = float(hour_trades.iloc[0]["price"])
            S_T = float(hour_trades.iloc[-1]["price"])
            Y = 1 if S_T > K else 0

            market_id = f"{asset}_{current.strftime('%Y%m%d_%H')}"
            rows.append({
                "market_id": market_id,
                "hour_start_ms": hour_start_ms,
                "hour_end_ms": hour_end_ms,
                "K": K,
                "S_T": S_T,
                "Y": Y,
            })

        current += timedelta(hours=1)

    return pd.DataFrame(rows)
