"""Build hourly contract labels from cached Binance trade-based labels.

K = price of first trade with ts_event >= hour_start
S_T = price of last trade with ts_event < hour_end
Y = 1{S_T > K}

Uses the project's standard label cache (data/resampled_data/binance_labels/).
On first access, labels are fetched from S3 and cached as lightweight daily
parquet files (~24 rows per day).  Subsequent calls read from cache.
"""

from datetime import datetime

import pandas as pd

from src.data import load_binance_labels


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
    labels = load_binance_labels(start=start_dt, end=end_dt, asset=asset)

    if labels.empty:
        return pd.DataFrame(
            columns=["market_id", "hour_start_ms", "hour_end_ms", "K", "S_T", "Y"]
        )

    # Add market_id column (matches pricer_calibration convention)
    labels["market_id"] = labels["hour_start_ms"].apply(
        lambda ms: f"{asset}_{datetime.utcfromtimestamp(ms / 1000).strftime('%Y%m%d_%H')}"
    )

    return labels[["market_id", "hour_start_ms", "hour_end_ms", "K", "S_T", "Y"]].reset_index(drop=True)
