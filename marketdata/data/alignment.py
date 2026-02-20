"""Time alignment utilities for joining Binance and Polymarket data."""

from typing import Literal

import numpy as np
import pandas as pd


def align_asof(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_ts_col: str = "ts_recv",
    right_ts_col: str = "ts_recv",
    direction: Literal["backward", "forward", "nearest"] = "backward",
    tolerance_ms: int | None = None,
    suffixes: tuple[str, str] = ("_left", "_right"),
) -> pd.DataFrame:
    """Align two DataFrames using ASOF join on timestamps.
    
    For each row in `left`, finds the closest matching row in `right`
    based on timestamp.
    
    Args:
        left: Left DataFrame (typically the one you want to iterate over)
        right: Right DataFrame (typically the reference data)
        left_ts_col: Timestamp column name in left DataFrame
        right_ts_col: Timestamp column name in right DataFrame
        direction: "backward" (right ts <= left ts), "forward" (right ts >= left ts),
                   or "nearest"
        tolerance_ms: Maximum time difference in milliseconds (None = no limit)
        suffixes: Suffixes for overlapping column names
        
    Returns:
        Merged DataFrame with all left rows and matched right columns
    """
    # Ensure sorted
    left_sorted = left.sort_values(left_ts_col).reset_index(drop=True)
    right_sorted = right.sort_values(right_ts_col).reset_index(drop=True)
    
    # Rename timestamp column in right if different
    if left_ts_col != right_ts_col:
        right_sorted = right_sorted.rename(columns={right_ts_col: left_ts_col})
    
    tolerance = pd.Timedelta(milliseconds=tolerance_ms) if tolerance_ms else None
    
    # Convert to datetime for merge_asof if needed
    left_sorted["_merge_ts"] = pd.to_datetime(left_sorted[left_ts_col], unit="ms")
    right_sorted["_merge_ts"] = pd.to_datetime(right_sorted[left_ts_col], unit="ms")
    
    result = pd.merge_asof(
        left_sorted,
        right_sorted,
        on="_merge_ts",
        direction=direction,
        tolerance=tolerance,
        suffixes=suffixes,
    )
    
    # Clean up
    result = result.drop(columns=["_merge_ts"])
    
    return result


def align_bucketed(
    left: pd.DataFrame,
    right: pd.DataFrame,
    bucket_ms: int = 1000,
    left_ts_col: str = "ts_recv",
    right_ts_col: str = "ts_recv",
    agg_method: Literal["last", "first", "mean"] = "last",
    suffixes: tuple[str, str] = ("_left", "_right"),
) -> pd.DataFrame:
    """Align two DataFrames by aggregating into time buckets.
    
    Both DataFrames are aggregated to fixed time buckets, then inner joined.
    
    Args:
        left: Left DataFrame
        right: Right DataFrame
        bucket_ms: Bucket size in milliseconds (default 1000 = 1 second)
        left_ts_col: Timestamp column name in left DataFrame
        right_ts_col: Timestamp column name in right DataFrame
        agg_method: How to aggregate within buckets ("last", "first", "mean")
        suffixes: Suffixes for overlapping column names
        
    Returns:
        Merged DataFrame with one row per bucket (inner join)
    """
    def bucket_df(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
        df = df.copy()
        df["_bucket"] = (df[ts_col] // bucket_ms) * bucket_ms
        
        # Determine aggregation - exclude _bucket from aggregation
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() 
                        if c != "_bucket"]
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols and c != "_bucket"]
        
        agg_dict = {}
        for col in numeric_cols:
            agg_dict[col] = agg_method
        for col in non_numeric_cols:
            agg_dict[col] = "first"  # Always take first for non-numeric
        
        result = df.groupby("_bucket", as_index=False).agg(agg_dict)
        return result
    
    left_bucketed = bucket_df(left, left_ts_col)
    right_bucketed = bucket_df(right, right_ts_col)
    
    # Join on bucket
    result = pd.merge(
        left_bucketed,
        right_bucketed,
        on="_bucket",
        how="inner",
        suffixes=suffixes,
    )
    
    # Rename bucket to a sensible name
    result = result.rename(columns={"_bucket": "ts_bucket"})
    
    return result.sort_values("ts_bucket").reset_index(drop=True)


def resample_to_grid(
    df: pd.DataFrame,
    grid_ms: int = 100,
    ts_col: str = "ts_recv",
    start_ms: int | None = None,
    end_ms: int | None = None,
    method: Literal["ffill", "bfill", "nearest"] = "ffill",
) -> pd.DataFrame:
    """Resample DataFrame to a fixed time grid.
    
    Creates a regular time grid and fills values using forward-fill,
    back-fill, or nearest neighbor.
    
    Args:
        df: Input DataFrame
        grid_ms: Grid spacing in milliseconds (default 100ms)
        ts_col: Timestamp column name
        start_ms: Start of grid (default: first timestamp in df)
        end_ms: End of grid (default: last timestamp in df)
        method: Fill method ("ffill", "bfill", "nearest")
        
    Returns:
        DataFrame resampled to regular grid
    """
    if df.empty:
        return df.copy()
    
    if start_ms is None:
        start_ms = int(df[ts_col].min())
    if end_ms is None:
        end_ms = int(df[ts_col].max())
    
    # Align to grid boundaries
    start_ms = (start_ms // grid_ms) * grid_ms
    end_ms = ((end_ms // grid_ms) + 1) * grid_ms
    
    # Create grid
    grid = pd.DataFrame({ts_col: np.arange(start_ms, end_ms + 1, grid_ms)})
    
    # Sort input and use merge_asof for proper alignment
    df_sorted = df.sort_values(ts_col).copy()
    
    # Convert to datetime for merge_asof
    grid["_dt"] = pd.to_datetime(grid[ts_col], unit="ms")
    df_sorted["_dt"] = pd.to_datetime(df_sorted[ts_col], unit="ms")
    
    # Use merge_asof to get latest value at or before each grid point
    result = pd.merge_asof(
        grid,
        df_sorted.drop(columns=[ts_col]),  # Avoid duplicate ts_col
        on="_dt",
        direction="backward",
    )
    
    # Clean up
    result = result.drop(columns=["_dt"])
    
    # merge_asof with direction="backward" is equivalent to ffill
    # For bfill or nearest, we need additional processing
    if method == "bfill":
        # Re-do with forward direction
        result = pd.merge_asof(
            grid.assign(_dt=pd.to_datetime(grid[ts_col], unit="ms")),
            df_sorted.assign(_dt=pd.to_datetime(df_sorted[ts_col], unit="ms")).drop(columns=[ts_col]),
            on="_dt",
            direction="forward",
        ).drop(columns=["_dt"])
    elif method == "nearest":
        # Use nearest direction
        result = pd.merge_asof(
            grid.assign(_dt=pd.to_datetime(grid[ts_col], unit="ms")),
            df_sorted.assign(_dt=pd.to_datetime(df_sorted[ts_col], unit="ms")).drop(columns=[ts_col]),
            on="_dt",
            direction="nearest",
        ).drop(columns=["_dt"])
    
    return result


def compute_derived_fields(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Add derived fields (mid, spread, microprice) to a BBO DataFrame.
    
    Args:
        df: DataFrame with bid_px, ask_px, bid_sz, ask_sz columns
                      OR {prefix}bid, {prefix}ask, {prefix}bid_sz, {prefix}ask_sz columns
        prefix: Prefix for new column names (e.g., "pm_" or "bnc_")
        
    Returns:
        DataFrame with added columns: {prefix}mid, {prefix}spread, {prefix}microprice
    """
    df = df.copy()
    
    # Try both naming conventions: bid_px/ask_px or {prefix}bid/{prefix}ask
    if prefix:
        # Check for {prefix}bid (e.g., pm_bid) first
        if f"{prefix}bid" in df.columns:
            bid_col = f"{prefix}bid"
            ask_col = f"{prefix}ask"
            bid_sz_col = f"{prefix}bid_sz"
            ask_sz_col = f"{prefix}ask_sz"
        else:
            bid_col = f"{prefix}bid_px"
            ask_col = f"{prefix}ask_px"
            bid_sz_col = f"{prefix}bid_sz"
            ask_sz_col = f"{prefix}ask_sz"
    else:
        bid_col = "bid_px"
        ask_col = "ask_px"
        bid_sz_col = "bid_sz"
        ask_sz_col = "ask_sz"
    
    # Check if columns exist
    if bid_col not in df.columns or ask_col not in df.columns:
        return df
    
    # Mid price
    df[f"{prefix}mid"] = (df[bid_col] + df[ask_col]) / 2
    
    # Spread
    df[f"{prefix}spread"] = df[ask_col] - df[bid_col]
    
    # Microprice (size-weighted mid)
    if bid_sz_col in df.columns and ask_sz_col in df.columns:
        total_size = df[bid_sz_col] + df[ask_sz_col]
        # Weight towards the side with less size (tighter quote)
        df[f"{prefix}microprice"] = np.where(
            total_size > 0,
            (df[bid_col] * df[ask_sz_col] + df[ask_col] * df[bid_sz_col]) / total_size,
            df[f"{prefix}mid"]
        )
    
    return df
