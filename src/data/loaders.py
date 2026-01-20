"""Raw data loading functions for Binance and Polymarket."""

from datetime import datetime, timedelta
from typing import Literal

import duckdb
import pandas as pd

from src.config import get_config
from src.data.connection import get_connection


def _build_s3_path(
    venue: str,
    stream_id: str,
    event_type: str,
    date: datetime,
    hour: int,
) -> str:
    """Build S3 path for a specific hour partition."""
    config = get_config()
    date_str = date.strftime("%Y-%m-%d")
    return (
        f"s3://{config.s3.bucket}/{config.s3.prefix}/"
        f"venue={venue}/stream_id={stream_id}/event_type={event_type}/"
        f"date={date_str}/hour={hour}/data.parquet"
    )


def _generate_hour_partitions(
    start_dt: datetime,
    end_dt: datetime,
) -> list[tuple[datetime, int]]:
    """Generate list of (date, hour) tuples covering a time range."""
    partitions = []
    current = start_dt.replace(minute=0, second=0, microsecond=0)
    
    while current <= end_dt:
        partitions.append((current, current.hour))
        current = current + timedelta(hours=1)
    
    return partitions


def _load_parquet_range(
    conn: duckdb.DuckDBPyConnection,
    venue: str,
    stream_id: str,
    event_type: str,
    start_dt: datetime,
    end_dt: datetime,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load parquet data for a time range, handling multiple hour partitions."""
    partitions = _generate_hour_partitions(start_dt, end_dt)
    
    if not partitions:
        return pd.DataFrame()
    
    # Build paths
    paths = [
        _build_s3_path(venue, stream_id, event_type, dt, hour)
        for dt, hour in partitions
    ]
    
    # Build column projection
    col_clause = ", ".join(columns) if columns else "*"
    
    # Build UNION query for all partitions
    queries = [
        f"SELECT {col_clause} FROM read_parquet('{path}')"
        for path in paths
    ]
    union_query = " UNION ALL ".join(queries)
    
    # Filter to exact time range
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    
    query = f"""
        SELECT * FROM ({union_query})
        WHERE ts_recv BETWEEN {start_ms} AND {end_ms}
        ORDER BY ts_recv, seq
    """
    
    try:
        return conn.execute(query).df()
    except duckdb.IOException as e:
        # Handle missing partitions gracefully
        if "Could not open file" in str(e) or "No such file" in str(e):
            # Try each partition individually
            dfs = []
            for path in paths:
                try:
                    single_query = f"""
                        SELECT {col_clause} FROM read_parquet('{path}')
                        WHERE ts_recv BETWEEN {start_ms} AND {end_ms}
                    """
                    dfs.append(conn.execute(single_query).df())
                except duckdb.IOException:
                    continue  # Skip missing partitions
            
            if dfs:
                result = pd.concat(dfs, ignore_index=True)
                return result.sort_values(["ts_recv", "seq"]).reset_index(drop=True)
            return pd.DataFrame()
        raise


def load_binance_bbo(
    start_dt: datetime,
    end_dt: datetime,
    asset: Literal["BTC", "ETH"] = "BTC",
    conn: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Load Binance BBO data for a time range.
    
    Args:
        start_dt: Start datetime (UTC)
        end_dt: End datetime (UTC)
        asset: "BTC" or "ETH"
        conn: DuckDB connection (uses global if not provided)
        
    Returns:
        DataFrame with columns: ts_event, ts_recv, seq, bid_px, bid_sz, 
        ask_px, ask_sz, update_id
    """
    if conn is None:
        conn = get_connection()
    
    config = get_config()
    stream_id = config.stream_id_for_asset(asset, "binance")
    
    return _load_parquet_range(
        conn=conn,
        venue="binance",
        stream_id=stream_id,
        event_type="bbo",
        start_dt=start_dt,
        end_dt=end_dt,
    )


def load_binance_trades(
    start_dt: datetime,
    end_dt: datetime,
    asset: Literal["BTC", "ETH"] = "BTC",
    conn: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Load Binance trade data for a time range.
    
    Args:
        start_dt: Start datetime (UTC)
        end_dt: End datetime (UTC)
        asset: "BTC" or "ETH"
        conn: DuckDB connection (uses global if not provided)
        
    Returns:
        DataFrame with columns: ts_event, ts_recv, seq, price, size, side, trade_id
    """
    if conn is None:
        conn = get_connection()
    
    config = get_config()
    stream_id = config.stream_id_for_asset(asset, "binance")
    
    return _load_parquet_range(
        conn=conn,
        venue="binance",
        stream_id=stream_id,
        event_type="trade",
        start_dt=start_dt,
        end_dt=end_dt,
    )


def load_polymarket_bbo(
    start_dt: datetime,
    end_dt: datetime,
    asset: Literal["BTC", "ETH"] = "BTC",
    token_id_prefix: str | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Load Polymarket BBO data for a time range.
    
    Args:
        start_dt: Start datetime (UTC)
        end_dt: End datetime (UTC)
        asset: "BTC" or "ETH"
        token_id_prefix: Filter to specific token (e.g., first 6 chars of Up token).
                         If None, returns both Up and Down tokens.
        conn: DuckDB connection (uses global if not provided)
        
    Returns:
        DataFrame with columns: ts_event, ts_recv, seq, bid_px, bid_sz,
        ask_px, ask_sz, token_id
    """
    if conn is None:
        conn = get_connection()
    
    config = get_config()
    stream_id = config.stream_id_for_asset(asset, "polymarket")
    
    df = _load_parquet_range(
        conn=conn,
        venue="polymarket",
        stream_id=stream_id,
        event_type="bbo",
        start_dt=start_dt,
        end_dt=end_dt,
    )
    
    if token_id_prefix and not df.empty:
        df = df[df["token_id"].str.startswith(token_id_prefix)].reset_index(drop=True)
    
    return df


def load_polymarket_trades(
    start_dt: datetime,
    end_dt: datetime,
    asset: Literal["BTC", "ETH"] = "BTC",
    token_id_prefix: str | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Load Polymarket trade data for a time range.
    
    Args:
        start_dt: Start datetime (UTC)
        end_dt: End datetime (UTC)
        asset: "BTC" or "ETH"
        token_id_prefix: Filter to specific token
        conn: DuckDB connection (uses global if not provided)
        
    Returns:
        DataFrame with columns: ts_event, ts_recv, seq, price, size, side, token_id
    """
    if conn is None:
        conn = get_connection()
    
    config = get_config()
    stream_id = config.stream_id_for_asset(asset, "polymarket")
    
    df = _load_parquet_range(
        conn=conn,
        venue="polymarket",
        stream_id=stream_id,
        event_type="trade",
        start_dt=start_dt,
        end_dt=end_dt,
    )
    
    if token_id_prefix and not df.empty:
        df = df[df["token_id"].str.startswith(token_id_prefix)].reset_index(drop=True)
    
    return df


def load_polymarket_book(
    start_dt: datetime,
    end_dt: datetime,
    asset: Literal["BTC", "ETH"] = "BTC",
    token_id_prefix: str | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Load Polymarket L2 order book snapshots for a time range.
    
    Args:
        start_dt: Start datetime (UTC)
        end_dt: End datetime (UTC)
        asset: "BTC" or "ETH"
        token_id_prefix: Filter to specific token
        conn: DuckDB connection (uses global if not provided)
        
    Returns:
        DataFrame with columns: ts_event, ts_recv, seq, token_id,
        bid_prices (list), bid_sizes (list), ask_prices (list), ask_sizes (list),
        book_hash
    """
    if conn is None:
        conn = get_connection()
    
    config = get_config()
    stream_id = config.stream_id_for_asset(asset, "polymarket")
    
    df = _load_parquet_range(
        conn=conn,
        venue="polymarket",
        stream_id=stream_id,
        event_type="book",
        start_dt=start_dt,
        end_dt=end_dt,
    )
    
    if token_id_prefix and not df.empty:
        df = df[df["token_id"].str.startswith(token_id_prefix)].reset_index(drop=True)
    
    return df


def get_unique_token_ids(
    start_dt: datetime,
    end_dt: datetime,
    asset: Literal["BTC", "ETH"] = "BTC",
    conn: duckdb.DuckDBPyConnection | None = None,
) -> list[str]:
    """Get unique token IDs for a time range.
    
    Useful for discovering which token_id corresponds to Up vs Down.
    The first token_id (alphabetically) is typically "Up".
    
    Args:
        start_dt: Start datetime (UTC)
        end_dt: End datetime (UTC)
        asset: "BTC" or "ETH"
        conn: DuckDB connection
        
    Returns:
        List of unique token_id values, sorted
    """
    if conn is None:
        conn = get_connection()
    
    df = load_polymarket_bbo(start_dt, end_dt, asset, conn=conn)
    
    if df.empty:
        return []
    
    return sorted(df["token_id"].unique().tolist())
