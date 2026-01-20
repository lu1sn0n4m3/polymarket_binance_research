"""DuckDB connection management with S3 support."""

import duckdb
from src.config import get_config, S3Config


_connection: duckdb.DuckDBPyConnection | None = None


def configure_s3(conn: duckdb.DuckDBPyConnection, s3_config: S3Config | None = None) -> None:
    """Configure S3 access for a DuckDB connection.
    
    Args:
        conn: DuckDB connection to configure
        s3_config: S3 configuration (uses global config if not provided)
    """
    if s3_config is None:
        s3_config = get_config().s3
    
    s3_config.validate()
    
    conn.execute("INSTALL httpfs;")
    conn.execute("LOAD httpfs;")
    conn.execute(f"SET s3_endpoint = '{s3_config.endpoint}';")
    conn.execute(f"SET s3_access_key_id = '{s3_config.access_key}';")
    conn.execute(f"SET s3_secret_access_key = '{s3_config.secret_key}';")
    conn.execute(f"SET s3_region = '{s3_config.region}';")
    conn.execute(f"SET s3_url_style = '{s3_config.url_style}';")


def get_connection(
    configure_s3_access: bool = True,
    fresh: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Get a configured DuckDB connection.
    
    Args:
        configure_s3_access: Whether to configure S3 access
        fresh: If True, create a new connection instead of reusing cached one
        
    Returns:
        Configured DuckDB connection
    """
    global _connection
    
    if fresh or _connection is None:
        _connection = duckdb.connect()
        if configure_s3_access:
            configure_s3(_connection)
    
    return _connection


def close_connection() -> None:
    """Close the global connection."""
    global _connection
    if _connection is not None:
        _connection.close()
        _connection = None
