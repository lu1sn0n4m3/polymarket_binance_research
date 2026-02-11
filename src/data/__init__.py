"""Data loading and alignment utilities."""

# =============================================================================
# PRIMARY USER-FACING API (Easy API - use these for most tasks)
# =============================================================================
from src.data.easy_api import (
    load_binance,
    load_polymarket_market,
    align_timestamps,
    get_cache_status,
    clear_cache as clear_cache_easy,
)

# =============================================================================
# LEGACY/INTERNAL APIs (for backwards compatibility and advanced use)
# =============================================================================
from src.data.connection import get_connection, configure_s3
from src.data.loaders import (
    load_binance_bbo,
    load_binance_trades,
    load_polymarket_bbo,
    load_polymarket_trades,
    load_polymarket_book,
    get_unique_token_ids,
)
from src.data.alignment import align_asof, align_bucketed, resample_to_grid
from src.data.session import HourlyMarketSession, load_session, load_sessions_range
from src.data.resampled_bbo import load_resampled_bbo, resample_bbo_to_interval
from src.data.resampled_polymarket import load_resampled_polymarket, resample_polymarket_to_interval
from src.data.cache_manager import get_cache_info, clear_cache

__all__ = [
    # ==========================================================================
    # PRIMARY API - Use these for most research tasks
    # ==========================================================================
    "load_binance",  # Load Binance data for any time range
    "load_polymarket_market",  # Load single Polymarket market
    "align_timestamps",  # Combine Binance and Polymarket data
    "get_cache_status",  # Check what's cached
    "clear_cache_easy",  # Clear cache (easy API version)
    # ==========================================================================
    # Session-based API - Alternative high-level interface
    # ==========================================================================
    "HourlyMarketSession",
    "load_session",
    "load_sessions_range",
    # ==========================================================================
    # Internal/Advanced APIs - For specialized use cases
    # ==========================================================================
    # Connection
    "get_connection",
    "configure_s3",
    # Raw loaders (S3 direct access)
    "load_binance_bbo",
    "load_binance_trades",
    "load_polymarket_bbo",
    "load_polymarket_trades",
    "load_polymarket_book",
    "get_unique_token_ids",
    # Alignment utilities
    "align_asof",
    "align_bucketed",
    "resample_to_grid",
    # Resampled data (internal - use easy API instead)
    "load_resampled_bbo",
    "load_resampled_polymarket",
    "resample_bbo_to_interval",
    "resample_polymarket_to_interval",
    "get_cache_info",
    "clear_cache",
]
