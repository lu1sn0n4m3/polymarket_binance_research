"""Data loading and alignment utilities."""

from src.data.connection import get_connection, configure_s3
from src.data.loaders import load_binance_bbo, load_binance_trades, load_polymarket_bbo, load_polymarket_trades, load_polymarket_book
from src.data.alignment import align_asof, align_bucketed, resample_to_grid
from src.data.session import HourlyMarketSession, load_session, load_sessions_range
from src.data.resampled_bbo import load_resampled_bbo, resample_bbo_to_interval
from src.data.cache_manager import get_cache_info, clear_cache

__all__ = [
    # Connection
    "get_connection",
    "configure_s3",
    # Loaders
    "load_binance_bbo",
    "load_binance_trades",
    "load_polymarket_bbo",
    "load_polymarket_trades",
    "load_polymarket_book",
    # Alignment
    "align_asof",
    "align_bucketed",
    "resample_to_grid",
    # Session
    "HourlyMarketSession",
    "load_session",
    "load_sessions_range",
    # Resampled BBO
    "load_resampled_bbo",
    "resample_bbo_to_interval",
    "get_cache_info",
    "clear_cache",
]
