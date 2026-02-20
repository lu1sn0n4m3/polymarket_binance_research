"""Market data infrastructure for Polymarket + Binance research.

Provides data loading, caching, alignment, features, and visualization.
"""

from marketdata.data.easy_api import (
    load_binance,
    load_binance_labels,
    load_polymarket_market,
    align_timestamps,
)
from marketdata.data.session import HourlyMarketSession, load_session

__version__ = "0.1.0"
__all__ = [
    "load_binance",
    "load_binance_labels",
    "load_polymarket_market",
    "align_timestamps",
    "HourlyMarketSession",
    "load_session",
]
