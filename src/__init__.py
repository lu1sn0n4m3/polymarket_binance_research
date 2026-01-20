"""Polymarket-Binance Research Framework.

A framework for analyzing Polymarket hourly binary options using Binance data.
"""

from src.data.session import HourlyMarketSession, load_session

__version__ = "0.1.0"
__all__ = ["HourlyMarketSession", "load_session"]
