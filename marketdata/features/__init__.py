"""Feature computation utilities."""

from marketdata.features.microstructure import (
    compute_microprice,
    compute_weighted_mid,
    compute_vwap,
    compute_trade_imbalance,
    compute_book_imbalance,
    compute_spread_bps,
)
from marketdata.features.volatility import (
    RealizedVolEstimator,
    SimpleRealizedVol,
    TradeBasedVol,
    compute_returns,
    compute_realized_vol,
    compute_yang_zhang_vol,
    compute_parkinson_vol,
)
from marketdata.features.historical import (
    compute_hourly_returns,
    get_historical_hourly_stats,
)

__all__ = [
    # Microstructure
    "compute_microprice",
    "compute_weighted_mid",
    "compute_vwap",
    "compute_trade_imbalance",
    "compute_book_imbalance",
    "compute_spread_bps",
    # Volatility
    "RealizedVolEstimator",
    "SimpleRealizedVol",
    "TradeBasedVol",
    "compute_returns",
    "compute_realized_vol",
    "compute_yang_zhang_vol",
    "compute_parkinson_vol",
    # Historical
    "compute_hourly_returns",
    "get_historical_hourly_stats",
]
