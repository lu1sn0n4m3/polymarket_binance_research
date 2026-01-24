"""Pricing models for binary options."""

from src.pricing.base import Pricer, PricerOutput, NaivePricer, MoneynessPricer
from src.pricing.fundamental import (
    FundamentalPricer,
    FundamentalPricerConfig,
    IncrementalPricer,
    VolatilityBlender,
    LinearBlender,
    HistoricalVolProfile,
    EmptyHistoricalProfile,
    SimpleHistoricalProfile,
    estimate_degrees_of_freedom,
    compare_prices,
    analyze_edge as analyze_edge_fundamental,
)
from src.pricing.gaussian_ewma import (
    GaussianEWMAPricer,
    GaussianEWMAConfig,
    TODProfile,
    VolatilityState,
    analyze_edge,
)

__all__ = [
    # Base
    "Pricer",
    "PricerOutput",
    "NaivePricer",
    "MoneynessPricer",
    # Gaussian EWMA (primary)
    "GaussianEWMAPricer",
    "GaussianEWMAConfig",
    "TODProfile",
    "VolatilityState",
    "analyze_edge",
    # Fundamental (legacy)
    "FundamentalPricer",
    "FundamentalPricerConfig",
    "IncrementalPricer",
    "VolatilityBlender",
    "LinearBlender",
    "HistoricalVolProfile",
    "EmptyHistoricalProfile",
    "SimpleHistoricalProfile",
    "estimate_degrees_of_freedom",
    "compare_prices",
    "analyze_edge_fundamental",
]
