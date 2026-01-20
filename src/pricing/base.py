"""Abstract base class for binary option pricers.

This module defines the interface that all pricing models should implement.
The actual pricing logic is left for you to implement based on your research.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PricerOutput:
    """Output from a pricer.
    
    Attributes:
        up_prob: Probability of "up" outcome (0-1)
        down_prob: Probability of "down" outcome (0-1)
        up_fair_bid: Fair bid price for "up" token
        up_fair_ask: Fair ask price for "up" token
        edge_up_bid: Edge if buying "up" at current market bid
        edge_up_ask: Edge if selling "up" at current market ask
        metadata: Optional additional data from the pricer
    """
    up_prob: float
    down_prob: float
    up_fair_bid: float
    up_fair_ask: float
    edge_up_bid: float | None = None
    edge_up_ask: float | None = None
    metadata: dict[str, Any] | None = None
    
    def __post_init__(self):
        # Ensure probabilities sum to 1
        total = self.up_prob + self.down_prob
        if not np.isclose(total, 1.0, atol=0.01):
            raise ValueError(f"Probabilities should sum to 1, got {total}")
    
    @classmethod
    def from_up_prob(
        cls,
        up_prob: float,
        spread: float = 0.02,
        market_up_bid: float | None = None,
        market_up_ask: float | None = None,
        metadata: dict | None = None,
    ) -> "PricerOutput":
        """Create PricerOutput from just an up probability.
        
        Args:
            up_prob: Probability of up outcome
            spread: Spread to apply for fair bid/ask (default 2%)
            market_up_bid: Current market bid for computing edge
            market_up_ask: Current market ask for computing edge
            metadata: Optional metadata
            
        Returns:
            PricerOutput instance
        """
        half_spread = spread / 2
        
        output = cls(
            up_prob=up_prob,
            down_prob=1 - up_prob,
            up_fair_bid=max(0, up_prob - half_spread),
            up_fair_ask=min(1, up_prob + half_spread),
            metadata=metadata,
        )
        
        # Compute edge if market prices provided
        if market_up_bid is not None:
            # Edge if selling up at market bid
            output.edge_up_bid = market_up_bid - output.up_fair_bid
        
        if market_up_ask is not None:
            # Edge if buying up at market ask
            output.edge_up_ask = output.up_fair_ask - market_up_ask
        
        return output


class Pricer(ABC):
    """Abstract base class for binary option pricers.
    
    Implement this interface to create custom pricing models.
    
    Example:
        class MyPricer(Pricer):
            def price(self, **kwargs) -> PricerOutput:
                vol = kwargs.get("realized_vol", 0.5)
                tte = kwargs.get("time_to_expiry_sec", 1800)
                
                # Your pricing logic here
                up_prob = 0.5  # Placeholder
                
                return PricerOutput.from_up_prob(up_prob)
    """
    
    @abstractmethod
    def price(
        self,
        time_to_expiry_sec: float,
        realized_vol: float,
        current_price: float | None = None,
        strike_price: float | None = None,
        **features,
    ) -> PricerOutput:
        """Compute theoretical price for the up outcome.
        
        Args:
            time_to_expiry_sec: Seconds until market close
            realized_vol: Annualized realized volatility estimate
            current_price: Current Binance mid price (optional)
            strike_price: Strike price (open price) (optional)
            **features: Additional features (e.g., book imbalance, momentum)
            
        Returns:
            PricerOutput with probabilities and fair prices
        """
        pass
    
    def compute_moneyness(
        self,
        current_price: float,
        strike_price: float,
    ) -> float:
        """Compute moneyness: (current - strike) / strike.
        
        Positive = currently "in the money" for up
        Negative = currently "in the money" for down
        
        Args:
            current_price: Current price
            strike_price: Strike (open) price
            
        Returns:
            Moneyness as a percentage
        """
        return (current_price - strike_price) / strike_price


class NaivePricer(Pricer):
    """Naive 50/50 pricer for testing.
    
    Always returns 50% probability for up.
    """
    
    def price(
        self,
        time_to_expiry_sec: float,
        realized_vol: float,
        current_price: float | None = None,
        strike_price: float | None = None,
        **features,
    ) -> PricerOutput:
        """Return 50/50 probability."""
        return PricerOutput.from_up_prob(0.5)


class MoneynessPricer(Pricer):
    """Simple pricer based on current moneyness.
    
    Uses a sigmoid function to map moneyness to probability.
    This is a placeholder - you should implement a proper model.
    """
    
    def __init__(self, sensitivity: float = 100.0):
        """
        Args:
            sensitivity: How sensitive to moneyness (higher = steeper sigmoid)
        """
        self.sensitivity = sensitivity
    
    def price(
        self,
        time_to_expiry_sec: float,
        realized_vol: float,
        current_price: float | None = None,
        strike_price: float | None = None,
        **features,
    ) -> PricerOutput:
        """Price based on moneyness with time decay."""
        if current_price is None or strike_price is None:
            return PricerOutput.from_up_prob(0.5)
        
        moneyness = self.compute_moneyness(current_price, strike_price)
        
        # Scale by volatility and time remaining
        # As time decreases, price moves have less time to reverse
        time_factor = np.sqrt(time_to_expiry_sec / 3600)  # Normalize to 1 hour
        vol_factor = realized_vol if realized_vol > 0 else 0.5
        
        # Adjusted moneyness (more extreme as time runs out)
        adj_moneyness = moneyness / (vol_factor * time_factor + 1e-6)
        
        # Sigmoid to probability
        up_prob = 1 / (1 + np.exp(-self.sensitivity * adj_moneyness))
        
        # Clamp to reasonable range
        up_prob = np.clip(up_prob, 0.01, 0.99)
        
        return PricerOutput.from_up_prob(
            up_prob,
            metadata={
                "moneyness": moneyness,
                "adj_moneyness": adj_moneyness,
                "time_factor": time_factor,
                "vol_factor": vol_factor,
            },
        )
