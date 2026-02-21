"""Core data classes for backtesting."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Trade:
    """A single executed trade."""

    ts: int  # epoch ms
    side: str  # "BUY" or "SELL"
    size: float  # dollars risked
    price: float  # execution price (pm_ask if BUY, pm_bid if SELL)
    model_price: float  # model P(Up) at time of trade
    edge: float  # model_price - price (BUY) or price - model_price (SELL)
    tau: float  # seconds to expiry at entry

    # Context fields (populated by simulator)
    sigma_rv: float = 0.0  # realized vol at entry
    hour_et: int = -1  # hour in ET at entry
    bankroll_at_entry: float = 0.0  # bankroll before this trade
    kelly_fraction: float = 0.0  # raw Kelly fraction used
    mfe: float = 0.0  # max favorable excursion (best unrealized PnL)
    mae: float = 0.0  # max adverse excursion (worst unrealized PnL, negative)

    def settle(self, outcome: int) -> float:
        """Compute PnL given binary outcome Y in {0, 1}."""
        if self.side == "BUY":
            return self.size * (outcome - self.price)
        else:
            return self.size * (self.price - outcome)

    @property
    def pnl_per_dollar(self) -> float:
        """Return per contract (not sized)."""
        # For a $1 contract: BUY at ask, win pays 1, lose pays 0
        # PnL / size = (outcome - price) for BUY
        # We don't know outcome here, so this is edge-based expected
        return self.edge


@dataclass
class MarketResult:
    """Result for a single market-hour."""

    market_id: str
    outcome: int  # Y: 1 (up) or 0 (down)
    trades: list[Trade]
    pnl: float
    gross_edge: float  # sum of edge * size

    @classmethod
    def from_trades(cls, market_id: str, outcome: int, trades: list[Trade]) -> MarketResult:
        pnl = sum(t.settle(outcome) for t in trades)
        gross_edge = sum(t.edge * t.size for t in trades)
        return cls(market_id=market_id, outcome=outcome, trades=trades, pnl=pnl,
                   gross_edge=gross_edge)


@dataclass
class BacktestResult:
    """Aggregate backtest results."""

    markets: list[MarketResult]
    config: dict = field(default_factory=dict)
    bankroll_curve: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def n_markets(self) -> int:
        return len(self.markets)

    @property
    def n_markets_traded(self) -> int:
        return sum(1 for m in self.markets if m.trades)

    @property
    def n_trades(self) -> int:
        return sum(len(m.trades) for m in self.markets)

    @property
    def total_pnl(self) -> float:
        return sum(m.pnl for m in self.markets)

    @property
    def all_trades(self) -> list[tuple[Trade, int]]:
        """All (trade, outcome) pairs."""
        return [(t, m.outcome) for m in self.markets for t in m.trades]

    @property
    def pnl_per_market(self) -> np.ndarray:
        """PnL array for traded markets only."""
        return np.array([m.pnl for m in self.markets if m.trades])

    @property
    def hit_rate(self) -> float:
        """Fraction of traded markets with positive PnL."""
        traded = self.pnl_per_market
        if len(traded) == 0:
            return 0.0
        return float(np.mean(traded > 0))

    @property
    def sharpe(self) -> float:
        """Sharpe ratio of per-market PnL (annualized assuming 24 markets/day)."""
        pnl = self.pnl_per_market
        if len(pnl) < 2:
            return 0.0
        mu = np.mean(pnl)
        sigma = np.std(pnl, ddof=1)
        if sigma < 1e-12:
            return 0.0
        return float(mu / sigma * np.sqrt(24 * 365))

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown of cumulative PnL curve (over traded markets)."""
        pnl = self.pnl_per_market
        if len(pnl) == 0:
            return 0.0
        cum = np.cumsum(pnl)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        return float(np.max(dd))

    @property
    def avg_edge(self) -> float:
        """Average edge at entry across all trades."""
        edges = [t.edge for m in self.markets for t in m.trades]
        return float(np.mean(edges)) if edges else 0.0
