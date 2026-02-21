"""Edge-threshold market-taker strategy with proper Kelly sizing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtesting.core import Trade
from backtesting.strategies import Strategy


@dataclass
class EdgeThresholdStrategy(Strategy):
    """Trade when model-vs-PM edge exceeds a threshold.

    Buy at pm_ask when model_p - pm_ask > min_edge.
    Sell at pm_bid when pm_bid - model_p > min_edge.

    Kelly sizing for binary options:
        BUY at ask a:  f* = (p - a) / (1 - a)   [fraction of bankroll]
        SELL at bid b: f* = (b - p) / b           [fraction of bankroll]

    With fractional Kelly (kelly_fraction < 1) for risk management.
    """

    min_edge: float = 0.03
    min_tau: float = 60.0  # skip last N seconds before expiry
    max_entries: int = 1  # max trades per market-hour
    sizing: str = "fixed"  # "fixed" or "kelly"
    fixed_size: float = 1.0
    max_kelly_fraction: float = 0.25  # cap raw Kelly at 25% of bankroll
    kelly_fraction: float = 0.5  # half-Kelly by default

    # Bankroll is injected by the simulator for Kelly sizing
    _bankroll: float = 100.0

    def generate_trades(self, snapshots: pd.DataFrame) -> list[Trade]:
        # Filter by minimum time-to-expiry
        df = snapshots[snapshots["tau"] > self.min_tau]
        if df.empty:
            return []

        ts = df["ts"].values
        tau = df["tau"].values
        model_p = df["model_p"].values
        pm_bid = df["pm_bid"].values
        pm_ask = df["pm_ask"].values
        sigma_rv = df["sigma_rv"].values if "sigma_rv" in df.columns else np.zeros(len(df))

        buy_edge = model_p - pm_ask
        sell_edge = pm_bid - model_p

        trades: list[Trade] = []
        for i in range(len(df)):
            if len(trades) >= self.max_entries:
                break

            # Skip NaN PM prices
            if np.isnan(pm_bid[i]) or np.isnan(pm_ask[i]):
                continue

            # Skip zero/negative spreads (bad data)
            spread = pm_ask[i] - pm_bid[i]
            if spread <= 0:
                continue

            if buy_edge[i] > self.min_edge:
                size = self._compute_size(
                    "BUY", float(model_p[i]), float(pm_ask[i]),
                )
                if size <= 0:
                    continue
                kelly_f = self._raw_kelly("BUY", float(model_p[i]), float(pm_ask[i]))
                trades.append(Trade(
                    ts=int(ts[i]), side="BUY", size=size,
                    price=float(pm_ask[i]), model_price=float(model_p[i]),
                    edge=float(buy_edge[i]), tau=float(tau[i]),
                    sigma_rv=float(sigma_rv[i]),
                    bankroll_at_entry=self._bankroll,
                    kelly_fraction=kelly_f,
                ))
            elif sell_edge[i] > self.min_edge:
                size = self._compute_size(
                    "SELL", float(model_p[i]), float(pm_bid[i]),
                )
                if size <= 0:
                    continue
                kelly_f = self._raw_kelly("SELL", float(model_p[i]), float(pm_bid[i]))
                trades.append(Trade(
                    ts=int(ts[i]), side="SELL", size=size,
                    price=float(pm_bid[i]), model_price=float(model_p[i]),
                    edge=float(sell_edge[i]), tau=float(tau[i]),
                    sigma_rv=float(sigma_rv[i]),
                    bankroll_at_entry=self._bankroll,
                    kelly_fraction=kelly_f,
                ))

        return trades

    def _raw_kelly(self, side: str, model_p: float, price: float) -> float:
        """Raw Kelly fraction for binary option."""
        if side == "BUY":
            # BUY at ask a: f* = (p - a) / (1 - a)
            return (model_p - price) / max(1 - price, 1e-6)
        else:
            # SELL at bid b: f* = (b - p) / b
            return (price - model_p) / max(price, 1e-6)

    def _compute_size(self, side: str, model_p: float, price: float) -> float:
        """Compute position size in dollars."""
        if self.sizing == "kelly":
            f_star = self._raw_kelly(side, model_p, price)
            # Apply fractional Kelly and cap
            f = min(f_star * self.kelly_fraction, self.max_kelly_fraction)
            f = max(f, 0.0)
            return f * self._bankroll
        return self.fixed_size

    @property
    def name(self) -> str:
        if self.sizing == "kelly":
            return (f"EdgeThreshold(edge={self.min_edge}, tau>{self.min_tau}s, "
                    f"kelly={self.kelly_fraction:.0%})")
        return f"EdgeThreshold(edge={self.min_edge}, tau>{self.min_tau}s, {self.sizing})"
