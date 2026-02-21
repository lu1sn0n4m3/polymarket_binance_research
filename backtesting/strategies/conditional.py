"""Conditional edge strategy with walk-forward hour scoring.

NO forward-looking bias: hour quality scores are passed in from the
walk-forward loop, computed only on past data at each fold.

Kelly fraction is always capped at kelly_fraction (default 0.5 = half-Kelly).
The condition multiplier scales DOWN from that cap, never up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from backtesting.core import Trade
from backtesting.strategies import Strategy

ET = ZoneInfo("America/New_York")


@dataclass
class ConditionalStrategy(Strategy):
    """Trade with condition-dependent sizing.

    Hour scores are injected externally (by the walk-forward loop) based
    on past performance only. No hardcoded hour lists.

    Multiplier logic (scales DOWN from kelly_fraction, never up):
      - hour_score > 0 (profitable in past): mult = 1.0 (full half-Kelly)
      - hour_score == 0 (no data or neutral): mult = skip_unknown_mult
      - hour_score < 0 (unprofitable in past): mult = 0 (skip)
    """

    min_edge: float = 0.05
    min_tau: float = 60.0
    max_entries: int = 1
    kelly_fraction: float = 0.5       # HARD CAP: never exceed this fraction of raw Kelly
    max_kelly_fraction: float = 0.15   # max fraction of bankroll per trade
    tp_frac: float = 0.5
    sl_frac: float = 0.2
    latency_ms: int = 1000
    skip_unknown_mult: float = 0.5    # multiplier for hours with no past data

    # Bankroll injected by simulator
    _bankroll: float = 100.0

    # Hour scores: dict hour_et -> float (positive = good, negative = bad)
    # Injected by walk-forward loop before each fold
    _hour_scores: dict = field(default_factory=dict)

    def generate_trades(self, snapshots: pd.DataFrame) -> list[Trade]:
        df = snapshots[snapshots["tau"] > self.min_tau]
        if df.empty:
            return []

        ts = df["ts"].values
        tau = df["tau"].values
        model_p = df["model_p"].values
        pm_bid = df["pm_bid"].values
        pm_ask = df["pm_ask"].values
        sigma_rv = df["sigma_rv"].values if "sigma_rv" in df.columns else np.zeros(len(df))
        sigma_rel = df["sigma_rel"].values if "sigma_rel" in df.columns else np.zeros(len(df))
        pm_spread = df["pm_spread"].values if "pm_spread" in df.columns else np.zeros(len(df))

        buy_edge = model_p - pm_ask
        sell_edge = pm_bid - model_p

        trades: list[Trade] = []
        n = len(df)
        for i in range(n):
            if len(trades) >= self.max_entries:
                break

            if np.isnan(pm_bid[i]) or np.isnan(pm_ask[i]):
                continue
            if pm_ask[i] - pm_bid[i] <= 0:
                continue

            # Latency: fill at t+latency
            if self.latency_ms > 0:
                fill_ts = int(ts[i]) + self.latency_ms
                fill_idx = i + 1
                while fill_idx < n and ts[fill_idx] < fill_ts:
                    fill_idx += 1
                if fill_idx >= n:
                    continue
                if np.isnan(pm_bid[fill_idx]) or np.isnan(pm_ask[fill_idx]):
                    continue
                fill_ask = float(pm_ask[fill_idx])
                fill_bid = float(pm_bid[fill_idx])
                fill_tau = float(tau[fill_idx])
                fill_ts_int = int(ts[fill_idx])
            else:
                fill_ask = float(pm_ask[i])
                fill_bid = float(pm_bid[i])
                fill_tau = float(tau[i])
                fill_ts_int = int(ts[i])

            # Detect signal
            is_buy = buy_edge[i] > self.min_edge
            is_sell = sell_edge[i] > self.min_edge
            if not is_buy and not is_sell:
                continue

            if is_buy:
                actual_edge = float(model_p[i]) - fill_ask
                fill_price = fill_ask
                side = "BUY"
            else:
                actual_edge = fill_bid - float(model_p[i])
                fill_price = fill_bid
                side = "SELL"

            if actual_edge <= 0:
                continue

            # ----------------------------------------------------------
            # Condition-based multiplier (0 to 1, never above 1)
            # ----------------------------------------------------------
            dt = datetime.fromtimestamp(int(ts[i]) / 1000, tz=timezone.utc)
            hour_et = dt.astimezone(ET).hour

            if self._hour_scores:
                score = self._hour_scores.get(hour_et, 0.0)
                if score < 0:
                    continue  # skip hours that have been unprofitable
                elif score > 0:
                    mult = 1.0
                else:
                    mult = self.skip_unknown_mult
            else:
                # No hour scores yet (first fold) — trade all hours at reduced size
                mult = self.skip_unknown_mult

            # ----------------------------------------------------------
            # Kelly sizing: mult scales DOWN from kelly_fraction, never up
            # effective_kelly = f_star * kelly_fraction * mult
            # where mult in [0, 1], kelly_fraction = 0.5 (half-Kelly cap)
            # ----------------------------------------------------------
            f_star = self._raw_kelly(side, float(model_p[i]), fill_price)
            effective_kelly_frac = self.kelly_fraction * mult  # always <= kelly_fraction
            f = min(f_star * effective_kelly_frac, self.max_kelly_fraction)
            f = max(f, 0.0)
            size = f * self._bankroll
            if size <= 0:
                continue

            trades.append(Trade(
                ts=fill_ts_int, side=side, size=size,
                price=fill_price, model_price=float(model_p[i]),
                edge=actual_edge, tau=fill_tau,
                sigma_rv=float(sigma_rv[i]),
                sigma_rel=float(sigma_rel[i]),
                pm_spread=float(pm_spread[i]),
                bankroll_at_entry=self._bankroll,
                kelly_fraction=f_star * effective_kelly_frac,
            ))

        return trades

    def _raw_kelly(self, side: str, model_p: float, price: float) -> float:
        if side == "BUY":
            return (model_p - price) / max(1 - price, 1e-6)
        else:
            return (price - model_p) / max(price, 1e-6)

    @property
    def name(self) -> str:
        parts = [f"edge={self.min_edge}"]
        parts.append(f"kelly={self.kelly_fraction:.0%}")
        parts.append(f"max={self.max_kelly_fraction:.0%}")
        if self.tp_frac > 0:
            parts.append(f"tp={self.tp_frac:.0%}")
        if self.sl_frac > 0:
            parts.append(f"sl={self.sl_frac:.0%}")
        parts.append(f"lat={self.latency_ms}ms")
        parts.append("conditional")
        return f"Conditional({', '.join(parts)})"
