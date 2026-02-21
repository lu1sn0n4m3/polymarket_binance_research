"""Strategy interface for backtesting."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from backtesting.core import Trade


class Strategy(ABC):
    """Base class for market-taker strategies.

    Receives a DataFrame of snapshots for one market-hour and returns
    a list of Trade objects to execute.

    Snapshot columns:
        ts, tau, S, K, model_p, pm_bid, pm_ask, pm_mid, sigma_rv
    """

    @abstractmethod
    def generate_trades(self, snapshots: pd.DataFrame) -> list[Trade]:
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
