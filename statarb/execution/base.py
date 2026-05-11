"""Broker interface. The paper broker and the CCXT live broker both
implement this protocol so the same `Strategy` can target either."""
from __future__ import annotations

from typing import Protocol

import pandas as pd


class Broker(Protocol):
    def equity(self, prices: dict[str, float]) -> float: ...
    def current_weights(self, prices: dict[str, float]) -> pd.Series: ...
    def rebalance_to_weights(
        self,
        ts: pd.Timestamp,
        target_weights: dict[str, float],
        prices: dict[str, float],
    ) -> None: ...
