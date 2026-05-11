"""Strategy interface used by both backtester and live runner.

A `Strategy` consumes a *price panel* and produces a *target-weight panel*
indexed identically — one column per symbol, values in [-1, 1].

Returning weights (not orders) keeps strategies pure and easy to test;
the execution layer translates weights into orders, accounting for the
existing portfolio.
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd


class Strategy(Protocol):
    name: str

    def fit(self, closes: pd.DataFrame) -> None: ...

    def generate_weights(self, closes: pd.DataFrame) -> pd.DataFrame: ...
