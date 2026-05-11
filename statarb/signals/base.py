"""Signal generator interface.

A signal generator turns a *spread* (a stationary series) into a
position-state series in {-1, 0, +1} representing target spread side.
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd


class SpreadSignal(Protocol):
    def generate(self, spread: pd.Series) -> pd.Series: ...
