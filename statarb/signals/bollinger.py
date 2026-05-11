"""Bollinger-band signal on the spread (alternate parameterization of z-score)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BollingerParams:
    lookback: int = 200
    n_std_entry: float = 2.0
    n_std_exit: float = 0.5


class BollingerSignal:
    def __init__(self, params: BollingerParams | None = None):
        self.p = params or BollingerParams()

    def generate(self, spread: pd.Series) -> pd.Series:
        mu = spread.rolling(self.p.lookback, min_periods=self.p.lookback // 4).mean()
        sd = spread.rolling(self.p.lookback, min_periods=self.p.lookback // 4).std()
        upper = mu + self.p.n_std_entry * sd
        lower = mu - self.p.n_std_entry * sd
        upper_exit = mu + self.p.n_std_exit * sd
        lower_exit = mu - self.p.n_std_exit * sd

        state = np.zeros(len(spread), dtype=np.int8)
        pos = 0
        s = spread.values
        for i in range(len(s)):
            if np.isnan(upper.iloc[i]):
                state[i] = pos
                continue
            if pos == 0:
                if s[i] >= upper.iloc[i]:
                    pos = -1
                elif s[i] <= lower.iloc[i]:
                    pos = 1
            elif pos == 1 and s[i] >= lower_exit.iloc[i]:
                pos = 0
            elif pos == -1 and s[i] <= upper_exit.iloc[i]:
                pos = 0
            state[i] = pos
        return pd.Series(state, index=spread.index, name="state")
