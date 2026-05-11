"""Classic z-score mean-reversion signal with hysteresis (entry/exit/stop)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..stats.spread import rolling_zscore


@dataclass
class ZScoreParams:
    entry: float = 2.0
    exit: float = 0.5
    stop: float = 4.0
    lookback: int = 200


class ZScoreSignal:
    """Position state machine on the spread z-score.

    - enter short when z >= +entry; exit when |z| <= exit
    - enter long  when z <= -entry; exit when |z| <= exit
    - stop out if |z| >= stop while in a position
    """

    def __init__(self, params: ZScoreParams | None = None):
        self.p = params or ZScoreParams()

    def zscore(self, spread: pd.Series) -> pd.Series:
        return rolling_zscore(spread, lookback=self.p.lookback)

    def generate(self, spread: pd.Series) -> pd.Series:
        z = self.zscore(spread)
        state = np.zeros(len(z), dtype=np.int8)
        pos = 0
        zv = z.values
        for i in range(len(zv)):
            zi = zv[i]
            if np.isnan(zi):
                state[i] = pos
                continue
            if pos == 0:
                if zi >= self.p.entry:
                    pos = -1  # short the spread
                elif zi <= -self.p.entry:
                    pos = 1  # long the spread
            else:
                if abs(zi) >= self.p.stop:
                    pos = 0
                elif abs(zi) <= self.p.exit:
                    pos = 0
            state[i] = pos
        return pd.Series(state, index=z.index, name="state")
