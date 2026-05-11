"""Kalman-dynamic signal: hedge ratio re-estimated each bar; spread = innovation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..stats.hedge_ratio import kalman_hedge


@dataclass
class KalmanParams:
    delta: float = 1e-4
    obs_var: float = 1e-3
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_stop: float = 4.0
    z_lookback: int = 100


class KalmanSpreadSignal:
    """Compute dynamic spread via Kalman, then trade z-score of the innovation."""

    def __init__(self, params: KalmanParams | None = None):
        self.p = params or KalmanParams()

    def generate(self, y: pd.Series, x: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        alpha, beta, spread = kalman_hedge(y, x, delta=self.p.delta, obs_var=self.p.obs_var)
        mu = spread.rolling(self.p.z_lookback, min_periods=self.p.z_lookback // 4).mean()
        sd = spread.rolling(self.p.z_lookback, min_periods=self.p.z_lookback // 4).std()
        z = (spread - mu) / sd.replace(0, np.nan)

        state = np.zeros(len(z), dtype=np.int8)
        pos = 0
        zv = z.values
        for i in range(len(zv)):
            zi = zv[i]
            if np.isnan(zi):
                state[i] = pos
                continue
            if pos == 0:
                if zi >= self.p.z_entry:
                    pos = -1
                elif zi <= -self.p.z_entry:
                    pos = 1
            else:
                if abs(zi) >= self.p.z_stop or abs(zi) <= self.p.z_exit:
                    pos = 0
            state[i] = pos
        return pd.Series(state, index=z.index, name="state"), beta, spread
