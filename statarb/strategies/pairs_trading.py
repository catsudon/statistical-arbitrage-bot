"""Pairs trading strategy: z-score on the OLS spread, with optional
Kalman-dynamic hedge ratio."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.types import PairSpec
from ..signals.kalman import KalmanParams, KalmanSpreadSignal
from ..signals.zscore import ZScoreParams, ZScoreSignal
from ..stats.spread import make_spread


@dataclass
class PairStrategyConfig:
    use_kalman: bool = False
    zscore: ZScoreParams = None
    kalman: KalmanParams = None
    leg_dollar_neutral: bool = True

    def __post_init__(self):
        if self.zscore is None:
            self.zscore = ZScoreParams()
        if self.kalman is None:
            self.kalman = KalmanParams()


class PairsStrategy:
    """One strategy instance per cointegrated pair."""

    def __init__(self, pair: PairSpec, cfg: PairStrategyConfig | None = None):
        self.name = f"pair[{pair.y}|{pair.x}]"
        self.pair = pair
        self.cfg = cfg or PairStrategyConfig()
        self._beta_t: pd.Series | None = None

    def fit(self, closes: pd.DataFrame) -> None:
        return  # nothing to fit for static-β; Kalman is online

    def generate_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        y = closes[self.pair.y]
        x = closes[self.pair.x]
        if self.cfg.use_kalman:
            sig = KalmanSpreadSignal(self.cfg.kalman)
            state, beta_t, _ = sig.generate(np.log(y), np.log(x))
            beta = beta_t
            self._beta_t = beta_t
        else:
            sig = ZScoreSignal(self.cfg.zscore)
            spread = make_spread(np.log(y), np.log(x), self.pair.beta, self.pair.alpha)
            state = sig.generate(spread)
            beta = pd.Series(self.pair.beta, index=state.index)

        # Translate spread-state {-1, 0, +1} into dollar weights per leg.
        # state = +1 ⇒ long y, short x  (long the spread); state = -1 ⇒ opposite.
        # For dollar-neutrality, the x-leg is sized by β times the dollar of y.
        if self.cfg.leg_dollar_neutral:
            denom = 1.0 + beta.abs()
            w_y = state / denom
            w_x = -state * beta / denom
        else:
            w_y = state.astype(float)
            w_x = -state * beta

        out = pd.DataFrame(
            {self.pair.y: w_y.reindex(closes.index).fillna(0.0),
             self.pair.x: w_x.reindex(closes.index).fillna(0.0)},
            index=closes.index,
        )
        # Make sure all other symbols are 0
        for s in closes.columns:
            if s not in out.columns:
                out[s] = 0.0
        return out[closes.columns]
