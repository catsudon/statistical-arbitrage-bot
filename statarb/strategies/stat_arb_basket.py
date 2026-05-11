"""Multi-asset stat-arb on a Johansen basket."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.types import BasketSpec
from ..signals.zscore import ZScoreParams, ZScoreSignal


@dataclass
class BasketStrategyConfig:
    zscore: ZScoreParams = None

    def __post_init__(self):
        if self.zscore is None:
            self.zscore = ZScoreParams()


class BasketStrategy:
    def __init__(self, basket: BasketSpec, cfg: BasketStrategyConfig | None = None):
        self.basket = basket
        self.cfg = cfg or BasketStrategyConfig()
        self.name = f"basket[{','.join(basket.symbols)}]"

    def fit(self, closes: pd.DataFrame) -> None:
        return

    def generate_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        syms = self.basket.symbols
        w = np.asarray(self.basket.weights, dtype=float)
        px = np.log(closes[syms])
        spread = pd.Series(px.values @ w, index=closes.index, name="spread")
        state = ZScoreSignal(self.cfg.zscore).generate(spread)

        # Scale weights so the gross exposure per unit-state == 1.
        gross = np.sum(np.abs(w))
        w_norm = w / gross if gross > 0 else w

        out = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
        for s, wi in zip(syms, w_norm):
            # state=+1 ⇒ long the spread direction => weights = +w_norm
            out[s] = state.astype(float) * wi
        return out
