"""Hard portfolio limits: drawdown, position count, single-name concentration."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RiskLimits:
    max_drawdown: float = 0.25       # halt trading when DD breaches this
    max_positions: int = 10
    max_single_weight: float = 0.5
    max_gross_leverage: float = 3.0


def apply_position_limits(weights: pd.DataFrame, limits: RiskLimits) -> pd.DataFrame:
    """Cap per-name weight and drop the smallest beyond `max_positions`."""
    w = weights.clip(lower=-limits.max_single_weight, upper=limits.max_single_weight)
    # row-wise: keep top-N by |weight|
    out = w.copy()
    for ts, row in w.iterrows():
        nz = row[row != 0]
        if len(nz) <= limits.max_positions:
            continue
        kept = nz.abs().nlargest(limits.max_positions).index
        zero_idx = nz.index.difference(kept)
        out.loc[ts, zero_idx] = 0.0
    return out


def drawdown_curve(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def halt_after_drawdown(equity: pd.Series, max_dd: float) -> pd.Series:
    """Returns a boolean mask: True = trading allowed, False = halted."""
    dd = drawdown_curve(equity)
    halted = (dd <= -abs(max_dd)).astype(bool)
    # once halted, stays halted (sticky)
    return ~halted.cummax()
