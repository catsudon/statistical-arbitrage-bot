"""Aggregate per-strategy weights into a portfolio target panel.

Implements: equal-risk allocation, inverse-vol allocation, max-gross-leverage
clamp, and per-symbol weight aggregation across overlapping strategies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_weights(strategy_weights: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Sum per-strategy weight panels into a portfolio target.

    All panels must share the same index and columns (use the same
    `closes`). Strategies that overlap on a symbol have their weights
    added (a long+short on the same symbol nets out, which is correct)."""
    if not strategy_weights:
        return pd.DataFrame()
    panels = list(strategy_weights.values())
    out = panels[0].copy()
    for p in panels[1:]:
        out = out.add(p, fill_value=0.0)
    return out


def inverse_vol_allocation(
    strategy_returns: pd.DataFrame, lookback: int = 240
) -> pd.Series:
    """Allocate budget inversely proportional to realised vol of each
    strategy's PnL stream. Returns a row of weights summing to 1."""
    vol = strategy_returns.rolling(lookback, min_periods=lookback // 4).std().iloc[-1]
    inv = 1.0 / vol.replace(0, np.nan)
    inv = inv.fillna(0.0)
    s = inv.sum()
    return (inv / s) if s > 0 else pd.Series(np.zeros_like(inv), index=inv.index)


def clamp_gross_leverage(weights: pd.DataFrame, max_gross: float) -> pd.DataFrame:
    """Row-wise scaling so that Σ|w| ≤ max_gross."""
    gross = weights.abs().sum(axis=1)
    scale = np.where(gross > max_gross, max_gross / gross.replace(0, np.nan), 1.0)
    scale = pd.Series(scale, index=weights.index).fillna(0.0)
    return weights.mul(scale, axis=0)
