"""Spread construction and z-score utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_spread(y: pd.Series, x: pd.Series, beta: float, alpha: float = 0.0) -> pd.Series:
    aligned = pd.concat([y, x], axis=1, join="inner").dropna()
    aligned.columns = ["y", "x"]
    return (aligned["y"] - alpha - beta * aligned["x"]).rename("spread")


def rolling_zscore(s: pd.Series, lookback: int = 200, min_periods: int | None = None) -> pd.Series:
    mp = min_periods or max(20, lookback // 4)
    mu = s.rolling(lookback, min_periods=mp).mean()
    sd = s.rolling(lookback, min_periods=mp).std()
    return ((s - mu) / sd.replace(0, np.nan)).rename("zscore")


def expanding_zscore(s: pd.Series, min_periods: int = 50) -> pd.Series:
    mu = s.expanding(min_periods).mean()
    sd = s.expanding(min_periods).std()
    return ((s - mu) / sd.replace(0, np.nan)).rename("zscore")
