"""Mean-reversion measures: OU fit, half-life, expected hitting time."""
from __future__ import annotations

import numpy as np
import pandas as pd


def half_life(spread: pd.Series | np.ndarray) -> float:
    """OU half-life of mean reversion: Δs_t = κ (μ - s_{t-1}) + ε.

    Regress Δs on lagged level; half-life = -ln(2) / β where β is the
    coefficient on the lag.
    """
    s = pd.Series(spread).dropna()
    if len(s) < 30:
        return np.inf
    s_lag = s.shift(1).dropna()
    d = (s - s.shift(1)).dropna()
    aligned = pd.concat([d, s_lag], axis=1, join="inner").values
    if aligned.shape[0] < 30:
        return np.inf
    X = np.column_stack([np.ones(aligned.shape[0]), aligned[:, 1]])
    coef, *_ = np.linalg.lstsq(X, aligned[:, 0], rcond=None)
    beta = coef[1]
    if beta >= 0:
        return np.inf
    return float(-np.log(2.0) / beta)


def ou_fit(spread: pd.Series) -> dict:
    """Fit a discrete OU process and return κ, μ, σ, and half-life."""
    s = spread.dropna()
    if len(s) < 30:
        return {"kappa": 0.0, "mu": s.mean() if len(s) else 0.0, "sigma": s.std(), "half_life": np.inf}
    s_lag = s.shift(1).dropna()
    s_cur = s.loc[s_lag.index]
    X = np.column_stack([np.ones(len(s_lag)), s_lag.values])
    coef, *_ = np.linalg.lstsq(X, s_cur.values, rcond=None)
    a, b = coef
    resid = s_cur.values - (a + b * s_lag.values)
    sigma = float(np.std(resid, ddof=1))
    if b <= 0 or b >= 1:
        return {"kappa": 0.0, "mu": float(s.mean()), "sigma": sigma, "half_life": np.inf}
    kappa = -np.log(b)
    mu = a / (1 - b)
    hl = np.log(2) / kappa
    return {"kappa": float(kappa), "mu": float(mu), "sigma": sigma, "half_life": float(hl)}
