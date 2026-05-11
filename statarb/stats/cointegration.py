"""Cointegration tests: Engle-Granger (pairs) and Johansen (basket)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from .hedge_ratio import ols_hedge
from .mean_reversion import half_life


@dataclass
class EGResult:
    pvalue: float
    tstat: float
    alpha: float
    beta: float
    half_life: float
    resid: pd.Series


def engle_granger(y: pd.Series, x: pd.Series) -> EGResult:
    """Engle-Granger two-step cointegration test.

    Returns the p-value of the residual stationarity test and the static
    hedge ratio. p < 0.05 is the conventional pass.
    """
    aligned = pd.concat([y, x], axis=1, join="inner").dropna()
    aligned.columns = ["y", "x"]
    y_, x_ = aligned["y"], aligned["x"]
    fit = ols_hedge(y_, x_)
    try:
        t, p, _ = coint(y_.values, x_.values, trend="c", autolag="AIC")
    except Exception:
        t, p = 0.0, 1.0
    return EGResult(
        pvalue=float(p),
        tstat=float(t),
        alpha=fit.alpha,
        beta=fit.beta,
        half_life=half_life(fit.resid),
        resid=fit.resid,
    )


@dataclass
class JohansenResult:
    eig_stats: np.ndarray
    crit_95: np.ndarray
    n_coint: int
    weights: np.ndarray  # leading eigenvector, normalized so weights[0] = 1
    spread: pd.Series
    half_life: float


def johansen(prices: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> JohansenResult:
    """Johansen test on a price panel. Returns the rank, the leading
    cointegration vector, and the implied spread."""
    df = prices.dropna()
    if df.shape[0] < 50 or df.shape[1] < 2:
        raise ValueError("need at least 50 rows and 2 columns")
    res = coint_johansen(df.values, det_order, k_ar_diff)
    crit = res.cvt[:, 1]  # 95% trace crit values
    n_coint = int(np.sum(res.lr1 > crit))
    v = res.evec[:, 0]
    if v[0] != 0:
        v = v / v[0]
    spread = pd.Series(df.values @ v, index=df.index, name="spread")
    return JohansenResult(
        eig_stats=res.lr1,
        crit_95=crit,
        n_coint=n_coint,
        weights=v,
        spread=spread,
        half_life=half_life(spread),
    )
