"""Stationarity / mean-reversion diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def adf_pvalue(x: pd.Series | np.ndarray, regression: str = "c") -> float:
    """Augmented Dickey-Fuller p-value. Null: unit root (non-stationary).
    Small p ⇒ stationary."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 20:
        return 1.0
    try:
        return float(adfuller(x, regression=regression, autolag="AIC")[1])
    except Exception:
        return 1.0


def kpss_pvalue(x: pd.Series | np.ndarray, regression: str = "c") -> float:
    """KPSS p-value. Null: stationary. Large p ⇒ stationary."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 20:
        return 0.0
    try:
        return float(kpss(x, regression=regression, nlags="auto")[1])
    except Exception:
        return 0.0


def hurst_exponent(x: pd.Series | np.ndarray, max_lag: int = 100) -> float:
    """Hurst exponent via R/S-style log-log fit of variance of lagged
    differences. H < 0.5 → mean-reverting, ~0.5 → random walk, > 0.5 →
    trending."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 50:
        return 0.5
    lags = np.unique(np.logspace(0, np.log10(min(max_lag, n // 4)), 20).astype(int))
    lags = lags[lags >= 2]
    if len(lags) < 4:
        return 0.5
    tau = []
    for lag in lags:
        d = x[lag:] - x[:-lag]
        tau.append(np.std(d, ddof=1))
    tau = np.asarray(tau)
    mask = tau > 0
    if mask.sum() < 4:
        return 0.5
    slope, _ = np.polyfit(np.log(lags[mask]), np.log(tau[mask]), 1)
    return float(slope)


def variance_ratio(x: pd.Series | np.ndarray, q: int = 2) -> float:
    """Lo-MacKinlay variance ratio. VR < 1 ⇒ mean-reverting."""
    x = np.asarray(x, dtype=float)
    r = np.diff(x)
    if len(r) < q * 4:
        return 1.0
    var1 = np.var(r, ddof=1)
    rq = np.convolve(r, np.ones(q), mode="valid")
    varq = np.var(rq, ddof=1) / q
    return float(varq / var1) if var1 > 0 else 1.0
