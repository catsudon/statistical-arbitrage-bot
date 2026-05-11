"""Estimators of the hedge ratio β in y_t = α + β x_t + ε_t.

Static: OLS, total-least-squares (TLS).
Dynamic: Kalman filter that treats (α_t, β_t) as a random walk.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class HedgeFit:
    alpha: float
    beta: float
    resid: pd.Series


def ols_hedge(y: pd.Series, x: pd.Series) -> HedgeFit:
    y_a, x_a = y.values.astype(float), x.values.astype(float)
    X = np.column_stack([np.ones_like(x_a), x_a])
    coef, *_ = np.linalg.lstsq(X, y_a, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    resid = pd.Series(y_a - (alpha + beta * x_a), index=y.index, name="resid")
    return HedgeFit(alpha=alpha, beta=beta, resid=resid)


def tls_hedge(y: pd.Series, x: pd.Series) -> HedgeFit:
    """Total least squares — symmetric in (y, x); good when both legs are noisy."""
    y_a, x_a = y.values.astype(float), x.values.astype(float)
    y_c, x_c = y_a - y_a.mean(), x_a - x_a.mean()
    cov = np.cov(np.vstack([x_c, y_c]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, np.argmax(eigvals)]
    beta = float(v[1] / v[0]) if v[0] != 0 else 0.0
    alpha = float(y_a.mean() - beta * x_a.mean())
    resid = pd.Series(y_a - (alpha + beta * x_a), index=y.index, name="resid")
    return HedgeFit(alpha=alpha, beta=beta, resid=resid)


def kalman_hedge(
    y: pd.Series,
    x: pd.Series,
    delta: float = 1e-4,
    obs_var: float = 1e-3,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Recursive Kalman filter on β_t (and α_t) as a random walk.

    Returns (alpha_t, beta_t, spread_t). `delta` controls how fast the
    hedge ratio adapts: smaller ⇒ slower / smoother.
    """
    y_a = y.values.astype(float)
    x_a = x.values.astype(float)
    n = len(y_a)
    Vw = delta / (1 - delta) * np.eye(2)
    Ve = obs_var

    theta = np.zeros(2)
    P = np.zeros((2, 2))

    alphas = np.empty(n)
    betas = np.empty(n)
    spreads = np.empty(n)

    for t in range(n):
        F = np.array([1.0, x_a[t]])
        if t > 0:
            P = P + Vw  # state prediction covariance
        y_hat = F @ theta
        S = F @ P @ F + Ve
        e = y_a[t] - y_hat
        K = (P @ F) / S
        theta = theta + K * e
        P = P - np.outer(K, F) @ P

        alphas[t] = theta[0]
        betas[t] = theta[1]
        spreads[t] = e  # innovation = spread on a normalized scale

    idx = y.index
    return (
        pd.Series(alphas, index=idx, name="alpha"),
        pd.Series(betas, index=idx, name="beta"),
        pd.Series(spreads, index=idx, name="spread"),
    )
