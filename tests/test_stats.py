"""Sanity tests for the quant primitives. Uses synthetic data so no
network calls are needed."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.stats import (
    adf_pvalue,
    engle_granger,
    half_life,
    hurst_exponent,
    johansen,
    kalman_hedge,
    ols_hedge,
    rolling_zscore,
    tls_hedge,
    variance_ratio,
)


def _ar1(n: int, phi: float, sigma: float = 1.0, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    e = rng.normal(0, sigma, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return pd.Series(x)


def _random_walk(n: int, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(np.cumsum(rng.normal(0, 1, n)))


def test_adf_distinguishes_stationary_from_rw():
    stationary = _ar1(2000, phi=0.5, seed=0)
    rw = _random_walk(2000, seed=1)
    assert adf_pvalue(stationary) < 0.05
    assert adf_pvalue(rw) > 0.10


def test_hurst_signs_for_mean_reverting_and_trending():
    mr = _ar1(2000, phi=-0.3, seed=2)
    assert hurst_exponent(mr) < 0.5
    rw = _random_walk(5000, seed=3)
    assert 0.4 < hurst_exponent(rw) < 0.6


def test_half_life_recovers_known_phi():
    phi = 0.9
    s = _ar1(5000, phi=phi, seed=4)
    hl = half_life(s)
    expected = np.log(2) / -np.log(phi)
    # within 30% — single sample, finite-sample noise
    assert 0.7 * expected < hl < 1.3 * expected


def test_ols_and_tls_recover_beta():
    rng = np.random.default_rng(5)
    x = pd.Series(np.cumsum(rng.normal(0, 1, 2000)))
    beta_true = 1.7
    y = beta_true * x + rng.normal(0, 0.5, 2000)
    y = pd.Series(y)
    fit = ols_hedge(y, x)
    assert abs(fit.beta - beta_true) < 0.05
    fit_tls = tls_hedge(y, x)
    assert abs(fit_tls.beta - beta_true) < 0.1


def test_engle_granger_finds_cointegration():
    rng = np.random.default_rng(6)
    x = pd.Series(np.cumsum(rng.normal(0, 1, 2000)))
    spread = _ar1(2000, phi=0.7, seed=7)
    y = 1.2 * x + spread
    eg = engle_granger(y, x)
    assert eg.pvalue < 0.05
    assert abs(eg.beta - 1.2) < 0.1


def test_johansen_basket():
    rng = np.random.default_rng(8)
    base = pd.Series(np.cumsum(rng.normal(0, 1, 2000)))
    s1 = _ar1(2000, phi=0.6, seed=9)
    s2 = _ar1(2000, phi=0.7, seed=10)
    a = base
    b = 0.5 * base + s1
    c = 0.8 * base + s2
    df = pd.DataFrame({"a": a, "b": b, "c": c})
    jr = johansen(df)
    assert jr.n_coint >= 1


def test_kalman_tracks_beta_drift():
    rng = np.random.default_rng(11)
    n = 3000
    x = pd.Series(np.cumsum(rng.normal(0, 1, n)))
    beta_path = np.linspace(1.0, 2.0, n)
    y = pd.Series(beta_path * x.values + rng.normal(0, 0.3, n))
    _, beta_t, _ = kalman_hedge(y, x, delta=1e-3)
    # estimate beta near the end should be close to 2.0
    assert abs(beta_t.iloc[-200:].mean() - 2.0) < 0.3


def test_rolling_zscore_centered():
    s = _ar1(2000, phi=0.5, seed=12)
    z = rolling_zscore(s, lookback=200).dropna()
    assert abs(z.mean()) < 0.2
    assert 0.8 < z.std() < 1.2


def test_variance_ratio():
    mr = _ar1(2000, phi=-0.3, seed=13)
    assert variance_ratio(mr.cumsum().values, q=4) < 1.0
