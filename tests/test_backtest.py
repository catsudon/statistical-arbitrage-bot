"""Smoke tests for the backtester on synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.backtest import CostModel, event_backtest, vectorized_backtest
from statarb.core.types import PairSpec
from statarb.signals import ZScoreParams
from statarb.strategies.pairs_trading import PairsStrategy, PairStrategyConfig


def _synthetic_pair(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Cointegrated *log* prices: log_y = 1.5 * log_x + AR(1) spread."""
    rng = np.random.default_rng(seed)
    log_x = np.cumsum(rng.normal(0, 0.01, n))
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = 0.8 * spread[i - 1] + rng.normal(0, 0.01)
    log_y = 1.5 * log_x + spread
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"Y": 100 * np.exp(log_y), "X": 100 * np.exp(log_x)}, index=idx)


def test_vectorized_runs_and_has_metrics():
    closes = _synthetic_pair()
    pair = PairSpec(y="Y", x="X", beta=1.5, alpha=0.0, half_life=10, pvalue=0.01)
    strat = PairsStrategy(pair, PairStrategyConfig(zscore=ZScoreParams(lookback=200)))
    w = strat.generate_weights(closes)
    res = vectorized_backtest(closes, w, cost=CostModel(fee_bps=2, slippage_bps=1),
                              initial_capital=10_000, timeframe="1h")
    assert res.equity.iloc[-1] > 0
    assert res.metrics.n_trades > 0


def test_event_driven_runs():
    closes = _synthetic_pair()
    pair = PairSpec(y="Y", x="X", beta=1.5, alpha=0.0, half_life=10, pvalue=0.01)
    strat = PairsStrategy(pair, PairStrategyConfig(zscore=ZScoreParams(lookback=200)))
    res = event_backtest(closes, strat, cost=CostModel(fee_bps=2, slippage_bps=1),
                        initial_capital=10_000, timeframe="1h", warmup_bars=300, rebalance_every=5)
    assert res.equity.iloc[-1] > 0
    assert len(res.fills) > 0
