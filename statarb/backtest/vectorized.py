"""Fast vectorized backtester.

Approach: target weights at bar t are applied to per-bar returns at t+1
(no look-ahead). Costs charged on weight turnover. Suitable for parameter
sweeps and quick iteration. For order-book accuracy, use `engine.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import CostModel
from .metrics import BacktestMetrics, compute_metrics


@dataclass
class VectorizedResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    pnl_by_symbol: pd.DataFrame
    costs: pd.Series
    metrics: BacktestMetrics


def vectorized_backtest(
    closes: pd.DataFrame,
    weights: pd.DataFrame,
    cost: CostModel | None = None,
    initial_capital: float = 100_000.0,
    timeframe: str = "1h",
    lag: int = 1,
) -> VectorizedResult:
    """Run a vectorized backtest.

    `weights[t]` is the target portfolio weight at the *close* of bar t,
    applied to the return from t to t+lag.
    """
    cost = cost or CostModel()
    closes = closes.sort_index()
    weights = weights.reindex_like(closes).fillna(0.0)
    ret = closes.pct_change().fillna(0.0)

    # Shift weights forward by `lag` to enforce no look-ahead.
    w = weights.shift(lag).fillna(0.0)

    # PnL by symbol = w_{t-1} * r_t
    pnl_by_symbol = w * ret
    gross_ret = pnl_by_symbol.sum(axis=1)

    # turnover = sum |Δw| per bar
    dw = w.diff().abs().sum(axis=1).fillna(0.0)
    cost_series = dw.apply(cost.trade_cost)

    net_ret = gross_ret - cost_series
    equity = (1 + net_ret).cumprod() * initial_capital

    metrics = compute_metrics(equity, weights=w, timeframe=timeframe)
    return VectorizedResult(
        equity=equity.rename("equity"),
        returns=net_ret.rename("ret"),
        weights=w,
        pnl_by_symbol=pnl_by_symbol,
        costs=cost_series.rename("cost"),
        metrics=metrics,
    )


def sweep(
    closes: pd.DataFrame,
    weight_fn,
    grid: list[dict],
    cost: CostModel | None = None,
    timeframe: str = "1h",
) -> pd.DataFrame:
    """Run `weight_fn(closes, **params)` over a parameter grid; return a
    DataFrame of metrics, one row per param-set."""
    rows = []
    for params in grid:
        w = weight_fn(closes, **params)
        res = vectorized_backtest(closes, w, cost=cost, timeframe=timeframe)
        rows.append({**params, **res.metrics.to_dict()})
    return pd.DataFrame(rows)
