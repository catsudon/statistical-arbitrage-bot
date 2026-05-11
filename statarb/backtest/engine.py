"""Event-driven backtester.

Iterates bar-by-bar, calls the strategy's `generate_weights` only on
data up to the current bar (so any look-ahead in the strategy itself is
exposed), then asks the paper broker to rebalance.

This is slower than the vectorized version but enforces realistic
fill mechanics, funding accrual, and supports stateful position logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.logging import get_logger
from ..core.types import Fill, Side
from ..execution.paper import PaperBroker
from ..strategies.base import Strategy
from .costs import CostModel
from .metrics import BacktestMetrics, compute_metrics

log = get_logger(__name__)


@dataclass
class EventResult:
    equity: pd.Series
    weights: pd.DataFrame
    fills: list[Fill]
    metrics: BacktestMetrics
    diagnostics: dict = field(default_factory=dict)


def event_backtest(
    closes: pd.DataFrame,
    strategy: Strategy,
    cost: CostModel | None = None,
    initial_capital: float = 100_000.0,
    timeframe: str = "1h",
    warmup_bars: int = 250,
    rebalance_every: int = 1,
) -> EventResult:
    cost = cost or CostModel()
    broker = PaperBroker(initial_capital=initial_capital, cost=cost)
    strategy.fit(closes.iloc[:warmup_bars])

    equity_curve: list[float] = []
    weights_log: list[pd.Series] = []
    ts_log: list[pd.Timestamp] = []

    for i, ts in enumerate(closes.index):
        if i < warmup_bars:
            equity_curve.append(broker.equity({s: closes[s].iloc[i] for s in closes.columns}))
            ts_log.append(ts)
            weights_log.append(pd.Series(0.0, index=closes.columns))
            continue

        prices = {s: closes[s].iloc[i] for s in closes.columns}

        if (i - warmup_bars) % rebalance_every == 0:
            # Generate weights on data up to and including current bar
            window = closes.iloc[: i + 1]
            target = strategy.generate_weights(window).iloc[-1]
            broker.rebalance_to_weights(ts=ts, target_weights=target.to_dict(), prices=prices)

        eq = broker.equity(prices)
        equity_curve.append(eq)
        ts_log.append(ts)
        weights_log.append(broker.current_weights(prices))

    equity = pd.Series(equity_curve, index=pd.Index(ts_log, name="ts"), name="equity")
    weights_df = pd.DataFrame(weights_log, index=ts_log).fillna(0.0)
    metrics = compute_metrics(equity, weights=weights_df, timeframe=timeframe)
    return EventResult(
        equity=equity,
        weights=weights_df,
        fills=list(broker.fills),
        metrics=metrics,
        diagnostics={"n_fills": len(broker.fills)},
    )
