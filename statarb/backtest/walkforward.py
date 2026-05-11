"""Walk-forward validation for stat-arb strategies.

Each fold:
    - train window: scan for cointegrated pairs
    - test  window: backtest *only* the pairs found in that fold's scan
                    using data the scanner never saw

Aggregates per-pair OOS metrics across folds to surface which pairs are
consistently profitable rather than in-sample flukes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..core.logging import get_logger
from ..core.types import PairSpec
from ..scanner.pairs_scanner import PairsScanConfig, scan_pairs
from ..signals.zscore import ZScoreParams
from ..strategies.pairs_trading import PairStrategyConfig, PairsStrategy
from .costs import CostModel
from .metrics import BacktestMetrics, compute_metrics
from .vectorized import vectorized_backtest

log = get_logger(__name__)


@dataclass
class FoldResult:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_pairs_found: int
    rows: list[dict] = field(default_factory=list)  # one row per pair in this fold


@dataclass
class WalkForwardResult:
    folds: list[FoldResult]
    pair_summary: pd.DataFrame   # per-pair across folds
    fold_summary: pd.DataFrame   # per-fold aggregated


def run_walk_forward(
    closes: pd.DataFrame,
    scan_cfg: PairsScanConfig,
    strategy_cfg: PairStrategyConfig,
    cost: CostModel,
    scan_bars: int,
    test_bars: int,
    step_bars: int | None = None,
    timeframe: str = "1h",
) -> WalkForwardResult:
    """
    Parameters
    ----------
    closes     : full aligned price panel
    scan_cfg   : pairs scanner config
    strategy_cfg: strategy params (entry/exit/stop/lookback)
    cost       : fee/slippage model
    scan_bars  : number of bars in each training window
    test_bars  : number of bars in each OOS test window
    step_bars  : how far to roll between folds (default = test_bars, non-overlapping)
    timeframe  : for annualizing metrics
    """
    step_bars = step_bars or test_bars
    n = len(closes)
    folds: list[FoldResult] = []

    fold_idx = 0
    train_start_i = 0
    while True:
        train_end_i = train_start_i + scan_bars
        test_end_i = train_end_i + test_bars
        if test_end_i > n:
            break

        train_close = closes.iloc[train_start_i:train_end_i]
        test_close = closes.iloc[train_end_i:test_end_i]

        log.info(
            "fold %d | train [%s → %s] | test [%s → %s]",
            fold_idx,
            train_close.index[0].date(), train_close.index[-1].date(),
            test_close.index[0].date(), test_close.index[-1].date(),
        )

        pairs = scan_pairs(train_close, scan_cfg)
        fold = FoldResult(
            fold=fold_idx,
            train_start=train_close.index[0],
            train_end=train_close.index[-1],
            test_start=test_close.index[0],
            test_end=test_close.index[-1],
            n_pairs_found=len(pairs),
        )

        for pair in pairs:
            if pair.y not in test_close.columns or pair.x not in test_close.columns:
                continue
            sub = test_close[[pair.y, pair.x]].dropna()
            if len(sub) < 50:
                continue
            try:
                strat = PairsStrategy(pair, strategy_cfg)
                w = strat.generate_weights(sub)
                res = vectorized_backtest(sub, w, cost=cost, timeframe=timeframe)
                m = res.metrics
                fold.rows.append({
                    "fold": fold_idx,
                    "y": pair.y, "x": pair.x,
                    "scan_half_life": round(pair.half_life, 1),
                    "scan_pvalue": round(pair.pvalue, 4),
                    "oos_sharpe": m.sharpe,
                    "oos_return": m.total_return,
                    "oos_max_dd": m.max_drawdown,
                    "oos_turnover": m.turnover_annual,
                    "oos_n_trades": m.n_trades,
                    "test_bars": len(sub),
                })
            except Exception as e:
                log.debug("fold %d pair %s/%s failed: %s", fold_idx, pair.y, pair.x, e)

        folds.append(fold)
        train_start_i += step_bars
        fold_idx += 1

    if not folds:
        return WalkForwardResult(folds=[], pair_summary=pd.DataFrame(), fold_summary=pd.DataFrame())

    all_rows = [r for f in folds for r in f.rows]
    if not all_rows:
        return WalkForwardResult(folds=folds, pair_summary=pd.DataFrame(), fold_summary=pd.DataFrame())

    df = pd.DataFrame(all_rows)

    # ---- per-pair summary ----
    pair_summary = (
        df.groupby(["y", "x"])
        .agg(
            n_folds=("fold", "count"),
            mean_oos_sharpe=("oos_sharpe", "mean"),
            mean_oos_return=("oos_return", "mean"),
            worst_oos_dd=("oos_max_dd", "min"),
            pct_folds_positive=("oos_return", lambda s: (s > 0).mean()),
            mean_turnover=("oos_turnover", "mean"),
            total_trades=("oos_n_trades", "sum"),
        )
        .reset_index()
        .sort_values("mean_oos_sharpe", ascending=False)
    )

    # ---- per-fold summary ----
    fold_summary = (
        df.groupby("fold")
        .agg(
            n_pairs=("y", "count"),
            mean_sharpe=("oos_sharpe", "mean"),
            pct_positive=("oos_return", lambda s: (s > 0).mean()),
        )
        .reset_index()
    )

    return WalkForwardResult(folds=folds, pair_summary=pair_summary, fold_summary=fold_summary)
