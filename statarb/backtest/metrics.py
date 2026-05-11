"""Performance & risk metrics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..risk.position_sizing import annualize_factor


@dataclass
class BacktestMetrics:
    total_return: float
    cagr: float
    annual_vol: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    hit_rate: float
    turnover_annual: float
    avg_exposure: float
    skew: float
    kurtosis: float
    n_trades: int

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def compute_metrics(
    equity: pd.Series,
    weights: pd.DataFrame | None = None,
    timeframe: str = "1h",
    rf: float = 0.0,
) -> BacktestMetrics:
    eq = equity.dropna()
    if len(eq) < 2:
        return BacktestMetrics(*([0.0] * 12), 0)

    af = annualize_factor(timeframe)
    ret = eq.pct_change().dropna()
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    years = max(len(ret) / af, 1e-9)
    cagr = (1 + total) ** (1 / years) - 1 if (1 + total) > 0 else -1.0
    vol = float(ret.std(ddof=1) * np.sqrt(af))
    excess = ret - rf / af
    sharpe = float(excess.mean() / ret.std(ddof=1) * np.sqrt(af)) if ret.std() > 0 else 0.0
    downside = ret[ret < 0]
    sortino = (
        float(excess.mean() / downside.std(ddof=1) * np.sqrt(af))
        if len(downside) > 1 and downside.std() > 0
        else 0.0
    )
    peak = eq.cummax()
    dd = (eq / peak - 1.0).min()
    calmar = float(cagr / abs(dd)) if dd < 0 else 0.0
    hit = float((ret > 0).mean())

    turnover_annual = 0.0
    avg_exposure = 0.0
    n_trades = 0
    if weights is not None and not weights.empty:
        dw = weights.diff().abs().sum(axis=1)
        turnover_annual = float(dw.mean() * af)
        avg_exposure = float(weights.abs().sum(axis=1).mean())
        # crude trade count: number of zero-crossings per symbol
        for c in weights.columns:
            sgn = np.sign(weights[c].values)
            n_trades += int(np.sum(np.diff(sgn) != 0))

    return BacktestMetrics(
        total_return=total,
        cagr=float(cagr),
        annual_vol=vol,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=float(dd),
        hit_rate=hit,
        turnover_annual=turnover_annual,
        avg_exposure=avg_exposure,
        skew=float(ret.skew()),
        kurtosis=float(ret.kurt()),
        n_trades=n_trades,
    )
