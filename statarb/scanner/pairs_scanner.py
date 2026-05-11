"""Parallel pair scanner.

For every (y, x) with sufficient correlation, we run an Engle-Granger
cointegration test on log prices and keep pairs with a stationary spread
and a tradable half-life.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from ..core.logging import get_logger
from ..core.types import PairSpec
from ..stats.cointegration import engle_granger

log = get_logger(__name__)


@dataclass
class PairsScanConfig:
    pvalue_max: float = 0.05
    min_abs_correlation: float = 0.7
    min_half_life: float = 6.0
    max_half_life: float = 240.0
    use_log_prices: bool = True
    workers: int = 8
    show_progress: bool = True


def _eval_pair(y_name: str, x_name: str, y: pd.Series, x: pd.Series, cfg: PairsScanConfig) -> PairSpec | None:
    if cfg.use_log_prices:
        y = np.log(y)
        x = np.log(x)
    aligned = pd.concat([y, x], axis=1, join="inner").dropna()
    if len(aligned) < 200:
        return None
    aligned.columns = ["y", "x"]
    corr = aligned["y"].corr(aligned["x"])
    if abs(corr) < cfg.min_abs_correlation:
        return None
    eg = engle_granger(aligned["y"], aligned["x"])
    if eg.pvalue > cfg.pvalue_max:
        return None
    if not (cfg.min_half_life <= eg.half_life <= cfg.max_half_life):
        return None
    # combined score: prefer small p, tradable half-life, high |corr|
    score = (1 - eg.pvalue) * abs(corr) / np.log(1 + max(eg.half_life, 1.0))
    return PairSpec(
        y=y_name,
        x=x_name,
        beta=eg.beta,
        alpha=eg.alpha,
        half_life=eg.half_life,
        pvalue=eg.pvalue,
        score=float(score),
        meta={"correlation": float(corr), "tstat": eg.tstat, "log_prices": cfg.use_log_prices},
    )


def scan_pairs(closes: pd.DataFrame, cfg: PairsScanConfig | None = None) -> list[PairSpec]:
    cfg = cfg or PairsScanConfig()
    symbols = list(closes.columns)
    log.info("scanning %d symbols → %d candidate pairs", len(symbols), len(symbols) * (len(symbols) - 1) // 2)

    pairs = list(combinations(symbols, 2))

    def _task(a: str, b: str):
        return _eval_pair(a, b, closes[a], closes[b], cfg)

    iterable = pairs
    if cfg.show_progress:
        iterable = tqdm(pairs, desc="scan_pairs", total=len(pairs))

    results = Parallel(n_jobs=cfg.workers, prefer="processes")(
        delayed(_task)(a, b) for a, b in iterable
    )
    kept = [r for r in results if r is not None]
    kept.sort(key=lambda p: p.score, reverse=True)
    log.info("kept %d / %d pairs", len(kept), len(pairs))
    return kept


def pairs_to_dataframe(pairs: list[PairSpec]) -> pd.DataFrame:
    rows = []
    for p in pairs:
        rows.append(
            {
                "y": p.y,
                "x": p.x,
                "beta": p.beta,
                "alpha": p.alpha,
                "half_life": p.half_life,
                "pvalue": p.pvalue,
                "correlation": p.meta.get("correlation"),
                "score": p.score,
            }
        )
    return pd.DataFrame(rows)
