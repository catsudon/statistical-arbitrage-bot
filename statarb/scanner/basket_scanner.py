"""Greedy basket scanner using the Johansen test.

Strategy: cluster symbols by return correlation, then run Johansen on
each cluster. Keep clusters with rank ≥ 1 and a tradable half-life.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from ..core.logging import get_logger
from ..core.types import BasketSpec
from ..stats.cointegration import johansen

log = get_logger(__name__)


@dataclass
class BasketScanConfig:
    n_clusters: int = 10
    min_basket_size: int = 3
    max_basket_size: int = 6
    min_half_life: float = 6.0
    max_half_life: float = 240.0


def _cluster_symbols(closes: pd.DataFrame, n_clusters: int) -> dict[int, list[str]]:
    returns = np.log(closes).diff().dropna()
    corr = returns.corr().fillna(0.0).values
    dist = 1.0 - corr
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0, 2)
    model = AgglomerativeClustering(
        n_clusters=min(n_clusters, len(returns.columns)),
        metric="precomputed",
        linkage="average",
    )
    labels = model.fit_predict(dist)
    out: dict[int, list[str]] = {}
    for sym, lbl in zip(returns.columns, labels):
        out.setdefault(int(lbl), []).append(sym)
    return out


def scan_baskets(closes: pd.DataFrame, cfg: BasketScanConfig | None = None) -> list[BasketSpec]:
    cfg = cfg or BasketScanConfig()
    log_px = np.log(closes)
    clusters = _cluster_symbols(closes, cfg.n_clusters)

    baskets: list[BasketSpec] = []
    for lbl, syms in clusters.items():
        if not (cfg.min_basket_size <= len(syms) <= cfg.max_basket_size):
            continue
        try:
            jr = johansen(log_px[syms])
        except Exception as e:
            log.debug("johansen failed on cluster %s: %s", lbl, e)
            continue
        if jr.n_coint < 1:
            continue
        if not (cfg.min_half_life <= jr.half_life <= cfg.max_half_life):
            continue
        baskets.append(
            BasketSpec(
                symbols=list(syms),
                weights=jr.weights.tolist(),
                half_life=jr.half_life,
                eig_stat=float(jr.eig_stats[0]),
                meta={"cluster": lbl, "n_coint": jr.n_coint},
            )
        )
    baskets.sort(key=lambda b: b.eig_stat, reverse=True)
    log.info("kept %d baskets", len(baskets))
    return baskets
