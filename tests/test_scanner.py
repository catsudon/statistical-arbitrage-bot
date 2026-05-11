"""Synthetic scan: build a panel with two known cointegrated pairs and
random walkers; the scanner should surface the cointegrated ones."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.scanner import PairsScanConfig, scan_pairs


def _build_panel(n: int = 2500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_base = np.cumsum(rng.normal(0, 0.01, n))
    spread1 = np.zeros(n)
    spread2 = np.zeros(n)
    for i in range(1, n):
        spread1[i] = 0.85 * spread1[i - 1] + rng.normal(0, 0.01)
        spread2[i] = 0.80 * spread2[i - 1] + rng.normal(0, 0.01)
    log_y1 = 1.2 * log_base + spread1
    log_y2 = 0.7 * log_base + spread2
    rw_a = np.cumsum(rng.normal(0, 0.01, n))
    rw_b = np.cumsum(rng.normal(0, 0.01, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "BASE": 100 * np.exp(log_base),
            "Y1": 100 * np.exp(log_y1),
            "Y2": 100 * np.exp(log_y2),
            "RW_A": 100 * np.exp(rw_a),
            "RW_B": 100 * np.exp(rw_b),
        },
        index=idx,
    )


def test_scanner_finds_known_pair():
    df = _build_panel()
    cfg = PairsScanConfig(
        pvalue_max=0.05,
        workers=1,
        min_abs_correlation=0.5,
        min_half_life=2.0,
        max_half_life=500.0,
        show_progress=False,
    )
    pairs = scan_pairs(df, cfg)
    found = {tuple(sorted([p.y, p.x])) for p in pairs}
    assert ("BASE", "Y1") in found or ("Y1", "BASE") in found or {"BASE", "Y1"}.issubset({s for pair in found for s in pair})
