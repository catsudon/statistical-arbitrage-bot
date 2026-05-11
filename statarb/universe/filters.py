"""Universe filters: liquidity, history length, stablecoin exclusion."""
from __future__ import annotations

import pandas as pd


def filter_universe(
    closes: pd.DataFrame,
    volumes: pd.DataFrame | None = None,
    min_history_bars: int = 2000,
    min_quote_volume_usd: float | None = None,
    exclude: list[str] | None = None,
    max_zero_volume_frac: float = 0.05,
) -> list[str]:
    """Apply liquidity and history filters to a price panel.

    `closes` and `volumes` are wide DataFrames keyed by symbol. Returns
    the filtered symbol list.
    """
    exclude = set(exclude or [])
    keep: list[str] = []
    for s in closes.columns:
        if s in exclude:
            continue
        col = closes[s].dropna()
        if len(col) < min_history_bars:
            continue
        if volumes is not None and s in volumes.columns:
            v = volumes[s].dropna()
            if (v == 0).mean() > max_zero_volume_frac:
                continue
            if min_quote_volume_usd is not None:
                avg_qv = (v * closes[s]).rolling(168, min_periods=24).mean().iloc[-1]
                if pd.isna(avg_qv) or avg_qv < min_quote_volume_usd:
                    continue
        keep.append(s)
    return keep
