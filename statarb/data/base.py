"""Provider interface. Any data source — REST API, CSV dump, on-disk
parquet — should implement these two methods."""
from __future__ import annotations

from typing import Protocol

import pandas as pd


class DataProvider(Protocol):
    """Read-only market-data provider."""

    def list_symbols(self, quote: str | None = None) -> list[str]: ...

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return a DataFrame indexed by UTC `ts` with columns
        `open, high, low, close, volume`."""
        ...
