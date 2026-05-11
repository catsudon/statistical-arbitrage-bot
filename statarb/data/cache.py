"""Parquet-backed OHLCV cache, keyed by (exchange, symbol, timeframe).

The cache is append-only and idempotent: re-fetching a window that's
already on disk costs one Parquet read.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from filelock import FileLock

_SAFE = re.compile(r"[^A-Za-z0-9_-]+")


def _safe(s: str) -> str:
    return _SAFE.sub("_", s)


class ParquetCache:
    def __init__(self, root: str | Path = ".cache/statarb"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, exchange: str, symbol: str, timeframe: str) -> Path:
        return self.root / _safe(exchange) / _safe(timeframe) / f"{_safe(symbol)}.parquet"

    def _lock(self, p: Path) -> FileLock:
        return FileLock(str(p) + ".lock")

    def read(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        p = self.path(exchange, symbol, timeframe)
        if not p.exists():
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.read_parquet(p)
        if start is not None:
            df = df.loc[df.index >= start]
        if end is not None:
            df = df.loc[df.index <= end]
        return df

    def write(self, exchange: str, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        p = self.path(exchange, symbol, timeframe)
        p.parent.mkdir(parents=True, exist_ok=True)
        with self._lock(p):
            if p.exists():
                existing = pd.read_parquet(p)
                df = pd.concat([existing, df])
                df = df[~df.index.duplicated(keep="last")].sort_index()
            df.to_parquet(p)

    def coverage(self, exchange: str, symbol: str, timeframe: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        p = self.path(exchange, symbol, timeframe)
        if not p.exists():
            return None
        df = pd.read_parquet(p, columns=["close"])
        if df.empty:
            return None
        return df.index.min(), df.index.max()
