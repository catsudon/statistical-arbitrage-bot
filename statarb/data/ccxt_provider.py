"""Synchronous CCXT-based provider. Handles pagination and rate-limits.

For very large universes prefer the async version (`ccxt.async_support`);
this sync version is the easier-to-debug default.
"""
from __future__ import annotations

import time
from typing import Any

import ccxt
import pandas as pd

from ..core.logging import get_logger
from ..core.secrets import ExchangeCredentials, get_credentials
from .cache import ParquetCache

log = get_logger(__name__)

_TF_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


class CCXTProvider:
    """Thin wrapper around a `ccxt` exchange object with Parquet caching."""

    def __init__(
        self,
        exchange: str = "binance",
        cache: ParquetCache | None = None,
        rate_limit_ms: int = 250,
        params: dict[str, Any] | None = None,
        credentials: ExchangeCredentials | None = None,
        use_env_credentials: bool = True,
    ):
        cls = getattr(ccxt, exchange)
        self.exchange_id = exchange
        creds = credentials if credentials is not None else (
            get_credentials(exchange) if use_env_credentials else ExchangeCredentials()
        )
        cfg: dict[str, Any] = {"enableRateLimit": True, **(params or {})}
        if creds.api_key:
            cfg["apiKey"] = creds.api_key
        if creds.api_secret:
            cfg["secret"] = creds.api_secret
        if creds.passphrase:
            cfg["password"] = creds.passphrase
        self.ex = cls(cfg)
        if creds.testnet and hasattr(self.ex, "set_sandbox_mode"):
            self.ex.set_sandbox_mode(True)
        self.credentials = creds
        self.rate_limit_ms = rate_limit_ms
        self.cache = cache or ParquetCache()
        self._markets: dict[str, Any] | None = None

    # ---------- universe ----------
    def _load_markets(self) -> dict[str, Any]:
        if self._markets is None:
            self._markets = self.ex.load_markets()
        return self._markets

    def list_symbols(self, quote: str | None = "USDT", spot_only: bool = True) -> list[str]:
        markets = self._load_markets()
        out = []
        for sym, m in markets.items():
            if quote and m.get("quote") != quote:
                continue
            if spot_only and not m.get("spot", True):
                continue
            if not m.get("active", True):
                continue
            out.append(sym)
        return sorted(out)

    def top_by_volume(self, n: int, quote: str = "USDT") -> list[str]:
        tickers = self.ex.fetch_tickers()
        rows = []
        for sym, t in tickers.items():
            mkt = self._load_markets().get(sym)
            if not mkt or mkt.get("quote") != quote:
                continue
            qv = t.get("quoteVolume") or 0.0
            rows.append((sym, qv))
        rows.sort(key=lambda r: r[1], reverse=True)
        return [s for s, _ in rows[:n]]

    # ---------- OHLCV ----------
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        if timeframe not in _TF_MS:
            raise ValueError(f"unsupported timeframe: {timeframe}")

        end = end or pd.Timestamp.utcnow().floor(timeframe.replace("m", "min"))
        start = start or (end - pd.Timedelta(days=365))

        if use_cache:
            cached = self.cache.read(self.exchange_id, symbol, timeframe)
            if not cached.empty:
                have_max = cached.index.max()
                if have_max >= end - pd.Timedelta(milliseconds=_TF_MS[timeframe]):
                    return cached.loc[(cached.index >= start) & (cached.index <= end)]
                start_fetch = have_max + pd.Timedelta(milliseconds=_TF_MS[timeframe])
            else:
                start_fetch = start
        else:
            start_fetch = start

        rows = self._paginated_fetch(symbol, timeframe, start_fetch, end)
        if not rows:
            return self.cache.read(self.exchange_id, symbol, timeframe, start, end)

        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.drop_duplicates(subset="ts").set_index("ts").sort_index()

        if use_cache:
            self.cache.write(self.exchange_id, symbol, timeframe, df)
            df = self.cache.read(self.exchange_id, symbol, timeframe, start, end)
        return df

    def _paginated_fetch(
        self,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> list[list[float]]:
        step_ms = _TF_MS[timeframe]
        since = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        limit = 1000
        all_rows: list[list[float]] = []
        while since < end_ms:
            try:
                batch = self.ex.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            except ccxt.NetworkError as e:
                log.warning("network error on %s: %s; sleeping", symbol, e)
                time.sleep(1.0)
                continue
            except ccxt.ExchangeError as e:
                log.warning("exchange error on %s: %s; bailing", symbol, e)
                break
            if not batch:
                break
            all_rows.extend(batch)
            last = batch[-1][0]
            if last <= since:
                break
            since = last + step_ms
            time.sleep(self.rate_limit_ms / 1000.0)
        return all_rows


def panel_closes(
    provider: CCXTProvider,
    symbols: list[str],
    timeframe: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    min_bars: int | None = None,
) -> pd.DataFrame:
    """Return a wide DataFrame of close prices: index=time, columns=symbols.

    `min_bars` pre-filters symbols that don't have enough individual history
    before building the intersection — prevents one new coin from shrinking
    the whole panel to its listing date.
    """
    out: dict[str, pd.Series] = {}
    for s in symbols:
        df = provider.fetch_ohlcv(s, timeframe, start, end)
        if df.empty:
            continue
        if min_bars and len(df) < min_bars:
            log.debug("dropping %s: only %d bars (min=%d)", s, len(df), min_bars)
            continue
        out[s] = df["close"]
    if not out:
        return pd.DataFrame()
    wide = pd.concat(out, axis=1).sort_index()
    wide = wide.ffill().dropna(how="any")
    return wide
