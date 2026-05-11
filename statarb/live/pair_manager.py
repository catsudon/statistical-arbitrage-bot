"""Periodic pair re-scanner for the live runner.

Every `rescan_interval_days` the manager fetches fresh universe data,
runs the cointegration scanner, and returns the best pair.  If the
best pair has changed the runner closes current positions and switches.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..core.logging import get_logger
from ..core.types import PairSpec
from ..data.ccxt_provider import CCXTProvider, panel_closes
from ..scanner.pairs_scanner import PairsScanConfig, scan_pairs

log = get_logger(__name__)

_TF_BARS: dict[str, int] = {
    "1m": 1440, "5m": 288, "15m": 96, "30m": 48,
    "1h": 24, "4h": 6, "1d": 1,
}


@dataclass
class PairManagerConfig:
    universe_top: int = 50
    quote: str = "USDT"
    scan_days: int = 90          # training window for each scan
    rescan_interval_days: int = 30   # how often to re-scan
    pvalue_max: float = 0.05
    workers: int = 4


class PairManager:
    """Tracks the active pair and decides when to re-scan."""

    def __init__(
        self,
        provider: CCXTProvider,
        cfg: PairManagerConfig,
        timeframe: str = "1h",
    ) -> None:
        self.provider = provider
        self.cfg = cfg
        self.timeframe = timeframe
        self._bars_per_day = _TF_BARS.get(timeframe, 24)
        self._rescan_every = cfg.rescan_interval_days * self._bars_per_day
        self._bars_since_scan: int = 0
        self.current_pair: PairSpec | None = None

    @property
    def rescan_due(self) -> bool:
        return self._bars_since_scan >= self._rescan_every

    def tick(self) -> None:
        """Advance internal bar counter by one."""
        self._bars_since_scan += 1

    def scan_now(self) -> PairSpec | None:
        """Fetch universe, run scanner, return best pair (lowest p-value)."""
        log.info(
            "PairManager: scanning top-%d universe over %dd …",
            self.cfg.universe_top, self.cfg.scan_days,
        )
        symbols = self.provider.top_by_volume(n=self.cfg.universe_top, quote=self.cfg.quote)
        end = pd.Timestamp.utcnow()
        start = end - pd.Timedelta(days=self.cfg.scan_days)
        min_bars = int(self.cfg.scan_days * self._bars_per_day * 0.85)

        closes = panel_closes(
            self.provider, symbols, self.timeframe,
            start=start, end=end, min_bars=min_bars,
        )
        if closes.empty or closes.shape[1] < 2:
            log.warning("PairManager: panel too small — no scan performed")
            return None

        pairs = scan_pairs(
            closes,
            PairsScanConfig(pvalue_max=self.cfg.pvalue_max, workers=self.cfg.workers, show_progress=False),
        )
        if not pairs:
            log.warning("PairManager: no cointegrated pairs found in this window")
            return None

        best = min(pairs, key=lambda p: p.pvalue)
        log.info(
            "PairManager: best pair = %s/%s  pvalue=%.4f  half_life=%.1f",
            best.y, best.x, best.pvalue, best.half_life,
        )
        self._bars_since_scan = 0
        return best

    def maybe_rescan(self) -> tuple[PairSpec | None, bool]:
        """
        Call once per bar.  Returns (pair, switched).

        - (current_pair, False) — rescan not due yet, no change
        - (current_pair, False) — rescan ran, same pair still best
        - (new_pair,     True)  — rescan ran, pair has changed
        """
        if not self.rescan_due:
            return self.current_pair, False

        new_pair = self.scan_now()
        if new_pair is None:
            return self.current_pair, False

        switched = (
            self.current_pair is None
            or new_pair.y != self.current_pair.y
            or new_pair.x != self.current_pair.x
        )
        self.current_pair = new_pair
        return new_pair, switched
