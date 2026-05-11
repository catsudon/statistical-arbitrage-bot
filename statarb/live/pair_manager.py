"""Multi-slot pair manager for the live runner.

Manages N independent pair slots.  Each slot:
- holds one cointegrated pair + its strategy
- has its own rescan countdown (staggered so slots don't all rescan together)
- contributes 1/N of equity to the combined weight vector
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from ..core.logging import get_logger
from ..core.types import PairSpec
from ..data.ccxt_provider import CCXTProvider, panel_closes
from ..scanner.pairs_scanner import PairsScanConfig, scan_pairs
from ..strategies.base import Strategy

log = get_logger(__name__)

_TF_BARS: dict[str, int] = {
    "1m": 1440, "5m": 288, "15m": 96, "30m": 48,
    "1h": 24, "4h": 6, "1d": 1,
}


@dataclass
class PairManagerConfig:
    n_slots: int = 1
    universe_top: int = 50
    quote: str = "USDT"
    scan_days: int = 90
    rescan_interval_days: int = 30
    pvalue_max: float = 0.05
    workers: int = 4


@dataclass
class _Slot:
    idx: int
    pair: PairSpec | None = None
    strategy: Strategy | None = None
    bars_since_scan: int = 0


@dataclass
class SlotSwitch:
    slot_idx: int
    old_pair: PairSpec | None
    new_pair: PairSpec
    exited_symbols: set[str]   # symbols that left the slot (need weight→0)


class PairManager:
    """Manages `n_slots` independent pair positions with periodic re-scanning."""

    def __init__(
        self,
        provider: CCXTProvider,
        cfg: PairManagerConfig,
        timeframe: str,
        strategy_factory: Callable[[PairSpec], Strategy],
    ) -> None:
        self.provider = provider
        self.cfg = cfg
        self.timeframe = timeframe
        self.strategy_factory = strategy_factory
        self._bars_per_day = _TF_BARS.get(timeframe, 24)
        self._rescan_every = cfg.rescan_interval_days * self._bars_per_day
        # stagger slots so they don't all rescan on the same bar
        self.slots: list[_Slot] = [
            _Slot(idx=i, bars_since_scan=i * (self._rescan_every // max(cfg.n_slots, 1)))
            for i in range(cfg.n_slots)
        ]

    # ---- public properties ----

    @property
    def all_symbols(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for s in self.slots:
            if s.pair:
                for sym in (s.pair.y, s.pair.x):
                    if sym not in seen:
                        seen.add(sym)
                        out.append(sym)
        return out

    @property
    def active_slots(self) -> list[_Slot]:
        return [s for s in self.slots if s.pair is not None]

    # ---- scan helpers ----

    def _fetch_closes(self) -> pd.DataFrame:
        symbols = self.provider.top_by_volume(n=self.cfg.universe_top, quote=self.cfg.quote)
        end = pd.Timestamp.utcnow()
        start = end - pd.Timedelta(days=self.cfg.scan_days)
        min_bars = int(self.cfg.scan_days * self._bars_per_day * 0.85)
        return panel_closes(
            self.provider, symbols, self.timeframe,
            start=start, end=end, min_bars=min_bars,
        )

    def _pick_non_overlapping(
        self,
        pairs: list[PairSpec],
        occupied: set[str],
        n: int,
    ) -> list[PairSpec]:
        selected: list[PairSpec] = []
        used = set(occupied)
        for p in sorted(pairs, key=lambda p: p.pvalue):
            if p.y in used or p.x in used:
                continue
            selected.append(p)
            used.add(p.y)
            used.add(p.x)
            if len(selected) >= n:
                break
        return selected

    # ---- lifecycle ----

    def initial_scan(self) -> None:
        """Fill all slots at startup.  Called once before the main loop."""
        log.info("PairManager: initial scan for %d slots …", self.cfg.n_slots)
        closes = self._fetch_closes()
        if closes.empty:
            log.warning("PairManager: empty panel — all slots start empty")
            return
        pairs = scan_pairs(
            closes,
            PairsScanConfig(pvalue_max=self.cfg.pvalue_max, workers=self.cfg.workers, show_progress=False),
        )
        selected = self._pick_non_overlapping(pairs, set(), self.cfg.n_slots)
        for i, pair in enumerate(selected):
            self.slots[i].pair = pair
            self.slots[i].strategy = self.strategy_factory(pair)
            log.info("slot %d: %s / %s  pvalue=%.4f  hl=%.1f", i, pair.y, pair.x, pair.pvalue, pair.half_life)
        if len(selected) < self.cfg.n_slots:
            log.warning("only %d / %d slots filled after initial scan", len(selected), self.cfg.n_slots)

    def tick(self) -> None:
        """Advance bar counters for all slots.  Call once per bar."""
        for s in self.slots:
            s.bars_since_scan += 1

    def check_rescans(self) -> list[SlotSwitch]:
        """
        For each slot whose counter has expired, run a targeted re-scan.
        Returns a list of SlotSwitch objects for every slot that changed pair.
        """
        switches: list[SlotSwitch] = []
        for slot in self.slots:
            if slot.bars_since_scan < self._rescan_every:
                continue

            log.info("PairManager: slot %d rescan due (after %d bars) …", slot.idx, slot.bars_since_scan)
            occupied = {sym for s in self.slots if s.idx != slot.idx and s.pair
                        for sym in (s.pair.y, s.pair.x)}

            closes = self._fetch_closes()
            slot.bars_since_scan = 0

            if closes.empty:
                log.warning("slot %d: empty panel — keeping current pair", slot.idx)
                continue

            pairs = scan_pairs(
                closes,
                PairsScanConfig(pvalue_max=self.cfg.pvalue_max, workers=self.cfg.workers, show_progress=False),
            )
            candidates = self._pick_non_overlapping(pairs, occupied, 1)
            if not candidates:
                log.warning("slot %d: no suitable pair — keeping current", slot.idx)
                continue

            new_pair = candidates[0]
            same = (
                slot.pair is not None
                and new_pair.y == slot.pair.y
                and new_pair.x == slot.pair.x
            )
            if same:
                log.info("slot %d: same pair still best (%s/%s)", slot.idx, new_pair.y, new_pair.x)
                continue

            old_pair = slot.pair
            exited = {old_pair.y, old_pair.x} if old_pair else set()
            log.info(
                "slot %d: switching  %s/%s → %s/%s",
                slot.idx,
                old_pair.y if old_pair else "none", old_pair.x if old_pair else "none",
                new_pair.y, new_pair.x,
            )
            slot.pair = new_pair
            slot.strategy = self.strategy_factory(new_pair)
            switches.append(SlotSwitch(
                slot_idx=slot.idx,
                old_pair=old_pair,
                new_pair=new_pair,
                exited_symbols=exited - {new_pair.y, new_pair.x},
            ))

        return switches

    # ---- weight generation ----

    def combined_weights(self, closes: pd.DataFrame) -> pd.Series:
        """
        Generate weights for every active slot, scale each by 1/n_slots,
        and sum.  Returns a single weight Series for the latest bar.
        """
        combined: dict[str, float] = {}
        scale = 1.0 / self.cfg.n_slots

        for slot in self.active_slots:
            y, x = slot.pair.y, slot.pair.x
            if y not in closes.columns or x not in closes.columns:
                continue
            sub = closes[[y, x]].dropna()
            if len(sub) < 50:
                continue
            try:
                w_df = slot.strategy.generate_weights(sub)
                last = w_df.iloc[-1]
                for sym in (y, x):
                    combined[sym] = combined.get(sym, 0.0) + float(last.get(sym, 0.0)) * scale
            except Exception as e:
                log.warning("slot %d weight error: %s", slot.idx, e)

        return pd.Series(combined)

    def slot_summary(self) -> str:
        parts = []
        for s in self.slots:
            if s.pair:
                parts.append(f"[{s.idx}] {s.pair.y}/{s.pair.x} (next rescan in {self._rescan_every - s.bars_since_scan} bars)")
            else:
                parts.append(f"[{s.idx}] empty")
        return " | ".join(parts)
