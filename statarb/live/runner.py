"""Live runner — production-ready loop.

Features:
- Kill switch: create `.killswitch` in CWD to stop gracefully
- Drawdown monitoring: halt + alert when max_dd breached
- Exponential backoff on errors (max 5 min)
- Startup checklist: credentials, reconciliation, alert on launch
- Per-cycle equity + drawdown logged every bar
- Multi-slot PairManager: N independent pair positions, auto re-scan
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..core.logging import get_logger
from ..data.ccxt_provider import CCXTProvider, panel_closes
from ..execution.alerts import Alerter
from ..execution.base import Broker
from ..strategies.base import Strategy

log = get_logger(__name__)

KILLSWITCH = Path(".killswitch")

_TF_SEC = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "4h": 14400, "1d": 86400,
}


@dataclass
class RunnerConfig:
    symbols: list[str]           # used only in fixed-pair mode
    timeframe: str = "1h"
    history_bars: int = 1500
    max_drawdown: float = 0.20
    max_consecutive_errors: int = 10


@dataclass
class RunnerState:
    peak_equity: float = 0.0
    cycle: int = 0
    consecutive_errors: int = 0
    equity_history: list[float] = field(default_factory=list)

    def update_equity(self, eq: float) -> None:
        self.equity_history.append(eq)
        if eq > self.peak_equity:
            self.peak_equity = eq

    @property
    def current_drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        eq = self.equity_history[-1] if self.equity_history else self.peak_equity
        return eq / self.peak_equity - 1.0


def _next_bar_close(timeframe: str) -> pd.Timestamp:
    sec = _TF_SEC[timeframe]
    now = pd.Timestamp.utcnow().timestamp()
    return pd.Timestamp(((int(now) // sec) + 1) * sec, unit="s", tz="UTC")


def _backoff(attempt: int, base: float = 5.0, cap: float = 300.0) -> float:
    return min(cap, base * (2 ** attempt))


def run_live(
    provider: CCXTProvider,
    broker: Broker,
    strategy: Strategy | None,
    cfg: RunnerConfig,
    alerter: Alerter | None = None,
    pair_manager=None,    # PairManager | None
) -> None:
    """
    Fixed-pair mode : pass a pre-built `strategy`, leave `pair_manager=None`
    Auto-scan mode  : pass `pair_manager` (PairManager), leave `strategy=None`
    """
    alerter = alerter or Alerter()
    state = RunnerState()
    mode = "DRY-RUN" if getattr(broker, "dry_run", True) else "LIVE"
    auto_mode = pair_manager is not None

    log.info("=== statarb %s starting (%s) ===", mode, "AUTO-SCAN" if auto_mode else "FIXED-PAIR")

    # ---- auto mode: initial scan ----
    if auto_mode:
        pair_manager.initial_scan()
        if not pair_manager.active_slots:
            log.error("initial scan found no pairs — aborting")
            alerter.alert_no_pair()
            return
        active_symbols = pair_manager.all_symbols
        log.info("slots after initial scan: %s", pair_manager.slot_summary())
    else:
        active_symbols = cfg.symbols

    if hasattr(broker, "reconcile_on_startup"):
        broker.reconcile_on_startup(active_symbols)

    prices_init = _latest_prices(provider, active_symbols, cfg.timeframe)
    eq0 = broker.equity(prices_init)
    state.peak_equity = eq0
    alerter.alert_startup(
        strategy.name if strategy else f"auto-{pair_manager.cfg.n_slots}-slots",
        mode,
        active_symbols,
    )
    log.info("starting equity: %.2f USDT", eq0)

    # ---- main loop ----
    while True:
        if KILLSWITCH.exists():
            log.info("kill switch active — shutting down")
            alerter.alert_shutdown("kill switch")
            break

        target = _next_bar_close(cfg.timeframe)
        sleep_for = max(0.0, target.timestamp() - pd.Timestamp.utcnow().timestamp() + 2.0)
        log.info("cycle %d — sleeping %.0fs until %s UTC", state.cycle, sleep_for, target.strftime("%H:%M:%S"))
        time.sleep(sleep_for)

        try:
            # ---- auto mode: tick + handle slot switches ----
            if auto_mode:
                pair_manager.tick()
                switches = pair_manager.check_rescans()
                for sw in switches:
                    # zero-out exited symbols before fetching new weights
                    if sw.exited_symbols:
                        exit_prices = _latest_prices(provider, list(sw.exited_symbols), cfg.timeframe)
                        broker.rebalance_to_weights(
                            pd.Timestamp.utcnow(),
                            {sym: 0.0 for sym in sw.exited_symbols},
                            exit_prices,
                        )
                    alerter.alert_pair_switch(
                        sw.old_pair.y if sw.old_pair else "none",
                        sw.old_pair.x if sw.old_pair else "none",
                        sw.new_pair.y, sw.new_pair.x,
                        sw.new_pair.pvalue, sw.new_pair.half_life,
                    )
                active_symbols = pair_manager.all_symbols
                log.info("slots: %s", pair_manager.slot_summary())

            # ---- fetch OHLCV for all active symbols ----
            end = pd.Timestamp.utcnow()
            start = end - pd.Timedelta(seconds=_TF_SEC[cfg.timeframe] * cfg.history_bars)
            closes = panel_closes(provider, active_symbols, cfg.timeframe, start=start, end=end)

            if closes.empty or len(closes) < 50:
                log.warning("insufficient data this cycle — skipping")
                state.consecutive_errors += 1
                continue

            prices = {s: float(closes[s].iloc[-1]) for s in closes.columns}

            # ---- generate weights ----
            if auto_mode:
                weights = pair_manager.combined_weights(closes)
            else:
                weights = strategy.generate_weights(closes).iloc[-1]

            broker.rebalance_to_weights(closes.index[-1], weights.to_dict(), prices)

            eq = broker.equity(prices)
            state.update_equity(eq)
            state.consecutive_errors = 0
            state.cycle += 1

            dd = state.current_drawdown
            log.info("equity=%.2f  drawdown=%.2f%%", eq, dd * 100)

            if dd <= -abs(cfg.max_drawdown):
                alerter.alert_halted(f"drawdown {dd:.1%} breached limit {-cfg.max_drawdown:.1%}", eq)
                log.critical("max drawdown breached — halting")
                break
            if dd <= -abs(cfg.max_drawdown) * 0.75:
                alerter.alert_drawdown(dd, cfg.max_drawdown, eq)

        except KeyboardInterrupt:
            log.info("keyboard interrupt — shutting down")
            alerter.alert_shutdown("keyboard interrupt")
            break
        except Exception as e:
            state.consecutive_errors += 1
            log.exception("error in cycle %d (%d consecutive)", state.cycle, state.consecutive_errors)
            alerter.alert_error(str(e), state.cycle)

            if state.consecutive_errors >= cfg.max_consecutive_errors:
                alerter.alert_halted(f"{cfg.max_consecutive_errors} consecutive errors", 0.0)
                log.critical("too many consecutive errors — halting")
                break

            backoff = _backoff(state.consecutive_errors - 1)
            log.info("backing off %.0fs", backoff)
            time.sleep(backoff)


def _latest_prices(provider: CCXTProvider, symbols: list[str], timeframe: str) -> dict[str, float]:
    if not symbols:
        return {}
    closes = panel_closes(
        provider, symbols, timeframe,
        start=pd.Timestamp.utcnow() - pd.Timedelta(hours=10),
        end=pd.Timestamp.utcnow(),
    )
    if closes.empty:
        return {}
    return {s: float(closes[s].iloc[-1]) for s in closes.columns}
