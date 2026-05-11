"""Live runner — production-ready loop.

Features:
- Kill switch: create `.killswitch` in CWD to stop gracefully
- Drawdown monitoring: halt + alert when max_dd breached
- Exponential backoff on errors (max 5 min)
- Startup checklist: credentials, reconciliation, alert on launch
- Per-cycle equity + drawdown logged every bar
- Auto pair re-scan: optional PairManager swaps the active pair every N days
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from ..core.logging import get_logger
from ..core.types import PairSpec
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
    symbols: list[str]           # universe for price fetching (pair symbols when fixed)
    timeframe: str = "1h"
    history_bars: int = 1500
    poll_seconds: int = 30
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
    pair_manager=None,                         # PairManager | None
    strategy_factory: Callable[[PairSpec], Strategy] | None = None,
) -> None:
    """
    Parameters
    ----------
    strategy        : pre-built strategy for fixed-pair mode; pass None for auto mode
    pair_manager    : PairManager instance for auto re-scan mode
    strategy_factory: callable (PairSpec) -> Strategy; required when pair_manager is set
    """
    alerter = alerter or Alerter()
    state = RunnerState()
    mode = "DRY-RUN" if getattr(broker, "dry_run", True) else "LIVE"
    auto_mode = pair_manager is not None

    log.info("=== statarb %s starting (mode=%s) ===", mode, "AUTO-SCAN" if auto_mode else "FIXED-PAIR")

    # ---- auto mode: initial scan ----
    if auto_mode:
        log.info("performing initial pair scan …")
        best = pair_manager.scan_now()
        if best is None:
            log.error("initial scan found no cointegrated pairs — aborting")
            alerter.alert_no_pair()
            return
        strategy = strategy_factory(best)
        pair_manager.current_pair = best
        active_symbols = [best.y, best.x]
        log.info("initial pair: %s / %s", best.y, best.x)
    else:
        active_symbols = cfg.symbols

    if hasattr(broker, "reconcile_on_startup"):
        broker.reconcile_on_startup(active_symbols)

    prices_init = _latest_prices(provider, active_symbols, cfg.timeframe)
    eq0 = broker.equity(prices_init)
    state.peak_equity = eq0
    alerter.alert_startup(strategy.name, mode, active_symbols)
    log.info("starting equity: %.2f USDT", eq0)

    # ---- main loop ----
    while True:
        if KILLSWITCH.exists():
            log.info("kill switch active — shutting down")
            alerter.alert_shutdown("kill switch")
            break

        target = _next_bar_close(cfg.timeframe)
        sleep_for = max(0.0, target.timestamp() - pd.Timestamp.utcnow().timestamp() + 2.0)
        log.info(
            "cycle %d — sleeping %.0fs until %s UTC",
            state.cycle, sleep_for, target.strftime("%H:%M:%S"),
        )
        time.sleep(sleep_for)

        try:
            # ---- auto mode: check for pair switch ----
            if auto_mode:
                pair_manager.tick()
                new_pair, switched = pair_manager.maybe_rescan()

                if new_pair is None:
                    log.warning("rescan found no pairs — keeping current pair or idling")
                    alerter.alert_no_pair()
                    state.consecutive_errors += 1
                    continue

                if switched:
                    old_pair = pair_manager.current_pair
                    old_y = old_pair.y if old_pair else "none"
                    old_x = old_pair.x if old_pair else "none"
                    log.info("switching pair: %s/%s → %s/%s", old_y, old_x, new_pair.y, new_pair.x)

                    # close existing positions before switching
                    old_prices = _latest_prices(provider, [old_y, old_x], cfg.timeframe)
                    try:
                        broker.rebalance_to_weights(
                            pd.Timestamp.utcnow(),
                            {old_y: 0.0, old_x: 0.0},
                            old_prices,
                        )
                    except Exception as e:
                        log.warning("could not close old positions: %s", e)

                    alerter.alert_pair_switch(
                        old_y, old_x, new_pair.y, new_pair.x,
                        new_pair.pvalue, new_pair.half_life,
                    )
                    strategy = strategy_factory(new_pair)
                    active_symbols = [new_pair.y, new_pair.x]
                    pair_manager.current_pair = new_pair

            # ---- fetch OHLCV ----
            end = pd.Timestamp.utcnow()
            start = end - pd.Timedelta(seconds=_TF_SEC[cfg.timeframe] * cfg.history_bars)
            closes = panel_closes(provider, active_symbols, cfg.timeframe, start=start, end=end)

            if closes.empty or len(closes) < 50:
                log.warning("insufficient data this cycle — skipping")
                state.consecutive_errors += 1
                continue

            prices = {s: float(closes[s].iloc[-1]) for s in closes.columns}
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
    closes = panel_closes(
        provider, symbols, timeframe,
        start=pd.Timestamp.utcnow() - pd.Timedelta(hours=10),
        end=pd.Timestamp.utcnow(),
    )
    if closes.empty:
        return {}
    return {s: float(closes[s].iloc[-1]) for s in closes.columns}
