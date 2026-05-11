"""CCXT live broker with limit orders, persistence, and exchange reconciliation.

dry_run=True (default) → log orders, never touch exchange.
dry_run=False → real orders. Requires valid credentials and calls
`reconcile_on_startup()` before the first rebalance.
"""
from __future__ import annotations

import math
import time
from typing import Any

import ccxt
import numpy as np
import pandas as pd

from ..core.logging import get_logger
from ..core.secrets import ExchangeCredentials, get_credentials
from ..core.types import Fill, Position, Side
from .alerts import Alerter
from .persistence import FillStore

log = get_logger(__name__)

_DUST_USD = 10.0          # orders smaller than this are skipped
_LIMIT_OFFSET_BPS = 3     # how aggressively inside the spread we post limits
_LIMIT_TIMEOUT_S = 45     # seconds before we cancel limit and go market
_POLL_INTERVAL_S = 2      # order status poll interval


class CCXTBroker:
    def __init__(
        self,
        exchange: str,
        api_key: str | None = None,
        secret: str | None = None,
        dry_run: bool = True,
        params: dict[str, Any] | None = None,
        credentials: ExchangeCredentials | None = None,
        use_env_credentials: bool = True,
        db_path: str = ".cache/statarb/fills.db",
        alerter: Alerter | None = None,
    ):
        cls = getattr(ccxt, exchange)
        creds = credentials if credentials is not None else (
            get_credentials(exchange) if use_env_credentials else ExchangeCredentials()
        )
        key = api_key or creds.api_key
        sec = secret or creds.api_secret
        cfg: dict[str, Any] = {"enableRateLimit": True, **(params or {})}
        if key:
            cfg["apiKey"] = key
        if sec:
            cfg["secret"] = sec
        if creds.passphrase:
            cfg["password"] = creds.passphrase
        self.ex = cls(cfg)
        if creds.testnet and hasattr(self.ex, "set_sandbox_mode"):
            self.ex.set_sandbox_mode(True)
        if not dry_run and not (key and sec):
            raise RuntimeError(
                f"live trading on {exchange} requires credentials — "
                f"set {exchange.upper()}_API_KEY / _API_SECRET in .env"
            )
        self.exchange_id = exchange
        self.dry_run = dry_run
        self.credentials = creds
        self.store = FillStore(db_path)
        self.alerter = alerter or Alerter()
        # load persisted positions
        self.positions: dict[str, Position] = self.store.load_positions()
        self.fills: list[Fill] = []
        log.info("loaded %d persisted positions from %s", len(self.positions), db_path)

    # ------------------------------------------------------------------ #
    # Accounting                                                           #
    # ------------------------------------------------------------------ #
    def equity(self, prices: dict[str, float]) -> float:
        if self.dry_run:
            usdt = 100_000.0
        else:
            bal = self.ex.fetch_balance()
            usdt = float(bal.get("total", {}).get("USDT", 0.0))
        for sym, pos in self.positions.items():
            px = prices.get(sym)
            if px:
                usdt += pos.market_value(px)
        return usdt

    def current_weights(self, prices: dict[str, float]) -> pd.Series:
        eq = self.equity(prices)
        if eq <= 0:
            return pd.Series(dtype=float)
        return pd.Series(
            {s: p.market_value(prices[s]) / eq for s, p in self.positions.items() if prices.get(s)},
        )

    # ------------------------------------------------------------------ #
    # Startup reconciliation                                               #
    # ------------------------------------------------------------------ #
    def reconcile_on_startup(self, symbols: list[str]) -> None:
        """Compare persisted positions against actual exchange balances and
        correct any discrepancies. Call once before the first rebalance."""
        if self.dry_run:
            log.info("dry_run — skipping exchange reconciliation")
            return
        try:
            self.ex.load_markets()
            bal = self.ex.fetch_balance()
            for sym in symbols:
                base = sym.split("/")[0]
                exchange_qty = float(bal.get("total", {}).get(base, 0.0))
                persisted = self.positions.get(sym)
                persisted_qty = persisted.qty if persisted else 0.0
                if abs(exchange_qty - abs(persisted_qty)) > max(0.001, 0.02 * max(exchange_qty, abs(persisted_qty))):
                    log.warning(
                        "position mismatch %s: persisted=%.6f exchange=%.6f → using exchange",
                        sym, persisted_qty, exchange_qty,
                    )
                    if exchange_qty > 1e-9:
                        self.positions[sym] = Position(symbol=sym, qty=exchange_qty, avg_price=0.0)
                    else:
                        self.positions.pop(sym, None)
            log.info("reconciliation complete — %d positions", len(self.positions))
        except Exception as e:
            log.error("reconciliation failed: %s — continuing with persisted state", e)

    # ------------------------------------------------------------------ #
    # Order execution                                                      #
    # ------------------------------------------------------------------ #
    def _round_amount(self, symbol: str, qty: float) -> float:
        if not self.ex.markets:
            try:
                self.ex.load_markets()
            except Exception:
                return qty
        m = self.ex.markets.get(symbol)
        if not m:
            return qty
        prec = m.get("precision", {}).get("amount")
        if prec is None:
            return qty
        step = 10 ** (-int(prec))
        return math.floor(qty / step) * step

    def _limit_with_fallback(
        self,
        symbol: str,
        qty: float,
        mid_price: float,
        ts: pd.Timestamp,
    ) -> Fill | None:
        """Submit an aggressive limit order; fall back to market if not filled
        within `_LIMIT_TIMEOUT_S` seconds."""
        side = "buy" if qty > 0 else "sell"
        amount = self._round_amount(symbol, abs(qty))
        if amount <= 0:
            return None

        offset = _LIMIT_OFFSET_BPS / 10_000.0
        limit_px = mid_price * (1 + offset) if side == "buy" else mid_price * (1 - offset)
        limit_px = round(limit_px, 8)

        if self.dry_run:
            log.info("[DRY] limit %s %.6f %s @ %.6f", side, amount, symbol, limit_px)
            return None

        order_id: str | None = None
        try:
            order = self.ex.create_order(symbol, "limit", side, amount, limit_px)
            order_id = order["id"]
            log.info("limit order %s: %s %.6f %s @ %.6f", order_id, side, amount, symbol, limit_px)

            deadline = time.monotonic() + _LIMIT_TIMEOUT_S
            while time.monotonic() < deadline:
                time.sleep(_POLL_INTERVAL_S)
                o = self.ex.fetch_order(order_id, symbol)
                if o["status"] == "closed":
                    log.info("limit filled: %s", order_id)
                    return self._order_to_fill(o, symbol, qty, ts)
                if o["status"] in ("canceled", "rejected", "expired"):
                    log.warning("limit %s %s — going market", order_id, o["status"])
                    break

            # timeout → cancel and market
            try:
                self.ex.cancel_order(order_id, symbol)
            except Exception:
                pass
            log.info("limit timed out, falling back to market for %s", symbol)

        except ccxt.BaseError as e:
            log.warning("limit order failed (%s), trying market: %s", symbol, e)

        # market fallback
        order = self.ex.create_order(symbol, "market", side, amount)
        return self._order_to_fill(order, symbol, qty, ts)

    @staticmethod
    def _order_to_fill(order: dict, symbol: str, signed_qty: float, ts: pd.Timestamp) -> Fill:
        fee_info = order.get("fee") or {}
        return Fill(
            ts=ts,
            symbol=symbol,
            side=Side.LONG if signed_qty > 0 else Side.SHORT,
            qty=float(order.get("filled") or order.get("amount") or abs(signed_qty)),
            price=float(order.get("average") or order.get("price") or 0.0),
            fee=float(fee_info.get("cost", 0.0)),
            order_id=order.get("id"),
        )

    # ------------------------------------------------------------------ #
    # Rebalance                                                            #
    # ------------------------------------------------------------------ #
    def rebalance_to_weights(
        self,
        ts: pd.Timestamp,
        target_weights: dict[str, float],
        prices: dict[str, float],
    ) -> None:
        eq = self.equity(prices)
        if eq <= 0:
            return
        for sym, target_w in target_weights.items():
            px = prices.get(sym)
            if not px or np.isnan(px) or px <= 0:
                continue
            cur = self.positions.get(sym, Position(symbol=sym))
            current_w = cur.qty * px / eq
            delta_qty = (float(target_w) - current_w) * eq / px
            if abs(delta_qty * px) < _DUST_USD:
                continue

            fill = self._limit_with_fallback(sym, delta_qty, px, ts)
            if fill is None:
                continue

            # update in-memory position
            signed = fill.qty if fill.side == Side.LONG else -fill.qty
            new_qty = cur.qty + signed
            if abs(new_qty) < 1e-9:
                self.positions.pop(sym, None)
            else:
                cur.qty = new_qty
                cur.avg_price = fill.price
                self.positions[sym] = cur

            # persist and alert
            self.store.record_fill(fill)
            self.fills.append(fill)
            self.alerter.alert_fill(fill, self.equity(prices))
            log.info("fill recorded: %s %s %.6f @ %.6f fee=%.6f",
                     fill.side.value, fill.symbol, fill.qty, fill.price, fill.fee)
