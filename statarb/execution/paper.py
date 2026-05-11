"""In-memory paper broker. Fills at the next price with a slippage model."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ..backtest.costs import CostModel
from ..core.types import Fill, Position, Side


class PaperBroker:
    def __init__(self, initial_capital: float = 100_000.0, cost: CostModel | None = None):
        self.cash = float(initial_capital)
        self.positions: dict[str, Position] = {}
        self.cost = cost or CostModel()
        self.fills: list[Fill] = []

    # ---- accounting ----
    def equity(self, prices: dict[str, float]) -> float:
        eq = self.cash
        for sym, pos in self.positions.items():
            px = prices.get(sym)
            if px is None or np.isnan(px):
                continue
            eq += pos.market_value(px)
        return eq

    def current_weights(self, prices: dict[str, float]) -> pd.Series:
        eq = self.equity(prices)
        if eq <= 0:
            return pd.Series(dtype=float)
        weights = {
            s: (p.market_value(prices[s]) / eq) if s in prices and prices[s] > 0 else 0.0
            for s, p in self.positions.items()
        }
        return pd.Series(weights)

    # ---- execution ----
    def _fill_price(self, mid: float, side: Side) -> float:
        slip = self.cost.slippage_bps / 10_000.0
        return mid * (1 + slip) if side == Side.LONG else mid * (1 - slip)

    def _execute(self, ts: pd.Timestamp, symbol: str, qty: float, price: float) -> None:
        if qty == 0 or price <= 0 or np.isnan(price):
            return
        side = Side.LONG if qty > 0 else Side.SHORT
        fill_px = self._fill_price(price, side)
        notional = abs(qty) * fill_px
        fee = notional * (self.cost.fee_bps / 10_000.0)
        self.cash -= qty * fill_px + fee
        pos = self.positions.get(symbol) or Position(symbol=symbol)
        new_qty = pos.qty + qty
        if pos.qty == 0 or np.sign(new_qty) != np.sign(pos.qty):
            pos.avg_price = fill_px if new_qty != 0 else 0.0
        elif np.sign(new_qty) == np.sign(pos.qty) and abs(new_qty) > abs(pos.qty):
            # increasing position: weighted-avg cost
            pos.avg_price = (pos.avg_price * abs(pos.qty) + fill_px * abs(qty)) / abs(new_qty)
        pos.qty = new_qty
        self.positions[symbol] = pos
        self.fills.append(Fill(ts=ts, symbol=symbol, side=side, qty=abs(qty), price=fill_px, fee=fee))

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
            if px is None or np.isnan(px) or px <= 0:
                continue
            current = self.positions.get(sym, Position(symbol=sym))
            current_w = (current.qty * px) / eq
            delta_w = float(target_w) - current_w
            if abs(delta_w) < 1e-6:
                continue
            delta_qty = (delta_w * eq) / px
            self._execute(ts, sym, delta_qty, px)
