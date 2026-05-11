"""Common dataclasses used across the package.

All types are immutable where it doesn't hurt performance. Timestamps are
always pandas-compatible UTC `pd.Timestamp` values; prices are `float`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass(frozen=True, slots=True)
class Bar:
    ts: pd.Timestamp
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class Signal:
    """Output of a strategy. `weight` is a target portfolio weight in [-1, 1]
    per leg; backtest/execution layer translates into orders."""

    ts: pd.Timestamp
    symbol: str
    weight: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Order:
    ts: pd.Timestamp
    symbol: str
    side: Side
    qty: float
    type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    id: str | None = None


@dataclass(slots=True)
class Fill:
    ts: pd.Timestamp
    symbol: str
    side: Side
    qty: float
    price: float
    fee: float
    order_id: str | None = None


@dataclass(slots=True)
class Position:
    symbol: str
    qty: float = 0.0
    avg_price: float = 0.0

    @property
    def side(self) -> Side:
        if self.qty > 0:
            return Side.LONG
        if self.qty < 0:
            return Side.SHORT
        return Side.FLAT

    def market_value(self, price: float) -> float:
        return self.qty * price

    def unrealized_pnl(self, price: float) -> float:
        return (price - self.avg_price) * self.qty


@dataclass(slots=True)
class PairSpec:
    """A cointegrated pair: y ~ beta * x + alpha. Trade `1` unit of y vs
    `beta` units of x to express the spread."""

    y: str
    x: str
    beta: float
    alpha: float
    half_life: float
    pvalue: float
    score: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BasketSpec:
    """A Johansen cointegrated basket. `weights` align with `symbols`; the
    spread is `prices @ weights`."""

    symbols: list[str]
    weights: list[float]
    half_life: float
    eig_stat: float
    meta: dict[str, Any] = field(default_factory=dict)
