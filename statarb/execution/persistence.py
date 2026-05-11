"""SQLite-backed fill log and position persistence.

Positions are derived by replaying fills (net qty per symbol), so the
database is the single source of truth — no separate position table to
get out of sync.

Thread-safe: one connection per thread via `threading.local`.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pandas as pd

from ..core.logging import get_logger
from ..core.types import Fill, Position, Side

log = get_logger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS fills (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT    NOT NULL,
    symbol    TEXT    NOT NULL,
    side      TEXT    NOT NULL,
    qty       REAL    NOT NULL,
    price     REAL    NOT NULL,
    fee       REAL    NOT NULL,
    order_id  TEXT
);
CREATE INDEX IF NOT EXISTS idx_fills_symbol ON fills(symbol);
CREATE INDEX IF NOT EXISTS idx_fills_ts     ON fills(ts);
"""


class FillStore:
    def __init__(self, db_path: str | Path = ".cache/statarb/fills.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    # ---- connection ----
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(str(self.db_path))
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        self._conn().executescript(_DDL)
        self._conn().commit()

    # ---- write ----
    def record_fill(self, fill: Fill) -> None:
        self._conn().execute(
            "INSERT INTO fills(ts,symbol,side,qty,price,fee,order_id) VALUES(?,?,?,?,?,?,?)",
            (fill.ts.isoformat(), fill.symbol, fill.side.value,
             fill.qty, fill.price, fill.fee, fill.order_id),
        )
        self._conn().commit()

    # ---- read ----
    def load_fills(self, symbol: str | None = None) -> list[Fill]:
        if symbol:
            rows = self._conn().execute(
                "SELECT * FROM fills WHERE symbol=? ORDER BY ts", (symbol,)
            ).fetchall()
        else:
            rows = self._conn().execute("SELECT * FROM fills ORDER BY ts").fetchall()
        out = []
        for r in rows:
            out.append(Fill(
                ts=pd.Timestamp(r["ts"]),
                symbol=r["symbol"],
                side=Side(r["side"]),
                qty=r["qty"],
                price=r["price"],
                fee=r["fee"],
                order_id=r["order_id"],
            ))
        return out

    def load_positions(self) -> dict[str, Position]:
        """Replay all fills to reconstruct net positions."""
        rows = self._conn().execute(
            "SELECT symbol, side, SUM(qty) as net_qty, "
            "SUM(qty*price)/NULLIF(SUM(qty),0) as avg_px "
            "FROM fills GROUP BY symbol, side"
        ).fetchall()

        net: dict[str, float] = {}
        cost: dict[str, float] = {}
        qty_abs: dict[str, float] = {}

        for r in rows:
            sym = r["symbol"]
            signed = r["net_qty"] if r["side"] == Side.LONG.value else -r["net_qty"]
            net[sym] = net.get(sym, 0.0) + signed
            # weighted avg price (only for the dominant direction)
            if r["side"] == Side.LONG.value:
                cost[sym] = float(r["avg_px"] or 0)
                qty_abs[sym] = float(r["net_qty"])

        positions = {}
        for sym, qty in net.items():
            if abs(qty) < 1e-9:
                continue
            positions[sym] = Position(symbol=sym, qty=qty, avg_price=cost.get(sym, 0.0))
        return positions

    def total_fees_paid(self) -> float:
        row = self._conn().execute("SELECT SUM(fee) FROM fills").fetchone()
        return float(row[0] or 0.0)

    def fills_as_dataframe(self) -> pd.DataFrame:
        rows = self._conn().execute("SELECT * FROM fills ORDER BY ts").fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])
