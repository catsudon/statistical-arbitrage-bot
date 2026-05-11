"""Real-time terminal dashboard for the live runner.

Uses rich.live to refresh a full-screen layout every bar.
Log messages are captured and shown in the bottom panel instead of stdout.
"""
from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

from rich.layout import Layout
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_TF_BPD: dict[str, int] = {
    "1m": 1440, "5m": 288, "15m": 96, "30m": 48,
    "1h": 24, "4h": 6, "1d": 1,
}


# ---------- state (updated by runner each cycle) ----------

@dataclass
class SlotInfo:
    idx: int
    pair_y: str = "—"
    pair_x: str = "—"
    position: str = "FLAT"      # LONG / SHORT / FLAT
    weight_y: float = 0.0
    zscore: float = float("nan")
    bars_to_rescan: int = 0


@dataclass
class DashboardState:
    exchange: str = "—"
    timeframe: str = "1h"
    mode: str = "—"
    cycle: int = 0
    equity: float = 0.0
    peak_equity: float = 0.0
    drawdown: float = 0.0
    slots: list[SlotInfo] = field(default_factory=list)
    recent_fills: list[str] = field(default_factory=list)
    log_lines: list[str] = field(default_factory=list)
    last_update: str = "—"
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_log(self, line: str) -> None:
        with self._lock:
            self.log_lines.append(line)
            self.log_lines = self.log_lines[-8:]

    def add_fill(self, line: str) -> None:
        with self._lock:
            self.recent_fills.append(line)
            self.recent_fills = self.recent_fills[-6:]

    def stamp(self) -> None:
        self.last_update = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")


# ---------- log handler that feeds into state ----------

class _DashHandler(logging.Handler):
    def __init__(self, state: DashboardState) -> None:
        super().__init__()
        self.state = state

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            plain = msg.split("\x1b")[0][:110]
            self.state.add_log(plain)
        except Exception:
            pass


# ---------- dynamic renderable so auto-refresh re-reads state each tick ----------

class _LiveRenderable:
    """Calls render_fn() fresh on every Rich refresh cycle."""
    def __init__(self, render_fn):
        self._render_fn = render_fn

    def __rich_console__(self, console, options):
        yield self._render_fn()


# ---------- renderer ----------

class Dashboard:
    """Context-manager wrapper around rich.Live."""

    def __init__(self, state: DashboardState) -> None:
        self.state = state
        self._live = Live(
            _LiveRenderable(self._render),
            refresh_per_second=2,
            screen=True,
        )
        self._handler = _DashHandler(state)
        self._handler.setFormatter(logging.Formatter("%(levelname)s  %(name)s  %(message)s"))
        self._removed_rich_handlers: list[logging.Handler] = []

    def __enter__(self) -> "Dashboard":
        # silence RichHandler while Live owns the screen — logs go to _DashHandler only
        root = logging.getLogger()
        self._removed_rich_handlers = [h for h in root.handlers if isinstance(h, RichHandler)]
        for h in self._removed_rich_handlers:
            root.removeHandler(h)
        root.addHandler(self._handler)
        self._live.__enter__()
        return self

    def __exit__(self, *args) -> None:
        root = logging.getLogger()
        root.removeHandler(self._handler)
        for h in self._removed_rich_handlers:
            root.addHandler(h)
        self._live.__exit__(*args)

    def refresh(self) -> None:
        pass  # auto-refresh handles it via _LiveRenderable

    # ---- layout ----

    def _render(self) -> Layout:
        root = Layout()
        root.split_column(
            Layout(self._header(),  name="header",  size=3),
            Layout(self._equity(),  name="equity",  size=3),
            Layout(self._slots(),   name="slots",   size=len(self.state.slots) + 4),
            Layout(self._fills(),   name="fills",   size=8),
            Layout(self._log(),     name="log",     size=10),
        )
        return root

    def _header(self) -> Panel:
        s = self.state
        t = Text()
        t.append("statarb ", style="bold white")
        t.append(s.mode, style="bold yellow" if s.mode == "DRY-RUN" else "bold green")
        t.append(f"  │  {s.exchange}  {s.timeframe}  │  cycle {s.cycle}  │  {s.last_update}")
        return Panel(t, style="bold blue")

    def _equity(self) -> Panel:
        s = self.state
        pnl = (s.equity / s.peak_equity - 1.0) if s.peak_equity > 0 else 0.0
        dd_color = "green" if s.drawdown > -0.05 else ("yellow" if s.drawdown > -0.10 else "red bold")
        pnl_color = "green" if pnl >= 0 else "red"
        t = Text()
        t.append(f"Equity  {s.equity:>12,.2f} USDT", style="bold")
        t.append(f"   Peak  {s.peak_equity:,.2f}")
        t.append(f"   P&L  ")
        t.append(f"{pnl:+.2%}", style=pnl_color)
        t.append(f"   Drawdown  ")
        t.append(f"{s.drawdown:.2%}", style=dd_color)
        return Panel(t, title="Portfolio")

    def _slots(self) -> Panel:
        tbl = Table(expand=True, show_header=True, header_style="bold cyan", box=None)
        tbl.add_column("#",           width=4)
        tbl.add_column("Y leg",       min_width=12)
        tbl.add_column("X leg",       min_width=12)
        tbl.add_column("Position",    width=8)
        tbl.add_column("Wt Y",        width=8,  justify="right")
        tbl.add_column("Z-score",     width=9,  justify="right")
        tbl.add_column("Next rescan", width=20)

        bpd = _TF_BPD.get(self.state.timeframe, 24)
        for sl in self.state.slots:
            pos_style = (
                "bold green" if sl.position == "LONG"
                else "bold red" if sl.position == "SHORT"
                else "dim"
            )
            if math.isnan(sl.zscore):
                z_text = Text("—", style="dim")
            else:
                az = abs(sl.zscore)
                z_style = (
                    "bold red"    if az >= 2.0
                    else "yellow" if az >= 1.0
                    else "green"
                )
                z_text = Text(f"{sl.zscore:+.2f}", style=z_style)
            days = sl.bars_to_rescan / bpd
            tbl.add_row(
                str(sl.idx),
                sl.pair_y,
                sl.pair_x,
                Text(sl.position, style=pos_style),
                f"{sl.weight_y:+.3f}",
                z_text,
                f"{sl.bars_to_rescan} bars  ({days:.1f}d)",
            )
        return Panel(tbl, title="Pair Slots")

    def _fills(self) -> Panel:
        lines = self.state.recent_fills or ["no fills yet"]
        return Panel("\n".join(lines), title="Recent Fills")

    def _log(self) -> Panel:
        lines = self.state.log_lines or ["waiting for first cycle…"]
        return Panel("\n".join(lines), title="Log")
