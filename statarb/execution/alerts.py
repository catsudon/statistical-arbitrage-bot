"""Alert dispatcher: Telegram first, falls back to structured log.

Configure via .env:
    TELEGRAM_BOT_TOKEN=...
    TELEGRAM_CHAT_ID=...

If either is missing, all alerts go to the log at WARNING level.
"""
from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass

from ..core.logging import get_logger
from ..core.secrets import _ensure_loaded
from ..core.types import Fill

log = get_logger(__name__)


@dataclass
class Alerter:
    bot_token: str | None = None
    chat_id: str | None = None
    _enabled: bool = False

    def __post_init__(self) -> None:
        _ensure_loaded()
        self.bot_token = self.bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = self.chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        self._enabled = bool(self.bot_token and self.chat_id)
        if self._enabled:
            log.info("Telegram alerts enabled (chat_id=%s)", self.chat_id)
        else:
            log.info("Telegram not configured — alerts go to log only")

    def send(self, text: str, level: str = "INFO") -> None:
        text = textwrap.dedent(text).strip()
        if self._enabled:
            self._post(text)
        else:
            log.warning("[ALERT/%s] %s", level, text)

    def _post(self, text: str) -> None:
        try:
            import requests
            resp = requests.post(
                f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                json={"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"},
                timeout=10,
            )
            if not resp.ok:
                log.warning("Telegram send failed: %s", resp.text[:200])
        except Exception as e:
            log.warning("Telegram error: %s", e)

    # ---- typed helpers ----
    def alert_startup(self, strategy_name: str, mode: str, symbols: list[str]) -> None:
        self.send(
            f"🟢 *statarb started*\n"
            f"strategy: `{strategy_name}`\n"
            f"mode: `{mode}`\n"
            f"symbols: {', '.join(symbols)}"
        )

    def alert_fill(self, fill: Fill, equity: float) -> None:
        emoji = "🔼" if fill.side.value == "long" else "🔽"
        self.send(
            f"{emoji} *Fill* `{fill.symbol}`\n"
            f"side={fill.side.value}  qty={fill.qty:.4f}  price={fill.price:.4f}\n"
            f"fee={fill.fee:.4f}  equity={equity:,.0f}"
        )

    def alert_drawdown(self, current_dd: float, max_dd: float, equity: float) -> None:
        self.send(
            f"⚠️ *Drawdown warning*\n"
            f"current={current_dd:.1%}  limit={max_dd:.1%}\n"
            f"equity={equity:,.0f}",
            level="WARN",
        )

    def alert_halted(self, reason: str, equity: float) -> None:
        self.send(
            f"🛑 *Trading HALTED*\n"
            f"reason: {reason}\n"
            f"equity={equity:,.0f}",
            level="CRITICAL",
        )

    def alert_error(self, error: str, cycle: int) -> None:
        self.send(f"❌ *Error* (cycle {cycle})\n`{error[:300]}`", level="ERROR")

    def alert_shutdown(self, reason: str) -> None:
        self.send(f"🔴 *statarb stopped*\nreason: {reason}")
