"""Credential loading from environment / .env file.

We never log or print the resolved key; only its presence is reported.

Convention: per-exchange keys are namespaced, e.g. `BINANCE_API_KEY` /
`BINANCE_API_SECRET`. Optional: `BINANCE_API_PASSPHRASE` (Coinbase et al.)
and `BINANCE_TESTNET=1` to flip the exchange into sandbox mode.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .logging import get_logger

log = get_logger(__name__)

_LOADED = False


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    # walk up from CWD looking for .env; falls back to project root
    for d in [Path.cwd(), *Path.cwd().parents]:
        candidate = d / ".env"
        if candidate.exists():
            load_dotenv(candidate, override=False)
            log.info("loaded .env from %s", candidate)
            break
    _LOADED = True


@dataclass
class ExchangeCredentials:
    api_key: str | None = None
    api_secret: str | None = None
    passphrase: str | None = None
    testnet: bool = False

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key and self.api_secret)


def get_credentials(exchange: str) -> ExchangeCredentials:
    """Read credentials for `exchange` from the environment.

    Looks for `<EXCHANGE>_API_KEY`, `<EXCHANGE>_API_SECRET`, and the
    optional `<EXCHANGE>_API_PASSPHRASE` and `<EXCHANGE>_TESTNET`.
    Returns an empty `ExchangeCredentials` if none are set (safe for
    read-only data flows).
    """
    _ensure_loaded()
    prefix = exchange.upper()
    creds = ExchangeCredentials(
        api_key=os.environ.get(f"{prefix}_API_KEY"),
        api_secret=os.environ.get(f"{prefix}_API_SECRET"),
        passphrase=os.environ.get(f"{prefix}_API_PASSPHRASE"),
        testnet=os.environ.get(f"{prefix}_TESTNET", "").lower() in {"1", "true", "yes"},
    )
    if creds.has_credentials:
        log.info("loaded credentials for %s (testnet=%s)", exchange, creds.testnet)
    return creds
