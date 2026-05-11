"""Rich-backed structured logging."""
from __future__ import annotations

import logging
import os

from rich.logging import RichHandler

_CONFIGURED = False


def get_logger(name: str = "statarb") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        level = os.environ.get("STATARB_LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="%H:%M:%S",
            handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        )
        _CONFIGURED = True
    return logging.getLogger(name)
