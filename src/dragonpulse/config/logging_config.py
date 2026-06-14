"""Centralized logging configuration for DragonPulse.

Security note
-------------
DragonPulse intentionally never logs full sensitive proposal text or full API
keys. Use :func:`get_logger` everywhere and pass already-redacted values. The
:func:`redact` helper is provided for convenience.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

_CONFIGURED = False
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once for the whole application.

    Parameters
    ----------
    level:
        Logging level name (e.g. ``"INFO"``, ``"DEBUG"``).
    """
    global _CONFIGURED
    if _CONFIGURED:
        logging.getLogger().setLevel(level)
        return

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet down noisy third-party loggers.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring logging on first use."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)


def redact(value: Optional[str], keep: int = 4) -> str:
    """Return a redacted version of ``value`` safe for logs.

    Examples
    --------
    >>> redact("supersecretkey")
    'supe…(redacted)'
    """
    if not value:
        return "<none>"
    if len(value) <= keep:
        return "*" * len(value)
    return f"{value[:keep]}…(redacted)"
