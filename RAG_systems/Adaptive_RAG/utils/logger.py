"""
Structured logging utility for Adaptive RAG.
"""

import logging
import os
import sys
from typing import Any

LOG_LEVEL = getattr(logging, (os.environ.get("LOG_LEVEL") or "INFO").upper(), logging.INFO)


def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """Return a configured logger with optional level override."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level if level is not None else LOG_LEVEL)
    return logger


def log_structured(
    logger: logging.Logger,
    level: int,
    message: str,
    **kwargs: Any,
) -> None:
    """Log a message with optional structured key-value fields."""
    extra = " ".join(f"{k}={v!r}" for k, v in kwargs.items()) if kwargs else ""
    full = f"{message} {extra}".strip()
    logger.log(level, full)
