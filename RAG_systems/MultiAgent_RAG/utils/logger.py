"""
Structured logging for Multi-Agent RAG.
"""

import logging
import os
import sys

LOG_LEVEL = getattr(
    logging,
    (os.environ.get("LOG_LEVEL") or "INFO").upper(),
    logging.INFO,
)


def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """Return a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level if level is not None else LOG_LEVEL)
    return logger
