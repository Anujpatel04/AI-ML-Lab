"""Logging."""

import logging
import sys

from utils.config import LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(h)
    return logger
