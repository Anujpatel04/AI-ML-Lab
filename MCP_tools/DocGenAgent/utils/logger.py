"""
Structured logging for the AI Documentation Generator.
"""

import logging
import sys
from typing import Any

from utils.config import LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with structured format."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logger.level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def log_repo_scanned(logger: logging.Logger, repo_path: str, file_count: int) -> None:
    """Log that a repository was scanned."""
    logger.info("repo_scanned | path=%s | file_count=%s", repo_path, file_count)


def log_files_processed(
    logger: logging.Logger, file_path: str, classes: int, functions: int
) -> None:
    """Log that a file was processed."""
    logger.info(
        "files_processed | file=%s | classes=%s | functions=%s",
        file_path,
        classes,
        functions,
    )


def log_llm_call(logger: logging.Logger, prompt_type: str, extra: dict[str, Any] | None = None) -> None:
    """Log an LLM call."""
    msg = f"llm_call | prompt_type={prompt_type}"
    if extra:
        msg += " | " + " | ".join(f"{k}={v}" for k, v in extra.items())
    logger.info(msg)


def log_docs_generated(logger: logging.Logger, output_path: str) -> None:
    """Log that documentation was generated."""
    logger.info("documentation_generated | output_path=%s", output_path)
