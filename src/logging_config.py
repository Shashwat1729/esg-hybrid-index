"""Centralised logging configuration for the multi-factor ESG pipeline.

Usage in any module or script::

    import logging
    from src.logging_config import setup_logging

    setup_logging()            # call once at startup
    logger = logging.getLogger(__name__)
    logger.info("Pipeline started")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    *,
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """Configure the root logger with console and (optionally) file handlers.

    Parameters
    ----------
    level : int
        Minimum log level (default ``logging.INFO``).
    log_file : str | Path | None
        If provided, also write log messages to this file.
    fmt : str
        Log message format string.
    datefmt : str
        Date format for timestamps.
    """
    root = logging.getLogger()

    # Avoid adding duplicate handlers on repeated calls
    if root.handlers:
        return

    root.setLevel(level)

    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # Console handler (stderr)
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler (optional)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
