"""
Logging configuration.
======================

A single ``setup_logging`` call configures a console handler plus a rotating
file handler under ``outputs/logs/``. Modules obtain loggers with
``get_logger(__name__)`` and never configure handlers themselves.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-38s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False


def setup_logging(
    logs_dir: Optional[Path] = None,
    level: int = logging.INFO,
    run_name: Optional[str] = None,
    force: bool = False,
) -> logging.Logger:
    """Configure the ``neurogenesis`` logger hierarchy.

    Idempotent: repeated calls are no-ops unless ``force=True``. This matters
    because Streamlit re-executes the whole script on every interaction, and
    without the guard the dashboard would accumulate duplicate handlers and
    print every line N times.

    Args:
        logs_dir: Directory for the log file. When ``None``, only console
            logging is configured.
        level: Root level for the ``neurogenesis`` logger.
        run_name: Optional label included in the log filename.
        force: Reconfigure even if already configured.

    Returns:
        The ``neurogenesis`` logger.
    """
    global _CONFIGURED

    root = logging.getLogger("neurogenesis")

    if _CONFIGURED and not force:
        return root

    for handler in list(root.handlers):
        root.removeHandler(handler)

    root.setLevel(level)
    # Do not propagate to the Python root logger: Streamlit and pytest install
    # their own root handlers, which would duplicate every record.
    root.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(level)
    root.addHandler(console)

    if logs_dir is not None:
        logs_dir = Path(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{run_name}" if run_name else ""
        log_path = logs_dir / f"neurogenesis_{stamp}{suffix}.log"
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        root.addHandler(file_handler)
        root.info("Log file: %s", log_path)

    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a module logger inside the ``neurogenesis`` hierarchy.

    Args:
        name: Usually ``__name__``. A leading ``modules.`` is stripped so that
            log lines read ``neurogenesis.m06_neuropropx.srve`` rather than
            ``neurogenesis.modules.m06_neuropropx.srve``.
    """
    clean = name
    for prefix in ("modules.", "neurogenesis."):
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
    return logging.getLogger(f"neurogenesis.{clean}")


def log_banner(logger: logging.Logger, title: str, width: int = 78) -> None:
    """Emit a visually separated section banner."""
    logger.info("=" * width)
    logger.info("  %s", title)
    logger.info("=" * width)


__all__ = ["setup_logging", "get_logger", "log_banner"]
