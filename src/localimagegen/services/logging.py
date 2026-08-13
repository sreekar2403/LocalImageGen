"""Structured logging for LocalImageGen using ``structlog``.

Provides a single :func:`configure_logging` entry point that sets up a
``structlog``-based logger with JSON output (or a readable console format when
running interactively) and a :func:`get_logger` helper to obtain a configured
logger bound to a module name.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

#: Default log level used when none is supplied.
DEFAULT_LOG_LEVEL = "INFO"


def configure_logging(
    level: str = DEFAULT_LOG_LEVEL,
    *,
    json: bool | None = None,
) -> None:
    """Configure the global structured logging pipeline.

    Args:
        level: Logging level (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``,
            ``CRITICAL``).
        json: When ``True`` emit JSON lines; when ``False`` emit a readable
            console format. Defaults to JSON when stdout is not a TTY.
    """
    if json is None:
        json = not sys.stdout.isatty()

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    renderer: Any
    if json:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None, **initial_values: Any) -> structlog.stdlib.BoundLogger:
    """Return a configured structured logger.

    Args:
        name: Optional logger name (typically the module ``__name__``).
        **initial_values: Key/value pairs bound to every log call.

    Returns:
        A bound ``structlog`` logger.
    """
    return structlog.get_logger(name, **initial_values)
