"""Logging setup. Use `get_logger(__name__)` in every module."""

import logging

from etic_genai.config import get_settings

_configured = False


def setup_logging(level: str | None = None) -> None:
    global _configured
    if _configured:
        return
    logging.basicConfig(
        level=(level or get_settings().log_level).upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
