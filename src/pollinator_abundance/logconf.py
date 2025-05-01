"""Logging configurations."""

import logging
import time

# Normal formatter
LOGGER_FORMATTER = logging.Formatter(
    "%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
LOGGER_FORMATTER.converter = time.gmtime


def create_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # If no handlers exist, add one; otherwise update existing
    if not logger.hasHandlers():
        logger_streamhandler = logging.StreamHandler()
        logger_streamhandler.setLevel(level)
        logger_streamhandler.setFormatter(LOGGER_FORMATTER)
        logger.addHandler(logger_streamhandler)
    else:
        for handler in logger.handlers:
            handler.setLevel(level)
            handler.setFormatter(LOGGER_FORMATTER)

    return logger


def set_handler_formatter(handler: logging.Handler) -> logging.Handler:
    """Set the formatter for a given logger handler."""
    handler.setFormatter(LOGGER_FORMATTER)
    return handler
