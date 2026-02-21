"""Logger utility for the Document Reconstruction Engine."""
import logging
from typing import Optional


class EngineLogger:
    """Wraps Python logging to provide a pre-configured logger factory."""

    _FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    _DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

    @staticmethod
    def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
        """Return a configured logger with StreamHandler and standard formatter.

        Args:
            name: The logger name (typically ``__name__``).
            level: The logging level. Defaults to ``logging.DEBUG``.

        Returns:
            A :class:`logging.Logger` instance with a stream handler attached.
        """
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                fmt=EngineLogger._FORMAT,
                datefmt=EngineLogger._DATE_FORMAT,
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(level)
        return logger
