"""Memory profiling utilities."""
import logging
import os

import psutil

logger = logging.getLogger(__name__)


def log_memory_usage(tag: str = "") -> None:
    """Log current process memory usage in MB.

    Args:
        tag: Optional label appended to the log message.
    """
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    logger.debug(f"[Memory{' ' + tag if tag else ''}] RSS: {mem_mb:.1f} MB")


def get_memory_usage_mb() -> float:
    """Return current RSS memory usage in MB.

    Returns:
        RSS memory in megabytes as a float.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024
