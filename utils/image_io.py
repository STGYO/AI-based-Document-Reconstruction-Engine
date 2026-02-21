"""Image loading and saving utilities."""
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as a BGR numpy array.

    Args:
        path: Filesystem path to the image file.

    Returns:
        Image as a BGR numpy array.

    Raises:
        FileNotFoundError: If the image cannot be read.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    logger.debug(f"Loaded image {path} shape={img.shape}")
    return img


def save_image(image: np.ndarray, path: str) -> None:
    """Save a numpy array image to disk.

    Args:
        image: BGR numpy array to save.
        path: Destination file path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, image)
    logger.debug(f"Saved image to {path}")


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB.

    Args:
        image: BGR numpy array.

    Returns:
        RGB numpy array.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert RGB to BGR.

    Args:
        image: RGB numpy array.

    Returns:
        BGR numpy array.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale.

    Args:
        image: BGR or grayscale numpy array.

    Returns:
        Single-channel grayscale numpy array.
    """
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
