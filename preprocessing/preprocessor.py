"""Image preprocessing module for document reconstruction."""
import logging
from typing import Optional

import cv2
import numpy as np

from utils.memory import log_memory_usage

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG: dict = {
    "skew_threshold": 0.5,
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid": (8, 8),
    "morph_kernel_size": 3,
    "shadow_dilate_kernel": 15,
}


class ImagePreprocessor:
    """Applies a configurable preprocessing pipeline to document images.

    Each step can be individually invoked or the full pipeline can be
    executed via :meth:`process`.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        """Initialise with optional configuration overrides.

        Args:
            config: Dictionary of config keys that override defaults.
        """
        self._config: dict = {**_DEFAULT_CONFIG, **(config or {})}
        logger.debug(f"ImagePreprocessor created with config={self._config}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def correct_skew(self, image: np.ndarray) -> np.ndarray:
        """Detect and correct image skew using Hough lines.

        Args:
            image: BGR or grayscale numpy array.

        Returns:
            De-skewed image as a numpy array.
        """
        try:
            gray = self._to_gray(image)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
            if lines is None:
                return image
            angles = []
            for line in lines:
                rho, theta = line[0]
                # Keep only near-horizontal lines
                if abs(theta - np.pi / 2) < np.deg2rad(45):
                    angles.append(np.rad2deg(theta) - 90)
            if not angles:
                return image
            median_angle = float(np.median(angles))
            if abs(median_angle) < self._config["skew_threshold"]:
                return image
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            logger.debug(f"Skew corrected by {median_angle:.2f} degrees")
            return rotated
        except Exception as exc:
            logger.error(f"correct_skew failed: {exc}")
            return image

    def perspective_transform(self, image: np.ndarray) -> np.ndarray:
        """Detect document contour and apply 4-point perspective transform.

        Returns the original image if no quadrilateral contour is found.

        Args:
            image: BGR numpy array.

        Returns:
            Perspective-corrected BGR numpy array.
        """
        try:
            gray = self._to_gray(image)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 75, 200)
            contours, _ = cv2.findContours(
                edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                return image
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            doc_contour = None
            for cnt in contours[:5]:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if len(approx) == 4:
                    doc_contour = approx
                    break
            if doc_contour is None:
                return image
            pts = doc_contour.reshape(4, 2).astype(np.float32)
            dst, (out_w, out_h) = self._order_points(pts)
            M = cv2.getPerspectiveTransform(dst, np.float32([
                [0, 0], [out_w - 1, 0],
                [out_w - 1, out_h - 1], [0, out_h - 1],
            ]))
            warped = cv2.warpPerspective(image, M, (out_w, out_h))
            logger.debug("Perspective transform applied")
            return warped
        except Exception as exc:
            logger.error(f"perspective_transform failed: {exc}")
            return image

    def adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive Gaussian thresholding.

        Args:
            image: BGR or grayscale numpy array.

        Returns:
            Binary (thresholded) single-channel numpy array.
        """
        try:
            gray = self._to_gray(image)
            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2,
            )
            logger.debug("Adaptive threshold applied")
            return thresh
        except Exception as exc:
            logger.error(f"adaptive_threshold failed: {exc}")
            return image

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to the LAB lightness channel.

        Args:
            image: BGR numpy array (or grayscale – handled gracefully).

        Returns:
            CLAHE-enhanced BGR numpy array.
        """
        try:
            clip = self._config["clahe_clip_limit"]
            tile = tuple(self._config["clahe_tile_grid"])
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
            if len(image.shape) == 2:
                return clahe.apply(image)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            l_ch = clahe.apply(l_ch)
            merged = cv2.merge([l_ch, a_ch, b_ch])
            result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            logger.debug("CLAHE applied")
            return result
        except Exception as exc:
            logger.error(f"apply_clahe failed: {exc}")
            return image

    def remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """Divide by dilated image to remove lighting shadows.

        Args:
            image: BGR or grayscale numpy array.

        Returns:
            Shadow-reduced numpy array (same channel count as input).
        """
        try:
            k = self._config["shadow_dilate_kernel"]
            kernel = np.ones((k, k), np.uint8)
            if len(image.shape) == 2:
                dilated = cv2.dilate(image, kernel)
                result = cv2.divide(image, dilated, scale=255)
                return result
            channels = cv2.split(image)
            processed = []
            for ch in channels:
                dilated = cv2.dilate(ch, kernel)
                processed.append(cv2.divide(ch, dilated, scale=255))
            result = cv2.merge(processed)
            logger.debug("Shadow removal applied")
            return result
        except Exception as exc:
            logger.error(f"remove_shadows failed: {exc}")
            return image

    def morphological_cleanup(self, image: np.ndarray) -> np.ndarray:
        """Apply morphological open+close to remove noise.

        Args:
            image: Single- or multi-channel numpy array.

        Returns:
            Morphologically cleaned numpy array.
        """
        try:
            ks = self._config["morph_kernel_size"]
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (ks, ks)
            )
            opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            logger.debug("Morphological cleanup applied")
            return closed
        except Exception as exc:
            logger.error(f"morphological_cleanup failed: {exc}")
            return image

    def process(self, image: np.ndarray) -> np.ndarray:
        """Run the full preprocessing pipeline on a document image.

        Pipeline order:
        1. Shadow removal
        2. CLAHE
        3. Skew correction
        4. Perspective transform
        5. Adaptive thresholding
        6. Morphological cleanup

        Args:
            image: BGR numpy array.

        Returns:
            Preprocessed numpy array.
        """
        log_memory_usage("preprocess_start")
        result = image.copy()
        result = self.remove_shadows(result)
        result = self.apply_clahe(result)
        result = self.correct_skew(result)
        result = self.perspective_transform(result)
        result = self.adaptive_threshold(result)
        result = self.morphological_cleanup(result)
        log_memory_usage("preprocess_end")
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        """Convert to grayscale if needed."""
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _order_points(pts: np.ndarray) -> tuple:
        """Order four points: top-left, top-right, bottom-right, bottom-left.

        Returns:
            Tuple of (ordered_pts, (width, height)).
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # top-left
        rect[2] = pts[np.argmax(s)]   # bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left

        (tl, tr, br, bl) = rect
        width = int(max(
            np.linalg.norm(br - bl),
            np.linalg.norm(tr - tl),
        ))
        height = int(max(
            np.linalg.norm(tr - br),
            np.linalg.norm(tl - bl),
        ))
        width = max(width, 1)
        height = max(height, 1)
        return rect, (width, height)
