"""Unit tests for ImagePreprocessor."""
import unittest

import cv2
import numpy as np

from preprocessing.preprocessor import ImagePreprocessor


def _make_color_image(h: int = 200, w: int = 200) -> np.ndarray:
    """Create a synthetic BGR image with a black rectangle on white."""
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (160, 160), (0, 0, 0), -1)
    return img


def _make_gray_image(h: int = 200, w: int = 200) -> np.ndarray:
    """Create a synthetic grayscale image with a black rectangle on white."""
    img = np.full((h, w), 255, dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (160, 160), 0, -1)
    return img


class TestImagePreprocessor(unittest.TestCase):
    """Tests for each preprocessing method and the full pipeline."""

    def setUp(self) -> None:
        self.preprocessor = ImagePreprocessor()
        self.color_img = _make_color_image()
        self.gray_img = _make_gray_image()

    # ------------------------------------------------------------------
    # correct_skew
    # ------------------------------------------------------------------

    def test_correct_skew_color(self) -> None:
        result = self.preprocessor.correct_skew(self.color_img)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[:2], self.color_img.shape[:2])

    def test_correct_skew_gray(self) -> None:
        result = self.preprocessor.correct_skew(self.gray_img)
        self.assertIsInstance(result, np.ndarray)

    # ------------------------------------------------------------------
    # adaptive_threshold
    # ------------------------------------------------------------------

    def test_adaptive_threshold_from_color(self) -> None:
        result = self.preprocessor.adaptive_threshold(self.color_img)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result.shape), 2)  # single channel

    def test_adaptive_threshold_from_gray(self) -> None:
        result = self.preprocessor.adaptive_threshold(self.gray_img)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result.shape), 2)

    # ------------------------------------------------------------------
    # apply_clahe
    # ------------------------------------------------------------------

    def test_apply_clahe_color(self) -> None:
        result = self.preprocessor.apply_clahe(self.color_img)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, self.color_img.shape)

    def test_apply_clahe_gray(self) -> None:
        result = self.preprocessor.apply_clahe(self.gray_img)
        self.assertIsInstance(result, np.ndarray)

    # ------------------------------------------------------------------
    # remove_shadows
    # ------------------------------------------------------------------

    def test_remove_shadows_color(self) -> None:
        result = self.preprocessor.remove_shadows(self.color_img)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, self.color_img.shape)

    def test_remove_shadows_gray(self) -> None:
        result = self.preprocessor.remove_shadows(self.gray_img)
        self.assertIsInstance(result, np.ndarray)

    # ------------------------------------------------------------------
    # morphological_cleanup
    # ------------------------------------------------------------------

    def test_morphological_cleanup_color(self) -> None:
        result = self.preprocessor.morphological_cleanup(self.color_img)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, self.color_img.shape)

    def test_morphological_cleanup_gray(self) -> None:
        result = self.preprocessor.morphological_cleanup(self.gray_img)
        self.assertIsInstance(result, np.ndarray)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def test_process_end_to_end(self) -> None:
        result = self.preprocessor.process(self.color_img)
        self.assertIsInstance(result, np.ndarray)
        self.assertGreater(result.size, 0)

    def test_process_does_not_raise(self) -> None:
        try:
            self.preprocessor.process(self.color_img)
        except Exception as exc:
            self.fail(f"process() raised an exception: {exc}")


if __name__ == "__main__":
    unittest.main()
