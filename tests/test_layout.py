"""Unit tests for LayoutDetector."""
import unittest

import cv2
import numpy as np

from layout.detector import LayoutBlock, LayoutDetector


def _make_text_image(h: int = 200, w: int = 400) -> np.ndarray:
    """Create a synthetic BGR image with dark text-like rectangles."""
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    # Simulate two text lines
    cv2.rectangle(img, (10, 20), (380, 40), (0, 0, 0), -1)
    cv2.rectangle(img, (10, 60), (350, 80), (0, 0, 0), -1)
    cv2.rectangle(img, (10, 120), (300, 140), (0, 0, 0), -1)
    return img


class TestLayoutDetector(unittest.TestCase):
    """Tests for LayoutDetector heuristic and public detect method."""

    def setUp(self) -> None:
        self.detector = LayoutDetector(use_layoutparser=False)
        self.image = _make_text_image()

    def test_heuristic_detect_returns_list(self) -> None:
        blocks = self.detector._heuristic_detect(self.image)
        self.assertIsInstance(blocks, list)

    def test_heuristic_detect_blocks_are_sorted(self) -> None:
        blocks = self.detector._heuristic_detect(self.image)
        if len(blocks) > 1:
            ys = [b.y1 for b in blocks]
            self.assertEqual(ys, sorted(ys))

    def test_heuristic_detect_block_properties(self) -> None:
        blocks = self.detector._heuristic_detect(self.image)
        for block in blocks:
            self.assertIsInstance(block, LayoutBlock)
            self.assertGreater(block.width, 0)
            self.assertGreater(block.height, 0)
            self.assertIn(block.block_type, {"text", "title", "table", "image", "list"})

    def test_detect_returns_list(self) -> None:
        blocks = self.detector.detect(self.image)
        self.assertIsInstance(blocks, list)

    def test_detect_does_not_crash_on_blank_image(self) -> None:
        blank = np.full((200, 200, 3), 255, dtype=np.uint8)
        try:
            blocks = self.detector.detect(blank)
        except Exception as exc:
            self.fail(f"detect() raised an exception on blank image: {exc}")
        self.assertIsInstance(blocks, list)

    def test_detect_sorted_by_y_then_x(self) -> None:
        blocks = self.detector.detect(self.image)
        if len(blocks) > 1:
            keys = [(b.y1, b.x1) for b in blocks]
            self.assertEqual(keys, sorted(keys))


if __name__ == "__main__":
    unittest.main()
