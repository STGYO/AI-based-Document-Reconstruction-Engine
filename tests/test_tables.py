"""Unit tests for TableDetector."""
import unittest

import cv2
import numpy as np

from tables.detector import TableDetector, TableExporter


class TestTableDetector(unittest.TestCase):
    """Tests for TableDetector."""

    def setUp(self) -> None:
        self.detector = TableDetector()

    def test_extract_table_region(self) -> None:
        img = np.full((200, 200, 3), 255, dtype=np.uint8)
        region = self.detector.extract_table_region(img, (10, 10, 100, 100))
        self.assertEqual(region.shape[:2], (90, 90))

    def test_extract_table_region_clamps_to_bounds(self) -> None:
        img = np.full((50, 50, 3), 255, dtype=np.uint8)
        region = self.detector.extract_table_region(img, (-5, -5, 100, 100))
        self.assertEqual(region.shape[:2], (50, 50))

    def test_detect_grid_lines_returns_dict(self) -> None:
        img = np.full((100, 100), 255, dtype=np.uint8)
        result = self.detector.detect_grid_lines(img)
        self.assertIn("horizontal", result)
        self.assertIn("vertical", result)

    def test_reconstruct_table_returns_2d_list(self) -> None:
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        table = self.detector.reconstruct_table(img)
        self.assertIsInstance(table, list)
        self.assertIsInstance(table[0], list)

    def test_find_cells_no_lines(self) -> None:
        cells = self.detector._find_cells([], [], (100, 100))
        self.assertIsInstance(cells, list)
        self.assertGreater(len(cells), 0)


class TestTableExporter(unittest.TestCase):
    """Tests for TableExporter."""

    def test_to_csv(self) -> None:
        import os
        import tempfile
        exporter = TableExporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "table.csv")
            exporter.to_csv([["A", "B"], ["1", "2"]], path)
            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
