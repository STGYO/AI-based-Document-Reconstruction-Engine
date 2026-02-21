"""Unit tests for DocxBuilder, JSONBuilder, and PDFBuilder."""
import json
import os
import tempfile
import unittest

import numpy as np

from reconstruction.docx_builder import DocxBuilder, JSONBuilder, PDFBuilder


class TestDocxBuilder(unittest.TestCase):
    """Tests for DocxBuilder."""

    def test_save_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.docx")
            builder = DocxBuilder(path)
            builder.add_paragraph("Hello, world!")
            builder.add_heading("Section 1", level=1)
            builder.add_table([["A", "B"], ["1", "2"]])
            saved = builder.save()
            self.assertEqual(saved, path)
            self.assertTrue(os.path.exists(path))

    def test_add_heading_levels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "headings.docx")
            builder = DocxBuilder(path)
            for level in range(1, 5):
                builder.add_heading(f"Heading {level}", level=level)
            builder.save()
            self.assertTrue(os.path.exists(path))

    def test_estimate_heading_level(self) -> None:
        builder = DocxBuilder("/tmp/unused.docx")
        self.assertEqual(builder._estimate_heading_level(100, 1000), 1)
        self.assertEqual(builder._estimate_heading_level(55, 1000), 2)
        self.assertEqual(builder._estimate_heading_level(35, 1000), 3)
        self.assertEqual(builder._estimate_heading_level(10, 1000), 4)


class TestJSONBuilder(unittest.TestCase):
    """Tests for JSONBuilder."""

    def test_save_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            builder = JSONBuilder(path)
            builder.add_page({"page_num": 0, "blocks": []})
            builder.add_page({"page_num": 1, "blocks": ["something"]})
            saved = builder.save()
            self.assertEqual(saved, path)
            self.assertTrue(os.path.exists(path))

    def test_json_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "struct.json")
            builder = JSONBuilder(path)
            builder.add_page({"page_num": 0})
            builder.save()
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertIn("pages", data)
            self.assertEqual(len(data["pages"]), 1)


class TestPDFBuilder(unittest.TestCase):
    """Tests for PDFBuilder."""

    def test_save_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pdf")
            builder = PDFBuilder(path)
            # Create a small synthetic BGR image
            img = np.full((100, 100, 3), 200, dtype=np.uint8)
            builder.add_page(img)
            saved = builder.save()
            self.assertEqual(saved, path)
            self.assertTrue(os.path.exists(path))

    def test_empty_pdf_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty.pdf")
            builder = PDFBuilder(path)
            # No pages added — should not crash
            try:
                builder.save()
            except Exception as exc:
                self.fail(f"save() raised with no pages: {exc}")


if __name__ == "__main__":
    unittest.main()
