"""Unit tests for OCREngine."""
import unittest

import numpy as np

from ocr.engine import OCREngine


class TestOCREngine(unittest.TestCase):
    """Tests for OCREngine initialization and language mapping."""

    def test_init_defaults(self) -> None:
        engine = OCREngine()
        self.assertEqual(engine._primary, "paddle")
        self.assertEqual(engine._fallback, "tesseract")
        self.assertEqual(engine._lang, "en")
        self.assertEqual(engine._tesseract_lang, "eng")

    def test_tesseract_lang_mapping(self) -> None:
        self.assertEqual(OCREngine._map_tesseract_lang("en"), "eng")
        self.assertEqual(OCREngine._map_tesseract_lang("fr"), "fra")
        self.assertEqual(OCREngine._map_tesseract_lang("de"), "deu")
        self.assertEqual(OCREngine._map_tesseract_lang("ja"), "jpn")
        # Unknown language passes through
        self.assertEqual(OCREngine._map_tesseract_lang("xyz"), "xyz")

    def test_recognize_returns_list(self) -> None:
        engine = OCREngine(primary="tesseract", fallback="")
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        result = engine.recognize(img)
        self.assertIsInstance(result, list)

    def test_recognize_empty_image(self) -> None:
        engine = OCREngine(primary="tesseract", fallback="")
        img = np.full((50, 50, 3), 255, dtype=np.uint8)
        result = engine.recognize(img)
        self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main()
