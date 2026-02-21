"""Unit tests for SemanticCorrector."""
import unittest

from intelligence.semantic_corrector import SemanticCorrector


class TestSemanticCorrector(unittest.TestCase):
    """Tests for SemanticCorrector."""

    def test_disabled_returns_unchanged(self) -> None:
        corrector = SemanticCorrector(enabled=False)
        blocks = [{"text": "0pen", "confidence": 0.8}]
        result = corrector.correct_page(blocks)
        self.assertEqual(result[0]["text"], "0pen")

    def test_enabled_fixes_ocr_errors(self) -> None:
        corrector = SemanticCorrector(enabled=True)
        blocks = [{"text": "0pen", "confidence": 0.8}]
        result = corrector.correct_page(blocks)
        self.assertEqual(result[0]["text"], "Open")

    def test_whitespace_normalization(self) -> None:
        corrector = SemanticCorrector(enabled=True)
        blocks = [{"text": "hello   world  test", "confidence": 0.9}]
        result = corrector.correct_page(blocks)
        self.assertEqual(result[0]["text"], "hello world test")

    def test_empty_blocks(self) -> None:
        corrector = SemanticCorrector(enabled=True)
        result = corrector.correct_page([])
        self.assertEqual(result, [])

    def test_preserves_other_keys(self) -> None:
        corrector = SemanticCorrector(enabled=True)
        blocks = [{"text": "hello", "confidence": 0.9, "bbox": (0, 0, 10, 10)}]
        result = corrector.correct_page(blocks)
        self.assertEqual(result[0]["confidence"], 0.9)
        self.assertEqual(result[0]["bbox"], (0, 0, 10, 10))


if __name__ == "__main__":
    unittest.main()
