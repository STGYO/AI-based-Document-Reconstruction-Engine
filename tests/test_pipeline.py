"""Unit tests for DocumentReconstructionPipeline.process_page."""
import unittest

import numpy as np

from core.config import PipelineConfig
from core.pipeline import DocumentReconstructionPipeline


def _make_synthetic_page(h: int = 200, w: int = 200) -> np.ndarray:
    """Create a simple BGR test image."""
    import cv2
    img = np.full((h, w, 3), 240, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (180, 50), (0, 0, 0), -1)
    cv2.rectangle(img, (20, 70), (160, 90), (50, 50, 50), -1)
    return img


class TestDocumentReconstructionPipeline(unittest.TestCase):
    """Tests for DocumentReconstructionPipeline."""

    def _make_config(self) -> PipelineConfig:
        """Return a minimal config with all heavy features disabled."""
        return PipelineConfig(
            enable_super_resolution=False,
            enable_semantic_correction=False,
            use_layoutparser=False,
            detect_tables=False,
            export_docx=False,
            export_pdf=False,
            export_json=False,
            ocr_primary="tesseract",
            ocr_fallback="",
            confidence_threshold=0.0,
            output_dir="/tmp/doc_recon_test",
        )

    def test_process_page_returns_dict(self) -> None:
        config = self._make_config()
        pipeline = DocumentReconstructionPipeline(config)
        image = _make_synthetic_page()
        result = pipeline.process_page(image, page_num=0)
        self.assertIsInstance(result, dict)

    def test_process_page_has_required_keys(self) -> None:
        config = self._make_config()
        pipeline = DocumentReconstructionPipeline(config)
        image = _make_synthetic_page()
        result = pipeline.process_page(image, page_num=0)
        for key in ("page_num", "blocks", "final_score"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_process_page_page_num(self) -> None:
        config = self._make_config()
        pipeline = DocumentReconstructionPipeline(config)
        image = _make_synthetic_page()
        result = pipeline.process_page(image, page_num=3)
        self.assertEqual(result["page_num"], 3)

    def test_process_page_blocks_is_list(self) -> None:
        config = self._make_config()
        pipeline = DocumentReconstructionPipeline(config)
        image = _make_synthetic_page()
        result = pipeline.process_page(image)
        self.assertIsInstance(result["blocks"], list)

    def test_process_page_final_score_is_float(self) -> None:
        config = self._make_config()
        pipeline = DocumentReconstructionPipeline(config)
        image = _make_synthetic_page()
        result = pipeline.process_page(image)
        self.assertIsInstance(result["final_score"], float)


if __name__ == "__main__":
    unittest.main()
