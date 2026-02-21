"""Document reconstruction pipeline orchestrator."""
import logging
import os
from typing import List, Optional

import numpy as np

from core.config import PipelineConfig
from enhancement.super_resolution import SuperResolutionEnhancer
from intelligence.semantic_corrector import SemanticCorrector
from layout.detector import LayoutDetector
from ocr.engine import OCREngine
from preprocessing.preprocessor import ImagePreprocessor
from reconstruction.docx_builder import DocxBuilder, JSONBuilder, PDFBuilder
from tables.detector import TableDetector
from utils.memory import log_memory_usage

logger = logging.getLogger(__name__)


class DocumentReconstructionPipeline:
    """Orchestrates the full document reconstruction pipeline.

    All heavy sub-modules are initialised lazily to minimise startup time.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialise the pipeline with the given configuration.

        Args:
            config: A :class:`~core.config.PipelineConfig` instance.
        """
        self._config = config
        self._preprocessor: Optional[ImagePreprocessor] = None
        self._enhancer: Optional[SuperResolutionEnhancer] = None
        self._layout_detector: Optional[LayoutDetector] = None
        self._ocr_engine: Optional[OCREngine] = None
        self._table_detector: Optional[TableDetector] = None
        self._semantic_corrector: Optional[SemanticCorrector] = None
        logger.info("DocumentReconstructionPipeline initialised")

    # ------------------------------------------------------------------
    # Lazy module accessors
    # ------------------------------------------------------------------

    def _get_preprocessor(self) -> ImagePreprocessor:
        """Return (or create) the :class:`ImagePreprocessor` instance."""
        if self._preprocessor is None:
            self._preprocessor = ImagePreprocessor()
        return self._preprocessor

    def _get_enhancer(self) -> SuperResolutionEnhancer:
        """Return (or create) the :class:`SuperResolutionEnhancer` instance."""
        if self._enhancer is None:
            self._enhancer = SuperResolutionEnhancer(
                scale=self._config.sr_scale,
                tile_size=self._config.sr_tile_size,
                enabled=self._config.enable_super_resolution,
                device="cuda" if self._config.use_gpu else "cpu",
            )
        return self._enhancer

    def _get_layout_detector(self) -> LayoutDetector:
        """Return (or create) the :class:`LayoutDetector` instance."""
        if self._layout_detector is None:
            self._layout_detector = LayoutDetector(
                use_layoutparser=self._config.use_layoutparser
            )
        return self._layout_detector

    def _get_ocr_engine(self) -> OCREngine:
        """Return (or create) the :class:`OCREngine` instance."""
        if self._ocr_engine is None:
            self._ocr_engine = OCREngine(
                primary=self._config.ocr_primary,
                fallback=self._config.ocr_fallback,
                lang=self._config.ocr_lang,
                use_gpu=self._config.use_gpu,
            )
        return self._ocr_engine

    def _get_table_detector(self) -> TableDetector:
        """Return (or create) the :class:`TableDetector` instance."""
        if self._table_detector is None:
            self._table_detector = TableDetector()
        return self._table_detector

    def _get_semantic_corrector(self) -> SemanticCorrector:
        """Return (or create) the :class:`SemanticCorrector` instance."""
        if self._semantic_corrector is None:
            self._semantic_corrector = SemanticCorrector(
                enabled=self._config.enable_semantic_correction
            )
        return self._semantic_corrector

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process_page(
        self, image: np.ndarray, page_num: int = 0
    ) -> dict:
        """Run the full pipeline for a single page image.

        If the final confidence score falls below the configured threshold,
        the preprocessing step is re-run with adjusted parameters.

        Args:
            image: BGR numpy array of the document page.
            page_num: Zero-based page index.

        Returns:
            Page data dict with keys: ``page_num``, ``blocks``, ``tables``,
            ``ocr_confidence``, ``layout_confidence``, ``final_score``.
        """
        log_memory_usage(f"page_{page_num}_start")
        try:
            # Step 1: Preprocess
            preprocessor = self._get_preprocessor()
            preprocessed = preprocessor.process(image)

            # Step 2: Optional super-resolution
            enhancer = self._get_enhancer()
            enhanced = enhancer.enhance(preprocessed)

            # Step 3: Layout detection
            # Layout detection — ensure input is BGR (preprocessing may output grayscale)
            layout_input = self._ensure_bgr(enhanced)
            layout_detector = self._get_layout_detector()
            layout_blocks = layout_detector.detect(layout_input)

            # Step 4: OCR
            ocr_engine = self._get_ocr_engine()
            ocr_results = ocr_engine.recognize(layout_input)

            # Step 5: Semantic correction
            corrector = self._get_semantic_corrector()
            ocr_results = corrector.correct_page(ocr_results)

            # Step 6: Table detection
            tables: List[List[List[str]]] = []
            if self._config.detect_tables:
                table_detector = self._get_table_detector()
                for block in layout_blocks:
                    if block.block_type == "table":
                        region = table_detector.extract_table_region(
                            layout_input, block.bbox
                        )
                        table = table_detector.reconstruct_table(region)
                        tables.append(table)

            # Step 7: Confidence scoring
            ocr_conf, layout_conf, final_score = self._compute_confidence(
                ocr_results, layout_blocks
            )

            # Step 8: Retry with relaxed preprocessing if score is too low
            if final_score < self._config.confidence_threshold:
                logger.warning(
                    f"Page {page_num} score {final_score:.2f} below threshold "
                    f"{self._config.confidence_threshold:.2f}; retrying"
                )
                relaxed = ImagePreprocessor(
                    config={"skew_threshold": 0.2, "clahe_clip_limit": 3.0}
                )
                preprocessed2 = relaxed.process(image)
                layout_input2 = self._ensure_bgr(preprocessed2)
                ocr_results = ocr_engine.recognize(layout_input2)
                ocr_results = corrector.correct_page(ocr_results)
                ocr_conf, layout_conf, final_score = self._compute_confidence(
                    ocr_results, layout_blocks
                )

            page_data = {
                "page_num": page_num,
                "blocks": [
                    {
                        "type": b.block_type,
                        "bbox": list(b.bbox),
                        "confidence": b.confidence,
                    }
                    for b in layout_blocks
                ],
                "ocr_results": ocr_results,
                "tables": tables,
                "ocr_confidence": ocr_conf,
                "layout_confidence": layout_conf,
                "final_score": final_score,
            }
            log_memory_usage(f"page_{page_num}_end")
            return page_data
        except Exception as exc:
            logger.error(f"process_page failed on page {page_num}: {exc}")
            return {
                "page_num": page_num,
                "blocks": [],
                "ocr_results": [],
                "tables": [],
                "ocr_confidence": 0.0,
                "layout_confidence": 0.0,
                "final_score": 0.0,
            }

    def process_document(
        self,
        image_paths: List[str],
        output_prefix: str = "document",
    ) -> dict:
        """Process a list of page image files and build outputs.

        Args:
            image_paths: Ordered list of page image file paths.
            output_prefix: Base name (without extension) for output files.

        Returns:
            Summary dict with keys ``pages``, ``outputs``, ``page_count``.
        """
        from utils.image_io import load_image

        pages: List[dict] = []
        for idx, path in enumerate(image_paths):
            try:
                img = load_image(path)
                page_data = self.process_page(img, page_num=idx)
                pages.append(page_data)
            except Exception as exc:
                logger.error(f"Failed to process page {idx} ({path}): {exc}")

        outputs = self._build_outputs(pages, output_prefix)
        summary = {
            "page_count": len(pages),
            "pages": pages,
            "outputs": outputs,
        }
        logger.info(
            f"Document processing complete: {len(pages)} pages, outputs={outputs}"
        )
        return summary

    def _compute_confidence(
        self,
        ocr_results: list,
        layout_blocks: list,
    ) -> tuple:
        """Compute OCR, layout, and combined confidence scores.

        Args:
            ocr_results: List of OCR result dicts.
            layout_blocks: List of :class:`~layout.detector.LayoutBlock` objects.

        Returns:
            Tuple of ``(ocr_confidence, layout_confidence, final_score)``.
        """
        try:
            if ocr_results:
                confs = [r.get("confidence", 0.0) for r in ocr_results]
                ocr_conf = float(sum(confs) / len(confs))
            else:
                ocr_conf = 0.0

            if layout_blocks:
                layout_conf = float(
                    sum(b.confidence for b in layout_blocks) / len(layout_blocks)
                )
            else:
                layout_conf = 0.5  # neutral when no blocks detected

            final_score = 0.7 * ocr_conf + 0.3 * layout_conf
            return ocr_conf, layout_conf, final_score
        except Exception as exc:
            logger.error(f"_compute_confidence failed: {exc}")
            return 0.0, 0.0, 0.0

    def _build_outputs(
        self, pages: List[dict], output_prefix: str
    ) -> dict:
        """Build DOCX, PDF, and JSON output files from page data.

        Args:
            pages: List of page data dicts from :meth:`process_page`.
            output_prefix: Base name for output files.

        Returns:
            Dict mapping format name to output file path.
        """
        out_dir = self._config.output_dir
        os.makedirs(out_dir, exist_ok=True)
        outputs: dict = {}

        try:
            if self._config.export_docx:
                docx_path = os.path.join(out_dir, f"{output_prefix}.docx")
                builder = DocxBuilder(docx_path)
                for page in pages:
                    for block in page.get("blocks", []):
                        block_type = block.get("type", "text")
                        text = ""
                        # Gather OCR text from matching bbox (best effort)
                        for res in page.get("ocr_results", []):
                            text += res.get("text", "") + " "
                        text = text.strip()
                        if block_type == "title":
                            builder.add_heading(text or "[Title]", level=1)
                        else:
                            if text:
                                builder.add_paragraph(text)
                    for table in page.get("tables", []):
                        builder.add_table(table)
                outputs["docx"] = builder.save()
        except Exception as exc:
            logger.error(f"DOCX build failed: {exc}")

        try:
            if self._config.export_json:
                json_path = os.path.join(out_dir, f"{output_prefix}.json")
                jbuilder = JSONBuilder(json_path)
                for page in pages:
                    # Exclude raw numpy data
                    safe_page = {
                        k: v for k, v in page.items()
                        if k != "image"
                    }
                    jbuilder.add_page(safe_page)
                outputs["json"] = jbuilder.save()
        except Exception as exc:
            logger.error(f"JSON build failed: {exc}")

        return outputs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_bgr(image: np.ndarray) -> np.ndarray:
        """Convert a single-channel image to 3-channel BGR if needed."""
        import cv2
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image
