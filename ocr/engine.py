"""OCR engine module supporting PaddleOCR and Tesseract backends."""
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from paddleocr import PaddleOCR
    _PADDLE_AVAILABLE = True
except ImportError:
    _PADDLE_AVAILABLE = False
    logger.warning("paddleocr not available")

try:
    import pytesseract
    from PIL import Image as PilImage
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False
    logger.warning("pytesseract / Pillow not available")


@dataclass
class OCRResult:
    """Container for a single OCR recognition result."""

    text: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)


class OCREngine:
    """Multi-backend OCR engine with automatic fallback.

    Primary backend is PaddleOCR; Tesseract is used as fallback.
    """

    def __init__(
        self,
        primary: str = "paddle",
        fallback: str = "tesseract",
        lang: str = "en",
        use_gpu: bool = False,
    ) -> None:
        """Initialise the OCR engine.

        Args:
            primary: Primary backend name (``"paddle"`` or ``"tesseract"``).
            fallback: Fallback backend name.
            lang: Language code (``"en"`` etc.).
            use_gpu: Whether to use GPU acceleration.
        """
        self._primary = primary
        self._fallback = fallback
        self._lang = lang
        self._use_gpu = use_gpu
        self._paddle_ocr = None
        self._tesseract_ready: bool = False
        logger.info(
            f"OCREngine primary={primary} fallback={fallback} lang={lang}"
        )

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load_paddle(self) -> None:
        """Lazy-load PaddleOCR instance."""
        if not _PADDLE_AVAILABLE:
            logger.warning("paddleocr is not installed; skipping paddle load")
            return
        try:
            self._paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self._lang,
                use_gpu=self._use_gpu,
                show_log=False,
            )
            logger.info("PaddleOCR loaded successfully")
        except Exception as exc:
            logger.error(f"Failed to load PaddleOCR: {exc}")
            self._paddle_ocr = None

    def _load_tesseract(self) -> None:
        """Lazy-check that pytesseract is functional."""
        if not _TESSERACT_AVAILABLE:
            logger.warning("pytesseract / Pillow not installed")
            return
        try:
            pytesseract.get_tesseract_version()
            self._tesseract_ready = True
            logger.info("Tesseract ready")
        except Exception as exc:
            logger.error(f"Tesseract not functional: {exc}")
            self._tesseract_ready = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recognize(self, image: np.ndarray) -> List[dict]:
        """Recognize text in an image.

        Tries the primary backend first, then falls back to the secondary.

        Args:
            image: BGR numpy array.

        Returns:
            List of dicts with keys ``text``, ``confidence``, ``bbox``.
        """
        result: Optional[List[dict]] = None
        if self._primary == "paddle":
            result = self._paddle_recognize(image)
        elif self._primary == "tesseract":
            result = self._tesseract_recognize(image)

        if not result and self._fallback:
            logger.debug("Primary OCR returned no results, trying fallback")
            if self._fallback == "tesseract":
                result = self._tesseract_recognize(image)
            elif self._fallback == "paddle":
                result = self._paddle_recognize(image)

        return result or []

    def _paddle_recognize(self, image: np.ndarray) -> List[dict]:
        """Run PaddleOCR recognition.

        Args:
            image: BGR numpy array.

        Returns:
            List of result dicts, or empty list on failure.
        """
        try:
            if self._paddle_ocr is None:
                self._load_paddle()
            if self._paddle_ocr is None:
                return []
            raw = self._paddle_ocr.ocr(image, cls=True)
            results: List[dict] = []
            if not raw:
                return results
            for page in raw:
                if not page:
                    continue
                for item in page:
                    box, (text, conf) = item
                    xs = [pt[0] for pt in box]
                    ys = [pt[1] for pt in box]
                    results.append({
                        "text": text,
                        "confidence": float(conf),
                        "bbox": (
                            int(min(xs)), int(min(ys)),
                            int(max(xs)), int(max(ys)),
                        ),
                    })
            return results
        except Exception as exc:
            logger.error(f"PaddleOCR recognition failed: {exc}")
            return []

    def _tesseract_recognize(self, image: np.ndarray) -> List[dict]:
        """Run Tesseract recognition.

        Args:
            image: BGR numpy array.

        Returns:
            List of result dicts, or empty list on failure.
        """
        try:
            if not self._tesseract_ready:
                self._load_tesseract()
            if not self._tesseract_ready:
                return []
            import cv2
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = PilImage.fromarray(rgb)
            data = pytesseract.image_to_data(
                pil_img,
                lang=self._lang,
                output_type=pytesseract.Output.DICT,
            )
            results: List[dict] = []
            for i, text in enumerate(data["text"]):
                text = text.strip()
                if not text:
                    continue
                conf = float(data["conf"][i])
                if conf < 0:
                    continue
                x = data["left"][i]
                y = data["top"][i]
                w = data["width"][i]
                h = data["height"][i]
                results.append({
                    "text": text,
                    "confidence": conf / 100.0,
                    "bbox": (x, y, x + w, y + h),
                })
            return results
        except Exception as exc:
            logger.error(f"Tesseract recognition failed: {exc}")
            return []
