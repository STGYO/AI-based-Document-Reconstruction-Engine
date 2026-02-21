"""Layout detection module."""
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LayoutBlock:
    """Represents a detected layout region."""

    x1: int
    y1: int
    x2: int
    y2: int
    block_type: str  # "text", "title", "table", "image", "list"
    confidence: float = 1.0

    @property
    def bbox(self):
        """Return (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> int:
        """Width of the block in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Height of the block in pixels."""
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """Area of the block in pixels squared."""
        return self.width * self.height


class LayoutDetector:
    """Detects layout blocks in a document image.

    Uses a heuristic OpenCV-based approach as fallback when
    LayoutParser models are unavailable (for offline/CPU use).
    Attempts to load a lightweight LayoutParser model first.
    """

    def __init__(
        self,
        use_layoutparser: bool = False,
        config: Optional[dict] = None,
    ) -> None:
        """Initialise the layout detector.

        Args:
            use_layoutparser: Whether to attempt loading a LayoutParser model.
            config: Optional configuration dictionary.
        """
        self._use_layoutparser = use_layoutparser
        self._config = config or {}
        self._model = None
        logger.info(f"LayoutDetector initialised (layoutparser={use_layoutparser})")

    def _load_layoutparser_model(self) -> bool:
        """Try to load LayoutParser model.

        Returns:
            ``True`` on success, ``False`` otherwise.
        """
        try:
            import layoutparser as lp
            self._model = lp.Detectron2LayoutModel(
                "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                label_map={
                    0: "text", 1: "title", 2: "list",
                    3: "table", 4: "figure",
                },
            )
            logger.info("LayoutParser model loaded")
            return True
        except Exception as exc:
            logger.warning(
                f"LayoutParser model unavailable: {exc}. Using heuristic fallback."
            )
            return False

    def _heuristic_detect(self, image: np.ndarray) -> List[LayoutBlock]:
        """Heuristic block detection via connected components.

        Args:
            image: BGR or grayscale numpy array.

        Returns:
            List of :class:`LayoutBlock` instances.
        """
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3
            else image
        )
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        blocks: List[LayoutBlock] = []
        h, w = image.shape[:2]
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < w * 0.02 or bh < h * 0.005:
                continue
            block_type = (
                "title" if bh > h * 0.05 and bw > w * 0.3 else "text"
            )
            blocks.append(
                LayoutBlock(x1=x, y1=y, x2=x + bw, y2=y + bh, block_type=block_type)
            )
        blocks.sort(key=lambda b: (b.y1, b.x1))
        return blocks

    def detect(self, image: np.ndarray) -> List[LayoutBlock]:
        """Detect layout blocks and return them sorted top-to-bottom.

        Args:
            image: BGR numpy array of the document page.

        Returns:
            List of :class:`LayoutBlock`, sorted by (y1, x1).
        """
        try:
            if self._use_layoutparser and self._model is None:
                self._load_layoutparser_model()
            if self._model is not None:
                layout = self._model.detect(image)
                blocks = [
                    LayoutBlock(
                        x1=int(b.block.x_1),
                        y1=int(b.block.y_1),
                        x2=int(b.block.x_2),
                        y2=int(b.block.y_2),
                        block_type=b.type.lower(),
                        confidence=b.score,
                    )
                    for b in layout
                ]
            else:
                blocks = self._heuristic_detect(image)
            blocks.sort(key=lambda b: (b.y1, b.x1))
            logger.info(f"Detected {len(blocks)} layout blocks")
            return blocks
        except Exception as exc:
            logger.error(f"Layout detection failed: {exc}")
            return []
