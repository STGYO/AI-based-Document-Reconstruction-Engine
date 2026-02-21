"""Document reconstruction output builders."""
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from docx import Document as DocxDocument
    from docx.shared import Cm
    _DOCX_AVAILABLE = True
except ImportError:
    _DOCX_AVAILABLE = False
    logger.warning("python-docx not available")

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas as rl_canvas
    _REPORTLAB_AVAILABLE = True
except ImportError:
    _REPORTLAB_AVAILABLE = False
    logger.warning("reportlab not available")


class DocxBuilder:
    """Builds a DOCX document incrementally."""

    def __init__(self, output_path: str) -> None:
        """Initialise the builder.

        Args:
            output_path: Destination ``.docx`` file path.
        """
        self._output_path = output_path
        if _DOCX_AVAILABLE:
            self._doc = DocxDocument()
        else:
            self._doc = None
        logger.debug(f"DocxBuilder output_path={output_path}")

    def add_paragraph(self, text: str, style: str = "Normal") -> None:
        """Add a paragraph with the given style.

        Args:
            text: Paragraph text.
            style: Word paragraph style name.
        """
        if self._doc is None:
            logger.warning("python-docx not available; skipping add_paragraph")
            return
        try:
            self._doc.add_paragraph(text, style=style)
        except Exception as exc:
            logger.error(f"add_paragraph failed: {exc}")

    def add_heading(self, text: str, level: int = 1) -> None:
        """Add a heading at the specified level (1–4).

        Args:
            text: Heading text.
            level: Heading level (1 = largest).
        """
        if self._doc is None:
            logger.warning("python-docx not available; skipping add_heading")
            return
        try:
            level = max(1, min(4, level))
            self._doc.add_heading(text, level=level)
        except Exception as exc:
            logger.error(f"add_heading failed: {exc}")

    def add_table(self, table_data: List[List[str]]) -> None:
        """Add a table to the document.

        Args:
            table_data: 2-D list of strings.
        """
        if self._doc is None:
            logger.warning("python-docx not available; skipping add_table")
            return
        try:
            rows = len(table_data)
            cols = max((len(row) for row in table_data), default=1)
            tbl = self._doc.add_table(rows=rows, cols=cols)
            tbl.style = "Table Grid"
            for r_idx, row in enumerate(table_data):
                for c_idx, cell_text in enumerate(row):
                    tbl.cell(r_idx, c_idx).text = str(cell_text)
        except Exception as exc:
            logger.error(f"add_table failed: {exc}")

    def add_image(self, image_path: str, width_cm: float = 15.0) -> None:
        """Add an image to the document.

        Args:
            image_path: Path to the image file on disk.
            width_cm: Displayed width in centimetres.
        """
        if self._doc is None:
            logger.warning("python-docx not available; skipping add_image")
            return
        try:
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                return
            self._doc.add_picture(image_path, width=Cm(width_cm))
        except Exception as exc:
            logger.error(f"add_image failed: {exc}")

    def save(self) -> str:
        """Save the document to disk.

        Returns:
            Absolute path of the saved file.
        """
        if self._doc is None:
            logger.warning("python-docx not available; skipping save")
            return self._output_path
        try:
            Path(self._output_path).parent.mkdir(parents=True, exist_ok=True)
            self._doc.save(self._output_path)
            logger.info(f"DOCX saved: {self._output_path}")
        except Exception as exc:
            logger.error(f"DocxBuilder.save failed: {exc}")
        return self._output_path

    def _estimate_heading_level(
        self, bbox_height: int, page_height: int
    ) -> int:
        """Estimate heading level from block height relative to page.

        Args:
            bbox_height: Height of the text bounding box in pixels.
            page_height: Total page height in pixels.

        Returns:
            Integer heading level between 1 and 4.
        """
        ratio = bbox_height / page_height if page_height > 0 else 0
        if ratio > 0.08:
            return 1
        if ratio > 0.05:
            return 2
        if ratio > 0.03:
            return 3
        return 4


class PDFBuilder:
    """Builds a multi-page PDF from cleaned page images."""

    def __init__(self, output_path: str) -> None:
        """Initialise the PDF builder.

        Args:
            output_path: Destination ``.pdf`` file path.
        """
        self._output_path = output_path
        self._pages: List[np.ndarray] = []
        logger.debug(f"PDFBuilder output_path={output_path}")

    def add_page(self, image: np.ndarray) -> None:
        """Add a cleaned image as a PDF page.

        Args:
            image: BGR numpy array representing one document page.
        """
        self._pages.append(image)

    def save(self) -> str:
        """Render all pages and save the PDF to disk.

        Returns:
            Absolute path of the saved file.
        """
        if not _REPORTLAB_AVAILABLE:
            logger.warning("reportlab not available; cannot save PDF")
            return self._output_path
        try:
            import io
            import cv2
            from PIL import Image as PilImage

            Path(self._output_path).parent.mkdir(parents=True, exist_ok=True)
            c = rl_canvas.Canvas(self._output_path, pagesize=A4)
            page_w, page_h = A4

            for img_bgr in self._pages:
                if len(img_bgr.shape) == 2:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
                else:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                pil_img = PilImage.fromarray(img_rgb)
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                buf.seek(0)
                reader = ImageReader(buf)
                c.drawImage(reader, 0, 0, width=page_w, height=page_h)
                c.showPage()

            c.save()
            logger.info(f"PDF saved: {self._output_path}")
        except Exception as exc:
            logger.error(f"PDFBuilder.save failed: {exc}")
        return self._output_path


class JSONBuilder:
    """Builds a structured JSON output from page data."""

    def __init__(self, output_path: str) -> None:
        """Initialise the JSON builder.

        Args:
            output_path: Destination ``.json`` file path.
        """
        self._output_path = output_path
        self._pages: List[dict] = []
        logger.debug(f"JSONBuilder output_path={output_path}")

    def add_page(self, page_data: dict) -> None:
        """Append a page data dictionary.

        Args:
            page_data: Arbitrary dict describing one document page.
        """
        self._pages.append(page_data)

    def save(self) -> str:
        """Write the accumulated pages to a JSON file.

        Returns:
            Absolute path of the saved file.
        """
        try:
            Path(self._output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self._output_path, "w", encoding="utf-8") as fh:
                json.dump({"pages": self._pages}, fh, indent=2, ensure_ascii=False)
            logger.info(f"JSON saved: {self._output_path}")
        except Exception as exc:
            logger.error(f"JSONBuilder.save failed: {exc}")
        return self._output_path
