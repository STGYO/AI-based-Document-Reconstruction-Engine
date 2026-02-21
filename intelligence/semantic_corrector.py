"""Semantic correction module for OCR post-processing."""
import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

# Common single-character OCR confusion pairs: (wrong, correct) in context.
_OCR_REPLACEMENTS = [
    # Digit-letter swaps that are clearly wrong in typical word context
    (r'\b0([a-z])', r'O\1'),   # 0 at start of word followed by letter → O
    (r'([a-z])0\b', r'\1o'),   # 0 at end of word after letter → o
    (r'\bl([A-Z])', r'I\1'),   # lowercase l before uppercase → I
]


class SemanticCorrector:
    """Applies lightweight heuristic corrections to OCR text blocks.

    No large language model is used; corrections are rule-based.
    """

    def __init__(
        self,
        enabled: bool = True,
        config: Optional[dict] = None,
    ) -> None:
        """Initialise the corrector.

        Args:
            enabled: Whether correction is active.
            config: Optional configuration dictionary.
        """
        self._enabled = enabled
        self._config = config or {}
        logger.debug(f"SemanticCorrector enabled={enabled}")

    def correct_page(self, text_blocks: List[dict]) -> List[dict]:
        """Apply corrections to all text blocks on a page.

        Each block dict is expected to have a ``"text"`` key.
        Other keys are passed through unchanged.

        Args:
            text_blocks: List of OCR result dicts with at least a ``"text"`` key.

        Returns:
            New list of dicts with corrected ``"text"`` values.
        """
        if not self._enabled:
            return text_blocks
        corrected: List[dict] = []
        for block in text_blocks:
            try:
                new_block = dict(block)
                text = new_block.get("text", "")
                text = self._fix_common_ocr_errors(text)
                text = self._normalize_whitespace(text)
                new_block["text"] = text
                corrected.append(new_block)
            except Exception as exc:
                logger.error(f"correct_page block processing failed: {exc}")
                corrected.append(block)
        return corrected

    def _fix_common_ocr_errors(self, text: str) -> str:
        """Fix common OCR character-substitution errors.

        Applies a set of regex-based replacements for typical digit/letter
        confusion (e.g. ``0`` ↔ ``O``, ``l`` ↔ ``I``).

        Args:
            text: Input string with potential OCR errors.

        Returns:
            Corrected string.
        """
        try:
            for pattern, replacement in _OCR_REPLACEMENTS:
                text = re.sub(pattern, replacement, text)
        except Exception as exc:
            logger.error(f"_fix_common_ocr_errors failed: {exc}")
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Collapse multiple whitespace characters into a single space.

        Args:
            text: Input string.

        Returns:
            String with normalised whitespace.
        """
        try:
            return re.sub(r'[ \t]+', ' ', text).strip()
        except Exception as exc:
            logger.error(f"_normalize_whitespace failed: {exc}")
            return text
