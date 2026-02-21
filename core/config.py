"""Global pipeline configuration."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PipelineConfig:
    """Configuration for the full document reconstruction pipeline."""

    # Preprocessing
    enable_skew_correction: bool = True
    enable_perspective_transform: bool = True
    enable_clahe: bool = True
    enable_shadow_removal: bool = True

    # Enhancement
    enable_super_resolution: bool = False
    sr_scale: int = 2
    sr_tile_size: int = 256

    # Layout
    use_layoutparser: bool = False

    # OCR
    ocr_primary: str = "paddle"
    ocr_fallback: str = "tesseract"
    ocr_lang: str = "en"
    use_gpu: bool = False

    # Tables
    detect_tables: bool = True

    # Intelligence
    enable_semantic_correction: bool = False

    # Confidence
    confidence_threshold: float = 0.6

    # Output
    output_dir: str = "output"
    export_docx: bool = True
    export_pdf: bool = True
    export_json: bool = True

    # Performance
    batch_size: int = 1
