# AI-based Document Reconstruction Engine

A modular Python pipeline for digitizing camera-scanned or photographed documents. The engine preprocesses noisy images, detects page layout, performs OCR with dual-backend support, reconstructs tables, and exports clean **DOCX**, **PDF**, and **JSON** outputs.

## Features

- **Multi-stage image preprocessing** – skew correction, perspective transform, CLAHE enhancement, shadow removal, adaptive thresholding, morphological cleanup
- **Dual-backend OCR** – PaddleOCR (primary) with automatic Tesseract fallback; supports PaddleOCR v2 and v3 APIs
- **Layout detection** – LayoutParser model or heuristic OpenCV fallback for text, title, table, and image block detection
- **Table reconstruction** – grid line detection, cell extraction, CSV and DOCX table export
- **Super-resolution** – optional tile-based bicubic upsampling via PyTorch (or OpenCV fallback)
- **Semantic correction** – rule-based OCR post-processing to fix common character substitution errors
- **Confidence scoring & retry** – weighted OCR + layout confidence score with automatic retry using relaxed preprocessing parameters
- **Multi-format output** – DOCX (styled paragraphs, headings, tables), structured JSON, and image-based PDF
- **Web UI** – Gradio-based interface for interactive batch processing
- **CLI** – command-line entry point for scripted workflows
- **Memory profiling** – built-in RSS memory tracking at each pipeline stage

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌────────────────┐
│ Preprocessing│───▸│ Enhancement  │───▸│ Layout Detect  │
│  (OpenCV)   │    │ (SR optional)│    │ (LP / heuristic)│
└─────────────┘    └──────────────┘    └───────┬────────┘
                                               │
                   ┌──────────────┐    ┌───────▼────────┐
                   │  Semantic    │◂───│    OCR Engine   │
                   │  Correction  │    │(Paddle/Tesseract)│
                   └──────┬───────┘    └────────────────┘
                          │
              ┌───────────▼───────────┐
              │  Table Detection      │
              │  & Reconstruction     │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │  Output Builders      │
              │  (DOCX / JSON / PDF)  │
              └───────────────────────┘
```

## Project Structure

```
├── core/
│   ├── config.py          # PipelineConfig dataclass
│   └── pipeline.py        # DocumentReconstructionPipeline orchestrator
├── preprocessing/
│   └── preprocessor.py    # ImagePreprocessor (skew, perspective, CLAHE, …)
├── enhancement/
│   └── super_resolution.py # SuperResolutionEnhancer (torch / OpenCV)
├── ocr/
│   └── engine.py          # OCREngine (PaddleOCR + Tesseract)
├── layout/
│   └── detector.py        # LayoutDetector + LayoutBlock
├── tables/
│   └── detector.py        # TableDetector + TableExporter
├── intelligence/
│   └── semantic_corrector.py # SemanticCorrector (regex rules)
├── reconstruction/
│   └── docx_builder.py    # DocxBuilder, PDFBuilder, JSONBuilder
├── ui/
│   └── app.py             # Gradio web interface
├── utils/
│   ├── image_io.py        # load_image, save_image, conversions
│   ├── logger.py          # EngineLogger factory
│   └── memory.py          # log_memory_usage, get_memory_usage_mb
├── tests/                 # Unit tests (pytest)
├── __main__.py            # CLI entry point
├── setup.py               # Package configuration
└── requirements.txt       # Full dependency list
```

## Installation

### Core (minimal)

```bash
pip install -e .
```

This installs the base dependencies: OpenCV, NumPy, Pillow, python-docx, reportlab, psutil.

### With OCR support

```bash
# Tesseract
sudo apt-get install tesseract-ocr   # or brew install tesseract
pip install -e ".[ocr]"

# PaddleOCR
pip install -e ".[ocr]"
```

### All extras (OCR + deep learning + layout + UI)

```bash
pip install -e ".[all]"
```

### Extra groups

| Group    | Packages                                         |
|----------|--------------------------------------------------|
| `ocr`    | paddleocr, paddlepaddle, pytesseract             |
| `dl`     | torch, torchvision (for super-resolution)        |
| `layout` | layoutparser (Detectron2-based layout detection) |
| `ui`     | gradio (web interface)                           |
| `all`    | All of the above + scipy                         |

## Quick Start

### Command Line

```bash
# Process document images and produce DOCX + JSON + PDF
python -m doc_reconstruction_engine page1.png page2.png --output-dir output

# With super-resolution and semantic correction
python -m doc_reconstruction_engine scan.jpg --enable-sr --enable-semantic --output-dir results

# Disable specific outputs
python -m doc_reconstruction_engine doc.png --no-pdf --no-docx
```

### Python API

```python
from core.config import PipelineConfig
from core.pipeline import DocumentReconstructionPipeline

config = PipelineConfig(
    enable_super_resolution=False,
    enable_semantic_correction=True,
    confidence_threshold=0.6,
    output_dir="output",
)

pipeline = DocumentReconstructionPipeline(config)
summary = pipeline.process_document(
    ["page1.png", "page2.png"],
    output_prefix="my_document",
)

print(f"Processed {summary['page_count']} pages")
for fmt, path in summary["outputs"].items():
    print(f"  {fmt}: {path}")
```

### Web UI

```bash
python -m ui.app
# Opens at http://localhost:7860
```

## Configuration

All pipeline settings are controlled via `PipelineConfig`:

| Parameter                    | Default    | Description                                |
|------------------------------|------------|--------------------------------------------|
| `enable_skew_correction`     | `True`     | Correct image rotation                     |
| `enable_perspective_transform`| `True`    | Apply 4-point perspective warp             |
| `enable_clahe`               | `True`     | CLAHE contrast enhancement                 |
| `enable_shadow_removal`      | `True`     | Remove lighting shadows                    |
| `enable_super_resolution`    | `False`    | Tile-based upsampling (2×/4×)              |
| `sr_scale`                   | `2`        | Super-resolution scale factor              |
| `use_layoutparser`           | `False`    | Use LayoutParser model (vs heuristic)      |
| `ocr_primary`                | `"paddle"` | Primary OCR backend                        |
| `ocr_fallback`               | `"tesseract"` | Fallback OCR backend                    |
| `ocr_lang`                   | `"en"`     | Language code                              |
| `use_gpu`                    | `False`    | GPU acceleration                           |
| `detect_tables`              | `True`     | Enable table detection                     |
| `enable_semantic_correction` | `False`    | OCR post-processing corrections            |
| `confidence_threshold`       | `0.6`      | Minimum score before retry                 |
| `output_dir`                 | `"output"` | Output directory                           |
| `export_docx`                | `True`     | Generate DOCX output                       |
| `export_pdf`                 | `True`     | Generate PDF output                        |
| `export_json`                | `True`     | Generate JSON output                       |

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## License

This project is licensed under the terms of the [MIT License](LICENSE).
