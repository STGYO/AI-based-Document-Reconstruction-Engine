"""CLI entry point for the Document Reconstruction Engine.

Usage::

    python -m doc_reconstruction_engine input1.png input2.png --output-dir output
"""
import argparse
import logging
import sys
from pathlib import Path

from core.config import PipelineConfig
from core.pipeline import DocumentReconstructionPipeline


def main() -> None:
    """Parse arguments and run the document reconstruction pipeline."""
    parser = argparse.ArgumentParser(
        description="AI-based Document Reconstruction Engine",
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Paths to document page images (PNG, JPG, etc.)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for output files (default: output)",
    )
    parser.add_argument(
        "--output-prefix",
        default="document",
        help="Base name for output files (default: document)",
    )
    parser.add_argument(
        "--enable-sr",
        action="store_true",
        help="Enable super-resolution enhancement",
    )
    parser.add_argument(
        "--enable-semantic",
        action="store_true",
        help="Enable semantic OCR correction",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.6,
        help="Minimum confidence threshold (default: 0.6)",
    )
    parser.add_argument(
        "--no-docx",
        action="store_true",
        help="Disable DOCX output",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Disable PDF output",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Disable JSON output",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Validate input files
    for path in args.images:
        if not Path(path).is_file():
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)

    config = PipelineConfig(
        output_dir=args.output_dir,
        enable_super_resolution=args.enable_sr,
        enable_semantic_correction=args.enable_semantic,
        confidence_threshold=args.confidence_threshold,
        export_docx=not args.no_docx,
        export_pdf=not args.no_pdf,
        export_json=not args.no_json,
    )

    pipeline = DocumentReconstructionPipeline(config)
    summary = pipeline.process_document(
        args.images, output_prefix=args.output_prefix
    )

    print(f"Processed {summary['page_count']} page(s)")
    for fmt, path in summary.get("outputs", {}).items():
        print(f"  {fmt}: {path}")


if __name__ == "__main__":
    main()
