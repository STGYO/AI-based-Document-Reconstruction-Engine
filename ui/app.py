"""Gradio-based web UI for the Document Reconstruction Engine."""
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import gradio as gr
    _GRADIO_AVAILABLE = True
except ImportError:
    _GRADIO_AVAILABLE = False
    logger.warning("gradio not available; UI cannot be launched")

from core.config import PipelineConfig
from core.pipeline import DocumentReconstructionPipeline


class DocumentReconstructionUI:
    """Gradio-based web interface for the Document Reconstruction Engine."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        """Initialise the UI.

        Args:
            config: Optional pipeline configuration. If ``None``, defaults are used.
        """
        self._base_config = config or PipelineConfig()
        self._interface: Optional[object] = None
        logger.info("DocumentReconstructionUI initialised")

    def _process_files(
        self,
        files: List,
        enable_sr: bool,
        enable_semantic: bool,
        confidence_threshold: float,
    ) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
        """Process uploaded files through the reconstruction pipeline.

        Args:
            files: List of uploaded file objects (Gradio UploadedFile).
            enable_sr: Whether to enable super-resolution.
            enable_semantic: Whether to enable semantic correction.
            confidence_threshold: Minimum confidence score.

        Returns:
            Tuple of ``(status_text, docx_path, pdf_path, json_path)``.
        """
        if not files:
            return "No files uploaded.", None, None, None
        try:
            config = PipelineConfig(
                enable_super_resolution=enable_sr,
                enable_semantic_correction=enable_semantic,
                confidence_threshold=confidence_threshold,
                output_dir=tempfile.mkdtemp(prefix="doc_recon_"),
                # PDF pages are built from raw BGR arrays; the pipeline stores
                # page images internally for that purpose, but the UI only
                # receives file paths, so PDF export is not available here.
                export_pdf=False,
            )
            pipeline = DocumentReconstructionPipeline(config)
            # Gradio returns file paths as strings or objects with .name
            paths = [
                f.name if hasattr(f, "name") else str(f)
                for f in files
            ]
            summary = pipeline.process_document(
                paths, output_prefix="reconstructed"
            )
            outputs = summary.get("outputs", {})
            status = (
                f"Processed {summary['page_count']} page(s). "
                f"Outputs: {list(outputs.keys())}"
            )
            return (
                status,
                outputs.get("docx"),
                outputs.get("pdf"),
                outputs.get("json"),
            )
        except Exception as exc:
            logger.error(f"_process_files failed: {exc}")
            return f"Error: {exc}", None, None, None

    def build_interface(self) -> "gr.Blocks":
        """Create and return the Gradio Blocks interface.

        Returns:
            A configured :class:`gradio.Blocks` instance.

        Raises:
            ImportError: If gradio is not installed.
        """
        if not _GRADIO_AVAILABLE:
            raise ImportError(
                "gradio is not installed. "
                "Install it with: pip install gradio>=4.0.0"
            )
        with gr.Blocks(title="Document Reconstruction Engine") as demo:
            gr.Markdown("# AI-based Document Reconstruction Engine")
            gr.Markdown(
                "Upload one or more document images. "
                "The engine will preprocess, detect layout, run OCR, "
                "and produce DOCX/JSON outputs."
            )
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(
                        label="Upload Document Images",
                        file_count="multiple",
                        file_types=["image"],
                    )
                    enable_sr = gr.Checkbox(
                        label="Enable Super-Resolution",
                        value=False,
                    )
                    enable_semantic = gr.Checkbox(
                        label="Enable Semantic Correction",
                        value=False,
                    )
                    confidence_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.6,
                        step=0.05,
                        label="Confidence Threshold",
                    )
                    process_btn = gr.Button("Process Documents", variant="primary")
                with gr.Column():
                    status_text = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=3,
                    )
                    docx_output = gr.File(label="Download DOCX")
                    pdf_output = gr.File(label="Download PDF")
                    json_output = gr.File(label="Download JSON")

            process_btn.click(
                fn=self._process_files,
                inputs=[file_input, enable_sr, enable_semantic, confidence_slider],
                outputs=[status_text, docx_output, pdf_output, json_output],
            )

        self._interface = demo
        return demo

    def launch(self, **kwargs) -> None:
        """Build (if needed) and launch the Gradio interface.

        Args:
            **kwargs: Additional keyword arguments forwarded to
                :meth:`gradio.Blocks.launch`.
        """
        if self._interface is None:
            self.build_interface()
        logger.info("Launching Gradio interface")
        self._interface.launch(**kwargs)


def main() -> None:
    """Entry point: create and launch the Document Reconstruction UI."""
    logging.basicConfig(level=logging.INFO)
    ui = DocumentReconstructionUI()
    ui.launch(share=False)


if __name__ == "__main__":
    main()
