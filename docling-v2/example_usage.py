"""Example: convert a PDF with LightOnOCR-2-1B as the OCR backend.

Run:
    python example_usage.py path/to/document.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# Local modules (must be on PYTHONPATH or in the same directory)
from lighton_ocr_model import LightOnOCRModel
from lighton_ocr_options import LightOnOCROptions


def build_converter(
    api_base_url: str = "http://localhost:8000",
    api_key: str = "none",
    model_name: str = "LightOnOCR-2-1B",
    scale: float = 3.0,
    confidence_threshold: float = 0.0,
) -> DocumentConverter:
    """Create a Docling DocumentConverter wired to the LightOnOCR endpoint."""

    options = LightOnOCROptions(
        api_base_url=api_base_url,
        api_key=api_key,
        model_name=model_name,
        scale=scale,
        confidence_threshold=confidence_threshold,
    )

    pipeline_options = PdfPipelineOptions()
    # Disable the default OCR model; we supply our own.
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = options

    converter = DocumentConverter(
        format_options={
            # PdfFormatOption wires the pipeline options (incl. our OCR model)
            # into the PDF conversion pipeline.
            "pdf": PdfFormatOption(pipeline_options=pipeline_options),
        }
    )

    # ---------------------------------------------------------------
    # Monkey-patch: replace whatever OCR model Docling instantiated
    # with our LightOnOCRModel.  This is necessary because Docling's
    # factory doesn't know about third-party OCR classes yet.
    # ---------------------------------------------------------------
    pdf_pipeline = converter.get_pipeline_for("pdf")  # type: ignore[attr-defined]
    pdf_pipeline.ocr_model = LightOnOCRModel(enabled=True, options=options)

    return converter


def main(pdf_path: str) -> None:
    converter = build_converter()

    result = converter.convert(pdf_path)
    doc = result.document

    print("=== Converted document (markdown preview) ===\n")
    print(doc.export_to_markdown())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python example_usage.py <path/to/document.pdf>")
        sys.exit(1)
    main(sys.argv[1])
