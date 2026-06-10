#!/usr/bin/env python3
"""
convert.py – Convert documents with Docling using the LightOnOCR-2 remote engine.

Usage examples
--------------
# Basic conversion (PDF → Markdown printed to stdout)
python convert.py document.pdf

# Specify a custom API endpoint and output file
python convert.py scan.pdf \
    --api-url http://192.168.1.10:8000 \
    --output result.md

# Convert an image file, force full-page OCR, save as JSON
python convert.py photo.png \
    --api-url http://localhost:8000 \
    --model lightonai/LightOnOCR-2-1B-bbox-soup \
    --format json \
    --output result.json

# Keep <bbox>…</bbox> tokens in the output
python convert.py document.pdf --keep-bbox-tokens
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from docling_lightonocr import LightOnOcrOptions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
)
_log = logging.getLogger("lightonocr.convert")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert a document with Docling + LightOnOCR-2 remote OCR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("source", help="Path to the document to convert (PDF, PNG, JPEG, TIFF, …)")
    p.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the OpenAI-compatible server (no trailing slash).",
    )
    p.add_argument(
        "--api-key",
        default="EMPTY",
        help="Bearer token / API key for the remote endpoint.",
    )
    p.add_argument(
        "--model",
        default="lightonai/LightOnOCR-2-1B-bbox-soup",
        help="Model name forwarded in the API request body.",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate per page.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI used when rasterising page crops.",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds.",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Number of retries on API failure.",
    )
    p.add_argument(
        "--keep-bbox-tokens",
        action="store_true",
        help="Do NOT strip <bbox>…</bbox> tokens from the OCR output.",
    )
    p.add_argument(
        "--format",
        choices=["markdown", "json", "text"],
        default="markdown",
        help="Output format.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Write output to this file (default: print to stdout).",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return p


# ---------------------------------------------------------------------------
# Converter factory
# ---------------------------------------------------------------------------

def build_converter(args: argparse.Namespace) -> DocumentConverter:
    """Construct a DocumentConverter configured with the LightOnOCR plugin."""
    ocr_opts = LightOnOcrOptions(
        api_base_url=args.api_url,
        api_key=args.api_key,
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        dpi=args.dpi,
        timeout=args.timeout,
        max_retries=args.max_retries,
        strip_bbox_tokens=not args.keep_bbox_tokens,
        force_full_page_ocr=True,
    )

    pipeline_opts = PdfPipelineOptions(
        do_ocr=True,
        ocr_options=ocr_opts,
        # Required so Docling loads third-party (non-bundled) plugins.
        allow_external_plugins=True,
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts),
        }
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    source = Path(args.source)
    if not source.exists():
        _log.error("File not found: %s", source)
        sys.exit(1)

    _log.info("Converting %s …", source)
    converter = build_converter(args)

    try:
        result = converter.convert(str(source))
    except Exception as exc:
        _log.error("Conversion failed: %s", exc, exc_info=args.verbose)
        sys.exit(2)

    # Serialise output
    doc = result.document
    if args.format == "markdown":
        output_text = doc.export_to_markdown()
    elif args.format == "json":
        output_text = json.dumps(doc.export_to_dict(), indent=2, ensure_ascii=False)
    else:  # text
        output_text = doc.export_to_text()

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(output_text, encoding="utf-8")
        _log.info("Written to %s", out_path)
    else:
        print(output_text)


if __name__ == "__main__":
    main()
