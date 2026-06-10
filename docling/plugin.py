"""
Docling plugin registration for the LightOnOCR-2 remote engine.

This module is discovered by Docling's pluggy-based plugin loader via the
``[project.entry-points."docling"]`` section of ``pyproject.toml``.

Docling calls the ``ocr_engines()`` function to obtain the list of OCR model
classes that this package contributes.
"""

from docling_lightonocr.ocr_model import LightOnOcrModel


def ocr_engines() -> dict[str, list]:
    """Return the OCR engine classes contributed by this plugin."""
    return {
        "ocr_engines": [
            LightOnOcrModel,
        ]
    }
