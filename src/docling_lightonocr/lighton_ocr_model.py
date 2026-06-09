"""
LightOnOCR-2-1B-bbox-soup Docling OCR Plugin
=============================================
Integrates the LightOnOCR-2-1B-bbox-soup model (served via an
OpenAI-compatible API) into Docling's standard PDF pipeline.

The model receives a full-page image and returns Markdown text with
optional bounding boxes for embedded images in the format:
    ![image](image_N.png) x1,y1,x2,y2   (coordinates in [0, 1000])

This plugin:
1. Renders each page (or OCR-required region) to PNG at ~1540px longest edge.
2. Sends it as a base64-encoded image_url to the chat/completions endpoint.
3. Parses the Markdown response into Docling TextCell objects.
4. Optionally parses bbox annotations and stores them as page metadata.
"""

from __future__ import annotations

import base64
import io
import logging
import re
from pathlib import Path
from typing import Generator, Iterable, Optional

import requests
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import OcrOptions
from docling.models.base_ocr_model import BaseOcrModel
from docling_core.types.doc.page import BoundingRectangle, TextCell
from PIL import Image
from pydantic import Field

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

class LightOnOcrOptions(OcrOptions):
    """Configuration for the LightOnOCR remote API OCR engine."""

    kind: str = "lighton-ocr-remote"

    # API settings
    api_base_url: str = Field(
        default="http://localhost:8000",
        description="Base URL of the OpenAI-compatible server (no trailing slash).",
    )
    model_name: str = Field(
        default="lightonai/LightOnOCR-2-1B-bbox-soup",
        description="Model identifier to pass in the API request.",
    )
    api_key: str = Field(
        default="EMPTY",
        description="API key for the server (use 'EMPTY' for unauthenticated servers).",
    )

    # Generation settings (matching the model card recommendations)
    max_tokens: int = Field(default=4096, ge=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)

    # Image rendering
    render_dpi: int = Field(
        default=200,
        description=(
            "DPI used when rendering PDF pages to images before sending to the API. "
            "The model card recommends a longest dimension of ~1540 px."
        ),
    )
    render_max_side: int = Field(
        default=1540,
        description="Maximum side length (px) for the rendered page image.",
    )

    # Behaviour
    timeout: float = Field(default=120.0, description="HTTP request timeout in seconds.")
    parse_bboxes: bool = Field(
        default=True,
        description=(
            "If True, strip and log bbox annotations produced by the bbox-soup model "
            "instead of including them verbatim in the recognised text."
        ),
    )

    class Config:
        # allow extra fields so future API versions don't break existing configs
        extra = "ignore"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Matches:  ![image](image_N.png) x1,y1,x2,y2
_BBOX_RE = re.compile(
    r"!\[image]\(image_\d+\.(?:png|jpg|jpeg)\)\s+(\d+),(\d+),(\d+),(\d+)",
    re.IGNORECASE,
)


def _strip_bboxes(text: str) -> tuple[str, list[tuple[int, int, int, int]]]:
    """Remove bbox annotations from the model output and return them separately."""
    bboxes: list[tuple[int, int, int, int]] = []

    def _replacer(m: re.Match) -> str:
        bboxes.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))))
        return ""

    clean = _BBOX_RE.sub(_replacer, text).strip()
    return clean, bboxes


def _resize_image(img: Image.Image, max_side: int) -> Image.Image:
    """Resize *img* so its longest side equals *max_side*, preserving aspect ratio."""
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def _image_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_text_cells(text: str, page_w: int, page_h: int) -> list[TextCell]:
    """
    Convert the raw Markdown string from LightOnOCR into a list of TextCell objects.

    LightOnOCR returns the *full page* as a single Markdown blob — it doesn't
    produce per-word bounding boxes.  We split the output into non-empty lines
    and assign each line a synthetic bounding box that spans the full page
    width and a proportional height slice.  This gives downstream Docling
    models something to work with while preserving reading order.
    """
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return []

    cells: list[TextCell] = []
    line_height = max(1, page_h // max(len(lines), 1))

    for idx, line in enumerate(lines):
        y0 = idx * line_height
        y1 = min(y0 + line_height, page_h)
        cells.append(
            TextCell(
                text=line,
                bbox=BoundingRectangle(
                    r_x0=0,
                    r_y0=y0,
                    r_x1=page_w,
                    r_y1=y1,
                    coord_origin="TOPLEFT",
                ),
                from_ocr=True,
                confidence=1.0,
            )
        )
    return cells


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LightOnOcrModel(BaseOcrModel):
    """
    Docling OCR engine that delegates recognition to a remote
    LightOnOCR-2-1B-bbox-soup model via an OpenAI-compatible API.
    """

    options: LightOnOcrOptions

    def __init__(
        self,
        *,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: LightOnOcrOptions,
        accelerator_options: AcceleratorOptions,
    ) -> None:
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {options.api_key}",
                "Content-Type": "application/json",
            }
        )
        self._endpoint = f"{options.api_base_url.rstrip('/')}/v1/chat/completions"
        _log.info(
            "LightOnOcrModel initialised — endpoint=%s  model=%s",
            self._endpoint,
            options.model_name,
        )

    # ------------------------------------------------------------------
    # Core interface expected by BaseOcrModel
    # ------------------------------------------------------------------

    def __call__(
        self,
        conv_res: ConversionResult,
        page_batch: Iterable[Page],
    ) -> Generator[Page, None, None]:
        for page in page_batch:
            if not self.enabled or page._backend is None:
                yield page
                continue

            ocr_rects = self.get_ocr_rects(page)

            if not ocr_rects:
                yield page
                continue

            all_cells: list[TextCell] = []

            for ocr_rect in ocr_rects:
                # --- 1. Render the region to an image ---
                try:
                    scale = self.options.render_dpi / 72.0  # PDF default 72 DPI
                    pil_image: Image.Image = page.get_image(
                        scale=scale, cropbox=ocr_rect
                    )
                except Exception:
                    # Fallback: render the full page
                    _log.warning(
                        "Could not render cropped region on page %d; "
                        "falling back to full-page render.",
                        page.page_no,
                    )
                    pil_image = page.get_image(scale=scale)

                pil_image = _resize_image(pil_image, self.options.render_max_side)
                img_b64 = _image_to_base64_png(pil_image)
                page_w, page_h = pil_image.size

                # --- 2. Call the API ---
                raw_text = self._call_api(img_b64)
                if raw_text is None:
                    continue

                # --- 3. Parse bbox annotations ---
                if self.options.parse_bboxes:
                    clean_text, bboxes = _strip_bboxes(raw_text)
                    if bboxes:
                        _log.debug(
                            "Page %d: found %d embedded-image bbox(es): %s",
                            page.page_no,
                            len(bboxes),
                            bboxes,
                        )
                        # Store on the page for downstream consumers (optional)
                        if not hasattr(page, "extra"):
                            page.extra = {}  # type: ignore[attr-defined]
                        page.extra["lighton_image_bboxes"] = bboxes  # type: ignore[attr-defined]
                else:
                    clean_text = raw_text

                # --- 4. Build TextCells ---
                cells = _build_text_cells(clean_text, page_w, page_h)
                all_cells.extend(cells)

            if all_cells:
                page.cells = self.post_process_cells(all_cells, page)

            yield page

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_api(self, image_b64: str) -> Optional[str]:
        """Send the base64 image to the LightOnOCR endpoint and return the text."""
        payload = {
            "model": self.options.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            },
                        }
                    ],
                }
            ],
            "max_tokens": self.options.max_tokens,
            "temperature": self.options.temperature,
            "top_p": self.options.top_p,
        }

        try:
            resp = self._session.post(
                self._endpoint,
                json=payload,
                timeout=self.options.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return text
        except requests.exceptions.Timeout:
            _log.error("LightOnOCR API request timed out (timeout=%.1fs).", self.options.timeout)
        except requests.exceptions.HTTPError as exc:
            _log.error("LightOnOCR API HTTP error: %s", exc)
        except (KeyError, IndexError, ValueError) as exc:
            _log.error("LightOnOCR API unexpected response format: %s", exc)
        except Exception as exc:  # noqa: BLE001
            _log.error("LightOnOCR API unexpected error: %s", exc)

        return None