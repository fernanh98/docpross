"""
Docling OCR plugin for LightOnOCR-2-1B-bbox-soup via an OpenAI-compatible API.

The model accepts a base64-encoded page image and returns Markdown text (with
optional normalised bounding boxes for embedded figures).  This plugin feeds
each page-crop that Docling identifies as needing OCR to the remote endpoint
and maps the response back to Docling's TextCell / BoundingBox data model.
"""

from __future__ import annotations

import base64
import io
import logging
import re
import time
from pathlib import Path
from typing import ClassVar, Iterable

import requests
from PIL import Image
from pydantic import Field

from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import AcceleratorOptions, OcrOptions
from docling.models.base_ocr_model import BaseOcrModel
from docling_core.types.doc import BoundingBox, CoordOrigin, TextCell

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex for the bbox tokens that the bbox-soup variant may emit.
# Format: <bbox>x1,y1,x2,y2</bbox>  (coords normalised to [0, 1000])
# ---------------------------------------------------------------------------
_BBOX_RE = re.compile(r"<bbox>\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*</bbox>")


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

class LightOnOcrOptions(OcrOptions):
    """Configuration for the LightOnOCR-2-1B remote OCR engine."""

    kind: ClassVar[str] = "lightonocr-remote"

    # --- connection ----------------------------------------------------------
    api_base_url: str = Field(
        default="http://localhost:8000",
        description=(
            "Base URL of the OpenAI-compatible server (no trailing slash). "
            "The plugin will POST to <api_base_url>/v1/chat/completions."
        ),
    )
    api_key: str = Field(
        default="EMPTY",
        description="Bearer token / API key.  Use 'EMPTY' for unauthenticated servers.",
    )
    model_name: str = Field(
        default="lightonai/LightOnOCR-2-1B-bbox-soup",
        description="Model string forwarded in the request body.",
    )

    # --- generation ----------------------------------------------------------
    max_tokens: int = Field(default=4096, ge=64, le=16384)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)

    # --- behaviour -----------------------------------------------------------
    force_full_page_ocr: bool = Field(
        default=True,
        description=(
            "Send the full page image rather than individual crop regions. "
            "Recommended for LightOnOCR because it is an end-to-end model "
            "trained on full pages."
        ),
    )
    dpi: int = Field(
        default=150,
        description="DPI used when rasterising a page crop to PNG before sending.",
    )
    timeout: float = Field(
        default=120.0,
        description="HTTP request timeout in seconds.",
    )
    strip_bbox_tokens: bool = Field(
        default=True,
        description=(
            "Remove <bbox>…</bbox> tokens from the returned text "
            "so that downstream Docling stages only see clean Markdown."
        ),
    )
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay: float = Field(default=2.0, ge=0.0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LightOnOcrModel(BaseOcrModel):
    """Docling OCR model that delegates to a remote LightOnOCR-2 endpoint."""

    def __init__(
        self,
        *,
        enabled: bool,
        artifacts_path: Path | None,
        options: LightOnOcrOptions,
        accelerator_options: AcceleratorOptions,
    ) -> None:
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: LightOnOcrOptions = options  # re-bind for type narrowing
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {options.api_key}",
                "Content-Type": "application/json",
            }
        )
        self._endpoint = options.api_base_url.rstrip("/") + "/v1/chat/completions"
        _log.info(
            "LightOnOcrModel initialised – endpoint=%s  model=%s",
            self._endpoint,
            options.model_name,
        )

    @classmethod
    def get_options_type(cls) -> type[LightOnOcrOptions]:
        return LightOnOcrOptions

    # ------------------------------------------------------------------
    # Public interface required by BaseOcrModel / BasePageModel
    # ------------------------------------------------------------------

    def __call__(
        self,
        conv_res: ConversionResult,
        page_batch: Iterable[Page],
    ) -> Iterable[Page]:
        for page in page_batch:
            if not self.enabled or page._backend is None:
                yield page
                continue

            # Ask the base class which rectangles need OCR.
            ocr_rects = self.get_ocr_rects(page)
            if not ocr_rects:
                yield page
                continue

            all_cells: list[TextCell] = []
            cell_idx = 0

            for rect in ocr_rects:
                # Render the page crop to a PIL image.
                try:
                    pil_img = self._crop_page(page, rect)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("Could not render page crop: %s", exc)
                    continue

                # Call the remote OCR endpoint.
                try:
                    raw_text = self._call_api(pil_img)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("LightOnOCR API call failed: %s", exc)
                    continue

                # Parse the response into TextCell objects.
                cells = self._text_to_cells(
                    raw_text,
                    bbox=rect,
                    page_height=page.size.height if page.size else 0.0,
                    start_idx=cell_idx,
                )
                cell_idx += len(cells)
                all_cells.extend(cells)

            # Merge OCR cells with native PDF cells.
            if all_cells:
                page.cells = self.post_process_cells(all_cells, page.cells)

            yield page

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _crop_page(self, page: Page, bbox: BoundingBox) -> Image.Image:
        """Return a PIL image for the given page bounding-box region."""
        scale = self.options.dpi / 72.0  # docling uses 72-dpi internal coords
        pil_img = page._backend.get_page_image(
            scale=scale,
            cropbox=bbox,
        )
        return pil_img

    def _image_to_b64(self, img: Image.Image) -> str:
        """Encode a PIL image as a base64 PNG data URI."""
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def _call_api(self, img: Image.Image) -> str:
        """Send the image to the remote endpoint and return the raw text."""
        data_uri = self._image_to_b64(img)
        payload = {
            "model": self.options.model_name,
            "max_tokens": self.options.max_tokens,
            "temperature": self.options.temperature,
            "top_p": self.options.top_p,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri},
                        }
                    ],
                }
            ],
        }

        last_exc: Exception | None = None
        for attempt in range(1, self.options.max_retries + 2):
            try:
                resp = self._session.post(
                    self._endpoint,
                    json=payload,
                    timeout=self.options.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt <= self.options.max_retries:
                    _log.warning(
                        "LightOnOCR API attempt %d/%d failed: %s – retrying in %.1fs",
                        attempt,
                        self.options.max_retries + 1,
                        exc,
                        self.options.retry_delay,
                    )
                    time.sleep(self.options.retry_delay)

        raise RuntimeError(f"LightOnOCR API failed after all retries: {last_exc}") from last_exc

    def _text_to_cells(
        self,
        raw_text: str,
        bbox: BoundingBox,
        page_height: float,
        start_idx: int,
    ) -> list[TextCell]:
        """
        Convert the model's Markdown output into Docling TextCell objects.

        Strategy
        --------
        LightOnOCR-2-bbox-soup returns a mix of plain Markdown text and optional
        <bbox>x1,y1,x2,y2</bbox> tokens for embedded figure locations.

        We produce one TextCell per non-empty line of clean text, distributing
        the cells evenly across the page crop vertically.  If the strip_bbox_tokens
        option is False, the raw bbox tokens are preserved in the text.
        """
        if self.options.strip_bbox_tokens:
            clean_text = _BBOX_RE.sub("", raw_text).strip()
        else:
            clean_text = raw_text.strip()

        lines = [ln for ln in clean_text.splitlines() if ln.strip()]
        if not lines:
            return []

        # The page crop occupies [bbox.l, bbox.r] × [bbox.t, bbox.b] in page coords.
        crop_w = bbox.r - bbox.l
        crop_h = bbox.b - bbox.t
        line_h = crop_h / len(lines) if lines else crop_h

        cells: list[TextCell] = []
        for i, line in enumerate(lines):
            y_top = bbox.t + i * line_h
            y_bot = y_top + line_h
            cell_bbox = BoundingBox(
                l=bbox.l,
                t=y_top,
                r=bbox.l + crop_w,
                b=y_bot,
                coord_origin=CoordOrigin.TOPLEFT,
            )
            cells.append(
                TextCell(
                    index=start_idx + i,
                    text=line,
                    orig=line,
                    from_ocr=True,
                    bbox=cell_bbox,
                )
            )

        return cells
