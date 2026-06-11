"""Docling OCR backend for LightOnOCR-2-1B (bbox soup) served via an OpenAI-compatible API.

Drop-in replacement for EasyOCRModel / RapidOCRModel.  The model is called
through ``OpenAI().beta.chat.completions.parse()``, which gives us automatic
structured-output parsing via the ``OCRResponse`` Pydantic schema.

Usage
-----
Instantiate ``LightOnOCRModel`` and pass it as the ``ocr_model`` argument to
``PdfPipelineOptions`` (see ``example_usage.py``).
"""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import Iterable

import numpy
from openai import OpenAI
from PIL import Image

from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.models.base_ocr_model import BaseOcrModel
from docling.datamodel.base_models import BoundingBox, BoundingRectangle, CoordOrigin, TextCell
from docling.utils.profiling import TimeRecorder
from docling.datamodel.settings import settings

from lighton_ocr_options import LightOnOCROptions
from lighton_ocr_schema import OCRResponse

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt sent to the model
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are an OCR engine. "
    "Given an image, detect all text regions and return them as structured JSON "
    "matching the OCRResponse schema. "
    "Each entry must contain 'bbox' ([x_min, y_min, x_max, y_max] in pixels, "
    "top-left origin), 'text' (the recognised string), and 'confidence' (float 0-1). "
    "Return every visible text region, preserving reading order where possible. "
    "Do NOT include any prose or explanation outside the JSON."
)

_USER_PROMPT = (
    "Please perform OCR on the attached image and return all detected text "
    "regions in the required JSON format."
)


def _pil_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image as a base64 string."""
    buf = BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class LightOnOCRModel(BaseOcrModel):
    """Docling OCR model backed by a LightOnOCR-2-1B endpoint."""

    def __init__(self, enabled: bool, options: LightOnOCROptions) -> None:
        super().__init__(enabled=enabled, options=options)
        self.options: LightOnOCROptions = options
        self.scale = options.scale

        if not enabled:
            return

        # Build the OpenAI-compatible client once and reuse it.
        self._client = OpenAI(
            base_url=options.api_base_url,
            api_key=options.api_key,
            timeout=options.timeout,
        )
        _log.info(
            "LightOnOCRModel initialised – endpoint: %s  model: %s",
            options.api_base_url,
            options.model_name,
        )

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _call_api(self, image: Image.Image) -> OCRResponse | None:
        """Send *image* to the LightOnOCR API and return a parsed ``OCRResponse``.

        Returns ``None`` on any recoverable error so callers can skip gracefully.
        """
        b64 = _pil_to_base64(image)
        media_type = "image/png"

        try:
            completion = self._client.beta.chat.completions.parse(
                model=self.options.model_name,
                max_tokens=self.options.max_tokens,
                response_format=OCRResponse,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{b64}",
                                    "detail": "high",
                                },
                            },
                            {"type": "text", "text": _USER_PROMPT},
                        ],
                    },
                ],
            )
        except Exception as exc:  # noqa: BLE001
            _log.error("LightOnOCR API call failed: %s", exc)
            return None

        parsed: OCRResponse | None = completion.choices[0].message.parsed
        if parsed is None:
            _log.warning(
                "LightOnOCR returned an unparseable response: %s",
                completion.choices[0].message.content,
            )
        return parsed

    # ------------------------------------------------------------------
    # Docling pipeline interface
    # ------------------------------------------------------------------

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
                continue

            with TimeRecorder(conv_res, "ocr"):
                ocr_rects = self.get_ocr_rects(page)
                all_ocr_cells: list[TextCell] = []

                for ocr_rect in ocr_rects:
                    if ocr_rect.area() == 0:
                        continue

                    # Render a high-resolution crop of the current OCR region.
                    high_res_image: Image.Image = page._backend.get_page_image(
                        scale=self.scale, cropbox=ocr_rect
                    )

                    ocr_response = self._call_api(high_res_image)

                    del high_res_image  # release memory early

                    if ocr_response is None or not ocr_response.results:
                        _log.warning(
                            "LightOnOCR returned empty result for rect %s", ocr_rect
                        )
                        continue

                    cells = [
                        TextCell(
                            index=ix,
                            text=box.text,
                            orig=box.text,
                            from_ocr=True,
                            confidence=box.confidence,
                            rect=BoundingRectangle.from_bounding_box(
                                BoundingBox.from_tuple(
                                    coord=(
                                        # Convert from crop-local pixels → page coords
                                        (box.bbox[0] / self.scale) + ocr_rect.l,
                                        (box.bbox[1] / self.scale) + ocr_rect.t,
                                        (box.bbox[2] / self.scale) + ocr_rect.l,
                                        (box.bbox[3] / self.scale) + ocr_rect.t,
                                    ),
                                    origin=CoordOrigin.TOPLEFT,
                                )
                            ),
                        )
                        for ix, box in enumerate(ocr_response.results)
                        if box.confidence >= self.options.confidence_threshold
                    ]
                    all_ocr_cells.extend(cells)

                # Let Docling handle NMS / deduplication / ordering.
                self.post_process_cells(all_ocr_cells, page)

                # Optional debug visualisation (enabled via settings).
                if settings.debug.visualize_ocr:
                    self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

            yield page
