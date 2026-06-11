"""Docling OCR backend for LightOnOCR-2-1B (bbox soup) served via an OpenAI-compatible API.

Drop-in replacement for EasyOCRModel / RapidOCRModel.

The model is called via ``OpenAI().chat.completions.create()``; its raw text
output is then parsed with ``lighton_ocr_parser.parse_lighton_output()``.

Output format (per the LightOnOCR docs):

    recognised text
    ![image](image_N.png) x1,y1,x2,y2

Coordinates are normalised to [0, 1000] with top-left origin.
"""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import Iterable

from openai import OpenAI
from PIL import Image

from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.models.base_ocr_model import BaseOcrModel
from docling.datamodel.base_models import BoundingBox, BoundingRectangle, CoordOrigin, TextCell
from docling.utils.profiling import TimeRecorder
from docling.datamodel.settings import settings

from lighton_ocr_options import LightOnOCROptions
from lighton_ocr_parser import parse_lighton_output

_log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an OCR engine. "
    "For each text region in the image output exactly two lines: "
    "first the recognised text, then the bounding box in the format "
    "![image](image_N.png) x1,y1,x2,y2 where coordinates are normalised "
    "to [0, 1000] with top-left origin and N is the 0-based region index. "
    "Do not add any other commentary."
)

_USER_PROMPT = "Perform OCR on this image."


def _pil_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_api(self, image: Image.Image) -> str | None:
        """Send *image* to the API and return the raw text response.

        Returns ``None`` on any recoverable error.
        """
        b64 = _pil_to_base64(image)

        try:
            completion = self._client.chat.completions.create(
                model=self.options.model_name,
                max_tokens=self.options.max_tokens,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64}",
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

        return completion.choices[0].message.content

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

                    high_res_image: Image.Image = page._backend.get_page_image(
                        scale=self.scale, cropbox=ocr_rect
                    )
                    img_w = high_res_image.width
                    img_h = high_res_image.height

                    raw_output = self._call_api(high_res_image)
                    del high_res_image

                    if not raw_output:
                        _log.warning("LightOnOCR returned empty result for rect %s", ocr_rect)
                        continue

                    boxes = parse_lighton_output(raw_output)

                    if not boxes:
                        _log.warning("No boxes parsed from LightOnOCR output for rect %s", ocr_rect)
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
                                        # Normalised [0,1] → pixel → page coords
                                        (box.x1 * img_w / self.scale) + ocr_rect.l,
                                        (box.y1 * img_h / self.scale) + ocr_rect.t,
                                        (box.x2 * img_w / self.scale) + ocr_rect.l,
                                        (box.y2 * img_h / self.scale) + ocr_rect.t,
                                    ),
                                    origin=CoordOrigin.TOPLEFT,
                                )
                            ),
                        )
                        for ix, box in enumerate(boxes)
                        if box.confidence >= self.options.confidence_threshold
                    ]
                    all_ocr_cells.extend(cells)

                self.post_process_cells(all_ocr_cells, page)

                if settings.debug.visualize_ocr:
                    self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

            yield page
