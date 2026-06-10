"""
Tests for docling-lightonocr.

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from docling_lightonocr.ocr_model import LightOnOcrModel, LightOnOcrOptions, _BBOX_RE
from docling_core.types.doc import BoundingBox, CoordOrigin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_options(**kwargs) -> LightOnOcrOptions:
    defaults = dict(
        api_base_url="http://localhost:8000",
        api_key="test-key",
        model_name="lightonai/LightOnOCR-2-1B-bbox-soup",
    )
    defaults.update(kwargs)
    return LightOnOcrOptions(**defaults)


def make_model(options: LightOnOcrOptions | None = None) -> LightOnOcrModel:
    opts = options or make_options()
    return LightOnOcrModel(
        enabled=True,
        artifacts_path=None,
        options=opts,
        accelerator_options=MagicMock(),
    )


def make_small_image() -> Image.Image:
    img = Image.new("RGB", (200, 100), color=(255, 255, 255))
    return img


def make_api_response(text: str) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": text,
                }
            }
        ]
    }


# ---------------------------------------------------------------------------
# Unit tests – options
# ---------------------------------------------------------------------------

class TestLightOnOcrOptions:
    def test_defaults(self):
        opts = LightOnOcrOptions()
        assert opts.api_base_url == "http://localhost:8000"
        assert opts.model_name == "lightonai/LightOnOCR-2-1B-bbox-soup"
        assert opts.max_tokens == 4096
        assert opts.strip_bbox_tokens is True
        assert opts.force_full_page_ocr is True

    def test_kind_discriminator(self):
        assert LightOnOcrOptions.kind == "lightonocr-remote"

    def test_custom_values(self):
        opts = make_options(temperature=0.5, dpi=200, timeout=60.0)
        assert opts.temperature == 0.5
        assert opts.dpi == 200
        assert opts.timeout == 60.0


# ---------------------------------------------------------------------------
# Unit tests – model initialisation
# ---------------------------------------------------------------------------

class TestLightOnOcrModelInit:
    def test_endpoint_construction(self):
        model = make_model(make_options(api_base_url="http://myserver:9000"))
        assert model._endpoint == "http://myserver:9000/v1/chat/completions"

    def test_endpoint_strips_trailing_slash(self):
        model = make_model(make_options(api_base_url="http://myserver:9000/"))
        assert model._endpoint == "http://myserver:9000/v1/chat/completions"

    def test_get_options_type(self):
        assert LightOnOcrModel.get_options_type() is LightOnOcrOptions


# ---------------------------------------------------------------------------
# Unit tests – image encoding
# ---------------------------------------------------------------------------

class TestImageEncoding:
    def test_b64_is_valid_png(self):
        model = make_model()
        img = make_small_image()
        data_uri = model._image_to_b64(img)
        assert data_uri.startswith("data:image/png;base64,")
        b64_part = data_uri.split(",", 1)[1]
        raw = base64.b64decode(b64_part)
        recovered = Image.open(BytesIO(raw))
        assert recovered.format == "PNG"
        assert recovered.size == (200, 100)


# ---------------------------------------------------------------------------
# Unit tests – API call with mocked requests
# ---------------------------------------------------------------------------

class TestApiCall:
    def test_successful_call(self):
        model = make_model()
        mock_response = MagicMock()
        mock_response.json.return_value = make_api_response("## Hello world\nSome text here.")
        mock_response.raise_for_status = MagicMock()

        with patch.object(model._session, "post", return_value=mock_response) as mock_post:
            text = model._call_api(make_small_image())

        assert text == "## Hello world\nSome text here."
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"]  # keyword argument
        assert payload["model"] == "lightonai/LightOnOCR-2-1B-bbox-soup"
        assert payload["messages"][0]["content"][0]["type"] == "image_url"
        assert payload["messages"][0]["content"][0]["image_url"]["url"].startswith(
            "data:image/png;base64,"
        )

    def test_retry_on_failure(self):
        model = make_model(make_options(max_retries=2, retry_delay=0.0))
        mock_ok = MagicMock()
        mock_ok.json.return_value = make_api_response("recovered")
        mock_ok.raise_for_status = MagicMock()

        call_count = 0

        def side_effect(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("timeout")
            return mock_ok

        with patch.object(model._session, "post", side_effect=side_effect):
            text = model._call_api(make_small_image())

        assert text == "recovered"
        assert call_count == 3

    def test_all_retries_exhausted_raises(self):
        model = make_model(make_options(max_retries=1, retry_delay=0.0))
        with patch.object(model._session, "post", side_effect=ConnectionError("fail")):
            with pytest.raises(RuntimeError, match="all retries"):
                model._call_api(make_small_image())


# ---------------------------------------------------------------------------
# Unit tests – text → cells mapping
# ---------------------------------------------------------------------------

class TestTextToCells:
    def _bbox(self) -> BoundingBox:
        return BoundingBox(l=10, t=20, r=210, b=120, coord_origin=CoordOrigin.TOPLEFT)

    def test_basic_lines(self):
        model = make_model()
        bbox = self._bbox()
        cells = model._text_to_cells("Line one\nLine two\nLine three", bbox, 300.0, 0)
        assert len(cells) == 3
        assert cells[0].text == "Line one"
        assert cells[1].text == "Line two"
        assert cells[2].text == "Line three"
        assert all(c.from_ocr for c in cells)

    def test_indices(self):
        model = make_model()
        cells = model._text_to_cells("A\nB\nC", self._bbox(), 300.0, start_idx=10)
        assert [c.index for c in cells] == [10, 11, 12]

    def test_bbox_tokens_stripped_by_default(self):
        model = make_model(make_options(strip_bbox_tokens=True))
        raw = "Some text\n<bbox>100,200,300,400</bbox>\nMore text"
        cells = model._text_to_cells(raw, self._bbox(), 300.0, 0)
        for c in cells:
            assert "<bbox>" not in c.text

    def test_bbox_tokens_kept_when_configured(self):
        model = make_model(make_options(strip_bbox_tokens=False))
        raw = "Some text\n<bbox>100,200,300,400</bbox>"
        cells = model._text_to_cells(raw, self._bbox(), 300.0, 0)
        combined = " ".join(c.text for c in cells)
        assert "<bbox>" in combined

    def test_empty_input_returns_no_cells(self):
        model = make_model()
        cells = model._text_to_cells("   \n\n  ", self._bbox(), 300.0, 0)
        assert cells == []

    def test_vertical_distribution(self):
        model = make_model()
        bbox = BoundingBox(l=0, t=0, r=100, b=90, coord_origin=CoordOrigin.TOPLEFT)
        cells = model._text_to_cells("A\nB\nC", bbox, 90.0, 0)
        # Each cell should occupy 30 units vertically
        for i, cell in enumerate(cells):
            assert abs(cell.bbox.t - i * 30) < 0.01
            assert abs(cell.bbox.b - (i + 1) * 30) < 0.01


# ---------------------------------------------------------------------------
# Unit tests – bbox regex
# ---------------------------------------------------------------------------

class TestBboxRegex:
    def test_parses_standard(self):
        m = _BBOX_RE.search("<bbox>100,200,300,400</bbox>")
        assert m is not None
        assert m.groups() == ("100", "200", "300", "400")

    def test_parses_with_spaces(self):
        m = _BBOX_RE.search("<bbox> 10 , 20 , 30 , 40 </bbox>")
        assert m is not None

    def test_no_match_on_plain_text(self):
        assert _BBOX_RE.search("no bounding box here") is None


# ---------------------------------------------------------------------------
# Integration smoke test (requires a live API – skipped by default)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Requires a running LightOnOCR-2 server")
def test_live_api_round_trip(tmp_path: Path):
    """
    End-to-end smoke test.  Set LIGHTONOCR_API_URL in your environment and
    run with ``pytest -k test_live_api_round_trip --no-header -s``.
    """
    import os

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    api_url = os.environ.get("LIGHTONOCR_API_URL", "http://localhost:8000")
    ocr_opts = LightOnOcrOptions(api_base_url=api_url)
    pipeline_opts = PdfPipelineOptions(
        do_ocr=True,
        ocr_options=ocr_opts,
        allow_external_plugins=True,
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)}
    )
    # Replace with a real PDF path
    result = converter.convert("tests/fixtures/sample.pdf")
    md = result.document.export_to_markdown()
    assert len(md) > 0
    print(md[:500])
