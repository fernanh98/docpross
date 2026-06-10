# docling-lightonocr

A [Docling](https://github.com/docling-project/docling) OCR plugin that replaces the built-in OCR stage with a call to a remote **LightOnOCR-2-1B-bbox-soup** model served behind an OpenAI-compatible API (e.g. vLLM, IONOS AI Model Hub).

---

## How it works

```
Document (PDF / image)
        │
        ▼
Docling StandardPdfPipeline
        │  layout detection, table structure, …
        │
        ▼  (OCR stage)
LightOnOcrModel  ◄──── this plugin
        │
        │  POST /v1/chat/completions
        │  { "model": "lightonai/LightOnOCR-2-1B-bbox-soup",
        │    "messages": [{ "role": "user",
        │                   "content": [{ "type": "image_url",
        │                                 "image_url": { "url": "data:image/png;base64,…" } }] }] }
        ▼
Remote LightOnOCR-2 server
        │  returns Markdown text (+ optional <bbox>…</bbox> tokens)
        ▼
Docling TextCell objects  →  merged back into the document
```

---

## Installation

```bash
# 1. Serve the model (example with vLLM)
vllm serve lightonai/LightOnOCR-2-1B-bbox-soup \
    --limit-mm-per-prompt '{"image": 1}' \
    --mm-processor-cache-gb 0 \
    --no-enable-prefix-caching \
    --port 8000

# 2. Install this plugin (editable for development)
pip install -e ".[dev]"

# If you get a docling version conflict, pin it:
pip install -e . "docling>=2.28.0"
```

> **Important:** Docling requires `allow_external_plugins=True` in the pipeline options whenever a third-party (non-bundled) plugin is used.  The `convert.py` script and the code examples below already set this flag.

---

## Quick start

### Python API

```python
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from docling_lightonocr import LightOnOcrOptions

ocr_opts = LightOnOcrOptions(
    api_base_url="http://localhost:8000",   # your vLLM / IONOS endpoint
    api_key="EMPTY",                         # or your real API key
    model_name="lightonai/LightOnOCR-2-1B-bbox-soup",
)

pipeline_opts = PdfPipelineOptions(
    do_ocr=True,
    ocr_options=ocr_opts,
    allow_external_plugins=True,            # required for third-party plugins
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts),
    }
)

result = converter.convert("my_document.pdf")
print(result.document.export_to_markdown())
```

### Command-line script

```bash
# Basic usage
python convert.py document.pdf --api-url http://localhost:8000

# Save output to a file
python convert.py scan.pdf --api-url http://localhost:8000 --output result.md

# JSON output
python convert.py document.pdf --format json --output result.json

# Keep <bbox>…</bbox> tokens (for downstream layout analysis)
python convert.py document.pdf --keep-bbox-tokens

# Full option list
python convert.py --help
```

---

## Configuration reference

| Option | Default | Description |
|---|---|---|
| `api_base_url` | `http://localhost:8000` | Base URL of the OpenAI-compatible server |
| `api_key` | `"EMPTY"` | Bearer token / API key |
| `model_name` | `lightonai/LightOnOCR-2-1B-bbox-soup` | Model string in request body |
| `max_tokens` | `4096` | Max generation tokens per page |
| `temperature` | `0.2` | Sampling temperature |
| `top_p` | `0.9` | Top-p sampling |
| `force_full_page_ocr` | `True` | Send full page instead of individual crops |
| `dpi` | `150` | Rasterisation DPI for page crops |
| `timeout` | `120.0` | HTTP timeout in seconds |
| `strip_bbox_tokens` | `True` | Remove `<bbox>…</bbox>` tokens from OCR text |
| `max_retries` | `3` | Retries on API failure |
| `retry_delay` | `2.0` | Seconds between retries |

---

## Running tests

```bash
pytest tests/ -v

# Including the live end-to-end test (requires a running server)
LIGHTONOCR_API_URL=http://localhost:8000 \
pytest tests/ -v -k test_live_api_round_trip --no-header -s
```

---

## Project layout

```
docling_lightonocr/
├── __init__.py          # public re-exports
├── ocr_model.py         # LightOnOcrModel + LightOnOcrOptions
└── plugin.py            # Docling plugin registration (ocr_engines())

convert.py               # ready-to-use CLI conversion script
tests/
└── test_lightonocr.py   # unit tests (no server required)
pyproject.toml           # package metadata + entry-point declaration
README.md
```

---

## Notes on the bbox-soup variant

The `LightOnOCR-2-1B-bbox-soup` checkpoint is a task-arithmetic merge of the OCR and bbox models.  It may emit lines like:

```
Some paragraph text.
<bbox>142,310,687,412</bbox>
More paragraph text.
```

The coordinates are normalised to `[0, 1000]`.  By default, the plugin strips these tokens (`strip_bbox_tokens=True`) so that Docling only sees clean Markdown.  Set `strip_bbox_tokens=False` if your downstream pipeline needs figure-location hints.
