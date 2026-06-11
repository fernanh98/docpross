# LightOnOCR-2-1B × Docling Integration

Drop-in Docling OCR backend for the **LightOnOCR-2-1B bbox soup** model,
called via an OpenAI-compatible API using
`OpenAI().beta.chat.completions.parse()` for automatic structured-output
parsing.

---

## File Overview

| File | Purpose |
|------|---------|
| `lighton_ocr_options.py` | `LightOnOCROptions` – all tuneable knobs (URL, key, scale, …) |
| `lighton_ocr_parser.py` | Parses the model's raw text output into `OCRBox` objects |
| `lighton_ocr_model.py` | `LightOnOCRModel` – the Docling-compatible OCR class |
| `example_usage.py` | End-to-end example: convert a PDF and print markdown |

---

## Requirements

```
docling>=2.95.0
openai>=1.0.0
pillow
```

Install:
```bash
pip install "docling>=2.95.0" "openai>=1.0.0" pillow
```

---

## Quick Start

### 1 – Place all four files in the same directory (or on `PYTHONPATH`).

### 2 – Start your LightOnOCR-2-1B server
Make sure it is reachable and exposes the `v1/chat/completions` endpoint.

### 3 – Convert a document

```python
from example_usage import build_converter

converter = build_converter(
    api_base_url="http://localhost:8000",   # your server
    api_key="none",                         # or your real key
    model_name="LightOnOCR-2-1B",
    scale=3.0,                              # rendering DPI multiplier
    confidence_threshold=0.0,              # drop boxes below this score
)

result = converter.convert("invoice.pdf")
print(result.document.export_to_markdown())
```

Or from the command line:
```bash
python example_usage.py invoice.pdf
```

---

## How it Works

```
Docling pipeline
  └─ LightOnOCRModel.__call__(page_batch)
       ├─ get_ocr_rects(page)            # Docling decides where OCR is needed
       ├─ page._backend.get_page_image() # render high-res crop (scale × DPI)
       ├─ _call_api(image)
       │    └─ OpenAI().chat.completions.create(
       │         model    = "LightOnOCR-2-1B",
       │         messages = [system + image + prompt]
       │       )  →  raw text
       ├─ parse_lighton_output(raw_text)
       │    parses lines of the form:
       │      recognised text
       │      ![image](image_N.png) x1,y1,x2,y2   ← coords in [0, 1000]
       │    into OCRBox(text, x1, y1, x2, y2)
       ├─ map normalised coords → Docling TextCell page coords
       └─ post_process_cells()           # Docling NMS / ordering
```

### Coordinate mapping

The model outputs coordinates normalised to `[0, 1000]`.
`LightOnOCRModel` converts them to full-page coordinates in two steps:

```
# Step 1: normalised → crop pixels
pixel_x = (coord_x / 1000) * crop_width_px

# Step 2: crop pixels → page space (undo scale + add crop offset)
page_x = (pixel_x / scale) + ocr_rect.left
```

---

## Options Reference

| Option | Default | Description |
|--------|---------|-------------|
| `api_base_url` | `http://localhost:8000` | Base URL of your OpenAI-compatible server |
| `api_key` | `"none"` | API key (set if your server requires one) |
| `model_name` | `"LightOnOCR-2-1B"` | Model identifier passed to the API |
| `max_tokens` | `4096` | Max tokens for the completion |
| `timeout` | `60.0` | HTTP timeout in seconds |
| `confidence_threshold` | `0.0` | Discard OCR boxes below this confidence |
| `scale` | `3.0` | Image upscaling factor before sending to the API |

---

## Troubleshooting

**`AttributeError: 'DocumentConverter' object has no attribute 'get_pipeline_for'`**  
Docling's internal API can vary between patch versions.  
In that case, access the pipeline directly:
```python
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
# … and assign ocr_model on the pipeline instance you hold a reference to.
```

**Empty OCR results**  
- Check that the server is reachable from your machine.  
- Lower `confidence_threshold` to `0.0` to keep all detections.  
- Increase `scale` (e.g. `4.0`) for low-resolution documents.

**`openai.BadRequestError`**  
Ensure your server accepts `response_format` / structured-output requests
and base64-encoded inline images.
