"""Options for the LightOnOCR Docling integration."""

from docling.datamodel.pipeline_options import OcrOptions


class LightOnOCROptions(OcrOptions):
    """Configuration options for LightOnOCR-2-1B bbox soup model."""

    kind: str = "lighton_ocr"

    # API connection
    api_base_url: str = "http://localhost:8000"
    api_key: str = "none"  # Set to your key if your server requires auth
    model_name: str = "LightOnOCR-2-1B"

    # Request settings
    max_tokens: int = 4096
    timeout: float = 60.0
    confidence_threshold: float = 0.0  # Min confidence to keep a cell (0 = keep all)

    # OCR scale factor (higher = better quality but slower)
    scale: float = 3.0
