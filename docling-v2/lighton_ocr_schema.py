"""Pydantic models for structured output parsing from LightOnOCR-2-1B bbox soup model.

The model returns a list of detected text regions, each with:
  - bbox: [x_min, y_min, x_max, y_max] in pixel coordinates (top-left origin)
  - text: the recognised string
  - confidence: float in [0, 1]
"""

from typing import List
from pydantic import BaseModel, Field


class OCRBox(BaseModel):
    """A single detected text region."""

    bbox: List[float] = Field(
        ...,
        description="Bounding box as [x_min, y_min, x_max, y_max] in pixel coords.",
        min_length=4,
        max_length=4,
    )
    text: str = Field(..., description="Recognised text for this region.")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score in [0, 1].",
    )


class OCRResponse(BaseModel):
    """Full structured response from the LightOnOCR model."""

    results: List[OCRBox] = Field(
        default_factory=list,
        description="All detected text boxes on the image.",
    )
