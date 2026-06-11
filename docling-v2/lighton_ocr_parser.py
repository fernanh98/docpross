"""Parser for LightOnOCR-2-1B bbox soup raw text output.

The model outputs one line per detected text region in this format:

    ![image](image_N.png) x1,y1,x2,y2

Where:
  - N          : 0-based region index (ignored; we use enumeration order)
  - x1,y1,x2,y2: bounding box corners normalised to [0, 1000]
                  top-left origin, i.e. (x1,y1) = top-left, (x2,y2) = bottom-right

The text content for each region is whatever follows on the same line after
the coordinate token, but in the bbox-soup variant the model interleaves text
lines and bbox lines.  The canonical pattern is:

    recognised text here
    ![image](image_N.png) x1,y1,x2,y2

So we parse pairs: a text line followed immediately by its bbox line.
Lines that do not match either pattern are silently skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Matches:  ![image](image_N.png) x1,y1,x2,y2
# Groups:   (N)  (x1) (y1) (x2) (y2)
_BBOX_RE = re.compile(
    r"!\[image\]\(image_(\d+)\.png\)\s+"
    r"(\d+(?:\.\d+)?),(\d+(?:\.\d+)?),(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)"
)

_NORM = 1000.0  # coordinates are in [0, 1000]


@dataclass
class OCRBox:
    """A single detected text region with normalised [0,1] coordinates."""

    text: str
    # Normalised to [0, 1] — caller scales to pixel / page space
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0  # model does not emit per-box scores


def parse_lighton_output(raw: str) -> list[OCRBox]:
    """Parse the raw string returned by LightOnOCR-2-1B into a list of OCRBox.

    The function is tolerant of extra whitespace, blank lines, and stray lines
    that match neither the text nor the bbox pattern.

    Parameters
    ----------
    raw:
        The full text content of ``completion.choices[0].message.content``.

    Returns
    -------
    list[OCRBox]
        One entry per recognised text region, in document order.
    """
    boxes: list[OCRBox] = []
    pending_text: str | None = None

    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = _BBOX_RE.match(line)
        if m:
            x1 = float(m.group(2)) / _NORM
            y1 = float(m.group(3)) / _NORM
            x2 = float(m.group(4)) / _NORM
            y2 = float(m.group(5)) / _NORM

            # Use the pending text line if we have one; otherwise fall back to
            # whatever the model wrote after the coordinate token (edge case).
            text = pending_text if pending_text is not None else ""
            pending_text = None

            if text:  # skip boxes with no associated text
                boxes.append(OCRBox(text=text, x1=x1, y1=y1, x2=x2, y2=y2))
        else:
            # Not a bbox line → treat as the text for the next bbox line.
            # If two text lines appear in a row the first is discarded; this
            # mirrors how the model actually formats its output.
            pending_text = line

    return boxes
