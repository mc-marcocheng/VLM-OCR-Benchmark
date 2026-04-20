"""Post-processing utilities for LightOnOCR-2 output.

Handles two distinct output formats:

* **Non-bbox models** (e.g. LightOnOCR-2-1B):
  Output is plain markdown text with no positional information.

* **Bbox models** (e.g. LightOnOCR-2-1B-bbox):
  Output is markdown text with inline image references annotated with
  bounding boxes: ``![image](image_N.png)x1,y1,x2,y2`` where coordinates
  are normalised to [0, 1000].
"""

from __future__ import annotations

import re
from typing import Any

# Pattern: ![image](image_N.png)x1,y1,x2,y2  (no space between ref and coords)
BBOX_PATTERN = re.compile(r"!\[image\]\((image_\d+\.png)\)\s*(\d+),(\d+),(\d+),(\d+)")

# Normalisation scale used by the bbox model
BBOX_SCALE = 1000


def clean_output_text(text: str) -> str:
    """Remove chat-template artifacts from raw decoded output.

    The model's tokenizer may leave ``system``, ``user``, or ``assistant``
    turn markers in the decoded text.  This function strips them so that
    only the assistant's response remains.

    Args:
        text: Raw decoded text (may include the full conversation).

    Returns:
        Cleaned assistant response.
    """
    # Fast path: if "assistant" marker is present, take everything after it
    lower = text.lower()
    if "assistant" in lower:
        idx = lower.index("assistant")
        text = text[idx + len("assistant") :].strip()

    # Strip any remaining bare turn markers that appear on their own line
    markers = {"system", "user", "assistant"}
    lines = text.split("\n")
    cleaned = [ln for ln in lines if ln.strip().lower() not in markers]
    return "\n".join(cleaned).strip()


# ---------------------------------------------------------------------------
# Bbox output parsing
# ---------------------------------------------------------------------------


def parse_bbox_output(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Parse bbox-annotated markdown into cleaned text and detections.

    Args:
        text: Model output containing ``![image](image_N.png)x1,y1,x2,y2``
              annotations.

    Returns:
        Tuple of:
        - ``cleaned_text``: markdown with coordinates stripped
          (image refs remain as ``![image](image_N.png)``).
        - ``detections``: list of dicts with ``ref`` (str) and
          ``coords`` (x1, y1, x2, y2) in normalised [0, 1000] values.
    """
    detections: list[dict[str, Any]] = []

    for match in BBOX_PATTERN.finditer(text):
        image_ref, x1, y1, x2, y2 = match.groups()
        detections.append(
            {
                "ref": image_ref,
                "coords": (int(x1), int(y1), int(x2), int(y2)),
            }
        )

    # Strip coordinates, keep the markdown image syntax
    cleaned = BBOX_PATTERN.sub(r"![image](\1)", text)
    return cleaned, detections


def normalised_to_pixel(
    coords: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    bbox_scale: int = BBOX_SCALE,
) -> tuple[float, float, float, float]:
    """Convert [0, bbox_scale] normalised coordinates to pixel values.

    Args:
        coords: (x1, y1, x2, y2) in normalised space.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        bbox_scale: Normalisation scale (default 1000).

    Returns:
        (x1, y1, x2, y2) in pixel coordinates, clamped to image bounds.
    """
    x1, y1, x2, y2 = coords
    px1 = max(0.0, min(x1 * image_width / bbox_scale, float(image_width)))
    py1 = max(0.0, min(y1 * image_height / bbox_scale, float(image_height)))
    px2 = max(0.0, min(x2 * image_width / bbox_scale, float(image_width)))
    py2 = max(0.0, min(y2 * image_height / bbox_scale, float(image_height)))
    return px1, py1, px2, py2


# ---------------------------------------------------------------------------
# Region builders
# ---------------------------------------------------------------------------


def parse_bbox_to_regions(
    text: str,
    image_width: int,
    image_height: int,
) -> tuple[str, list[dict[str, Any]]]:
    """Parse bbox model output into full_text and region dicts.

    Returns a cleaned markdown string (``full_text``) and a list of region
    dicts ready to be converted to ``OCRRegion`` objects.  Each detected
    image becomes a ``picture`` region with pixel-coordinate bbox.  A single
    ``text`` region covering the full page holds the cleaned markdown.

    Args:
        text: Raw model output with bbox annotations.
        image_width: Original image width in pixels.
        image_height: Original image height in pixels.

    Returns:
        Tuple of (full_text, regions) where regions is a list of dicts
        with keys: text, category, bbox, order, text_format.
    """
    cleaned_text, detections = parse_bbox_output(text)

    regions: list[dict[str, Any]] = []
    order = 0

    # Add a text region for the full OCR content
    regions.append(
        {
            "text": cleaned_text,
            "category": "text",
            "bbox": None,  # covers whole page
            "order": order,
            "text_format": "markdown",
        }
    )
    order += 1

    # Add a picture region for each detected image bbox
    for det in detections:
        px1, py1, px2, py2 = normalised_to_pixel(
            det["coords"], image_width, image_height
        )
        # Skip degenerate bboxes
        if px2 <= px1 or py2 <= py1:
            continue

        regions.append(
            {
                "text": det["ref"],  # e.g. "image_1.png"
                "category": "picture",
                "bbox": (px1, py1, px2, py2),
                "order": order,
                "text_format": "plain",
            }
        )
        order += 1

    return cleaned_text, regions


def parse_plain_to_regions(
    text: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Parse non-bbox model output into full_text and region dicts.

    The non-bbox model outputs plain markdown with no positional data,
    so we return a single ``text`` region covering the whole page.

    Args:
        text: Raw model output (plain markdown).

    Returns:
        Tuple of (full_text, regions).
    """
    regions = [
        {
            "text": text,
            "category": "text",
            "bbox": None,
            "order": 0,
            "text_format": "markdown",
        }
    ]
    return text, regions
