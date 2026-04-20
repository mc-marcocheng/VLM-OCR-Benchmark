"""Post-processing utilities for Chandra OCR-2 HTML output.

Parses the model's HTML output (with data-bbox and data-label div attributes)
into structured layout blocks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag

# BBox scale used by the model (normalised coordinates)
BBOX_SCALE = 1000


@dataclass
class LayoutBlock:
    """A single detected layout block from the HTML output."""

    bbox: list[int]  # [x1, y1, x2, y2] in pixel coordinates
    label: str  # Category label
    content: str  # Inner HTML content of the div


def parse_html_layout(
    html: str,
    image_width: int,
    image_height: int,
    bbox_scale: int = BBOX_SCALE,
) -> list[LayoutBlock]:
    """Parse the model's raw HTML output into layout blocks.

    The model outputs top-level `<div>` elements, each with:
      - `data-bbox`: "x0 y0 x1 y1" normalised to [0, bbox_scale]
      - `data-label`: category label string

    This function extracts those divs, converts normalised bboxes to
    pixel coordinates, and returns structured LayoutBlock objects.

    Args:
        html: Raw HTML string from model output.
        image_width: Original image width in pixels.
        image_height: Original image height in pixels.
        bbox_scale: The normalisation scale used by the model (default 1000).

    Returns:
        List of LayoutBlock instances in reading order.
    """
    soup = BeautifulSoup(html, "html.parser")
    top_level_divs = soup.find_all("div", recursive=False)

    width_scaler = image_width / bbox_scale
    height_scaler = image_height / bbox_scale

    layout_blocks: list[LayoutBlock] = []

    for div in top_level_divs:
        if not isinstance(div, Tag):
            continue

        label = div.get("data-label", "")
        if not label:
            label = "block"

        # Skip blank pages
        if label == "Blank-Page":
            continue

        # Parse bbox
        bbox_str = div.get("data-bbox", "")
        bbox = _parse_bbox(
            bbox_str, width_scaler, height_scaler, image_width, image_height
        )

        # Get inner HTML content (strip nested data-bbox attributes)
        content = str(div.decode_contents())
        content_soup = BeautifulSoup(content, "html.parser")
        for tag in content_soup.find_all(attrs={"data-bbox": True}):
            del tag["data-bbox"]
        content = str(content_soup)

        layout_blocks.append(LayoutBlock(bbox=bbox, label=label, content=content))

    return layout_blocks


def _parse_bbox(
    bbox_str: str,
    width_scaler: float,
    height_scaler: float,
    image_width: int,
    image_height: int,
) -> list[int]:
    """Parse a bbox string and convert to pixel coordinates.

    Args:
        bbox_str: Space-separated "x0 y0 x1 y1" string (normalised).
        width_scaler: Multiplier for x coordinates (image_width / bbox_scale).
        height_scaler: Multiplier for y coordinates (image_height / bbox_scale).
        image_width: Image width for clamping.
        image_height: Image height for clamping.

    Returns:
        List [x1, y1, x2, y2] in pixel coordinates.
    """
    try:
        parts = bbox_str.strip().split()
        coords = list(map(int, parts))
        if len(coords) != 4:
            raise ValueError(f"Expected 4 values, got {len(coords)}")
    except (ValueError, AttributeError):
        # Default to full image if bbox is invalid
        return [0, 0, image_width, image_height]

    # Convert normalised → pixel coordinates and clamp
    x1 = max(0, min(int(coords[0] * width_scaler), image_width))
    y1 = max(0, min(int(coords[1] * height_scaler), image_height))
    x2 = max(0, min(int(coords[2] * width_scaler), image_width))
    y2 = max(0, min(int(coords[3] * height_scaler), image_height))

    return [x1, y1, x2, y2]


def extract_text_from_html(html_content: str) -> str:
    """Extract plain text from HTML content, stripping all tags.

    Args:
        html_content: HTML string.

    Returns:
        Plain text with whitespace normalised.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_latex_from_math_tags(html_content: str) -> str:
    """Extract LaTeX from <math> tags in HTML content.

    If multiple <math> tags are present, they are joined with newlines.

    Args:
        html_content: HTML string potentially containing <math> tags.

    Returns:
        LaTeX string, or the plain text if no <math> tags found.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    math_tags = soup.find_all("math")

    if not math_tags:
        return extract_text_from_html(html_content)

    latex_parts = []
    for tag in math_tags:
        latex_parts.append(tag.get_text(strip=True))

    return "\n".join(latex_parts)


def extract_table_html(html_content: str) -> str:
    """Extract <table> element from HTML content.

    Returns the full table HTML, or the original content if no table found.

    Args:
        html_content: HTML string potentially containing a <table>.

    Returns:
        Table HTML string or the original content.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    table = soup.find("table")
    if table:
        return str(table)
    return html_content
