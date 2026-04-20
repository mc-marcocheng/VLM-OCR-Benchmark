"""Post-processing utilities for Infinity-Parser2-Pro output.

Adapted from the official Infinity-Parser2 repository (infinity_parser2/utils/utils.py).
Handles JSON extraction, bbox coordinate restoration, and markdown conversion.
"""

from __future__ import annotations

import json
import re
from typing import Union

from PIL import Image

# ---------------------------------------------------------------------------
# JSON extraction & cleanup
# ---------------------------------------------------------------------------


def extract_json_content(text: str) -> str:
    """Extract the JSON block from a markdown-wrapped LLM response.

    Handles responses wrapped in ```json ... ``` code fences, including
    cases where the closing fence is missing (truncated output).

    Args:
        text: Raw model output text.

    Returns:
        Extracted JSON string, or the original text if no fence is found.
    """
    match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    partial = re.search(r"```json\n(.*)", text, re.DOTALL)
    if partial:
        return partial.group(1).strip()
    return text


def truncate_last_incomplete_element(text: str) -> tuple[str, bool]:
    """Truncate the response at the last complete dict entry.

    When the model hits max_new_tokens, the JSON may be cut off mid-element.
    This function removes the last incomplete element so the JSON is always
    parseable.

    Args:
        text: JSON string that may be truncated.

    Returns:
        Tuple of (cleaned_text, was_truncated).
    """
    needs_truncation = len(text) > 50_000 or not text.rstrip().endswith("]")

    if not needs_truncation:
        return text, False

    if text.count('{"bbox":') <= 1:
        return text, False

    last_bbox_pos = text.rfind('{"bbox":')
    truncated = text[:last_bbox_pos].rstrip()
    if truncated.endswith(","):
        truncated = truncated[:-1] + "]"
    elif not truncated.endswith("]"):
        truncated = truncated + "]"
    return truncated, True


# ---------------------------------------------------------------------------
# Coordinate normalisation
# ---------------------------------------------------------------------------


def obtain_origin_hw(image: Union[str, Image.Image]) -> tuple[int, int]:
    """Return (height, width) of the image.

    Accepts a file path (str) or a PIL Image object.

    Args:
        image: File path or PIL Image.

    Returns:
        Tuple of (height, width) in pixels.
    """
    if isinstance(image, Image.Image):
        w, h = image.size
        return h, w
    try:
        img = Image.open(image).convert("RGB")
        w, h = img.size
        return h, w
    except Exception:
        return 1000, 1000


def restore_abs_bbox_coordinates(ans: str, origin_h: float, origin_w: float) -> str:
    """Convert normalised [0-1000] bboxes back to pixel coordinates.

    The model outputs bounding boxes normalised to a [0, 1000] coordinate
    system. This function restores them to absolute pixel coordinates
    based on the original image dimensions.

    Args:
        ans: JSON string containing layout elements with normalised bboxes.
        origin_h: Original image height in pixels.
        origin_w: Original image width in pixels.

    Returns:
        JSON string with bboxes converted to pixel coordinates.
    """
    try:
        data = json.loads(ans)
    except json.JSONDecodeError:
        return ans

    valid = True
    for item in data:
        for key in item:
            if "bbox" not in key:
                continue
            bbox = item[key]
            if len(bbox) == 4 and all(isinstance(c, (int, float)) for c in bbox):
                x1, y1, x2, y2 = bbox
                item[key] = [
                    int(x1 / 1000.0 * origin_w),
                    int(y1 / 1000.0 * origin_h),
                    int(x2 / 1000.0 * origin_w),
                    int(y2 / 1000.0 * origin_h),
                ]
            else:
                valid = False

    return json.dumps(data, ensure_ascii=False) if valid else ans


# ---------------------------------------------------------------------------
# JSON → Markdown
# ---------------------------------------------------------------------------


def convert_json_to_markdown(ans: str, keep_header_footer: bool = False) -> str:
    """Convert the layout JSON list into a markdown string.

    Concatenates text from all elements (optionally excluding headers/footers)
    into a single markdown document.

    Args:
        ans: JSON string containing layout element list.
        keep_header_footer: Whether to include header/footer/page_footnote text.

    Returns:
        Markdown string, or the original string if parsing fails.
    """
    try:
        items = json.loads(ans)
        if not isinstance(items, list):
            return ans
        lines = []
        for sub in items:
            if "text" not in sub or not sub["text"]:
                continue
            if keep_header_footer:
                lines.append(sub["text"])
            else:
                if sub.get("category") not in ("header", "footer", "page_footnote"):
                    lines.append(sub["text"])
        return "\n\n".join(lines) if lines else ans
    except Exception:
        return ans


# ---------------------------------------------------------------------------
# Combined post-processing pipeline
# ---------------------------------------------------------------------------


def postprocess_doc2json_result(
    raw_text: str,
    image: Union[str, Image.Image],
) -> str:
    """Full post-processing pipeline for doc2json mode.

    1. Extract JSON block from markdown-wrapped response
    2. Truncate last incomplete element for parseable JSON
    3. Restore normalised [0-1000] bboxes to pixel coordinates

    Args:
        raw_text: Raw model output text.
        image: Original image (path or PIL Image) for obtaining dimensions.

    Returns:
        Post-processed JSON string with pixel-coordinate bboxes.
    """
    text = extract_json_content(raw_text)
    text, _ = truncate_last_incomplete_element(text)
    origin_h, origin_w = obtain_origin_hw(image)
    text = restore_abs_bbox_coordinates(text, origin_h, origin_w)
    return text
