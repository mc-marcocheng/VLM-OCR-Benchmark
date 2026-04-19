"""
Convert a Label Studio JSON export into benchmark ground-truth files.

Usage:
    python scripts/convert_label_studio.py <export.json> <test_set>

Produces one .json per annotated image under:
    data/groundtruths/<test_set>/
"""

from __future__ import annotations

import json
import math
import os
import re
import sys

GT_BASE = os.path.join("data", "groundtruths")

# Label Studio prefixes uploads with 8 hex chars + dash
_LS_PREFIX_RE = re.compile(r"^[0-9a-f]{8}-")


def strip_ls_prefix(file_upload: str) -> str:
    """'8d4fe007-IMG_0298.png' → 'IMG_0298.png'"""
    return _LS_PREFIX_RE.sub("", file_upload)


# ── Rotated bbox → axis-aligned bbox ───────────────────────


def _rotate_point(
    px: float, py: float, cx: float, cy: float, angle_deg: float
) -> tuple[float, float]:
    """Rotate point (px, py) around centre (cx, cy)."""
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    dx, dy = px - cx, py - cy
    return (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)


def _to_pixel_bbox(
    x_pct: float,
    y_pct: float,
    w_pct: float,
    h_pct: float,
    rotation: float,
    orig_w: int,
    orig_h: int,
) -> dict[str, float]:
    """
    Convert Label Studio percentage-based (possibly rotated) bbox
    to an axis-aligned pixel bbox {x1, y1, x2, y2}.
    """
    x = x_pct / 100 * orig_w
    y = y_pct / 100 * orig_h
    w = w_pct / 100 * orig_w
    h = h_pct / 100 * orig_h

    if rotation == 0:
        return {
            "x1": round(x, 1),
            "y1": round(y, 1),
            "x2": round(x + w, 1),
            "y2": round(y + h, 1),
        }

    # Compute axis-aligned envelope of the rotated rectangle
    cx, cy = x + w / 2, y + h / 2
    corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    rotated = [_rotate_point(px, py, cx, cy, rotation) for px, py in corners]
    xs = [p[0] for p in rotated]
    ys = [p[1] for p in rotated]
    return {
        "x1": round(min(xs), 1),
        "y1": round(min(ys), 1),
        "x2": round(max(xs), 1),
        "y2": round(max(ys), 1),
    }


# ── Task conversion ────────────────────────────────────────


def convert_task(task: dict) -> dict | None:
    """Convert one Label Studio task → benchmark ground-truth dict."""
    annotations = task.get("annotations", [])
    if not annotations:
        return None

    results = annotations[0].get("result", [])
    if not results:
        return None

    # ── Group results by region id ──
    # Each annotated region produces TWO result entries sharing the same id:
    #   type="rectanglelabels"  →  bbox + category
    #   type="textarea"         →  transcription text
    rects: dict[str, dict] = {}
    texts: dict[str, list[str]] = {}

    for r in results:
        rid = r["id"]
        if r["type"] == "rectanglelabels":
            rects[rid] = r
        elif r["type"] == "textarea":
            texts[rid] = r["value"].get("text", [])

    # ── Build regions in annotation order ──
    seen: list[str] = []
    for r in results:
        if r["type"] == "rectanglelabels" and r["id"] not in seen:
            seen.append(r["id"])

    regions: list[dict] = []
    for order, rid in enumerate(seen):
        rect = rects[rid]
        val = rect["value"]
        orig_w = rect["original_width"]
        orig_h = rect["original_height"]

        bbox = _to_pixel_bbox(
            val["x"],
            val["y"],
            val["width"],
            val["height"],
            val.get("rotation", 0),
            orig_w,
            orig_h,
        )

        category = val["rectanglelabels"][0].lower()
        text = "\n".join(texts.get(rid, []))

        text_format = "plain"
        if category == "table":
            text_format = "html"
        elif category == "formula":
            text_format = "latex"

        regions.append(
            {
                "text": text,
                "category": category,
                "bbox": bbox,
                "text_format": text_format,
                "order": order,
            }
        )

    full_text = "\n".join(
        r["text"] for r in regions if r["text"] and r["category"] != "picture"
    )

    # Dimensions from first result
    first = results[0]

    return {
        "pages": [
            {
                "page_number": 1,
                "full_text": full_text,
                "width": first["original_width"],
                "height": first["original_height"],
                "regions": regions,
            }
        ],
    }


# ── Main ────────────────────────────────────────────────────


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/convert_label_studio.py <export.json> <test_set>")
        print()
        print("Example:")
        print(
            "  python scripts/convert_label_studio.py "
            "project-1-at-2026-01-01.json test_1"
        )
        sys.exit(1)

    export_path = sys.argv[1]
    test_set = sys.argv[2]

    with open(export_path, encoding="utf-8") as f:
        tasks = json.load(f)

    output_dir = os.path.join(GT_BASE, test_set)
    os.makedirs(output_dir, exist_ok=True)

    converted = 0
    for task in tasks:
        file_upload = task.get("file_upload", "")
        original_name = strip_ls_prefix(file_upload)
        stem = os.path.splitext(original_name)[0]

        gt = convert_task(task)
        if gt is None:
            print(f"  ⚠️  Skipped {original_name} (no annotations)")
            continue

        gt["source_file"] = original_name

        out_path = os.path.join(output_dir, f"{stem}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(gt, f, ensure_ascii=False, indent=2)

        n_regions = len(gt["pages"][0]["regions"])
        print(f"  ✅ {out_path} ({n_regions} regions)")
        converted += 1

    print(f"\nDone — {converted}/{len(tasks)} file(s) → {output_dir}")


if __name__ == "__main__":
    main()
