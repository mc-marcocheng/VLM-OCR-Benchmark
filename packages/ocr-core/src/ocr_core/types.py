"""
Canonical data types shared across the entire framework.

Every model worker serialises predictions into these types (via ``to_dict`` /
``from_dict``), and every metric receives them as input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "BBox",
    "REGION_CATEGORIES",
    "TEXT_FORMATS",
    "OCRRegion",
    "OCRPage",
    "GroundTruth",
    "WorkerTask",
    "WorkerPageResult",
    "WorkerResponse",
]


# ── Bounding box ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned bounding box — coordinates are in *pixels*."""

    x1: float
    y1: float
    x2: float
    y2: float

    # ── geometry helpers ──

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: BBox) -> float:
        """Intersection-over-union with *other*."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    # ── serialisation ──

    def to_dict(self) -> dict:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    @classmethod
    def from_dict(cls, d: dict | list) -> BBox:
        if isinstance(d, (list, tuple)):
            if len(d) < 4:
                raise ValueError(f"BBox requires 4 values, got {len(d)}: {d}")
            return cls(x1=d[0], y1=d[1], x2=d[2], y2=d[3])
        return cls(x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"])


# ── OCR region ──────────────────────────────────────────────

REGION_CATEGORIES = frozenset({
    "text",
    "title",
    "section-header",
    "caption",
    "footnote",
    "page-header",
    "page-footer",
    "list-item",
    "table",
    "formula",
    "picture",
})

TEXT_FORMATS = frozenset({"plain", "markdown", "latex", "html"})


@dataclass
class OCRRegion:
    """A single detected region on a page."""

    text: str = ""
    category: str = "text"
    bbox: BBox | None = None
    confidence: float = -1.0
    text_format: str = "plain"
    order: int = -1
    children: list[OCRRegion] = field(default_factory=list)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "text": self.text,
            "category": self.category,
            "confidence": self.confidence,
            "text_format": self.text_format,
            "order": self.order,
        }
        if self.bbox is not None:
            d["bbox"] = self.bbox.to_dict()
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> OCRRegion:
        bbox = BBox.from_dict(d["bbox"]) if d.get("bbox") else None
        children = [cls.from_dict(c) for c in d.get("children", [])]
        return cls(
            text=d.get("text", ""),
            category=d.get("category", "text"),
            bbox=bbox,
            confidence=d.get("confidence", -1.0),
            text_format=d.get("text_format", "plain"),
            order=d.get("order", -1),
            children=children,
        )


# ── Page ────────────────────────────────────────────────────


@dataclass
class OCRPage:
    """One page of OCR output (or ground truth)."""

    page_number: int = 1
    full_text: str = ""
    regions: list[OCRRegion] = field(default_factory=list)
    width: int = 0
    height: int = 0

    # ── convenience accessors ──

    def regions_by_category(self, *categories: str) -> list[OCRRegion]:
        return [r for r in self.regions if r.category in categories]

    def has_regions(self) -> bool:
        return len(self.regions) > 0

    def has_bboxes(self) -> bool:
        return any(r.bbox is not None for r in self.regions)

    # ── serialisation ──

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "full_text": self.full_text,
            "width": self.width,
            "height": self.height,
            "regions": [r.to_dict() for r in self.regions],
        }

    @classmethod
    def from_dict(cls, d: dict) -> OCRPage:
        return cls(
            page_number=d.get("page_number", d.get("page", 1)),
            full_text=d.get("full_text", ""),
            width=d.get("width", 0),
            height=d.get("height", 0),
            regions=[OCRRegion.from_dict(r) for r in d.get("regions", [])],
        )


# ── Ground truth (alias — same structure, distinct semantics) ──


@dataclass
class GroundTruth:
    """Ground-truth annotation for one input file."""

    source_file: str = ""
    pages: list[OCRPage] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.full_text for p in self.pages)

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "pages": [p.to_dict() for p in self.pages],
        }

    @classmethod
    def from_dict(cls, d: dict) -> GroundTruth:
        return cls(
            source_file=d.get("source_file", d.get("file", "")),
            pages=[OCRPage.from_dict(p) for p in d.get("pages", [])],
        )


# ── Worker-protocol types ───────────────────────────────────


@dataclass
class WorkerTask:
    """JSON payload sent *to* a model worker."""

    image_paths: list[str] = field(default_factory=list)
    device: str = "cpu"
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> WorkerTask:
        return cls(
            image_paths=d.get("image_paths", []),
            device=d.get("device", "cpu"),
            params=d.get("params", {}),
        )


@dataclass
class WorkerPageResult:
    image_path: str = ""
    prediction_time_seconds: float = 0.0
    ram_after_mb: float = 0.0
    result: OCRPage = field(default_factory=OCRPage)
    error: str | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "image_path": self.image_path,
            "prediction_time_seconds": self.prediction_time_seconds,
            "ram_after_mb": self.ram_after_mb,
            "result": self.result.to_dict(),
        }
        if self.error:
            d["error"] = self.error
        return d

    @classmethod
    def from_dict(cls, d: dict) -> WorkerPageResult:
        return cls(
            image_path=d.get("image_path", ""),
            prediction_time_seconds=d.get("prediction_time_seconds", 0.0),
            ram_after_mb=d.get("ram_after_mb", 0.0),
            result=OCRPage.from_dict(d.get("result", {})),
            error=d.get("error"),
        )


@dataclass
class WorkerResponse:
    """JSON payload returned *from* a model worker."""

    model_load_time_seconds: float = 0.0
    ram_before_load_mb: float = 0.0
    ram_after_load_mb: float = 0.0
    peak_ram_mb: float = 0.0
    vram_before_load_mb: float | None = None
    vram_after_load_mb: float | None = None
    peak_vram_mb: float | None = None
    pages: list[WorkerPageResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_load_time_seconds": self.model_load_time_seconds,
            "ram_before_load_mb": self.ram_before_load_mb,
            "ram_after_load_mb": self.ram_after_load_mb,
            "peak_ram_mb": self.peak_ram_mb,
            "vram_before_load_mb": self.vram_before_load_mb,
            "vram_after_load_mb": self.vram_after_load_mb,
            "peak_vram_mb": self.peak_vram_mb,
            "pages": [p.to_dict() for p in self.pages],
        }

    @classmethod
    def from_dict(cls, d: dict) -> WorkerResponse:
        return cls(
            model_load_time_seconds=d.get("model_load_time_seconds", 0.0),
            ram_before_load_mb=d.get("ram_before_load_mb", 0.0),
            ram_after_load_mb=d.get("ram_after_load_mb", 0.0),
            peak_ram_mb=d.get("peak_ram_mb", 0.0),
            vram_before_load_mb=d.get("vram_before_load_mb"),
            vram_after_load_mb=d.get("vram_after_load_mb"),
            peak_vram_mb=d.get("peak_vram_mb"),
            pages=[WorkerPageResult.from_dict(p) for p in d.get("pages", [])],
        )
