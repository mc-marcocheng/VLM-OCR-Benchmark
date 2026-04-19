from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ocr_core.normalisation import NormalisationPipeline
from ocr_core.types import OCRPage


@dataclass
class MetricResult:
    """Container for one metric's output on a single page pair."""

    scores: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


class Metric(ABC):
    """Base class every metric must subclass."""

    name: str = ""
    primary_key: str = ""  # Key used for aggregation; defaults to name

    def __init__(self, apply_to: list[str] | None = None, **params: Any):
        self.apply_to = apply_to or []
        self.params = params
        if not self.primary_key:
            self.primary_key = self.name

    def is_applicable(self, gt_page: OCRPage, pred_page: OCRPage) -> bool:
        """Override to restrict when the metric is computed."""
        if not self.apply_to:
            return True
        # At least one GT region must match a requested category
        return any(r.category in self.apply_to for r in gt_page.regions)

    @abstractmethod
    def compute(
        self,
        gt_page: OCRPage,
        pred_page: OCRPage,
        normaliser: NormalisationPipeline,
    ) -> MetricResult: ...
