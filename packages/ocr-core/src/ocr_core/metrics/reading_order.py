"""
Reading-order evaluation via Kendall's τ on matched region indices.
"""

from __future__ import annotations

import numpy as np

from ocr_core.metrics.base import Metric, MetricResult
from ocr_core.metrics.layout_iou import _hungarian_match
from ocr_core.normalisation import NormalisationPipeline
from ocr_core.types import OCRPage


def _kendall_tau(a: list[int], b: list[int]) -> float:
    """Kendall τ between two rankings of the same items."""
    n = len(a)
    if n < 2:
        return 1.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff_a = a[i] - a[j]
            diff_b = b[i] - b[j]
            product = diff_a * diff_b
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total > 0 else 1.0


class ReadingOrderMetric(Metric):
    name = "reading_order"
    primary_key = "reading_order_tau"

    def is_applicable(self, gt_page: OCRPage, pred_page: OCRPage) -> bool:
        gt_ordered = [r for r in gt_page.regions if r.order >= 0]
        return len(gt_ordered) >= 2

    def compute(
        self,
        gt_page: OCRPage,
        pred_page: OCRPage,
        normaliser: NormalisationPipeline,
    ) -> MetricResult:
        gt_regions = [r for r in gt_page.regions if r.order >= 0]
        pred_regions = [r for r in pred_page.regions if r.order >= 0]

        if len(gt_regions) < 2:
            return MetricResult(scores={"reading_order_tau": float("nan")})
        if len(pred_regions) < 2:
            return MetricResult(scores={"reading_order_tau": 0.0})

        # Build similarity matrix
        m, n = len(gt_regions), len(pred_regions)
        sim = np.zeros((m, n))
        for i, gt_r in enumerate(gt_regions):
            for j, pred_r in enumerate(pred_regions):
                if gt_r.bbox and pred_r.bbox:
                    sim[i, j] = gt_r.bbox.iou(pred_r.bbox)
                else:
                    # Fallback: Jaccard on normalised text (word-level)
                    g_set = set(normaliser.tokenise_for_wer(gt_r.text).split())
                    p_set = set(normaliser.tokenise_for_wer(pred_r.text).split())
                    union = g_set | p_set
                    sim[i, j] = len(g_set & p_set) / len(union) if union else 0.0

        # Use Hungarian matching for unique assignments
        matches = _hungarian_match(sim)

        # Extract orders for matched pairs with non-zero similarity
        gt_orders = [gt_regions[i].order for i, j in matches if sim[i, j] > 0]
        pred_orders = [pred_regions[j].order for i, j in matches if sim[i, j] > 0]

        if len(gt_orders) < 2:
            return MetricResult(scores={"reading_order_tau": float("nan")})

        tau = _kendall_tau(gt_orders, pred_orders)
        return MetricResult(scores={"reading_order_tau": tau})
