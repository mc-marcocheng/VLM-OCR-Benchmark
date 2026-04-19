"""
Bounding-box IoU metrics with Hungarian matching.

Falls back to greedy matching if scipy is unavailable.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from loguru import logger
from ocr_core.metrics.base import Metric, MetricResult
from ocr_core.normalisation import NormalisationPipeline
from ocr_core.types import BBox, OCRPage


def _iou_matrix(gt: Sequence[BBox], pred: Sequence[BBox]) -> np.ndarray:
    m, n = len(gt), len(pred)
    mat = np.zeros((m, n), dtype=np.float64)
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            mat[i, j] = g.iou(p)
    return mat


def _hungarian_match(cost: np.ndarray) -> list[tuple[int, int]]:
    """Optimal assignment. Falls back to greedy if scipy is unavailable."""
    try:
        from scipy.optimize import linear_sum_assignment

        row_idx, col_idx = linear_sum_assignment(-cost)
        return list(zip(row_idx.tolist(), col_idx.tolist()))
    except ImportError:
        logger.warning(
            "scipy not installed — using greedy matching for layout IoU. "
            "Install with: uv sync --extra optimal-matching"
        )
        matches: list[tuple[int, int]] = []
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        m, n = cost.shape
        flat = [(cost[i, j], i, j) for i in range(m) for j in range(n)]
        flat.sort(reverse=True)
        for _, i, j in flat:
            if i not in used_rows and j not in used_cols:
                matches.append((i, j))
                used_rows.add(i)
                used_cols.add(j)
        return matches


class LayoutIOUMetric(Metric):
    name = "layout_iou"
    primary_key = "layout_mean_iou"

    def __init__(self, iou_threshold: float = 0.5, **params):
        super().__init__(**params)
        self.iou_threshold = iou_threshold

    def is_applicable(self, gt_page: OCRPage, pred_page: OCRPage) -> bool:
        return gt_page.has_bboxes()

    def compute(
        self,
        gt_page: OCRPage,
        pred_page: OCRPage,
        normaliser: NormalisationPipeline,
    ) -> MetricResult:
        gt_boxes = [r.bbox for r in gt_page.regions if r.bbox]
        pred_boxes = [r.bbox for r in pred_page.regions if r.bbox]
        gt_cats = [r.category for r in gt_page.regions if r.bbox]
        pred_cats = [r.category for r in pred_page.regions if r.bbox]

        if not gt_boxes:
            return MetricResult(
                scores={
                    "layout_mean_iou": float("nan"),
                    "layout_precision": float("nan"),
                    "layout_recall": float("nan"),
                }
            )
        if not pred_boxes:
            return MetricResult(
                scores={
                    "layout_mean_iou": 0.0,
                    "layout_precision": 0.0,
                    "layout_recall": 0.0,
                }
            )

        iou_mat = _iou_matrix(gt_boxes, pred_boxes)
        matches = _hungarian_match(iou_mat)

        ious: list[float] = []
        tp = 0
        for i, j in matches:
            iou_val = iou_mat[i, j]
            ious.append(iou_val)
            if iou_val >= self.iou_threshold and gt_cats[i] == pred_cats[j]:
                tp += 1

        precision = tp / len(pred_boxes) if pred_boxes else 0.0
        recall = tp / len(gt_boxes) if gt_boxes else 0.0
        mean_iou = float(np.mean(ious)) if ious else 0.0

        return MetricResult(
            scores={
                "layout_mean_iou": mean_iou,
                "layout_precision": precision,
                "layout_recall": recall,
            },
            details={"matched_ious": ious},
        )
