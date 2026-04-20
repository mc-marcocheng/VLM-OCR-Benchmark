"""Tests for layout IoU metric."""

from __future__ import annotations

import math

import pytest

from ocr_core.metrics.layout_iou import LayoutIOUMetric
from ocr_core.types import BBox, OCRPage, OCRRegion


def _page_with_regions(regions: list[OCRRegion]) -> OCRPage:
    return OCRPage(regions=regions)


class TestLayoutIOU:
    def setup_method(self):
        self.metric = LayoutIOUMetric(iou_threshold=0.5)

    def test_perfect_match(self, default_normaliser):
        regions = [
            OCRRegion(text="a", category="text", bbox=BBox(0, 0, 100, 100)),
            OCRRegion(text="b", category="title", bbox=BBox(200, 200, 300, 300)),
        ]
        gt = _page_with_regions(regions)
        pred = _page_with_regions(regions)
        r = self.metric.compute(gt, pred, default_normaliser)
        assert r.scores["layout_mean_iou"] == pytest.approx(1.0)
        assert r.scores["layout_precision"] == pytest.approx(1.0)
        assert r.scores["layout_recall"] == pytest.approx(1.0)

    def test_no_overlap(self, default_normaliser):
        gt = _page_with_regions(
            [OCRRegion(text="a", category="text", bbox=BBox(0, 0, 10, 10))]
        )
        pred = _page_with_regions(
            [OCRRegion(text="b", category="text", bbox=BBox(100, 100, 200, 200))]
        )
        r = self.metric.compute(gt, pred, default_normaliser)
        assert r.scores["layout_mean_iou"] == pytest.approx(0.0)
        assert r.scores["layout_precision"] == pytest.approx(0.0)

    def test_both_empty(self, default_normaliser):
        gt = _page_with_regions([])
        pred = _page_with_regions([])
        r = self.metric.compute(gt, pred, default_normaliser)
        assert r.scores["layout_mean_iou"] == pytest.approx(1.0)

    def test_empty_gt(self, default_normaliser):
        gt = _page_with_regions([])
        pred = _page_with_regions(
            [OCRRegion(text="x", category="text", bbox=BBox(0, 0, 10, 10))]
        )
        r = self.metric.compute(gt, pred, default_normaliser)
        assert math.isnan(r.scores["layout_mean_iou"])

    def test_empty_pred(self, default_normaliser):
        gt = _page_with_regions(
            [OCRRegion(text="x", category="text", bbox=BBox(0, 0, 10, 10))]
        )
        pred = _page_with_regions([])
        r = self.metric.compute(gt, pred, default_normaliser)
        assert r.scores["layout_mean_iou"] == 0.0
        assert r.scores["layout_recall"] == 0.0

    def test_category_mismatch(self, default_normaliser):
        """Boxes overlap perfectly but categories differ → tp=0."""
        gt = _page_with_regions(
            [OCRRegion(text="a", category="text", bbox=BBox(0, 0, 100, 100))]
        )
        pred = _page_with_regions(
            [OCRRegion(text="a", category="title", bbox=BBox(0, 0, 100, 100))]
        )
        r = self.metric.compute(gt, pred, default_normaliser)
        assert r.scores["layout_mean_iou"] == pytest.approx(1.0)
        assert r.scores["layout_precision"] == pytest.approx(0.0)

    def test_applicability_requires_bboxes(self, default_normaliser):
        gt_no_bbox = OCRPage(regions=[OCRRegion(text="a")])
        pred = OCRPage()
        assert not self.metric.is_applicable(gt_no_bbox, pred)

        gt_with_bbox = OCRPage(regions=[OCRRegion(bbox=BBox(0, 0, 1, 1))])
        assert self.metric.is_applicable(gt_with_bbox, pred)

    def test_regions_without_bbox_ignored(self, default_normaliser):
        gt = _page_with_regions(
            [
                OCRRegion(text="a", category="text", bbox=BBox(0, 0, 100, 100)),
                OCRRegion(text="b", category="text"),  # no bbox
            ]
        )
        pred = _page_with_regions(
            [OCRRegion(text="a", category="text", bbox=BBox(0, 0, 100, 100))]
        )
        r = self.metric.compute(gt, pred, default_normaliser)
        assert r.scores["layout_mean_iou"] == pytest.approx(1.0)
