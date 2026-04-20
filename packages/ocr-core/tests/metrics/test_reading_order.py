"""Tests for reading order metric."""

from __future__ import annotations

import math

import pytest

from ocr_core.metrics.reading_order import ReadingOrderMetric, _kendall_tau
from ocr_core.types import BBox, OCRPage, OCRRegion


class TestKendallTau:
    def test_identical(self):
        assert _kendall_tau([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_reversed(self):
        assert _kendall_tau([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)

    def test_single_element(self):
        assert _kendall_tau([1], [1]) == 1.0

    def test_empty(self):
        assert _kendall_tau([], []) == 1.0

    def test_partial_agreement(self):
        tau = _kendall_tau([1, 2, 3, 4], [1, 3, 2, 4])
        assert -1 <= tau <= 1
        # One swap: concordant=5, discordant=1, tau=4/6
        assert tau == pytest.approx(4 / 6)


class TestReadingOrderMetric:
    def setup_method(self):
        self.metric = ReadingOrderMetric()

    def _make_page(self, orders, base_x=0, step=100) -> OCRPage:
        regions = []
        for i, o in enumerate(orders):
            regions.append(
                OCRRegion(
                    text=f"region_{i}",
                    order=o,
                    bbox=BBox(base_x + i * step, 0, base_x + (i + 1) * step, 50),
                )
            )
        return OCRPage(regions=regions)

    def test_perfect_order(self, default_normaliser):
        gt = self._make_page([0, 1, 2, 3])
        pred = self._make_page([0, 1, 2, 3])
        r = self.metric.compute(gt, pred, default_normaliser)
        assert r.scores["reading_order_tau"] == pytest.approx(1.0)

    def test_reversed_order(self, default_normaliser):
        gt = self._make_page([0, 1, 2, 3])
        pred = self._make_page([3, 2, 1, 0])
        r = self.metric.compute(gt, pred, default_normaliser)
        assert r.scores["reading_order_tau"] == pytest.approx(-1.0)

    def test_not_applicable_single_region(self):
        gt = OCRPage(regions=[OCRRegion(order=0)])
        pred = OCRPage()
        assert not self.metric.is_applicable(gt, pred)

    def test_not_applicable_no_ordered_regions(self):
        gt = OCRPage(regions=[OCRRegion(order=-1), OCRRegion(order=-1)])
        pred = OCRPage()
        assert not self.metric.is_applicable(gt, pred)

    def test_applicable_two_ordered(self):
        gt = OCRPage(regions=[OCRRegion(order=0), OCRRegion(order=1)])
        pred = OCRPage()
        assert self.metric.is_applicable(gt, pred)

    def test_too_few_pred_regions(self, default_normaliser):
        gt = self._make_page([0, 1, 2])
        pred = OCRPage(regions=[OCRRegion(order=0, bbox=BBox(0, 0, 100, 50))])
        r = self.metric.compute(gt, pred, default_normaliser)
        assert r.scores["reading_order_tau"] == pytest.approx(0.0)

    def test_too_few_gt_regions(self, default_normaliser):
        gt = OCRPage(regions=[OCRRegion(order=0)])
        pred = self._make_page([0, 1, 2])
        r = self.metric.compute(gt, pred, default_normaliser)
        assert math.isnan(r.scores["reading_order_tau"])
