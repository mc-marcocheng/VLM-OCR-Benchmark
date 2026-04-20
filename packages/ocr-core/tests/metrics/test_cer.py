"""Tests for CER metric."""

from __future__ import annotations

import math

import pytest

from ocr_core.metrics.cer import CERMetric
from ocr_core.types import OCRPage


class TestCER:
    def setup_method(self):
        self.metric = CERMetric()

    def test_perfect_match(self, default_normaliser):
        gt = OCRPage(full_text="hello world")
        pred = OCRPage(full_text="hello world")
        result = self.metric.compute(gt, pred, default_normaliser)
        assert result.scores["cer"] == pytest.approx(0.0)

    def test_completely_wrong(self, default_normaliser):
        gt = OCRPage(full_text="abc")
        pred = OCRPage(full_text="xyz")
        result = self.metric.compute(gt, pred, default_normaliser)
        assert result.scores["cer"] == pytest.approx(1.0)

    def test_empty_ground_truth(self, default_normaliser):
        gt = OCRPage(full_text="")
        pred = OCRPage(full_text="something")
        result = self.metric.compute(gt, pred, default_normaliser)
        assert math.isnan(result.scores["cer"])

    def test_empty_prediction(self, default_normaliser):
        gt = OCRPage(full_text="hello")
        pred = OCRPage(full_text="")
        result = self.metric.compute(gt, pred, default_normaliser)
        assert result.scores["cer"] == pytest.approx(1.0)

    def test_both_empty(self, default_normaliser):
        gt = OCRPage(full_text="")
        pred = OCRPage(full_text="")
        result = self.metric.compute(gt, pred, default_normaliser)
        assert math.isnan(result.scores["cer"])

    def test_partial_match(self, default_normaliser):
        gt = OCRPage(full_text="abcdef")
        pred = OCRPage(full_text="abcxyz")
        result = self.metric.compute(gt, pred, default_normaliser)
        assert 0 < result.scores["cer"] < 1.0

    def test_details_present(self, default_normaliser):
        gt = OCRPage(full_text="hello")
        pred = OCRPage(full_text="hallo")
        result = self.metric.compute(gt, pred, default_normaliser)
        assert "substitutions" in result.details
        assert "deletions" in result.details
        assert "insertions" in result.details

    def test_name_and_key(self):
        assert self.metric.name == "cer"
        assert self.metric.primary_key == "cer"
