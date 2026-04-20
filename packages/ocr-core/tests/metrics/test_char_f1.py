"""Tests for character-level F1 metric."""

from __future__ import annotations

import pytest

from ocr_core.metrics.char_f1 import CharF1Metric
from ocr_core.types import OCRPage


class TestCharF1:
    def setup_method(self):
        self.metric = CharF1Metric()

    def test_perfect_match(self, default_normaliser):
        gt = OCRPage(full_text="hello")
        pred = OCRPage(full_text="hello")
        result = self.metric.compute(gt, pred, default_normaliser)
        assert result.scores["char_f1"] == pytest.approx(1.0)
        assert result.scores["char_precision"] == pytest.approx(1.0)
        assert result.scores["char_recall"] == pytest.approx(1.0)

    def test_both_empty(self, default_normaliser):
        gt = OCRPage(full_text="")
        pred = OCRPage(full_text="")
        result = self.metric.compute(gt, pred, default_normaliser)
        assert result.scores["char_f1"] == pytest.approx(1.0)

    def test_empty_gt_nonempty_pred(self, default_normaliser):
        gt = OCRPage(full_text="")
        pred = OCRPage(full_text="hello")
        result = self.metric.compute(gt, pred, default_normaliser)
        assert result.scores["char_f1"] == pytest.approx(0.0)

    def test_nonempty_gt_empty_pred(self, default_normaliser):
        gt = OCRPage(full_text="hello")
        pred = OCRPage(full_text="")
        result = self.metric.compute(gt, pred, default_normaliser)
        assert result.scores["char_f1"] == pytest.approx(0.0)
        assert result.scores["char_precision"] == pytest.approx(0.0)
        assert result.scores["char_recall"] == pytest.approx(0.0)

    def test_partial_match(self, default_normaliser):
        gt = OCRPage(full_text="abcdef")
        pred = OCRPage(full_text="abcxyz")
        result = self.metric.compute(gt, pred, default_normaliser)
        f1 = result.scores["char_f1"]
        assert 0 < f1 < 1

    def test_f1_between_precision_and_recall(self, default_normaliser):
        gt = OCRPage(full_text="abcdef")
        pred = OCRPage(full_text="abc")
        result = self.metric.compute(gt, pred, default_normaliser)
        p = result.scores["char_precision"]
        r = result.scores["char_recall"]
        f1 = result.scores["char_f1"]
        # F1 is harmonic mean, should be between min(p,r) and max(p,r)
        assert min(p, r) <= f1 + 1e-9
        assert f1 <= max(p, r) + 1e-9

    def test_details_present(self, default_normaliser):
        gt = OCRPage(full_text="ab")
        pred = OCRPage(full_text="ac")
        result = self.metric.compute(gt, pred, default_normaliser)
        assert "hits" in result.details
        assert "substitutions" in result.details

    def test_primary_key(self):
        assert self.metric.primary_key == "char_f1"
