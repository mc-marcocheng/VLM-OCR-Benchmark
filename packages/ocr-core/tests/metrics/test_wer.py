"""Tests for WER metric."""

from __future__ import annotations

import math

import pytest

from ocr_core.metrics.wer import WERMetric
from ocr_core.types import OCRPage


class TestWER:
    def setup_method(self):
        self.metric = WERMetric()

    def test_perfect_match(self, default_normaliser):
        gt = OCRPage(full_text="the cat sat on the mat")
        pred = OCRPage(full_text="the cat sat on the mat")
        result = self.metric.compute(gt, pred, default_normaliser)
        assert result.scores["wer"] == pytest.approx(0.0)

    def test_completely_wrong(self, default_normaliser):
        gt = OCRPage(full_text="a b c")
        pred = OCRPage(full_text="x y z")
        result = self.metric.compute(gt, pred, default_normaliser)
        assert result.scores["wer"] == pytest.approx(1.0)

    def test_empty_ground_truth(self, default_normaliser):
        gt = OCRPage(full_text="")
        pred = OCRPage(full_text="something")
        result = self.metric.compute(gt, pred, default_normaliser)
        assert math.isnan(result.scores["wer"])

    def test_empty_prediction(self, default_normaliser):
        gt = OCRPage(full_text="word")
        pred = OCRPage(full_text="")
        result = self.metric.compute(gt, pred, default_normaliser)
        assert result.scores["wer"] == pytest.approx(1.0)

    def test_insertion(self, default_normaliser):
        gt = OCRPage(full_text="hello world")
        pred = OCRPage(full_text="hello beautiful world")
        result = self.metric.compute(gt, pred, default_normaliser)
        # 1 insertion out of 2 ref words = 0.5
        assert result.scores["wer"] == pytest.approx(0.5)

    def test_cjk_characters_as_words(self, default_normaliser):
        gt = OCRPage(full_text="你好世界")
        pred = OCRPage(full_text="你好世界")
        result = self.metric.compute(gt, pred, default_normaliser)
        assert result.scores["wer"] == pytest.approx(0.0)
