"""Tests for bag-of-words metric."""

from __future__ import annotations

import math

import pytest

from ocr_core.metrics.bag_of_words import BagOfWordsMetric
from ocr_core.types import OCRPage


class TestBagOfWords:
    def setup_method(self):
        self.metric = BagOfWordsMetric()

    def test_perfect_match(self, default_normaliser):
        gt = OCRPage(full_text="hello world")
        pred = OCRPage(full_text="hello world")
        r = self.metric.compute(gt, pred, default_normaliser)
        assert r.scores["bow_f1"] == pytest.approx(1.0)

    def test_reordered_words(self, default_normaliser):
        gt = OCRPage(full_text="hello world foo")
        pred = OCRPage(full_text="foo hello world")
        r = self.metric.compute(gt, pred, default_normaliser)
        # Bag of words is order-insensitive → perfect
        assert r.scores["bow_f1"] == pytest.approx(1.0)

    def test_both_empty(self, default_normaliser):
        gt = OCRPage(full_text="")
        pred = OCRPage(full_text="")
        r = self.metric.compute(gt, pred, default_normaliser)
        assert r.scores["bow_f1"] == pytest.approx(1.0)

    def test_empty_ground_truth(self, default_normaliser):
        gt = OCRPage(full_text="")
        pred = OCRPage(full_text="hello")
        r = self.metric.compute(gt, pred, default_normaliser)
        assert math.isnan(r.scores["bow_f1"])

    def test_empty_prediction(self, default_normaliser):
        gt = OCRPage(full_text="hello world")
        pred = OCRPage(full_text="")
        r = self.metric.compute(gt, pred, default_normaliser)
        assert r.scores["bow_f1"] == pytest.approx(0.0)

    def test_partial_overlap(self, default_normaliser):
        gt = OCRPage(full_text="a b c d")
        pred = OCRPage(full_text="a b x y")
        r = self.metric.compute(gt, pred, default_normaliser)
        assert 0 < r.scores["bow_f1"] < 1

    def test_jaccard(self, default_normaliser):
        gt = OCRPage(full_text="a b c")
        pred = OCRPage(full_text="a b d")
        r = self.metric.compute(gt, pred, default_normaliser)
        # unique sets: {a,b,c} & {a,b,d} → intersection={a,b}, union={a,b,c,d}
        assert r.scores["bow_jaccard"] == pytest.approx(2 / 4)

    def test_duplicate_words(self, default_normaliser):
        gt = OCRPage(full_text="a a a b")
        pred = OCRPage(full_text="a a b b")
        r = self.metric.compute(gt, pred, default_normaliser)
        # overlap: min(3,2)=2 for 'a', min(1,2)=1 for 'b' → 3
        # recall = 3/4, precision = 3/4
        assert r.scores["bow_precision"] == pytest.approx(3 / 4)
        assert r.scores["bow_recall"] == pytest.approx(3 / 4)

    def test_primary_key(self):
        assert self.metric.primary_key == "bow_f1"
