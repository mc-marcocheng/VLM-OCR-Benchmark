"""Tests for BLEU metric."""

from __future__ import annotations

import pytest

from ocr_core.metrics.bleu import BLEUMetric, bleu_score
from ocr_core.types import OCRPage


class TestBleuScore:
    def test_identical(self):
        text = "the cat sat on the mat"
        assert bleu_score(text, text) == pytest.approx(1.0)

    def test_empty_reference(self):
        assert bleu_score("", "some words") == 0.0

    def test_empty_hypothesis(self):
        assert bleu_score("some words", "") == 0.0

    def test_both_empty(self):
        assert bleu_score("", "") == 0.0

    def test_no_overlap(self):
        assert bleu_score("a b c d", "x y z w") == 0.0

    def test_partial_overlap(self):
        ref = "the quick brown fox jumps over the lazy dog"
        hyp = "the quick brown cat jumps over the lazy dog"
        score = bleu_score(ref, hyp)
        assert 0 < score < 1

    def test_brevity_penalty(self):
        ref = "a b c d e f g h"
        hyp = "a b c"  # much shorter
        score_short = bleu_score(ref, hyp)
        score_full = bleu_score(ref, ref)
        assert score_short < score_full

    def test_max_n_1(self):
        ref = "a b c d"
        hyp = "a b c d"
        assert bleu_score(ref, hyp, max_n=1) == pytest.approx(1.0)

    def test_single_word_matching(self):
        # Only 1 word each — needs max_n=1 otherwise higher n-gram precision = 0
        ref = "hello"
        hyp = "hello"
        assert bleu_score(ref, hyp, max_n=1) == pytest.approx(1.0)
        # max_n=4: no 2/3/4-grams exist, so bleu returns 0
        assert bleu_score(ref, hyp, max_n=4) == 0.0


class TestBLEUMetric:
    def setup_method(self):
        self.metric = BLEUMetric(max_n=4)

    def test_perfect(self, default_normaliser):
        text = "the quick brown fox jumps over the lazy dog"
        gt = OCRPage(full_text=text)
        pred = OCRPage(full_text=text)
        result = self.metric.compute(gt, pred, default_normaliser)
        assert result.scores["bleu"] == pytest.approx(1.0)

    def test_name(self):
        assert self.metric.name == "bleu"

    def test_custom_max_n(self, default_normaliser):
        m = BLEUMetric(max_n=2)
        gt = OCRPage(full_text="a b c d")
        pred = OCRPage(full_text="a b c d")
        result = m.compute(gt, pred, default_normaliser)
        assert result.scores["bleu"] == pytest.approx(1.0)
