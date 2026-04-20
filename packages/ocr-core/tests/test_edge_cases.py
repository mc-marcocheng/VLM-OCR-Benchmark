"""
Edge-case and boundary-condition tests across multiple modules.
"""

from __future__ import annotations

import math

import pytest

from ocr_core.config import NormalisationConfig
from ocr_core.metrics.bleu import bleu_score
from ocr_core.metrics.cer import CERMetric
from ocr_core.metrics.wer import WERMetric
from ocr_core.normalisation import NormalisationPipeline
from ocr_core.types import BBox, GroundTruth, OCRPage, OCRRegion


class TestVeryLongTexts:
    def test_cer_long_text(self, default_normaliser):
        text = "a " * 10_000
        gt = OCRPage(full_text=text)
        pred = OCRPage(full_text=text)
        result = CERMetric().compute(gt, pred, default_normaliser)
        assert result.scores["cer"] == pytest.approx(0.0)

    def test_wer_long_text(self, default_normaliser):
        text = " ".join(f"word{i}" for i in range(5000))
        gt = OCRPage(full_text=text)
        pred = OCRPage(full_text=text)
        result = WERMetric().compute(gt, pred, default_normaliser)
        assert result.scores["wer"] == pytest.approx(0.0)


class TestSpecialCharacters:
    def test_unicode_emoji(self, default_normaliser):
        gt = OCRPage(full_text="Hello 🌍 World 🎉")
        pred = OCRPage(full_text="Hello 🌍 World 🎉")
        result = CERMetric().compute(gt, pred, default_normaliser)
        assert result.scores["cer"] == pytest.approx(0.0)

    def test_newlines_and_tabs(self, default_normaliser):
        gt = OCRPage(full_text="line1\nline2\tline3")
        pred = OCRPage(full_text="line1\nline2\tline3")
        result = CERMetric().compute(gt, pred, default_normaliser)
        assert result.scores["cer"] == pytest.approx(0.0)

    def test_only_whitespace(self, default_normaliser):
        gt = OCRPage(full_text="   ")
        pred = OCRPage(full_text="   ")
        # After normalisation, both become "" → nan
        result = CERMetric().compute(gt, pred, default_normaliser)
        assert math.isnan(result.scores["cer"])

    def test_mixed_scripts(self, default_normaliser):
        gt = OCRPage(full_text="Hello مرحبا こんにちは 你好")
        pred = OCRPage(full_text="Hello مرحبا こんにちは 你好")
        result = CERMetric().compute(gt, pred, default_normaliser)
        assert result.scores["cer"] == pytest.approx(0.0)


class TestBoundaryBBoxes:
    def test_touching_boxes(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(10, 0, 20, 10)
        assert a.iou(b) == 0.0

    def test_one_pixel_overlap(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(9, 9, 20, 20)
        iou = a.iou(b)
        assert iou > 0
        assert iou < 0.01

    def test_contained_box(self):
        outer = BBox(0, 0, 100, 100)
        inner = BBox(25, 25, 75, 75)
        iou = outer.iou(inner)
        # inner area = 2500, outer area = 10000, inter = 2500
        # union = 10000 + 2500 - 2500 = 10000
        assert iou == pytest.approx(2500 / 10000)

    def test_large_coordinates(self):
        a = BBox(0, 0, 1e6, 1e6)
        b = BBox(0, 0, 1e6, 1e6)
        assert a.iou(b) == pytest.approx(1.0)

    def test_float_coordinates(self):
        a = BBox(0.5, 0.5, 10.5, 10.5)
        b = BBox(0.5, 0.5, 10.5, 10.5)
        assert a.iou(b) == pytest.approx(1.0)


class TestNormalisationWithMetrics:
    """Test that normalisation is properly applied before metric computation."""

    def test_case_insensitive_cer(self):
        normaliser = NormalisationPipeline(NormalisationConfig(lowercase=True))
        gt = OCRPage(full_text="HELLO")
        pred = OCRPage(full_text="hello")
        result = CERMetric().compute(gt, pred, normaliser)
        assert result.scores["cer"] == pytest.approx(0.0)

    def test_punctuation_ignored(self):
        normaliser = NormalisationPipeline(NormalisationConfig(remove_punctuation=True))
        gt = OCRPage(full_text="Hello, world!")
        pred = OCRPage(full_text="Hello world")
        result = CERMetric().compute(gt, pred, normaliser)
        assert result.scores["cer"] == pytest.approx(0.0)

    def test_whitespace_normalised(self):
        normaliser = NormalisationPipeline(
            NormalisationConfig(collapse_whitespace=True, strip_whitespace=True)
        )
        gt = OCRPage(full_text="  hello    world  ")
        pred = OCRPage(full_text="hello world")
        result = CERMetric().compute(gt, pred, normaliser)
        assert result.scores["cer"] == pytest.approx(0.0)


class TestBleuEdgeCases:
    def test_single_word_match(self):
        assert bleu_score("cat", "cat", max_n=1) == pytest.approx(1.0)

    def test_hypothesis_longer_than_reference(self):
        ref = "a b"
        hyp = "a b c d e f"
        score = bleu_score(ref, hyp, max_n=1)
        # No brevity penalty (hyp longer), unigram precision = 2/6
        assert score == pytest.approx(2 / 6)

    def test_repeated_words(self):
        ref = "a a a a"
        hyp = "a a a a"
        assert bleu_score(ref, hyp, max_n=1) == pytest.approx(1.0)


class TestGroundTruthEdgeCases:
    def test_single_empty_page(self):
        gt = GroundTruth(pages=[OCRPage(full_text="")])
        assert gt.full_text == ""

    def test_many_pages(self):
        pages = [OCRPage(full_text=f"p{i}") for i in range(100)]
        gt = GroundTruth(pages=pages)
        assert gt.full_text.count("\n\n") == 99

    def test_roundtrip_preserves_nested_regions(self):
        region = OCRRegion(
            text="outer",
            children=[
                OCRRegion(
                    text="inner",
                    children=[OCRRegion(text="innermost")],
                )
            ],
        )
        page = OCRPage(regions=[region])
        gt = GroundTruth(pages=[page])
        d = gt.to_dict()
        restored = GroundTruth.from_dict(d)
        assert restored.pages[0].regions[0].children[0].children[0].text == "innermost"
