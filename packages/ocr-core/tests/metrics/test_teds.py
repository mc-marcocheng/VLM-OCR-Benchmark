"""Tests for TEDS metric."""

from __future__ import annotations

import math

import pytest

from ocr_core.metrics.teds import TEDSMetric, parse_html_table, teds
from ocr_core.types import OCRPage, OCRRegion


class TestParseHtmlTable:
    def test_simple_table(self):
        html = "<table><tr><td>A</td><td>B</td></tr></table>"
        tree = parse_html_table(html)
        assert tree is not None
        assert tree.tag == "table"
        assert len(tree.children) == 1  # one <tr>
        assert len(tree.children[0].children) == 2  # two <td>

    def test_empty_string(self):
        tree = parse_html_table("")
        assert tree is None

    def test_text_in_cells(self):
        html = "<table><tr><td>hello</td></tr></table>"
        tree = parse_html_table(html)
        assert tree.children[0].children[0].text == "hello"

    def test_nested_tags(self):
        html = "<table><tr><td><b>bold</b></td></tr></table>"
        tree = parse_html_table(html)
        td = tree.children[0].children[0]
        assert len(td.children) == 1
        assert td.children[0].tag == "b"
        assert td.children[0].text == "bold"


class TestTedsScore:
    def test_identical(self):
        html = "<table><tr><td>A</td><td>B</td></tr></table>"
        assert teds(html, html) == pytest.approx(1.0)

    def test_completely_different(self):
        gt = "<table><tr><td>A</td><td>B</td></tr></table>"
        pred = (
            "<table><tr><td>X</td><td>Y</td><td>Z</td></tr><tr><td>W</td></tr></table>"
        )
        score = teds(gt, pred)
        assert 0 <= score <= 1

    def test_both_empty(self):
        assert teds("", "") == 1.0

    def test_one_empty(self):
        html = "<table><tr><td>A</td></tr></table>"
        assert teds(html, "") == 0.0
        assert teds("", html) == 0.0

    def test_text_difference(self):
        gt = "<table><tr><td>hello</td></tr></table>"
        pred = "<table><tr><td>world</td></tr></table>"
        score = teds(gt, pred)
        assert 0 < score < 1

    def test_structural_difference(self):
        gt = "<table><tr><td>A</td></tr></table>"
        pred = "<table><tr><td>A</td><td>B</td></tr></table>"
        score = teds(gt, pred)
        assert 0 < score < 1


class TestTEDSMetric:
    def setup_method(self):
        self.metric = TEDSMetric()

    def test_applicability_requires_tables(self):
        gt_no_table = OCRPage(regions=[OCRRegion(category="text")])
        pred = OCRPage()
        assert not self.metric.is_applicable(gt_no_table, pred)

        gt_with_table = OCRPage(regions=[OCRRegion(category="table")])
        assert self.metric.is_applicable(gt_with_table, pred)

    def test_perfect_table_match(self, default_normaliser):
        html = "<table><tr><td>A</td><td>B</td></tr></table>"
        gt = OCRPage(regions=[OCRRegion(text=html, category="table")])
        pred = OCRPage(regions=[OCRRegion(text=html, category="table")])
        r = self.metric.compute(gt, pred, default_normaliser)
        assert r.scores["teds"] == pytest.approx(1.0)

    def test_no_pred_tables(self, default_normaliser):
        html = "<table><tr><td>A</td></tr></table>"
        gt = OCRPage(regions=[OCRRegion(text=html, category="table")])
        pred = OCRPage(regions=[])
        r = self.metric.compute(gt, pred, default_normaliser)
        assert r.scores["teds"] == pytest.approx(0.0)

    def test_no_gt_tables(self, default_normaliser):
        gt = OCRPage(regions=[])
        pred = OCRPage(regions=[OCRRegion(text="<table></table>", category="table")])
        r = self.metric.compute(gt, pred, default_normaliser)
        assert math.isnan(r.scores["teds"])

    def test_multiple_tables(self, default_normaliser):
        html_a = "<table><tr><td>A</td></tr></table>"
        html_b = "<table><tr><td>B</td></tr></table>"
        gt = OCRPage(
            regions=[
                OCRRegion(text=html_a, category="table"),
                OCRRegion(text=html_b, category="table"),
            ]
        )
        pred = OCRPage(
            regions=[
                OCRRegion(text=html_a, category="table"),
                OCRRegion(text=html_b, category="table"),
            ]
        )
        r = self.metric.compute(gt, pred, default_normaliser)
        assert r.scores["teds"] == pytest.approx(1.0)
