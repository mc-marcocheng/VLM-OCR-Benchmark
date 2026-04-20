"""
Integration-style tests that combine multiple components.

These tests don't invoke real model workers but exercise the data pipeline
and metric computation end-to-end.
"""

from __future__ import annotations

import json
import os

import pytest
from PIL import Image

from ocr_core.config import MetricConfig, NormalisationConfig
from ocr_core.data_loader import DataLoader
from ocr_core.metrics.registry import MetricRegistry
from ocr_core.normalisation import NormalisationPipeline
from ocr_core.types import BBox, OCRPage, OCRRegion


def _create_test_set(base_dir: str, set_name: str, files: dict[str, str]):
    """Create a test set with images and ground truths.

    files: mapping of stem → ground truth text
    """
    input_dir = os.path.join(base_dir, "inputs", set_name)
    gt_dir = os.path.join(base_dir, "groundtruths", set_name)
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    for stem, gt_text in files.items():
        # Create a dummy image
        img = Image.new("RGB", (100, 50), "white")
        img.save(os.path.join(input_dir, f"{stem}.png"))
        # Create ground truth
        with open(os.path.join(gt_dir, f"{stem}.txt"), "w", encoding="utf-8") as f:
            f.write(gt_text)


class TestDataLoaderToMetricsPipeline:
    def test_full_pipeline(self, tmp_dir):
        _create_test_set(
            tmp_dir,
            "english",
            {
                "doc1": "The quick brown fox",
                "doc2": "Hello world",
            },
        )

        loader = DataLoader(
            input_dir=os.path.join(tmp_dir, "inputs"),
            groundtruth_dir=os.path.join(tmp_dir, "groundtruths"),
            processed_dir=os.path.join(tmp_dir, "processed"),
        )

        files = loader.list_files("english")
        assert len(files) == 2

        normaliser = NormalisationPipeline(NormalisationConfig())
        metrics = MetricRegistry.from_config(
            [
                MetricConfig(name="cer"),
                MetricConfig(name="wer"),
                MetricConfig(name="char_f1"),
            ]
        )

        for fname in files:
            paths = loader.ensure_cached("english", fname)
            assert len(paths) == 1

            gt = loader.load_ground_truth("english", fname)
            assert gt is not None

            # Simulate perfect prediction
            pred_page = OCRPage(full_text=gt.pages[0].full_text)

            for metric in metrics:
                result = metric.compute(gt.pages[0], pred_page, normaliser)
                # Perfect match → CER=0, WER=0, F1=1
                if metric.name == "cer":
                    assert result.scores["cer"] == pytest.approx(0.0)
                elif metric.name == "wer":
                    assert result.scores["wer"] == pytest.approx(0.0)
                elif metric.name == "char_f1":
                    assert result.scores["char_f1"] == pytest.approx(1.0)


class TestJsonGroundTruthWithLayoutMetrics:
    def test_layout_metrics(self, tmp_dir):
        gt_dir = os.path.join(tmp_dir, "groundtruths", "layout_test")
        os.makedirs(gt_dir)

        gt_data = {
            "pages": [
                {
                    "page_number": 1,
                    "full_text": "Title\nBody text here",
                    "regions": [
                        {
                            "text": "Title",
                            "category": "title",
                            "bbox": {"x1": 10, "y1": 10, "x2": 200, "y2": 50},
                            "order": 0,
                        },
                        {
                            "text": "Body text here",
                            "category": "text",
                            "bbox": {"x1": 10, "y1": 60, "x2": 200, "y2": 200},
                            "order": 1,
                        },
                    ],
                }
            ]
        }
        with open(os.path.join(gt_dir, "doc.json"), "w") as f:
            json.dump(gt_data, f)

        loader = DataLoader(
            input_dir=os.path.join(tmp_dir, "inputs"),
            groundtruth_dir=os.path.join(tmp_dir, "groundtruths"),
            processed_dir=os.path.join(tmp_dir, "processed"),
        )

        gt = loader.load_ground_truth("layout_test", "doc.pdf")
        assert gt is not None

        gt_page = gt.pages[0]
        assert gt_page.has_bboxes()
        assert len(gt_page.regions) == 2

        # Simulate prediction with same layout
        pred_page = OCRPage(
            page_number=1,
            full_text="Title\nBody text here",
            regions=[
                OCRRegion(
                    text="Title",
                    category="title",
                    bbox=BBox(10, 10, 200, 50),
                    order=0,
                ),
                OCRRegion(
                    text="Body text here",
                    category="text",
                    bbox=BBox(10, 60, 200, 200),
                    order=1,
                ),
            ],
        )

        normaliser = NormalisationPipeline()

        from ocr_core.metrics.layout_iou import LayoutIOUMetric
        from ocr_core.metrics.reading_order import ReadingOrderMetric

        layout_metric = LayoutIOUMetric(iou_threshold=0.5)
        assert layout_metric.is_applicable(gt_page, pred_page)
        lr = layout_metric.compute(gt_page, pred_page, normaliser)
        assert lr.scores["layout_mean_iou"] == pytest.approx(1.0)

        order_metric = ReadingOrderMetric()
        assert order_metric.is_applicable(gt_page, pred_page)
        orr = order_metric.compute(gt_page, pred_page, normaliser)
        assert orr.scores["reading_order_tau"] == pytest.approx(1.0)


class TestAllMetricsOnSamePair:
    """Ensure all built-in metrics can run on the same GT/pred pair without errors."""

    def test_all_metrics_run(self, default_normaliser):
        gt = OCRPage(
            page_number=1,
            full_text="The quick brown fox jumps over the lazy dog",
            regions=[
                OCRRegion(
                    text="The quick brown fox",
                    category="text",
                    bbox=BBox(0, 0, 200, 50),
                    order=0,
                ),
                OCRRegion(
                    text="jumps over the lazy dog",
                    category="text",
                    bbox=BBox(0, 50, 200, 100),
                    order=1,
                ),
                OCRRegion(
                    text="<table><tr><td>A</td></tr></table>",
                    category="table",
                    bbox=BBox(0, 100, 200, 200),
                    order=2,
                ),
            ],
        )
        pred = OCRPage(
            page_number=1,
            full_text="The quick brown fox jumps over the lazy dog",
            regions=[
                OCRRegion(
                    text="The quick brown fox",
                    category="text",
                    bbox=BBox(0, 0, 200, 50),
                    order=0,
                ),
                OCRRegion(
                    text="jumps over the lazy dog",
                    category="text",
                    bbox=BBox(0, 50, 200, 100),
                    order=1,
                ),
                OCRRegion(
                    text="<table><tr><td>A</td></tr></table>",
                    category="table",
                    bbox=BBox(0, 100, 200, 200),
                    order=2,
                ),
            ],
        )

        all_metric_configs = [
            MetricConfig(name="cer"),
            MetricConfig(name="wer"),
            MetricConfig(name="char_f1"),
            MetricConfig(name="bleu", params={"max_n": 4}),
            MetricConfig(name="bag_of_words"),
            MetricConfig(name="teds"),
            MetricConfig(name="layout_iou", params={"iou_threshold": 0.5}),
            MetricConfig(name="reading_order"),
        ]
        registry = MetricRegistry.from_config(all_metric_configs)

        for metric in registry:
            if metric.is_applicable(gt, pred):
                result = metric.compute(gt, pred, default_normaliser)
                assert isinstance(result.scores, dict)
                assert len(result.scores) > 0
                # For perfect match, primary scores should be optimal
                primary = result.scores.get(metric.primary_key)
                assert primary is not None
