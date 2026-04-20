"""Tests for the metric registry."""

from __future__ import annotations

from ocr_core.config import MetricConfig
from ocr_core.metrics.base import Metric, MetricResult
from ocr_core.metrics.registry import MetricRegistry
from ocr_core.types import OCRPage


class TestMetricRegistry:
    def test_from_config_builtin(self):
        configs = [MetricConfig(name="cer"), MetricConfig(name="wer")]
        reg = MetricRegistry.from_config(configs)
        assert len(reg) == 2
        assert reg.names == ["cer", "wer"]

    def test_from_config_unknown_skipped(self, suppress_loguru):
        configs = [MetricConfig(name="nonexistent"), MetricConfig(name="cer")]
        reg = MetricRegistry.from_config(configs)
        assert len(reg) == 1
        assert reg.names == ["cer"]

    def test_from_config_with_params(self):
        configs = [MetricConfig(name="bleu", params={"max_n": 2})]
        reg = MetricRegistry.from_config(configs)
        assert len(reg) == 1
        m = list(reg)[0]
        assert m.max_n == 2

    def test_iter(self):
        configs = [MetricConfig(name="cer"), MetricConfig(name="wer")]
        reg = MetricRegistry.from_config(configs)
        metrics_list = list(reg)
        assert len(metrics_list) == 2

    def test_register_custom(self):
        class DummyMetric(Metric):
            name = "dummy"

            def compute(self, gt, pred, norm):
                return MetricResult(scores={"dummy": 1.0})

        MetricRegistry.register_custom("dummy", DummyMetric)
        reg = MetricRegistry.from_config([MetricConfig(name="dummy")])
        assert len(reg) == 1
        assert reg.names == ["dummy"]

    def test_empty_config(self):
        reg = MetricRegistry.from_config([])
        assert len(reg) == 0

    def test_all_builtins(self):
        """Ensure every built-in metric can be instantiated."""
        names = [
            "cer",
            "wer",
            "char_f1",
            "bleu",
            "bag_of_words",
            "teds",
            "layout_iou",
            "reading_order",
        ]
        configs = [MetricConfig(name=n) for n in names]
        reg = MetricRegistry.from_config(configs)
        assert len(reg) == len(names)


class TestMetricApplicability:
    def test_default_always_applicable(self, default_normaliser):
        from ocr_core.metrics.cer import CERMetric

        m = CERMetric()
        gt = OCRPage(full_text="a")
        pred = OCRPage(full_text="b")
        assert m.is_applicable(gt, pred)

    def test_apply_to_filter(self, default_normaliser):
        from ocr_core.metrics.cer import CERMetric
        from ocr_core.types import OCRRegion

        m = CERMetric(apply_to=["table"])
        gt_no_table = OCRPage(regions=[OCRRegion(category="text")])
        assert not m.is_applicable(gt_no_table, OCRPage())

        gt_with_table = OCRPage(regions=[OCRRegion(category="table")])
        assert m.is_applicable(gt_with_table, OCRPage())
