"""Central registry that maps metric names to classes."""

from __future__ import annotations

from typing import Sequence

from loguru import logger
from ocr_core.config import MetricConfig
from ocr_core.metrics.bag_of_words import BagOfWordsMetric
from ocr_core.metrics.base import Metric
from ocr_core.metrics.bleu import BLEUMetric
from ocr_core.metrics.cer import CERMetric
from ocr_core.metrics.char_f1 import CharF1Metric
from ocr_core.metrics.layout_iou import LayoutIOUMetric
from ocr_core.metrics.reading_order import ReadingOrderMetric
from ocr_core.metrics.teds import TEDSMetric
from ocr_core.metrics.wer import WERMetric

_BUILTIN: dict[str, type[Metric]] = {
    "cer": CERMetric,
    "wer": WERMetric,
    "char_f1": CharF1Metric,
    "bleu": BLEUMetric,
    "bag_of_words": BagOfWordsMetric,
    "teds": TEDSMetric,
    "layout_iou": LayoutIOUMetric,
    "reading_order": ReadingOrderMetric,
}


class MetricRegistry:
    """Instantiate and hold a set of ``Metric`` objects."""

    def __init__(self):
        self._metrics: list[Metric] = []

    def register(self, metric: Metric) -> None:
        self._metrics.append(metric)

    @classmethod
    def register_custom(cls, name: str, metric_class: type[Metric]) -> None:
        _BUILTIN[name] = metric_class

    @classmethod
    def from_config(cls, metric_configs: Sequence[MetricConfig]) -> MetricRegistry:
        reg = cls()
        for mc in metric_configs:
            klass = _BUILTIN.get(mc.name)
            if klass is None:
                logger.warning(f"Unknown metric: {mc.name!r}")
                continue
            reg.register(klass(apply_to=mc.apply_to, **mc.params))
        return reg

    def __iter__(self):
        return iter(self._metrics)

    def __len__(self):
        return len(self._metrics)

    @property
    def names(self) -> list[str]:
        return [m.name for m in self._metrics]
