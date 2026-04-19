"""OCR metrics for benchmarking."""

from ocr_core.metrics.base import Metric, MetricResult
from ocr_core.metrics.registry import MetricRegistry
from ocr_core.metrics.cer import CERMetric
from ocr_core.metrics.wer import WERMetric
from ocr_core.metrics.char_f1 import CharF1Metric
from ocr_core.metrics.bleu import BLEUMetric
from ocr_core.metrics.bag_of_words import BagOfWordsMetric
from ocr_core.metrics.teds import TEDSMetric
from ocr_core.metrics.layout_iou import LayoutIOUMetric
from ocr_core.metrics.reading_order import ReadingOrderMetric

__all__ = [
    "Metric",
    "MetricResult",
    "MetricRegistry",
    "CERMetric",
    "WERMetric",
    "CharF1Metric",
    "BLEUMetric",
    "BagOfWordsMetric",
    "TEDSMetric",
    "LayoutIOUMetric",
    "ReadingOrderMetric",
]
