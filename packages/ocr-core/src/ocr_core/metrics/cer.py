from __future__ import annotations

from jiwer import process_characters
from ocr_core.metrics.base import Metric, MetricResult
from ocr_core.normalisation import NormalisationPipeline
from ocr_core.types import OCRPage


class CERMetric(Metric):
    name = "cer"

    def compute(
        self,
        gt_page: OCRPage,
        pred_page: OCRPage,
        normaliser: NormalisationPipeline,
    ) -> MetricResult:
        reference = normaliser.apply(gt_page.full_text)
        hypothesis = normaliser.apply(pred_page.full_text)

        if not reference:
            return MetricResult(scores={"cer": float("nan")})
        if not hypothesis:
            return MetricResult(scores={"cer": 1.0})

        out = process_characters(reference, hypothesis)

        return MetricResult(
            scores={"cer": out.cer},
            details={
                "substitutions": out.substitutions,
                "deletions": out.deletions,
                "insertions": out.insertions,
            },
        )
