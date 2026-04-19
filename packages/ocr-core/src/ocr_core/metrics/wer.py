from __future__ import annotations

from jiwer import process_words

from ocr_core.metrics.base import Metric, MetricResult
from ocr_core.normalisation import NormalisationPipeline
from ocr_core.types import OCRPage


class WERMetric(Metric):
    name = "wer"

    def compute(
        self, gt_page: OCRPage, pred_page: OCRPage,
        normaliser: NormalisationPipeline,
    ) -> MetricResult:
        # CJK-aware: use tokenise_for_wer which inserts spaces around each
        # CJK character so each becomes a separate "word"
        reference = normaliser.tokenise_for_wer(gt_page.full_text)
        hypothesis = normaliser.tokenise_for_wer(pred_page.full_text)

        if not reference.strip():
            return MetricResult(scores={"wer": float('nan')})
        if not hypothesis.strip():
            return MetricResult(scores={"wer": 1.0})

        out = process_words(reference, hypothesis)

        return MetricResult(
            scores={"wer": out.wer},
            details={
                "substitutions": out.substitutions,
                "deletions": out.deletions,
                "insertions": out.insertions,
            },
        )
