from __future__ import annotations

from jiwer import process_characters

from ocr_core.metrics.base import Metric, MetricResult
from ocr_core.normalisation import NormalisationPipeline
from ocr_core.types import OCRPage


class CharF1Metric(Metric):
    name = "char_f1"
    primary_key = "char_f1"

    def compute(
        self, gt_page: OCRPage, pred_page: OCRPage,
        normaliser: NormalisationPipeline,
    ) -> MetricResult:
        reference = normaliser.apply(gt_page.full_text)
        hypothesis = normaliser.apply(pred_page.full_text)

        if not reference:
            score = 1.0 if not hypothesis else 0.0
            return MetricResult(scores={
                "char_precision": score, "char_recall": score, "char_f1": score,
            })
        if not hypothesis:
            return MetricResult(scores={
                "char_precision": 0.0, "char_recall": 0.0, "char_f1": 0.0,
            })

        out = process_characters(reference, hypothesis)

        # Assert jiwer >= 3.0 (needs 'hits' field)
        if not hasattr(out, 'hits'):
            raise RuntimeError(
                "jiwer >= 3.0 required for char_f1 (needs 'hits' field)"
            )

        # out.hits = correctly matched characters
        hits = out.hits
        insertions = out.insertions
        deletions = out.deletions
        substitutions = out.substitutions

        hyp_len = hits + substitutions + insertions  # total chars in hypothesis
        ref_len = hits + deletions + substitutions   # total chars in reference

        # Precision = correctly matched / total hypothesis chars
        # Recall    = correctly matched / total reference chars
        precision = hits / hyp_len if hyp_len > 0 else 0.0
        recall    = hits / ref_len if ref_len > 0 else 0.0

        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        return MetricResult(
            scores={
                "char_precision": precision,
                "char_recall": recall,
                "char_f1": f1,
            },
            details={
                "hits": hits,
                "insertions": insertions,
                "deletions": deletions,
                "substitutions": substitutions,
            },
        )
