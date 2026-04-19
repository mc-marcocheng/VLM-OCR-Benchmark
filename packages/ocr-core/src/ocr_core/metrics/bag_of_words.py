"""Order-insensitive word overlap (Jaccard + precision/recall)."""
from __future__ import annotations

from collections import Counter

from ocr_core.metrics.base import Metric, MetricResult
from ocr_core.normalisation import NormalisationPipeline
from ocr_core.types import OCRPage


class BagOfWordsMetric(Metric):
    name = "bag_of_words"
    primary_key = "bow_f1"

    def compute(
        self, gt_page: OCRPage, pred_page: OCRPage,
        normaliser: NormalisationPipeline,
    ) -> MetricResult:
        ref = normaliser.tokenise_for_wer(gt_page.full_text).split()
        hyp = normaliser.tokenise_for_wer(pred_page.full_text).split()

        if not ref and not hyp:
            return MetricResult(scores={
                "bow_precision": 1.0, "bow_recall": 1.0,
                "bow_f1": 1.0, "bow_jaccard": 1.0,
            })
        if not ref:
            return MetricResult(scores={
                "bow_precision": 0.0, "bow_recall": float('nan'),
                "bow_f1": float('nan'), "bow_jaccard": 0.0,
            })

        ref_c = Counter(ref)
        hyp_c = Counter(hyp)
        overlap = sum((ref_c & hyp_c).values())

        precision = overlap / sum(hyp_c.values()) if hyp_c else 0.0
        recall = overlap / sum(ref_c.values())
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        ref_set = set(ref)
        hyp_set = set(hyp)
        jaccard = (len(ref_set & hyp_set) / len(ref_set | hyp_set)
                   if ref_set | hyp_set else 0.0)

        return MetricResult(scores={
            "bow_precision": precision,
            "bow_recall": recall,
            "bow_f1": f1,
            "bow_jaccard": jaccard,
        })
