"""
BLEU score metric (self-contained implementation).
"""

from __future__ import annotations

import math
from collections import Counter

from ocr_core.metrics.base import Metric, MetricResult
from ocr_core.normalisation import NormalisationPipeline
from ocr_core.types import OCRPage


def _get_ngrams(text: str, n: int) -> list[tuple]:
    """Get n-grams from text."""
    words = text.split()
    if len(words) < n:
        return []
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def _compute_precision(reference: str, hypothesis: str, n: int) -> float:
    """Compute n-gram precision for a single n-value."""
    ref_ngrams = _get_ngrams(reference, n)
    hyp_ngrams = _get_ngrams(hypothesis, n)

    if not hyp_ngrams:
        return 0.0

    ref_counts = Counter(ref_ngrams)
    hyp_counts = Counter(hyp_ngrams)

    # Clip counts
    clipped = {}
    for ngram, count in hyp_counts.items():
        clipped[ngram] = min(count, ref_counts.get(ngram, 0))

    matched = sum(clipped.values())
    total = len(hyp_ngrams)

    return matched / total if total > 0 else 0.0


def bleu_score(
    reference: str,
    hypothesis: str,
    max_n: int = 4,
) -> float:
    """
    Compute BLEU score between reference and hypothesis.

    BLEU = BP * exp(sum(w_i * log(p_i)))
    where BP is brevity penalty and p_i are n-gram precisions.
    """
    if not reference or not hypothesis:
        return 0.0

    # Compute precisions for n-grams 1 to max_n
    precisions = []
    for n in range(1, max_n + 1):
        p = _compute_precision(reference, hypothesis, n)
        if p == 0.0:
            return 0.0  # Standard BLEU: zero if any n-gram precision is zero
        precisions.append(p)

    # Compute brevity penalty
    ref_len = len(reference.split())
    hyp_len = len(hypothesis.split())

    if hyp_len == 0:
        bp = 0.0
    elif hyp_len < ref_len:
        bp = math.exp(1 - ref_len / hyp_len)
    else:
        bp = 1.0

    # Compute BLEU
    weights = [1 / max_n] * max_n
    log_bleu = sum(w * math.log(p) for w, p in zip(weights, precisions))
    bleu = bp * math.exp(log_bleu)

    return bleu


class BLEUMetric(Metric):
    """BLEU (Bilingual Evaluation Understudy) score."""

    name = "bleu"

    def __init__(self, max_n: int = 4, **params):
        super().__init__(**params)
        self.max_n = max_n

    def compute(
        self,
        gt_page: OCRPage,
        pred_page: OCRPage,
        normaliser: NormalisationPipeline,
    ) -> MetricResult:
        reference = normaliser.tokenise_for_wer(gt_page.full_text)
        hypothesis = normaliser.tokenise_for_wer(pred_page.full_text)

        score = bleu_score(reference, hypothesis, max_n=self.max_n)
        return MetricResult(scores={"bleu": score})
