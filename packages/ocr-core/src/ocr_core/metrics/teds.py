"""
Tree-Edit-Distance-based Similarity (TEDS) for HTML tables.

Reference:
    Zhong, ShaoLab & Jimeno Yepes (2020) — PubTabNet / ICDAR 2019.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser

import numpy as np

from ocr_core.metrics.base import Metric, MetricResult
from ocr_core.metrics.layout_iou import _hungarian_match
from ocr_core.normalisation import NormalisationPipeline
from ocr_core.types import OCRPage

# ── HTML → tree ─────────────────────────────────────────────


@dataclass
class TreeNode:
    tag: str = ""
    text: str = ""
    children: list[TreeNode] = field(default_factory=list)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)


class _TableParser(HTMLParser):
    """Parse an HTML table string into a ``TreeNode`` tree."""

    def __init__(self):
        super().__init__()
        self.root: TreeNode | None = None
        self._stack: list[TreeNode] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        node = TreeNode(tag=tag)
        if self._stack:
            self._stack[-1].children.append(node)
        else:
            self.root = node
        self._stack.append(node)

    def handle_endtag(self, tag: str):
        # Pop back to the matching tag (tolerant of interleaved tags)
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i].tag == tag:
                self._stack = self._stack[:i]
                return

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]):
        """Handle self-closing tags like <br/> or <img/>."""
        node = TreeNode(tag=tag)
        if self._stack:
            self._stack[-1].children.append(node)
        elif self.root is None:
            self.root = node
        else:
            # Document has multiple roots. Create a dummy wrapping root.
            dummy = TreeNode(tag="root")
            dummy.children.extend([self.root, node])
            self.root = dummy
            self._stack.append(dummy)

        self._stack.append(node)

    def handle_data(self, data: str):
        text = data.strip()
        if text and self._stack:
            node = self._stack[-1]
            if node.text:
                node.text += " " + text
            else:
                node.text = text


def parse_html_table(html: str) -> TreeNode | None:
    parser = _TableParser()
    try:
        parser.feed(html)
    except Exception:
        return None
    return parser.root


# ── Tree-edit distance (Zhang-Shasha simplified) ───────────


def _tree_edit_distance(t1: TreeNode | None, t2: TreeNode | None) -> int:
    """
    Tree alignment distance with memoisation.

    Children are aligned via DP preserving left-to-right order
    (equivalent to constrained edit distance on ordered trees).
    This matches the standard TEDS formulation for HTML tables.
    """
    if t1 is None and t2 is None:
        return 0
    if t1 is None:
        return t2.size()  # type: ignore[union-attr]
    if t2 is None:
        return t1.size()

    cache: dict[tuple[int, int], int] = {}

    def _dist(a: TreeNode, b: TreeNode) -> int:
        key = (id(a), id(b))
        if key in cache:
            return cache[key]

        # Cost of renaming root
        rename_cost = 0
        if a.tag != b.tag:
            rename_cost += 1
        if a.text != b.text:
            rename_cost += 1

        # Align children via DP
        m, n = len(a.children), len(b.children)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] + a.children[i - 1].size()
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] + b.children[j - 1].size()

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                del_cost = dp[i - 1][j] + a.children[i - 1].size()
                ins_cost = dp[i][j - 1] + b.children[j - 1].size()
                sub_cost = dp[i - 1][j - 1] + _dist(
                    a.children[i - 1], b.children[j - 1]
                )
                dp[i][j] = min(del_cost, ins_cost, sub_cost)

        result = rename_cost + dp[m][n]
        cache[key] = result
        return result

    return _dist(t1, t2)


def teds(html_gt: str, html_pred: str) -> float:
    """
    TEDS = 1 − (tree_edit_distance / max(|T_gt|, |T_pred|)).
    Returns a value in [0, 1], where 1 is a perfect match.
    """
    t1 = parse_html_table(html_gt)
    t2 = parse_html_table(html_pred)
    if t1 is None and t2 is None:
        return 1.0
    if t1 is None or t2 is None:
        return 0.0

    dist = _tree_edit_distance(t1, t2)
    max_size = max(t1.size(), t2.size())
    return max(0.0, 1.0 - dist / max_size) if max_size > 0 else 1.0


# ── Metric class ────────────────────────────────────────────


class TEDSMetric(Metric):
    name = "teds"

    def is_applicable(self, gt_page: OCRPage, pred_page: OCRPage) -> bool:
        return bool(gt_page.regions_by_category("table"))

    def compute(
        self,
        gt_page: OCRPage,
        pred_page: OCRPage,
        normaliser: NormalisationPipeline,
    ) -> MetricResult:
        gt_tables = gt_page.regions_by_category("table")
        pred_tables = pred_page.regions_by_category("table")

        if not gt_tables:
            return MetricResult(scores={"teds": float("nan")})
        if not pred_tables:
            return MetricResult(scores={"teds": 0.0})

        n_gt, n_pred = len(gt_tables), len(pred_tables)
        score_mat = np.zeros((n_gt, n_pred))
        for i, gt_t in enumerate(gt_tables):
            for j, pred_t in enumerate(pred_tables):
                score_mat[i, j] = teds(gt_t.text, pred_t.text)

        matches = _hungarian_match(score_mat)
        scores: list[float] = [score_mat[i, j] for i, j in matches]

        # Unmatched GT tables score 0.0
        unmatched_gt = n_gt - len(matches)
        scores.extend([0.0] * unmatched_gt)

        avg_teds = sum(scores) / len(scores) if scores else 0.0
        return MetricResult(
            scores={"teds": avg_teds},
            details={"per_table_teds": scores},
        )
