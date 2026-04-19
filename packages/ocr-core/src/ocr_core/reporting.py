# packages/ocr-core/src/ocr_core/reporting.py
"""Generate Markdown and JSON reports from benchmark results."""
from __future__ import annotations

import json
import os
from typing import Sequence

from ocr_core.benchmark import BenchmarkResult
from ocr_core.metrics import MetricRegistry
from ocr_core.utils import fmt

__all__ = ["generate_markdown_report"]


def generate_markdown_report(
    results: Sequence[BenchmarkResult],
    metrics: MetricRegistry,
    output_path: str,
) -> str:
    """Write a multi-model comparison report in Markdown."""
    lines: list[str] = ["# OCR Benchmark Report\n"]

    for result in results:
        lines.append(f"## {result.model_name} — {result.test_set} ({result.device})")
        lines.append(f"*{result.timestamp}*\n")

        # Metric summary table
        header = "| Metric | Mean | Std | 95% CI |"
        sep = "|--------|------|-----|--------|"
        lines += [header, sep]
        for m in metrics:
            ss = result.score_summary(m.primary_key)
            if ss.n > 0:
                lines.append(
                    f"| {m.name.upper()} | {ss.mean:.4f} | {ss.std:.4f} | "
                    f"[{ss.ci_lower:.4f}, {ss.ci_upper:.4f}] |"
                )

        ts = result.timing_summary()
        if ts.n > 0:
            lines.append(
                f"| Speed (s/page) | {ts.mean:.4f} | {ts.std:.4f} | "
                f"[{ts.ci_lower:.4f}, {ts.ci_upper:.4f}] |"
            )

        # Resources
        measured = result.measured_runs
        if measured:
            res = measured[-1].resources
            lines.append("\n### Resources\n")
            lines.append(f"- Load time: {measured[-1].model_load_time_seconds:.2f}s")
            lines.append(f"- Peak RAM: {fmt(res.get('peak_ram_mb'), '.0f', ' MB')}")
            lines.append(f"- Peak VRAM: {fmt(res.get('peak_vram_mb'), '.0f', ' MB')}")

        # Degradation
        if result.degradation_runs:
            lines.append("\n### Degradation Robustness\n")
            header = "| Degradation |"
            sep = "|-------------|"
            for m in metrics:
                header += f" {m.name.upper()} |"
                sep += "------|"
            lines += [header, sep]
            for dr in result.degradation_runs:
                row = f"| {dr.degradation_label} |"
                for m in metrics:
                    v = dr.aggregate_score(m.primary_key)
                    row += f" {fmt(v)} |"
                lines.append(row)

        lines.append("\n---\n")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    return report
