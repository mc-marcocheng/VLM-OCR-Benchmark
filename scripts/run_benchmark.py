# scripts/run_benchmark.py
"""
CLI entry-point for running benchmarks.

Usage:
    python scripts/run_benchmark.py --model GLMOCR --test_set test_1 --device cuda
    python scripts/run_benchmark.py --model GLMOCR --test_set test_1 --config config/default.yaml
    python scripts/run_benchmark.py --model GLMOCR --test_set test_1 --report results/report.md
"""

import argparse
import sys

from loguru import logger

from ocr_core.benchmark import BenchmarkRunner
from ocr_core.config import load_config
from ocr_core.reporting import generate_markdown_report
from ocr_core.utils import resolve_device


def main():
    parser = argparse.ArgumentParser(
        description="Run OCR benchmarks.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Model name (must match config)")
    parser.add_argument("--test_set", required=True, help="Test set folder name")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "gpu"])
    parser.add_argument("--config", default="config/default.yaml", help="Config YAML")
    parser.add_argument("--report", default=None, help="Path for Markdown report")
    parser.add_argument("--runs", type=int, default=None, help="Override config runs")
    parser.add_argument("--warmup", type=int, default=None, help="Override warmup runs")
    parser.add_argument(
        "--degradation", action="store_true", help="Enable degradation testing"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI overrides
    if args.runs is not None:
        cfg.runs = args.runs
    if args.warmup is not None:
        cfg.warmup_runs = args.warmup
    if args.degradation:
        cfg.degradation.enabled = True

    device = resolve_device(args.device)

    runner = BenchmarkRunner(config=cfg)

    try:
        result = runner.run(
            model_name=args.model,
            test_set=args.test_set,
            device=device,
        )

        # Log top-level results
        for m in runner.metrics:
            ss = result.score_summary(m.primary_key)
            if ss.n > 0:
                logger.success(f"{m.name.upper()}: {ss.mean:.4f} ± {ss.std:.4f}")

        # Optional report
        if args.report:
            generate_markdown_report([result], runner.metrics, args.report)
            logger.success(f"Report → {args.report}")

    except Exception:
        logger.exception("Benchmark failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
