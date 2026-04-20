"""Tests for ocr_core.reporting."""

from __future__ import annotations

import os

import pytest

from ocr_core.benchmark import BenchmarkResult, PageResult, RunResult
from ocr_core.config import MetricConfig
from ocr_core.metrics.registry import MetricRegistry
from ocr_core.reporting import generate_markdown_report


@pytest.fixture
def sample_result():
    pr = PageResult(
        file="test.png",
        page=1,
        prediction_time_seconds=0.5,
        predicted_text="hello world",
        ground_truth_text="hello world",
        scores={"cer": 0.0, "wer": 0.0},
    )
    rr = RunResult(
        run_index=0,
        is_warmup=False,
        model_load_time_seconds=2.0,
        total_prediction_time_seconds=0.5,
        resources={
            "peak_ram_mb": 1024.0,
            "peak_vram_mb": 2048.0,
        },
        page_results=[pr],
    )
    return BenchmarkResult(
        model_name="TestModel",
        test_set="test_set",
        device="cuda",
        timestamp="2025-01-01T00:00:00",
        runs=[rr],
    )


@pytest.fixture
def sample_metrics():
    return MetricRegistry.from_config(
        [MetricConfig(name="cer"), MetricConfig(name="wer")]
    )


class TestGenerateMarkdownReport:
    def test_basic_report(self, sample_result, sample_metrics, tmp_dir):
        path = os.path.join(tmp_dir, "report.md")
        report = generate_markdown_report([sample_result], sample_metrics, path)
        assert os.path.isfile(path)
        assert "TestModel" in report
        assert "test_set" in report
        assert "cuda" in report
        assert "CER" in report
        assert "WER" in report

    def test_empty_results(self, sample_metrics, tmp_dir):
        path = os.path.join(tmp_dir, "empty_report.md")
        report = generate_markdown_report([], sample_metrics, path)
        assert "# OCR Benchmark Report" in report

    def test_creates_directory(self, sample_result, sample_metrics, tmp_dir):
        path = os.path.join(tmp_dir, "subdir", "nested", "report.md")
        generate_markdown_report([sample_result], sample_metrics, path)
        assert os.path.isfile(path)

    def test_multiple_results(self, sample_result, sample_metrics, tmp_dir):
        result2 = BenchmarkResult(
            model_name="Model2",
            test_set="test_set",
            device="cpu",
            timestamp="2025-01-02",
            runs=[
                RunResult(
                    run_index=0,
                    is_warmup=False,
                    page_results=[PageResult(scores={"cer": 0.1, "wer": 0.2})],
                    resources={},
                )
            ],
        )
        path = os.path.join(tmp_dir, "multi.md")
        report = generate_markdown_report(
            [sample_result, result2], sample_metrics, path
        )
        assert "TestModel" in report
        assert "Model2" in report

    def test_degradation_section(self, sample_metrics, tmp_dir):
        dr = RunResult(
            run_index=0,
            is_warmup=False,
            degradation_label="noise_sigma=25",
            page_results=[PageResult(scores={"cer": 0.15, "wer": 0.25})],
            resources={},
        )
        result = BenchmarkResult(
            model_name="M",
            test_set="s",
            device="cpu",
            timestamp="t",
            runs=[
                RunResult(
                    is_warmup=False,
                    page_results=[PageResult(scores={"cer": 0.0, "wer": 0.0})],
                    resources={},
                )
            ],
            degradation_runs=[dr],
        )
        path = os.path.join(tmp_dir, "deg.md")
        report = generate_markdown_report([result], sample_metrics, path)
        assert "noise_sigma=25" in report
        assert "Degradation" in report
