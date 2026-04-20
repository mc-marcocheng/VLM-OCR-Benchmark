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
def sample_page_result():
    return PageResult(
        file="test.pdf",
        page=1,
        prediction_time_seconds=0.5,
        predicted_text="hello world",
        ground_truth_text="hello world",
        scores={"cer": 0.0, "wer": 0.0},
    )


@pytest.fixture
def sample_run_result(sample_page_result):
    return RunResult(
        run_index=0,
        is_warmup=False,
        model_load_time_seconds=1.5,
        total_prediction_time_seconds=0.5,
        resources={
            "ram_before_load_mb": 500.0,
            "ram_after_load_mb": 800.0,
            "peak_ram_mb": 1000.0,
            "vram_before_load_mb": 100.0,
            "vram_after_load_mb": 500.0,
            "peak_vram_mb": 2000.0,
        },
        page_results=[sample_page_result],
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

    def test_report_contains_resource_metrics(
        self, sample_run_result, sample_metrics, tmp_dir
    ):
        """Test that resource metrics are included in the report."""
        result = BenchmarkResult(
            model_name="TestModel",
            test_set="test_set",
            device="cuda",
            timestamp="2024-01-01T00:00:00",
            runs=[sample_run_result],
        )
        path = os.path.join(tmp_dir, "report.md")
        report = generate_markdown_report([result], sample_metrics, path)
        assert "Load time" in report
        assert "Peak RAM" in report
        assert "Peak VRAM" in report

    def test_report_with_degradation_robustness(
        self, sample_page_result, sample_metrics, tmp_dir
    ):
        """Test markdown report with degradation results."""
        measured_run = RunResult(
            run_index=0,
            is_warmup=False,
            model_load_time_seconds=1.0,
            page_results=[sample_page_result],
            resources={"peak_ram_mb": 1000.0, "peak_vram_mb": None},
        )
        degradation_run = RunResult(
            run_index=0,
            is_warmup=False,
            page_results=[
                PageResult(file="test.pdf", page=1, scores={"cer": 0.15, "wer": 0.2})
            ],
            degradation_label="noise_sigma=25",
        )
        result = BenchmarkResult(
            model_name="TestModel",
            test_set="test_set",
            device="cpu",
            timestamp="2024-01-01T00:00:00",
            runs=[measured_run],
            degradation_runs=[degradation_run],
        )
        path = os.path.join(tmp_dir, "report.md")
        report = generate_markdown_report([result], sample_metrics, path)
        assert "Degradation Robustness" in report
        assert "noise_sigma=25" in report

    def test_report_with_only_warmup_runs(self, sample_metrics, tmp_dir):
        """Test report when no measured runs exist (only warmup)."""
        warmup_run = RunResult(run_index=0, is_warmup=True, page_results=[])
        result = BenchmarkResult(
            model_name="TestModel",
            test_set="test_set",
            device="cpu",
            timestamp="2024-01-01T00:00:00",
            runs=[warmup_run],
        )
        path = os.path.join(tmp_dir, "report.md")
        report = generate_markdown_report([result], sample_metrics, path)
        assert "TestModel" in report

    def test_report_with_nan_scores(self, sample_metrics, tmp_dir):
        """Test report handles NaN scores gracefully."""
        page_result = PageResult(
            file="test.pdf",
            page=1,
            scores={"cer": float("nan"), "wer": float("nan")},
        )
        run_result = RunResult(
            run_index=0,
            is_warmup=False,
            page_results=[page_result],
            resources={},
        )
        result = BenchmarkResult(
            model_name="TestModel",
            test_set="test_set",
            device="cpu",
            timestamp="2024-01-01",
            runs=[run_result],
        )
        path = os.path.join(tmp_dir, "report.md")
        # Should not raise
        report = generate_markdown_report([result], sample_metrics, path)
        assert "TestModel" in report
