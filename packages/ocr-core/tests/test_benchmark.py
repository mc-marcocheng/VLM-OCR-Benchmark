"""Tests for ocr_core.benchmark — data classes and helpers."""

from __future__ import annotations

import math
import os

import pytest

from ocr_core.benchmark import BenchmarkResult, BenchmarkRunner, PageResult, RunResult
from ocr_core.types import BBox, OCRPage, OCRRegion


class TestPageResult:
    def test_to_dict_basic(self):
        pr = PageResult(
            file="doc.pdf",
            page=2,
            prediction_time_seconds=1.5,
            predicted_text="hello",
            ground_truth_text="hello",
            scores={"cer": 0.0},
        )
        d = pr.to_dict()
        assert d["file"] == "doc.pdf"
        assert d["page"] == 2
        assert d["scores"]["cer"] == 0.0

    def test_to_dict_no_regions(self):
        pr = PageResult()
        d = pr.to_dict()
        assert "predicted_regions" not in d

    def test_to_dict_with_regions(self):
        from ocr_core.types import OCRPage, OCRRegion

        page = OCRPage(regions=[OCRRegion(text="r1")])
        pr = PageResult(predicted_page=page)
        d = pr.to_dict()
        assert "predicted_regions" in d
        assert len(d["predicted_regions"]) == 1

    def test_to_dict_with_regions_detailed(self):
        """Test to_dict includes regions when predicted_page has them."""
        region = OCRRegion(
            text="Hello",
            category="text",
            bbox=BBox(0, 0, 100, 50),
        )
        pred_page = OCRPage(
            page_number=1,
            full_text="Hello",
            regions=[region],
        )
        pr = PageResult(
            file="test.pdf",
            page=1,
            predicted_text="Hello",
            predicted_page=pred_page,
        )
        d = pr.to_dict()
        assert "predicted_regions" in d
        assert len(d["predicted_regions"]) == 1
        assert d["predicted_regions"][0]["text"] == "Hello"

    def test_to_dict_without_regions(self):
        """Test to_dict excludes regions when none exist."""
        pred_page = OCRPage(page_number=1, full_text="Hello", regions=[])
        pr = PageResult(
            file="test.pdf",
            page=1,
            predicted_page=pred_page,
        )
        d = pr.to_dict()
        assert "predicted_regions" not in d


class TestRunResult:
    def test_n_pages(self):
        rr = RunResult(page_results=[PageResult(), PageResult(), PageResult()])
        assert rr.n_pages == 3

    def test_n_pages_empty(self):
        rr = RunResult()
        assert rr.n_pages == 0

    def test_aggregate_score(self):
        rr = RunResult(
            page_results=[
                PageResult(scores={"cer": 0.1}),
                PageResult(scores={"cer": 0.3}),
            ]
        )
        assert rr.aggregate_score("cer") == pytest.approx(0.2)

    def test_aggregate_score_missing(self):
        rr = RunResult(page_results=[PageResult(scores={})])
        assert math.isnan(rr.aggregate_score("cer"))

    def test_aggregate_score_nan_skipped(self):
        rr = RunResult(
            page_results=[
                PageResult(scores={"cer": float("nan")}),
                PageResult(scores={"cer": 0.5}),
            ]
        )
        assert rr.aggregate_score("cer") == pytest.approx(0.5)

    def test_aggregate_score_all_nan(self):
        rr = RunResult(
            page_results=[
                PageResult(scores={"cer": float("nan")}),
            ]
        )
        assert math.isnan(rr.aggregate_score("cer"))

    def test_aggregate_score_basic(self):
        rr = RunResult(
            run_index=0,
            page_results=[
                PageResult(file="a.pdf", page=1, scores={"cer": 0.1}),
                PageResult(file="a.pdf", page=2, scores={"cer": 0.2}),
                PageResult(file="b.pdf", page=1, scores={"cer": 0.3}),
            ],
        )
        assert rr.aggregate_score("cer") == pytest.approx(0.2)

    def test_aggregate_score_with_nan(self):
        """NaN values should be excluded from aggregation."""
        rr = RunResult(
            run_index=0,
            page_results=[
                PageResult(file="a.pdf", page=1, scores={"cer": 0.1}),
                PageResult(file="a.pdf", page=2, scores={"cer": float("nan")}),
                PageResult(file="b.pdf", page=1, scores={"cer": 0.3}),
            ],
        )
        assert rr.aggregate_score("cer") == pytest.approx(0.2)

    def test_aggregate_score_empty_results(self):
        rr = RunResult(run_index=0, page_results=[])
        result = rr.aggregate_score("cer")
        assert math.isnan(result)


class TestBenchmarkResult:
    def _make_result(self, warmup=1, measured=2) -> BenchmarkResult:
        runs = []
        for i in range(warmup):
            runs.append(
                RunResult(
                    run_index=i,
                    is_warmup=True,
                    page_results=[PageResult(scores={"cer": 0.99})],
                )
            )
        for i in range(measured):
            runs.append(
                RunResult(
                    run_index=warmup + i,
                    is_warmup=False,
                    total_prediction_time_seconds=1.0,
                    page_results=[
                        PageResult(
                            prediction_time_seconds=0.5,
                            scores={"cer": 0.1 * (i + 1)},
                        )
                    ],
                )
            )
        return BenchmarkResult(
            model_name="M",
            test_set="s",
            device="cpu",
            runs=runs,
        )

    def test_measured_runs_excludes_warmup(self):
        br = self._make_result(warmup=2, measured=3)
        assert len(br.measured_runs) == 3

    def test_score_summary(self):
        br = self._make_result(warmup=0, measured=3)
        ss = br.score_summary("cer")
        assert ss.n == 3
        assert ss.mean > 0

    def test_score_summary_no_measured(self):
        br = BenchmarkResult(runs=[RunResult(is_warmup=True)])
        ss = br.score_summary("cer")
        assert ss.n == 0

    def test_timing_summary(self):
        br = self._make_result(warmup=0, measured=2)
        ts = br.timing_summary()
        assert ts.n == 2
        assert ts.mean == pytest.approx(1.0)  # 1.0s / 1 page

    def test_timing_summary_no_pages(self):
        br = BenchmarkResult(runs=[RunResult(is_warmup=False, page_results=[])])
        ts = br.timing_summary()
        assert ts.n == 0

    def test_measured_runs(self):
        result = BenchmarkResult(
            model_name="Test",
            runs=[
                RunResult(run_index=0, is_warmup=True),
                RunResult(run_index=1, is_warmup=False),
                RunResult(run_index=2, is_warmup=False),
            ],
        )
        measured = result.measured_runs
        assert len(measured) == 2
        assert all(not r.is_warmup for r in measured)

    def test_score_summary_basic(self):
        result = BenchmarkResult(
            model_name="Test",
            runs=[
                RunResult(
                    run_index=0,
                    is_warmup=False,
                    page_results=[PageResult(scores={"cer": 0.1})],
                ),
                RunResult(
                    run_index=1,
                    is_warmup=False,
                    page_results=[PageResult(scores={"cer": 0.2})],
                ),
            ],
        )
        ss = result.score_summary("cer")
        assert ss.n == 2
        assert ss.mean == pytest.approx(0.15)

    def test_score_summary_empty(self):
        result = BenchmarkResult(model_name="Test", runs=[])
        ss = result.score_summary("cer")
        assert ss.n == 0

    def test_timing_summary_detailed(self):
        result = BenchmarkResult(
            model_name="Test",
            runs=[
                RunResult(
                    run_index=0,
                    is_warmup=False,
                    total_prediction_time_seconds=1.0,
                    page_results=[PageResult(), PageResult()],
                ),
                RunResult(
                    run_index=1,
                    is_warmup=False,
                    total_prediction_time_seconds=2.0,
                    page_results=[PageResult(), PageResult()],
                ),
            ],
        )
        ts = result.timing_summary()
        assert ts.n == 2
        # (1.0/2 + 2.0/2) / 2 = 0.75
        assert ts.mean == pytest.approx(0.75)


class TestBenchmarkRunnerInit:
    def test_creates_results_dir(self, minimal_config):
        BenchmarkRunner(config=minimal_config)
        assert os.path.isdir(minimal_config.data.results_dir)

    def test_unknown_model_raises_on_invoke(self, minimal_config):
        runner = BenchmarkRunner(config=minimal_config)
        with pytest.raises(ValueError, match="not in config"):
            runner._invoke_worker("NonExistent", "cpu", ["/fake.png"])


class TestBenchmarkRunnerCompare:
    def test_compare_identical(self):
        pages = [
            PageResult(file="a.png", page=1, scores={"cer": 0.1}),
            PageResult(file="b.png", page=1, scores={"cer": 0.2}),
            PageResult(file="c.png", page=1, scores={"cer": 0.3}),
        ]
        rr = RunResult(is_warmup=False, page_results=pages)
        br = BenchmarkResult(runs=[rr])
        result = BenchmarkRunner.compare(br, br, "cer")
        assert result["p_value"] == pytest.approx(1.0)
        assert result["n_paired_samples"] == 3

    def test_compare_different(self):
        pages_a = [
            PageResult(file=f"f{i}.png", page=1, scores={"cer": 0.1}) for i in range(20)
        ]
        pages_b = [
            PageResult(file=f"f{i}.png", page=1, scores={"cer": 0.9}) for i in range(20)
        ]
        rr_a = RunResult(is_warmup=False, page_results=pages_a)
        rr_b = RunResult(is_warmup=False, page_results=pages_b)
        br_a = BenchmarkResult(runs=[rr_a])
        br_b = BenchmarkResult(runs=[rr_b])
        result = BenchmarkRunner.compare(br_a, br_b, "cer")
        assert result["significant_at_005"] is True

    def test_compare_not_enough_samples(self):
        rr = RunResult(
            is_warmup=False,
            page_results=[PageResult(file="a.png", page=1, scores={"cer": 0.5})],
        )
        br = BenchmarkResult(runs=[rr])
        br2 = BenchmarkResult(runs=[RunResult(is_warmup=False, page_results=[])])
        result = BenchmarkRunner.compare(br, br2, "cer")
        assert "error" in result

    def test_compare_with_metric_key(self):
        pages = [
            PageResult(file="a.png", page=1, scores={"char_f1": 0.9}),
            PageResult(file="b.png", page=1, scores={"char_f1": 0.85}),
        ]
        rr = RunResult(is_warmup=False, page_results=pages)
        br = BenchmarkResult(runs=[rr])
        result = BenchmarkRunner.compare(br, br, "char_f1", metric_key="char_f1")
        assert result["metric"] == "char_f1"

    def test_compare_no_measured_runs(self):
        br = BenchmarkResult(runs=[RunResult(is_warmup=True)])
        result = BenchmarkRunner.compare(br, br, "cer")
        assert "error" in result

    def test_compare_basic(self):
        pages_a = [
            PageResult(file="test.pdf", page=1, scores={"cer": 0.1}),
            PageResult(file="test.pdf", page=2, scores={"cer": 0.15}),
        ]
        pages_b = [
            PageResult(file="test.pdf", page=1, scores={"cer": 0.2}),
            PageResult(file="test.pdf", page=2, scores={"cer": 0.25}),
        ]

        result_a = BenchmarkResult(
            model_name="ModelA",
            runs=[RunResult(is_warmup=False, page_results=pages_a)],
        )
        result_b = BenchmarkResult(
            model_name="ModelB",
            runs=[RunResult(is_warmup=False, page_results=pages_b)],
        )

        comparison = BenchmarkRunner.compare(result_a, result_b, "cer")
        assert "p_value" in comparison
        assert "model_a" in comparison
        assert "model_b" in comparison
        assert comparison["n_paired_samples"] == 2

    def test_compare_multiple_pages(self):
        pages_a = [
            PageResult(file="a.pdf", page=1, scores={"cer": 0.1}),
            PageResult(file="a.pdf", page=2, scores={"cer": 0.15}),
            PageResult(file="b.pdf", page=1, scores={"cer": 0.2}),
        ]
        pages_b = [
            PageResult(file="a.pdf", page=1, scores={"cer": 0.12}),
            PageResult(file="a.pdf", page=2, scores={"cer": 0.18}),
            PageResult(file="b.pdf", page=1, scores={"cer": 0.22}),
        ]

        result_a = BenchmarkResult(
            model_name="ModelA",
            runs=[RunResult(is_warmup=False, page_results=pages_a)],
        )
        result_b = BenchmarkResult(
            model_name="ModelB",
            runs=[RunResult(is_warmup=False, page_results=pages_b)],
        )

        comparison = BenchmarkRunner.compare(result_a, result_b, "cer")
        assert comparison["n_paired_samples"] == 3
        assert "significant_at_005" in comparison

    def test_compare_insufficient_samples(self):
        result_a = BenchmarkResult(
            model_name="ModelA",
            runs=[RunResult(is_warmup=False, page_results=[])],
        )
        result_b = BenchmarkResult(
            model_name="ModelB",
            runs=[RunResult(is_warmup=False, page_results=[])],
        )

        comparison = BenchmarkRunner.compare(result_a, result_b, "cer")
        assert "error" in comparison

    def test_compare_nan_excluded(self):
        pages_a = [
            PageResult(file="a.pdf", page=1, scores={"cer": 0.1}),
            PageResult(file="a.pdf", page=2, scores={"cer": 0.13}),
            PageResult(file="a.pdf", page=3, scores={"cer": float("nan")}),
        ]
        pages_b = [
            PageResult(file="a.pdf", page=1, scores={"cer": 0.12}),
            PageResult(file="a.pdf", page=2, scores={"cer": 0.15}),
            PageResult(file="a.pdf", page=3, scores={"cer": 0.18}),
        ]

        result_a = BenchmarkResult(
            model_name="ModelA",
            runs=[RunResult(is_warmup=False, page_results=pages_a)],
        )
        result_b = BenchmarkResult(
            model_name="ModelB",
            runs=[RunResult(is_warmup=False, page_results=pages_b)],
        )

        comparison = BenchmarkRunner.compare(result_a, result_b, "cer")
        # Two valid pairs (a.pdf page 1 and 2; page 3 excluded due to NaN)
        assert comparison["n_paired_samples"] == 2


class TestBenchmarkRunnerDegradeImages:
    def test_degrade_images(self, sample_image, tmp_dir):
        # Save sample to disk
        img_path = os.path.join(tmp_dir, "test.png")
        sample_image.save(img_path)

        file_images = {"doc.pdf": [img_path]}

        def dummy_apply(img):
            return img.copy()

        degraded = BenchmarkRunner._degrade_images(file_images, dummy_apply)
        assert "doc.pdf" in degraded
        assert len(degraded["doc.pdf"]) == 1
        assert os.path.isfile(degraded["doc.pdf"][0])

        # Cleanup
        for paths in degraded.values():
            for p in paths:
                os.unlink(p)
