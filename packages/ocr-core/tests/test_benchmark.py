"""Tests for ocr_core.benchmark — data classes and helpers."""

from __future__ import annotations

import math
import os

import pytest

from ocr_core.benchmark import BenchmarkResult, BenchmarkRunner, PageResult, RunResult


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
