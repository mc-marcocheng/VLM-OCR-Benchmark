# packages/ocr-core/src/ocr_core/benchmark.py
"""
Main benchmark runner.

Orchestrates data loading, model invocation (via subprocess), metric
computation, multi-run statistics, and degradation sweeps.
"""
from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger
from PIL import Image

from ocr_core.config import BenchmarkConfig, load_config
from ocr_core.data_loader import DataLoader
from ocr_core.degradation import DegradationPipeline
from ocr_core.metrics import MetricRegistry
from ocr_core.normalisation import NormalisationPipeline
from ocr_core.statistics import SummaryStats, paired_bootstrap_test, summarise
from ocr_core.types import GroundTruth, OCRPage, WorkerResponse, WorkerTask
from ocr_core.utils import resolve_device, safe_filename

__all__ = ["BenchmarkResult", "BenchmarkRunner", "PageResult", "RunResult"]

_REPO_ROOT = os.path.abspath(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
    )
)

# ── Per-page detail saved to JSON ───────────────────────────


@dataclass
class PageResult:
    file: str = ""
    page: int = 1
    prediction_time_seconds: float = 0.0
    predicted_text: str = ""
    ground_truth_text: str | None = None
    scores: dict[str, float] = field(default_factory=dict)
    predicted_page: OCRPage | None = None

    def to_dict(self) -> dict:
        d = {
            "file": self.file,
            "page": self.page,
            "prediction_time_seconds": self.prediction_time_seconds,
            "predicted_text": self.predicted_text,
            "ground_truth_text": self.ground_truth_text,
            "scores": self.scores,
        }
        if self.predicted_page and self.predicted_page.has_regions():
            d["predicted_regions"] = [r.to_dict() for r in self.predicted_page.regions]
        return d


# ── Single-run result ───────────────────────────────────────


@dataclass
class RunResult:
    run_index: int = 0
    is_warmup: bool = False
    model_load_time_seconds: float = 0.0
    total_prediction_time_seconds: float = 0.0
    resources: dict[str, Any] = field(default_factory=dict)
    page_results: list[PageResult] = field(default_factory=list)
    degradation_label: str = ""  # empty = clean (no degradation)

    @property
    def n_pages(self) -> int:
        return len(self.page_results)

    def aggregate_score(self, metric_name: str) -> float:
        vals = [
            pr.scores[metric_name]
            for pr in self.page_results
            if metric_name in pr.scores and not math.isnan(pr.scores[metric_name])
        ]
        return sum(vals) / len(vals) if vals else float("nan")


# ── Full benchmark result ───────────────────────────────────


@dataclass
class BenchmarkResult:
    model_name: str = ""
    test_set: str = ""
    device: str = ""
    timestamp: str = ""
    config_path: str = ""
    runs: list[RunResult] = field(default_factory=list)
    degradation_runs: list[RunResult] = field(default_factory=list)

    @property
    def measured_runs(self) -> list[RunResult]:
        return [r for r in self.runs if not r.is_warmup]

    def score_summary(self, metric_name: str) -> SummaryStats:
        raw = [r.aggregate_score(metric_name) for r in self.measured_runs]
        values = [v for v in raw if not math.isnan(v)]
        return summarise(values)

    def timing_summary(self) -> SummaryStats:
        vals: list[float] = []
        for r in self.measured_runs:
            if r.n_pages > 0:
                vals.append(r.total_prediction_time_seconds / r.n_pages)
        return summarise(vals)


# ── Runner ──────────────────────────────────────────────────


class BenchmarkRunner:
    def __init__(
        self, config: BenchmarkConfig | None = None, config_path: str | None = None
    ):
        if config is None:
            config = load_config(config_path)
        self.cfg = config
        self.data = DataLoader(
            input_dir=config.data.input_dir,
            processed_dir=config.data.processed_dir,
            groundtruth_dir=config.data.groundtruth_dir,
            pdf_dpi=config.data.pdf_dpi,
        )
        self.normaliser = NormalisationPipeline(config.normalisation)
        self.metrics = MetricRegistry.from_config(config.metrics)
        os.makedirs(config.data.results_dir, exist_ok=True)

    # ── public entry-point ──────────────────────────────────

    def run(
        self,
        model_name: str,
        test_set: str,
        device: str = "cpu",
    ) -> BenchmarkResult:
        device = resolve_device(device)

        logger.info("=" * 60)
        logger.info(f"  {model_name}  |  {test_set}  |  {device}")
        logger.info(f"  metrics: {self.metrics.names}")
        logger.info(f"  runs: {self.cfg.warmup_runs}w + {self.cfg.runs}m")
        logger.info("=" * 60)

        files = self.data.list_files(test_set)
        if not files:
            raise FileNotFoundError(
                f"No input files in {self.cfg.data.input_dir}/{test_set}"
            )

        # Pre-process all inputs (cache page images)
        file_images: dict[str, list[str]] = {}
        file_gts: dict[str, GroundTruth | None] = {}
        for fname in files:
            try:
                file_images[fname] = self.data.ensure_cached(test_set, fname)
                file_gts[fname] = self.data.load_ground_truth(test_set, fname)
            except Exception:
                logger.exception(f"Skipping {fname} — failed to load/cache")

        if not file_images:
            raise FileNotFoundError(
                f"No usable input files in {self.cfg.data.input_dir}/{test_set}"
            )

        result = BenchmarkResult(
            model_name=model_name,
            test_set=test_set,
            device=device,
            timestamp=datetime.now().isoformat(),
            config_path=self.cfg._source_path,
        )

        total_runs = self.cfg.warmup_runs + self.cfg.runs
        for run_idx in range(total_runs):
            is_warmup = run_idx < self.cfg.warmup_runs
            label = "warmup" if is_warmup else "measured"
            logger.info(f"--- Run {run_idx + 1}/{total_runs} ({label}) ---")

            rr = self._single_run(
                model_name=model_name,
                device=device,
                file_images=file_images,
                file_gts=file_gts,
                run_index=run_idx,
                is_warmup=is_warmup,
            )
            result.runs.append(rr)

        # Degradation sweeps
        if self.cfg.degradation.enabled:
            pipeline = DegradationPipeline(
                [
                    {"name": s.name, "params": s.params}
                    for s in self.cfg.degradation.pipelines
                ]
            )
            for variant in pipeline:
                logger.info(f"--- Degradation: {variant.label} ---")
                try:
                    degraded_images = self._degrade_images(file_images, variant.apply)
                except Exception:
                    logger.exception(
                        f"Failed to apply degradation {variant.label} — skipping"
                    )
                    continue
                try:
                    rr = self._single_run(
                        model_name=model_name,
                        device=device,
                        file_images=degraded_images,
                        file_gts=file_gts,
                        run_index=0,
                        is_warmup=False,
                        degradation_label=variant.label,
                    )
                    result.degradation_runs.append(rr)
                except Exception:
                    logger.exception(
                        f"Degradation run {variant.label} failed — skipping"
                    )
                finally:
                    for paths in degraded_images.values():
                        for tmp in paths:
                            try:
                                os.unlink(tmp)
                            except OSError:
                                pass

        # Save
        self._save(result)
        self._update_summary_csv(result)

        # Log summary
        for m in self.metrics:
            ss = result.score_summary(m.primary_key)
            if ss.n > 0:
                logger.info(
                    f"  {m.name}: {ss.mean:.4f}  "
                    f"(±{ss.std:.4f}, 95% CI [{ss.ci_lower:.4f}, {ss.ci_upper:.4f}])"
                )
        ts = result.timing_summary()
        if ts.n > 0:
            logger.info(
                f"  speed: {ts.mean:.4f} s/page  "
                f"(±{ts.std:.4f}, 95% CI [{ts.ci_lower:.4f}, {ts.ci_upper:.4f}])"
            )

        return result

    # ── single run (invoke worker) ──────────────────────────

    def _single_run(
        self,
        model_name: str,
        device: str,
        file_images: dict[str, list[str]],
        file_gts: dict[str, GroundTruth | None],
        run_index: int,
        is_warmup: bool,
        degradation_label: str = "",
    ) -> RunResult:
        all_image_paths = [p for paths in file_images.values() for p in paths]

        worker_resp = self._invoke_worker(model_name, device, all_image_paths)

        if len(worker_resp.pages) != len(all_image_paths):
            logger.warning(
                f"Worker returned {len(worker_resp.pages)} pages "
                f"but {len(all_image_paths)} images were sent"
            )

        # Map worker page results back to files
        resp_idx = 0
        page_results: list[PageResult] = []
        total_pred_time = 0.0

        for fname, img_paths in file_images.items():
            gt = file_gts.get(fname)
            for page_idx, img_path in enumerate(img_paths):
                if resp_idx >= len(worker_resp.pages):
                    break
                wp = worker_resp.pages[resp_idx]
                resp_idx += 1

                total_pred_time += wp.prediction_time_seconds

                pred_page = wp.result
                gt_page = (
                    gt.pages[page_idx] if gt and page_idx < len(gt.pages) else None
                )

                scores: dict[str, float] = {}
                if wp.error:
                    logger.warning(
                        f"Worker reported error on {fname} p{page_idx+1}: {wp.error}"
                    )
                elif gt_page:
                    for metric in self.metrics:
                        if metric.is_applicable(gt_page, pred_page):
                            try:
                                mr = metric.compute(gt_page, pred_page, self.normaliser)
                                scores.update(mr.scores)
                            except Exception:
                                logger.exception(
                                    f"Metric {metric.name} failed on {fname}, "
                                    f"page {page_idx+1}"
                                )

                page_results.append(
                    PageResult(
                        file=fname,
                        page=page_idx + 1,
                        prediction_time_seconds=wp.prediction_time_seconds,
                        predicted_text=pred_page.full_text,
                        ground_truth_text=gt_page.full_text if gt_page else None,
                        scores=scores,
                        predicted_page=pred_page,
                    )
                )

        return RunResult(
            run_index=run_index,
            is_warmup=is_warmup,
            model_load_time_seconds=worker_resp.model_load_time_seconds,
            total_prediction_time_seconds=total_pred_time,
            resources={
                "ram_before_load_mb": worker_resp.ram_before_load_mb,
                "ram_after_load_mb": worker_resp.ram_after_load_mb,
                "peak_ram_mb": worker_resp.peak_ram_mb,
                "vram_before_load_mb": worker_resp.vram_before_load_mb,
                "vram_after_load_mb": worker_resp.vram_after_load_mb,
                "peak_vram_mb": worker_resp.peak_vram_mb,
            },
            page_results=page_results,
            degradation_label=degradation_label,
        )

    # ── Worker invocation ───────────────────────────────────

    def _invoke_worker(
        self, model_name: str, device: str, image_paths: list[str]
    ) -> WorkerResponse:
        model_cfg = self.cfg.models.get(model_name)
        if not model_cfg:
            raise ValueError(
                f"Model {model_name!r} not in config. "
                f"Available: {list(self.cfg.models)}"
            )

        task = WorkerTask(
            image_paths=image_paths,
            device=device,
            params=model_cfg.params,
        )

        task_fd, task_path = tempfile.mkstemp(suffix=".json", prefix="ocr_task_")
        out_fd, output_path = tempfile.mkstemp(suffix=".json", prefix="ocr_result_")
        os.close(out_fd)

        try:
            with os.fdopen(task_fd, "w", encoding="utf-8") as fh:
                json.dump(task.to_dict(), fh)

            # ── key change: --directory instead of --package ──
            project_dir = model_cfg.project_dir
            if not os.path.isabs(project_dir):
                project_dir = os.path.join(_REPO_ROOT, project_dir)

            cmd = [
                "uv",
                "run",
                "--directory",
                project_dir,
                "python",
                "-m",
                model_cfg.module,
                "--task",
                task_path,
                "--output",
                output_path,
            ]

            logger.info(f"Spawning: {' '.join(cmd)}")
            t0 = time.perf_counter()

            try:
                if os.name != "nt":
                    preexec_fn: Any = os.setsid  # type: ignore[attr-defined]
                else:
                    preexec_fn = None
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.cfg.timeout_seconds,
                    cwd=_REPO_ROOT,
                    encoding="utf-8",
                    errors="replace",
                    preexec_fn=preexec_fn,
                )
            except subprocess.TimeoutExpired as exc:
                if os.name != "nt":
                    pid = exc.pid  # type: ignore[attr-defined]
                    if pid is not None:
                        os.killpg(  # type: ignore[attr-defined]
                            os.getpgid(pid),  # type: ignore[attr-defined]
                            signal.SIGKILL,  # type: ignore[attr-defined]
                        )
                raise RuntimeError(f"Worker {model_name} timed out") from exc

            elapsed = time.perf_counter() - t0
            logger.info(f"Worker finished in {elapsed:.1f}s (exit={proc.returncode})")

            # Log worker output even on success (debug/warning level)
            if proc.stdout and proc.stdout.strip():
                logger.debug(f"Worker STDOUT:\n{proc.stdout}")
            if proc.stderr and proc.stderr.strip():
                logger.warning(f"Worker STDERR:\n{proc.stderr}")

            if proc.returncode != 0:
                logger.error(f"STDOUT:\n{proc.stdout}")
                logger.error(f"STDERR:\n{proc.stderr}")
                # Truncate to avoid enormous exception messages
                stderr_tail = proc.stderr[-3000:] if proc.stderr else "(empty)"
                raise RuntimeError(
                    f"Worker {model_name} failed (exit {proc.returncode})\n"
                    f"--- STDERR (last 3000 chars) ---\n{stderr_tail}"
                )

            if not os.path.isfile(output_path):
                raise RuntimeError(
                    f"Worker {model_name} exited successfully (code 0) but produced "
                    f"no output file.\nExpected: {output_path}\n"
                    f"STDOUT: {(proc.stdout or '')[-1000:]}\n"
                    f"STDERR: {(proc.stderr or '')[-1000:]}"
                )

            with open(output_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)

            return WorkerResponse.from_dict(raw)

        finally:
            for p in (task_path, output_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    # ── Degradation helpers ─────────────────────────────────

    @staticmethod
    def _degrade_images(
        file_images: dict[str, list[str]],
        apply_fn,
    ) -> dict[str, list[str]]:
        """Apply degradation and save to temp files."""
        degraded: dict[str, list[str]] = {}
        for fname, paths in file_images.items():
            new_paths: list[str] = []
            for p in paths:
                img = Image.open(p).convert("RGB")
                img = apply_fn(img)
                fd, tmp = tempfile.mkstemp(suffix=".png", prefix="ocr_deg_")
                os.close(fd)
                img.save(tmp)
                new_paths.append(tmp)
            degraded[fname] = new_paths
        return degraded

    # ── Persistence ─────────────────────────────────────────

    def _save(self, result: BenchmarkResult) -> None:
        out_dir = os.path.join(self.cfg.data.results_dir, result.test_set)
        os.makedirs(out_dir, exist_ok=True)

        # Pick the last measured run for the detailed result file
        measured = result.measured_runs
        last_run = (
            measured[-1] if measured else (result.runs[-1] if result.runs else None)
        )

        data: dict[str, Any] = {
            "model_name": result.model_name,
            "test_set": result.test_set,
            "device": result.device,
            "timestamp": result.timestamp,
            "total_runs": len(result.runs),
            "warmup_runs": sum(1 for r in result.runs if r.is_warmup),
            "measured_runs": len(measured),
        }

        if last_run:
            data.update(
                {
                    "model_load_time_seconds": last_run.model_load_time_seconds,
                    "total_prediction_time_s": last_run.total_prediction_time_seconds,
                    "total_pages_processed": last_run.n_pages,
                    "average_prediction_time_s_per_page": (
                        last_run.total_prediction_time_seconds / last_run.n_pages
                        if last_run.n_pages > 0
                        else 0.0
                    ),
                    "resources": last_run.resources,
                    "metrics": [pr.to_dict() for pr in last_run.page_results],
                }
            )

        # Aggregate metric summaries across runs
        metric_summaries: dict[str, dict] = {}
        for m in self.metrics:
            ss = result.score_summary(m.primary_key)
            if ss.n > 0:
                metric_summaries[m.name] = {
                    "mean": round(ss.mean, 6),
                    "std": round(ss.std, 6),
                    "ci_lower": round(ss.ci_lower, 6),
                    "ci_upper": round(ss.ci_upper, 6),
                    "n_runs": ss.n,
                    "primary_key": m.primary_key,
                }
                data[f"average_{m.name}"] = round(ss.mean, 6)
        data["metric_summaries"] = metric_summaries

        ts = result.timing_summary()
        if ts.n > 0:
            data["timing_summary"] = {
                "mean_s_per_page": round(ts.mean, 6),
                "std": round(ts.std, 6),
                "ci_lower": round(ts.ci_lower, 6),
                "ci_upper": round(ts.ci_upper, 6),
            }

        # Degradation summaries
        if result.degradation_runs:
            deg_data: list[dict] = []
            for dr in result.degradation_runs:
                entry: dict[str, Any] = {"label": dr.degradation_label, "scores": {}}
                for m in self.metrics:
                    v = dr.aggregate_score(m.primary_key)
                    if not math.isnan(v):
                        entry["scores"][m.name] = round(v, 6)  # type: ignore[index]
                deg_data.append(entry)
            data["degradation_results"] = deg_data

        json_path = os.path.join(
            out_dir,
            f"{safe_filename(result.model_name)}_"
            f"{safe_filename(result.device)}_results.json",
        )
        fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, json_path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        logger.success(f"Results → {json_path}")

    def _update_summary_csv(self, result: BenchmarkResult) -> None:
        csv_path = os.path.join(self.cfg.data.results_dir, "summary.csv")

        measured = result.measured_runs
        last_run = measured[-1] if measured else None
        ts = result.timing_summary()

        row: dict[str, Any] = {
            "Model": result.model_name,
            "Test Set": result.test_set,
            "Device": result.device,
            "Timestamp": result.timestamp,
            "Runs": len(measured),
            "Pages": last_run.n_pages if last_run else 0,
            "Avg Speed (s/page)": round(ts.mean, 4) if ts.n > 0 else None,
            "Speed 95% CI Low": round(ts.ci_lower, 4) if ts.n > 1 else None,
            "Speed 95% CI High": round(ts.ci_upper, 4) if ts.n > 1 else None,
        }

        for m in self.metrics:
            ss = result.score_summary(m.primary_key)
            if ss.n > 0:
                row[f"Avg {m.name.upper()}"] = round(ss.mean, 4)
                row[f"{m.name.upper()} CI Low"] = (
                    round(ss.ci_lower, 4) if ss.n > 1 else None
                )
                row[f"{m.name.upper()} CI High"] = (
                    round(ss.ci_upper, 4) if ss.n > 1 else None
                )
            else:
                row[f"Avg {m.name.upper()}"] = None

        if last_run:
            res = last_run.resources
            row["Load Time (s)"] = last_run.model_load_time_seconds
            row["RAM After Load (MB)"] = res.get("ram_after_load_mb")
            row["Peak RAM (MB)"] = res.get("peak_ram_mb")
            row["VRAM After Load (MB)"] = res.get("vram_after_load_mb")
            row["Peak VRAM (MB)"] = res.get("peak_vram_mb")

        new_row = pd.DataFrame([row])

        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path, keep_default_na=False)
            mask = (
                (df["Model"] == row["Model"])
                & (df["Test Set"] == row["Test Set"])
                & (df["Device"] == row["Device"])
            )
            df = df[~mask]
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = new_row

        # Atomic write via temp + rename
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(csv_path) or ".", suffix=".csv")
        os.close(fd)
        df.to_csv(tmp, index=False)
        os.replace(tmp, csv_path)
        logger.debug(f"Summary CSV → {csv_path}")

    # ── Static comparison helper ────────────────────────────

    @staticmethod
    def compare(
        result_a: BenchmarkResult,
        result_b: BenchmarkResult,
        metric_name: str,
        metric_key: str | None = None,
    ) -> dict[str, Any]:
        """Paired bootstrap significance test between two results."""
        key = metric_key or metric_name
        pages_a = (
            result_a.measured_runs[-1].page_results if result_a.measured_runs else []
        )
        pages_b = (
            result_b.measured_runs[-1].page_results if result_b.measured_runs else []
        )

        map_a = {
            (p.file, p.page): p.scores[key]
            for p in pages_a
            if key in p.scores and not math.isnan(p.scores[key])
        }
        map_b = {
            (p.file, p.page): p.scores[key]
            for p in pages_b
            if key in p.scores and not math.isnan(p.scores[key])
        }

        common_keys = sorted(set(map_a) & set(map_b))
        scores_a = [map_a[k] for k in common_keys]
        scores_b = [map_b[k] for k in common_keys]

        if len(scores_a) < 2:
            return {"error": "Not enough paired samples"}

        sa = summarise(scores_a)
        sb = summarise(scores_b)
        p_value = paired_bootstrap_test(scores_a, scores_b)

        return {
            "metric": metric_name,
            "model_a": {
                "mean": sa.mean,
                "std": sa.std,
                "ci": (sa.ci_lower, sa.ci_upper),
            },
            "model_b": {
                "mean": sb.mean,
                "std": sb.std,
                "ci": (sb.ci_lower, sb.ci_upper),
            },
            "p_value": p_value,
            "significant_at_005": p_value < 0.05,
            "n_paired_samples": len(scores_a),
        }
