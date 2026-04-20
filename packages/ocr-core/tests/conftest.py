"""Shared fixtures for ocr-core tests."""

from __future__ import annotations

import os
import sys
import tempfile

import pytest
from loguru import logger
from PIL import Image

from ocr_core.config import (
    BenchmarkConfig,
    DataConfig,
    MetricConfig,
    ModelConfig,
    NormalisationConfig,
)
from ocr_core.normalisation import NormalisationPipeline
from ocr_core.types import BBox, OCRPage

# ── Logging / warnings setup ───────────────────────────────


def pytest_configure(config):
    """Reduce loguru noise and suppress third-party warnings."""
    # Keep only WARNING+ so DEBUG/INFO don't flood output
    logger.remove()
    logger.add(sys.stderr, level="WARNING")


@pytest.fixture
def suppress_loguru():
    """Temporarily silence ALL loguru output for a single test.

    Usage:
        def test_something_that_logs_errors(suppress_loguru):
            ...
    """
    logger.disable("ocr_core")
    yield
    logger.enable("ocr_core")


# ── Reusable lightweight objects ────────────────────────────


@pytest.fixture
def default_normaliser() -> NormalisationPipeline:
    return NormalisationPipeline(NormalisationConfig())


@pytest.fixture
def identity_normaliser() -> NormalisationPipeline:
    """Normaliser that changes nothing."""
    return NormalisationPipeline(
        NormalisationConfig(
            unicode_form="",
            lowercase=False,
            strip_whitespace=False,
            collapse_whitespace=False,
            remove_punctuation=False,
        )
    )


@pytest.fixture
def simple_gt_page() -> OCRPage:
    return OCRPage(
        page_number=1, full_text="The quick brown fox jumps over the lazy dog"
    )


@pytest.fixture
def simple_pred_page() -> OCRPage:
    return OCRPage(
        page_number=1, full_text="The quick brown fox jumps over the lazy dog"
    )


@pytest.fixture
def bbox_a() -> BBox:
    return BBox(x1=0, y1=0, x2=100, y2=100)


@pytest.fixture
def bbox_b() -> BBox:
    return BBox(x1=50, y1=50, x2=150, y2=150)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_image() -> Image.Image:
    return Image.new("RGB", (200, 100), color=(255, 255, 255))


@pytest.fixture
def minimal_config(tmp_dir) -> BenchmarkConfig:
    cfg = BenchmarkConfig(
        runs=1,
        warmup_runs=0,
        timeout_seconds=120,
        metrics=[MetricConfig(name="cer")],
        data=DataConfig(
            input_dir=os.path.join(tmp_dir, "inputs"),
            groundtruth_dir=os.path.join(tmp_dir, "groundtruths"),
            processed_dir=os.path.join(tmp_dir, "processed"),
            results_dir=os.path.join(tmp_dir, "results"),
        ),
        models={
            "TestModel": ModelConfig(
                project_dir="models/test_model",
                module="test_model.worker",
            )
        },
    )
    cfg._source_path = ""
    return cfg
