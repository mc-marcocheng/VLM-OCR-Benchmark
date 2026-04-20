"""Tests for ocr_core.degradation."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from ocr_core.degradation import (
    DEGRADATION_FUNCTIONS,
    DegradationPipeline,
    add_gaussian_noise,
    apply_blur,
    jpeg_compress,
    reduce_dpi,
    rotate,
    salt_and_pepper,
)


@pytest.fixture
def white_image():
    return Image.new("RGB", (100, 100), (255, 255, 255))


@pytest.fixture
def color_image():
    img = Image.new("RGB", (200, 200))
    pixels = img.load()
    for x in range(200):
        for y in range(200):
            pixels[x, y] = (x % 256, y % 256, (x + y) % 256)
    return img


class TestGaussianNoise:
    def test_output_same_size(self, white_image):
        result = add_gaussian_noise(white_image, sigma=25)
        assert result.size == white_image.size

    def test_output_is_rgb(self, white_image):
        result = add_gaussian_noise(white_image, sigma=25)
        assert result.mode == "RGB"

    def test_sigma_zero_unchanged(self, white_image):
        result = add_gaussian_noise(white_image, sigma=0)
        assert np.array_equal(np.array(result), np.array(white_image))

    def test_higher_sigma_more_noise(self, white_image):
        r10 = add_gaussian_noise(white_image, sigma=10)
        r50 = add_gaussian_noise(white_image, sigma=50)
        diff10 = np.abs(np.array(r10, dtype=float) - 255).mean()
        diff50 = np.abs(np.array(r50, dtype=float) - 255).mean()
        assert diff50 > diff10

    def test_deterministic_with_seed(self, white_image):
        a = add_gaussian_noise(white_image, sigma=25, seed=123)
        b = add_gaussian_noise(white_image, sigma=25, seed=123)
        assert np.array_equal(np.array(a), np.array(b))

    def test_different_seeds(self, white_image):
        a = add_gaussian_noise(white_image, sigma=25, seed=1)
        b = add_gaussian_noise(white_image, sigma=25, seed=2)
        assert not np.array_equal(np.array(a), np.array(b))


class TestBlur:
    def test_output_same_size(self, color_image):
        result = apply_blur(color_image, radius=2.0)
        assert result.size == color_image.size

    def test_blurred_less_sharp(self, color_image):
        blurred = apply_blur(color_image, radius=5.0)
        orig_arr = np.array(color_image, dtype=float)
        blur_arr = np.array(blurred, dtype=float)
        orig_var = np.var(orig_arr)
        blur_var = np.var(blur_arr)
        assert blur_var < orig_var


class TestJpegCompress:
    def test_output_same_size(self, color_image):
        result = jpeg_compress(color_image, quality=50)
        assert result.size == color_image.size

    def test_output_rgb(self, color_image):
        result = jpeg_compress(color_image, quality=10)
        assert result.mode == "RGB"

    def test_low_quality_lossy(self, color_image):
        result = jpeg_compress(color_image, quality=1)
        assert not np.array_equal(np.array(result), np.array(color_image))


class TestRotate:
    def test_output_same_size(self, white_image):
        result = rotate(white_image, degrees=5)
        assert result.size == white_image.size

    def test_zero_rotation_unchanged(self, white_image):
        result = rotate(white_image, degrees=0)
        assert np.array_equal(np.array(result), np.array(white_image))


class TestReduceDpi:
    def test_output_same_size(self, color_image):
        result = reduce_dpi(color_image, factor=0.5)
        assert result.size == color_image.size

    def test_severe_reduction_loses_detail(self, color_image):
        result = reduce_dpi(color_image, factor=0.1)
        assert not np.array_equal(np.array(result), np.array(color_image))

    def test_factor_one_nearly_unchanged(self, color_image):
        result = reduce_dpi(color_image, factor=1.0)
        diff = np.abs(
            np.array(result, dtype=float) - np.array(color_image, dtype=float)
        )
        assert diff.mean() < 1.0  # very small rounding


class TestSaltAndPepper:
    def test_output_same_size(self, white_image):
        result = salt_and_pepper(white_image, amount=0.05)
        assert result.size == white_image.size

    def test_deterministic(self, white_image):
        a = salt_and_pepper(white_image, amount=0.05, seed=42)
        b = salt_and_pepper(white_image, amount=0.05, seed=42)
        assert np.array_equal(np.array(a), np.array(b))

    def test_zero_amount_unchanged(self, white_image):
        result = salt_and_pepper(white_image, amount=0.0)
        assert np.array_equal(np.array(result), np.array(white_image))


class TestDegradationRegistry:
    def test_all_known(self):
        expected = {
            "noise",
            "blur",
            "jpeg",
            "rotate",
            "dpi_reduction",
            "salt_and_pepper",
        }
        assert set(DEGRADATION_FUNCTIONS.keys()) == expected


class TestDegradationPipeline:
    def test_single_fixed_params(self):
        steps = [{"name": "noise", "params": {"sigma": 25}}]
        pipeline = DegradationPipeline(steps)
        assert len(pipeline) == 1
        assert pipeline.variants[0].params == {"sigma": 25}

    def test_list_params_expand(self):
        steps = [{"name": "noise", "params": {"sigma": [10, 25, 50]}}]
        pipeline = DegradationPipeline(steps)
        assert len(pipeline) == 3
        sigmas = [v.params["sigma"] for v in pipeline.variants]
        assert sigmas == [10, 25, 50]

    def test_cartesian_product(self):
        steps = [
            {
                "name": "noise",
                "params": {"sigma": [10, 25], "seed": [1, 2]},
            }
        ]
        pipeline = DegradationPipeline(steps)
        assert len(pipeline) == 4  # 2 × 2

    def test_unknown_degradation_skipped(self, suppress_loguru):
        steps = [{"name": "unknown_degradation", "params": {}}]
        pipeline = DegradationPipeline(steps)
        assert len(pipeline) == 0

    def test_multiple_steps(self):
        steps = [
            {"name": "noise", "params": {"sigma": [10, 25]}},
            {"name": "blur", "params": {"radius": [1.0, 2.0]}},
        ]
        pipeline = DegradationPipeline(steps)
        assert len(pipeline) == 4

    def test_variant_apply(self, white_image):
        steps = [{"name": "noise", "params": {"sigma": 25}}]
        pipeline = DegradationPipeline(steps)
        result = pipeline.variants[0].apply(white_image)
        assert result.size == white_image.size

    def test_iter(self):
        steps = [{"name": "blur", "params": {"radius": [1.0, 2.0]}}]
        pipeline = DegradationPipeline(steps)
        variants = list(pipeline)
        assert len(variants) == 2

    def test_empty_params(self):
        steps = [{"name": "blur", "params": {}}]
        pipeline = DegradationPipeline(steps)
        assert len(pipeline) == 1

    def test_labels_are_unique(self):
        steps = [{"name": "noise", "params": {"sigma": [10, 25, 50]}}]
        pipeline = DegradationPipeline(steps)
        labels = [v.label for v in pipeline.variants]
        assert len(labels) == len(set(labels))
