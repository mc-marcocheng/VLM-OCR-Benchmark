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


@pytest.fixture
def sample_image():
    return Image.new("RGB", (100, 100), (128, 128, 128))


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

    def test_default_seed(self, sample_image):
        """Test that default seed is deterministic based on sigma."""
        # Default seed is 42 + int(sigma * 1_000_000)
        a = add_gaussian_noise(sample_image, sigma=25)
        b = add_gaussian_noise(sample_image, sigma=25)
        assert np.array_equal(np.array(a), np.array(b))


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

    def test_positive_rotation(self, sample_image):
        """Test positive (counter-clockwise) rotation."""
        result = rotate(sample_image, degrees=45)
        assert result.size == sample_image.size
        assert result.mode == "RGB"

    def test_negative_rotation(self, sample_image):
        """Test negative (clockwise) rotation."""
        result = rotate(sample_image, degrees=-45)
        assert result.size == sample_image.size
        assert result.mode == "RGB"


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

    def test_half_resolution(self, sample_image):
        """Test reducing to half resolution."""
        result = reduce_dpi(sample_image, factor=0.5)
        assert result.size == sample_image.size
        assert result.mode == "RGB"

    def test_quarter_resolution(self, color_image):
        """Test reducing to quarter resolution."""
        result = reduce_dpi(color_image, factor=0.25)
        assert result.size == color_image.size
        # Should be different from original (color_image has gradients)
        diff = np.abs(
            np.array(result, dtype=float) - np.array(color_image, dtype=float)
        )
        assert diff.mean() > 1.0

    def test_very_small_factor(self, sample_image):
        """Test with very small reduction factor."""
        result = reduce_dpi(sample_image, factor=0.01)
        assert result.size == sample_image.size
        # Should be heavily degraded
        assert result.mode == "RGB"


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

    def test_different_noise_amounts(self, sample_image):
        """Test that different noise amounts produce different results."""
        low_noise = salt_and_pepper(sample_image, amount=0.01, seed=42)
        high_noise = salt_and_pepper(sample_image, amount=0.2, seed=42)
        low_arr = np.array(low_noise)
        high_arr = np.array(high_noise)
        # High noise should have more pixels at 0 or 255
        low_extremes = np.sum((low_arr == 0) | (low_arr == 255))
        high_extremes = np.sum((high_arr == 0) | (high_arr == 255))
        assert high_extremes > low_extremes


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


class TestDegradationFunctionsRegistry:
    def test_all_functions_registered(self):
        """Test that all degradation functions are properly registered."""
        expected = {
            "noise",
            "blur",
            "jpeg",
            "rotate",
            "dpi_reduction",
            "salt_and_pepper",
        }
        assert set(DEGRADATION_FUNCTIONS.keys()) == expected

    def test_all_functions_callable(self, sample_image):
        """Test that all registered functions are callable and work."""
        test_cases = {
            "noise": {"sigma": 10, "seed": 42},
            "blur": {"radius": 2.0},
            "jpeg": {"quality": 80},
            "rotate": {"degrees": 10},
            "dpi_reduction": {"factor": 0.5},
            "salt_and_pepper": {"amount": 0.01, "seed": 42},
        }
        for name, params in test_cases.items():
            func = DEGRADATION_FUNCTIONS[name]
            result = func(sample_image, **params)
            assert isinstance(result, Image.Image)
            assert result.size == sample_image.size


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

    def test_single_value_params(self):
        """Test pipeline with single (non-list) parameter values."""
        steps = [
            {"name": "noise", "params": {"sigma": 25, "seed": 42}},
            {"name": "blur", "params": {"radius": 2.0}},
        ]
        pipeline = DegradationPipeline(steps)
        # Each step creates one variant (no list expansion)
        assert len(pipeline) == 2
        # First variant is noise
        assert pipeline.variants[0].name == "noise"
        assert pipeline.variants[0].params["sigma"] == 25
        # Second variant is blur
        assert pipeline.variants[1].name == "blur"
        assert pipeline.variants[1].params["radius"] == 2.0

    def test_list_expansion(self):
        """Test that list parameters are expanded into multiple variants."""
        steps = [{"name": "blur", "params": {"radius": [1.0, 2.0, 3.0]}}]
        pipeline = DegradationPipeline(steps)
        assert len(pipeline) == 3
        radii = [v.params["radius"] for v in pipeline.variants]
        assert radii == [1.0, 2.0, 3.0]

    def test_unknown_degradation_skipped(self, suppress_loguru):
        """Test that unknown degradation functions are skipped with warning."""
        steps = [
            {"name": "unknown_func", "params": {}},
            {"name": "noise", "params": {"sigma": 25}},
        ]
        pipeline = DegradationPipeline(steps)
        # Only the valid noise step should be included
        assert len(pipeline) == 1
        assert pipeline.variants[0].name == "noise"

    def test_variant_label_generation(self):
        """Test that variant labels are generated correctly."""
        steps = [
            {"name": "noise", "params": {"sigma": [10, 25]}},
            {"name": "blur", "params": {"radius": [1.0, 2.0]}},
        ]
        pipeline = DegradationPipeline(steps)
        labels = [v.label for v in pipeline.variants]
        # Each step creates separate variants with their own labels
        # noise creates: noise_sigma=10, noise_sigma=25
        # blur creates: blur_radius=1.0, blur_radius=2.0
        assert len(labels) == 4
        assert any("noise" in label for label in labels)
        assert any("blur" in label for label in labels)
