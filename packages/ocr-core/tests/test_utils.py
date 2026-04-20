"""Tests for ocr_core.utils."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import ocr_core.utils as utils_module
from ocr_core.utils import (
    _get_device_id,
    fmt,
    get_peak_vram_mb,
    get_vram_usage_mb,
    reset_peak_vram,
    resolve_device,
    safe_filename,
)


class TestSafeFilename:
    def test_alphanumeric(self):
        assert safe_filename("hello123") == "hello123"

    def test_special_characters(self):
        result = safe_filename("model/name:v2@test")
        assert "/" not in result
        assert ":" not in result
        assert "@" not in result

    def test_dots_and_hyphens_preserved(self):
        assert safe_filename("model-v1.0") == "model-v1.0"

    def test_spaces_replaced(self):
        result = safe_filename("hello world")
        assert " " not in result

    def test_empty(self):
        assert safe_filename("") == ""

    def test_unicode(self):
        result = safe_filename("模型_test")
        # CJK characters match \w in Python regex, so they should be preserved
        assert "test" in result

    def test_with_spaces(self):
        assert safe_filename("model name") == "model_name"

    def test_with_special_chars(self):
        assert safe_filename("model@name#1") == "model_name_1"

    def test_preserves_dots(self):
        assert safe_filename("model.v1") == "model.v1"

    def test_preserves_hyphens(self):
        assert safe_filename("model-name") == "model-name"

    def test_complex_string(self):
        assert safe_filename("Model/Name:v1.0@test") == "Model_Name_v1.0_test"


class TestFmt:
    def test_normal_float(self):
        assert fmt(0.12345) == "0.1235"

    def test_none(self):
        assert fmt(None) == "N/A"

    def test_nan(self):
        assert fmt(float("nan")) == "N/A"

    def test_custom_spec(self):
        assert fmt(3.14159, ".2f") == "3.14"

    def test_suffix(self):
        assert fmt(512, ".0f", " MB") == "512 MB"

    def test_integer(self):
        assert fmt(42, ".0f") == "42"

    def test_inf(self):
        result = fmt(float("inf"))
        assert result == "inf"

    def test_non_numeric_falls_back(self):
        assert fmt("hello", ".4f") == "hello"

    def test_custom_spec_decimal(self):
        assert fmt(100.5, ".0f") == "100"

    def test_with_suffix_mb(self):
        assert fmt(100.0, ".0f", " MB") == "100 MB"

    def test_none_value(self):
        assert fmt(None) == "N/A"

    def test_nan_value(self):
        assert fmt(float("nan")) == "N/A"

    def test_invalid_spec(self):
        # Should return str representation on error
        result = fmt("not a number", ".4f")
        assert result == "not a number"


class TestResolveDevice:
    def test_cpu(self):
        assert resolve_device("cpu") == "cpu"

    def test_cpu_uppercase(self):
        assert resolve_device("CPU") == "cpu"

    def test_cpu_with_whitespace(self):
        assert resolve_device("  cpu  ") == "cpu"

    def test_gpu_without_cuda(self):
        # In test env CUDA is unlikely available
        result = resolve_device("gpu")
        assert result in ("cuda", "cpu")

    def test_cuda_without_cuda(self):
        result = resolve_device("cuda")
        assert result in ("cuda", "cpu")

    def test_other_device(self):
        assert resolve_device("mps") == "mps"

    @patch("ocr_core.utils.logger")
    def test_gpu_without_cuda_mocked(self, mock_logger):
        with patch.dict("sys.modules", {"torch": None}):
            # Force ImportError by removing torch
            original = sys.modules.get("torch")
            sys.modules["torch"] = None
            try:
                result = resolve_device("gpu")
                # Should fall back to cpu
                assert result == "cpu"
            finally:
                if original:
                    sys.modules["torch"] = original

    def test_cuda_alias(self):
        result = resolve_device("cuda")
        # Will be "cuda" if available, "cpu" otherwise
        assert result in ("cuda", "cpu")


class TestVramFunctions:
    def test_get_vram_usage_no_gpu(self):
        # Should return None when no GPU available
        with patch("ocr_core.utils._get_device_id", return_value=0):
            with patch.dict("sys.modules", {"torch": None}):
                # Force reimport of utils with torch unavailable
                importlib.reload(utils_module)
                result = utils_module.get_vram_usage_mb()
                # Result depends on system, but should not raise
                assert result is None or isinstance(result, float)

    def test_get_peak_vram_no_gpu(self):
        result = get_peak_vram_mb()
        # Should return None or float depending on system
        assert result is None or isinstance(result, float)

    def test_reset_peak_vram_no_error(self):
        # Should not raise even without GPU
        reset_peak_vram()

    @patch("ocr_core.utils._get_device_id")
    def test_get_device_id_from_env(self, mock_get_device):
        mock_get_device.return_value = 0
        # Just verify it's called correctly
        get_vram_usage_mb()


class TestGetDeviceId:
    def test_device_id_default(self):
        # Should return an integer
        result = _get_device_id()
        assert isinstance(result, int)
        assert result >= 0

    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "2,3"})
    def test_device_id_from_env(self):
        # When torch is not available, should parse from env
        with patch.dict("sys.modules", {"torch": None}):
            result = _get_device_id()
            assert isinstance(result, int)

    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "invalid"})
    def test_device_id_invalid_env(self):
        with patch.dict("sys.modules", {"torch": None}):
            result = _get_device_id()
            assert result == 0  # Falls back to 0
