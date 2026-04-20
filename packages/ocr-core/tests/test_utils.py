"""Tests for ocr_core.utils."""

from __future__ import annotations

from ocr_core.utils import fmt, resolve_device, safe_filename


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
