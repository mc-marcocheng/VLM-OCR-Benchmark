"""Tests for ocr_core.config — loading, validation, edge cases."""

from __future__ import annotations

import os

import pytest
import yaml

from ocr_core.config import (
    BenchmarkConfig,
    ConfigValidationError,
    DataConfig,
    DegradationConfig,
    MetricConfig,
    ModelConfig,
    NormalisationConfig,
    load_config,
)


class TestLoadConfigDefaults:
    def test_load_none_returns_defaults(self):
        cfg = load_config(None)
        assert isinstance(cfg, BenchmarkConfig)
        assert cfg.runs == 1
        assert cfg.warmup_runs == 0
        assert len(cfg.metrics) > 0

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")


class TestLoadConfigFromFile:
    def _write_yaml(self, data: dict, path: str):
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(data, fh)

    def test_minimal_yaml(self, tmp_dir):
        path = os.path.join(tmp_dir, "cfg.yaml")
        self._write_yaml(
            {
                "benchmark": {"runs": 2, "warmup_runs": 1, "timeout_seconds": 120},
                "metrics": [{"name": "cer"}],
            },
            path,
        )
        cfg = load_config(path)
        assert cfg.runs == 2
        assert cfg.warmup_runs == 1
        assert cfg.timeout_seconds == 120
        assert cfg._source_path == path

    def test_normalisation_section(self, tmp_dir):
        path = os.path.join(tmp_dir, "cfg.yaml")
        self._write_yaml(
            {
                "normalisation": {
                    "unicode_form": "NFC",
                    "lowercase": True,
                    "remove_punctuation": True,
                },
                "metrics": [{"name": "wer"}],
            },
            path,
        )
        cfg = load_config(path)
        assert cfg.normalisation.unicode_form == "NFC"
        assert cfg.normalisation.lowercase is True
        assert cfg.normalisation.remove_punctuation is True

    def test_models_section(self, tmp_dir):
        path = os.path.join(tmp_dir, "cfg.yaml")
        self._write_yaml(
            {
                "metrics": [{"name": "cer"}],
                "models": {
                    "MyModel": {
                        "project_dir": "models/my_model",
                        "module": "my_model.worker",
                        "params": {"batch_size": 4},
                    }
                },
            },
            path,
        )
        cfg = load_config(path)
        assert "MyModel" in cfg.models
        m = cfg.models["MyModel"]
        assert m.project_dir == "models/my_model"
        assert m.module == "my_model.worker"
        assert m.params == {"batch_size": 4}

    def test_degradation_section(self, tmp_dir):
        path = os.path.join(tmp_dir, "cfg.yaml")
        self._write_yaml(
            {
                "metrics": [{"name": "cer"}],
                "degradation": {
                    "enabled": True,
                    "pipelines": [
                        {"name": "noise", "params": {"sigma": [10, 25]}},
                    ],
                },
            },
            path,
        )
        cfg = load_config(path)
        assert cfg.degradation.enabled is True
        assert len(cfg.degradation.pipelines) == 1
        assert cfg.degradation.pipelines[0].name == "noise"

    def test_data_section(self, tmp_dir):
        path = os.path.join(tmp_dir, "cfg.yaml")
        self._write_yaml(
            {
                "metrics": [{"name": "cer"}],
                "data": {
                    "input_dir": "/custom/inputs",
                    "groundtruth_dir": "/custom/gt",
                    "pdf_dpi": 300,
                },
            },
            path,
        )
        cfg = load_config(path)
        assert cfg.data.input_dir == "/custom/inputs"
        assert cfg.data.pdf_dpi == 300

    def test_metric_as_string(self, tmp_dir):
        path = os.path.join(tmp_dir, "cfg.yaml")
        self._write_yaml({"metrics": ["cer", "wer"]}, path)
        cfg = load_config(path)
        assert len(cfg.metrics) == 2
        assert cfg.metrics[0].name == "cer"
        assert cfg.metrics[1].name == "wer"

    def test_empty_yaml(self, tmp_dir):
        path = os.path.join(tmp_dir, "cfg.yaml")
        with open(path, "w") as fh:
            fh.write("")
        cfg = load_config(path)
        assert isinstance(cfg, BenchmarkConfig)

    def test_metric_with_params_and_apply_to(self, tmp_dir):
        path = os.path.join(tmp_dir, "cfg.yaml")
        self._write_yaml(
            {
                "metrics": [
                    {
                        "name": "bleu",
                        "params": {"max_n": 2},
                        "apply_to": ["text", "title"],
                    }
                ]
            },
            path,
        )
        cfg = load_config(path)
        assert cfg.metrics[0].params == {"max_n": 2}
        assert cfg.metrics[0].apply_to == ["text", "title"]


class TestValidation:
    def test_valid_default_config_passes(self):
        cfg = BenchmarkConfig()
        cfg.validate()  # should not raise

    def test_runs_zero_fails(self):
        cfg = BenchmarkConfig(runs=0)
        with pytest.raises(ConfigValidationError, match="runs must be >= 1"):
            cfg.validate()

    def test_negative_warmup_fails(self):
        cfg = BenchmarkConfig(warmup_runs=-1)
        with pytest.raises(ConfigValidationError, match="warmup_runs"):
            cfg.validate()

    def test_timeout_too_low_fails(self):
        cfg = BenchmarkConfig(timeout_seconds=10)
        with pytest.raises(ConfigValidationError, match="timeout_seconds"):
            cfg.validate()

    def test_empty_metrics_fails(self):
        cfg = BenchmarkConfig(metrics=[])
        with pytest.raises(ConfigValidationError, match="metrics list cannot be empty"):
            cfg.validate()

    def test_metric_empty_name_fails(self):
        cfg = BenchmarkConfig(metrics=[MetricConfig(name="")])
        with pytest.raises(ConfigValidationError, match="metric name cannot be empty"):
            cfg.validate()

    def test_degradation_enabled_no_pipelines_fails(self):
        cfg = BenchmarkConfig(degradation=DegradationConfig(enabled=True, pipelines=[]))
        with pytest.raises(ConfigValidationError, match="at least one pipeline"):
            cfg.validate()

    def test_empty_input_dir_fails(self):
        cfg = BenchmarkConfig(data=DataConfig(input_dir=""))
        with pytest.raises(ConfigValidationError, match="input_dir cannot be empty"):
            cfg.validate()

    def test_pdf_dpi_too_low_fails(self):
        cfg = BenchmarkConfig(data=DataConfig(pdf_dpi=50))
        with pytest.raises(ConfigValidationError, match="pdf_dpi must be >= 72"):
            cfg.validate()

    def test_invalid_unicode_form_fails(self):
        cfg = BenchmarkConfig(
            normalisation=NormalisationConfig(unicode_form="INVALID")  # type: ignore
        )
        with pytest.raises(ConfigValidationError, match="unicode_form"):
            cfg.validate()

    def test_model_empty_project_dir_fails(self):
        cfg = BenchmarkConfig(
            models={"M": ModelConfig(project_dir="", module="m.worker")}
        )
        with pytest.raises(ConfigValidationError, match="project_dir cannot be empty"):
            cfg.validate()

    def test_model_empty_module_fails(self):
        cfg = BenchmarkConfig(
            models={"M": ModelConfig(project_dir="models/m", module="")}
        )
        with pytest.raises(ConfigValidationError, match="module cannot be empty"):
            cfg.validate()

    def test_multiple_errors_reported(self):
        cfg = BenchmarkConfig(
            runs=0,
            warmup_runs=-1,
            metrics=[],
        )
        with pytest.raises(ConfigValidationError) as exc_info:
            cfg.validate()
        msg = str(exc_info.value)
        assert "runs must be >= 1" in msg
        assert "warmup_runs" in msg
        assert "metrics" in msg
