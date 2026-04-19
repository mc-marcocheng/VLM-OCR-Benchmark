"""
YAML-driven configuration with typed dataclass defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml
from loguru import logger

__all__ = [
    "BenchmarkConfig",
    "ConfigValidationError",
    "DataConfig",
    "DegradationConfig",
    "DegradationStep",
    "MetricConfig",
    "ModelConfig",
    "NormalisationConfig",
    "load_config",
]


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails."""
    pass


@dataclass
class NormalisationConfig:
    unicode_form: str = "NFKC"
    lowercase: bool = False
    strip_whitespace: bool = True
    collapse_whitespace: bool = True
    remove_punctuation: bool = False
    fullwidth_to_halfwidth: bool = False
    traditional_to_simplified: bool = False
    custom_replacements: dict[str, str] = field(default_factory=dict)


@dataclass
class MetricConfig:
    name: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    apply_to: list[str] = field(default_factory=list)  # region categories


@dataclass
class DegradationStep:
    name: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DegradationConfig:
    enabled: bool = False
    pipelines: list[DegradationStep] = field(default_factory=list)


@dataclass
class ModelConfig:
    project_dir: str = ""
    module: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    input_dir: str = "data/inputs"
    groundtruth_dir: str = "data/groundtruths"
    processed_dir: str = "data/processed"
    results_dir: str = "results"
    pdf_dpi: int = 200


@dataclass
class BenchmarkConfig:
    runs: int = 1
    warmup_runs: int = 0
    timeout_seconds: int = 3600
    normalisation: NormalisationConfig = field(default_factory=NormalisationConfig)
    metrics: list[MetricConfig] = field(default_factory=lambda: [
        MetricConfig(name="cer"),
        MetricConfig(name="wer"),
        MetricConfig(name="char_f1"),
    ])
    degradation: DegradationConfig = field(default_factory=DegradationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    models: dict[str, ModelConfig] = field(default_factory=dict)
    _source_path: str = field(default="", init=False, repr=False, compare=False)

    def validate(self) -> None:
        """Validate configuration and raise ConfigValidationError if invalid."""
        errors: list[str] = []

        if self.runs < 1:
            errors.append(f"runs must be >= 1, got {self.runs}")
        if self.warmup_runs < 0:
            errors.append(f"warmup_runs must be >= 0, got {self.warmup_runs}")
        if self.timeout_seconds < 60:
            errors.append(f"timeout_seconds must be >= 60, got {self.timeout_seconds}")

        if not self.metrics:
            errors.append("metrics list cannot be empty")
        for m in self.metrics:
            if not m.name:
                errors.append("metric name cannot be empty")

        if self.degradation.enabled and not self.degradation.pipelines:
            errors.append("degradation.enabled=True requires at least one pipeline step")

        if not self.data.input_dir:
            errors.append("data.input_dir cannot be empty")
        if not self.data.groundtruth_dir:
            errors.append("data.groundtruth_dir cannot be empty")
        if self.data.pdf_dpi < 72:
            errors.append(f"data.pdf_dpi must be >= 72, got {self.data.pdf_dpi}")

        valid_unicode_forms = {"NFC", "NFD", "NFKC", "NFKD", ""}
        if self.normalisation.unicode_form not in valid_unicode_forms:
            errors.append(
                f"normalisation.unicode_form must be one of {valid_unicode_forms}, "
                f"got {self.normalisation.unicode_form!r}"
            )

        for name, mcfg in self.models.items():
            if not mcfg.project_dir:
                errors.append(f"model {name!r}: project_dir cannot be empty")
            if not mcfg.module:
                errors.append(f"model {name!r}: module cannot be empty")

        if errors:
            raise ConfigValidationError("\n".join(f"  - {e}" for e in errors))


def _build_normalisation(d: dict) -> NormalisationConfig:
    return NormalisationConfig(**{k: v for k, v in d.items()
                                  if k in NormalisationConfig.__dataclass_fields__})


def _build_metric(d: dict | str) -> MetricConfig:
    if isinstance(d, str):
        return MetricConfig(name=d)
    return MetricConfig(
        name=d.get("name", ""),
        params=d.get("params", {}),
        apply_to=d.get("apply_to", []),
    )


def _build_degradation(d: dict) -> DegradationConfig:
    return DegradationConfig(
        enabled=d.get("enabled", False),
        pipelines=[DegradationStep(**s) for s in d.get("pipelines", [])],
    )


def _build_model(d: dict) -> ModelConfig:
    return ModelConfig(
        project_dir=d.get("project_dir", ""),
        module=d.get("module", ""),
        params=d.get("params", {}),
    )


def load_config(path: str | None = None) -> BenchmarkConfig:
    """Load configuration from *path* (or return defaults)."""
    if path is None:
        logger.info("No config file — using defaults")
        return BenchmarkConfig()

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    logger.info(f"Loaded config from {path}")

    cfg = BenchmarkConfig(
        runs=raw.get("benchmark", {}).get("runs", 1),
        warmup_runs=raw.get("benchmark", {}).get("warmup_runs", 0),
        timeout_seconds=raw.get("benchmark", {}).get("timeout_seconds", 3600),
    )

    if "normalisation" in raw:
        cfg.normalisation = _build_normalisation(raw["normalisation"])

    if "metrics" in raw:
        cfg.metrics = [_build_metric(m) for m in raw["metrics"]]

    if "degradation" in raw:
        cfg.degradation = _build_degradation(raw["degradation"])

    if "data" in raw:
        d = raw["data"]
        cfg.data = DataConfig(**{k: v for k, v in d.items()
                                  if k in DataConfig.__dataclass_fields__})

    if "models" in raw:
        cfg.models = {name: _build_model(body)
                      for name, body in raw["models"].items()}

    cfg._source_path = path or ""
    cfg.validate()
    logger.info("Config validation passed")

    return cfg
