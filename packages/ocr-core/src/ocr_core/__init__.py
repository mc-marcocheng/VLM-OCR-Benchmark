"""ocr-core — general-purpose OCR benchmarking framework."""

from ocr_core.base_model import AbstractOCRModel
from ocr_core.benchmark import BenchmarkRunner
from ocr_core.config import BenchmarkConfig, load_config
from ocr_core.normalisation import NormalisationPipeline
from ocr_core.types import BBox, GroundTruth, OCRPage, OCRRegion

__all__ = [
    "BBox",
    "GroundTruth",
    "OCRPage",
    "OCRRegion",
    "AbstractOCRModel",
    "BenchmarkConfig",
    "load_config",
    "NormalisationPipeline",
    "BenchmarkRunner",
]
