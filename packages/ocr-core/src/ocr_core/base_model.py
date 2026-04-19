"""
Abstract interface that every OCR model adapter implements.

Workers in separate venvs may inherit this (since ocr-core is a dependency),
or they can simply produce the correct JSON output without importing it.
"""

from abc import ABC, abstractmethod

import psutil
from ocr_core.types import OCRPage
from PIL import Image

__all__ = ["AbstractOCRModel"]


class AbstractOCRModel(ABC):
    """Base class for in-process model adapters."""

    def __init__(self, model_path: str = "", device: str = "cpu"):
        self.model_path = model_path
        self.device = device

    @abstractmethod
    def load_model(self) -> None:
        """Load weights and processor into memory."""

    @abstractmethod
    def predict(self, image: Image.Image) -> OCRPage:
        """
        Run inference on a single page image.

        Returns an ``OCRPage`` containing at minimum ``full_text``.
        When the model supports layout detection, ``regions`` should
        be populated with bounding boxes and categories.
        """

    @staticmethod
    def get_ram_usage_mb() -> float:
        return psutil.Process().memory_info().rss / (1024 * 1024)
