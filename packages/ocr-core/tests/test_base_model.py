"""Tests for ocr_core.base_model."""

from __future__ import annotations

import pytest
from PIL import Image

from ocr_core.base_model import AbstractOCRModel
from ocr_core.types import OCRPage


class ConcreteModel(AbstractOCRModel):
    """Minimal concrete implementation for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._loaded = False

    def load_model(self):
        self._loaded = True

    def predict(self, image: Image.Image) -> OCRPage:
        return OCRPage(full_text="mock prediction")


class TestAbstractOCRModel:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            AbstractOCRModel()

    def test_concrete_instantiation(self):
        m = ConcreteModel(model_path="/path", device="cuda")
        assert m.model_path == "/path"
        assert m.device == "cuda"

    def test_load_model(self):
        m = ConcreteModel()
        m.load_model()
        assert m._loaded

    def test_predict(self):
        m = ConcreteModel()
        img = Image.new("RGB", (10, 10))
        result = m.predict(img)
        assert isinstance(result, OCRPage)
        assert result.full_text == "mock prediction"

    def test_get_ram_usage(self):
        ram = AbstractOCRModel.get_ram_usage_mb()
        assert isinstance(ram, float)
        assert ram > 0
