"""Tests for ocr_core.data_loader."""

from __future__ import annotations

import json
import os

import pytest
from PIL import Image

from ocr_core.data_loader import DataLoader


@pytest.fixture
def data_dirs(tmp_dir):
    input_dir = os.path.join(tmp_dir, "inputs")
    gt_dir = os.path.join(tmp_dir, "groundtruths")
    proc_dir = os.path.join(tmp_dir, "processed")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    return input_dir, gt_dir, proc_dir


@pytest.fixture
def loader(data_dirs):
    input_dir, gt_dir, proc_dir = data_dirs
    return DataLoader(
        input_dir=input_dir,
        groundtruth_dir=gt_dir,
        processed_dir=proc_dir,
        pdf_dpi=72,
    )


def _create_image(path: str, size=(100, 100)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size, "white").save(path)


class TestListTestSets:
    def test_empty(self, loader):
        assert loader.list_test_sets() == []

    def test_with_subdirectories(self, loader, data_dirs):
        input_dir = data_dirs[0]
        os.makedirs(os.path.join(input_dir, "setA"))
        os.makedirs(os.path.join(input_dir, "setB"))
        result = loader.list_test_sets()
        assert result == ["setA", "setB"]

    def test_files_ignored(self, loader, data_dirs):
        input_dir = data_dirs[0]
        os.makedirs(os.path.join(input_dir, "testset"))
        with open(os.path.join(input_dir, "readme.txt"), "w") as f:
            f.write("not a dir")
        assert loader.list_test_sets() == ["testset"]


class TestListFiles:
    def test_empty_set(self, loader):
        assert loader.list_files("nonexistent") == []

    def test_images_only(self, loader, data_dirs):
        input_dir = data_dirs[0]
        set_dir = os.path.join(input_dir, "test")
        os.makedirs(set_dir)
        _create_image(os.path.join(set_dir, "a.png"))
        _create_image(os.path.join(set_dir, "b.jpg"))
        with open(os.path.join(set_dir, "readme.txt"), "w") as f:
            f.write("ignore")
        result = loader.list_files("test")
        assert set(result) == {"a.png", "b.jpg"}

    def test_sorted(self, loader, data_dirs):
        input_dir = data_dirs[0]
        set_dir = os.path.join(input_dir, "s")
        os.makedirs(set_dir)
        _create_image(os.path.join(set_dir, "c.png"))
        _create_image(os.path.join(set_dir, "a.png"))
        _create_image(os.path.join(set_dir, "b.png"))
        assert loader.list_files("s") == ["a.png", "b.png", "c.png"]


class TestEnsureCached:
    def test_image_cached(self, loader, data_dirs):
        input_dir = data_dirs[0]
        set_dir = os.path.join(input_dir, "s1")
        os.makedirs(set_dir)
        _create_image(os.path.join(set_dir, "img.png"))

        paths = loader.ensure_cached("s1", "img.png")
        assert len(paths) == 1
        assert os.path.isfile(paths[0])
        assert paths[0].endswith(".png")

    def test_cache_reused(self, loader, data_dirs):
        input_dir = data_dirs[0]
        set_dir = os.path.join(input_dir, "s1")
        os.makedirs(set_dir)
        _create_image(os.path.join(set_dir, "img.png"))

        paths1 = loader.ensure_cached("s1", "img.png")
        paths2 = loader.ensure_cached("s1", "img.png")
        assert paths1 == paths2

    def test_missing_file_raises(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.ensure_cached("s1", "nonexistent.png")

    def test_unsupported_format_raises(self, loader, data_dirs):
        input_dir = data_dirs[0]
        set_dir = os.path.join(input_dir, "s1")
        os.makedirs(set_dir)
        path = os.path.join(set_dir, "data.csv")
        with open(path, "w") as f:
            f.write("a,b,c")
        with pytest.raises(ValueError, match="Unsupported"):
            loader.ensure_cached("s1", "data.csv")

    def test_heic_format(self, loader, data_dirs):
        """Test that HEIC images can be listed (actual decode needs pillow-heif)."""
        input_dir = data_dirs[0]
        set_dir = os.path.join(input_dir, "s1")
        os.makedirs(set_dir)
        # Create a dummy HEIC file — won't decode, but listing should work
        path = os.path.join(set_dir, "photo.heic")
        with open(path, "wb") as f:
            f.write(b"\x00" * 100)
        files = loader.list_files("s1")
        assert "photo.heic" in files


class TestLoadInput:
    def test_returns_paths_and_images(self, loader, data_dirs):
        input_dir = data_dirs[0]
        set_dir = os.path.join(input_dir, "s1")
        os.makedirs(set_dir)
        _create_image(os.path.join(set_dir, "test.png"), size=(50, 30))

        paths, images = loader.load_input("s1", "test.png")
        assert len(paths) == 1
        assert len(images) == 1
        assert images[0].size == (50, 30)
        assert images[0].mode == "RGB"


class TestLoadGroundTruth:
    def test_no_gt_returns_none(self, loader):
        result = loader.load_ground_truth("test", "nonexistent.png")
        assert result is None

    def test_txt_single_page(self, loader, data_dirs):
        gt_dir = data_dirs[1]
        os.makedirs(os.path.join(gt_dir, "s1"))
        with open(os.path.join(gt_dir, "s1", "doc.txt"), "w", encoding="utf-8") as f:
            f.write("Hello world")
        gt = loader.load_ground_truth("s1", "doc.png")
        assert gt is not None
        assert len(gt.pages) == 1
        assert gt.pages[0].full_text == "Hello world"

    def test_txt_multi_page(self, loader, data_dirs):
        gt_dir = data_dirs[1]
        os.makedirs(os.path.join(gt_dir, "s1"))
        with open(os.path.join(gt_dir, "s1", "doc.txt"), "w", encoding="utf-8") as f:
            f.write("Page one\n[PAGE_BREAK]\nPage two\n[PAGE_BREAK]\nPage three")
        gt = loader.load_ground_truth("s1", "doc.pdf")
        assert gt is not None
        assert len(gt.pages) == 3
        assert gt.pages[0].full_text == "Page one"
        assert gt.pages[2].full_text == "Page three"

    def test_json_gt(self, loader, data_dirs):
        gt_dir = data_dirs[1]
        os.makedirs(os.path.join(gt_dir, "s1"))
        data = {
            "source_file": "doc.pdf",
            "pages": [
                {
                    "page_number": 1,
                    "full_text": "Hello JSON",
                    "regions": [
                        {
                            "text": "Hello JSON",
                            "category": "text",
                            "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 50},
                        }
                    ],
                }
            ],
        }
        with open(os.path.join(gt_dir, "s1", "doc.json"), "w", encoding="utf-8") as f:
            json.dump(data, f)
        gt = loader.load_ground_truth("s1", "doc.pdf")
        assert gt is not None
        assert len(gt.pages) == 1
        assert gt.pages[0].full_text == "Hello JSON"
        assert len(gt.pages[0].regions) == 1
        assert gt.pages[0].regions[0].bbox is not None

    def test_json_takes_priority_over_txt(self, loader, data_dirs):
        gt_dir = data_dirs[1]
        os.makedirs(os.path.join(gt_dir, "s1"))
        with open(os.path.join(gt_dir, "s1", "doc.txt"), "w") as f:
            f.write("From TXT")
        data = {"pages": [{"full_text": "From JSON"}]}
        with open(os.path.join(gt_dir, "s1", "doc.json"), "w") as f:
            json.dump(data, f)
        gt = loader.load_ground_truth("s1", "doc.pdf")
        assert gt.pages[0].full_text == "From JSON"

    def test_malformed_json(self, loader, data_dirs, suppress_loguru):
        gt_dir = data_dirs[1]
        os.makedirs(os.path.join(gt_dir, "s1"))
        with open(os.path.join(gt_dir, "s1", "doc.json"), "w") as f:
            f.write("{broken json")
        result = loader.load_ground_truth("s1", "doc.pdf")
        assert result is None

    def test_source_file_set_from_filename(self, loader, data_dirs):
        gt_dir = data_dirs[1]
        os.makedirs(os.path.join(gt_dir, "s1"))
        with open(os.path.join(gt_dir, "s1", "img.txt"), "w") as f:
            f.write("text")
        gt = loader.load_ground_truth("s1", "img.png")
        assert gt.source_file == "img.png"

    def test_empty_txt(self, loader, data_dirs):
        gt_dir = data_dirs[1]
        os.makedirs(os.path.join(gt_dir, "s1"))
        with open(os.path.join(gt_dir, "s1", "empty.txt"), "w") as f:
            f.write("")
        gt = loader.load_ground_truth("s1", "empty.png")
        assert gt is not None
        assert len(gt.pages) == 1
        assert gt.pages[0].full_text == ""

    def test_load_txt_gt_with_page_breaks(self, loader, data_dirs):
        set_dir = os.path.join(data_dirs[1], "test_set")
        os.makedirs(set_dir)

        gt_content = (
            "Page 1 content[PAGE_BREAK]Page 2 content[PAGE_BREAK]Page 3 content"
        )
        with open(os.path.join(set_dir, "doc.txt"), "w") as f:
            f.write(gt_content)

        gt = loader.load_ground_truth("test_set", "doc.pdf")
        assert gt is not None
        assert len(gt.pages) == 3
        assert gt.pages[0].full_text == "Page 1 content"
        assert gt.pages[1].full_text == "Page 2 content"
        assert gt.pages[2].full_text == "Page 3 content"

    def test_load_json_gt_invalid_structure(self, loader, data_dirs):
        set_dir = os.path.join(data_dirs[1], "test_set")
        os.makedirs(set_dir)

        # Valid JSON but invalid structure for GroundTruth
        with open(os.path.join(set_dir, "doc.json"), "w") as f:
            json.dump({"pages": [{"invalid_field": "value"}]}, f)

        gt = loader.load_ground_truth("test_set", "doc.pdf")
        # Should still work due to defaults in from_dict
        assert gt is not None
