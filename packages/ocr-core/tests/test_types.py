"""Tests for ocr_core.types — serialisation round-trips and edge cases."""

from __future__ import annotations

import pytest

from ocr_core.types import (
    BBox,
    GroundTruth,
    OCRPage,
    OCRRegion,
    WorkerPageResult,
    WorkerResponse,
    WorkerTask,
)

# ═══════════════════════════════════════════════════════════════
#  BBox
# ═══════════════════════════════════════════════════════════════


class TestBBox:
    def test_width_height_area(self):
        b = BBox(10, 20, 110, 70)
        assert b.width == 100
        assert b.height == 50
        assert b.area == 5000

    def test_zero_area(self):
        b = BBox(5, 5, 5, 5)
        assert b.width == 0
        assert b.height == 0
        assert b.area == 0

    def test_inverted_coords_clamp_to_zero(self):
        b = BBox(100, 100, 50, 50)
        assert b.width == 0
        assert b.height == 0
        assert b.area == 0

    def test_iou_identical(self, bbox_a):
        assert bbox_a.iou(bbox_a) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(20, 20, 30, 30)
        assert a.iou(b) == 0.0

    def test_iou_partial(self, bbox_a, bbox_b):
        iou = bbox_a.iou(bbox_b)
        # intersection = 50*50 = 2500, union = 10000+10000-2500 = 17500
        assert iou == pytest.approx(2500 / 17500)

    def test_iou_zero_area_box(self):
        a = BBox(0, 0, 0, 0)
        b = BBox(0, 0, 10, 10)
        assert a.iou(b) == 0.0

    def test_iou_both_zero_area(self):
        a = BBox(0, 0, 0, 0)
        b = BBox(0, 0, 0, 0)
        assert a.iou(b) == 0.0

    def test_to_dict_from_dict_roundtrip(self):
        original = BBox(1.5, 2.5, 3.5, 4.5)
        d = original.to_dict()
        restored = BBox.from_dict(d)
        assert restored == original

    def test_from_dict_list(self):
        b = BBox.from_dict([10, 20, 30, 40])
        assert b == BBox(10, 20, 30, 40)

    def test_from_dict_tuple(self):
        b = BBox.from_dict((10, 20, 30, 40))
        assert b == BBox(10, 20, 30, 40)

    def test_from_dict_list_too_short(self):
        with pytest.raises(ValueError, match="4 values"):
            BBox.from_dict([1, 2])

    def test_frozen(self):
        b = BBox(0, 0, 10, 10)
        with pytest.raises(AttributeError):
            b.x1 = 5  # type: ignore[misc]

    def test_hashable(self):
        b1 = BBox(0, 0, 10, 10)
        b2 = BBox(0, 0, 10, 10)
        assert hash(b1) == hash(b2)
        assert {b1, b2} == {b1}

    def test_from_dict_list_format(self):
        bbox = BBox.from_dict([10, 20, 100, 200])
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 100
        assert bbox.y2 == 200

    def test_from_dict_list_short(self):
        with pytest.raises(ValueError, match="requires 4 values"):
            BBox.from_dict([10, 20, 100])

    def test_from_dict_tuple_format(self):
        bbox = BBox.from_dict((10, 20, 100, 200))
        assert bbox.x1 == 10

    def test_iou_complete_overlap(self):
        b1 = BBox(0, 0, 10, 10)
        assert b1.iou(b1) == 1.0


# ═══════════════════════════════════════════════════════════════
#  OCRRegion
# ═══════════════════════════════════════════════════════════════


class TestOCRRegion:
    def test_defaults(self):
        r = OCRRegion()
        assert r.text == ""
        assert r.category == "text"
        assert r.bbox is None
        assert r.confidence == -1.0
        assert r.order == -1
        assert r.children == []

    def test_roundtrip(self):
        r = OCRRegion(
            text="hello",
            category="title",
            bbox=BBox(1, 2, 3, 4),
            confidence=0.95,
            text_format="markdown",
            order=0,
            children=[OCRRegion(text="child", category="text")],
        )
        d = r.to_dict()
        restored = OCRRegion.from_dict(d)
        assert restored.text == "hello"
        assert restored.category == "title"
        assert restored.bbox == BBox(1, 2, 3, 4)
        assert restored.confidence == 0.95
        assert restored.order == 0
        assert len(restored.children) == 1
        assert restored.children[0].text == "child"

    def test_from_dict_missing_fields(self):
        r = OCRRegion.from_dict({})
        assert r.text == ""
        assert r.category == "text"
        assert r.bbox is None

    def test_to_dict_no_bbox_no_children(self):
        r = OCRRegion(text="x")
        d = r.to_dict()
        assert "bbox" not in d
        assert "children" not in d

    def test_from_dict_with_children(self):
        data = {
            "text": "Parent",
            "category": "text",
            "children": [
                {"text": "Child1", "category": "text"},
                {"text": "Child2", "category": "text"},
            ],
        }
        region = OCRRegion.from_dict(data)
        assert len(region.children) == 2
        assert region.children[0].text == "Child1"

    def test_to_dict_with_bbox(self):
        region = OCRRegion(
            text="Test",
            bbox=BBox(0, 0, 100, 50),
        )
        d = region.to_dict()
        assert "bbox" in d
        assert d["bbox"]["x1"] == 0

    def test_to_dict_with_children(self):
        child = OCRRegion(text="Child")
        parent = OCRRegion(text="Parent", children=[child])
        d = parent.to_dict()
        assert "children" in d
        assert len(d["children"]) == 1


# ═══════════════════════════════════════════════════════════════
#  OCRPage
# ═══════════════════════════════════════════════════════════════


class TestOCRPage:
    def test_empty_page(self):
        p = OCRPage()
        assert p.full_text == ""
        assert not p.has_regions()
        assert not p.has_bboxes()
        assert p.regions_by_category("text") == []

    def test_regions_by_category(self):
        p = OCRPage(
            regions=[
                OCRRegion(text="Title", category="title"),
                OCRRegion(text="Body", category="text"),
                OCRRegion(text="Footer", category="page-footer"),
                OCRRegion(text="Body2", category="text"),
            ]
        )
        texts = p.regions_by_category("text")
        assert len(texts) == 2
        titles = p.regions_by_category("title", "page-footer")
        assert len(titles) == 2

    def test_has_bboxes(self):
        p = OCRPage(regions=[OCRRegion(bbox=BBox(0, 0, 1, 1))])
        assert p.has_bboxes()
        p2 = OCRPage(regions=[OCRRegion()])
        assert not p2.has_bboxes()

    def test_roundtrip(self):
        p = OCRPage(
            page_number=3,
            full_text="test",
            width=800,
            height=600,
            regions=[OCRRegion(text="r1")],
        )
        d = p.to_dict()
        restored = OCRPage.from_dict(d)
        assert restored.page_number == 3
        assert restored.full_text == "test"
        assert restored.width == 800
        assert len(restored.regions) == 1

    def test_from_dict_legacy_page_key(self):
        p = OCRPage.from_dict({"page": 5, "full_text": "hi"})
        assert p.page_number == 5

    def test_from_dict_with_page_alias(self):
        """Test that 'page' is accepted as alias for 'page_number'."""
        data = {"page": 5, "full_text": "Test"}
        page = OCRPage.from_dict(data)
        assert page.page_number == 5

    def test_regions_by_category_multiple(self):
        page = OCRPage(
            regions=[
                OCRRegion(category="text"),
                OCRRegion(category="title"),
                OCRRegion(category="table"),
                OCRRegion(category="text"),
            ]
        )
        text_regions = page.regions_by_category("text", "title")
        assert len(text_regions) == 3

    def test_has_bboxes_false(self):
        page = OCRPage(
            regions=[
                OCRRegion(text="No bbox"),
            ]
        )
        assert not page.has_bboxes()

    def test_has_bboxes_true(self):
        page = OCRPage(
            regions=[
                OCRRegion(text="With bbox", bbox=BBox(0, 0, 10, 10)),
            ]
        )
        assert page.has_bboxes()


# ═══════════════════════════════════════════════════════════════
#  GroundTruth
# ═══════════════════════════════════════════════════════════════


class TestGroundTruth:
    def test_full_text_joins_pages(self):
        gt = GroundTruth(
            pages=[
                OCRPage(full_text="Page one"),
                OCRPage(full_text="Page two"),
            ]
        )
        assert gt.full_text == "Page one\n\nPage two"

    def test_empty_ground_truth(self):
        gt = GroundTruth()
        assert gt.full_text == ""

    def test_roundtrip(self):
        gt = GroundTruth(
            source_file="doc.pdf",
            pages=[OCRPage(page_number=1, full_text="hello")],
        )
        d = gt.to_dict()
        restored = GroundTruth.from_dict(d)
        assert restored.source_file == "doc.pdf"
        assert len(restored.pages) == 1

    def test_from_dict_legacy_file_key(self):
        gt = GroundTruth.from_dict({"file": "x.pdf", "pages": []})
        assert gt.source_file == "x.pdf"

    def test_full_text_property(self):
        gt = GroundTruth(
            pages=[
                OCRPage(full_text="Page 1"),
                OCRPage(full_text="Page 2"),
            ]
        )
        assert gt.full_text == "Page 1\n\nPage 2"

    def test_from_dict_with_file_alias(self):
        """Test that 'file' is accepted as alias for 'source_file'."""
        data = {"file": "test.pdf", "pages": []}
        gt = GroundTruth.from_dict(data)
        assert gt.source_file == "test.pdf"

    def test_to_dict(self):
        gt = GroundTruth(
            source_file="test.pdf",
            pages=[OCRPage(page_number=1, full_text="Content")],
        )
        d = gt.to_dict()
        assert d["source_file"] == "test.pdf"
        assert len(d["pages"]) == 1


# ═══════════════════════════════════════════════════════════════
#  Worker protocol types
# ═══════════════════════════════════════════════════════════════


class TestWorkerTask:
    def test_roundtrip(self):
        t = WorkerTask(
            image_paths=["/a.png", "/b.png"], device="cuda", params={"k": "v"}
        )
        d = t.to_dict()
        restored = WorkerTask.from_dict(d)
        assert restored.image_paths == ["/a.png", "/b.png"]
        assert restored.device == "cuda"
        assert restored.params == {"k": "v"}

    def test_defaults(self):
        t = WorkerTask.from_dict({})
        assert t.image_paths == []
        assert t.device == "cpu"

    def test_worker_task_round_trip(self):
        task = WorkerTask(
            image_paths=["/path/to/image.png"],
            device="cuda",
            params={"max_tokens": 1000},
        )
        d = task.to_dict()
        restored = WorkerTask.from_dict(d)
        assert restored.image_paths == task.image_paths
        assert restored.device == task.device
        assert restored.params == task.params


class TestWorkerPageResult:
    def test_roundtrip(self):
        wpr = WorkerPageResult(
            image_path="/img.png",
            prediction_time_seconds=1.23,
            ram_after_mb=512.0,
            result=OCRPage(full_text="detected"),
            error=None,
        )
        d = wpr.to_dict()
        assert "error" not in d  # None errors omitted
        restored = WorkerPageResult.from_dict(d)
        assert restored.image_path == "/img.png"
        assert restored.result.full_text == "detected"

    def test_with_error(self):
        wpr = WorkerPageResult(error="OOM")
        d = wpr.to_dict()
        assert d["error"] == "OOM"
        restored = WorkerPageResult.from_dict(d)
        assert restored.error == "OOM"

    def test_worker_page_result_with_error(self):
        result = WorkerPageResult(
            image_path="/path/image.png",
            error="Model failed",
        )
        d = result.to_dict()
        assert d["error"] == "Model failed"

    def test_worker_page_result_no_error(self):
        result = WorkerPageResult(image_path="/path/image.png")
        d = result.to_dict()
        assert "error" not in d


class TestWorkerResponse:
    def test_roundtrip(self):
        resp = WorkerResponse(
            model_load_time_seconds=5.0,
            ram_before_load_mb=100,
            ram_after_load_mb=500,
            peak_ram_mb=600,
            vram_before_load_mb=0,
            vram_after_load_mb=2000,
            peak_vram_mb=3000,
            pages=[
                WorkerPageResult(
                    image_path="/a.png",
                    prediction_time_seconds=0.5,
                    result=OCRPage(full_text="text"),
                )
            ],
        )
        d = resp.to_dict()
        restored = WorkerResponse.from_dict(d)
        assert restored.model_load_time_seconds == 5.0
        assert restored.peak_vram_mb == 3000
        assert len(restored.pages) == 1

    def test_defaults(self):
        resp = WorkerResponse.from_dict({})
        assert resp.model_load_time_seconds == 0.0
        assert resp.vram_before_load_mb is None
        assert resp.pages == []

    def test_worker_response_round_trip(self):
        response = WorkerResponse(
            model_load_time_seconds=1.5,
            ram_before_load_mb=1000,
            ram_after_load_mb=2000,
            peak_ram_mb=2500,
            vram_before_load_mb=100,
            vram_after_load_mb=500,
            peak_vram_mb=800,
            pages=[
                WorkerPageResult(
                    image_path="/path/image.png",
                    prediction_time_seconds=0.5,
                    result=OCRPage(full_text="Hello"),
                )
            ],
        )
        d = response.to_dict()
        restored = WorkerResponse.from_dict(d)
        assert restored.model_load_time_seconds == 1.5
        assert len(restored.pages) == 1
        assert restored.pages[0].result.full_text == "Hello"
