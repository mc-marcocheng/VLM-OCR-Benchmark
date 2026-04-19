"""
Load input files (PDF / images), convert to page images, and load
ground-truth annotations in both legacy ``.txt`` and structured ``.json``
formats.
"""

from __future__ import annotations

import glob
import json
import os
import shutil

import fitz  # PyMuPDF
import pillow_heif
from loguru import logger
from ocr_core.types import GroundTruth, OCRPage
from PIL import Image

pillow_heif.register_heif_opener()

__all__ = ["DataLoader"]

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".heic",
    ".heif",
    ".webp",
}
DOC_EXTS = {".pdf"}


class DataLoader:
    def __init__(
        self,
        input_dir: str,
        processed_dir: str,
        groundtruth_dir: str,
        pdf_dpi: int = 200,
    ):
        self.input_dir = os.path.abspath(input_dir)
        self.processed_dir = os.path.abspath(processed_dir)
        self.groundtruth_dir = os.path.abspath(groundtruth_dir)
        self.pdf_dpi = pdf_dpi
        os.makedirs(self.processed_dir, exist_ok=True)

    # ── PDF → images ────────────────────────────────────────

    def _convert_pdf(self, pdf_path: str, out_dir: str) -> list[str]:
        os.makedirs(out_dir, exist_ok=True)
        zoom = self.pdf_dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        paths: list[str] = []
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                out = os.path.join(out_dir, f"page_{i + 1:04d}.png")
                img.save(out, optimize=True)
                paths.append(out)
        return paths

    # ── Load input ──────────────────────────────────────────

    def ensure_cached(self, test_set: str, file_name: str) -> list[str]:
        """
        Ensure page images are cached and return their paths.
        Does NOT load images into memory.
        """
        input_path = os.path.join(self.input_dir, test_set, file_name)
        if not os.path.isfile(input_path):
            raise FileNotFoundError(input_path)

        stem, ext = os.path.splitext(file_name)
        cache_dir = os.path.join(self.processed_dir, test_set, stem)

        # Serve from cache
        if os.path.isdir(cache_dir):
            cached = sorted(glob.glob(os.path.join(cache_dir, "*.png")))
            if cached:
                source_mtime = os.path.getmtime(input_path)
                cache_mtime = min(os.path.getmtime(c) for c in cached)
                if cache_mtime >= source_mtime:
                    return cached
                logger.info(f"Cache stale for {file_name}, regenerating")
                shutil.rmtree(cache_dir, ignore_errors=True)

        os.makedirs(cache_dir, exist_ok=True)

        if ext.lower() in DOC_EXTS:
            return self._convert_pdf(input_path, cache_dir)

        if ext.lower() in IMAGE_EXTS:
            img = Image.open(input_path).convert("RGB")
            out = os.path.join(cache_dir, f"{stem}.png")
            img.save(out, optimize=True)
            return [out]

        raise ValueError(f"Unsupported file type: {ext}")

    def load_input(
        self, test_set: str, file_name: str
    ) -> tuple[list[str], list[Image.Image]]:
        """
        Returns ``(image_paths, pil_images)`` for every page.
        Results are cached under ``processed_dir``.
        """
        paths = self.ensure_cached(test_set, file_name)
        return paths, [Image.open(p).convert("RGB") for p in paths]

    # ── Ground truth ────────────────────────────────────────

    def load_ground_truth(self, test_set: str, file_name: str) -> GroundTruth | None:
        """
        Try structured ``.json`` first, fall back to legacy ``.txt``.
        Returns ``None`` when no ground truth exists.
        """
        stem = os.path.splitext(file_name)[0]
        gt_dir = os.path.join(self.groundtruth_dir, test_set)

        # ── Structured JSON ──
        json_path = os.path.join(gt_dir, f"{stem}.json")
        if os.path.isfile(json_path):
            return self._load_json_gt(json_path, file_name)

        # ── Legacy plain-text ──
        txt_path = os.path.join(gt_dir, f"{stem}.txt")
        if os.path.isfile(txt_path):
            return self._load_txt_gt(txt_path, file_name)

        logger.debug(f"No ground truth for {test_set}/{file_name}")
        return None

    @staticmethod
    def _load_json_gt(path: str, file_name: str) -> GroundTruth | None:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError:
            logger.error(f"Malformed JSON ground truth: {path}")
            return None

        try:
            gt = GroundTruth.from_dict(data)
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Invalid ground truth structure in {path}: {e}")
            return None

        gt.source_file = gt.source_file or file_name
        logger.debug(f"Loaded JSON GT ({len(gt.pages)} pages): {path}")
        return gt

    @staticmethod
    def _load_txt_gt(path: str, file_name: str) -> GroundTruth:
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
        if "[PAGE_BREAK]" in content:
            texts = [t.strip() for t in content.split("[PAGE_BREAK]") if t.strip()]
        else:
            texts = [content.strip()]

        pages = [OCRPage(page_number=i + 1, full_text=t) for i, t in enumerate(texts)]
        logger.debug(f"Loaded TXT GT ({len(pages)} pages): {path}")
        return GroundTruth(source_file=file_name, pages=pages)

    # ── Discovery ───────────────────────────────────────────

    def list_test_sets(self) -> list[str]:
        if not os.path.isdir(self.input_dir):
            return []
        return sorted(
            d
            for d in os.listdir(self.input_dir)
            if os.path.isdir(os.path.join(self.input_dir, d))
        )

    def list_files(self, test_set: str) -> list[str]:
        dir_path = os.path.join(self.input_dir, test_set)
        if not os.path.isdir(dir_path):
            return []
        files = []
        for name in sorted(os.listdir(dir_path)):
            path = os.path.join(dir_path, name)
            if os.path.isfile(path):
                ext = os.path.splitext(name)[1].lower()
                if ext in IMAGE_EXTS | DOC_EXTS:
                    files.append(name)
        return files
