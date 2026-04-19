"""
Configurable text-normalisation pipeline.

Each step is independently toggleable via ``NormalisationConfig``.
"""

from __future__ import annotations

import re
import threading
import unicodedata
from typing import Optional

from loguru import logger

from ocr_core.config import NormalisationConfig

__all__ = ["NormalisationPipeline"]

# ── CJK detection ───────────────────────────────────────────

_CJK_RANGES = (
    "\u4e00-\u9fff"
    "\u3400-\u4dbf"
    "\U00020000-\U0002a6df"
    "\U0002a700-\U0002b73f"
    "\U0002b740-\U0002b81f"
    "\U0002b820-\U0002ceaf"
    "\U0002ceb0-\U0002ebef"
    "\U00030000-\U0003134f"
    "\u3000-\u303f"
    "\uff00-\uffef"
)
_CJK_RE = re.compile(f"([{_CJK_RANGES}])")

# ── Full-width → half-width table ───────────────────────────

_FW_OFFSET = 0xFEE0  # ord('！') - ord('!')


def _fullwidth_to_halfwidth(text: str) -> str:
    out: list[str] = []
    for ch in text:
        cp = ord(ch)
        if 0xFF01 <= cp <= 0xFF5E:
            out.append(chr(cp - _FW_OFFSET))
        elif cp == 0x3000:  # ideographic space
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


# ── Optional: Traditional → Simplified Chinese ──────────────

_opencc_converter: Optional[object] = None
_opencc_unavailable: bool = False
_opencc_lock = threading.Lock()


def _t2s(text: str) -> str:
    global _opencc_converter, _opencc_unavailable
    if _opencc_unavailable:
        return text
    if _opencc_converter is None:
        with _opencc_lock:
            if _opencc_converter is None and not _opencc_unavailable:
                try:
                    from opencc import OpenCC
                    _opencc_converter = OpenCC("t2s")
                except ImportError:
                    _opencc_unavailable = True
                    logger.warning(
                        "opencc-python-reimplemented not installed — "
                        "traditional_to_simplified will be skipped"
                    )
                    return text
    return _opencc_converter.convert(text)  # type: ignore[union-attr]


# ── Punctuation removal ─────────────────────────────────────

_PUNCT_RE = re.compile(
    r"[\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E"
    r"\u2000-\u206F\u3000-\u303F\uFF00-\uFF0F\uFF1A-\uFF20"
    r"\uFF3B-\uFF40\uFF5B-\uFF65\uFE30-\uFE4F]"
)


# ── Pipeline ────────────────────────────────────────────────


class NormalisationPipeline:
    """Applies a sequence of normalisations controlled by config."""

    def __init__(self, config: NormalisationConfig | None = None):
        self.cfg = config or NormalisationConfig()

    def __call__(self, text: str) -> str:
        return self.apply(text)

    def apply(self, text: str) -> str:
        if not text:
            return ""

        # 1. Unicode normalisation
        if self.cfg.unicode_form:
            text = unicodedata.normalize(self.cfg.unicode_form, text)

        # 2. Full-width → half-width
        if self.cfg.fullwidth_to_halfwidth:
            text = _fullwidth_to_halfwidth(text)

        # 3. Traditional → Simplified Chinese
        if self.cfg.traditional_to_simplified:
            text = _t2s(text)

        # 4. Custom replacements
        for old, new in self.cfg.custom_replacements.items():
            text = text.replace(old, new)

        # 5. Lowercase
        if self.cfg.lowercase:
            text = text.lower()

        # 6. Remove punctuation
        if self.cfg.remove_punctuation:
            text = _PUNCT_RE.sub("", text)

        # 7. Whitespace
        if self.cfg.strip_whitespace:
            text = text.strip()
        if self.cfg.collapse_whitespace:
            text = re.sub(r"\s+", " ", text)

        return text

    def tokenise_for_wer(self, text: str) -> str:
        """
        Normalise then insert spaces around CJK characters so each
        character becomes a separate "word" for WER computation.
        """
        text = self.apply(text)
        return self._insert_cjk_spaces(text)

    def _insert_cjk_spaces(self, text: str) -> str:
        """Insert spaces around CJK characters for word-based WER."""
        text = _CJK_RE.sub(r" \1 ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def has_cjk(text: str) -> bool:
        return bool(_CJK_RE.search(text))
