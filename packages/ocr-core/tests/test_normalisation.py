"""Tests for ocr_core.normalisation."""

from __future__ import annotations

from ocr_core.config import NormalisationConfig
from ocr_core.normalisation import NormalisationPipeline


class TestBasicNormalisation:
    def test_empty_string(self, default_normaliser):
        assert default_normaliser.apply("") == ""

    def test_strip_whitespace(self):
        n = NormalisationPipeline(NormalisationConfig(strip_whitespace=True))
        assert n.apply("  hello  ") == "hello"

    def test_no_strip_whitespace(self):
        n = NormalisationPipeline(
            NormalisationConfig(
                strip_whitespace=False,
                collapse_whitespace=False,
                unicode_form="",
            )
        )
        assert n.apply("  hello  ") == "  hello  "

    def test_collapse_whitespace(self):
        n = NormalisationPipeline(NormalisationConfig(collapse_whitespace=True))
        assert n.apply("hello   world\t\nfoo") == "hello world foo"

    def test_collapse_whitespace_newlines(self):
        n = NormalisationPipeline(NormalisationConfig(collapse_whitespace=True))
        assert n.apply("a\n\n\nb") == "a b"

    def test_no_collapse_whitespace(self):
        n = NormalisationPipeline(
            NormalisationConfig(collapse_whitespace=False, strip_whitespace=False)
        )
        result = n.apply("hello   world")
        assert "   " in result

    def test_lowercase(self):
        n = NormalisationPipeline(NormalisationConfig(lowercase=True))
        assert n.apply("Hello WORLD") == "hello world"

    def test_no_lowercase(self):
        n = NormalisationPipeline(NormalisationConfig(lowercase=False))
        assert "WORLD" in n.apply("Hello WORLD")

    def test_remove_punctuation(self):
        n = NormalisationPipeline(NormalisationConfig(remove_punctuation=True))
        assert n.apply("Hello, world! How's it?") == "Hello world Hows it"

    def test_remove_punctuation_ellipsis(self):
        n = NormalisationPipeline(NormalisationConfig(remove_punctuation=True))
        assert n.apply("Test... done?") == "Test done"

    def test_no_remove_punctuation(self):
        n = NormalisationPipeline(NormalisationConfig(remove_punctuation=False))
        result = n.apply("Hello, world!")
        assert "," in result
        assert "!" in result

    def test_custom_replacements(self):
        n = NormalisationPipeline(
            NormalisationConfig(custom_replacements={"foo": "bar", "hello": "world"})
        )
        assert n.apply("foo says hello") == "bar says world"


class TestUnicodeNormalisation:
    def test_nfkc(self):
        n = NormalisationPipeline(NormalisationConfig(unicode_form="NFKC"))
        # ﬁ (U+FB01) → fi under NFKC
        assert n.apply("\ufb01") == "fi"

    def test_nfc(self):
        n = NormalisationPipeline(NormalisationConfig(unicode_form="NFC"))
        # Composed form: é (U+00E9) should remain as-is
        assert n.apply("\u00e9") == "\u00e9"

    def test_empty_unicode_form(self):
        n = NormalisationPipeline(NormalisationConfig(unicode_form=""))
        # Should not normalise — ligature stays
        assert "\ufb01" in n.apply("\ufb01")


class TestFullwidthToHalfwidth:
    def test_enabled(self):
        n = NormalisationPipeline(NormalisationConfig(fullwidth_to_halfwidth=True))
        # Ａ (U+FF21) → A
        assert n.apply("\uff21\uff22\uff23") == "ABC"

    def test_ideographic_space(self):
        n = NormalisationPipeline(NormalisationConfig(fullwidth_to_halfwidth=True))
        # Ideographic space (U+3000) → regular space
        assert n.apply("hello\u3000world") == "hello world"

    def test_disabled(self):
        n = NormalisationPipeline(
            NormalisationConfig(fullwidth_to_halfwidth=False, unicode_form="")
        )
        assert "\uff21" in n.apply("\uff21")


class TestCustomReplacements:
    def test_single_replacement(self):
        n = NormalisationPipeline(
            NormalisationConfig(custom_replacements={"foo": "bar"})
        )
        assert n.apply("foo baz foo") == "bar baz bar"

    def test_multiple_replacements(self):
        n = NormalisationPipeline(
            NormalisationConfig(custom_replacements={"a": "x", "b": "y"})
        )
        assert n.apply("ab") == "xy"

    def test_empty_replacements(self):
        n = NormalisationPipeline(NormalisationConfig(custom_replacements={}))
        assert n.apply("hello") == "hello"


class TestCJK:
    def test_has_cjk(self):
        assert NormalisationPipeline.has_cjk("中文")
        assert NormalisationPipeline.has_cjk("hello 世界")
        assert not NormalisationPipeline.has_cjk("hello world")

    def test_has_cjk_chinese(self):
        assert NormalisationPipeline.has_cjk("你好")

    def test_has_cjk_japanese(self):
        assert NormalisationPipeline.has_cjk("こんにちは")

    def test_has_cjk_mixed(self):
        assert NormalisationPipeline.has_cjk("Hello 世界")

    def test_has_cjk_english_only(self):
        assert not NormalisationPipeline.has_cjk("Hello World")

    def test_has_cjk_empty(self):
        assert not NormalisationPipeline.has_cjk("")

    def test_tokenise_for_wer_cjk(self, default_normaliser):
        result = default_normaliser.tokenise_for_wer("你好世界")
        # Each CJK character should become a separate word
        assert result == "你 好 世 界"

    def test_tokenise_for_wer_cjk_word_count(self):
        pipeline = NormalisationPipeline()
        result = pipeline.tokenise_for_wer("你好世界")
        words = result.split()
        assert len(words) == 4

    def test_tokenise_for_wer_mixed(self, default_normaliser):
        result = default_normaliser.tokenise_for_wer("Hello 世界 World")
        assert "Hello" in result
        assert "World" in result
        assert "世" in result
        assert "界" in result

    def test_tokenise_for_wer_mixed_word_check(self):
        pipeline = NormalisationPipeline()
        result = pipeline.tokenise_for_wer("Hello 世界 World")
        words = result.split()
        assert "世" in words
        assert "界" in words

    def test_tokenise_for_wer_latin_only(self, default_normaliser):
        result = default_normaliser.tokenise_for_wer("hello world")
        assert result == "hello world"

    def test_tokenise_for_wer_empty(self, default_normaliser):
        assert default_normaliser.tokenise_for_wer("") == ""


class TestCallable:
    def test_call_delegates_to_apply(self, default_normaliser):
        text = "  Hello   World  "
        assert default_normaliser(text) == default_normaliser.apply(text)


class TestCombinedSteps:
    def test_all_steps(self):
        n = NormalisationPipeline(
            NormalisationConfig(
                unicode_form="NFKC",
                lowercase=True,
                strip_whitespace=True,
                collapse_whitespace=True,
                remove_punctuation=True,
                fullwidth_to_halfwidth=True,
                custom_replacements={"@": "at"},
            )
        )
        result = n.apply("  Ｈello, World!  @home  ")
        assert result == "hello world athome"

    def test_fullwidth_to_halfwidth(self):
        n = NormalisationPipeline(NormalisationConfig(fullwidth_to_halfwidth=True))
        # Fullwidth A is \uff21
        assert n.apply("\uff21\uff22\uff23") == "ABC"
        # Ideographic space
        assert n.apply("A\u3000B") == "A B"

    def test_unicode_normalisation_nfc(self):
        n = NormalisationPipeline(NormalisationConfig(unicode_form="NFC"))
        # é can be composed or decomposed
        result = n.apply("café")
        assert "é" in result or "e" in result

    def test_empty_unicode_form(self):
        n = NormalisationPipeline(NormalisationConfig(unicode_form=""))
        # Should not raise
        result = n.apply("test")
        assert result == "test"

    def test_callable_interface(self):
        n = NormalisationPipeline()
        result = n("  hello  world  ")
        assert result == "hello world"


class TestTraditionalToSimplified:
    def test_t2s_without_opencc(self):
        """Test that t2s gracefully handles missing opencc."""
        n = NormalisationPipeline(NormalisationConfig(traditional_to_simplified=True))
        # Should not raise, even if opencc is not installed
        result = n.apply("測試")
        assert isinstance(result, str)
