"""Klein prompt normalizer. CPU only."""

from app.prompts import normalize_klein_prompt


def test_strips_quality_tags():
    clean, warns = normalize_klein_prompt("a lighthouse, masterpiece, 8k, ultra detailed")
    assert "masterpiece" not in clean and "8k" not in clean
    assert any("quality tag" in w for w in warns)


def test_folds_negative_to_positive():
    clean, warns = normalize_klein_prompt("a portrait, no blur, no extra fingers")
    assert "sharp, crisp focus" in clean
    assert "five fingers" in clean
    assert any("positive" in w for w in warns)


def test_truncates_long_prompt():
    clean, warns = normalize_klein_prompt("word " * 300)
    assert len(clean.split()) <= 140
    assert any("truncated" in w for w in warns)


def test_short_prompt_unchanged_except_spacing():
    clean, warns = normalize_klein_prompt("a red fox sticker")
    assert clean == "a red fox sticker"
    assert warns == []


def test_empty_prompt():
    clean, warns = normalize_klein_prompt("")
    assert clean == "" and warns == []
