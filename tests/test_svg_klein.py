"""SVG klein-era upgrades. CPU only (no LLM, no vtracer, no GPU)."""

from app.backends.svg import SvgAuthorBackend, SvgTraceBackend
from app.prompts import SVG_PROMPTS


def test_author_and_trace_share_svg_kind():
    assert "svg" in SvgAuthorBackend.kinds
    assert "svg" in SvgTraceBackend(None).kinds


def test_trace_adaptive_palette():
    t = SvgTraceBackend(None)
    assert t.KIND_COLORS["icon"] <= 8
    assert t.KIND_COLORS["illustration"] >= 12


def test_prompts_have_fewshot_budgets():
    assert "Budget" not in SVG_PROMPTS["icon"]  # budget lives in user message, not system
    assert "viewBox" in SVG_PROMPTS["icon"]
    assert "hexagonal" in SVG_PROMPTS["logo"] or "64" in SVG_PROMPTS["logo"]
    assert "Login" in SVG_PROMPTS["diagram"]


def test_coordinate_rounding_regex():
    import re

    raw = "M12.34567 L13.88888"
    out = re.sub(r"(M-?\d+\.\d{3})\d+", lambda m: m.group(1), raw)
    assert out == "M12.345 L13.88888"
