import pytest

from app import svgtool
from app.svgtool import SvgError

GOOD = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="40" fill="tomato"/></svg>'


def test_extract_from_markdown_fence():
    raw = "Here you go:\n```xml\n" + GOOD + "\n```\nHope that helps!"
    assert svgtool.extract(raw).startswith("<svg")
    assert svgtool.extract(raw).endswith("</svg>")


def test_extract_with_prose_around_it():
    assert svgtool.extract("blah " + GOOD + " trailing words").startswith("<svg")


@pytest.mark.parametrize("raw", ["", "   ", "no svg here at all", "<div>nope</div>"])
def test_extract_rejects_garbage(raw):
    with pytest.raises(SvgError):
        svgtool.extract(raw)


def test_script_is_removed():
    dirty = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><script>alert(1)</script><rect width="5" height="5"/></svg>'
    clean, notes = svgtool.sanitize(dirty)
    assert "script" not in clean.lower()
    assert any("script" in n for n in notes)


def test_event_handler_attribute_is_removed():
    dirty = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><rect width="5" height="5" onload="alert(1)" onclick="x()"/></svg>'
    clean, notes = svgtool.sanitize(dirty)
    assert "onload" not in clean.lower() and "onclick" not in clean.lower()
    assert any("unsafe attribute" in n for n in notes)


def test_external_href_is_removed_but_internal_kept():
    dirty = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<image href="https://evil.example/x.png" width="5" height="5"/>'
        '<use href="#frag"/></svg>'
    )
    clean, _ = svgtool.sanitize(dirty)
    assert "evil.example" not in clean


def test_foreignobject_is_removed():
    dirty = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><foreignObject><body/></foreignObject></svg>'
    clean, _ = svgtool.sanitize(dirty)
    assert "foreignobject" not in clean.lower()


def test_style_import_is_stripped():
    dirty = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><style>@import url(https://evil.example/a.css);</style></svg>'
    clean, _ = svgtool.sanitize(dirty)
    assert "evil.example" not in clean


def test_xxe_entity_is_refused():
    xxe = (
        '<?xml version="1.0"?><!DOCTYPE svg [<!ENTITY xxe SYSTEM "file:///c:/windows/win.ini">]>'
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><text>&xxe;</text></svg>'
    )
    with pytest.raises(SvgError):
        svgtool.sanitize(xxe)


def test_missing_viewbox_is_synthesized():
    clean, notes = svgtool.sanitize('<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect width="10" height="10"/></svg>')
    assert "viewBox" in clean
    assert any("viewBox" in n for n in notes)


def test_non_svg_root_rejected():
    with pytest.raises(SvgError):
        svgtool.sanitize("<html><body/></html>")


def test_malformed_xml_rejected():
    with pytest.raises(SvgError):
        svgtool.sanitize('<svg xmlns="http://www.w3.org/2000/svg"><rect></svg>')


def test_rasterize_produces_png():
    clean, _ = svgtool.sanitize(GOOD)
    png = svgtool.rasterize(clean)
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "resvg did not return a PNG"


def test_validate_end_to_end():
    clean, png, _ = svgtool.validate(GOOD, 256, 256)
    assert 'width="256"' in clean
    assert png[:4] == b"\x89PNG"


def test_path_count():
    assert svgtool.path_count('<svg><path d="M0 0"/><path d="M1 1"/></svg>') == 2
