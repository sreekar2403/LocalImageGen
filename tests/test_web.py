"""Web UI contract. CPU only — parses the served index.html, no browser."""

from html.parser import HTMLParser
from pathlib import Path

INDEX = Path(__file__).parent.parent / "app" / "web" / "index.html"


class _Collector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.missing_alt_img = 0
        self._img_stack = 0

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if d.get("id"):
            self.ids.add(d["id"])
        if tag == "img" and "alt" not in d:
            self.missing_alt_img += 1


def _text():
    return INDEX.read_text(encoding="utf-8")


def _ids():
    c = _Collector()
    c.feed(_text())
    return c


# IDs the JS wires up — renaming one in HTML without the JS (or vice versa)
# silently breaks the page, so pin them here.
REQUIRED_IDS = {
    "messages", "prompt", "sendBtn", "settingsBtn", "settingsPanel",
    "settingsGrid", "attachBtn", "fileInput", "chips", "cmdPalette",
    "cmdPaletteList", "statusPill", "statusText", "toasts", "jobsBtn",
    "jobsBadge", "jobsDrawer", "jobsList", "jobsClose", "overlay",
    "themeBtn", "wordCount", "emptyState", "quickActions",
}


def test_index_exists_and_parses():
    c = _ids()
    assert len(c.ids) > 20


def test_js_hooks_present():
    missing = REQUIRED_IDS - _ids().ids
    assert not missing, f"HTML ids missing for JS hooks: {missing}"


def test_no_blocking_alert():
    assert "alert(" not in _text(), "use toast() instead of alert()"


def test_no_external_cdn_dependency():
    text = _text()
    assert "cdn.tailwindcss.com" not in text
    assert "unpkg.com" not in text
    assert "cdnjs.cloudflare.com" not in text


def test_video_presets_include_medium():
    assert "medium-480p" in _text()


def test_images_have_alt():
    assert _ids().missing_alt_img == 0


def test_toast_region_is_live():
    assert 'id="toasts"' in _text() and 'aria-live="polite"' in _text()
