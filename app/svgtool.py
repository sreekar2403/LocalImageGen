"""SVG extraction, sanitization and render-validation. No LLM in here.

Keeping this pure makes the security-critical part directly testable without a
model in the loop.
"""

from __future__ import annotations

import re
from typing import Any

from defusedxml import ElementTree as DefusedET

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"

# Elements that can execute, fetch, or embed foreign content.
BANNED_TAGS = {
    "script", "foreignobject", "iframe", "embed", "object",
    "audio", "video", "handler", "set", "animate", "animatetransform",
    "animatemotion", "use",
}

# href values we allow: internal refs and inline data-image payloads only.
_SAFE_HREF = re.compile(r"^(#|data:image/)", re.IGNORECASE)
_URL_EXTERNAL = re.compile(r"url\(\s*['\"]?\s*(https?:|//)", re.IGNORECASE)
_IMPORT = re.compile(r"@import", re.IGNORECASE)
_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")


class SvgError(ValueError):
    """SVG could not be extracted, parsed, sanitized or rendered."""


def extract(raw: str) -> str:
    """Pull the SVG document out of an LLM response."""
    if not raw or not raw.strip():
        raise SvgError("empty response")
    text = _FENCE.sub("", raw.strip())
    start = text.find("<svg")
    end = text.rfind("</svg>")
    if start == -1 or end == -1 or end < start:
        raise SvgError("no <svg>...</svg> element found in response")
    return text[start : end + len("</svg>")]


def _localname(tag: Any) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1].lower()


def sanitize(svg: str, width: int | None = None, height: int | None = None) -> tuple[str, list[str]]:
    """Parse, strip anything active or external, and re-serialize.

    Returns (clean_svg, notes). Re-serializing from the parsed tree is itself a
    defence: anything that survived as raw text cannot come back out.
    """
    notes: list[str] = []
    try:
        root = DefusedET.fromstring(svg.encode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SvgError(f"XML parse failed: {type(exc).__name__}: {exc}") from exc

    if _localname(root.tag) != "svg":
        raise SvgError(f"root element is <{_localname(root.tag)}>, expected <svg>")

    # Drop banned elements (walk parents so we can remove children).
    for parent in list(root.iter()):
        for child in list(parent):
            name = _localname(child.tag)
            if name in BANNED_TAGS:
                parent.remove(child)
                notes.append(f"removed <{name}>")

    removed_attrs = 0
    for el in root.iter():
        for attr in list(el.attrib):
            local = attr.rsplit("}", 1)[-1].lower()
            value = el.attrib[attr]

            if local.startswith("on"):  # onload, onclick, ...
                del el.attrib[attr]
                removed_attrs += 1
                continue
            if local == "href" and not _SAFE_HREF.match(value.strip()):
                del el.attrib[attr]
                removed_attrs += 1
                continue
            if isinstance(value, str) and _URL_EXTERNAL.search(value):
                del el.attrib[attr]
                removed_attrs += 1
                continue

        if _localname(el.tag) == "style" and el.text and (_IMPORT.search(el.text) or _URL_EXTERNAL.search(el.text)):
            el.text = ""
            notes.append("stripped external refs from <style>")

    if removed_attrs:
        notes.append(f"removed {removed_attrs} unsafe attribute(s)")

    # Geometry: a viewBox is what makes the result scale properly.
    if "viewBox" not in root.attrib:
        w = root.attrib.get("width", str(width or 512))
        h = root.attrib.get("height", str(height or 512))
        digits = re.compile(r"[\d.]+")
        wm, hm = digits.search(w), digits.search(h)
        root.set("viewBox", f"0 0 {wm.group() if wm else 512} {hm.group() if hm else 512}")
        notes.append("synthesized missing viewBox")

    # Normalize every tag to its local name and drop namespaced attributes.
    # Without this, ElementTree serializes as <ns0:svg ns0:...>, which renders
    # but is unpleasant to hand-edit -- and hand-editability is the point.
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.rsplit("}", 1)[-1]
        for attr in list(el.attrib):
            if "}" in attr:
                local = attr.rsplit("}", 1)[-1]
                value = el.attrib.pop(attr)
                if local not in el.attrib and not local.startswith("xmlns"):
                    el.attrib[local] = value

    root.set("xmlns", SVG_NS)
    if width:
        root.set("width", str(width))
    if height:
        root.set("height", str(height))

    out = DefusedET.tostring(root, encoding="unicode")
    if not out.lstrip().startswith("<?xml"):
        out = '<?xml version="1.0" encoding="UTF-8"?>\n' + out
    return out, notes


def rasterize(svg: str) -> bytes:
    """Render to PNG. This is the real validation gate.

    resvg rejects structurally invalid or unsupported SVG, so a successful
    render is meaningful evidence the file is good -- far stronger than an XML
    parse alone.
    """
    import resvg_py

    try:
        png = resvg_py.svg_to_bytes(svg_string=svg)
    except Exception as exc:  # noqa: BLE001
        raise SvgError(f"resvg render failed: {type(exc).__name__}: {exc}") from exc
    if isinstance(png, list):  # resvg_py returns a list of ints
        png = bytes(png)
    if not png:
        raise SvgError("resvg produced an empty image")
    return png


def path_count(svg: str) -> int:
    return len(re.findall(r"<path\b", svg, re.IGNORECASE))


def validate(svg: str, width: int | None = None, height: int | None = None) -> tuple[str, bytes, list[str]]:
    """extract-free full gate: sanitize then render. Raises SvgError."""
    clean, notes = sanitize(svg, width, height)
    png = rasterize(clean)
    return clean, png, notes
