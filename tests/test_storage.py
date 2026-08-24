from pathlib import Path

import pytest

from app import storage
from app.config import snap_dimension, resolve_dimensions


def test_unique_name_does_not_collide():
    names = {storage.unique_name("image") for _ in range(500)}
    assert len(names) == 500, "timestamp-only names collided; uuid suffix missing"


def test_kind_extensions():
    assert storage.resolve_output_path(None, "image").suffix == ".png"
    assert storage.resolve_output_path(None, "svg").suffix == ".svg"
    assert storage.resolve_output_path(None, "video").suffix == ".mp4"


def test_explicit_file_path_is_used_as_is(tmp_path):
    target = tmp_path / "nested" / "poster.png"
    assert storage.resolve_output_path(str(target), "image") == target
    assert target.parent.is_dir()


def test_directory_path_gets_generated_name(tmp_path):
    out = storage.resolve_output_path(str(tmp_path), "image")
    assert out.parent == tmp_path and out.suffix == ".png"


@pytest.mark.parametrize(
    "bad",
    ["../x", r"..\x", "a/b", r"a\b", "../../Windows/win.ini", "..%2fetc", ".."],
)
def test_safe_join_blocks_traversal(tmp_path, bad):
    with pytest.raises(ValueError):
        storage.safe_join(tmp_path, bad)


def test_safe_join_allows_plain_filename(tmp_path):
    assert storage.safe_join(tmp_path, "image.png") == (tmp_path / "image.png").resolve()


@pytest.mark.parametrize(
    "raw,expected",
    [(1000, 992), (1024, 1024), (5000, 1024), (100, 256), (577, 576)],
)
def test_snap_dimension(raw, expected):
    assert snap_dimension(raw) == expected


def test_resolve_dimensions_reports_adjustment():
    warns: list[str] = []
    assert resolve_dimensions("default", 1000, 1000, warns) == (992, 992)
    assert warns and "multiple of 16" in warns[0]


def test_resolve_dimensions_preset_is_clean():
    warns: list[str] = []
    assert resolve_dimensions("youtube", None, None, warns) == (1024, 576)
    assert warns == []
