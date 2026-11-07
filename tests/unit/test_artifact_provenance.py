from __future__ import annotations

from congestion_lab.data.artifacts import source_tree_sha256


def test_source_tree_hash_changes_with_executable_source(tmp_path) -> None:
    source = tmp_path / "src" / "package" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")

    first = source_tree_sha256(tmp_path)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    second = source_tree_sha256(tmp_path)

    assert first != second


def test_source_tree_hash_ignores_runtime_cache_files(tmp_path) -> None:
    source = tmp_path / "src" / "package" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    first = source_tree_sha256(tmp_path)

    cache = source.parent / "__pycache__" / "module.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"runtime cache")

    assert source_tree_sha256(tmp_path) == first
