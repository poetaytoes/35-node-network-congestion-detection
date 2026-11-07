from __future__ import annotations

from pathlib import Path

import pytest

from congestion_lab.data import sha256_file


def test_authoritative_sources_are_unchanged(project_root: Path) -> None:
    expected = {
        (
            "Machine Learning-Driven Congestion Prediction for Resource-Efficient "
            "Centralized and Distributed 5G Data Center Models.md"
        ): "8712f436ebc157feebfc2ce4519c25c75435062eecab928a82b70ec4e2eb1750",
        (
            "Goi y ket qua Machine Learning-Driven Congestion Prediction for "
            "Resource-Efficient Centralized and Distributed 5G Data Center Models.md"
        ): "f8da6a28a0b622b4a8c93f495c263d11bb6e4fcddb8b924ef7c387cceb0a220a",
        "Mo hinh.jpg": "59a4eb2cdd02008efe88503cdb4c8e786c26be4cf40db1f79d9ee7e485610b0f",
    }
    missing = [name for name in expected if not (project_root / "docs" / name).is_file()]
    if missing:
        pytest.skip("Private authority documents are not included in the source-only release.")

    for name, digest in expected.items():
        assert sha256_file(project_root / "docs" / name) == digest


def test_result_code_does_not_embed_demo_table_values(project_root: Path) -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in (project_root / "src").rglob("*.py")
    )
    assert "0.925" not in source
    assert "0.885" not in source
    assert "DEMO values only" not in source
