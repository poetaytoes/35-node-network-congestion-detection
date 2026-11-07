from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from congestion_lab.data.artifacts import ArtifactStore, verify_manifest
from congestion_lab.features import FEATURE_COLUMNS
from congestion_lab.models import fit_binary_models
from congestion_lab.models.training import load_bundle, save_bundle


def _model_frame(rows_per_split: int = 48) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    rows: list[dict[str, object]] = []
    for split in ("train", "validation", "locked_test"):
        for index in range(rows_per_split):
            signal = float(rng.normal() + (index % 3))
            row: dict[str, object] = {
                feature: signal + float(rng.normal(scale=0.1)) for feature in FEATURE_COLUMNS
            }
            row.update(
                {
                    "episode_id": f"{split}-{index // 12}",
                    "topology": "centralized" if index % 2 == 0 else "distributed",
                    "object_id": f"object-{index % 2}",
                    "window_index": index,
                    "split": split,
                    "future_congestion_h2": signal > 1.0,
                    "target_valid_h2": True,
                    "congestion_state": signal > 1.5,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_model_save_load_score_equality_and_schema_rejection(tmp_path: Path) -> None:
    frame = _model_frame()
    bundles, _ = fit_binary_models(frame, compact_grid=True)
    bundle = bundles["xgboost"]
    validation = frame.loc[frame["split"] == "validation"]
    expected = bundle.predict_score(validation)
    path = save_bundle(bundle, tmp_path / "xgboost.joblib")
    loaded = load_bundle(path)
    np.testing.assert_allclose(expected, loaded.predict_score(validation), rtol=0, atol=0)
    with pytest.raises(ValueError, match="schema mismatch"):
        loaded.predict_score(validation.drop(columns=[FEATURE_COLUMNS[0]]))


def test_artifact_manifest_hash_chain_and_tamper_detection(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text('{"name":"fixture"}', encoding="utf-8")
    store = ArtifactStore(tmp_path / "runs", "contract")
    input_path = store.write_text("data/input.txt", "alpha\n")
    output_path = store.write_json("data/output.json", {"value": 1})
    stage = store.write_stage_manifest(
        "fixture",
        inputs=[input_path],
        outputs=[output_path],
        config_path=config,
        statistics={"rows": 1},
    )
    store.finalize(config_hash="0" * 64)
    assert verify_manifest(store.run_dir / "run_manifest.json") == []
    payload = json.loads(stage.read_text(encoding="utf-8"))
    assert payload["outputs"][0]["sha256"]
    output_path.write_text('{"value":2}', encoding="utf-8")
    assert verify_manifest(store.run_dir / "run_manifest.json") == ["hash:data/output.json"]


def test_immutable_store_refuses_overwrite(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, "immutable")
    store.write_text("item.txt", "first")
    with pytest.raises(FileExistsError):
        store.write_text("item.txt", "second")
