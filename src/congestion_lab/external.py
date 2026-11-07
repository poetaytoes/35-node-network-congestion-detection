"""Reproducible privacy-safe external-profile replay experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from congestion_lab.config import load_experiment, load_topology
from congestion_lab.data import validate_raw_events, validate_telemetry
from congestion_lab.data.artifacts import ArtifactStore, sha256_file, stable_frame_hash
from congestion_lab.features import FEATURE_COLUMNS, build_feature_table, leakage_audit
from congestion_lab.models import load_bundle
from congestion_lab.simulation import simulate_episode
from congestion_lab.traffic import external_sample_to_events, verify_external_file


def _manifest_entry(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    normalized = path.resolve().as_posix().lower()
    for entry in payload.get("sources", []) + payload.get("derived", []):
        candidate = Path(str(entry.get("local_file", entry.get("path", "")))).resolve()
        if candidate.as_posix().lower() == normalized:
            return dict(entry)
    raise ValueError(f"external manifest does not pin file: {path}")


def _assert_privacy_contract(events: pd.DataFrame) -> dict[str, Any]:
    identity_tokens = (
        "raw_ip",
        "src_ip",
        "dst_ip",
        "address",
        "subscriber",
        "src_port",
        "dst_port",
    )
    identity_columns = sorted(
        column
        for column in events.columns
        if any(token in column.lower() for token in identity_tokens)
    )
    feature_metadata_overlap = sorted(
        set(FEATURE_COLUMNS)
        & {
            "traffic_source",
            "episode_id",
            "event_id",
            "source_group_id",
            "scenario_family",
            "attack_family",
            "is_attack_ground_truth",
        }
    )
    report = {
        "identity_columns_in_events": identity_columns,
        "identity_or_source_metadata_in_model_features": feature_metadata_overlap,
        "training_use": False,
        "status": "pass" if not identity_columns and not feature_metadata_overlap else "fail",
    }
    if report["status"] != "pass":
        raise AssertionError(f"external privacy contract failed: {report}")
    return report


def run_external_profile_replay(
    experiment_path: str | Path,
    *,
    manifest_path: str | Path,
    profile_path: str | Path,
    sample_path: str | Path,
    run_id: str,
    model_path: str | Path | None = None,
) -> Path:
    """Replay a seeded profile and evaluate an optional frozen core model."""
    if not run_id.startswith("external_"):
        raise ValueError("external replay run_id must start with 'external_'")
    experiment_file = Path(experiment_path).resolve()
    source_manifest_file = Path(manifest_path).resolve()
    profile_file = Path(profile_path).resolve()
    sample_file = Path(sample_path).resolve()
    model_file = Path(model_path).resolve() if model_path is not None else None
    source_manifest = json.loads(source_manifest_file.read_text(encoding="utf-8"))
    profile_entry = _manifest_entry(source_manifest, profile_file)
    sample_entry = _manifest_entry(source_manifest, sample_file)
    verifications = {
        "profile": verify_external_file(profile_file, expected_sha256=str(profile_entry["sha256"])),
        "sample": verify_external_file(sample_file, expected_sha256=str(sample_entry["sha256"])),
    }
    if any(result["status"] != "pass" for result in verifications.values()):
        raise ValueError(f"external checksum verification failed: {verifications}")

    config = load_experiment(experiment_file)
    topology = load_topology(config.topology_path)
    regions = sorted(node.id for node in topology.nodes if node.role == "region")
    events = external_sample_to_events(
        sample_file,
        episode_id="EX01",
        duration_s=config.episode_duration_s,
        regions=regions,
        seed=config.root_seed,
    )
    raw_schema = validate_raw_events(events)
    privacy_report = _assert_privacy_contract(events)

    telemetry_parts: list[pd.DataFrame] = []
    ledger_rows: list[dict[str, object]] = []
    for mode in ("centralized", "distributed"):
        result = simulate_episode(events, config, topology, mode)
        telemetry_parts.append(result.telemetry)
        ledger_rows.append(dict(result.ledger))
    telemetry = pd.concat(telemetry_parts, ignore_index=True)
    telemetry_schema = validate_telemetry(telemetry)
    ledgers = pd.DataFrame(ledger_rows)
    features = build_feature_table(telemetry, config, {"EX01": "external_profile"})
    feature_audit = leakage_audit(features)
    if feature_audit["status"] != "pass":
        raise AssertionError(f"external feature leakage audit failed: {feature_audit}")
    offered = ledgers.set_index("topology")["offered_bytes"]
    paired_equal = bool(
        np.isclose(float(offered["centralized"]), float(offered["distributed"]), rtol=0, atol=1e-6)
    )
    if not paired_equal:
        raise AssertionError("external replay offered workload differs between C/D")

    external_predictions = pd.DataFrame()
    external_metrics = pd.DataFrame()
    model_sha256: str | None = None
    if model_file is not None:
        bundle = load_bundle(model_file)
        eligible = features.loc[features["model_eligible"]].copy()
        scores = bundle.predict_score(eligible)
        eligible["score"] = scores
        eligible["prediction"] = (scores >= bundle.threshold).astype(int)
        external_predictions = eligible[
            [
                "episode_id",
                "topology",
                "object_id",
                "object_role",
                "window_index",
                "score",
                "prediction",
            ]
        ].copy()
        metric_rows: list[dict[str, object]] = []
        for topology_name, group in external_predictions.groupby("topology", sort=True):
            alarm_events = 0
            for _, object_group in group.groupby("object_id", sort=True):
                predicted = object_group.sort_values("window_index")["prediction"].to_numpy(
                    dtype=bool
                )
                previous = np.concatenate(([False], predicted[:-1]))
                alarm_events += int((predicted & ~previous).sum())
            metric_rows.append(
                {
                    "model": bundle.name,
                    "topology": str(topology_name),
                    "rows": len(group),
                    "frozen_threshold": bundle.threshold,
                    "mean_score": float(group["score"].mean()),
                    "p95_score": float(group["score"].quantile(0.95)),
                    "max_score": float(group["score"].max()),
                    "predicted_positive_windows": int(group["prediction"].sum()),
                    "predicted_positive_rate": float(group["prediction"].mean()),
                    "alarm_events": alarm_events,
                    "truth_scope": "unlabeled benign aggregate profile",
                }
            )
        external_metrics = pd.DataFrame(metric_rows)
        model_sha256 = sha256_file(model_file)

    store = ArtifactStore(config.output_root, run_id)
    experiment_snapshot = store.write_json("config/experiment.json", config.model_dump(mode="json"))
    topology_snapshot = store.write_json("config/topology.json", topology.model_dump(mode="json"))
    source_snapshot = store.write_json("external/source_manifest.json", source_manifest)
    profile_snapshot = store.write_json(
        "external/profile.json", json.loads(profile_file.read_text(encoding="utf-8"))
    )
    checks_path = store.write_json("external/checksum_verification.json", verifications)
    events_path = store.write_frame("external/raw_events.parquet", events)
    telemetry_path = store.write_frame("external/telemetry.parquet", telemetry)
    features_path = store.write_frame("external/features.parquet", features)
    ledgers_path = store.write_frame("external/ledgers.csv", ledgers)
    model_output_paths: list[Path] = []
    if model_file is not None:
        model_output_paths.extend(
            [
                store.write_frame(
                    "external/frozen_model_predictions.parquet", external_predictions
                ),
                store.write_frame("external/frozen_model_metrics.csv", external_metrics),
            ]
        )
    contract = {
        "namespace": "external_profile_only",
        "episode_id": "EX01",
        "source_manifest_sha256": sha256_file(source_manifest_file),
        "profile_sha256": sha256_file(profile_file),
        "sample_sha256": sha256_file(sample_file),
        "event_rows": len(events),
        "event_frame_sha256": stable_frame_hash(events),
        "telemetry_rows": len(telemetry),
        "feature_rows": len(features),
        "topologies": sorted(telemetry["topology"].unique().tolist()),
        "paired_offered_bytes_equal": paired_equal,
        "privacy": privacy_report,
        "feature_leakage": feature_audit,
        "raw_event_schema": raw_schema,
        "telemetry_schema": telemetry_schema,
        "core_training_use": False,
        "frozen_model_evaluated": model_file is not None,
        "frozen_model_sha256": model_sha256,
        "external_truth_scope": "unlabeled benign aggregate profile",
        "external_model_metric_rows": len(external_metrics),
        "status": "pass",
    }
    contract_path = store.write_json("verification/external_replay_contract.json", contract)
    store.write_stage_manifest(
        "m11_external_profile_replay",
        inputs=[
            experiment_file,
            Path(config.topology_path),
            source_manifest_file,
            profile_file,
            sample_file,
            *([model_file] if model_file is not None else []),
        ],
        outputs=[
            experiment_snapshot,
            topology_snapshot,
            source_snapshot,
            profile_snapshot,
            checks_path,
            events_path,
            telemetry_path,
            features_path,
            ledgers_path,
            *model_output_paths,
            contract_path,
        ],
        config_path=experiment_file,
        statistics=contract,
    )
    store.finalize(
        config_hash=sha256_file(experiment_file),
        parents={
            "external_source_manifest": sha256_file(source_manifest_file),
            "external_profile": sha256_file(profile_file),
            "external_sample": sha256_file(sample_file),
            **({"frozen_core_model": model_sha256} if model_sha256 is not None else {}),
        },
        random_seeds={"root_seed": config.root_seed},
    )
    return store.run_dir
