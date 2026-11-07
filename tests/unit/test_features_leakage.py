from __future__ import annotations

import pandas as pd

from congestion_lab.config.models import ExperimentConfig, TopologyConfig
from congestion_lab.features import FEATURE_COLUMNS, build_feature_table, leakage_audit
from congestion_lab.features.builder import feature_dictionary
from congestion_lab.simulation import simulate_episode
from congestion_lab.traffic import generate_workloads


def _pilot_telemetry(
    pilot_config: ExperimentConfig, minimal_topology: TopologyConfig
) -> tuple[pd.DataFrame, dict[str, str]]:
    events, manifest = generate_workloads(pilot_config, minimal_topology)
    parts: list[pd.DataFrame] = []
    for _, episode in events.groupby("episode_id", sort=True):
        for mode in ("centralized", "distributed"):
            parts.append(
                simulate_episode(
                    episode.reset_index(drop=True), pilot_config, minimal_topology, mode
                ).telemetry
            )
    split_map = dict(zip(manifest["episode_id"], manifest["split"], strict=True))
    return pd.concat(parts, ignore_index=True), split_map


def _pilot_features(
    pilot_config: ExperimentConfig, minimal_topology: TopologyConfig
) -> pd.DataFrame:
    telemetry, split_map = _pilot_telemetry(pilot_config, minimal_topology)
    return build_feature_table(telemetry, pilot_config, split_map)


def test_window_count_and_group_boundaries(
    pilot_config: ExperimentConfig, minimal_topology: TopologyConfig
) -> None:
    features = _pilot_features(pilot_config, minimal_topology)
    assert len(features) == 2 * 2 * 24 * 8
    first_rows = features.groupby(["episode_id", "topology", "object_id"]).head(1)
    assert first_rows["util_lag1"].isna().all()
    last_rows = features.groupby(["episode_id", "topology", "object_id"]).tail(2)
    assert last_rows["target_valid_h2"].eq(False).all()
    assert set(features["cause_label"]) >= {
        "normal",
        "attack_without_congestion",
        "attack_associated_congestion",
    }


def test_feature_allowlist_blocks_metadata(
    pilot_config: ExperimentConfig, minimal_topology: TopologyConfig
) -> None:
    features = _pilot_features(pilot_config, minimal_topology)
    report = leakage_audit(features)
    assert report["status"] == "pass"
    assert len(FEATURE_COLUMNS) == 40
    assert "topology" not in FEATURE_COLUMNS
    assert "episode_id" not in FEATURE_COLUMNS
    assert "attack_bytes_window" not in FEATURE_COLUMNS
    dictionary = feature_dictionary()
    assert dictionary["ground_truth_dependency"].eq(False).all()
    assert dictionary["source_scope"].eq("observable_current_or_past").all()


def test_future_telemetry_cannot_change_past_causal_derivatives(
    pilot_config: ExperimentConfig, minimal_topology: TopologyConfig
) -> None:
    telemetry, split_map = _pilot_telemetry(pilot_config, minimal_topology)
    original = build_feature_table(telemetry, pilot_config, split_map)
    perturbed = telemetry.copy()
    perturbed.loc[perturbed["timestamp_s"] >= 20, "utilization"] = 0.99
    changed = build_feature_table(perturbed, pilot_config, split_map)
    keys = ["episode_id", "topology", "object_id", "window_index"]
    before = original.loc[original["window_index"] <= 3, keys + list(FEATURE_COLUMNS)].sort_values(keys)
    after = changed.loc[changed["window_index"] <= 3, keys + list(FEATURE_COLUMNS)].sort_values(keys)
    pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))
