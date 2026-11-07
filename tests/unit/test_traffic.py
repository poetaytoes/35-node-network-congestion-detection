from __future__ import annotations

import pandas as pd

from congestion_lab.config.models import ExperimentConfig, TopologyConfig
from congestion_lab.traffic import generate_workloads


def test_pilot_workload_is_deterministic(
    pilot_config: ExperimentConfig, minimal_topology: TopologyConfig
) -> None:
    first, first_manifest = generate_workloads(pilot_config, minimal_topology)
    second, second_manifest = generate_workloads(pilot_config, minimal_topology)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_manifest, second_manifest)
    assert first["episode_id"].nunique() == 2
    assert first["is_attack_ground_truth"].any()
    assert (~first["is_attack_ground_truth"]).any()
    assert (first["traffic_source"] == "synthetic").all()
    assert first["event_id"].is_unique
    assert first["arrival_time_s"].between(0, pilot_config.episode_duration_s - 1).all()
    benign = first.loc[~first["is_attack_ground_truth"]]
    assert (
        benign["duration_s"]
        .between(pilot_config.traffic.duration_min_s, pilot_config.traffic.duration_max_s)
        .all()
    )
    assert set(first["protocol_family"]) <= {"tcp", "udp", "http", "syn", "scan"}
    assert first["seed_path"].str.startswith(f"{pilot_config.root_seed}/").all()


def test_core_episode_matrix_and_split(
    core_config: ExperimentConfig, minimal_topology: TopologyConfig
) -> None:
    _, manifest = generate_workloads(core_config, minimal_topology)
    assert manifest["family"].value_counts().to_dict() == {
        "S3_udp_syn_flood": 4,
        "S0_stable": 3,
        "S1_gradual": 3,
        "S2_flash_crowd": 3,
        "S4_http_slow_rate": 3,
        "S5_hard_negative": 2,
    }
    assert manifest["split"].value_counts().to_dict() == {
        "train": 10,
        "validation": 4,
        "locked_test": 4,
    }
    assert manifest["workload_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    durations = manifest["attack_end_s"] - manifest["attack_start_s"]
    assert durations.between(30, 120).all()
