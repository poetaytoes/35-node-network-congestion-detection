from __future__ import annotations

import pandas as pd

from congestion_lab.config.models import ExperimentConfig, TopologyConfig
from congestion_lab.features import build_feature_table
from congestion_lab.simulation import simulate_episode
from congestion_lab.traffic import generate_workloads


def test_pilot_raw_to_feature_pipeline(
    pilot_config: ExperimentConfig, minimal_topology: TopologyConfig
) -> None:
    events, manifest = generate_workloads(pilot_config, minimal_topology)
    telemetry: list[pd.DataFrame] = []
    for _, episode in events.groupby("episode_id", sort=True):
        for mode in ("centralized", "distributed"):
            telemetry.append(
                simulate_episode(
                    episode.reset_index(drop=True), pilot_config, minimal_topology, mode
                ).telemetry
            )
    split_map = dict(zip(manifest["episode_id"], manifest["split"], strict=True))
    features = build_feature_table(pd.concat(telemetry, ignore_index=True), pilot_config, split_map)
    stable = features.loc[features["episode_id"] == "P01"]
    attack = features.loc[features["episode_id"] == "P02"]
    assert not stable["congestion_state"].any()
    assert attack["congestion_state"].any()
    assert attack["attack_in_h2"].any()
