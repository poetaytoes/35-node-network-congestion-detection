from __future__ import annotations

import numpy as np
import pandas as pd

from congestion_lab.config.models import ExperimentConfig, TopologyConfig
from congestion_lab.data import validate_raw_events, validate_telemetry
from congestion_lab.features import FEATURE_COLUMNS, build_feature_table
from congestion_lab.simulation import simulate_episode
from congestion_lab.simulation.engine import QueueState
from congestion_lab.traffic import generate_workloads


def test_queue_hand_calculated_cases() -> None:
    state = QueueState()
    transmitted, dropped = state.step(arrivals=80, service=100, buffer=50)
    assert (transmitted, dropped, state.queue_bytes) == (80, 0, 0)
    transmitted, dropped = state.step(arrivals=140, service=100, buffer=50)
    assert (transmitted, dropped, state.queue_bytes) == (100, 0, 40)
    transmitted, dropped = state.step(arrivals=140, service=100, buffer=50)
    assert (transmitted, dropped, state.queue_bytes) == (100, 30, 50)
    transmitted, dropped = state.step(arrivals=0, service=100, buffer=50)
    assert (transmitted, dropped, state.queue_bytes) == (50, 0, 0)


def test_paired_simulation_preserves_offered_workload(
    pilot_config: ExperimentConfig, minimal_topology: TopologyConfig
) -> None:
    events, _ = generate_workloads(pilot_config, minimal_topology)
    episode = events.loc[events["episode_id"] == "P02"].reset_index(drop=True)
    central = simulate_episode(episode, pilot_config, minimal_topology, "centralized")
    distributed = simulate_episode(episode, pilot_config, minimal_topology, "distributed")
    assert np.isclose(central.ledger["offered_bytes"], distributed.ledger["offered_bytes"])
    assert validate_raw_events(episode)["status"] == "pass"
    assert validate_telemetry(central.telemetry)["status"] == "pass"
    assert central.ledger["events"] == distributed.ledger["events"]
    assert central.telemetry.shape[0] == distributed.telemetry.shape[0] == 960
    assert {
        "propagation_delay_ms",
        "queue_delay_ms",
        "delay_ms",
        "compute_consumed_bytes",
        "e2e_mean_latency_ms",
        "e2e_loss_bytes",
        "e2e_accepted_requests",
    } <= set(central.telemetry)
    assert np.allclose(
        central.telemetry["delay_ms"],
        central.telemetry["propagation_delay_ms"]
        + central.telemetry["queue_delay_ms"]
        + central.telemetry["processing_delay_ms"],
    )
    assert distributed.ledger["backbone_bytes"] < central.ledger["backbone_bytes"]
    for result in (central, distributed):
        assert np.isclose(
            result.ledger["network_offered_bytes"],
            result.ledger["e2e_delivered_bytes"] + result.ledger["e2e_loss_bytes"],
        )
        assert result.ledger["bandwidth_hop_bytes"] > result.ledger["backbone_bytes"]
        assert result.ledger["compute_budget_fair"] is True
        assert result.ledger["active_compute_budget_mbps"] == 360.0


def test_attack_ground_truth_counterfactual_cannot_change_model_features(
    pilot_config: ExperimentConfig,
    minimal_topology: TopologyConfig,
) -> None:
    events, _ = generate_workloads(pilot_config, minimal_topology)
    episode = events.loc[events["episode_id"] == "P02"].reset_index(drop=True)
    counterfactual = episode.copy()
    counterfactual["is_attack_ground_truth"] = False
    counterfactual["attack_family"] = "none"
    observed = simulate_episode(episode, pilot_config, minimal_topology, "centralized").telemetry
    relabeled = simulate_episode(
        counterfactual, pilot_config, minimal_topology, "centralized"
    ).telemetry
    observed_features = build_feature_table(observed, pilot_config, {"P02": "test"})
    relabeled_features = build_feature_table(relabeled, pilot_config, {"P02": "test"})
    pd.testing.assert_frame_equal(
        observed_features.loc[:, FEATURE_COLUMNS],
        relabeled_features.loc[:, FEATURE_COLUMNS],
        check_exact=True,
    )
    assert not observed["attack_bytes"].equals(relabeled["attack_bytes"])
