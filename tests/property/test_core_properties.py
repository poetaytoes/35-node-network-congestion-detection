from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from congestion_lab.config.models import ExperimentConfig, TopologyConfig
from congestion_lab.data.artifacts import stable_frame_hash
from congestion_lab.policies import NoActionPolicy
from congestion_lab.simulation import simulate_episode
from congestion_lab.simulation.engine import QueueState
from congestion_lab.traffic import generate_workloads


@given(
    queue=st.floats(min_value=0, max_value=1_000_000, allow_nan=False),
    arrivals=st.floats(min_value=0, max_value=1_000_000, allow_nan=False),
    service=st.floats(min_value=0, max_value=1_000_000, allow_nan=False),
    buffer=st.floats(min_value=1, max_value=1_000_000, allow_nan=False),
)
@settings(max_examples=100, deadline=None)
def test_queue_conservation_and_bounds(
    queue: float, arrivals: float, service: float, buffer: float
) -> None:
    state = QueueState(queue_bytes=min(queue, buffer))
    before = state.queue_bytes
    transmitted, dropped = state.step(arrivals, service, buffer)
    assert 0 <= state.queue_bytes <= buffer
    assert transmitted >= 0 and dropped >= 0
    assert np.isclose(before + arrivals, transmitted + dropped + state.queue_bytes)


def test_same_seed_hash_and_changed_seed_difference(
    pilot_config: ExperimentConfig, minimal_topology: TopologyConfig
) -> None:
    first, _ = generate_workloads(pilot_config, minimal_topology)
    second, _ = generate_workloads(pilot_config, minimal_topology)
    changed = ExperimentConfig.model_validate(
        {**pilot_config.model_dump(), "root_seed": pilot_config.root_seed + 1}
    )
    third, _ = generate_workloads(changed, minimal_topology)
    assert stable_frame_hash(first) == stable_frame_hash(second)
    assert stable_frame_hash(first) != stable_frame_hash(third)


def test_no_action_policy_matches_no_policy(
    pilot_config: ExperimentConfig, minimal_topology: TopologyConfig
) -> None:
    events, _ = generate_workloads(pilot_config, minimal_topology)
    episode = events.loc[events["episode_id"] == "P01"].reset_index(drop=True)
    plain = simulate_episode(episode, pilot_config, minimal_topology, "centralized")
    explicit = simulate_episode(
        episode,
        pilot_config,
        minimal_topology,
        "centralized",
        policy=NoActionPolicy(),
    )
    pd.testing.assert_frame_equal(plain.telemetry, explicit.telemetry)
    assert plain.ledger == explicit.ledger
    assert explicit.actions.empty
