from __future__ import annotations

import pandas as pd
import pytest

from congestion_lab.features import FEATURE_COLUMNS
from congestion_lab.models.training import ModelBundle
from congestion_lab.policies import (
    NoActionPolicy,
    OnlineShapingPolicy,
    PolicyConfig,
    PolicyDecision,
)


def test_no_action_is_exact_noop() -> None:
    policy = NoActionPolicy()
    for second in range(20):
        assert policy.shaping_fraction(second, "R1") == 0.0
        policy.record_shaped(second, "R1", 100.0)
    policy.on_window(20, pd.DataFrame())
    assert policy.action_frame().empty


def _threshold_bundle() -> ModelBundle:
    return ModelBundle(
        name="threshold",
        task="binary",
        estimator=None,
        feature_columns=FEATURE_COLUMNS,
        threshold=0.5,
        parameters={},
    )


def test_policy_cap_expiry_cooldown_and_precomputed_path_only(
    pilot_config, minimal_topology
) -> None:
    config = PolicyConfig(
        name="P1",
        kind="low_priority_shaping",
        threshold=0.5,
        shape_fraction=0.25,
        duration_windows=2,
        cooldown_windows=1,
        tau_low=0.25,
    )
    policy = OnlineShapingPolicy(
        _threshold_bundle(),
        config,
        pilot_config,
        minimal_topology,
        episode_id="P01",
        topology_mode="centralized",
    )
    decision = PolicyDecision("B1--B2", 0.9, ("R1",))
    policy.apply(10, decision)
    assert policy.tau_high == _threshold_bundle().threshold
    assert policy.tau_low == 0.25
    assert policy.action_frame().loc[0, ["tau_high", "tau_low"]].tolist() == [0.5, 0.25]
    assert policy.shaping_fraction(10, "R1") == 0.25
    assert policy.shaping_fraction(20, "R1") == 0.0
    policy.apply(20, decision)
    assert len(policy.action_frame()) == 1
    policy.apply(25, decision)
    assert len(policy.action_frame()) == 2

    p2 = OnlineShapingPolicy(
        _threshold_bundle(),
        PolicyConfig("P2", "precomputed_alternate_path", 0.5, 0.0, 1, 0, 0.15),
        pilot_config,
        minimal_topology,
        episode_id="P01",
        topology_mode="centralized",
    )
    p2.apply(10, decision)
    assert p2.shaping_fraction(10, "R1") == 0.0
    assert p2.central_reroute_fraction(10, "R1") == 0.15
    assert p2.action_frame()["route_changes"].tolist() == [1]


def test_invalid_policy_config_rejected() -> None:
    with pytest.raises(ValueError, match="within"):
        PolicyConfig("bad", "low_priority_shaping", 1.1, 0.2, 1, 0)
