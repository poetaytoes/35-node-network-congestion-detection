from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from congestion_lab.config import load_topology
from congestion_lab.config.models import ExperimentConfig, LinkConfig, NodeConfig, TopologyConfig
from congestion_lab.topology import NetworkTopology


def test_minimal_topology_matches_markdown_contract(minimal_topology: TopologyConfig) -> None:
    report = NetworkTopology(minimal_topology).validate_contract()
    assert report == {
        "name": "student_minimal",
        "nodes": 15,
        "physical_links": 14,
        "role_backbone": 3,
        "role_central_dc": 1,
        "role_edge_dc": 3,
        "role_metro": 4,
        "role_region": 4,
    }
    assert len(minimal_topology.monitored_links) + 2 == 8


def test_thirty_node_profile_is_separate_and_complete(project_root: Path) -> None:
    config = load_topology(project_root / "configs/topologies/vietnam_30node.yaml")
    report = NetworkTopology(config).validate_contract()
    assert report["nodes"] == 30
    assert report["physical_links"] == 42
    assert report["role_edge_dc"] == 6
    assert report["role_central_dc"] == 2
    assert config.central_dc == "30"
    assert {link.capacity_mbps for link in config.links if link.link_class == "metro_internal"} == {
        100_000.0
    }
    assert {link.capacity_mbps for link in config.links if link.link_class != "metro_internal"} == {
        400_000.0
    }
    roles = {node.id: node.role for node in config.nodes}
    for mode in ("centralized", "distributed"):
        for route in NetworkTopology(config).routes(mode).values():
            assert not {
                node for node in route.nodes[1:-1] if roles[node] in {"edge_dc", "central_dc"}
            }


def test_routes_differ_only_by_destination(minimal_topology: TopologyConfig) -> None:
    network = NetworkTopology(minimal_topology)
    centralized = network.routes("centralized")
    distributed = network.routes("distributed")
    assert set(centralized) == set(distributed) == {"R1", "R2", "R3", "R4"}
    assert {route.destination for route in centralized.values()} == {"C0"}
    assert {route.destination for route in distributed.values()} <= {"E1", "E2", "E3"}
    assert network.route_for_service("R1", "distributed", "svc_video").destination == "E1"
    assert network.route_for_service("R1", "distributed", "svc_not_replicated").destination == "C0"


def test_unknown_config_field_is_rejected(core_config: ExperimentConfig) -> None:
    payload = core_config.model_dump()
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"simulation_step_s": 2}, "one-second step"),
        ({"horizons": [1, 2, 2]}, "horizons must be unique"),
        ({"primary_horizon": 3}, "primary_horizon"),
        ({"episode_duration_s": 121}, "divisible"),
    ],
)
def test_invalid_experiment_ranges_are_rejected(
    pilot_config: ExperimentConfig, changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        ExperimentConfig.model_validate({**pilot_config.model_dump(), **changes})


def test_routing_rejects_unknown_mode_and_is_repeatable(
    minimal_topology: TopologyConfig,
) -> None:
    network = NetworkTopology(minimal_topology)
    first = network.route("R1", "centralized")
    second = network.route("R1", "centralized")
    assert first == second
    with pytest.raises(ValueError, match="unsupported topology mode"):
        network.route("R1", "hybrid")


def test_shortest_path_uses_lexicographic_full_path_tie_break() -> None:
    nodes = (
        NodeConfig(id="R", role="region", domain="x"),
        NodeConfig(id="A", role="metro", domain="x"),
        NodeConfig(id="B", role="metro", domain="x"),
        NodeConfig(id="E", role="edge_dc", domain="x"),
        NodeConfig(id="C", role="central_dc", domain="x"),
    )

    def link(source: str, target: str) -> LinkConfig:
        return LinkConfig(
            source=source,
            target=target,
            link_class="dc_access",
            capacity_mbps=10,
            distance_km=1,
            buffer_mb=1,
        )

    topology = TopologyConfig(
        name="diamond",
        description="equal-distance tie fixture",
        nodes=nodes,
        links=(link("R", "A"), link("R", "B"), link("A", "E"), link("B", "E"), link("E", "C")),
        region_edge_map={"R": "E"},
        central_dc="C",
        edge_dcs=("E",),
        monitored_links=("A--R",),
        service_replicas={"svc": ("E",)},
    )
    route = NetworkTopology(topology).route("R", "centralized")
    assert route.nodes == ("R", "A", "C")
    assert route.physical_nodes == ("R", "A", "E", "C")
