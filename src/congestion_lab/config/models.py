"""Typed YAML configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base model that rejects unknown configuration fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class NodeConfig(StrictModel):
    id: str
    role: Literal["region", "metro", "backbone", "edge_dc", "central_dc"]
    domain: str
    compute_capacity_mbps: float = Field(default=0.0, ge=0.0)
    queue_capacity_mb: float = Field(default=0.0, ge=0.0)
    primary: bool = False


class LinkConfig(StrictModel):
    source: str
    target: str
    link_class: Literal[
        "access", "metro_internal", "backbone_access", "national_backbone", "dc_access"
    ]
    capacity_mbps: float = Field(gt=0.0)
    distance_km: float = Field(ge=0.0)
    buffer_mb: float = Field(gt=0.0)

    @property
    def id(self) -> str:
        return "--".join(sorted((self.source, self.target)))


class TopologyConfig(StrictModel):
    name: str
    description: str
    nodes: tuple[NodeConfig, ...]
    links: tuple[LinkConfig, ...]
    region_edge_map: dict[str, str]
    central_dc: str
    edge_dcs: tuple[str, ...]
    monitored_links: tuple[str, ...]
    service_replicas: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_references(self) -> TopologyConfig:
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("node IDs must be unique")
        known = set(node_ids)
        roles = {node.id: node.role for node in self.nodes}
        link_ids = [link.id for link in self.links]
        if len(link_ids) != len(set(link_ids)):
            raise ValueError("physical links must be unique")
        for link in self.links:
            if link.source not in known or link.target not in known:
                raise ValueError(f"unknown endpoint in link {link.id}")
        if self.central_dc not in known:
            raise ValueError("central_dc must reference a node")
        if roles.get(self.central_dc) != "central_dc":
            raise ValueError("central_dc must reference a central_dc role")
        if not set(self.edge_dcs).issubset(known):
            raise ValueError("edge_dcs contains an unknown node")
        if any(roles[edge] != "edge_dc" for edge in self.edge_dcs):
            raise ValueError("edge_dcs must reference edge_dc roles")
        if not set(self.region_edge_map).issubset(known):
            raise ValueError("region_edge_map contains an unknown region")
        if any(roles[region] != "region" for region in self.region_edge_map):
            raise ValueError("region_edge_map keys must reference region roles")
        if not set(self.region_edge_map.values()).issubset(set(self.edge_dcs)):
            raise ValueError("region_edge_map must target configured edge DCs")
        if not set(self.monitored_links).issubset(set(link_ids)):
            raise ValueError("monitored_links contains an unknown link")
        for service_id, replicas in self.service_replicas.items():
            if not service_id or not replicas:
                raise ValueError("service replica entries require an ID and at least one replica")
            if not set(replicas).issubset(known):
                raise ValueError(f"service {service_id} contains an unknown replica")
            if any(roles[replica] not in {"edge_dc", "central_dc"} for replica in replicas):
                raise ValueError(f"service {service_id} replicas must reference DC roles")
        return self


class CongestionConfig(StrictModel):
    utilization_threshold: float = Field(default=0.8, gt=0.0, le=1.0)
    queue_threshold: float = Field(default=0.7, gt=0.0, le=1.0)
    loss_threshold: float = Field(default=0.01, ge=0.0, le=1.0)
    delay_sla_ms: float = Field(default=40.0, gt=0.0)
    persistence_windows: int = Field(default=2, ge=1)


class SplitConfig(StrictModel):
    train: tuple[str, ...]
    validation: tuple[str, ...]
    locked_test: tuple[str, ...]

    @model_validator(mode="after")
    def validate_disjoint(self) -> SplitConfig:
        sets = (set(self.train), set(self.validation), set(self.locked_test))
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError("episode splits overlap")
        return self


class TrafficConfig(StrictModel):
    base_arrival_rate: float = Field(default=1.05, gt=0.0)
    zipf_alpha: float = Field(default=1.0, ge=0.8, le=1.2)
    duration_shape: float = Field(default=2.2, gt=1.0)
    duration_min_s: float = Field(default=8.0, gt=0.0)
    duration_max_s: float = Field(default=120.0, gt=0.0)
    bitrate_lognormal_mean: float = 13.9
    bitrate_lognormal_sigma: float = Field(default=0.55, gt=0.0)
    low_priority_fraction: float = Field(default=0.4, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_duration_bounds(self) -> TrafficConfig:
        if self.duration_max_s <= self.duration_min_s:
            raise ValueError("duration_max_s must exceed duration_min_s")
        return self


class ExperimentConfig(StrictModel):
    name: str
    topology_path: Path
    output_root: Path = Path("runs")
    root_seed: int = Field(default=20260814, ge=0)
    episode_duration_s: int = Field(default=720, gt=0)
    simulation_step_s: int = Field(default=1, gt=0)
    window_s: int = Field(default=5, gt=0)
    horizons: tuple[int, ...] = (1, 2, 4)
    primary_horizon: int = 2
    episodes: int = Field(default=18, gt=0)
    monitored_object_count: int = Field(default=8, gt=0)
    traffic: TrafficConfig = TrafficConfig()
    congestion: CongestionConfig = CongestionConfig()
    split: SplitConfig
    sealed: bool = False

    @model_validator(mode="after")
    def validate_timing(self) -> ExperimentConfig:
        if self.simulation_step_s != 1:
            raise ValueError("the declared simulator contract requires a one-second step")
        if self.window_s % self.simulation_step_s:
            raise ValueError("window size must be divisible by simulation step")
        if self.episode_duration_s % self.window_s:
            raise ValueError("episode duration must be divisible by window size")
        if not self.horizons or any(horizon <= 0 for horizon in self.horizons):
            raise ValueError("horizons must be positive")
        if len(self.horizons) != len(set(self.horizons)):
            raise ValueError("horizons must be unique")
        if self.primary_horizon not in self.horizons:
            raise ValueError("primary_horizon must be listed in horizons")
        episode_ids = set(self.split.train + self.split.validation + self.split.locked_test)
        if len(episode_ids) != self.episodes:
            raise ValueError("split episode count must equal episodes")
        if self.sealed and (
            self.episodes != 18
            or len(self.split.train) != 10
            or len(self.split.validation) != 4
            or len(self.split.locked_test) != 4
        ):
            raise ValueError("sealed core experiments require the frozen 10/4/4 split")
        return self

    def resolve_paths(self, base_dir: Path) -> ExperimentConfig:
        values = self.model_dump()
        values["topology_path"] = (base_dir / self.topology_path).resolve()
        values["output_root"] = (base_dir / self.output_root).resolve()
        return ExperimentConfig.model_validate(values)
