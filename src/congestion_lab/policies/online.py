"""Causal window-boundary policies with expiry, cooldown and action accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd

from congestion_lab.config.models import ExperimentConfig, TopologyConfig
from congestion_lab.features import build_feature_table
from congestion_lab.models.training import ModelBundle


@dataclass(frozen=True)
class PolicyConfig:
    name: str
    kind: str
    threshold: float
    shape_fraction: float
    duration_windows: int
    cooldown_windows: int
    reroute_fraction: float = 0.0
    tau_low: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in {
            "no_action",
            "low_priority_shaping",
            "precomputed_alternate_path",
        }:
            raise ValueError(f"unsupported policy kind: {self.kind}")
        for name, value in (
            ("threshold", self.threshold),
            ("shape_fraction", self.shape_fraction),
            ("reroute_fraction", self.reroute_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.duration_windows < 0 or self.cooldown_windows < 0:
            raise ValueError("policy window counts must be non-negative")
        if self.kind != "no_action" and self.duration_windows == 0:
            raise ValueError("active policies require a positive duration")
        if self.tau_low is not None and not 0.0 <= self.tau_low <= 1.0:
            raise ValueError("tau_low must be within [0, 1]")


@dataclass(frozen=True)
class PolicyDecision:
    trigger_object: str
    trigger_probability: float
    regions: tuple[str, ...]


class NoActionPolicy:
    def shaping_fraction(self, second: int, region: str) -> float:
        return 0.0

    def central_reroute_fraction(self, second: int, region: str) -> float:
        return 0.0

    def decide(self, current_features: pd.DataFrame) -> tuple[PolicyDecision, ...]:
        return ()

    def apply(self, window_end_s: int, decision: PolicyDecision) -> None:
        return None

    def expire(self, window_end_s: int) -> None:
        return None

    def on_window(self, window_end_s: int, telemetry: pd.DataFrame) -> None:
        return None

    def record_shaped(self, second: int, region: str, shaped_bytes: float) -> None:
        return None

    def action_frame(self) -> pd.DataFrame:
        return pd.DataFrame()


class OnlineShapingPolicy:
    """Apply a bounded rate reduction to low-priority traffic after model alarms."""

    def __init__(
        self,
        bundle: ModelBundle,
        policy: PolicyConfig,
        experiment: ExperimentConfig,
        topology: TopologyConfig,
        *,
        episode_id: str,
        topology_mode: str,
    ) -> None:
        if policy.kind not in {"low_priority_shaping", "precomputed_alternate_path"}:
            raise ValueError(f"unsupported policy kind: {policy.kind}")
        self.bundle = bundle
        self.policy = policy
        self.experiment = experiment
        self.topology = topology
        self.episode_id = episode_id
        self.topology_mode = topology_mode
        self.regions = sorted(node.id for node in topology.nodes if node.role == "region")
        self.active_until = {region: -1 for region in self.regions}
        self.cooldown_until = {region: -1 for region in self.regions}
        self.active_action: dict[str, int] = {}
        self.actions: list[dict[str, object]] = []
        
        # The model bundle is the frozen source of the alarm threshold.  The
        # policy may only set the lower, release threshold for hysteresis.
        self.tau_high = float(bundle.threshold)
        self.tau_low = float(policy.tau_low if policy.tau_low is not None else policy.threshold)
        if self.tau_low > self.tau_high:
            raise ValueError("tau_low cannot exceed the frozen bundle threshold")
        self.hysteresis_active = False

    def shaping_fraction(self, second: int, region: str) -> float:
        if self.policy.kind != "low_priority_shaping":
            return 0.0
        return self.policy.shape_fraction if second < self.active_until[region] else 0.0

    def central_reroute_fraction(self, second: int, region: str) -> float:
        if self.policy.kind != "precomputed_alternate_path":
            return 0.0
        return self.policy.reroute_fraction if second < self.active_until[region] else 0.0

    def record_shaped(self, second: int, region: str, shaped_bytes: float) -> None:
        action_index = self.active_action.get(region)
        if action_index is not None and second < self.active_until[region]:
            self.actions[action_index]["shaped_bytes"] = (
                cast(float, self.actions[action_index]["shaped_bytes"]) + shaped_bytes
            )

    def _regions_for_object(self, object_id: str) -> list[str]:
        if object_id.startswith("dc_") or object_id in {"B1--B2", "B2--B3"}:
            return self.regions
        for region_index, region in enumerate(self.regions, start=1):
            if f"PE{region_index}" in object_id:
                return [region]
        return self.regions

    def expire(self, window_end_s: int) -> None:
        expired = [
            region
            for region, action_index in self.active_action.items()
            if window_end_s >= self.active_until[region] and action_index >= 0
        ]
        for region in expired:
            self.active_action.pop(region, None)

    def decide(self, current_features: pd.DataFrame) -> tuple[PolicyDecision, ...]:
        if current_features.empty:
            return ()
        current = current_features.copy()
        current["score"] = self.bundle.predict_score(current)
        
        # Sort to find the most congested object
        sorted_features = current.sort_values(["score", "object_id"], ascending=[False, True])
        strongest = sorted_features.iloc[0]
        max_score = float(strongest["score"])

        # --- Dual-Threshold Hysteresis Logic ---
        if max_score >= self.tau_high:
            self.hysteresis_active = True
        elif max_score <= self.tau_low:
            self.hysteresis_active = False

        if self.hysteresis_active:
            return (
                PolicyDecision(
                    trigger_object=str(strongest["object_id"]),
                    trigger_probability=max_score,
                    regions=tuple(self._regions_for_object(str(strongest["object_id"]))),
                ),
            )
        return ()

    def apply(self, window_end_s: int, decision: PolicyDecision) -> None:
        for region in decision.regions:
            if window_end_s < self.cooldown_until[region]:
                continue
            expiry = window_end_s + self.policy.duration_windows * self.experiment.window_s
            cooldown = expiry + self.policy.cooldown_windows * self.experiment.window_s
            self.active_until[region] = expiry
            self.cooldown_until[region] = cooldown
            action = {
                "episode_id": self.episode_id,
                "topology": self.topology_mode,
                "policy": self.policy.name,
                "kind": self.policy.kind,
                "start_s": window_end_s,
                "expiry_s": expiry,
                "cooldown_until_s": cooldown,
                "region": region,
                "trigger_object": decision.trigger_object,
                "trigger_probability": decision.trigger_probability,
                "tau_high": self.tau_high,
                "tau_low": self.tau_low,
                "shape_fraction": self.policy.shape_fraction,
                "reroute_fraction": self.policy.reroute_fraction,
                "shaped_bytes": 0.0,
                "signaling_bytes": 128.0,
                "route_changes": 1 if self.policy.kind == "precomputed_alternate_path" else 0,
            }
            self.actions.append(action)
            self.active_action[region] = len(self.actions) - 1

    def on_window(self, window_end_s: int, telemetry: pd.DataFrame) -> None:
        self.expire(window_end_s)
        split_map = {self.episode_id: "online"}
        features = build_feature_table(telemetry, self.experiment, split_map)
        current_window = int(features["window_index"].max())
        current = features.loc[features["window_index"] == current_window].copy()
        for decision in self.decide(current):
            self.apply(window_end_s, decision)

    def action_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.actions)
