"""Deterministic benign and attack flow/session generator."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from congestion_lab.config.models import ExperimentConfig, TopologyConfig
from congestion_lab.data.artifacts import stable_frame_hash

DEMAND_VERSION = "synthetic-flow-v1"
ATTACK_VERSION = "bounded-research-profiles-v1"


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    family: str
    attack_family: str
    attack_intensity: str
    variant: str
    seed: int


CORE_EPISODE_LAYOUT: tuple[tuple[str, str, str], ...] = (
    ("S0_stable", "none", "nominal"),
    ("S0_stable", "none", "nominal"),
    ("S0_stable", "none", "validation_seed"),
    ("S1_gradual", "none", "ramp_a"),
    ("S1_gradual", "none", "ramp_b"),
    ("S1_gradual", "none", "validation_ramp"),
    ("S2_flash_crowd", "none", "service_a"),
    ("S2_flash_crowd", "none", "service_b"),
    ("S2_flash_crowd", "none", "locked_unseen_shape"),
    ("S3_udp_syn_flood", "udp_flood", "medium"),
    ("S3_udp_syn_flood", "syn_flood", "medium"),
    ("S3_udp_syn_flood", "udp_flood", "validation_high"),
    ("S3_udp_syn_flood", "syn_flood", "locked_unseen_high"),
    ("S4_http_slow_rate", "http_flood", "medium"),
    ("S4_http_slow_rate", "slow_rate_dos", "medium"),
    ("S4_http_slow_rate", "http_flood", "locked_unseen_bursty"),
    ("S5_hard_negative", "port_scan", "low"),
    ("S5_hard_negative", "port_scan", "locked_unseen_wide"),
)

# Final Vietnam-topology protocol: four independently seeded episodes from
# every scenario family.  The experiment config assigns two to training and
# one each to validation and locked test, so every split retains all six
# families while preserving episode-level isolation.
CORE_24_EPISODE_LAYOUT: tuple[tuple[str, str, str], ...] = (
    ("S0_stable", "none", "nominal"),
    ("S0_stable", "none", "train_seed_b"),
    ("S0_stable", "none", "validation_seed"),
    ("S0_stable", "none", "locked_seed"),
    ("S1_gradual", "none", "ramp_a"),
    ("S1_gradual", "none", "ramp_b"),
    ("S1_gradual", "none", "validation_ramp"),
    ("S1_gradual", "none", "locked_ramp"),
    ("S2_flash_crowd", "none", "service_a"),
    ("S2_flash_crowd", "none", "service_b"),
    ("S2_flash_crowd", "none", "validation_service_a"),
    ("S2_flash_crowd", "none", "locked_unseen_shape"),
    ("S3_udp_syn_flood", "udp_flood", "medium"),
    ("S3_udp_syn_flood", "syn_flood", "medium"),
    ("S3_udp_syn_flood", "udp_flood", "validation_high"),
    ("S3_udp_syn_flood", "syn_flood", "locked_unseen_high"),
    ("S4_http_slow_rate", "http_flood", "medium"),
    ("S4_http_slow_rate", "slow_rate_dos", "medium"),
    ("S4_http_slow_rate", "http_flood", "validation_high"),
    ("S4_http_slow_rate", "slow_rate_dos", "locked_unseen_bursty"),
    ("S5_hard_negative", "port_scan", "low"),
    ("S5_hard_negative", "port_scan", "train_wide"),
    ("S5_hard_negative", "port_scan", "validation_wide"),
    ("S5_hard_negative", "port_scan", "locked_unseen_wide"),
)


def _episode_sort_key(value: str) -> tuple[str, int]:
    match = re.search(r"\d+$", value)
    return re.sub(r"\d+$", "", value), int(match.group()) if match else -1


def episode_specs(config: ExperimentConfig) -> list[EpisodeSpec]:
    ids = list(config.split.train + config.split.validation + config.split.locked_test)
    ids = sorted(ids, key=_episode_sort_key)
    seed_sequence = np.random.SeedSequence(config.root_seed)
    child_sequences = seed_sequence.spawn(config.episodes)
    if config.episodes == 18:
        layout = CORE_EPISODE_LAYOUT
    elif config.episodes == 24:
        layout = CORE_24_EPISODE_LAYOUT
    else:
        layout = (
            ("S0_stable", "none", "pilot_stable"),
            ("S3_udp_syn_flood", "udp_flood", "pilot_attack"),
        )[: config.episodes]
    return [
        EpisodeSpec(
            episode_id=episode_id,
            family=family,
            attack_family=attack_family,
            attack_intensity=(
                "high"
                if "high" in variant
                else "low"
                if "low" in variant or "wide" in variant
                else "medium"
                if attack_family != "none"
                else "none"
            ),
            variant=variant,
            seed=int(child.generate_state(1)[0]),
        )
        for episode_id, (family, attack_family, variant), child in zip(
            ids, layout, child_sequences, strict=True
        )
    ]


def _bounded_pareto(
    rng: np.random.Generator,
    count: int,
    shape: float,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    values = minimum * (1.0 + rng.pareto(shape, count))
    return np.clip(values, minimum, maximum)


def _scenario_multiplier(spec: EpisodeSpec, second: int, duration: int, region_index: int) -> float:
    progress = second / max(duration - 1, 1)
    if spec.family == "S0_stable":
        return float(0.85 + 0.06 * np.sin(second / 37.0 + region_index))
    if spec.family == "S1_gradual":
        return 0.55 + 1.45 * progress
    if spec.family == "S2_flash_crowd":
        start = int(duration * 0.38)
        flash_duration = min(180, max(60, int(duration * 0.25)))
        end = min(duration, start + flash_duration)
        target = 0 if "service_a" in spec.variant else 2
        if start <= second < end and region_index == target:
            phase = (second - start) / max(end - start, 1)
            return float(1.0 + 4.2 * np.sin(np.pi * phase) ** 2)
        return 0.82
    return float(0.76 + 0.05 * np.sin(second / 43.0 + region_index))


def _attack_parameters(spec: EpisodeSpec, duration: int) -> dict[str, float | int | str]:
    locked = "locked" in spec.variant
    start = int(duration * (0.36 if not locked else 0.31))
    attack_duration = min(120, max(30, int(duration * (0.18 if not locked else 0.24))))
    factor = 1.35 if "high" in spec.attack_intensity or locked else 1.0
    parameters: dict[str, dict[str, float | str]] = {
        "udp_flood": {"rate": 9.0, "bitrate": 4_000_000.0, "duration": 2.5, "protocol": "udp"},
        "syn_flood": {"rate": 24.0, "bitrate": 80_000.0, "duration": 3.0, "protocol": "syn"},
        "http_flood": {"rate": 7.0, "bitrate": 900_000.0, "duration": 12.0, "protocol": "http"},
        "slow_rate_dos": {"rate": 1.8, "bitrate": 220_000.0, "duration": 100.0, "protocol": "tcp"},
        "port_scan": {"rate": 13.0, "bitrate": 18_000.0, "duration": 1.2, "protocol": "scan"},
    }
    selected = parameters.get(
        spec.attack_family, {"rate": 0.0, "bitrate": 0.0, "duration": 0.0, "protocol": "none"}
    )
    return {
        "start": start,
        "end": min(duration, start + attack_duration),
        "rate": float(selected["rate"]) * factor,
        "bitrate": float(selected["bitrate"]) * factor,
        "flow_duration": float(selected["duration"]),
        "protocol": str(selected["protocol"]),
    }


def _make_rows(
    *,
    spec: EpisodeSpec,
    region: str,
    second: int,
    count: int,
    rng: np.random.Generator,
    config: ExperimentConfig,
    attack: bool,
    attack_params: dict[str, float | int | str] | None = None,
) -> list[dict[str, object]]:
    if count <= 0:
        return []
    if attack and attack_params is not None:
        durations = np.clip(
            rng.lognormal(np.log(float(attack_params["flow_duration"])), 0.18, count),
            1.0,
            config.traffic.duration_max_s,
        )
        bitrates = rng.lognormal(np.log(float(attack_params["bitrate"])), 0.22, count)
        protocol = str(attack_params["protocol"])
        protocols = np.repeat(protocol, count)
        low_priority = np.repeat(False, count)
        service_ids = np.repeat("svc_victim", count)
    else:
        durations = _bounded_pareto(
            rng,
            count,
            config.traffic.duration_shape,
            config.traffic.duration_min_s,
            config.traffic.duration_max_s,
        )
        bitrates = rng.lognormal(
            config.traffic.bitrate_lognormal_mean,
            config.traffic.bitrate_lognormal_sigma,
            count,
        )
        protocols = rng.choice(["tcp", "udp", "http"], count, p=[0.55, 0.25, 0.20])
        low_priority = rng.random(count) < config.traffic.low_priority_fraction
        services = np.asarray(["svc_video", "svc_api", "svc_web"])
        ranks = np.arange(1, len(services) + 1, dtype=float)
        popularity = np.power(ranks, -config.traffic.zipf_alpha)
        popularity /= popularity.sum()
        service_ids = rng.choice(services, count, p=popularity)
    packet_profiles = np.where(
        np.isin(protocols, ["syn", "scan"]),
        "small",
        np.where(protocols == "udp", "mixed", "large"),
    )
    compute_cost = np.where(
        protocols == "syn",
        600_000.0,
        np.where(protocols == "http", 260_000.0, np.where(protocols == "scan", 12_000.0, 70_000.0)),
    )
    rows: list[dict[str, object]] = []
    for index in range(count):
        duration = float(durations[index])
        bitrate = float(bitrates[index])
        rows.append(
            {
                "episode_id": spec.episode_id,
                "arrival_time_s": second,
                "source_region": region,
                "service_id": str(service_ids[index]),
                "service_class": "low" if bool(low_priority[index]) else "high",
                "duration_s": duration,
                "mean_bitrate_bps": bitrate,
                "flow_size_bytes": bitrate * duration / 8.0,
                "packet_size_profile": str(packet_profiles[index]),
                "protocol_family": str(protocols[index]),
                "source_group_id": f"src-{rng.integers(0, 4096):04x}",
                "is_attack_ground_truth": bool(attack),
                "attack_family": spec.attack_family if attack else "none",
                "traffic_source": "synthetic",
                "compute_cost_bytes": float(compute_cost[index]),
                "scenario_family": spec.family,
                "seed": spec.seed,
                "seed_path": f"{config.root_seed}/{spec.episode_id}/{spec.seed}",
            }
        )
    return rows


def generate_episode(
    spec: EpisodeSpec,
    config: ExperimentConfig,
    topology: TopologyConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(spec.seed)
    regions = sorted(node.id for node in topology.nodes if node.role == "region")
    rows: list[dict[str, object]] = []
    attack_params = _attack_parameters(spec, config.episode_duration_s)
    for second in range(config.episode_duration_s):
        for region_index, region in enumerate(regions):
            multiplier = _scenario_multiplier(spec, second, config.episode_duration_s, region_index)
            count = int(rng.poisson(config.traffic.base_arrival_rate * multiplier))
            rows.extend(
                _make_rows(
                    spec=spec,
                    region=region,
                    second=second,
                    count=count,
                    rng=rng,
                    config=config,
                    attack=False,
                )
            )
        if int(attack_params["start"]) <= second < int(attack_params["end"]):
            target_region = regions[0] if spec.attack_family != "http_flood" else regions[-1]
            attack_count = int(rng.poisson(float(attack_params["rate"])))
            rows.extend(
                _make_rows(
                    spec=spec,
                    region=target_region,
                    second=second,
                    count=attack_count,
                    rng=rng,
                    config=config,
                    attack=True,
                    attack_params=attack_params,
                )
            )
    frame = pd.DataFrame(rows)
    frame.insert(1, "event_id", [f"{spec.episode_id}-{index:07d}" for index in range(len(frame))])
    return frame


def generate_workloads(
    config: ExperimentConfig,
    topology: TopologyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = episode_specs(config)
    frames = [generate_episode(spec, config, topology) for spec in specs]
    events = pd.concat(frames, ignore_index=True)
    manifest = pd.DataFrame([asdict(spec) for spec in specs])
    split_map = {
        **{episode: "train" for episode in config.split.train},
        **{episode: "validation" for episode in config.split.validation},
        **{episode: "locked_test" for episode in config.split.locked_test},
    }
    manifest["split"] = manifest["episode_id"].map(split_map)
    manifest["demand_version"] = DEMAND_VERSION
    manifest["attack_version"] = ATTACK_VERSION
    manifest["pattern"] = manifest["variant"]
    episode_details: dict[str, dict[str, object]] = {}
    for spec, frame in zip(specs, frames, strict=True):
        attack_parameters = _attack_parameters(spec, config.episode_duration_s)
        episode_details[spec.episode_id] = {
            "seed_path": f"{config.root_seed}/{spec.episode_id}/{spec.seed}",
            "attack_start_s": int(attack_parameters["start"]),
            "attack_end_s": int(attack_parameters["end"]),
            "event_count": len(frame),
            "offered_bytes": float(frame["flow_size_bytes"].sum()),
            "workload_sha256": stable_frame_hash(frame),
        }
    for column in (
        "seed_path",
        "attack_start_s",
        "attack_end_s",
        "event_count",
        "offered_bytes",
        "workload_sha256",
    ):
        manifest[column] = manifest["episode_id"].map(
            {episode: details[column] for episode, details in episode_details.items()}
        )
    return events, manifest


def workload_summary(events: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, object]:
    return {
        "episodes": int(events["episode_id"].nunique()),
        "events": len(events),
        "attack_events": int(events["is_attack_ground_truth"].sum()),
        "offered_bytes": float(events["flow_size_bytes"].sum()),
        "families": manifest["family"].value_counts().sort_index().to_dict(),
        "split_counts": manifest["split"].value_counts().sort_index().to_dict(),
    }


def iter_episode_frames(events: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    for episode_id, frame in events.groupby("episode_id", sort=True):
        yield str(episode_id), frame.reset_index(drop=True)
