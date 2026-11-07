"""One-second deterministic flow-level aggregate simulator."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Protocol

import numpy as np
import pandas as pd

from congestion_lab.config.models import ExperimentConfig, TopologyConfig
from congestion_lab.simulation.network import (
    LinkTick,
    QueueState,
    RouteDemand,
    TrafficConservingNetwork,
)
from congestion_lab.simulation.observability import (
    active_distribution_statistics,
    mixed_distribution,
)
from congestion_lab.topology import NetworkTopology


class OnlinePolicy(Protocol):
    """Minimal online policy interface used by the simulator."""

    def shaping_fraction(self, second: int, region: str) -> float: ...

    def central_reroute_fraction(self, second: int, region: str) -> float: ...

    def on_window(self, window_end_s: int, telemetry: pd.DataFrame) -> None: ...

    def record_shaped(self, second: int, region: str, shaped_bytes: float) -> None: ...

    def action_frame(self) -> pd.DataFrame: ...


@dataclass(frozen=True)
class SimulationResult:
    telemetry: pd.DataFrame
    actions: pd.DataFrame
    ledger: dict[str, float | int | str | bool]


@dataclass(frozen=True)
class RegionSeries:
    total_bytes: np.ndarray
    low_bytes: np.ndarray
    attack_bytes: np.ndarray
    tcp_bytes: np.ndarray
    udp_bytes: np.ndarray
    syn_bytes: np.ndarray
    short_bytes: np.ndarray
    active_flows: np.ndarray
    new_flows: np.ndarray
    completed_flows: np.ndarray
    compute_cost_bytes: np.ndarray
    mean_duration_s: np.ndarray
    packet_rate: np.ndarray
    mean_size_bytes: np.ndarray
    p95_size_bytes: np.ndarray
    src_count: np.ndarray
    src_entropy: np.ndarray
    dst_entropy: np.ndarray
    top_talker_share: np.ndarray


@dataclass(frozen=True)
class ObservedAggregate:
    total_bytes: float
    tcp_bytes: float
    udp_bytes: float
    syn_bytes: float
    short_bytes: float
    attack_bytes: float
    active_flows: float
    new_flows: float
    completed_flows: float
    packet_rate: float
    mean_size_bytes: float
    p95_size_bytes: float
    mean_duration_s: float
    src_count: float
    src_entropy: float
    dst_entropy: float
    top_talker_share: float


def _interval_add(
    diff: np.ndarray, starts: np.ndarray, ends: np.ndarray, values: np.ndarray
) -> None:
    np.add.at(diff, starts, values)
    valid_ends = ends < len(diff)
    np.add.at(diff, ends[valid_ends], -values[valid_ends])


def _active_values(
    *,
    duration: int,
    frame_length: int,
    starts: np.ndarray,
    ends: np.ndarray,
    default_values: np.ndarray,
    mask: np.ndarray | None = None,
    values: np.ndarray | None = None,
) -> np.ndarray:
    """Aggregate values over half-open active intervals."""

    diff = np.zeros(duration + 1, dtype=float)
    selected = np.ones(frame_length, dtype=bool) if mask is None else mask
    selected_values = default_values if values is None else values
    _interval_add(diff, starts[selected], ends[selected], selected_values[selected])
    return np.cumsum(diff[:-1])


def _region_series(
    events: pd.DataFrame, regions: list[str], duration: int
) -> dict[str, RegionSeries]:
    result: dict[str, RegionSeries] = {}
    for region in regions:
        frame = events.loc[events["source_region"] == region].reset_index(drop=True)
        starts = frame["arrival_time_s"].to_numpy(dtype=int)
        raw_ends = (frame["arrival_time_s"] + frame["duration_s"]).to_numpy(dtype=float)
        ends = np.minimum(duration, np.ceil(raw_ends).astype(int))
        bytes_per_s = frame["mean_bitrate_bps"].to_numpy(dtype=float) / 8.0

        active_values = partial(
            _active_values,
            duration=duration,
            frame_length=len(frame),
            starts=starts,
            ends=ends,
            default_values=bytes_per_s,
        )
        total = active_values()
        low = active_values(mask=frame["service_class"].to_numpy() == "low")
        # Ground truth remains available only for target/cause construction.
        attack = active_values(mask=frame["is_attack_ground_truth"].to_numpy(dtype=bool))
        protocol = frame["protocol_family"].to_numpy(dtype=str)
        tcp = active_values(mask=np.isin(protocol, ["tcp", "http", "syn", "scan"]))
        udp = active_values(mask=protocol == "udp")
        syn = active_values(mask=np.isin(protocol, ["syn", "scan"]))
        short = active_values(mask=frame["duration_s"].to_numpy(dtype=float) <= 5.0)
        active = active_values(values=np.ones(len(frame), dtype=float))

        new = np.zeros(duration, dtype=float)
        complete = np.zeros(duration, dtype=float)
        compute = np.zeros(duration, dtype=float)
        np.add.at(new, starts, 1.0)
        valid_ends = ends < duration
        np.add.at(complete, ends[valid_ends], 1.0)
        np.add.at(compute, starts, frame["compute_cost_bytes"].to_numpy(dtype=float))

        duration_weighted = active_values(values=frame["duration_s"].to_numpy(dtype=float))
        mean_duration = np.divide(
            duration_weighted,
            active,
            out=np.zeros_like(active),
            where=active > 0,
        )
        packets = np.divide(total, 900.0, out=np.zeros_like(total), where=total > 0)
        grouped = frame.groupby("arrival_time_s")["flow_size_bytes"]
        mean_size = np.zeros(duration, dtype=float)
        p95_size = np.zeros(duration, dtype=float)
        for second, values in grouped:
            index = int(str(second))
            if 0 <= index < duration:
                mean_size[index] = float(values.mean())
                p95_size[index] = float(values.quantile(0.95))

        source_stats = active_distribution_statistics(
            frame, key_column="source_group_id", duration=duration
        )
        destination_stats = active_distribution_statistics(
            frame, key_column="service_id", duration=duration
        )
        result[region] = RegionSeries(
            total_bytes=total,
            low_bytes=low,
            attack_bytes=attack,
            tcp_bytes=tcp,
            udp_bytes=udp,
            syn_bytes=syn,
            short_bytes=short,
            active_flows=active,
            new_flows=new,
            completed_flows=complete,
            compute_cost_bytes=compute,
            mean_duration_s=mean_duration,
            packet_rate=packets,
            mean_size_bytes=mean_size,
            p95_size_bytes=p95_size,
            src_count=source_stats.count,
            src_entropy=source_stats.entropy_bits,
            dst_entropy=destination_stats.entropy_bits,
            top_talker_share=source_stats.top_share,
        )
    return result


def _node_compute(topology: TopologyConfig, node_id: str) -> tuple[float, float]:
    node = next(item for item in topology.nodes if item.id == node_id)
    return node.compute_capacity_mbps * 1_000_000.0 / 8.0, node.queue_capacity_mb * 1_000_000.0


def _edge_split(
    series: dict[str, RegionSeries],
    topology: TopologyConfig,
    second: int,
) -> dict[str, float]:
    """Return the edge fraction; overload remainder is eligible for central fallback."""

    fractions = {region: 1.0 for region in series}
    by_edge: dict[str, list[str]] = {}
    for region, edge in topology.region_edge_map.items():
        by_edge.setdefault(edge, []).append(region)
    for edge, regions in by_edge.items():
        offered = sum(series[region].total_bytes[second] for region in regions)
        capacity, _ = _node_compute(topology, edge)
        if offered > capacity and offered > 0:
            ratio = capacity / offered
            for region in regions:
                fractions[region] = max(0.55, ratio)
    return fractions


def _weighted_nonzero(values: list[float], weights: list[float]) -> float:
    selected = [
        (value, weight) for value, weight in zip(values, weights, strict=True) if weight > 0
    ]
    if not selected:
        return 0.0
    numerator = sum(value * weight for value, weight in selected)
    denominator = sum(weight for _, weight in selected)
    return float(numerator / denominator) if denominator else 0.0


def _observed_aggregate(
    *,
    series: dict[str, RegionSeries],
    effective: dict[str, dict[str, float]],
    route_values: dict[str, float],
    route_regions: dict[str, str],
    second: int,
) -> ObservedAggregate:
    bytes_by_region: dict[str, float] = {}
    for key, value in route_values.items():
        region = route_regions[key]
        bytes_by_region[region] = bytes_by_region.get(region, 0.0) + max(value, 0.0)
    regions = sorted(bytes_by_region)
    scales = {
        region: (
            bytes_by_region[region] / effective[region]["total_bytes"]
            if effective[region]["total_bytes"] > 0.0
            else 0.0
        )
        for region in regions
    }

    def scaled(field: str) -> float:
        return float(sum(effective[region][field] * scales[region] for region in regions))

    totals = [bytes_by_region[region] for region in regions]
    counts = [series[region].src_count[second] * scales[region] for region in regions]
    entropies = [series[region].src_entropy[second] for region in regions]
    top_shares = [series[region].top_talker_share[second] for region in regions]
    src_count, src_entropy, top_share = mixed_distribution(
        totals=totals,
        counts=counts,
        entropies=entropies,
        top_shares=top_shares,
    )
    weights = totals
    return ObservedAggregate(
        total_bytes=float(sum(totals)),
        tcp_bytes=scaled("tcp_bytes"),
        udp_bytes=scaled("udp_bytes"),
        syn_bytes=scaled("syn_bytes"),
        short_bytes=scaled("short_bytes"),
        attack_bytes=scaled("attack_bytes"),
        active_flows=float(
            sum(series[region].active_flows[second] * scales[region] for region in regions)
        ),
        new_flows=float(
            sum(series[region].new_flows[second] * scales[region] for region in regions)
        ),
        completed_flows=float(
            sum(series[region].completed_flows[second] * scales[region] for region in regions)
        ),
        packet_rate=float(
            sum(series[region].packet_rate[second] * scales[region] for region in regions)
        ),
        mean_size_bytes=_weighted_nonzero(
            [series[region].mean_size_bytes[second] for region in regions], weights
        ),
        p95_size_bytes=max(
            (series[region].p95_size_bytes[second] for region in regions), default=0.0
        ),
        mean_duration_s=_weighted_nonzero(
            [series[region].mean_duration_s[second] for region in regions], weights
        ),
        src_count=src_count,
        src_entropy=src_entropy,
        dst_entropy=_weighted_nonzero(
            [series[region].dst_entropy[second] for region in regions], weights
        ),
        top_talker_share=top_share,
    )


def _weighted_quantile(values: list[float], weights: list[float], quantile: float) -> float:
    pairs = sorted(
        (value, weight) for value, weight in zip(values, weights, strict=True) if weight > 0.0
    )
    if not pairs:
        return 0.0
    threshold = quantile * sum(weight for _, weight in pairs)
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    return float(pairs[-1][0])


def _telemetry_row(
    *,
    episode_id: str,
    mode: str,
    second: int,
    object_id: str,
    object_role: str,
    scenario_family: str,
    attack_family: str,
    capacity: float,
    arrivals: float,
    transmitted: float,
    queue_bytes: float,
    buffer_bytes: float,
    dropped: float,
    propagation_delay_ms: float,
    queue_delay_ms: float,
    processing_delay_ms: float,
    observed: ObservedAggregate,
    backbone_bytes: float,
    bhp: float,
) -> dict[str, object]:
    denominator = max(observed.total_bytes, 1.0)
    loss_rate = min(1.0, max(0.0, dropped / arrivals)) if arrivals > 0 else 0.0
    return {
        "episode_id": episode_id,
        "topology": mode,
        "timestamp_s": second,
        "object_id": object_id,
        "object_role": object_role,
        "scenario_family": scenario_family,
        "attack_family": attack_family,
        "capacity_bytes_per_s": capacity,
        "offered_bytes": arrivals,
        "bytes_out": transmitted,
        "utilization": min(1.0, transmitted / capacity) if capacity > 0 else 0.0,
        "queue_bytes": queue_bytes,
        "queue_ratio": min(1.0, queue_bytes / buffer_bytes) if buffer_bytes > 0 else 0.0,
        "loss_bytes": dropped,
        "loss_rate": loss_rate,
        "delay_ms": propagation_delay_ms + queue_delay_ms + processing_delay_ms,
        "propagation_delay_ms": propagation_delay_ms,
        "queue_delay_ms": queue_delay_ms,
        "processing_delay_ms": processing_delay_ms,
        "active_flows": observed.active_flows,
        "new_flows": observed.new_flows,
        "completed_flows": observed.completed_flows,
        "mean_size_bytes": observed.mean_size_bytes,
        "p95_size_bytes": observed.p95_size_bytes,
        "packet_rate": observed.packet_rate,
        "byte_rate": arrivals,
        "tcp_ratio": observed.tcp_bytes / denominator,
        "udp_ratio": observed.udp_bytes / denominator,
        "syn_ratio": observed.syn_bytes / denominator,
        "short_flow_ratio": observed.short_bytes / denominator,
        "top_talker_share": observed.top_talker_share,
        "src_count": observed.src_count,
        "src_entropy": observed.src_entropy,
        "dst_entropy": observed.dst_entropy,
        "mean_duration_s": observed.mean_duration_s,
        "cpu_proxy": 0.0,
        "request_queue": 0.0,
        "rejection_rate": 0.0,
        "accepted": 0.0,
        "rejected": 0.0,
        "accepted_requests": 0.0,
        "rejected_requests": 0.0,
        "compute_consumed_bytes": 0.0,
        "attack_bytes": observed.attack_bytes,
        "backbone_bytes": backbone_bytes,
        "bhp_bytes": bhp,
        "source_offered_bytes": 0.0,
        "e2e_offered_bytes": 0.0,
        "e2e_delivered_bytes": 0.0,
        "e2e_loss_bytes": 0.0,
        "e2e_mean_latency_ms": 0.0,
        "e2e_p95_latency_ms": 0.0,
        "e2e_request_count": 0.0,
        "e2e_accepted_requests": 0.0,
        "e2e_rejected_requests": 0.0,
    }


def _dc_row(
    *,
    episode_id: str,
    mode: str,
    second: int,
    object_id: str,
    scenario_family: str,
    attack_family: str,
    capacity: float,
    arrivals: float,
    transmitted: float,
    queue_bytes: float,
    buffer_bytes: float,
    dropped: float,
    accepted_requests: float,
    rejected_requests: float,
    observed: ObservedAggregate,
) -> dict[str, object]:
    processing_delay = queue_bytes / capacity * 1000.0 if capacity > 0 else 0.0
    base = _telemetry_row(
        episode_id=episode_id,
        mode=mode,
        second=second,
        object_id=object_id,
        object_role="dc_ingress",
        scenario_family=scenario_family,
        attack_family=attack_family,
        capacity=capacity,
        arrivals=arrivals,
        transmitted=transmitted,
        queue_bytes=queue_bytes,
        buffer_bytes=buffer_bytes,
        dropped=dropped,
        propagation_delay_ms=0.0,
        queue_delay_ms=0.0,
        processing_delay_ms=processing_delay,
        observed=observed,
        backbone_bytes=0.0,
        bhp=0.0,
    )
    base["cpu_proxy"] = min(1.0, arrivals / capacity) if capacity > 0 else float(arrivals > 0)
    base["request_queue"] = queue_bytes
    base["rejection_rate"] = dropped / arrivals if arrivals > 0 else 0.0
    base["accepted"] = transmitted
    base["rejected"] = dropped
    base["accepted_requests"] = accepted_requests
    base["rejected_requests"] = rejected_requests
    base["compute_consumed_bytes"] = transmitted
    return base


def simulate_episode(
    events: pd.DataFrame,
    config: ExperimentConfig,
    topology: TopologyConfig,
    mode: str,
    *,
    capacity_scale: float = 1.0,
    policy: OnlinePolicy | None = None,
) -> SimulationResult:
    if mode not in {"centralized", "distributed"}:
        raise ValueError(f"unsupported mode: {mode}")
    episode_ids = events["episode_id"].unique()
    if len(episode_ids) != 1:
        raise ValueError("simulate_episode requires exactly one episode")
    episode_id = str(episode_ids[0])
    scenario_family = str(events["scenario_family"].iloc[0])
    attack_family = (
        str(events.loc[events["is_attack_ground_truth"], "attack_family"].iloc[0])
        if events["is_attack_ground_truth"].any()
        else "none"
    )
    duration = config.episode_duration_s
    network_topology = NetworkTopology(topology)
    network = TrafficConservingNetwork(topology)
    regions = sorted(node.id for node in topology.nodes if node.role == "region")
    series = _region_series(events, regions, duration)
    routes_edge = network_topology.routes("distributed")
    routes_central = network_topology.routes("centralized")

    central_state = QueueState()
    edge_states = {edge: QueueState() for edge in topology.edge_dcs}
    telemetry_rows: list[dict[str, object]] = []
    total_source_offered = 0.0
    total_network_offered = 0.0
    total_shaped = 0.0
    total_backbone = 0.0
    total_bhp = 0.0
    total_processed_compute = 0.0
    total_rejected_compute = 0.0
    total_e2e_delivered = 0.0
    total_e2e_loss = 0.0
    total_accepted_requests = 0.0
    total_rejected_requests = 0.0
    link_by_id = {link.id: link for link in topology.links}
    monitored = set(topology.monitored_links)
    fair_compute_capacity, _ = _node_compute(topology, topology.central_dc)
    fair_compute_capacity *= capacity_scale
    edge_config_capacity = sum(_node_compute(topology, edge)[0] for edge in topology.edge_dcs)
    if not np.isclose(edge_config_capacity, fair_compute_capacity / capacity_scale):
        raise ValueError(
            "distributed edge compute capacity must equal the declared centralized budget"
        )

    for second in range(duration):
        shape_by_region = {
            region: 0.0 if policy is None else policy.shaping_fraction(second, region)
            for region in regions
        }
        effective: dict[str, dict[str, float]] = {}
        source_offered_tick = 0.0
        for region in regions:
            source = series[region]
            shaped = source.low_bytes[second] * shape_by_region[region]
            total = max(0.0, source.total_bytes[second] - shaped)
            ratio = total / source.total_bytes[second] if source.total_bytes[second] > 0 else 1.0
            effective[region] = {
                "total_bytes": total,
                "low_bytes": max(0.0, source.low_bytes[second] - shaped),
                "attack_bytes": source.attack_bytes[second],
                "tcp_bytes": source.tcp_bytes[second] * ratio,
                "udp_bytes": source.udp_bytes[second] * ratio,
                "syn_bytes": source.syn_bytes[second] * ratio,
                "short_bytes": source.short_bytes[second] * ratio,
            }
            source_offered_tick += source.total_bytes[second]
            total_source_offered += source.total_bytes[second]
            total_shaped += shaped
            if policy is not None:
                policy.record_shaped(second, region, shaped)

        edge_fraction = (
            {region: 0.0 for region in regions}
            if mode == "centralized"
            else _edge_split(series, topology, second)
        )
        if policy is not None and mode == "distributed":
            for region in regions:
                edge_fraction[region] = max(
                    0.0,
                    edge_fraction[region] - policy.central_reroute_fraction(second, region),
                )

        demands: list[RouteDemand] = []
        route_regions: dict[str, str] = {}
        route_fractions: dict[str, float] = {}
        route_destinations: dict[str, str] = {}
        for region in regions:
            parts = (
                [(routes_central[region], 1.0)]
                if mode == "centralized"
                else [
                    (routes_edge[region], edge_fraction[region]),
                    (routes_central[region], 1.0 - edge_fraction[region]),
                ]
            )
            for route, fraction in parts:
                key = f"{region}|{route.destination}"
                route_regions[key] = region
                route_fractions[key] = fraction
                route_destinations[key] = route.destination
                demands.append(
                    RouteDemand(
                        key=key,
                        region=region,
                        destination=route.destination,
                        route=route,
                        offered_bytes=effective[region]["total_bytes"] * fraction,
                    )
                )
        network_tick = network.step(demands, capacity_scale=capacity_scale)
        network_offered_tick = float(sum(demand.offered_bytes for demand in demands))
        total_network_offered += network_offered_tick
        total_backbone += network_tick.backbone_bytes
        total_bhp += network_tick.bandwidth_hop_bytes

        tick_rows: list[dict[str, object]] = []
        for link_id, link in link_by_id.items():
            tick = network_tick.links.get(link_id, LinkTick(link_id=link_id))
            if link_id not in monitored:
                continue
            observed = _observed_aggregate(
                series=series,
                effective=effective,
                route_values=tick.route_arrivals,
                route_regions=route_regions,
                second=second,
            )
            capacity = tick.capacity_bytes_per_s or (
                link.capacity_mbps * capacity_scale * 1_000_000.0 / 8.0
            )
            buffer = tick.buffer_bytes or link.buffer_mb * 1_000_000.0
            tick_rows.append(
                _telemetry_row(
                    episode_id=episode_id,
                    mode=mode,
                    second=second,
                    object_id=link_id,
                    object_role=(
                        "backbone_link"
                        if link.link_class in {"backbone_access", "national_backbone"}
                        else "metro_link"
                    ),
                    scenario_family=scenario_family,
                    attack_family=attack_family,
                    capacity=capacity,
                    arrivals=tick.arrivals,
                    transmitted=tick.transmitted,
                    queue_bytes=tick.queue_bytes,
                    buffer_bytes=buffer,
                    dropped=tick.dropped,
                    propagation_delay_ms=tick.propagation_delay_ms,
                    queue_delay_ms=tick.queue_delay_ms,
                    processing_delay_ms=0.0,
                    observed=observed,
                    backbone_bytes=tick.backbone_ingress_bytes,
                    bhp=tick.bandwidth_hop_bytes,
                )
            )

        delivered_by_route = {key: path.delivered_bytes for key, path in network_tick.paths.items()}
        compute_by_destination: dict[str, float] = {}
        requests_by_destination: dict[str, float] = {}
        route_compute: dict[str, float] = {}
        route_requests: dict[str, float] = {}
        for key, path in network_tick.paths.items():
            region = route_regions[key]
            fraction = route_fractions[key]
            network_survival = (
                path.delivered_bytes / path.offered_bytes if path.offered_bytes > 0 else 1.0
            )
            compute = (
                path.delivered_bytes
                + series[region].compute_cost_bytes[second] * fraction * network_survival
            )
            requests = series[region].new_flows[second] * fraction * network_survival
            route_compute[key] = compute
            route_requests[key] = requests
            destination = route_destinations[key]
            compute_by_destination[destination] = (
                compute_by_destination.get(destination, 0.0) + compute
            )
            requests_by_destination[destination] = (
                requests_by_destination.get(destination, 0.0) + requests
            )

        destination_drop_ratio: dict[str, float] = {}
        destination_processing_delay: dict[str, float] = {}
        destination_accepted_requests: dict[str, float] = {}
        destination_rejected_requests: dict[str, float] = {}
        edge_rows: list[tuple[str, float, float, float, float, float, float, float]] = []
        edge_transmitted_total = 0.0
        for edge in topology.edge_dcs:
            arrivals = compute_by_destination.get(edge, 0.0)
            requests = requests_by_destination.get(edge, 0.0)
            capacity, buffer = _node_compute(topology, edge)
            capacity *= capacity_scale
            transmitted, dropped = edge_states[edge].step(arrivals, capacity, buffer)
            drop_ratio = min(1.0, max(0.0, dropped / arrivals)) if arrivals > 0 else 0.0
            accepted_requests = requests * (1.0 - drop_ratio)
            rejected_requests = requests - accepted_requests
            destination_drop_ratio[edge] = drop_ratio
            destination_processing_delay[edge] = (
                edge_states[edge].queue_bytes / capacity * 1000.0 if capacity > 0 else 0.0
            )
            destination_accepted_requests[edge] = accepted_requests
            destination_rejected_requests[edge] = rejected_requests
            edge_transmitted_total += transmitted
            edge_rows.append(
                (
                    edge,
                    capacity,
                    buffer,
                    arrivals,
                    transmitted,
                    dropped,
                    accepted_requests,
                    rejected_requests,
                )
            )

        central_arrivals = compute_by_destination.get(topology.central_dc, 0.0)
        central_requests = requests_by_destination.get(topology.central_dc, 0.0)
        _, central_buffer = _node_compute(topology, topology.central_dc)
        central_capacity = (
            fair_compute_capacity
            if mode == "centralized"
            else max(0.0, fair_compute_capacity - edge_transmitted_total)
        )
        central_transmitted, central_dropped = central_state.step(
            central_arrivals, central_capacity, central_buffer
        )
        central_drop_ratio = (
            min(1.0, max(0.0, central_dropped / central_arrivals)) if central_arrivals > 0 else 0.0
        )
        central_accepted_requests = central_requests * (1.0 - central_drop_ratio)
        central_rejected_requests = central_requests - central_accepted_requests
        destination_drop_ratio[topology.central_dc] = central_drop_ratio
        destination_processing_delay[topology.central_dc] = (
            central_state.queue_bytes / central_capacity * 1000.0 if central_capacity > 0 else 0.0
        )
        destination_accepted_requests[topology.central_dc] = central_accepted_requests
        destination_rejected_requests[topology.central_dc] = central_rejected_requests

        central_route_values = {
            key: value
            for key, value in delivered_by_route.items()
            if route_destinations[key] == topology.central_dc
        }
        central_observed = _observed_aggregate(
            series=series,
            effective=effective,
            route_values=central_route_values,
            route_regions=route_regions,
            second=second,
        )
        tick_rows.append(
            _dc_row(
                episode_id=episode_id,
                mode=mode,
                second=second,
                object_id="dc_central_ingress",
                scenario_family=scenario_family,
                attack_family=attack_family,
                capacity=central_capacity,
                arrivals=central_arrivals,
                transmitted=central_transmitted,
                queue_bytes=central_state.queue_bytes,
                buffer_bytes=central_buffer,
                dropped=central_dropped,
                accepted_requests=central_accepted_requests,
                rejected_requests=central_rejected_requests,
                observed=central_observed,
            )
        )

        edge_capacity_total = sum(row[1] for row in edge_rows)
        edge_buffer_total = sum(row[2] for row in edge_rows)
        edge_arrivals_total = sum(row[3] for row in edge_rows)
        edge_transmitted_total = sum(row[4] for row in edge_rows)
        edge_dropped_total = sum(row[5] for row in edge_rows)
        edge_accepted_requests = sum(row[6] for row in edge_rows)
        edge_rejected_requests = sum(row[7] for row in edge_rows)
        edge_queue_total = sum(edge_states[row[0]].queue_bytes for row in edge_rows)
        edge_route_values = {
            key: value
            for key, value in delivered_by_route.items()
            if route_destinations[key] in set(topology.edge_dcs)
        }
        edge_observed = _observed_aggregate(
            series=series,
            effective=effective,
            route_values=edge_route_values,
            route_regions=route_regions,
            second=second,
        )
        tick_rows.append(
            _dc_row(
                episode_id=episode_id,
                mode=mode,
                second=second,
                object_id="dc_edge_pool_ingress",
                scenario_family=scenario_family,
                attack_family=attack_family,
                capacity=edge_capacity_total,
                arrivals=edge_arrivals_total,
                transmitted=edge_transmitted_total,
                queue_bytes=edge_queue_total,
                buffer_bytes=edge_buffer_total,
                dropped=edge_dropped_total,
                accepted_requests=edge_accepted_requests,
                rejected_requests=edge_rejected_requests,
                observed=edge_observed,
            )
        )

        path_latencies: list[float] = []
        path_weights: list[float] = []
        e2e_delivered_tick = 0.0
        accepted_requests_tick = 0.0
        for key, path in network_tick.paths.items():
            destination = route_destinations[key]
            delivery = path.delivered_bytes * (1.0 - destination_drop_ratio[destination])
            latency = path.delay_ms + destination_processing_delay[destination]
            e2e_delivered_tick += delivery
            path_latencies.append(latency)
            path_weights.append(delivery)
            accepted_requests_tick += route_requests[key] * (
                1.0 - destination_drop_ratio[destination]
            )
        e2e_loss_tick = max(0.0, network_offered_tick - e2e_delivered_tick)
        source_requests_tick = float(sum(series[region].new_flows[second] for region in regions))
        rejected_requests_tick = max(0.0, source_requests_tick - accepted_requests_tick)
        mean_latency_tick = _weighted_nonzero(path_latencies, path_weights)
        p95_latency_tick = _weighted_quantile(path_latencies, path_weights, 0.95)
        e2e_payload = {
            "source_offered_bytes": source_offered_tick,
            "e2e_offered_bytes": network_offered_tick,
            "e2e_delivered_bytes": e2e_delivered_tick,
            "e2e_loss_bytes": e2e_loss_tick,
            "e2e_mean_latency_ms": mean_latency_tick,
            "e2e_p95_latency_ms": p95_latency_tick,
            "e2e_request_count": source_requests_tick,
            "e2e_accepted_requests": accepted_requests_tick,
            "e2e_rejected_requests": rejected_requests_tick,
        }
        for row in tick_rows:
            row.update(e2e_payload)
        telemetry_rows.extend(tick_rows)

        total_processed_compute += central_transmitted + edge_transmitted_total
        total_rejected_compute += central_dropped + edge_dropped_total
        total_e2e_delivered += e2e_delivered_tick
        total_e2e_loss += e2e_loss_tick
        total_accepted_requests += accepted_requests_tick
        total_rejected_requests += rejected_requests_tick

        if policy is not None and (second + 1) % config.window_s == 0:
            history_windows = 4
            history_rows = history_windows * config.window_s * config.monitored_object_count
            policy.on_window(second + 1, pd.DataFrame(telemetry_rows[-history_rows:]))

    telemetry = pd.DataFrame(telemetry_rows)
    actions = policy.action_frame() if policy is not None else pd.DataFrame()
    ledger: dict[str, float | int | str | bool] = {
        "episode_id": episode_id,
        "topology": mode,
        "offered_bytes": float(total_source_offered),
        "network_offered_bytes": float(total_network_offered),
        "shaped_bytes": float(total_shaped),
        "e2e_delivered_bytes": float(total_e2e_delivered),
        "e2e_loss_bytes": float(total_e2e_loss),
        "backbone_bytes": float(total_backbone),
        "bandwidth_hop_bytes": float(total_bhp),
        "accepted_compute_bytes": float(total_processed_compute),
        "rejected_compute_bytes": float(total_rejected_compute),
        "accepted_requests": float(total_accepted_requests),
        "rejected_requests": float(total_rejected_requests),
        "active_compute_budget_mbps": float(fair_compute_capacity * 8.0 / 1_000_000.0),
        "compute_budget_fair": True,
        "events": len(events),
    }
    return SimulationResult(telemetry=telemetry, actions=actions, ledger=ledger)
