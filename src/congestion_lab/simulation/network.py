"""Traffic-conserving fluid routing primitives for the one-second simulator.

The simulator is intentionally aggregate rather than packet-level.  Each tick admits
the bytes that survive a link's finite buffer to the next hop while the queue state
tracks service backlog.  That gives a deterministic, traffic-conserving path ledger
without pretending that independently repeated per-link demand is end-to-end traffic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

import networkx as nx

from congestion_lab.config.models import LinkConfig, TopologyConfig
from congestion_lab.topology.core import Route

BACKBONE_CLASSES = frozenset({"backbone_access", "national_backbone"})


@dataclass
class QueueState:
    """Finite-buffer work-conserving queue measured in bytes."""

    queue_bytes: float = 0.0
    total_dropped_bytes: float = 0.0
    total_transmitted_bytes: float = 0.0

    def step(self, arrivals: float, service: float, buffer: float) -> tuple[float, float]:
        available = self.queue_bytes + max(arrivals, 0.0)
        transmitted = min(available, max(service, 0.0))
        remaining = max(0.0, available - transmitted)
        dropped = min(max(arrivals, 0.0), max(0.0, remaining - buffer))
        self.queue_bytes = min(buffer, remaining)
        self.total_dropped_bytes += dropped
        self.total_transmitted_bytes += transmitted
        return transmitted, dropped


@dataclass(frozen=True)
class RouteDemand:
    """One region-to-DC commodity offered during the current tick."""

    key: str
    region: str
    destination: str
    route: Route
    offered_bytes: float


@dataclass(frozen=True)
class PathTick:
    offered_bytes: float
    delivered_bytes: float
    dropped_bytes: float
    delay_ms: float


@dataclass
class LinkTick:
    link_id: str
    capacity_bytes_per_s: float = 0.0
    arrivals: float = 0.0
    admitted: float = 0.0
    transmitted: float = 0.0
    dropped: float = 0.0
    queue_bytes: float = 0.0
    buffer_bytes: float = 0.0
    propagation_delay_ms: float = 0.0
    queue_delay_ms: float = 0.0
    backbone_ingress_bytes: float = 0.0
    bandwidth_hop_bytes: float = 0.0
    route_arrivals: dict[str, float] = field(default_factory=dict)
    route_admitted: dict[str, float] = field(default_factory=dict)

    @property
    def delay_ms(self) -> float:
        return self.propagation_delay_ms + self.queue_delay_ms


@dataclass(frozen=True)
class NetworkTick:
    paths: dict[str, PathTick]
    links: dict[str, LinkTick]
    backbone_bytes: float
    bandwidth_hop_bytes: float


class TrafficConservingNetwork:
    """Persist directed queue state and propagate every commodity hop by hop."""

    def __init__(self, topology: TopologyConfig) -> None:
        self.topology = topology
        self._links = {link.id: link for link in topology.links}
        self._states: dict[tuple[str, str], QueueState] = {}

    @staticmethod
    def _physical_nodes(route: Route) -> tuple[str, ...]:
        return route.physical_nodes or route.nodes

    def _link_for_arc(self, source: str, target: str) -> LinkConfig:
        return self._links["--".join(sorted((source, target)))]

    def _route_graph(self, demands: list[RouteDemand]) -> nx.DiGraph:
        graph = nx.DiGraph()
        for demand in demands:
            physical_nodes = self._physical_nodes(demand.route)
            graph.add_nodes_from(physical_nodes)
            graph.add_edges_from(pairwise(physical_nodes))
        if not nx.is_directed_acyclic_graph(graph):
            cycles = list(nx.simple_cycles(graph))[:3]
            raise ValueError(f"route union must be acyclic for one-tick propagation: {cycles}")
        return graph

    def step(self, demands: list[RouteDemand], *, capacity_scale: float = 1.0) -> NetworkTick:
        if capacity_scale <= 0.0:
            raise ValueError("capacity_scale must be positive")
        if not demands:
            return NetworkTick(paths={}, links={}, backbone_bytes=0.0, bandwidth_hop_bytes=0.0)

        route_by_key = {demand.key: demand for demand in demands}
        if len(route_by_key) != len(demands):
            raise ValueError("route demand keys must be unique")
        graph = self._route_graph(demands)
        node_order = tuple(nx.lexicographical_topological_sort(graph, key=str))
        current = {demand.key: max(0.0, demand.offered_bytes) for demand in demands}
        positions = {demand.key: 0 for demand in demands}
        delays = {demand.key: 0.0 for demand in demands}
        link_ticks = {link_id: LinkTick(link_id=link_id) for link_id in self._links}

        first_backbone: dict[str, str | None] = {}
        for demand in demands:
            first_backbone[demand.key] = next(
                (
                    link_id
                    for link_id in demand.route.link_ids
                    if self._links[link_id].link_class in BACKBONE_CLASSES
                ),
                None,
            )

        for node in node_order:
            by_arc: dict[tuple[str, str], dict[str, float]] = {}
            for key, demand in sorted(route_by_key.items()):
                physical_nodes = self._physical_nodes(demand.route)
                position = positions[key]
                if position >= len(physical_nodes) - 1 or physical_nodes[position] != node:
                    continue
                arc = (node, physical_nodes[position + 1])
                by_arc.setdefault(arc, {})[key] = current[key]

            for arc, route_arrivals in sorted(by_arc.items()):
                source, target = arc
                link = self._link_for_arc(source, target)
                state = self._states.setdefault(arc, QueueState())
                arrivals = float(sum(route_arrivals.values()))
                service = link.capacity_mbps * capacity_scale * 1_000_000.0 / 8.0
                buffer = link.buffer_mb * 1_000_000.0
                transmitted, dropped = state.step(arrivals, service, buffer)
                admitted = max(0.0, arrivals - dropped)
                survival = admitted / arrivals if arrivals > 0.0 else 1.0
                queue_delay_ms = state.queue_bytes / service * 1000.0 if service > 0.0 else 0.0
                propagation_delay_ms = link.distance_km * 0.005

                tick = link_ticks[link.id]
                tick.capacity_bytes_per_s += service
                tick.arrivals += arrivals
                tick.admitted += admitted
                tick.transmitted += transmitted
                tick.dropped += dropped
                tick.queue_bytes += state.queue_bytes
                tick.buffer_bytes += buffer
                tick.propagation_delay_ms = max(tick.propagation_delay_ms, propagation_delay_ms)
                tick.queue_delay_ms = max(tick.queue_delay_ms, queue_delay_ms)
                tick.bandwidth_hop_bytes += admitted
                for key, value in route_arrivals.items():
                    survived = value * survival
                    tick.route_arrivals[key] = tick.route_arrivals.get(key, 0.0) + value
                    tick.route_admitted[key] = tick.route_admitted.get(key, 0.0) + survived
                    if first_backbone[key] == link.id:
                        tick.backbone_ingress_bytes += survived
                    current[key] = survived
                    positions[key] += 1
                    delays[key] += propagation_delay_ms + queue_delay_ms

        unfinished = {
            key: (positions[key], len(self._physical_nodes(demand.route)) - 1)
            for key, demand in route_by_key.items()
            if positions[key] != len(self._physical_nodes(demand.route)) - 1
        }
        if unfinished:
            raise RuntimeError(f"not every route reached its destination: {unfinished}")

        path_ticks = {
            key: PathTick(
                offered_bytes=float(demand.offered_bytes),
                delivered_bytes=current[key],
                dropped_bytes=max(0.0, float(demand.offered_bytes) - current[key]),
                delay_ms=delays[key],
            )
            for key, demand in route_by_key.items()
        }
        return NetworkTick(
            paths=path_ticks,
            links=link_ticks,
            backbone_bytes=float(sum(tick.backbone_ingress_bytes for tick in link_ticks.values())),
            bandwidth_hop_bytes=float(
                sum(tick.bandwidth_hop_bytes for tick in link_ticks.values())
            ),
        )
