"""Validated undirected physical graph and deterministic routing strategies."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import pairwise

import networkx as nx

from congestion_lab.config.models import LinkConfig, TopologyConfig


@dataclass(frozen=True)
class Route:
    region: str
    destination: str
    nodes: tuple[str, ...]
    link_ids: tuple[str, ...]
    physical_nodes: tuple[str, ...] = ()


class NetworkTopology:
    """Builds a physical graph and exposes deterministic C/D routes."""

    def __init__(self, config: TopologyConfig) -> None:
        self.config = config
        graph = nx.Graph(name=config.name)
        for node in config.nodes:
            graph.add_node(node.id, **node.model_dump())
        for link in config.links:
            graph.add_edge(
                link.source,
                link.target,
                id=link.id,
                weight=link.distance_km,
                **link.model_dump(),
            )
        if not nx.is_connected(graph):
            raise ValueError(f"topology {config.name} is disconnected")
        self.graph = graph
        self._links = {link.id: link for link in config.links}

    @property
    def link_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._links))

    def link(self, link_id: str) -> LinkConfig:
        return self._links[link_id]

    def _deterministic_shortest_path(self, source: str, target: str) -> tuple[str, ...]:
        """Return the distance-minimal path with a lexical full-path tie break."""
        frontier: list[tuple[float, tuple[str, ...], str]] = [(0.0, (source,), source)]
        best: dict[str, tuple[float, tuple[str, ...]]] = {source: (0.0, (source,))}
        while frontier:
            distance, path, node = heappop(frontier)
            if best.get(node) != (distance, path):
                continue
            if node == target:
                return path
            for neighbor in sorted(self.graph.neighbors(node)):
                edge = self.graph.edges[node, neighbor]
                candidate = (distance + float(edge["weight"]), (*path, neighbor))
                if neighbor not in best or candidate < best[neighbor]:
                    best[neighbor] = candidate
                    heappush(frontier, (candidate[0], candidate[1], neighbor))
        raise nx.NetworkXNoPath(f"no path from {source} to {target}")

    def route(self, region: str, topology: str) -> Route:
        if topology == "centralized":
            destination = self.config.central_dc
        elif topology == "distributed":
            destination = self.config.region_edge_map[region]
        else:
            raise ValueError(f"unsupported topology mode: {topology}")
        return self._route_to(region, destination)

    def route_for_service(self, region: str, topology: str, service_id: str) -> Route:
        """Use the fixed edge replica when present, otherwise fall back to the central DC."""
        if topology == "centralized":
            destination = self.config.central_dc
        elif topology == "distributed":
            preferred = self.config.region_edge_map[region]
            replicas = self.config.service_replicas.get(service_id, ())
            destination = preferred if preferred in replicas else self.config.central_dc
        else:
            raise ValueError(f"unsupported topology mode: {topology}")
        return self._route_to(region, destination)

    def _route_to(self, region: str, destination: str) -> Route:
        physical_nodes = self._deterministic_shortest_path(region, destination)
        roles = {node.id: node.role for node in self.config.nodes}
        # The 30-node authority image connects transport segments at co-located DC sites.
        # A non-target DC is therefore a passive physical junction, not a service hop.
        # Excluding it from ``nodes`` keeps service routing auditable while
        # ``physical_nodes`` and ``link_ids`` preserve every visible segment.
        service_nodes = tuple(
            node
            for node in physical_nodes
            if node in {region, destination} or roles[node] not in {"edge_dc", "central_dc"}
        )
        link_ids = tuple(
            "--".join(sorted((source, target))) for source, target in pairwise(physical_nodes)
        )
        return Route(
            region=region,
            destination=destination,
            nodes=service_nodes,
            link_ids=link_ids,
            physical_nodes=physical_nodes,
        )

    def routes(self, topology: str) -> dict[str, Route]:
        regions = sorted(node.id for node in self.config.nodes if node.role == "region")
        return {region: self.route(region, topology) for region in regions}

    def validate_contract(self) -> dict[str, int | str]:
        role_counts: dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            role = str(data["role"])
            role_counts[role] = role_counts.get(role, 0) + 1
        for mode in ("centralized", "distributed"):
            for route in self.routes(mode).values():
                if not route.link_ids:
                    raise ValueError(f"empty route in {mode}")
        return {
            "name": self.config.name,
            "nodes": self.graph.number_of_nodes(),
            "physical_links": self.graph.number_of_edges(),
            **{f"role_{key}": value for key, value in sorted(role_counts.items())},
        }
