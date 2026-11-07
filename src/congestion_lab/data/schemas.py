"""Versioned dataframe contracts for events and one-second telemetry."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from congestion_lab.constants import SCHEMA_VERSION, TOPOLOGIES
from congestion_lab.exceptions import ContractViolation

RAW_EVENT_COLUMNS: tuple[str, ...] = (
    "episode_id",
    "event_id",
    "arrival_time_s",
    "source_region",
    "service_id",
    "service_class",
    "duration_s",
    "mean_bitrate_bps",
    "flow_size_bytes",
    "packet_size_profile",
    "protocol_family",
    "source_group_id",
    "is_attack_ground_truth",
    "attack_family",
    "traffic_source",
    "seed_path",
)

TELEMETRY_COLUMNS: tuple[str, ...] = (
    "episode_id",
    "topology",
    "timestamp_s",
    "object_id",
    "object_role",
    "scenario_family",
    "capacity_bytes_per_s",
    "offered_bytes",
    "bytes_out",
    "utilization",
    "queue_bytes",
    "queue_ratio",
    "loss_bytes",
    "loss_rate",
    "propagation_delay_ms",
    "queue_delay_ms",
    "processing_delay_ms",
    "delay_ms",
    "active_flows",
    "new_flows",
    "completed_flows",
    "mean_size_bytes",
    "p95_size_bytes",
    "tcp_ratio",
    "udp_ratio",
    "syn_ratio",
    "short_flow_ratio",
    "top_talker_share",
    "src_count",
    "src_entropy",
    "dst_entropy",
    "request_queue",
    "cpu_proxy",
    "accepted",
    "rejected",
    "accepted_requests",
    "rejected_requests",
    "backbone_bytes",
    "bhp_bytes",
    "compute_consumed_bytes",
    "source_offered_bytes",
    "e2e_offered_bytes",
    "e2e_delivered_bytes",
    "e2e_loss_bytes",
    "e2e_mean_latency_ms",
    "e2e_p95_latency_ms",
    "e2e_request_count",
    "e2e_accepted_requests",
    "e2e_rejected_requests",
)


def _base_report(frame: pd.DataFrame, required: Iterable[str]) -> dict[str, Any]:
    required_columns = tuple(required)
    missing = sorted(set(required_columns) - set(frame.columns))
    nulls = {
        column: int(frame[column].isna().sum())
        for column in required_columns
        if column in frame and frame[column].isna().any()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "rows": len(frame),
        "columns": len(frame.columns),
        "missing": missing,
        "nulls": nulls,
        "violations": [],
    }


def validate_raw_events(frame: pd.DataFrame, *, raise_on_error: bool = True) -> dict[str, Any]:
    report = _base_report(frame, RAW_EVENT_COLUMNS)
    violations: list[str] = report["violations"]
    if not report["missing"]:
        if not frame["event_id"].is_unique:
            violations.append("event_id_not_unique")
        for column in ("arrival_time_s",):
            if (frame[column] < 0).any():
                violations.append(f"negative:{column}")
        for column in ("duration_s", "mean_bitrate_bps", "flow_size_bytes"):
            if (frame[column] <= 0).any():
                violations.append(f"non_positive:{column}")
        if not set(frame["service_class"].unique()).issubset({"low", "high"}):
            violations.append("service_class_domain")
    report["status"] = (
        "pass" if not report["missing"] and not report["nulls"] and not violations else "fail"
    )
    if raise_on_error and report["status"] != "pass":
        raise ContractViolation(f"raw event schema failed: {report}")
    return report


def validate_telemetry(frame: pd.DataFrame, *, raise_on_error: bool = True) -> dict[str, Any]:
    report = _base_report(frame, TELEMETRY_COLUMNS)
    violations: list[str] = report["violations"]
    if not report["missing"]:
        if not set(frame["topology"].unique()).issubset(set(TOPOLOGIES)):
            violations.append("topology_domain")
        nonnegative = (
            "timestamp_s",
            "capacity_bytes_per_s",
            "offered_bytes",
            "bytes_out",
            "queue_bytes",
            "loss_bytes",
            "propagation_delay_ms",
            "queue_delay_ms",
            "processing_delay_ms",
            "delay_ms",
            "accepted",
            "rejected",
            "accepted_requests",
            "rejected_requests",
            "compute_consumed_bytes",
            "source_offered_bytes",
            "e2e_offered_bytes",
            "e2e_delivered_bytes",
            "e2e_loss_bytes",
            "e2e_mean_latency_ms",
            "e2e_p95_latency_ms",
            "e2e_request_count",
            "e2e_accepted_requests",
            "e2e_rejected_requests",
        )
        for column in nonnegative:
            if (frame[column] < 0).any():
                violations.append(f"negative:{column}")
        for column in ("utilization", "queue_ratio", "loss_rate"):
            if not frame[column].between(0, 1 + 1e-12).all():
                violations.append(f"ratio_domain:{column}")
        if not (
            frame["delay_ms"]
            .sub(
                frame["propagation_delay_ms"]
                + frame["queue_delay_ms"]
                + frame["processing_delay_ms"]
            )
            .abs()
            .le(1e-9)
            .all()
        ):
            violations.append("delay_component_sum")
        if not frame["compute_consumed_bytes"].equals(frame["accepted"]):
            violations.append("compute_consumed_not_accepted")
    report["status"] = (
        "pass" if not report["missing"] and not report["nulls"] and not violations else "fail"
    )
    if raise_on_error and report["status"] != "pass":
        raise ContractViolation(f"telemetry schema failed: {report}")
    return report
