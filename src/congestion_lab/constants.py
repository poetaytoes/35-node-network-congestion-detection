"""Stable schema constants shared across the pipeline."""

from __future__ import annotations

SCHEMA_VERSION = "1.0.0"
TOPOLOGIES = ("centralized", "distributed")
CAUSE_LABELS = (
    "normal",
    "benign_high_load",
    "attack_without_congestion",
    "attack_associated_congestion",
)
CAUSE_TO_INDEX = {label: index for index, label in enumerate(CAUSE_LABELS)}
INDEX_TO_CAUSE = {index: label for label, index in CAUSE_TO_INDEX.items()}
DEFAULT_FEATURE_GROUPS = (
    "utilization",
    "trend",
    "queue",
    "delay_loss",
    "flow",
    "volume",
    "protocol",
    "distribution",
    "short_flow",
    "object_context",
    "dc_context",
)
