"""Blocking feature-list leakage checks."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from congestion_lab.features.builder import FEATURE_COLUMNS

DENY_EXACT = {
    "episode_id",
    "topology",
    "object_id",
    "scenario_family",
    "attack_family",
    "attack_bytes_window",
    "attack_in_h2",
    "seed",
    "traffic_source",
    "seed_path",
    "source_ip",
    "destination_ip",
    "source_port",
    "destination_port",
    "congestion_state",
    "congestion_onset",
    "cause_label",
    "cause_index",
}
DENY_PREFIXES = (
    "future_",
    "target_",
    "selected_path_future",
    "ground_truth_",
    "attack_start",
    "attack_remaining",
    "raw_ip",
    "raw_port",
)


def leakage_audit(
    frame: pd.DataFrame, features: Sequence[str] = FEATURE_COLUMNS
) -> dict[str, object]:
    feature_set = set(features)
    missing = sorted(feature_set - set(frame.columns))
    blocked = sorted(
        feature
        for feature in feature_set
        if feature in DENY_EXACT or any(feature.startswith(prefix) for prefix in DENY_PREFIXES)
    )
    non_numeric = sorted(
        feature
        for feature in feature_set
        if feature in frame and not pd.api.types.is_numeric_dtype(frame[feature])
    )
    duplicates = sorted({feature for feature in features if list(features).count(feature) > 1})
    return {
        "status": "pass"
        if not missing and not blocked and not non_numeric and not duplicates
        else "fail",
        "feature_count": len(features),
        "missing": missing,
        "blocked": blocked,
        "non_numeric": non_numeric,
        "duplicates": duplicates,
    }


def assert_no_leakage(frame: pd.DataFrame, features: Sequence[str] = FEATURE_COLUMNS) -> None:
    report = leakage_audit(frame, features)
    if report["status"] != "pass":
        raise ValueError(f"blocking leakage audit: {report}")
