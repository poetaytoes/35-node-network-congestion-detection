"""Build five-second object-window features and future labels."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from congestion_lab.config.models import ExperimentConfig
from congestion_lab.constants import CAUSE_TO_INDEX

FEATURE_COLUMNS: tuple[str, ...] = (
    "util_mean",
    "util_max",
    "util_end",
    "util_lag1",
    "util_lag2",
    "util_rolling3",
    "util_slope3",
    "util_acceleration",
    "queue_ratio",
    "queue_growth",
    "queue_rolling3",
    "queue_to_rate",
    "delay_mean_ms",
    "delay_p95_ms",
    "loss_rate",
    "active_flows",
    "new_flow_rate",
    "completed_rate",
    "packet_rate",
    "byte_rate",
    "mean_size_bytes",
    "p95_size_bytes",
    "udp_ratio",
    "tcp_ratio",
    "syn_ratio",
    "src_count",
    "src_entropy",
    "dst_entropy",
    "top_talker_share",
    "short_flow_ratio",
    "mean_duration_s",
    "capacity_norm",
    "path_centrality",
    "cpu_proxy",
    "request_queue",
    "rejection_rate",
    "fan_in_ratio",
    "burst_score",
    "cross_layer_gap",
    "object_role_code",
)

METADATA_COLUMNS: tuple[str, ...] = (
    "episode_id",
    "topology",
    "object_id",
    "object_role",
    "scenario_family",
    "attack_family",
    "window_index",
    "window_start_s",
    "window_end_s",
    "split",
)


def _last(values: pd.Series) -> float:
    return float(values.iloc[-1])


def _weighted_loss(group: pd.DataFrame) -> float:
    offered = float(group["offered_bytes"].sum())
    return float(group["loss_bytes"].sum()) / offered if offered > 0 else 0.0


def _time_to_next_onset(onset: np.ndarray, window_s: int) -> np.ndarray:
    result = np.full(len(onset), np.nan, dtype=float)
    next_index: int | None = None
    for index in range(len(onset) - 1, -1, -1):
        if onset[index]:
            next_index = index
        if next_index is not None and next_index > index:
            result[index] = (next_index - index) * window_s
    return result


def _within_future(values: pd.Series, horizon: int) -> pd.Series:
    array = values.to_numpy(dtype=bool)
    result = np.zeros(len(array), dtype=bool)
    for index in range(len(array)):
        end = min(len(array), index + horizon + 1)
        result[index] = bool(array[index + 1 : end].any())
    return pd.Series(result, index=values.index)


def _future_attack(values: pd.Series, horizon: int) -> pd.Series:
    array = values.to_numpy(dtype=float)
    result = np.zeros(len(array), dtype=bool)
    for index in range(len(array)):
        end = min(len(array), index + horizon + 1)
        result[index] = bool((array[index:end] > 0).any())
    return pd.Series(result, index=values.index)


def relabel_feature_table(
    features: pd.DataFrame,
    config: ExperimentConfig,
    *,
    utilization_threshold: float | None = None,
) -> pd.DataFrame:
    """Rebuild congestion and cause targets without re-aggregating telemetry."""
    windows = features.copy()
    group_keys = ["episode_id", "topology", "object_id"]
    windows = windows.sort_values([*group_keys, "window_index"]).reset_index(drop=True)
    util_threshold = (
        config.congestion.utilization_threshold
        if utilization_threshold is None
        else utilization_threshold
    )
    windows["congestion_breach"] = (
        (windows["util_mean"] >= util_threshold)
        | (windows["queue_ratio"] >= config.congestion.queue_threshold)
        | (windows["loss_rate"] >= config.congestion.loss_threshold)
        | (windows["delay_p95_ms"] > config.congestion.delay_sla_ms)
    )
    groups = windows.groupby(group_keys, sort=False)
    persistence = config.congestion.persistence_windows
    persistent_count = groups["congestion_breach"].transform(
        lambda values: values.astype(int).rolling(persistence, min_periods=persistence).sum()
    )
    windows["congestion_state"] = persistent_count.ge(persistence).fillna(False)
    previous_state = groups["congestion_state"].shift(1).fillna(False).astype(bool)
    windows["congestion_onset"] = windows["congestion_state"] & ~previous_state
    windows["congestion_offset"] = ~windows["congestion_state"] & previous_state

    groups = windows.groupby(group_keys, sort=False)
    for horizon in config.horizons:
        target_name = f"future_congestion_h{horizon}"
        valid_name = f"target_valid_h{horizon}"
        within_name = f"congestion_within_h{horizon}"
        windows[target_name] = groups["congestion_state"].shift(-horizon)
        windows[valid_name] = windows[target_name].notna()
        windows[target_name] = windows[target_name].astype("boolean")
        windows[within_name] = groups["congestion_onset"].transform(
            lambda values, h=horizon: _within_future(values, h)
        )
    windows["attack_in_h2"] = groups["attack_bytes_window"].transform(
        lambda values: _future_attack(values, config.primary_horizon)
    )
    primary = windows[f"future_congestion_h{config.primary_horizon}"].fillna(False).astype(bool)
    windows["cause_label"] = np.select(
        [
            windows["attack_in_h2"] & primary,
            windows["attack_in_h2"] & ~primary,
            ~windows["attack_in_h2"] & primary,
        ],
        ["attack_associated_congestion", "attack_without_congestion", "benign_high_load"],
        default="normal",
    )
    windows["cause_index"] = windows["cause_label"].map(CAUSE_TO_INDEX).astype(int)
    windows["time_to_congestion_s"] = groups["congestion_onset"].transform(
        lambda values: pd.Series(
            _time_to_next_onset(values.to_numpy(dtype=bool), config.window_s),
            index=values.index,
        )
    )
    windows["model_eligible"] = windows[f"target_valid_h{config.primary_horizon}"]
    return windows


def build_feature_table(
    telemetry: pd.DataFrame,
    config: ExperimentConfig,
    split_map: dict[str, str] | None = None,
    *,
    utilization_threshold: float | None = None,
) -> pd.DataFrame:
    required = {
        "episode_id",
        "topology",
        "timestamp_s",
        "object_id",
        "object_role",
        "utilization",
        "queue_ratio",
        "loss_bytes",
        "offered_bytes",
        "delay_ms",
        "attack_bytes",
    }
    missing = required - set(telemetry.columns)
    if missing:
        raise ValueError(f"telemetry missing columns: {sorted(missing)}")
    frame = telemetry.copy()
    frame["window_index"] = (frame["timestamp_s"] // config.window_s).astype(int)
    group_keys = ["episode_id", "topology", "object_id", "window_index"]
    records: list[dict[str, object]] = []
    
    for keys, group in frame.groupby(group_keys, sort=True):
        group = group.sort_values("timestamp_s")
        episode_id, topology, object_id, window_index = keys
        
        # --- Normalized Source IP Shannon Entropy ---
        mean_src_count = float(group["src_count"].mean())
        raw_entropy = float(group["src_entropy"].mean())
        norm_entropy = raw_entropy / np.log2(max(mean_src_count, 2.0)) if mean_src_count > 1.0 else 0.0

        record: dict[str, object] = {
            "episode_id": str(episode_id),
            "topology": str(topology),
            "object_id": str(object_id),
            "object_role": str(group["object_role"].iloc[0]),
            "scenario_family": str(group["scenario_family"].iloc[0]),
            "attack_family": str(group["attack_family"].iloc[0]),
            "window_index": int(str(window_index)),
            "window_start_s": int(group["timestamp_s"].min()),
            "window_end_s": int(group["timestamp_s"].max()) + 1,
            "ticks": len(group),
            "util_mean": float(group["utilization"].mean()),
            "util_max": float(group["utilization"].max()),
            "util_end": _last(group["utilization"]),
            "queue_ratio": _last(group["queue_ratio"]),
            "delay_mean_ms": float(group["delay_ms"].mean()),
            "delay_p95_ms": float(group["delay_ms"].quantile(0.95)),
            "loss_rate": _weighted_loss(group),
            "active_flows": float(group["active_flows"].mean()),
            "new_flow_rate": float(group["new_flows"].sum() / config.window_s),
            "completed_rate": float(group["completed_flows"].sum() / config.window_s),
            "packet_rate": float(group["packet_rate"].mean()),
            "byte_rate": float(group["byte_rate"].mean()),
            "mean_size_bytes": float(group["mean_size_bytes"].replace(0, np.nan).mean()),
            "p95_size_bytes": float(group["p95_size_bytes"].max()),
            "udp_ratio": float(group["udp_ratio"].mean()),
            "tcp_ratio": float(group["tcp_ratio"].mean()),
            "syn_ratio": float(group["syn_ratio"].mean()),
            "src_count": mean_src_count,
            "src_entropy": norm_entropy,
            "dst_entropy": float(group["dst_entropy"].mean()),
            "top_talker_share": float(group["top_talker_share"].mean()),
            "short_flow_ratio": float(group["short_flow_ratio"].mean()),
            "mean_duration_s": float(group["mean_duration_s"].replace(0, np.nan).mean()),
            "capacity_norm": float(group["capacity_bytes_per_s"].iloc[0] / 50_000_000.0),
            "path_centrality": {"metro_link": 0.35, "backbone_link": 0.85, "dc_ingress": 0.65}.get(
                str(group["object_role"].iloc[0]), 0.5
            ),
            "cpu_proxy": float(group["cpu_proxy"].mean()),
            "request_queue": _last(group["request_queue"]),
            "rejection_rate": float(group["rejection_rate"].mean()),
            "attack_bytes_window": float(group["attack_bytes"].sum()),
            "backbone_bytes_window": float(group["backbone_bytes"].sum()),
            "bhp_bytes_window": float(group["bhp_bytes"].sum()),
            "accepted_window": float(group["accepted"].sum()),
            "rejected_window": float(group["rejected"].sum()),
            "loss_bytes_window": float(group["loss_bytes"].sum()),
            "offered_bytes_window": float(group["offered_bytes"].sum()),
        }
        records.append(record)
        
    windows = pd.DataFrame(records).sort_values(group_keys).reset_index(drop=True)
    windows["mean_size_bytes"] = windows["mean_size_bytes"].fillna(0.0)
    windows["mean_duration_s"] = windows["mean_duration_s"].fillna(0.0)
    windows["object_role_code"] = (
        windows["object_role"]
        .map({"metro_link": 0.0, "backbone_link": 1.0, "dc_ingress": 2.0})
        .astype(float)
    )
    windows["split"] = windows["episode_id"].map(split_map or {}).fillna("unassigned")

    causal_groups = windows.groupby(["episode_id", "topology", "object_id"], sort=False)
    windows["util_lag1"] = causal_groups["util_end"].shift(1)
    windows["util_lag2"] = causal_groups["util_end"].shift(2)
    windows["util_rolling3"] = causal_groups["util_end"].transform(
        lambda values: values.rolling(3, min_periods=1).mean()
    )
    
    # Use only current and preceding windows. The first two rows per object
    # intentionally remain NaN and are handled by the model imputer.
    windows["util_slope3"] = (
        (windows["util_end"] - windows["util_lag2"]) / (2.0 * config.window_s)
    )
    windows["util_acceleration"] = (
        (windows["util_end"] - 2.0 * windows["util_lag1"] + windows["util_lag2"])
        / float(config.window_s**2)
    )
    
    windows["queue_growth"] = causal_groups["queue_ratio"].diff()
    windows["queue_rolling3"] = causal_groups["queue_ratio"].transform(
        lambda values: values.rolling(3, min_periods=1).mean()
    )
    windows["queue_to_rate"] = windows["queue_ratio"] / (windows["byte_rate"] + 1.0)
    windows["fan_in_ratio"] = windows["src_count"] / (windows["active_flows"] + 1.0)
    windows["burst_score"] = windows["util_max"] / (windows["util_mean"] + 1e-9)
    windows["cross_layer_gap"] = windows["cpu_proxy"] - windows["util_mean"]

    return relabel_feature_table(
        windows,
        config,
        utilization_threshold=utilization_threshold,
    )


def feature_dictionary() -> pd.DataFrame:
    groups: dict[str, Iterable[str]] = {
        "utilization": ["util_mean", "util_max", "util_end"],
        "trend": ["util_lag1", "util_lag2", "util_rolling3", "util_slope3", "util_acceleration"],
        "queue": ["queue_ratio", "queue_growth", "queue_rolling3", "queue_to_rate"],
        "delay_loss": ["delay_mean_ms", "delay_p95_ms", "loss_rate"],
        "flow": ["active_flows", "new_flow_rate", "completed_rate"],
        "volume": ["packet_rate", "byte_rate", "mean_size_bytes", "p95_size_bytes", "burst_score"],
        "protocol": ["udp_ratio", "tcp_ratio", "syn_ratio"],
        "distribution": [
            "src_count",
            "src_entropy",
            "dst_entropy",
            "top_talker_share",
            "fan_in_ratio",
        ],
        "short_flow": ["short_flow_ratio", "mean_duration_s"],
        "object_context": ["capacity_norm", "path_centrality", "object_role_code"],
        "dc_context": ["cpu_proxy", "request_queue", "rejection_rate", "cross_layer_gap"],
    }
    definitions: dict[str, tuple[str, str]] = {
        "util_mean": ("ratio", "mean(utilization) over current 5 s window"),
        "util_max": ("ratio", "max(utilization) over current 5 s window"),
        "util_end": ("ratio", "last utilization in current window"),
        "util_lag1": ("ratio", "util_end at t-1 within episode/topology/object"),
        "util_lag2": ("ratio", "util_end at t-2 within episode/topology/object"),
        "util_rolling3": ("ratio", "causal mean of current and previous two util_end values"),
        
        "util_slope3": ("ratio/s", "causal two-window backward difference of util_end"),
        "util_acceleration": ("ratio/s^2", "causal second backward difference of util_end"),
        
        "queue_ratio": ("ratio", "last queue_bytes / buffer_bytes in current window"),
        "queue_growth": ("ratio/window", "first difference of queue_ratio"),
        "queue_rolling3": ("ratio", "causal three-window mean of queue_ratio"),
        "queue_to_rate": ("s/byte", "queue_ratio / (byte_rate + 1)"),
        "delay_mean_ms": ("ms", "mean local object delay in current window"),
        "delay_p95_ms": ("ms", "95th percentile local object delay in current window"),
        "loss_rate": ("ratio", "sum(loss_bytes) / sum(offered_bytes)"),
        "active_flows": ("flows", "mean active flow count in current window"),
        "new_flow_rate": ("flows/s", "sum(new_flows) / window_s"),
        "completed_rate": ("flows/s", "sum(completed_flows) / window_s"),
        "packet_rate": ("packets/s", "mean packet_rate in current window"),
        "byte_rate": ("bytes/s", "mean byte_rate in current window"),
        "mean_size_bytes": ("bytes", "mean nonzero mean flow size in current window"),
        "p95_size_bytes": ("bytes", "max per-tick p95 flow size in current window"),
        "burst_score": ("ratio", "util_max / (util_mean + 1e-9)"),
        "udp_ratio": ("ratio", "mean UDP byte share in current window"),
        "tcp_ratio": ("ratio", "mean TCP-like byte share in current window"),
        "syn_ratio": ("ratio", "mean SYN/scan byte share in current window"),
        "src_count": ("count", "mean unique active source_group_id count from raw events"),
        
        # --- Updated Dictionary Definitions for Audits ---
        "src_entropy": ("ratio", "normalized Shannon entropy of active source-group byte shares"),
        
        "dst_entropy": ("bits", "mean Shannon entropy of active service byte shares"),
        "top_talker_share": ("ratio", "mean largest observable source-group byte share"),
        "fan_in_ratio": ("ratio", "src_count / (active_flows + 1)"),
        "short_flow_ratio": ("ratio", "mean short-flow byte share"),
        "mean_duration_s": ("s", "mean nonzero active-flow duration"),
        "capacity_norm": ("ratio", "capacity_bytes_per_s / 50,000,000"),
        "path_centrality": ("score", "fixed role-derived path centrality score"),
        "object_role_code": ("category code", "metro=0, backbone=1, dc_ingress=2"),
        "cpu_proxy": ("ratio", "mean compute utilization proxy"),
        "request_queue": ("bytes", "last compute request queue in current window"),
        "rejection_rate": ("ratio", "mean compute rejection share"),
        "cross_layer_gap": ("ratio", "cpu_proxy - util_mean"),
    }
    rows = []
    for group, features in groups.items():
        for feature in features:
            unit, formula = definitions[feature]
            rows.append(
                {
                    "feature": feature,
                    "group": group,
                    "unit": unit,
                    "formula": formula,
                    "causal": True,
                    "source_scope": "observable_current_or_past",
                    "ground_truth_dependency": False,
                }
            )
    return pd.DataFrame(rows)
