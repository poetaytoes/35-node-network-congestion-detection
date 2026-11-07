"""Auditable row-level, alarm-event, lead-time and system metrics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from congestion_lab.config.models import ExperimentConfig
from congestion_lab.constants import CAUSE_LABELS
from congestion_lab.models.training import ModelBundle


class EventStatistics(TypedDict):
    false_alarm_events: int
    alarm_events: int
    congestion_events: int
    anticipated_events: int
    lead_times: list[float]


def _alarm_onsets(prediction: np.ndarray) -> np.ndarray:
    previous = np.concatenate(([False], prediction[:-1].astype(bool)))
    return prediction.astype(bool) & ~previous


def compute_tick_level_lead_time(alarm_tick: int, congestion_onset_tick: int) -> float:
    """Return lead time for the alarm event matched to this onset only."""
    return float(congestion_onset_tick - alarm_tick)


def _event_statistics(
    group: pd.DataFrame,
    prediction: np.ndarray,
    window_s: int,
    lookback_windows: int = 4,
) -> EventStatistics:
    alarm = _alarm_onsets(prediction)
    onset = group["congestion_onset"].to_numpy(dtype=bool)
    
    tick_timestamps = group["window_start_s"].to_numpy(dtype=int)
    onset_ticks = tick_timestamps[onset]
    
    alarm_indices = np.flatnonzero(alarm)
    onset_indices = np.flatnonzero(onset)
    
    false_alarms = 0
    matched_alarm_indices: set[int] = set()
    lead_times: list[float] = []
    
    for onset_index, onset_tick in zip(onset_indices, onset_ticks):
        candidates = alarm_indices[
            (alarm_indices < onset_index) & (alarm_indices >= onset_index - lookback_windows)
        ]
        if len(candidates):
            alarm_index = int(candidates[0])
            matched_alarm_indices.add(alarm_index)
            matched_tick = int(tick_timestamps[alarm_index])
            exact_lead_time = compute_tick_level_lead_time(matched_tick, int(onset_tick))
            if 0.0 < exact_lead_time <= lookback_windows * window_s:
                lead_times.append(exact_lead_time)
            
    for candidate_alarm_index in alarm_indices:
        future = onset_indices[
            (onset_indices > candidate_alarm_index)
            & (onset_indices <= candidate_alarm_index + lookback_windows)
        ]
        if not len(future):
            false_alarms += 1
            
    return {
        "false_alarm_events": false_alarms,
        "alarm_events": len(alarm_indices),
        "congestion_events": len(onset_indices),
        "anticipated_events": len(lead_times),
        "lead_times": lead_times,
    }


def evaluate_binary_models(
    bundles: Mapping[str, ModelBundle],
    features: pd.DataFrame,
    *,
    split: str = "locked_test",
    window_s: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    lead_rows: list[dict[str, object]] = []
    for model_name, bundle in bundles.items():
        target = f"future_congestion_h{bundle.horizon}"
        valid = f"target_valid_h{bundle.horizon}"
        evaluation = features.loc[(features["split"] == split) & features[valid]].copy()
        score = bundle.predict_score(evaluation)
        prediction = (score >= bundle.threshold).astype(int)
        evaluation["model"] = model_name
        evaluation["score"] = score
        evaluation["prediction"] = prediction
        prediction_frames.append(
            evaluation[
                [
                    "episode_id",
                    "topology",
                    "object_id",
                    "object_role",
                    "scenario_family",
                    "window_index",
                    "window_start_s",
                    target,
                    "congestion_onset",
                    "cause_label",
                    "model",
                    "score",
                    "prediction",
                ]
            ].copy()
        )
        for topology, topology_frame in evaluation.groupby("topology", sort=True):
            index = topology_frame.index
            topo_prediction = pd.Series(prediction, index=evaluation.index).loc[index].to_numpy()
            y_true = topology_frame[target].astype(int).to_numpy()
            topo_score = pd.Series(score, index=evaluation.index).loc[index].to_numpy()
            total_false = 0
            total_alarm = 0
            total_congestion = 0
            total_anticipated = 0
            all_leads: list[float] = []
            prediction_series = pd.Series(topo_prediction, index=topology_frame.index)
            for keys, group in topology_frame.groupby(["episode_id", "object_id"], sort=True):
                group = group.sort_values("window_index")
                stats = _event_statistics(
                    group,
                    prediction_series.loc[group.index].to_numpy(),
                    window_s,
                    bundle.horizon,
                )
                total_false += int(stats["false_alarm_events"])
                total_alarm += int(stats["alarm_events"])
                total_congestion += int(stats["congestion_events"])
                total_anticipated += int(stats["anticipated_events"])
                for lead in stats["lead_times"]:
                    lead_rows.append(
                        {
                            "model": model_name,
                            "topology": topology,
                            "episode_id": keys[0],
                            "object_id": keys[1],
                            "lead_time_s": float(lead),
                        }
                    )
                    all_leads.append(float(lead))
            evaluated_system_windows = topology_frame[
                ["episode_id", "window_index"]
            ].drop_duplicates()
            hours = len(evaluated_system_windows) * window_s / 3600.0
            object_hours = len(topology_frame) * window_s / 3600.0
            rows.append(
                {
                    "model": model_name,
                    "topology": topology,
                    "train_scope": bundle.train_scope,
                    "horizon": bundle.horizon,
                    "threshold": bundle.threshold,
                    "rows": len(topology_frame),
                    "prevalence": float(y_true.mean()),
                    "pr_auc": float(average_precision_score(y_true, topo_score)),
                    "precision": float(precision_score(y_true, topo_prediction, zero_division=0)),
                    "recall": float(recall_score(y_true, topo_prediction, zero_division=0)),
                    "f1": float(f1_score(y_true, topo_prediction, zero_division=0)),
                    "false_positive_windows": int(((topo_prediction == 1) & (y_true == 0)).sum()),
                    "false_alarm_events": total_false,
                    "false_alarms_per_hour": total_false / hours if hours > 0 else 0.0,
                    "evaluation_hours": hours,
                    "object_hours": object_hours,
                    "false_alarms_per_object_hour": (
                        total_false / object_hours if object_hours > 0 else 0.0
                    ),
                    "alarm_events": total_alarm,
                    "congestion_events": total_congestion,
                    "anticipated_events": total_anticipated,
                    "event_coverage": total_anticipated / total_congestion
                    if total_congestion
                    else 0.0,
                    "lead_time_median_s": float(np.median(all_leads)) if all_leads else np.nan,
                    "lead_time_p25_s": float(np.quantile(all_leads, 0.25)) if all_leads else np.nan,
                    "lead_time_p75_s": float(np.quantile(all_leads, 0.75)) if all_leads else np.nan,
                }
            )
    predictions = (
        pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    )
    return pd.DataFrame(rows), predictions, pd.DataFrame(lead_rows)


def evaluate_cause_model(
    bundle: ModelBundle,
    features: pd.DataFrame,
    *,
    split: str = "locked_test",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluation = features.loc[(features["split"] == split) & features["model_eligible"]].copy()
    prediction = bundle.predict_label(evaluation)
    truth = evaluation["cause_index"].to_numpy(dtype=int)
    matrix = confusion_matrix(truth, prediction, labels=range(4), normalize="true")
    metrics = pd.DataFrame(
        [
            {
                "model": bundle.name,
                "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
                "rows": len(evaluation),
                **{
                    f"support_{label}": int((truth == index).sum())
                    for index, label in enumerate(CAUSE_LABELS)
                },
                **{
                    f"precision_{label}": float(
                        precision_score(truth == index, prediction == index, zero_division=0)
                    )
                    for index, label in enumerate(CAUSE_LABELS)
                },
                **{
                    f"recall_{label}": float(
                        recall_score(truth == index, prediction == index, zero_division=0)
                    )
                    for index, label in enumerate(CAUSE_LABELS)
                },
                **{
                    f"f1_{label}": float(
                        f1_score(truth == index, prediction == index, zero_division=0)
                    )
                    for index, label in enumerate(CAUSE_LABELS)
                },
            }
        ]
    )
    matrix_frame = pd.DataFrame(matrix, index=CAUSE_LABELS, columns=CAUSE_LABELS)
    return metrics, matrix_frame


def cause_slice_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report four-class performance by topology and scenario family."""

    rows: list[dict[str, object]] = []
    dimensions = ("topology", "scenario_family")
    for dimension in dimensions:
        for value, group in predictions.groupby(dimension, sort=True):
            truth = group["cause_index"].to_numpy(dtype=int)
            predicted = group["predicted_cause_index"].to_numpy(dtype=int)
            row: dict[str, object] = {
                "slice_dimension": dimension,
                "slice_value": str(value),
                "rows": len(group),
                "macro_f1": float(
                    f1_score(truth, predicted, average="macro", labels=range(4), zero_division=0)
                ),
            }
            for index, label in enumerate(CAUSE_LABELS):
                row[f"support_{label}"] = int((truth == index).sum())
                row[f"precision_{label}"] = float(
                    precision_score(truth == index, predicted == index, zero_division=0)
                )
                row[f"recall_{label}"] = float(
                    recall_score(truth == index, predicted == index, zero_division=0)
                )
                row[f"f1_{label}"] = float(
                    f1_score(truth == index, predicted == index, zero_division=0)
                )
            rows.append(row)
    return pd.DataFrame(rows)


def cause_prediction_rows(
    bundle: ModelBundle,
    features: pd.DataFrame,
    *,
    split: str = "locked_test",
) -> pd.DataFrame:
    """Return auditable locked cause predictions with metadata kept out of X."""
    evaluation = features.loc[(features["split"] == split) & features["model_eligible"]].copy()
    prediction = bundle.predict_label(evaluation)
    result = evaluation.loc[
        :,
        [
            "episode_id",
            "topology",
            "object_id",
            "object_role",
            "scenario_family",
            "window_index",
            "window_start_s",
            "cause_label",
            "cause_index",
        ],
    ].copy()
    result["predicted_cause_index"] = prediction
    result["predicted_cause_label"] = [CAUSE_LABELS[int(index)] for index in prediction]
    result["correct"] = result["cause_index"] == result["predicted_cause_index"]
    return result


def binary_slice_metrics(
    predictions: pd.DataFrame,
    *,
    target: str = "future_congestion_h2",
) -> pd.DataFrame:
    """Compute locked binary metrics by scenario, object role, cause, and topology."""
    rows: list[dict[str, object]] = []
    dimensions = ("scenario_family", "object_role", "cause_label")
    for model, model_frame in predictions.groupby("model", sort=True):
        for dimension in dimensions:
            for value, group in model_frame.groupby(dimension, sort=True):
                truth = group[target].astype(int).to_numpy()
                score = group["score"].to_numpy(dtype=float)
                predicted = group["prediction"].astype(int).to_numpy()
                positive_support = int(truth.sum())
                pr_auc_defined = positive_support > 0
                rows.append(
                    {
                        "model": model,
                        "slice_dimension": dimension,
                        "slice_value": value,
                        "rows": len(group),
                        "positive_support": positive_support,
                        "prevalence": float(truth.mean()),
                        "pr_auc": (
                            float(average_precision_score(truth, score))
                            if pr_auc_defined
                            else float("nan")
                        ),
                        "pr_auc_defined": pr_auc_defined,
                        "precision": float(precision_score(truth, predicted, zero_division=0)),
                        "recall": float(recall_score(truth, predicted, zero_division=0)),
                        "f1": float(f1_score(truth, predicted, zero_division=0)),
                    }
                )
    return pd.DataFrame(rows)


def binary_error_analysis(
    predictions: pd.DataFrame,
    *,
    target: str = "future_congestion_h2",
) -> pd.DataFrame:
    """Return only locked false-positive and false-negative rows."""
    result = predictions.copy()
    truth = result[target].astype(int)
    predicted = result["prediction"].astype(int)
    result["error_type"] = np.select(
        [(truth == 0) & (predicted == 1), (truth == 1) & (predicted == 0)],
        ["false_positive", "false_negative"],
        default="correct",
    )
    return result.loc[result["error_type"] != "correct"].reset_index(drop=True)


def system_metrics(
    telemetry: pd.DataFrame,
    actions: pd.DataFrame | None = None,
    *,
    config_name: str,
    experiment: ExperimentConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (episode_id, topology), group in telemetry.groupby(["episode_id", "topology"], sort=True):
        windowed = group.copy()
        windowed["window_index"] = (windowed["timestamp_s"] // experiment.window_s).astype(int)
        windowed = windowed.sort_values(["object_id", "window_index", "timestamp_s"])
        object_windows = (
            windowed.groupby(["object_id", "window_index"], sort=True)
            .agg(
                util_mean=("utilization", "mean"),
                queue_ratio=("queue_ratio", "last"),
                loss_bytes=("loss_bytes", "sum"),
                offered_bytes=("offered_bytes", "sum"),
                delay_p95_ms=("delay_ms", lambda values: values.quantile(0.95)),
            )
            .reset_index()
        )
        object_windows["loss_rate"] = np.divide(
            object_windows["loss_bytes"],
            object_windows["offered_bytes"],
            out=np.zeros(len(object_windows), dtype=float),
            where=object_windows["offered_bytes"].to_numpy(dtype=float) > 0,
        )
        object_windows["breach"] = (
            (object_windows["util_mean"] >= experiment.congestion.utilization_threshold)
            | (object_windows["queue_ratio"] >= experiment.congestion.queue_threshold)
            | (object_windows["loss_rate"] >= experiment.congestion.loss_threshold)
            | (object_windows["delay_p95_ms"] > experiment.congestion.delay_sla_ms)
        )
        object_windows["congestion_state"] = object_windows.groupby("object_id", sort=False)[
            "breach"
        ].transform(
            lambda values: values.astype(int)
            .rolling(
                experiment.congestion.persistence_windows,
                min_periods=experiment.congestion.persistence_windows,
            )
            .sum()
            .ge(experiment.congestion.persistence_windows)
            .fillna(False)
        )
        by_second = group.groupby("timestamp_s", sort=True).agg(
            e2e_mean_latency_ms=("e2e_mean_latency_ms", "first"),
            e2e_p95_latency_ms=("e2e_p95_latency_ms", "first"),
            e2e_loss_bytes=("e2e_loss_bytes", "first"),
            e2e_offered_bytes=("e2e_offered_bytes", "first"),
            e2e_delivered_bytes=("e2e_delivered_bytes", "first"),
            source_offered_bytes=("source_offered_bytes", "first"),
            accepted_requests=("e2e_accepted_requests", "first"),
            rejected_requests=("e2e_rejected_requests", "first"),
            backbone_bytes=("backbone_bytes", "sum"),
            bhp_bytes=("bhp_bytes", "sum"),
        )
        congestion_by_window = object_windows.groupby("window_index", sort=True)[
            "congestion_state"
        ].any()
        onset = congestion_by_window & ~congestion_by_window.shift(1, fill_value=False)
        action_subset = (
            actions.loc[(actions["episode_id"] == episode_id) & (actions["topology"] == topology)]
            if actions is not None and not actions.empty
            else pd.DataFrame()
        )
        offered = float(by_second["e2e_offered_bytes"].sum())
        delivered = by_second["e2e_delivered_bytes"].to_numpy(dtype=float)
        latency = by_second["e2e_mean_latency_ms"].to_numpy(dtype=float)
        latency_denominator = float(delivered.sum())
        mean_latency = (
            float(np.average(latency, weights=delivered)) if latency_denominator > 0 else 0.0
        )
        shaped_bytes = (
            float(action_subset["shaped_bytes"].sum()) if not action_subset.empty else 0.0
        )
        source_offered = float(by_second["source_offered_bytes"].sum())
        rows.append(
            {
                "config": config_name,
                "episode_id": episode_id,
                "topology": topology,
                "p95_latency_ms": float(by_second["e2e_p95_latency_ms"].quantile(0.95)),
                "mean_latency_ms": mean_latency,
                "loss_percent": float(by_second["e2e_loss_bytes"].sum() / offered * 100.0)
                if offered
                else 0.0,
                "congestion_events": int(onset.sum()),
                "congestion_duration_s": int(congestion_by_window.sum() * experiment.window_s),
                "time_to_first_congestion_s": int(
                    congestion_by_window.index[np.flatnonzero(congestion_by_window.to_numpy())[0]]
                    * experiment.window_s
                )
                if congestion_by_window.any()
                else np.nan,
                "backbone_traffic_gb": float(by_second["backbone_bytes"].sum() / 1_000_000_000.0),
                "bandwidth_hop_gb_hop": float(by_second["bhp_bytes"].sum() / 1_000_000_000.0),
                "accepted_requests": float(by_second["accepted_requests"].sum()),
                "rejected_requests": float(by_second["rejected_requests"].sum()),
                "actions": len(action_subset),
                "shaped_gb": shaped_bytes / 1_000_000_000.0,
                "shaped_percent": shaped_bytes / source_offered * 100.0
                if source_offered > 0
                else 0.0,
                "signaling_bytes": float(action_subset["signaling_bytes"].sum())
                if not action_subset.empty
                else 0.0,
                "route_changes": int(action_subset["route_changes"].sum())
                if not action_subset.empty
                else 0,
            }
        )
    return pd.DataFrame(rows)
