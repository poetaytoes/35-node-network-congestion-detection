from __future__ import annotations

import numpy as np
import pandas as pd

from congestion_lab.evaluation import (
    binary_slice_metrics,
    evaluate_binary_models,
    evaluate_cause_model,
    system_metrics,
)
from congestion_lab.features import FEATURE_COLUMNS
from congestion_lab.models import fit_binary_models, fit_cause_models, fit_xgboost_model


def synthetic_features(rows_per_split: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(20260814)
    rows: list[dict[str, object]] = []
    for split in ("train", "validation", "locked_test"):
        for index in range(rows_per_split):
            signal = rng.normal() + (index % 4) * 0.75
            row: dict[str, object] = {
                feature: float(signal + rng.normal(scale=0.35)) for feature in FEATURE_COLUMNS
            }
            target = bool(signal > 1.1)
            cause_index = index % 4
            row.update(
                {
                    "episode_id": f"{split[0].upper()}{index // 30:02d}",
                    "topology": "centralized" if index % 2 == 0 else "distributed",
                    "object_id": f"obj{index % 3}",
                    "object_role": "backbone_link",
                    "scenario_family": f"S{cause_index}",
                    "window_index": index,
                    "window_start_s": index * 5,
                    "split": split,
                    "congestion_state": bool(index % 7 == 0),
                    "congestion_onset": bool(index % 31 == 0),
                    "future_congestion_h2": target,
                    "target_valid_h2": True,
                    "congestion_within_h2": target,
                    "model_eligible": True,
                    "cause_index": cause_index,
                    "cause_label": (
                        "normal",
                        "benign_high_load",
                        "attack_without_congestion",
                        "attack_associated_congestion",
                    )[cause_index],
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_binary_model_contract_and_locked_metrics() -> None:
    features = synthetic_features()
    bundles, selection = fit_binary_models(features, compact_grid=True)
    assert set(bundles) == {
        "threshold",
        "persistence",
        "ewma",
        "logistic_regression",
        "random_forest",
        "xgboost",
    }
    assert selection["model"].nunique() == 6
    assert {
        "validation_false_alarm_events",
        "validation_system_hours",
        "validation_false_alarms_per_hour",
        "validation_false_alarm_budget_per_hour",
    }.issubset(selection.columns)
    assert (selection["validation_false_alarms_per_hour"] <= 5.0).all()
    metrics, predictions, _ = evaluate_binary_models(bundles, features)
    assert len(metrics) == 12
    assert predictions["model"].nunique() == 6
    assert metrics["pr_auc"].between(0, 1).all()


def test_cause_model_contract() -> None:
    features = synthetic_features()
    bundles, selection = fit_cause_models(features)
    assert set(bundles) == {"cause_logistic", "cause_random_forest", "cause_xgboost"}
    assert selection["validation_macro_f1"].between(0, 1).all()
    metrics, matrix = evaluate_cause_model(bundles["cause_xgboost"], features)
    assert metrics["macro_f1"].between(0, 1).all()
    assert matrix.shape == (4, 4)


def test_false_alarm_rate_uses_system_hours_not_object_hours() -> None:
    base = synthetic_features(rows_per_split=80)
    expanded = pd.concat(
        [base.assign(object_id=f"obj{object_index}") for object_index in range(4)],
        ignore_index=True,
    )
    bundles, _ = fit_binary_models(expanded, compact_grid=True)
    metrics, predictions, _ = evaluate_binary_models({"xgboost": bundles["xgboost"]}, expanded)
    for row in metrics.itertuples():
        group = predictions.loc[
            (predictions["model"] == "xgboost") & (predictions["topology"] == row.topology)
        ]
        expected_hours = len(group[["episode_id", "window_index"]].drop_duplicates()) * 5 / 3600
        assert np.isclose(row.evaluation_hours, expected_hours)
        assert np.isclose(row.object_hours, expected_hours * 4)
        assert np.isclose(
            row.false_alarms_per_hour,
            row.false_alarm_events / expected_hours if expected_hours else 0.0,
        )


def test_binary_slice_marks_pr_auc_undefined_without_positive_support() -> None:
    predictions = pd.DataFrame(
        {
            "model": ["xgboost", "xgboost"],
            "scenario_family": ["negative_only", "negative_only"],
            "object_role": ["link", "link"],
            "cause_label": ["normal", "normal"],
            "future_congestion_h2": [0, 0],
            "score": [0.1, 0.2],
            "prediction": [0, 0],
        }
    )

    slices = binary_slice_metrics(predictions)

    assert not slices["pr_auc_defined"].any()
    assert slices["pr_auc"].isna().all()


def test_fixed_xgboost_protocol_records_one_matched_candidate() -> None:
    features = synthetic_features()
    primary, _ = fit_xgboost_model(features, compact_grid=True)
    fixed, selection = fit_xgboost_model(
        features,
        feature_columns=tuple(FEATURE_COLUMNS[:-1]),
        fixed_parameters=primary.parameters,
    )
    assert selection.loc[0, "grid_profile"] == "fixed_primary_parameters"
    assert selection.loc[0, "candidate_count"] == 1
    assert {key: value for key, value in fixed.parameters.items() if key != "scale_pos_weight"} == {
        key: value for key, value in primary.parameters.items() if key != "scale_pos_weight"
    }


def test_system_metrics_uses_deduplicated_end_to_end_fields(pilot_config) -> None:
    rows = []
    for second in range(10):
        for object_id in ("a", "b"):
            rows.append(
                {
                    "episode_id": "E",
                    "topology": "centralized",
                    "timestamp_s": second,
                    "object_id": object_id,
                    "utilization": 0.1,
                    "queue_ratio": 0.0,
                    "loss_rate": 0.0,
                    "loss_bytes": 0.0,
                    "offered_bytes": 1_000.0,
                    "delay_ms": 9_999.0,
                    "e2e_mean_latency_ms": 10.0 + second,
                    "e2e_p95_latency_ms": 20.0 + second,
                    "e2e_loss_bytes": 10.0,
                    "e2e_offered_bytes": 1_000.0,
                    "e2e_delivered_bytes": 990.0,
                    "source_offered_bytes": 1_000.0,
                    "e2e_accepted_requests": 9.0,
                    "e2e_rejected_requests": 1.0,
                    "backbone_bytes": 100.0 if object_id == "a" else 0.0,
                    "bhp_bytes": 250.0 if object_id == "a" else 300.0,
                }
            )
    metrics = system_metrics(pd.DataFrame(rows), config_name="C0", experiment=pilot_config).iloc[0]
    assert metrics["p95_latency_ms"] < 30.0
    assert np.isclose(metrics["loss_percent"], 1.0)
    assert metrics["accepted_requests"] == 90.0
    assert metrics["rejected_requests"] == 10.0
    assert metrics["bandwidth_hop_gb_hop"] > metrics["backbone_traffic_gb"]
    assert metrics["congestion_events"] == 1
    assert metrics["congestion_duration_s"] == pilot_config.window_s
    assert metrics["time_to_first_congestion_s"] == pilot_config.window_s
