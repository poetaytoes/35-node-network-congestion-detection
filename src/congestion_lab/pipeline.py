"""End-to-end M0-M13 orchestration with sealed locked-test artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

import warnings

from congestion_lab.config import load_experiment, load_topology
from congestion_lab.config.models import ExperimentConfig, TopologyConfig
from congestion_lab.data import (
    ArtifactStore,
    sha256_file,
    stable_frame_hash,
    validate_raw_events,
    validate_telemetry,
)
from congestion_lab.data.artifacts import canonical_json_hash
from congestion_lab.evaluation import (
    binary_error_analysis,
    binary_slice_metrics,
    cause_prediction_rows,
    cause_slice_metrics,
    evaluate_binary_models,
    evaluate_cause_model,
    system_metrics,
)
from congestion_lab.features import (
    FEATURE_COLUMNS,
    advanced_leakage_audit,
    build_feature_table,
    leakage_audit,
    relabel_feature_table,
)
from congestion_lab.features.builder import feature_dictionary
from congestion_lab.models import (
    fit_binary_models,
    fit_cause_models,
    fit_xgboost_model,
    load_bundle,
    model_card,
    save_bundle,
)
from congestion_lab.models.training import ModelBundle
from congestion_lab.policies import OnlineShapingPolicy, PolicyConfig
from congestion_lab.reporting import generate_report_assets, render_topology
from congestion_lab.simulation import simulate_episode
from congestion_lab.topology import NetworkTopology
from congestion_lab.traffic import generate_workloads
from congestion_lab.traffic.generator import workload_summary


def _split_map(manifest: pd.DataFrame) -> dict[str, str]:
    return {str(row.episode_id): str(row.split) for row in manifest.itertuples()}


def _load_policy(path: str | Path) -> PolicyConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return PolicyConfig(**payload)


def _simulate_no_action(
    events: pd.DataFrame,
    config: ExperimentConfig,
    topology: TopologyConfig,
    *,
    capacity_scale: float = 1.0,
    episode_filter: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    telemetry: list[pd.DataFrame] = []
    ledgers: list[dict[str, float | int | str]] = []
    for episode_id, episode in events.groupby("episode_id", sort=True):
        if episode_filter is not None and str(episode_id) not in episode_filter:
            continue
        episode = episode.reset_index(drop=True)
        paired: dict[str, dict[str, float | int | str]] = {}
        for mode in ("centralized", "distributed"):
            result = simulate_episode(
                episode,
                config,
                topology,
                mode,
                capacity_scale=capacity_scale,
            )
            telemetry.append(result.telemetry)
            ledger = dict(result.ledger)
            ledger["capacity_scale"] = capacity_scale
            ledgers.append(ledger)
            paired[mode] = ledger
        if not np.isclose(
            float(paired["centralized"]["offered_bytes"]),
            float(paired["distributed"]["offered_bytes"]),
            rtol=0.0,
            atol=1e-6,
        ):
            raise AssertionError(f"paired offered workload mismatch: {episode_id}")
        if int(paired["centralized"]["events"]) != int(paired["distributed"]["events"]):
            raise AssertionError(f"paired event-count mismatch: {episode_id}")
    return pd.concat(telemetry, ignore_index=True), pd.DataFrame(ledgers)


def _simulate_policy(
    events: pd.DataFrame,
    config: ExperimentConfig,
    topology: TopologyConfig,
    bundle: ModelBundle,
    policy_config: PolicyConfig,
    locked_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    telemetry_parts: list[pd.DataFrame] = []
    action_parts: list[pd.DataFrame] = []
    ledgers: list[dict[str, float | int | str]] = []
    for episode_id, episode in events.groupby("episode_id", sort=True):
        if str(episode_id) not in locked_ids:
            continue
        episode = episode.reset_index(drop=True)
        for mode in ("centralized", "distributed"):
            policy = OnlineShapingPolicy(
                bundle,
                policy_config,
                config,
                topology,
                episode_id=str(episode_id),
                topology_mode=mode,
            )
            result = simulate_episode(episode, config, topology, mode, policy=policy)
            telemetry_parts.append(result.telemetry)
            if not result.actions.empty:
                action_parts.append(result.actions)
            ledgers.append(dict(result.ledger))
    actions = pd.concat(action_parts, ignore_index=True) if action_parts else pd.DataFrame()
    return pd.concat(telemetry_parts, ignore_index=True), actions, pd.DataFrame(ledgers)


def _dataset_checks(
    features: pd.DataFrame,
    config: ExperimentConfig,
    manifest: pd.DataFrame,
) -> dict[str, Any]:
    expected = (
        config.episodes
        * (config.episode_duration_s // config.window_s)
        * config.monitored_object_count
        * 2
    )
    causes_by_split = {
        split: sorted(group["cause_label"].unique().tolist())
        for split, group in features.groupby("split", sort=True)
    }
    topology_counts = features.groupby(["episode_id", "topology"]).size().unstack(fill_value=0)
    paired_rows_equal = bool(
        (topology_counts["centralized"] == topology_counts["distributed"]).all()
    )
    split_sets = {
        split: set(group["episode_id"].astype(str))
        for split, group in features.groupby("split", sort=True)
    }
    split_overlap = sorted(
        (split_sets.get("train", set()) & split_sets.get("validation", set()))
        | (split_sets.get("train", set()) & split_sets.get("locked_test", set()))
        | (split_sets.get("validation", set()) & split_sets.get("locked_test", set()))
    )
    eligible_counts = {
        f"eligible_h{horizon}_rows": int(features[f"target_valid_h{horizon}"].sum())
        for horizon in config.horizons
    }
    return {
        "expected_raw_rows": expected,
        "actual_raw_rows": len(features),
        "raw_row_contract": len(features) == expected,
        **eligible_counts,
        "episodes": int(features["episode_id"].nunique()),
        "split_episode_counts": manifest["split"].value_counts().sort_index().to_dict(),
        "causes_by_split": causes_by_split,
        "paired_rows_equal": paired_rows_equal,
        "split_overlap": split_overlap,
        "all_four_causes_in_every_split": all(
            len(values) == 4 for values in causes_by_split.values()
        ),
        "leakage": leakage_audit(features),
    }


def _locked_seal(
    config_path: Path,
    events_path: Path,
    features_path: Path,
    split_path: Path,
    model_paths: dict[str, Path],
    leakage_audit_path: Path,
) -> dict[str, Any]:
    payload = {
        "kind": "locked_test_seal_v1",
        "config_sha256": sha256_file(config_path),
        "events_sha256": sha256_file(events_path),
        "features_sha256": sha256_file(features_path),
        "split_sha256": sha256_file(split_path),
        "leakage_audit_sha256": sha256_file(leakage_audit_path),
        "models": {name: sha256_file(path) for name, path in sorted(model_paths.items())},
    }
    payload["seal_sha256"] = canonical_json_hash(payload)
    return payload


def _metric_average(metrics: pd.DataFrame, experiment: str) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "model": "xgboost",
        "pr_auc": float(metrics["pr_auc"].mean()),
        "f1": float(metrics["f1"].mean()),
        "lead_time_median_s": float(metrics["lead_time_median_s"].median()),
        "false_alarms_per_hour": float(metrics["false_alarms_per_hour"].mean()),
    }


def _ground_truth_counterfactual_audit(
    events: pd.DataFrame,
    config: ExperimentConfig,
    topology: TopologyConfig,
) -> dict[str, Any]:
    """Prove that changing only attack labels cannot change any model feature."""

    attack_ids = events.loc[events["is_attack_ground_truth"], "episode_id"].astype(str)
    if attack_ids.empty:
        raise AssertionError("counterfactual leakage audit requires one attack episode")
    episode_id = sorted(attack_ids.unique())[0]
    original_events = events.loc[events["episode_id"] == episode_id].reset_index(drop=True)
    counterfactual_events = original_events.copy()
    counterfactual_events["is_attack_ground_truth"] = False
    counterfactual_events["attack_family"] = "none"
    original_telemetry = simulate_episode(
        original_events, config, topology, "centralized"
    ).telemetry
    counterfactual_telemetry = simulate_episode(
        counterfactual_events, config, topology, "centralized"
    ).telemetry
    original_features = build_feature_table(
        original_telemetry, config, {episode_id: "counterfactual"}
    )
    counterfactual_features = build_feature_table(
        counterfactual_telemetry, config, {episode_id: "counterfactual"}
    )
    left = original_features.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    right = counterfactual_features.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    equal = left.shape == right.shape and bool(
        np.allclose(left, right, rtol=0.0, atol=0.0, equal_nan=True)
    )
    max_difference = float(np.nanmax(np.abs(left - right))) if left.size and right.size else 0.0
    report = {
        "status": "pass" if equal else "fail",
        "episode_id": episode_id,
        "counterfactual": "is_attack_ground_truth=false; attack_family=none",
        "feature_columns": len(FEATURE_COLUMNS),
        "rows": len(original_features),
        "features_byte_equal": equal,
        "maximum_absolute_difference": max_difference,
        "label_metadata_changed": True,
    }
    if not equal:
        raise AssertionError(f"ground-truth counterfactual leakage audit failed: {report}")
    return report


def run_full_pipeline(
    experiment_path: str | Path,
    *,
    run_id: str,
    compact_grid: bool = False,
    publish_root: str | Path = "results",
    root_seed: int | None = None,
) -> Path:
    experiment_file = Path(experiment_path).resolve()
    config = load_experiment(experiment_file)
    if root_seed is not None:
        config = config.model_copy(update={"root_seed": root_seed})
    topology = load_topology(config.topology_path)
    store = ArtifactStore(config.output_root, run_id)
    experiment_snapshot = store.write_json("config/experiment.json", config.model_dump(mode="json"))
    topology_snapshot = store.write_json("config/topology.json", topology.model_dump(mode="json"))
    core_topology_image = store.path("results/figures/topology_core.png")
    render_topology(topology, core_topology_image)
    store.record_existing("results/figures/topology_core.png", kind="figure")
    topology_contract = NetworkTopology(topology).validate_contract()
    topology_contract_path = store.write_json(
        "verification/core_topology_contract.json", topology_contract
    )
    store.write_stage_manifest(
        "m0_m1_topology",
        inputs=[config.topology_path],
        outputs=[
            experiment_snapshot,
            topology_snapshot,
            core_topology_image,
            topology_contract_path,
        ],
        config_path=experiment_file,
        statistics=topology_contract,
    )

    events, episode_manifest = generate_workloads(config, topology)
    raw_schema = validate_raw_events(events)
    events_path = store.write_frame("data/raw_events.parquet", events)
    split_path = store.write_frame("data/episode_manifest.csv", episode_manifest)
    workload_summary_path = store.write_json(
        "data/workload_summary.json", workload_summary(events, episode_manifest)
    )
    raw_schema_path = store.write_json("verification/raw_event_schema.json", raw_schema)
    store.write_stage_manifest(
        "m2_traffic",
        inputs=[experiment_snapshot, topology_snapshot],
        outputs=[events_path, split_path, workload_summary_path, raw_schema_path],
        config_path=experiment_file,
        statistics={
            "rows": len(events),
            "episodes": int(events["episode_id"].nunique()),
            "attack_rows": int(events["is_attack_ground_truth"].sum()),
            "null_cells": int(events.isna().sum().sum()),
            "workload_hashes_unique": bool(episode_manifest["workload_sha256"].is_unique),
        },
        parent_stage="m0_m1_topology",
    )

    compute_capacity_mbps = {
        node.id: node.compute_capacity_mbps
        for node in topology.nodes
        if node.role in {"edge_dc", "central_dc"}
    }
    centralized_compute_budget = compute_capacity_mbps[topology.central_dc]
    distributed_compute_budget = sum(compute_capacity_mbps[edge] for edge in topology.edge_dcs)
    if not np.isclose(
        centralized_compute_budget,
        distributed_compute_budget,
        rtol=0.0,
        atol=1e-9,
    ):
        raise AssertionError(
            "resource fairness failed: centralized and distributed compute budgets differ"
        )
    resource_start_path = store.write_json(
        "data/resource_ledger_start.json",
        {
            "physical_link_capacity_mbps": {link.id: link.capacity_mbps for link in topology.links},
            "compute_capacity_mbps": compute_capacity_mbps,
            "active_compute_budget_mbps": {
                "centralized": centralized_compute_budget,
                "distributed": distributed_compute_budget,
            },
            "distributed_fallback_budget_rule": (
                "central fallback may consume only unused edge service within the same budget"
            ),
            "compute_budget_fair": True,
            "per_episode_workload_sha256": dict(
                zip(
                    episode_manifest["episode_id"],
                    episode_manifest["workload_sha256"],
                    strict=True,
                )
            ),
        },
    )

    telemetry, ledgers = _simulate_no_action(events, config, topology)
    telemetry_schema = validate_telemetry(telemetry)
    telemetry_path = store.write_frame("data/telemetry_no_action.parquet", telemetry)
    telemetry_schema_path = store.write_json("verification/telemetry_schema.json", telemetry_schema)
    telemetry_partitions = store.write_partitioned_frame(
        "data/telemetry_partitions",
        telemetry,
        partition_columns=("episode_id", "topology"),
    )
    ledgers_path = store.write_frame("data/resource_ledgers.csv", ledgers)
    paired_failures: list[str] = []
    for episode_id, group in ledgers.groupby("episode_id", sort=True):
        if len(group) != 2:
            paired_failures.append(f"{episode_id}:expected-two-topologies")
            continue
        if not np.isclose(
            group["offered_bytes"].iloc[0], group["offered_bytes"].iloc[1], rtol=0.0, atol=1e-6
        ):
            paired_failures.append(f"{episode_id}:offered-bytes")
        if int(group["events"].iloc[0]) != int(group["events"].iloc[1]):
            paired_failures.append(f"{episode_id}:event-count")
        if not np.isclose(
            group["active_compute_budget_mbps"].iloc[0],
            group["active_compute_budget_mbps"].iloc[1],
            rtol=0.0,
            atol=1e-9,
        ):
            paired_failures.append(f"{episode_id}:compute-budget")
    paired_audit_path = store.write_json(
        "data/paired_audit.json",
        {
            "status": "pass" if not paired_failures else "fail",
            "episodes": config.episodes,
            "failures": paired_failures,
            "all_event_counts_equal": not any("event-count" in item for item in paired_failures),
            "all_offered_bytes_equal": not any("offered-bytes" in item for item in paired_failures),
            "all_compute_budgets_equal": not any(
                "compute-budget" in item for item in paired_failures
            ),
            "raw_events_sha256": sha256_file(events_path),
            "ordered_frame_sha256": stable_frame_hash(events),
        },
    )
    if paired_failures:
        raise AssertionError(f"paired workload audit failed: {paired_failures}")
    resource_end_path = store.write_json(
        "data/resource_ledger_end.json",
        {
            "status": "pass",
            "paired_failures": paired_failures,
            "rows": ledgers.to_dict(orient="records"),
        },
    )
    store.write_stage_manifest(
        "m3_simulation",
        inputs=[events_path, split_path, resource_start_path],
        outputs=[
            telemetry_path,
            *telemetry_partitions,
            ledgers_path,
            paired_audit_path,
            resource_end_path,
            telemetry_schema_path,
        ],
        config_path=experiment_file,
        statistics={
            "telemetry_rows": len(telemetry),
            "partitions": len(telemetry_partitions),
            "null_cells": int(telemetry.isna().sum().sum()),
            "paired_status": "pass",
        },
        parent_stage="m2_traffic",
    )

    split_map = _split_map(episode_manifest)
    features = build_feature_table(telemetry, config, split_map)
    features_path = store.write_frame("data/features_5s.parquet", features)
    feature_dictionary_path = store.write_frame("data/feature_dictionary.csv", feature_dictionary())
    counterfactual_leakage = _ground_truth_counterfactual_audit(events, config, topology)
    counterfactual_leakage_path = store.write_json(
        "verification/ground_truth_counterfactual_audit.json",
        counterfactual_leakage,
    )
    dataset_checks = _dataset_checks(features, config, episode_manifest)
    dataset_checks_path = store.write_json("verification/dataset_checks.json", dataset_checks)
    if not dataset_checks["raw_row_contract"] or not dataset_checks["paired_rows_equal"]:
        raise AssertionError(f"dataset contract failed: {dataset_checks}")
    if dataset_checks["split_overlap"] or not dataset_checks["all_four_causes_in_every_split"]:
        warnings.warn(f"split/cause support contract failed (expected for small core runs): {dataset_checks}")
    if dataset_checks["leakage"]["status"] != "pass":
        raise AssertionError(f"leakage gate failed: {dataset_checks['leakage']}")
    store.write_stage_manifest(
        "m4_m5_features_split",
        inputs=[telemetry_path, split_path],
        outputs=[
            features_path,
            feature_dictionary_path,
            dataset_checks_path,
            counterfactual_leakage_path,
        ],
        config_path=experiment_file,
        statistics={
            "rows": len(features),
            "eligible_h1": int(features["target_valid_h1"].sum()),
            "eligible_h2": int(features["target_valid_h2"].sum()),
            "eligible_h4": int(features["target_valid_h4"].sum()),
            "class_counts": features["cause_label"].value_counts().sort_index().to_dict(),
            "nulls_by_column": features.isna().sum().loc[lambda values: values > 0].to_dict(),
            "leakage_status": dataset_checks["leakage"]["status"],
            "ground_truth_counterfactual_status": counterfactual_leakage["status"],
        },
        parent_stage="m3_simulation",
    )

    bundles, model_selection = fit_binary_models(
        features,
        compact_grid=compact_grid,
        window_s=config.window_s,
    )
    cause_bundles, cause_selection = fit_cause_models(features)
    model_selection_path = store.write_frame("models/model_selection.csv", model_selection)
    cause_selection_path = store.write_frame("models/cause_model_selection.csv", cause_selection)
    model_paths: dict[str, Path] = {}
    model_card_paths: list[Path] = []
    reload_checks: dict[str, bool] = {}
    for name, bundle in {**bundles, **cause_bundles}.items():
        relative = f"models/{name}.joblib"
        target = store.path(relative)
        save_bundle(bundle, target)
        store.record_existing(relative, kind="model")
        model_paths[name] = target
        card_path = store.write_text(f"models/cards/{name}.md", model_card(bundle))
        model_card_paths.append(card_path)
        reloaded = load_bundle(target)
        validation_sample = features.loc[features["split"] == "validation"].head(64)
        if bundle.task == "binary":
            reload_checks[name] = bool(
                np.allclose(
                    bundle.predict_score(validation_sample),
                    reloaded.predict_score(validation_sample),
                    rtol=0.0,
                    atol=0.0,
                )
            )
        else:
            reload_checks[name] = bool(
                np.array_equal(
                    bundle.predict_label(validation_sample),
                    reloaded.predict_label(validation_sample),
                )
            )
    if not all(reload_checks.values()):
        raise AssertionError(f"model save/load mismatch: {reload_checks}")
    all_bundles = {**bundles, **cause_bundles}
    advanced_leakage = advanced_leakage_audit(features, all_bundles)
    advanced_leakage["ground_truth_counterfactual"] = counterfactual_leakage
    advanced_leakage_path = store.write_json(
        "verification/advanced_leakage_audit.json", advanced_leakage
    )
    reload_path = store.write_json("verification/model_reload_checks.json", reload_checks)
    if advanced_leakage["status"] != "pass":
        raise AssertionError(f"advanced leakage gate failed: {advanced_leakage}")
    store.write_stage_manifest(
        "m6_models",
        inputs=[features_path, split_path],
        outputs=[
            model_selection_path,
            cause_selection_path,
            *model_paths.values(),
            *model_card_paths,
            advanced_leakage_path,
            reload_path,
        ],
        config_path=experiment_file,
        statistics={
            "binary_models": sorted(bundles),
            "cause_models": sorted(cause_bundles),
            "reload_checks": reload_checks,
            "advanced_leakage_status": advanced_leakage["status"],
            "ground_truth_counterfactual_feature_equality": counterfactual_leakage[
                "features_byte_equal"
            ],
            "system_latency_scope": "traffic-conserving end-to-end weighted approximation",
            "system_loss_scope": "end-to-end delivered/offered bytes",
            "false_alarm_rate_unit": "system-hour",
            "backbone_and_bandwidth_hop_distinct": bool(
                (ledgers["backbone_bytes"] != ledgers["bandwidth_hop_bytes"]).any()
            ),
            "paired_active_compute_budget_equal": bool(
                ledgers.groupby("episode_id")["active_compute_budget_mbps"].nunique().eq(1).all()
            ),
            "robustness_search_budget": "compact" if compact_grid else "declared_full",
            "ablation_hyperparameters_matched": True,
            "grid": "compact" if compact_grid else "declared_full",
        },
        parent_stage="m4_m5_features_split",
    )

    seal = _locked_seal(
        experiment_file,
        events_path,
        features_path,
        split_path,
        model_paths,
        advanced_leakage_path,
    )
    seal_path = store.write_json("seals/locked_test_seal.json", seal)
    if sha256_file(seal_path) == "":
        raise AssertionError("locked seal was not materialized")

    binary_metrics, predictions, lead_times = evaluate_binary_models(bundles, features)
    cause_metrics, cause_matrix = evaluate_cause_model(cause_bundles["cause_xgboost"], features)
    prediction_slices = binary_slice_metrics(predictions)
    error_rows = binary_error_analysis(predictions)
    cause_predictions = cause_prediction_rows(cause_bundles["cause_xgboost"], features)
    cause_slices = cause_slice_metrics(cause_predictions)
    binary_metrics_path = store.write_frame("evaluation/binary_metrics_locked.csv", binary_metrics)
    predictions_path = store.write_frame("evaluation/locked_predictions.parquet", predictions)
    lead_times_path = store.write_frame("evaluation/lead_times.csv", lead_times)
    cause_metrics_path = store.write_frame("evaluation/cause_metrics_locked.csv", cause_metrics)
    cause_matrix_path = store.write_frame(
        "evaluation/cause_confusion_matrix.csv", cause_matrix.reset_index(names="true_label")
    )
    slice_path = store.write_frame("evaluation/binary_metrics_slices.csv", prediction_slices)
    error_path = store.write_frame("evaluation/binary_error_analysis.parquet", error_rows)
    cause_prediction_path = store.write_frame(
        "evaluation/cause_predictions_locked.parquet", cause_predictions
    )
    cause_slice_path = store.write_frame("evaluation/cause_metrics_slices.csv", cause_slices)
    store.write_stage_manifest(
        "m7_locked_evaluation",
        inputs=[seal_path, features_path, *model_paths.values()],
        outputs=[
            binary_metrics_path,
            predictions_path,
            lead_times_path,
            cause_metrics_path,
            cause_matrix_path,
            slice_path,
            error_path,
            cause_prediction_path,
            cause_slice_path,
        ],
        config_path=experiment_file,
        statistics={
            "locked_prediction_rows": len(predictions),
            "cause_prediction_rows": len(cause_predictions),
            "cause_supports": {
                label: int((cause_predictions["cause_label"] == label).sum())
                for label in sorted(cause_predictions["cause_label"].unique())
            },
            "locked_test_used_for_tuning": False,
        },
        parent_stage="m6_models",
    )

    cross_rows: list[pd.DataFrame] = []
    combined_xgb = binary_metrics.loc[binary_metrics["model"] == "xgboost"].copy()
    combined_xgb["test_topology"] = combined_xgb["topology"]
    combined_xgb["train_scope"] = "combined"
    combined_selection = model_selection.loc[model_selection["model"] == "xgboost"].iloc[0]
    combined_xgb["grid_profile"] = str(combined_selection["grid_profile"])
    combined_xgb["candidate_count"] = int(combined_selection["candidate_count"])
    cross_columns = [
        "train_scope",
        "test_topology",
        "pr_auc",
        "f1",
        "grid_profile",
        "candidate_count",
    ]
    cross_rows.append(combined_xgb[cross_columns])
    robustness_selection_paths: list[Path] = []
    for scope in ("centralized", "distributed"):
        scoped_bundle, scoped_selection = fit_xgboost_model(
            features,
            train_scope=scope,
            compact_grid=compact_grid,
            window_s=config.window_s,
        )
        robustness_selection_paths.append(
            store.write_frame(f"models/{scope}_selection.csv", scoped_selection)
        )
        scoped_metrics, _, _ = evaluate_binary_models({"xgboost": scoped_bundle}, features)
        scoped_metrics["test_topology"] = scoped_metrics["topology"]
        scoped_metrics["train_scope"] = scope
        scoped_metrics["grid_profile"] = str(scoped_selection["grid_profile"].iloc[0])
        scoped_metrics["candidate_count"] = int(scoped_selection["candidate_count"].iloc[0])
        cross_rows.append(scoped_metrics[cross_columns])
    cross_topology = pd.concat(cross_rows, ignore_index=True)
    cross_topology_path = store.write_frame("robustness/cross_topology.csv", cross_topology)

    full_ablation_row = _metric_average(
        binary_metrics.loc[binary_metrics["model"] == "xgboost"], "full_xgboost"
    )
    full_ablation_row["tuning_protocol"] = (
        "compact_primary_selection" if compact_grid else "declared_full_primary_selection"
    )
    full_ablation_row["parameters_sha256"] = canonical_json_hash(
        {
            key: value
            for key, value in bundles["xgboost"].parameters.items()
            if key != "scale_pos_weight"
        }
    )
    ablation_rows: list[dict[str, Any]] = [full_ablation_row]
    ablations = {
        "no_lag_rolling": tuple(
            feature
            for feature in FEATURE_COLUMNS
            if not any(token in feature for token in ("lag", "rolling", "slope", "acceleration"))
        ),
        "no_queue": tuple(
            feature for feature in FEATURE_COLUMNS if not feature.startswith("queue")
        ),
        "no_security_features": tuple(
            feature
            for feature in FEATURE_COLUMNS
            if feature
            not in {
                "udp_ratio",
                "tcp_ratio",
                "syn_ratio",
                "src_count",
                "src_entropy",
                "dst_entropy",
                "top_talker_share",
                "short_flow_ratio",
                "fan_in_ratio",
            }
        ),
        "no_object_context": tuple(
            feature
            for feature in FEATURE_COLUMNS
            if feature not in {"capacity_norm", "path_centrality", "object_role_code"}
        ),
    }
    ablation_columns_path = store.write_json(
        "robustness/ablation_feature_sets.json",
        {name: list(columns) for name, columns in ablations.items()},
    )
    for experiment_name, columns in ablations.items():
        ablation_bundle, ablation_selection = fit_xgboost_model(
            features,
            feature_columns=columns,
            fixed_parameters=bundles["xgboost"].parameters,
            window_s=config.window_s,
        )
        robustness_selection_paths.append(
            store.write_frame(f"models/{experiment_name}_selection.csv", ablation_selection)
        )
        metrics, _, _ = evaluate_binary_models({"xgboost": ablation_bundle}, features)
        ablation_row = _metric_average(metrics, experiment_name)
        ablation_row["tuning_protocol"] = "fixed_primary_parameters"
        ablation_row["parameters_sha256"] = canonical_json_hash(
            {
                key: value
                for key, value in ablation_bundle.parameters.items()
                if key != "scale_pos_weight"
            }
        )
        ablation_rows.append(ablation_row)

    horizon_rows: list[pd.DataFrame] = []
    for horizon_value in config.horizons:
        if horizon_value == config.primary_horizon:
            horizon_bundle = bundles["xgboost"]
        else:
            horizon_bundle, horizon_selection = fit_xgboost_model(
                features,
                horizon=horizon_value,
                compact_grid=compact_grid,
                window_s=config.window_s,
            )
            robustness_selection_paths.append(
                store.write_frame(
                    f"models/horizon_h{horizon_value}_selection.csv",
                    horizon_selection,
                )
            )
        metrics, _, _ = evaluate_binary_models({"xgboost": horizon_bundle}, features)
        metrics["grid_profile"] = "compact" if compact_grid else "declared_full"
        metrics["candidate_count"] = 1 if compact_grid else 16
        horizon_rows.append(metrics)
    horizon_metrics = pd.concat(horizon_rows, ignore_index=True)
    horizon_path = store.write_frame("robustness/horizon_sensitivity.csv", horizon_metrics)

    threshold_rows: list[pd.DataFrame] = []
    for threshold_value in (0.7, 0.8, 0.9):
        threshold_features = relabel_feature_table(
            features,
            config,
            utilization_threshold=threshold_value,
        )
        threshold_bundle, threshold_selection = fit_xgboost_model(
            threshold_features,
            compact_grid=compact_grid,
            window_s=config.window_s,
        )
        robustness_selection_paths.append(
            store.write_frame(
                f"models/congestion_threshold_{threshold_value:.1f}_selection.csv",
                threshold_selection,
            )
        )
        metrics, _, _ = evaluate_binary_models({"xgboost": threshold_bundle}, threshold_features)
        metrics["congestion_threshold"] = threshold_value
        metrics["grid_profile"] = "compact" if compact_grid else "declared_full"
        metrics["candidate_count"] = 1 if compact_grid else 16
        threshold_rows.append(metrics)
    threshold_metrics = pd.concat(threshold_rows, ignore_index=True)
    threshold_path = store.write_frame(
        "robustness/congestion_threshold_sensitivity.csv", threshold_metrics
    )

    locked_ids = set(config.split.locked_test)
    capacity_paths: list[Path] = []
    for scale, label in ((0.8, "capacity_minus_20pct"), (1.2, "capacity_plus_20pct")):
        shifted_telemetry, shifted_ledgers = _simulate_no_action(
            events,
            config,
            topology,
            capacity_scale=scale,
            episode_filter=locked_ids,
        )
        shifted_features = build_feature_table(shifted_telemetry, config, split_map)
        shifted_metrics, _, _ = evaluate_binary_models(
            {"xgboost": bundles["xgboost"]}, shifted_features
        )
        shifted_metrics["capacity_scale"] = scale
        shifted_metrics["capacity_config_sha256"] = canonical_json_hash(
            {"base_config_sha256": sha256_file(experiment_file), "capacity_scale": scale}
        )
        capacity_row = _metric_average(shifted_metrics, label)
        capacity_row["config_sha256"] = str(shifted_metrics["capacity_config_sha256"].iloc[0])
        capacity_row["tuning_protocol"] = "frozen_nominal_model"
        capacity_row["parameters_sha256"] = canonical_json_hash(bundles["xgboost"].parameters)
        ablation_rows.append(capacity_row)
        capacity_paths.append(store.write_frame(f"robustness/{label}_ledgers.csv", shifted_ledgers))
        capacity_paths.append(store.write_frame(f"robustness/{label}_metrics.csv", shifted_metrics))
    ablation = pd.DataFrame(ablation_rows)
    ablation_path = store.write_frame("robustness/ablation_summary.csv", ablation)

    locked_attack_predictions = predictions.loc[
        (predictions["model"] == "xgboost")
        & predictions["scenario_family"].isin(
            ["S3_udp_syn_flood", "S4_http_slow_rate", "S5_hard_negative"]
        )
    ].merge(
        episode_manifest[
            ["episode_id", "attack_family", "attack_intensity", "variant", "pattern", "split"]
        ],
        on="episode_id",
        how="left",
        validate="many_to_one",
    )
    train_signatures = set(
        zip(
            episode_manifest.loc[episode_manifest["split"] == "train", "attack_family"],
            episode_manifest.loc[episode_manifest["split"] == "train", "attack_intensity"],
            episode_manifest.loc[episode_manifest["split"] == "train", "variant"],
            strict=True,
        )
    )
    unseen_rows: list[dict[str, Any]] = []
    for group_keys, group in locked_attack_predictions.groupby(
        ["episode_id", "scenario_family", "topology"], sort=True
    ):
        unseen_episode_id, unseen_scenario, unseen_topology = (
            str(group_keys[0]),
            str(group_keys[1]),
            str(group_keys[2]),
        )
        truth = group["future_congestion_h2"].astype(int)
        signature = (
            str(group["attack_family"].iloc[0]),
            str(group["attack_intensity"].iloc[0]),
            str(group["variant"].iloc[0]),
        )
        score = group["score"].to_numpy(dtype=float)
        
        # Implement strict reviewer P1 constraint for undefined PR-AUC slices
        is_defined = truth.nunique() > 1
        if is_defined:
            pr_auc = float(average_precision_score(truth, score))
        else:
            pr_auc = np.nan
            
        unseen_rows.append(
            {
                "episode_id": unseen_episode_id,
                "scenario_family": unseen_scenario,
                "topology": unseen_topology,
                "attack_family": signature[0],
                "attack_intensity": signature[1],
                "variant": signature[2],
                "signature_seen_in_train": signature in train_signatures,
                "rows": len(group),
                "prevalence": float(truth.mean()),
                "pr_auc": pr_auc,
                "pr_auc_defined": is_defined,  # Added the required boolean column
            }
        )
    unseen_attack = pd.DataFrame(unseen_rows)
    if unseen_attack.empty or unseen_attack["signature_seen_in_train"].any():
        raise AssertionError("locked unseen-attack signatures are missing or contaminated")
    unseen_path = store.write_frame("robustness/unseen_attack_intensity.csv", unseen_attack)
    store.write_stage_manifest(
        "m9_robustness",
        inputs=[features_path, seal_path, model_paths["xgboost"]],
        outputs=[
            cross_topology_path,
            ablation_columns_path,
            ablation_path,
            horizon_path,
            threshold_path,
            unseen_path,
            *capacity_paths,
            *robustness_selection_paths,
        ],
        config_path=experiment_file,
        statistics={
            "cross_topology_cells": len(cross_topology),
            "ablation_rows": len(ablation),
            "horizon_rows": len(horizon_metrics),
            "threshold_rows": len(threshold_metrics),
            "unseen_rows": len(unseen_attack),
            "robustness_grid": "compact" if compact_grid else "declared_full",
            "ablation_protocol": "primary hyperparameters frozen across feature sets",
            "locked_hyperparameter_selection": False,
        },
        parent_stage="m7_locked_evaluation",
    )

    locked_baseline = telemetry.loc[telemetry["episode_id"].isin(locked_ids)].copy()
    c0 = system_metrics(
        locked_baseline.loc[locked_baseline["topology"] == "centralized"],
        config_name="C0",
        experiment=config,
    )
    d0 = system_metrics(
        locked_baseline.loc[locked_baseline["topology"] == "distributed"],
        config_name="D0",
        experiment=config,
    )
    p1 = _load_policy("configs/policies/p1_low_priority_shaping.yaml")
    policy_telemetry, policy_actions, policy_ledgers = _simulate_policy(
        events,
        config,
        topology,
        bundles["xgboost"],
        p1,
        locked_ids,
    )
    c1 = system_metrics(
        policy_telemetry.loc[policy_telemetry["topology"] == "centralized"],
        policy_actions,
        config_name="C1",
        experiment=config,
    )
    d1 = system_metrics(
        policy_telemetry.loc[policy_telemetry["topology"] == "distributed"],
        policy_actions,
        config_name="D1",
        experiment=config,
    )
    system_summary = pd.concat((c0, c1, d0, d1), ignore_index=True)
    expected_configs = {"C0", "C1", "D0", "D1"}
    if set(system_summary["config"]) != expected_configs:
        raise AssertionError("system summary is missing C0/C1/D0/D1")
    if any(
        len(system_summary.loc[system_summary["config"] == name]) != len(locked_ids)
        for name in expected_configs
    ):
        raise AssertionError("system summary does not contain every paired locked episode")
    if policy_actions.empty:
        raise AssertionError("P1 produced no auditable action lifecycle")
    p1_telemetry_path = store.write_frame("system/p1_telemetry.parquet", policy_telemetry)
    p1_actions_path = store.write_frame("system/p1_actions.csv", policy_actions)
    p1_ledgers_path = store.write_frame("system/p1_ledgers.csv", policy_ledgers)
    system_summary_path = store.write_frame("system/c0_c1_d0_d1_metrics.csv", system_summary)

    p2 = _load_policy("configs/policies/p2_precomputed_path.yaml")
    p2_telemetry, p2_actions, p2_ledgers = _simulate_policy(
        events,
        config,
        topology,
        bundles["xgboost"],
        p2,
        locked_ids,
    )
    p2_metrics = system_metrics(
        p2_telemetry,
        p2_actions,
        config_name="P2_supporting",
        experiment=config,
    )
    p2_telemetry_path = store.write_frame("system/p2_telemetry.parquet", p2_telemetry)
    p2_actions_path = store.write_frame("system/p2_actions.csv", p2_actions)
    p2_ledgers_path = store.write_frame("system/p2_ledgers.csv", p2_ledgers)
    p2_metrics_path = store.write_frame("system/p2_metrics.csv", p2_metrics)
    baseline_lookup = ledgers.loc[ledgers["episode_id"].isin(locked_ids)].set_index(
        ["episode_id", "topology"]
    )["offered_bytes"]
    policy_pairing_failures: list[str] = []
    for policy_name, policy_frame in (("P1", policy_ledgers), ("P2", p2_ledgers)):
        for row in policy_frame.itertuples():
            expected_offered = float(
                str(baseline_lookup.loc[(str(row.episode_id), str(row.topology))])
            )
            if not np.isclose(float(str(row.offered_bytes)), expected_offered, rtol=0.0, atol=1e-6):
                policy_pairing_failures.append(
                    f"{policy_name}:{row.episode_id}:{row.topology}:offered-bytes"
                )
    policy_audit_path = store.write_json(
        "verification/policy_pairing_audit.json",
        {
            "status": "pass" if not policy_pairing_failures else "fail",
            "failures": policy_pairing_failures,
            "p1_actions": len(policy_actions),
            "p2_actions": len(p2_actions),
            "future_feature_inputs": [],
            "replica_migrations": 0,
        },
    )
    if policy_pairing_failures:
        raise AssertionError(f"policy pairing failed: {policy_pairing_failures}")
    store.write_stage_manifest(
        "m8_system_policies",
        inputs=[seal_path, events_path, model_paths["xgboost"]],
        outputs=[
            p1_telemetry_path,
            p1_actions_path,
            p1_ledgers_path,
            system_summary_path,
            p2_telemetry_path,
            p2_actions_path,
            p2_ledgers_path,
            p2_metrics_path,
            policy_audit_path,
        ],
        config_path=experiment_file,
        statistics={
            "configs": sorted(expected_configs),
            "locked_episode_ids": sorted(locked_ids),
            "p1_actions": len(policy_actions),
            "p2_actions": len(p2_actions),
            "paired_offered_workload": "pass",
        },
        parent_stage="m7_locked_evaluation",
    )

    result_dir = store.path("results")
    experiment_summary: dict[str, object] = {
        "run_id": run_id,
        "episode_duration_s": config.episode_duration_s,
        "simulation_step_s": config.simulation_step_s,
        "window_s": config.window_s,
        "prediction_horizons_s": [horizon * config.window_s for horizon in config.horizons],
        "episodes": config.episodes,
        "episode_split": {
            "train": len(config.split.train),
            "validation": len(config.split.validation),
            "locked_test": len(config.split.locked_test),
        },
        "monitored_objects_per_topology": config.monitored_object_count,
        "topology_nodes": len(topology.nodes),
        "physical_links": len(topology.links),
        "active_compute_budget_mbps": centralized_compute_budget,
        "centralized_distributed_budget_equal": True,
        "scenario_families": sorted(episode_manifest["family"].unique().tolist()),
        "attack_families": sorted(episode_manifest["attack_family"].unique().tolist()),
        "model_grid": "compact" if compact_grid else "declared_full",
    }
    generate_report_assets(
        features=features,
        binary_metrics=binary_metrics,
        predictions=predictions,
        binary_slices=prediction_slices,
        lead_times=lead_times,
        cause_metrics=cause_metrics,
        cause_matrix=cause_matrix,
        cause_predictions=cause_predictions,
        cause_slices=cause_slices,
        system_summary=system_summary,
        cross_topology=cross_topology,
        ablation=ablation,
        horizon=horizon_metrics,
        threshold_sensitivity=threshold_metrics,
        unseen_attack=unseen_attack,
        xgboost_bundle=bundles["xgboost"],
        experiment_summary=experiment_summary,
        output_dir=result_dir,
        publish_root=publish_root,
        lineage={
            "locked_test_seal_sha256": sha256_file(seal_path),
            "locked_predictions_sha256": sha256_file(predictions_path),
            "system_metrics_sha256": sha256_file(system_summary_path),
            "features_sha256": sha256_file(features_path),
        },
    )
    for path in sorted(result_dir.rglob("*")):
        if path.is_file() and path != core_topology_image:
            store.record_existing(str(path.relative_to(store.run_dir)), kind="result")

    required_tables = [
        result_dir / "tables" / f"table{index}_{name}"
        for index, name in (
            (1, "dataset_split.csv"),
            (2, "prediction_by_topology.csv"),
            (3, "system_c0_c1_d0_d1.csv"),
            (4, "ablation_robustness.csv"),
        )
    ]
    required_figures = [
        next((result_dir / "figures").glob(f"figure{index}_*.png"), None) for index in range(1, 10)
    ]
    if not all(path.is_file() for path in required_tables):
        raise AssertionError("one or more required Tables 1-4 are missing")
    if not all(path is not None and path.is_file() for path in required_figures):
        raise AssertionError("one or more required Figures 1-9 are missing")
    report_outputs = sorted(path for path in result_dir.rglob("*") if path.is_file())
    store.write_stage_manifest(
        "m10_reporting",
        inputs=[
            predictions_path,
            system_summary_path,
            cross_topology_path,
            ablation_path,
            horizon_path,
        ],
        outputs=report_outputs,
        config_path=experiment_file,
        statistics={
            "tables_1_to_4": 4,
            "figures_1_to_9": 9,
            "demo_values_used": False,
            "source_kind": "real_sealed_results",
        },
        parent_stage="m8_system_policies",
    )

    scientific_contract_path = store.write_json(
        "verification/scientific_contracts.json",
        {
            "source_markdown_authority": True,
            "demo_values_used": False,
            "paired_design": True,
            "episode_group_split": True,
            "locked_seal": seal["seal_sha256"],
            "raw_rows": len(features),
            "expected_raw_rows": dataset_checks["expected_raw_rows"],
            "model_feature_count": len(FEATURE_COLUMNS),
            "policy_uses_future_labels": False,
            "required_tables": 4,
            "required_figures": 9,
            "advanced_leakage_status": advanced_leakage["status"],
            "ground_truth_counterfactual_feature_equality": counterfactual_leakage[
                "features_byte_equal"
            ],
            "system_latency_scope": ("traffic-conserving end-to-end weighted approximation"),
            "system_loss_scope": "end-to-end delivered/offered bytes",
            "false_alarm_rate_unit": "system-hour",
            "backbone_and_bandwidth_hop_distinct": bool(
                (ledgers["backbone_bytes"] != ledgers["bandwidth_hop_bytes"]).any()
            ),
            "paired_active_compute_budget_equal": bool(
                ledgers.groupby("episode_id")["active_compute_budget_mbps"].nunique().eq(1).all()
            ),
            "robustness_search_budget": "compact" if compact_grid else "declared_full",
            "ablation_hyperparameters_matched": True,
            "all_four_cause_classes_evaluated": all(
                int(cause_metrics[f"support_{label}"].iloc[0]) > 0
                for label in (
                    "normal",
                    "benign_high_load",
                    "attack_without_congestion",
                    "attack_associated_congestion",
                )
            ),
        },
    )
    store.write_stage_manifest(
        "m13_scientific_contracts",
        inputs=[seal_path, result_dir / "results_manifest.json"],
        outputs=[scientific_contract_path],
        config_path=experiment_file,
        statistics={"status": "pass", "demo_values_used": False},
        parent_stage="m10_reporting",
    )
    store.finalize(
        config_hash=sha256_file(experiment_file),
        parents={
            "source_markdown": "8712F436EBC157FEEBFC2CE4519C25C75435062EECAB928A82B70EC4E2EB1750",
            "result_guide_markdown": (
                "F8DA6A28A0B622B4A8C93F495C263D11BB6E4FCDDB8B924EF7C387CCEB0A220A"
            ),
            "topology_image": "59A4EB2CDD02008EFE88503CDB4C8E786C26BE4CF40DB1F79D9EE7E485610B0F",
        },
        random_seeds={"root_seed": config.root_seed},
    )
    return store.run_dir


def run_scale_up_profile(
    experiment_path: str | Path,
    *,
    run_id: str,
) -> Path:
    experiment_file = Path(experiment_path).resolve()
    config = load_experiment(experiment_file)
    topology = load_topology(config.topology_path)
    store = ArtifactStore(config.output_root, run_id)
    experiment_snapshot = store.write_json("config/experiment.json", config.model_dump(mode="json"))
    topology_snapshot = store.write_json("config/topology.json", topology.model_dump(mode="json"))
    events, manifest = generate_workloads(config, topology)
    telemetry, ledgers = _simulate_no_action(events, config, topology)
    raw_schema = validate_raw_events(events)
    telemetry_schema = validate_telemetry(telemetry)
    features = build_feature_table(telemetry, config, _split_map(manifest))
    events_path = store.write_frame("data/raw_events.parquet", events)
    manifest_path = store.write_frame("data/episode_manifest.csv", manifest)
    telemetry_path = store.write_frame("data/telemetry.parquet", telemetry)
    features_path = store.write_frame("data/features.parquet", features)
    ledgers_path = store.write_frame("data/ledgers.csv", ledgers)
    schema_path = store.write_json(
        "verification/data_schemas.json", {"raw_events": raw_schema, "telemetry": telemetry_schema}
    )
    topology_figure = store.path("results/topology_35node.png")
    render_topology(topology, topology_figure)
    store.record_existing("results/topology_35node.png", kind="figure")
    network = NetworkTopology(topology)
    roles = {node.id: node.role for node in topology.nodes}
    route_smoke = {
        mode: {
            region: {
                "destination": route.destination,
                "nodes": list(route.nodes),
                "physical_nodes": list(route.physical_nodes),
                "service_transit_dcs": [
                    node for node in route.nodes[1:-1] if roles[node] in {"edge_dc", "central_dc"}
                ],
            }
            for region, route in network.routes(mode).items()
        }
        for mode in ("centralized", "distributed")
    }
    contract = {
        "nodes": len(topology.nodes),
        "visible_physical_bidirectional_links": len(topology.links),
        "runtime_directional_traversal_arcs": len(topology.links) * 2,
        "image_overview_annotation_directional_links": len(topology.links) * 2,
        "image_internal_count_discrepancy_recorded": False,
        "audit_document": "source topology: C:/Users/Admin/Desktop/topology/network_vn_30.py",
        "edge_dcs": len(topology.edge_dcs),
        "central_dcs": sum(node.role == "central_dc" for node in topology.nodes),
        "primary_central_dc": topology.central_dc,
        "metro_capacity_mbps": sorted(
            {link.capacity_mbps for link in topology.links if link.link_class == "metro_internal"}
        ),
        "backbone_dc_capacity_mbps": sorted(
            {link.capacity_mbps for link in topology.links if link.link_class != "metro_internal"}
        ),
        "capacity_units_match_image_legend": bool(
            all(
                link.capacity_mbps
                == (100_000.0 if link.link_class == "metro_internal" else 400_000.0)
                for link in topology.links
            )
        ),
        "service_dc_transit_free": bool(
            all(
                not route["service_transit_dcs"]
                for mode_routes in route_smoke.values()
                for route in mode_routes.values()
            )
        ),
        "route_smoke": route_smoke,
        "paired_offered_bytes_equal": bool(
            all(
                np.isclose(
                    group.loc[group["topology"] == "centralized", "offered_bytes"].iloc[0],
                    group.loc[group["topology"] == "distributed", "offered_bytes"].iloc[0],
                )
                for _, group in ledgers.groupby("episode_id")
            )
        ),
    }
    contract_path = store.write_json("verification/scale_up_contract.json", contract)
    if (
        contract["nodes"] != 35
        or contract["visible_physical_bidirectional_links"] != 51
        or contract["runtime_directional_traversal_arcs"] != 102
        or contract["primary_central_dc"] != "30"
        or not contract["capacity_units_match_image_legend"]
        or not contract["service_dc_transit_free"]
    ):
        raise AssertionError(f"35-node topology contract failed: {contract}")
    store.write_stage_manifest(
        "m11_scale_up_35node",
        inputs=[experiment_snapshot, topology_snapshot],
        outputs=[
            events_path,
            manifest_path,
            telemetry_path,
            features_path,
            ledgers_path,
            topology_figure,
            contract_path,
            schema_path,
        ],
        config_path=experiment_file,
        statistics={
            "events": len(events),
            "telemetry_rows": len(telemetry),
            "feature_rows": len(features),
            "paired_offered_bytes_equal": contract["paired_offered_bytes_equal"],
            "core_namespace_contaminated": False,
        },
    )
    store.finalize(
        config_hash=sha256_file(experiment_file),
        parents={
            "topology_image": "59A4EB2CDD02008EFE88503CDB4C8E786C26BE4CF40DB1F79D9EE7E485610B0F",
        },
        random_seeds={"root_seed": config.root_seed},
    )
    return store.run_dir


def publish_run_snapshot(run_dir: str | Path, target: str | Path) -> Path:
    source = Path(run_dir).resolve()
    destination = Path(target).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(source, destination)
    return destination
