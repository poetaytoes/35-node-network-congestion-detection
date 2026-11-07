"""Independently audit the final sealed experiment and release artifacts.

The audit is intentionally read-only except for its JSON report.  It reopens the
published files instead of trusting console output from the pipeline run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from dataclasses import dataclass, field
from email.parser import BytesParser
from pathlib import Path
from typing import Any, cast

import pandas as pd

from congestion_lab.data.artifacts import verify_manifest

SOURCE_MD = (
    "docs/Machine Learning-Driven Congestion Prediction for Resource-Efficient Centralized "
    "and Distributed 5G Data Center Models.md"
)
RESULT_GUIDE_MD = (
    "docs/Goi y ket qua Machine Learning-Driven Congestion Prediction for Resource-Efficient "
    "Centralized and Distributed 5G Data Center Models.md"
)
SOURCE_HASHES = {
    SOURCE_MD: ("8712f436ebc157feebfc2ce4519c25c75435062eecab928a82b70ec4e2eb1750"),
    RESULT_GUIDE_MD: ("f8da6a28a0b622b4a8c93f495c263d11bb6e4fcddb8b924ef7c387cceb0a220a"),
    "docs/Mo hinh.jpg": "59a4eb2cdd02008efe88503cdb4c8e786c26be4cf40db1f79d9ee7e485610b0f",
}

REQUIRED_BINARY_MODELS = {
    "threshold",
    "persistence",
    "ewma",
    "logistic_regression",
    "random_forest",
    "xgboost",
}
REQUIRED_CAUSE_MODELS = {"cause_logistic", "cause_random_forest", "cause_xgboost"}
REQUIRED_CAUSES = {
    "normal",
    "benign_high_load",
    "attack_without_congestion",
    "attack_induced_congestion",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


@dataclass
class Audit:
    checks: list[dict[str, Any]] = field(default_factory=list)

    def check(self, name: str, passed: bool, **evidence: Any) -> None:
        self.checks.append({"name": name, "status": "pass" if passed else "fail", **evidence})

    @property
    def failures(self) -> list[str]:
        return [str(item["name"]) for item in self.checks if item["status"] != "pass"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--core-pilot-run", type=Path, required=True)
    parser.add_argument("--core-run", type=Path, required=True)
    parser.add_argument("--published-results", type=Path, required=True)
    parser.add_argument("--external-run", type=Path, required=True)
    parser.add_argument("--scaleup-pilot-run", type=Path, required=True)
    parser.add_argument("--scaleup-declared-run", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def resolve(workspace: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (workspace / path).resolve()


def audit_sources(audit: Audit, workspace: Path) -> None:
    observed = {relative: sha256(workspace / relative) for relative in SOURCE_HASHES}
    audit.check(
        "M13 source authority SHA-256",
        observed == SOURCE_HASHES,
        expected=SOURCE_HASHES,
        observed=observed,
    )


def audit_core_pilot(audit: Audit, pilot: Path) -> None:
    """Reopen the two-episode pilot and enforce its pre-full-run sanity gate."""
    manifest = pd.read_csv(pilot / "episode_manifest.csv")
    raw_events = pd.read_parquet(pilot / "raw_events.parquet")
    telemetry = pd.read_parquet(pilot / "telemetry.parquet")
    features = pd.read_parquet(pilot / "features_5s.parquet")
    ledgers = pd.read_csv(pilot / "ledgers.csv")
    leakage = read_json(pilot / "leakage_audit.json")

    topologies_per_episode = ledgers.groupby("episode_id")["topology"].apply(set)
    paired_offered_counts = ledgers.groupby("episode_id")["network_offered_bytes"].nunique()
    paired_compute_counts = ledgers.groupby("episode_id")["active_compute_budget_mbps"].nunique()
    conservation_residual = (
        ledgers["network_offered_bytes"]
        - ledgers["e2e_delivered_bytes"]
        - ledgers["e2e_loss_bytes"]
    ).abs()
    physical_bounds = (
        telemetry.select_dtypes(include="number").notna().all().all()
        and telemetry["utilization"].between(0, 1).all()
        and telemetry["queue_ratio"].between(0, 1).all()
        and telemetry["loss_rate"].between(0, 1).all()
        and telemetry["delay_ms"].ge(0).all()
    )
    expected_feature_rows = 2 * 2 * (120 // 5) * 8

    audit.check(
        "M2-M4 two-episode core pilot sanity gate",
        len(manifest) == 2
        and set(manifest["episode_id"]) == {"P01", "P02"}
        and set(manifest["split"]) == {"train", "validation"}
        and int(manifest["event_count"].sum()) == len(raw_events)
        and len(telemetry) == 2 * 2 * 120 * 8
        and len(features) == expected_feature_rows
        and all(value == {"centralized", "distributed"} for value in topologies_per_episode)
        and bool(paired_offered_counts.eq(1).all())
        and bool(paired_compute_counts.eq(1).all())
        and bool(ledgers["compute_budget_fair"].all())
        and float(conservation_residual.max()) < 1e-3
        and bool(physical_bounds)
        and bool(features["congestion_state"].any())
        and leakage["status"] == "pass"
        and not leakage["blocked"],
        pilot=str(pilot),
        episode_ids=sorted(manifest["episode_id"].tolist()),
        raw_event_rows=len(raw_events),
        telemetry_rows=len(telemetry),
        feature_rows=len(features),
        paired_topologies={key: sorted(value) for key, value in topologies_per_episode.items()},
        maximum_conservation_residual_bytes=float(conservation_residual.max()),
        physical_bounds=bool(physical_bounds),
        observed_congestion=bool(features["congestion_state"].any()),
        leakage=leakage,
    )


def audit_run_manifest(audit: Audit, name: str, run_dir: Path) -> None:
    failures = verify_manifest(run_dir / "run_manifest.json")
    audit.check(f"{name} manifest hash chain", not failures, failures=failures)


def audit_core(audit: Audit, core: Path, published: Path) -> dict[str, Any]:
    audit_run_manifest(audit, "core sealed run", core)

    stage_names = {path.stem for path in (core / "manifests").glob("*.json")}
    required_stages = {
        "m0_m1_topology",
        "m2_traffic",
        "m3_simulation",
        "m4_m5_features_split",
        "m6_models",
        "m7_locked_evaluation",
        "m8_system_policies",
        "m9_robustness",
        "m10_reporting",
        "m13_scientific_contracts",
    }
    audit.check(
        "M0-M10 and M13 immutable stage manifests",
        required_stages <= stage_names,
        required=sorted(required_stages),
        observed=sorted(stage_names),
    )

    dataset = read_json(core / "verification" / "dataset_checks.json")
    audit.check(
        "M4 raw and eligible row contracts",
        dataset["actual_raw_rows"] == dataset["expected_raw_rows"] == 41472
        and dataset["eligible_h1_rows"] < 41472
        and dataset["eligible_h2_rows"] < dataset["eligible_h1_rows"]
        and dataset["eligible_h4_rows"] < dataset["eligible_h2_rows"],
        raw_rows=dataset["actual_raw_rows"],
        eligible_h1_rows=dataset["eligible_h1_rows"],
        eligible_h2_rows=dataset["eligible_h2_rows"],
        eligible_h4_rows=dataset["eligible_h4_rows"],
    )
    audit.check(
        "M5 episode split and leakage",
        dataset["split_episode_counts"] == {"train": 10, "validation": 4, "locked_test": 4}
        and not dataset["split_overlap"]
        and dataset["paired_rows_equal"]
        and dataset["leakage"]["status"] == "pass"
        and not dataset["leakage"]["blocked"],
        split_episode_counts=dataset["split_episode_counts"],
        leakage=dataset["leakage"],
    )
    all_split_causes = {split: set(causes) for split, causes in dataset["causes_by_split"].items()}
    audit.check(
        "M4 four cause labels in every split",
        dataset["all_four_causes_in_every_split"]
        and all(causes == REQUIRED_CAUSES for causes in all_split_causes.values()),
        causes_by_split={key: sorted(value) for key, value in all_split_causes.items()},
    )

    paired = read_json(core / "data" / "paired_audit.json")
    audit.check(
        "M2/M3 paired offered workload",
        paired["status"] == "pass"
        and paired["episodes"] == 18
        and paired["all_event_counts_equal"]
        and paired["all_offered_bytes_equal"],
        evidence=paired,
    )

    resource_start = read_json(core / "data" / "resource_ledger_start.json")
    resource_ledger = read_json(core / "data" / "resource_ledger_end.json")
    resource_rows = pd.DataFrame(resource_ledger["rows"])
    paired_compute_counts = resource_rows.groupby("episode_id")[
        "active_compute_budget_mbps"
    ].nunique()
    audit.check(
        "M3 traffic conservation and fair active compute budget",
        resource_ledger["status"] == "pass"
        and not resource_ledger["paired_failures"]
        and resource_start["compute_budget_fair"]
        and set(resource_start["active_compute_budget_mbps"].values()) == {360.0}
        and bool(resource_rows["compute_budget_fair"].all())
        and bool(paired_compute_counts.eq(1).all())
        and bool(
            (
                resource_rows["e2e_delivered_bytes"]
                <= resource_rows["network_offered_bytes"] + 1e-6
            ).all()
        )
        and bool((resource_rows["e2e_loss_bytes"] >= 0).all())
        and bool((resource_rows["backbone_bytes"] != resource_rows["bandwidth_hop_bytes"]).any()),
        active_compute_budgets=sorted(
            resource_rows["active_compute_budget_mbps"].unique().tolist()
        ),
        paired_compute_counts=paired_compute_counts.to_dict(),
        backbone_distinct_from_byte_hop=bool(
            (resource_rows["backbone_bytes"] != resource_rows["bandwidth_hop_bytes"]).any()
        ),
    )

    schemas = {
        "raw_events": read_json(core / "verification" / "raw_event_schema.json"),
        "telemetry": read_json(core / "verification" / "telemetry_schema.json"),
    }
    audit.check(
        "M2/M3 typed data schemas",
        all(value["status"] == "pass" and not value["violations"] for value in schemas.values()),
        schemas=schemas,
    )

    feature_dictionary = pd.read_csv(core / "data" / "feature_dictionary.csv")
    dictionary_ok = (
        len(feature_dictionary) == 40
        and feature_dictionary["unit"].fillna("").str.len().gt(0).all()
        and feature_dictionary["formula"].fillna("").str.len().gt(0).all()
        and feature_dictionary["causal"].astype(bool).all()
    )
    audit.check(
        "M4 causal feature dictionary",
        bool(dictionary_ok),
        rows=len(feature_dictionary),
        columns=list(feature_dictionary.columns),
    )

    leakage = read_json(core / "verification" / "advanced_leakage_audit.json")
    audit.check(
        "M5 advanced leakage and shuffle-label audit",
        leakage["status"] == "pass"
        and not leakage["blocking_issues"]
        and not leakage["fit_scope_issues"]
        and leakage["shuffle_label"]["status"] == "pass",
        evidence=leakage,
    )
    counterfactual = read_json(core / "verification" / "ground_truth_counterfactual_audit.json")
    audit.check(
        "M5 attack-ground-truth counterfactual feature equality",
        counterfactual["status"] == "pass"
        and counterfactual["features_byte_equal"]
        and counterfactual["label_metadata_changed"]
        and counterfactual["feature_columns"] == 40
        and counterfactual["maximum_absolute_difference"] == 0.0,
        evidence=counterfactual,
    )

    seal = read_json(core / "seals" / "locked_test_seal.json")
    audit.check(
        "M5 locked-test seal",
        seal["kind"] == "locked_test_seal_v1"
        and set(seal["models"]) == REQUIRED_BINARY_MODELS | REQUIRED_CAUSE_MODELS
        and len(seal["seal_sha256"]) == 64,
        seal_sha256=seal["seal_sha256"],
        sealed_models=sorted(seal["models"]),
    )

    reloads = read_json(core / "verification" / "model_reload_checks.json")
    audit.check(
        "M6 model save/load score equality",
        set(reloads) == REQUIRED_BINARY_MODELS | REQUIRED_CAUSE_MODELS and all(reloads.values()),
        reload_checks=reloads,
    )

    binary = pd.read_csv(core / "evaluation" / "binary_metrics_locked.csv")
    binary_models = set(binary["model"])
    binary_topologies = set(binary["topology"])
    audit.check(
        "M6/M7 sealed binary model evaluation",
        binary_models >= REQUIRED_BINARY_MODELS
        and binary_topologies == {"centralized", "distributed"}
        and len(binary) == len(REQUIRED_BINARY_MODELS) * 2,
        models=sorted(binary_models),
        topologies=sorted(binary_topologies),
        rows=len(binary),
    )
    expected_false_alarm_rate = binary["false_alarm_events"].astype(float) / binary[
        "evaluation_hours"
    ].astype(float)
    false_alarm_math = bool(
        (binary["false_alarms_per_hour"].astype(float) - expected_false_alarm_rate)
        .abs()
        .le(1e-12)
        .all()
    )
    audit.check(
        "M7 false-alarm denominator is unique system-hour",
        false_alarm_math
        and bool((binary["evaluation_hours"] > 0).all())
        and bool((binary["object_hours"] > binary["evaluation_hours"]).all())
        and bool((binary["false_alarms_per_object_hour"] <= binary["false_alarms_per_hour"]).all()),
        evaluation_hours=sorted(binary["evaluation_hours"].unique().tolist()),
        object_hours=sorted(binary["object_hours"].unique().tolist()),
        formula="false_alarm_events / unique episode-window system hours",
    )
    binary_slices = pd.read_csv(core / "evaluation" / "binary_metrics_slices.csv")
    zero_positive_slices = binary_slices.loc[binary_slices["positive_support"] == 0]
    supported_slices = binary_slices.loc[binary_slices["positive_support"] > 0]
    audit.check(
        "M7 undefined PR-AUC slices remain explicit",
        len(zero_positive_slices) > 0
        and not bool(zero_positive_slices["pr_auc_defined"].any())
        and bool(zero_positive_slices["pr_auc"].isna().all())
        and bool(supported_slices["pr_auc_defined"].all())
        and bool(supported_slices["pr_auc"].between(0, 1).all()),
        undefined_rows=len(zero_positive_slices),
        supported_rows=len(supported_slices),
    )

    cause = pd.read_csv(core / "evaluation" / "cause_metrics_locked.csv")
    support_columns = [f"support_{label}" for label in sorted(REQUIRED_CAUSES)]
    audit.check(
        "M7 four-class cause evaluation",
        len(cause) >= 1
        and set(support_columns) <= set(cause.columns)
        and all(
            {f"precision_{label}", f"recall_{label}", f"f1_{label}"} <= set(cause.columns)
            for label in REQUIRED_CAUSES
        )
        and bool((cause[support_columns] > 0).all().all()),
        rows=len(cause),
        support={column: int(cause[column].sum()) for column in support_columns},
    )
    cause_slices = pd.read_csv(core / "evaluation" / "cause_metrics_slices.csv")
    audit.check(
        "M7 cause topology/scenario slices",
        {"topology", "scenario_family"} <= set(cause_slices["slice_dimension"])
        and {"centralized", "distributed"}
        <= set(cause_slices.loc[cause_slices["slice_dimension"] == "topology", "slice_value"]),
        dimensions=sorted(cause_slices["slice_dimension"].unique().tolist()),
        rows=len(cause_slices),
    )

    system = pd.read_csv(core / "system" / "c0_c1_d0_d1_metrics.csv")
    policies = read_json(core / "verification" / "policy_pairing_audit.json")
    required_system_columns = {
        "mean_latency_ms",
        "p95_latency_ms",
        "loss_percent",
        "backbone_traffic_gb",
        "bandwidth_hop_gb_hop",
        "accepted_requests",
        "rejected_requests",
        "shaped_percent",
    }
    audit.check(
        "M8 C0/C1/D0/D1 and policy lifecycle",
        set(system["config"]) == {"C0", "C1", "D0", "D1"}
        and len(system) == 16
        and required_system_columns <= set(system.columns)
        and bool((system["accepted_requests"] >= 0).all())
        and bool((system["rejected_requests"] >= 0).all())
        and bool((system["bandwidth_hop_gb_hop"] != system["backbone_traffic_gb"]).any())
        and policies["status"] == "pass"
        and policies["p1_actions"] > 0
        and policies["p2_actions"] > 0
        and policies["replica_migrations"] == 0
        and not policies["future_feature_inputs"],
        configs=sorted(system["config"].unique()),
        system_columns=sorted(required_system_columns),
        policy_audit=policies,
    )

    cross = pd.read_csv(core / "robustness" / "cross_topology.csv")
    ablation = pd.read_csv(core / "robustness" / "ablation_summary.csv")
    horizons = pd.read_csv(core / "robustness" / "horizon_sensitivity.csv")
    thresholds = pd.read_csv(core / "robustness" / "congestion_threshold_sensitivity.csv")
    unseen = pd.read_csv(core / "robustness" / "unseen_attack_intensity.csv")
    capacity_minus = pd.read_csv(core / "robustness" / "capacity_minus_20pct_metrics.csv")
    capacity_plus = pd.read_csv(core / "robustness" / "capacity_plus_20pct_metrics.csv")
    full_grid_selection_files = [
        "model_selection.csv",
        "centralized_selection.csv",
        "distributed_selection.csv",
        "horizon_h1_selection.csv",
        "horizon_h4_selection.csv",
        "congestion_threshold_0.7_selection.csv",
        "congestion_threshold_0.8_selection.csv",
        "congestion_threshold_0.9_selection.csv",
    ]
    full_grid_counts: dict[str, int] = {}
    full_grid_profiles: dict[str, str] = {}
    for name in full_grid_selection_files:
        selection = pd.read_csv(core / "models" / name)
        xgboost_rows = selection.loc[selection["model"] == "xgboost"]
        full_grid_counts[name] = int(xgboost_rows["candidate_count"].iloc[0])
        full_grid_profiles[name] = str(xgboost_rows["grid_profile"].iloc[0])
    ablation_selection_files = [
        "no_lag_rolling_selection.csv",
        "no_queue_selection.csv",
        "no_security_features_selection.csv",
    ]
    ablation_candidate_counts = {
        name: int(pd.read_csv(core / "models" / name)["candidate_count"].iloc[0])
        for name in ablation_selection_files
    }
    primary_structural_hash = str(
        ablation.loc[ablation["experiment"] == "full_xgboost", "parameters_sha256"].iloc[0]
    )
    ablation_structural_hashes = set(
        ablation.loc[
            ablation["experiment"].isin({"no_lag_rolling", "no_queue", "no_security_features"}),
            "parameters_sha256",
        ]
    )
    robustness_ok = (
        set(cross["train_scope"]) == {"centralized", "distributed", "combined"}
        and set(cross["test_topology"]) == {"centralized", "distributed"}
        and {
            "full_xgboost",
            "no_lag_rolling",
            "no_queue",
            "no_security_features",
            "capacity_minus_20pct",
            "capacity_plus_20pct",
        }
        <= set(ablation["experiment"])
        and set(horizons["horizon"]) == {1, 2, 4}
        and set(thresholds["congestion_threshold"]) == {0.7, 0.8, 0.9}
        and len(unseen) > 0
        and bool((~unseen["signature_seen_in_train"]).all())
        and set(capacity_minus["capacity_scale"]) == {0.8}
        and set(capacity_plus["capacity_scale"]) == {1.2}
        and set(full_grid_counts.values()) == {16}
        and set(full_grid_profiles.values()) == {"declared_full"}
        and set(ablation_candidate_counts.values()) == {1}
        and ablation_structural_hashes == {primary_structural_hash}
    )
    audit.check(
        "M9 robustness matrix",
        robustness_ok,
        cross_topology_rows=len(cross),
        ablations=sorted(ablation["experiment"].unique()),
        horizons_s=sorted((horizons["horizon"] * 5).unique().tolist()),
        thresholds=sorted(thresholds["congestion_threshold"].unique().tolist()),
        unseen_rows=len(unseen),
        capacity_scales=[0.8, 1.2],
        full_grid_candidate_counts=full_grid_counts,
        full_grid_profiles=full_grid_profiles,
        ablation_candidate_counts=ablation_candidate_counts,
        ablation_structural_parameters_match_primary=(
            ablation_structural_hashes == {primary_structural_hash}
        ),
    )

    report_manifest = read_json(published / "results_manifest.json")
    figure_names = {path.name for path in (published / "figures").glob("figure*.png")}
    figure_numbers = {
        int(path.name.split("_")[0].removeprefix("figure"))
        for path in (published / "figures").glob("figure*.png")
    }
    table_numbers = {
        int(path.name.split("_")[0].removeprefix("table"))
        for path in (published / "tables").glob("table[1-4]_*.csv")
    }
    report_hash_failures: list[str] = []
    for item in report_manifest["files"]:
        candidate = published / item["path"]
        if not candidate.is_file() or candidate.stat().st_size != item["bytes"]:
            report_hash_failures.append(item["path"] + ":missing-or-size")
        elif sha256(candidate) != item["sha256"]:
            report_hash_failures.append(item["path"] + ":sha256")
        run_candidate = core / "results" / item["path"]
        if not run_candidate.is_file() or sha256(run_candidate) != item["sha256"]:
            report_hash_failures.append(item["path"] + ":run-publish-mismatch")
    table3 = pd.read_csv(published / "tables" / "table3_system_c0_c1_d0_d1.csv")
    results_readme = (published / "RESULTS_README.md").read_text(encoding="utf-8")
    metric_semantics = report_manifest.get("metric_semantics", {})
    audit.check(
        "M10 real sealed Tables 1-4 and Figures 1-9",
        report_manifest["kind"] == "real_sealed_results"
        and report_manifest["demo_values_used"] is False
        and figure_numbers == set(range(1, 10))
        and table_numbers == set(range(1, 5))
        and required_system_columns <= set(table3.columns)
        and {"RQ1", "RQ2", "RQ3", "RQ4", "RQ5"}
        <= {token for token in ("RQ1", "RQ2", "RQ3", "RQ4", "RQ5") if token in results_readme}
        and metric_semantics.get("false_alarm_rate") == "events per elapsed topology system-hour"
        and metric_semantics.get("backbone_traffic") == "unique first-backbone-ingress bytes"
        and metric_semantics.get("bandwidth_hop") == "sum of admitted bytes over physical hops"
        and not report_hash_failures,
        figures=sorted(figure_names),
        figure_numbers=sorted(figure_numbers),
        table_numbers=sorted(table_numbers),
        manifest_files=len(report_manifest["files"]),
        table3_columns=sorted(table3.columns.tolist()),
        metric_semantics=metric_semantics,
        hash_failures=report_hash_failures,
        lineage=report_manifest["lineage"],
    )

    contracts = read_json(core / "verification" / "scientific_contracts.json")
    audit.check(
        "M13 scientific contracts",
        contracts["advanced_leakage_status"] == "pass"
        and contracts["demo_values_used"] is False
        and contracts["paired_design"]
        and not contracts["policy_uses_future_labels"]
        and contracts["ground_truth_counterfactual_feature_equality"]
        and contracts["false_alarm_rate_unit"] == "system-hour"
        and contracts["backbone_and_bandwidth_hop_distinct"]
        and contracts["paired_active_compute_budget_equal"]
        and contracts["robustness_search_budget"] == "declared_full"
        and contracts["ablation_hyperparameters_matched"]
        and contracts["required_figures"] == 9
        and contracts["required_tables"] == 4,
        evidence=contracts,
    )

    means = binary.groupby("model", as_index=True)["pr_auc"].mean().sort_values(ascending=False)
    xgboost_mean = float(means["xgboost"])
    comparator = means.drop("xgboost").index[0]
    comparator_mean = float(means.drop("xgboost").iloc[0])
    negative_result = xgboost_mean <= comparator_mean
    audit.check(
        "Scientific result reported without forcing XGBoost superiority",
        math.isfinite(xgboost_mean)
        and math.isfinite(comparator_mean)
        and 0.0 <= xgboost_mean <= 1.0
        and 0.0 <= comparator_mean <= 1.0,
        hypothesis_supported=not negative_result,
        observed_result="supported" if not negative_result else "not_supported",
        xgboost_mean_locked_pr_auc=xgboost_mean,
        strongest_non_xgboost_model=str(comparator),
        strongest_non_xgboost_mean_locked_pr_auc=comparator_mean,
    )
    return {
        "hypothesis_supported": not negative_result,
        "xgboost_mean_locked_pr_auc": xgboost_mean,
        "strongest_non_xgboost_model": str(comparator),
        "strongest_non_xgboost_mean_locked_pr_auc": comparator_mean,
    }


def audit_m11(audit: Audit, external: Path, pilot: Path, declared: Path) -> None:
    for name, directory in (
        ("M11 external profile", external),
        ("M11 30-node pilot", pilot),
        ("M11 30-node declared", declared),
    ):
        audit_run_manifest(audit, name, directory)

    external_contract = read_json(external / "verification" / "external_replay_contract.json")
    audit.check(
        "M11 external privacy-safe replay",
        external_contract["status"] == "pass"
        and external_contract["event_rows"] == 256
        and external_contract["paired_offered_bytes_equal"]
        and external_contract["privacy"]["status"] == "pass"
        and external_contract["privacy"]["training_use"] is False
        and not external_contract["privacy"]["identity_columns_in_events"]
        and not external_contract["privacy"]["identity_or_source_metadata_in_model_features"]
        and external_contract["frozen_model_evaluated"]
        and external_contract["external_model_metric_rows"] == 2
        and external_contract["external_truth_scope"] == "unlabeled benign aggregate profile"
        and len(external_contract["frozen_model_sha256"]) == 64,
        evidence=external_contract,
    )

    contracts = {
        "pilot": read_json(pilot / "verification" / "scale_up_contract.json"),
        "declared": read_json(declared / "verification" / "scale_up_contract.json"),
    }
    topology_ok = all(
        contract["nodes"] == 30
        and contract["visible_physical_bidirectional_links"] == 42
        and contract["runtime_directional_traversal_arcs"] == 84
        and contract["image_overview_annotation_directional_links"] == 120
        and contract["image_internal_count_discrepancy_recorded"]
        and contract["paired_offered_bytes_equal"]
        and contract["primary_central_dc"] == "30"
        and contract["metro_capacity_mbps"] == [100000.0]
        and contract["backbone_dc_capacity_mbps"] == [400000.0]
        and contract["capacity_units_match_image_legend"]
        and contract["service_dc_transit_free"]
        for contract in contracts.values()
    )
    audit.check(
        "M11 separate 30-node pilot and declared profiles",
        topology_ok,
        contracts=contracts,
    )


def audit_packaging(audit: Audit, workspace: Path, wheel: Path) -> None:
    lock_hashes = {
        "requirements.lock": sha256(workspace / "requirements.lock"),
        "requirements-dev.lock": sha256(workspace / "requirements-dev.lock"),
    }
    runtime_header = (workspace / "requirements.lock").read_text(encoding="utf-8")[:1000]
    dev_header = (workspace / "requirements-dev.lock").read_text(encoding="utf-8")[:1000]
    audit.check(
        "M12 Python 3.11 compiled hash locks",
        "Python 3.11" in runtime_header and "Python 3.11" in dev_header,
        lock_sha256=lock_hashes,
    )

    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(name for name in archive.namelist() if name.endswith("METADATA"))
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
    requirements = metadata.get_all("Requires-Dist", [])
    audit.check(
        "M12 Python 3.11+ wheel metadata",
        metadata["Requires-Python"] == ">=3.11"
        and any(item.startswith("numpy<2.5") for item in requirements)
        and any(item.startswith("scipy<1.18") for item in requirements),
        wheel=str(wheel),
        wheel_sha256=sha256(wheel),
        requires_python=metadata["Requires-Python"],
        requirements=requirements,
    )

    qa_logs = {
        "ruff_check": workspace / "artifacts" / "qa" / "ruff_check.log",
        "ruff_format": workspace / "artifacts" / "qa" / "ruff_format.log",
        "mypy": workspace / "artifacts" / "qa" / "mypy.log",
        "pytest_cov": workspace / "artifacts" / "qa" / "pytest_cov.log",
        "build": workspace / "artifacts" / "qa" / "build.log",
        "clean_wheel_py311": workspace / "artifacts" / "qa" / "clean_wheel_smoke_py311_pass.log",
        "clean_editable_py311": workspace / "artifacts" / "qa" / "clean_editable_py311.log",
    }
    success_markers = {
        "ruff_check": "All checks passed!",
        "ruff_format": "files already formatted",
        "mypy": "Success: no issues found",
        "pytest_cov": "passed in",
        "build": "Successfully built congestion_lab-0.1.0.tar.gz",
        "clean_wheel_py311": "EXIT_STATUS=0",
        "clean_editable_py311": "EXIT_STATUS=0",
    }
    qa_status = {}
    for name, path in qa_logs.items():
        content = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        qa_status[name] = success_markers[name] in content
    audit.check(
        "M0/M12/M13 clean install and quality evidence",
        all(qa_status.values()),
        logs=qa_status,
    )


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    core_pilot = resolve(workspace, args.core_pilot_run)
    core = resolve(workspace, args.core_run)
    published = resolve(workspace, args.published_results)
    external = resolve(workspace, args.external_run)
    pilot = resolve(workspace, args.scaleup_pilot_run)
    declared = resolve(workspace, args.scaleup_declared_run)
    wheel = resolve(workspace, args.wheel)
    output = resolve(workspace, args.output)

    audit = Audit()
    audit_sources(audit, workspace)
    audit_core_pilot(audit, core_pilot)
    result_statement = audit_core(audit, core, published)
    audit_m11(audit, external, pilot, declared)
    audit_packaging(audit, workspace, wheel)
    audit.check("Workspace intentionally has no Git repository", not (workspace / ".git").exists())

    report = {
        "schema_version": "final-audit-v1",
        "status": "pass" if not audit.failures else "fail",
        "workspace": str(workspace),
        "core_pilot_run": str(core_pilot),
        "core_run": str(core),
        "published_results": str(published),
        "external_run": str(external),
        "scaleup_pilot_run": str(pilot),
        "scaleup_declared_run": str(declared),
        "scientific_result": result_statement,
        "checks": audit.checks,
        "failures": audit.failures,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "checks": len(audit.checks),
                "failures": audit.failures,
                "output": str(output),
                "output_sha256": sha256(output),
                "scientific_result": result_statement,
            },
            indent=2,
        )
    )
    return 0 if not audit.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
