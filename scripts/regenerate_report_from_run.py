"""Regenerate print-ready report assets from immutable artifacts of a completed run.

This avoids rerunning simulation, training, or locked inference when only report
presentation code has changed.  It deliberately writes to a new output folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from congestion_lab.config.models import TopologyConfig
from congestion_lab.models import load_bundle
from congestion_lab.reporting import generate_report_assets, render_topology


def _read_design_summary(path: Path) -> dict[str, object]:
    """Restore the report summary values from the original companion table."""

    summary: dict[str, object] = {}
    for row in pd.read_csv(path).itertuples(index=False):
        value = str(row.value)
        try:
            summary[str(row.parameter)] = json.loads(value)
        except json.JSONDecodeError:
            summary[str(row.parameter)] = value
    return summary


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required run artifact is missing: {path}")
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Immutable run directory, including a run interrupted during reporting",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for regenerated report assets",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    run_manifest = run_dir / "run_manifest.json"
    recovery_mode = not run_manifest.is_file()
    if recovery_mode and not (run_dir / "config" / "experiment.json").is_file():
        raise FileNotFoundError(f"Not a recoverable run directory: {run_dir}")
    if output_dir == run_dir / "results":
        raise ValueError(
            "Choose a new output directory; do not overwrite the immutable run results."
        )

    evaluation = run_dir / "evaluation"
    robustness = run_dir / "robustness"
    system = run_dir / "system"
    source_results = run_dir / "results"
    xgboost_bundle = load_bundle(run_dir / "models" / "xgboost.joblib")

    cause_matrix = _read_csv(evaluation / "cause_confusion_matrix.csv").set_index("true_label")
    generate_report_assets(
        features=pd.read_parquet(run_dir / "data" / "features_5s.parquet"),
        binary_metrics=_read_csv(evaluation / "binary_metrics_locked.csv"),
        predictions=pd.read_parquet(evaluation / "locked_predictions.parquet"),
        binary_slices=_read_csv(evaluation / "binary_metrics_slices.csv"),
        lead_times=_read_csv(evaluation / "lead_times.csv"),
        cause_metrics=_read_csv(evaluation / "cause_metrics_locked.csv"),
        cause_matrix=cause_matrix,
        cause_predictions=pd.read_parquet(evaluation / "cause_predictions_locked.parquet"),
        cause_slices=_read_csv(evaluation / "cause_metrics_slices.csv"),
        system_summary=_read_csv(system / "c0_c1_d0_d1_metrics.csv"),
        cross_topology=_read_csv(robustness / "cross_topology.csv"),
        ablation=_read_csv(robustness / "ablation_summary.csv"),
        horizon=_read_csv(robustness / "horizon_sensitivity.csv"),
        threshold_sensitivity=_read_csv(robustness / "congestion_threshold_sensitivity.csv"),
        unseen_attack=_read_csv(robustness / "unseen_attack_intensity.csv"),
        xgboost_bundle=xgboost_bundle,
        experiment_summary=_read_design_summary(
            source_results / "tables" / "companion_experiment_design.csv"
        ),
        output_dir=output_dir,
        publish_root=None,
        lineage={
            "source_run": str(run_dir),
            "regeneration": "report-only; no simulation, training, or inference rerun",
            "source_run_complete": not recovery_mode,
        },
    )

    topology_snapshot = run_dir / "config" / "topology.json"
    if topology_snapshot.is_file():
        topology = TopologyConfig.model_validate_json(topology_snapshot.read_text(encoding="utf-8"))
        render_topology(topology, output_dir / "figures" / "topology_core.png")
    (output_dir / "regeneration_metadata.json").write_text(
        json.dumps(
            {
                "source_run": str(run_dir),
                "purpose": "print-ready report rendering only",
                "simulation_training_or_inference_rerun": False,
                "source_run_complete": not recovery_mode,
                "recovery_mode": recovery_mode,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Regenerated print-ready report assets in {output_dir}")


if __name__ == "__main__":
    main()
