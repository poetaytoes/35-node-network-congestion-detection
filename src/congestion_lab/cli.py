"""Typer command-line interface for the reproducible research pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console

from congestion_lab.config import load_experiment, load_topology
from congestion_lab.data.artifacts import verify_manifest
from congestion_lab.external import run_external_profile_replay
from congestion_lab.features import build_feature_table, leakage_audit
from congestion_lab.logging_config import configure_logging
from congestion_lab.pipeline import run_full_pipeline, run_scale_up_profile
from congestion_lab.reporting import render_pilot_sanity, render_topology
from congestion_lab.simulation import simulate_episode
from congestion_lab.topology import NetworkTopology
from congestion_lab.traffic import generate_workloads
from congestion_lab.traffic.generator import workload_summary

app = typer.Typer(help="Paired centralized/distributed congestion-prediction laboratory.")
config_app = typer.Typer(help="Validate typed experiment configuration.")
topology_app = typer.Typer(help="Validate or render topology profiles.")
traffic_app = typer.Typer(help="Generate deterministic flow/session workloads.")
simulate_app = typer.Typer(help="Run paired no-action simulations.")
dataset_app = typer.Typer(help="Build five-second causal datasets.")
pipeline_app = typer.Typer(help="Run complete sealed experiment pipelines.")
external_app = typer.Typer(help="Build and replay privacy-safe external traffic profiles.")
verify_app = typer.Typer(help="Verify run manifests and scientific contracts.")
app.add_typer(config_app, name="config")
app.add_typer(topology_app, name="topology")
app.add_typer(traffic_app, name="traffic")
app.add_typer(simulate_app, name="simulate")
app.add_typer(dataset_app, name="dataset")
app.add_typer(pipeline_app, name="pipeline")
app.add_typer(external_app, name="external")
app.add_typer(verify_app, name="verify")
console = Console()
configure_logging()


@config_app.command("validate")
def validate_config(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
    dry_run: Annotated[bool, typer.Option(help="Inspect inputs without writing outputs.")] = False,
) -> None:
    parsed = load_experiment(config)
    payload = {
        "status": "planned" if dry_run else "valid",
        "dry_run": dry_run,
        "name": parsed.name,
        "episodes": parsed.episodes,
        "duration_s": parsed.episode_duration_s,
        "window_s": parsed.window_s,
        "locked": parsed.sealed,
    }
    console.print_json(data=payload)


@topology_app.command("validate")
def validate_topology(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
    dry_run: Annotated[bool, typer.Option(help="Inspect inputs without writing outputs.")] = False,
) -> None:
    parsed = load_topology(config)
    payload = NetworkTopology(parsed).validate_contract()
    payload.update({"status": "planned" if dry_run else "valid", "dry_run": dry_run})
    console.print_json(data=payload)


@topology_app.command("render")
def render_topology_command(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
    dry_run: Annotated[bool, typer.Option(help="Inspect inputs without writing outputs.")] = False,
) -> None:
    parsed = load_topology(config)
    if dry_run:
        console.print_json(
            data={"status": "planned", "dry_run": True, "output": str(output.resolve())}
        )
        return
    result = render_topology(parsed, output)
    console.print_json(data={"status": "rendered", "output": str(result.resolve())})


@traffic_app.command("generate")
def generate_traffic(
    experiment: Annotated[Path, typer.Option(exists=True, readable=True)],
    output_dir: Annotated[Path, typer.Option()] = Path("runs/manual_traffic"),
    dry_run: Annotated[bool, typer.Option(help="Inspect inputs without writing outputs.")] = False,
) -> None:
    config = load_experiment(experiment)
    topology = load_topology(config.topology_path)
    if dry_run:
        console.print_json(
            data={
                "status": "planned",
                "dry_run": True,
                "episodes": config.episodes,
                "topology": topology.name,
                "output": str(output_dir.resolve()),
            }
        )
        return
    events, manifest = generate_workloads(config, topology)
    output_dir.mkdir(parents=True, exist_ok=False)
    events.to_parquet(output_dir / "raw_events.parquet", index=False)
    manifest.to_csv(output_dir / "episode_manifest.csv", index=False)
    console.print_json(data=workload_summary(events, manifest))


@simulate_app.command("paired")
def simulate_paired(
    experiment: Annotated[Path, typer.Option(exists=True, readable=True)],
    output_dir: Annotated[Path, typer.Option()] = Path("runs/manual_simulation"),
    dry_run: Annotated[bool, typer.Option(help="Inspect inputs without writing outputs.")] = False,
) -> None:
    config = load_experiment(experiment)
    topology = load_topology(config.topology_path)
    if dry_run:
        console.print_json(
            data={
                "status": "planned",
                "dry_run": True,
                "paired_runs": config.episodes * 2,
                "topology": topology.name,
                "output": str(output_dir.resolve()),
            }
        )
        return
    events, manifest = generate_workloads(config, topology)
    telemetry: list[pd.DataFrame] = []
    ledgers: list[dict[str, object]] = []
    for _, episode in events.groupby("episode_id", sort=True):
        for mode in ("centralized", "distributed"):
            result = simulate_episode(episode.reset_index(drop=True), config, topology, mode)
            telemetry.append(result.telemetry)
            ledgers.append(dict(result.ledger))
    output_dir.mkdir(parents=True, exist_ok=False)
    events.to_parquet(output_dir / "raw_events.parquet", index=False)
    manifest.to_csv(output_dir / "episode_manifest.csv", index=False)
    combined_telemetry = pd.concat(telemetry, ignore_index=True)
    combined_telemetry.to_parquet(output_dir / "telemetry.parquet", index=False)
    render_pilot_sanity(combined_telemetry, output_dir / "pilot_sanity.png")
    pd.DataFrame(ledgers).to_csv(output_dir / "ledgers.csv", index=False)
    console.print_json(
        data={"status": "complete", "output": str(output_dir.resolve()), "runs": len(ledgers)}
    )


@dataset_app.command("build")
def build_dataset(
    experiment: Annotated[Path, typer.Option(exists=True, readable=True)],
    run_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    dry_run: Annotated[bool, typer.Option(help="Inspect inputs without writing outputs.")] = False,
) -> None:
    config = load_experiment(experiment)
    telemetry_path = run_dir / "telemetry.parquet"
    manifest_path = run_dir / "episode_manifest.csv"
    if dry_run:
        console.print_json(
            data={
                "status": "planned",
                "dry_run": True,
                "inputs": [str(telemetry_path.resolve()), str(manifest_path.resolve())],
                "output": str((run_dir / "features_5s.parquet").resolve()),
            }
        )
        return
    telemetry = pd.read_parquet(telemetry_path)
    manifest = pd.read_csv(manifest_path)
    split_map = dict(zip(manifest["episode_id"], manifest["split"], strict=True))
    features = build_feature_table(telemetry, config, split_map)
    features.to_parquet(run_dir / "features_5s.parquet", index=False)
    report = leakage_audit(features)
    (run_dir / "leakage_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    console.print_json(data={"rows": len(features), "leakage": report})


@pipeline_app.command("full")
def full_pipeline(
    experiment: Annotated[Path, typer.Option(exists=True, readable=True)],
    run_id: Annotated[str, typer.Option()],
    compact_grid: Annotated[bool, typer.Option()] = False,
    publish_root: Annotated[Path, typer.Option()] = Path("results"),
    root_seed: Annotated[
        int | None,
        typer.Option(help="Override the experiment root seed and record it in the run snapshot."),
    ] = None,
    dry_run: Annotated[bool, typer.Option(help="Inspect inputs without writing outputs.")] = False,
) -> None:
    config = load_experiment(experiment)
    if dry_run:
        console.print_json(
            data={
                "status": "planned",
                "dry_run": True,
                "run_id": run_id,
                "run_dir": str((config.output_root / run_id).resolve()),
                "episodes": config.episodes,
                "compact_grid": compact_grid,
                "publish_root": str(publish_root.resolve()),
                "root_seed": root_seed if root_seed is not None else config.root_seed,
            }
        )
        return
    output = run_full_pipeline(
        experiment,
        run_id=run_id,
        compact_grid=compact_grid,
        publish_root=publish_root,
        root_seed=root_seed,
    )
    console.print_json(data={"status": "complete", "run_dir": str(output)})


@pipeline_app.command("scale-up")
def scale_up_pipeline(
    experiment: Annotated[Path, typer.Option(exists=True, readable=True)],
    run_id: Annotated[str, typer.Option()],
    dry_run: Annotated[bool, typer.Option(help="Inspect inputs without writing outputs.")] = False,
) -> None:
    config = load_experiment(experiment)
    if dry_run:
        console.print_json(
            data={
                "status": "planned",
                "dry_run": True,
                "run_id": run_id,
                "run_dir": str((config.output_root / run_id).resolve()),
                "topology": str(config.topology_path),
            }
        )
        return
    output = run_scale_up_profile(experiment, run_id=run_id)
    console.print_json(data={"status": "complete", "run_dir": str(output)})


@external_app.command("replay")
def external_replay(
    experiment: Annotated[Path, typer.Option(exists=True, readable=True)],
    manifest: Annotated[Path, typer.Option(exists=True, readable=True)],
    profile: Annotated[Path, typer.Option(exists=True, readable=True)],
    sample: Annotated[Path, typer.Option(exists=True, readable=True)],
    run_id: Annotated[str, typer.Option()],
    model: Annotated[
        Path | None,
        typer.Option(exists=True, readable=True, help="Optional frozen core model bundle."),
    ] = None,
    dry_run: Annotated[bool, typer.Option(help="Inspect inputs without writing outputs.")] = False,
) -> None:
    config = load_experiment(experiment)
    if dry_run:
        console.print_json(
            data={
                "status": "planned",
                "dry_run": True,
                "run_id": run_id,
                "run_dir": str((config.output_root / run_id).resolve()),
                "inputs": [
                    str(manifest.resolve()),
                    str(profile.resolve()),
                    str(sample.resolve()),
                    *([str(model.resolve())] if model is not None else []),
                ],
            }
        )
        return
    output = run_external_profile_replay(
        experiment,
        manifest_path=manifest,
        profile_path=profile,
        sample_path=sample,
        run_id=run_id,
        model_path=model,
    )
    console.print_json(data={"status": "complete", "run_dir": str(output)})


@verify_app.command("run")
def verify_run(
    run_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    dry_run: Annotated[bool, typer.Option(help="Inspect inputs without validation.")] = False,
) -> None:
    manifest = run_dir / "run_manifest.json"
    if dry_run:
        console.print_json(
            data={"status": "planned", "dry_run": True, "manifest": str(manifest.resolve())}
        )
        return
    failures = verify_manifest(manifest)
    payload = {"status": "pass" if not failures else "fail", "failures": failures}
    console.print_json(data=payload)
    if failures:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
