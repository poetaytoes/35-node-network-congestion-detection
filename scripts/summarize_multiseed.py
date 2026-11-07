"""Aggregate independent 35-node runs into seed-level uncertainty artifacts.

This script deliberately treats each complete run as one experimental unit.  It
does not pool object windows or locked episodes across seeds when forming the
confidence intervals, so the resulting intervals are not pseudo-replicated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POLICY_COMPARISONS = {
    "C1 − C0": ("C0", "C1", "centralized"),
    "D1 − D0": ("D0", "D1", "distributed"),
}
SYSTEM_METRICS = {
    "p95_latency_ms": "p95 latency (ms)",
    "loss_percent": "End-to-end loss (%)",
    "congestion_duration_s": "Congestion duration (s)",
    "backbone_traffic_gb": "Backbone traffic (GB)",
}


def _save_figure(fig: plt.Figure, output: Path) -> None:
    """Write matching raster-preview and editable-vector versions."""

    png_output = output.with_suffix(".png")
    svg_output = output.with_suffix(".svg")
    png_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_output, bbox_inches="tight")
    fig.savefig(svg_output, format="svg", bbox_inches="tight")
    plt.close(fig)


def _seed_bootstrap_ci(values: pd.Series, *, seed: int = 20260822) -> tuple[float, float, float]:
    """Return a seed-level mean and fixed-seed percentile bootstrap interval."""

    array = values.to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(array.mean())
    if array.size == 1:
        return (mean, mean, mean)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(array, size=(20_000, array.size), replace=True)
    lower, upper = np.quantile(sampled.mean(axis=1), [0.025, 0.975])
    return (mean, float(lower), float(upper))


def _load_seed_run(run_dir: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = run_dir / "evaluation" / "binary_metrics_locked.csv"
    system_path = run_dir / "system" / "c0_c1_d0_d1_metrics.csv"
    for path in (metrics_path, system_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing completed seed artifact: {path}")
    metrics = pd.read_csv(metrics_path)
    system = pd.read_csv(system_path)
    metrics["seed"] = seed
    system["seed"] = seed
    return metrics, system


def _prediction_summary(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    xgboost = metrics.loc[metrics["model"] == "xgboost"].copy()
    estimates = xgboost[["seed", "topology", "pr_auc", "f1"]].melt(
        id_vars=["seed", "topology"], var_name="metric", value_name="estimate"
    )
    rows: list[dict[str, object]] = []
    for (topology, metric), group in estimates.groupby(["topology", "metric"], sort=True):
        mean, low, high = _seed_bootstrap_ci(group["estimate"])
        rows.append(
            {
                "topology": topology,
                "metric": metric,
                "seed_count": int(group["seed"].nunique()),
                "mean": mean,
                "ci95_low": low,
                "ci95_high": high,
                "interval_method": "seed-level percentile bootstrap (20,000 resamples; fixed seed 20260822)",
            }
        )
    return pd.DataFrame(rows), estimates


def _policy_summary(system: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_estimates: list[dict[str, object]] = []
    for seed, seed_rows in system.groupby("seed", sort=True):
        for comparison, (baseline, policy, topology) in POLICY_COMPARISONS.items():
            paired = seed_rows.loc[
                (seed_rows["topology"] == topology)
                & (seed_rows["config"].isin([baseline, policy]))
            ]
            for metric, label in SYSTEM_METRICS.items():
                pivot = paired.pivot(index="episode_id", columns="config", values=metric)
                if not {baseline, policy}.issubset(pivot.columns):
                    raise ValueError(
                        f"seed {seed} lacks matched {comparison} values for {metric}"
                    )
                deltas = pivot[policy] - pivot[baseline]
                seed_estimates.append(
                    {
                        "seed": int(seed),
                        "comparison": comparison,
                        "metric": metric,
                        "metric_label": label,
                        "paired_episodes": int(deltas.notna().sum()),
                        "seed_mean_delta": float(deltas.mean()),
                    }
                )
    estimates = pd.DataFrame(seed_estimates)
    rows: list[dict[str, object]] = []
    for (comparison, metric, label), group in estimates.groupby(
        ["comparison", "metric", "metric_label"], sort=False
    ):
        mean, low, high = _seed_bootstrap_ci(group["seed_mean_delta"])
        rows.append(
            {
                "comparison": comparison,
                "metric": metric,
                "metric_label": label,
                "seed_count": int(group["seed"].nunique()),
                "episodes_per_seed": int(group["paired_episodes"].min()),
                "mean_seed_delta": mean,
                "ci95_low": low,
                "ci95_high": high,
                "interval_method": "seed-level percentile bootstrap (20,000 resamples; fixed seed 20260822)",
                "interpretation": "negative delta = lower impact with policy",
            }
        )
    return pd.DataFrame(rows), estimates


def _plot_prediction(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=220, sharey=True)
    for axis, metric, title in zip(axes, ("pr_auc", "f1"), ("PR-AUC", "F1 score"), strict=True):
        rows = summary.loc[summary["metric"] == metric].sort_values("topology")
        positions = np.arange(len(rows))
        means = rows["mean"].to_numpy(dtype=float)
        lows = rows["ci95_low"].to_numpy(dtype=float)
        highs = rows["ci95_high"].to_numpy(dtype=float)
        axis.errorbar(
            positions,
            means,
            yerr=np.vstack((means - lows, highs - means)),
            fmt="o",
            markersize=7,
            capsize=5,
            color="#2563eb",
        )
        axis.set_xticks(positions, [item.title() for item in rows["topology"]])
        axis.set_ylim(0, 1.05)
        axis.set_title(title, weight="bold")
        axis.grid(True, axis="y", alpha=0.3)
        for position, row in zip(positions, rows.itertuples(), strict=True):
            axis.annotate(
                f"{row.mean:.3f}\nn={row.seed_count}",
                (position, row.mean),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    axes[0].set_ylabel("Seed-level estimate")
    fig.suptitle("Locked-Test Prediction Quality Across Independent Seeds", weight="bold")
    fig.text(0.5, 0.01, "Points/error bars = mean and 95% seed-bootstrap CI.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    _save_figure(fig, output)


def _plot_policy(summary: pd.DataFrame, estimates: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=220)
    colours = {"C1 − C0": "#2563eb", "D1 − D0": "#ea580c"}
    for axis, (metric, label) in zip(axes.flat, SYSTEM_METRICS.items(), strict=True):
        rows = summary.loc[summary["metric"] == metric].set_index("comparison")
        positions = np.arange(len(POLICY_COMPARISONS))
        for position, comparison in enumerate(POLICY_COMPARISONS):
            seed_rows = estimates.loc[
                (estimates["metric"] == metric) & (estimates["comparison"] == comparison)
            ]
            jitter = np.linspace(-0.06, 0.06, len(seed_rows)) if len(seed_rows) > 1 else [0.0]
            axis.scatter(
                np.full(len(seed_rows), position) + jitter,
                seed_rows["seed_mean_delta"],
                color=colours[comparison],
                edgecolor="white",
                linewidth=0.6,
                s=48,
                zorder=3,
            )
            row = rows.loc[comparison]
            mean = float(row["mean_seed_delta"])
            low = float(row["ci95_low"])
            high = float(row["ci95_high"])
            axis.errorbar(
                position,
                mean,
                yerr=[[mean - low], [high - mean]],
                fmt="D",
                markersize=6,
                markerfacecolor="white",
                markeredgecolor="#252525",
                ecolor="#252525",
                capsize=5,
                zorder=4,
            )
        axis.axhline(0, color="#333333", linewidth=1)
        axis.set_xticks(positions, list(POLICY_COMPARISONS))
        axis.set_title(label, weight="bold")
        axis.set_ylabel("Seed mean paired change")
        axis.grid(True, alpha=0.3)
    fig.suptitle("Paired Policy Effect Across Independent Seeds", weight="bold")
    fig.text(
        0.5,
        0.01,
        "Points = per-seed paired episode means; diamonds/error bars = mean and 95% seed-bootstrap CI. "
        "Negative values indicate lower system impact.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    _save_figure(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--seeds", required=True, help="Comma-separated independent root seeds.")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--publish-root", type=Path, default=Path("results"))
    args = parser.parse_args()

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("provide at least three distinct seeds")

    metrics_frames: list[pd.DataFrame] = []
    system_frames: list[pd.DataFrame] = []
    for seed in seeds:
        metrics, system = _load_seed_run(args.runs_root / f"{args.run_prefix}_seed{seed}", seed)
        metrics_frames.append(metrics)
        system_frames.append(system)

    output = args.publish_root / f"{args.run_prefix}_summary"
    if output.exists():
        raise FileExistsError(f"summary namespace already exists: {output}")
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True)
    figures.mkdir(parents=True)

    prediction_summary, prediction_estimates = _prediction_summary(pd.concat(metrics_frames, ignore_index=True))
    policy_summary, policy_estimates = _policy_summary(pd.concat(system_frames, ignore_index=True))
    prediction_summary.to_csv(tables / "multiseed_locked_prediction_summary.csv", index=False)
    prediction_estimates.to_csv(tables / "multiseed_locked_prediction_seed_estimates.csv", index=False)
    policy_summary.to_csv(tables / "multiseed_paired_policy_summary.csv", index=False)
    policy_estimates.to_csv(tables / "multiseed_paired_policy_seed_estimates.csv", index=False)
    (output / "README.md").write_text(
        "# Multi-seed 35-node summary\n\n"
        f"Independent root seeds: {', '.join(str(seed) for seed in seeds)}.\n\n"
        "Each complete run is one resampling unit. The 95% intervals are percentile "
        "bootstrap intervals over seed-level estimates; they are not episode-level or "
        "window-level pseudo-replicates.\n",
        encoding="utf-8",
    )
    _plot_prediction(prediction_summary, figures / "figure_multiseed_prediction_quality.png")
    _plot_policy(policy_summary, policy_estimates, figures / "figure_multiseed_policy_effects.png")
    print(f"Wrote multi-seed summary to {output}")


if __name__ == "__main__":
    main()
