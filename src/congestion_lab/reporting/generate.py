"""Generate Tables 1-4 and Figures 1-9 from real run artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Preserve labels as editable SVG text for Inkscape instead of converting each
# glyph to a vector path.  Hatches, lines, markers, and other artists remain
# native vector objects.
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
from sklearn.metrics import average_precision_score, precision_recall_curve

from congestion_lab.config.models import TopologyConfig
from congestion_lab.constants import CAUSE_LABELS
from congestion_lab.data.artifacts import sha256_file
from congestion_lab.models.training import ModelBundle
from congestion_lab.topology import NetworkTopology

MODEL_LABELS = {
    "threshold": "Threshold",
    "persistence": "Persistence",
    "ewma": "EWMA",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

CLEAN_CAUSE_LABELS = {
    "normal": "Normal",
    "benign_high_load": "Benign high load",
    "attack_without_congestion": "Attack w/o congestion",
    "attack_associated_congestion": "Attack-associated congestion",
}

VIETNAM_30NODE_POSITIONS = {
    "1": (-5.2, 4.1),
    "2": (-3.6, 4.1),
    "3": (-1.3, 4.1),
    "4": (0.8, 4.1),
    "5": (2.3, 4.1),
    "6": (3.9, 4.1),
    "7": (-4.4, 3.1),
    "8": (-5.1, 2.1),
    "9": (-3.6, 2.1),
    "10": (-1.3, 3.1),
    "11": (-1.3, 2.1),
    "12": (0.8, 3.1),
    "13": (2.3, 3.1),
    "14": (1.7, 2.1),
    "15": (3.9, 3.1),
    "16": (3.3, 2.1),
    "23": (-5.1, 0.9),
    "24": (-4.2, 0.9),
    "25": (-3.3, 0.9),
    "26": (-1.3, 0.9),
    "27": (1.7, 0.9),
    "28": (3.0, 0.9),
    "17": (-4.6, -0.4),
    "18": (-3.1, -0.4),
    "19": (-1.2, -0.4),
    "20": (0.8, -0.4),
    "21": (2.7, -0.4),
    "22": (4.2, -0.4),
    "29": (-3.8, -1.8),
    "30": (3.2, -1.8),
}

# Fixed coordinates from the authoritative ``draw_vn_topology.py`` topology
# drawing.  Although the original module is named ``*_30``, it explicitly
# defines nodes 1--35 in three regional domains and four network layers.
VIETNAM_35NODE_POSITIONS = {
    "1": (1.5, 12.8), "2": (3.5, 12.8), "3": (7.75, 12.8),
    "4": (11.5, 12.8), "5": (13.0, 12.8), "6": (14.5, 12.8),
    "7": (2.5, 11.6), "8": (1.5, 10.5), "9": (3.5, 10.5),
    "10": (7.75, 11.6), "11": (7.75, 10.5),
    "12": (11.5, 11.6), "13": (13.0, 11.6), "14": (12.25, 10.5),
    "15": (14.5, 11.6), "16": (13.75, 10.5),
    "17": (1.8, 5.2), "18": (3.5, 5.2), "19": (7.75, 5.2),
    "20": (11.8, 5.2), "21": (13.2, 5.2), "22": (14.5, 5.2),
    "23": (1.2, 8.0), "24": (2.5, 8.0), "25": (3.8, 8.0),
    "26": (7.75, 8.0), "27": (12.0, 8.0), "28": (14.0, 8.0),
    "29": (1.8, 1.8), "30": (14.2, 1.8),
    "31": (2.8, 2.5), "32": (3.5, 2.5), "33": (4.2, 2.5),
    "34": (12.2, 2.5), "35": (13.5, 2.5),
}


def _save_figure(fig: Figure, path: Path) -> None:
    """Save every figure as a preview PNG and an editable vector SVG."""

    png_path = path.with_suffix(".png")
    svg_path = path.with_suffix(".svg")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _as_float(value: object) -> float:
    return float(str(value))


def _operating_point(
    truth: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> tuple[float | None, float, int]:
    """Return point precision/recall without calling abstention perfect precision."""

    predicted = np.asarray(scores >= threshold, dtype=int)
    target = np.asarray(truth, dtype=int)
    predicted_positive = int(predicted.sum())
    true_positive = int(((predicted == 1) & (target == 1)).sum())
    false_positive = int(((predicted == 1) & (target == 0)).sum())
    false_negative = int(((predicted == 0) & (target == 1)).sum())
    precision = (
        true_positive / (true_positive + false_positive)
        if predicted_positive > 0
        else None
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )
    return precision, float(recall), predicted_positive


def _early_warning_metrics(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    binary_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate frozen scores on currently non-congested rows for future onsets."""

    keys = ["episode_id", "topology", "object_id", "window_index"]
    rows: list[dict[str, object]] = []
    for (model, topology), group in predictions.groupby(["model", "topology"], sort=True):
        metric = binary_metrics.loc[
            (binary_metrics["model"] == model) & (binary_metrics["topology"] == topology)
        ]
        if metric.empty:
            continue
        horizon = int(metric.iloc[0]["horizon"])
        onset_target = f"congestion_within_h{horizon}"
        required = {*keys, "congestion_state", onset_target}
        missing = required - set(features.columns)
        if missing:
            raise ValueError(f"early-warning features missing: {sorted(missing)}")
        context = features.loc[:, [*keys, "congestion_state", onset_target]].drop_duplicates(keys)
        merged = group.merge(context, on=keys, how="left", validate="one_to_one")
        if merged[["congestion_state", onset_target]].isna().any().any():
            raise ValueError("early-warning context did not match every locked prediction row")
        at_risk = merged.loc[~merged["congestion_state"].astype(bool)].copy()
        truth = at_risk[onset_target].astype(int).to_numpy()
        score = at_risk["score"].to_numpy(dtype=float)
        predicted = at_risk["prediction"].astype(int).to_numpy()
        predicted_positive = int(predicted.sum())
        true_positive = int(((predicted == 1) & (truth == 1)).sum())
        false_positive = int(((predicted == 1) & (truth == 0)).sum())
        false_negative = int(((predicted == 0) & (truth == 1)).sum())
        precision = (
            true_positive / (true_positive + false_positive)
            if predicted_positive > 0
            else float("nan")
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if np.isfinite(precision) and (precision + recall) > 0
            else 0.0
        )
        pr_auc = (
            float(average_precision_score(truth, score))
            if len(np.unique(truth)) == 2
            else float("nan")
        )
        rows.append(
            {
                "model": model,
                "topology": topology,
                "horizon": horizon,
                "at_risk_rows": len(at_risk),
                "future_onset_rows": int(truth.sum()),
                "future_onset_prevalence": float(truth.mean()) if len(truth) else float("nan"),
                "pr_auc": pr_auc,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "predicted_positive_rows": predicted_positive,
                "operating_status": "active" if predicted_positive > 0 else "no alarms",
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_mean_ci(
    values: pd.Series | np.ndarray,
    *,
    seed: int = 20250822,
    resamples: int = 10_000,
) -> tuple[float, float, float]:
    """Return the mean and a reproducible percentile bootstrap interval.

    The unit of resampling is a locked episode, so the uncertainty stays aligned
    with the paired experimental design. This quantifies episode-to-episode
    variability; it is deliberately not presented as a multi-seed interval.
    """

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(array.mean())
    if array.size == 1:
        return (mean, mean, mean)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(array, size=(resamples, array.size), replace=True)
    lower, upper = np.quantile(sampled.mean(axis=1), [0.025, 0.975])
    return (mean, float(lower), float(upper))


def _bootstrap_median_ci(
    values: pd.Series | np.ndarray,
    *,
    seed: int = 20250822,
    resamples: int = 10_000,
) -> tuple[float, float, float]:
    """Return a median and a percentile bootstrap interval for matched events."""

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    median = float(np.median(array))
    if array.size == 1:
        return (median, median, median)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(array, size=(resamples, array.size), replace=True)
    lower, upper = np.quantile(np.median(sampled, axis=1), [0.025, 0.975])
    return (median, float(lower), float(upper))


def _wilson_proportion_ci(successes: int, total: int) -> tuple[float, float]:
    """Return a two-sided 95% Wilson interval for an event-coverage ratio."""

    if total <= 0:
        return (float("nan"), float("nan"))
    if successes < 0 or successes > total:
        raise ValueError("successes must be between zero and total")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    centre = (proportion + z**2 / (2.0 * total)) / denominator
    margin = z * np.sqrt(
        proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2)
    ) / denominator
    # At the exact boundaries, floating-point cancellation can make the Wilson
    # lower bound a few ulps greater than 0 (or the upper bound less than 1).
    # Guarantee that the observed proportion is inside its interval so plotting
    # asymmetric error bars can never receive a negative length.
    lower = min(proportion, max(0.0, float(centre - margin)))
    upper = max(proportion, min(1.0, float(centre + margin)))
    return (lower, upper)


def render_topology(config: TopologyConfig, output: str | Path) -> Path:
    network = NetworkTopology(config)
    graph = network.graph
    positions: dict[str, tuple[float, float]]
    if config.name == "student_minimal":
        positions = {
            "R1": (0, 3),
            "R2": (0, 2),
            "R3": (0, 1),
            "R4": (0, 0),
            "PE1": (1, 3),
            "PE2": (1, 2),
            "PE3": (1, 1),
            "PE4": (1, 0),
            "B1": (2, 1.5),
            "B2": (3, 1.5),
            "B3": (4, 1.5),
            "E1": (2, 3),
            "E2": (4, 3),
            "E3": (2, 0),
            "C0": (5, 1.5),
        }
    elif config.name == "vietnam_30node":
        positions = VIETNAM_30NODE_POSITIONS
    elif config.name == "vietnam_35node":
        positions = VIETNAM_35NODE_POSITIONS
    else:
        positions = nx.spring_layout(graph, seed=20260814, weight="weight")
    colors = {
        "region": "#dbeafe",
        "metro": "#dcfce7",
        "backbone": "#fef3c7",
        "edge_dc": "#ede9fe",
        "central_dc": "#fecaca",
    }
    is_vietnam_layered = config.name in {"vietnam_30node", "vietnam_35node"}
    figure_size = (
        (20, 14)
        if config.name == "vietnam_35node"
        else (16, 9)
        if config.name == "vietnam_30node"
        else (14, 8)
    )
    fig, ax = plt.subplots(figsize=figure_size)
    if is_vietnam_layered:
        edge_colors = {
            "metro_internal": "#2f855a",
            "backbone_access": "#2563eb",
            "national_backbone": "#b91c1c",
            "dc_access": "#2563eb",
            "access": "#2f855a",
        }
        for link_class in sorted({link.link_class for link in config.links}):
            edges = [
                (link.source, link.target)
                for link in config.links
                if link.link_class == link_class
                and not (
                    config.name == "vietnam_35node"
                    and {link.source, link.target} == {"17", "21"}
                )
            ]
            nx.draw_networkx_edges(
                graph,
                positions,
                edgelist=edges,
                edge_color=edge_colors[link_class],
                style="dashed" if link_class == "dc_access" else "solid",
                width=1.5,
                alpha=0.82,
                ax=ax,
            )
        if config.name == "vietnam_35node":
            # The source topology explicitly renders the 17--21 national
            # backbone bypass as an arc above the core layer.
            bypass = FancyArrowPatch(
                positions["17"],
                positions["21"],
                connectionstyle="arc3,rad=-0.35",
                color="#b91c1c",
                linewidth=2.5,
                arrowstyle="-",
                zorder=1,
            )
            ax.add_patch(bypass)
        for role in colors:
            nodes = [node for node in graph if graph.nodes[node]["role"] == role]
            nx.draw_networkx_nodes(
                graph,
                positions,
                nodelist=nodes,
                node_color=colors[role],
                edgecolors="#0f172a",
                linewidths=1.0,
                node_size=650 if role not in {"edge_dc", "central_dc"} else 780,
                node_shape="s" if role in {"edge_dc", "central_dc"} else "o",
                ax=ax,
            )
        nx.draw_networkx_labels(graph, positions, font_size=8, font_weight="bold", ax=ax)
        frames = (
            (
                (0.5, 9.8, 14.8, 3.4, "Metro / access layer", "#dcfce7"),
                (0.5, 7.0, 14.8, 2.2, "Upper edge data-center layer", "#ccfbf1"),
                (0.5, 4.2, 14.8, 2.2, "Backbone / core layer", "#f3e8ff"),
                (0.5, 1.0, 14.8, 2.6, "Lower data-center layer", "#ffedd5"),
            )
            if config.name == "vietnam_35node"
            else (
                (-5.65, 1.65, 2.55, 3.0, "HCM Metro", "#dcfce7"),
                (-1.75, 1.65, 0.9, 3.0, "Da Nang Metro", "#dbeafe"),
                (0.3, 1.65, 4.15, 3.0, "Ha Noi Metro", "#fef3c7"),
                (-5.65, -0.85, 10.35, 1.0, "Backbone / Core", "#eff6ff"),
            )
        )
        for x, y, width, height, label, color in frames:
            ax.add_patch(
                Rectangle(
                    (x, y),
                    width,
                    height,
                    facecolor=color,
                    edgecolor="#94a3b8",
                    alpha=0.18,
                    zorder=-2,
                )
            )
            ax.text(x + 0.08, y + height - 0.2, label, fontsize=9, weight="bold")
        if config.name == "vietnam_35node":
            for divider in (5.2, 10.2):
                ax.axvline(x=divider, color="#64748b", linestyle=":", linewidth=1.2)
            for x, label in (
                (2.8, "REGION 1: HO CHI MINH"),
                (7.75, "REGION 2: DA NANG"),
                (13.0, "REGION 3: HA NOI"),
            ):
                ax.text(
                    x,
                    13.5,
                    label,
                    color="#1e3a8a",
                    fontweight="bold",
                    fontsize=11,
                    ha="center",
                )
        legend = [
            Patch(facecolor=colors["region"], edgecolor="#0f172a", label="Access region"),
            Patch(facecolor=colors["metro"], edgecolor="#0f172a", label="Metro"),
            Patch(facecolor=colors["backbone"], edgecolor="#0f172a", label="Backbone"),
            Patch(facecolor=colors["edge_dc"], edgecolor="#0f172a", label="Edge DC"),
            Patch(facecolor=colors["central_dc"], edgecolor="#0f172a", label="Central DC"),
            Line2D([0], [0], color="#2f855a", label="Metro 100 Gbps"),
            Line2D([0], [0], color="#2563eb", label="Access 400 Gbps"),
            Line2D([0], [0], color="#b91c1c", label="Backbone 400 Gbps"),
        ]
        ax.legend(handles=legend, loc="lower center", ncol=4, frameon=False, fontsize=8)
        ax.set_title(
            "Vietnam 35-node topology (3 regional domains x 4 layers; 51 physical links)"
            if config.name == "vietnam_35node"
            else "Vietnam 30-node scale-up topology (42 physical links / 84 arcs)"
        )
    else:
        node_colors = [colors[str(graph.nodes[node]["role"])] for node in graph.nodes]
        nx.draw_networkx(
            graph,
            positions,
            node_color=node_colors,
            node_size=900,
            font_size=8,
            edge_color="#64748b",
            width=1.2,
            ax=ax,
        )
        ax.set_title(f"Topology profile: {config.name}")
    ax.axis("off")
    target = Path(output)
    _save_figure(fig, target)
    return target


def render_pilot_sanity(telemetry: pd.DataFrame, output: str | Path) -> Path:
    """Render deterministic pilot utilization/queue traces for pre-full-run inspection."""
    required = {"timestamp_s", "topology", "utilization", "queue_ratio"}
    missing = required - set(telemetry.columns)
    if missing:
        raise ValueError(f"pilot sanity telemetry missing columns: {sorted(missing)}")
    summary = (
        telemetry.groupby(["topology", "timestamp_s"], sort=True)
        .agg(max_utilization=("utilization", "max"), max_queue_ratio=("queue_ratio", "max"))
        .reset_index()
    )
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for topology, group in summary.groupby("topology", sort=True):
        axes[0].plot(group["timestamp_s"], group["max_utilization"], label=str(topology))
        axes[1].plot(group["timestamp_s"], group["max_queue_ratio"], label=str(topology))
    axes[0].axhline(0.8, color="#dc2626", linestyle="--", linewidth=1, label="0.80 gate")
    axes[1].axhline(0.7, color="#dc2626", linestyle="--", linewidth=1, label="0.70 gate")
    axes[0].set_ylabel("Max utilization (ratio)")
    axes[1].set_ylabel("Max queue (ratio)")
    axes[1].set_xlabel("Simulation time (s)")
    axes[0].set_title("Pilot physical-sanity traces (same offered workload, C/D replay)")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right")
    target = Path(output)
    _save_figure(fig, target)
    return target


def _table_markdown(frame: pd.DataFrame, path: Path, *, footnote: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = frame.to_markdown(index=False)
    if footnote:
        payload = f"{payload}\n\n_{footnote}_"
    path.write_text(f"{payload}\n", encoding="utf-8")


def _select_timeline_group(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    cause_predictions: pd.DataFrame,
    cause: str,
) -> pd.DataFrame:
    xgb = predictions.loc[predictions["model"] == "xgboost"]
    merged = features.merge(
        xgb[
            [
                "episode_id",
                "topology",
                "object_id",
                "window_index",
                "score",
                "prediction",
            ]
        ],
        on=["episode_id", "topology", "object_id", "window_index"],
        how="inner",
    )
    merged = merged.merge(
        cause_predictions[
            [
                "episode_id",
                "topology",
                "object_id",
                "window_index",
                "predicted_cause_index",
                "predicted_cause_label",
            ]
        ],
        on=["episode_id", "topology", "object_id", "window_index"],
        how="inner",
    )
    candidates: list[tuple[str, str, str]] = []
    for keys, group in merged.groupby(["episode_id", "topology", "object_id"], sort=True):
        ordered = group.sort_values("window_index")
        onset_positions = np.flatnonzero(ordered["congestion_onset"].to_numpy(dtype=bool))
        has_context = any(2 <= int(position) <= len(ordered) - 3 for position in onset_positions)
        if (ordered["cause_label"] == cause).any() and has_context:
            candidates.append((str(keys[0]), str(keys[1]), str(keys[2])))
    if not candidates:
        return pd.DataFrame()
    selected = candidates[0]
    return merged.loc[
        (merged["episode_id"] == selected[0])
        & (merged["topology"] == selected[1])
        & (merged["object_id"] == selected[2])
    ].sort_values("window_index")


def generate_report_assets(
    *,
    features: pd.DataFrame,
    binary_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    binary_slices: pd.DataFrame,
    lead_times: pd.DataFrame,
    cause_metrics: pd.DataFrame,
    cause_matrix: pd.DataFrame,
    cause_predictions: pd.DataFrame,
    cause_slices: pd.DataFrame,
    system_summary: pd.DataFrame,
    cross_topology: pd.DataFrame,
    ablation: pd.DataFrame,
    horizon: pd.DataFrame,
    threshold_sensitivity: pd.DataFrame,
    unseen_attack: pd.DataFrame,
    xgboost_bundle: ModelBundle,
    experiment_summary: dict[str, object],
    output_dir: str | Path,
    publish_root: str | Path | None = None,
    lineage: dict[str, str] | None = None,
) -> dict[str, str]:
    output = Path(output_dir)
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    design_table = pd.DataFrame(
        [
            {
                "parameter": key,
                "value": json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list, tuple))
                else value,
            }
            for key, value in experiment_summary.items()
        ]
    )
    design_table.to_csv(tables / "companion_experiment_design.csv", index=False)
    _table_markdown(
        design_table,
        tables / "companion_experiment_design.md",
        footnote=(
            "Link bytes and compute/request ledgers use separate units; C/D active compute "
            "budgets are equal by contract."
        ),
    )

    episode_split = experiment_summary.get("episode_split", {})
    split_label = " / ".join(
        str(episode_split.get(name, "?"))
        for name in ("train", "validation", "locked_test")
    )
    pipeline_labels = (
        "Raw flow\nevents",
        "Paired C/D\nreplay",
        "1 s\ntelemetry",
        "5 s causal\nwindows",
        f"Episode split\n{split_label}",
        "Train +\nvalidation",
        "Sealed locked\ninference",
        "Tables / figures\n/ policy",
    )
    fig, ax = plt.subplots(figsize=(16, 3.5))
    ax.set_xlim(-0.5, len(pipeline_labels) - 0.5)
    ax.set_ylim(-0.6, 0.8)
    for index, label in enumerate(pipeline_labels):
        ax.text(
            index,
            0,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "#eff6ff" if index < 5 else "#ecfdf5",
                "edgecolor": "#334155",
            },
        )
        if index:
            ax.annotate(
                "",
                xy=(index - 0.42, 0),
                xytext=(index - 0.58, 0),
                arrowprops={"arrowstyle": "->", "color": "#475569", "lw": 1.5},
            )
    ax.text(
        4,
        -0.42,
        "Locked rows never select features, hyperparameters, or thresholds",
        ha="center",
        fontsize=9,
        color="#991b1b",
    )
    ax.set_title("Auditable Experiment Pipeline (correct execution order)")
    ax.axis("off")
    _save_figure(fig, figures / "dataset_pipeline.png")

    table1 = (
        features.groupby(["split", "topology"], sort=False)
        .agg(
            episodes=("episode_id", "nunique"),
            raw_rows=("episode_id", "size"),
            eligible_h1=("target_valid_h1", "sum"),
            eligible_h2=("model_eligible", "sum"),
            eligible_h4=("target_valid_h4", "sum"),
            normal=("cause_label", lambda values: int((values == "normal").sum())),
            benign_high_load=(
                "cause_label",
                lambda values: int((values == "benign_high_load").sum()),
            ),
            attack_without_congestion=(
                "cause_label",
                lambda values: int((values == "attack_without_congestion").sum()),
            ),
            attack_associated_congestion=(
                "cause_label",
                lambda values: int((values == "attack_associated_congestion").sum()),
            ),
        )
        .reset_index()
    )
    table1.to_csv(tables / "table1_dataset_split.csv", index=False)
    _table_markdown(
        table1,
        tables / "table1_dataset_split.md",
        footnote=(
            "Raw rows include every 5-second object-window. Eligible h1/h2/h4 counts "
            "exclude only group-tail rows whose exact future target is unavailable."
        ),
    )

    table2 = binary_metrics.sort_values(["model", "topology"])
    table2 = table2.assign(
        missed_events=table2["congestion_events"] - table2["anticipated_events"],
        operating_status=np.where(table2["alarm_events"] > 0, "active", "no alarms"),
    )
    table2.to_csv(tables / "table2_prediction_by_topology.csv", index=False)
    _table_markdown(table2.round(5), tables / "table2_prediction_by_topology.md")

    early_warning = _early_warning_metrics(features, predictions, binary_metrics)
    early_warning.to_csv(tables / "companion_early_warning_metrics.csv", index=False)
    _table_markdown(
        early_warning.round(5),
        tables / "companion_early_warning_metrics.md",
        footnote=(
            "Diagnostic early-warning evaluation only: rows already congested at prediction "
            "time are excluded, the target is a new onset within the frozen horizon, and all "
            "scores and thresholds remain unchanged. Precision is undefined when no alarms fire."
        ),
    )

    system_aggregate = system_summary.groupby(["config", "topology"], as_index=False).agg(
        p95_latency_ms=("p95_latency_ms", "mean"),
        mean_latency_ms=("mean_latency_ms", "mean"),
        loss_percent=("loss_percent", "mean"),
        congestion_duration_s=("congestion_duration_s", "mean"),
        backbone_traffic_gb=("backbone_traffic_gb", "mean"),
        bandwidth_hop_gb_hop=("bandwidth_hop_gb_hop", "mean"),
        accepted_requests=("accepted_requests", "mean"),
        rejected_requests=("rejected_requests", "mean"),
        actions=("actions", "mean"),
        shaped_gb=("shaped_gb", "mean"),
        shaped_percent=("shaped_percent", "mean"),
        signaling_bytes=("signaling_bytes", "mean"),
        route_changes=("route_changes", "mean"),
    )
    system_aggregate.to_csv(tables / "table3_system_c0_c1_d0_d1.csv", index=False)
    _table_markdown(system_aggregate.round(5), tables / "table3_system_c0_c1_d0_d1.md")
    paired_wide = system_summary.pivot(
        index="episode_id",
        columns="config",
        values=[
            "p95_latency_ms",
            "loss_percent",
            "congestion_duration_s",
            "backbone_traffic_gb",
            "bandwidth_hop_gb_hop",
            "accepted_requests",
            "rejected_requests",
        ],
    )
    paired_rows: list[dict[str, object]] = []
    comparisons = (("C1", "C0"), ("D1", "D0"), ("D0", "C0"), ("D1", "C1"))
    for episode_id in paired_wide.index:
        for left, right in comparisons:
            if left not in paired_wide.columns.get_level_values(
                1
            ) or right not in paired_wide.columns.get_level_values(1):
                continue
            row: dict[str, object] = {
                "episode_id": episode_id,
                "comparison": f"{left}-{right}",
            }
            for metric in (
                "p95_latency_ms",
                "loss_percent",
                "congestion_duration_s",
                "backbone_traffic_gb",
                "bandwidth_hop_gb_hop",
                "accepted_requests",
                "rejected_requests",
            ):
                row[f"delta_{metric}"] = float(
                    paired_wide.loc[episode_id, (metric, left)]
                    - paired_wide.loc[episode_id, (metric, right)]
                )
            paired_rows.append(row)
    paired_deltas = pd.DataFrame(paired_rows)
    paired_deltas.to_csv(tables / "table3_paired_episode_deltas.csv", index=False)

    # Policy effects are calculated within the same locked episode. This is
    # more informative than comparing four independent-looking means because
    # it controls for each episode's offered workload and attack schedule.
    policy_comparisons = ("C1-C0", "D1-D0")
    policy_metrics = (
        "p95_latency_ms",
        "loss_percent",
        "congestion_duration_s",
        "backbone_traffic_gb",
    )
    paired_effect_rows: list[dict[str, object]] = []
    for comparison in policy_comparisons:
        comparison_rows = paired_deltas.loc[paired_deltas["comparison"] == comparison]
        for metric in policy_metrics:
            mean, ci_low, ci_high = _bootstrap_mean_ci(
                comparison_rows[f"delta_{metric}"],
            )
            paired_effect_rows.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "paired_episodes": int(len(comparison_rows)),
                    "mean_delta": mean,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "interval_method": "episode-level percentile bootstrap (10,000 resamples; fixed seed 20250822)",
                    "interpretation": "negative delta = lower impact with policy",
                }
            )
    paired_effects = pd.DataFrame(paired_effect_rows)
    paired_effects.to_csv(tables / "table3_paired_policy_effects.csv", index=False)
    _table_markdown(
        paired_effects.round(5),
        tables / "table3_paired_policy_effects.md",
        footnote=(
            "Each delta is policy minus its paired no-policy baseline in the same "
            "locked episode. Intervals quantify episode-to-episode variability, not "
            "multi-seed uncertainty."
        ),
    )

    ablation.to_csv(tables / "table4_ablation_robustness.csv", index=False)
    _table_markdown(ablation.round(5), tables / "table4_ablation_robustness.md")

    cross_topology.to_csv(tables / "companion_cross_topology.csv", index=False)
    horizon.to_csv(tables / "companion_horizon_sensitivity.csv", index=False)
    threshold_sensitivity.to_csv(
        tables / "companion_congestion_threshold_sensitivity.csv", index=False
    )
    unseen_attack.to_csv(tables / "companion_unseen_attack_intensity.csv", index=False)
    cause_metrics.to_csv(tables / "companion_cause_metrics.csv", index=False)
    binary_slices.to_csv(tables / "companion_binary_metrics_slices.csv", index=False)
    cause_slices.to_csv(tables / "companion_cause_metrics_slices.csv", index=False)

    # --- Figure 1: Class Distribution (model_eligible only, percentage layout) ---
    eligible = features.loc[features["model_eligible"]].copy()
    class_counts = (
        eligible.groupby(["split", "topology", "cause_label"]).size().unstack(fill_value=0)
    )
    class_counts = class_counts.reindex(columns=CAUSE_LABELS, fill_value=0)
    ordered_index = pd.MultiIndex.from_product(
        [
            ["train", "validation", "locked_test"],
            ["centralized", "distributed"],
        ],
        names=["split", "topology"],
    )
    class_counts = class_counts.reindex(ordered_index, fill_value=0)
    class_percentages = class_counts.div(class_counts.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    class_percentages.plot(kind="bar", stacked=False, ax=ax, colormap="tab10", width=0.8)
    ax.set_title("Class Distribution by Dataset Split (model_eligible only)")
    ax.set_xlabel("Split / topology")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(
        np.arange(len(class_counts.index)),
        [f"{split}\n{topology}" for split, topology in class_counts.index],
        rotation=0,
    )
    ax.legend(
        labels=[CLEAN_CAUSE_LABELS.get(c, c) for c in class_counts.columns],
        title="Cause",
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    _save_figure(fig, figures / "figure1_class_distribution.png")

    # --- Figure 2: Precision-Recall Curves with AP in Legend & Threshold Operating Points ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True, dpi=300)
    target = "future_congestion_h2"
    recall_grid = np.linspace(0, 1, 100)

    for ax, topology in zip(axes, ("centralized", "distributed"), strict=True):
        topology_predictions = predictions.loc[predictions["topology"] == topology]
        
        xgb_sample = topology_predictions.loc[topology_predictions["model"] == "xgboost"]
        if not xgb_sample.empty:
            prevalence = float(xgb_sample[target].astype(int).mean())
            ax.axhline(
                prevalence,
                color="#64748b",
                linestyle="--",
                alpha=0.7,
                label=f"Prevalence ({prevalence:.3f})",
            )

        for model, group in topology_predictions.groupby("model", sort=True):
            truth = group[target].astype(int).to_numpy()
            scores = group["score"].to_numpy()
            if truth.sum() == 0:
                continue
            
            ap_score = average_precision_score(truth, scores)
            precision, recall, _thresholds = precision_recall_curve(truth, scores)
            
            sort_idx = np.argsort(recall)
            recall_sorted, precision_sorted = recall[sort_idx], precision[sort_idx]
            precision_interp = np.interp(recall_grid, recall_sorted, precision_sorted)
            
            model_name = MODEL_LABELS.get(str(model), str(model))
            (line,) = ax.plot(
                recall_grid,
                precision_interp,
                label=f"{model_name} (AP={ap_score:.4f})",
                lw=2,
            )

            m_thresh = binary_metrics.loc[
                (binary_metrics["model"] == model) & (binary_metrics["topology"] == topology),
                "threshold",
            ]
            if not m_thresh.empty:
                t_val = float(m_thresh.iloc[0])
                prec_pt, rec_pt, _ = _operating_point(truth, scores, t_val)
                if prec_pt is None:
                    line.set_label(f"{model_name} (AP={ap_score:.4f}; no alarms)")
                else:
                    ax.plot(
                        rec_pt,
                        prec_pt,
                        marker="o",
                        markersize=6,
                        color=line.get_color(),
                    )

        ax.set_title(topology.title(), fontsize=12, weight="bold")
        ax.set_xlabel("Recall")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
        
    axes[0].set_ylabel("Precision")
    axes[1].legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "Precision-Recall Curves on the Sealed Locked Test (with Operating Points)",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_figure(fig, figures / "figure2_precision_recall.png")

    # --- Figure 2b: Prediction Quality Summary ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Preserve colour for screen/PDF viewing, but do not rely on it: hatches
    # and black outlines keep the topology comparison readable in monochrome.
    model_order = list(binary_metrics["model"].drop_duplicates())
    topology_order = list(binary_metrics["topology"].drop_duplicates())
    topology_styles = {
        "centralized": {"facecolor": "#4C78A8", "hatch": "///"},
        "distributed": {"facecolor": "#F58518", "hatch": "..."},
    }

    def style_topology_bars(axis: plt.Axes) -> None:
        for topology, container in zip(topology_order, axis.containers, strict=False):
            style = topology_styles.get(
                topology, {"facecolor": "#E6E6E6", "hatch": "xx"}
            )
            for patch in container:
                patch.set_facecolor(style["facecolor"])
                patch.set_hatch(style["hatch"])
                patch.set_edgecolor("#202020")
                patch.set_linewidth(0.8)

    sns.barplot(
        data=binary_metrics,
        x="model",
        y="pr_auc",
        hue="topology",
        ax=axes[0],
        palette="tab10",
    )
    style_topology_bars(axes[0])
    axes[0].set_title("PR-AUC on locked test (ranking metric)", weight="bold")
    # Both metrics are ratios.  Keep a common zero-based scale so that small
    # differences near one are not visually exaggerated by a truncated axis.
    axes[0].set_ylim(0, 1.10)
    model_tick_positions = np.arange(len(model_order))
    model_tick_labels = [MODEL_LABELS.get(model, model) for model in model_order]
    axes[0].set_xticks(model_tick_positions, model_tick_labels, rotation=25, ha="right")
    for p in axes[0].patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            axes[0].annotate(
                f"{h:.4f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=7,
                xytext=(0, 3),
                textcoords="offset points",
            )
    axes[0].set_ylabel("PR-AUC (average precision)")

    sns.barplot(
        data=binary_metrics,
        x="model",
        y="f1",
        hue="topology",
        ax=axes[1],
        palette="tab10",
    )
    style_topology_bars(axes[1])
    axes[1].set_title("F1-score at frozen threshold", weight="bold")
    axes[1].set_ylim(0, 1.10)
    axes[1].set_xticks(model_tick_positions, model_tick_labels, rotation=25, ha="right")
    axes[1].set_ylabel("F1 score")
    fully_abstaining_models = {
        model
        for model in model_order
        if not binary_metrics.loc[binary_metrics["model"] == model].empty
        and (binary_metrics.loc[binary_metrics["model"] == model, "alarm_events"] == 0).all()
    }
    for topology, container in zip(topology_order, axes[1].containers, strict=False):
        for model, patch in zip(model_order, container.patches, strict=False):
            height = patch.get_height()
            metric = binary_metrics.loc[
                (binary_metrics["model"] == model)
                & (binary_metrics["topology"] == topology)
            ]
            no_alarms = not metric.empty and int(metric.iloc[0]["alarm_events"]) == 0
            x_position = patch.get_x() + patch.get_width() / 2.0
            if not np.isnan(height) and height > 0:
                axes[1].annotate(
                    f"{height:.3f}",
                    (x_position, height),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    xytext=(0, 3),
                    textcoords="offset points",
                )
            elif no_alarms and model not in fully_abstaining_models:
                axes[1].annotate(
                    "No alarms",
                    (x_position, 0.015),
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.8,
                        "pad": 0.5,
                    },
                )
    for model in fully_abstaining_models:
        axes[1].annotate(
            "No alarms (both)",
            (model_order.index(model), 0.015),
            ha="center",
            va="bottom",
            fontsize=6,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.8,
                "pad": 0.5,
            },
        )

    fig.suptitle(
        "Prediction Quality by Model and Topology",
        fontsize=14,
        weight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        "PR-AUC summarizes score ranking, not point precision. "
        "'No alarms' means the frozen threshold produced no positive predictions.",
        ha="center",
        va="bottom",
        fontsize=7,
    )
    for axis in axes:
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
    fig.legend(
        handles=[
            Patch(
                facecolor=topology_styles.get(
                    topology, topology_styles["centralized"]
                )["facecolor"],
                edgecolor="#202020",
                hatch=topology_styles.get(
                    topology, topology_styles["centralized"]
                )["hatch"],
                label=topology.title(),
            )
            for topology in topology_order
        ],
        title="Topology",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(topology_order),
        frameon=True,
        fontsize=8,
        title_fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.86))
    _save_figure(fig, figures / "figure2b_prediction_quality.png")

    benign = _select_timeline_group(features, predictions, cause_predictions, "benign_high_load")
    attack = _select_timeline_group(
        features, predictions, cause_predictions, "attack_associated_congestion"
    )
    if benign.empty or attack.empty:
        missing = [
            label
            for label, frame in (
                ("benign_high_load", benign),
                ("attack_associated_congestion", attack),
            )
            if frame.empty
        ]
        raise RuntimeError(f"locked test lacks deterministic Figure 3 event(s): {missing}")

    def onset_context(frame: pd.DataFrame) -> pd.DataFrame:
        ordered = frame.sort_values("window_index").reset_index(drop=True)
        onset_positions = np.flatnonzero(ordered["congestion_onset"].to_numpy(dtype=bool))
        center = int(onset_positions[0])
        return ordered.iloc[max(0, center - 8) : min(len(ordered), center + 9)].copy()

    benign = onset_context(benign)
    attack = onset_context(attack)
    selection_payload = {
        label: {
            "episode_id": str(frame["episode_id"].iloc[0]),
            "topology": str(frame["topology"].iloc[0]),
            "object_id": str(frame["object_id"].iloc[0]),
            "onset_window_indices": frame.loc[frame["congestion_onset"], "window_index"]
            .astype(int)
            .tolist(),
            "binary_threshold": xgboost_bundle.threshold,
            "predicted_cause_at_onset": str(
                frame.loc[frame["congestion_onset"], "predicted_cause_label"].iloc[0]
            ),
        }
        for label, frame in (
            ("benign_high_load", benign),
            ("attack_associated_congestion", attack),
        )
    }
    (tables / "figure3_deterministic_selection.json").write_text(
        json.dumps(selection_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    
    # --- Figure 3: Early Warning Timeline (with Shaded Warning Window & Lead Time Arrow) ---
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 9),
        gridspec_kw={"width_ratios": [3.4, 1.6]},
        sharex="row",
        dpi=300,
    )
    # Keep colour for digital viewing.  The primary time-series also use line
    # styles and markers, so they remain distinguishable after B/W printing.
    cause_shades = ["#66C2A5", "#8DA0CB", "#FFD92F", "#B3B3B3"]
    cause_cmap = ListedColormap(cause_shades)
    timeline_styles = {
        "utilization": {"linestyle": "-", "marker": "o", "label": "Utilization"},
        "queue": {"linestyle": "--", "marker": "s", "label": "Queue ratio"},
        "score": {"linestyle": "-.", "marker": "^", "label": "Predicted probability"},
    }
    
    for row_index, (group, title) in enumerate(
        zip(
            (benign, attack),
            ("Benign Flash Crowd", "Attack-Associated Congestion"),
            strict=True,
        )
    ):
        ax = axes[row_index, 0]
        time = group["window_start_s"]
        ax.plot(
            time,
            group["util_end"],
            color="#1F4E9A",
            linewidth=1.8,
            markersize=3.2,
            markevery=2,
            **timeline_styles["utilization"],
        )
        ax.plot(
            time,
            group["queue_ratio"],
            color="#D97706",
            linewidth=1.8,
            markersize=3.2,
            markevery=2,
            **timeline_styles["queue"],
        )
        ax.plot(
            time,
            group["score"],
            color="#228B22",
            linewidth=1.8,
            markersize=3.6,
            markevery=2,
            **timeline_styles["score"],
        )
        
        onset_val: float | None = None
        warning_start: float | None = None
        first_alarm: float | None = None
        onset_rows = group.loc[group["congestion_onset"]]
        if not onset_rows.empty:
            onset_val = float(onset_rows["window_start_s"].iloc[0])
            warning_start = max(float(time.min()), onset_val - 10.0)
            
            # Warning window
            warning_patch = ax.axvspan(
                warning_start,
                onset_val,
                facecolor="#FFF2A8",
                edgecolor="#C9A227",
                hatch="///",
                alpha=0.65,
                label="Warning window (10s)",
            )
            warning_patch.set_linewidth(0.0)
            ax.axvline(onset_val, color="#C62828", linestyle=":", linewidth=1.6, label="Onset")
            
            # Only an alarm inside the declared 10 s warning window can be
            # presented as lead time.  Older alarms are not retrospectively
            # credited to the event.
            pred_binary = group["score"] >= xgboost_bundle.threshold
            alarms = group.loc[
                pred_binary
                & (group["window_start_s"] >= onset_val - 10.0)
                & (group["window_start_s"] < onset_val)
            ]
            if not alarms.empty:
                first_alarm = float(alarms["window_start_s"].iloc[0])
                if first_alarm < onset_val:
                    lead_time_s = onset_val - first_alarm
                    annotation_y = 0.52
                    ax.annotate(
                        "",
                        xy=(onset_val, annotation_y),
                        xytext=(first_alarm, annotation_y),
                        arrowprops=dict(arrowstyle="<->", color="black", lw=1.5)
                    )
                    # Keep explanatory text beside the event marker instead of
                    # drawing it across the telemetry curves inside the warning
                    # window.  The white callout remains legible in B/W print.
                    ax.text(
                        onset_val + 1.2,
                        0.61,
                        f"Lead time = {lead_time_s:g} s",
                        ha="left",
                        va="center",
                        fontsize=8,
                        weight="bold",
                        bbox={
                            "boxstyle": "round,pad=0.25",
                            "facecolor": "white",
                            "edgecolor": "#555555",
                            "alpha": 0.92,
                        },
                        zorder=6,
                    )
            
        ax.axhline(0.8, color="#666666", linestyle=(0, (4, 2)), label="Utilization threshold")
        ax.axhline(
            xgboost_bundle.threshold,
            color="#6D28D9",
            linestyle=(0, (6, 2, 1, 2)),
            label=f"Frozen threshold ({xgboost_bundle.threshold:.3f})",
        )
        ax.set_title(f"{title}: {group['episode_id'].iloc[0]} / {group['object_id'].iloc[0]}")
        ax.set_ylabel("Normalized value")
        ax.grid(True, alpha=0.3)

        cause_axis = axes[row_index, 1]
        cause_values = np.vstack(
            [
                group["cause_index"].to_numpy(dtype=int),
                group["predicted_cause_index"].to_numpy(dtype=int),
            ]
        )
        cause_axis.imshow(
            cause_values,
            aspect="auto",
            interpolation="nearest",
            cmap=cause_cmap,
            vmin=0,
            vmax=3,
            extent=(float(time.min()), float(time.max()) + 5, 1.5, -0.5),
        )
        # The cause timeline must carry the same early-warning evidence as the
        # telemetry panel.  Otherwise it cannot show whether a cause label was
        # available inside the declared warning window.
        if onset_val is not None and warning_start is not None:
            cause_warning = cause_axis.axvspan(
                warning_start,
                onset_val,
                facecolor="#FFF2A8",
                edgecolor="#C9A227",
                hatch="///",
                alpha=0.35,
                zorder=3,
            )
            cause_warning.set_linewidth(0.0)
            cause_axis.axvline(onset_val, color="#C62828", linestyle=":", linewidth=1.6, zorder=4)
        if first_alarm is not None:
            cause_axis.axvline(first_alarm, color="#202020", linestyle="--", linewidth=1.3, zorder=4)
            cause_axis.plot(first_alarm, 1.38, marker="v", color="#202020", markersize=5, zorder=5)
        cause_axis.set_yticks([0, 1], labels=["True cause", "Predicted cause"])
        cause_axis.set_title("Cause classifier timeline")
        cause_axis.grid(False)
        cause_axis.set_xlabel("Time (s)")
        
    axes[-1, 0].set_xlabel("Time (s)")
    cause_legend = [
        Patch(facecolor=cause_shades[index], edgecolor="#202020", label=CLEAN_CAUSE_LABELS.get(label, label))
        for index, label in enumerate(CAUSE_LABELS)
    ]
    timeline_handles, timeline_labels = axes[0, 0].get_legend_handles_labels()
    timeline_handles.append(Line2D([0], [0], color="#202020", linestyle="--", label="First alarm"))
    timeline_labels.append("First alarm")
    fig.legend(
        timeline_handles,
        timeline_labels,
        loc="upper left",
        bbox_to_anchor=(0.05, 0.96),
        ncol=3,
        frameon=True,
        fontsize=7,
    )
    fig.legend(
        handles=cause_legend,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle("Deterministic Early Warning and Cause Predictions (with 10s Window)", y=0.99)
    fig.tight_layout(rect=(0, 0.07, 1, 0.84))
    _save_figure(fig, figures / "figure3_early_warning_timeline.png")

    # --- Figure 4: Operational Usefulness (3-Panel Bar Chart with NA Handling & Values) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    model_order = list(binary_metrics["model"].drop_duplicates())
    topology_order = list(binary_metrics["topology"].drop_duplicates())
    model_tick_positions = np.arange(len(model_order))
    model_tick_labels = [MODEL_LABELS.get(model, model) for model in model_order]
    
    # Panel 1: False alarms
    sns.barplot(data=binary_metrics, x="model", y="false_alarms_per_hour", hue="topology", ax=axes[0], palette="tab10")
    style_topology_bars(axes[0])
    axes[0].set_title("False alarms / system-hour", weight="bold")
    axes[0].set_ylabel("False alarms / system-hour")
    axes[0].set_xticks(model_tick_positions, model_tick_labels, rotation=25, ha="right")
    axes[0].grid(True, alpha=0.3)
    for p in axes[0].patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            axes[0].annotate(
                f"{h:.1f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )
    
    # Panel 2: Event coverage
    coverage = binary_metrics.copy()
    coverage["event_coverage"] = coverage["event_coverage"].fillna(0.0)
    coverage[["coverage_ci95_low", "coverage_ci95_high"]] = coverage.apply(
        lambda row: pd.Series(
            _wilson_proportion_ci(
                int(row["anticipated_events"]), int(row["congestion_events"])
            )
        ),
        axis=1,
    )
    sns.barplot(data=coverage, x="model", y="event_coverage", hue="topology", ax=axes[1], palette="tab10")
    style_topology_bars(axes[1])
    axes[1].set_title("Event coverage (Wilson 95% CI; anticipated / total)", weight="bold")
    axes[1].set_ylabel("Event coverage (ratio)")
    axes[1].set_ylim(0, 1.15)
    axes[1].set_xticks(model_tick_positions, model_tick_labels, rotation=25, ha="right")
    axes[1].grid(True, alpha=0.3)
    coverage_lookup = coverage.set_index(["model", "topology"])
    for topology, container in zip(topology_order, axes[1].containers, strict=False):
        for model, patch in zip(model_order, container, strict=False):
            row = coverage_lookup.loc[(model, topology)]
            height = float(patch.get_height())
            low = float(row["coverage_ci95_low"])
            high = float(row["coverage_ci95_high"])
            axes[1].errorbar(
                patch.get_x() + patch.get_width() / 2.0,
                height,
                yerr=[[max(0.0, height - low)], [max(0.0, high - height)]],
                fmt="none",
                ecolor="#252525",
                capsize=3,
                linewidth=1,
                zorder=5,
            )
            coverage_label = (
                f"{height:.2f}\n{int(row['anticipated_events'])}/{int(row['congestion_events'])}"
                if height > 0
                else f"0/{int(row['congestion_events'])}"
            )
            axes[1].annotate(
                coverage_label,
                (patch.get_x() + patch.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=7,
                xytext=(0, 2),
                textcoords="offset points",
            )
    
    # Panel 3: Median lead time (Capped to 30s to avoid heuristic distortions)
    lead_clean = binary_metrics.copy()
    # Mask lead time for models with 0 coverage to show NA
    lead_clean.loc[lead_clean["event_coverage"] == 0, "lead_time_median_s"] = np.nan
    lead_interval_rows: list[dict[str, object]] = []
    for (model, topology), group in lead_times.groupby(["model", "topology"], sort=False):
        median, lower, upper = _bootstrap_median_ci(group["lead_time_s"])
        lead_interval_rows.append(
            {
                "model": model,
                "topology": topology,
                "lead_ci95_low": lower,
                "lead_ci95_high": upper,
                "matched_events": int(len(group)),
                "lead_median": median,
            }
        )
    lead_intervals = pd.DataFrame(lead_interval_rows)
    lead_clean = lead_clean.merge(lead_intervals, on=["model", "topology"], how="left")
    
    sns.barplot(data=lead_clean, x="model", y="lead_time_median_s", hue="topology", ax=axes[2], palette="tab10")
    style_topology_bars(axes[2])
    axes[2].set_title("Median lead time (bootstrap 95% CI)", weight="bold")
    axes[2].set_ylabel("Median lead time (s)")
    axes[2].set_ylim(0, 15)
    axes[2].set_xticks(model_tick_positions, model_tick_labels, rotation=25, ha="right")
    axes[2].grid(True, alpha=0.3)
    lead_lookup = lead_clean.set_index(["model", "topology"])
    for topology, container in zip(topology_order, axes[2].containers, strict=False):
        for model, patch in zip(model_order, container, strict=False):
            row = lead_lookup.loc[(model, topology)]
            height = float(patch.get_height())
            centre = float(row["lead_median"]) if pd.notna(row["lead_median"]) else float("nan")
            if np.isfinite(centre) and centre > 0:
                low = float(row["lead_ci95_low"])
                high = float(row["lead_ci95_high"])
                axes[2].errorbar(
                    patch.get_x() + patch.get_width() / 2.0,
                    centre,
                    yerr=[[max(0.0, centre - low)], [max(0.0, high - centre)]],
                    fmt="none",
                    ecolor="#252525",
                    capsize=3,
                    linewidth=1,
                    zorder=5,
                )
                axes[2].annotate(
                    f"{centre:.1f}\nn={int(row['matched_events'])}",
                    (patch.get_x() + patch.get_width() / 2.0, min(centre, 14.0)),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    xytext=(0, 2),
                    textcoords="offset points",
                )
            else:
                axes[2].annotate("NA", (patch.get_x() + patch.get_width() / 2.0, 0.2), ha="center", va="bottom", fontsize=8, color="gray")

    for axis in axes:
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
    fig.legend(
        handles=[
            Patch(
                facecolor=topology_styles.get(topology, topology_styles["centralized"])["facecolor"],
                edgecolor="#202020",
                hatch=topology_styles.get(topology, topology_styles["centralized"])["hatch"],
                label=topology,
            )
            for topology in topology_order
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=max(1, len(topology_order)),
        frameon=False,
        title="Topology",
    )
    fig.suptitle(
        "Operational Usefulness of Predictions (NA lead time = no anticipated event)",
        fontsize=14,
        weight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    _save_figure(fig, figures / "figure4_lead_time_distribution.png")

    # --- Figure 5: Confusion Matrix (Clean Labels + Sample Size n) ---
    clean_cause_names = [CLEAN_CAUSE_LABELS.get(c, c) for c in cause_matrix.index]
    # Build support-annotated tick labels
    class_supports = [int((cause_predictions["cause_label"] == c).sum()) if "cause_label" in cause_predictions.columns else 0 for c in cause_matrix.index]
    annotated_y_ticks = [f"{name}\n(n={sup})" if sup > 0 else name for name, sup in zip(clean_cause_names, class_supports)]

    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=300)
    sns.heatmap(
        cause_matrix * 100.0,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        vmin=0,
        vmax=100,
        xticklabels=clean_cause_names,
        yticklabels=annotated_y_ticks,
        ax=ax,
    )
    ax.set_title("Cause Classification Confusion Matrix (%) — locked test", weight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    _save_figure(fig, figures / "figure5_cause_confusion_matrix.png")

    # --- NEW: Figure 5b / Suggested C: Class-Wise Precision, Recall, and F1 ---
    class_rows = []
    for c_idx, c_label in enumerate(CAUSE_LABELS):
        # Extract row from confusion matrix
        pred_label = cause_predictions["predicted_cause_label"] == c_label
        true_label = cause_predictions["cause_label"] == c_label
        tp = int((pred_label & true_label).sum())
        fp = int((pred_label & ~true_label).sum())
        fn = int((~pred_label & true_label).sum())
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_c = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        class_rows.append({"Class": CLEAN_CAUSE_LABELS.get(c_label, c_label), "Metric": "Precision", "Score": prec})
        class_rows.append({"Class": CLEAN_CAUSE_LABELS.get(c_label, c_label), "Metric": "Recall", "Score": rec})
        class_rows.append({"Class": CLEAN_CAUSE_LABELS.get(c_label, c_label), "Metric": "F1", "Score": f1_c})
        
    class_metric_df = pd.DataFrame(class_rows)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    sns.barplot(data=class_metric_df, x="Class", y="Score", hue="Metric", ax=ax, palette=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_title("Class-Wise Precision, Recall, and F1", weight="bold")
    ax.set_ylim(0, 1.12)
    ax.grid(True, axis="y", alpha=0.3)
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2.0, h), ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    plt.tight_layout()
    _save_figure(fig, figures / "figure5b_cause_class_metrics.png")

    classifier = None
    if xgboost_bundle.estimator is not None and hasattr(xgboost_bundle.estimator, "named_steps"):
        classifier = xgboost_bundle.estimator.named_steps.get("classifier")
    importances = getattr(
        classifier, "feature_importances_", np.zeros(len(xgboost_bundle.feature_columns))
    )
    importance_frame = (
        pd.DataFrame({"feature": xgboost_bundle.feature_columns, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )
    importance_frame.to_csv(tables / "feature_importance.csv", index=False)
    (tables / "figure6_method.json").write_text(
        json.dumps(
            {
                "method": "xgboost_gain_feature_importance",
                "fallback_reason": (
                    "Deterministic gain-based importance is the declared student-scope "
                    "fallback; the locked model is not refit."
                ),
                "model_state_sha256": xgboost_bundle.state_sha256,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    
    # --- Figure 6: XGBoost Feature Importance ---
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    sns.barplot(
        data=importance_frame,
        x="importance",
        y="feature",
        hue="feature",
        palette="Blues_r",
        legend=False,
        ax=ax,
    )
    ax.set_title("XGBoost Normalized Gain Feature Importance")
    ax.set_xlabel("Normalized gain importance")
    plt.tight_layout()
    _save_figure(fig, figures / "figure6_feature_importance.png")

    # --- Figure 7: paired policy effect with episode-level uncertainty ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    metrics_to_plot = [
        ("p95_latency_ms", "p95 latency (ms)"),
        ("loss_percent", "End-to-end loss (%)"),
        ("congestion_duration_s", "Congestion duration (s)"),
        ("backbone_traffic_gb", "Backbone traffic (GB)"),
    ]
    comparison_labels = {"C1-C0": "Centralized\n(C1 − C0)", "D1-D0": "Distributed\n(D1 − D0)"}
    comparison_colors = {"C1-C0": "#4C78A8", "D1-D0": "#F58518"}
    for ax, (metric_key, metric_title) in zip(axes.flatten(), metrics_to_plot):
        metric_effects = (
            paired_effects.loc[paired_effects["metric"] == metric_key]
            .set_index("comparison")
            .reindex(policy_comparisons)
        )
        means = metric_effects["mean_delta"].to_numpy(dtype=float)
        lows = metric_effects["ci95_low"].to_numpy(dtype=float)
        highs = metric_effects["ci95_high"].to_numpy(dtype=float)
        positions = np.arange(len(policy_comparisons))
        for position, comparison in enumerate(policy_comparisons):
            episode_values = paired_deltas.loc[
                paired_deltas["comparison"] == comparison, f"delta_{metric_key}"
            ].to_numpy(dtype=float)
            episode_values = episode_values[np.isfinite(episode_values)]
            jitter = np.linspace(-0.08, 0.08, episode_values.size) if episode_values.size > 1 else np.array([0.0])
            ax.scatter(
                np.full(episode_values.size, position) + jitter,
                episode_values,
                color=comparison_colors[comparison],
                edgecolor="white",
                linewidth=0.6,
                s=42,
                alpha=0.9,
                zorder=3,
            )
        ax.errorbar(
            positions,
            means,
            yerr=np.vstack((means - lows, highs - means)),
            fmt="D",
            markersize=6,
            markerfacecolor="white",
            markeredgecolor="#252525",
            ecolor="#252525",
            capsize=5,
            linewidth=1.4,
            zorder=5,
        )
        ax.axhline(0, color="#333333", linewidth=1, zorder=2)
        ax.set_xticks(positions, [comparison_labels[item] for item in policy_comparisons])
        ax.set_title(metric_title, fontsize=12, weight="bold")
        ax.set_xlabel("Paired policy comparison")
        ax.set_ylabel("Paired change")
        ax.grid(True, alpha=0.3)

    paired_episode_count = int(paired_effects["paired_episodes"].max()) if not paired_effects.empty else 0
    fig.suptitle(
        "Paired Policy Effect on Locked Episodes (mean Δ with 95% bootstrap CI)",
        fontsize=14,
        weight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        f"Points = locked episodes; diamonds/error bars = mean and 95% episode-bootstrap CI (n = {paired_episode_count}). "
        "Δ = policy − no-policy; negative values indicate lower system impact. Intervals are not repeated-seed intervals.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.95))
    _save_figure(fig, figures / "figure7_system_impact.png")
    shutil.copy2(
        figures / "figure7_system_impact.png",
        figures / "figure7_paired_policy_effects.png",
    )
    shutil.copy2(
        figures / "figure7_system_impact.svg",
        figures / "figure7_paired_policy_effects.svg",
    )

    # --- Figure 8: Cross-Topology Generalization Matrix (PR-AUC & F1 with Formatted Labels) ---
    matrix_pr = cross_topology.pivot(index="train_scope", columns="test_topology", values="pr_auc").reindex(
        index=["centralized", "distributed", "combined"], columns=["centralized", "distributed"]
    )
    matrix_f1 = cross_topology.pivot(index="train_scope", columns="test_topology", values="f1").reindex(
        index=["centralized", "distributed", "combined"], columns=["centralized", "distributed"]
    )
    
    c_to_d_drop_pr = (_as_float(matrix_pr.loc["centralized", "centralized"]) - _as_float(matrix_pr.loc["centralized", "distributed"])) * 100.0
    d_to_c_drop_pr = (_as_float(matrix_pr.loc["distributed", "distributed"]) - _as_float(matrix_pr.loc["distributed", "centralized"])) * 100.0
    
    c_to_d_drop_f1 = (_as_float(matrix_f1.loc["centralized", "centralized"]) - _as_float(matrix_f1.loc["centralized", "distributed"])) * 100.0
    d_to_c_drop_f1 = (_as_float(matrix_f1.loc["distributed", "distributed"]) - _as_float(matrix_f1.loc["distributed", "centralized"])) * 100.0

    clean_train_labels = ["Train Centralized", "Train Distributed", "Train Combined"]
    clean_test_labels = ["Test Centralized", "Test Distributed"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)
    
    sns.heatmap(matrix_pr, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, ax=axes[0], yticklabels=clean_train_labels, xticklabels=clean_test_labels, cbar_kws={"label": "PR-AUC"})
    axes[0].set_title(f"PR-AUC\nC -> D drop: {c_to_d_drop_pr:.1f} pts | D -> C drop: {d_to_c_drop_pr:.1f} pts", fontsize=11, weight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")

    sns.heatmap(matrix_f1, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, ax=axes[1], yticklabels=clean_train_labels, xticklabels=clean_test_labels, cbar_kws={"label": "F1-Score"})
    axes[1].set_title(f"F1-Score\nC -> D drop: {c_to_d_drop_f1:.1f} pts | D -> C drop: {d_to_c_drop_f1:.1f} pts", fontsize=11, weight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    
    fig.suptitle("Cross-Topology Generalization (PR-AUC & F1)", fontsize=14, weight="bold", y=0.98)
    plt.tight_layout()
    _save_figure(fig, figures / "figure8_cross_topology.png")

    # --- Figure 9: Sensitivity to Prediction Horizon (2-Panel Subplots) ---
    horizon_xgb = horizon.loc[horizon["model"] == "xgboost"].sort_values(["topology", "horizon"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    coverage_axis = ax2.twinx()
    
    for topology_key, group in horizon_xgb.groupby("topology", sort=True):
        topo_name = str(topology_key).title()
        ax1.plot(group["horizon"] * 5, group["pr_auc"], marker="o", lw=2, label=f"{topo_name}")
        horizon_seconds = group["horizon"] * 5
        ax2.plot(
            horizon_seconds,
            group["lead_time_median_s"],
            marker="s",
            linestyle="--",
            lw=2,
            label=f"{topo_name} lead time",
        )
        coverage_axis.plot(
            horizon_seconds,
            group["event_coverage"],
            marker="o",
            linestyle=":",
            lw=2,
            alpha=0.85,
            label=f"{topo_name} coverage",
        )
        for seconds, covered, total in zip(
            horizon_seconds,
            group["anticipated_events"],
            group["congestion_events"],
            strict=True,
        ):
            coverage_axis.annotate(
                f"{int(covered)}/{int(total)}",
                (seconds, float(covered) / float(total) if total else 0.0),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        
    ax1.set_title("PR-AUC vs. Prediction Horizon", fontsize=12, weight="bold")
    ax1.set_xlabel("Prediction horizon (s)")
    ax1.set_ylabel("PR-AUC")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left")

    ax2.set_title("Lead Time and Event Coverage vs. Prediction Horizon", fontsize=12, weight="bold")
    ax2.set_xlabel("Prediction horizon (s)")
    ax2.set_ylabel("Median Lead Time (s)")
    ax2.grid(True, alpha=0.3)
    coverage_axis.set_ylabel("Event coverage (anticipated / total)")
    coverage_axis.set_ylim(0, 1.08)
    lead_handles, lead_labels = ax2.get_legend_handles_labels()
    coverage_handles, coverage_labels = coverage_axis.get_legend_handles_labels()
    ax2.legend(
        lead_handles + coverage_handles,
        lead_labels + coverage_labels,
        loc="lower right",
        fontsize=8,
    )

    fig.suptitle("Sensitivity to Prediction Horizon (5 s, 10 s, 20 s)", fontsize=13, weight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_figure(fig, figures / "figure9_horizon_sensitivity.png")

    xgb_mean = float(binary_metrics.loc[binary_metrics["model"] == "xgboost", "pr_auc"].mean())
    baseline_means = (
        binary_metrics.loc[binary_metrics["model"] != "xgboost"]
        .groupby("model")["pr_auc"]
        .mean()
        .sort_values(ascending=False)
    )
    best_comparator_name = str(baseline_means.index[0])
    best_comparator_value = float(baseline_means.iloc[0])
    rq1_result = (
        "supported" if xgb_mean > best_comparator_value else "not supported by this locked run"
    )
    top_feature = str(importance_frame.iloc[0]["feature"])
    top_feature_importance = float(importance_frame.iloc[0]["importance"])
    ablation_index = ablation.set_index("experiment")
    full_pr_auc = _as_float(ablation_index.loc["full_xgboost", "pr_auc"])
    no_queue_pr_auc = _as_float(ablation_index.loc["no_queue", "pr_auc"])
    no_security_pr_auc = _as_float(ablation_index.loc["no_security_features", "pr_auc"])
    cause_row = cause_metrics.iloc[0]
    cause_macro_f1 = float(cause_row["macro_f1"])
    system_index = system_aggregate.set_index("config")
    c0_row = system_index.loc["C0"]
    d0_row = system_index.loc["D0"]
    backbone_reduction = (
        (_as_float(c0_row["backbone_traffic_gb"]) - _as_float(d0_row["backbone_traffic_gb"]))
        / _as_float(c0_row["backbone_traffic_gb"])
        * 100.0
        if _as_float(c0_row["backbone_traffic_gb"]) > 0
        else 0.0
    )
    c_to_d = _as_float(matrix_pr.loc["centralized", "distributed"])
    d_to_c = _as_float(matrix_pr.loc["distributed", "centralized"])
    combined_to_c = _as_float(matrix_pr.loc["combined", "centralized"])
    combined_to_d = _as_float(matrix_pr.loc["combined", "distributed"])
    horizon_means = (
        horizon_xgb.assign(horizon_s=horizon_xgb["horizon"] * 5)
        .groupby("horizon_s")["pr_auc"]
        .mean()
        .sort_index()
    )
    horizon_summary = ", ".join(
        f"{int(str(seconds))} s={value:.6f}" for seconds, value in horizon_means.items()
    )
    captions = [
        "# Figure and table captions",
        "",
        "- Figure 1: Real generated object-window cause counts by frozen split and topology.",
        (
            "- Figure 2 / Table 2: Sealed locked-test prediction only; thresholds "
            "were selected on validation."
        ),
        (
            "- Companion early-warning table: frozen scores evaluated only before "
            "congestion, with a new onset inside the prediction horizon as the target."
        ),
        (
            "- Figure 3: First sorted eligible locked episode/topology/object with "
            "pre/post onset context, frozen binary threshold, and true/predicted cause strips; "
            "no cherry-picking."
        ),
        (
            "- Figure 4: Event-matched lead-time samples plus event coverage; missed "
            "event counts remain explicit in Table 2."
        ),
        (
            "- Figure 5: Row-normalized four-class locked-test cause confusion matrix "
            "with class supports in the companion table."
        ),
        "- Figure 6: Frozen XGBoost gain importance fallback; it is associative, not causal.",
        (
            "- Figure 7 / Table 3: Same locked episode workload and equal active compute "
            "budget for C0/C1/D0/D1; latency/loss are end-to-end, backbone GB is unique "
            "ingress volume, and GB-hop is a separate path-work ledger."
        ),
        (
            "- Figure 8: Train scope C, D, or C+D used the same declared search grid and "
            "was evaluated on untouched C/D locked partitions."
        ),
        (
            "- Figure 9: Independently trained 5/10/20-second target horizons using "
            "the same episode split."
        ),
        "",
    ]
    (output / "CAPTIONS.md").write_text("\n".join(captions), encoding="utf-8")

    report_lines = [
        "# Real experimental result artifact index",
        "",
        "All tables and figures in this directory were generated from the sealed run artifacts.",
        "No value was copied from the DEMO result-guide Markdown.",
        "",
        "## Research-question evidence",
        "",
        "- RQ1: Figure 2 and Table 2.",
        "- RQ2: Figure 6 and Table 4.",
        "- RQ3: Figures 3 and 5.",
        "- RQ4: Table 3 and Figure 7.",
        "- RQ5: Figure 8.",
        "",
        "## Locked-run result statement",
        "",
        (
            f"RQ1 is **{rq1_result}**: mean locked PR-AUC for XGBoost is "
            f"{xgb_mean:.6f}; the strongest non-XGBoost comparator is "
            f"{best_comparator_name} at {best_comparator_value:.6f}."
        ),
        (
            "This statement reports the observed sealed result even when it differs "
            "from the desired hypothesis."
        ),
        "",
        "## RQ1-RQ5 observed conclusions",
        "",
        (
            f"- RQ1 — **{rq1_result}**. XGBoost mean PR-AUC={xgb_mean:.6f}; "
            f"best comparator {best_comparator_name}={best_comparator_value:.6f}."
        ),
        (
            f"- RQ2 — The strongest frozen gain association is `{top_feature}` "
            f"({top_feature_importance:.6f}). With primary hyperparameters held fixed, "
            f"full/no-queue/no-security PR-AUC={full_pr_auc:.6f}/{no_queue_pr_auc:.6f}/"
            f"{no_security_pr_auc:.6f}; this is associative ablation evidence, not causality."
        ),
        (
            f"- RQ3 — Leakage-free four-class cause XGBoost Macro-F1={cause_macro_f1:.6f}; "
            "all class supports and topology/scenario slices are in companion tables."
        ),
        (
            f"- RQ4 — Under equal active compute budget, D0 changes mean p95 end-to-end "
            f"latency from {_as_float(c0_row['p95_latency_ms']):.3f} to "
            f"{_as_float(d0_row['p95_latency_ms']):.3f} ms and loss from "
            f"{_as_float(c0_row['loss_percent']):.3f}% to "
            f"{_as_float(d0_row['loss_percent']):.3f}%; "
            f"unique backbone ingress volume falls {backbone_reduction:.3f}%."
        ),
        (
            f"- RQ5 — Topology shift is asymmetric: C-to-D PR-AUC={c_to_d:.6f}, "
            f"D-to-C={d_to_c:.6f}; combined training gives C={combined_to_c:.6f}, "
            f"D={combined_to_d:.6f}."
        ),
        (
            f"- Horizon sensitivity — mean XGBoost PR-AUC by frozen target: "
            f"{horizon_summary}; lead time and event coverage must be read together."
        ),
        "",
        "## Metric semantics",
        "",
        (
            "- False alarms/hour uses elapsed topology system-hours; object-hours are "
            "reported separately."
        ),
        (
            "- End-to-end loss propagates admitted bytes hop by hop and includes terminal "
            "compute rejection; shaped bytes are reported separately rather than relabeled as loss."
        ),
        (
            "- End-to-end latency is the documented deterministic byte-weighted approximation "
            "of path propagation, link queueing, and terminal processing delay."
        ),
    ]
    (output / "RESULTS_README.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    files = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {
        "kind": "real_sealed_results",
        "demo_values_used": False,
        "lineage": lineage or {},
        "figure3_selection": selection_payload,
        "figure6_method": "xgboost_gain_feature_importance",
        "metric_semantics": {
            "false_alarm_rate": "events per elapsed topology system-hour",
            "latency": "traffic-conserving end-to-end weighted approximation",
            "loss": "end-to-end lost bytes divided by network-offered bytes",
            "backbone_traffic": "unique first-backbone-ingress bytes",
            "bandwidth_hop": "sum of admitted bytes over physical hops",
        },
        "files": [
            {
                "path": path.relative_to(output).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        ],
    }
    (output / "results_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if publish_root is not None:
        publish = Path(publish_root) / output.parent.name
        if publish.exists():
            raise FileExistsError(f"published result namespace already exists: {publish}")
        for directory in ("tables", "figures"):
            publish_target = publish / directory
            publish_target.mkdir(parents=True, exist_ok=True)
            for source in (output / directory).glob("*"):
                if source.is_file():
                    shutil.copy2(source, publish_target / source.name)
        shutil.copy2(output / "RESULTS_README.md", publish / "RESULTS_README.md")
        shutil.copy2(output / "CAPTIONS.md", publish / "CAPTIONS.md")
        shutil.copy2(output / "results_manifest.json", publish / "results_manifest.json")
    return {path.name: str(path) for path in files}
