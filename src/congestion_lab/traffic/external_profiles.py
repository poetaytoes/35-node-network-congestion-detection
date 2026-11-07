"""Adapters that reduce external traces to simulator-safe statistical profiles."""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from congestion_lab.data.artifacts import sha256_file


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        return " ".join(" ".join(self.parts).split())


def verify_external_file(
    path: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    source = Path(path).resolve()
    actual = sha256_file(source)
    return {
        "path": str(source),
        "expected_sha256": expected_sha256.lower(),
        "actual_sha256": actual,
        "bytes": source.stat().st_size,
        "status": "pass" if actual == expected_sha256.lower() else "fail",
    }


def profile_mawi_trace_info_html(
    path: str | Path,
    *,
    source_url: str,
) -> dict[str, Any]:
    """Extract privacy-safe aggregate statistics from an official MAWI info page."""
    source = Path(path).resolve()
    parser = _TextExtractor()
    parser.feed(source.read_text(encoding="utf-8", errors="replace"))
    text = parser.text()

    def number(pattern: str) -> float:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"MAWI trace-info field did not match: {pattern}")
        return float(match.group(1))

    trace_seconds = number(r"TotalTime:\s*([0-9.]+)\s*seconds")
    average_mbps = number(r"AvgRate:\s*([0-9.]+)Mbps")
    stddev_mbps = number(r"stddev:\s*([0-9.]+)M")
    packets = int(number(r"# of packets:\s*([0-9]+)"))
    tcp_ratio = number(r"\btcp\s+[0-9]+\s*\(\s*([0-9.]+)%\)") / 100.0
    udp_ratio = number(r"\budp\s+[0-9]+\s*\(\s*([0-9.]+)%\)") / 100.0
    icmp_ratio = number(r"\bicmp\s+[0-9]+\s*\(\s*([0-9.]+)%\)") / 100.0
    selected_total = tcp_ratio + udp_ratio + icmp_ratio
    protocol_ratio = {
        "tcp": tcp_ratio / selected_total,
        "udp": udp_ratio / selected_total,
        "icmp": icmp_ratio / selected_total,
    }
    return {
        "source_name": "MAWI samplepoint-F 202608011400 aggregate trace info",
        "source_url": source_url,
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "identity_fields_retained": [],
        "duplicate_rows": 0,
        "trace": {
            "duration_s": trace_seconds,
            "packets": packets,
            "average_rate_mbps": average_mbps,
            "rate_stddev_mbps": stddev_mbps,
        },
        "bitrate_bps": {
            "median": average_mbps * 1_000_000.0,
            "mean": average_mbps * 1_000_000.0,
            "p95": (average_mbps + 1.645 * stddev_mbps) * 1_000_000.0,
            "method": "aggregate-link-rate; scale before flow-level replay",
        },
        "duration_s": {
            "median": 30.0,
            "mean": 35.0,
            "p95": 60.0,
            "method": "declared fallback because the trace-info page has no flow durations",
        },
        "protocol_ratio": protocol_ratio,
    }


def profile_flow_csv(
    path: str | Path,
    *,
    source_name: str,
    bitrate_column: str,
    duration_column: str,
    protocol_column: str,
    label_column: str | None = None,
) -> dict[str, Any]:
    source = Path(path).resolve()
    frame = pd.read_csv(source)
    required = {bitrate_column, duration_column, protocol_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"external profile missing columns: {sorted(missing)}")
    bitrate = pd.to_numeric(frame[bitrate_column], errors="coerce").dropna()
    duration = pd.to_numeric(frame[duration_column], errors="coerce").dropna()
    protocol = frame[protocol_column].astype(str).str.lower()
    duplicate_rows = int(frame.duplicated().sum())
    identity_columns = sorted(
        column
        for column in frame.columns
        if any(token in column.lower() for token in ("ip", "address", "port", "subscriber"))
    )
    profile: dict[str, Any] = {
        "source_name": source_name,
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "rows": len(frame),
        "duplicate_rows": duplicate_rows,
        "identity_columns_discarded": identity_columns,
        "bitrate_bps": {
            "median": float(bitrate.median()),
            "p95": float(bitrate.quantile(0.95)),
            "mean": float(bitrate.mean()),
        },
        "duration_s": {
            "median": float(duration.median()),
            "p95": float(duration.quantile(0.95)),
            "mean": float(duration.mean()),
        },
        "protocol_ratio": protocol.value_counts(normalize=True).sort_index().to_dict(),
    }
    if label_column is not None and label_column in frame:
        profile["label_ratio"] = (
            frame[label_column].astype(str).value_counts(normalize=True).sort_index().to_dict()
        )
    return profile


def write_profile(profile: dict[str, Any], output: str | Path) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    return target


def profile_to_synthetic_sample(
    profile: dict[str, Any],
    output: str | Path,
    *,
    rows: int = 512,
    seed: int = 20260814,
    bitrate_scale: float = 1.0,
) -> Path:
    if bitrate_scale <= 0.0:
        raise ValueError("bitrate_scale must be positive")
    rng = np.random.default_rng(seed)
    median_bitrate = max(float(profile["bitrate_bps"]["median"]), 1.0)
    median_duration = max(float(profile["duration_s"]["median"]), 1.0)
    protocols = list(profile["protocol_ratio"])
    probabilities = np.asarray(list(profile["protocol_ratio"].values()), dtype=float)
    probabilities /= probabilities.sum()
    frame = pd.DataFrame(
        {
            "mean_bitrate_bps": rng.lognormal(np.log(median_bitrate * bitrate_scale), 0.45, rows),
            "duration_s": rng.lognormal(np.log(median_duration), 0.35, rows),
            "protocol_family": rng.choice(protocols, rows, p=probabilities),
            "traffic_source": str(profile["source_name"]),
        }
    )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return target


def external_sample_to_events(
    sample_path: str | Path,
    *,
    episode_id: str,
    duration_s: int,
    regions: list[str],
    seed: int = 20260814,
) -> pd.DataFrame:
    """Convert a privacy-safe profile sample into the simulator raw-event contract."""
    sample = pd.read_csv(sample_path)
    required = {"mean_bitrate_bps", "duration_s", "protocol_family", "traffic_source"}
    missing = required - set(sample.columns)
    if missing:
        raise ValueError(f"external replay sample missing columns: {sorted(missing)}")
    if not regions:
        raise ValueError("external replay requires at least one region")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for index, (_, row) in enumerate(sample.reset_index(drop=True).iterrows()):
        flow_duration = max(1.0, min(float(row["duration_s"]), float(duration_s)))
        bitrate = max(1.0, float(row["mean_bitrate_bps"]))
        protocol = str(row["protocol_family"]).lower()
        rows.append(
            {
                "episode_id": episode_id,
                "event_id": f"{episode_id}-{index:07d}",
                "arrival_time_s": int(index % duration_s),
                "source_region": regions[index % len(regions)],
                "service_id": "svc_external_profile",
                "service_class": "low" if index % 3 == 0 else "high",
                "duration_s": flow_duration,
                "mean_bitrate_bps": bitrate,
                "flow_size_bytes": bitrate * flow_duration / 8.0,
                "packet_size_profile": "mixed",
                "protocol_family": protocol,
                "source_group_id": f"external-group-{index % 64:02d}",
                "is_attack_ground_truth": False,
                "attack_family": "none",
                "traffic_source": str(row["traffic_source"]),
                "compute_cost_bytes": float(rng.choice([70_000.0, 260_000.0])),
                "scenario_family": "external_profile_replay",
                "seed": seed,
                "seed_path": f"external/{episode_id}/{seed}",
            }
        )
    return pd.DataFrame(rows)
