"""Observable active-flow distribution statistics.

Only fields available in raw flow records are used here.  Attack labels and attack
families are deliberately absent so the resulting model features cannot branch on
ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DistributionSeries:
    count: np.ndarray
    entropy_bits: np.ndarray
    top_share: np.ndarray


def _active_key_bytes(
    starts: np.ndarray,
    ends: np.ndarray,
    values: np.ndarray,
    duration: int,
) -> np.ndarray:
    diff = np.zeros(duration + 1, dtype=float)
    np.add.at(diff, starts, values)
    valid_ends = ends < len(diff)
    np.add.at(diff, ends[valid_ends], -values[valid_ends])
    return np.cumsum(diff[:-1])


def active_distribution_statistics(
    frame: pd.DataFrame,
    *,
    key_column: str,
    duration: int,
) -> DistributionSeries:
    """Compute unique-active count, Shannon entropy and largest byte share per tick."""

    count = np.zeros(duration, dtype=float)
    total = np.zeros(duration, dtype=float)
    weighted_log = np.zeros(duration, dtype=float)
    top = np.zeros(duration, dtype=float)
    if frame.empty:
        return DistributionSeries(count=count, entropy_bits=total, top_share=top)

    starts_all = frame["arrival_time_s"].to_numpy(dtype=int)
    raw_ends = (frame["arrival_time_s"] + frame["duration_s"]).to_numpy(dtype=float)
    ends_all = np.minimum(duration, np.ceil(raw_ends).astype(int))
    values_all = frame["mean_bitrate_bps"].to_numpy(dtype=float) / 8.0
    for _, indices in frame.groupby(key_column, sort=True).indices.items():
        selected = np.asarray(indices, dtype=int)
        active_bytes = _active_key_bytes(
            starts_all[selected],
            ends_all[selected],
            values_all[selected],
            duration,
        )
        positive = active_bytes > 0.0
        count += positive
        total += active_bytes
        weighted_log += np.where(
            positive,
            active_bytes * np.log2(np.maximum(active_bytes, 1.0)),
            0.0,
        )
        top = np.maximum(top, active_bytes)

    entropy = np.zeros(duration, dtype=float)
    positive_total = total > 0.0
    entropy[positive_total] = (
        np.log2(total[positive_total]) - weighted_log[positive_total] / total[positive_total]
    )
    top_share = np.divide(top, total, out=np.zeros_like(top), where=positive_total)
    return DistributionSeries(count=count, entropy_bits=entropy, top_share=top_share)


def mixed_distribution(
    *,
    totals: list[float],
    counts: list[float],
    entropies: list[float],
    top_shares: list[float],
) -> tuple[float, float, float]:
    """Combine disjoint region source groups using byte-weighted entropy."""

    overall = float(sum(max(value, 0.0) for value in totals))
    if overall <= 0.0:
        return 0.0, 0.0, 0.0
    weights = np.asarray([max(value, 0.0) / overall for value in totals], dtype=float)
    positive = weights > 0.0
    region_entropy = float(-(weights[positive] * np.log2(weights[positive])).sum())
    within_entropy = float(
        sum(weight * max(entropy, 0.0) for weight, entropy in zip(weights, entropies, strict=True))
    )
    top_share = max(
        (weight * max(share, 0.0) for weight, share in zip(weights, top_shares, strict=True)),
        default=0.0,
    )
    return float(sum(counts)), region_entropy + within_entropy, float(top_share)
