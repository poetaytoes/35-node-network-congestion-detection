from __future__ import annotations

import numpy as np
import pandas as pd

from congestion_lab.config.models import ExperimentConfig
from congestion_lab.evaluation.metrics import _event_statistics
from congestion_lab.features import relabel_feature_table


def _label_fixture() -> pd.DataFrame:
    util = [0.1, 0.85, 0.90, 0.95, 0.2, 0.1, 0.1]
    attack = [0.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0]
    return pd.DataFrame(
        {
            "episode_id": ["E01"] * len(util),
            "topology": ["centralized"] * len(util),
            "object_id": ["B1--B2"] * len(util),
            "window_index": range(len(util)),
            "util_mean": util,
            "queue_ratio": [0.0] * len(util),
            "loss_rate": [0.0] * len(util),
            "delay_p95_ms": [1.0] * len(util),
            "attack_bytes_window": attack,
        }
    )


def test_golden_persistence_future_shift_and_cause_precedence(
    pilot_config: ExperimentConfig,
) -> None:
    labeled = relabel_feature_table(_label_fixture(), pilot_config)
    assert labeled["congestion_state"].tolist() == [False, False, True, True, False, False, False]
    assert labeled["congestion_onset"].tolist() == [False, False, True, False, False, False, False]
    assert labeled["congestion_offset"].tolist() == [False, False, False, False, True, False, False]
    assert labeled["future_congestion_h2"].astype(object).tolist()[:5] == [
        True,
        True,
        False,
        False,
        False,
    ]
    assert labeled["target_valid_h2"].tolist() == [True, True, True, True, True, False, False]
    assert labeled.loc[1, "cause_label"] == "attack_associated_congestion"
    assert labeled.loc[3, "cause_label"] == "attack_without_congestion"


def test_alarm_collapse_lead_time_and_missed_event_behavior() -> None:
    group = pd.DataFrame(
        {
            "congestion_onset": [False, False, False, True, False, False, True],
            "window_start_s": [0, 5, 10, 15, 20, 25, 30],
        }
    )
    stats = _event_statistics(group, np.array([1, 1, 0, 0, 0, 0, 0]), window_s=5)
    assert stats["alarm_events"] == 1
    assert stats["congestion_events"] == 2
    assert stats["anticipated_events"] == 1
    assert stats["lead_times"] == [15.0]
    assert stats["false_alarm_events"] == 0


def test_lead_time_uses_the_matched_recent_alarm_not_an_old_alarm() -> None:
    group = pd.DataFrame(
        {
            "congestion_onset": [False, False, False, False, False, True],
            "window_start_s": [0, 5, 10, 15, 20, 25],
        }
    )
    stats = _event_statistics(group, np.array([1, 0, 0, 0, 1, 0]), window_s=5, lookback_windows=2)
    assert stats["anticipated_events"] == 1
    assert stats["lead_times"] == [5.0]
