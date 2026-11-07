from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from congestion_lab.reporting.generate import (
    _early_warning_metrics,
    _operating_point,
    _wilson_proportion_ci,
)


@pytest.mark.parametrize(
    ("successes", "total", "proportion"),
    [(0, 4, 0.0), (1, 7, 1 / 7), (4, 4, 1.0)],
)
def test_wilson_interval_contains_observed_proportion(
    successes: int, total: int, proportion: float
) -> None:
    lower, upper = _wilson_proportion_ci(successes, total)

    assert 0.0 <= lower <= proportion <= upper <= 1.0
    assert np.isfinite([lower, upper]).all()


def test_wilson_interval_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError, match="successes"):
        _wilson_proportion_ci(5, 4)


def test_operating_point_marks_abstention_as_undefined_precision() -> None:
    precision, recall, predicted_positive = _operating_point(
        np.array([0, 1]),
        np.array([0.0, 1.0]),
        np.nextafter(1.0, np.inf),
    )

    assert precision is None
    assert recall == 0.0
    assert predicted_positive == 0


def test_early_warning_metrics_exclude_already_congested_rows() -> None:
    keys = {
        "episode_id": ["E1"] * 4,
        "topology": ["centralized"] * 4,
        "object_id": ["L1"] * 4,
        "window_index": [0, 1, 2, 3],
    }
    features = pd.DataFrame(
        {
            **keys,
            "congestion_state": [False, False, True, False],
            "congestion_within_h2": [False, True, False, True],
        }
    )
    predictions = pd.DataFrame(
        {
            **keys,
            "model": ["xgboost"] * 4,
            "score": [0.1, 0.9, 0.95, 0.2],
            "prediction": [0, 1, 1, 0],
        }
    )
    binary_metrics = pd.DataFrame(
        [{"model": "xgboost", "topology": "centralized", "horizon": 2}]
    )

    result = _early_warning_metrics(features, predictions, binary_metrics).iloc[0]

    assert result["at_risk_rows"] == 3
    assert result["future_onset_rows"] == 2
    assert result["predicted_positive_rows"] == 1
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5
    assert np.isclose(result["f1"], 2.0 / 3.0)
