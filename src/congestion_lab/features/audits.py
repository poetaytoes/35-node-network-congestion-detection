"""Statistical and provenance leakage audits beyond the name allowlist."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from congestion_lab.features.builder import FEATURE_COLUMNS
from congestion_lab.features.leakage import leakage_audit


def advanced_leakage_audit(
    features: pd.DataFrame,
    bundles: Mapping[str, Any],
    *,
    random_state: int = 20260814,
) -> dict[str, Any]:
    """Run shuffled-label, correlation, episode-proxy, and fit-scope audits."""
    target = "future_congestion_h2"
    valid = "target_valid_h2"
    train = features.loc[(features["split"] == "train") & features[valid]].copy()
    validation = features.loc[(features["split"] == "validation") & features[valid]].copy()
    x_train = train.loc[:, list(FEATURE_COLUMNS)]
    x_validation = validation.loc[:, list(FEATURE_COLUMNS)]
    y_train = train[target].astype(int).to_numpy()
    y_validation = validation[target].astype(int).to_numpy()

    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(y_train)
    shuffle_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=random_state),
            ),
        ]
    )
    shuffle_model.fit(x_train, shuffled)
    shuffle_score = shuffle_model.predict_proba(x_validation)[:, 1]
    shuffle_pr_auc = float(average_precision_score(y_validation, shuffle_score))
    validation_prevalence = float(y_validation.mean())
    shuffle_limit = max(0.35, validation_prevalence + 0.25)

    correlations = train.loc[:, [*FEATURE_COLUMNS, target]].corr(numeric_only=True)[target]
    high_correlations = {
        str(feature): float(value)
        for feature, value in correlations.drop(target).abs().sort_values(ascending=False).items()
        if float(value) >= 0.995
    }

    episode_x = train.loc[:, list(FEATURE_COLUMNS)]
    episode_y = train["episode_id"].astype(str)
    proxy_accuracy: float | None = None
    if episode_y.value_counts().min() >= 2:
        proxy_train_x, proxy_test_x, proxy_train_y, proxy_test_y = train_test_split(
            episode_x,
            episode_y,
            test_size=0.25,
            random_state=random_state,
            stratify=episode_y,
        )
        proxy = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=8,
                        n_jobs=1,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        proxy.fit(proxy_train_x, proxy_train_y)
        proxy_accuracy = float(proxy.score(proxy_test_x, proxy_test_y))

    expected_train = set(features.loc[features["split"] == "train", "episode_id"].astype(str))
    expected_validation = set(
        features.loc[features["split"] == "validation", "episode_id"].astype(str)
    )
    locked = set(features.loc[features["split"] == "locked_test", "episode_id"].astype(str))
    fit_scope_issues: list[str] = []
    preprocessing_hashes: dict[str, str] = {}
    for name, bundle in bundles.items():
        training_ids = set(str(value) for value in bundle.training_episode_ids)
        validation_ids = set(str(value) for value in bundle.validation_episode_ids)
        if training_ids - expected_train:
            fit_scope_issues.append(f"{name}:unexpected-train-episodes")
        if validation_ids - expected_validation:
            fit_scope_issues.append(f"{name}:unexpected-validation-episodes")
        if (training_ids | validation_ids) & locked:
            fit_scope_issues.append(f"{name}:locked-test-fit-overlap")
        if not bundle.state_sha256 or not bundle.fit_data_sha256:
            fit_scope_issues.append(f"{name}:missing-fit-state-hash")
        preprocessing_hashes[str(name)] = str(bundle.state_sha256)

    name_report = leakage_audit(features)
    blockers = list(fit_scope_issues)
    if name_report["status"] != "pass":
        blockers.append("feature-name-allowlist")
    if shuffle_pr_auc > shuffle_limit:
        blockers.append("shuffled-label-performance-too-high")
    return {
        "status": "pass" if not blockers else "fail",
        "blocking_issues": blockers,
        "feature_name_audit": name_report,
        "shuffle_label": {
            "validation_prevalence": validation_prevalence,
            "pr_auc": shuffle_pr_auc,
            "pass_limit": shuffle_limit,
            "status": "pass" if shuffle_pr_auc <= shuffle_limit else "fail",
        },
        "high_absolute_correlations": high_correlations,
        "episode_id_proxy_accuracy_within_train": proxy_accuracy,
        "episode_id_proxy_interpretation": (
            "diagnostic warning only; scenario dynamics can legitimately identify episodes"
        ),
        "fit_scope_issues": fit_scope_issues,
        "preprocessing_state_hashes": preprocessing_hashes,
    }
