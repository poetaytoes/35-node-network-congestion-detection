"""Small, reproducible model grids and serializable bundles."""

from __future__ import annotations

import itertools
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from congestion_lab.data.artifacts import canonical_json_hash, sha256_bytes, stable_frame_hash
from congestion_lab.features.builder import FEATURE_COLUMNS
from congestion_lab.features.leakage import assert_no_leakage


@dataclass
class ModelBundle:
    name: str
    task: str
    estimator: BaseEstimator | None
    feature_columns: tuple[str, ...]
    threshold: float
    parameters: dict[str, Any]
    train_scope: str = "combined"
    horizon: int = 2
    training_episode_ids: tuple[str, ...] = ()
    validation_episode_ids: tuple[str, ...] = ()
    fit_data_sha256: str = ""
    state_sha256: str = ""
    threshold_selection: dict[str, float | int | bool] | None = None

    def _feature_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(self.feature_columns) - set(frame.columns))
        if missing:
            raise ValueError(f"model schema mismatch; missing features: {missing}")
        if frame.columns.duplicated().any():
            raise ValueError("model schema mismatch; duplicate dataframe columns")
        return frame.loc[:, list(self.feature_columns)]

    def predict_score(self, frame: pd.DataFrame) -> np.ndarray:
        values = self._feature_frame(frame)
        if self.estimator is None:
            return baseline_score(self.name, frame)
        probabilities = self.estimator.predict_proba(values)
        if self.task == "binary":
            return np.asarray(probabilities[:, 1], dtype=float)
        return np.asarray(probabilities, dtype=float)

    def predict_label(self, frame: pd.DataFrame) -> np.ndarray:
        scores = self.predict_score(frame)
        if self.task == "binary":
            return (scores >= self.threshold).astype(int)
        return np.asarray(np.argmax(scores, axis=1), dtype=int)


def baseline_score(name: str, frame: pd.DataFrame) -> np.ndarray:
    if name == "threshold":
        risk = np.maximum(
            frame["util_end"].to_numpy(dtype=float) / 0.8,
            frame["queue_ratio"].to_numpy(dtype=float) / 0.7,
        )
        return np.clip(risk, 0.0, 1.0)
    if name == "persistence":
        return frame["congestion_state"].to_numpy(dtype=float)
    if name == "ewma":
        return np.clip(frame["util_rolling3"].to_numpy(dtype=float) / 0.8, 0.0, 1.0)
    raise ValueError(f"unknown baseline: {name}")


def _alarm_onsets(prediction: np.ndarray) -> np.ndarray:
    previous = np.concatenate(([False], prediction[:-1].astype(bool)))
    return prediction.astype(bool) & ~previous


def _validation_alarm_metrics(
    metadata: pd.DataFrame,
    prediction: np.ndarray,
    *,
    horizon: int,
    window_s: int,
) -> dict[str, float | int]:
    """Use the evaluation event definition for validation threshold selection."""
    required = {"episode_id", "topology", "object_id", "window_index"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"threshold selection metadata missing: {sorted(missing)}")
    frame = metadata.loc[:, list(required)].copy()
    if "congestion_onset" in metadata:
        frame["congestion_onset"] = metadata["congestion_onset"].astype(bool)
    elif "congestion_state" in metadata:
        ordered = metadata.loc[:, [*required, "congestion_state"]].copy()
        ordered = ordered.sort_values(["episode_id", "topology", "object_id", "window_index"])
        ordered["congestion_onset"] = ordered["congestion_state"].astype(bool) & ~ordered.groupby(
            ["episode_id", "topology", "object_id"], sort=False
        )["congestion_state"].shift(1).fillna(False).astype(bool)
        frame = ordered.loc[:, [*required, "congestion_onset"]].reindex(metadata.index)
    else:
        raise ValueError("threshold selection metadata requires congestion_onset or congestion_state")
    frame["prediction"] = np.asarray(prediction, dtype=bool)
    false_alarm_events = 0
    alarm_events = 0
    for _, group in frame.groupby(["episode_id", "topology", "object_id"], sort=True):
        group = group.sort_values("window_index")
        alarm_indices = np.flatnonzero(_alarm_onsets(group["prediction"].to_numpy()))
        onset_indices = np.flatnonzero(group["congestion_onset"].to_numpy(dtype=bool))
        alarm_events += len(alarm_indices)
        for alarm_index in alarm_indices:
            future = onset_indices[
                (onset_indices > alarm_index) & (onset_indices <= alarm_index + horizon)
            ]
            if not len(future):
                false_alarm_events += 1
    system_hours = (
        len(frame[["episode_id", "topology", "window_index"]].drop_duplicates())
        * window_s
        / 3600.0
    )
    rate = false_alarm_events / system_hours if system_hours > 0 else 0.0
    return {
        "validation_false_alarm_events": int(false_alarm_events),
        "validation_alarm_events": int(alarm_events),
        "validation_system_hours": float(system_hours),
        "validation_false_alarms_per_hour": float(rate),
    }


def _select_threshold(
    y_true: np.ndarray,
    score: np.ndarray,
    metadata: pd.DataFrame,
    *,
    horizon: int,
    window_s: int = 5,
    fa_budget_per_hour: float = 5.0,
) -> tuple[float, dict[str, float | int | bool]]:
    """Maximize validation F1 subject to event false alarms per system-hour."""
    candidates = np.unique(np.asarray(score, dtype=float))
    candidates = np.concatenate((candidates, [np.nextafter(float(np.max(candidates)), np.inf)]))
    best: tuple[float, float, dict[str, float | int]] | None = None
    for threshold in candidates:
        prediction = np.asarray(score >= threshold, dtype=int)
        operating_point = _validation_alarm_metrics(
            metadata, prediction, horizon=horizon, window_s=window_s
        )
        if float(operating_point["validation_false_alarms_per_hour"]) > fa_budget_per_hour:
            continue
        candidate = (f1_score(y_true, prediction, zero_division=0), float(threshold), operating_point)
        if best is None or candidate[0] > best[0] or (
            candidate[0] == best[0] and candidate[1] < best[1]
        ):
            best = candidate
    if best is None:
        raise RuntimeError("no threshold candidate satisfied the false-alarm budget")
    selection: dict[str, float | int | bool] = {
        **best[2],
        "validation_false_alarm_budget_per_hour": float(fa_budget_per_hour),
        "validation_false_alarm_budget_met": True,
        "validation_f1": float(best[0]),
    }
    return best[1], selection


def _binary_xy(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    horizon: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    target = f"future_congestion_h{horizon}"
    valid = f"target_valid_h{horizon}"
    subset = frame.loc[frame[valid]].copy()
    x = subset.loc[:, list(features)].copy()
    y = np.asarray(subset[target].astype(int).to_numpy(), dtype=int)
    return x, y


def _fit_data_hash(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    target: str,
) -> str:
    columns = ["episode_id", "topology", "object_id", "window_index", *features, target]
    return stable_frame_hash(frame.loc[:, columns].reset_index(drop=True))


def _estimator_state_hash(estimator: BaseEstimator) -> str:
    return sha256_bytes(pickle.dumps(estimator, protocol=5))


def _best_candidate(
    candidates: list[tuple[BaseEstimator, dict[str, Any]]],
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    validation_x: pd.DataFrame,
    validation_y: np.ndarray,
    validation_metadata: pd.DataFrame,
    horizon: int,
    window_s: int,
) -> tuple[BaseEstimator, dict[str, Any], float, float, dict[str, float | int | bool]]:
    best: tuple[float, BaseEstimator, dict[str, Any], float, dict[str, float | int | bool]] | None = None
    for estimator, parameters in candidates:
        estimator.fit(train_x, train_y)
        score = np.asarray(estimator.predict_proba(validation_x)[:, 1], dtype=float)
        pr_auc = float(average_precision_score(validation_y, score))
        threshold, threshold_selection = _select_threshold(
            validation_y,
            score,
            validation_metadata,
            horizon=horizon,
            window_s=window_s,
        )
        if best is None or pr_auc > best[0]:
            best = (pr_auc, estimator, parameters, threshold, threshold_selection)
    if best is None:
        raise RuntimeError("candidate grid was empty")
    return best[1], best[2], best[3], best[0], best[4]


def _xgboost_candidates(
    train_y: np.ndarray,
    *,
    random_state: int,
    compact_grid: bool,
    fixed_parameters: dict[str, Any] | None = None,
) -> list[tuple[BaseEstimator, dict[str, Any]]]:
    positive = max(1, int(train_y.sum()))
    scale_pos_weight = max(1.0, float((len(train_y) - positive) / positive))
    if fixed_parameters is not None:
        grid = [
            (
                int(fixed_parameters["max_depth"]),
                int(fixed_parameters["n_estimators"]),
                float(fixed_parameters["learning_rate"]),
                float(fixed_parameters["subsample"]),
            )
        ]
    else:
        grid = (
            [(3, 200, 0.07, 0.8)]
            if compact_grid
            else list(itertools.product((3, 5), (200, 350), (0.03, 0.07), (0.8, 1.0)))
        )
    candidates: list[tuple[BaseEstimator, dict[str, Any]]] = []
    for depth, estimators, learning_rate, subsample in grid:
        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    XGBClassifier(
                        max_depth=depth,
                        n_estimators=estimators,
                        learning_rate=learning_rate,
                        subsample=subsample,
                        colsample_bytree=0.7,   # Adjusted for regularization
                        reg_alpha=1.5,          # L1 Regularization
                        reg_lambda=3.0,         # L2 Regularization
                        min_child_weight=5,     # Added complexity limit
                        scale_pos_weight=scale_pos_weight,
                        eval_metric="logloss",
                        tree_method="hist",
                        n_jobs=1,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        candidates.append(
            (
                estimator,
                {
                    "max_depth": depth,
                    "n_estimators": estimators,
                    "learning_rate": learning_rate,
                    "subsample": subsample,
                    "scale_pos_weight": scale_pos_weight,
                },
            )
        )
    return candidates


def fit_xgboost_model(
    features: pd.DataFrame,
    *,
    horizon: int = 2,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    train_scope: str = "combined",
    random_state: int = 20260814,
    compact_grid: bool = False,
    fixed_parameters: dict[str, Any] | None = None,
    window_s: int = 5,
) -> tuple[ModelBundle, pd.DataFrame]:
    """Fit only XGBoost for matched-budget robustness experiments."""

    assert_no_leakage(features, feature_columns)
    scoped = (
        features if train_scope == "combined" else features.loc[features["topology"] == train_scope]
    )
    train = scoped.loc[scoped["split"] == "train"]
    validation = scoped.loc[scoped["split"] == "validation"]
    train_x, train_y = _binary_xy(train, feature_columns, horizon)
    validation_x, validation_y = _binary_xy(validation, feature_columns, horizon)
    if len(np.unique(train_y)) < 2 or len(np.unique(validation_y)) < 2:
        raise ValueError(f"binary task requires two classes in train/validation for {train_scope}")
    candidates = _xgboost_candidates(
        train_y,
        random_state=random_state,
        compact_grid=compact_grid,
        fixed_parameters=fixed_parameters,
    )
    validation_valid = validation.loc[validation[f"target_valid_h{horizon}"]].copy()
    estimator, parameters, threshold, pr_auc, threshold_selection = _best_candidate(
        candidates,
        train_x,
        train_y,
        validation_x,
        validation_y,
        validation_valid,
        horizon,
        window_s,
    )
    training_episode_ids = tuple(sorted(str(value) for value in train["episode_id"].unique()))
    validation_episode_ids = tuple(
        sorted(str(value) for value in validation["episode_id"].unique())
    )
    fit_hash = _fit_data_hash(
        train.loc[train[f"target_valid_h{horizon}"]].copy(),
        feature_columns,
        f"future_congestion_h{horizon}",
    )
    bundle = ModelBundle(
        name="xgboost",
        task="binary",
        estimator=estimator,
        feature_columns=feature_columns,
        threshold=threshold,
        parameters=parameters,
        train_scope=train_scope,
        horizon=horizon,
        training_episode_ids=training_episode_ids,
        validation_episode_ids=validation_episode_ids,
        fit_data_sha256=fit_hash,
        state_sha256=_estimator_state_hash(estimator),
        threshold_selection=threshold_selection,
    )
    protocol = (
        "fixed_primary_parameters"
        if fixed_parameters is not None
        else "compact"
        if compact_grid
        else "declared_full"
    )
    selection = pd.DataFrame(
        [
            {
                "model": "xgboost",
                "train_scope": train_scope,
                "horizon": horizon,
                "validation_pr_auc": pr_auc,
                "threshold": threshold,
                **threshold_selection,
                "parameters": repr(parameters),
                "grid_profile": protocol,
                "candidate_count": len(candidates),
            }
        ]
    )
    return bundle, selection


def fit_binary_models(
    features: pd.DataFrame,
    *,
    horizon: int = 2,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    train_scope: str = "combined",
    random_state: int = 20260814,
    compact_grid: bool = False,
    window_s: int = 5,
) -> tuple[dict[str, ModelBundle], pd.DataFrame]:
    assert_no_leakage(features, feature_columns)
    scoped = (
        features if train_scope == "combined" else features.loc[features["topology"] == train_scope]
    )
    train = scoped.loc[scoped["split"] == "train"]
    validation = scoped.loc[scoped["split"] == "validation"]
    train_x, train_y = _binary_xy(train, feature_columns, horizon)
    validation_x, validation_y = _binary_xy(validation, feature_columns, horizon)
    if len(np.unique(train_y)) < 2 or len(np.unique(validation_y)) < 2:
        raise ValueError(f"binary task requires two classes in train/validation for {train_scope}")

    bundles: dict[str, ModelBundle] = {}
    rows: list[dict[str, Any]] = []
    validation_valid = validation.loc[validation[f"target_valid_h{horizon}"]].copy()
    training_episode_ids = tuple(sorted(str(value) for value in train["episode_id"].unique()))
    validation_episode_ids = tuple(
        sorted(str(value) for value in validation["episode_id"].unique())
    )
    fit_hash = _fit_data_hash(
        train.loc[train[f"target_valid_h{horizon}"]].copy(),
        feature_columns,
        f"future_congestion_h{horizon}",
    )
    for name in ("threshold", "persistence", "ewma"):
        score = baseline_score(name, validation_valid)
        threshold, threshold_selection = _select_threshold(
            validation_y,
            score,
            validation_valid,
            horizon=horizon,
            window_s=window_s,
        )
        bundles[name] = ModelBundle(
            name=name,
            task="binary",
            estimator=None,
            feature_columns=feature_columns,
            threshold=threshold,
            parameters={},
            train_scope=train_scope,
            horizon=horizon,
            training_episode_ids=training_episode_ids,
            validation_episode_ids=validation_episode_ids,
            fit_data_sha256=fit_hash,
            state_sha256=canonical_json_hash(
                {
                    "name": name,
                    "threshold": threshold,
                    "features": feature_columns,
                    "fit_data_sha256": fit_hash,
                }
            ),
            threshold_selection=threshold_selection,
        )
        rows.append(
            {
                "model": name,
                "train_scope": train_scope,
                "horizon": horizon,
                "validation_pr_auc": average_precision_score(validation_y, score),
                "threshold": threshold,
                **threshold_selection,
                "parameters": "{}",
            }
        )

    logistic_candidates: list[tuple[BaseEstimator, dict[str, Any]]] = []
    logistic_grid = (1.0,) if compact_grid else (0.1, 1.0, 10.0)
    for c_value in logistic_grid:
        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=c_value,
                        class_weight="balanced",
                        max_iter=1500,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        logistic_candidates.append((estimator, {"C": c_value}))

    rf_candidates: list[tuple[BaseEstimator, dict[str, Any]]] = []
    rf_grid = (
        [(200, 10, 2)]
        if compact_grid
        else list(itertools.product((200, 400), (6, 10, None), (2, 5)))
    )
    for estimators, depth, leaf in rf_grid:
        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=estimators,
                        max_depth=depth,
                        min_samples_leaf=leaf,
                        class_weight="balanced",
                        n_jobs=1,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        rf_candidates.append(
            (estimator, {"n_estimators": estimators, "max_depth": depth, "min_samples_leaf": leaf})
        )

    xgb_candidates = _xgboost_candidates(
        train_y,
        random_state=random_state,
        compact_grid=compact_grid,
    )

    for name, candidates in (
        ("logistic_regression", logistic_candidates),
        ("random_forest", rf_candidates),
        ("xgboost", xgb_candidates),
    ):
        estimator, parameters, threshold, pr_auc, threshold_selection = _best_candidate(
            candidates,
            train_x,
            train_y,
            validation_x,
            validation_y,
            validation_valid,
            horizon,
            window_s,
        )
        bundles[name] = ModelBundle(
            name=name,
            task="binary",
            estimator=estimator,
            feature_columns=feature_columns,
            threshold=threshold,
            parameters=parameters,
            train_scope=train_scope,
            horizon=horizon,
            training_episode_ids=training_episode_ids,
            validation_episode_ids=validation_episode_ids,
            fit_data_sha256=fit_hash,
            state_sha256=_estimator_state_hash(estimator),
            threshold_selection=threshold_selection,
        )
        rows.append(
            {
                "model": name,
                "train_scope": train_scope,
                "horizon": horizon,
                "validation_pr_auc": pr_auc,
                "threshold": threshold,
                **threshold_selection,
                "parameters": repr(parameters),
                "grid_profile": "compact" if compact_grid else "declared_full",
                "candidate_count": len(candidates),
            }
        )
    return bundles, pd.DataFrame(rows)


def fit_cause_models(
    features: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    random_state: int = 20260814,
) -> tuple[dict[str, ModelBundle], pd.DataFrame]:
    assert_no_leakage(features, feature_columns)
    train = features.loc[(features["split"] == "train") & features["model_eligible"]]
    validation = features.loc[(features["split"] == "validation") & features["model_eligible"]]
    train_x = train.loc[:, feature_columns]
    train_y = train["cause_index"].to_numpy(dtype=int)
    validation_x = validation.loc[:, feature_columns]
    validation_y = validation["cause_index"].to_numpy(dtype=int)
    required_classes = set(range(4))
    if (
        set(np.unique(train_y)) != required_classes
        or set(np.unique(validation_y)) != required_classes
    ):
        raise ValueError("cause task requires all four classes in train and validation")
    training_episode_ids = tuple(sorted(str(value) for value in train["episode_id"].unique()))
    validation_episode_ids = tuple(
        sorted(str(value) for value in validation["episode_id"].unique())
    )
    fit_hash = _fit_data_hash(train.copy(), feature_columns, "cause_index")
    
    model_specs: dict[str, BaseEstimator] = {
        "cause_logistic": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "cause_random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=10,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        n_jobs=1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "cause_xgboost": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    XGBClassifier(
                        objective="multi:softprob",
                        num_class=4,
                        max_depth=5,
                        n_estimators=200,      # Adjusted to match new standards
                        learning_rate=0.03,    # Adjusted to prevent overfitting
                        subsample=0.8,
                        colsample_bytree=0.7,  # Adjusted to force feature exploration
                        reg_alpha=1.5,         # Added L1 Penalty
                        reg_lambda=3.0,        # Added L2 Penalty
                        min_child_weight=5,    # Added complexity constraint
                        eval_metric="mlogloss",
                        tree_method="hist",
                        n_jobs=1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
    
    bundles: dict[str, ModelBundle] = {}
    rows: list[dict[str, Any]] = []

    # Calculate balanced sample weights for multi-class XGBoost imbalance
    sample_weights = compute_sample_weight(class_weight="balanced", y=train_y)

    for name, estimator in model_specs.items():
        if "xgboost" in name:
            # Pass the computed weights to the XGBoost step in the pipeline
            estimator.fit(train_x, train_y, classifier__sample_weight=sample_weights)
        else:
            estimator.fit(train_x, train_y)
            
        prediction = estimator.predict(validation_x)
        macro_f1 = f1_score(validation_y, prediction, average="macro", zero_division=0)
        bundles[name] = ModelBundle(
            name=name,
            task="cause",
            estimator=estimator,
            feature_columns=feature_columns,
            threshold=0.0,
            parameters={},
            training_episode_ids=training_episode_ids,
            validation_episode_ids=validation_episode_ids,
            fit_data_sha256=fit_hash,
            state_sha256=_estimator_state_hash(estimator),
        )
        rows.append({"model": name, "validation_macro_f1": float(macro_f1)})
    return bundles, pd.DataFrame(rows)


def save_bundle(bundle: ModelBundle, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, target)
    return target


def load_bundle(path: str | Path) -> ModelBundle:
    bundle = joblib.load(Path(path))
    if not isinstance(bundle, ModelBundle):
        raise TypeError("model artifact is not a ModelBundle")
    return bundle


def model_card(bundle: ModelBundle) -> str:
    """Render a compact auditable Markdown model card."""
    return "\n".join(
        [
            f"# Model card: {bundle.name}",
            "",
            f"- Task: `{bundle.task}`",
            f"- Train scope: `{bundle.train_scope}`",
            f"- Horizon windows: `{bundle.horizon}`",
            f"- Frozen decision threshold: `{bundle.threshold:.12g}`",
            f"- Validation alarm operating point: `{bundle.threshold_selection!r}`",
            f"- Feature count: `{len(bundle.feature_columns)}`",
            f"- Train episodes: `{', '.join(bundle.training_episode_ids)}`",
            f"- Validation episodes: `{', '.join(bundle.validation_episode_ids)}`",
            f"- Fit-data SHA-256: `{bundle.fit_data_sha256}`",
            f"- Fitted-state SHA-256: `{bundle.state_sha256}`",
            f"- Parameters: `{bundle.parameters!r}`",
            "",
            "The feature matrix is selected only from the frozen feature allowlist; "
            "locked-test rows are not used for fitting or threshold selection.",
            "",
        ]
    )
