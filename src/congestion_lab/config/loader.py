"""YAML configuration loaders."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from congestion_lab.config.models import ExperimentConfig, TopologyConfig
from congestion_lab.exceptions import ConfigurationError

ModelT = TypeVar("ModelT", bound=BaseModel)


def _load(path: Path, model: type[ModelT]) -> ModelT:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return model.model_validate(payload)
    except (OSError, yaml.YAMLError, ValidationError, TypeError) as exc:
        raise ConfigurationError(f"invalid {model.__name__} file {path}: {exc}") from exc


def load_topology(path: str | Path) -> TopologyConfig:
    return _load(Path(path).resolve(), TopologyConfig)


def load_experiment(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    config = _load(config_path, ExperimentConfig)
    project_root = config_path.parents[2]
    return config.resolve_paths(project_root)
