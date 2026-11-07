from __future__ import annotations

from pathlib import Path

import pytest

from congestion_lab.config import load_experiment, load_topology
from congestion_lab.config.models import ExperimentConfig, TopologyConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def pilot_config() -> ExperimentConfig:
    return load_experiment(ROOT / "configs/experiments/pilot.yaml")


@pytest.fixture(scope="session")
def core_config() -> ExperimentConfig:
    return load_experiment(ROOT / "configs/experiments/core_18_episodes.yaml")


@pytest.fixture(scope="session")
def minimal_topology(pilot_config: ExperimentConfig) -> TopologyConfig:
    return load_topology(pilot_config.topology_path)
