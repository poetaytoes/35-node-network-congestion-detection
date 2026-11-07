"""Immutable run-directory and cryptographic lineage helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from congestion_lab.constants import SCHEMA_VERSION


def source_tree_sha256(project_root: str | Path | None = None) -> str:
    """Hash executable project sources and configuration in a stable order."""

    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[3]
    )
    candidates: set[Path] = set()
    for directory, suffixes in (
        (root / "src", {".py"}),
        (root / "scripts", {".py", ".ps1"}),
        (root / "configs", {".yaml", ".yml"}),
    ):
        if directory.is_dir():
            candidates.update(
                path for path in directory.rglob("*") if path.is_file() and path.suffix in suffixes
            )
    for filename in ("pyproject.toml", "requirements.lock", "requirements-dev.lock"):
        path = root / filename
        if path.is_file():
            candidates.add(path)

    digest = hashlib.sha256()
    for path in sorted(candidates, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return sha256_bytes(encoded)


def stable_frame_hash(frame: pd.DataFrame) -> str:
    """Hash an ordered dataframe for deterministic workload-pairing checks."""
    row_hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype="uint64")
    schema = canonical_json_hash([(str(column), str(frame[column].dtype)) for column in frame])
    return sha256_bytes(schema.encode("ascii") + row_hashes.tobytes())


@dataclass(frozen=True)
class ArtifactRecord:
    kind: str
    relative_path: str
    sha256: str
    bytes: int
    rows: int | None = None


class ArtifactStore:
    """Owns one immutable-by-convention experiment run directory."""

    def __init__(self, root: str | Path, run_id: str, *, allow_existing: bool = False) -> None:
        self.root = Path(root).resolve()
        self.run_id = run_id
        self.run_dir = self.root / run_id
        if self.run_dir.exists() and not allow_existing:
            raise FileExistsError(f"run already exists: {self.run_dir}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[ArtifactRecord] = []
        self.started_at = datetime.now(UTC)
        self.source_tree_sha256 = source_tree_sha256()

    def path(self, relative: str | Path) -> Path:
        target = (self.run_dir / relative).resolve()
        if self.run_dir not in target.parents and target != self.run_dir:
            raise ValueError("artifact path escapes run directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def write_json(self, relative: str, payload: Any) -> Path:
        target = self.path(relative)
        self._ensure_new(target)
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        self._record("json", target)
        return target

    def write_text(self, relative: str, payload: str) -> Path:
        target = self.path(relative)
        self._ensure_new(target)
        target.write_text(payload, encoding="utf-8")
        self._record("text", target)
        return target

    def write_frame(self, relative: str, frame: pd.DataFrame) -> Path:
        target = self.path(relative)
        self._ensure_new(target)
        suffix = target.suffix.lower()
        if suffix == ".parquet":
            frame.to_parquet(target, index=False)
        elif suffix == ".csv":
            frame.to_csv(target, index=False)
        else:
            raise ValueError(f"unsupported frame format: {suffix}")
        self._record("dataframe", target, rows=len(frame))
        return target

    def write_partitioned_frame(
        self,
        relative_root: str,
        frame: pd.DataFrame,
        *,
        partition_columns: tuple[str, ...],
    ) -> list[Path]:
        """Write deterministic single-file Parquet partitions and record each file."""
        if not partition_columns:
            raise ValueError("partition_columns cannot be empty")
        missing = set(partition_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"partition columns missing: {sorted(missing)}")
        root = self.path(relative_root)
        if root.exists():
            raise FileExistsError(f"partition root already exists: {root}")
        outputs: list[Path] = []
        group_key: str | list[str] = (
            partition_columns[0] if len(partition_columns) == 1 else list(partition_columns)
        )
        for keys, group in frame.groupby(group_key, sort=True, dropna=False):
            key_values = (
                (keys,)
                if len(partition_columns) == 1
                else tuple(cast(tuple[object, ...], keys))
            )
            directory = root
            for column, value in zip(partition_columns, key_values, strict=True):
                token = str(value).replace("/", "_").replace("\\", "_")
                directory /= f"{column}={token}"
            target = directory / "part-000.parquet"
            target.parent.mkdir(parents=True, exist_ok=False)
            group.sort_index().to_parquet(target, index=False)
            self._record("partitioned_dataframe", target, rows=len(group))
            outputs.append(target)
        return outputs

    @staticmethod
    def _ensure_new(target: Path) -> None:
        if target.exists():
            raise FileExistsError(f"immutable artifact already exists: {target}")

    def _record(self, kind: str, target: Path, rows: int | None = None) -> None:
        relative = target.relative_to(self.run_dir).as_posix()
        self.records = [record for record in self.records if record.relative_path != relative]
        self.records.append(
            ArtifactRecord(
                kind=kind,
                relative_path=relative,
                sha256=sha256_file(target),
                bytes=target.stat().st_size,
                rows=rows,
            )
        )

    def record_existing(self, relative: str, *, kind: str, rows: int | None = None) -> Path:
        target = self.path(relative)
        if not target.is_file():
            raise FileNotFoundError(target)
        self._record(kind, target, rows=rows)
        return target

    def write_stage_manifest(
        self,
        stage: str,
        *,
        inputs: list[Path],
        outputs: list[Path],
        config_path: Path,
        statistics: dict[str, Any],
        parent_stage: str | None = None,
    ) -> Path:
        """Write a parent/child hash record for one pipeline stage."""

        def describe(path: Path) -> dict[str, Any]:
            resolved = path.resolve()
            try:
                name = resolved.relative_to(self.run_dir).as_posix()
            except ValueError:
                name = str(resolved)
            return {
                "path": name,
                "sha256": sha256_file(resolved),
                "bytes": resolved.stat().st_size,
            }

        try:
            package_version = importlib.metadata.version("congestion-lab")
        except importlib.metadata.PackageNotFoundError:
            package_version = "source-tree"
        payload = {
            "schema_version": SCHEMA_VERSION,
            "code_package_version": package_version,
            "source_tree_sha256": self.source_tree_sha256,
            "stage": stage,
            "run_id": self.run_id,
            "parent_run_id": self.run_id,
            "parent_stage": parent_stage,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "config": describe(config_path),
            "inputs": [describe(path) for path in inputs],
            "outputs": [describe(path) for path in outputs],
            "statistics": statistics,
        }
        return self.write_json(f"manifests/{stage}.json", payload)

    def finalize(
        self,
        *,
        config_hash: str,
        parents: dict[str, str] | None = None,
        random_seeds: dict[str, int] | None = None,
    ) -> Path:
        finished_at = datetime.now(UTC)
        distributions = (
            "congestion-lab",
            "numpy",
            "pandas",
            "networkx",
            "scikit-learn",
            "xgboost",
            "pyarrow",
            "pydantic",
        )
        package_versions: dict[str, str] = {}
        for distribution in distributions:
            try:
                package_versions[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                package_versions[distribution] = "not-installed"
        lock_hashes = {
            lock.name: sha256_file(lock)
            for lock in (Path("requirements.lock"), Path("requirements-dev.lock"))
            if lock.is_file()
        }
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "started_at_utc": self.started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "duration_s": (finished_at - self.started_at).total_seconds(),
            "exit_status": 0,
            "command": subprocess.list2cmdline(sys.argv),
            "executable": sys.executable,
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "package_versions": package_versions,
            "source_tree_sha256": self.source_tree_sha256,
            "lock_sha256": lock_hashes,
            "random_seeds": random_seeds or {},
            "config_sha256": config_hash,
            "parents": parents or {},
            "artifacts": [
                record.__dict__
                for record in sorted(self.records, key=lambda item: item.relative_path)
            ],
        }
        target = self.path("run_manifest.json")
        self._ensure_new(target)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return target


def verify_manifest(path: str | Path) -> list[str]:
    manifest_path = Path(path).resolve()
    run_dir = manifest_path.parent
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for record in payload["artifacts"]:
        artifact = run_dir / record["relative_path"]
        if not artifact.is_file():
            failures.append(f"missing:{record['relative_path']}")
        elif sha256_file(artifact) != record["sha256"]:
            failures.append(f"hash:{record['relative_path']}")
    return failures
