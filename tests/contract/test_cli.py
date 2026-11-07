from __future__ import annotations

from typer.testing import CliRunner

from congestion_lab.cli import app

runner = CliRunner()


def test_cli_help_and_config_validation(project_root) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "pipeline" in result.stdout
    config = project_root / "configs/experiments/core_18_episodes.yaml"
    result = runner.invoke(app, ["config", "validate", "--config", str(config)])
    assert result.exit_code == 0
    assert '"status": "valid"' in result.stdout


def test_cli_dry_run_is_machine_readable_and_does_not_write(project_root, tmp_path) -> None:
    config = project_root / "configs/experiments/pilot.yaml"
    output = tmp_path / "must_not_exist"
    result = runner.invoke(
        app,
        [
            "simulate",
            "paired",
            "--experiment",
            str(config),
            "--output-dir",
            str(output),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert '"status": "planned"' in result.stdout
    assert '"dry_run": true' in result.stdout
    assert not output.exists()
