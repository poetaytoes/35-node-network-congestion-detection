from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from congestion_lab.traffic.external_profiles import (
    external_sample_to_events,
    profile_flow_csv,
    profile_mawi_trace_info_html,
    profile_to_synthetic_sample,
    verify_external_file,
    write_profile,
)


def test_external_profile_retains_hash_not_identity(tmp_path: Path, project_root: Path) -> None:
    source = project_root / "tests/fixtures/external_profile_sample.csv"
    profile = profile_flow_csv(
        source,
        source_name="5g_nidd_fixture",
        bitrate_column="bitrate_bps",
        duration_column="duration_s",
        protocol_column="protocol",
        label_column="label",
    )
    assert profile["rows"] == 8
    assert len(profile["source_sha256"]) == 64
    output = write_profile(profile, tmp_path / "profile.json")
    sample = profile_to_synthetic_sample(profile, tmp_path / "sample.csv", rows=32)
    assert output.is_file() and sample.is_file()
    assert "raw_ip" not in sample.read_text(encoding="utf-8")
    assert profile["duplicate_rows"] == 0
    assert profile["identity_columns_discarded"] == []


def test_external_profile_detects_duplicates_and_discards_identity(tmp_path: Path) -> None:
    source = tmp_path / "identity_fixture.csv"
    pd.DataFrame(
        {
            "bitrate": [100.0, 100.0],
            "duration": [2.0, 2.0],
            "protocol": ["tcp", "tcp"],
            "raw_ip": ["192.0.2.1", "192.0.2.1"],
        }
    ).to_csv(source, index=False)
    profile = profile_flow_csv(
        source,
        source_name="fixture",
        bitrate_column="bitrate",
        duration_column="duration",
        protocol_column="protocol",
    )
    assert profile["duplicate_rows"] == 1
    assert profile["identity_columns_discarded"] == ["raw_ip"]


def test_official_mawi_aggregate_parser_and_checksum(project_root: Path) -> None:
    source = project_root / "data/external/sources/mawi_202608011400_trace_info.html"
    if not source.is_file():
        pytest.skip("The optional MAWI snapshot is not included in the source-only release.")
    expected = "fde452532c1e1703bebb1fe04e20e10089fb0e517cf8fe3d2528cc1dad7fdb64"
    verification = verify_external_file(source, expected_sha256=expected)
    profile = profile_mawi_trace_info_html(
        source,
        source_url="https://mawi.nezu.wide.ad.jp/mawi/samplepoint-F/2026/202608011400.html",
    )
    assert verification["status"] == "pass"
    assert profile["trace"] == {
        "duration_s": 900.14,
        "packets": 239338095,
        "average_rate_mbps": 2215.77,
        "rate_stddev_mbps": 441.73,
    }
    assert abs(sum(profile["protocol_ratio"].values()) - 1.0) < 1e-12
    assert profile["identity_fields_retained"] == []


def test_external_sample_event_schema_has_no_raw_identity(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    pd.DataFrame(
        {
            "mean_bitrate_bps": [1_000_000.0, 2_000_000.0],
            "duration_s": [5.0, 9.0],
            "protocol_family": ["tcp", "udp"],
            "traffic_source": ["profile", "profile"],
        }
    ).to_csv(sample, index=False)
    events = external_sample_to_events(
        sample, episode_id="EX01", duration_s=20, regions=["R1", "R2"]
    )
    assert len(events) == 2
    assert set(events["source_region"]) == {"R1", "R2"}
    assert events["event_id"].is_unique
    assert not any("ip" in column.lower() for column in events)
