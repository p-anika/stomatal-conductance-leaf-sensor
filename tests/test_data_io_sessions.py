"""Tests for the T4.1 parts of processing/data_io.py: list_sessions, load_session,
load_ground_truth, default path of load_reference_cards.

Done-when (spec): a fabricated session folder loads. Schema problems raise SchemaError
naming the file and the column.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from acquisition.session_metadata import FIELDNAMES
from processing.data_io import (
    EVENTS_COLUMNS,
    MANIFEST_FIELDS,
    SENSOR_LOG_COLUMNS,
    SchemaError,
    Session,
    list_sessions,
    load_ground_truth,
    load_reference_cards,
    load_session,
)

SID = "HK-01_2026-11-09T10-00-00+00-00"


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def fabricate(raw: Path, sid: str = SID, n: int = 3) -> Path:
    d = raw / sid
    (d / "images").mkdir(parents=True)
    meta = {k: "" for k in FIELDNAMES}
    meta.update(session_id=sid, timestamp_start="2026-11-09T10:00:00+00:00", plant_id="HK-01",
                acclimation_minutes=40.0)
    (d / "session.json").write_text(json.dumps(meta), encoding="utf-8")
    manifest = {k: None for k in MANIFEST_FIELDS}
    manifest.update(session_id=sid, start_iso="2026-11-09T10:00:00+00:00", clock_status="ntp_synced",
                    n_samples=n, sensor_settings={})
    (d / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = []
    for i in range(n):
        r = {c: "" for c in SENSOR_LOG_COLUMNS}
        r.update(session_id=sid, timestamp_iso=f"2026-11-09T10:00:{5 * i:02d}+00:00",
                 monotonic_s=5 * i, air_temp_c=22.0, humidity_pct=50, leaf_temp_c=23.1 + i / 10)
        rows.append(r)
    _write_csv(d / "sensor_log.csv", list(SENSOR_LOG_COLUMNS), rows)
    _write_csv(d / "events.csv", list(EVENTS_COLUMNS),
               [{"timestamp_iso": "2026-11-09T10:00:00+00:00", "label": "measurement_start", "note": ""}])
    return d


def test_fabricated_session_loads(tmp_path: Path):
    raw = tmp_path / "raw"
    d = fabricate(raw)
    s = load_session(SID, raw)
    assert isinstance(s, Session)
    assert s.session_id == SID and s.metadata["plant_id"] == "HK-01"
    assert len(s.sensor) == 3
    assert s.sensor["leaf_temp_c"].iloc[2] == pytest.approx(23.3)
    assert pd.isna(s.sensor["wind_ms"].iloc[0])                 # blank -> NaN
    assert s.sensor["timestamp_iso"].iloc[1] == pd.Timestamp("2026-11-09T10:00:05+00:00")
    assert list(s.events["label"]) == ["measurement_start"]
    assert s.manifest["clock_status"] == "ntp_synced"
    assert s.paths["thermal_frames"] == d / "thermal_frames.f32"
    assert s.synthetic is False


def test_list_sessions_needs_manifest(tmp_path: Path):
    raw = tmp_path / "raw"
    fabricate(raw, "B_session")
    fabricate(raw, "A_session")
    (raw / "debug_logs").mkdir()
    (raw / "half_done").mkdir()
    (raw / "session_metadata.csv").write_text("x\n")
    assert list_sessions(raw) == ["A_session", "B_session"]
    assert list_sessions(tmp_path / "missing") == []


def test_missing_events_gives_empty_table(tmp_path: Path):
    raw = tmp_path / "raw"
    (fabricate(raw) / "events.csv").unlink()
    s = load_session(SID, raw)
    assert list(s.events.columns) == list(EVENTS_COLUMNS) and len(s.events) == 0


@pytest.mark.parametrize("breaker, match", [
    (lambda d: (d / "run_manifest.json").unlink(), "run_manifest.json"),
    (lambda d: (d / "session.json").unlink(), "session.json"),
    (lambda d: (d / "sensor_log.csv").unlink(), "sensor_log.csv"),
])
def test_missing_files(tmp_path: Path, breaker, match):
    raw = tmp_path / "raw"
    breaker(fabricate(raw))
    with pytest.raises(SchemaError, match=match):
        load_session(SID, raw)


def test_sensor_log_missing_column(tmp_path: Path):
    raw = tmp_path / "raw"
    d = fabricate(raw)
    df = pd.read_csv(d / "sensor_log.csv", dtype=str).drop(columns=["lux"])
    df.to_csv(d / "sensor_log.csv", index=False)
    with pytest.raises(SchemaError, match="lux"):
        load_session(SID, raw)


def test_sensor_log_not_a_number(tmp_path: Path):
    raw = tmp_path / "raw"
    d = fabricate(raw)
    df = pd.read_csv(d / "sensor_log.csv", dtype=str, keep_default_na=False)
    df.loc[1, "air_temp_c"] = "warm"
    df.to_csv(d / "sensor_log.csv", index=False)
    with pytest.raises(SchemaError, match="air_temp_c.*row 3"):
        load_session(SID, raw)


def test_sensor_log_naive_timestamp(tmp_path: Path):
    raw = tmp_path / "raw"
    d = fabricate(raw)
    df = pd.read_csv(d / "sensor_log.csv", dtype=str, keep_default_na=False)
    df.loc[0, "timestamp_iso"] = "2026-11-09T10:00:00"
    df.to_csv(d / "sensor_log.csv", index=False)
    with pytest.raises(SchemaError, match="timestamp_iso"):
        load_session(SID, raw)


def test_manifest_problems(tmp_path: Path):
    raw = tmp_path / "raw"
    d = fabricate(raw)
    m = json.loads((d / "run_manifest.json").read_text())
    m["clock_status"] = "guess"
    (d / "run_manifest.json").write_text(json.dumps(m))
    with pytest.raises(SchemaError, match="clock_status"):
        load_session(SID, raw)
    m["clock_status"] = "rtc"
    del m["git_commit"]
    (d / "run_manifest.json").write_text(json.dumps(m))
    with pytest.raises(SchemaError, match="git_commit"):
        load_session(SID, raw)


def test_session_id_mismatch(tmp_path: Path):
    raw = tmp_path / "raw"
    d = fabricate(raw)
    meta = json.loads((d / "session.json").read_text())
    meta["session_id"] = "other"
    (d / "session.json").write_text(json.dumps(meta))
    with pytest.raises(SchemaError, match="session_id"):
        load_session(SID, raw)


def test_metadata_missing_field(tmp_path: Path):
    raw = tmp_path / "raw"
    d = fabricate(raw)
    meta = json.loads((d / "session.json").read_text())
    del meta["genotype"]
    (d / "session.json").write_text(json.dumps(meta))
    with pytest.raises(SchemaError, match="genotype"):
        load_session(SID, raw)


def test_unknown_session(tmp_path: Path):
    with pytest.raises(SchemaError, match="nope"):
        load_session("nope", tmp_path)


# ---------- ground truth ----------

GT_COLS = ["plant_id", "timestamp_iso", "instrument", "surface", "g_s_mol", "leaf_position", "notes"]


def test_ground_truth_loads(tmp_path: Path):
    p = tmp_path / "gt.csv"
    _write_csv(p, GT_COLS, [{"plant_id": "HK-01", "timestamp_iso": "2026-11-09T10:05:00+00:00",
                             "instrument": "porometer", "surface": "abaxial", "g_s_mol": "0.31",
                             "leaf_position": "top", "notes": ""}])
    df = load_ground_truth(p)
    assert df["g_s_mol"].iloc[0] == pytest.approx(0.31)
    assert df["timestamp_iso"].iloc[0] == pd.Timestamp("2026-11-09T10:05:00+00:00")


@pytest.mark.parametrize("col, value", [("instrument", "guess"), ("surface", "top"),
                                        ("g_s_mol", "-0.1"), ("g_s_mol", "lots"), ("plant_id", "")])
def test_ground_truth_bad_values(tmp_path: Path, col, value):
    row = {"plant_id": "HK-01", "timestamp_iso": "2026-11-09T10:05:00+00:00", "instrument": "porometer",
           "surface": "abaxial", "g_s_mol": "0.31", "leaf_position": "top", "notes": ""}
    row[col] = value
    p = tmp_path / "gt.csv"
    _write_csv(p, GT_COLS, [row])
    with pytest.raises(SchemaError, match=col):
        load_ground_truth(p)


def test_ground_truth_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_ground_truth(tmp_path / "none.csv")


def test_reference_cards_default_path(repo_root: Path, monkeypatch):
    monkeypatch.chdir(repo_root)
    assert isinstance(load_reference_cards(), dict)
