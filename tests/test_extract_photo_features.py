"""Tests for processing/extract_photo_features.py (T3.5) and data_io.load_photo_features.

Done-when (spec): processes a folder of test images and the CSV loads through data_io.
Owner decisions 2026-11-09: mm_per_pixel from --mm-per-pixel, else images/scale.csv;
photo timestamp from the sensor_log row whose image_file matches (blank if none).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from processing.data_io import PhotoFeaturesError, load_photo_features
from processing.extract_photo_features import main

SID = "S1"


def leaf_image(a: int = 300, b: int = 150) -> np.ndarray:
    img = np.full((800, 1000, 3), 200, np.uint8)
    cv2.ellipse(img, (500, 400), (a, b), 0, 0, 360, (40, 140, 50), -1)
    return img


def write_png(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    path.write_bytes(buf.tobytes())


@pytest.fixture
def session(tmp_path: Path) -> Path:
    """tmp_path is the repo root; data/raw/S1/ with 3 images, a sensor log and session.json."""
    sdir = tmp_path / "data" / "raw" / SID
    img = sdir / "images"
    img.mkdir(parents=True)
    write_png(img / "leaf_a.png", leaf_image())
    write_png(img / "leaf_b.png", leaf_image(280, 150))
    write_png(img / "blank.png", np.full((800, 1000, 3), 200, np.uint8))
    (img / "notes.txt").write_text("not an image")
    rows = [
        {"timestamp_iso": "2026-11-09T10:00:00+00:00", "timestamp_corrected_iso": "", "image_file": "images/leaf_a.png"},
        {"timestamp_iso": "2026-11-09T10:00:05+00:00", "timestamp_corrected_iso": "", "image_file": ""},
        {"timestamp_iso": "2026-11-09T10:01:00+00:00", "timestamp_corrected_iso": "2026-11-09T10:31:00+00:00",
         "image_file": "blank.png"},
    ]
    with open(sdir / "sensor_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (sdir / "session.json").write_text(json.dumps({"session_id": SID, "incidence_angle_deg": 0.0}))
    return tmp_path


def _hashes(folder: Path) -> dict[str, str]:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob("*") if p.is_file()}


def test_processes_folder_and_loads_through_data_io(session: Path):
    raw_before = _hashes(session / "data" / "raw")
    assert main(["--session", SID, "--root", str(session), "--mm-per-pixel", "0.1"]) == 0
    out = session / "data" / "processed" / SID
    df = load_photo_features(out / "photo_features.csv")
    assert list(df["image_file"]) == ["blank.png", "leaf_a.png", "leaf_b.png"]   # sorted; txt skipped
    a = df.set_index("image_file")
    assert bool(a.loc["leaf_a.png", "ok"]) and a.loc["leaf_a.png", "reason"] == "ok"
    assert a.loc["leaf_a.png", "d_min_mm"] == pytest.approx(30.0, rel=0.02)
    assert a.loc["leaf_a.png", "timestamp"] == pd.Timestamp("2026-11-09T10:00:00+00:00")
    assert pd.isna(a.loc["leaf_b.png", "timestamp"])                 # not in the sensor log
    assert not bool(a.loc["blank.png", "ok"]) and a.loc["blank.png", "reason"] == "segmentation_failed"
    assert a.loc["blank.png", "timestamp"] == pd.Timestamp("2026-11-09T10:31:00+00:00")  # corrected time
    assert "foreshortening_corrected" in a.loc["leaf_a.png", "flags"]   # angle from session.json
    assert (out / "overlays" / "leaf_a_overlay.png").is_file()
    assert _hashes(session / "data" / "raw") == raw_before           # raw data untouched


def test_scale_csv_used_when_no_flag(session: Path):
    img = session / "data" / "raw" / SID / "images"
    (img / "scale.csv").write_text("image_file,mm_per_pixel,notes\nleaf_a.png,0.2,\nleaf_b.png,0.1,\n")
    assert main(["--session", SID, "--root", str(session)]) == 0
    df = load_photo_features(session / "data" / "processed" / SID / "photo_features.csv").set_index("image_file")
    assert df.loc["leaf_a.png", "d_min_mm"] == pytest.approx(60.0, rel=0.02)
    assert df.loc["leaf_a.png", "mm_per_pixel"] == pytest.approx(0.2)
    # blank.png is missing from scale.csv: kept as a row, not measured
    assert df.loc["blank.png", "reason"] == "missing_input" and not bool(df.loc["blank.png", "ok"])


def test_no_scale_at_all_fails_clearly(session: Path):
    with pytest.raises(SystemExit, match="mm_per_pixel"):
        main(["--session", SID, "--root", str(session)])


def test_missing_session_folder(tmp_path: Path):
    with pytest.raises(SystemExit, match="images"):
        main(["--session", "nope", "--root", str(tmp_path), "--mm-per-pixel", "0.1"])


def test_without_session_json_no_tilt(session: Path):
    (session / "data" / "raw" / SID / "session.json").unlink()
    assert main(["--session", SID, "--root", str(session), "--mm-per-pixel", "0.1"]) == 0
    df = load_photo_features(session / "data" / "processed" / SID / "photo_features.csv").set_index("image_file")
    assert "foreshortening_corrected" not in str(df.loc["leaf_a.png", "flags"])


def test_help_runs():
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0


# ---------- data_io.load_photo_features ----------

def test_loader_rejects_missing_columns(tmp_path: Path):
    p = tmp_path / "pf.csv"
    p.write_text("timestamp,image_file,ok\n,a.png,true\n")
    with pytest.raises(PhotoFeaturesError, match="d_min_mm"):
        load_photo_features(p)


def test_loader_rejects_bad_ok(tmp_path: Path, session: Path):
    main(["--session", SID, "--root", str(session), "--mm-per-pixel", "0.1"])
    src = session / "data" / "processed" / SID / "photo_features.csv"
    text = src.read_text(encoding="utf-8").replace(",True,", ",maybe,", 1)
    p = tmp_path / "bad.csv"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(PhotoFeaturesError, match="ok"):
        load_photo_features(p)
