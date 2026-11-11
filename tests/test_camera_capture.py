"""Tests for acquisition/camera_capture.py (N1) on the laptop. The real Arducam is replaced by a fake
picamera2 object; the real camera test is in test_sensors_hardware.py (marked hardware)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from acquisition import camera_capture as cc


def test_null_camera_returns_none(tmp_path):
    assert cc.NullCamera().capture(tmp_path, "2026-11-11T13:30:00+00:00") is None


def test_file_camera_copies_and_writes_sidecar(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.jpg").write_bytes(b"\xff\xd8fake")
    (src / "b.jpg").write_bytes(b"\xff\xd8fake2")
    sess = tmp_path / "S1"
    cam = cc.FileCamera(src)
    rel = cam.capture(sess, "2026-11-11T13:30:00+00:00", label="leaf")
    assert rel == "images/2026-11-11T13-30-00+00-00_leaf.jpg"
    assert (sess / rel).read_bytes() == b"\xff\xd8fake"
    meta = json.loads((sess / rel).with_suffix(".json").read_text())
    assert meta["source_file"] == "a.jpg" and meta["camera"] == "file"
    rel2 = cam.capture(sess, "2026-11-11T13:31:00+00:00")
    assert (sess / rel2).read_bytes() == b"\xff\xd8fake2"


def test_lens_position_from_working_distance(default_config):
    cfg = copy.deepcopy(default_config)
    cfg["sensors"]["arducam"]["working_distance_mm"] = 300
    assert cc.lens_position_dioptres(cfg) == pytest.approx(1000 / 300)
    cfg["sensors"]["arducam"]["lens_position"] = 2.5                  # explicit value wins
    assert cc.lens_position_dioptres(cfg) == 2.5
    cfg["sensors"]["arducam"].update(lens_position=None, working_distance_mm=None)
    with pytest.raises(ValueError, match="working_distance_mm"):
        cc.lens_position_dioptres(cfg)


class FakePicam:
    def __init__(self):
        self.controls, self.started, self.files = {}, False, []

    def create_still_configuration(self, main):
        return {"main": main}

    def configure(self, c):
        self.config = c

    def set_controls(self, c):
        self.controls.update(c)

    def start(self):
        self.started = True

    def capture_file(self, path):
        Path(path).write_bytes(b"\xff\xd8jpeg")
        self.files.append(path)

    def capture_metadata(self):
        return {"ExposureTime": 8000, "AnalogueGain": 1.5, "LensPosition": 3.33}


def test_arducam_manual_focus_and_sidecar(tmp_path, default_config):
    cfg = copy.deepcopy(default_config)
    cfg["sensors"]["arducam"]["working_distance_mm"] = 300
    pc = FakePicam()
    cam = cc.ArducamCamera(cfg, picam=pc, af_manual="MANUAL")
    assert pc.controls["AfMode"] == "MANUAL" and pc.controls["LensPosition"] == pytest.approx(1000 / 300)
    rel = cam.capture(tmp_path, "2026-11-11T13:30:00+00:00", label="card")
    meta = json.loads((tmp_path / rel).with_suffix(".json").read_text())
    assert meta["exposure_us"] == 8000 and meta["analogue_gain"] == 1.5
    assert meta["working_distance_mm"] == 300 and meta["resolution"] == [2304, 1296]


def test_arducam_without_picamera2_raises_clear_error(default_config):
    cfg = copy.deepcopy(default_config)
    cfg["sensors"]["arducam"]["working_distance_mm"] = 300
    with pytest.raises(cc.CameraUnavailable, match="picamera2"):
        cc.ArducamCamera(cfg)
