"""Real-sensor checks for the Raspberry Pi. Skipped on the laptop; run on the Pi with
    .venv/bin/python -m pytest -q tests/test_sensors_hardware.py --hardware
(see docs/pi_first_run_checklist.md). Each reads once and checks a plausible value."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from acquisition import camera_capture as cc
from acquisition import sensors as S

pytestmark = pytest.mark.hardware


def test_bme280_real(default_config):
    r = S.Bme280Sensor(default_config).read()
    assert r.status == S.OK
    assert 5 < r.air_temp_c < 45 and 5 < r.humidity_pct < 95 and 900 < r.pressure_hpa < 1100


def test_tsl2591_real(default_config):
    r = S.Tsl2591Sensor(default_config).read()
    assert r.status in (S.OK, S.OVERFLOW)
    if r.status == S.OK:
        assert r.lux >= 0 and r.gain in ("high", "medium", "low")


def test_mlx90640_real(default_config):
    r = S.Mlx90640Sensor(default_config).read()
    assert r.status == S.OK and r.frame_c.shape == (24, 32)
    assert 0 < float(np.median(r.frame_c)) < 50


def test_fs3000_real(default_config):
    cfg = copy.deepcopy(default_config)
    cfg["sensors"]["fs3000"]["enabled"] = True
    r = S.Fs3000Sensor(cfg).read()
    assert r.status == S.OK and 0 <= r.wind_ms <= 7.23


def test_arducam_real(tmp_path, default_config):
    cfg = copy.deepcopy(default_config)
    cfg["sensors"]["arducam"]["working_distance_mm"] = cfg["sensors"]["arducam"]["working_distance_mm"] or 300
    cam = cc.ArducamCamera(cfg)
    rel = cam.capture(tmp_path, "2026-11-11T13-30-00", label="test")
    assert (tmp_path / rel).stat().st_size > 10_000
