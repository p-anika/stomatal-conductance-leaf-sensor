"""Tests for processing/config.py (T1.1).

Covers the three light-source paths the spec names (calibrated, window or sunlight
fallback, LED refusal), the missing-calibration warning, and position factors (A16).
"""
import copy
import json
import logging
from pathlib import Path

import pytest

from processing.config import (
    CalibrationMissing,
    get_light_source,
    get_position_factor,
    load_calibration,
    load_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def cfg():
    return load_config(REPO_ROOT / "config.yaml")


def allow_fallback(cfg):
    c = copy.deepcopy(cfg)
    c["physics"]["allow_uncalibrated_k"] = True
    return c


# ---------- loading ----------

def test_load_config_reads_repo_file(cfg):
    assert cfg["physics"]["faces"] in (1, 2)
    assert cfg["physics"]["air_density_kg_m3"] == pytest.approx(1.204)


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_load_calibration_absent_gives_empty_dict_and_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        cal = load_calibration(tmp_path / "calibration.json")
    assert cal == {}
    assert "calibration" in caplog.text.lower()


def test_load_calibration_reads_json(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"light_sources": {"led_lamp": {"k_wm2_per_lux": 0.0042}}}))
    cal = load_calibration(path)
    assert cal["light_sources"]["led_lamp"]["k_wm2_per_lux"] == 0.0042


def test_load_calibration_rejects_non_object(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text("[1, 2]")
    with pytest.raises(ValueError):
        load_calibration(path)


# ---------- path 1: calibrated ----------

@pytest.mark.parametrize("name", ["window", "led_lamp", "sunlight"])
def test_calibrated_k_is_used_without_flags(cfg, name):
    # Test value only; not a real calibration.
    cal = {"light_sources": {name: {"k_wm2_per_lux": 0.0042, "a_s_leaf": 0.8}}}
    src = get_light_source(cfg, cal, name)
    assert src.k_wm2_per_lux == pytest.approx(0.0042)
    assert src.k_source == "calibration"
    assert src.a_s_leaf == pytest.approx(0.8)
    assert src.flags == ()
    assert src.sigma_lux_fraction == cfg["uncertainty"]["sigma_lux_fraction"]


def test_a_s_from_config_adds_a_s_default_flag(cfg):
    cal = {"light_sources": {"led_lamp": {"k_wm2_per_lux": 0.0042}}}
    src = get_light_source(cfg, cal, "led_lamp")
    assert src.a_s_leaf == pytest.approx(cfg["light_sources"]["led_lamp"]["a_s_leaf"])
    assert "a_s_default" in src.flags
    assert "k_uncalibrated" not in src.flags


# ---------- path 2: window or sunlight fallback ----------

@pytest.mark.parametrize("name", ["window", "sunlight"])
def test_fallback_uses_sunlight_reference_with_flag(cfg, name):
    src = get_light_source(allow_fallback(cfg), {}, name)
    assert src.k_wm2_per_lux == pytest.approx(cfg["light_sources"]["sunlight"]["k_wm2_per_lux"])
    assert src.k_source == "sunlight_reference"
    assert "k_uncalibrated" in src.flags
    assert src.sigma_lux_fraction == cfg["uncertainty"]["sigma_lux_fraction_uncalibrated"]


@pytest.mark.parametrize("name", ["window", "sunlight"])
def test_no_fallback_without_allow_flag(cfg, name):
    with pytest.raises(CalibrationMissing, match="light_calibration.py"):
        get_light_source(cfg, {}, name)


# ---------- path 3: LED refusal ----------

def test_led_without_calibration_raises(cfg):
    with pytest.raises(CalibrationMissing, match="light_calibration.py"):
        get_light_source(cfg, {}, "led_lamp")


def test_led_refused_even_when_fallback_allowed(cfg):
    with pytest.raises(CalibrationMissing, match="light_calibration.py"):
        get_light_source(allow_fallback(cfg), {}, "led_lamp")


def test_led_null_in_calibration_still_raises(cfg):
    cal = {"light_sources": {"led_lamp": {"k_wm2_per_lux": None}}}
    with pytest.raises(CalibrationMissing):
        get_light_source(allow_fallback(cfg), cal, "led_lamp")


def test_led_config_default_is_null(cfg):
    # Guard: the LED k must never get a default value in config.yaml.
    assert cfg["light_sources"]["led_lamp"]["k_wm2_per_lux"] is None


def test_unlisted_source_refused_even_when_fallback_allowed(cfg):
    with pytest.raises(CalibrationMissing):
        get_light_source(allow_fallback(cfg), {}, "desk_lamp")


def test_non_positive_k_is_an_error(cfg):
    cal = {"light_sources": {"led_lamp": {"k_wm2_per_lux": 0.0}}}
    with pytest.raises(ValueError):
        get_light_source(cfg, cal, "led_lamp")


# ---------- position factors (A16) ----------

@pytest.mark.parametrize("which", ["leaf", "card"])
def test_position_factor_defaults_to_one_with_flag(cfg, which):
    c, flags = get_position_factor(cfg, {}, which)
    assert c == 1.0
    assert flags == ("c_position_default",)


def test_position_factor_from_calibration(cfg):
    cal = {"spatial": {"irradiance_factor_card": 1.03}}
    c, flags = get_position_factor(cfg, cal, "card")
    assert c == pytest.approx(1.03)
    assert flags == ()


def test_position_factor_bad_name(cfg):
    with pytest.raises(ValueError):
        get_position_factor(cfg, {}, "stem")
