"""Tests for processing/physics_baseline.py (T1.3, Appendix A eq. 9).

Expected numbers come from hand arithmetic written next to each test.
Key rule under test: the card gate looks only at exogenous inputs (light), never at
T_card - T_air or the sign of r_bH.
"""
import copy
import dataclasses
import math
from pathlib import Path

import pytest

from processing.config import CalibrationMissing, load_config
from processing.physics_baseline import (
    CardResult,
    ReferenceCardReading,
    count_reasons,
    solve_card_rbh,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SIGMA = 5.670374419e-8


@pytest.fixture
def cfg():
    return load_config(REPO_ROOT / "config.yaml")


# Calibration with every card-side value filled in: k = 0.01 W m-2 per lux for the
# LED lamp, a_s_card = 0.95, card position factor 1.0.
@pytest.fixture
def cal():
    return {
        "light_sources": {"led_lamp": {"k_wm2_per_lux": 0.01}},
        "card": {"a_s_card": 0.95},
        "spatial": {"irradiance_factor_card": 1.0},
    }


def no_longwave(cfg):
    c = copy.deepcopy(cfg)
    c["physics"]["include_longwave"] = False
    return c


def reading(**kw):
    base = dict(card_id="round_1", card_temp_c=26.0, air_temp_c=24.0,
                surroundings_temp_c=24.0, lux=10000.0, light_source="led_lamp", c_card=1.0)
    base.update(kw)
    return ReferenceCardReading(**base)


# ---------- ok path ----------

def test_ok_shortwave_only_hand_value(cfg, cal):
    # E_s = 1.0 x 0.01 x 10000 = 100 W m-2; Rn = 0.95 x 100 = 95 W m-2 (no longwave)
    # r_bH = 1.204 x 1005 x (26 - 24) / 95 = 1210.02 x 2 / 95 = 25.474 s m-1
    res = solve_card_rbh(reading(), no_longwave(cfg), cal)
    assert res.valid and res.reason == "ok"
    assert res.rn_wm2 == pytest.approx(95.0, rel=1e-9)
    assert res.r_bh == pytest.approx(1210.02 * 2 / 95, rel=1e-9)
    assert res.flags == ()


def test_ok_with_longwave_matches_independent_formula(cfg, cal):
    # faces = 2, emissivity 0.95 (config card.emissivity), T_sur 24, T_card 26
    # about -22.8 W m-2 (linearised: 2 x 0.95 x 4 sigma T^3 x 2 K, T = 298.15 K)
    lw = 2 * 0.95 * SIGMA * (297.15 ** 4 - 299.15 ** 4)
    rn = 95.0 + lw
    res = solve_card_rbh(reading(), cfg, cal)
    assert res.valid
    assert res.rn_wm2 == pytest.approx(rn, rel=1e-9)
    assert res.r_bh == pytest.approx(1210.02 * 2 / rn, rel=1e-9)
    assert -23.5 < lw < -22.0


def test_result_is_frozen(cfg, cal):
    res = solve_card_rbh(reading(), cfg, cal)
    assert isinstance(res, CardResult)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.valid = False  # type: ignore[misc]


def test_reading_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        reading().lux = 1.0  # type: ignore[misc]


# ---------- missing_input ----------

@pytest.mark.parametrize("field", ["card_temp_c", "air_temp_c", "surroundings_temp_c", "lux"])
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
def test_missing_input(cfg, cal, field, bad):
    res = solve_card_rbh(reading(**{field: bad}), cfg, cal)
    assert not res.valid and res.reason == "missing_input"
    assert res.r_bh is None and res.rn_wm2 is None


def test_surroundings_not_needed_without_longwave(cfg, cal):
    res = solve_card_rbh(reading(surroundings_temp_c=None), no_longwave(cfg), cal)
    assert res.valid


# ---------- low_card_signal (exogenous gate) ----------

def test_low_card_signal_below_threshold(cfg, cal):
    # absorbed = 0.95 x 0.01 x 4200 = 39.9 W m-2 < 40
    res = solve_card_rbh(reading(lux=4200.0), cfg, cal)
    assert not res.valid and res.reason == "low_card_signal"
    assert res.r_bh is None


def test_low_card_signal_threshold_is_inclusive(cfg, cal):
    # absorbed = 0.95 x 0.01 x lux = 40.0 exactly -> passes (spec: "at least 40")
    lux = 40.0 / (0.95 * 0.01)
    res = solve_card_rbh(reading(lux=lux), no_longwave(cfg), cal)
    assert res.reason == "ok"


def test_threshold_read_from_config(cfg, cal):
    c = copy.deepcopy(cfg)
    c["physics"]["min_card_absorbed_sw_wm2"] = 200.0
    assert solve_card_rbh(reading(), c, cal).reason == "low_card_signal"


def test_c_card_scales_gate(cfg, cal):
    # absorbed = 0.95 x 0.5 x 0.01 x 6000 = 28.5 -> low; with c_card 1.0 it is 57 -> ok
    assert solve_card_rbh(reading(lux=6000.0, c_card=0.5), cfg, cal).reason == "low_card_signal"
    assert solve_card_rbh(reading(lux=6000.0, c_card=1.0), cfg, cal).reason == "ok"


@pytest.mark.parametrize("lux", [1000.0, 10000.0])
def test_gate_ignores_card_temperature(cfg, cal, lux):
    """Same light, very different card temperatures: the gate decision must not change.

    Readings that fail the gate fail it for every T_card; readings that pass it are
    never re-labelled low_card_signal because of a small or negative difference.
    """
    reasons = {solve_card_rbh(reading(lux=lux, card_temp_c=t), no_longwave(cfg), cal).reason
               for t in (22.0, 24.0, 24.01, 26.0, 30.0)}
    if lux == 1000.0:
        assert reasons == {"low_card_signal"}
    else:
        assert "low_card_signal" not in reasons


# ---------- rn_near_zero (divide-by-zero guard) ----------

def test_rn_near_zero(cfg, cal):
    # Absorbed shortwave 47.5 W m-2 passes the gate. Choose T_sur so that the
    # longwave loss cancels it: T_sur^4 = T_card^4 - 47.5 / (2 x 0.95 x sigma).
    t_card_k = 299.15
    t_sur_k = (t_card_k ** 4 - 47.5 / (2 * 0.95 * SIGMA)) ** 0.25
    res = solve_card_rbh(reading(lux=5000.0, surroundings_temp_c=t_sur_k - 273.15), cfg, cal)
    assert not res.valid and res.reason == "rn_near_zero"
    assert abs(res.rn_wm2) < 1.0
    assert res.r_bh is None


# ---------- suspect_card_rbh ----------

def test_card_colder_than_air_is_suspect_not_gated(cfg, cal):
    # Strong light, card 0.5 C below air: passes the exogenous gate, r_bH < 0.
    # It is kept (value returned) with reason suspect_card_rbh, never low_card_signal.
    res = solve_card_rbh(reading(card_temp_c=23.5), cfg, cal)
    assert not res.valid and res.reason == "suspect_card_rbh"
    assert res.r_bh is not None and res.r_bh < 0


def test_zero_difference_is_suspect(cfg, cal):
    res = solve_card_rbh(reading(card_temp_c=24.0), no_longwave(cfg), cal)
    assert res.reason == "suspect_card_rbh" and res.r_bh == 0.0


# ---------- a_s_card and emissivity sources ----------

def test_missing_a_s_card_raises(cfg, cal):
    cal = copy.deepcopy(cal)
    del cal["card"]
    with pytest.raises(CalibrationMissing):
        solve_card_rbh(reading(), cfg, cal)


def test_default_a_s_card_when_allowed(cfg, cal):
    cal = copy.deepcopy(cal)
    del cal["card"]
    c = copy.deepcopy(cfg)
    c["card"]["allow_default_a_s_card"] = True
    res = solve_card_rbh(reading(), no_longwave(c), cal)
    placeholder = c["card"]["a_s_card_placeholder"]
    assert res.rn_wm2 == pytest.approx(placeholder * 100.0)
    assert "a_s_default" in res.flags


def test_config_default_off():
    cfg = load_config(REPO_ROOT / "config.yaml")
    assert cfg["card"]["allow_default_a_s_card"] is False


def test_calibrated_a_s_card_has_no_default_flag(cfg, cal):
    c = copy.deepcopy(cfg)
    c["card"]["allow_default_a_s_card"] = True
    cal = copy.deepcopy(cal)
    cal["card"]["a_s_card"] = 0.9
    res = solve_card_rbh(reading(), no_longwave(c), cal)
    assert res.rn_wm2 == pytest.approx(90.0)
    assert "a_s_default" not in res.flags


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.2])
def test_a_s_card_out_of_range_raises(cfg, cal, bad):
    cal = copy.deepcopy(cal)
    cal["card"]["a_s_card"] = bad
    with pytest.raises(ValueError):
        solve_card_rbh(reading(), cfg, cal)


def test_emissivity_from_calibration(cfg, cal):
    cal = copy.deepcopy(cal)
    cal["card"]["emissivity"] = 0.90
    lw = 2 * 0.90 * SIGMA * (297.15 ** 4 - 299.15 ** 4)
    res = solve_card_rbh(reading(), cfg, cal)
    assert res.rn_wm2 == pytest.approx(95.0 + lw, rel=1e-9)


# ---------- flags from light source and position ----------

def test_c_card_none_uses_position_factor_and_flags(cfg, cal):
    cal = copy.deepcopy(cal)
    del cal["spatial"]
    res = solve_card_rbh(reading(c_card=None), no_longwave(cfg), cal)
    assert res.rn_wm2 == pytest.approx(95.0)
    assert "c_position_default" in res.flags


def test_c_card_none_uses_calibrated_position(cfg, cal):
    cal = copy.deepcopy(cal)
    cal["spatial"]["irradiance_factor_card"] = 1.1
    res = solve_card_rbh(reading(c_card=None), no_longwave(cfg), cal)
    assert res.rn_wm2 == pytest.approx(0.95 * 1.1 * 100.0)
    assert "c_position_default" not in res.flags


def test_k_uncalibrated_flag_passes_through(cfg, cal):
    c = copy.deepcopy(cfg)
    c["physics"]["allow_uncalibrated_k"] = True
    res = solve_card_rbh(reading(light_source="window"), c, cal)
    assert "k_uncalibrated" in res.flags
    # a_s_default is about the LEAF absorptivity, so it must not be copied to the card
    assert "a_s_default" not in res.flags


def test_led_without_k_raises(cfg, cal):
    cal = copy.deepcopy(cal)
    del cal["light_sources"]
    with pytest.raises(CalibrationMissing):
        solve_card_rbh(reading(), cfg, cal)


def test_flags_kept_on_invalid_results(cfg, cal):
    cal = copy.deepcopy(cal)
    del cal["spatial"]
    res = solve_card_rbh(reading(c_card=None, lux=100.0), cfg, cal)
    assert res.reason == "low_card_signal"
    assert "c_position_default" in res.flags


# ---------- counting ----------

def test_count_reasons(cfg, cal):
    results = [solve_card_rbh(r, cfg, cal) for r in (
        reading(), reading(), reading(card_temp_c=23.0), reading(lux=100.0),
        reading(card_temp_c=float("nan")))]
    counts = count_reasons(results)
    assert counts == {"ok": 2, "suspect_card_rbh": 1, "low_card_signal": 1, "missing_input": 1}
    assert math.isclose(sum(counts.values()), len(results))


def test_count_reasons_empty():
    assert count_reasons([]) == {}
