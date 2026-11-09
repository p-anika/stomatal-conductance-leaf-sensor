"""Tests for processing/gs_calculator.py (T1.4, Appendix A eqs. 1-8, validity rules, A15).

The worked-example numbers and the 1% tolerance are from spec Appendix A.
Censored results (lambdaE <= 0, r_s <= 0) must come back labelled and valid, never dropped.
"""
import copy
import dataclasses
from pathlib import Path

import pytest
from scipy.optimize import brentq

from processing import energy_balance_utils as ebu
from processing.config import CalibrationMissing, load_config
from processing.gs_calculator import ConductanceResult, LeafReading, count_censor, solve_gs

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def cfg():
    return load_config(REPO_ROOT / "config.yaml")


@pytest.fixture
def cfg1(cfg):
    """Worked example sets faces = 1 explicitly (config default is 2)."""
    c = copy.deepcopy(cfg)
    c["physics"]["faces"] = 1
    return c


# k = 0.01 W m-2 per lux and lux 12000 give E_s = 120 W m-2 (worked example).
# a_s_leaf 0.85 for led_lamp comes from config (so flag a_s_default is expected).
@pytest.fixture
def cal():
    return {"light_sources": {"led_lamp": {"k_wm2_per_lux": 0.01}},
            "spatial": {"irradiance_factor_leaf": 1.0}}


def leaf(**kw):
    base = dict(leaf_temp_c=23.0, air_temp_c=24.0, surroundings_temp_c=24.0,
                relative_humidity_pct=50.0, pressure_hpa=995.0, lux=12000.0,
                light_source="led_lamp", r_bh=70.0, r_bh_method="card_direct")
    base.update(kw)
    return LeafReading(**base)


# ---------- worked example (Appendix A, 1% tolerance) ----------

def test_worked_example(cfg1, cal):
    r = solve_gs(leaf(), cfg1, cal)
    assert r.valid and r.reason == "ok" and r.censor == "none"
    assert r.rn_wm2 == pytest.approx(107.7, rel=0.01)
    assert r.h_wm2 == pytest.approx(-17.3, rel=0.01)
    assert r.latent_wm2 == pytest.approx(125.0, rel=0.01)
    assert r.vpd_pa == pytest.approx(1317, rel=0.01)
    assert r.r_s == pytest.approx(127.6, rel=0.01)
    assert r.g_s_ms == pytest.approx(0.00784, rel=0.01)
    assert r.g_s_mol == pytest.approx(0.316, rel=0.01)


@pytest.mark.parametrize("leaf_temp_c, expected_mol", [(22.5, 0.402), (23.5, 0.250)])
def test_worked_example_half_degree_error(cfg1, cal, leaf_temp_c, expected_mol):
    r = solve_gs(leaf(leaf_temp_c=leaf_temp_c), cfg1, cal)
    assert r.g_s_mol == pytest.approx(expected_mol, rel=0.01)


def test_flags_from_light_and_position(cfg1, cal):
    r = solve_gs(leaf(), cfg1, cal)
    assert "a_s_default" in r.flags          # a_s_leaf not in calibration
    cal2 = copy.deepcopy(cal)
    del cal2["spatial"]
    r2 = solve_gs(leaf(), cfg1, cal2)
    assert "c_position_default" in r2.flags
    assert r2.g_s_mol == pytest.approx(r.g_s_mol)   # factor 1.0 either way


def test_k_uncalibrated_flag(cfg1, cal):
    c = copy.deepcopy(cfg1)
    c["physics"]["allow_uncalibrated_k"] = True
    assert "k_uncalibrated" in solve_gs(leaf(light_source="window"), c, cal).flags


def test_led_without_k_raises(cfg1):
    with pytest.raises(CalibrationMissing):
        solve_gs(leaf(), cfg1, {})


def test_interval_fields_empty_until_t15(cfg1, cal):
    r = solve_gs(leaf(), cfg1, cal)
    assert r.g_s_mol_dl is None and r.g_s_lo is None and r.g_s_hi is None


def test_result_frozen(cfg1, cal):
    r = solve_gs(leaf(), cfg1, cal)
    assert isinstance(r, ConductanceResult)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.valid = False  # type: ignore[misc]


# ---------- round trip ----------

def forward_leaf_temp(r_s, r_bh, air_c, sur_c, rh, p_hpa, e_s, a_s, cfg):
    """Solve Rn(T) - H(T) - lambdaE(T) = 0 for T_leaf, independently of gs_calculator."""
    p = cfg["physics"]
    rc = ebu.rho_cp_j_m3_k(cfg)
    gamma = ebu.psychrometric_constant_pa(p_hpa, cfg)
    e_a = ebu.saturation_vapor_pressure_pa(air_c, cfg) * rh / 100.0
    r_bw = p["rbw_to_rbh_ratio"] * r_bh

    def residual(t):
        rn = ebu.net_radiation_wm2(e_s, a_s, t, sur_c, p["leaf_emissivity_longwave"],
                                   p["faces"], p["include_longwave"], cfg)
        h = rc * (t - air_c) / r_bh
        le = rc / gamma * (ebu.saturation_vapor_pressure_pa(t, cfg) - e_a) / (r_s + r_bw)
        return rn - h - le

    return brentq(residual, air_c - 20.0, air_c + 20.0, xtol=1e-12)


@pytest.mark.parametrize("faces", [1, 2])
@pytest.mark.parametrize("r_s_true", [50.0, 127.6, 400.0, 2000.0])
def test_round_trip(cfg, cal, faces, r_s_true):
    c = copy.deepcopy(cfg)
    c["physics"]["faces"] = faces
    t_leaf = forward_leaf_temp(r_s_true, 70.0, 24.0, 24.0, 50.0, 995.0, 120.0, 0.85, c)
    r = solve_gs(leaf(leaf_temp_c=t_leaf), c, cal)
    assert r.censor == "none"
    assert r.r_s == pytest.approx(r_s_true, rel=1e-6)


# ---------- censoring (A15): kept, labelled, valid ----------

def test_negative_latent_is_below_detection(cfg1, cal):
    # Low light, warm leaf: E_s = 10 W m-2, Rn = 8.5 - longwave (about 17) < 0,
    # H = 1210 x 3 / 70 = 51.9 W m-2, so lambdaE = Rn - H is clearly negative.
    r = solve_gs(leaf(leaf_temp_c=27.0, lux=1000.0), cfg1, cal)
    assert r.latent_wm2 <= 0
    assert r.valid is True and r.reason == "ok" and r.censor == "below_detection"
    assert r.r_s is None and r.g_s_ms is None and r.g_s_mol is None
    assert r.rn_wm2 is not None and r.h_wm2 is not None and r.vpd_pa is not None


def test_zero_latent_is_below_detection(cfg1, cal):
    # Find the leaf temperature where lambdaE = 0 exactly, then nudge just above it.
    def le(t):
        return solve_gs(leaf(leaf_temp_c=t), cfg1, cal).latent_wm2
    t0 = brentq(le, 24.0, 40.0, xtol=1e-12)
    r = solve_gs(leaf(leaf_temp_c=t0 + 1e-6), cfg1, cal)
    assert r.censor == "below_detection" and r.valid


def test_nonpositive_rs_is_above_range(cfg1, cal):
    # Very cold leaf -> large lambdaE -> r_s + r_bw < r_bw.
    r = solve_gs(leaf(leaf_temp_c=15.0), cfg1, cal)
    assert r.latent_wm2 > 0
    assert r.valid is True and r.reason == "ok" and r.censor == "above_range"
    assert r.r_s is not None and r.r_s <= 0
    assert r.g_s_ms is None and r.g_s_mol is None


def test_censored_results_are_kept_in_batch(cfg1, cal):
    readings = [leaf(), leaf(leaf_temp_c=27.0, lux=1000.0), leaf(leaf_temp_c=15.0),
                leaf(air_temp_c=float("nan"))]
    results = [solve_gs(x, cfg1, cal) for x in readings]
    assert len(results) == 4
    # count_censor counts valid results only; the NaN reading is invalid (missing_input).
    assert count_censor(results) == {"none": 1, "below_detection": 1, "above_range": 1}
    assert [r.valid for r in results] == [True, True, True, False]


# ---------- VPD <= 0 (spec, Session 8 review: censored above_range + flag) ----------

@pytest.mark.parametrize("leaf_temp_c", [
    23.0,   # VPD < 0 (e_s(23) 2809 < e_a 2984 Pa) with lambdaE > 0
    24.0,   # VPD = 0 exactly (leaf = air, RH 100)
])
def test_vpd_nonpositive_is_above_range_and_kept(cfg1, cal, leaf_temp_c):
    r = solve_gs(leaf(leaf_temp_c=leaf_temp_c, relative_humidity_pct=100.0), cfg1, cal)
    assert r.latent_wm2 > 0 and r.vpd_pa <= 0
    assert r.valid is True and r.reason == "ok" and r.censor == "above_range"
    assert "vpd_nonpositive" in r.flags
    # With VPD <= 0 and lambdaE > 0, eq. 8 gives r_s = (<= 0) - r_bw < 0.
    assert r.r_s < 0 and r.g_s_ms is None and r.g_s_mol is None


def test_vpd_nonpositive_with_nonpositive_latent_is_below_detection(cfg1, cal):
    # The spec rule covers lambdaE > 0 only. With lambdaE <= 0 the usual order applies
    # (below_detection) and the flag is still set. No light, 10 C surroundings:
    # Rn about -75, H about -8.6, lambdaE about -66 W m-2.
    r = solve_gs(leaf(leaf_temp_c=23.5, lux=0.0, surroundings_temp_c=10.0,
                      relative_humidity_pct=100.0), cfg1, cal)
    assert r.vpd_pa <= 0 and r.latent_wm2 <= 0
    assert r.valid is True and r.reason == "ok" and r.censor == "below_detection"
    assert "vpd_nonpositive" in r.flags


def test_small_positive_vpd_is_not_flagged(cfg1, cal):
    # No threshold: any VPD > 0 is solved normally (its uncertainty is T1.5's job).
    r = solve_gs(leaf(leaf_temp_c=24.05, relative_humidity_pct=100.0), cfg1, cal)
    assert 0 < r.vpd_pa < 20
    assert r.reason == "ok" and r.valid and "vpd_nonpositive" not in r.flags


def test_negative_lux_is_missing_input(cfg1, cal):
    assert solve_gs(leaf(lux=-1.0), cfg1, cal).reason == "missing_input"
    assert solve_gs(leaf(lux=0.0), cfg1, cal).valid     # darkness is a real reading


# ---------- independent hand calculation, faces = 2 (Session 8 review item 7) ----------

def test_hand_calculation_two_faces(cfg, cal):
    """Expected numbers were worked out by hand from the config constants, NOT with any
    project function. Inputs: leaf 23.0, air 24.0, surroundings 24.0 C, RH 50 %, 995 hPa,
    E_s 120 W m-2 (12000 lux x 0.01), a_s 0.85, e_L 0.97, faces 2, r_bH 70, ratio 0.93.

      shortwave  0.85 x 120 = 102.000
      T_s^4 = 296.15^4 = 7.69214e9 K^4;  T_sur^4 = 297.15^4 = 7.79656e9 K^4
      longwave   2 x 0.97 x 5.670374419e-8 x (7.79656e9 - 7.69214e9) = 11.487
      Rn         102.000 + 11.487 = 113.487 W m-2
      rho cp     1.204 x 1005 = 1210.02;  H = 1210.02 x (23 - 24) / 70 = -17.286 W m-2
      lambdaE    113.487 + 17.286 = 130.773 W m-2
      e_s(23)    610.78 exp(17.27 x 23 / 260.3) = 2809.35 Pa
      e_s(24)    610.78 exp(17.27 x 24 / 261.3) = 2983.82 Pa;  e_a = 1491.91 Pa
      VPD        2809.35 - 1491.91 = 1317.44 Pa
      gamma      0.0665 x 995 = 66.1675 Pa K-1
      r_s+r_bw   1210.02 x 1317.44 / (66.1675 x 130.773) = 184.229 s m-1
      r_s        184.229 - 0.93 x 70 = 119.129 s m-1;  g_s = 1/119.129 = 0.00839423 m s-1
      molar      99500 / (8.314 x 297.15) = 40.2752 mol m-3;  g_s = 0.338079 mol m-2 s-1
    """
    c = copy.deepcopy(cfg)
    assert c["physics"]["faces"] == 2          # the config default, not overridden
    r = solve_gs(leaf(), c, cal)
    assert r.valid and r.censor == "none"
    assert r.rn_wm2 == pytest.approx(113.487, rel=1e-4)
    assert r.h_wm2 == pytest.approx(-17.286, rel=1e-4)
    assert r.latent_wm2 == pytest.approx(130.773, rel=1e-4)
    assert r.vpd_pa == pytest.approx(1317.44, rel=1e-4)
    assert r.r_s == pytest.approx(119.129, rel=1e-4)
    assert r.g_s_ms == pytest.approx(0.00839423, rel=1e-4)
    assert r.g_s_mol == pytest.approx(0.338079, rel=1e-4)


# ---------- missing_input ----------

@pytest.mark.parametrize("field", ["leaf_temp_c", "air_temp_c", "surroundings_temp_c",
                                   "relative_humidity_pct", "pressure_hpa", "lux", "r_bh"])
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
def test_missing_input(cfg1, cal, field, bad):
    r = solve_gs(leaf(**{field: bad}), cfg1, cal)
    assert r.valid is False and r.reason == "missing_input" and r.censor == "none"
    assert r.g_s_mol is None


@pytest.mark.parametrize("field, bad", [("r_bh", 0.0), ("r_bh", -5.0), ("pressure_hpa", 0.0),
                                        ("relative_humidity_pct", -1.0),
                                        ("relative_humidity_pct", 100.5)])
def test_out_of_range_inputs_are_missing(cfg1, cal, field, bad):
    r = solve_gs(leaf(**{field: bad}), cfg1, cal)
    assert r.valid is False and r.reason == "missing_input"


def test_surroundings_optional_without_longwave(cfg1, cal):
    c = copy.deepcopy(cfg1)
    c["physics"]["include_longwave"] = False
    assert solve_gs(leaf(surroundings_temp_c=None), c, cal).valid


def test_unknown_rbh_method_raises(cfg1, cal):
    with pytest.raises(ValueError):
        solve_gs(leaf(r_bh_method="guess"), cfg1, cal)


@pytest.mark.parametrize("method", ["card_direct", "ml_corrected", "flat_plate"])
def test_known_rbh_methods(cfg1, cal, method):
    assert solve_gs(leaf(r_bh_method=method), cfg1, cal).valid
