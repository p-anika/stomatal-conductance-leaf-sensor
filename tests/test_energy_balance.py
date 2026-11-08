"""Tests for processing/energy_balance_utils.py (T1.2; helper and Rn parts of T1.7).

Expected numbers come from the spec (T1.7 items 1 and 6, Appendix A worked example)
or from hand arithmetic written out in the comment next to each one.
Tolerances are the spec's (1% worked example, 3% for C) and must not be loosened.
"""
import copy
import math
from pathlib import Path

import pytest

from processing.config import load_config
from processing import energy_balance_utils as ebu

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def cfg():
    return load_config(REPO_ROOT / "config.yaml")


def with_faces(cfg, faces):
    c = copy.deepcopy(cfg)
    c["physics"]["faces"] = faces
    return c


def with_b(cfg, b):
    c = copy.deepcopy(cfg)
    c["convection"]["enhancement_factor"] = b
    return c


# ---------- helpers (T1.7 item 6) ----------

def test_saturation_vapor_pressure_20c(cfg):
    assert ebu.saturation_vapor_pressure_pa(20.0, cfg) == pytest.approx(2338, rel=0.001)


def test_saturation_vapor_pressure_0c_is_tetens_a(cfg):
    assert ebu.saturation_vapor_pressure_pa(0.0, cfg) == pytest.approx(610.78)


def test_molar_density_20c_1013hpa(cfg):
    assert ebu.molar_density_mol_m3(1013.0, 20.0, cfg) == pytest.approx(41.6, rel=0.001)


def test_psychrometric_constant(cfg):
    # 0.0665 Pa/K/hPa x 1013 hPa = 67.3645 Pa/K
    assert ebu.psychrometric_constant_pa(1013.0, cfg) == pytest.approx(67.3645)


def test_rho_cp(cfg):
    # 1.204 x 1005 = 1210.02 J m-3 K-1 (Appendix A eq. 3 says 1210)
    assert ebu.rho_cp_j_m3_k(cfg) == pytest.approx(1210.02)


def test_lux_to_irradiance(cfg):
    # E_s = c_pos x k x lux = 0.9 x 0.01 x 10000 = 90 W m-2
    assert ebu.lux_to_irradiance_wm2(10000.0, 0.01, c_position=0.9) == pytest.approx(90.0)
    assert ebu.lux_to_irradiance_wm2(10000.0, 0.01) == pytest.approx(100.0)


# ---------- net radiation ----------

def test_net_radiation_worked_example(cfg):
    # Appendix A worked example: Rn ~= 107.7 W m-2 (1% tolerance)
    rn = ebu.net_radiation_wm2(120.0, 0.85, 23.0, 24.0, 0.97, 1, True, cfg)
    assert rn == pytest.approx(107.7, rel=0.01)


def test_net_radiation_without_longwave(cfg):
    rn = ebu.net_radiation_wm2(120.0, 0.85, 23.0, 24.0, 0.97, 1, False, cfg)
    assert rn == pytest.approx(0.85 * 120.0)


def test_net_radiation_two_faces_doubles_longwave(cfg):
    rn1 = ebu.net_radiation_wm2(120.0, 0.85, 23.0, 24.0, 0.97, 1, True, cfg)
    rn2 = ebu.net_radiation_wm2(120.0, 0.85, 23.0, 24.0, 0.97, 2, True, cfg)
    assert (rn2 - 102.0) == pytest.approx(2 * (rn1 - 102.0))


def test_net_radiation_bad_faces(cfg):
    with pytest.raises(ValueError):
        ebu.net_radiation_wm2(120.0, 0.85, 23.0, 24.0, 0.97, 3, True, cfg)


# ---------- equivalent coefficient C (T1.7 item 6, 3% tolerance) ----------

@pytest.mark.parametrize("s, b, expected", [(2, 1.0, 156), (1, 1.5, 208), (2, 1.5, 104), (1, 1.0, 313)])
def test_equivalent_coefficient_c(cfg, s, b, expected):
    c = ebu.equivalent_coefficient_c(with_b(with_faces(cfg, s), b))
    assert c == pytest.approx(expected, rel=0.03)


def test_forced_only_matches_c_shortcut(cfg):
    # Large wind, tiny free term check: r from the Nusselt form equals C sqrt(d/U)
    # when free convection is removed (free coefficient set to 0).
    c = copy.deepcopy(cfg)
    c["convection"]["free_nu_coeff"] = 0.0
    d, u = 0.05, 1.0
    r, regime = ebu.flat_plate_rbh(d, u, 2.0, 20.0, c)
    assert r == pytest.approx(ebu.equivalent_coefficient_c(c) * math.sqrt(d / u), rel=1e-9)
    assert regime == "forced"


# ---------- flat plate behaviour ----------

def test_rbh_finite_at_zero_wind(cfg):
    r, regime = ebu.flat_plate_rbh(0.05, 0.0, 2.0, 22.0, cfg)
    assert math.isfinite(r) and r > 0
    assert regime == "free"


def test_rbh_finite_at_zero_wind_and_zero_delta_t(cfg):
    # The Grashof floor (min_grashof_delta_t_c) keeps r_bH finite.
    r, regime = ebu.flat_plate_rbh(0.05, 0.0, 0.0, 22.0, cfg)
    assert math.isfinite(r) and r > 0
    r_floor, _ = ebu.flat_plate_rbh(0.05, 0.0, cfg["convection"]["min_grashof_delta_t_c"], 22.0, cfg)
    assert r == pytest.approx(r_floor)


def test_wind_below_noise_floor_counts_as_zero(cfg):
    floor = cfg["convection"]["wind_noise_floor_ms"]
    r0, _ = ebu.flat_plate_rbh(0.05, 0.0, 2.0, 22.0, cfg)
    r_below, regime = ebu.flat_plate_rbh(0.05, floor * 0.5, 2.0, 22.0, cfg)
    assert r_below == pytest.approx(r0)
    assert regime == "free"


def test_rbh_falls_smoothly_with_wind(cfg):
    floor = cfg["convection"]["wind_noise_floor_ms"]
    winds = [floor + 0.01 * i for i in range(1, 400)]
    rs = [ebu.flat_plate_rbh(0.05, u, 2.0, 22.0, cfg)[0] for u in winds]
    assert all(b < a for a, b in zip(rs, rs[1:]))
    # Smooth, with a bound derived from the physics: g_forced grows as U^(1/2) and g_free
    # does not depend on U, so d ln(r) / d ln(U) lies between -1/2 and 0. No step may
    # change ln(r) by more than half the change in ln(U).
    for (u1, r1), (u2, r2) in zip(zip(winds, rs), zip(winds[1:], rs[1:])):
        assert 0 < math.log(r1 / r2) <= 0.5 * math.log(u2 / u1) + 1e-12


def test_regime_switches_at_configured_ratio(cfg):
    d, dt, ta = 0.05, 2.0, 22.0
    ratio = cfg["convection"]["mixed_convection_ratio"]
    g_forced, g_free = ebu.flat_plate_conductances(d, 1.0, dt, ta, cfg)
    # g_forced grows as sqrt(U), so g_free/g_forced = ratio at U = (g_free/(ratio*g_forced(1)))^2
    u_switch = (g_free / (ratio * g_forced)) ** 2
    assert ebu.flat_plate_rbh(d, u_switch * 1.05, dt, ta, cfg)[1] == "forced"
    assert ebu.flat_plate_rbh(d, u_switch * 0.95, dt, ta, cfg)[1] == "mixed"


def test_regime_free_when_forced_small(cfg):
    # With the real 0.1 m/s noise floor, this switch happens at winds of a few mm/s,
    # i.e. below the floor. The noise floor is set to 0 here only to test the rule itself.
    c = copy.deepcopy(cfg)
    c["convection"]["wind_noise_floor_ms"] = 0.0
    d, dt, ta = 0.05, 2.0, 22.0
    ratio = c["convection"]["mixed_convection_ratio"]
    g_forced, g_free = ebu.flat_plate_conductances(d, 1.0, dt, ta, c)
    # free regime when g_forced/g_free < ratio: U = (ratio*g_free/g_forced(1))^2
    u_free = (ratio * g_free / g_forced) ** 2
    assert ebu.flat_plate_rbh(d, u_free * 0.9, dt, ta, c)[1] == "free"
    assert ebu.flat_plate_rbh(d, u_free * 1.1, dt, ta, c)[1] == "mixed"


def test_conductances_are_summed(cfg):
    d, u, dt, ta = 0.05, 0.3, 2.0, 22.0
    g_forced, g_free = ebu.flat_plate_conductances(d, u, dt, ta, cfg)
    r, _ = ebu.flat_plate_rbh(d, u, dt, ta, cfg)
    assert r == pytest.approx(1.0 / (g_forced + g_free))


def test_combination_other_than_sum_refused(cfg):
    c = copy.deepcopy(cfg)
    c["convection"]["combination"] = "max"
    with pytest.raises(ValueError, match="sum"):
        ebu.flat_plate_rbh(0.05, 1.0, 2.0, 22.0, c)


def test_sidedness_must_come_from_faces(cfg):
    c = copy.deepcopy(cfg)
    c["convection"]["sidedness_factor"] = 2
    with pytest.raises(ValueError, match="faces"):
        ebu.flat_plate_rbh(0.05, 1.0, 2.0, 22.0, c)


def test_two_faces_halves_rbh(cfg):
    r1, _ = ebu.flat_plate_rbh(0.05, 0.5, 2.0, 22.0, with_faces(cfg, 1))
    r2, _ = ebu.flat_plate_rbh(0.05, 0.5, 2.0, 22.0, with_faces(cfg, 2))
    assert r2 == pytest.approx(r1 / 2)


def test_bad_inputs_raise(cfg):
    with pytest.raises(ValueError):
        ebu.flat_plate_rbh(0.0, 1.0, 2.0, 22.0, cfg)
    with pytest.raises(ValueError):
        ebu.flat_plate_rbh(0.05, -1.0, 2.0, 22.0, cfg)
    with pytest.raises(ValueError):
        ebu.flat_plate_rbh(0.05, float("nan"), 2.0, 22.0, cfg)


@pytest.mark.parametrize("regime, flags", [("forced", ()), ("mixed", ("baseline_unreliable",)),
                                           ("free", ("baseline_unreliable",))])
def test_convection_flags(regime, flags):
    assert ebu.convection_flags(regime) == flags


@pytest.mark.skip(reason="UNVERIFIED: needs a worked forced/free Nusselt value checked in Campbell & Norman (T1.7 item 6)")
def test_nusselt_matches_campbell_norman_worked_value():
    pass
