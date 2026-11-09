"""Tests for processing/uncertainty.py (T1.5; T1.7 items 3 and 5).

T1.7 coverage: item 3 (+-0.5 C sensitivity) is test_half_degree_sensitivity; the pooled
flux-space estimate used by the truncation test (item 5, tests/test_physics_roundtrip.py)
is unit-tested at the end of this file (owner decision A, 2026-11-08).

Owner decisions tested here (2026-11-08): lognormal light and r_b errors; humidity draws
clipped to 0-100; VPD <= 0 draws excluded from percentiles but counted as not resolved;
lambdaE <= 0 draws count as g_s = 0 and r_s <= 0 draws as the cap; nothing is dropped.
"""
import copy
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from processing.config import load_config
from processing.energy_balance_utils import saturation_vapor_pressure_pa
from processing.gs_calculator import LeafReading, solve_gs
from processing.physics_baseline import ReferenceCardReading, solve_card_rbh
from processing.uncertainty import (
    ABOVE,
    BELOW,
    OK,
    VPD,
    Sigmas,
    attach_uncertainty,
    card_rbh_uncertainty,
    esat_pa_array,
    get_sigmas,
    gs_draws,
    gs_uncertainty,
    pooled_gs_mol,
    solve_core,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def cfg1():
    c = load_config(REPO_ROOT / "config.yaml")
    c["physics"]["faces"] = 1      # worked example
    return c


@pytest.fixture
def cal():
    return {"light_sources": {"led_lamp": {"k_wm2_per_lux": 0.01}},
            "spatial": {"irradiance_factor_leaf": 1.0, "irradiance_factor_card": 1.0},
            "card": {"a_s_card": 0.95}}


def leaf(**kw):
    base = dict(leaf_temp_c=23.0, air_temp_c=24.0, surroundings_temp_c=24.0,
                relative_humidity_pct=50.0, pressure_hpa=995.0, lux=12000.0,
                light_source="led_lamp", r_bh=70.0, r_bh_method="card_direct")
    base.update(kw)
    return LeafReading(**base)


def only(leaf_temp_c=0.0, air_temp_c=0.0, rh_pct=0.0, lux_fraction=0.0, rb_fraction=0.0,
         rn_card_fraction=0.0, rho_leaf_air=0.0):
    return Sigmas(leaf_temp_c=leaf_temp_c, air_temp_c=air_temp_c, rh_pct=rh_pct,
                  lux_fraction=lux_fraction, rb_fraction=rb_fraction,
                  rn_card_fraction=rn_card_fraction, rho_leaf_air=rho_leaf_air,
                  sources=(("all", "test"),))


# ---------- the vectorised core must equal solve_gs ----------

def test_esat_array_matches_scalar(cfg1):
    t = np.array([-5.0, 0.0, 20.0, 23.0, 35.0])
    expected = [saturation_vapor_pressure_pa(x, cfg1) for x in t]
    assert np.allclose(esat_pa_array(t, cfg1), expected, rtol=1e-12)


@pytest.mark.parametrize("kw", [{}, {"leaf_temp_c": 27.0, "lux": 1000.0},
                                {"leaf_temp_c": 15.0},
                                {"relative_humidity_pct": 100.0}])
@pytest.mark.parametrize("faces", [1, 2])
def test_core_matches_solve_gs(cfg1, cal, kw, faces):
    c = copy.deepcopy(cfg1)
    c["physics"]["faces"] = faces
    r = leaf(**kw)
    ref = solve_gs(r, c, cal)
    out = solve_core(np.array([r.leaf_temp_c]), np.array([r.air_temp_c]), r.surroundings_temp_c,
                     np.array([r.relative_humidity_pct]), r.pressure_hpa,
                     np.array([0.01 * r.lux]), 0.85, np.array([r.r_bh]), c)
    assert out["rn"][0] == pytest.approx(ref.rn_wm2, rel=1e-9)
    assert out["latent"][0] == pytest.approx(ref.latent_wm2, rel=1e-9)
    assert out["vpd"][0] == pytest.approx(ref.vpd_pa, rel=1e-9)
    if ref.reason == "vpd_nonpositive":
        assert out["status"][0] == VPD
    elif ref.censor == "below_detection":
        assert out["status"][0] == BELOW
    elif ref.censor == "above_range":
        assert out["status"][0] == ABOVE
    else:
        assert out["status"][0] == OK
        assert out["g_mol"][0] == pytest.approx(ref.g_s_mol, rel=1e-9)


# ---------- T1.7 item 3: +-0.5 C sensitivity ----------

def test_half_degree_sensitivity(cfg1, cal):
    """sigma_leaf = 0.5 only. g_s falls monotonically with T_leaf, so the 15.87th and
    84.13th percentiles of g_s are g_s at T_leaf + 0.5 and - 0.5: about -21% and +27%."""
    d = gs_draws(leaf(), cfg1, cal, sigmas=only(leaf_temp_c=0.5), n=20000, seed=0)
    lo, hi = np.percentile(d.g_mol, [15.87, 84.13])
    g0 = solve_gs(leaf(), cfg1, cal).g_s_mol
    assert (hi / g0 - 1) * 100 == pytest.approx(27, abs=3)
    assert (lo / g0 - 1) * 100 == pytest.approx(-21, abs=3)


# ---------- reproducibility and basic behaviour ----------

def test_same_seed_same_result(cfg1, cal):
    a = gs_uncertainty(leaf(), cfg1, cal, seed=3)
    b = gs_uncertainty(leaf(), cfg1, cal, seed=3)
    c = gs_uncertainty(leaf(), cfg1, cal, seed=4)
    assert a == b
    assert a.g_s_mol_hi != c.g_s_mol_hi


def test_zero_sigmas_collapse_to_point(cfg1, cal):
    u = gs_uncertainty(leaf(), cfg1, cal, sigmas=only(), n=50)
    g0 = solve_gs(leaf(), cfg1, cal).g_s_mol
    assert u.g_s_mol_lo == pytest.approx(g0) and u.g_s_mol_hi == pytest.approx(g0)
    assert u.frac_resolved == 1.0


def test_n_defaults_to_config(cfg1, cal):
    u = gs_uncertainty(leaf(), cfg1, cal)
    assert u.n_draws == cfg1["uncertainty"]["n_draws"] and u.seed == 0


def test_correlation_narrows_interval(cfg1, cal):
    s0 = only(leaf_temp_c=0.5, air_temp_c=0.5, rho_leaf_air=0.0)
    s9 = only(leaf_temp_c=0.5, air_temp_c=0.5, rho_leaf_air=0.9)
    w0 = gs_uncertainty(leaf(), cfg1, cal, sigmas=s0).relative_width
    w9 = gs_uncertainty(leaf(), cfg1, cal, sigmas=s9).relative_width
    assert w9 < w0


def test_correlation_is_applied(cfg1, cal):
    d = gs_draws(leaf(), cfg1, cal, sigmas=only(leaf_temp_c=1.0, air_temp_c=1.0,
                                                 rho_leaf_air=0.8), n=20000)
    assert np.corrcoef(d.t_leaf - 23.0, d.t_air - 24.0)[0, 1] == pytest.approx(0.8, abs=0.02)


def test_light_and_rb_are_lognormal(cfg1, cal):
    d = gs_draws(leaf(), cfg1, cal, sigmas=only(lux_fraction=0.5, rb_fraction=0.5), n=20000)
    assert (d.e_s > 0).all() and (d.r_bh > 0).all()
    assert np.median(d.e_s) == pytest.approx(120.0, rel=0.02)
    assert np.std(np.log(d.e_s / 120.0)) == pytest.approx(0.5, rel=0.03)


def test_humidity_draws_clipped(cfg1, cal):
    d = gs_draws(leaf(relative_humidity_pct=99.0), cfg1, cal, sigmas=only(rh_pct=3.0), n=5000)
    assert d.rh.max() == 100.0 and d.rh.min() >= 0.0
    assert (d.rh == 100.0).sum() > 0


# ---------- censored draws are kept ----------

def test_below_detection_window_keeps_all_draws(cfg1, cal):
    r = leaf(leaf_temp_c=27.0, lux=1000.0)
    u = gs_uncertainty(r, cfg1, cal)
    assert u.valid and u.frac_latent_nonpositive > 0.5
    assert u.g_s_mol_lo == 0.0              # closed stomata are inside the interval
    assert u.n_used == u.n_draws            # nothing dropped (no VPD <= 0 here)
    assert u.resolved is False


def test_above_range_draws_at_cap(cfg1, cal):
    u = gs_uncertainty(leaf(leaf_temp_c=16.0), cfg1, cal)
    assert u.frac_above_range > 0.5
    assert u.g_s_mol_hi == cfg1["uncertainty"]["g_s_cap_mol"]
    assert u.hi_at_cap is True and u.resolved is False


def test_cap_cannot_fake_resolved(cfg1, cal):
    # True g_s 1.5 mol with sigma 0.3/0.2: p97.5 hits the cap, which shrinks the width
    # below 1.0; it must still be labelled not resolved.
    from processing.uncertainty import forward_leaf_temp
    t = forward_leaf_temp(leaf(), 1.5, cfg1, cal)
    u = gs_uncertainty(leaf(leaf_temp_c=t), cfg1, cal, sigmas=only(**CALIBRATED_LIKE))
    assert u.hi_at_cap and u.relative_width < 1.0 and u.resolved is False


def test_vpd_draws_excluded_but_counted(cfg1, cal):
    # Air near saturation, leaf barely warmer: many draws have VPD <= 0.
    r = leaf(leaf_temp_c=24.1, relative_humidity_pct=99.0)
    u = gs_uncertainty(r, cfg1, cal)
    assert 0 < u.frac_vpd_nonpositive < 1
    assert u.n_used == round(u.n_draws * (1 - u.frac_vpd_nonpositive))
    assert u.frac_resolved <= 1 - u.frac_vpd_nonpositive


def test_invalid_reading_returns_invalid(cfg1, cal):
    u = gs_uncertainty(leaf(air_temp_c=float("nan")), cfg1, cal)
    assert u.valid is False and u.reason == "missing_input" and u.g_s_mol_dl is None


def test_vpd_nonpositive_window_is_invalid(cfg1, cal):
    u = gs_uncertainty(leaf(relative_humidity_pct=100.0), cfg1, cal)
    assert u.valid is False and u.reason == "vpd_nonpositive"


# ---------- detection limit ----------

CALIBRATED_LIKE = dict(leaf_temp_c=0.3, air_temp_c=0.2)


def test_placeholder_sigmas_resolve_nothing(cfg1, cal):
    # With the 1 C leaf placeholder no g_s on the grid has a 95 % width under 100 %:
    # the detection limit is undefined until thermal calibration (T7.5).
    u = gs_uncertainty(leaf(), cfg1, cal)
    assert u.g_s_mol_dl is None and u.resolved is False
    assert u.relative_width > 1.0


def test_detection_limit_in_grid_and_brackets_resolved(cfg1, cal):
    u = gs_uncertainty(leaf(), cfg1, cal, sigmas=only(**CALIBRATED_LIKE))
    unc = cfg1["uncertainty"]
    assert unc["dl_grid_min_mol"] < u.g_s_mol_dl < unc["g_s_cap_mol"]
    # The worked-example window (0.316 mol) sits above its detection limit here.
    assert u.g_s_mol_dl < 0.316


def test_detection_limit_grows_with_noise(cfg1, cal):
    lo = gs_uncertainty(leaf(), cfg1, cal, sigmas=only(leaf_temp_c=0.1, air_temp_c=0.05))
    hi = gs_uncertainty(leaf(), cfg1, cal, sigmas=only(leaf_temp_c=0.4, air_temp_c=0.2))
    assert lo.g_s_mol_dl < hi.g_s_mol_dl


def test_detection_limit_meaning(cfg1, cal):
    """Leaf temperatures produced by a true g_s well above / below the DL should be
    resolved / not resolved when tested on their own draws."""
    s = only(**CALIBRATED_LIKE)
    u = gs_uncertainty(leaf(), cfg1, cal, sigmas=s)
    dl = u.g_s_mol_dl
    from processing.uncertainty import forward_leaf_temp
    for factor, expect in ((3.0, True), (0.3, False)):
        t = forward_leaf_temp(leaf(), dl * factor, cfg1, cal)
        assert gs_uncertainty(leaf(leaf_temp_c=t), cfg1, cal, sigmas=s).resolved is expect


# ---------- sigmas ----------

def test_sigmas_from_config_are_placeholders(cfg1, cal):
    u = gs_uncertainty(leaf(), cfg1, cal)
    assert u.sigma_placeholder is True
    assert u.sigmas.leaf_temp_c == 1.0 and u.sigmas.lux_fraction == 0.15


def test_sigmas_from_calibration_override(cfg1, cal):
    from processing.config import get_light_source
    cal2 = copy.deepcopy(cal)
    cal2["uncertainty"] = {"sigma_leaf_temp_c": 0.3}
    s = get_sigmas(cfg1, cal2, get_light_source(cfg1, cal2, "led_lamp"))
    assert s.leaf_temp_c == 0.3 and dict(s.sources)["sigma_leaf_temp_c"] == "calibration"
    assert dict(s.sources)["sigma_air_temp_c"] == "config_placeholder"


def test_uncalibrated_k_uses_wide_light_sigma(cfg1, cal):
    from processing.config import get_light_source
    c = copy.deepcopy(cfg1)
    c["physics"]["allow_uncalibrated_k"] = True
    s = get_sigmas(c, cal, get_light_source(c, cal, "window"))
    assert s.lux_fraction == 0.5


@pytest.mark.parametrize("bad", [{"sigma_leaf_temp_c": -1.0}, {"rho_leaf_air_temp": 1.5}])
def test_bad_sigma_raises(cfg1, cal, bad):
    from processing.config import get_light_source
    cal2 = copy.deepcopy(cal)
    cal2["uncertainty"] = bad
    with pytest.raises(ValueError):
        get_sigmas(cfg1, cal2, get_light_source(cfg1, cal2, "led_lamp"))


# ---------- attach ----------

def test_attach_fills_fields_and_keeps_censor(cfg1, cal):
    r = leaf(leaf_temp_c=27.0, lux=1000.0)
    res = solve_gs(r, cfg1, cal)
    out = attach_uncertainty(res, gs_uncertainty(r, cfg1, cal, sigmas=only(**CALIBRATED_LIKE)))
    assert out.censor == "below_detection" and out.valid
    assert out.g_s_mol_dl is not None and out.g_s_lo == 0.0 and out.g_s_hi is not None
    assert dataclasses.replace(out, g_s_mol_dl=None, g_s_lo=None, g_s_hi=None) == res


# ---------- card helper ----------

def card(**kw):
    base = dict(card_id="round_1", card_temp_c=28.0, air_temp_c=24.0,
                surroundings_temp_c=24.0, lux=12000.0, light_source="led_lamp", c_card=1.0)
    base.update(kw)
    return ReferenceCardReading(**base)


def test_card_rn_fraction_only(cfg1, cal):
    # r = rho cp dT / (Rn (1 + 0.1 z)); p84 = r/0.9, p16 = r/1.1 -> robust sigma = 0.101 r
    r0 = solve_card_rbh(card(), cfg1, cal).r_bh
    u = card_rbh_uncertainty(card(), cfg1, cal, sigmas=only(rn_card_fraction=0.1), n=20000)
    assert u.r_bh_median == pytest.approx(r0, rel=0.01)
    assert u.r_bh_sigma == pytest.approx(0.101 * r0, rel=0.05)


def test_card_temperature_errors_dominate(cfg1, cal):
    rn_only = card_rbh_uncertainty(card(), cfg1, cal, sigmas=only(rn_card_fraction=0.1))
    full = card_rbh_uncertainty(card(), cfg1, cal)
    assert full.r_bh_sigma > 2 * rn_only.r_bh_sigma
    assert full.sigma_placeholder is True


def test_card_keeps_nonpositive_draws(cfg1, cal):
    # Card only 0.5 C above air with sigma 1 C: many draws give r_bH <= 0; kept and counted.
    u = card_rbh_uncertainty(card(card_temp_c=24.5), cfg1, cal)
    assert u.frac_nonpositive > 0.2 and u.r_bh_lo < 0


def test_card_invalid_passthrough(cfg1, cal):
    u = card_rbh_uncertainty(card(lux=100.0), cfg1, cal)
    assert u.valid is False and u.reason == "low_card_signal"


def test_card_same_seed_same_result(cfg1, cal):
    assert card_rbh_uncertainty(card(), cfg1, cal) == card_rbh_uncertainty(card(), cfg1, cal)


# ---------- pooled flux-space estimate (T1.7 item 5, owner decision A) ----------

def _pool(readings, cfg, cal):
    return pooled_gs_mol(readings, [solve_gs(x, cfg, cal) for x in readings], cfg)


def test_pooled_single_window_equals_solve_gs(cfg1, cal):
    p = _pool([leaf()], cfg1, cal)
    ref = solve_gs(leaf(), cfg1, cal)
    assert p.valid and p.reason == "ok" and p.censor == "none"
    assert p.g_s_mol == pytest.approx(ref.g_s_mol, rel=1e-12)
    assert p.r_s == pytest.approx(ref.r_s, rel=1e-12)
    assert p.n_windows == 1 and p.n_excluded == 0 and p.n_censored_windows == 0


def test_pooled_identical_windows_equal_one(cfg1, cal):
    p = _pool([leaf()] * 5, cfg1, cal)
    assert p.g_s_mol == pytest.approx(solve_gs(leaf(), cfg1, cal).g_s_mol, rel=1e-12)
    assert p.n_windows == 5


def test_pooled_uses_censored_windows_in_flux_space(cfg1, cal):
    """Hand check: mean lambdaE and mean VPD over both windows (including the one with
    lambdaE <= 0), then eq. 8 once with the shared r_bH, pressure and air temperature."""
    # Second window: E_s 60 W m-2, warm leaf -> lambdaE about -18 W m-2 (censored);
    # the pooled mean lambdaE stays positive.
    readings = [leaf(), leaf(leaf_temp_c=27.0, lux=6000.0)]
    results = [solve_gs(x, cfg1, cal) for x in readings]
    le = (results[0].latent_wm2 + results[1].latent_wm2) / 2
    vpd = (results[0].vpd_pa + results[1].vpd_pa) / 2
    rho_cp = 1.204 * 1005.0
    gamma = 0.0665 * 995.0
    r_s = rho_cp * vpd / (gamma * le) - 0.93 * 70.0
    g_mol = (1.0 / r_s) * 99500.0 / (8.314 * (24.0 + 273.15))
    p = pooled_gs_mol(readings, results, cfg1)
    assert results[1].censor == "below_detection"
    assert p.n_censored_windows == 1
    assert p.latent_wm2 == pytest.approx(le) and p.vpd_pa == pytest.approx(vpd)
    assert p.g_s_mol == pytest.approx(g_mol, rel=1e-9)


def test_pooled_below_detection_when_mean_latent_nonpositive(cfg1, cal):
    p = _pool([leaf(leaf_temp_c=27.0, lux=1000.0)] * 3, cfg1, cal)
    assert p.valid and p.censor == "below_detection"
    assert p.g_s_mol is None and p.r_s is None and p.latent_wm2 <= 0


def test_pooled_above_range(cfg1, cal):
    p = _pool([leaf(leaf_temp_c=15.0)] * 3, cfg1, cal)
    assert p.valid and p.censor == "above_range"
    assert p.g_s_mol is None and p.r_s <= 0


def test_pooled_excludes_and_counts_invalid_windows(cfg1, cal):
    p = _pool([leaf(), leaf(leaf_temp_c=None), leaf(relative_humidity_pct=100.0)], cfg1, cal)
    assert p.n_windows == 1 and p.n_excluded == 2
    assert p.g_s_mol == pytest.approx(solve_gs(leaf(), cfg1, cal).g_s_mol, rel=1e-12)


def test_pooled_all_invalid_is_invalid(cfg1, cal):
    p = _pool([leaf(leaf_temp_c=None)], cfg1, cal)
    assert p.valid is False and p.reason == "missing_input" and p.g_s_mol is None
    p0 = pooled_gs_mol([], [], cfg1)
    assert p0.valid is False and p0.n_windows == 0


def test_pooled_nonpositive_mean_vpd(cfg1, cal):
    """Each window has VPD > 0 only if solved; force a pooled VPD <= 0 by hand-made results."""
    r = solve_gs(leaf(), cfg1, cal)
    bad = dataclasses.replace(r, vpd_pa=-1.0)
    p = pooled_gs_mol([leaf()], [bad], cfg1)
    assert p.valid is False and p.reason == "vpd_nonpositive"


def test_pooled_length_mismatch_raises(cfg1, cal):
    with pytest.raises(ValueError, match="same length"):
        pooled_gs_mol([leaf()], [], cfg1)
