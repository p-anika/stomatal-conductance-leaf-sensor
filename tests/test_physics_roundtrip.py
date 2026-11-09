"""Physics round trips, edge cases and the truncation-bias test (spec T1.7 items 2, 4, 5).

The forward models here are written independently of gs_calculator / physics_baseline
(they only share the small helpers in energy_balance_utils, tested in test_energy_balance).
Tolerances are the spec's (0.1% for the g_s round trip, 10% for the censoring-aware
estimate) and must not be loosened.

Truncation-test conditions are the owner's (2026-11-08): true g_s 0.05 mol m-2 s-1,
sigma_leaf 0.5 C, sigma_air 0.25 C, 200 windows, 20 seeds.
"""
import copy
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

from processing import energy_balance_utils as ebu
from processing.config import CalibrationMissing, load_config
from processing.gs_calculator import CENSOR_LABELS, LeafReading, solve_gs
from processing.jones_index import conductance_index
from processing.physics_baseline import ReferenceCardReading, solve_card_rbh
from processing.uncertainty import pooled_gs_mol

REPO_ROOT = Path(__file__).resolve().parent.parent
K_WM2_PER_LUX = 0.01        # test calibration: lux = E_s / 0.01
A_S_CARD = 0.95


@pytest.fixture
def cfg():
    return load_config(REPO_ROOT / "config.yaml")


@pytest.fixture
def cal():
    return {"light_sources": {"led_lamp": {"k_wm2_per_lux": K_WM2_PER_LUX}},
            "spatial": {"irradiance_factor_leaf": 1.0, "irradiance_factor_card": 1.0},
            "card": {"a_s_card": A_S_CARD}}


def with_faces(cfg, faces):
    c = copy.deepcopy(cfg)
    c["physics"]["faces"] = faces
    return c


def leaf(**kw):
    base = dict(leaf_temp_c=23.0, air_temp_c=24.0, surroundings_temp_c=24.0,
                relative_humidity_pct=50.0, pressure_hpa=995.0, lux=12000.0,
                light_source="led_lamp", r_bh=70.0, r_bh_method="card_direct")
    base.update(kw)
    return LeafReading(**base)


def card(**kw):
    base = dict(card_id="c1", card_temp_c=28.0, air_temp_c=24.0, surroundings_temp_c=24.0,
                lux=12000.0, light_source="led_lamp")
    base.update(kw)
    return ReferenceCardReading(**base)


# ---------- independent forward models ----------

def forward_leaf_temp(g_s_mol, r_bh_of_t, air_c, sur_c, rh, p_hpa, e_s_wm2, cfg):
    """Root of Rn(T) - H(T) - lambdaE(T) = 0 for the leaf temperature T [C].

    r_bh_of_t(T) gives r_bH [s m-1]; it may depend on T (flat plate with wind).
    r_s from g_s: r_s = molar density / g_s,mol (eq. 8 inverted).
    """
    p = cfg["physics"]
    a_s = cfg["light_sources"]["led_lamp"]["a_s_leaf"]
    rc = ebu.rho_cp_j_m3_k(cfg)
    gamma = ebu.psychrometric_constant_pa(p_hpa, cfg)
    e_a = ebu.saturation_vapor_pressure_pa(air_c, cfg) * rh / 100.0
    r_s = ebu.molar_density_mol_m3(p_hpa, air_c, cfg) / g_s_mol

    def residual(t):
        r_bh = r_bh_of_t(t)
        rn = ebu.net_radiation_wm2(e_s_wm2, a_s, t, sur_c, p["leaf_emissivity_longwave"],
                                   p["faces"], p["include_longwave"], cfg)
        h = rc * (t - air_c) / r_bh
        vpd = ebu.saturation_vapor_pressure_pa(t, cfg) - e_a
        le = rc / gamma * vpd / (r_s + p["rbw_to_rbh_ratio"] * r_bh)
        return rn - h - le

    return brentq(residual, air_c - 20.0, air_c + 20.0, xtol=1e-12)


def forward_card_temp(r_bh, air_c, sur_c, e_s_wm2, cfg):
    """Root of Rn_card(T) - rho cp (T - T_a) / r_bH = 0 (eq. 9 with lambdaE = 0)."""
    p = cfg["physics"]
    rc = ebu.rho_cp_j_m3_k(cfg)

    def residual(t):
        rn = ebu.net_radiation_wm2(e_s_wm2, A_S_CARD, t, sur_c, cfg["card"]["emissivity"],
                                   p["faces"], p["include_longwave"], cfg)
        return rn - rc * (t - air_c) / r_bh

    return brentq(residual, air_c - 20.0, air_c + 40.0, xtol=1e-12)


# ---------- T1.7 item 2: leaf round trip (0.1%) ----------

G_S_MOL = [0.02, 0.05, 0.1, 0.3, 0.6]


@pytest.mark.parametrize("faces", [1, 2])
@pytest.mark.parametrize("g_s_mol", G_S_MOL)
@pytest.mark.parametrize("r_bh", [30.0, 70.0, 150.0])
@pytest.mark.parametrize("e_s_wm2", [50.0, 120.0, 300.0])
def test_leaf_round_trip_fixed_rb(cfg, cal, faces, g_s_mol, r_bh, e_s_wm2):
    c = with_faces(cfg, faces)
    t_leaf = forward_leaf_temp(g_s_mol, lambda t: r_bh, 24.0, 24.0, 50.0, 995.0, e_s_wm2, c)
    r = solve_gs(leaf(leaf_temp_c=t_leaf, r_bh=r_bh, lux=e_s_wm2 / K_WM2_PER_LUX), c, cal)
    assert r.valid and r.censor == "none"
    assert r.g_s_mol == pytest.approx(g_s_mol, rel=1e-3)


@pytest.mark.parametrize("faces", [1, 2])
@pytest.mark.parametrize("g_s_mol", G_S_MOL)
@pytest.mark.parametrize("wind_ms", [0.2, 0.5, 1.0, 2.0])
def test_leaf_round_trip_wind(cfg, cal, faces, g_s_mol, wind_ms):
    """r_bH from the flat plate at this wind, with the leaf's own T_s - T_a inside the
    Grashof number, so leaf temperature and r_bH are solved together."""
    c = with_faces(cfg, faces)
    d_m, air_c = 0.05, 24.0

    def r_bh_of_t(t):
        return ebu.flat_plate_rbh(d_m, wind_ms, t - air_c, air_c, c)[0]

    t_leaf = forward_leaf_temp(g_s_mol, r_bh_of_t, air_c, 24.0, 50.0, 995.0, 120.0, c)
    r = solve_gs(leaf(leaf_temp_c=t_leaf, r_bh=r_bh_of_t(t_leaf), r_bh_method="flat_plate"),
                 c, cal)
    assert r.valid and r.censor == "none"
    assert r.g_s_mol == pytest.approx(g_s_mol, rel=1e-3)


# ---------- T1.7 item 4: card round trip ----------

@pytest.mark.parametrize("faces", [1, 2])
@pytest.mark.parametrize("r_bh", [30.0, 70.0, 150.0])
@pytest.mark.parametrize("e_s_wm2", [60.0, 120.0, 300.0])
@pytest.mark.parametrize("air_c", [15.0, 24.0, 32.0])
def test_card_round_trip(cfg, cal, faces, r_bh, e_s_wm2, air_c):
    c = with_faces(cfg, faces)
    t_card = forward_card_temp(r_bh, air_c, air_c, e_s_wm2, c)
    r = solve_card_rbh(card(card_temp_c=t_card, air_temp_c=air_c, surroundings_temp_c=air_c,
                            lux=e_s_wm2 / K_WM2_PER_LUX), c, cal)
    assert r.valid and r.reason == "ok"
    assert r.r_bh == pytest.approx(r_bh, rel=1e-6)


# ---------- T1.7 item 5: every reason code and censor label ----------

def _rn_zero_card_temp(cfg):
    """Card temperature at which Rn_card = 0 with E_s 120 W m-2 (longwave balances SW)."""
    p = cfg["physics"]
    return brentq(lambda t: ebu.net_radiation_wm2(120.0, A_S_CARD, t, 24.0,
                                                  cfg["card"]["emissivity"], p["faces"],
                                                  True, cfg), 24.0, 80.0, xtol=1e-12)


def test_every_physics_reason_code_and_censor_label(cfg, cal):
    c = with_faces(cfg, 1)
    gs_cases = {
        # input -> (reason, censor, valid)
        "worked example": (leaf(), ("ok", "none", True)),
        "missing leaf temp": (leaf(leaf_temp_c=None), ("missing_input", "none", False)),
        "saturated air": (leaf(relative_humidity_pct=100.0), ("ok", "above_range", True)),
        "warm leaf, low light": (leaf(leaf_temp_c=27.0, lux=1000.0),
                                 ("ok", "below_detection", True)),
        "very cold leaf": (leaf(leaf_temp_c=15.0), ("ok", "above_range", True)),
    }
    card_cases = {
        "good card": (card(), "ok"),
        "missing card temp": (card(card_temp_c=None), "missing_input"),
        "dim light": (card(lux=3000.0), "low_card_signal"),     # 0.95 x 30 = 28.5 < 40
        "Rn about 0": (card(card_temp_c=_rn_zero_card_temp(c)), "rn_near_zero"),
        "card colder than air": (card(card_temp_c=23.0), "suspect_card_rbh"),
    }
    reasons, censors = set(), set()
    for name, (reading, expected) in gs_cases.items():
        r = solve_gs(reading, c, cal)
        assert (r.reason, r.censor, r.valid) == expected, name
        reasons.add(r.reason)
        censors.add(r.censor)
    for name, (reading, expected) in card_cases.items():
        r = solve_card_rbh(reading, c, cal)
        assert r.reason == expected, name
        reasons.add(r.reason)
    jones = conductance_index(25.0, 24.8, 24.5, c)      # dry - wet = 0.5 C < 1 C
    assert jones.cwsi.reason == "references_not_bracketing"
    reasons.add(jones.cwsi.reason)

    assert "vpd_nonpositive" in solve_gs(gs_cases["saturated air"][0], c, cal).flags
    assert reasons == {"ok", "missing_input", "low_card_signal",
                       "rn_near_zero", "suspect_card_rbh", "references_not_bracketing"}
    assert censors == set(CENSOR_LABELS)


def test_below_detection_is_valid_and_kept(cfg, cal):
    r = solve_gs(leaf(leaf_temp_c=27.0, lux=1000.0), with_faces(cfg, 1), cal)
    assert r.latent_wm2 <= 0
    assert r.valid and r.reason == "ok" and r.censor == "below_detection"
    assert r.latent_wm2 is not None and r.vpd_pa is not None   # fluxes kept for pooling


@pytest.mark.parametrize("solver, reading", [(solve_gs, leaf()), (solve_card_rbh, card())])
def test_missing_k_raises(cfg, solver, reading):
    with pytest.raises(CalibrationMissing):
        solver(reading, cfg, {"card": {"a_s_card": A_S_CARD}})


# ---------- T1.7 item 5: truncation bias ----------

N_WINDOWS, N_SEEDS = 200, 20
SIGMA_LEAF_C, SIGMA_AIR_C = 0.5, 0.25      # owner's test conditions (2026-11-08)


def simulate_stressed_windows(cfg, cal, g_true_mol, seed):
    """200 windows of one leaf with true g_s, each with independent Gaussian errors on the
    window-mean leaf and air temperatures. Returns (readings, results)."""
    t_leaf_true = forward_leaf_temp(g_true_mol, lambda t: 70.0, 24.0, 24.0, 50.0, 995.0,
                                    120.0, cfg)
    rng = np.random.default_rng(seed)
    readings = [leaf(leaf_temp_c=t_leaf_true + SIGMA_LEAF_C * rng.normal(),
                     air_temp_c=24.0 + SIGMA_AIR_C * rng.normal())
                for _ in range(N_WINDOWS)]
    return readings, [solve_gs(x, cfg, cal) for x in readings]


def truncated_mean_mol(results):
    """The WRONG estimate: mean of the non-censored windows only."""
    return float(np.mean([r.g_s_mol for r in results if r.valid and r.censor == "none"]))


def test_truncation_bias_pooled_recovers_truth(cfg, cal):
    c = with_faces(cfg, 1)
    g_true = 0.05
    pooled, truncated = [], []
    for seed in range(N_SEEDS):
        readings, results = simulate_stressed_windows(c, cal, g_true, seed)
        p = pooled_gs_mol(readings, results, c)
        assert p.valid and p.censor == "none"
        assert p.n_windows == N_WINDOWS and p.n_excluded == 0
        pooled.append(p.g_s_mol)
        truncated.append(truncated_mean_mol(results))
    pooled_err = np.abs(np.array(pooled) / g_true - 1.0)
    assert np.all(pooled_err <= 0.10), f"worst pooled error {pooled_err.max():.1%}"
    assert np.mean(truncated) > g_true


def test_vpd_nonpositive_windows_kept_and_dropping_them_biases_low(cfg, cal):
    """Session 8 review: humid air (RH 97 %), dim light (E_s 30 W m-2), true g_s 0.3 mol.
    The true leaf is only about 0.4 C above air, so noise pushes about 5 % of windows to
    VPD <= 0. They must be kept (censored above_range, flagged). Dropping them removes the
    cold, high-g_s windows and pulls the pooled mean low (about -15 %); keeping them is
    closer to the truth (about +6 %). Per-seed accuracy is NOT asserted: at these
    conditions the noise is too large for the 10 % target (worst seed about 35 %)."""
    c = with_faces(cfg, 1)
    g_true, rh, e_s = 0.3, 97.0, 30.0
    t_true = forward_leaf_temp(g_true, lambda t: 70.0, 24.0, 24.0, rh, 995.0, e_s, c)
    kept, dropped, n_flagged = [], [], 0
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        readings = [leaf(leaf_temp_c=t_true + SIGMA_LEAF_C * rng.normal(),
                         air_temp_c=24.0 + SIGMA_AIR_C * rng.normal(),
                         relative_humidity_pct=rh, lux=e_s / K_WM2_PER_LUX)
                    for _ in range(N_WINDOWS)]
        results = [solve_gs(x, c, cal) for x in readings]
        flagged = ["vpd_nonpositive" in r.flags for r in results]
        n_flagged += sum(flagged)
        for r, f in zip(results, flagged):
            if f:
                assert r.valid and r.censor in ("above_range", "below_detection")
        p = pooled_gs_mol(readings, results, c)
        assert p.n_windows == N_WINDOWS and p.n_excluded == 0
        kept.append(p.g_s_mol)
        keep = [i for i, f in enumerate(flagged) if not f]
        dropped.append(pooled_gs_mol([readings[i] for i in keep],
                                     [results[i] for i in keep], c).g_s_mol)
    assert n_flagged > 0
    assert np.mean(dropped) < g_true
    assert abs(np.mean(kept) / g_true - 1) < abs(np.mean(dropped) / g_true - 1)


def test_truncation_bias_with_many_censored_windows(cfg, cal):
    """At g_s 0.02 mol (same noise) about 5% of windows are below_detection. The
    truncated mean must be high on average. The pooled estimate is not asserted here:
    its worst seed is about 11% off (see the uncertainty explainer)."""
    c = with_faces(cfg, 1)
    g_true = 0.02
    truncated, n_censored = [], 0
    for seed in range(N_SEEDS):
        _, results = simulate_stressed_windows(c, cal, g_true, seed)
        n_censored += sum(r.censor == "below_detection" for r in results)
        truncated.append(truncated_mean_mol(results))
    assert n_censored > 0
    assert np.mean(truncated) > g_true
