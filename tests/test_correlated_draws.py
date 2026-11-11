"""Tests for correlated card and leaf Monte Carlo draws (A43, uncertainty.card_leaf_shared_draws):
with a fully shared thermal and air offset the card_direct detection limit falls and the
interval narrows; with no sharing the old lognormal r_bH draws are unchanged."""
from __future__ import annotations

import copy
import math
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from analysis.aggregate_results import rb_sigma_from_interval, session_results
from ml.synthetic_data import make_synthetic_cards, simulate_leaf_session, synthetic_calibration
from processing.data_io import Session
from processing.gs_calculator import LeafReading
from processing.physics_baseline import ReferenceCardReading, solve_card_rbh
from processing.uncertainty import (card_link, card_rbh_uncertainty, detection_limit_mol, get_sigmas, gs_draws,
                                    gs_uncertainty)
from processing.config import get_light_source
from test_validate_rb_vaseline import cards_dict

TARGET = {"sigma_leaf_temp_c": 0.2, "sigma_lux_fraction": 0.05}


@pytest.fixture()
def setup(default_config):
    cfg = copy.deepcopy(default_config)
    cal = synthetic_calibration(cfg)
    cal["uncertainty"] = dict(TARGET)
    card = ReferenceCardReading("C1", card_temp_c=26.5, air_temp_c=24.0, surroundings_temp_c=24.0, lux=12000.0,
                                light_source="led_lamp", c_card=1.0)
    r_card = solve_card_rbh(card, cfg, cal).r_bh
    leaf = LeafReading(24.2, 24.0, 24.0, 50.0, 995.0, 12000.0, "led_lamp", float(r_card), "card_direct")
    cu = card_rbh_uncertainty(card, cfg, cal, n=2000, seed=0)
    base = get_sigmas(cfg, cal, get_light_source(cfg, cal, "led_lamp"))
    from dataclasses import replace
    s = replace(base, rb_fraction=rb_sigma_from_interval(cu.r_bh_lo, cu.r_bh_hi))
    return cfg, cal, card, leaf, s


def test_no_sharing_draws_unchanged(setup):
    cfg, cal, _, leaf, s = setup
    d = gs_draws(leaf, cfg, cal, s, n=500, seed=3)
    z = np.random.default_rng(3).standard_normal((5, 500))
    assert np.allclose(d.r_bh, leaf.r_bh * np.exp(s.rb_fraction * z[4]))
    assert np.allclose(d.t_leaf, leaf.leaf_temp_c + s.leaf_temp_c * z[0])


def test_shared_draws_reuse_leaf_thermal_and_air_draws(setup):
    cfg, cal, card, leaf, s = setup
    d = gs_draws(leaf, cfg, cal, s, n=500, seed=3, card=card_link(card, cfg, cal))
    z = np.random.default_rng(3).standard_normal((5, 500))
    dt = d.t_leaf - leaf.leaf_temp_c
    # every kept draw: the leaf offset is one of the z1 draws (same stream), card r_bH is re-solved
    assert np.isin(np.round(dt, 12), np.round(s.leaf_temp_c * z[0], 12)).all()
    assert len(d.r_bh) <= 500 and (d.r_bh > 0).all()


def test_fully_shared_offset_lowers_card_direct_detection_limit(setup):
    """Scenario: air 0.2 C and RH 2 % calibrated too. (With the 0.5 C air placeholder the shared air
    error works against the leaf at high g_s and nothing resolves; recorded as a T6.1 finding.)"""
    from dataclasses import replace
    cfg, cal, card, leaf, s = setup
    s = replace(s, air_temp_c=0.2, rh_pct=2.0)
    dl_indep, nr1 = detection_limit_mol(leaf, cfg, cal, s, 2000, 0)
    dl_shared, nr2 = detection_limit_mol(leaf, cfg, cal, s, 2000, 0, card=card_link(card, cfg, cal))
    assert nr1 == nr2 == 0
    assert dl_shared is not None and dl_indep is not None
    assert dl_shared < 0.6 * dl_indep                               # 0.054 vs 0.118 mol m-2 s-1
    u_indep = gs_uncertainty(leaf, cfg, cal, sigmas=s, n=2000)
    u_shared = gs_uncertainty(leaf, cfg, cal, sigmas=s, n=2000, card=card_link(card, cfg, cal))
    assert (u_shared.g_s_mol_hi - u_shared.g_s_mol_lo) < (u_indep.g_s_mol_hi - u_indep.g_s_mol_lo)


def _leaf_session(cfg):
    cards, truth = make_synthetic_cards(2, seed=4, cfg=cfg)
    sim = simulate_leaf_session(0.3, cards.iloc[1], truth.iloc[1], card=cards.iloc[0], card_truth=truth.iloc[0],
                                cfg=cfg, seed=2, conditions={"duration_min": 6.0, "wind_ms": 0.8})
    return Session(sim.session_id, asdict(sim.metadata), sim.sensor_log, sim.events, sim.manifest, {}, True), \
        sim.photo_features, cards_dict(cards)


def test_config_key_switches_only_card_direct(default_config):
    cfg = copy.deepcopy(default_config)
    cfg["uncertainty"].update(n_draws=300, dl_grid_points=10, **TARGET)
    cal = synthetic_calibration(cfg)
    s, pf, cards = _leaf_session(cfg)
    off = pd.DataFrame(session_results(s, pf, cards, cfg, cal, None))
    cfg_absent = copy.deepcopy(cfg)
    del cfg_absent["uncertainty"]["card_leaf_shared_draws"]
    absent = pd.DataFrame(session_results(s, pf, cards, cfg_absent, cal, None))
    pd.testing.assert_frame_equal(off, absent)                       # key false == old behaviour
    cfg["uncertainty"]["card_leaf_shared_draws"] = True
    on = pd.DataFrame(session_results(s, pf, cards, cfg, cal, None))
    fp = lambda df: df[df["method"] == "flat_plate"].reset_index(drop=True)
    pd.testing.assert_frame_equal(fp(off), fp(on))
    cd_off = off[(off["method"] == "card_direct") & off["valid"].astype(bool)]
    cd_on = on[(on["method"] == "card_direct") & on["valid"].astype(bool)]
    assert np.allclose(cd_off["g_s_mol"], cd_on["g_s_mol"])          # point estimate unchanged
    assert not np.allclose(cd_off["g_s_mol_hi"], cd_on["g_s_mol_hi"])
