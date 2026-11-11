"""Tests for the session-pooled card r_bH (option C, decision_log 2026-11-11): selection only on
exogenous criteria, kept-and-counted suspect sessions, wind-variation flag, and the A-vs-C
selection-bias experiment (direction only)."""
from __future__ import annotations

import copy
from dataclasses import asdict

import pandas as pd
import pytest

from analysis import aggregate_results as ar
from analysis import card_selection_bias as csb
from analysis import genotype_comparison as gc
from ml.synthetic_data import synthetic_calibration
from processing.data_io import Session
from test_genotype_comparison import PLANTED, make_results


def rec(card, air, rn, pool_ok=True):
    return {"card_temp_c": card, "air_temp_c": air, "rn_wm2": rn, "pool_ok": pool_ok}


def test_pooled_is_ratio_of_sums_and_keeps_low_windows(default_config):
    wins = [rec(26.0, 24.0, 60.0), rec(23.5, 24.0, 55.0), rec(30.0, 24.0, 10.0, pool_ok=False)]
    r, reason, n = ar.pooled_card_rbh(wins, default_config)
    rho_cp = 1.204 * 1005.0
    assert n == 2 and reason == "ok"
    assert r == pytest.approx(rho_cp * ((26.0 - 24.0) + (23.5 - 24.0)) / (60.0 + 55.0))   # window with dT < 0 kept


def test_pooled_nonpositive_is_suspect_and_empty_is_missing(default_config):
    r, reason, _ = ar.pooled_card_rbh([rec(23.0, 24.0, 60.0), rec(23.8, 24.0, 60.0)], default_config)
    assert reason == "suspect_card_rbh" and r < 0
    r, reason, n = ar.pooled_card_rbh([rec(26.0, 24.0, 60.0, pool_ok=False)], default_config)
    assert r is None and reason == "missing_input" and n == 0


def _session(cfg, delta_c=0.0, noise=False, irradiance=120.0, wind=0.8, seed=1):
    sim, cards = csb.simulate_session(cfg, g_s_mol=0.3, delta_c=delta_c, irradiance_wm2=irradiance, wind_ms=wind,
                                      seed=seed, noise=noise, plant_id="P1", metadata={})
    return sim, cards


def test_pooled_matches_truth_without_offset(default_config):
    cfg = copy.deepcopy(default_config)
    cal = synthetic_calibration(cfg)
    sim, cards = _session(cfg, delta_c=0.0, noise=False)
    s = Session(sim.session_id, asdict(sim.metadata), sim.sensor_log, sim.events, sim.manifest, {}, True)
    card = ar.session_card(ar.card_windows(s, cfg, cal), cfg, cal)
    assert card["valid"]
    assert card["r_bh"] == pytest.approx(sim.truth["r_bh_card_true"], rel=0.005)


def test_suspect_session_rows_kept_with_reason(default_config):
    cfg = copy.deepcopy(default_config)
    cfg["uncertainty"].update(n_draws=100, dl_grid_points=6)
    cal = synthetic_calibration(cfg)
    sim, cards = _session(cfg, delta_c=-10.0)        # card reads 10 C low: pooled card r_bH < 0
    s = Session(sim.session_id, asdict(sim.metadata), sim.sensor_log, sim.events, sim.manifest, {}, True)
    rows = pd.DataFrame(ar.session_results(s, sim.photo_features, cards, cfg, cal, None))
    cd = rows[rows["method"] == "card_direct"]
    steady = cd[cd["steady_state"].astype(bool)]
    assert len(steady) > 0 and (steady["reason"] == "suspect_card_rbh").all() and not steady["valid"].any()


def test_card_source_switch_keeps_old_method(default_config):
    cfg = copy.deepcopy(default_config)
    assert cfg["windows"]["card_source"] == "session_pooled"
    cfg["uncertainty"].update(n_draws=100, dl_grid_points=6)
    cal = synthetic_calibration(cfg)
    sim, cards = _session(cfg, noise=True)
    s = Session(sim.session_id, asdict(sim.metadata), sim.sensor_log, sim.events, sim.manifest, {}, True)
    pooled = pd.DataFrame(ar.session_results(s, sim.photo_features, cards, cfg, cal, None))
    cfg["windows"]["card_source"] = "nearest_window"
    nearest = pd.DataFrame(ar.session_results(s, sim.photo_features, cards, cfg, cal, None))
    p = pooled[(pooled["method"] == "card_direct") & pooled["valid"].astype(bool)]["r_bh"]
    q = nearest[(nearest["method"] == "card_direct") & nearest["valid"].astype(bool)]["r_bh"]
    assert p.nunique() == 1                          # one card r_bH per session
    assert q.nunique() > 1                           # old: one per matched window


def test_wind_variation_flag(default_config):
    cfg = copy.deepcopy(default_config)
    cfg["uncertainty"].update(n_draws=100, dl_grid_points=6)
    cal = synthetic_calibration(cfg)
    sim, cards = _session(cfg, noise=False)
    log = sim.sensor_log.copy()
    half = len(log) // 2
    log.loc[log.index[half:], "wind_ms"] = 2.0       # fan setting changes mid-session (0.8 -> 2.0)
    s = Session(sim.session_id, asdict(sim.metadata), log, sim.events, sim.manifest, {}, True)
    rows = pd.DataFrame(ar.session_results(s, sim.photo_features, cards, cfg, cal, None))
    cd = rows[(rows["method"] == "card_direct") & rows["valid"].astype(bool)]
    assert cd["flags"].str.contains("wind_varies_in_session").all()


def test_suspect_share_by_group_in_t53(default_config):
    df = make_results(PLANTED, n_plants=3, n_windows=4)
    idx = df.index[(df["method"] == "card_direct") & (df["genotype"] == "Williams_82")][:6]
    df.loc[idx, ["valid", "reason"]] = [False, "suspect_card_rbh"]
    t = gc.run(df, default_config, campaign_dl=None, label="SYNTHETIC")["excluded_by_group"]
    t = t[(t["outcome"] == "card_direct") & (t["subset"] == "all")]
    assert t["n_suspect_card_rbh"].sum() == 6
    assert {"share_suspect_card_rbh"} <= set(t.columns)


def test_selection_bias_direction_at_sigma_05_low_light(default_config):
    """The only experiment assertion (decision_log design row): at sigma 0.5 C, random per-session
    offsets, low light, |bias C| < |bias A|. Everything else is reported."""
    res = csb.run_design(default_config, design="random", sigma_c=0.5, irradiance_wm2=60.0, n_plants=10, seed=0)
    a = res.loc[res["method"] == "A_nearest_window", "bias_mol"].iloc[0]
    c = res.loc[res["method"] == "C_session_pooled", "bias_mol"].iloc[0]
    assert abs(c) < abs(a)
