"""Tests for the 2026-11-11 review fixes: unusable Monte Carlo draws are kept and counted (shared card
draws and normal r_bH draws), the r_bH sigma uses the real interval level and distribution with a
rb_sigma_fallback flag, dry-run no-effect settings come from config, card_buoyancy flags a default
c_card, and T5.3 reports not_steady_state exclusions by group."""
from __future__ import annotations

import copy
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from analysis import card_buoyancy_check as cb
from analysis import genotype_comparison as gc
from analysis.aggregate_results import RESULT_COLUMNS, ml_rb_fraction, rb_sigma_from_interval, session_results
from processing import pipeline_dry_run as pdr
from processing.config import get_light_source
from processing.gs_calculator import LeafReading
from processing.uncertainty import (UNUSABLE, CardLink, detection_limit_mol, get_sigmas, gs_draws,
                                    gs_uncertainty)
from ml.synthetic_data import synthetic_calibration
from test_correlated_draws import TARGET, _leaf_session
from test_genotype_comparison import PLANTED, make_results


@pytest.fixture()
def base(default_config):
    cfg = copy.deepcopy(default_config)
    cal = synthetic_calibration(cfg)
    cal["uncertainty"] = dict(TARGET)
    s = get_sigmas(cfg, cal, get_light_source(cfg, cal, "led_lamp"))
    leaf = LeafReading(24.2, 24.0, 24.0, 50.0, 995.0, 12000.0, "led_lamp", 35.0, "card_direct")
    return cfg, cal, s, leaf


def _link(card_temp_c: float) -> CardLink:
    return CardLink(card_temp_c=card_temp_c, surroundings_temp_c=24.0, e_s_card_wm2=120.0, a_s_card=0.95,
                    emissivity=0.95)


# ---------------------------------------------------------------- fix 1: unusable draws kept and counted

def test_partial_unusable_card_draws_counted_and_fail_usable_test(base):
    cfg, cal, s, leaf = base
    link = _link(24.25)                                   # card only 0.25 C above air: many draws give r_bH <= 0
    d = gs_draws(leaf, cfg, cal, s, n=2000, seed=0, card=link)
    assert len(d.status) == 2000                          # nothing dropped
    n_bad = int((d.status == UNUSABLE).sum())
    assert 0 < n_bad < 2000
    u = gs_uncertainty(leaf, cfg, cal, sigmas=s, n=2000, card=link)
    assert u.n_draws_unusable == n_bad
    assert u.frac_resolved <= 1.0 - n_bad / 2000 + 1e-12  # unusable draws count as not usable
    if n_bad / 2000 > 1 - cfg["uncertainty"]["dl_min_resolved_fraction"]:
        assert not u.resolved


def test_all_unusable_card_draws_do_not_crash(base):
    cfg, cal, s, leaf = base
    link = _link(19.0)                                    # card 5 C below air: every r_bH < 0
    u = gs_uncertainty(leaf, cfg, cal, sigmas=s, n=500, card=link)
    assert u.valid and not u.resolved
    assert u.n_draws_unusable == 500 and u.all_draws_unusable
    assert u.g_s_mol_lo is None and u.g_s_mol_hi is None and u.g_s_mol_median is None
    dl, no_root = detection_limit_mol(leaf, cfg, cal, s, 200, 0, card=link)
    assert dl is None


def test_lognormal_path_has_no_unusable_draws(base):
    cfg, cal, s, leaf = base
    u = gs_uncertainty(leaf, cfg, cal, sigmas=s, n=500)
    assert u.n_draws_unusable == 0 and not u.all_draws_unusable


def test_normal_rbh_draws_exact_and_nonpositive_counted(base):
    cfg, cal, s, leaf = base
    s_n = replace(s, rb_fraction=0.6, rb_distribution="normal")
    d = gs_draws(leaf, cfg, cal, s_n, n=1000, seed=4)
    z = np.random.default_rng(4).standard_normal((5, 1000))
    r = leaf.r_bh * (1.0 + 0.6 * z[4])
    ok = r > 0
    assert np.allclose(d.r_bh[ok], r[ok])
    assert int((d.status == UNUSABLE).sum()) == int((~ok).sum()) > 0


# ---------------------------------------------------------------- fix 2: r_bH sigma

def test_rb_sigma_uses_interval_level():
    lo, hi = 50.0, 50.0 * math.exp(2 * 1.6448536 * 0.2)        # a 90 % lognormal interval, sigma 0.2
    assert rb_sigma_from_interval(lo, hi, level=0.90) == pytest.approx(0.2, rel=1e-5)
    assert rb_sigma_from_interval(lo, hi) != pytest.approx(0.2, rel=1e-3)   # default 95 % would be wrong


def test_ml_rb_fraction_gaussian_in_f():
    # f_hat 1.2, sd 0.1, 90 % interval -> f_lo/hi = 1.2 -/+ 1.6449 x 0.1
    z = 1.6448536
    assert ml_rb_fraction(1.2, 1.2 - z * 0.1, 1.2 + z * 0.1, 0.90) == pytest.approx(0.1 / 1.2, rel=1e-5)
    assert ml_rb_fraction(1.2, -0.1, 2.5, 0.95) is None              # f_lo <= 0 -> fallback


class _Const:
    def __init__(self, f):
        self.f = f

    def predict(self, x):
        return np.full(len(x), self.f)


def _stub(cfg, f, sd, level=0.95):
    return {"features": list(cfg["ml"]["features"]), "model": _Const(f), "bootstrap_models": [_Const(f)] * 2,
            "residual_sd_f": sd, "interval_level": level, "status": "ok", "feature_ranges": {}}


def test_ml_rows_fallback_flag_and_column(default_config):
    cfg = copy.deepcopy(default_config)
    cfg["uncertainty"].update(n_draws=200, dl_grid_points=8, **TARGET)
    cal = synthetic_calibration(cfg)
    s, pf, cards = _leaf_session(cfg)
    wide = pd.DataFrame(session_results(s, pf, cards, cfg, cal, _stub(cfg, 1.0, 0.8)))     # f_lo < 0
    ml = wide[(wide["method"] == "ml_corrected") & wide["valid"].astype(bool)]
    assert len(ml) and ml["flags"].str.contains("rb_sigma_fallback").all()
    narrow = pd.DataFrame(session_results(s, pf, cards, cfg, cal, _stub(cfg, 1.0, 0.05)))
    ml = narrow[(narrow["method"] == "ml_corrected") & narrow["valid"].astype(bool)]
    assert len(ml) and not ml["flags"].str.contains("rb_sigma_fallback").any()
    assert "n_draws_unusable" in RESULT_COLUMNS
    assert narrow["n_draws_unusable"].notna().all() or narrow["valid"].eq(False).any()


# ---------------------------------------------------------------- fix 3: config constants, c_card flag

def test_null_check_settings_from_config(default_config):
    cfg = copy.deepcopy(default_config)
    assert set(cfg["dry_run"]["null_check"]) >= {"gs_mol", "plant_sd_mol", "window_sd_mol", "n_plants",
                                                 "n_windows", "alpha"}
    cfg["dry_run"]["null_check"]["alpha"] = 1.01                   # every p (<= 1) is below 1.01 -> rate 1
    r = pdr.null_rank_rates(cfg, n_frames=3, seed=0)
    assert all(v == 1.0 for k, v in r.items() if k != "n_frames")


def test_card_buoyancy_flags_default_c_card(default_config):
    from test_card_buoyancy_check import training_rows
    ts, cal = training_rows(copy.deepcopy(default_config))
    ts.loc[ts.index[:3], "c_card"] = np.nan
    out = cb.run(ts, default_config, cal, label="SYNTHETIC")
    first = ts.loc[ts.index[0], "card_id"]
    assert "c_position_default" in out.loc[out["card_id"] == first, "flags"].iloc[0]


# ---------------------------------------------------------------- fix 5: steady-state exclusions by group

def test_not_steady_share_by_group(default_config):
    df = make_results(PLANTED, n_plants=3, n_windows=4)
    idx = df.index[(df["method"] == "card_direct") & (df["genotype"] == "Williams_82")][:5]
    df.loc[idx, ["valid", "reason", "steady_state"]] = [False, "not_steady_state", False]
    t = gc.run(df, default_config, campaign_dl=None, label="SYNTHETIC")["excluded_by_group"]
    t = t[(t["outcome"] == "card_direct") & (t["subset"] == "all")].set_index(["genotype", "water_treatment"])
    assert t["n_not_steady_state"].sum() == 5
    assert t.loc[("Houjaku_Kuwazu", "well_watered"), "share_not_steady_state"] == 0.0
    assert t["n_windows_in"].sum() == 4 * 3 * 4
