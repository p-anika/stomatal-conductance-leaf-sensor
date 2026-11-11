"""Tests for analysis/card_buoyancy_check.py (T5.9): synthetic card sessions generated with the
forced-plus-free Nusselt model at several lamp levels and one wind; the planted delta-T
dependence must be recovered."""
from __future__ import annotations

import copy
from dataclasses import asdict

import numpy as np
import pytest

from analysis import card_buoyancy_check as cb
from ml.build_training_set import build_training_set
from ml.synthetic_data import make_synthetic_cards, simulate_card_session, synthetic_calibration
from processing.data_io import Session
from test_validate_rb_vaseline import cards_dict

LAMPS = (100.0, 200.0, 400.0, 800.0)


def training_rows(cfg, lamps=LAMPS, wind=0.3, f_true=(1.2, 0.9)):
    cards, truth = make_synthetic_cards(len(f_true), seed=7, cfg=cfg)
    truth["f_true"] = list(f_true)
    sessions = []
    for i in range(len(f_true)):
        for j, irr in enumerate(lamps):
            sim = simulate_card_session(cards.iloc[i], truth.iloc[i], cfg=cfg, seed=10 * i + j, noise=False,
                                        conditions={"irradiance_wm2": irr, "wind_ms": wind},
                                        metadata={"plant_id": f"CB{i}{j}"})
            sessions.append(Session(sim.session_id, asdict(sim.metadata), sim.sensor_log, sim.events,
                                    sim.manifest, {}, True))
    cal = synthetic_calibration(cfg)
    return build_training_set(sessions, cards_dict(cards), cfg, cal, force=True).kept, cal


@pytest.fixture()
def cfg(default_config):
    return copy.deepcopy(default_config)


def test_planted_nusselt_dependence_recovered(cfg):
    ts, cal = training_rows(cfg)
    out = cb.run(ts, cfg, cal, label="SYNTHETIC")
    assert len(out) == 2 and (out["status"] == "ok").all()
    for _, r in out.iterrows():
        assert r["n_lamp_levels"] == 4
        # the fit has the same form as the planted model, so it recovers it
        assert r["frac_change_fit"] == pytest.approx(r["frac_change_model"], abs=0.005)
        assert r["frac_change_fit_low"] == pytest.approx(r["frac_change_model_low"], abs=0.01)
        assert r["frac_change_fit"] > 0                     # smaller delta T -> less buoyancy -> larger r_bH
        assert r["sw_slope"] < 0                            # more light -> warmer card -> lower r_bH
        expected = abs(r["frac_change_fit"]) > cfg["convection"]["card_transfer_max_change"]
        assert bool(r["card_transfer_unreliable"]) == expected
    assert set(out["data_label"]) == {"SYNTHETIC"}


def test_no_buoyancy_no_change_not_flagged(cfg):
    cfg["convection"]["free_nu_coeff"] = 1e-9                # planted: no free convection
    ts, cal = training_rows(cfg, wind=1.0)
    out = cb.run(ts, cfg, cal, label="SYNTHETIC")
    for _, r in out.iterrows():
        assert abs(r["frac_change_fit"]) < 0.005
        assert not bool(r["card_transfer_unreliable"])
        assert abs(r["sw_rel_change"]) < 0.005


def test_too_few_lamp_levels(cfg):
    ts, cal = training_rows(cfg, lamps=(200.0, 400.0))
    out = cb.run(ts, cfg, cal, label="SYNTHETIC")
    assert (out["status"] == "too_few_lamp_levels").all()
    assert out["card_transfer_unreliable"].isna().all()


def test_wls_line_exact():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = 2.0 + 0.5 * x + np.array([0.01, -0.01, 0.01, -0.01])
    fit = cb.wls_line(x, y, np.ones(4), ci_level=0.95)
    assert fit["slope"] == pytest.approx(0.5, abs=0.01)
    assert fit["slope_lo"] < fit["slope"] < fit["slope_hi"]
