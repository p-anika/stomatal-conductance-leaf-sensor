"""Tests for analysis/validate_rb_vaseline.py (T5.5): coated-leaf r_bH solved like a card with leaf
optics, compared with flat plate and ML. Synthetic coated-leaf sessions only."""
from __future__ import annotations

import copy
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from analysis import validate_rb_vaseline as vr
from ml.synthetic_data import make_synthetic_cards, simulate_card_session, synthetic_calibration
from processing.data_io import CARD_COLUMNS_ORDER, ReferenceCard, Session

F_TRUE = [1.3, 0.8, 1.0]


class ConstModel:
    def __init__(self, f: float) -> None:
        self.f = f

    def predict(self, x):
        return np.full(len(x), self.f)


def stub_model(cfg, f: float) -> dict:
    return {"features": list(cfg["ml"]["features"]), "model": ConstModel(f), "bootstrap_models": [ConstModel(f)] * 2,
            "residual_sd_f": 0.0, "interval_level": 0.95, "status": "ok", "feature_ranges": {}}


def cards_dict(table: pd.DataFrame) -> dict[str, ReferenceCard]:
    out = {}
    for _, r in table.iterrows():
        vals = {c: r[c] for c in CARD_COLUMNS_ORDER}
        vals["photo_path"] = vals["photo_path"] or None
        out[r["card_id"]] = ReferenceCard(**vals)
    return out


def coated_leaves(cfg):
    """Synthetic coated leaves: leaf a_s and emissivity, planted f_true, vaseline_leaf sessions."""
    cards, truth = make_synthetic_cards(len(F_TRUE), seed=5, cfg=cfg)
    cards["a_s_card"] = cfg["light_sources"]["led_lamp"]["a_s_leaf"]
    cards["emissivity"] = cfg["physics"]["leaf_emissivity_longwave"]
    cards["source"], cards["coating"] = "traced_leaf", "vaseline"
    truth["f_true"] = F_TRUE
    sessions = []
    for i in range(len(F_TRUE)):
        sim = simulate_card_session(cards.iloc[i], truth.iloc[i], cfg=cfg, seed=i, noise=False,
                                    metadata={"dry_reference_type": "vaseline_leaf", "plant_id": f"VAS{i}"})
        sessions.append(Session(sim.session_id, asdict(sim.metadata), sim.sensor_log, sim.events,
                                sim.manifest, {}, True))
    return sessions, cards_dict(cards)


@pytest.fixture()
def cfg(default_config):
    return copy.deepcopy(default_config)


def test_flat_plate_bias_recovers_planted_f(cfg):
    sessions, cards = coated_leaves(cfg)
    out = vr.run(sessions, cards, cfg, synthetic_calibration(cfg), model=None, label="SYNTHETIC")
    cases = out["cases"].set_index("session_id")
    for s, f in zip(sessions, F_TRUE):
        assert cases.loc[s.session_id, "ratio_flat_plate"] == pytest.approx(1.0 / f, rel=1e-6)
    s = out["summary"]
    flat = s[(s["method"] == "flat_plate") & (s["level"] == vr.LEVEL_CASE)].iloc[0]
    assert flat["n"] == 3
    assert flat["mean_ratio"] == pytest.approx(np.mean([1 / f for f in F_TRUE]), rel=1e-6)
    assert "p_value" not in s.columns                                   # case-level numbers only
    assert set(s["data_label"]) == {"SYNTHETIC"}


def test_ml_prediction_compared(cfg):
    sessions, cards = coated_leaves(cfg)
    out = vr.run(sessions, cards, cfg, synthetic_calibration(cfg), model=stub_model(cfg, 1.3), label="SYNTHETIC")
    cases = out["cases"].set_index("session_id")
    assert cases.loc[sessions[0].session_id, "ratio_ml_corrected"] == pytest.approx(1.0, rel=1e-6)
    assert cases.loc[sessions[1].session_id, "ratio_ml_corrected"] == pytest.approx(1.3 / 0.8, rel=1e-6)


def test_no_model_ml_rows_invalid_and_flagged(cfg):
    sessions, cards = coated_leaves(cfg)
    out = vr.run(sessions, cards, cfg, synthetic_calibration(cfg), model=None, label="SYNTHETIC")
    w = out["windows"]
    assert w["r_bh_ml_corrected"].isna().all()
    assert w.loc[w["valid"], "flags"].str.contains("ml_model_missing").all()
    assert w["flags"].str.contains("coated_leaf_optics_assumed").all()


def test_only_vaseline_sessions_used(cfg):
    sessions, cards = coated_leaves(cfg)
    other = copy.copy(sessions[0])
    meta = dict(other.metadata, dry_reference_type="card")
    other = Session("OTHER", meta, other.sensor, other.events, other.manifest, {}, True)
    out = vr.run([other] + sessions[1:], cards, cfg, synthetic_calibration(cfg), model=None, label="SYNTHETIC")
    assert "OTHER" not in set(out["cases"]["session_id"])
    assert len(out["cases"]) == 2


def test_leaf_optics_override_does_not_touch_inputs(cfg):
    cal = synthetic_calibration(cfg)
    cfg2, cal2 = vr.leaf_optics(cfg, cal, "led_lamp")
    assert cfg2["card"]["a_s_card"] == cfg["light_sources"]["led_lamp"]["a_s_leaf"]
    assert cal2["card"]["a_s_card"] == cfg["light_sources"]["led_lamp"]["a_s_leaf"]
    assert cfg2["card"]["emissivity"] == cfg["physics"]["leaf_emissivity_longwave"]
    assert cal["card"]["a_s_card"] == cfg["synthetic"]["a_s_card"] and cfg["card"]["a_s_card"] is None
