"""Tests for ml/synthetic_data.py (T4.2).

Done-when (spec): the outputs pass the same schema checks as real data. Also checked:
the forward physics inverts exactly (noise off), noise and the optional per-session bias
behave as configured, the null card set has f_true = 1, the held-out set differs, and the
real config is never given a lamp calibration (owner decisions 2026-11-09).

Synthetic results prove the code works. They say nothing about real plants.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ml.synthetic_data import (
    hidden_correction,
    make_card_sets,
    make_synthetic_cards,
    simulate_card_session,
    simulate_leaf_session,
    write_reference_cards,
    write_session,
)
from processing.config import load_config
from processing.data_io import load_photo_features, load_reference_cards, load_session
from processing.gs_calculator import LeafReading, solve_gs
from processing.measurement_windows import build_windows
from processing.physics_baseline import ReferenceCardReading, solve_card_rbh

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def cfg() -> dict[str, Any]:
    return load_config(REPO / "config.yaml")


@pytest.fixture
def cards(cfg):
    return make_synthetic_cards(cfg=cfg)


# ---------- cards ----------

def test_cards_pass_real_manifest_checks(tmp_path, cfg, cards):
    table, truth = cards
    assert len(table) == 20 and len(truth) == 20
    path = write_reference_cards(table, tmp_path / "reference_cards.csv")
    loaded = load_reference_cards(path)
    assert set(loaded) == set(table["card_id"])
    assert "f_true" not in table.columns                      # truth kept out of the schema table


def test_hidden_correction_formula(cfg, cards):
    table, truth = cards
    s = cfg["synthetic"]
    merged = table.merge(truth, on="card_id")
    det = 1 + s["correction_a"] * (1 - merged["solidity"]) + s["correction_b"] * (merged["aspect_ratio"] - 1) \
        + s["correction_c"] * merged["edge_density"]
    assert np.allclose(merged["f_deterministic"], det)
    np.testing.assert_allclose(hidden_correction(merged, cfg), det)
    resid = merged["f_true"] - det
    assert 0.02 < resid.std() < 0.1                           # noise sd 0.05


def test_geometry_consistent(cards):
    t, _ = cards
    assert np.allclose(t["aspect_ratio"], t["d_max_mm"] / t["d_min_mm"]) if "d_max_mm" in t else True
    assert np.allclose(t["d_eq_mm"], np.sqrt(4 * t["area_mm2"] / np.pi))
    assert np.allclose(t["eccentricity"], np.sqrt(1 - 1 / t["aspect_ratio"] ** 2))
    assert ((t["circularity"] > 0) & (t["circularity"] <= 1)).all()
    assert t["shape_group"].is_unique


def test_seeds_reproducible_and_distinct(cfg):
    a, _ = make_synthetic_cards(seed=0, cfg=cfg)
    b, _ = make_synthetic_cards(seed=0, cfg=cfg)
    c, _ = make_synthetic_cards(seed=1, cfg=cfg)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["solidity"], c["solidity"])
    assert set(a["card_id"]).isdisjoint(c["card_id"])


def test_card_sets_train_test_null(cfg):
    sets = make_card_sets(cfg=cfg)
    assert set(sets) == {"train", "test", "null"}
    null_truth = sets["null"][1]
    assert (null_truth["f_true"] == 1.0).all()
    assert set(sets["train"][0]["card_id"]).isdisjoint(sets["test"][0]["card_id"])
    assert set(sets["train"][0]["shape_group"]).isdisjoint(sets["test"][0]["shape_group"])


# ---------- forward physics round trips (noise off) ----------

def test_card_session_inverts_exactly(cfg, cards):
    table, truth = cards
    sim = simulate_card_session(table.iloc[0], truth.iloc[0], cfg=cfg, noise=False)
    row = sim.sensor_log.iloc[0]
    r = solve_card_rbh(ReferenceCardReading(card_id=row_card(sim), card_temp_c=row["card_temp_c"],
                                            air_temp_c=row["air_temp_c"],
                                            surroundings_temp_c=row["air_temp_c"], lux=row["lux"],
                                            light_source="led_lamp"), cfg, sim.cal)
    assert r.valid
    assert r.r_bh == pytest.approx(sim.truth["r_bh_card_true"], rel=1e-6)
    assert sim.truth["r_bh_card_true"] == pytest.approx(
        sim.truth["f_card_true"] * sim.truth["r_bh_card_flat"], rel=1e-9)


def row_card(sim) -> str:
    return sim.metadata.card_id


@pytest.mark.parametrize("g_s_mol", [0.05, 0.3, 0.6])
def test_leaf_session_inverts_exactly(cfg, cards, g_s_mol):
    table, truth = cards
    sim = simulate_leaf_session(g_s_mol, table.iloc[3], truth.iloc[3], cfg=cfg, noise=False)
    row = sim.sensor_log.iloc[0]
    r = solve_gs(LeafReading(leaf_temp_c=row["leaf_temp_c"], air_temp_c=row["air_temp_c"],
                             surroundings_temp_c=row["air_temp_c"], relative_humidity_pct=row["humidity_pct"],
                             pressure_hpa=row["pressure_hpa"], lux=row["lux"], light_source="led_lamp",
                             r_bh=sim.truth["r_bh_leaf_true"], r_bh_method="card_direct"), cfg, sim.cal)
    assert r.valid and r.g_s_mol == pytest.approx(g_s_mol, rel=1e-6)
    assert pd.isna(row["card_temp_c"])                        # no card given


def test_leaf_session_with_card(cfg, cards):
    table, truth = cards
    sim = simulate_leaf_session(0.3, table.iloc[3], truth.iloc[3], card=table.iloc[0],
                                card_truth=truth.iloc[0], cfg=cfg, noise=False)
    assert sim.sensor_log["card_temp_c"].notna().all()
    assert sim.metadata.card_id == table.iloc[0]["card_id"]
    assert sim.truth["r_bh_card_true"] > 0 and sim.truth["r_bh_leaf_true"] > 0


# ---------- schema: outputs pass the real loaders ----------

def test_written_session_loads_like_real_data(tmp_path, cfg, cards):
    table, truth = cards
    sim = simulate_leaf_session(0.3, table.iloc[1], truth.iloc[1], card=table.iloc[0],
                                card_truth=truth.iloc[0], cfg=cfg, seed=4)
    write_session(sim, tmp_path)
    s = load_session(sim.session_id, tmp_path / "data" / "raw")
    assert s.synthetic is True
    assert len(s.sensor) == len(sim.sensor_log)
    pf = load_photo_features(tmp_path / "data" / "processed" / sim.session_id / "photo_features.csv")
    assert pf["ok"].all() and pf["timestamp"].notna().all()
    assert (tmp_path / "data" / "processed" / sim.session_id / "synthetic_truth.json").is_file()
    assert not (tmp_path / "data" / "raw" / sim.session_id / "synthetic_truth.json").exists()
    w = build_windows(s.sensor, s.events, s.metadata, cfg, photo_features=pf)   # loaded tables work too
    assert w.iloc[3:]["valid"].all()
    with pytest.raises(FileExistsError):
        write_session(sim, tmp_path)                           # raw folders are never overwritten


def test_synthetic_session_runs_through_windows(cfg, cards):
    table, truth = cards
    sim = simulate_leaf_session(0.3, table.iloc[1], truth.iloc[1], card=table.iloc[0],
                                card_truth=truth.iloc[0], cfg=cfg, seed=4)
    w = build_windows(sim.sensor_log, sim.events, sim.metadata, cfg, photo_features=sim.photo_features)
    later = w.iloc[3:]                                        # first windows lack 3 min of history
    assert later["valid"].all(), later[["reason", "flags"]]
    assert later["d_min_mm"].notna().all()


# ---------- noise and per-session bias ----------

def test_random_noise_averages_out(cfg, cards):
    table, truth = cards
    sim = simulate_card_session(table.iloc[0], truth.iloc[0], cfg=cfg, seed=2)
    log = sim.sensor_log
    sd = cfg["synthetic"]["noise"]["card_temp_c"]
    assert log["card_temp_c"].std() == pytest.approx(sd, rel=0.3)
    assert log["card_temp_c"].mean() == pytest.approx(sim.truth["card_temp_true_c"], abs=4 * sd / np.sqrt(len(log)))
    assert sim.truth["bias"] == {"thermal_c": 0.0, "air_temp_c": 0.0, "humidity_pct": 0.0, "lux_factor": 1.0}


def test_session_bias_shifts_and_is_recorded(cfg, cards):
    table, truth = cards
    c = copy.deepcopy(cfg)
    c["synthetic"]["session_bias"] = True
    sim = simulate_leaf_session(0.3, table.iloc[2], truth.iloc[2], card=table.iloc[0],
                                card_truth=truth.iloc[0], cfg=c, seed=7)
    b = sim.truth["bias"]
    assert b["thermal_c"] != 0.0 and b["lux_factor"] != 1.0
    log = sim.sensor_log
    n = len(log)
    sd = c["synthetic"]["noise"]
    # one shared camera offset on leaf and card; separate offsets on air, RH, lux
    assert log["leaf_temp_c"].mean() - sim.truth["leaf_temp_true_c"] == pytest.approx(
        b["thermal_c"], abs=4 * sd["leaf_temp_c"] / np.sqrt(n))
    assert log["card_temp_c"].mean() - sim.truth["card_temp_true_c"] == pytest.approx(
        b["thermal_c"], abs=4 * sd["card_temp_c"] / np.sqrt(n))
    assert log["air_temp_c"].mean() - sim.truth["air_temp_true_c"] == pytest.approx(
        b["air_temp_c"], abs=4 * sd["air_temp_c"] / np.sqrt(n))
    # an offset does not average away: g_s from the biased session-mean misses the truth
    m = log.mean(numeric_only=True)
    r = solve_gs(LeafReading(leaf_temp_c=m["leaf_temp_c"], air_temp_c=m["air_temp_c"],
                             surroundings_temp_c=m["air_temp_c"], relative_humidity_pct=m["humidity_pct"],
                             pressure_hpa=m["pressure_hpa"], lux=m["lux"], light_source="led_lamp",
                             r_bh=sim.truth["r_bh_leaf_true"], r_bh_method="card_direct"), c, sim.cal)
    assert abs(r.g_s_mol - 0.3) / 0.3 > 0.05 or r.censor != "none"


def test_bias_off_by_default(cfg):
    assert cfg["synthetic"]["session_bias"] is False


# ---------- real config untouched ----------

def test_real_config_lamp_k_still_null(cfg, cards):
    assert cfg["light_sources"]["led_lamp"]["k_wm2_per_lux"] is None
    before = copy.deepcopy(cfg)
    table, truth = cards
    sim = simulate_card_session(table.iloc[0], truth.iloc[0], cfg=cfg, seed=0)
    assert cfg == before                                       # generator does not edit cfg
    assert sim.cal["light_sources"]["led_lamp"]["k_wm2_per_lux"] == cfg["synthetic"]["k_wm2_per_lux"]
    assert load_config(REPO / "config.yaml")["light_sources"]["led_lamp"]["k_wm2_per_lux"] is None
