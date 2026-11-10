"""Tests for analysis/aggregate_results.py (T5.1) on a synthetic campaign.

Owner decisions 2026-11-10: every leaf window gets one row per method (card_direct primary,
ml_corrected secondary, flat_plate baseline), including non-steady windows (invalid, with their
reason); censored rows are kept with their label and detection limit; the g_s interval uses each
method's own r_bH interval (sigma = ln(hi/lo) / 3.92) for card_direct and ml_corrected and
uncertainty.sigma_rb_fraction for flat_plate; Jones I_g only with a vaseline_leaf dry reference
(flag jones_no_leaf_dry_reference otherwise); jones_ig_censor column; missing model ->
missing_input + ml_model_missing. Synthetic results prove the code works, nothing about plants.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from analysis.aggregate_results import (
    APPENDIX_B_COLUMNS,
    RESULT_COLUMNS,
    aggregate,
    dl_flags,
    main as agg_main,
    rb_sigma_from_interval,
)
from ml.build_training_set import build_training_set
from ml.synthetic_data import (
    make_card_sets,
    simulate_card_session,
    simulate_leaf_session,
    synthetic_calibration,
    write_reference_cards,
    write_session,
)
from ml.train_correction_model import save_model, train_correction_model
from processing.config import load_config
from processing.data_io import list_sessions, load_reference_cards, load_session

REPO = Path(__file__).resolve().parent.parent
G_TRUE = 0.3                     # mol m-2 s-1, invented
WIND = 2.0                       # m s-1: forced convection, so the card transfers (A17)
DURATION_MIN = 8.0               # windows 0-7; windows 0 and 1 lack 3 min of history
N_TRAIN_SHAPES = 12


def fast_cfg() -> dict[str, Any]:
    cfg = load_config(REPO / "config.yaml")
    cfg["uncertainty"]["n_draws"] = 300          # speed only; production values stay in config
    cfg["uncertainty"]["dl_grid_points"] = 12
    # Calibration-sized sigmas (TEST ONLY, invented). With the config placeholders (leaf 1.0 C,
    # lux 15 %) no g_s is resolved at all, so the detection limit would always be blank.
    cfg["uncertainty"].update(sigma_leaf_temp_c=0.2, sigma_air_temp_c=0.1, sigma_rh_pct=1.0,
                              sigma_lux_fraction=0.05, sigma_rb_fraction=0.1)
    cfg["ml"]["forest_n_estimators"] = 20
    cfg["ml"]["n_bootstrap"] = 30
    return cfg


@pytest.fixture(scope="module")
def cfg() -> dict[str, Any]:
    return fast_cfg()


@pytest.fixture(scope="module")
def sets(cfg):
    return make_card_sets(cfg)


@pytest.fixture(scope="module")
def model(tmp_path_factory, cfg, sets):
    """A correction model trained on synthetic card sessions (12 shapes x 2 winds)."""
    root = tmp_path_factory.mktemp("cards")
    table, truth = sets["train"]
    table, truth = table.iloc[:N_TRAIN_SHAPES], truth.iloc[:N_TRAIN_SHAPES]
    write_reference_cards(table, root / "reference_cards.csv")
    k = 0
    for i in range(len(table)):
        for w in (1.0, 2.0):
            sim = simulate_card_session(table.iloc[i], truth.iloc[i], cfg=cfg, seed=k,
                                        conditions={"wind_ms": w, "duration_min": 6.0},
                                        metadata={"plant_id": f"CARD-{k}"})
            write_session(sim, root)
            k += 1
    raw = root / "data" / "raw"
    sessions = [load_session(s, raw) for s in list_sessions(raw)]
    ts = build_training_set(sessions, load_reference_cards(root / "reference_cards.csv"), cfg,
                            synthetic_calibration(cfg))
    trained = train_correction_model(ts.kept, cfg)
    out = root / "models"
    save_model(trained, out)
    return {"bundle": trained.bundle, "dir": out}


def _leaf_shape(sets) -> tuple[pd.Series, pd.Series]:
    """Held-out shape with the largest hidden correction (flat plate is most wrong there)."""
    table, truth = sets["test"]
    j = int(truth["f_deterministic"].idxmax())
    return table.loc[j], truth.loc[j]


def _set(sim, **cols) -> None:
    for c, v in cols.items():
        sim.sensor_log[c] = v


@pytest.fixture(scope="module")
def campaign(tmp_path_factory, cfg, sets):
    """Seven leaf sessions that exercise every branch; returns root and plant -> session_id."""
    root = tmp_path_factory.mktemp("campaign")
    leaf, leaf_t = _leaf_shape(sets)
    test_table = sets["test"][0]
    write_reference_cards(pd.concat([sets["train"][0], test_table], ignore_index=True),
                          root / "reference_cards.csv")
    cond = {"wind_ms": WIND, "duration_min": DURATION_MIN}
    ids: dict[str, str] = {}

    def make(plant: str, seed: int, card: bool = True, noise: bool = True, g: float = G_TRUE,
             meta: dict[str, Any] | None = None):
        return simulate_leaf_session(g, leaf, leaf_t, card=leaf if card else None,
                                     card_truth=leaf_t if card else None, cfg=cfg, seed=seed,
                                     noise=noise, conditions=cond, plant_id=plant, metadata=meta)

    sims = {}
    sims["A"] = make("A", 1, noise=False)                       # card = same shape as the leaf
    sims["B"] = make("B", 2, card=False)                         # no card in view
    sims["C"] = make("C", 3)                                     # wind sensor missing
    _set(sims["C"], wind_ms=np.nan)
    sims["D"] = make("D", 4, noise=False, g=0.02)                # leaf read 2 C too warm
    sims["D"].sensor_log["leaf_temp_c"] = sims["D"].sensor_log["leaf_temp_c"] + 2.0
    vas = {"dry_reference_type": "vaseline_leaf", "wet_reference_present": "true", "card_id": ""}
    for plant, seed, wet_offset in (("E", 5, -2.0), ("G", 7, -0.1)):
        sims[plant] = make(plant, seed, card=False, noise=False, meta=vas)
        leaf_c = sims[plant].sensor_log["leaf_temp_c"]
        _set(sims[plant], card_temp_c=leaf_c + 1.5, card_n_px=20.0,
             wet_ref_temp_c=leaf_c + wet_offset, wet_ref_n_px=20.0)
    sims["F"] = make("F", 6, noise=False, meta={"wet_reference_present": "true"})
    _set(sims["F"], wet_ref_temp_c=sims["F"].sensor_log["leaf_temp_c"] - 2.0, wet_ref_n_px=20.0)
    for plant, sim in sims.items():
        write_session(sim, root)
        ids[plant] = sim.session_id
    return {"root": root, "ids": ids, "cal": sims["A"].cal}


def _run(campaign, cfg, model_bundle, plants=None) -> pd.DataFrame:
    root = campaign["root"]
    raw = root / "data" / "raw"
    wanted = set(campaign["ids"][p] for p in plants) if plants else None
    sessions = [load_session(s, raw) for s in list_sessions(raw) if wanted is None or s in wanted]
    return aggregate(sessions, root / "data" / "processed", load_reference_cards(root / "reference_cards.csv"),
                     cfg, campaign["cal"], model_bundle)


@pytest.fixture(scope="module")
def results(campaign, cfg, model) -> pd.DataFrame:
    return _run(campaign, cfg, model["bundle"])


def _rows(results: pd.DataFrame, campaign, plant: str, method: str | None = None) -> pd.DataFrame:
    r = results[results["session_id"] == campaign["ids"][plant]]
    return r if method is None else r[r["method"] == method]


def _steady(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["steady_state"].astype(bool)]


# ---------------------------------------------------------------- schema and completeness

def test_columns_are_appendix_b_plus_logged_deviations(results):
    assert list(results.columns) == list(RESULT_COLUMNS)
    assert list(RESULT_COLUMNS[:len(APPENDIX_B_COLUMNS)]) == list(APPENDIX_B_COLUMNS)
    assert set(RESULT_COLUMNS) - set(APPENDIX_B_COLUMNS) == {"light_source", "jones_ig_censor"}


def test_three_rows_per_window_none_missing(results, campaign):
    n_windows = int(round(DURATION_MIN))
    for plant in campaign["ids"]:
        r = _rows(results, campaign, plant)
        assert len(r) == 3 * n_windows, plant
        for _, w in r.groupby("window_start_iso"):
            assert sorted(w["method"]) == ["card_direct", "flat_plate", "ml_corrected"]


def test_method_roles_from_config(results, cfg):
    roles = {cfg["methods"][k]: k for k in ("primary", "secondary", "baseline")}
    for m, role in roles.items():
        assert (results.loc[results["method"] == m, "method_role"] == role).all()
    assert roles["card_direct"] == "primary"


def test_non_steady_windows_written_invalid(results, campaign):
    r = _rows(results, campaign, "A")
    early = r[~r["steady_state"].astype(bool)]
    assert len(early) == 3 * 2
    assert (~early["valid"].astype(bool)).all()
    assert (early["reason"] == "not_steady_state").all()


def test_every_invalid_row_has_reason(results):
    bad = results[~results["valid"].astype(bool)]
    assert len(bad) > 0
    assert (bad["reason"].astype(str).str.len() > 0).all()
    assert (bad["reason"] != "ok").all()


# ---------------------------------------------------------------- methods

def test_card_direct_recovers_true_gs(results, campaign):
    r = _steady(_rows(results, campaign, "A", "card_direct"))
    assert len(r) == 6 and r["valid"].astype(bool).all() and (r["censor"] == "none").all()
    rel = (r["g_s_mol"].astype(float) - G_TRUE).abs() / G_TRUE
    assert rel.max() < 0.10, rel.tolist()
    assert (r["convection_regime"] == "forced").all()


def test_ml_closer_than_flat_plate_on_atypical_shape(results, campaign):
    ml = _steady(_rows(results, campaign, "A", "ml_corrected"))["g_s_mol"].astype(float)
    flat = _steady(_rows(results, campaign, "A", "flat_plate"))["g_s_mol"].astype(float)
    assert (ml - G_TRUE).abs().median() < (flat - G_TRUE).abs().median()


def test_no_card_gives_missing_input_others_run(results, campaign):
    r = _steady(_rows(results, campaign, "B"))
    cd = r[r["method"] == "card_direct"]
    assert (~cd["valid"].astype(bool)).all() and (cd["reason"] == "missing_input").all()
    others = r[r["method"] != "card_direct"]
    assert others["valid"].astype(bool).all()


def test_no_wind_card_direct_still_runs(results, campaign):
    r = _steady(_rows(results, campaign, "C"))
    cd = r[r["method"] == "card_direct"]
    assert cd["valid"].astype(bool).all()
    assert (cd["convection_regime"] == "unknown").all()
    assert cd["flags"].str.contains("regime_unknown").all()
    for m in ("flat_plate", "ml_corrected"):
        x = r[r["method"] == m]
        assert (~x["valid"].astype(bool)).all() and (x["reason"] == "no_wind").all()


def test_missing_model_flagged(campaign, cfg):
    r = _steady(_run(campaign, cfg, None, plants=["B"]))
    ml = r[r["method"] == "ml_corrected"]
    assert (~ml["valid"].astype(bool)).all()
    assert (ml["reason"] == "missing_input").all()
    assert ml["flags"].str.contains("ml_model_missing").all()


# ---------------------------------------------------------------- censoring and uncertainty

def test_censored_rows_kept_with_label_and_limit(results, campaign):
    r = _steady(_rows(results, campaign, "D"))
    assert len(r) == 3 * 6
    cen = r[r["censor"] == "below_detection"]
    assert len(cen) >= 6                      # at least every card_direct window
    assert cen["valid"].astype(bool).all()
    assert cen["g_s_mol"].isna().all()
    assert cen["g_s_mol_dl"].notna().all() and (cen["g_s_mol_dl"].astype(float) > 0).all()
    assert (cen["latent_wm2"].astype(float) <= 0).all()


def test_valid_rows_have_interval_and_resolved(results):
    ok = results[results["valid"].astype(bool)]
    assert ok["g_s_mol_lo"].notna().all() and ok["g_s_mol_hi"].notna().all()
    assert ok["resolved"].isin([True, False]).all()
    assert ok["r_bh"].notna().all() and ok["r_bh_lo"].notna().all() and ok["r_bh_hi"].notna().all()


def test_rb_sigma_from_interval():
    assert rb_sigma_from_interval(50.0, 50.0 * math.exp(2 * 1.959964 * 0.2)) == pytest.approx(0.2, rel=1e-5)
    assert rb_sigma_from_interval(0.0, 60.0) is None
    assert rb_sigma_from_interval(-5.0, 60.0) is None
    assert rb_sigma_from_interval(float("nan"), 60.0) is None


def test_flat_plate_interval_uses_config_sigma(results, cfg):
    flat = results[(results["method"] == "flat_plate") & results["valid"].astype(bool)]
    s = cfg["uncertainty"]["sigma_rb_fraction"]
    assert s == 0.1                           # the test override, not a method interval
    ratio = flat["r_bh_hi"].astype(float) / flat["r_bh_lo"].astype(float)
    assert np.allclose(np.log(ratio) / (2 * 1.959964), s, rtol=1e-4)


def test_card_direct_interval_is_card_interval(results, campaign):
    """card_direct r_bh_lo/hi come from the card Monte Carlo, so they differ from the flat 0.2."""
    cd = _steady(_rows(results, campaign, "A", "card_direct"))
    sig = np.log(cd["r_bh_hi"].astype(float) / cd["r_bh_lo"].astype(float)) / (2 * 1.959964)
    assert not np.allclose(sig, 0.1, rtol=0.02)


def test_dl_flags_rule():
    assert dl_flags(0.09, 0) == [] and dl_flags(0.09, 3) == []
    assert dl_flags(None, 0) == ["dl_not_resolved"]
    assert dl_flags(float("nan"), 0) == ["dl_not_resolved"]
    assert dl_flags(None, 2) == ["dl_search_failed"]


def _check_dl_flags(df: pd.DataFrame) -> None:
    """Valid rows: blank limit <-> exactly one dl flag; invalid rows never carry one."""
    flags = df["flags"].fillna("").str.split(";")
    n_dl = flags.apply(lambda f: sum(x in ("dl_not_resolved", "dl_search_failed") for x in f))
    valid = df["valid"].astype(bool)
    blank = df["g_s_mol_dl"].isna()
    assert (n_dl[valid & blank] == 1).all()
    assert (n_dl[valid & ~blank] == 0).all()
    assert (n_dl[~valid] == 0).all()


def test_dl_flags_on_results(results):
    _check_dl_flags(results)


def test_placeholder_sigmas_give_dl_not_resolved(campaign, model):
    """Config placeholder sigmas resolve nothing (A44): every valid row says so."""
    cfg = load_config(REPO / "config.yaml")
    cfg["uncertainty"]["n_draws"] = 300
    cfg["uncertainty"]["dl_grid_points"] = 12
    r = _run(campaign, cfg, model["bundle"], plants=["A"])
    ok = r[r["valid"].astype(bool)]
    assert len(ok) > 0 and ok["g_s_mol_dl"].isna().all()
    assert ok["flags"].str.contains("dl_not_resolved").all()
    assert not ok["flags"].str.contains("dl_search_failed").any()
    _check_dl_flags(r)


def test_failed_root_search_gives_dl_search_failed(campaign, cfg, model, monkeypatch):
    """If the forward root search errors, the run continues and the row says the search failed."""
    import processing.uncertainty as unc

    def broken(*args, **kwargs):
        raise RuntimeError("simulated non-convergence")

    monkeypatch.setattr(unc, "brentq", broken)
    r = _run(campaign, cfg, model["bundle"], plants=["A"])
    ok = r[r["valid"].astype(bool)]
    assert len(ok) > 0 and ok["g_s_mol_dl"].isna().all()
    assert ok["flags"].str.contains("dl_search_failed").all()
    assert not ok["flags"].str.contains("dl_not_resolved").any()
    assert ok["g_s_mol_lo"].notna().all()          # the interval itself is unaffected
    _check_dl_flags(r)


# ---------------------------------------------------------------- Jones index

def test_jones_with_vaseline_leaf(results, campaign):
    r = _steady(_rows(results, campaign, "E"))
    assert np.allclose(r["jones_ig"].astype(float), 1.5 / 2.0, rtol=1e-6)
    assert np.allclose(r["cwsi"].astype(float), 2.0 / 3.5, rtol=1e-6)
    assert (r["jones_ig_censor"] == "none").all()
    cd = r[r["method"] == "card_direct"]     # a vaseline leaf is not a card
    assert (cd["reason"] == "missing_input").all()


def test_jones_above_range_kept_with_label(results, campaign):
    r = _steady(_rows(results, campaign, "G"))
    assert r["jones_ig"].isna().all()
    assert (r["jones_ig_censor"] == "above_range").all()
    assert r["cwsi"].notna().all()


def test_card_is_not_a_jones_dry_reference(results, campaign):
    r = _steady(_rows(results, campaign, "F"))
    assert r["jones_ig"].isna().all() and r["cwsi"].isna().all()
    assert r["flags"].str.contains("jones_no_leaf_dry_reference").all()


def test_no_references_no_jones_no_flag(results, campaign):
    r = _rows(results, campaign, "A")
    assert r["jones_ig"].isna().all() and (r["jones_ig_censor"].fillna("") == "").all()
    assert not r["flags"].str.contains("jones").any()


# ---------------------------------------------------------------- command line

def test_cli_synthetic(campaign, cfg, model, tmp_path, capsys):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(copy.deepcopy(cfg)), encoding="utf-8")
    root = campaign["root"]
    out = tmp_path / "leaf_gs_results_synthetic.csv"
    args = ["--raw-dir", str(root / "data" / "raw"), "--processed-dir", str(root / "data" / "processed"),
            "--cards", str(root / "reference_cards.csv"), "--model", str(model["dir"] / "correction_model.joblib"),
            "--config", str(cfg_path), "--out", str(out)]
    assert agg_main(args + ["--synthetic"]) == 0
    df = pd.read_csv(out)
    assert list(df.columns) == list(RESULT_COLUMNS)
    assert len(df) == 3 * int(round(DURATION_MIN)) * len(campaign["ids"])
    assert "below_detection" in set(df["censor"])
    text = capsys.readouterr().out
    assert "SYNTHETIC" in text and "below_detection" in text
    with pytest.raises(SystemExit):           # real-data mode never picks up synthetic sessions
        agg_main(args)
