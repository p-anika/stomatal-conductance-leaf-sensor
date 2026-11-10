"""ML pipeline tests (T4.6): build_training_set (T4.3), train_correction_model (T4.4),
predict_correction (T4.5), on synthetic data from ml/synthetic_data.py (T4.2).

Owner decisions 2026-11-09: card windows gated only on exogenous conditions; r_bH <= 0 rejected
and counted, with the truncation bias of that rejection measured and reported; target f on a
linear scale; unweighted training; nested leave-one-shape-out CV against the null model f = 1;
interval = bootstrap spread and CV residual SD in quadrature; Arducam-only card geometry;
card_angle_default flag. Synthetic results prove the code works, nothing about plants.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ml.build_training_set import (
    TRAINING_COLUMNS,
    InsufficientShapes,
    build_training_set,
    format_report,
    main as build_main,
)
from ml.predict_correction import load_model, predict_f, predict_rbh
from ml.synthetic_data import make_card_sets, simulate_card_session, write_reference_cards, write_session
from ml.train_correction_model import logo_splits, main as train_main, save_model, train_correction_model
from processing import energy_balance_utils as ebu
from processing.config import load_config
from processing.data_io import list_sessions, load_reference_cards, load_session

REPO = Path(__file__).resolve().parent.parent
WINDS = (0.3, 0.8, 1.5)
DURATION_MIN = 6.0           # windows 0-5; windows 0 and 1 lack 3 min of history
STEADY_PER_SESSION = 4


def fast_cfg() -> dict[str, Any]:
    cfg = load_config(REPO / "config.yaml")
    cfg["ml"]["forest_n_estimators"] = 20        # speed only; production value stays in config
    cfg["ml"]["n_bootstrap"] = 30
    return cfg


@pytest.fixture(scope="module")
def cfg() -> dict[str, Any]:
    return fast_cfg()


@pytest.fixture(scope="module")
def sets(cfg):
    return make_card_sets(cfg)


def make_root(root: Path, table: pd.DataFrame, truth: pd.DataFrame, cfg: dict[str, Any],
              winds=WINDS, conditions: dict[str, Any] | None = None, seed0: int = 0,
              geometry_source: str = "camera") -> Path:
    t = table.copy()
    t["geometry_source"] = geometry_source
    write_reference_cards(t, root / "reference_cards.csv")
    k = 0
    for i in range(len(t)):
        for w in winds:
            cond = {"wind_ms": w, "duration_min": DURATION_MIN, **(conditions or {})}
            sim = simulate_card_session(t.iloc[i], truth.iloc[i], cfg=cfg, seed=seed0 + k, conditions=cond,
                                        metadata={"plant_id": f"CARD-{t.iloc[i]['card_id']}-{k}"})
            write_session(sim, root)
            k += 1
    (root / "calibration.json").write_text(json.dumps(sim.cal))
    return root


def load_all(root: Path) -> list:
    raw = root / "data" / "raw"
    return [load_session(s, raw) for s in list_sessions(raw)]


def run_build(root: Path, cfg, force: bool = False):
    cal = json.loads((root / "calibration.json").read_text())
    return build_training_set(load_all(root), load_reference_cards(root / "reference_cards.csv"),
                              cfg, cal, force=force)


@pytest.fixture(scope="module")
def train_root(tmp_path_factory, cfg, sets):
    return make_root(tmp_path_factory.mktemp("train"), *sets["train"], cfg)


@pytest.fixture(scope="module")
def train_ts(train_root, cfg):
    return run_build(train_root, cfg)


# ---------------------------------------------------------------- T4.3 build_training_set

def test_report_matches_known_counts(train_ts, sets):
    n_sessions = len(sets["train"][0]) * len(WINDS)
    r = train_ts.report
    # windows 0-1 of every session lack history; a flat card fails the 0.05 C/min slope test
    # about 1 % of the time by chance (T3.4 explainer), so a few steady windows are lost too
    assert set(r["rejects_by_reason"]) == {"not_steady_state"}
    assert r["n_kept"] + r["rejects_by_reason"]["not_steady_state"] == n_sessions * 6
    assert r["n_kept"] >= 0.95 * n_sessions * STEADY_PER_SESSION
    assert r["n_shapes"] == len(sets["train"][0])
    assert r["wind_ms_range"][0] == pytest.approx(min(WINDS), abs=0.05)
    assert r["suspect_count"] == 0 and r["truncation_bias_f"] == pytest.approx(0.0)
    text = format_report(r)
    for word in ("rows kept", "rejects", "shapes", "wind", "suspect", "truncation", "r_bh_card_sigma"):
        assert word in text


def test_training_rows_schema_and_values(train_ts, sets, cfg):
    df = train_ts.kept
    assert list(df.columns) == list(TRAINING_COLUMNS)
    assert df["valid"].all() and (df["reason"] == "ok").all()
    truth = sets["train"][1].set_index("card_id")["f_true"]
    err = (df["f_ratio"] / df["card_id"].map(truth) - 1).abs()
    assert err.mean() < 0.02 and err.max() < 0.08  # window noise only (small delta T at high wind)
    assert np.allclose(df["f_ratio"], df["r_bh_card"] / df["r_bh_flat"])
    assert np.allclose(df["r_bh_card_sigma"], cfg["uncertainty"]["sigma_rn_card_fraction"] * df["r_bh_card"])
    assert (df["incidence_angle_deg"] == 0.0).all()
    assert df["flags"].str.contains("card_angle_default").all()
    assert df["row_id"].is_unique
    rej = train_ts.rejects
    assert set(rej["reason"]) == {"not_steady_state"} and "note" in rej.columns


def test_exogenous_gates(tmp_path, cfg, sets):
    table, truth = sets["train"]
    t3, tr3 = table.head(3), truth.head(3)
    root = make_root(tmp_path, t3, tr3, cfg, winds=(0.8,), conditions={"irradiance_wm2": 30.0})
    ts = run_build(root, cfg, force=True)
    # absorbed shortwave 0.95 x 30 = 28.5 W m-2 < 40: rejected however warm the card is
    r = ts.report
    assert r["n_kept"] == 0
    assert r["rejects_by_reason"]["low_card_signal"] + r["rejects_by_reason"]["not_steady_state"] == 3 * 6
    assert r["rejects_by_reason"]["low_card_signal"] >= 3 * STEADY_PER_SESSION - 1


def test_no_wind_rejected(train_root, cfg):
    cal = json.loads((train_root / "calibration.json").read_text())
    sessions = load_all(train_root)[:2]
    s0 = sessions[0]
    sensor = s0.sensor.copy()
    sensor["wind_ms"] = np.nan
    sessions[0] = dataclasses.replace(s0, sensor=sensor)
    ts = build_training_set(sessions, load_reference_cards(train_root / "reference_cards.csv"), cfg, cal,
                            force=True)
    assert 1 <= ts.report["rejects_by_reason"]["no_wind"] <= STEADY_PER_SESSION
    assert s0.session_id not in set(ts.kept["session_id"])


def test_phone_geometry_rejected(tmp_path, cfg, sets):
    table, truth = sets["train"]
    root = make_root(tmp_path, table.head(2), truth.head(2), cfg, winds=(0.8,), geometry_source="phone")
    ts = run_build(root, cfg, force=True)
    assert ts.report["n_kept"] == 0
    rej = ts.rejects[ts.rejects["reason"] == "missing_input"]
    assert len(rej) >= 2 * STEADY_PER_SESSION - 1
    assert rej["note"].str.contains("camera").all()


def test_min_shapes_and_force(tmp_path, cfg, sets, capsys):
    table, truth = sets["train"]
    root = make_root(tmp_path, table.head(3), truth.head(3), cfg, winds=(0.8,))
    with pytest.raises(InsufficientShapes, match="3"):
        run_build(root, cfg)
    ts = run_build(root, cfg, force=True)
    assert ts.report["n_shapes"] == 3
    assert "overfit" in " ".join(ts.report["warnings"]).lower()


def test_truncation_bias_measured(tmp_path, sets):
    """Weak signal + per-session calibration offsets: some card windows get r_bH <= 0. Rejecting them
    (spec) pushes the mean f up; the report measures that from the rows themselves."""
    cfg = fast_cfg()
    cfg["synthetic"]["session_bias"] = True
    table, truth = sets["train"]
    # delta T only about 1 C; thermal offset sigma 1.0 C, air 0.5 C; three sessions per card
    root = make_root(tmp_path, table, truth, cfg, winds=(2.0, 2.0, 2.0),
                     conditions={"irradiance_wm2": 60.0})
    ts = run_build(root, cfg)
    r = ts.report
    assert r["suspect_count"] > 0
    assert r["suspect_fraction"] > cfg["ml"]["max_suspect_fraction"]
    assert any("suspect" in w for w in r["warnings"])
    assert r["truncation_bias_f"] > 0                                   # kept mean > all-gated mean
    # against the known truth: keeping the suspect rows is closer to the true mean f
    sus = ts.rejects[ts.rejects["reason"] == "suspect_card_rbh"]
    true_f = truth.set_index("card_id")["f_true"]
    all_rows = pd.concat([ts.kept, sus])
    bias_kept = (ts.kept["f_ratio"] - ts.kept["card_id"].map(true_f)).mean()
    bias_all = (all_rows["f_ratio"] - all_rows["card_id"].map(true_f)).mean()
    assert bias_kept > abs(bias_all)
    print(f"\ntruncation bias of mean f (kept - all gated): {r['truncation_bias_f']:.3f}; "
          f"vs truth: kept {bias_kept:+.3f}, all gated {bias_all:+.3f}")


def test_build_cli(train_root, tmp_path):
    out = tmp_path / "rb.csv"
    code = build_main(["--out", str(out), "--raw-dir", str(train_root / "data" / "raw"),
                       "--cards", str(train_root / "reference_cards.csv"),
                       "--calibration", str(train_root / "calibration.json"), "--include-synthetic"])
    assert code == 0
    df = pd.read_csv(out)
    assert list(df.columns) == list(TRAINING_COLUMNS)
    assert (tmp_path / "rb_rejects.csv").is_file()
    # synthetic sessions are skipped unless asked for
    with pytest.raises(SystemExit, match="no card windows"):
        build_main(["--out", str(out), "--raw-dir", str(train_root / "data" / "raw"),
                    "--cards", str(train_root / "reference_cards.csv"),
                    "--calibration", str(train_root / "calibration.json")])


# ---------------------------------------------------------------- T4.4 train_correction_model

def test_logo_never_mixes_shape_groups(train_ts):
    df = train_ts.kept
    splits = list(logo_splits(df))
    assert len(splits) == df["shape_group"].nunique()
    for tr, te in splits:
        assert set(df["shape_group"].iloc[tr]).isdisjoint(df["shape_group"].iloc[te])
        assert df["shape_group"].iloc[te].nunique() == 1


@pytest.fixture(scope="module")
def trained(train_ts, cfg):
    return train_correction_model(train_ts.kept, cfg)


def test_recovers_hidden_correction(trained, sets, cfg, tmp_path):
    card = trained.card
    assert card["status"] == "ok"
    assert card["metrics"]["nested"]["improvement_vs_null"] >= cfg["ml"]["min_improvement"]
    assert card["chosen_model"] in ("ridge", "forest")
    assert set(card["metrics"]["logo"]) == {"null", "ridge", "forest"}
    # held-out shapes (seed 1) the model never saw
    save_model(trained, tmp_path)
    model = load_model(tmp_path / "correction_model.joblib")
    table, truth = sets["test"]
    preds = []
    for _, row in table.iterrows():
        feats = {c: row[c] for c in ("d_min_mm", "d_eq_mm", "aspect_ratio", "eccentricity", "solidity",
                                     "circularity", "edge_density")}
        feats.update(incidence_angle_deg=0.0, wind_ms=0.8)
        preds.append(predict_f(feats, model).f_hat)
    rmse = math.sqrt(np.mean((np.array(preds) - truth["f_deterministic"].to_numpy()) ** 2))
    null_rmse = math.sqrt(np.mean((1.0 - truth["f_deterministic"].to_numpy()) ** 2))
    assert rmse < 0.06 and rmse < 0.25 * null_rmse
    print(f"\nheld-out f RMSE {rmse:.3f} (null {null_rmse:.3f})")


def test_model_card_contents(trained, tmp_path):
    save_model(trained, tmp_path)
    card = json.loads((tmp_path / "correction_model_card.json").read_text())
    for key in ("status", "chosen_model", "target", "features", "metrics", "training_data_sha256",
                "n_rows", "n_shapes", "library_versions", "date_utc", "n_bootstrap", "residual_sd_f",
                "interval_note"):
        assert key in card
    m = card["metrics"]
    assert {"rmse_s_m", "mae_s_m"} <= set(m["logo"]["null"])
    assert "inner_selection" in m and "nested" in m
    assert m["nested"]["outer_rmse_s_m"] > 0 and m["nested"]["null_rmse_s_m"] > 0
    assert card["target"] == "f (linear scale)"


def test_null_set_reports_no_improvement(tmp_path_factory, cfg, sets):
    root = make_root(tmp_path_factory.mktemp("null"), *sets["null"], cfg)
    ts = run_build(root, cfg)
    t = train_correction_model(ts.kept, cfg)
    assert t.card["status"] == "no_improvement"
    out = tmp_path_factory.mktemp("nullmodel")
    save_model(t, out)
    p = predict_f({c: 1.0 for c in cfg["ml"]["features"]} | {"wind_ms": 0.8, "aspect_ratio": 1.5},
                  load_model(out / "correction_model.joblib"))
    assert p.f_hat == 1.0 and "ml_no_improvement" in p.flags and p.valid
    assert p.f_lo < 1.0 < p.f_hi


def test_train_cli(train_root, tmp_path):
    out = tmp_path / "rb.csv"
    build_main(["--out", str(out), "--raw-dir", str(train_root / "data" / "raw"),
                "--cards", str(train_root / "reference_cards.csv"),
                "--calibration", str(train_root / "calibration.json"), "--include-synthetic"])
    cfg_path = tmp_path / "config.yaml"
    import yaml
    cfg_path.write_text(yaml.safe_dump(fast_cfg()))
    assert train_main(["--data", str(out), "--out-dir", str(tmp_path / "models"),
                       "--config", str(cfg_path)]) == 0
    assert (tmp_path / "models" / "correction_model.joblib").is_file()


# ---------------------------------------------------------------- T4.5 predict_correction

@pytest.fixture(scope="module")
def saved_model(trained, tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    save_model(trained, d)
    return load_model(d / "correction_model.joblib")


def feats_of(sets, i: int = 0, wind: float = 0.8) -> dict[str, float]:
    row = sets["train"][0].iloc[i]
    f = {c: float(row[c]) for c in ("d_min_mm", "d_eq_mm", "aspect_ratio", "eccentricity", "solidity",
                                    "circularity", "edge_density")}
    return f | {"incidence_angle_deg": 0.0, "wind_ms": wind}


def test_predict_f_interval_and_flags(saved_model, sets):
    p = predict_f(feats_of(sets), saved_model)
    assert p.valid and p.reason == "ok" and p.flags == ()
    assert p.f_lo < p.f_hat < p.f_hi
    far = feats_of(sets) | {"solidity": 0.3}
    assert "outside_training_range" in predict_f(far, saved_model).flags


def test_predict_f_missing_inputs(saved_model, sets):
    f = feats_of(sets)
    f["wind_ms"] = float("nan")
    assert predict_f(f, saved_model).reason == "no_wind"
    g = feats_of(sets)
    del g["solidity"]
    p = predict_f(g, saved_model)
    assert not p.valid and p.reason == "missing_input" and p.f_hat is None


def test_predict_rbh(saved_model, sets, cfg):
    f = feats_of(sets)
    p = predict_rbh(f, wind_ms=0.8, delta_t_c=-1.0, air_temp_c=24.0, cfg=cfg, model=saved_model)
    flat, regime = ebu.flat_plate_rbh(f["d_min_mm"] / 1000.0, 0.8, -1.0, 24.0, cfg)
    pf = predict_f(f, saved_model)
    assert p.valid and p.r_bh == pytest.approx(pf.f_hat * flat)
    assert p.r_bh_lo == pytest.approx(pf.f_lo * flat) and p.r_bh_hi == pytest.approx(pf.f_hi * flat)
    assert p.r_bh_flat == pytest.approx(flat) and p.convection_regime == regime
    nw = predict_rbh(f, wind_ms=float("nan"), delta_t_c=-1.0, air_temp_c=24.0, cfg=cfg, model=saved_model)
    assert not nw.valid and nw.reason == "no_wind"


@pytest.mark.parametrize("bad_wind", [float("nan"), None, float("inf"), "", "calm"])
def test_predict_rbh_refuses_without_wind(saved_model, sets, cfg, bad_wind):
    """T4.5: no wind -> reason no_wind and no r_bH, even if the features dict holds a wind value."""
    f = feats_of(sets, wind=0.8)                     # features carry a wind; the argument is missing
    p = predict_rbh(f, wind_ms=bad_wind, delta_t_c=-1.0, air_temp_c=24.0, cfg=cfg, model=saved_model)
    assert not p.valid and p.reason == "no_wind"
    assert p.r_bh is None and p.r_bh_lo is None and p.r_bh_hi is None and p.r_bh_flat is None
    assert p.f_hat is None


# ---------------------------------------------------------------- T4.2 outputs in this pipeline file

def test_synthetic_card_sets_pass_real_checks(sets, tmp_path, cfg):
    """T4.2: train, held-out and null card sets load through the real manifest loader, and a
    synthetic card session loads through the real session loader."""
    for name, (table, truth) in sets.items():
        path = write_reference_cards(table, tmp_path / f"{name}.csv")
        assert set(load_reference_cards(path)) == set(table["card_id"])
        assert "f_true" not in table.columns and len(truth) == len(table)
    table, truth = sets["train"]
    sim = simulate_card_session(table.iloc[0], truth.iloc[0], cfg=cfg, seed=0,
                                conditions={"duration_min": DURATION_MIN})
    write_session(sim, tmp_path)
    s = load_session(sim.session_id, tmp_path / "data" / "raw")
    assert s.synthetic and s.metadata["card_id"] == table.iloc[0]["card_id"]


# ---------------------------------------------------------------- every CV split, outer and inner

def test_no_cv_split_mixes_shape_groups(train_ts, cfg, monkeypatch):
    """T4.6: record every split train_correction_model makes (plain LOGO, nested outer and nested
    inner) and check that no shape_group is ever in both the training and the test part."""
    import ml.train_correction_model as tcm
    real = tcm.logo_splits
    seen: list[tuple[set, set]] = []

    def spy(df):
        groups = df["shape_group"].to_numpy()
        for tr, te in real(df):
            seen.append((set(groups[tr]), set(groups[te])))
            yield tr, te

    monkeypatch.setattr(tcm, "logo_splits", spy)
    shapes = sorted(train_ts.kept["shape_group"].unique())[:5]          # small: speed only
    small = train_ts.kept[train_ts.kept["shape_group"].isin(shapes)]
    tcm.train_correction_model(small, cfg)
    n = len(shapes)
    # plain LOGO for 3 candidates (n each) + nested: n outer splits, each with 2 inner models x (n - 1)
    assert len(seen) == 3 * n + n + n * 2 * (n - 1)
    for train_groups, test_groups in seen:
        assert len(test_groups) == 1
        assert train_groups.isdisjoint(test_groups)


def test_model_reloads_and_predicts_same(trained, tmp_path, sets):
    save_model(trained, tmp_path)
    a = predict_f(feats_of(sets, 2), load_model(tmp_path / "correction_model.joblib"))
    b = predict_f(feats_of(sets, 2), load_model(tmp_path / "correction_model.joblib"))
    assert a == b
