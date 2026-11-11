"""Tests for analysis/validate_gs_ground_truth.py (T5.6, E4): whole-leaf rule, pairing, statistics,
detection table, gravimetric trend table, CLI. Synthetic only."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import validate_gs_ground_truth as v

T0 = pd.Timestamp("2026-11-09T10:00:00+00:00")
METHODS = (("card_direct", "primary"), ("ml_corrected", "secondary"), ("flat_plate", "baseline"))


def gt_row(plant: str, minutes: float, surface: str, g: float, instrument: str = "porometer",
           leaf: str = "top") -> dict:
    return {"plant_id": plant, "timestamp_iso": T0 + pd.Timedelta(minutes=minutes), "instrument": instrument,
            "surface": surface, "g_s_mol": g, "leaf_position": leaf, "notes": ""}


def gt_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(v.GROUND_TRUTH_COLUMNS))


def res_row(plant: str, minutes: float, method: str, g: float, *, valid: bool = True, censor: str = "none",
            dl: float = 0.05, regime: str = "forced", leaf: str = "top", session: str = "S1") -> dict:
    start = T0 + pd.Timedelta(minutes=minutes) - pd.Timedelta(seconds=30)
    role = dict(METHODS)[method]
    return {"session_id": session, "plant_id": plant, "leaf_position": leaf,
            "window_start_iso": start.isoformat(), "window_end_iso": (start + pd.Timedelta(seconds=60)).isoformat(),
            "method": method, "method_role": role, "convection_regime": regime, "g_s_mol": g,
            "censor": censor, "g_s_mol_dl": dl, "valid": valid, "reason": "ok" if valid else "missing_input"}


def res_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- whole-leaf rule (E4)

def test_two_faces_within_limit_are_summed():
    gt = gt_frame([gt_row("P1", 0, "adaxial", 0.10), gt_row("P1", 3, "abaxial", 0.20)])
    whole, one = v.whole_leaf_ground_truth(gt, surface_pair_minutes=5)
    assert len(whole) == 1 and one.empty
    assert whole["g_true_mol"].iloc[0] == pytest.approx(0.30)
    assert whole["truth_source"].iloc[0] == "summed_faces"
    assert whole["truth_time"].iloc[0] == T0 + pd.Timedelta(minutes=1.5)      # midpoint


def test_faces_too_far_apart_are_listed_never_compared():
    gt = gt_frame([gt_row("P1", 0, "adaxial", 0.10), gt_row("P1", 9, "abaxial", 0.20)])
    whole, one = v.whole_leaf_ground_truth(gt, surface_pair_minutes=5)
    assert whole.empty
    assert len(one) == 2 and set(one["reason"]) == {"one_surface_only"}


def test_faces_of_different_plants_or_leaves_are_not_summed():
    gt = gt_frame([gt_row("P1", 0, "adaxial", 0.1), gt_row("P2", 1, "abaxial", 0.2),
                   gt_row("P3", 0, "adaxial", 0.1, leaf="top"), gt_row("P3", 1, "abaxial", 0.2, leaf="mid")])
    whole, one = v.whole_leaf_ground_truth(gt, surface_pair_minutes=5)
    assert whole.empty and len(one) == 4


def test_each_face_reading_used_once_closest_first():
    # two adaxial readings, one abaxial: the closer adaxial is used, the other is left over
    gt = gt_frame([gt_row("P1", 0, "adaxial", 0.10), gt_row("P1", 4, "adaxial", 0.11),
                   gt_row("P1", 5, "abaxial", 0.20)])
    whole, one = v.whole_leaf_ground_truth(gt, surface_pair_minutes=5)
    assert len(whole) == 1 and whole["g_true_mol"].iloc[0] == pytest.approx(0.31)
    assert len(one) == 1 and one["g_s_mol"].iloc[0] == pytest.approx(0.10)


def test_whole_leaf_and_both_summed_used_directly():
    gt = gt_frame([gt_row("P1", 0, "whole_leaf", 0.4), gt_row("P2", 0, "both_summed", 0.5)])
    whole, one = v.whole_leaf_ground_truth(gt, surface_pair_minutes=5)
    assert sorted(whole["g_true_mol"]) == [0.4, 0.5] and one.empty


def test_gravimetric_rows_are_split_off():
    gt = gt_frame([gt_row("P1", 0, "whole_leaf", 0.4), gt_row("P1", 0, "whole_leaf", 0.3, "gravimetric")])
    poro, grav = v.split_ground_truth(gt)
    assert list(poro["instrument"]) == ["porometer"] and list(grav["instrument"]) == ["gravimetric"]


def test_refuses_one_sided_device_convention(default_config):
    cfg = copy.deepcopy(default_config)
    cfg["physics"]["faces"] = 1
    with pytest.raises(ValueError, match="faces"):
        v.check_two_sided(cfg)
    cfg["physics"]["faces"] = 2
    v.check_two_sided(cfg)


# ---------------------------------------------------------------- pairing

def _whole(plant: str, minutes: float, g: float) -> pd.DataFrame:
    gt = gt_frame([gt_row(plant, minutes, "whole_leaf", g)])
    return v.whole_leaf_ground_truth(gt, 5)[0]


def test_pairs_nearest_valid_window_same_plant_within_limit():
    res = res_frame([res_row("P1", 2, "card_direct", 0.31), res_row("P1", 8, "card_direct", 0.99),
                     res_row("P2", 0, "card_direct", 0.77)])
    pairs = v.pair_ground_truth(_whole("P1", 0, 0.30), res, pair_max_minutes=10)
    p = pairs[pairs["method"] == "card_direct"].iloc[0]
    assert p["status"] == "paired" and p["g_device_mol"] == pytest.approx(0.31)
    assert p["gap_min"] == pytest.approx(2.0)


def test_no_window_beyond_limit():
    res = res_frame([res_row("P1", 11, "card_direct", 0.31)])
    pairs = v.pair_ground_truth(_whole("P1", 0, 0.30), res, pair_max_minutes=10)
    assert pairs.loc[pairs["method"] == "card_direct", "status"].iloc[0] == "no_window"


def test_censored_and_invalid_status():
    res = res_frame([res_row("P1", 1, "card_direct", 0.05, censor="below_detection"),
                     res_row("P1", 1, "flat_plate", math.nan, valid=False)])
    pairs = v.pair_ground_truth(_whole("P1", 0, 0.02), res, pair_max_minutes=10)
    st = dict(zip(pairs["method"], pairs["status"]))
    assert st["card_direct"] == "device_censored" and st["flat_plate"] == "device_invalid"


def test_valid_window_preferred_over_nearer_invalid():
    res = res_frame([res_row("P1", 0.5, "card_direct", math.nan, valid=False),
                     res_row("P1", 4, "card_direct", 0.33)])
    pairs = v.pair_ground_truth(_whole("P1", 0, 0.30), res, pair_max_minutes=10)
    assert pairs["status"].iloc[0] == "paired" and pairs["g_device_mol"].iloc[0] == pytest.approx(0.33)


# ---------------------------------------------------------------- statistics

def test_agreement_stats_exact():
    truth = np.array([0.1, 0.2, 0.3, 0.4])
    device = 2 * truth + 0.05                      # slope 2, intercept 0.05
    s = v.agreement_stats(device, truth, loa_z=1.96)
    d = device - truth
    assert s["n"] == 4
    assert s["bias"] == pytest.approx(d.mean())
    assert s["mae"] == pytest.approx(np.abs(d).mean())
    assert s["rmse"] == pytest.approx(math.sqrt((d ** 2).mean()))
    assert s["slope"] == pytest.approx(2.0) and s["intercept"] == pytest.approx(0.05)
    assert s["r2"] == pytest.approx(1.0)
    assert s["loa_hi"] == pytest.approx(d.mean() + 1.96 * d.std(ddof=1))


def test_agreement_stats_too_few():
    s = v.agreement_stats(np.array([0.2]), np.array([0.1]), 1.96)
    assert s["n"] == 1 and s["bias"] == pytest.approx(0.1) and math.isnan(s["slope"])


def _synthetic_campaign(bias: float, sd: float, n_plants: int = 40, per_plant: int = 3, seed: int = 1):
    rng = np.random.default_rng(seed)
    gt_rows, res_rows = [], []
    for i in range(n_plants):
        plant = f"P{i:02d}"
        for k in range(per_plant):
            minutes = 60 * k
            g_true = rng.uniform(0.1, 0.6)
            # truth measured as two faces (1 : 2 split), summed by the module
            gt_rows += [gt_row(plant, minutes, "adaxial", g_true / 3),
                        gt_row(plant, minutes + 2, "abaxial", 2 * g_true / 3)]
            for m, _ in METHODS:
                res_rows.append(res_row(plant, minutes + 3, m, g_true + bias + rng.normal(0, sd)))
    return gt_frame(gt_rows), res_frame(res_rows)


def test_synthetic_known_error_is_recovered(default_config):
    gt, res = _synthetic_campaign(bias=0.03, sd=0.02)
    out = v.run(res, gt, default_config, label="SYNTHETIC")
    s = out["summary"]
    row = s[(s["method"] == "card_direct") & (s["level"] == v.LEVEL_WINDOW)].iloc[0]
    assert row["n"] == 120 and row["n_plants"] == 40
    assert row["bias"] == pytest.approx(0.03, abs=0.006)
    assert row["sd_diff"] == pytest.approx(0.02, abs=0.005)
    assert row["slope"] == pytest.approx(1.0, abs=0.05)
    plant = s[(s["method"] == "card_direct") & (s["level"] == v.LEVEL_PLANT)].iloc[0]
    assert plant["n"] == 40 and plant["bias"] == pytest.approx(0.03, abs=0.006)
    assert set(s["data_label"]) == {"SYNTHETIC"}
    assert out["one_surface"].empty


def test_plant_level_is_primary_and_window_level_labelled():
    assert "primary" in v.LEVEL_PLANT and "non-independent" in v.LEVEL_WINDOW


# ---------------------------------------------------------------- detection table

def test_detection_table_counts():
    whole = pd.concat([_whole("P1", 0, 0.01), _whole("P2", 0, 0.02), _whole("P3", 0, 0.30)], ignore_index=True)
    res = res_frame([res_row("P1", 1, "card_direct", 0.05, censor="below_detection", dl=0.05),
                     res_row("P2", 1, "card_direct", 0.06, dl=0.05),          # missed: reported a value
                     res_row("P3", 1, "card_direct", 0.05, censor="below_detection", dl=0.05)])  # false censor
    det = v.detection_table(v.pair_ground_truth(whole, res, 10))
    r = det[det["method"] == "card_direct"].iloc[0]
    assert r["n_truth_below_dl"] == 2 and r["n_device_below_detection"] == 1
    assert r["fraction_correct"] == pytest.approx(0.5)
    assert r["n_truth_above_dl"] == 1 and r["n_false_below_detection"] == 1


# ---------------------------------------------------------------- gravimetric trend table

def test_gravimetric_trend_table_spearman():
    grav = gt_frame([gt_row(f"P{i}", 0, "whole_leaf", 0.1 * (i + 1), "gravimetric") for i in range(4)])
    res = res_frame([res_row(f"P{i}", 20, "card_direct", 0.2 * (i + 1)) for i in range(4)]
                    + [res_row("P0", 300, "card_direct", 9.9)])                 # outside the hour: ignored
    per_plant, summary = v.gravimetric_trend(grav, res, match_hours=1.0)
    p0 = per_plant[(per_plant["method"] == "card_direct") & (per_plant["plant_id"] == "P0")].iloc[0]
    assert p0["g_device_mol"] == pytest.approx(0.2)
    s = summary[summary["method"] == "card_direct"].iloc[0]
    assert s["n_plants"] == 4 and s["spearman_rho"] == pytest.approx(1.0)


def test_gravimetric_never_in_bias_tables(default_config):
    gt = gt_frame([gt_row("P1", 0, "whole_leaf", 0.3, "gravimetric")])
    res = res_frame([res_row("P1", 1, "card_direct", 0.3)])
    out = v.run(res, gt, default_config, label="SYNTHETIC")
    assert out["pairs"].empty
    assert (out["summary"]["n"] == 0).all()


# ---------------------------------------------------------------- CLI

def test_cli_writes_outputs(tmp_path: Path, default_config):
    gt, res = _synthetic_campaign(bias=0.0, sd=0.02, n_plants=6, per_plant=2)
    gt_path, res_path = tmp_path / "gt.csv", tmp_path / "res.csv"
    gt_out = gt.copy()
    gt_out["timestamp_iso"] = gt_out["timestamp_iso"].map(lambda t: t.isoformat())
    gt_out.to_csv(gt_path, index=False)
    res.to_csv(res_path, index=False)
    out_dir, fig_dir = tmp_path / "out", tmp_path / "fig"
    rc = v.main(["--synthetic", "--results", str(res_path), "--ground-truth", str(gt_path),
                 "--out-dir", str(out_dir), "--figure-dir", str(fig_dir)])
    assert rc == 0
    summary = pd.read_csv(out_dir / "validation_gs_summary_synthetic.csv")
    assert set(summary["data_label"]) == {"SYNTHETIC"}
    for name in ("pairs", "one_surface", "detection", "gravimetric_plants", "gravimetric_summary"):
        assert (out_dir / f"validation_gs_{name}_synthetic.csv").is_file()
    assert (fig_dir / "bland_altman_synthetic.png").stat().st_size > 1000
