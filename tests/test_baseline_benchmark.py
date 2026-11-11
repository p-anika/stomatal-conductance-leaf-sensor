"""Tests for analysis/baseline_benchmark.py (T5.2): shape join, splits, forced-only flat plate,
pre-registered plant-level primary analysis, Wilcoxon. Synthetic only."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import baseline_benchmark as b
from analysis import validate_gs_ground_truth as v
from processing.data_io import PHOTO_REQUIRED_COLUMNS
from test_validate_gs_ground_truth import T0, gt_frame, gt_row, res_frame, res_row


# ---------------------------------------------------------------- shape join

def _photo_csv(path: Path, minutes: float, solidity: float, aspect: float) -> None:
    row = {c: "1.0" for c in PHOTO_REQUIRED_COLUMNS}
    row.update(timestamp=(T0 + pd.Timedelta(minutes=minutes)).isoformat(), image_file="a.jpg",
               solidity=str(solidity), aspect_ratio=str(aspect), ok="true", reason="ok")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row], columns=list(PHOTO_REQUIRED_COLUMNS)).to_csv(path, index=False)


def test_attach_shape_uses_window_photo_pairing(tmp_path: Path):
    _photo_csv(tmp_path / "S1" / "photo_features.csv", minutes=1.0, solidity=0.9, aspect=2.0)
    gt = gt_frame([gt_row("P1", 0, "whole_leaf", 0.3), gt_row("P2", 0, "whole_leaf", 0.3)])
    res = res_frame([res_row("P1", 1, "card_direct", 0.3, session="S1"),
                     res_row("P2", 1, "card_direct", 0.3, session="S2")])        # S2 has no photos
    pairs = v.pair_ground_truth(v.whole_leaf_ground_truth(gt, 5)[0], res, 10)
    out = b.attach_shape(pairs, tmp_path, max_photo_gap_sec=120)
    p1 = out[out["plant_id"] == "P1"].iloc[0]
    p2 = out[out["plant_id"] == "P2"].iloc[0]
    assert p1["solidity"] == pytest.approx(0.9) and p1["aspect_ratio"] == pytest.approx(2.0)
    assert p1["shape_status"] == "ok" and p2["shape_status"] == "shape_unknown"


def test_shape_deviation_zero_at_median():
    geom = pd.DataFrame({"solidity": [0.8, 0.9, 1.0], "aspect_ratio": [1.0, 2.0, 3.0]})
    dev = b.shape_deviation(geom)
    assert dev.iloc[1] == pytest.approx(0.0)
    assert dev.iloc[0] == pytest.approx(dev.iloc[2]) and dev.iloc[0] > 0


# ---------------------------------------------------------------- planted synthetic campaign

def _campaign(n_plants: int = 60, seed: int = 3) -> pd.DataFrame:
    """Paired rows with shape: ML error independent of shape; flat-plate error grows with shape
    deviation. Returns the pairs table with solidity / aspect_ratio already attached."""
    rng = np.random.default_rng(seed)
    gt_rows, res_rows, geom = [], [], {}
    for i in range(n_plants):
        plant, session = f"P{i:02d}", f"S{i:02d}"
        dev = i / (n_plants - 1)                           # 0 (typical) .. 1 (atypical)
        geom[session] = (0.9 - 0.1 * dev * (1 if i % 2 else -1), 2.0 + dev * (1 if i % 2 else -1))
        for k in range(2):
            g = rng.uniform(0.15, 0.6)
            gt_rows.append(gt_row(plant, 60 * k, "whole_leaf", g))
            noise = rng.normal(0, 0.01, 3)
            res_rows += [res_row(plant, 60 * k + 1, "card_direct", g + noise[0], session=session),
                         res_row(plant, 60 * k + 1, "ml_corrected", g + noise[1], session=session),
                         res_row(plant, 60 * k + 1, "flat_plate", g * (1 + 0.6 * dev ** 2) + noise[2],
                                 session=session)]
    pairs = v.pair_ground_truth(v.whole_leaf_ground_truth(gt_frame(gt_rows), 5)[0], res_frame(res_rows), 10)
    pairs["solidity"] = pairs["session_id"].map(lambda s: geom[s][0])
    pairs["aspect_ratio"] = pairs["session_id"].map(lambda s: geom[s][1])
    pairs["shape_status"] = "ok"
    return pairs


def test_done_ml_better_on_atypical_no_difference_on_typical(default_config):
    pairs = b.add_split_labels(_campaign(), n_bins=3)
    t = b.split_table(pairs, "shape_bin", loa_z=1.96)
    t = t[t["level"] == v.LEVEL_PLANT].set_index(["shape_bin", "method"])
    assert t.loc[("q3", "ml_corrected"), "mae"] < 0.5 * t.loc[("q3", "flat_plate"), "mae"]
    assert abs(t.loc[("q1", "ml_corrected"), "mae"] - t.loc[("q1", "flat_plate"), "mae"]) < 0.01
    assert "p_value" not in t.columns                          # descriptive only


def test_split_bins_and_unknown_shape():
    pairs = _campaign(n_plants=9)
    pairs.loc[pairs["plant_id"] == "P00", "shape_status"] = "shape_unknown"
    pairs.loc[pairs["plant_id"] == "P00", ["solidity", "aspect_ratio"]] = math.nan
    out = b.add_split_labels(pairs, n_bins=3)
    assert set(out.loc[out["plant_id"] == "P00", "shape_bin"]) == {"shape_unknown"}
    assert set(out["shape_bin"]) == {"q1", "q2", "q3", "shape_unknown"}
    assert set(out["gs_bin"]) == {"q1", "q2", "q3"}


def test_flat_plate_judged_on_forced_only():
    gt = gt_frame([gt_row("P1", 0, "whole_leaf", 0.3)])
    res = res_frame([res_row("P1", 1, "flat_plate", 0.9, regime="free"),
                     res_row("P1", 1, "card_direct", 0.3, regime="free")])
    pairs = v.pair_ground_truth(v.whole_leaf_ground_truth(gt, 5)[0], res, 10)
    judged = b.judged_pairs(pairs)
    assert list(judged["method"]) == ["card_direct"]


def test_primary_analysis_plant_level_forced_only():
    pairs = _campaign(n_plants=20)
    pairs.loc[pairs["plant_id"] == "P19", "convection_regime"] = "mixed"
    prim = b.primary_rmse(pairs, loa_z=1.96)
    row = prim[prim["method"] == "flat_plate"].iloc[0]
    assert row["n"] == 19 and row["level"] == v.LEVEL_PLANT


def test_wilcoxon_plant_level_detects_planted_difference():
    w = b.wilcoxon_ml_vs_flat(_campaign(), level="plant")
    assert w["n"] == 60 and w["median_diff_abs_error"] < 0 and w["p_value"] < 0.001
    assert "primary" in w["label"]
    ww = b.wilcoxon_ml_vs_flat(_campaign(), level="window")
    assert ww["n"] == 120 and "non-independent" in ww["label"]


def test_wilcoxon_too_few_plants_is_blank_not_error():
    w = b.wilcoxon_ml_vs_flat(_campaign().head(0), level="plant")
    assert w["n"] == 0 and math.isnan(w["p_value"])


def test_cli_writes_outputs(tmp_path: Path):
    gt_rows, res_rows = [], []
    for i in range(6):
        gt_rows.append(gt_row(f"P{i}", 0, "whole_leaf", 0.2 + 0.05 * i))
        for m in ("card_direct", "ml_corrected", "flat_plate"):
            res_rows.append(res_row(f"P{i}", 1, m, 0.21 + 0.05 * i, session=f"S{i}"))
    gt = gt_frame(gt_rows)
    gt["timestamp_iso"] = gt["timestamp_iso"].map(lambda t: t.isoformat())
    gt.to_csv(tmp_path / "gt.csv", index=False)
    res_frame(res_rows).to_csv(tmp_path / "res.csv", index=False)
    rc = b.main(["--synthetic", "--results", str(tmp_path / "res.csv"), "--ground-truth", str(tmp_path / "gt.csv"),
                 "--processed-dir", str(tmp_path), "--out-dir", str(tmp_path / "out")])
    assert rc == 0
    for name in ("primary", "wilcoxon", "by_shape", "by_gs_range", "by_regime"):
        df = pd.read_csv(tmp_path / "out" / f"benchmark_{name}_synthetic.csv")
        assert set(df["data_label"]) == {"SYNTHETIC"}
