"""Tests for analysis/make_figures.py (T5.7): one command, every figure and table built or listed as
missing with a reason, SYNTHETIC stamping. Synthetic data only."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis import genotype_comparison as gc
from analysis import make_figures as mf
from analysis import validate_gs_ground_truth as v
from test_genotype_comparison import PLANTED, make_results
from test_validate_gs_ground_truth import gt_frame, gt_row, res_frame, res_row


def _write_inputs(root: Path, cfg) -> None:
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    results = make_results(PLANTED, n_plants=4, n_windows=3)
    results.to_csv(processed / "leaf_gs_results_synthetic.csv", index=False)
    for name, df in gc.run(results, cfg, campaign_dl=None, label="SYNTHETIC").items():
        df.to_csv(processed / f"genotype_{name}_synthetic.csv", index=False)
    gt = gt_frame([gt_row(f"P{i}", 0, "whole_leaf", 0.2 + 0.05 * i) for i in range(5)])
    res = res_frame([res_row(f"P{i}", 1, "card_direct", 0.22 + 0.05 * i) for i in range(5)])
    for name, df in v.run(res, gt, cfg, label="SYNTHETIC").items():
        df.to_csv(processed / f"validation_gs_{name}_synthetic.csv", index=False)
    models = root / "data" / "models"
    models.mkdir(parents=True)
    card = {"status": "ok", "n_shapes": 12, "metrics": {
        "logo": {"null_f1": {"rmse_s_m": 30.0}, "ridge": {"rmse_s_m": 20.0}, "forest": {"rmse_s_m": 22.0}},
        "nested": {"outer_rmse_s_m": 21.0, "null_rmse_s_m": 30.0}}}
    (models / "correction_model_card.json").write_text(json.dumps(card), encoding="utf-8")


def _index(root: Path) -> pd.DataFrame:
    text = (root / "docs" / "tables" / "index_synthetic.md").read_text(encoding="utf-8")
    assert "SYNTHETIC" in text
    rows = [l.strip("|").split("|") for l in text.splitlines() if l.startswith("| ") and "---" not in l][1:]
    return pd.DataFrame([[x.strip() for x in r] for r in rows], columns=["kind", "name", "status", "detail"])


def test_figures_only_builds_what_has_inputs(tmp_path: Path, default_config):
    _write_inputs(tmp_path, default_config)
    assert mf.main(["--synthetic", "--figures-only", "--root", str(tmp_path)]) == 0
    figs = tmp_path / "docs" / "figures"
    for name in ("gs_by_group", "method_comparison", "ml_cv", "below_detection"):
        assert (figs / f"{name}_synthetic.png").stat().st_size > 1000, name
    idx = _index(tmp_path).set_index(["kind", "name"])
    assert idx.loc[("figure", "calibration_residuals"), "status"] == "missing"
    assert "T7" in idx.loc[("figure", "calibration_residuals"), "detail"]
    assert idx.loc[("figure", "vaseline_rb"), "status"] == "missing"
    assert idx.loc[("table", "genotype_effects"), "status"] == "built"
    assert (tmp_path / "docs" / "tables" / "genotype_effects_synthetic.csv").is_file()
    assert idx.loc[("table", "card_buoyancy_check"), "status"] == "missing"


def test_every_listed_item_is_in_the_index(tmp_path: Path, default_config):
    _write_inputs(tmp_path, default_config)
    mf.main(["--synthetic", "--figures-only", "--root", str(tmp_path)])
    idx = _index(tmp_path)
    assert set(idx.loc[idx["kind"] == "figure", "name"]) == set(mf.OWN_FIGURES) | set(mf.STEP_FIGURES)
    assert set(idx.loc[idx["kind"] == "table", "name"]) == set(mf.TABLES)


def test_full_run_on_empty_root_skips_steps_with_reasons(tmp_path: Path):
    assert mf.main(["--synthetic", "--root", str(tmp_path)]) == 0
    idx = _index(tmp_path)
    steps = idx[idx["kind"] == "step"]
    assert len(steps) == 7
    assert set(steps["status"]) == {"skipped"}
    assert (steps["detail"] != "").all()
