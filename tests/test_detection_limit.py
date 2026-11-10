"""Tests for analysis/detection_limit.py (T5.4).

Resolved = the A33 three-part rule (fraction of usable draws >= dl_min_resolved_fraction,
relative 95 % width < max_relative_width, p97.5 < g_s_cap_mol). The sweep reports a resolved
BAND (lowest and highest resolved true g_s), because near the cap the upper bound is not bounded.
Synthetic numbers prove the code works, nothing about plants.
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

from analysis.aggregate_results import RESULT_COLUMNS
from analysis.detection_limit import (
    bracket_table,
    main as dl_main,
    representative_reading,
    resolved_band,
    session_brackets,
    summary_line,
    sweep,
    synthetic_brackets,
)
from ml.synthetic_data import make_card_sets, simulate_leaf_session, synthetic_calibration, write_session
from processing.config import get_light_source, load_config
from processing.data_io import list_sessions, load_session
from processing.gs_calculator import LeafReading
from processing.uncertainty import detection_limit_mol, dl_grid, get_sigmas

REPO = Path(__file__).resolve().parent.parent
TIGHT = dict(sigma_leaf_temp_c=0.2, sigma_air_temp_c=0.1, sigma_rh_pct=1.0, sigma_lux_fraction=0.05,
             sigma_rb_fraction=0.1)                     # calibration-sized, TEST ONLY (invented)


def make_cfg(tight: bool = True) -> dict[str, Any]:
    cfg = load_config(REPO / "config.yaml")
    cfg["uncertainty"]["n_draws"] = 300                 # speed only
    cfg["uncertainty"]["dl_grid_points"] = 12
    if tight:
        cfg["uncertainty"].update(TIGHT)
    return cfg


@pytest.fixture(scope="module")
def cfg() -> dict[str, Any]:
    return make_cfg()


@pytest.fixture(scope="module")
def cal(cfg) -> dict[str, Any]:
    return synthetic_calibration(cfg)


def reading(r_bh: float = 40.0) -> LeafReading:
    # synthetic.conditions: 24 C, 50 %, 995 hPa, 120 W m-2 = 12000 lux x k 0.01
    return LeafReading(25.0, 24.0, 24.0, 50.0, 995.0, 12000.0, "led_lamp", r_bh, "card_direct")


def sigmas(cfg, cal):
    return get_sigmas(cfg, cal, get_light_source(cfg, cal, "led_lamp"))


@pytest.fixture(scope="module")
def swept(cfg, cal) -> pd.DataFrame:
    return sweep(reading(), cfg, cal, sigmas(cfg, cal), seed=0)


def test_sweep_columns_and_grid(swept, cfg):
    assert np.allclose(swept["g_s_true_mol"], dl_grid(cfg))
    for c in ("leaf_temp_c", "g_s_mol_lo", "g_s_mol_median", "g_s_mol_hi", "frac_ok", "relative_width",
              "resolved", "root_found"):
        assert c in swept.columns
    assert swept["root_found"].all()


def test_lowest_resolved_equals_per_window_limit(swept, cfg, cal):
    """Same rule, same draws: T5.4 and the limit stored on each censored window agree."""
    lo, _ = resolved_band(swept)
    dl, no_root = detection_limit_mol(reading(), cfg, cal, sigmas(cfg, cal), 300, 0)
    assert no_root == 0
    assert lo == pytest.approx(dl)


def test_band_has_an_upper_end(swept, cfg):
    lo, hi = resolved_band(swept)
    assert lo is not None and hi is not None and lo < hi < cfg["uncertainty"]["g_s_cap_mol"]
    assert not bool(swept["resolved"].iloc[-1])            # at the cap: not bounded above
    assert not bool(swept["resolved"].iloc[0])             # near zero: not resolved


def test_usable_fraction_rises_at_low_gs(swept):
    low = swept[swept["g_s_true_mol"] < 0.05]["frac_ok"].to_numpy()
    assert np.all(np.diff(low) >= -0.02)                    # Monte Carlo wobble allowed


def test_placeholder_sigmas_resolve_nothing(cal):
    """With the pre-calibration placeholder sigmas nothing is resolved; the summary must say so."""
    cfg = make_cfg(tight=False)
    sw = sweep(reading(), cfg, cal, sigmas(cfg, cal), seed=0)
    assert resolved_band(sw) == (None, None)
    line = summary_line(sw, None, "SYNTHETIC", "card_direct", 3)
    assert "no true g_s" in line and "\n" not in line


def test_synthetic_brackets_unresolved(cfg, cal):
    br = bracket_table(synthetic_brackets(reading(), cfg, cal), cfg, cal, sigmas(cfg, cal))
    assert list(br["reference"]) == ["petroleum_jelly_leaf", "wet_reference"]
    assert not br["resolved"].astype(bool).any()
    assert br["as_expected"].astype(bool).all()
    dry = br.iloc[0]
    assert dry["censor"] == "below_detection" or dry["g_s_mol_hi"] < 0.1
    wet = br.iloc[1]
    assert wet["censor"] == "above_range" or wet["g_s_mol_hi"] >= cfg["uncertainty"]["g_s_cap_mol"]


def _results_df(n: int = 4) -> pd.DataFrame:
    rows = []
    for i in range(n):
        row = {c: math.nan for c in RESULT_COLUMNS}
        row.update(session_id=f"S{i}", method="card_direct", method_role="primary", valid=True,
                   censor="below_detection" if i == 0 else "none", reason="ok", light_source="led_lamp",
                   air_temp_c=23.0 + i, rh_pct=40.0 + 10 * i, pressure_hpa=995.0, lux=12000.0,
                   leaf_temp_c=24.0, r_bh=30.0 + 10 * i,
                   r_bh_lo=(30.0 + 10 * i) * math.exp(-1.959964 * 0.15),
                   r_bh_hi=(30.0 + 10 * i) * math.exp(1.959964 * 0.15))
        rows.append(row)
    bad = dict(rows[0], valid=False, reason="missing_input", air_temp_c=99.0, r_bh=999.0)
    other = dict(rows[1], method="flat_plate", method_role="baseline", air_temp_c=-5.0)
    return pd.DataFrame(rows + [bad, other], columns=list(RESULT_COLUMNS))


def test_representative_reading_uses_primary_valid_rows(cfg):
    rd, sigma_rb, n = representative_reading(_results_df(), cfg)
    assert n == 4                                           # censored row counted, invalid/other not
    assert rd.air_temp_c == pytest.approx(24.5)
    assert rd.relative_humidity_pct == pytest.approx(55.0)
    assert rd.r_bh == pytest.approx(45.0)
    assert rd.surroundings_temp_c == rd.air_temp_c
    assert rd.light_source == "led_lamp" and rd.r_bh_method == "card_direct"
    assert sigma_rb == pytest.approx(0.15, rel=1e-4)


def test_representative_reading_needs_rows(cfg):
    df = _results_df().assign(valid=False)
    with pytest.raises(ValueError, match="no valid card_direct rows"):
        representative_reading(df, cfg)


def test_session_brackets_from_vaseline_and_wet_sessions(tmp_path, cfg):
    table, truth = make_card_sets(cfg)["test"]
    meta = {"dry_reference_type": "vaseline_leaf", "wet_reference_present": "true", "card_id": ""}
    sim = simulate_leaf_session(0.3, table.iloc[0], truth.iloc[0], cfg=cfg, seed=3, noise=False,
                                conditions={"duration_min": 6.0}, plant_id="VAS", metadata=meta)
    leaf_c = sim.sensor_log["leaf_temp_c"]
    sim.sensor_log["card_temp_c"] = leaf_c + 1.5
    sim.sensor_log["wet_ref_temp_c"] = leaf_c - 2.0
    write_session(sim, tmp_path)
    raw = tmp_path / "data" / "raw"
    sessions = [load_session(s, raw) for s in list_sessions(raw)]
    pairs = session_brackets(sessions, cfg, r_bh=40.0)
    kinds = [k for k, _ in pairs]
    assert kinds.count("petroleum_jelly_leaf") == 4 and kinds.count("wet_reference") == 4   # steady windows
    for k, rd in pairs:
        assert rd.r_bh == 40.0 and rd.surroundings_temp_c == rd.air_temp_c
    dry = [rd.leaf_temp_c for k, rd in pairs if k == "petroleum_jelly_leaf"]
    wet = [rd.leaf_temp_c for k, rd in pairs if k == "wet_reference"]
    assert np.allclose(np.array(dry) - np.array(wet), 3.5)


def test_cli_writes_figure_table_and_one_line(tmp_path, cfg, capsys):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(copy.deepcopy(cfg)), encoding="utf-8")
    res = tmp_path / "leaf_gs_results_synthetic.csv"
    _results_df().to_csv(res, index=False)
    fig, csv, brk = tmp_path / "dl.png", tmp_path / "sweep.csv", tmp_path / "brackets.csv"
    (tmp_path / "raw").mkdir()
    code = dl_main(["--synthetic", "--results", str(res), "--config", str(cfg_path), "--raw-dir",
                    str(tmp_path / "raw"), "--figure", str(fig), "--out-csv", str(csv), "--brackets-csv", str(brk)])
    assert code == 0
    assert fig.is_file() and fig.stat().st_size > 10_000
    assert len(pd.read_csv(csv)) == cfg["uncertainty"]["dl_grid_points"]
    assert len(pd.read_csv(brk)) == 2
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("Detection limit")]
    assert len(lines) == 1 and "SYNTHETIC" in lines[0]
