"""Tests for ml/delta_t_sweep.py: bias of mean card f versus card delta T (synthetic only).

Owner decisions 2026-11-09: lamp levels x two winds x offset scales (0, 0.5, 1 x placeholder
sigmas); report true delta T, incident and absorbed irradiance, sessions surviving the gate,
bias with and without the r_bH <= 0 rejection and with the gate off, each with a standard error.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.delta_t_sweep import TABLE_COLUMNS, format_table, main, run_sweep
from processing.config import load_config

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def table():
    cfg = load_config(REPO / "config.yaml")
    return run_sweep(cfg, irradiances=[60.0, 400.0], winds=[0.8], scales=[0.0, 1.0], repeats=2, n_cards=8)


def test_table_produced(table):
    assert list(table.columns) == list(TABLE_COLUMNS)
    assert len(table) == 2 * 1 * 2
    assert (table["n_sessions"] == 16).all()
    assert np.allclose(table["absorbed_sw_wm2"], 0.95 * table["irradiance_incident_wm2"])
    for col in ("bias_kept", "bias_gated_all", "bias_nogate_all"):
        assert table[f"se_{col.removeprefix('bias_')}"].notna().all()
    text = format_table(table)
    assert "SYNTHETIC" in text and "T7.5" in text and "incident" in text and "absorbed" in text


def test_delta_t_grows_with_lamp(table):
    t = table[table["offset_scale"] == 1.0].set_index("irradiance_incident_wm2")
    assert t.loc[400.0, "delta_t_true_c"] > 4 * t.loc[60.0, "delta_t_true_c"]


def test_bias_falls_as_delta_t_grows(table):
    t = table[table["offset_scale"] == 1.0].set_index("irradiance_incident_wm2")
    assert t.loc[60.0, "suspect_fraction"] > t.loc[400.0, "suspect_fraction"]
    assert abs(t.loc[400.0, "bias_kept"]) < abs(t.loc[60.0, "bias_kept"])
    assert t.loc[400.0, "spread_kept"] < t.loc[60.0, "spread_kept"]


def test_noise_only_has_small_bias(table):
    """Offset scale 0: per-sample noise only, so no suspects and almost no bias at either level."""
    t = table[table["offset_scale"] == 0.0]
    assert (t["suspect_fraction"] == 0).all()
    assert (t["bias_kept"].abs() < 0.02).all()


def test_cli_writes_csv(tmp_path):
    out = tmp_path / "sweep.csv"
    assert main(["--out", str(out), "--irradiance", "100", "--wind", "0.8", "--scales", "1.0",
                 "--repeats", "1", "--n-cards", "4"]) == 0
    assert out.is_file()
