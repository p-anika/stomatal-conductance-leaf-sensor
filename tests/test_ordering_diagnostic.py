"""Tests for the dry-run ordering diagnostic (T6.1, definition in decision_log 2026-11-11): rank tests
and the share-below-detection table on card_direct, reported only, never a check."""
from __future__ import annotations

from math import comb

import pytest

from processing import pipeline_dry_run as pdr
from test_genotype_comparison import G1, G2, DR, WW, make_results


def diag(df, cfg):
    return pdr.ordering_diagnostic(df, cfg).set_index("item")


def test_planted_order_recovered(default_config):
    # no censoring: G1 drought 0.30 is lowest; share condition needs a share above 0, so it is "no" here
    d = diag(make_results({(G1, WW): 0.50, (G1, DR): 0.30, (G2, WW): 0.50, (G2, DR): 0.45},
                          plant_sd=0.005, window_sd=0.005), default_config)
    assert d.loc["first_genotype_drought_lowest", "value"] == "no"
    assert "share" in d.loc["first_genotype_drought_lowest", "detail"]
    assert float(d.loc["second_genotype_reduction_pct", "value"]) == pytest.approx(10, abs=2)


def test_near_zero_first_genotype_lowest_and_second_reduction(default_config):
    means = {(G1, WW): 0.40, (G1, DR): 0.004, (G2, WW): 0.40, (G2, DR): 0.24}
    d = diag(make_results(means, plant_sd=0.005, window_sd=0.005), default_config)
    assert d.loc["first_genotype_drought_lowest", "value"] == "yes"
    assert float(d.loc["second_genotype_reduction_pct", "value"]) == pytest.approx(40, abs=3)
    assert float(d.loc[f"mann_whitney_p:{G1}", "value"]) == pytest.approx(2 / comb(10, 5))
    assert "drought lower" in d.loc[f"mann_whitney_p:{G1}", "detail"]
    assert set(d["status"]) == {"DIAG"}


def test_reversed_order_not_recovered(default_config):
    means = {(G1, WW): 0.40, (G1, DR): 0.30, (G2, WW): 0.40, (G2, DR): 0.004}
    d = diag(make_results(means, plant_sd=0.005, window_sd=0.005), default_config)
    assert d.loc["first_genotype_drought_lowest", "value"] == "no"


def test_tie_at_censored_value_is_no_and_reduction_not_computable(default_config):
    means = {(G1, WW): 0.40, (G1, DR): 0.004, (G2, WW): 0.40, (G2, DR): 0.004}
    d = diag(make_results(means, plant_sd=0.0005, window_sd=0.0005), default_config)
    assert d.loc["first_genotype_drought_lowest", "value"] == "no"
    assert "tie" in d.loc["first_genotype_drought_lowest", "detail"]
    assert d.loc["second_genotype_reduction_pct", "value"] == "not computable"


def test_null_rank_rates_reported(default_config):
    r = pdr.null_rank_rates(default_config, n_frames=10, seed=0)
    assert set(r) >= {"n_frames", f"mann_whitney:{G1}", f"mann_whitney:{G2}", "kruskal_wallis"}
    assert all(0.0 <= r[k] <= 1.0 for k in r if k != "n_frames")


def test_diagnostic_rows_never_change_checks(default_config):
    import pandas as pd
    checks = pd.DataFrame([{"check": "f_exit_code", "status": "PASS", "value": 0, "detail": ""}])
    means = {(G1, WW): 0.40, (G1, DR): 0.30, (G2, WW): 0.40, (G2, DR): 0.004}       # order NOT recovered
    out = pdr.append_diagnostic(checks, make_results(means), default_config)
    assert out.loc[out["check"] == "f_exit_code", "status"].iloc[0] == "PASS"
    assert set(out.loc[out["check"] != "f_exit_code", "status"]) == {"DIAG"}
