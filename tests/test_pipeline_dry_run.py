"""Tests for processing/pipeline_dry_run.py (T6.1). The full 20-plant run is the command itself
(python -m processing.pipeline_dry_run); these tests run a small version to check the flow,
the outputs, the labels and the placeholder-sigma behaviour."""
from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest

from processing import pipeline_dry_run as pdr

CHECKS = {"a_completes", "b_ml_rbh_beats_flat_on_atypical", "c_interaction_recovered",
          "d_censoring_effect_reported", "e_runtime", "f_exit_code"}


def small_cfg(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["dry_run"]["n_plants_per_group"] = 2
    cfg["synthetic"]["n_shapes"] = 10            # training needs >= ml.min_shapes (10)
    cfg["dry_run"]["fast"]["ml"]["n_bootstrap"] = 5
    cfg["uncertainty"].update(n_draws=200, dl_grid_points=12)   # test speed only; the dry run itself uses config
    return cfg


@pytest.fixture(scope="module")
def target_run(tmp_path_factory, request):
    from processing.config import load_config
    cfg = small_cfg(load_config(pdr.REPO_ROOT / "config.yaml"))
    out = tmp_path_factory.mktemp("dry")
    return pdr.run(cfg, sigmas="target", settings="fast", seed=0, k_error=0.0, out_dir=out, figure_dir=out), out


def test_completes_with_every_check_listed(target_run):
    res, _ = target_run
    checks = res.checks.set_index("check")
    assert CHECKS <= set(checks.index)
    assert set(checks.loc[sorted(CHECKS), "status"]) <= {"PASS", "FAIL", "NOT EVALUATED"}
    assert set(checks["status"]) <= {"PASS", "FAIL", "NOT EVALUATED", "INFO", "DIAG"}  # INFO, DIAG = reported only
    assert checks.loc["a_completes", "status"] == "PASS"


def test_three_methods_with_uncertainty_and_truth(target_run):
    res, _ = target_run
    r = res.results
    assert set(r["method"]) == {"card_direct", "ml_corrected", "flat_plate"}
    ok = r[r["valid"].astype(bool) & (r["censor"] == "none")]
    assert ok["g_s_mol_lo"].notna().all() and ok["g_s_mol_hi"].notna().all()
    assert r["g_s_true_mol"].notna().all() and r["shape_deviation"].notna().all()
    assert ok.loc[ok["method"] == "ml_corrected"].shape[0] > 0     # the trained model was used


def test_outputs_labelled_conditional(target_run):
    res, out = target_run
    for name in ("pipeline_dry_run_checks_synthetic.csv", "pipeline_dry_run_results_synthetic.csv"):
        df = pd.read_csv(out / name)
        labels = set(df["data_label"])
        assert len(labels) == 1
        label = labels.pop()
        assert label.startswith(pdr.LABEL_TARGET) and label.endswith("not a prediction")
        assert "air 0.5 C, RH 3.0 %" in label                         # default scenario = placeholder air/RH
    assert (out / "pipeline_dry_run_synthetic.png").stat().st_size > 1000


def test_placeholder_sigmas_make_no_claims(tmp_path: Path, default_config):
    cfg = small_cfg(default_config)
    res = pdr.run(cfg, sigmas="placeholder", settings="fast", seed=0, k_error=0.0, out_dir=tmp_path,
                  figure_dir=tmp_path)
    checks = res.checks.set_index("check")
    assert checks.loc["g_placeholder_no_claims", "status"] == "PASS"
    assert "nothing resolved" in checks.loc["g_placeholder_no_claims", "detail"]
    for c in ("b_ml_rbh_beats_flat_on_atypical", "c_interaction_recovered", "d_censoring_effect_reported"):
        assert checks.loc[c, "status"] == "NOT EVALUATED"
    r = res.results
    assert not r.loc[r["valid"].astype(bool), "resolved"].astype(bool).any()


def test_dry_run_config_does_not_touch_input(default_config):
    before = copy.deepcopy(default_config)
    cfg2, cal = pdr.dry_run_config(default_config, sigmas="target", settings="fast", k_error=0.1)
    assert default_config == before
    assert cfg2["synthetic"]["session_bias"] is True
    assert cal["uncertainty"]["sigma_leaf_temp_c"] == pytest.approx(0.2)
    k_true = default_config["synthetic"]["k_wm2_per_lux"]
    assert cal["light_sources"]["led_lamp"]["k_wm2_per_lux"] == pytest.approx(1.1 * k_true)
    assert cfg2["ml"]["n_bootstrap"] == default_config["dry_run"]["fast"]["ml"]["n_bootstrap"]


def test_dry_run_pins_full_detection_limit_settings_and_shared_draws(default_config):
    for settings in ("fast", "full"):
        cfg2, _ = pdr.dry_run_config(default_config, sigmas="target", settings=settings, k_error=0.0)
        assert cfg2["uncertainty"]["n_draws"] == default_config["uncertainty"]["n_draws"]
        assert cfg2["uncertainty"]["dl_grid_points"] == default_config["uncertainty"]["dl_grid_points"]
        assert cfg2["uncertainty"]["card_leaf_shared_draws"] is True
    assert default_config["uncertainty"]["card_leaf_shared_draws"] is False      # real pipeline default unchanged


def test_check_b_uses_rbh_and_reports_gs_with_offsets_on_and_off(target_run):
    res, _ = target_run
    checks = res.checks.set_index("check")
    detail = checks.loc["b_ml_rbh_beats_flat_on_atypical", "detail"]
    assert "r_bH rel. RMSE" in detail and "offsets on" in detail and "offsets off" in detail


def test_check_d_reports_size_and_direction_only(target_run):
    res, _ = target_run
    d = res.checks.set_index("check").loc["d_censoring_effect_reported"]
    if d["status"] == "PASS":
        assert ("changes it by" in d["detail"]) or ("loses the group" in d["detail"])


@pytest.mark.parametrize("scenario, air, rh", [("placeholder", 0.5, 3.0), ("intermediate", 0.3, 2.5),
                                               ("target", 0.2, 2.0), ("relaxed", 0.3, 3.0)])
def test_air_rh_scenarios(default_config, scenario, air, rh):
    cfg2, cal = pdr.dry_run_config(default_config, sigmas="target", settings="fast", k_error=0.0, air_rh=scenario)
    assert cal["uncertainty"]["sigma_air_temp_c"] == air and cal["uncertainty"]["sigma_rh_pct"] == rh
    assert cfg2["uncertainty"]["sigma_air_temp_c"] == air                       # session offsets use them too
    assert f"air {air} C, RH {rh} %" in pdr.target_label(cfg2)


def test_campaign_width_curve_written(target_run):
    _, out = target_run
    sw = pd.read_csv(out / "pipeline_dry_run_dl_sweep_synthetic.csv")
    assert {"g_s_true_mol", "relative_width", "resolved"} <= set(sw.columns)


def test_relaxed_scenario_overrides_leaf_only(default_config):
    cfg2, cal = pdr.dry_run_config(default_config, sigmas="target", settings="fast", k_error=0.0, air_rh="relaxed")
    u = cal["uncertainty"]
    assert (u["sigma_leaf_temp_c"], u["sigma_air_temp_c"], u["sigma_rh_pct"]) == (0.4, 0.3, 3.0)
    assert u["sigma_lux_fraction"] == default_config["dry_run"]["target_sigmas"]["sigma_lux_fraction"]
    label = pdr.target_label(cfg2)
    assert "leaf temperature 0.4 C" in label and label.endswith("not a prediction")
    for other in ("placeholder", "intermediate", "target"):                     # other scenarios unchanged
        _, c = pdr.dry_run_config(default_config, sigmas="target", settings="fast", k_error=0.0, air_rh=other)
        assert c["uncertainty"]["sigma_leaf_temp_c"] == 0.2
