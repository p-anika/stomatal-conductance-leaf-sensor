"""Tests for analysis/genotype_comparison.py (T5.3): plant-level inference, censoring handling,
effect sizes, rank tests, sensitivity outcomes. Planted synthetic data only."""
from __future__ import annotations

import math
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import genotype_comparison as gc

G1, G2 = "Houjaku_Kuwazu", "Williams_82"
WW, DR = "well_watered", "drought_stressed"


def make_results(means: dict[tuple[str, str], float], n_plants: int = 5, n_windows: int = 6,
                 plant_sd: float = 0.03, window_sd: float = 0.03, dl: float = 0.05, seed: int = 0,
                 regime: str = "forced") -> pd.DataFrame:
    """leaf_gs_results-like rows (card_direct, ml_corrected, flat_plate per window) with planted
    group means. True g_s below dl -> censor below_detection with g_s_mol blank."""
    rng = np.random.default_rng(seed)
    rows = []
    for (geno, trt), mu in means.items():
        for p in range(n_plants):
            plant = f"{geno[:2]}-{trt[:2]}-{p}"
            plant_mu = mu + rng.normal(0, plant_sd)
            for w in range(n_windows):
                g = plant_mu + rng.normal(0, window_sd)
                for method, role in (("card_direct", "primary"), ("ml_corrected", "secondary"),
                                     ("flat_plate", "baseline")):
                    below = g < dl
                    rows.append({"plant_id": plant, "genotype": geno, "water_treatment": trt, "block": "",
                                 "session_id": f"S-{plant}", "window_start_iso": f"2026-11-09T10:{w:02d}:00+00:00",
                                 "method": method, "method_role": role, "valid": True, "reason": "ok",
                                 "steady_state": True, "censor": "below_detection" if below else "none",
                                 "g_s_mol": math.nan if below else g, "g_s_mol_dl": dl, "flags": "",
                                 "convection_regime": regime,
                                 "jones_ig": max(g, 0.0) * 2.0, "cwsi": 1.0 - min(max(g, 0.0), 1.0),
                                 "jones_ig_censor": "none"})
    return pd.DataFrame(rows)


PLANTED = {(G1, WW): 0.50, (G1, DR): 0.30, (G2, WW): 0.50, (G2, DR): 0.45}   # 40 % and 10 % drops


def _effect(tables, outcome="card_direct", subset="all", substitution="half_dl", contrast=None):
    e = tables["effects"]
    e = e[(e["outcome"] == outcome) & (e["subset"] == subset) & (e["substitution"] == substitution)]
    return e.set_index("contrast").loc[contrast] if contrast else e


# ---------------------------------------------------------------- planted effect recovered

def test_planted_drop_recovered(default_config):
    t = gc.run(make_results(PLANTED), default_config, campaign_dl=None, label="SYNTHETIC")
    d1 = _effect(t, contrast=f"drop_under_drought:{G1}")
    d2 = _effect(t, contrast=f"drop_under_drought:{G2}")
    # one dataset: a 95 % CI covers the planted 40 % only 95 % of the time, so check estimate and CI shape
    assert d1["pct_drop"] == pytest.approx(40, abs=8) and 0 < d1["ci_lo_pct"] < d1["pct_drop"] < d1["ci_hi_pct"]
    assert d2["pct_drop"] == pytest.approx(10, abs=8)
    diff = _effect(t, contrast="difference_of_drops")
    assert diff["ci_lo_pct"] > 0 or diff["ci_hi_pct"] < 0            # interaction found (distribution-free)
    a = t["anova"]
    a = a[(a["outcome"] == "card_direct") & (a["substitution"] == "half_dl") & (a["subset"] == "all")]
    inter = a[a["term"].str.contains(":")].iloc[0]
    assert inter["p_value"] < 0.05 and inter["df"] == 1
    resid = a[a["term"] == "Residual"].iloc[0]
    assert resid["df"] == 20 - 4                                     # plants, not windows, are the replicates
    assert set(t["summary"]["data_label"]) == {"SYNTHETIC"}


def test_mixed_model_supportive_and_reports_convergence(default_config):
    t = gc.run(make_results(PLANTED), default_config, campaign_dl=None, label="SYNTHETIC")
    m = t["mixed_model"]
    m = m[(m["outcome"] == "card_direct") & (m["substitution"] == "half_dl") & (m["subset"] == "all")]
    assert "converged" in m.columns and m["converged"].notna().all()
    inter = m[m["term"].str.contains(":")].iloc[0]
    # G2 drops 0.05 instead of 0.20: interaction about +0.15 mol m-2 s-1
    assert inter["estimate"] == pytest.approx(0.15, abs=0.06)
    assert inter["ci_lo"] < inter["estimate"] < inter["ci_hi"]


def test_near_zero_stressed_group_recovered_and_robust(default_config):
    means = {(G1, WW): 0.50, (G1, DR): 0.01, (G2, WW): 0.50, (G2, DR): 0.40}
    t = gc.run(make_results(means, plant_sd=0.005, window_sd=0.005), default_config, campaign_dl=None,
               label="SYNTHETIC")
    drops = {s: _effect(t, substitution=s, contrast=f"drop_under_drought:{G1}")["pct_drop"]
             for s in ("zero", "half_dl", "dl")}
    assert drops["zero"] == pytest.approx(100, abs=2)
    assert drops["half_dl"] == pytest.approx(95, abs=2)
    assert drops["dl"] == pytest.approx(90, abs=2)
    rob = t["robustness"]
    r = rob[(rob["outcome"] == "card_direct") & (rob["contrast"] == f"drop_under_drought:{G1}")].iloc[0]
    assert bool(r["robust"])
    share = t["censor_share"]
    s = share[(share["outcome"] == "card_direct") & (share["genotype"] == G1) & (share["water_treatment"] == DR)]
    assert s["share_below_detection"].iloc[0] == pytest.approx(1.0)
    rank = t["rank_tests"]
    mw = rank[(rank["outcome"] == "card_direct") & (rank["test"] == f"mann_whitney:{G1}")].iloc[0]
    assert mw["p_value"] == pytest.approx(2 / comb(10, 5), rel=1e-6)   # all 5 stressed below all 5 watered


# ---------------------------------------------------------------- censoring without a limit

def _blank_dl(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[df["censor"] == "below_detection", "g_s_mol_dl"] = math.nan
    return df


def test_campaign_dl_fallback_flagged_and_counted(default_config):
    means = {(G1, WW): 0.50, (G1, DR): 0.02, (G2, WW): 0.50, (G2, DR): 0.40}
    t = gc.run(_blank_dl(make_results(means, plant_sd=0.005, window_sd=0.005)), default_config,
               campaign_dl=0.08, label="SYNTHETIC")
    c = t["counts"]
    c = c[(c["outcome"] == "card_direct") & (c["subset"] == "all")].set_index("item")["n"]
    assert c["dl_campaign"] == 30 and c.get("dl_none", 0) == 0
    d = _effect(t, substitution="dl", contrast=f"drop_under_drought:{G1}")
    assert d["pct_drop"] == pytest.approx(100 * (1 - 0.08 / 0.5), abs=2)


def test_no_limit_refuses_numeric_but_rank_and_share_run(default_config):
    means = {(G1, WW): 0.50, (G1, DR): 0.02, (G2, WW): 0.50, (G2, DR): 0.40}
    t = gc.run(_blank_dl(make_results(means, plant_sd=0.005, window_sd=0.005)), default_config,
               campaign_dl=None, label="SYNTHETIC")
    notes = t["notes"]
    n = notes[(notes["outcome"] == "card_direct") & (notes["subset"] == "all")]
    assert n["message"].str.contains("no detection limit").any()
    assert _effect(t, outcome="card_direct").empty
    assert not t["rank_tests"][t["rank_tests"]["outcome"] == "card_direct"].empty
    assert not t["censor_share"][t["censor_share"]["outcome"] == "card_direct"].empty
    c = t["counts"]
    c = c[(c["outcome"] == "card_direct") & (c["subset"] == "all")].set_index("item")["n"]
    assert c["dl_none"] == 30


# ---------------------------------------------------------------- exclusions, outcomes, subsets

def test_invalid_rows_excluded_and_counted_by_reason(default_config):
    df = make_results(PLANTED)
    idx = df.index[(df["method"] == "card_direct")][:3]
    df.loc[idx, ["valid", "reason"]] = [False, "not_steady_state"]
    t = gc.run(df, default_config, campaign_dl=None, label="SYNTHETIC")
    c = t["counts"]
    c = c[(c["outcome"] == "card_direct") & (c["subset"] == "all")].set_index("item")["n"]
    assert c["excluded:not_steady_state"] == 3 and c["windows_used"] == 120 - 3


def test_sensitivity_outcomes_and_roles(default_config):
    t = gc.run(make_results(PLANTED), default_config, campaign_dl=None, label="SYNTHETIC")
    roles = t["summary"].drop_duplicates("outcome").set_index("outcome")["role"]
    assert roles["card_direct"] == "primary"
    for o in ("ml_corrected", "flat_plate", "jones_ig", "cwsi"):
        assert roles[o] == "sensitivity"


def test_cwsi_signed_with_direction(default_config):
    t = gc.run(make_results(PLANTED), default_config, campaign_dl=None, label="SYNTHETIC")
    e = _effect(t, outcome="cwsi", substitution="none", contrast=f"drop_under_drought:{G1}")
    assert e["difference"] > 0 and math.isnan(e["pct_drop"])          # CWSI rises under stress
    assert "rises" in e["direction"]


def test_jones_counted_once_per_window(default_config):
    t = gc.run(make_results(PLANTED), default_config, campaign_dl=None, label="SYNTHETIC")
    c = t["counts"]
    c = c[(c["outcome"] == "jones_ig") & (c["subset"] == "all")].set_index("item")["n"]
    assert c["windows_used"] == 120


def test_forced_only_subset(default_config):
    df = make_results(PLANTED)
    df.loc[df["plant_id"].str.endswith("-0"), "convection_regime"] = "free"
    t = gc.run(df, default_config, campaign_dl=None, label="SYNTHETIC")
    c = t["counts"]
    c = c[(c["outcome"] == "card_direct") & (c["subset"] == "forced_only")].set_index("item")["n"]
    assert c["windows_used"] == 120 - 4 * 6


def test_no_forced_windows_is_a_note_not_an_error(default_config):
    t = gc.run(make_results(PLANTED, regime=""), default_config, campaign_dl=None, label="SYNTHETIC")
    n = t["notes"]
    assert n[(n["subset"] == "forced_only")]["message"].str.contains("no forced").any()


# ---------------------------------------------------------------- building blocks

def test_exact_permutation_smallest_p():
    ww = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4])
    dr = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    p = gc.permutation_p(ww.to_numpy(), dr.to_numpy(), n_permutations=10_000, rng=np.random.default_rng(0))
    assert p == pytest.approx(2 / comb(10, 5))


def test_campaign_detection_limit_reads_sweep(tmp_path: Path):
    path = tmp_path / "sweep.csv"
    pd.DataFrame({"g_s_true_mol": [0.01, 0.05, 0.1, 1.0], "resolved": [False, False, True, True]}).to_csv(path, index=False)
    assert gc.campaign_detection_limit(path) == pytest.approx(0.1)
    pd.DataFrame({"g_s_true_mol": [0.01], "resolved": [False]}).to_csv(path, index=False)
    assert gc.campaign_detection_limit(path) is None
    assert gc.campaign_detection_limit(tmp_path / "missing.csv") is None


def test_cli_writes_outputs(tmp_path: Path):
    make_results(PLANTED, n_plants=4, n_windows=3).to_csv(tmp_path / "res.csv", index=False)
    rc = gc.main(["--synthetic", "--results", str(tmp_path / "res.csv"),
                  "--detection-limit-sweep", str(tmp_path / "none.csv"), "--out-dir", str(tmp_path / "out")])
    assert rc == 0
    for name in ("summary", "effects", "anova", "mixed_model", "rank_tests", "censor_share", "robustness",
                 "counts", "notes"):
        df = pd.read_csv(tmp_path / "out" / f"genotype_{name}_synthetic.csv")
        assert df.empty or set(df["data_label"]) == {"SYNTHETIC"}
