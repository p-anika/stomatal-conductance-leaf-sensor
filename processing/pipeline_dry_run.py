"""End-to-end dry run on synthetic data (T6.1). Run it before every merge to main.

    python -m processing.pipeline_dry_run [--seed N] [--sigmas target|placeholder]
                                          [--settings fast|full] [--k-error FRACTION]
                                          [--air-rh placeholder|intermediate|target|relaxed]

Everything happens in memory (nothing is written to data/raw):
1. Cards: the synthetic "train" card shapes (T4.2, hidden correction f), each simulated at
   dry_run.card_winds_ms and dry_run.card_irradiance_wm2 -> build_training_set solves card r_bH
   (eq. 9) -> train_correction_model (T4.4/T4.5).
2. Leaves: genotype x treatment x dry_run.n_plants_per_group plants; each plant gets one
   held-out "test" leaf shape and a planted true g_s (dry_run.true_gs_mol, x (1 + N(0, plant_sd))),
   with the most typical train card in the frame; simulate_leaf_session (forward physics).
3. g_s three ways with uncertainty: aggregate_results.session_results (card_direct,
   ml_corrected, flat_plate; Monte Carlo interval, resolved label, detection limit).
4. Statistics: genotype_comparison.run (plant level; 0 / half / full detection-limit substitutions).
5. Figure: device vs true g_s per method, atypical leaf shapes highlighted.
6. Checks (PASS / FAIL / NOT EVALUATED), exit code 0 only if none FAILS:
   a  the pipeline completes (all three methods give rows; statistics run)
   b  r_bH relative RMSE vs the true leaf r_bH: ml_corrected < flat_plate on atypical leaves
      (top tertile of shape deviation, T5.2 definition), valid uncensored windows. CHANGED after
      run 1 (owner 2026-11-10): run 1 judged plant-level g_s RMSE, which per-session thermal
      offsets dominate. The plant-level g_s RMSE is still reported, with the offsets on and
      (atypical plants re-simulated) with the offsets off.
   c  planted interaction recovered: card_direct, primary substitution, bootstrap CI of the
      difference of drops excludes 0 and the first genotype's drop >= min_first_genotype_drop_pct
   d  the change in the first genotype's drop when censored windows are dropped is reported
      (size and direction, or "group lost"); the direction is NOT asserted (reworded after run 1)
   e  runtime < dry_run.max_runtime_s (300 s; was 120 s for the old fast settings; machine-dependent;
      fast settings only, full settings: reported)
   f  exit code 0 (no check failed)
   g  placeholder sigmas only: completes, nothing resolved, and checks b-d are NOT EVALUATED
      (no claims); the statistics refuse (no detection limit) or run only on flagged rows.
   With no resolved window at all, b-d are NOT EVALUATED: the run makes no claim.
Every dry run sets dry_run.overrides (card_leaf_shared_draws: true, A43) and uses the full
uncertainty.n_draws and dl_grid_points: the detection limit depends on n_draws (T6.1 run 1).
Air/RH scenarios (--air-rh, owner 2026-11-11): placeholder (0.5 C, 3 %), intermediate (0.3 C, 2.5 %),
target (0.2 C, 2 %) are added to the target sigmas; the default is placeholder (not chosen by which passes).
Sigmas: "target" = dry_run.target_sigmas (leaf 0.2 C, light 5 %) placed in the calibration,
labelled "if this calibration is achieved, not a prediction"; "placeholder" = config values.
synthetic.session_bias is on: each session gets one thermal (leaf and card share it), air,
humidity and light offset drawn from the sigmas (A38, A43).
--k-error e: the analysis uses k x (1 + e) for every card and leaf solve while the data were
made with the true k (one shared light-calibration error; spec open item). Reported, not asserted.
Outputs: data/processed/dry_run/pipeline_dry_run_{checks,results}_synthetic[tag].csv and
docs/figures/pipeline_dry_run_synthetic[tag].png (tag empty for the default run).
"""
from __future__ import annotations

import argparse
import copy
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from processing.config import load_config
from processing.data_io import CARD_COLUMNS_ORDER, ReferenceCard, Session

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent
LABEL_TARGET = "SYNTHETIC: if this calibration is achieved"           # completed by target_label()
AIR_RH_SCENARIOS = ("placeholder", "intermediate", "target", "relaxed")   # relaxed also sets the leaf sigma


def target_label(cfg: dict[str, Any]) -> str:
    """'SYNTHETIC: if this calibration is achieved (leaf .., light .., air .., RH ..), not a prediction'."""
    u = cfg["uncertainty"]
    return (f"{LABEL_TARGET} (leaf temperature {u['sigma_leaf_temp_c']} C, light {100 * u['sigma_lux_fraction']:.0f} %, "
            f"air {u['sigma_air_temp_c']} C, RH {u['sigma_rh_pct']} %), not a prediction")
LABEL_PLACEHOLDER = "SYNTHETIC: placeholder sigmas (leaf temperature 1.0 C), nothing is expected to resolve"
PASS, FAIL, NOT_EVAL, INFO = "PASS", "FAIL", "NOT EVALUATED", "INFO"
CHECK_NAMES = ("a_completes", "b_ml_rbh_beats_flat_on_atypical", "c_interaction_recovered",
               "d_censoring_effect_reported", "e_runtime", "g_placeholder_no_claims", "f_exit_code")


@dataclass
class DryRun:
    checks: pd.DataFrame
    results: pd.DataFrame
    runtime_s: float
    exit_code: int


def _merge(base: dict[str, Any], over: dict[str, Any]) -> None:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v


def dry_run_config(cfg: dict[str, Any], sigmas: str, settings: str, k_error: float,
                   air_rh: str = "placeholder") -> tuple[dict[str, Any], dict[str, Any]]:
    """(config copy, analysis calibration). The input cfg is not changed. With sigmas = target, the
    air/RH scenario dry_run.air_rh_scenarios[air_rh] is added to dry_run.target_sigmas."""
    from ml.synthetic_data import LIGHT_SOURCE, synthetic_calibration
    if air_rh not in AIR_RH_SCENARIOS:
        raise ValueError(f"air_rh must be one of {AIR_RH_SCENARIOS}, got {air_rh}")
    if sigmas not in ("target", "placeholder") or settings not in ("fast", "full"):
        raise ValueError(f"sigmas must be target or placeholder, settings fast or full; got {sigmas}, {settings}")
    c = copy.deepcopy(cfg)
    c["synthetic"]["session_bias"] = True
    _merge(c, copy.deepcopy(c["dry_run"]["overrides"]))
    if settings == "fast":
        _merge(c, copy.deepcopy(c["dry_run"]["fast"]))
    cal = synthetic_calibration(c)
    if sigmas == "target":
        chosen = {**c["dry_run"]["target_sigmas"], **c["dry_run"]["air_rh_scenarios"][air_rh]}
        cal["uncertainty"] = dict(chosen)
        c["uncertainty"].update(chosen)                               # the session offsets use these too
    cal["light_sources"][LIGHT_SOURCE]["k_wm2_per_lux"] = float(c["synthetic"]["k_wm2_per_lux"]) * (1.0 + k_error)
    return c, cal


def cards_dict(table: pd.DataFrame) -> dict[str, ReferenceCard]:
    out = {}
    for _, r in table.iterrows():
        vals = {k: r[k] for k in CARD_COLUMNS_ORDER}
        vals["photo_path"] = vals["photo_path"] or None
        out[r["card_id"]] = ReferenceCard(**vals)
    return out


def _session(sim) -> Session:
    return Session(sim.session_id, asdict(sim.metadata), sim.sensor_log, sim.events, sim.manifest, {}, True)


# ---------------------------------------------------------------- stages

def train_stage(cfg: dict[str, Any], cal: dict[str, Any], seed: int, cards: pd.DataFrame, truth: pd.DataFrame
                ) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Card sessions -> training set -> model. Returns (bundle, model card, training rows)."""
    from ml.build_training_set import build_training_set
    from ml.synthetic_data import simulate_card_session
    from ml.train_correction_model import train_correction_model
    d = cfg["dry_run"]
    sessions = []
    for i in range(len(cards)):
        for j, wind in enumerate(d["card_winds_ms"]):
            sim = simulate_card_session(cards.iloc[i], truth.iloc[i], cfg=cfg, seed=seed * 100_000 + 100 * i + j,
                                        conditions={"wind_ms": float(wind), "irradiance_wm2": float(d["card_irradiance_wm2"]),
                                                    "duration_min": float(d["leaf_duration_min"])},
                                        metadata={"plant_id": f"CARD{i:02d}W{j}"})
            sessions.append(_session(sim))
    ts = build_training_set(sessions, cards_dict(cards), cfg, cal)
    trained = train_correction_model(ts.kept, cfg)
    return trained.bundle, trained.card, len(ts.kept)


def leaf_stage(cfg: dict[str, Any], cal: dict[str, Any], seed: int, model: dict[str, Any] | None,
               train_cards: pd.DataFrame, train_truth: pd.DataFrame, leaf_cards: pd.DataFrame,
               leaf_truth: pd.DataFrame, only_atypical: bool = False) -> pd.DataFrame:
    """All leaf windows x methods with the planted truth and shape deviation attached.
    only_atypical: simulate only plants with an atypical leaf (same plants, same true g_s)."""
    from analysis.aggregate_results import session_results
    from analysis.baseline_benchmark import shape_deviation
    from ml.synthetic_data import simulate_leaf_session
    d = cfg["dry_run"]
    rng = np.random.default_rng(seed)
    card_dev = shape_deviation(train_cards[["solidity", "aspect_ratio"]].astype(float))
    k_card = int(np.argmin(card_dev.to_numpy()))         # the most typical card is the one in the frame
    leaf_dev = shape_deviation(leaf_cards[["solidity", "aspect_ratio"]].astype(float)).to_numpy()
    cards = cards_dict(pd.concat([train_cards.iloc[[k_card]], leaf_cards], ignore_index=True))
    genos = cfg["enumerations"]["genotype"]
    n_total = len(genos) * len(cfg["enumerations"]["water_treatment"]) * int(d["n_plants_per_group"])
    used = sorted({i % len(leaf_cards) for i in range(n_total)})
    threshold = float(np.quantile(leaf_dev[used], 2 / 3))     # top tertile of the leaf shapes used = atypical
    rows, n_leaf = [], 0
    for gi, geno in enumerate(genos):
        for trt in cfg["enumerations"]["water_treatment"]:
            base = float(d["true_gs_mol"][gi][trt])
            for p in range(int(d["n_plants_per_group"])):
                li = n_leaf % len(leaf_cards)
                g_true = base * max(1.0 + rng.normal(0, float(d["plant_sd_fraction"])), 0.1)
                plant = f"{geno[:2].upper()}-{trt[:2].upper()}-{p}"
                if only_atypical and leaf_dev[li] < threshold:
                    n_leaf += 1
                    continue
                sim = simulate_leaf_session(
                    g_true, leaf_cards.iloc[li], leaf_truth.iloc[li], card=train_cards.iloc[k_card],
                    card_truth=train_truth.iloc[k_card], cfg=cfg, seed=seed * 100_000 + 50_000 + n_leaf,
                    conditions={"wind_ms": float(d["leaf_wind_ms"]), "duration_min": float(d["leaf_duration_min"])},
                    plant_id=plant, metadata={"genotype": geno, "water_treatment": trt})
                for r in session_results(_session(sim), sim.photo_features, cards, cfg, cal, model):
                    r.update(g_s_true_mol=g_true, r_bh_true=float(sim.truth.get("r_bh_leaf_true", math.nan)),
                             leaf_card_id=leaf_cards.iloc[li]["card_id"],
                             shape_deviation=float(leaf_dev[li]))
                    rows.append(r)
                n_leaf += 1
    df = pd.DataFrame(rows)
    df["atypical"] = df["shape_deviation"] >= threshold
    return df


def campaign_detection_limit(df: pd.DataFrame, cfg: dict[str, Any], cal: dict[str, Any]
                             ) -> tuple[float | None, int, pd.DataFrame]:
    """(T5.4 campaign detection limit [mol m-2 s-1] or None, grid points with no forward root) at
    the run's representative conditions: the same steps as python -m analysis.detection_limit
    (the fallback T5.3 uses for blank row limits; lognormal r_bH, not the correlated draws)."""
    from analysis import detection_limit as dlm
    reading, rb_sigma, _ = dlm.representative_reading(df, cfg)
    sw = dlm.sweep(reading, cfg, cal, dlm.sweep_sigmas(cfg, cal, reading, rb_sigma))
    return dlm.resolved_band(sw)[0], int((~sw["root_found"].astype(bool)).sum()), sw


def rbh_rel_rmse(df: pd.DataFrame, method: str, atypical_only: bool) -> float:
    """sqrt(mean(((r_bH - r_bH,true) / r_bH,true)^2)) over valid uncensored windows (dimensionless)."""
    u = _uncensored(df[df["method"] == method])
    if atypical_only:
        u = u[u["atypical"]]
    rel = (u["r_bh"] - u["r_bh_true"]) / u["r_bh_true"]
    return float(np.sqrt((rel ** 2).mean())) if len(rel) else math.nan


# ---------------------------------------------------------------- checks

def _uncensored(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["valid"].astype(bool) & (df["censor"] == "none") & df["g_s_mol"].notna()]


def plant_rmse(df: pd.DataFrame, method: str, plants: set[str] | None = None) -> tuple[float, int]:
    """Plant-level RMSE [mol m-2 s-1]: plant mean of uncensored windows vs the plant's true g_s."""
    u = _uncensored(df[df["method"] == method])
    if plants is not None:
        u = u[u["plant_id"].isin(plants)]
    pm = u.groupby("plant_id").agg(dev=("g_s_mol", "mean"), true=("g_s_true_mol", "first"))
    if pm.empty:
        return math.nan, 0
    return float(np.sqrt(((pm["dev"] - pm["true"]) ** 2).mean())), len(pm)


def first_genotype_drop_uncensored(df: pd.DataFrame, cfg: dict[str, Any]) -> float | None:
    """pct drop of the first genotype using only uncensored valid windows (what dropping censored
    windows would give); None if its drought group has no uncensored window at all."""
    g0 = cfg["enumerations"]["genotype"][0]
    ref = cfg["genotype_comparison"]["reference_treatment"]
    u = _uncensored(df[(df["method"] == cfg["methods"]["primary"]) & (df["genotype"] == g0)])
    pm = u.groupby(["plant_id", "water_treatment"], as_index=False)["g_s_mol"].mean()
    ww = pm.loc[pm["water_treatment"] == ref, "g_s_mol"]
    dr = pm.loc[pm["water_treatment"] != ref, "g_s_mol"]
    if dr.empty or ww.empty:
        return None
    return 100.0 * (1.0 - dr.mean() / ww.mean())


def evaluate(df: pd.DataFrame, tables: dict[str, pd.DataFrame] | None, cfg: dict[str, Any], sigmas: str,
             settings: str, runtime_s: float, error: str | None, df_no_offsets: pd.DataFrame | None = None,
             dl_info: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Every check row (see module docstring)."""
    rows: list[dict[str, Any]] = []

    def add(check: str, status: str, value: Any = math.nan, detail: str = "") -> None:
        rows.append({"check": check, "status": status, "value": value, "detail": detail})

    d = cfg["dry_run"]
    primary = cfg["methods"]["primary"]
    genos = cfg["enumerations"]["genotype"]
    if error is not None:
        add("a_completes", FAIL, detail=error)
    else:
        methods = set(_uncensored(df)["method"])
        missing = {primary, "ml_corrected", "flat_plate"} - set(df["method"])
        add("a_completes", PASS if not missing else FAIL,
            detail=f"methods with uncensored rows: {sorted(methods)}" + (f"; missing {sorted(missing)}" if missing else ""))
    valid = df[df["valid"].astype(bool)] if len(df) else df
    n_resolved = int(valid["resolved"].astype(bool).sum()) if len(valid) else 0
    add("n_resolved_windows", INFO, n_resolved, f"of {len(valid)} valid rows")
    claims = error is None and n_resolved > 0

    if not claims:
        why = "pipeline failed" if error else "nothing resolved: no accuracy or genotype claim is made"
        for c in ("b_ml_rbh_beats_flat_on_atypical", "c_interaction_recovered", "d_censoring_effect_reported"):
            add(c, NOT_EVAL, detail=why)
    else:
        atyp = set(df.loc[df["atypical"], "plant_id"])
        b_ml, b_fp = rbh_rel_rmse(df, "ml_corrected", True), rbh_rel_rmse(df, "flat_plate", True)
        g_ml, n_ml = plant_rmse(df, "ml_corrected", atyp)
        g_fp, n_fp = plant_rmse(df, "flat_plate", atyp)
        text = (f"r_bH rel. RMSE atypical: ml_corrected {b_ml:.3f} vs flat_plate {b_fp:.3f}; g_s plant RMSE "
                f"atypical, offsets on: ml {g_ml:.4f} vs flat {g_fp:.4f} (n {n_ml}, {n_fp})")
        if df_no_offsets is not None and len(df_no_offsets):
            o_ml, _ = plant_rmse(df_no_offsets, "ml_corrected")
            o_fp, _ = plant_rmse(df_no_offsets, "flat_plate")
            text += f"; offsets off: ml {o_ml:.4f} vs flat {o_fp:.4f}"
            add("rmse_ml_corrected_atypical_offsets_off", INFO, o_ml, "g_s plant level, session offsets off")
            add("rmse_flat_plate_atypical_offsets_off", INFO, o_fp, "g_s plant level, session offsets off")
        add("b_ml_rbh_beats_flat_on_atypical", PASS if b_ml < b_fp else FAIL, b_ml - b_fp, text)
        for m in (primary, "ml_corrected", "flat_plate"):
            for subset, plants in (("all", None), ("atypical", atyp)):
                r, n = plant_rmse(df, m, plants)
                add(f"rmse_{m}_{subset}", INFO, r, f"g_s plant level, n {n}")
                add(f"rbh_rel_rmse_{m}_{subset}", INFO, rbh_rel_rmse(df, m, subset == "atypical"),
                    "r_bH relative RMSE vs true leaf r_bH, windows")

        e = tables["effects"] if tables else pd.DataFrame()
        sub = cfg["genotype_comparison"]["primary_substitution"]
        e = e[(e["outcome"] == primary) & (e["subset"] == "all") & (e["substitution"] == sub)] if len(e) else e
        if e.empty:
            notes = tables["notes"]["message"].tolist() if tables and len(tables["notes"]) else []
            add("c_interaction_recovered", FAIL, detail="no effect sizes: " + "; ".join(notes[:2]))
            add("d_censoring_effect_reported", FAIL, detail="no effect sizes")
        else:
            e = e.set_index("contrast")
            drop0 = float(e.loc[f"drop_under_drought:{genos[0]}", "pct_drop"])
            diff = e.loc["difference_of_drops"]
            excl = diff["ci_lo_pct"] > 0 or diff["ci_hi_pct"] < 0
            ok = excl and drop0 >= float(d["min_first_genotype_drop_pct"])
            add("c_interaction_recovered", PASS if ok else FAIL, float(diff["pct_drop"]),
                f"difference of drops {diff['pct_drop']:.1f} pct points, 95 % CI {diff['ci_lo_pct']:.1f} to "
                f"{diff['ci_hi_pct']:.1f}; {genos[0]} drop {drop0:.1f} % (need >= {d['min_first_genotype_drop_pct']})")
            for g in genos:
                r = e.loc[f"drop_under_drought:{g}"]
                add(f"drop_{g}", INFO, float(r["pct_drop"]), f"95 % CI {r['ci_lo_pct']:.1f} to {r['ci_hi_pct']:.1f}")
            # reworded after run 1: the change is reported (size and direction); its direction is not asserted
            dropped = first_genotype_drop_uncensored(df, cfg)
            if dropped is None:
                add("d_censoring_effect_reported", PASS, math.nan,
                    f"every {genos[0]} drought window is censored: dropping them loses the group entirely")
            else:
                change = dropped - drop0
                add("d_censoring_effect_reported", PASS, change,
                    f"{genos[0]} drop: censored kept ({sub}) {drop0:.1f} %, censored dropped {dropped:.1f} %; "
                    f"dropping changes it by {change:+.1f} pct points ({'larger' if change > 0 else 'smaller'})")
        cen = df[(df["method"] == primary) & (df["censor"] == "below_detection")]
        add("n_below_detection_primary", INFO, len(cen), "card_direct windows censored below_detection")
        for m in (primary, "ml_corrected", "flat_plate"):
            v = valid[valid["method"] == m]
            fl = v["flags"].astype(str)
            add(f"dl_rows_{m}", INFO, int(v["g_s_mol_dl"].notna().sum()),
                f"of {len(v)} valid rows have a detection limit; dl_not_resolved {int(fl.str.contains('dl_not_resolved').sum())}, "
                f"dl_search_failed {int(fl.str.contains('dl_search_failed').sum())}")

    for k, v in (dl_info or {}).items():
        add(k, INFO, v)
    if settings == "fast":
        add("e_runtime", PASS if runtime_s < float(d["max_runtime_s"]) else FAIL, runtime_s,
            f"limit {d['max_runtime_s']} s")
    else:
        add("e_runtime", NOT_EVAL, runtime_s, "full settings: runtime reported, not limited")

    if sigmas == "placeholder":
        notes = tables["notes"]["message"] if tables and len(tables["notes"]) else pd.Series(dtype=str)
        refused = bool(notes.str.contains("no detection limit").any())
        flagged = bool(len(valid)) and valid["flags"].astype(str).str.contains("dl_not_resolved|dl_search_failed").all()
        ok = error is None and n_resolved == 0 and (refused or flagged)
        stats = "statistics refused (no detection limit)" if refused else \
            "statistics ran only on rows flagged dl_not_resolved / dl_search_failed"
        add("g_placeholder_no_claims", PASS if ok else FAIL, n_resolved,
            f"nothing resolved ({n_resolved} resolved rows); checks b-d not evaluated; {stats}")
    else:
        add("g_placeholder_no_claims", NOT_EVAL, detail="placeholder-sigma configuration only")
    failed = any(r["status"] == FAIL for r in rows)
    add("f_exit_code", FAIL if failed else PASS, 1 if failed else 0)
    return rows


# ---------------------------------------------------------------- ordering diagnostic (never a check)

DIAG = "DIAG"
NULL_GS_MOL, NULL_PLANT_SD, NULL_WINDOW_SD = 0.40, 0.03, 0.03     # no-effect frames: invented, mol m-2 s-1
NULL_PLANTS, NULL_WINDOWS = 5, 6


def ordering_diagnostic(results: pd.DataFrame, cfg: dict[str, Any], rng_seed: int = 0) -> pd.DataFrame:
    """Do the rank tests and the share-below-detection table recover the planted ordering?
    Definition: decision_log 2026-11-11 (T6.1 diagnostic). card_direct, valid windows only.
    Rows: item, status DIAG, value, detail. Reported only; never PASS/FAIL."""
    from analysis import genotype_comparison as gc
    oc = next(o for o in gc.outcomes(cfg) if o.role == "primary")
    w, _ = gc.prepare(results, oc, campaign_dl=None)
    rows: list[dict[str, Any]] = []

    def add(item: str, value: Any, detail: str = "") -> None:
        rows.append({"item": item, "status": DIAG, "value": value, "detail": detail})

    if w.empty:
        add("first_genotype_drought_lowest", "no data", "no valid card_direct windows")
        return pd.DataFrame(rows)
    measured_min = w["value"].min()                   # smallest measured g_s; tie values sit outside it
    w = w.assign(o=gc.ordinal_values(w))
    pm = w.groupby("plant_id").agg(genotype=("genotype", "first"), water_treatment=("water_treatment", "first"),
                                   med=("o", "median"))
    gv = pm.groupby(["genotype", "water_treatment"])["med"].median()
    share = gc.censor_share(w).set_index(["genotype", "water_treatment"])["share_below_detection"]
    genos = cfg["enumerations"]["genotype"]
    ref = cfg["genotype_comparison"]["reference_treatment"]
    other = next(t for t in cfg["enumerations"]["water_treatment"] if t != ref)
    key = (genos[0], other)

    def show(k: tuple[str, str]) -> str:
        v = gv.get(k, math.nan)
        return "below-detection tie" if v < measured_min else f"{v:.4f}"

    for k in gv.index:
        add(f"group_value:{k[0]}:{k[1]}", show(k), f"median of plant medians; share below detection {share.get(k, 0):.2f}")
    others = [k for k in gv.index if k != key]
    if key not in gv.index or not others:
        add("first_genotype_drought_lowest", "no data", "group missing")
    else:
        tie = any(gv[k] == gv[key] for k in others)
        lowest = all(gv[key] < gv[k] for k in others)
        share_top = share.get(key, 0) > 0 and all(share.get(key, 0) >= share.get(k, 0) for k in others)
        why = []
        if tie:
            why.append("tie with another group's value")
        elif not lowest:
            why.append("not the lowest group value")
        if not share_top:
            why.append("share below detection not the highest or zero")
        add("first_genotype_drought_lowest", "yes" if (lowest and share_top) else "no",
            "; ".join(why) if why else f"value {show(key)}, share {share.get(key, 0):.2f}")
    if len(genos) > 1 and (genos[1], other) in gv.index and (genos[1], ref) in gv.index:
        dr, ww = gv[(genos[1], other)], gv[(genos[1], ref)]
        if dr < measured_min or ww < measured_min:
            add("second_genotype_reduction_pct", "not computable", "a group value is set by below-detection windows")
        else:
            add("second_genotype_reduction_pct", round(100.0 * (1.0 - dr / ww), 1), "planted about 40 %")
    rt = gc.rank_tests(w.drop(columns="o"), cfg, np.random.default_rng(rng_seed))
    from scipy.stats import rankdata
    for g, sub in pm.groupby("genotype"):
        r = rankdata(sub["med"])
        is_dr = (sub["water_treatment"] != ref).to_numpy()
        lower = r[is_dr].mean() < r[~is_dr].mean() if is_dr.any() and (~is_dr).any() else None
        p = rt.loc[rt["test"] == f"mann_whitney:{g}", "p_value"]
        add(f"mann_whitney_p:{g}", float(p.iloc[0]) if len(p) else math.nan,
            "exact; " + ("drought lower" if lower else "drought not lower" if lower is not None else "one group missing"))
    kw = rt.loc[rt["test"] == "kruskal_wallis:4_groups", "p_value"]
    add("kruskal_wallis_p", float(kw.iloc[0]) if len(kw) else math.nan, "asymptotic, four groups")
    return pd.DataFrame(rows)


def append_diagnostic(checks: pd.DataFrame, results: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """checks plus DIAG rows (column check = item). The checks and the exit code are not touched."""
    d = ordering_diagnostic(results, cfg).rename(columns={"item": "check"})
    return pd.concat([checks, d], ignore_index=True)


def null_rank_rates(cfg: dict[str, Any], n_frames: int = 200, seed: int = 0) -> dict[str, float]:
    """No-effect check: n_frames synthetic card_direct tables with four IDENTICAL groups (NULL_PLANTS
    plants x NULL_WINDOWS windows, true g_s NULL_GS_MOL with plant and window noise); share of frames
    with p < 0.05 for each rank test. Expected near 0.05 (exact Mann-Whitney is discrete). Reported only."""
    from analysis import genotype_comparison as gc
    oc = next(o for o in gc.outcomes(cfg) if o.role == "primary")
    rng = np.random.default_rng(seed)
    genos, trts = cfg["enumerations"]["genotype"], cfg["enumerations"]["water_treatment"]
    hits: dict[str, int] = {}
    for _ in range(n_frames):
        rows = []
        for g in genos:
            for t in trts:
                for p in range(NULL_PLANTS):
                    mu = NULL_GS_MOL + rng.normal(0, NULL_PLANT_SD)
                    for _w in range(NULL_WINDOWS):
                        rows.append({"plant_id": f"{g}-{t}-{p}", "genotype": g, "water_treatment": t, "block": "",
                                     "method": oc.method, "valid": True, "reason": "ok", "censor": "none",
                                     "g_s_mol": mu + rng.normal(0, NULL_WINDOW_SD), "g_s_mol_dl": math.nan,
                                     "convection_regime": "forced"})
        w, _ = gc.prepare(pd.DataFrame(rows), oc, campaign_dl=None)
        for _, r in gc.rank_tests(w, cfg, rng).iterrows():
            name = "kruskal_wallis" if r["test"].startswith("kruskal") else r["test"]
            hits[name] = hits.get(name, 0) + int(r["p_value"] < 0.05)
    return {"n_frames": n_frames, **{k: v / n_frames for k, v in hits.items()}}


# ---------------------------------------------------------------- figure

def plot(df: pd.DataFrame, cfg: dict[str, Any], path: Path, label: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, faint, ref = "#1f4e79", "#b8b8b8", "#555555"
    methods = [cfg["methods"]["primary"], "ml_corrected", "flat_plate"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharex=True, sharey=True)
    u = _uncensored(df) if len(df) else df
    hi = float(max(df["g_s_true_mol"].max(), u["g_s_mol"].max() if len(u) else 0)) * 1.1 if len(df) else 1.0
    for ax, m in zip(axes, methods):
        um = u[u["method"] == m] if len(u) else u
        for atyp, color, name in ((False, faint, "typical shape"), (True, ink, "atypical shape (top tertile)")):
            s = um[um["atypical"] == atyp] if len(um) else um
            ax.scatter(s["g_s_true_mol"], s["g_s_mol"], s=12, color=color, label=name, zorder=3 if atyp else 2)
        ax.plot([0, hi], [0, hi], color=ref, linestyle="--", linewidth=1, label="1:1")
        n_cen = int(((df["method"] == m) & (df["censor"] == "below_detection")).sum())
        ax.set_title(f"{m} ({n_cen} windows below detection, not shown)", fontsize=9)
        ax.set_xlabel("planted true g_s (mol m$^{-2}$ s$^{-1}$)")
        ax.grid(color="#e6e6e6", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("device g_s, window (mol m$^{-2}$ s$^{-1}$)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Pipeline dry run: {label}", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- run and command line

def _tag(sigmas: str, settings: str, seed: int, k_error: float, air_rh: str = "placeholder") -> str:
    if (sigmas, settings, seed, k_error, air_rh) == ("target", "fast", 0, 0.0, "placeholder"):
        return ""
    air = f"_airrh-{air_rh}" if sigmas == "target" else ""
    return f"_{sigmas}{air}_{settings}_seed{seed}" + (f"_k{k_error:+.2f}" if k_error else "")


def run(cfg: dict[str, Any], sigmas: str, settings: str, seed: int, k_error: float, out_dir: Path,
        figure_dir: Path, air_rh: str = "placeholder") -> DryRun:
    """The whole dry run; writes the checks and results CSVs and the figure."""
    from analysis import genotype_comparison as gc
    from ml.synthetic_data import make_card_sets
    t0 = time.perf_counter()
    c, cal = dry_run_config(cfg, sigmas, settings, k_error, air_rh)
    label = target_label(c) if sigmas == "target" else LABEL_PLACEHOLDER
    df, tables, error, info, df_off, dl_info, sweep = pd.DataFrame(), None, None, {}, None, {}, None
    try:
        sets = make_card_sets(c)
        (train_cards, train_truth), (leaf_cards, leaf_truth) = sets["train"], sets["test"]
        bundle, card, n_rows = train_stage(c, cal, seed, train_cards, train_truth)
        info = {"model_status": card["status"], "model_chosen": card["chosen_model"], "n_training_rows": n_rows}
        df = leaf_stage(c, cal, seed, bundle, train_cards, train_truth, leaf_cards, leaf_truth)
        campaign_dl, no_root, sweep = campaign_detection_limit(df, c, cal)
        cd = df[(df["method"] == c["methods"]["primary"]) & df["valid"].astype(bool)]
        dl_info = {"campaign_dl_mol": math.nan if campaign_dl is None else campaign_dl,
                   "campaign_dl_grid_no_root": no_root,
                   "campaign_min_relative_width": float(sweep["relative_width"].min()),
                   "card_direct_median_row_dl_mol": float(cd["g_s_mol_dl"].median()),
                   "card_direct_share_rows_with_dl": float(cd["g_s_mol_dl"].notna().mean()) if len(cd) else math.nan}
        tables = gc.run(df, c, campaign_dl=campaign_dl, label=label)
        # check b also reports g_s with the per-session offsets off (atypical plants re-simulated)
        c_off = copy.deepcopy(c)
        c_off["synthetic"]["session_bias"] = False
        df_off = leaf_stage(c_off, cal, seed, bundle, train_cards, train_truth, leaf_cards, leaf_truth,
                            only_atypical=True)
    except Exception as e:                    # noqa: BLE001 - a dry run reports the failure as check a
        logger.exception("dry run failed")
        error = f"{type(e).__name__}: {e}"
    runtime = time.perf_counter() - t0
    rows = evaluate(df, tables, c, sigmas, settings, runtime, error, df_off, dl_info)
    rows += [{"check": k, "status": INFO, "value": v, "detail": ""} for k, v in info.items()]
    checks = pd.DataFrame(rows)
    if len(df):
        checks = append_diagnostic(checks, df, c)       # DIAG rows: reported only, never a check
    for col, v in (("seed", seed), ("sigmas", sigmas), ("air_rh", air_rh if sigmas == "target" else "placeholder"),
                   ("settings", settings), ("k_error", k_error), ("data_label", label)):
        checks[col] = v
    tag = _tag(sigmas, settings, seed, k_error, air_rh)
    out_dir.mkdir(parents=True, exist_ok=True)
    if sweep is not None:      # width vs true g_s at the campaign conditions (threshold review, open item)
        sweep.assign(data_label=label).to_csv(out_dir / f"pipeline_dry_run_dl_sweep_synthetic{tag}.csv", index=False)
    checks.to_csv(out_dir / f"pipeline_dry_run_checks_synthetic{tag}.csv", index=False)
    if len(df):
        df.assign(data_label=label).to_csv(out_dir / f"pipeline_dry_run_results_synthetic{tag}.csv", index=False)
        plot(df, c, figure_dir / f"pipeline_dry_run_synthetic{tag}.png", label)
    exit_code = int(checks.loc[checks["check"] == "f_exit_code", "value"].iloc[0])
    return DryRun(checks, df, runtime, exit_code)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m processing.pipeline_dry_run",
                                description="End-to-end dry run on synthetic data (T6.1).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigmas", choices=("target", "placeholder"), default="target")
    p.add_argument("--settings", choices=("fast", "full"), default="fast")
    p.add_argument("--k-error", type=float, default=0.0, help="shared fractional error in k (e.g. 0.10)")
    p.add_argument("--air-rh", choices=AIR_RH_SCENARIOS, default="placeholder",
                   help="air-temperature / RH sigma scenario added to the target sigmas (dry_run.air_rh_scenarios)")
    p.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "processed" / "dry_run"))
    p.add_argument("--figure-dir", default=str(REPO_ROOT / "docs" / "figures"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    res = run(load_config(args.config), args.sigmas, args.settings, args.seed, args.k_error,
              Path(args.out_dir), Path(args.figure_dir), args.air_rh)
    print(res.checks["data_label"].iloc[0])
    print(f"seed {args.seed}, sigmas {args.sigmas}, air_rh {args.air_rh}, settings {args.settings}, k_error {args.k_error}, "
          f"runtime {res.runtime_s:.1f} s")
    with pd.option_context("display.max_colwidth", 140, "display.width", 220):
        print(res.checks[["check", "status", "value", "detail"]].to_string(index=False))
    print(f"exit code {res.exit_code}")
    return res.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
