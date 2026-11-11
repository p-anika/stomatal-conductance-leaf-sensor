"""Does choosing card windows on their measured result bias card_direct g_s? (decision_log 2026-11-11)

    python -m analysis.card_selection_bias

Synthetic experiment, design fixed in the decision log BEFORE computing. A shared thermal offset
delta (one camera) is planted on BOTH leaf_temp_c and card_temp_c of a session. The same sessions
are analysed two ways:
    A_nearest_window   old rule: nearest VALID card window within card_match_minutes (a card window
                       with delta T <= 0 is skipped, a leaf row with no valid card window is invalid)
    C_session_pooled   new rule (option C): one card r_bH per session,
                       r_bH = rho cp sum(T_card - T_air) / sum(Rn_card) over all steady, light-gated
                       card windows including delta T <= 0; a session with pooled r_bH <= 0 keeps its
                       rows, labelled suspect_card_rbh
Designs: R = random delta ~ N(0, sigma) per session; F = one fixed delta for every session.
Metrics (REPORTED, not thresholded), card_direct, steady windows:
    bias_mol      mean over plants of (plant mean of valid uncensored g_s - true g_s)   [mol m-2 s-1]
    invalid_share share of steady card_direct rows that are invalid
    spearman_invalid_vs_minus_delta  (R) do sessions reading low lose more rows?
    group_diff_mol  plant-mean group 2 - group 1 (truth 0 unless a planted design)
    drop_pct      100 x (1 - group 2 / group 1) of plant means
    bias_e_mol    option E sensitivity: flat_plate g_s substituted for invalid card_direct rows
Settings: config card_selection_bias (invented); reduced Monte Carlo draws because only point
estimates are measured. Output: data/processed/card_selection_bias_synthetic.csv (SYNTHETIC).
"""
from __future__ import annotations

import argparse
import copy
import logging
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis import aggregate_results as ar
from processing.config import load_config
from processing.data_io import Session

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent
METHODS = {"A_nearest_window": "nearest_window", "C_session_pooled": "session_pooled"}
LABEL = "SYNTHETIC: planted shared thermal offsets, not a prediction"


def _shapes(cfg: dict[str, Any]):
    """(card row, card truth, leaf row, leaf truth, cards dict): the two most typical synthetic shapes."""
    from analysis.baseline_benchmark import shape_deviation
    from ml.synthetic_data import make_synthetic_cards
    from processing.pipeline_dry_run import cards_dict
    cards, truth = make_synthetic_cards(20, seed=int(cfg["card_selection_bias"]["shape_seed"]), cfg=cfg)
    order = np.argsort(shape_deviation(cards[["solidity", "aspect_ratio"]].astype(float)).to_numpy())
    k, l = int(order[0]), int(order[1])
    return cards.iloc[k], truth.iloc[k], cards.iloc[l], truth.iloc[l], cards_dict(cards.iloc[[k, l]])


def simulate_session(cfg: dict[str, Any], g_s_mol: float, delta_c: float, irradiance_wm2: float, wind_ms: float,
                     seed: int, noise: bool, plant_id: str, metadata: dict[str, Any]):
    """One leaf session with the typical card in the frame and the shared offset delta_c [deg C]
    added to both leaf_temp_c and card_temp_c. Returns (simulated session, cards dict)."""
    from ml.synthetic_data import simulate_leaf_session
    card, card_t, leaf, leaf_t, cards = _shapes(cfg)
    c = copy.deepcopy(cfg)
    c["synthetic"]["session_bias"] = False
    sim = simulate_leaf_session(g_s_mol, leaf, leaf_t, card=card, card_truth=card_t, cfg=c, seed=seed, noise=noise,
                                conditions={"irradiance_wm2": irradiance_wm2, "wind_ms": wind_ms,
                                            "duration_min": float(cfg["card_selection_bias"]["duration_min"])},
                                plant_id=plant_id, metadata=metadata)
    log = sim.sensor_log.copy()
    for col in ("leaf_temp_c", "card_temp_c"):
        log[col] = log[col] + delta_c              # ONE camera: the same offset on leaf and card
    return replace(sim, sensor_log=log), cards


def _metrics(rows: pd.DataFrame) -> dict[str, float]:
    cd = rows[(rows["method"] == "card_direct") & rows["steady_state"].astype(bool)]
    fp = rows[(rows["method"] == "flat_plate") & rows["steady_state"].astype(bool)]
    ok = cd["valid"].astype(bool) & (cd["censor"] == "none") & cd["g_s_mol"].notna()
    pm = cd[ok].groupby("plant_id").agg(g=("g_s_mol", "mean"), t=("g_true", "first"), grp=("group", "first"))
    out = {"bias_mol": float((pm["g"] - pm["t"]).mean()) if len(pm) else math.nan,
           "bias_pct": float(100 * (pm["g"] - pm["t"]).mean() / pm["t"].mean()) if len(pm) else math.nan,
           "invalid_share": float((~cd["valid"].astype(bool)).mean()) if len(cd) else math.nan,
           "n_plants_lost": int(cd["plant_id"].nunique() - len(pm))}
    g = pm.groupby("grp")["g"].mean()
    out["group_diff_mol"] = float(g.get("G2", math.nan) - g.get("G1", math.nan))
    out["drop_pct"] = float(100 * (1 - g.get("G2", math.nan) / g.get("G1", math.nan))) if "G1" in g else math.nan
    per_s = cd.groupby("plant_id").agg(inv=("valid", lambda v: float((~v.astype(bool)).mean())),
                                       d=("delta_c", "first"))
    if per_s["inv"].nunique() > 1 and per_s["d"].nunique() > 1:
        from scipy.stats import spearmanr
        out["spearman_invalid_vs_minus_delta"] = float(spearmanr(per_s["inv"], -per_s["d"]).statistic)
    else:
        out["spearman_invalid_vs_minus_delta"] = math.nan
    # option E (sensitivity): flat_plate g_s replaces an invalid card_direct row, same window
    key = ["plant_id", "window_start_iso"]
    e = cd.merge(fp[key + ["g_s_mol", "valid", "censor"]], on=key, suffixes=("", "_fp"))
    use_fp = ~e["valid"].astype(bool) & e["valid_fp"].astype(bool) & (e["censor_fp"] == "none")
    e["g_e"] = np.where(ok.reindex(e.index, fill_value=False).to_numpy() & e["valid"].astype(bool), e["g_s_mol"],
                        np.where(use_fp, e["g_s_mol_fp"], np.nan))
    pe = e.dropna(subset=["g_e"]).groupby("plant_id").agg(g=("g_e", "mean"), t=("g_true", "first"))
    out["bias_e_mol"] = float((pe["g"] - pe["t"]).mean()) if len(pe) else math.nan
    return out


def run_design(cfg: dict[str, Any], design: str, sigma_c: float, irradiance_wm2: float, n_plants: int, seed: int,
               fixed_delta_c: float | None = None, group_irradiance: tuple[float, float] | None = None,
               group_gs: tuple[float, float] | None = None, name: str | None = None) -> pd.DataFrame:
    """One design, both methods on the SAME simulated sessions. One row per method."""
    from ml.synthetic_data import synthetic_calibration
    x = cfg["card_selection_bias"]
    c = copy.deepcopy(cfg)
    c["uncertainty"].update(n_draws=int(x["n_draws"]), dl_grid_points=int(x["dl_grid_points"]))
    cal = synthetic_calibration(c)
    rng = np.random.default_rng(seed)
    irr = group_irradiance or (irradiance_wm2, irradiance_wm2)
    gs = group_gs or (float(x["g_s_mol"]), float(x["g_s_mol"]))
    sessions = []
    for gi, grp in enumerate(("G1", "G2")):
        for p in range(n_plants):
            delta = float(fixed_delta_c) if design == "fixed" else float(rng.normal(0.0, sigma_c))
            sim, cards = simulate_session(c, gs[gi], delta, irr[gi], float(x["wind_ms"]), seed * 1000 + gi * 100 + p,
                                          True, f"{grp}-{p}", {})
            sessions.append((sim, cards, grp, delta, gs[gi]))
    out = []
    for label, source in METHODS.items():
        cm = copy.deepcopy(c)
        cm["windows"]["card_source"] = source
        rows = []
        for sim, cards, grp, delta, g_true in sessions:
            s = Session(sim.session_id, asdict(sim.metadata), sim.sensor_log, sim.events, sim.manifest, {}, True)
            for r in ar.session_results(s, sim.photo_features, cards, cm, cal, None):
                r.update(group=grp, delta_c=delta, g_true=g_true)
                rows.append(r)
        out.append({"design": name or design, "method": label, "sigma_c": sigma_c,
                    "fixed_delta_c": fixed_delta_c, "irradiance_wm2": irradiance_wm2,
                    **_metrics(pd.DataFrame(rows))})
    return pd.DataFrame(out)


def run_all(cfg: dict[str, Any], seed: int = 0) -> pd.DataFrame:
    """Every design in the decision-log row."""
    x = cfg["card_selection_bias"]
    n, low, high = int(x["n_plants"]), float(x["low_irradiance_wm2"]), float(x["high_irradiance_wm2"])
    parts = []
    for sigma in x["sigmas_c"]:
        parts.append(run_design(cfg, "random", float(sigma), low, n, seed))
        for d in (-float(sigma), float(sigma)):
            parts.append(run_design(cfg, "fixed", float(sigma), low, n, seed, fixed_delta_c=d))
    parts.append(run_design(cfg, "random", 0.5, low, n, seed, group_irradiance=(low, high), name="light_difference"))
    parts.append(run_design(cfg, "random", 0.5, low, n, seed,
                            group_gs=(float(x["g_s_mol"]), float(x["g_s_drought_mol"])), name="planted_effect"))
    df = pd.concat(parts, ignore_index=True)
    df["data_label"] = LABEL
    return df


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m analysis.card_selection_bias",
                                description="Synthetic test of card-window selection bias (A vs C).")
    p.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(REPO_ROOT / "data" / "processed" / "card_selection_bias_synthetic.csv"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    df = run_all(load_config(args.config), args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    with pd.option_context("display.width", 250):
        print(df.drop(columns="data_label").round(4).to_string(index=False))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
