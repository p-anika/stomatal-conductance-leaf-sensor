"""Boundary-layer check with a petroleum-jelly (vaseline) coated leaf (T5.5, change E3).

    python -m analysis.validate_rb_vaseline [--synthetic]

A coated leaf barely transpires, so (like a dry card) lambdaE ~ 0 and its heat resistance can be
solved from its temperature (Appendix A eq. 9):
    r_bH,solved = rho cp (T_coated - T_air) / Rn_coated        [s m-1]
    Rn_coated = a_s,leaf x c_card x E_s + n x e_leaf x sigma x (T_sur^4 - T_coated^4)   (eq. 2)
That r_bH is compared with the two predictions for the same leaf shape and wind:
    flat_plate:   r_bH,flat = 1 / g_bH(d, U, dT)               (eq. 10)
    ml_corrected: r_bH,ml = f_hat(geometry, U) x r_bH,flat     (eq. 11)
ratio = predicted / solved (1 = perfect), difference = predicted - solved [s m-1].

Inputs and choices:
- sessions with dry_reference_type = vaseline_leaf; the coated leaf sits in the card position, so
  its temperature is card_temp_c (as in T5.1) and its irradiance factor is c_card;
- its geometry is the reference_cards.csv row named by the session's card_id (source traced_leaf,
  coating vaseline);
- it is solved like a card (ml.build_training_set.session_rows: same exogenous gates: steady
  state, wind present, absorbed shortwave >= physics.min_card_absorbed_sw_wm2), but with LEAF
  optics: a_s = light_sources.<source>.a_s_leaf, emissivity = physics.leaf_emissivity_longwave.
  The coating may change both; every row is flagged coated_leaf_optics_assumed.
- no correction model -> ml_corrected blank, flag ml_model_missing.
Statistics are case-level (one coated leaf = one session = one case: mean over its windows);
window-level numbers are secondary. Descriptive only: no p-values.
Outputs: data/processed/vaseline_rb_{windows,cases,summary}[_synthetic].csv (data_label column).
"""
from __future__ import annotations

import argparse
import copy
import logging
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ml.build_training_set import session_rows
from ml.predict_correction import predict_rbh
from processing.config import load_calibration, load_config
from processing.data_io import ReferenceCard, Session, list_sessions, load_reference_cards, load_session

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent

VASELINE = "vaseline_leaf"
METHODS = ("flat_plate", "ml_corrected")
LEVEL_CASE = "coated leaf (primary)"
LEVEL_WINDOW = "window (secondary, non-independent)"
OPTICS_FLAG = "coated_leaf_optics_assumed"


def leaf_optics(cfg: dict[str, Any], cal: dict[str, Any], light_source: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copies of cfg and cal whose card a_s and emissivity are the leaf's (a_s_leaf of this light
    source, physics.leaf_emissivity_longwave). The inputs are not changed."""
    cfg2, cal2 = copy.deepcopy(cfg), copy.deepcopy(cal)
    a_s = float(cfg["light_sources"][light_source]["a_s_leaf"])
    e_leaf = float(cfg["physics"]["leaf_emissivity_longwave"])
    cfg2["card"]["a_s_card"], cfg2["card"]["emissivity"] = a_s, e_leaf
    cal2.setdefault("card", {})
    cal2["card"]["a_s_card"], cal2["card"]["emissivity"] = a_s, e_leaf
    return cfg2, cal2


def _add_flag(flags: str, flag: str) -> str:
    parts = [f for f in str(flags or "").split(";") if f]
    return ";".join(dict.fromkeys(parts + [flag]))


def session_windows(session: Session, cards: Mapping[str, ReferenceCard], cfg: dict[str, Any],
                    cal: dict[str, Any], model: dict[str, Any] | None) -> pd.DataFrame:
    """One row per window of a vaseline_leaf session: r_bh_solved, r_bh_flat_plate, r_bh_ml_corrected."""
    meta = session.metadata
    cfg2, cal2 = leaf_optics(cfg, cal, str(meta.get("light_source", "")).strip())
    kept, rejected = session_rows(session, dict(cards), cfg2, cal2)
    rows = []
    d_key = f"{cfg['physics']['characteristic_dimension']}_mm"
    for r in kept + rejected:
        out = {"session_id": session.session_id, "plant_id": meta.get("plant_id", ""), "card_id": r.get("card_id"),
               "window_start_iso": r.get("window_start_iso"), "wind_ms": r.get("wind_ms"),
               "delta_t_c": r.get("card_temp_c", math.nan) - r.get("air_temp_c", math.nan),
               "convection_regime": r.get("convection_regime", ""), "valid": bool(r.get("valid")),
               "reason": r.get("reason"), "flags": _add_flag(r.get("flags", ""), OPTICS_FLAG),
               "r_bh_solved": r.get("r_bh_card", math.nan) if r.get("valid") else math.nan,
               "r_bh_flat_plate": r.get("r_bh_flat", math.nan) if r.get("valid") else math.nan,
               "r_bh_ml_corrected": math.nan, d_key: r.get(d_key, math.nan)}
        if out["valid"]:
            if model is None:
                out["flags"] = _add_flag(out["flags"], "ml_model_missing")
            else:
                feats = {k: r.get(k) for k in model["features"] if k in r}
                pred = predict_rbh(feats, r["wind_ms"], out["delta_t_c"], r["air_temp_c"], cfg, model)
                if pred.valid:
                    out["r_bh_ml_corrected"] = pred.r_bh
                for f in pred.flags:
                    out["flags"] = _add_flag(out["flags"], f)
        rows.append(out)
    return pd.DataFrame(rows)


def _agreement(pred: np.ndarray, solved: np.ndarray) -> dict[str, float]:
    """difference = pred - solved [s m-1]; ratio = pred / solved."""
    ok = np.isfinite(pred) & np.isfinite(solved)
    p, s = pred[ok], solved[ok]
    if not len(p):
        return {"n": 0, "bias_s_m": math.nan, "mae_s_m": math.nan, "rmse_s_m": math.nan, "mean_ratio": math.nan,
                "min_ratio": math.nan, "max_ratio": math.nan}
    d, ratio = p - s, p / s
    return {"n": int(len(p)), "bias_s_m": float(d.mean()), "mae_s_m": float(np.abs(d).mean()),
            "rmse_s_m": float(math.sqrt((d ** 2).mean())), "mean_ratio": float(ratio.mean()),
            "min_ratio": float(ratio.min()), "max_ratio": float(ratio.max())}


def case_table(windows: pd.DataFrame) -> pd.DataFrame:
    """One row per coated leaf (session): mean r_bH per source over its valid windows, ratios."""
    v = windows[windows["valid"]]
    cols = ["r_bh_solved", *[f"r_bh_{m}" for m in METHODS]]
    if v.empty:
        return pd.DataFrame(columns=["session_id", "plant_id", "card_id", "n_windows", *cols,
                                     *[f"ratio_{m}" for m in METHODS]])
    c = v.groupby(["session_id", "plant_id", "card_id"], as_index=False).agg(
        n_windows=("r_bh_solved", "size"), wind_ms=("wind_ms", "median"), delta_t_c=("delta_t_c", "mean"),
        **{k: (k, "mean") for k in cols})
    for m in METHODS:
        c[f"ratio_{m}"] = c[f"r_bh_{m}"] / c["r_bh_solved"]
    return c


def summary_table(windows: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in METHODS:
        for level, frame in ((LEVEL_CASE, cases), (LEVEL_WINDOW, windows[windows["valid"]] if len(windows) else windows)):
            pred = frame[f"r_bh_{m}"].to_numpy(float) if len(frame) else np.array([])
            solved = frame["r_bh_solved"].to_numpy(float) if len(frame) else np.array([])
            rows.append({"method": m, "level": level, **_agreement(pred, solved)})
    return pd.DataFrame(rows)


def run(sessions: list[Session], cards: Mapping[str, ReferenceCard], cfg: dict[str, Any], cal: dict[str, Any],
        model: dict[str, Any] | None, label: str) -> dict[str, pd.DataFrame]:
    """Tables for every vaseline_leaf session in sessions."""
    chosen = [s for s in sessions if str(s.metadata.get("dry_reference_type", "")).strip() == VASELINE]
    frames = [session_windows(s, cards, cfg, cal, model) for s in chosen]
    windows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["session_id", "plant_id", "card_id", "valid", "r_bh_solved", *[f"r_bh_{m}" for m in METHODS], "flags"])
    if len(windows):
        windows["valid"] = windows["valid"].astype(bool)
    cases = case_table(windows)
    out = {"windows": windows, "cases": cases, "summary": summary_table(windows, cases)}
    for df in out.values():
        df["data_label"] = label
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m analysis.validate_rb_vaseline",
                                description="Coated-leaf r_bH vs flat-plate and ML predictions (T5.5).")
    p.add_argument("--synthetic", action="store_true", help="use only synthetic sessions and the synthetic "
                   "calibration; stamp SYNTHETIC on every output")
    p.add_argument("--raw-dir", default=str(REPO_ROOT / "data" / "raw"))
    p.add_argument("--cards", default=str(REPO_ROOT / "data" / "reference_cards.csv"))
    p.add_argument("--calibration", default=str(REPO_ROOT / "calibration" / "calibration.json"))
    p.add_argument("--model", default=str(REPO_ROOT / "data" / "models" / "correction_model.joblib"))
    p.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "processed"))
    return p


def main(argv: list[str] | None = None) -> int:
    from ml.predict_correction import load_model
    from ml.synthetic_data import synthetic_calibration

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    label, suffix = ("SYNTHETIC", "_synthetic") if args.synthetic else ("real", "")
    raw = Path(args.raw_dir)
    sessions = [load_session(s, raw) for s in list_sessions(raw)] if raw.is_dir() else []
    sessions = [s for s in sessions if s.synthetic == args.synthetic
                and str(s.metadata.get("dry_reference_type", "")).strip() == VASELINE]
    if not sessions:
        raise SystemExit(f"no {label} {VASELINE} sessions under {raw}")
    cal = synthetic_calibration(cfg) if args.synthetic else load_calibration(args.calibration)
    cards = load_reference_cards(args.cards) if Path(args.cards).is_file() else {}
    model = load_model(args.model) if Path(args.model).is_file() else None
    out = run(sessions, cards, cfg, cal, model, label)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df.to_csv(out_dir / f"vaseline_rb_{name}{suffix}.csv", index=False)
    s = out["summary"]
    print(f"{label} data: {len(out['cases'])} coated leaves")
    print(s[s["level"] == LEVEL_CASE][["method", "n", "bias_s_m", "rmse_s_m", "mean_ratio"]].to_string(index=False))
    print(f"wrote {out_dir} (vaseline_rb_*{suffix}.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
