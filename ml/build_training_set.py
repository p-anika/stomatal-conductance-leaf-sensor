"""Build the card training set for the r_bH correction model (T4.3, Appendix A eq. 9-11).

    python -m ml.build_training_set --out data/processed/rb_training_set.csv [--force]

For every session with a card_id: 60 s windows judged steady on card_temp_c (T3.4,
temp_column="card_temp_c"); then, per window and only on exogenous conditions:
  steady state -> wind present (else no_wind) -> card geometry from data/reference_cards.csv
  measured by the camera (else missing_input) -> solve_card_rbh (T1.3: absorbed shortwave gate
  low_card_signal, |Rn| guard rn_near_zero, r_bH <= 0 suspect_card_rbh).
The measured T_card - T_air is never used to select. Rejecting r_bH <= 0 is selection on the
sign of the result, so its effect is measured and reported (owner decision 2026-11-09):
    truncation_bias_f = mean f (kept rows) - mean f (kept + suspect rows)
Kept rows:
    r_bH,flat = flat-plate r_bH(d, wind, T_card - T_air, T_air)    s m-1   (eq. 10)
    f         = r_bH,card / r_bH,flat                               (-)     (eq. 11 target)
    r_bh_card_sigma = sigma_rn_card_fraction x r_bH,card            s m-1   (stored and reported;
              NOT used as a training weight: its size depends on the measured result)
d = the card's d_min_mm or d_eq_mm (physics.characteristic_dimension).
incidence_angle_deg = the session's card_angle_deg; blank -> 0 with flag card_angle_default.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from processing import energy_balance_utils as ebu
from processing.config import get_position_factor, load_calibration, load_config
from processing.data_io import ReferenceCard, Session, list_sessions, load_reference_cards, load_session
from processing.measurement_windows import build_windows
from processing.physics_baseline import ReferenceCardReading, solve_card_rbh

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent
MM_PER_M = 1000.0
CARD_TEMP_COLUMN = "card_temp_c"
GEOMETRY_COLUMNS = ("d_min_mm", "d_eq_mm", "aspect_ratio", "eccentricity", "solidity", "circularity",
                    "edge_density")
# Appendix B rb_training_set columns, then flags (deviation: card flags need a place, decision_log).
TRAINING_COLUMNS = (
    "row_id", "session_id", "card_id", "shape_group", "window_start_iso", "window_end_iso",
    "n_samples", "card_temp_c", "air_temp_c", "surroundings_temp_c", "lux", "light_source",
    "rn_wm2", "c_card", "wind_ms", "convection_regime", "r_bh_card", "r_bh_card_sigma", "r_bh_flat",
    "f_ratio") + GEOMETRY_COLUMNS + ("photo_gap_sec", "incidence_angle_deg", "valid", "reason", "flags")
REJECT_COLUMNS = TRAINING_COLUMNS + ("note",)
CAMERA = "camera"


class InsufficientShapes(Exception):
    """Fewer than ml.min_shapes distinct shape_group values and --force not given."""


@dataclass(frozen=True)
class TrainingSet:
    kept: pd.DataFrame       # TRAINING_COLUMNS, valid rows only
    rejects: pd.DataFrame    # REJECT_COLUMNS, every rejected window with its reason and a note
    report: dict[str, Any]


def _blank_row(session: Session, card_id: str, win: pd.Series) -> dict[str, Any]:
    row: dict[str, Any] = {c: math.nan for c in REJECT_COLUMNS}
    row.update(session_id=session.session_id, card_id=card_id, shape_group="",
               window_start_iso=win["window_start_iso"], window_end_iso=win["window_end_iso"],
               n_samples=int(win["n_samples"]), light_source=session.metadata.get("light_source", ""),
               valid=False, reason="", flags="", note="", convection_regime="")
    for col in ("card_temp_c", "air_temp_c", "lux", "wind_ms"):
        row[col] = float(win[col]) if col in win and pd.notna(win[col]) else math.nan
    row["surroundings_temp_c"] = row["air_temp_c"]          # assumption A4: T_sur = T_air
    return row


def _card_problem(card: ReferenceCard | None, card_id: str) -> str:
    if card is None:
        return f"card_id {card_id!r} is not in reference_cards.csv"
    if card.geometry_source != CAMERA:
        return (f"card {card_id!r} geometry_source is {card.geometry_source!r}, not 'camera': ML features "
                "must come from Arducam photos (owner decision 2026-11-09)")
    missing = [c for c in GEOMETRY_COLUMNS if getattr(card, c) is None]
    return f"card {card_id!r} has no {missing} in reference_cards.csv" if missing else ""


def session_rows(session: Session, cards: dict[str, ReferenceCard], cfg: dict[str, Any],
                 cal: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(kept rows, rejected rows) for one session; ([], []) if the session has no card."""
    meta = session.metadata
    card_id = str(meta.get("card_id") or "").strip()
    if not card_id or CARD_TEMP_COLUMN not in session.sensor.columns:
        return [], []
    card = cards.get(card_id)
    problem = _card_problem(card, card_id)
    windows = build_windows(session.sensor, session.events, meta, cfg, temp_column=CARD_TEMP_COLUMN)
    angle = meta.get("card_angle_deg")
    base_flags = []
    if angle is None or angle == "" or (isinstance(angle, float) and math.isnan(angle)):
        angle, base_flags = 0.0, ["card_angle_default"]
    d_col = f"{cfg['physics']['characteristic_dimension']}_mm"
    c_card, c_flags = get_position_factor(cfg, cal, "card")
    sigma_frac = float(cfg["uncertainty"]["sigma_rn_card_fraction"])

    kept, rejected = [], []
    for _, win in windows.iterrows():
        row = _blank_row(session, card_id, win)
        row["incidence_angle_deg"] = float(angle)
        row["c_card"] = c_card
        if card is not None:
            row["shape_group"] = card.shape_group
            for c in GEOMETRY_COLUMNS:
                v = getattr(card, c)
                row[c] = math.nan if v is None else float(v)
        flags = list(base_flags) + list(c_flags)

        # Exogenous gates only, in order. Nothing below looks at T_card - T_air.
        if win["reason"] == "missing_input" or not math.isfinite(row["card_temp_c"]):
            row.update(reason="missing_input", note="fewer than min_samples clean rows or no card temperature")
        elif not bool(win["steady_state"]):
            row.update(reason="not_steady_state", note=str(win["flags"]))
        elif not math.isfinite(row["wind_ms"]):
            row.update(reason="no_wind", note="no valid wind measurement in this window")
        elif problem:
            row.update(reason="missing_input", note=problem)
        else:
            res = solve_card_rbh(ReferenceCardReading(
                card_id=card_id, card_temp_c=row["card_temp_c"], air_temp_c=row["air_temp_c"],
                surroundings_temp_c=row["surroundings_temp_c"], lux=row["lux"],
                light_source=row["light_source"], c_card=c_card), cfg, cal)
            flags += [f for f in res.flags if f not in flags]
            row["rn_wm2"] = res.rn_wm2 if res.rn_wm2 is not None else math.nan
            if res.r_bh is not None:
                d_m = row[d_col] / MM_PER_M
                flat, regime = ebu.flat_plate_rbh(d_m, row["wind_ms"], row["card_temp_c"] - row["air_temp_c"],
                                                  row["air_temp_c"], cfg)
                flags += list(ebu.convection_flags(regime))
                row.update(r_bh_card=res.r_bh, r_bh_flat=flat, f_ratio=res.r_bh / flat,
                           r_bh_card_sigma=sigma_frac * abs(res.r_bh), convection_regime=regime)
            row.update(valid=res.valid, reason=res.reason)
            if not res.valid:
                row["note"] = {"suspect_card_rbh": "r_bH <= 0 (kept here and counted; see truncation bias)",
                               "low_card_signal": "absorbed shortwave below physics.min_card_absorbed_sw_wm2",
                               "rn_near_zero": "|Rn| below physics.min_abs_rn_wm2"}.get(res.reason, "")
        row["flags"] = ";".join(dict.fromkeys(flags))
        (kept if row["valid"] else rejected).append(row)
    return kept, rejected


def build_training_set(sessions: Iterable[Session], cards: dict[str, ReferenceCard], cfg: dict[str, Any],
                       cal: dict[str, Any], force: bool = False) -> TrainingSet:
    """Training rows from all card sessions, plus rejects and the report (module docstring)."""
    kept_rows: list[dict[str, Any]] = []
    rej_rows: list[dict[str, Any]] = []
    for s in sessions:
        k, r = session_rows(s, cards, cfg, cal)
        kept_rows += k
        rej_rows += r
    kept = pd.DataFrame(kept_rows, columns=list(REJECT_COLUMNS))[list(TRAINING_COLUMNS)]
    rejects = pd.DataFrame(rej_rows, columns=list(REJECT_COLUMNS))
    kept["row_id"] = [f"{s}|{w}" for s, w in zip(kept["session_id"], kept["window_start_iso"])]
    rejects["row_id"] = [f"{s}|{w}" for s, w in zip(rejects["session_id"], rejects["window_start_iso"])]
    kept["valid"] = kept["valid"].astype(bool)
    report = make_report(kept, rejects, cfg)
    n_min = int(cfg["ml"]["min_shapes"])
    if report["n_shapes"] < n_min:
        if not force:
            raise InsufficientShapes(f"only {report['n_shapes']} distinct shape_group values; at least {n_min} "
                                     "(ml.min_shapes) are needed. Use --force to continue anyway.")
        report["warnings"].append(f"--force: only {report['n_shapes']} shapes (< {n_min}); a model trained on "
                                  "so few shapes will overfit and its cross-validation is unreliable")
    for w in report["warnings"]:
        logger.warning(w)
    return TrainingSet(kept, rejects, report)


def make_report(kept: pd.DataFrame, rejects: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    sus = rejects[rejects["reason"] == "suspect_card_rbh"]
    n_gated = len(kept) + len(sus)            # windows that reached the sign test
    frac = len(sus) / n_gated if n_gated else 0.0
    mean_kept = float(kept["f_ratio"].mean()) if len(kept) else math.nan
    all_f = pd.concat([kept["f_ratio"], sus["f_ratio"]]).dropna()
    mean_all = float(all_f.mean()) if len(all_f) else math.nan
    warnings: list[str] = []
    if frac > float(cfg["ml"]["max_suspect_fraction"]):
        warnings.append(f"{frac:.1%} of gated card windows are suspect_card_rbh (r_bH <= 0), above "
                        f"ml.max_suspect_fraction {cfg['ml']['max_suspect_fraction']:.0%}: rejecting them "
                        f"biases mean f by {mean_kept - mean_all:+.3f}; check thermal calibration")
    def by(col: str) -> dict[str, int]:
        return {str(k): int(v) for k, v in sus.groupby(col).size().items()} if len(sus) else {}
    wind_bin = sus["wind_ms"].round(1) if len(sus) else sus["wind_ms"]
    return {
        "n_kept": int(len(kept)),
        "rejects_by_reason": {str(k): int(v) for k, v in rejects["reason"].value_counts().items()},
        "n_shapes": int(kept["shape_group"].nunique()),
        "wind_ms_range": [float(kept["wind_ms"].min()), float(kept["wind_ms"].max())] if len(kept) else [],
        "lux_range": [float(kept["lux"].min()), float(kept["lux"].max())] if len(kept) else [],
        "suspect_count": int(len(sus)),
        "suspect_fraction": frac,
        "suspect_by_shape_group": by("shape_group"),
        "suspect_by_wind_ms": {str(k): int(v) for k, v in sus.groupby(wind_bin).size().items()} if len(sus) else {},
        "mean_f_kept": mean_kept,
        "mean_f_all_gated": mean_all,
        "truncation_bias_f": (mean_kept - mean_all) if len(sus) else 0.0,
        "r_bh_card_sigma_range": ([float(kept["r_bh_card_sigma"].min()), float(kept["r_bh_card_sigma"].max())]
                                  if len(kept) else []),
        "warnings": warnings,
    }


def format_report(r: dict[str, Any]) -> str:
    lines = [
        f"rows kept: {r['n_kept']}",
        f"rejects by reason: {r['rejects_by_reason']}",
        f"distinct shapes (shape_group): {r['n_shapes']}",
        f"wind range (m/s): {r['wind_ms_range']}",
        f"lux range: {r['lux_range']}",
        f"suspect_card_rbh (r_bH <= 0): {r['suspect_count']} ({r['suspect_fraction']:.1%} of gated windows)",
        f"  suspect by shape_group: {r['suspect_by_shape_group']}  (should not depend on shape)",
        f"  suspect by wind (m/s): {r['suspect_by_wind_ms']}  (should not depend on wind)",
        f"truncation bias of mean f from rejecting r_bH <= 0: {r['truncation_bias_f']:+.4f} "
        f"(mean f kept {r['mean_f_kept']:.4f}, kept + suspect {r['mean_f_all_gated']:.4f})",
        f"label uncertainty r_bh_card_sigma range (s/m, not used as weight): {r['r_bh_card_sigma_range']}",
    ]
    lines += [f"WARNING: {w}" for w in r["warnings"]]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ml.build_training_set",
                                description="Build the card r_bH training set (T4.3).")
    p.add_argument("--out", default=str(REPO_ROOT / "data" / "processed" / "rb_training_set.csv"))
    p.add_argument("--rejects", default=None, help="rejects CSV (default: <out stem>_rejects.csv; "
                   "for rb_training_set.csv this is rb_training_set_rejects.csv)")
    p.add_argument("--raw-dir", default=str(REPO_ROOT / "data" / "raw"))
    p.add_argument("--cards", default=str(REPO_ROOT / "data" / "reference_cards.csv"))
    p.add_argument("--calibration", default=str(REPO_ROOT / "calibration" / "calibration.json"))
    p.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    p.add_argument("--force", action="store_true", help="continue with fewer than ml.min_shapes shapes")
    p.add_argument("--include-synthetic", action="store_true",
                   help="also use sessions made by ml/synthetic_data.py (testing only)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    cal = load_calibration(args.calibration)
    sessions = [load_session(s, args.raw_dir) for s in list_sessions(args.raw_dir)]
    sessions = [s for s in sessions if args.include_synthetic or not s.synthetic]
    if not any(str(s.metadata.get("card_id") or "").strip() for s in sessions):
        raise SystemExit(f"no card windows: no session under {args.raw_dir} has a card_id"
                         + ("" if args.include_synthetic else " (synthetic sessions skipped)"))
    try:
        ts = build_training_set(sessions, load_reference_cards(args.cards), cfg, cal, force=args.force)
    except InsufficientShapes as e:
        raise SystemExit(str(e)) from e
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rejects = Path(args.rejects) if args.rejects else out.with_name(f"{out.stem}_rejects.csv")
    ts.kept.to_csv(out, index=False)
    ts.rejects.to_csv(rejects, index=False)
    print(format_report(ts.report))
    print(f"wrote {out} and {rejects}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
