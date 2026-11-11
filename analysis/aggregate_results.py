"""Leaf g_s for every window and every r_bH method, with uncertainty and censoring (T5.1).

    python -m analysis.aggregate_results [--synthetic] [--out data/processed/leaf_gs_results.csv]

For each session: leaf windows (T3.4, photo geometry joined by time). Every window gets one row
per method in config `methods` (card_direct = primary, ml_corrected = secondary, flat_plate =
baseline). The roles are fixed BEFORE real data are collected (decision_log 2026-11-10).
Nothing is ever dropped: a method that cannot run gives an invalid row with a reason code, and
censored results (below_detection, above_range) keep their label and detection limit.

r_bH per method [s m-1]:
    card_direct   r_bH = r_bH,card = rho cp (T_card - T_a) / Rn_card   (eq. 9) from the nearest
                  valid card window of the same session within windows.card_match_minutes.
                  Needs no wind and no photo. Only sessions with dry_reference_type = card.
    flat_plate    r_bH = 1 / g_bH(d, U, T_leaf - T_a)                   (eq. 10)
    ml_corrected  r_bH = f_hat(geometry, U) x r_bH,flat                 (eq. 11)
Then g_s from eqs. 1-8 (gs_calculator.solve_gs) and the Monte Carlo interval, resolved label
and per-window detection limit (uncertainty.gs_uncertainty). Surroundings = air (A4).

r_bH spread used in the Monte Carlo (owner decision A, 2026-11-10), as a log-normal sigma:
    sigma_rb = ln(r_bH,hi / r_bH,lo) / (2 z),  z = Phi^-1(0.975) = 1.96   (95 % interval)
from the card Monte Carlo (card_direct) or the ML interval (ml_corrected). flat_plate uses
uncertainty.sigma_rb_fraction (0.2, known to be too small: decision 20 allows up to 3x).
If a method's interval has lo <= 0, sigma_rb_fraction is used and the row is flagged marginal.
For flat_plate, r_bh_lo/hi = r_bH x exp(-/+ z sigma_rb), the same log-normal.

Convection regime: flat-plate regime from the leaf's d and wind; for card_direct without leaf
geometry, from the card's d (reference_cards.csv); no wind -> "unknown" + flag regime_unknown.
Flags: card_transfer_unreliable (card_direct, mixed or free, A17); baseline_unreliable (flat
plate and the flat-plate part of ml_corrected). A valid row with a blank g_s_mol_dl always has
exactly one of dl_not_resolved (no grid point resolved at these conditions) or dl_search_failed
(a forward root search failed or errored, so "nothing resolved" cannot be claimed).

Jones index (eqs. 12-13), steady windows only (owner decision B, 2026-11-10): T_dry = card_temp_c
ONLY when dry_reference_type = vaseline_leaf; T_wet = wet_ref_temp_c. A wet reference without a
vaseline-leaf dry reference -> I_g and CWSI blank, flag jones_no_leaf_dry_reference.
I_g above_range is blank with jones_ig_censor = above_range.

Schema: Appendix B leaf_gs_results columns, then two logged deviations: light_source (needed to
re-run the physics, T5.4) and jones_ig_censor (owner decision C).
"""
from __future__ import annotations

import argparse
import logging
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm

from ml.predict_correction import load_model, predict_rbh
from processing import energy_balance_utils as ebu
from processing.config import get_light_source, load_calibration, load_config
from processing.data_io import (
    ReferenceCard,
    Session,
    list_sessions,
    load_photo_features,
    load_reference_cards,
    load_session,
)
from processing.gs_calculator import LeafReading, solve_gs
from processing.jones_index import conductance_index
from processing.measurement_windows import build_windows
from processing.physics_baseline import ReferenceCardReading, solve_card_rbh
from processing.uncertainty import (CardLink, Sigmas, attach_uncertainty, card_link, card_rbh_uncertainty, get_sigmas,
                                    gs_uncertainty)

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent
MM_PER_M = 1000.0
SECONDS_PER_MINUTE = 60.0
# Two-sided 95 % interval: the Monte Carlo reports 2.5 and 97.5 percentiles.
Z95 = float(norm.ppf(0.975))
ROLES = ("primary", "secondary", "baseline")
CARD_TEMP = "card_temp_c"
WET_TEMP = "wet_ref_temp_c"
NO_PHOTO = "no_photo_in_window"

APPENDIX_B_COLUMNS = (
    "session_id", "plant_id", "genotype", "water_treatment", "block", "position", "leaf_position",
    "reproductive_stage", "window_start_iso", "window_end_iso", "n_samples", "steady_state", "method",
    "method_role", "convection_regime", "r_bh", "r_bh_lo", "r_bh_hi", "leaf_temp_c", "air_temp_c",
    "rh_pct", "pressure_hpa", "lux", "rn_wm2", "latent_wm2", "vpd_pa", "r_s", "g_s_ms", "g_s_mol",
    "g_s_mol_lo", "g_s_mol_hi", "resolved", "censor", "g_s_mol_dl", "valid", "reason", "flags",
    "photo_gap_sec", "jones_ig", "cwsi")
# Deviations from Appendix B (decision_log 2026-11-10).
RESULT_COLUMNS = APPENDIX_B_COLUMNS + ("light_source", "jones_ig_censor")
META_COLUMNS = ("plant_id", "genotype", "water_treatment", "block", "position", "leaf_position",
                "reproductive_stage")


# ---------------------------------------------------------------- small helpers

def _num(x: Any) -> float:
    """float, or nan for blank / text / None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v


def _finite(x: Any) -> bool:
    return math.isfinite(_num(x))


def _truthy(x: Any) -> bool:
    return x is True or str(x).strip().lower() == "true"


def _join(flags: list[str]) -> str:
    return ";".join(dict.fromkeys(f for f in flags if f))


def rb_sigma_from_interval(lo: float | None, hi: float | None) -> float | None:
    """Log-normal sigma of r_bH from its 95 % interval: ln(hi / lo) / (2 x 1.96) (dimensionless).
    None when the interval cannot be read that way (lo <= 0, hi <= lo, or not finite)."""
    lo_f, hi_f = _num(lo), _num(hi)
    if not (math.isfinite(lo_f) and math.isfinite(hi_f)) or lo_f <= 0 or hi_f <= lo_f:
        return None
    return math.log(hi_f / lo_f) / (2.0 * Z95)


def _sigmas_for(base: Sigmas, rb_fraction: float | None, source: str) -> Sigmas:
    """Same sigmas with the r_bH spread replaced (source recorded)."""
    if rb_fraction is None:
        return base
    sources = tuple((k, source if k == "sigma_rb_fraction" else v) for k, v in base.sources)
    return replace(base, rb_fraction=float(rb_fraction), sources=sources)


def dl_flags(g_s_mol_dl: float | None, dl_grid_no_root: int) -> list[str]:
    """Why a detection limit is blank (owner decision 2026-11-10). Empty when a limit was found.
    dl_search_failed: at least one grid point had no forward leaf temperature (root search failed
    or errored), so "nothing resolved" cannot be claimed. dl_not_resolved: every grid point was
    searched and none was resolved at these conditions."""
    if g_s_mol_dl is not None and math.isfinite(g_s_mol_dl):
        return []
    return ["dl_search_failed"] if dl_grid_no_root > 0 else ["dl_not_resolved"]


def method_roles(cfg: dict[str, Any]) -> list[tuple[str, str]]:
    """[(method, role)] in role order primary, secondary, baseline (config `methods`)."""
    return [(str(cfg["methods"][r]), r) for r in ROLES]


# ---------------------------------------------------------------- card windows

def card_windows(session: Session, cfg: dict[str, Any], cal: dict[str, Any]) -> list[dict[str, Any]]:
    """Card r_bH per card window (steady on card_temp_c, T3.4/T4.3), with its Monte Carlo interval.
    Only for dry_reference_type = card with a card_id; [] otherwise. Wind is not needed."""
    meta = session.metadata
    card_id = str(meta.get("card_id") or "").strip()
    if (str(meta.get("dry_reference_type", "")).strip() != "card" or not card_id
            or CARD_TEMP not in session.sensor.columns or not session.sensor[CARD_TEMP].notna().any()):
        return []
    wins = build_windows(session.sensor, session.events, meta, cfg, temp_column=CARD_TEMP)
    n = int(cfg["uncertainty"]["n_draws"])
    out = []
    for _, w in wins.iterrows():
        start = pd.Timestamp(w["window_start_iso"])
        end = pd.Timestamp(w["window_end_iso"])
        rec: dict[str, Any] = {"centre": start + (end - start) / 2, "card_id": card_id, "valid": False,
                               "reason": str(w["reason"]), "r_bh": None, "r_bh_lo": None, "r_bh_hi": None,
                               "card_temp_c": _num(w.get(CARD_TEMP)), "air_temp_c": _num(w.get("air_temp_c")),
                               "flags": []}
        if rec["reason"] == "ok":
            reading = ReferenceCardReading(card_id=card_id, card_temp_c=rec["card_temp_c"],
                                           air_temp_c=rec["air_temp_c"], surroundings_temp_c=rec["air_temp_c"],
                                           lux=_num(w.get("lux")), light_source=str(meta.get("light_source")))
            res = solve_card_rbh(reading, cfg, cal)
            rec.update(valid=res.valid, reason=res.reason, flags=list(res.flags))
            if res.valid:
                cu = card_rbh_uncertainty(reading, cfg, cal, n=n, seed=0)
                rec.update(r_bh=res.r_bh, r_bh_lo=cu.r_bh_lo, r_bh_hi=cu.r_bh_hi, reading=reading)
        out.append(rec)
    return out


def match_card(cards: list[dict[str, Any]], centre: pd.Timestamp, max_minutes: float) -> dict[str, Any] | None:
    """Nearest VALID card window within max_minutes of the leaf window centre; if none is valid,
    the nearest card window in range (its reason explains why); None if no card window in range."""
    in_range = [(abs((c["centre"] - centre).total_seconds()), c) for c in cards]
    in_range = [(d, c) for d, c in in_range if d <= max_minutes * SECONDS_PER_MINUTE]
    if not in_range:
        return None
    valid = [(d, c) for d, c in in_range if c["valid"]]
    pool = valid or in_range
    return min(pool, key=lambda dc: dc[0])[1]


# ---------------------------------------------------------------- one session

def _regime(d_mm: float, wind: float, delta_t: float, air: float, cfg: dict[str, Any]) -> tuple[float, str] | None:
    if not all(math.isfinite(v) for v in (d_mm, wind, delta_t, air)) or d_mm <= 0:
        return None
    return ebu.flat_plate_rbh(d_mm / MM_PER_M, wind, delta_t, air, cfg)


def _jones(meta: Mapping[str, Any], w: pd.Series, steady: bool, cfg: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """jones_ig, cwsi, jones_ig_censor and extra flags for one window (rules in module docstring)."""
    out: dict[str, Any] = {"jones_ig": math.nan, "cwsi": math.nan, "jones_ig_censor": ""}
    t_wet = _num(w.get(WET_TEMP))
    if not steady or not _truthy(meta.get("wet_reference_present")) or not math.isfinite(t_wet):
        return out, []
    if str(meta.get("dry_reference_type", "")).strip() != "vaseline_leaf":
        return out, ["jones_no_leaf_dry_reference"]
    j = conductance_index(_num(w.get(CARD_TEMP)), _num(w.get("leaf_temp_c")), t_wet, cfg)
    if not j.cwsi.valid:
        # Not a g_s reason, so it goes in flags (decision_log 2026-11-10).
        return out, [j.cwsi.reason] if j.cwsi.reason == "references_not_bracketing" else []
    out.update(cwsi=j.cwsi.value, jones_ig_censor=j.i_g.censor,
               jones_ig=j.i_g.value if j.i_g.value is not None else math.nan)
    return out, []


def session_results(session: Session, photo_features: pd.DataFrame | None, cards: Mapping[str, ReferenceCard],
                    cfg: dict[str, Any], cal: dict[str, Any], model: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Rows (RESULT_COLUMNS) for every window x method of one session."""
    meta = session.metadata
    light_source = str(meta.get("light_source"))
    wins = build_windows(session.sensor, session.events, meta, cfg, photo_features=photo_features)
    card_wins = card_windows(session, cfg, cal)
    max_min = float(cfg["windows"]["card_match_minutes"])
    d_col = f"{cfg['physics']['characteristic_dimension']}_mm"
    card_ref = cards.get(str(meta.get("card_id") or "").strip())
    base_sigmas = get_sigmas(cfg, cal, get_light_source(cfg, cal, light_source))
    sigma_flat = float(base_sigmas.rb_fraction)
    feature_names = list(model["features"]) if model is not None else []
    angle = _num(meta.get("incidence_angle_deg"))

    rows: list[dict[str, Any]] = []
    for _, w in wins.iterrows():
        start, end = pd.Timestamp(w["window_start_iso"]), pd.Timestamp(w["window_end_iso"])
        centre = start + (end - start) / 2
        steady = bool(w["steady_state"])
        win_reason = str(w["reason"])
        win_flags = [f for f in str(w.get("flags", "") or "").split(";") if f]
        leaf, air = _num(w.get("leaf_temp_c")), _num(w.get("air_temp_c"))
        wind, d_mm = _num(w.get("wind_ms")), _num(w.get(d_col))
        leaf_flat = _regime(d_mm, wind, leaf - air, air, cfg)
        jones, jones_flags = _jones(meta, w, steady, cfg)
        common: dict[str, Any] = {
            "session_id": session.session_id, **{c: meta.get(c, "") for c in META_COLUMNS},
            "window_start_iso": w["window_start_iso"], "window_end_iso": w["window_end_iso"],
            "n_samples": int(w["n_samples"]), "steady_state": steady, "leaf_temp_c": leaf, "air_temp_c": air,
            "rh_pct": _num(w.get("humidity_pct")), "pressure_hpa": _num(w.get("pressure_hpa")),
            "lux": _num(w.get("lux")), "photo_gap_sec": _num(w.get("photo_gap_sec")),
            "light_source": light_source, **jones}
        wind_flags = [] if math.isfinite(wind) else ["regime_unknown"]

        for method, role in method_roles(cfg):
            row = {c: math.nan for c in RESULT_COLUMNS}
            row.update(common, method=method, method_role=role, valid=False, reason="", censor="none",
                       resolved=False, convection_regime="unknown")
            flags = win_flags + wind_flags + jones_flags
            r_bh = r_lo = r_hi = None
            link: CardLink | None = None
            sigma: float | None = None
            reason = ""
            regime = leaf_flat[1] if leaf_flat else None

            if method == "card_direct":
                reason = "" if win_reason in ("ok", NO_PHOTO) else win_reason
                if not reason:
                    card = match_card(card_wins, centre, max_min)
                    if card is None:
                        reason = "missing_input"          # no card in this session or none in time
                    elif not card["valid"]:
                        reason = card["reason"]
                    else:
                        r_bh, r_lo, r_hi = card["r_bh"], card["r_bh_lo"], card["r_bh_hi"]
                        flags += card["flags"]
                        sigma = rb_sigma_from_interval(r_lo, r_hi)
                        if cfg["uncertainty"].get("card_leaf_shared_draws", False):
                            link = card_link(card["reading"], cfg, cal)   # one camera, one thermometer (A43)
                        if regime is None and card_ref is not None and getattr(card_ref, d_col, None):
                            fb = _regime(float(getattr(card_ref, d_col)), wind,
                                         card["card_temp_c"] - card["air_temp_c"], card["air_temp_c"], cfg)
                            regime = fb[1] if fb else None
                        if regime in ("mixed", "free"):
                            flags.append("card_transfer_unreliable")
            elif win_reason != "ok":
                reason = win_reason
            elif not math.isfinite(wind):
                reason = "no_wind"
            elif not math.isfinite(d_mm):
                reason = NO_PHOTO
            elif method == "flat_plate":
                r_bh = leaf_flat[0]
                sigma = sigma_flat
                r_lo, r_hi = r_bh * math.exp(-Z95 * sigma), r_bh * math.exp(Z95 * sigma)
                flags += list(ebu.convection_flags(regime))
            else:                                          # ml_corrected
                if model is None:
                    reason = "missing_input"
                    flags.append("ml_model_missing")
                else:
                    feats = {n: _num(w.get(n)) for n in feature_names}
                    feats["incidence_angle_deg"] = angle
                    pred = predict_rbh(feats, wind, leaf - air, air, cfg, model)
                    flags += list(pred.flags)
                    if not pred.valid:
                        reason = pred.reason
                    else:
                        r_bh, r_lo, r_hi = pred.r_bh, pred.r_bh_lo, pred.r_bh_hi
                        sigma = rb_sigma_from_interval(r_lo, r_hi)

            if regime is not None:
                row["convection_regime"] = regime
            if r_bh is None:
                row["reason"] = reason or "missing_input"
                row["flags"] = _join(flags)
                rows.append(row)
                continue

            if sigma is None:                              # method interval not usable as a ratio
                flags.append("marginal")
            s = _sigmas_for(base_sigmas, sigma if sigma is not None else sigma_flat,
                            "method_interval" if sigma is not None and method != "flat_plate"
                            else dict(base_sigmas.sources)["sigma_rb_fraction"])
            reading = LeafReading(leaf, air, air, row["rh_pct"], row["pressure_hpa"], row["lux"],
                                  light_source, float(r_bh), method)
            res = solve_gs(reading, cfg, cal)
            unc = gs_uncertainty(reading, cfg, cal, sigmas=s, seed=0, card=link)
            res = attach_uncertainty(res, unc)
            flags += list(res.flags)
            if res.valid and unc.valid:
                flags += dl_flags(unc.g_s_mol_dl, unc.dl_grid_no_root)
            row.update(r_bh=r_bh, r_bh_lo=_num(r_lo), r_bh_hi=_num(r_hi), rn_wm2=_num(res.rn_wm2),
                       latent_wm2=_num(res.latent_wm2), vpd_pa=_num(res.vpd_pa), r_s=_num(res.r_s),
                       g_s_ms=_num(res.g_s_ms), g_s_mol=_num(res.g_s_mol), g_s_mol_lo=_num(res.g_s_lo),
                       g_s_mol_hi=_num(res.g_s_hi), g_s_mol_dl=_num(res.g_s_mol_dl), censor=res.censor,
                       resolved=bool(unc.resolved), valid=res.valid, reason=res.reason, flags=_join(flags))
            rows.append(row)
    return rows


# ---------------------------------------------------------------- campaign

def _photo_features(processed_dir: Path, session_id: str) -> pd.DataFrame | None:
    path = processed_dir / session_id / "photo_features.csv"
    return load_photo_features(path) if path.is_file() else None


def aggregate(sessions: list[Session], processed_dir: str | Path, cards: Mapping[str, ReferenceCard],
              cfg: dict[str, Any], cal: dict[str, Any], model: dict[str, Any] | None) -> pd.DataFrame:
    """All rows for all sessions (RESULT_COLUMNS), sorted by session, window and role.
    model = None means no correction model exists (ml_corrected rows: missing_input,
    flag ml_model_missing)."""
    processed_dir = Path(processed_dir)
    rows: list[dict[str, Any]] = []
    for s in sessions:
        logger.info("session %s", s.session_id)
        rows += session_results(s, _photo_features(processed_dir, s.session_id), cards, cfg, cal, model)
    df = pd.DataFrame(rows, columns=list(RESULT_COLUMNS))
    order = {r: i for i, r in enumerate(ROLES)}
    df["_k"] = df["method_role"].map(order)
    df = df.sort_values(["session_id", "window_start_iso", "_k"], kind="stable").drop(columns="_k")
    return df.reset_index(drop=True)


def summarise(df: pd.DataFrame) -> str:
    """Counts by method: valid / censor, and invalid by reason."""
    lines = []
    for m, g in df.groupby("method", sort=False):
        ok = g[g["valid"].astype(bool)]
        cen = ", ".join(f"{k} {v}" for k, v in ok["censor"].value_counts().items())
        bad = ", ".join(f"{k} {v}" for k, v in g.loc[~g["valid"].astype(bool), "reason"].value_counts().items())
        lines.append(f"{m} ({g['method_role'].iloc[0]}): {len(g)} rows; valid {len(ok)} [{cen}]; "
                     f"invalid {len(g) - len(ok)} [{bad}]")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m analysis.aggregate_results",
                                description="g_s for every leaf window and r_bH method (T5.1).")
    p.add_argument("--raw-dir", default=str(REPO_ROOT / "data" / "raw"))
    p.add_argument("--processed-dir", default=str(REPO_ROOT / "data" / "processed"))
    p.add_argument("--cards", default=str(REPO_ROOT / "data" / "reference_cards.csv"))
    p.add_argument("--calibration", default=str(REPO_ROOT / "calibration" / "calibration.json"))
    p.add_argument("--model", default=str(REPO_ROOT / "data" / "models" / "correction_model.joblib"))
    p.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    p.add_argument("--out", default=None, help="default data/processed/leaf_gs_results.csv "
                   "(leaf_gs_results_synthetic.csv with --synthetic)")
    p.add_argument("--synthetic", action="store_true",
                   help="use ONLY sessions from ml/synthetic_data.py and the synthetic calibration "
                        "(testing). Without it, synthetic sessions are skipped.")
    return p


def main(argv: list[str] | None = None) -> int:
    from ml.synthetic_data import synthetic_calibration      # only needed with --synthetic

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    sessions = [load_session(s, args.raw_dir) for s in list_sessions(args.raw_dir)]
    sessions = [s for s in sessions if s.synthetic == args.synthetic]
    label = "SYNTHETIC" if args.synthetic else "real"
    if not sessions:
        raise SystemExit(f"no {label} sessions under {args.raw_dir}"
                         + ("" if args.synthetic else " (synthetic sessions are skipped; use --synthetic)"))
    cal = synthetic_calibration(cfg) if args.synthetic else load_calibration(args.calibration)
    cards = load_reference_cards(args.cards) if Path(args.cards).is_file() else {}
    model = None
    if Path(args.model).is_file():
        model = load_model(args.model)
    else:
        logger.warning("no correction model at %s: ml_corrected rows will be missing_input "
                       "(flag ml_model_missing)", args.model)
    df = aggregate(sessions, args.processed_dir, cards, cfg, cal, model)
    name = "leaf_gs_results_synthetic.csv" if args.synthetic else "leaf_gs_results.csv"
    out = Path(args.out) if args.out else Path(args.processed_dir) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{label} data: {len(sessions)} sessions, {len(df)} rows")
    print(summarise(df))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
