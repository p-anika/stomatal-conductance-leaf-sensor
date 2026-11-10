"""Split a session's sensor log into measurement windows, judge steady state, and attach photo
geometry by time (T3.4, spec B5 and B15). Settings from config.yaml section `windows`.

Windows: consecutive, window_sec long, starting at the first logged timestamp; a row belongs to
[start, end). Row time = timestamp_corrected_iso when filled (clock fix, B14), else timestamp_iso.
Rows with any qc flag (qc.add_qc_flags, applied here so nobody forgets) are left out.
Each window reports the mean (column keeps its name, e.g. air_temp_c) and the sample standard
deviation (<name>_sd, ddof = 1) of every numeric variable.

Steady state:
  slope = least-squares slope of leaf_temp_c against time [deg C min-1], over unflagged rows in
          [end - stability_minutes, end)
  acclimation [min] = acclimation_minutes (session metadata) + (end - session start), or, after
          an events.csv row labelled acclimation_reset_label (lamp_on), end - that event
  steady_state = |slope| < stability_slope_c_per_min  AND  acclimation >= min_acclimation_minutes
                 AND the session has data back to end - stability_minutes
Photo pairing (attach_photo_features): ok photos inside the window -> median of each feature;
  else the photo nearest the window centre if within max_photo_gap_sec; matched on time only.
  photo_gap_sec = smallest |photo time - window centre| among the photos used.
  leaf_moved (whole session): (max - min) / median of d_min_mm over ok photos > leaf_moved_range_frac.

One reason per window, first that applies: missing_input (fewer than min_samples unflagged rows
or no leaf temperature), not_steady_state, no_photo_in_window, else ok. Every other problem is
also written to flags (";"-joined), so nothing is hidden. Detail flags: stability_window_incomplete,
leaf_temp_drifting, acclimation_too_short, leaf_moved.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from processing.data_io import PHOTO_GEOMETRY_COLUMNS
from processing.qc import FLAG_COLUMN, SEP, add_qc_flags

logger = logging.getLogger(__name__)

TIME_COLUMNS = ("timestamp_corrected_iso", "timestamp_iso")
LEAF_COLUMN = "leaf_temp_c"
# Counters and indexes, not measurements: averaging them means nothing.
NOT_AVERAGED = ("monotonic_s", "thermal_frame_idx")
REASON_ORDER = ("missing_input", "not_steady_state", "no_photo_in_window")
SECONDS_PER_MINUTE = 60.0


def _parse_times(values: pd.Series) -> pd.Series:
    text = values.astype("string").fillna("").str.strip()
    return pd.to_datetime(text.where(text != "", None), utc=True, format="ISO8601")


def row_times(df: pd.DataFrame) -> pd.Series:
    """Timezone-aware UTC time of each row: timestamp_corrected_iso if filled, else timestamp_iso."""
    present = [c for c in TIME_COLUMNS if c in df.columns]
    if not present:
        raise KeyError(f"sensor log needs a timestamp column ({' or '.join(TIME_COLUMNS)})")
    times = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    for col in reversed(present):          # later assignment wins: corrected time has priority
        parsed = _parse_times(df[col])
        times = parsed.where(parsed.notna(), times)
    return times


def _meta_value(meta: Any, name: str) -> Any:
    if isinstance(meta, Mapping):
        return meta[name]
    return getattr(meta, name)


def leaf_slope_c_per_min(t_min: np.ndarray, temp_c: np.ndarray) -> float:
    """Least-squares slope of temperature against time, deg C min-1 (NaN if < 2 distinct times)."""
    ok = np.isfinite(t_min) & np.isfinite(temp_c)
    if ok.sum() < 2 or np.ptp(t_min[ok]) == 0:
        return math.nan
    return float(np.polyfit(t_min[ok], temp_c[ok], 1)[0])


def _join(items: list[str]) -> str:
    out: list[str] = []
    for i in items:
        if i and i not in out:
            out.append(i)
    return SEP.join(out)


def _split(text: Any) -> list[str]:
    return [x for x in str(text).split(SEP) if x] if isinstance(text, str) else []


def _resolve(problems: list[str], flags: list[str]) -> tuple[str, str]:
    """First problem in REASON_ORDER becomes the reason; the rest join the flags."""
    ranked = sorted(set(problems), key=REASON_ORDER.index)
    reason = ranked[0] if ranked else "ok"
    return reason, _join(ranked[1:] + flags)


def build_windows(sensor_df: pd.DataFrame, events_df: pd.DataFrame | None, meta: Any,
                  cfg: dict[str, Any], photo_features: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per window (rules in the module docstring). meta: session metadata (mapping or
    object) with acclimation_minutes. photo_features: optional, from load_photo_features; when
    given, attach_photo_features is applied with the windows settings in cfg."""
    w_cfg = cfg["windows"]
    window_s = float(w_cfg["window_sec"])
    min_samples = int(w_cfg["min_samples"])
    stab_min = float(w_cfg["stability_minutes"])
    slope_max = float(w_cfg["stability_slope_c_per_min"])
    acc_min = float(w_cfg["min_acclimation_minutes"])
    reset_label = str(w_cfg["acclimation_reset_label"])
    logged_acc = float(_meta_value(meta, "acclimation_minutes"))

    df = add_qc_flags(sensor_df, cfg)
    times = row_times(df)
    has_time = times.notna()
    if not has_time.any():
        raise ValueError("sensor log has no readable timestamps")
    df, times = df[has_time], times[has_time]
    clean = df[FLAG_COLUMN].fillna("").astype(str).str.strip() == ""
    t0 = times.min()
    elapsed_s = (times - t0).dt.total_seconds().to_numpy()
    window_idx = np.floor(elapsed_s / window_s).astype(int)
    t_min_all = elapsed_s / SECONDS_PER_MINUTE

    numeric = [c for c in df.columns
               if c not in NOT_AVERAGED and pd.api.types.is_numeric_dtype(df[c])]
    resets = pd.Series(dtype="datetime64[ns, UTC]")
    if events_df is not None and len(events_df) and "label" in events_df.columns:
        ev = events_df[events_df["label"].astype(str).str.strip() == reset_label]
        resets = _parse_times(ev["timestamp_iso"]).dropna().sort_values()

    leaf = df[LEAF_COLUMN].to_numpy(float) if LEAF_COLUMN in df.columns else np.full(len(df), np.nan)
    clean_np = clean.to_numpy()
    rows: list[dict[str, Any]] = []
    for k in range(int(window_idx.max()) + 1):
        start = t0 + pd.Timedelta(seconds=k * window_s)
        end = start + pd.Timedelta(seconds=window_s)
        in_win = window_idx == k
        good = in_win & clean_np
        row: dict[str, Any] = {"window_start_iso": start.isoformat(), "window_end_iso": end.isoformat(),
                               "n_samples": int(good.sum()), "n_excluded": int((in_win & ~clean_np).sum())}
        sub = df[good]
        for c in numeric:
            vals = sub[c].astype(float)
            row[c] = float(vals.mean()) if len(vals) else math.nan
            row[f"{c}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else math.nan

        end_min = (end - t0).total_seconds() / SECONDS_PER_MINUTE
        trail = clean_np & (t_min_all >= end_min - stab_min) & (t_min_all < end_min)
        slope = leaf_slope_c_per_min(t_min_all[trail], leaf[trail])
        past = resets[resets <= end]
        if len(past):
            acclimation = (end - past.iloc[-1]).total_seconds() / SECONDS_PER_MINUTE
        else:
            acclimation = logged_acc + end_min

        problems: list[str] = []
        flags: list[str] = []
        if row["n_samples"] < min_samples or not np.isfinite(leaf[good]).any():
            problems.append("missing_input")
        complete = end_min - stab_min >= -1e-9
        if not complete:
            flags.append("stability_window_incomplete")
        drifting = not (math.isfinite(slope) and abs(slope) < slope_max)
        if complete and drifting:
            flags.append("leaf_temp_drifting")
        if acclimation < acc_min:
            flags.append("acclimation_too_short")
        steady = complete and not drifting and acclimation >= acc_min
        if not steady:
            problems.append("not_steady_state")
        reason, flag_text = _resolve(problems, flags)
        row.update(leaf_slope_c_per_min=slope, acclimation_minutes=acclimation,
                   steady_state=steady, valid=reason == "ok", reason=reason, flags=flag_text)
        rows.append(row)

    windows = pd.DataFrame(rows)
    if photo_features is not None:
        windows = attach_photo_features(windows, photo_features, float(w_cfg["max_photo_gap_sec"]),
                                        float(w_cfg["leaf_moved_range_frac"]))
    return windows


def attach_photo_features(windows: pd.DataFrame, photo_features: pd.DataFrame, max_gap_sec: float,
                          leaf_moved_range_frac: float) -> pd.DataFrame:
    """Return a copy of windows with photo geometry joined by time (rules in module docstring).
    Adds every geometry column present in photo_features, n_photos and photo_gap_sec."""
    out = windows.copy()
    pf = photo_features.copy()
    pf["_t"] = _parse_times(pf["timestamp"]) if not pd.api.types.is_datetime64_any_dtype(pf["timestamp"]) \
        else pf["timestamp"]
    ok = pf["ok"].astype(str).str.strip().str.lower() == "true"
    feats = [c for c in PHOTO_GEOMETRY_COLUMNS if c in pf.columns]

    moved = False      # judged on every ok photo of the session, timestamped or not
    if "d_min_mm" in feats:
        d = pd.to_numeric(pf.loc[ok, "d_min_mm"], errors="coerce").dropna()
        if len(d) >= 2 and d.median() > 0:
            moved = (d.max() - d.min()) / d.median() > leaf_moved_range_frac

    pf = pf[ok & pf["_t"].notna()]
    t_ph = pf["_t"]
    vals = pf[feats].apply(pd.to_numeric, errors="coerce")

    starts = pd.to_datetime(out["window_start_iso"], utc=True, format="ISO8601")
    ends = pd.to_datetime(out["window_end_iso"], utc=True, format="ISO8601")
    new_cols: dict[str, list[float]] = {c: [] for c in feats + ["n_photos", "photo_gap_sec"]}
    reasons, flags = list(out["reason"]), list(out["flags"])
    for i in range(len(out)):
        centre = starts.iloc[i] + (ends.iloc[i] - starts.iloc[i]) / 2
        gap = (t_ph - centre).dt.total_seconds().abs()
        inside = (t_ph >= starts.iloc[i]) & (t_ph < ends.iloc[i])
        if inside.any():
            chosen = vals[inside].median()
            n, g = int(inside.sum()), float(gap[inside].min())
        elif len(gap) and gap.min() <= max_gap_sec:
            j = gap.idxmin()
            chosen, n, g = vals.loc[j], 1, float(gap.loc[j])
        else:
            chosen, n, g = pd.Series(math.nan, index=feats), 0, math.nan
        for c in feats:
            new_cols[c].append(float(chosen[c]))
        new_cols["n_photos"].append(n)
        new_cols["photo_gap_sec"].append(g)

        problems = [reasons[i]] if reasons[i] != "ok" else []
        extra = _split(flags[i])
        if n == 0:
            problems.append("no_photo_in_window")
        if moved:
            extra.append("leaf_moved")
        # flags may already hold lower-ranked reasons; re-rank them together with the new one
        problems += [f for f in extra if f in REASON_ORDER]
        reasons[i], flags[i] = _resolve(problems, [f for f in extra if f not in REASON_ORDER])
    for c, v in new_cols.items():
        out[c] = v
    out["reason"], out["flags"] = reasons, flags
    out["valid"] = out["reason"] == "ok"
    if moved:
        logger.info("leaf_moved: d_min range exceeds %.0f %% of its median", 100 * leaf_moved_range_frac)
    return out
