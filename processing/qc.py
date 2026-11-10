"""Quality-control flags for sensor logs (T3.3, spec B12). Flags rows; never deletes or edits them.

add_qc_flags(df, cfg) returns a copy of df with a qc_flags column: flag names joined by ";"
("" = clean). Flags already in the column are kept. Columns that are not in df are skipped,
so the old 8-column debug log works as well as the Appendix B sensor_log.csv.

Flags (all thresholds in config `qc`):
  range_<col>         value below low or above high (inclusive limits); +/-inf counts as out of
                      range; NaN is missing data, not a range failure (handled downstream)
  spike_<col>         Hampel filter on spike_columns: centred window of hampel_window rows
                      (shrinks at the ends), median m and MAD = median(|x - m|) of the window;
                      flag if |x - m| > max(n_sigma x 1.4826 x MAD, min_abs, min_rel x |m|)
  sensor_overflow     a TSL2591 raw channel >= overflow_count, or lux is +/-inf
  duplicate_timestamp the timestamp appears more than once (every copy is flagged, because
                      there is no way to tell which copy is right)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

FLAG_COLUMN = "qc_flags"
SEP = ";"


def hampel_flags(x: np.ndarray, window: int, n_sigma: float, mad_to_sigma: float,
                 min_abs: float = 0.0, min_rel: float = 0.0) -> np.ndarray:
    """Boolean spike flags for one series (see module docstring).

    A window needs at least window // 2 + 1 finite values to judge a point (the size of the
    window at either end of the series); NaN points are never flagged.
    """
    if window < 3 or window % 2 == 0:
        raise ValueError(f"hampel window must be an odd number of 3 or more, got {window}")
    x = np.asarray(x, dtype=float)
    half = window // 2
    min_points = half + 1
    out = np.zeros(x.size, dtype=bool)
    for i in range(x.size):
        if not np.isfinite(x[i]):
            continue
        w = x[max(0, i - half): i + half + 1]
        w = w[np.isfinite(w)]
        if w.size < min_points:
            continue
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        limit = max(n_sigma * mad_to_sigma * mad, min_abs, min_rel * abs(med))
        out[i] = abs(x[i] - med) > limit
    return out


def _range_flags(x: np.ndarray, low: float | None, high: float | None) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        bad = np.isinf(x)
        if low is not None:
            bad |= x < low
        if high is not None:
            bad |= x > high
    return bad


def _column(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def qc_flag_table(df: pd.DataFrame, cfg: dict[str, Any]) -> list[list[str]]:
    """New flag names for each row, in a fixed order (range, spike, overflow, duplicate)."""
    q = cfg["qc"]
    flags: list[list[str]] = [[] for _ in range(len(df))]

    def add(mask: np.ndarray, name: str) -> None:
        for i in np.flatnonzero(mask):
            flags[i].append(name)

    ranges = dict(q["ranges"])
    for col in q["thermal_columns"]:
        ranges.setdefault(col, q["thermal_range_c"])
    for col, (low, high) in ranges.items():
        if col in df.columns:
            add(_range_flags(_column(df, col), low, high), f"range_{col}")

    for col in q["spike_columns"]:
        if col in df.columns:
            add(hampel_flags(_column(df, col), int(q["hampel_window"]), float(q["hampel_n_sigma"]),
                             float(q["mad_to_sigma"]),
                             min_abs=float(q["spike_min_abs"].get(col, 0.0)),
                             min_rel=float(q["spike_min_rel"].get(col, 0.0))),
                f"spike_{col}")

    overflow = np.zeros(len(df), dtype=bool)
    for col in q["overflow_columns"]:
        if col in df.columns:
            overflow |= _column(df, col) >= q["overflow_count"]
    if "lux" in df.columns:
        overflow |= np.isinf(_column(df, "lux"))
    add(overflow, "sensor_overflow")

    ts_col = next((c for c in q["timestamp_columns"] if c in df.columns), None)
    if ts_col is not None:
        add(df[ts_col].duplicated(keep=False).to_numpy(), "duplicate_timestamp")
    return flags


def add_qc_flags(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Copy of df with the qc_flags column filled in. Rows and values are unchanged."""
    out = df.copy()
    new = qc_flag_table(df, cfg)
    if FLAG_COLUMN in df.columns:
        old = [[] if pd.isna(v) else [t for t in str(v).split(SEP) if t] for v in df[FLAG_COLUMN]]
    else:
        old = [[] for _ in range(len(df))]
    merged = [list(dict.fromkeys(a + b)) for a, b in zip(old, new)]   # keep order, no repeats
    out[FLAG_COLUMN] = pd.Series([SEP.join(m) for m in merged], index=df.index, dtype=object)
    return out
