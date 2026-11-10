"""Detection limit and resolved range of g_s at representative conditions (T5.4, A15, A33).

    python -m analysis.detection_limit [--synthetic]

1. Representative conditions = medians over the VALID rows of the primary method (card_direct)
   in leaf_gs_results.csv: air temperature, RH, pressure, lux, r_bH (censored rows included:
   their conditions are as real as any other; choosing on the result would bias them).
   r_bH spread sigma_rb = median of ln(r_bh_hi / r_bh_lo) / (2 x 1.96) over those rows.
2. Sweep: for each true g_s on the uncertainty grid (dl_grid_min_mol .. g_s_cap_mol, log spaced)
   find the leaf temperature it would give (energy balance, forward), apply the same Monte
   Carlo draws as every window (uncertainty.resolution_at), and record the 95 % interval,
   the fraction of usable draws and the relative width.
3. Resolved (A33): usable fraction >= dl_min_resolved_fraction AND relative width
   (p97.5 - p2.5) / median < max_relative_width AND p97.5 < g_s_cap_mol.
   Detection limit = lowest resolved true g_s. Because high g_s runs into the cap (not bounded
   above), the HIGHEST resolved g_s is reported too: the resolved band.
   This is the same rule and code as the per-window g_s_mol_dl stored on censored rows (T5.1).
4. Empirical brackets: a petroleum-jelly (vaseline) leaf (g_s about 0) and a wet reference
   (r_s about 0) are solved as if they were leaves; neither should be resolved.
   Real data: T_dry = card_temp_c in vaseline_leaf sessions, T_wet = wet_ref_temp_c, each at its
   own window's air, RH, pressure and lux, with the representative r_bH. Synthetic: forward leaf
   temperatures at synthetic.bracket_dry_g_s_mol and bracket_wet_g_s_mol.
Outputs: sweep CSV, brackets CSV, docs/figures/detection_limit[_synthetic].png, one summary line.
"""
from __future__ import annotations

import argparse
import logging
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.aggregate_results import rb_sigma_from_interval
from processing.config import get_light_source, load_calibration, load_config
from processing.data_io import Session, list_sessions, load_session
from processing.gs_calculator import LeafReading, solve_gs
from processing.measurement_windows import build_windows
from processing.uncertainty import Sigmas, dl_grid, forward_leaf_temp, get_sigmas, gs_uncertainty, resolution_at

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_COLUMNS = ("g_s_true_mol", "root_found", "leaf_temp_c", "g_s_mol_lo", "g_s_mol_median", "g_s_mol_hi",
                 "frac_ok", "relative_width", "resolved")
BRACKET_COLUMNS = ("reference", "temp_c", "air_temp_c", "censor", "g_s_mol", "g_s_mol_lo", "g_s_mol_hi",
                   "resolved", "expected", "as_expected")
DRY, WET = "petroleum_jelly_leaf", "wet_reference"


# ---------------------------------------------------------------- conditions

def representative_reading(results: pd.DataFrame, cfg: dict[str, Any]) -> tuple[LeafReading, float | None, int]:
    """(median-conditions reading, median r_bH log-sigma or None, number of rows used) from the
    valid rows of the primary method. Raises ValueError if there are none."""
    primary = str(cfg["methods"]["primary"])
    valid = results["valid"].astype(str).str.lower().isin(["true", "1"])
    rows = results[valid & (results["method"] == primary)]
    if rows.empty:
        raise ValueError(f"no valid {primary} rows in the results table; run "
                         "python -m analysis.aggregate_results first")
    med = {c: float(pd.to_numeric(rows[c], errors="coerce").median())
           for c in ("leaf_temp_c", "air_temp_c", "rh_pct", "pressure_hpa", "lux", "r_bh")}
    light = rows["light_source"].mode().iloc[0]
    sig = [rb_sigma_from_interval(lo, hi) for lo, hi in zip(rows["r_bh_lo"], rows["r_bh_hi"])]
    sig = [s for s in sig if s is not None]
    reading = LeafReading(med["leaf_temp_c"], med["air_temp_c"], med["air_temp_c"], med["rh_pct"],
                          med["pressure_hpa"], med["lux"], str(light), med["r_bh"], primary)
    return reading, (float(np.median(sig)) if sig else None), int(len(rows))


def sweep_sigmas(cfg: dict[str, Any], cal: dict[str, Any], reading: LeafReading,
                 rb_sigma: float | None) -> Sigmas:
    """Uncertainty sigmas with the r_bH spread of the primary method (if known)."""
    s = get_sigmas(cfg, cal, get_light_source(cfg, cal, reading.light_source))
    if rb_sigma is None:
        return s
    return replace(s, rb_fraction=rb_sigma)


# ---------------------------------------------------------------- sweep

def sweep(reading: LeafReading, cfg: dict[str, Any], cal: dict[str, Any], sigmas: Sigmas,
          n: int | None = None, seed: int = 0) -> pd.DataFrame:
    """One row per true g_s on the uncertainty grid (SWEEP_COLUMNS). g_s in mol m-2 s-1."""
    n = int(cfg["uncertainty"]["n_draws"]) if n is None else n
    rows = []
    for g in dl_grid(cfg):
        r = resolution_at(reading, float(g), cfg, cal, sigmas, n, seed)
        if r is None:
            rows.append({"g_s_true_mol": float(g), "root_found": False, "resolved": False})
            continue
        rows.append({"g_s_true_mol": float(g), "root_found": True, "leaf_temp_c": r.leaf_temp_c,
                     "g_s_mol_lo": r.g_s_mol_lo, "g_s_mol_median": r.g_s_mol_median,
                     "g_s_mol_hi": r.g_s_mol_hi, "frac_ok": r.frac_ok,
                     "relative_width": r.relative_width, "resolved": r.resolved})
    return pd.DataFrame(rows, columns=list(SWEEP_COLUMNS))


def resolved_band(sw: pd.DataFrame) -> tuple[float | None, float | None]:
    """(lowest, highest) resolved true g_s [mol m-2 s-1]; (None, None) if nothing is resolved."""
    r = sw.loc[sw["resolved"].astype(bool), "g_s_true_mol"]
    return (float(r.min()), float(r.max())) if len(r) else (None, None)


# ---------------------------------------------------------------- brackets

def synthetic_brackets(reading: LeafReading, cfg: dict[str, Any], cal: dict[str, Any]) -> list[tuple[str, LeafReading]]:
    """Stand-in brackets: forward leaf temperatures at synthetic.bracket_dry/wet_g_s_mol."""
    s = cfg["synthetic"]
    out = []
    for kind, g in ((DRY, s["bracket_dry_g_s_mol"]), (WET, s["bracket_wet_g_s_mol"])):
        t = forward_leaf_temp(reading, float(g), cfg, cal)
        if t is None:
            raise ValueError(f"no forward temperature for the {kind} bracket at g_s = {g}")
        out.append((kind, LeafReading(t, reading.air_temp_c, reading.surroundings_temp_c,
                                      reading.relative_humidity_pct, reading.pressure_hpa, reading.lux,
                                      reading.light_source, reading.r_bh, reading.r_bh_method)))
    return out


def session_brackets(sessions: list[Session], cfg: dict[str, Any], r_bh: float) -> list[tuple[str, LeafReading]]:
    """Measured brackets, one per steady window: petroleum-jelly leaf (card_temp_c in a
    vaseline_leaf session) and wet reference (wet_ref_temp_c when wet_reference_present),
    each at its own window's conditions, surroundings = air, with r_bH = r_bh [s m-1]."""
    primary = str(cfg["methods"]["primary"])
    out: list[tuple[str, LeafReading]] = []
    for s in sessions:
        m = s.metadata
        vas = str(m.get("dry_reference_type", "")).strip() == "vaseline_leaf"
        wet = m.get("wet_reference_present") is True or str(m.get("wet_reference_present")).lower() == "true"
        if not (vas or wet):
            continue
        for _, w in build_windows(s.sensor, s.events, m, cfg).iterrows():
            if not bool(w["steady_state"]):
                continue
            air = float(w["air_temp_c"])
            for kind, col, use in ((DRY, "card_temp_c", vas), (WET, "wet_ref_temp_c", wet)):
                t = float(w.get(col, math.nan))
                if use and math.isfinite(t):
                    out.append((kind, LeafReading(t, air, air, float(w["humidity_pct"]), float(w["pressure_hpa"]),
                                                  float(w["lux"]), str(m.get("light_source")), r_bh, primary)))
    return out


def bracket_table(pairs: list[tuple[str, LeafReading]], cfg: dict[str, Any], cal: dict[str, Any],
                  sigmas: Sigmas) -> pd.DataFrame:
    """Solve each bracket as a leaf (eqs. 1-8 + Monte Carlo). Expected: not resolved."""
    rows = []
    for kind, rd in pairs:
        res = solve_gs(rd, cfg, cal)
        unc = gs_uncertainty(rd, cfg, cal, sigmas=sigmas, seed=0)
        rows.append({"reference": kind, "temp_c": rd.leaf_temp_c, "air_temp_c": rd.air_temp_c,
                     "censor": res.censor, "g_s_mol": res.g_s_mol, "g_s_mol_lo": unc.g_s_mol_lo,
                     "g_s_mol_hi": unc.g_s_mol_hi, "resolved": bool(unc.resolved),
                     "expected": "unresolved", "as_expected": not unc.resolved})
    return pd.DataFrame(rows, columns=list(BRACKET_COLUMNS))


# ---------------------------------------------------------------- report

def summary_line(sw: pd.DataFrame, brackets: pd.DataFrame | None, label: str, method: str, n_rows: int) -> str:
    """One line for the paper's notes."""
    lo, hi = resolved_band(sw)
    head = f"Detection limit ({label}, {method}, conditions from {n_rows} windows): "
    if lo is None:
        body = (f"no true g_s from {sw['g_s_true_mol'].min():.3g} to {sw['g_s_true_mol'].max():.3g} "
                "mol m-2 s-1 is resolved with the current uncertainty sigmas")
    else:
        body = f"lowest resolved g_s {lo:.3g} mol m-2 s-1; resolved band {lo:.3g} to {hi:.3g}"
    if brackets is None or brackets.empty:
        tail = "; brackets: none measured yet"
    else:
        parts = []
        for kind, g in brackets.groupby("reference", sort=False):
            parts.append(f"{kind} unresolved in {int(g['as_expected'].sum())}/{len(g)}")
        tail = "; brackets: " + ", ".join(parts)
    return head + body + tail


def plot(sw: pd.DataFrame, band: tuple[float | None, float | None], cfg: dict[str, Any], path: Path,
         title: str, brackets: pd.DataFrame | None = None) -> Path:
    """Three panels sharing the true-g_s axis (log): estimate with 95 % interval, usable
    fraction, relative width. Thresholds dashed; resolved band shaded."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, blue, band_fill, shade = "#0b0b0b", "#52514e", "#2a78d6", "#b7d3f6", "#e9e8e3"
    unc = cfg["uncertainty"]
    floor = float(unc["dl_grid_min_mol"]) / 2.0     # lower bounds of 0 drawn at the axis floor
    ok = sw[sw["root_found"].astype(bool)]
    x = ok["g_s_true_mol"].to_numpy(float)
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1, 1]})
    for ax in axes:
        ax.set_xscale("log")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(ink2)
        ax.spines["bottom"].set_color(ink2)
        ax.tick_params(colors=ink2, labelsize=8)
        ax.grid(True, which="major", color="#ecebe7", linewidth=0.6)
        if band[0] is not None:
            ax.axvspan(band[0], band[1], color=shade, zorder=0, linewidth=0)

    a = axes[0]
    a.fill_between(x, np.maximum(ok["g_s_mol_lo"], floor), np.maximum(ok["g_s_mol_hi"], floor),
                   color=band_fill, linewidth=0, label="95 % interval")
    a.plot(x, np.maximum(ok["g_s_mol_median"], floor), color=blue, linewidth=2, label="median estimate")
    a.plot(x, x, color=ink2, linewidth=1, linestyle="--", label="estimate = true")
    a.set_yscale("log")
    a.set_ylabel("estimated g_s (mol m-2 s-1)", color=ink, fontsize=9)
    if band[0] is not None:
        for ax in axes:
            ax.axvline(band[0], color=ink, linewidth=1)
        a.annotate(f"detection limit {band[0]:.3g}", (band[0], a.get_ylim()[1]), xytext=(4, -12),
                   textcoords="offset points", fontsize=8, color=ink)
    a.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.92))
    if brackets is not None and not brackets.empty:
        text = "\n".join(f"{k.replace('_', ' ')}: unresolved {int(g['as_expected'].sum())}/{len(g)}"
                         for k, g in brackets.groupby("reference", sort=False))
        a.text(0.98, 0.04, text, transform=a.transAxes, ha="right", va="bottom", fontsize=8, color=ink2)

    b = axes[1]
    b.plot(x, ok["frac_ok"], color=blue, linewidth=2)
    b.axhline(unc["dl_min_resolved_fraction"], color=ink2, linewidth=1, linestyle="--")
    b.set_ylim(0, 1.05)
    b.set_ylabel("usable draws", color=ink, fontsize=9)

    c = axes[2]
    c.plot(x, ok["relative_width"], color=blue, linewidth=2)
    c.axhline(unc["max_relative_width"], color=ink2, linewidth=1, linestyle="--")
    c.set_yscale("log")
    c.set_ylabel("relative 95 % width", color=ink, fontsize=9)
    c.set_xlabel("true g_s (mol m-2 s-1)", color=ink, fontsize=9)

    fig.suptitle(title, fontsize=10, color=ink, x=0.02, ha="left")
    fig.text(0.02, 0.005, "Shaded: resolved band. Dashed: thresholds (A33). Lower bounds of 0 drawn at the "
             "axis floor. Near the cap the width shrinks only because the upper bound is held at the cap.",
             fontsize=7, color=ink2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m analysis.detection_limit",
                                description="Detection limit and resolved g_s band (T5.4).")
    p.add_argument("--results", default=None, help="default data/processed/leaf_gs_results.csv "
                   "(leaf_gs_results_synthetic.csv with --synthetic)")
    p.add_argument("--raw-dir", default=str(REPO_ROOT / "data" / "raw"))
    p.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    p.add_argument("--calibration", default=str(REPO_ROOT / "calibration" / "calibration.json"))
    p.add_argument("--out-csv", default=None, help="default data/processed/detection_limit_sweep[_synthetic].csv")
    p.add_argument("--brackets-csv", default=None,
                   help="default data/processed/detection_limit_brackets[_synthetic].csv")
    p.add_argument("--figure", default=None, help="default docs/figures/detection_limit[_synthetic].png")
    p.add_argument("--synthetic", action="store_true",
                   help="synthetic results and calibration; brackets from synthetic.bracket_*_g_s_mol")
    return p


def main(argv: list[str] | None = None) -> int:
    from ml.synthetic_data import synthetic_calibration

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    suffix = "_synthetic" if args.synthetic else ""
    label = "SYNTHETIC" if args.synthetic else "real"
    processed = REPO_ROOT / "data" / "processed"
    results_path = Path(args.results) if args.results else processed / f"leaf_gs_results{suffix}.csv"
    if not results_path.is_file():
        raise SystemExit(f"{results_path} not found; run python -m analysis.aggregate_results"
                         + (" --synthetic" if args.synthetic else "") + " first")
    cal = synthetic_calibration(cfg) if args.synthetic else load_calibration(args.calibration)
    reading, rb_sigma, n_rows = representative_reading(pd.read_csv(results_path), cfg)
    s = sweep_sigmas(cfg, cal, reading, rb_sigma)
    sw = sweep(reading, cfg, cal, s)
    if args.synthetic:
        pairs = synthetic_brackets(reading, cfg, cal)
    else:
        raw = Path(args.raw_dir)
        sessions = [load_session(x, raw) for x in list_sessions(raw)] if raw.is_dir() else []
        pairs = session_brackets([x for x in sessions if not x.synthetic], cfg, reading.r_bh)
    brackets = bracket_table(pairs, cfg, cal, s)

    out_csv = Path(args.out_csv) if args.out_csv else processed / f"detection_limit_sweep{suffix}.csv"
    br_csv = Path(args.brackets_csv) if args.brackets_csv else processed / f"detection_limit_brackets{suffix}.csv"
    fig = Path(args.figure) if args.figure else REPO_ROOT / "docs" / "figures" / f"detection_limit{suffix}.png"
    for p in (out_csv, br_csv):
        p.parent.mkdir(parents=True, exist_ok=True)
    sw.to_csv(out_csv, index=False)
    brackets.to_csv(br_csv, index=False)
    method = str(cfg["methods"]["primary"])
    plot(sw, resolved_band(sw), cfg, fig,
         f"Resolved g_s at representative conditions ({label}, {method}, r_bH {reading.r_bh:.0f} s m-1)",
         brackets)
    print(summary_line(sw, brackets, label, method, n_rows))
    print(f"wrote {out_csv}, {br_csv}, {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
