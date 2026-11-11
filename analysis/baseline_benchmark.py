"""Compare each r_b method with whole-leaf ground truth, split by shape, g_s range and regime (T5.2).

    python -m analysis.baseline_benchmark [--synthetic]

Pairing, the whole-leaf rule (E4) and every statistic come from analysis.validate_gs_ground_truth
(same definitions, same plant-mean primary level). On top of that:

1. Shape: each paired window gets the leaf geometry its photos give, using the SAME photo pairing
   as build_windows (measurement_windows.attach_photo_features: photos inside the window -> median,
   else the nearest photo within windows.max_photo_gap_sec). No photo -> shape_status shape_unknown.
2. Shape deviation of a window from the median leaf, in robust z units (MAD = median absolute
   deviation, x 1.4826 so it equals the SD for normal data):
       z_sol = (solidity - median) / MAD_sol,  z_ar = (aspect_ratio - median) / MAD_ar
       deviation = sqrt(z_sol^2 + z_ar^2)
   Medians and MADs are over the distinct paired windows with known shape. Scaling is needed
   because aspect ratio (about 1.5-3) varies far more than solidity (about 0.8-1).
3. Splits (validation.n_split_bins quantile bins, q1 = lowest): shape_bin on deviation (q1 = most
   typical), gs_bin on the true g_s, and convection regime. DESCRIPTIVE ONLY: n and statistics,
   no p-values (owner decision 2026-11-10).
4. The flat-plate baseline is judged on forced-convection windows only (spec T5.2), except in
   the by-regime table, which shows every regime.
5. PRE-REGISTERED primary analysis (decision_log 2026-11-10, before any real data):
   a. plant-level RMSE per method on forced-convection windows;
   b. two-sided paired Wilcoxon signed-rank test of ML vs flat-plate on plant-level mean
      absolute error (forced windows; plants with both methods paired on the same truth values).
   A window-level Wilcoxon is also run and labelled secondary, non-independent.
Outputs (data/processed/, suffix _synthetic with --synthetic; every row carries data_label):
benchmark_{primary,wilcoxon,by_shape,by_gs_range,by_regime}.csv
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from analysis import validate_gs_ground_truth as v
from processing.config import load_config
from processing.data_io import load_ground_truth, load_photo_features
from processing.measurement_windows import attach_photo_features

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent

# MAD x this = SD for normal data: half of a normal lies within 0.6745 SD of the median,
# so SD = MAD / Phi^-1(0.75) = 1.4826 MAD (derived, not typed in).
MAD_TO_SD = 1.0 / float(norm.ppf(0.75))
FORCED = "forced"
ML, FLAT = "ml_corrected", "flat_plate"
WINDOW_KEY = ["session_id", "window_start_iso", "window_end_iso"]


# ---------------------------------------------------------------- 1-2. shape

def attach_shape(pairs: pd.DataFrame, processed_dir: str | Path, max_photo_gap_sec: float) -> pd.DataFrame:
    """Add solidity, aspect_ratio, shape_status to pairs, using attach_photo_features per session."""
    out = pairs.copy()
    out["solidity"], out["aspect_ratio"], out["shape_status"] = math.nan, math.nan, "shape_unknown"
    windows = out.dropna(subset=["session_id", "window_start_iso"]).drop_duplicates(WINDOW_KEY)[WINDOW_KEY]
    for session, w in windows.groupby("session_id"):
        path = Path(processed_dir) / str(session) / "photo_features.csv"
        if not path.is_file():
            continue
        w = w.assign(reason="ok", flags="")       # only geometry is wanted, not a new reason
        # leaf_moved_range_frac = inf: the leaf_moved flag belongs to T3.4, not to this join
        geo = attach_photo_features(w, load_photo_features(path), max_photo_gap_sec, math.inf)
        for _, g in geo.iterrows():
            hit = (out["session_id"] == session) & (out["window_start_iso"] == g["window_start_iso"])
            for c in ("solidity", "aspect_ratio"):
                out.loc[hit, c] = g.get(c, math.nan)
    known = out["solidity"].notna() & out["aspect_ratio"].notna()
    out.loc[known, "shape_status"] = "ok"
    return out


def _robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    scale = MAD_TO_SD * (x - med).abs().median()
    if not scale > 0:
        scale = x.std(ddof=1)
    if not scale > 0:
        return pd.Series(0.0, index=x.index)
    return (x - med) / scale


def shape_deviation(geom: pd.DataFrame) -> pd.Series:
    """sqrt(z_solidity^2 + z_aspect^2) with robust z (median, MAD); dimensionless."""
    return np.sqrt(_robust_z(geom["solidity"]) ** 2 + _robust_z(geom["aspect_ratio"]) ** 2)


# ---------------------------------------------------------------- 3-4. splits

def _qbins(x: pd.Series, n_bins: int) -> pd.Series:
    labels = [f"q{i + 1}" for i in range(n_bins)]
    return pd.qcut(x.rank(method="first"), n_bins, labels=labels).astype(str)


def add_split_labels(pairs: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """Add shape_deviation, shape_bin, gs_bin and regime (forced / mixed / free / unknown)."""
    out = pairs.copy()
    out["shape_deviation"], out["shape_bin"] = math.nan, "shape_unknown"
    known = out[(out["shape_status"] == "ok") & out["session_id"].notna()]
    win = known.drop_duplicates(WINDOW_KEY)
    if len(win) >= n_bins:
        dev = shape_deviation(win[["solidity", "aspect_ratio"]].astype(float))
        table = win[WINDOW_KEY].assign(_dev=dev.to_numpy(), _bin=_qbins(dev, n_bins).to_numpy())
        merged = out[WINDOW_KEY].merge(table, on=WINDOW_KEY, how="left")
        out["shape_deviation"] = merged["_dev"].to_numpy()
        out["shape_bin"] = merged["_bin"].fillna("shape_unknown").to_numpy()
    truth = out.drop_duplicates("truth_id")[["truth_id", "g_true_mol"]]
    if len(truth) >= n_bins:
        out["gs_bin"] = out["truth_id"].map(dict(zip(truth["truth_id"], _qbins(truth["g_true_mol"], n_bins))))
    else:
        out["gs_bin"] = "too_few"
    out["regime"] = v._text(out["convection_regime"]).replace("", "unknown")
    return out


def judged_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Drop flat_plate rows outside forced convection (spec T5.2: baseline judged on forced only)."""
    regime = v._text(pairs["convection_regime"])
    return pairs[(pairs["method"] != FLAT) | (regime == FORCED)].reset_index(drop=True)


def split_table(pairs: pd.DataFrame, by: str, loa_z: float) -> pd.DataFrame:
    """Descriptive statistics per bin of column `by` x method x level. No p-values."""
    methods = v.method_roles(pairs)
    frames = []
    for b, g in pairs.groupby(by, sort=True):
        t = pd.DataFrame(v.stats_rows(g, methods, loa_z))
        t.insert(0, by, b)
        frames.append(t)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[by, "method", "level"])


# ---------------------------------------------------------------- 5. pre-registered primary analysis

def primary_rmse(pairs: pd.DataFrame, loa_z: float) -> pd.DataFrame:
    """Plant-level statistics (RMSE is the headline) per method, forced-convection windows only."""
    forced = pairs[v._text(pairs["convection_regime"]) == FORCED]
    t = pd.DataFrame(v.stats_rows(forced, v.method_roles(pairs), loa_z))
    t = t[t["level"] == v.LEVEL_PLANT].reset_index(drop=True)
    t.insert(2, "subset", "forced convection")
    return t


def wilcoxon_ml_vs_flat(pairs: pd.DataFrame, level: str) -> dict[str, Any]:
    """Two-sided paired Wilcoxon of |error| ML minus flat plate, forced windows, same truth values.
    level 'plant': one value per plant (mean |error|), primary. level 'window': secondary."""
    p = pairs[(pairs["status"] == v.PAIRED) & (v._text(pairs["convection_regime"]) == FORCED)]
    p = p.assign(abs_err=(p["g_device_mol"] - p["g_true_mol"]).abs())
    wide = p[p["method"].isin((ML, FLAT))].pivot_table(index=["truth_id", "plant_id"], columns="method",
                                                      values="abs_err").dropna()
    if level == "plant":
        wide = wide.groupby(level="plant_id").mean()
        label = "plant mean |error| (primary, pre-registered)"
    else:
        label = "window |error| (secondary, non-independent)"
    n = len(wide)
    out = {"comparison": f"{ML} - {FLAT}", "level": label, "label": label, "n": n,
           "median_diff_abs_error": math.nan, "statistic": math.nan, "p_value": math.nan}
    if n == 0 or ML not in wide or FLAT not in wide:
        return out
    diff = wide[ML] - wide[FLAT]
    out["median_diff_abs_error"] = float(diff.median())
    if (diff != 0).sum() >= 1:
        from scipy.stats import wilcoxon
        r = wilcoxon(wide[ML], wide[FLAT], alternative="two-sided")
        out.update(statistic=float(r.statistic), p_value=float(r.pvalue))
    return out


# ---------------------------------------------------------------- run and command line

def run(results: pd.DataFrame, gt: pd.DataFrame, cfg: dict[str, Any], processed_dir: str | Path,
        label: str) -> dict[str, pd.DataFrame]:
    v.check_two_sided(cfg)
    vc = cfg["validation"]
    loa_z, n_bins = float(vc["loa_z"]), int(vc["n_split_bins"])
    porometer, _ = v.split_ground_truth(gt)            # gravimetric: never in error tables (A45)
    whole, _ = v.whole_leaf_ground_truth(porometer, float(vc["surface_pair_minutes"]))
    pairs = v.pair_ground_truth(whole, results, float(vc["pair_max_minutes"]))
    pairs = attach_shape(pairs, processed_dir, float(cfg["windows"]["max_photo_gap_sec"]))
    pairs = add_split_labels(pairs, n_bins)
    judged = judged_pairs(pairs)
    out = {"primary": primary_rmse(pairs, loa_z),
           "wilcoxon": pd.DataFrame([wilcoxon_ml_vs_flat(pairs, "plant"), wilcoxon_ml_vs_flat(pairs, "window")]),
           "by_shape": split_table(judged, "shape_bin", loa_z),
           "by_gs_range": split_table(judged, "gs_bin", loa_z),
           "by_regime": split_table(pairs, "regime", loa_z)}
    for df in out.values():
        df["data_label"] = label
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m analysis.baseline_benchmark",
                                description="Method benchmark against whole-leaf ground truth (T5.2).")
    p.add_argument("--synthetic", action="store_true",
                   help="read leaf_gs_results_synthetic.csv and ground_truth_gs_synthetic.csv; "
                        "stamp SYNTHETIC on every output")
    p.add_argument("--results", default=None, help="default data/processed/leaf_gs_results[_synthetic].csv")
    p.add_argument("--ground-truth", default=None, help="default data/validation/ground_truth_gs[_synthetic].csv")
    p.add_argument("--processed-dir", default=str(REPO_ROOT / "data" / "processed"),
                   help="where <session>/photo_features.csv files are")
    p.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "processed"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    label, suffix = ("SYNTHETIC", "_synthetic") if args.synthetic else ("real", "")
    res_path, gt_path = v.default_paths(args.synthetic, args.results, args.ground_truth)
    out = run(pd.read_csv(res_path), load_ground_truth(gt_path), cfg, args.processed_dir, label)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df.to_csv(out_dir / f"benchmark_{name}{suffix}.csv", index=False)
    print(f"{label} data. Primary (pre-registered): plant-level, forced convection")
    print(out["primary"][["method", "n", "rmse", "bias", "mae"]].to_string(index=False))
    print(out["wilcoxon"][["level", "n", "median_diff_abs_error", "p_value"]].to_string(index=False))
    print(f"wrote {out_dir} (benchmark_*{suffix}.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
