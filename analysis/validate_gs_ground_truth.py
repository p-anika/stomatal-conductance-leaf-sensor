"""Device g_s versus porometer ground truth, whole-leaf basis (T5.6, change E4).

    python -m analysis.validate_gs_ground_truth [--synthetic]

Units: every g_s is mol m-2 s-1 per one-sided (projected) leaf area. With physics.faces = 2 the
device's g_s is BOTH faces combined (gs_calculator), so the porometer must be too. The script
refuses to run with faces != 2.

1. Ground truth (data/validation/ground_truth_gs.csv, load_ground_truth):
   - gravimetric rows are set aside: the spec does not say whether they are leaf-level or
     whole-plant (A45), so they never enter the bias/RMSE tables; they go to a separate
     trend-consistency table (step 6).
   - porometer rows with surface whole_leaf or both_summed are used as written.
   - an adaxial and an abaxial reading of the same plant_id and leaf_position taken within
     validation.surface_pair_minutes are summed (conductances of the two faces act in parallel):
         g_true = g_adaxial + g_abaxial,  time = midpoint of the two readings.
     Closest pairs are matched first and each reading is used once.
   - every face reading left over is listed with reason one_surface_only and NEVER compared.
2. Pairing: for each whole-leaf value and each method, the device rows (leaf_gs_results.csv) of
   the same plant_id (and leaf_position when both are filled) whose window midpoint is within
   validation.pair_max_minutes. The nearest VALID row is used (validity does not depend on the
   g_s value, so this does not pick on the result); if none is valid, the nearest invalid row
   is recorded. status: paired (valid, censor none), device_censored, device_invalid, no_window.
3. Statistics on pairs, difference d = g_device - g_true:
       bias = mean(d); MAE = mean|d|; RMSE = sqrt(mean d^2); sd_diff = SD(d) (n - 1)
       ordinary least squares g_device = slope x g_true + intercept; R^2 = 1 - SS_res / SS_tot
       Bland-Altman limits of agreement = bias +/- loa_z x sd_diff
   Primary level: plant means (one point per plant: mean device and mean truth over its
   pairs), because the plant is the experimental unit (E7). Window level is secondary and is
   labelled non-independent.
4. Detection table: among valid device rows with a known detection limit g_s_mol_dl, how often
   a true g_s below the limit was reported below_detection, and how often a true g_s above it
   was wrongly reported below_detection.
5. Figure: per method, device vs truth (with 1:1 and OLS line) and Bland-Altman, plant means
   in front, windows faint behind, the median detection limit as a dotted line.
6. Gravimetric trend table: per method and plant, mean gravimetric g_s and mean valid,
   uncensored device g_s within validation.gravimetric_match_hours of a weighing; across plants,
   Spearman rank correlation only (does the device rank plants the same way?). No bias.
Outputs (data/processed/, suffix _synthetic with --synthetic; every row carries data_label):
validation_gs_{summary,pairs,one_surface,detection,gravimetric_plants,gravimetric_summary}.csv
and docs/figures/bland_altman[_synthetic].png.
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from processing.config import load_config
from processing.data_io import GROUND_TRUTH_COLUMNS, load_ground_truth

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent

LEVEL_PLANT = "plant_mean (primary)"
LEVEL_WINDOW = "window (secondary, non-independent)"
WHOLE_SURFACES = ("whole_leaf", "both_summed")
FACE_SURFACES = ("adaxial", "abaxial")
STAT_KEYS = ("n", "bias", "sd_diff", "mae", "rmse", "r2", "slope", "intercept", "loa_lo", "loa_hi")
WHOLE_COLUMNS = ("truth_id", "plant_id", "leaf_position", "truth_time", "g_true_mol", "truth_source",
                 "g_adaxial_mol", "g_abaxial_mol", "notes")
PAIR_COLUMNS = ("truth_id", "plant_id", "leaf_position", "truth_time", "g_true_mol", "truth_source",
                "method", "method_role", "status", "gap_min", "session_id", "window_start_iso",
                "window_end_iso", "g_device_mol", "censor", "g_s_mol_dl", "convection_regime")
PAIRED, CENSORED, INVALID, NO_WINDOW = "paired", "device_censored", "device_invalid", "no_window"
SECONDS_PER_MINUTE = 60.0
MINUTES_PER_HOUR = 60.0
MIN_N_FIT = 3          # a line through 2 points always fits perfectly: R^2 would mean nothing
MIN_N_RANK = 3         # Spearman rho of 2 plants is always +/-1


# ---------------------------------------------------------------- checks and helpers

def check_two_sided(cfg: dict[str, Any]) -> None:
    """Raise ValueError unless physics.faces = 2 (device g_s is two faces combined, E4)."""
    faces = cfg["physics"]["faces"]
    if faces != 2:
        raise ValueError(f"physics.faces is {faces}: the device g_s is then one-sided and cannot be "
                         "compared with whole-leaf ground truth (E4). Set faces: 2.")


def _text(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return _text(s).str.lower().isin(("true", "1"))


def split_ground_truth(gt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(porometer rows, gravimetric rows) of a load_ground_truth table."""
    grav = gt["instrument"] == "gravimetric"
    return gt[~grav].reset_index(drop=True), gt[grav].reset_index(drop=True)


# ---------------------------------------------------------------- 1. whole-leaf rule (E4)

def whole_leaf_ground_truth(porometer: pd.DataFrame, surface_pair_minutes: float
                            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(whole-leaf values, leftover one-surface rows). g in mol m-2 s-1; g_true = g_ad + g_ab."""
    gt = porometer.reset_index(drop=True).copy()
    gt["leaf_position"] = _text(gt["leaf_position"])
    gt["plant_id"] = _text(gt["plant_id"])
    rows: list[dict[str, Any]] = []
    for _, r in gt[gt["surface"].isin(WHOLE_SURFACES)].iterrows():
        rows.append({"plant_id": r["plant_id"], "leaf_position": r["leaf_position"],
                     "truth_time": r["timestamp_iso"], "g_true_mol": float(r["g_s_mol"]),
                     "truth_source": r["surface"], "g_adaxial_mol": math.nan,
                     "g_abaxial_mol": math.nan, "notes": r.get("notes", "")})

    faces = gt[gt["surface"].isin(FACE_SURFACES)]
    candidates: list[tuple[float, int, int]] = []
    for _, grp in faces.groupby(["plant_id", "leaf_position"]):
        ad, ab = grp[grp["surface"] == "adaxial"], grp[grp["surface"] == "abaxial"]
        for i, a in ad.iterrows():
            for j, c in ab.iterrows():
                gap = abs((c["timestamp_iso"] - a["timestamp_iso"]).total_seconds()) / SECONDS_PER_MINUTE
                if gap <= surface_pair_minutes:
                    candidates.append((gap, i, j))
    used: set[int] = set()
    for _, i, j in sorted(candidates):           # closest first; each reading used once
        if i in used or j in used:
            continue
        used.update((i, j))
        a, c = gt.loc[i], gt.loc[j]
        rows.append({"plant_id": a["plant_id"], "leaf_position": a["leaf_position"],
                     "truth_time": a["timestamp_iso"] + (c["timestamp_iso"] - a["timestamp_iso"]) / 2,
                     "g_true_mol": float(a["g_s_mol"]) + float(c["g_s_mol"]), "truth_source": "summed_faces",
                     "g_adaxial_mol": float(a["g_s_mol"]), "g_abaxial_mol": float(c["g_s_mol"]),
                     "notes": "; ".join(x for x in (str(a.get("notes", "")), str(c.get("notes", ""))) if x)})
    one = faces.loc[[i for i in faces.index if i not in used]].copy()
    one["reason"] = "one_surface_only"
    if len(one):
        logger.warning("%d one-surface porometer reading(s) have no partner face within %.0f min; "
                       "listed, never compared (E4)", len(one), surface_pair_minutes)

    whole = pd.DataFrame(rows, columns=[c for c in WHOLE_COLUMNS if c != "truth_id"])
    if len(whole):
        whole["truth_time"] = pd.to_datetime(whole["truth_time"], utc=True)
        whole = whole.sort_values(["plant_id", "truth_time"], kind="stable").reset_index(drop=True)
    whole.insert(0, "truth_id", range(len(whole)))
    return whole, one.reset_index(drop=True)


# ---------------------------------------------------------------- 2. pairing

def device_table(results: pd.DataFrame) -> pd.DataFrame:
    """leaf_gs_results rows with parsed types and the window midpoint t_mid (UTC)."""
    d = results.copy()
    for c in ("plant_id", "leaf_position", "censor", "method", "method_role", "convection_regime", "session_id"):
        d[c] = _text(d[c]) if c in d.columns else ""
    d["valid"] = _as_bool(d["valid"])
    for c in ("g_s_mol", "g_s_mol_dl"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    start = pd.to_datetime(d["window_start_iso"], utc=True, format="ISO8601")
    end = pd.to_datetime(d["window_end_iso"], utc=True, format="ISO8601")
    d["t_mid"] = start + (end - start) / 2
    return d


def method_roles(results: pd.DataFrame) -> dict[str, str]:
    """method -> method_role, in the order methods appear in the results."""
    d = results.drop_duplicates("method")
    return dict(zip(_text(d["method"]), _text(d["method_role"]) if "method_role" in d else [""] * len(d)))


def pair_ground_truth(whole: pd.DataFrame, results: pd.DataFrame, pair_max_minutes: float) -> pd.DataFrame:
    """One row per (whole-leaf truth value, method); see module docstring step 2."""
    dev = device_table(results)
    roles = method_roles(results)
    rows: list[dict[str, Any]] = []
    for _, t in whole.iterrows():
        same_plant = dev[dev["plant_id"] == str(t["plant_id"]).strip()]
        if t["leaf_position"]:
            same_plant = same_plant[(same_plant["leaf_position"] == t["leaf_position"])
                                    | (same_plant["leaf_position"] == "")]
        gap_all = (same_plant["t_mid"] - t["truth_time"]).abs().dt.total_seconds() / SECONDS_PER_MINUTE
        for method, role in roles.items():
            base = {k: t[k] for k in ("truth_id", "plant_id", "leaf_position", "truth_time", "g_true_mol",
                                      "truth_source")}
            base.update(method=method, method_role=role)
            gap = gap_all[(same_plant["method"] == method) & (gap_all <= pair_max_minutes)]
            if gap.empty:
                rows.append({**base, "status": NO_WINDOW})
                continue
            valid_gap = gap[same_plant.loc[gap.index, "valid"]]
            pick = valid_gap.idxmin() if len(valid_gap) else gap.idxmin()
            w = same_plant.loc[pick]
            status = (PAIRED if w["censor"] in ("none", "") else CENSORED) if w["valid"] else INVALID
            rows.append({**base, "status": status, "gap_min": float(gap[pick]), "session_id": w["session_id"],
                         "window_start_iso": w["window_start_iso"], "window_end_iso": w["window_end_iso"],
                         "g_device_mol": w["g_s_mol"], "censor": w["censor"], "g_s_mol_dl": w["g_s_mol_dl"],
                         "convection_regime": w["convection_regime"]})
    return pd.DataFrame(rows, columns=list(PAIR_COLUMNS))


# ---------------------------------------------------------------- 3. statistics

def agreement_stats(device: np.ndarray, truth: np.ndarray, loa_z: float) -> dict[str, float]:
    """Agreement of device with truth (both mol m-2 s-1); d = device - truth. See module docstring."""
    device, truth = np.asarray(device, dtype=float), np.asarray(truth, dtype=float)
    n = len(device)
    out: dict[str, float] = {k: math.nan for k in STAT_KEYS}
    out["n"] = n
    if n == 0:
        return out
    d = device - truth
    out.update(bias=float(d.mean()), mae=float(np.abs(d).mean()), rmse=float(math.sqrt((d ** 2).mean())))
    if n >= 2:
        sd = float(d.std(ddof=1))
        out.update(sd_diff=sd, loa_lo=out["bias"] - loa_z * sd, loa_hi=out["bias"] + loa_z * sd)
    if n >= MIN_N_FIT and np.ptp(truth) > 0:
        slope, intercept = np.polyfit(truth, device, 1)
        ss_tot = float(((device - device.mean()) ** 2).sum())
        ss_res = float(((device - (slope * truth + intercept)) ** 2).sum())
        out.update(slope=float(slope), intercept=float(intercept),
                   r2=1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan)
    return out


def plant_means(paired: pd.DataFrame) -> pd.DataFrame:
    """One row per (method, plant_id): mean g_device_mol and g_true_mol over the plant's pairs."""
    return (paired.groupby(["method", "plant_id"], as_index=False)
            .agg(g_device_mol=("g_device_mol", "mean"), g_true_mol=("g_true_mol", "mean"),
                 n_windows=("g_device_mol", "size")))


def stats_rows(pairs: pd.DataFrame, methods: dict[str, str], loa_z: float) -> list[dict[str, Any]]:
    """Plant-level and window-level agreement rows per method, on status == paired."""
    rows = []
    for method, role in methods.items():
        p = pairs[(pairs["method"] == method) & (pairs["status"] == PAIRED)]
        pm = plant_means(p)
        for level, frame in ((LEVEL_PLANT, pm), (LEVEL_WINDOW, p)):
            s = agreement_stats(frame["g_device_mol"].to_numpy(float), frame["g_true_mol"].to_numpy(float), loa_z)
            rows.append({"method": method, "method_role": role, "level": level, **s,
                         "n_plants": int(frame["plant_id"].nunique())})
    return rows


def summary_table(pairs: pd.DataFrame, methods: dict[str, str], loa_z: float) -> pd.DataFrame:
    """stats_rows plus how many truth values were not paired, and why, per method."""
    df = pd.DataFrame(stats_rows(pairs, methods, loa_z),
                      columns=["method", "method_role", "level", *STAT_KEYS, "n_plants"])
    for status in (CENSORED, INVALID, NO_WINDOW):
        counts = pairs[pairs["status"] == status].groupby("method").size()
        df[f"n_{status}"] = df["method"].map(counts).fillna(0).astype(int)
    return df


# ---------------------------------------------------------------- 4. detection table

def detection_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per method: does the device say below_detection when the true g_s is below its limit?"""
    rows = []
    for method, p in pairs.groupby("method", sort=False):
        p = p[p["status"].isin((PAIRED, CENSORED))]
        dl = pd.to_numeric(p["g_s_mol_dl"], errors="coerce")
        known = p[dl.notna()]
        below = known["g_true_mol"] < dl[dl.notna()]
        said_below = known["censor"] == "below_detection"
        n_below = int(below.sum())
        rows.append({"method": method, "n_truth_below_dl": n_below,
                     "n_device_below_detection": int((below & said_below).sum()),
                     "fraction_correct": (below & said_below).sum() / n_below if n_below else math.nan,
                     "n_truth_above_dl": int((~below).sum()),
                     "n_false_below_detection": int((~below & said_below).sum()),
                     "n_dl_unknown": int(dl.isna().sum())})
    return pd.DataFrame(rows, columns=["method", "n_truth_below_dl", "n_device_below_detection", "fraction_correct",
                                       "n_truth_above_dl", "n_false_below_detection", "n_dl_unknown"])


# ---------------------------------------------------------------- 6. gravimetric trend table

def gravimetric_trend(grav: pd.DataFrame, results: pd.DataFrame, match_hours: float
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-plant table, per-method Spearman summary). No bias or RMSE: the gravimetric basis is
    not defined as leaf-level (A45), so only the ranking of plants is compared."""
    dev = device_table(results)
    dev = dev[dev["valid"] & dev["censor"].isin(("none", "")) & dev["g_s_mol"].notna()]
    roles = method_roles(results)
    rows = []
    for method in roles:
        dm = dev[dev["method"] == method]
        for plant, g in grav.groupby(_text(grav["plant_id"])):
            d = dm[dm["plant_id"] == plant]
            near = pd.Series(False, index=d.index)
            for t in g["timestamp_iso"]:
                near |= (d["t_mid"] - t).abs().dt.total_seconds() <= match_hours * MINUTES_PER_HOUR * SECONDS_PER_MINUTE
            rows.append({"method": method, "plant_id": plant, "g_grav_mol": float(g["g_s_mol"].mean()),
                         "g_device_mol": float(d.loc[near, "g_s_mol"].mean()) if near.any() else math.nan,
                         "n_weighings": len(g), "n_windows": int(near.sum())})
    per_plant = pd.DataFrame(rows, columns=["method", "plant_id", "g_grav_mol", "g_device_mol",
                                            "n_weighings", "n_windows"])
    summary = []
    for method in roles:
        p = per_plant[(per_plant["method"] == method)].dropna(subset=["g_device_mol"])
        rho = math.nan
        if len(p) >= MIN_N_RANK:
            from scipy.stats import spearmanr
            rho = float(spearmanr(p["g_grav_mol"], p["g_device_mol"]).statistic)
        summary.append({"method": method, "n_plants": len(p), "spearman_rho": rho})
    return per_plant, pd.DataFrame(summary, columns=["method", "n_plants", "spearman_rho"])


# ---------------------------------------------------------------- 5. figure

def plot_bland_altman(pairs: pd.DataFrame, summary: pd.DataFrame, methods: Sequence[str], path: Path,
                      label: str) -> Path:
    """Two rows per method column: device vs truth, and Bland-Altman (difference vs mean)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, faint, ref = "#1f4e79", "#b8b8b8", "#555555"
    methods = list(methods) or ["(none)"]
    fig, axes = plt.subplots(2, len(methods), figsize=(4.2 * len(methods), 7.6), squeeze=False)
    for k, method in enumerate(methods):
        ax_s, ax_b = axes[0, k], axes[1, k]
        p = pairs[(pairs["method"] == method) & (pairs["status"] == PAIRED)]
        pm = plant_means(p)
        dl = pd.to_numeric(pairs.loc[(pairs["method"] == method) & pairs["status"].isin((PAIRED, CENSORED)),
                                     "g_s_mol_dl"], errors="coerce").median()
        ax_s.set_title(f"{method}\n({len(pm)} plants, {len(p)} windows)", fontsize=10)
        for ax in (ax_s, ax_b):
            ax.grid(color="#e6e6e6", linewidth=0.6)
            ax.set_axisbelow(True)
        if p.empty:
            for ax in (ax_s, ax_b):
                ax.text(0.5, 0.5, "no pairs", ha="center", va="center", transform=ax.transAxes, color=ref)
            continue
        ax_s.scatter(p["g_true_mol"], p["g_device_mol"], s=14, color=faint, label="window (non-independent)")
        ax_s.scatter(pm["g_true_mol"], pm["g_device_mol"], s=36, color=ink, edgecolor="white",
                     linewidth=1, label="plant mean (primary)")
        hi = float(np.nanmax([p["g_true_mol"].max(), p["g_device_mol"].max()])) * 1.05
        ax_s.plot([0, hi], [0, hi], color=ref, linewidth=1, linestyle="--", label="1:1")
        s = summary[(summary["method"] == method) & (summary["level"] == LEVEL_PLANT)].iloc[0]
        if not math.isnan(s["slope"]):
            ax_s.plot([0, hi], [s["intercept"], s["intercept"] + s["slope"] * hi], color=ink, linewidth=1.5,
                      label=f"fit: {s['slope']:.2f}x + {s['intercept']:.3f}")
        ax_s.set_xlabel("ground truth g_s (mol m$^{-2}$ s$^{-1}$)")
        ax_s.set_ylabel("device g_s (mol m$^{-2}$ s$^{-1}$)")

        ax_b.scatter((p["g_true_mol"] + p["g_device_mol"]) / 2, p["g_device_mol"] - p["g_true_mol"],
                     s=14, color=faint, label="window (non-independent)")
        ax_b.scatter((pm["g_true_mol"] + pm["g_device_mol"]) / 2, pm["g_device_mol"] - pm["g_true_mol"],
                     s=36, color=ink, edgecolor="white", linewidth=1, label="plant mean (primary)")
        ax_b.axhline(0, color=ref, linewidth=0.8)
        if not math.isnan(s["bias"]):
            ax_b.axhline(s["bias"], color=ink, linewidth=1.5, label=f"bias {s['bias']:+.3f}")
        if not math.isnan(s["loa_lo"]):
            for y in (s["loa_lo"], s["loa_hi"]):
                ax_b.axhline(y, color=ink, linewidth=1, linestyle="--")
            ax_b.plot([], [], color=ink, linestyle="--", label="limits of agreement")
        ax_b.set_xlabel("mean of device and truth (mol m$^{-2}$ s$^{-1}$)")
        ax_b.set_ylabel("device − truth (mol m$^{-2}$ s$^{-1}$)")
        if not math.isnan(dl):
            for ax in (ax_s, ax_b):
                ax.axvline(dl, color=ref, linewidth=1, linestyle=":", label=f"detection limit {dl:.3f}")
        for ax in (ax_s, ax_b):
            ax.legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle(f"Device vs whole-leaf porometer g_s — {label}", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- run and command line

def run(results: pd.DataFrame, gt: pd.DataFrame, cfg: dict[str, Any], label: str) -> dict[str, pd.DataFrame]:
    """Every table of steps 1-4 and 6. label: SYNTHETIC or real, copied into a data_label column."""
    check_two_sided(cfg)
    vc = cfg["validation"]
    porometer, grav = split_ground_truth(gt)
    whole, one = whole_leaf_ground_truth(porometer, float(vc["surface_pair_minutes"]))
    pairs = pair_ground_truth(whole, results, float(vc["pair_max_minutes"]))
    methods = method_roles(results)
    grav_plants, grav_summary = gravimetric_trend(grav, results, float(vc["gravimetric_match_hours"]))
    out = {"summary": summary_table(pairs, methods, float(vc["loa_z"])), "pairs": pairs, "one_surface": one,
           "detection": detection_table(pairs), "gravimetric_plants": grav_plants,
           "gravimetric_summary": grav_summary}
    for df in out.values():
        df["data_label"] = label
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m analysis.validate_gs_ground_truth",
                                description="Device g_s vs whole-leaf porometer ground truth (T5.6, E4).")
    p.add_argument("--synthetic", action="store_true",
                   help="read leaf_gs_results_synthetic.csv and ground_truth_gs_synthetic.csv; "
                        "stamp SYNTHETIC on every output")
    p.add_argument("--results", default=None, help="default data/processed/leaf_gs_results[_synthetic].csv")
    p.add_argument("--ground-truth", default=None,
                   help="default data/validation/ground_truth_gs[_synthetic].csv")
    p.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "processed"))
    p.add_argument("--figure-dir", default=str(REPO_ROOT / "docs" / "figures"))
    return p


def default_paths(synthetic: bool, results: str | None, ground_truth: str | None) -> tuple[Path, Path]:
    suffix = "_synthetic" if synthetic else ""
    return (Path(results) if results else REPO_ROOT / "data" / "processed" / f"leaf_gs_results{suffix}.csv",
            Path(ground_truth) if ground_truth else REPO_ROOT / "data" / "validation" / f"ground_truth_gs{suffix}.csv")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    label, suffix = ("SYNTHETIC", "_synthetic") if args.synthetic else ("real", "")
    res_path, gt_path = default_paths(args.synthetic, args.results, args.ground_truth)
    results = pd.read_csv(res_path, keep_default_na=True)
    out = run(results, load_ground_truth(gt_path), cfg, label)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df.to_csv(out_dir / f"validation_gs_{name}{suffix}.csv", index=False)
    fig = plot_bland_altman(out["pairs"], out["summary"], list(method_roles(results)),
                            Path(args.figure_dir) / f"bland_altman{suffix}.png", label)
    s = out["summary"]
    print(f"{label} data: {len(out['pairs']['truth_id'].unique())} whole-leaf truth values, "
          f"{len(out['one_surface'])} one-surface readings not compared")
    print(s[s["level"] == LEVEL_PLANT][["method", "n", "bias", "mae", "rmse", "r2", "slope", "intercept"]]
          .to_string(index=False))
    print(f"wrote {out_dir} (validation_gs_*{suffix}.csv) and {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
