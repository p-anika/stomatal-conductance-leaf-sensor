"""Rebuild every figure and table from data/ alone (T5.7).

    python -m analysis.make_figures [--synthetic] [--figures-only] [--root DIR]

1. Analysis steps, in dependency order (each is that module's own command, with paths under --root):
   aggregate_results (T5.1) -> detection_limit (T5.4) -> validate_gs_ground_truth (T5.6) ->
   baseline_benchmark (T5.2) -> genotype_comparison (T5.3) -> validate_rb_vaseline (T5.5) ->
   card_buoyancy_check (T5.9). A step whose inputs are missing is SKIPPED (reason recorded); a
   step that crashes for another reason is FAILED (the command then exits 1). Earlier outputs
   on disk are used when a step is skipped. --figures-only skips this part.
   Not re-run here: the ML training set and model (T4.3-T4.5, python -m ml.*): their files in
   data/processed and data/models are inputs.
2. Figures in docs/figures/ (suffix _synthetic and a SYNTHETIC title with --synthetic):
   gs_by_group, method_comparison, bland_altman (T5.6), detection_limit (T5.4), ml_cv,
   calibration_residuals, below_detection, vaseline_rb, card_buoyancy.
3. Tables: every analysis CSV copied to docs/tables/.
4. docs/tables/index[_synthetic].md lists every step, figure and table as built, skipped or missing,
   with the reason.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from processing.config import CalibrationMissing, load_config
from processing.data_io import SchemaError

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent
INK, FAINT, REF = "#1f4e79", "#b8b8b8", "#555555"     # one hue + neutral grey, as in validate_gs_ground_truth

TABLES = ("validation_gs_summary", "validation_gs_detection", "validation_gs_one_surface",
          "validation_gs_gravimetric_summary", "benchmark_primary", "benchmark_wilcoxon", "benchmark_by_shape",
          "benchmark_by_gs_range", "benchmark_by_regime", "genotype_summary", "genotype_effects", "genotype_anova",
          "genotype_mixed_model", "genotype_rank_tests", "genotype_censor_share", "genotype_robustness",
          "genotype_counts", "genotype_notes", "genotype_excluded_by_group", "detection_limit_sweep", "detection_limit_brackets",
          "vaseline_rb_summary", "vaseline_rb_cases", "card_buoyancy_check")
SKIP_ERRORS = (FileNotFoundError, SchemaError, CalibrationMissing)


@dataclass
class Ctx:
    root: Path
    synthetic: bool
    config: Path
    cfg: dict[str, Any]
    log: list[dict[str, str]] = field(default_factory=list)

    @property
    def sfx(self) -> str:
        return "_synthetic" if self.synthetic else ""

    @property
    def label(self) -> str:
        return "SYNTHETIC" if self.synthetic else "real"

    def p(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    @property
    def processed(self) -> Path:
        return self.p("data", "processed")

    @property
    def figures(self) -> Path:
        return self.p("docs", "figures")

    @property
    def tables(self) -> Path:
        return self.p("docs", "tables")

    def record(self, kind: str, name: str, status: str, detail: str = "") -> None:
        self.log.append({"kind": kind, "name": name, "status": status, "detail": detail})


# ---------------------------------------------------------------- 1. analysis steps

def _steps(c: Ctx) -> list[tuple[str, Callable[[], int]]]:
    from analysis import (aggregate_results, baseline_benchmark, card_buoyancy_check, detection_limit,
                          genotype_comparison, validate_gs_ground_truth, validate_rb_vaseline)
    syn = ["--synthetic"] if c.synthetic else []
    pr, s = c.processed, c.sfx
    res = str(pr / f"leaf_gs_results{s}.csv")
    common = {"raw": str(c.p("data", "raw")), "cards": str(c.p("data", "reference_cards.csv")),
              "cal": str(c.p("calibration", "calibration.json")),
              "model": str(c.p("data", "models", "correction_model.joblib")), "config": str(c.config)}
    gt = str(c.p("data", "validation", f"ground_truth_gs{s}.csv"))
    return [
        ("aggregate_results", lambda: aggregate_results.main(
            ["--raw-dir", common["raw"], "--processed-dir", str(pr), "--cards", common["cards"], "--calibration",
             common["cal"], "--model", common["model"], "--config", common["config"],
             "--out", res] + syn)),
        ("detection_limit", lambda: detection_limit.main(
            ["--results", res, "--raw-dir", common["raw"], "--config", common["config"], "--calibration", common["cal"],
             "--out-csv", str(pr / f"detection_limit_sweep{s}.csv"),
             "--brackets-csv", str(pr / f"detection_limit_brackets{s}.csv"),
             "--figure", str(c.figures / f"detection_limit{s}.png")] + syn)),
        ("validate_gs_ground_truth", lambda: validate_gs_ground_truth.main(
            ["--results", res, "--ground-truth", gt, "--config", common["config"], "--out-dir", str(pr),
             "--figure-dir", str(c.figures)] + syn)),
        ("baseline_benchmark", lambda: baseline_benchmark.main(
            ["--results", res, "--ground-truth", gt, "--processed-dir", str(pr), "--config", common["config"],
             "--out-dir", str(pr)] + syn)),
        ("genotype_comparison", lambda: genotype_comparison.main(
            ["--results", res, "--detection-limit-sweep", str(pr / f"detection_limit_sweep{s}.csv"),
             "--config", common["config"], "--out-dir", str(pr)] + syn)),
        ("validate_rb_vaseline", lambda: validate_rb_vaseline.main(
            ["--raw-dir", common["raw"], "--cards", common["cards"], "--calibration", common["cal"],
             "--model", common["model"], "--config", common["config"], "--out-dir", str(pr)] + syn)),
        ("card_buoyancy_check", lambda: card_buoyancy_check.main(
            ["--training", str(pr / f"rb_training_set{s}.csv"), "--calibration", common["cal"],
             "--config", common["config"], "--out-dir", str(pr)] + syn)),
    ]


def run_steps(c: Ctx) -> bool:
    """Run every analysis step; True if none FAILED (skipped steps are fine)."""
    ok = True
    for name, fn in _steps(c):
        try:
            fn()
            c.record("step", name, "built")
        except SystemExit as e:                  # the analysis commands exit with a message when inputs are missing
            if isinstance(e.code, str):
                c.record("step", name, "skipped", e.code)
            else:
                c.record("step", name, "failed", f"exit code {e.code}")
                ok = False
        except SKIP_ERRORS as e:
            c.record("step", name, "skipped", f"{type(e).__name__}: {e}")
        except Exception as e:                   # noqa: BLE001 - recorded and reported, the run continues
            logger.exception("step %s failed", name)
            c.record("step", name, "failed", f"{type(e).__name__}: {e}")
            ok = False
    return ok


# ---------------------------------------------------------------- 2. figures

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _finish(fig, ax_list, path: Path, title: str) -> Path:
    for ax in ax_list:
        ax.grid(color="#e6e6e6", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    _plt().close(fig)
    return path


def _read(c: Ctx, stem: str) -> pd.DataFrame:
    path = c.processed / f"{stem}{c.sfx}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path.name} not found")
    return pd.read_csv(path)


def fig_gs_by_group(c: Ctx) -> Path:
    """Plant means of the primary g_s per genotype x treatment, group mean with bootstrap 95 % CI."""
    from analysis import genotype_comparison as gc
    res = _read(c, "leaf_gs_results")
    oc = next(o for o in gc.outcomes(c.cfg) if o.role == "primary")
    gcfg = c.cfg["genotype_comparison"]
    w, _ = gc.prepare(res, oc, gc.campaign_detection_limit(c.processed / f"detection_limit_sweep{c.sfx}.csv"))
    fraction = gcfg["censor_substitution"][gcfg["primary_substitution"]]
    pm = gc.plant_means(w, gc.substitute(w, oc, fraction, float(c.cfg["uncertainty"]["g_s_cap_mol"])))
    rng = np.random.default_rng(int(gcfg["random_seed"]))
    alpha = 1.0 - float(gcfg["ci_level"])
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    groups = [(g, t) for g in c.cfg["enumerations"]["genotype"] for t in c.cfg["enumerations"]["water_treatment"]]
    for i, (g, t) in enumerate(groups):
        v = pm[(pm["genotype"] == g) & (pm["water_treatment"] == t)]["mean"].to_numpy(float)
        if not len(v):
            continue
        ax.scatter(np.full(len(v), i) + rng.uniform(-0.08, 0.08, len(v)), v, s=18, color=FAINT, zorder=2)
        boots = v[rng.integers(0, len(v), (int(gcfg["n_bootstrap"]), len(v)))].mean(axis=1)
        lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        ax.errorbar(i + 0.25, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], fmt="o", color=INK, capsize=3, zorder=3)
    ax.set_xticks(range(len(groups)), [f"{g}\n{t.replace('_', ' ')}" for g, t in groups], fontsize=7)
    ax.set_ylabel(f"{oc.name} g_s, plant mean (mol m$^{{-2}}$ s$^{{-1}}$)")
    ax.scatter([], [], s=18, color=FAINT, label="plant mean")
    ax.errorbar([], [], yerr=[], fmt="o", color=INK, label=f"group mean, {int(100 * gcfg['ci_level'])} % bootstrap CI")
    ax.legend(fontsize=7, frameon=False)
    return _finish(fig, [ax], c.figures / f"gs_by_group{c.sfx}.png",
                   f"g_s by genotype and treatment ({c.label}; censored at {gcfg['primary_substitution']})")


def fig_method_comparison(c: Ctx) -> Path:
    """Plant-level bias and RMSE of each method against whole-leaf ground truth (T5.6)."""
    from analysis.validate_gs_ground_truth import LEVEL_PLANT
    s = _read(c, "validation_gs_summary")
    s = s[(s["level"] == LEVEL_PLANT) & (s["n"] > 0)]
    if s.empty:
        raise FileNotFoundError("validation_gs_summary has no plant-level pairs")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    y = np.arange(len(s))
    ax.scatter(s["rmse"], y, color=INK, s=40, label="RMSE", zorder=3)
    ax.scatter(s["bias"], y, color=FAINT, s=40, marker="D", label="bias", zorder=3)
    ax.axvline(0, color=REF, linewidth=0.8)
    ax.set_yticks(y, [f"{m} (n = {int(n)} plants)" for m, n in zip(s["method"], s["n"])], fontsize=8)
    ax.set_xlabel("device − porometer (mol m$^{-2}$ s$^{-1}$)")
    ax.legend(fontsize=7, frameon=False)
    return _finish(fig, [ax], c.figures / f"method_comparison{c.sfx}.png", f"Method comparison, plant level ({c.label})")


def fig_ml_cv(c: Ctx) -> Path:
    """Leave-one-shape-out RMSE of r_bH for each candidate model, and the nested estimate vs f = 1."""
    path = c.p("data", "models", "correction_model_card.json")
    if not path.is_file():
        raise FileNotFoundError(f"{path.name} not found (run python -m ml.train_correction_model)")
    card = json.loads(path.read_text(encoding="utf-8"))
    m = card["metrics"]
    names = list(m["logo"])
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    vals = [m["logo"][n]["rmse_s_m"] for n in names] + [m["nested"]["outer_rmse_s_m"], m["nested"]["null_rmse_s_m"]]
    labels = [f"{n} (LOGO)" for n in names] + ["chosen, nested (honest)", "f = 1, nested"]
    y = np.arange(len(vals))
    ax.scatter(vals, y, color=[INK] * len(names) + [INK, FAINT], s=40, zorder=3)
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel("RMSE of r_bH (s m$^{-1}$)")
    return _finish(fig, [ax], c.figures / f"ml_cv{c.sfx}.png",
                   f"ML correction cross-validation ({card.get('n_shapes', '?')} shapes; status {card.get('status')})")


def fig_calibration_residuals(c: Ctx) -> Path:
    raise FileNotFoundError("no calibration residual table yet: thermal and light calibration (T7.x) not run")


def fig_below_detection(c: Ctx) -> Path:
    """Share of below-detection windows per genotype x treatment (primary outcome, all windows)."""
    s = _read(c, "genotype_censor_share")
    primary = c.cfg["methods"]["primary"]
    s = s[(s["outcome"] == primary) & (s["subset"] == "all")]
    if s.empty:
        raise FileNotFoundError("genotype_censor_share has no primary rows")
    order = {(g, t): i for i, (g, t) in enumerate((g, t) for g in c.cfg["enumerations"]["genotype"]
                                                   for t in c.cfg["enumerations"]["water_treatment"])}
    s = s.assign(_k=[order.get(k, len(order)) for k in zip(s["genotype"], s["water_treatment"])]).sort_values("_k")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = np.arange(len(s))
    ax.bar(x, s["share_below_detection"], color=INK, width=0.6)
    for i, (sh, n) in enumerate(zip(s["share_below_detection"], s["n_windows"])):
        ax.text(i, sh + 0.02, f"{100 * sh:.0f} % of {int(n)}", ha="center", fontsize=7, color=REF)
    ax.set_xticks(x, [f"{g}\n{t.replace('_', ' ')}" for g, t in zip(s["genotype"], s["water_treatment"])], fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("share of windows below detection")
    return _finish(fig, [ax], c.figures / f"below_detection{c.sfx}.png", f"Windows below detection ({c.label}, {primary})")


def fig_vaseline(c: Ctx) -> Path:
    """Predicted / solved r_bH for each coated leaf (T5.5)."""
    cases = _read(c, "vaseline_rb_cases")
    if cases.empty:
        raise FileNotFoundError("vaseline_rb_cases is empty")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for k, (m, color) in enumerate((("flat_plate", FAINT), ("ml_corrected", INK))):
        r = cases[f"ratio_{m}"].to_numpy(float)
        ax.scatter(np.full(len(r), k), r, color=color, s=30, zorder=3, label=m)
    ax.axhline(1.0, color=REF, linewidth=0.8, linestyle="--")
    ax.set_xticks([0, 1], ["flat_plate", "ml_corrected"])
    ax.set_ylabel("predicted / coated-leaf r_bH")
    return _finish(fig, [ax], c.figures / f"vaseline_rb{c.sfx}.png", f"Coated-leaf r_bH check ({c.label}; one point per leaf)")


def fig_card_buoyancy(c: Ctx) -> Path:
    """Fractional change in card r_bH at a leaf-like delta T, per card x wind, with threshold."""
    t = _read(c, "card_buoyancy_check")
    t = t[t["status"] == "ok"]
    if t.empty:
        raise FileNotFoundError("card_buoyancy_check has no group with enough lamp levels")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.0, 0.6 + 0.35 * len(t) + 2))
    y = np.arange(len(t))
    ax.errorbar(t["frac_change_fit"], y, xerr=[t["frac_change_fit"] - t["frac_change_fit_lo"],
                                                t["frac_change_fit_hi"] - t["frac_change_fit"]],
                fmt="o", color=INK, capsize=3, label="fit (decides)")
    ax.scatter(t["frac_change_model"], y, color=FAINT, marker="D", s=24, label="flat-plate model", zorder=3)
    mc = float(c.cfg["convection"]["card_transfer_max_change"])
    for v in (mc, -mc):
        ax.axvline(v, color=REF, linewidth=0.8, linestyle="--")
    ax.set_yticks(y, [f"{a} @ {w:.1f} m/s" for a, w in zip(t["card_id"], t["wind_setting_ms"])], fontsize=7)
    ax.set_xlabel(f"r_bH change, card ΔT → {c.cfg['card_buoyancy']['leaf_delta_t_c']} °C (fraction)")
    ax.legend(fontsize=7, frameon=False)
    return _finish(fig, [ax], c.figures / f"card_buoyancy{c.sfx}.png", f"Card buoyancy check ({c.label})")


OWN_FIGURES: dict[str, Callable[[Ctx], Path]] = {
    "gs_by_group": fig_gs_by_group, "method_comparison": fig_method_comparison, "ml_cv": fig_ml_cv,
    "calibration_residuals": fig_calibration_residuals, "below_detection": fig_below_detection,
    "vaseline_rb": fig_vaseline, "card_buoyancy": fig_card_buoyancy}
STEP_FIGURES = ("bland_altman", "detection_limit")       # drawn by the analysis steps themselves


def make_all_figures(c: Ctx) -> None:
    from analysis.genotype_comparison import NoDetectionLimit
    for name, fn in OWN_FIGURES.items():
        try:
            path = fn(c)
            c.record("figure", name, "built", str(path.relative_to(c.root)))
        except (FileNotFoundError, NoDetectionLimit, KeyError) as e:
            c.record("figure", name, "missing", str(e).strip("'\""))
    for name in STEP_FIGURES:
        path = c.figures / f"{name}{c.sfx}.png"
        if path.is_file():
            c.record("figure", name, "built", str(path.relative_to(c.root)))
        else:
            c.record("figure", name, "missing", "its analysis step has not produced it")


# ---------------------------------------------------------------- 3-4. tables and index

def copy_tables(c: Ctx) -> None:
    c.tables.mkdir(parents=True, exist_ok=True)
    for stem in TABLES:
        src = c.processed / f"{stem}{c.sfx}.csv"
        if src.is_file():
            shutil.copyfile(src, c.tables / src.name)
            c.record("table", stem, "built", f"docs/tables/{src.name}")
        else:
            c.record("table", stem, "missing", f"{src.name} not in data/processed")


def write_index(c: Ctx) -> Path:
    path = c.tables / f"index{c.sfx}.md"
    lines = [f"# Figures and tables ({c.label})", "",
             "Generated by `python -m analysis.make_figures" + (" --synthetic" if c.synthetic else "") + "`. "
             "Do not edit by hand.", ""]
    if c.synthetic:
        lines += ["**SYNTHETIC DATA: not results.**", ""]
    lines += ["| Kind | Name | Status | File or reason |", "| --- | --- | --- | --- |"]
    lines += [f"| {r['kind']} | {r['name']} | {r['status']} | {r['detail'].replace('|', '/')} |" for r in c.log]
    c.tables.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m analysis.make_figures",
                                description="Rebuild every figure and table from data/ (T5.7).")
    p.add_argument("--synthetic", action="store_true", help="synthetic inputs and outputs (_synthetic, SYNTHETIC)")
    p.add_argument("--figures-only", action="store_true", help="do not re-run the analysis steps")
    p.add_argument("--root", default=str(REPO_ROOT), help="folder holding data/, calibration/ and docs/")
    p.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    c = Ctx(Path(args.root), args.synthetic, Path(args.config), load_config(args.config))
    ok = True if args.figures_only else run_steps(c)
    make_all_figures(c)
    copy_tables(c)
    index = write_index(c)
    counts = pd.Series([r["status"] for r in c.log]).value_counts().to_dict()
    print(f"{c.label}: " + ", ".join(f"{k} {v}" for k, v in counts.items()) + f"; index {index}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
