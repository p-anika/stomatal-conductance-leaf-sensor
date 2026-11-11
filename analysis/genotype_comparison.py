"""Genotype x water-treatment comparison of g_s with the plant as the unit of replication (T5.3).

    python -m analysis.genotype_comparison [--synthetic]

Outcomes: card_direct g_s is PRIMARY; ml_corrected and flat_plate g_s, the Jones index I_g and
CWSI are SENSITIVITY analyses (I_g and CWSI from the primary method's rows, once per window).
Each is analysed on all windows ("all"); the primary is also run on forced-convection windows
only ("forced_only", A17).

1. Windows used: valid rows (I_g / CWSI: steady rows with a value). Everything else is excluded
   and counted by reason. Censored windows are KEPT (A15).
2. Censored values (the censoring-aware Tobit fit is DEFERRED; until then):
   - below_detection -> fraction x detection limit, fraction 0, 1/2 and 1 (config
     genotype_comparison.censor_substitution); above_range -> uncertainty.g_s_cap_mol.
   - detection limit = the row's g_s_mol_dl; if blank, the T5.4 campaign limit (lowest resolved
     g_s in detection_limit_sweep.csv), flag dl_campaign; if neither exists, the numeric analyses
     (mixed model, ANOVA, bootstrap, permutation) refuse with a message. Every case is counted.
   - The rank tests and the share-below-detection table need no number and always run.
   - A conclusion is robust only if it is the same under all three substitutions.
3. Plant means (mean of a plant's windows) -> PRIMARY INFERENCE (pre-registered 2026-11-10):
   - two-way ANOVA (type II): mean ~ genotype x treatment (+ block if it has >= 2 levels).
   - effect sizes with bootstrap percentile CIs (plants resampled within each group):
       drop_under_drought:<genotype>: difference = mean_drought - mean_watered [mol m-2 s-1],
           pct_drop = 100 x (1 - mean_drought / mean_watered)
       genotype_difference:<treatment>: mean_G2 - mean_G1
       difference_of_drops: pct_drop(G2) - pct_drop(G1) (and the same in mol m-2 s-1) - the
           distribution-free interaction result.
     with a permutation p-value (exact enumeration when small) for the first two kinds.
4. Mixed model (SUPPORTIVE): statsmodels MixedLM, REML, value ~ genotype x treatment (+ block),
   random intercept per plant_id, on windows; Wald CIs; convergence reported.
5. Rank tests on plant medians with below_detection tied below every measured value and
   above_range tied above: exact permutation Mann-Whitney (on mid-ranks) of treatment within each
   genotype; Kruskal-Wallis over the four groups (asymptotic).
6. Share of below_detection (and above_range) windows per genotype x treatment.
CWSI rises under stress: it is reported as a signed difference, direction labelled, no pct_drop.
Outputs: data/processed/genotype_{summary,effects,anova,mixed_model,rank_tests,censor_share,
robustness,counts,notes,windows}[_synthetic].csv; every row carries data_label.
"""
from __future__ import annotations

import argparse
import itertools
import logging
import math
import warnings
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.detection_limit import resolved_band
from processing.config import load_config

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent

BELOW, ABOVE, NONE = "below_detection", "above_range", "none"
NO_SUBSTITUTION = "none"
EXACT_TOL = 1e-12          # floating-point tolerance when comparing permutation statistics


class NoDetectionLimit(ValueError):
    """A below_detection window has neither its own nor a campaign detection limit."""


@dataclass(frozen=True)
class Outcome:
    name: str
    method: str          # rows of this method are used
    column: str          # value column
    kind: str            # gs, jones_ig or cwsi
    role: str            # primary or sensitivity
    direction: str


def outcomes(cfg: dict[str, Any]) -> list[Outcome]:
    m = cfg["methods"]
    gs_dir = "g_s falls under stress (positive pct_drop = lower under drought)"
    out = [Outcome(m["primary"], m["primary"], "g_s_mol", "gs", "primary", gs_dir)]
    out += [Outcome(m[k], m[k], "g_s_mol", "gs", "sensitivity", gs_dir) for k in ("secondary", "baseline")]
    out += [Outcome("jones_ig", m["primary"], "jones_ig", "jones_ig", "sensitivity",
                    "I_g is proportional to conductance: falls under stress"),
            Outcome("cwsi", m["primary"], "cwsi", "cwsi", "sensitivity",
                    "CWSI rises under stress (signed difference; positive = higher under drought)")]
    return out


def campaign_detection_limit(sweep_path: str | Path) -> float | None:
    """Lowest resolved true g_s [mol m-2 s-1] in a T5.4 detection_limit_sweep CSV, or None."""
    path = Path(sweep_path)
    if not path.is_file():
        return None
    sw = pd.read_csv(path)
    sw["resolved"] = sw["resolved"].astype(str).str.lower().isin(("true", "1"))
    return resolved_band(sw)[0]


def _text(s: pd.Series) -> pd.Series:
    return s.astype(object).where(s.notna(), "").astype(str).str.strip()


def _bool(s: pd.Series) -> pd.Series:
    return s if s.dtype == bool else _text(s).str.lower().isin(("true", "1"))


# ---------------------------------------------------------------- 1-2. windows and censoring

def prepare(results: pd.DataFrame, oc: Outcome, campaign_dl: float | None) -> tuple[pd.DataFrame, dict[str, int]]:
    """(windows used, counts). windows: plant_id, genotype, water_treatment, block, value,
    censor, dl, dl_source (row, campaign, none, n/a), flags, convection_regime."""
    r = results[_text(results["method"]) == oc.method].copy()
    counts: dict[str, int] = {"windows_in": len(r)}
    if oc.kind == "gs":
        keep = _bool(r["valid"])
        reasons = _text(r.loc[~keep, "reason"]).replace("", "unknown")
        censor = _text(r["censor"]).replace("", NONE)
    else:
        steady = _bool(r["steady_state"])
        value = pd.to_numeric(r[oc.column], errors="coerce")
        censor = (_text(r["jones_ig_censor"]).replace("", NONE) if oc.kind == "jones_ig"
                  else pd.Series(NONE, index=r.index))
        has = value.notna() | (censor == ABOVE)
        keep = steady & has
        reasons = pd.Series(np.where(~steady, "not_steady_state", "not_computed"), index=r.index)[~keep]
    for reason, n in reasons.value_counts().items():
        counts[f"excluded:{reason}"] = int(n)
    r, censor = r[keep], censor[keep]
    w = pd.DataFrame({"plant_id": _text(r["plant_id"]), "genotype": _text(r["genotype"]),
                      "water_treatment": _text(r["water_treatment"]),
                      "block": _text(r["block"]) if "block" in r else "",
                      "value": pd.to_numeric(r[oc.column], errors="coerce"), "censor": censor,
                      "convection_regime": _text(r["convection_regime"]) if "convection_regime" in r else ""})
    w.loc[w["censor"] != NONE, "value"] = math.nan
    w["dl"], w["dl_source"], w["flags"] = math.nan, "n/a", ""
    if oc.kind == "gs":
        row_dl = pd.to_numeric(r["g_s_mol_dl"], errors="coerce")
        below = w["censor"] == BELOW
        has_row = below & (row_dl > 0)
        w.loc[has_row, ["dl", "dl_source"]] = pd.DataFrame({"dl": row_dl[has_row], "dl_source": "row"})
        need = below & ~has_row
        if campaign_dl is not None:
            w.loc[need, ["dl", "dl_source", "flags"]] = [campaign_dl, "campaign", "dl_campaign"]
        else:
            w.loc[need, "dl_source"] = "none"
        counts.update(dl_row=int(has_row.sum()), dl_campaign=int((need & (campaign_dl is not None)).sum()),
                      dl_none=int((need & (campaign_dl is None)).sum()))
    counts.update(windows_used=len(w), n_plants=int(w["plant_id"].nunique()),
                  censored_below=int((w["censor"] == BELOW).sum()), censored_above=int((w["censor"] == ABOVE).sum()))
    return w.reset_index(drop=True), counts


def substitute(w: pd.DataFrame, oc: Outcome, fraction: float | None, cap_mol: float) -> pd.Series:
    """Numeric value per window: measured, fraction x dl (below), cap (above, g_s only).
    Raises NoDetectionLimit if a below_detection window has no limit."""
    v = w["value"].copy()
    if oc.kind != "gs":
        if (w["censor"] == ABOVE).any():
            # I_g above_range has no numeric ceiling: numeric analyses drop it (counted), rank tests keep it
            v = v[w["censor"] != ABOVE]
        return v
    below = w["censor"] == BELOW
    if (below & w["dl"].isna()).any():
        raise NoDetectionLimit(
            f"no detection limit for {int((below & w['dl'].isna()).sum())} below_detection window(s) "
            "(no row g_s_mol_dl and no T5.4 campaign limit): mixed model, ANOVA and bootstrap refused; "
            "run python -m analysis.detection_limit first")
    v[below] = float(fraction) * w.loc[below, "dl"]
    v[w["censor"] == ABOVE] = cap_mol
    return v


def plant_means(w: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    """One row per plant: mean of its windows' values (mol m-2 s-1), group labels, n_windows."""
    d = w.loc[values.index].assign(v=values)
    return (d.groupby("plant_id", as_index=False)
            .agg(genotype=("genotype", "first"), water_treatment=("water_treatment", "first"),
                 block=("block", "first"), mean=("v", "mean"), n_windows=("v", "size")))


def _use_block(pm: pd.DataFrame) -> bool:
    return bool((pm["block"] != "").all() and pm["block"].nunique() >= 2)


# ---------------------------------------------------------------- 3. plant-level inference

def anova_plant_means(pm: pd.DataFrame) -> pd.DataFrame:
    """Type II two-way ANOVA on plant means."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    formula = "mean ~ C(genotype) * C(water_treatment)" + (" + C(block)" if _use_block(pm) else "")
    table = sm.stats.anova_lm(smf.ols(formula, data=pm).fit(), typ=2)
    return (table.reset_index().rename(columns={"index": "term", "PR(>F)": "p_value"})
            [["term", "sum_sq", "df", "F", "p_value"]])


def permutation_p(a: np.ndarray, b: np.ndarray, n_permutations: int, rng: np.random.Generator) -> float:
    """Two-sided permutation p for mean(b) - mean(a). Exact enumeration when C(n, len(b)) <=
    n_permutations (p = share of arrangements at least as extreme), else Monte Carlo ((k+1)/(N+1))."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return math.nan
    pooled, nb = np.concatenate([a, b]), len(b)
    obs = abs(b.mean() - a.mean())
    total = pooled.sum()
    n = len(pooled)
    if comb(n, nb) <= n_permutations:
        hits = 0
        for idx in itertools.combinations(range(n), nb):
            sb = pooled[list(idx)].sum()
            hits += abs(sb / nb - (total - sb) / (n - nb)) >= obs - EXACT_TOL
        return hits / comb(n, nb)
    hits = 0
    for _ in range(n_permutations):
        p = rng.permutation(pooled)
        hits += abs(p[:nb].mean() - p[nb:].mean()) >= obs - EXACT_TOL
    return (hits + 1) / (n_permutations + 1)


def _contrasts(gm: dict[tuple[str, str], float], genos: list[str], ref: str, other: str) -> dict[str, tuple[float, float]]:
    """contrast -> (difference in units of the outcome, percent value)."""
    out: dict[str, tuple[float, float]] = {}
    pct: dict[str, float] = {}
    for g in genos:
        ww, dr = gm.get((g, ref), math.nan), gm.get((g, other), math.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            pct[g] = 100.0 * (1.0 - np.divide(dr, ww))       # scalars or bootstrap arrays; ww = 0 -> inf/nan
        out[f"drop_under_drought:{g}"] = (dr - ww, pct[g])
    for t in (ref, other):
        if len(genos) == 2:
            out[f"genotype_difference:{t}"] = (gm.get((genos[1], t), math.nan) - gm.get((genos[0], t), math.nan),
                                               math.nan)
    if len(genos) == 2:
        d0, d1 = out[f"drop_under_drought:{genos[0]}"][0], out[f"drop_under_drought:{genos[1]}"][0]
        out["difference_of_drops"] = (d1 - d0, pct[genos[1]] - pct[genos[0]])
    return out


def effect_sizes(pm: pd.DataFrame, oc: Outcome, cfg: dict[str, Any], rng: np.random.Generator) -> pd.DataFrame:
    """Effect sizes from plant means with bootstrap percentile CIs and permutation p-values."""
    gcfg = cfg["genotype_comparison"]
    ref = gcfg["reference_treatment"]
    trts = [t for t in cfg["enumerations"]["water_treatment"] if t != ref]
    other = trts[0]
    genos = [g for g in cfg["enumerations"]["genotype"] if g in set(pm["genotype"])]
    groups = {k: g["mean"].to_numpy(float) for k, g in pm.groupby(["genotype", "water_treatment"])}
    obs = _contrasts({k: v.mean() for k, v in groups.items()}, genos, ref, other)

    boot: dict[str, np.ndarray] = {}
    if all(len(v) >= 2 for v in groups.values()):
        n_boot = int(gcfg["n_bootstrap"])
        # all replicates at once: resample plants within each group, take each group's mean
        gm = {k: v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1) for k, v in groups.items()}
        boot = {c: np.column_stack([np.broadcast_to(np.asarray(x, float), n_boot) for x in val])
                for c, val in _contrasts(gm, genos, ref, other).items()}
    alpha = 1.0 - float(gcfg["ci_level"])
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    is_pct = oc.kind != "cwsi"
    rows = []
    for c, (diff, pct) in obs.items():
        b = boot.get(c)
        b = None if b is None else np.where(np.isfinite(b), b, np.nan)
        lo, hi = np.nanpercentile(b[:, 0], q) if b is not None else (math.nan, math.nan)
        plo, phi = np.nanpercentile(b[:, 1], q) if b is not None and is_pct and not np.isnan(b[:, 1]).all() \
            else (math.nan, math.nan)
        p = math.nan
        if c.startswith("drop_under_drought:"):
            g = c.split(":", 1)[1]
            p = permutation_p(groups.get((g, ref), []), groups.get((g, other), []), int(gcfg["n_permutations"]), rng)
        elif c.startswith("genotype_difference:"):
            t = c.split(":", 1)[1]
            p = permutation_p(groups.get((genos[0], t), []), groups.get((genos[1], t), []),
                              int(gcfg["n_permutations"]), rng)
        rows.append({"contrast": c, "direction": oc.direction, "difference": diff, "ci_lo": lo, "ci_hi": hi,
                     "pct_drop": pct if is_pct else math.nan, "ci_lo_pct": plo, "ci_hi_pct": phi,
                     "permutation_p": p})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 4. mixed model (supportive)

def mixed_model(w: pd.DataFrame, values: pd.Series, cfg: dict[str, Any]) -> pd.DataFrame:
    """MixedLM (REML) value ~ genotype x treatment (+ block), random intercept per plant_id."""
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    d = w.loc[values.index].assign(v=values.to_numpy(float))
    g0 = cfg["enumerations"]["genotype"][0]
    ref = cfg["genotype_comparison"]["reference_treatment"]
    formula = (f"v ~ C(genotype, Treatment('{g0}')) * C(water_treatment, Treatment('{ref}'))"
               + (" + C(block)" if _use_block(plant_means(w, values)) else ""))
    alpha = 1.0 - float(cfg["genotype_comparison"]["ci_level"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fit = smf.mixedlm(formula, d, groups=d["plant_id"]).fit(reml=True)
    warned = any(issubclass(x.category, ConvergenceWarning) for x in caught)
    converged = bool(fit.converged) and not warned
    ci = fit.conf_int(alpha=alpha)
    rows = [{"term": t, "estimate": float(fit.fe_params[t]), "ci_lo": float(ci.loc[t, 0]),
             "ci_hi": float(ci.loc[t, 1]), "p_value": float(fit.pvalues[t]), "converged": converged}
            for t in fit.fe_params.index]
    rows.append({"term": "plant_variance", "estimate": float(fit.cov_re.iloc[0, 0]), "ci_lo": math.nan,
                 "ci_hi": math.nan, "p_value": math.nan, "converged": converged})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 5-6. rank tests, censor share

def ordinal_values(w: pd.DataFrame) -> pd.Series:
    """Measured values; below_detection tied below the smallest, above_range tied above the largest."""
    v = w["value"].copy()
    finite = v.dropna()
    lo = finite.min() - 1.0 if len(finite) else -1.0
    hi = finite.max() + 1.0 if len(finite) else 1.0
    v[w["censor"] == BELOW] = lo
    v[w["censor"] == ABOVE] = hi
    return v


def rank_tests(w: pd.DataFrame, cfg: dict[str, Any], rng: np.random.Generator) -> pd.DataFrame:
    """Exact permutation Mann-Whitney (mid-ranks of plant medians) per genotype; Kruskal-Wallis."""
    from scipy.stats import kruskal, rankdata
    gcfg = cfg["genotype_comparison"]
    ref = gcfg["reference_treatment"]
    d = w.assign(o=ordinal_values(w))
    pm = d.groupby("plant_id", as_index=False).agg(genotype=("genotype", "first"),
                                                    water_treatment=("water_treatment", "first"), med=("o", "median"))
    rows = []
    for g, sub in pm.groupby("genotype"):
        ranks = rankdata(sub["med"])
        a = ranks[(sub["water_treatment"] == ref).to_numpy()]
        b = ranks[(sub["water_treatment"] != ref).to_numpy()]
        u = float(b.sum() - len(b) * (len(b) + 1) / 2) if len(b) else math.nan
        rows.append({"test": f"mann_whitney:{g}", "statistic": u, "n_plants": len(sub),
                     "p_value": permutation_p(a, b, int(gcfg["n_permutations"]), rng)})
    samples = [grp["med"].to_numpy() for _, grp in pm.groupby(["genotype", "water_treatment"])]
    if len(samples) >= 2 and all(len(s) for s in samples) and pm["med"].nunique() > 1:
        k = kruskal(*samples)
        rows.append({"test": "kruskal_wallis:4_groups", "statistic": float(k.statistic), "n_plants": len(pm),
                     "p_value": float(k.pvalue)})
    return pd.DataFrame(rows)


def censor_share(w: pd.DataFrame) -> pd.DataFrame:
    """Per genotype x treatment: windows, share below_detection, share above_range."""
    g = w.groupby(["genotype", "water_treatment"])
    return pd.DataFrame({"n_windows": g.size(), "share_below_detection": g["censor"].apply(lambda c: (c == BELOW).mean()),
                         "share_above_range": g["censor"].apply(lambda c: (c == ABOVE).mean())}).reset_index()


# ---------------------------------------------------------------- run

TABLES = ("summary", "effects", "anova", "mixed_model", "rank_tests", "censor_share", "robustness",
          "counts", "notes", "windows", "excluded_by_group")


def excluded_by_group(results: pd.DataFrame, oc: Outcome) -> pd.DataFrame:
    """Per genotype x treatment: windows in, windows excluded as not_steady_state and their share
    (review 2026-11-11: the steady-state test selects on the measured leaf temperature)."""
    r = results[_text(results["method"]) == oc.method]
    if r.empty:
        return pd.DataFrame(columns=["genotype", "water_treatment", "n_windows_in", "n_not_steady_state",
                                     "n_suspect_card_rbh", "share_not_steady_state", "share_suspect_card_rbh"])
    not_steady = _text(r["reason"]) == "not_steady_state" if oc.kind == "gs" else ~_bool(r["steady_state"])
    # card_direct rows of a session whose pooled card r_bH is <= 0 are kept and counted, never hidden
    suspect = _text(r["reason"]) == "suspect_card_rbh"
    g = pd.DataFrame({"genotype": _text(r["genotype"]), "water_treatment": _text(r["water_treatment"]),
                      "ns": not_steady.to_numpy(), "sc": suspect.to_numpy()}).groupby(["genotype", "water_treatment"])
    out = pd.DataFrame({"n_windows_in": g.size(), "n_not_steady_state": g["ns"].sum().astype(int),
                        "n_suspect_card_rbh": g["sc"].sum().astype(int)}).reset_index()
    out["share_not_steady_state"] = out["n_not_steady_state"] / out["n_windows_in"]
    out["share_suspect_card_rbh"] = out["n_suspect_card_rbh"] / out["n_windows_in"]
    return out


def _tag(df: pd.DataFrame, **cols: Any) -> pd.DataFrame:
    df = df.copy()
    for i, (k, v) in enumerate(cols.items()):
        df.insert(i, k, v)
    return df


def analyse(results: pd.DataFrame, oc: Outcome, subset: str, cfg: dict[str, Any], campaign_dl: float | None,
            rng: np.random.Generator) -> dict[str, list[pd.DataFrame]]:
    """Every table for one outcome and subset (lists of frames, tagged)."""
    out: dict[str, list[pd.DataFrame]] = {t: [] for t in TABLES}
    tag = {"outcome": oc.name, "role": oc.role, "subset": subset}

    def note(msg: str) -> None:
        logger.warning("%s / %s: %s", oc.name, subset, msg)
        out["notes"].append(pd.DataFrame([{**tag, "message": msg}]))

    w, counts = prepare(results, oc, campaign_dl)
    if subset == "all":
        out["excluded_by_group"].append(_tag(excluded_by_group(results, oc), **tag))
    if subset == "forced_only":
        before = len(w)
        w = w[w["convection_regime"] == "forced"].reset_index(drop=True)
        counts.update(excluded_not_forced=before - len(w), windows_used=len(w), n_plants=int(w["plant_id"].nunique()))
        counts["dl_campaign"] = int((w["dl_source"] == "campaign").sum()) if oc.kind == "gs" else 0
        counts["dl_none"] = int((w["dl_source"] == "none").sum()) if oc.kind == "gs" else 0
    if oc.kind == "jones_ig":
        counts["jones_above_range_not_in_numeric"] = int((w["censor"] == ABOVE).sum())
    out["counts"].append(_tag(pd.DataFrame(list(counts.items()), columns=["item", "n"]), **tag))
    if w.empty:
        note("no forced-convection windows (wind data needed)" if subset == "forced_only" else "no usable windows")
        out["summary"].append(pd.DataFrame([{**tag, "substitution": "", "n_plants": 0, "n_windows": 0,
                                             "status": "no data"}]))
        return out
    out["windows"].append(_tag(w.drop(columns=["convection_regime"]), **tag))
    out["rank_tests"].append(_tag(rank_tests(w, cfg, rng), **tag))
    out["censor_share"].append(_tag(censor_share(w), **tag))

    gcfg = cfg["genotype_comparison"]
    subs = dict(gcfg["censor_substitution"]) if oc.kind == "gs" else {NO_SUBSTITUTION: None}
    cap = float(cfg["uncertainty"]["g_s_cap_mol"])
    for name, fraction in subs.items():
        stag = {**tag, "substitution": name}
        try:
            values = substitute(w, oc, fraction, cap)
        except NoDetectionLimit as e:
            note(str(e))
            out["summary"].append(pd.DataFrame([{**stag, "n_plants": int(w["plant_id"].nunique()),
                                                 "n_windows": len(w), "status": f"refused: {e}"}]))
            continue
        pm = plant_means(w, values)
        status = "ok"
        try:
            out["anova"].append(_tag(anova_plant_means(pm), **stag))
        except (ValueError, np.linalg.LinAlgError) as e:
            note(f"ANOVA failed ({name}): {e}")
            status = "anova_failed"
        out["effects"].append(_tag(effect_sizes(pm, oc, cfg, rng), **stag))
        try:
            mm = mixed_model(w, values, cfg)
            if not mm["converged"].all():
                note(f"mixed model did not converge ({name}); supportive only, plant-level results stand")
            out["mixed_model"].append(_tag(mm, **stag))
        except (ValueError, np.linalg.LinAlgError) as e:
            note(f"mixed model failed ({name}): {e}")
        out["summary"].append(pd.DataFrame([{**stag, "n_plants": len(pm), "n_windows": len(values),
                                             "status": status}]))
    return out


def robustness(effects: pd.DataFrame, anova: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Per g_s outcome, subset and contrast: conclusion under each substitution; robust = identical
    under all of them. Conclusion from the bootstrap CI of the difference (or of pct for
    difference_of_drops); ANOVA interaction: p < 1 - ci_level."""
    subs = list(cfg["genotype_comparison"]["censor_substitution"])
    alpha = 1.0 - float(cfg["genotype_comparison"]["ci_level"])
    rows = []
    e = effects[effects["substitution"].isin(subs)] if len(effects) else effects
    for (o, s, c), g in (e.groupby(["outcome", "subset", "contrast"], sort=False) if len(e) else []):
        concl = {}
        for _, r in g.iterrows():
            lo, hi = (r["ci_lo_pct"], r["ci_hi_pct"]) if c == "difference_of_drops" else (r["ci_lo"], r["ci_hi"])
            concl[r["substitution"]] = ("higher" if lo > 0 else "lower" if hi < 0 else
                                        "no clear difference" if not (math.isnan(lo) or math.isnan(hi)) else "no CI")
        rows.append({"outcome": o, "subset": s, "contrast": c, **{f"conclusion_{k}": concl.get(k, "refused") for k in subs},
                     "robust": len(concl) == len(subs) and len(set(concl.values())) == 1})
    a = anova[anova["substitution"].isin(subs) & anova["term"].str.contains(":")] if len(anova) else anova
    for (o, s), g in (a.groupby(["outcome", "subset"], sort=False) if len(a) else []):
        concl = {r["substitution"]: ("interaction" if r["p_value"] < alpha else "no clear interaction")
                 for _, r in g.iterrows()}
        rows.append({"outcome": o, "subset": s, "contrast": "anova:genotype_x_treatment",
                     **{f"conclusion_{k}": concl.get(k, "refused") for k in subs},
                     "robust": len(concl) == len(subs) and len(set(concl.values())) == 1})
    return pd.DataFrame(rows, columns=["outcome", "subset", "contrast", *[f"conclusion_{k}" for k in subs], "robust"])


def run(results: pd.DataFrame, cfg: dict[str, Any], campaign_dl: float | None, label: str) -> dict[str, pd.DataFrame]:
    """All tables for every outcome; the primary outcome also on forced-convection windows."""
    rng = np.random.default_rng(int(cfg["genotype_comparison"]["random_seed"]))
    parts: dict[str, list[pd.DataFrame]] = {t: [] for t in TABLES}
    for oc in outcomes(cfg):
        for subset in (("all", "forced_only") if oc.role == "primary" else ("all",)):
            for k, frames in analyse(results, oc, subset, cfg, campaign_dl, rng).items():
                parts[k] += frames
    out = {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame(columns=["outcome", "role", "subset"]))
           for k, v in parts.items() if k != "robustness"}
    out["robustness"] = robustness(out["effects"], out["anova"], cfg)
    for df in out.values():
        df["data_label"] = label
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m analysis.genotype_comparison",
                                description="Genotype x treatment comparison, plant as the unit (T5.3).")
    p.add_argument("--synthetic", action="store_true",
                   help="read leaf_gs_results_synthetic.csv and detection_limit_sweep_synthetic.csv; stamp SYNTHETIC")
    p.add_argument("--results", default=None, help="default data/processed/leaf_gs_results[_synthetic].csv")
    p.add_argument("--detection-limit-sweep", default=None,
                   help="T5.4 sweep CSV for the campaign detection limit; default data/processed/detection_limit_sweep[_synthetic].csv")
    p.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "processed"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    label, suffix = ("SYNTHETIC", "_synthetic") if args.synthetic else ("real", "")
    processed = REPO_ROOT / "data" / "processed"
    res_path = Path(args.results) if args.results else processed / f"leaf_gs_results{suffix}.csv"
    sweep = Path(args.detection_limit_sweep) if args.detection_limit_sweep else processed / f"detection_limit_sweep{suffix}.csv"
    campaign_dl = campaign_detection_limit(sweep)
    logger.info("campaign detection limit: %s", "none" if campaign_dl is None else f"{campaign_dl:.3g} mol m-2 s-1")
    out = run(pd.read_csv(res_path), cfg, campaign_dl, label)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df.to_csv(out_dir / f"genotype_{name}{suffix}.csv", index=False)
    e = out["effects"]
    primary = cfg["methods"]["primary"]
    head = e[(e["outcome"] == primary) & (e["subset"] == "all")
             & (e["substitution"] == cfg["genotype_comparison"]["primary_substitution"])
             & e["contrast"].str.startswith("drop_under_drought")] if len(e) else e
    print(f"{label} data, primary {primary}, plant level:")
    print(head[["contrast", "pct_drop", "ci_lo_pct", "ci_hi_pct", "permutation_p"]].to_string(index=False)
          if len(head) else "  no numeric result (see genotype_notes)")
    print(f"wrote {out_dir} (genotype_*{suffix}.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
