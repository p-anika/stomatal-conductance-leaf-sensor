"""Does the card's r_bH depend on its temperature difference? (T5.9, A17)

    python -m analysis.card_buoyancy_check [--synthetic]

A card in lamp light runs several degrees above air; a leaf runs within about 1 deg C. Free
convection (buoyancy) grows with the temperature difference, so the card's r_bH may be lower
than the r_bH the same shape would have at a leaf's temperature difference. card_direct uses
the card's r_bH for the leaf, so this check asks how big that transfer error is.

Input: data/processed/rb_training_set[_synthetic].csv (valid card windows, T4.3).
Group: card_id x wind setting (session median wind_ms rounded to card_buoyancy.wind_setting_step_ms).
Each session is one lamp level; a group needs card_buoyancy.min_lamp_levels of them.
Per group (weighted least squares, weights 1 / r_bh_card_sigma^2, 95 % CIs):
1. PRIMARY check: r_bH against absorbed shortwave (the exogenous driver; the lamp is set,
   the temperature follows)
       SW_abs = a_s_card x c_card x k x lux   [W m-2]
       r_bH = c0 + slope x SW_abs; sw_rel_change = slope x (max - min SW_abs) / mean r_bH
2. SECONDARY: r_bH against |T_card - T_air|. Caveat: r_bH = rho cp dT / Rn is itself computed
   from dT, so temperature noise enters both axes and can create a slope (also A42).
3. Nusselt-form fit, in conductance (eq. 10: g = g_forced + g_free, g_free ~ Gr^(1/4) ~ dT^(1/4)):
       g_bH = 1 / r_bH = a + b x |dT|^(1/4),   sigma_g = sigma_r / r_bH^2
   fractional change in r_bH from the card's median |dT| to a leaf-like dT:
       frac_change = g(dT_card) / g(dT_leaf) - 1      (positive: the leaf's r_bH is larger)
   with an interval from n_param_draws draws of (a, b) from their fitted covariance.
   PRE-REGISTERED decision (decision_log 2026-11-10): card_transfer_unreliable when
   |frac_change| at card_buoyancy.leaf_delta_t_c (1.0 deg C) exceeds
   convection.card_transfer_max_change (0.15). The value at leaf_delta_t_low_c is also reported.
4. The same fractional change predicted by the flat-plate Nusselt model (frac_change_model; f
   cancels in the ratio) for comparison.
This writes the per-(shape, wind) table only. OPEN ITEM: wiring the flag into aggregate_results
(card_direct windows of an unreliable shape x wind) must be done before real leaf results are analysed.
Output: data/processed/card_buoyancy_check[_synthetic].csv (data_label column).
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from processing import energy_balance_utils as ebu
from processing.config import get_light_source, load_calibration, load_config
from processing.physics_baseline import card_absorptivity

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent
MM_PER_M = 1000.0
QUARTER = 0.25          # free-convection exponent: Nu_free ~ Gr^(1/4) (Appendix A eq. 10)
MIN_WINDOWS_FIT = 3     # a line through 2 points has no uncertainty


def wls_line(x: np.ndarray, y: np.ndarray, sigma: np.ndarray, ci_level: float) -> dict[str, Any]:
    """Weighted least squares y = intercept + slope x, weights 1 / sigma^2 (statsmodels WLS,
    covariance scaled by the residual variance). Returns params, CI of the slope and covariance."""
    import statsmodels.api as sm
    x, y, sigma = (np.asarray(v, float) for v in (x, y, sigma))
    nan = {"intercept": math.nan, "slope": math.nan, "slope_lo": math.nan, "slope_hi": math.nan,
           "cov": np.full((2, 2), np.nan)}
    if len(x) < MIN_WINDOWS_FIT or np.ptp(x) == 0:
        return nan
    fit = sm.WLS(y, sm.add_constant(x), weights=1.0 / sigma ** 2).fit()
    ci = np.asarray(fit.conf_int(alpha=1.0 - ci_level))
    return {"intercept": float(fit.params[0]), "slope": float(fit.params[1]), "slope_lo": float(ci[1, 0]),
            "slope_hi": float(ci[1, 1]), "cov": np.asarray(fit.cov_params(), float)}


def absorbed_shortwave_wm2(df: pd.DataFrame, cfg: dict[str, Any], cal: dict[str, Any]) -> pd.Series:
    """SW_abs = a_s_card x c_card x k(light source) x lux  [W m-2] (Appendix A eqs. 1-2)."""
    a_s, _ = card_absorptivity(cfg, cal)
    k = df["light_source"].map(lambda s: get_light_source(cfg, cal, str(s)).k_wm2_per_lux)
    # a blank c_card becomes 1.0 and the group is flagged c_position_default (see run)
    c = pd.to_numeric(df["c_card"], errors="coerce").fillna(1.0)
    return a_s * c * k * pd.to_numeric(df["lux"], errors="coerce")


def wind_settings(df: pd.DataFrame, step_ms: float) -> pd.Series:
    """Session median wind rounded to step_ms, per row."""
    med = df.groupby("session_id")["wind_ms"].transform("median")
    return (med / step_ms).round() * step_ms


def nusselt_fraction(a: float, b: float, dt_card: float, dt_leaf: float) -> float:
    """g(dT_card) / g(dT_leaf) - 1 with g = a + b |dT|^(1/4)  (= r_leaf / r_card - 1)."""
    return (a + b * abs(dt_card) ** QUARTER) / (a + b * abs(dt_leaf) ** QUARTER) - 1.0


def check_group(g: pd.DataFrame, cfg: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """All numbers for one card x wind setting (see module docstring)."""
    cb = cfg["card_buoyancy"]
    ci = float(cb["ci_level"])
    leaf_dt, leaf_dt_low = float(cb["leaf_delta_t_c"]), float(cb["leaf_delta_t_low_c"])
    r = g["r_bh_card"].to_numpy(float)
    sig = g["r_bh_card_sigma"].to_numpy(float)
    dt = (g["card_temp_c"] - g["air_temp_c"]).abs().to_numpy(float)
    sw = g["sw_abs_wm2"].to_numpy(float)
    out: dict[str, Any] = {"n_windows": len(g), "n_lamp_levels": int(g["session_id"].nunique()),
                           "median_delta_t_card_c": float(np.median(dt)), "sw_abs_min_wm2": float(sw.min()),
                           "sw_abs_max_wm2": float(sw.max()), "mean_r_bh_s_m": float(r.mean())}
    if out["n_lamp_levels"] < int(cb["min_lamp_levels"]):
        return {**out, "status": "too_few_lamp_levels", "card_transfer_unreliable": None}
    if len(g) < MIN_WINDOWS_FIT:
        return {**out, "status": "too_few_windows", "card_transfer_unreliable": None}

    s = wls_line(sw, r, sig, ci)
    out.update(sw_slope=s["slope"], sw_slope_lo=s["slope_lo"], sw_slope_hi=s["slope_hi"],
               sw_rel_change=s["slope"] * (sw.max() - sw.min()) / r.mean())
    t = wls_line(dt, r, sig, ci)
    out.update(dt_slope=t["slope"], dt_slope_lo=t["slope_lo"], dt_slope_hi=t["slope_hi"])

    nf = wls_line(dt ** QUARTER, 1.0 / r, sig / r ** 2, ci)
    a, b = nf["intercept"], nf["slope"]
    dt_card = out["median_delta_t_card_c"]
    out.update(nusselt_a_m_s=a, nusselt_b_m_s_per_c4=b,
               frac_change_fit=nusselt_fraction(a, b, dt_card, leaf_dt),
               frac_change_fit_low=nusselt_fraction(a, b, dt_card, leaf_dt_low))
    cov = np.nan_to_num(nf["cov"])
    draws = rng.multivariate_normal([a, b], cov, size=int(cb["n_param_draws"]), check_valid="ignore")
    fr = (draws[:, 0] + draws[:, 1] * dt_card ** QUARTER) / (draws[:, 0] + draws[:, 1] * leaf_dt ** QUARTER) - 1
    q = [100 * (1 - ci) / 2, 100 * (1 + ci) / 2]
    out["frac_change_fit_lo"], out["frac_change_fit_hi"] = (float(v) for v in np.nanpercentile(fr, q))

    d_m = float(np.median(g[f"{cfg['physics']['characteristic_dimension']}_mm"])) / MM_PER_M
    wind, air = float(np.median(g["wind_ms"])), float(np.median(g["air_temp_c"]))
    r_card = ebu.flat_plate_rbh(d_m, wind, dt_card, air, cfg)[0]
    out.update(frac_change_model=ebu.flat_plate_rbh(d_m, wind, leaf_dt, air, cfg)[0] / r_card - 1.0,
               frac_change_model_low=ebu.flat_plate_rbh(d_m, wind, leaf_dt_low, air, cfg)[0] / r_card - 1.0)
    max_change = float(cfg["convection"]["card_transfer_max_change"])
    out.update(status="ok", max_change=max_change,
               card_transfer_unreliable=bool(abs(out["frac_change_fit"]) > max_change))
    return out


COLUMNS = ("card_id", "shape_group", "wind_setting_ms", "status", "n_windows", "n_lamp_levels",
           "sw_abs_min_wm2", "sw_abs_max_wm2", "sw_slope", "sw_slope_lo", "sw_slope_hi", "sw_rel_change",
           "dt_slope", "dt_slope_lo", "dt_slope_hi", "median_delta_t_card_c", "mean_r_bh_s_m",
           "nusselt_a_m_s", "nusselt_b_m_s_per_c4", "frac_change_fit", "frac_change_fit_lo", "frac_change_fit_hi",
           "frac_change_fit_low", "frac_change_model", "frac_change_model_low", "max_change",
           "card_transfer_unreliable", "flags")


def run(training: pd.DataFrame, cfg: dict[str, Any], cal: dict[str, Any], label: str) -> pd.DataFrame:
    """One row per card x wind setting from valid training-set rows."""
    cb = cfg["card_buoyancy"]
    rng = np.random.default_rng(int(cb["random_seed"]))
    df = training.copy()
    valid = df["valid"] if df["valid"].dtype == bool else df["valid"].astype(str).str.lower().eq("true")
    df = df[valid & df["r_bh_card"].notna() & df["wind_ms"].notna()].copy()
    rows = []
    if len(df):
        df["sw_abs_wm2"] = absorbed_shortwave_wm2(df, cfg, cal)
        df["wind_setting_ms"] = wind_settings(df, float(cb["wind_setting_step_ms"]))
        for (card, wind), g in df.groupby(["card_id", "wind_setting_ms"]):
            blank_c = pd.to_numeric(g["c_card"], errors="coerce").isna().any()
            rows.append({"card_id": card, "shape_group": g["shape_group"].iloc[0], "wind_setting_ms": wind,
                         **check_group(g, cfg, rng), "flags": "c_position_default" if blank_c else ""})
    out = pd.DataFrame(rows, columns=list(COLUMNS))
    out["card_transfer_unreliable"] = out["card_transfer_unreliable"].astype(object)
    out["data_label"] = label
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m analysis.card_buoyancy_check",
                                description="Card r_bH dependence on its temperature difference (T5.9).")
    p.add_argument("--synthetic", action="store_true", help="read rb_training_set_synthetic.csv, use the "
                   "synthetic calibration, stamp SYNTHETIC")
    p.add_argument("--training", default=None, help="default data/processed/rb_training_set[_synthetic].csv")
    p.add_argument("--calibration", default=str(REPO_ROOT / "calibration" / "calibration.json"))
    p.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "processed"))
    return p


def main(argv: list[str] | None = None) -> int:
    from ml.synthetic_data import synthetic_calibration

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    label, suffix = ("SYNTHETIC", "_synthetic") if args.synthetic else ("real", "")
    path = Path(args.training) if args.training else REPO_ROOT / "data" / "processed" / f"rb_training_set{suffix}.csv"
    if not path.is_file():
        raise SystemExit(f"{path} not found; run python -m ml.build_training_set first")
    cal = synthetic_calibration(cfg) if args.synthetic else load_calibration(args.calibration)
    out = run(pd.read_csv(path), cfg, cal, label)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"card_buoyancy_check{suffix}.csv"
    out.to_csv(dest, index=False)
    print(f"{label} data: {len(out)} card x wind groups; unreliable: "
          f"{int((out['card_transfer_unreliable'] == True).sum())}")  # noqa: E712 (object column)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
