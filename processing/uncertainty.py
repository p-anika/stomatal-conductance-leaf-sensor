"""Monte Carlo uncertainty and detection limit for g_s, and card r_b labels (T1.5, A12, A15).

Error model per draw (n draws, numpy default_rng(seed)):
    T_leaf = T_leaf_meas + sigma_leaf z1
    T_air  = T_air_meas  + sigma_air (rho z1 + sqrt(1 - rho^2) z2)   (shared thermometer)
    RH     = clip(RH_meas + sigma_rh z3, 0, 100)                     (true RH cannot leave 0-100)
    E_s    = E_s_meas x exp(sigma_lux z4)                            lognormal (owner, deviation
    r_bH   = r_bH_meas x exp(sigma_rb z5)                            from spec Gaussian)
Each draw is re-solved with eqs. 1-8 (solve_core, checked against gs_calculator.solve_gs).
Pressure and surroundings temperature are not perturbed.

Draw outcomes and the value used for percentiles (mol m-2 s-1):
    VPD <= 0      -> excluded from percentiles, counted, and counts as NOT resolved
    lambdaE <= 0  -> 0 (closed stomata)
    r_s <= 0      -> g_s_cap_mol (top of measurable range)
    otherwise     -> min(g_s, g_s_cap_mol)
An upper bound equal to the cap means "not bounded above" (hi_at_cap = True).

Resolved (a label, never a filter): fraction of ALL draws that are OK >= dl_min_resolved_fraction
and (p97.5 - p2.5) / median < max_relative_width and p97.5 < g_s_cap_mol.
Detection limit g_s_dl: on a log grid of true g_s (dl_grid_min_mol to g_s_cap_mol), find the
leaf temperature each would give under this window's conditions, apply the same draws, and
return the smallest true g_s that is resolved. None if no grid value is resolved.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.optimize import brentq

from processing.config import (
    LightSource,
    _calibrated,
    get_light_source,
    get_position_factor,
)
from processing.energy_balance_utils import (
    lux_to_irradiance_wm2,
    molar_density_mol_m3,
    net_radiation_wm2,
    psychrometric_constant_pa,
    rho_cp_j_m3_k,
)
from processing.gs_calculator import ConductanceResult, LeafReading, solve_gs
from processing.physics_baseline import (
    ReferenceCardReading,
    card_absorptivity,
    card_emissivity,
    solve_card_rbh,
)

# Per-draw status codes.
OK, BELOW, ABOVE, VPD = 0, 1, 2, 3

# Percentiles: 2.5 / 50 / 97.5 for the 95 % interval; 15.87 / 84.13 are +-1 sigma of a normal.
_P_LO, _P_MID, _P_HI = 2.5, 50.0, 97.5
_P_M1, _P_P1 = 15.87, 84.13
# Search span for the forward leaf temperature (deg C either side of air); numerical only.
_FORWARD_SPAN_C = 40.0


# ---------------------------------------------------------------- sigmas

@dataclass(frozen=True)
class Sigmas:
    """1-sigma errors. Temperatures deg C, RH %, others are fractions (spread of the log
    for lux and r_b; Gaussian fraction for card Rn). sources: (config key, origin) pairs."""
    leaf_temp_c: float
    air_temp_c: float
    rh_pct: float
    lux_fraction: float
    rb_fraction: float
    rn_card_fraction: float
    rho_leaf_air: float
    sources: tuple[tuple[str, str], ...]

    @property
    def placeholder(self) -> bool:
        return any(src == "config_placeholder" for _, src in self.sources)


def get_sigmas(cfg: dict[str, Any], cal: dict[str, Any], light: LightSource) -> Sigmas:
    """Each sigma from calibration.json `uncertainty.<key>`, else config (a placeholder).

    The light sigma key is sigma_lux_fraction when k is calibrated, otherwise
    sigma_lux_fraction_uncalibrated.
    """
    lux_key = ("sigma_lux_fraction" if light.k_source == "calibration"
               else "sigma_lux_fraction_uncalibrated")
    keys = {"leaf_temp_c": "sigma_leaf_temp_c", "air_temp_c": "sigma_air_temp_c",
            "rh_pct": "sigma_rh_pct", "lux_fraction": lux_key, "rb_fraction": "sigma_rb_fraction",
            "rn_card_fraction": "sigma_rn_card_fraction", "rho_leaf_air": "rho_leaf_air_temp"}
    values: dict[str, float] = {}
    sources = []
    for field, key in keys.items():
        v = _calibrated(cal, "uncertainty", key)
        src = "calibration"
        if v is None:
            v = cfg["uncertainty"][key]
            src = "config_placeholder"
        v = float(v)
        if field == "rho_leaf_air":
            if not -1.0 <= v <= 1.0:
                raise ValueError(f"{key} must be in [-1, 1], got {v}")
        elif not (math.isfinite(v) and v >= 0):
            raise ValueError(f"{key} must be a non-negative number, got {v}")
        values[field] = v
        sources.append((key, src))
    return Sigmas(**values, sources=tuple(sources))


# ---------------------------------------------------------------- vectorised solver

def esat_pa_array(temp_c: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Tetens (eq. 5) for arrays: e_s = a exp(b T / (T + c)), T deg C, Pa."""
    p = cfg["physics"]
    return p["tetens_a_pa"] * np.exp(p["tetens_b"] * temp_c / (temp_c + p["tetens_c_c"]))


def solve_core(t_s: np.ndarray, t_a: np.ndarray, t_sur: float | None, rh: np.ndarray,
               p_hpa: float, e_s_wm2: np.ndarray, a_s: float, r_bh: np.ndarray,
               cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """Eqs. 1-8 on arrays, same order of checks as gs_calculator.solve_gs.

    Returns rn, latent [W m-2], vpd [Pa], g_mol [mol m-2 s-1, nan unless status OK], status.
    """
    phys = cfg["physics"]
    lw = bool(phys["include_longwave"])
    rho_cp = rho_cp_j_m3_k(cfg)
    rn = net_radiation_wm2(e_s_wm2, a_s, t_s, t_sur if lw else t_a,
                           phys["leaf_emissivity_longwave"], phys["faces"], lw, cfg)
    latent = rn - rho_cp * (t_s - t_a) / r_bh
    vpd = esat_pa_array(t_s, cfg) - esat_pa_array(t_a, cfg) * rh / 100.0
    gamma = psychrometric_constant_pa(p_hpa, cfg)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_s = rho_cp * vpd / (gamma * latent) - phys["rbw_to_rbh_ratio"] * r_bh
        g_mol = molar_density_mol_m3(p_hpa, t_a, cfg) / r_s
    status = np.select([vpd <= 0, latent <= 0, r_s <= 0], [VPD, BELOW, ABOVE], OK)
    g_mol = np.where(status == OK, g_mol, np.nan)
    return {"rn": rn, "latent": latent, "vpd": vpd, "g_mol": g_mol, "status": status}


# ---------------------------------------------------------------- draws

@dataclass(frozen=True)
class GsDraws:
    """All n draws (arrays). g_mol is nan unless status == OK."""
    t_leaf: np.ndarray
    t_air: np.ndarray
    rh: np.ndarray
    e_s: np.ndarray
    r_bh: np.ndarray
    g_mol: np.ndarray
    status: np.ndarray


def _light_inputs(reading: LeafReading, cfg: dict[str, Any], cal: dict[str, Any]
                  ) -> tuple[LightSource, float]:
    light = get_light_source(cfg, cal, reading.light_source)
    c_leaf, _ = get_position_factor(cfg, cal, "leaf")
    return light, lux_to_irradiance_wm2(reading.lux, light.k_wm2_per_lux, c_leaf)


def _draw_at(t_leaf_c: float, reading: LeafReading, e_s_wm2: float, a_s: float,
             s: Sigmas, n: int, seed: int, cfg: dict[str, Any]) -> GsDraws:
    z = np.random.default_rng(seed).standard_normal((5, n))
    t_leaf = t_leaf_c + s.leaf_temp_c * z[0]
    t_air = reading.air_temp_c + s.air_temp_c * (
        s.rho_leaf_air * z[0] + math.sqrt(1.0 - s.rho_leaf_air ** 2) * z[1])
    rh = np.clip(reading.relative_humidity_pct + s.rh_pct * z[2], 0.0, 100.0)
    e_s = e_s_wm2 * np.exp(s.lux_fraction * z[3])
    r_bh = reading.r_bh * np.exp(s.rb_fraction * z[4])
    out = solve_core(t_leaf, t_air, reading.surroundings_temp_c, rh, reading.pressure_hpa,
                     e_s, a_s, r_bh, cfg)
    return GsDraws(t_leaf, t_air, rh, e_s, r_bh, out["g_mol"], out["status"])


def gs_draws(reading: LeafReading, cfg: dict[str, Any], cal: dict[str, Any],
             sigmas: Sigmas | None = None, n: int | None = None, seed: int = 0) -> GsDraws:
    """Raw Monte Carlo draws for one window (for tests and diagnostics)."""
    light, e_s = _light_inputs(reading, cfg, cal)
    s = sigmas if sigmas is not None else get_sigmas(cfg, cal, light)
    n = n if n is not None else int(cfg["uncertainty"]["n_draws"])
    return _draw_at(reading.leaf_temp_c, reading, e_s, light.a_s_leaf, s, n, seed, cfg)


@dataclass(frozen=True)
class _Summary:
    lo: float | None
    median: float | None
    hi: float | None
    n_used: int
    frac_ok: float
    rel_width: float
    resolved: bool


def _summarise(d: GsDraws, cfg: dict[str, Any]) -> _Summary:
    unc = cfg["uncertainty"]
    cap = float(unc["g_s_cap_mol"])
    st = d.status
    frac_ok = float(np.mean(st == OK))
    used = st != VPD
    if not used.any():
        return _Summary(None, None, None, 0, frac_ok, math.inf, False)
    values = np.where(st == BELOW, 0.0, np.where(st == ABOVE, cap, np.minimum(d.g_mol, cap)))
    lo, med, hi = (float(v) for v in np.percentile(values[used], [_P_LO, _P_MID, _P_HI]))
    rel_width = (hi - lo) / med if med > 0 else math.inf
    # An upper bound at the cap means "not bounded above", so it can never count as
    # resolved; otherwise capping would shrink the width and fake a resolved value.
    resolved = (frac_ok >= unc["dl_min_resolved_fraction"]
                and rel_width < unc["max_relative_width"] and hi < cap)
    return _Summary(lo, med, hi, int(used.sum()), frac_ok, rel_width, resolved)


# ---------------------------------------------------------------- detection limit

def forward_leaf_temp(reading: LeafReading, g_s_mol_true: float, cfg: dict[str, Any],
                      cal: dict[str, Any]) -> float | None:
    """Leaf temperature [deg C] that a true g_s would produce under this window's air,
    humidity, light, r_bH and pressure (energy balance root, brentq). None if no root."""
    light, e_s = _light_inputs(reading, cfg, cal)
    phys = cfg["physics"]
    rho_cp = rho_cp_j_m3_k(cfg)
    gamma = psychrometric_constant_pa(reading.pressure_hpa, cfg)
    r_s = molar_density_mol_m3(reading.pressure_hpa, reading.air_temp_c, cfg) / g_s_mol_true
    r_total = r_s + phys["rbw_to_rbh_ratio"] * reading.r_bh
    t_a = reading.air_temp_c

    def residual(t: float) -> float:
        out = solve_core(np.array([t]), np.array([t_a]), reading.surroundings_temp_c,
                         np.array([reading.relative_humidity_pct]), reading.pressure_hpa,
                         np.array([e_s]), light.a_s_leaf, np.array([reading.r_bh]), cfg)
        vpd = out["vpd"][0]
        return float(out["latent"][0] - rho_cp / gamma * vpd / r_total)

    lo, hi = t_a - _FORWARD_SPAN_C, t_a + _FORWARD_SPAN_C
    if residual(lo) * residual(hi) > 0:
        return None
    return brentq(residual, lo, hi, xtol=1e-10)


def detection_limit_mol(reading: LeafReading, cfg: dict[str, Any], cal: dict[str, Any],
                        s: Sigmas, n: int, seed: int) -> float | None:
    """Smallest true g_s on the grid that is resolved under this window's conditions."""
    unc = cfg["uncertainty"]
    light, e_s = _light_inputs(reading, cfg, cal)
    grid = np.geomspace(unc["dl_grid_min_mol"], unc["g_s_cap_mol"], int(unc["dl_grid_points"]))
    for g_true in grid:
        t_leaf = forward_leaf_temp(reading, float(g_true), cfg, cal)
        if t_leaf is None:
            continue
        if _summarise(_draw_at(t_leaf, reading, e_s, light.a_s_leaf, s, n, seed, cfg), cfg).resolved:
            return float(g_true)
    return None


# ---------------------------------------------------------------- public result

@dataclass(frozen=True)
class GsUncertainty:
    """Monte Carlo summary for one window. g_s values in mol m-2 s-1."""
    valid: bool
    reason: str
    g_s_mol_median: float | None
    g_s_mol_lo: float | None
    g_s_mol_hi: float | None
    hi_at_cap: bool
    frac_latent_nonpositive: float | None
    frac_above_range: float | None
    frac_vpd_nonpositive: float | None
    frac_resolved: float | None
    relative_width: float | None
    resolved: bool
    g_s_mol_dl: float | None
    n_draws: int
    n_used: int
    seed: int
    sigmas: Sigmas | None
    sigma_placeholder: bool


def gs_uncertainty(reading: LeafReading, cfg: dict[str, Any], cal: dict[str, Any],
                   sigmas: Sigmas | None = None, n: int | None = None,
                   seed: int = 0) -> GsUncertainty:
    """Interval, censoring fractions, resolved label and detection limit for one window.

    Windows that solve_gs marks invalid (missing_input, vpd_nonpositive) return an
    invalid result with the same reason. Censored windows ARE processed.
    """
    n = n if n is not None else int(cfg["uncertainty"]["n_draws"])
    base = solve_gs(reading, cfg, cal)
    if not base.valid:
        return GsUncertainty(False, base.reason, None, None, None, False, None, None, None,
                             None, None, False, None, n, 0, seed, sigmas,
                             sigmas.placeholder if sigmas else False)
    light, _ = _light_inputs(reading, cfg, cal)
    s = sigmas if sigmas is not None else get_sigmas(cfg, cal, light)
    d = gs_draws(reading, cfg, cal, s, n, seed)
    sm = _summarise(d, cfg)
    cap = float(cfg["uncertainty"]["g_s_cap_mol"])
    return GsUncertainty(
        valid=True, reason="ok",
        g_s_mol_median=sm.median, g_s_mol_lo=sm.lo, g_s_mol_hi=sm.hi,
        hi_at_cap=sm.hi is not None and sm.hi >= cap,
        frac_latent_nonpositive=float(np.mean(d.status == BELOW)),
        frac_above_range=float(np.mean(d.status == ABOVE)),
        frac_vpd_nonpositive=float(np.mean(d.status == VPD)),
        frac_resolved=sm.frac_ok, relative_width=sm.rel_width, resolved=sm.resolved,
        g_s_mol_dl=detection_limit_mol(reading, cfg, cal, s, n, seed),
        n_draws=n, n_used=sm.n_used, seed=seed, sigmas=s, sigma_placeholder=s.placeholder)


def attach_uncertainty(result: ConductanceResult, unc: GsUncertainty) -> ConductanceResult:
    """Copy g_s_mol_dl, g_s_lo, g_s_hi (mol m-2 s-1) onto a T1.4 result. Never filters."""
    return replace(result, g_s_mol_dl=unc.g_s_mol_dl, g_s_lo=unc.g_s_mol_lo,
                   g_s_hi=unc.g_s_mol_hi)


# ---------------------------------------------------------------- card r_b labels

@dataclass(frozen=True)
class CardUncertainty:
    """r_bH draws summary [s m-1]. r_bh_sigma = (p84.13 - p15.87) / 2 (robust)."""
    valid: bool
    reason: str
    r_bh_median: float | None
    r_bh_lo: float | None
    r_bh_hi: float | None
    r_bh_sigma: float | None
    frac_nonpositive: float | None
    frac_rn_near_zero: float | None
    n_draws: int
    seed: int
    sigma_placeholder: bool


def card_rbh_uncertainty(reading: ReferenceCardReading, cfg: dict[str, Any],
                         cal: dict[str, Any], sigmas: Sigmas | None = None,
                         n: int | None = None, seed: int = 0) -> CardUncertainty:
    """Monte Carlo over card temperature, air temperature (correlated, rho) and an Rn fraction:

        T_card = T_card_meas + sigma_leaf z1   (same thermal camera as the leaf)
        T_air  = T_air_meas + sigma_air (rho z1 + sqrt(1 - rho^2) z2)
        Rn     = Rn(T_card, eq. 2) x (1 + sigma_rn_card z3)          (Gaussian, as spec)
        r_bH   = rho cp (T_card - T_air) / Rn                        (eq. 9)

    Draws with r_bH <= 0 are kept and counted. Draws with |Rn| < min_abs_rn_wm2 are
    excluded from percentiles and counted. Readings that fail the T1.3 gate return invalid.
    """
    n = n if n is not None else int(cfg["uncertainty"]["n_draws"])
    base = solve_card_rbh(reading, cfg, cal)
    if base.reason not in ("ok", "suspect_card_rbh"):
        return CardUncertainty(False, base.reason, None, None, None, None, None, None,
                               n, seed, sigmas.placeholder if sigmas else False)
    light = get_light_source(cfg, cal, reading.light_source)
    s = sigmas if sigmas is not None else get_sigmas(cfg, cal, light)
    a_s_card, _ = card_absorptivity(cfg, cal)
    emissivity = card_emissivity(cfg, cal)
    c_card = (reading.c_card if reading.c_card is not None
              else get_position_factor(cfg, cal, "card")[0])
    e_s = lux_to_irradiance_wm2(reading.lux, light.k_wm2_per_lux, c_card)
    phys = cfg["physics"]
    lw = bool(phys["include_longwave"])

    z = np.random.default_rng(seed).standard_normal((3, n))
    t_card = reading.card_temp_c + s.leaf_temp_c * z[0]
    t_air = reading.air_temp_c + s.air_temp_c * (
        s.rho_leaf_air * z[0] + math.sqrt(1.0 - s.rho_leaf_air ** 2) * z[1])
    t_sur = reading.surroundings_temp_c if lw else reading.air_temp_c
    rn = net_radiation_wm2(e_s, a_s_card, t_card, t_sur, emissivity, phys["faces"], lw, cfg)
    rn = rn * (1.0 + s.rn_card_fraction * z[2])
    near_zero = np.abs(rn) < phys["min_abs_rn_wm2"]
    with np.errstate(divide="ignore", invalid="ignore"):
        r_bh = rho_cp_j_m3_k(cfg) * (t_card - t_air) / rn
    used = r_bh[~near_zero]
    if used.size == 0:
        return CardUncertainty(True, base.reason, None, None, None, None, None, 1.0,
                               n, seed, s.placeholder)
    lo, med, hi, m1, p1 = np.percentile(used, [_P_LO, _P_MID, _P_HI, _P_M1, _P_P1])
    return CardUncertainty(
        valid=True, reason=base.reason, r_bh_median=float(med), r_bh_lo=float(lo),
        r_bh_hi=float(hi), r_bh_sigma=float((p1 - m1) / 2.0),
        frac_nonpositive=float(np.mean(used <= 0)), frac_rn_near_zero=float(np.mean(near_zero)),
        n_draws=n, seed=seed, sigma_placeholder=s.placeholder)


# ---------------------------------------------------------------- pooled estimate

@dataclass(frozen=True)
class PooledGs:
    """Censoring-aware g_s for a group of windows that share one true g_s (T1.7 item 5).

    Units: g_s_mol mol m-2 s-1, r_s s m-1, latent_wm2 W m-2, vpd_pa Pa, r_bh s m-1,
    pressure_hpa hPa, air_temp_c deg C (all are means over the windows used).
    """
    g_s_mol: float | None
    r_s: float | None
    latent_wm2: float | None
    vpd_pa: float | None
    r_bh: float | None
    pressure_hpa: float | None
    air_temp_c: float | None
    valid: bool
    reason: str
    censor: str
    n_windows: int
    n_excluded: int
    n_censored_windows: int


def pooled_gs_mol(readings: list[LeafReading], results: list[ConductanceResult],
                  cfg: dict[str, Any]) -> PooledGs:
    """Pool windows in flux space, then convert to g_s once (owner decision A, 2026-11-08).

        mean lambdaE, mean VPD, mean r_bH, mean P, mean T_a over the valid windows
        r_s = rho cp x VPD / (gamma(P) x lambdaE) - ratio x r_bH        s m-1   (eq. 8)
        g_s,mol = (1 / r_s) x P_Pa / (R x T_a_K)                        mol m-2 s-1

    Every valid window is used, INCLUDING censored ones: lambdaE is always measured, even
    when it is <= 0, so nothing is truncated. lambdaE is close to linear in the temperature
    errors, while g_s = 1/r_s is strongly curved, so averaging fluxes avoids the bias of
    averaging g_s. Invalid windows (missing_input, vpd_nonpositive) are excluded and counted.

    Valid ONLY for windows that share one true g_s and one set of conditions (replicate
    windows on one leaf or plant); see docs/assumptions_and_limitations.md A37.
    Pooled lambdaE <= 0 -> censor below_detection; r_s <= 0 -> above_range (both valid).
    """
    if len(readings) != len(results):
        raise ValueError(f"readings and results must have the same length, "
                         f"got {len(readings)} and {len(results)}")
    used = [(x, r) for x, r in zip(readings, results) if r.valid]
    n_excluded = len(results) - len(used)
    n_censored = sum(r.censor != "none" for _, r in used)
    if not used:
        return PooledGs(None, None, None, None, None, None, None, False, "missing_input",
                        "none", 0, n_excluded, 0)

    latent = float(np.mean([r.latent_wm2 for _, r in used]))
    vpd = float(np.mean([r.vpd_pa for _, r in used]))
    r_bh = float(np.mean([x.r_bh for x, _ in used]))
    p_hpa = float(np.mean([x.pressure_hpa for x, _ in used]))
    t_a = float(np.mean([x.air_temp_c for x, _ in used]))
    means = dict(latent_wm2=latent, vpd_pa=vpd, r_bh=r_bh, pressure_hpa=p_hpa,
                 air_temp_c=t_a, n_windows=len(used), n_excluded=n_excluded,
                 n_censored_windows=n_censored)

    if vpd <= 0:
        return PooledGs(None, None, valid=False, reason="vpd_nonpositive", censor="none", **means)
    if latent <= 0:
        return PooledGs(None, None, valid=True, reason="ok", censor="below_detection", **means)
    gamma = psychrometric_constant_pa(p_hpa, cfg)
    r_s = rho_cp_j_m3_k(cfg) * vpd / (gamma * latent) - cfg["physics"]["rbw_to_rbh_ratio"] * r_bh
    if r_s <= 0:
        return PooledGs(None, r_s, valid=True, reason="ok", censor="above_range", **means)
    g_mol = molar_density_mol_m3(p_hpa, t_a, cfg) / r_s
    return PooledGs(g_mol, r_s, valid=True, reason="ok", censor="none", **means)
