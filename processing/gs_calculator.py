"""Stomatal conductance from the leaf energy balance (T1.4, Appendix A eqs. 1-8, A15).

Steps (per one-sided leaf area; with faces = 2, g_s is the two faces combined):
    E_s = c_leaf x k x lux                                           W m-2   (eq. 1)
    Rn  = a_s x E_s + n x e_L x sigma x (T_sur^4 - T_s^4)            W m-2   (eq. 2)
    H   = rho cp (T_s - T_a) / r_bH                                  W m-2   (eq. 3)
    lambdaE = Rn - H                                                 W m-2   (eq. 4)
    VPD = e_s(T_s) - e_s(T_a) x RH / 100                             Pa      (eq. 5)
    gamma = 0.0665 x P_hPa                                           Pa K-1  (eq. 6)
    r_s = rho cp x VPD / (gamma x lambdaE) - r_bw,  r_bw = 0.93 r_bH s m-1   (eq. 8)
    g_s = 1 / r_s [m s-1];  g_s,mol = g_s x P_Pa / (R x T_a_K) [mol m-2 s-1]

Censoring (A15): censored results are VALID and must never be dropped.
  - lambdaE <= 0 -> censor = below_detection. r_s, g_s_ms, g_s_mol are None
    (g_s is at or below the detection limit g_s_mol_dl, which T1.5 fills in).
  - r_s <= 0     -> censor = above_range. r_s holds the (<= 0) value; g_s_ms, g_s_mol None.
Rn, H, lambdaE and VPD are always filled when inputs are usable.

VPD <= 0 (spec, Session 8 review): NOT a failure. Flag vpd_nonpositive is added and the
usual order applies. With lambdaE > 0, eq. 8 gives r_s = (<= 0) - r_bw < 0, so the result
is censored above_range (valid). With lambdaE <= 0 it is below_detection (the spec covers
only lambdaE > 0; decision_log 2026-11-08). These windows stay in every table.

Bad data (missing_input): non-finite input, r_bh <= 0, pressure <= 0, RH outside 0-100,
lux < 0.

g_s_mol_dl, g_s_lo, g_s_hi are None here; uncertainty.py (T1.5) fills them.
Inputs must be window means (T3.4 checks steady state and wind upstream).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from processing.config import get_light_source, get_position_factor
from processing.energy_balance_utils import (
    lux_to_irradiance_wm2,
    molar_density_mol_m3,
    net_radiation_wm2,
    psychrometric_constant_pa,
    rho_cp_j_m3_k,
    saturation_vapor_pressure_pa,
)

CENSOR_LABELS = ("none", "below_detection", "above_range")


@dataclass(frozen=True)
class LeafReading:
    """Window-mean readings for one leaf.

    Temperatures deg C, RH %, pressure hPa, lux in lux, r_bh in s m-1.
    r_bh_method is one of the values under config `methods` (card_direct,
    ml_corrected, flat_plate). surroundings_temp_c may be None only when
    physics.include_longwave is false.
    """
    leaf_temp_c: float | None
    air_temp_c: float | None
    surroundings_temp_c: float | None
    relative_humidity_pct: float | None
    pressure_hpa: float | None
    lux: float | None
    light_source: str
    r_bh: float | None
    r_bh_method: str


@dataclass(frozen=True)
class ConductanceResult:
    """valid = inputs were usable. censor = none, below_detection or above_range.

    Units: r_s s m-1, g_s_ms m s-1, g_s_mol / g_s_mol_dl / g_s_lo / g_s_hi mol m-2 s-1,
    rn_wm2 / h_wm2 / latent_wm2 W m-2, vpd_pa Pa.
    """
    r_s: float | None
    g_s_ms: float | None
    g_s_mol: float | None
    rn_wm2: float | None
    h_wm2: float | None
    latent_wm2: float | None
    vpd_pa: float | None
    valid: bool
    reason: str
    censor: str
    g_s_mol_dl: float | None
    flags: tuple[str, ...]
    g_s_lo: float | None
    g_s_hi: float | None


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _result(flags: tuple[str, ...], *, valid: bool, reason: str = "ok", censor: str = "none",
            r_s: float | None = None, g_s_ms: float | None = None, g_s_mol: float | None = None,
            rn_wm2: float | None = None, h_wm2: float | None = None,
            latent_wm2: float | None = None, vpd_pa: float | None = None) -> ConductanceResult:
    return ConductanceResult(r_s=r_s, g_s_ms=g_s_ms, g_s_mol=g_s_mol, rn_wm2=rn_wm2,
                             h_wm2=h_wm2, latent_wm2=latent_wm2, vpd_pa=vpd_pa, valid=valid,
                             reason=reason, censor=censor, g_s_mol_dl=None, flags=flags,
                             g_s_lo=None, g_s_hi=None)


def _inputs_usable(reading: LeafReading, include_longwave: bool) -> bool:
    needed = [reading.leaf_temp_c, reading.air_temp_c, reading.relative_humidity_pct,
              reading.pressure_hpa, reading.lux, reading.r_bh]
    if include_longwave:
        needed.append(reading.surroundings_temp_c)
    if not all(_is_number(x) for x in needed):
        return False
    # Physical range checks: bad data -> missing_input, never an exception.
    return (reading.r_bh > 0 and reading.pressure_hpa > 0 and reading.lux >= 0
            and 0.0 <= reading.relative_humidity_pct <= 100.0)


def solve_gs(reading: LeafReading, cfg: dict[str, Any], cal: dict[str, Any]) -> ConductanceResult:
    """Solve the leaf energy balance for g_s (eqs. 1-8). See module docstring for steps.

    Raises CalibrationMissing for missing light calibration and ValueError for an
    unknown r_bh_method (setup or programming errors). Bad data returns a result.
    """
    allowed_methods = set(cfg["methods"][k] for k in ("primary", "secondary", "baseline"))
    if reading.r_bh_method not in allowed_methods:
        raise ValueError(f"r_bh_method must be one of {sorted(allowed_methods)}, "
                         f"got '{reading.r_bh_method}'")

    light = get_light_source(cfg, cal, reading.light_source)
    c_leaf, c_flags = get_position_factor(cfg, cal, "leaf")
    flags = light.flags + c_flags

    phys = cfg["physics"]
    include_lw = bool(phys["include_longwave"])
    if not _inputs_usable(reading, include_lw):
        return _result(flags, valid=False, reason="missing_input")

    t_s, t_a = reading.leaf_temp_c, reading.air_temp_c
    # Surroundings only enter the longwave term; any number works when it is off.
    t_sur = reading.surroundings_temp_c if include_lw else t_a
    rho_cp = rho_cp_j_m3_k(cfg)

    e_s_wm2 = lux_to_irradiance_wm2(reading.lux, light.k_wm2_per_lux, c_leaf)
    rn = net_radiation_wm2(e_s_wm2, light.a_s_leaf, t_s, t_sur,
                           phys["leaf_emissivity_longwave"], phys["faces"], include_lw, cfg)
    h = rho_cp * (t_s - t_a) / reading.r_bh
    latent = rn - h
    e_a = saturation_vapor_pressure_pa(t_a, cfg) * reading.relative_humidity_pct / 100.0
    vpd = saturation_vapor_pressure_pa(t_s, cfg) - e_a
    fluxes = dict(rn_wm2=rn, h_wm2=h, latent_wm2=latent, vpd_pa=vpd)

    # A flag, not a rejection: dropping cold-leaf windows would cut the high-g_s end
    # and bias means low (selection on the measured leaf temperature). Exactly <= 0.
    if vpd <= 0:
        flags = flags + ("vpd_nonpositive",)

    if latent <= 0:
        # A15: closed stomata within noise. Keep the window, label it.
        return _result(flags, valid=True, censor="below_detection", **fluxes)

    gamma = psychrometric_constant_pa(reading.pressure_hpa, cfg)
    r_bw = phys["rbw_to_rbh_ratio"] * reading.r_bh
    r_s = rho_cp * vpd / (gamma * latent) - r_bw
    if r_s <= 0:
        return _result(flags, valid=True, censor="above_range", r_s=r_s, **fluxes)

    g_s_ms = 1.0 / r_s
    g_s_mol = g_s_ms * molar_density_mol_m3(reading.pressure_hpa, t_a, cfg)
    return _result(flags, valid=True, r_s=r_s, g_s_ms=g_s_ms, g_s_mol=g_s_mol, **fluxes)


def count_censor(results: Iterable[ConductanceResult]) -> dict[str, int]:
    """Count valid results by censor label (invalid results are not counted)."""
    return dict(Counter(r.censor for r in results if r.valid))
