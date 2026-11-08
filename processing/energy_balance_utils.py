"""Pure helper functions for the leaf and card energy balance (T1.2, Appendix A eqs. 1-10).

Every constant comes from the cfg dict (config.yaml loaded by processing.config).
Nothing here reads files. Units follow spec Section 6: temperatures in deg C (kelvin
only inside radiation and buoyancy terms), pressure in hPa at the interface and Pa
inside the math, resistances in s m-1, fluxes in W m-2, lengths in m, wind in m s-1.

Deviation from the spec signatures (owner-approved, decision_log 2026-11-08):
saturation_vapor_pressure_pa, psychrometric_constant_pa, molar_density_mol_m3 and
net_radiation_wm2 take cfg as an explicit argument; lux_to_irradiance_wm2 takes an
optional c_position (default 1.0).
"""
from __future__ import annotations

import math
from typing import Any

REGIMES = ("forced", "mixed", "free")


def c_to_k(temp_c: float, cfg: dict[str, Any]) -> float:
    """T_K = T_C + kelvin_offset."""
    return temp_c + cfg["physics"]["kelvin_offset"]


def rho_cp_j_m3_k(cfg: dict[str, Any]) -> float:
    """Volumetric heat capacity of air, rho x cp [kg m-3 x J kg-1 K-1 = J m-3 K-1] (eq. 3)."""
    return cfg["physics"]["air_density_kg_m3"] * cfg["physics"]["specific_heat_air_j_kg_k"]


def saturation_vapor_pressure_pa(temp_c: float, cfg: dict[str, Any]) -> float:
    """Tetens (eq. 5): e_s = a x exp(b T / (T + c)), T in deg C, e_s in Pa."""
    p = cfg["physics"]
    denom = temp_c + p["tetens_c_c"]
    if denom <= 0:
        raise ValueError(f"temperature {temp_c} C is outside the Tetens formula's range")
    return p["tetens_a_pa"] * math.exp(p["tetens_b"] * temp_c / denom)


def psychrometric_constant_pa(pressure_hpa: float, cfg: dict[str, Any]) -> float:
    """gamma [Pa K-1] = 0.0665 [Pa K-1 hPa-1] x P [hPa] (eq. 6, FAO-56)."""
    if not pressure_hpa > 0:
        raise ValueError(f"pressure must be positive, got {pressure_hpa} hPa")
    return cfg["physics"]["gamma_pa_per_k_per_hpa"] * pressure_hpa


def molar_density_mol_m3(pressure_hpa: float, air_temp_c: float, cfg: dict[str, Any]) -> float:
    """Ideal gas (eq. 8): n/V = P / (R T), P in Pa (= 100 x hPa), T in K, result mol m-3."""
    if not pressure_hpa > 0:
        raise ValueError(f"pressure must be positive, got {pressure_hpa} hPa")
    pressure_pa = pressure_hpa * 100.0  # unit conversion hPa -> Pa
    return pressure_pa / (cfg["physics"]["gas_constant_j_mol_k"] * c_to_k(air_temp_c, cfg))


def lux_to_irradiance_wm2(lux: float, k: float, c_position: float = 1.0) -> float:
    """Shortwave at the surface (eq. 1): E_s [W m-2] = c_position x k [W m-2 lux-1] x lux.

    c_position is the card or leaf position factor (A16); use
    processing.config.get_position_factor, which flags c_position_default when unmeasured.
    """
    return c_position * k * lux


def net_radiation_wm2(irradiance_wm2: float, a_s: float, surface_temp_c: float,
                      surroundings_temp_c: float, emissivity: float, faces: int,
                      include_longwave: bool, cfg: dict[str, Any]) -> float:
    """Net radiation per one-sided area (eq. 2), W m-2:

        Rn = a_s x E_s + faces x emissivity x sigma x (T_sur^4 - T_s^4),  T in K.

    The longwave term is left out when include_longwave is False.
    """
    if faces not in (1, 2):
        raise ValueError(f"faces must be 1 or 2, got {faces}")
    rn = a_s * irradiance_wm2
    if include_longwave:
        t_sur_k = c_to_k(surroundings_temp_c, cfg)
        t_s_k = c_to_k(surface_temp_c, cfg)
        rn += faces * emissivity * cfg["physics"]["stefan_boltzmann"] * (t_sur_k ** 4 - t_s_k ** 4)
    return rn


def _sidedness(cfg: dict[str, Any]) -> int:
    """s is always physics.faces; config must say so (spec T1.2)."""
    if cfg["convection"]["sidedness_factor"] != "from_faces":
        raise ValueError("convection.sidedness_factor must be 'from_faces'; "
                         "s is read from physics.faces and never set separately")
    s = cfg["physics"]["faces"]
    if s not in (1, 2):
        raise ValueError(f"physics.faces must be 1 or 2, got {s}")
    return s


def prandtl_number(cfg: dict[str, Any]) -> float:
    """Pr = nu / D_H (dimensionless)."""
    c = cfg["convection"]
    return c["kinematic_viscosity_m2_s"] / c["thermal_diffusivity_m2_s"]


def equivalent_coefficient_c(cfg: dict[str, Any]) -> float:
    """C in the forced-only shortcut r_bH = C sqrt(d/U), units s^0.5 m-1 (eq. 10):

        C = sqrt(nu) / (s x b x a_forced x D_H x Pr^(1/3))
    """
    c = cfg["convection"]
    s = _sidedness(cfg)
    return math.sqrt(c["kinematic_viscosity_m2_s"]) / (
        s * c["enhancement_factor"] * c["forced_nu_coeff"]
        * c["thermal_diffusivity_m2_s"] * prandtl_number(cfg) ** (1.0 / 3.0))


def flat_plate_conductances(d_m: float, wind_ms: float, delta_t_c: float, air_temp_c: float,
                            cfg: dict[str, Any]) -> tuple[float, float]:
    """Forced and free heat conductances (m s-1) of a flat plate, both sides counted by s (eq. 10).

        Re = U d / nu;   Gr = g beta d^3 dT / nu^2,  beta = 1 / T_air[K]
        Nu_forced = b x 0.66 x Pr^(1/3) x Re^(1/2);   Nu_free = 0.54 x Pr^(1/4) x Gr^(1/4)
        g_x = s x D_H x Nu_x / d

    U below wind_noise_floor_ms is set to 0. dT inside Gr is max(|delta_t_c|,
    min_grashof_delta_t_c) so the free term never vanishes.
    """
    if not (d_m > 0 and math.isfinite(d_m)):
        raise ValueError(f"d_m must be a positive length in metres, got {d_m}")
    if not (math.isfinite(wind_ms) and wind_ms >= 0):
        raise ValueError(f"wind_ms must be a finite non-negative speed, got {wind_ms} "
                         "(missing wind is reason no_wind, handled by the caller)")
    if not math.isfinite(delta_t_c):
        raise ValueError(f"delta_t_c must be finite, got {delta_t_c}")
    c = cfg["convection"]
    s = _sidedness(cfg)
    nu = c["kinematic_viscosity_m2_s"]
    d_h = c["thermal_diffusivity_m2_s"]
    pr = prandtl_number(cfg)

    u = wind_ms if wind_ms >= c["wind_noise_floor_ms"] else 0.0
    re = u * d_m / nu
    nu_forced = c["enhancement_factor"] * c["forced_nu_coeff"] * pr ** (1.0 / 3.0) * math.sqrt(re)

    dt_gr = max(abs(delta_t_c), c["min_grashof_delta_t_c"])
    beta = 1.0 / c_to_k(air_temp_c, cfg)
    gr = cfg["physics"]["gravity_m_s2"] * beta * d_m ** 3 * dt_gr / nu ** 2
    nu_free = c["free_nu_coeff"] * pr ** 0.25 * gr ** 0.25

    return s * d_h * nu_forced / d_m, s * d_h * nu_free / d_m


def flat_plate_rbh(d_m: float, wind_ms: float, delta_t_c: float, air_temp_c: float,
                   cfg: dict[str, Any]) -> tuple[float, str]:
    """Flat-plate heat boundary-layer resistance and regime (eq. 10).

        r_bH [s m-1] = 1 / (g_forced + g_free)

    Regime: "forced" if g_free/g_forced < mixed_convection_ratio, "free" if
    g_forced/g_free < mixed_convection_ratio (including U = 0), else "mixed".
    Use convection_flags(regime) for the baseline_unreliable flag.
    """
    if cfg["convection"]["combination"] != "sum":
        raise ValueError("convection.combination must be 'sum' (Nu_forced + Nu_free, Appendix A eq. 10); "
                         f"got '{cfg['convection']['combination']}'")
    g_forced, g_free = flat_plate_conductances(d_m, wind_ms, delta_t_c, air_temp_c, cfg)
    ratio = cfg["convection"]["mixed_convection_ratio"]
    if g_free < ratio * g_forced:
        regime = "forced"
    elif g_forced < ratio * g_free:
        regime = "free"
    else:
        regime = "mixed"
    return 1.0 / (g_forced + g_free), regime


def convection_flags(regime: str) -> tuple[str, ...]:
    """Flag baseline_unreliable for mixed or free convection (spec Section 6 flags)."""
    if regime not in REGIMES:
        raise ValueError(f"unknown regime '{regime}'")
    return () if regime == "forced" else ("baseline_unreliable",)
