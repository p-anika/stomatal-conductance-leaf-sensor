"""
Shared physical constants and helper functions for the leaf energy
balance calculations used by physics_baseline.py and gs_calculator.py.

Units:
  - temperatures in degrees C (converted to Kelvin internally where needed)
  - resistances in s/m
  - fluxes in W/m^2
  - vapor pressures in Pa
"""

import math

# --- Physical constants ---
STEFAN_BOLTZMANN = 5.670374419e-8    # W / m^2 / K^4
AIR_DENSITY = 1.204                   # kg / m^3, ~20C at sea level (documented approx)
SPECIFIC_HEAT_AIR = 1010.0            # J / kg / K
PSYCHROMETRIC_CONSTANT = 66.5         # Pa / K, approx at sea level (~101.3 kPa)

# Ratio of boundary-layer resistance to water vapor vs. to heat. Water
# vapor diffuses slightly differently than heat is transferred across the
# same boundary layer; this Lewis-number-based correction is a standard,
# documented approximation in leaf energy balance work (see e.g. Jones,
# "Plants and Microclimate"). It is treated as fixed -- the ML model
# corrects r_bH itself, not this ratio.
R_BW_TO_R_BH_RATIO = 0.93

# Approximate conversion from illuminance (lux) to shortwave irradiance
# (W/m^2) under daylight conditions. This stands in for a calibrated
# pyranometer -- a documented tradeoff (see project limitations).
LUX_TO_WM2 = 1.0 / 126.0


def c_to_k(temp_c: float) -> float:
    """Celsius to Kelvin."""
    return temp_c + 273.15


def saturation_vapor_pressure_pa(temp_c: float) -> float:
    """
    Saturation vapor pressure (Pa) via the Tetens formula. Accurate to
    within ~0.1% over typical leaf/air temperature ranges (0-50C).
    """
    return 610.78 * math.exp((17.27 * temp_c) / (temp_c + 237.3))


def net_radiation_wm2(lux: float, air_temp_c: float, surface_temp_c: float,
                       absorptivity: float = 0.5, emissivity: float = 0.95) -> float:
    """
    Estimate net radiation absorbed by a surface (leaf or reference
    card), in W/m^2.

        Rn = absorbed shortwave + net longwave

    Shortwave is estimated from the TSL2591 lux reading via the
    documented lux->W/m2 approximation above. Longwave exchange assumes
    the sky/surroundings radiate as a blackbody at air temperature --
    itself a simplification, since real sky temperature typically runs a
    few degrees below air temperature under clear skies. Flagged here as
    a known source of systematic error, not hidden.

    absorptivity: fraction of incoming shortwave absorbed by this
        surface. ~0.5 is typical for a green leaf; use the same value
        for the reference card, since it is deliberately coated to
        approximate the leaf's shortwave absorptivity (see project
        description) -- this is what makes the two surfaces comparable.
    emissivity: longwave emissivity, ~0.95-0.98 for both leaf tissue and
        a matte-black-coated card.
    """
    shortwave_in = lux * LUX_TO_WM2
    absorbed_shortwave = absorptivity * shortwave_in

    t_air_k = c_to_k(air_temp_c)
    t_surface_k = c_to_k(surface_temp_c)
    net_longwave = emissivity * STEFAN_BOLTZMANN * (t_air_k**4 - t_surface_k**4)

    return absorbed_shortwave + net_longwave
