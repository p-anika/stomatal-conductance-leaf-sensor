"""
Module 7: Stomatal conductance (g_s) calculator.

Takes a real leaf's own thermal + environmental reading, plus the
ML-corrected boundary-layer resistance for that specific leaf (from
ml/predict_correction.py, module 6), and solves the leaf's energy
balance for the one remaining unknown -- stomatal resistance r_s -- then
inverts it to get stomatal conductance g_s.

Run once per real leaf measurement, after leaf_segmentation.py
(module 4) and predict_correction.py (module 6) have produced that
leaf's corrected r_bH.
"""

from dataclasses import dataclass
from typing import Optional

from processing.energy_balance_utils import (
    AIR_DENSITY,
    SPECIFIC_HEAT_AIR,
    PSYCHROMETRIC_CONSTANT,
    R_BW_TO_R_BH_RATIO,
    net_radiation_wm2,
    saturation_vapor_pressure_pa,
)


@dataclass
class LeafReading:
    """One real-leaf measurement, matched to a moment in time."""
    leaf_temp_c: float
    air_temp_c: float
    relative_humidity_pct: float
    lux: float
    r_bh_corrected: float       # from predict_correction.py, this leaf's geometry-corrected value
    absorptivity: float = 0.5
    emissivity: float = 0.95


@dataclass
class ConductanceResult:
    r_s: Optional[float]              # stomatal resistance, s/m
    g_s: Optional[float]              # stomatal conductance, m/s
    latent_heat_flux_wm2: Optional[float]
    valid: bool
    reason: str = ""                  # why a reading was rejected, if invalid


def solve_stomatal_conductance(reading: LeafReading) -> ConductanceResult:
    rn = net_radiation_wm2(
        lux=reading.lux,
        air_temp_c=reading.air_temp_c,
        surface_temp_c=reading.leaf_temp_c,
        absorptivity=reading.absorptivity,
        emissivity=reading.emissivity,
    )

    sensible_heat_flux = (
        AIR_DENSITY * SPECIFIC_HEAT_AIR
        * (reading.leaf_temp_c - reading.air_temp_c)
        / reading.r_bh_corrected
    )

    latent_heat_flux = rn - sensible_heat_flux

    if latent_heat_flux <= 0:
        # No usable transpiration signal in this reading (e.g. leaf
        # strongly heated relative to available Rn, or stomata closed
        # enough that this steady-state model breaks down). Reject
        # rather than report a fabricated conductance value.
        return ConductanceResult(
            r_s=None, g_s=None, latent_heat_flux_wm2=latent_heat_flux,
            valid=False, reason="non-positive latent heat flux",
        )

    e_sat_leaf = saturation_vapor_pressure_pa(reading.leaf_temp_c)
    e_air = saturation_vapor_pressure_pa(reading.air_temp_c) * (reading.relative_humidity_pct / 100.0)
    vapor_pressure_deficit = e_sat_leaf - e_air

    r_bw = reading.r_bh_corrected * R_BW_TO_R_BH_RATIO

    r_s_plus_r_bw = (AIR_DENSITY * SPECIFIC_HEAT_AIR * vapor_pressure_deficit) / (
        PSYCHROMETRIC_CONSTANT * latent_heat_flux
    )
    r_s = r_s_plus_r_bw - r_bw

    if r_s <= 0:
        # Under this simplified model, very wide-open stomata can push
        # this slightly negative at the sensor noise floor. Treat as
        # "resistance below detectable floor" rather than crash or
        # report a negative g_s.
        return ConductanceResult(
            r_s=r_s, g_s=None, latent_heat_flux_wm2=latent_heat_flux,
            valid=False, reason="non-positive stomatal resistance (near-zero floor)",
        )

    g_s = 1.0 / r_s
    return ConductanceResult(
        r_s=r_s, g_s=g_s, latent_heat_flux_wm2=latent_heat_flux, valid=True,
    )


if __name__ == "__main__":
    # Sanity check with synthetic numbers -- no hardware needed.
    test_reading = LeafReading(
        leaf_temp_c=23.0,
        air_temp_c=22.0,
        relative_humidity_pct=55.0,
        lux=9000,
        r_bh_corrected=120.0,  # s/m, placeholder until the ML model exists
    )
    print(solve_stomatal_conductance(test_reading))
