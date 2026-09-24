"""
Module 5: Physics baseline (dry reference card).

On the dry reference card, latent heat flux is zero -- there's no water
to evaporate -- so its energy balance equation

    Rn = H + lambda*E

collapses to Rn = H, leaving exactly one unknown: the boundary-layer
resistance to heat transfer, r_bH. This module solves for it directly
from one reference-card reading (its temperature + the environmental
readings taken at the same moment).

Run once per reference-card reading. Each valid result, paired with that
card's known geometry features (see leaf_segmentation.py), becomes one
labeled (geometry -> r_bH) training point -- accumulated by
build_training_set.py for the ML correction model.
"""

from dataclasses import dataclass
from typing import Optional

from processing.energy_balance_utils import (
    AIR_DENSITY,
    SPECIFIC_HEAT_AIR,
    net_radiation_wm2,
)


@dataclass
class ReferenceCardReading:
    """One reference-card measurement, matched to a moment in time."""
    card_id: str                # e.g. "round_1", "elongated_2" -- matches the geometry dataset
    card_temp_c: float          # from MLX90640, same frame as the leaf
    air_temp_c: float
    lux: float
    absorptivity: float = 0.5   # coating chosen to approximate leaf absorptivity
    emissivity: float = 0.95


def solve_boundary_layer_resistance(reading: ReferenceCardReading) -> Optional[float]:
    """
    Solve for the reference card's boundary-layer resistance to heat
    transfer, r_bH, in s/m:

        r_bH = rho * cp * (T_card - T_air) / Rn

    Returns None (instead of a nonsense number) when the reading isn't
    usable:
      - Rn near zero (e.g. at night, or card/air near thermal
        equilibrium) -- the equation is undefined/unstable there.
      - r_bH would come out non-positive -- physically invalid under
        this simplified model, e.g. card cooler than air while still
        absorbing net radiation.
    Reject these rather than let bad points into the training set.
    """
    rn = net_radiation_wm2(
        lux=reading.lux,
        air_temp_c=reading.air_temp_c,
        surface_temp_c=reading.card_temp_c,
        absorptivity=reading.absorptivity,
        emissivity=reading.emissivity,
    )

    if abs(rn) < 1.0:  # W/m^2 -- effectively no driving flux
        return None

    r_bh = (AIR_DENSITY * SPECIFIC_HEAT_AIR * (reading.card_temp_c - reading.air_temp_c)) / rn

    if r_bh <= 0:
        return None

    return r_bh


def process_reference_readings(readings: list) -> list:
    """
    Batch-process a list of ReferenceCardReading into
    {"card_id": ..., "r_bh": ...} records, skipping invalid ones.
    Call this from build_training_set.py once real multi-shape
    reference-card data exists.
    """
    results = []
    for reading in readings:
        r_bh = solve_boundary_layer_resistance(reading)
        if r_bh is not None:
            results.append({"card_id": reading.card_id, "r_bh": r_bh})
    return results


if __name__ == "__main__":
    # Sanity check with synthetic numbers -- no hardware needed.
    # A card a couple degrees warmer than air, moderate daylight.
    test_reading = ReferenceCardReading(
        card_id="round_1",
        card_temp_c=24.5,
        air_temp_c=22.0,
        lux=9000,
    )
    print(f"Test r_bH: {solve_boundary_layer_resistance(test_reading)}")
