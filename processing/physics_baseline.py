"""Boundary-layer resistance from the dry reference card (T1.3, Appendix A eq. 9).

The card does not evaporate water, so lambdaE = 0 and its energy balance Rn = H + lambdaE
becomes Rn = H = rho cp (T_card - T_air) / r_bH. Solving for r_bH:

    r_bH,card = rho cp (T_card - T_air) / Rn_card          [s m-1]

with Rn_card from eq. 2 using the card's own absorptivity a_s_card and emissivity.

Gate (exogenous only, spec T1.3 and Appendix A validity rules):
  - absorbed shortwave a_s_card x E_s,card < physics.min_card_absorbed_sw_wm2 -> low_card_signal
  - |Rn| < physics.min_abs_rn_wm2 -> rn_near_zero. Rn contains T_card through the
    longwave term, so this is a divide-by-zero guard, not a purely exogenous gate.
  - r_bH <= 0 -> suspect_card_rbh. The value is kept so it can go to the rejects file
    and be counted.
T_card - T_air is never used to accept or reject a reading.

Call this on window means only (windows.window_sec), never on single 5-second samples.
This function cannot check that; T3.4 and T4.3 must enforce it.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from processing.config import (
    CalibrationMissing,
    _calibrated,
    get_light_source,
    get_position_factor,
)
from processing.energy_balance_utils import (
    lux_to_irradiance_wm2,
    net_radiation_wm2,
    rho_cp_j_m3_k,
)


@dataclass(frozen=True)
class ReferenceCardReading:
    """Window-mean readings for one dry reference card.

    Temperatures in deg C, lux in lux. c_card is the card position's irradiance factor
    (A16); None means "look it up" (calibration.json, then config, else 1.0 flagged).
    surroundings_temp_c may be None only when physics.include_longwave is false.
    """
    card_id: str
    card_temp_c: float | None
    air_temp_c: float | None
    surroundings_temp_c: float | None
    lux: float | None
    light_source: str
    c_card: float | None = None


@dataclass(frozen=True)
class CardResult:
    """r_bh [s m-1] and rn_wm2 [W m-2] are None when they could not be computed.

    For reason suspect_card_rbh, r_bh holds the (<= 0) value so it can be logged.
    """
    r_bh: float | None
    rn_wm2: float | None
    valid: bool
    reason: str
    flags: tuple[str, ...]


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def card_absorptivity(cfg: dict[str, Any], cal: dict[str, Any]) -> tuple[float, tuple[str, ...]]:
    """Return (a_s_card, flags). Dimensionless, in (0, 1].

    Order: calibration.json card.a_s_card, then config card.a_s_card. If both are null:
    raise CalibrationMissing, unless card.allow_default_a_s_card is true, in which case
    card.a_s_card_placeholder is used and flagged a_s_default.
    """
    value = _calibrated(cal, "card", "a_s_card")
    if value is None:
        value = cfg["card"].get("a_s_card")
    flags: tuple[str, ...] = ()
    if value is None:
        if not cfg["card"].get("allow_default_a_s_card", False):
            raise CalibrationMissing(
                "card a_s_card is not measured (null in calibration.json and config.yaml). "
                "Measure the card's shortwave absorptivity, or set "
                "card.allow_default_a_s_card: true to use the UNVERIFIED placeholder.")
        value = cfg["card"]["a_s_card_placeholder"]
        flags = ("a_s_default",)
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"a_s_card must be in (0, 1], got {value}")
    return value, flags


def card_emissivity(cfg: dict[str, Any], cal: dict[str, Any]) -> float:
    """Card longwave emissivity: calibration.json card.emissivity, else config."""
    value = _calibrated(cal, "card", "emissivity")
    if value is None:
        value = cfg["card"]["emissivity"]
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"card emissivity must be in (0, 1], got {value}")
    return value


def solve_card_rbh(reading: ReferenceCardReading, cfg: dict[str, Any],
                   cal: dict[str, Any]) -> CardResult:
    """Solve eq. 9 for one window-mean card reading.

        E_s,card = c_card x k x lux                                    W m-2   (eq. 1)
        Rn = a_s_card x E_s,card + n x e_card x sigma x (T_sur^4 - T_card^4)   (eq. 2)
        r_bH = rho cp (T_card - T_air) / Rn                            s m-1   (eq. 9)

    Bad data returns an invalid CardResult. Missing calibration (k, a_s_card) raises
    CalibrationMissing because it is a setup error, not bad data.
    """
    light = get_light_source(cfg, cal, reading.light_source)
    a_s_card, a_s_flags = card_absorptivity(cfg, cal)
    emissivity = card_emissivity(cfg, cal)

    if reading.c_card is None:
        c_card, c_flags = get_position_factor(cfg, cal, "card")
    else:
        c_card, c_flags = float(reading.c_card), ()

    # a_s_default from the light source is about the LEAF absorptivity, so only
    # k_uncalibrated is carried over to the card.
    light_flags = tuple(f for f in light.flags if f == "k_uncalibrated")
    flags = light_flags + a_s_flags + c_flags

    phys = cfg["physics"]
    include_lw = bool(phys["include_longwave"])
    needed = [reading.card_temp_c, reading.air_temp_c, reading.lux]
    if include_lw:
        needed.append(reading.surroundings_temp_c)
    if not all(_is_number(x) for x in needed):
        return CardResult(None, None, False, "missing_input", flags)

    e_s_wm2 = lux_to_irradiance_wm2(reading.lux, light.k_wm2_per_lux, c_card)
    absorbed_sw_wm2 = a_s_card * e_s_wm2
    # Surroundings only matter in the longwave term; any number works when it is off.
    t_sur_c = reading.surroundings_temp_c if include_lw else reading.air_temp_c
    rn_wm2 = net_radiation_wm2(e_s_wm2, a_s_card, reading.card_temp_c, t_sur_c,
                               emissivity, phys["faces"], include_lw, cfg)

    # Exogenous gate: depends only on light, never on T_card - T_air.
    if absorbed_sw_wm2 < phys["min_card_absorbed_sw_wm2"]:
        return CardResult(None, rn_wm2, False, "low_card_signal", flags)
    # Divide-by-zero guard (Rn includes T_card via longwave).
    if abs(rn_wm2) < phys["min_abs_rn_wm2"]:
        return CardResult(None, rn_wm2, False, "rn_near_zero", flags)

    r_bh = rho_cp_j_m3_k(cfg) * (reading.card_temp_c - reading.air_temp_c) / rn_wm2
    if r_bh <= 0:
        # Kept with its value for the rejects file; a rising count signals a
        # temperature-calibration problem (spec T1.3).
        return CardResult(r_bh, rn_wm2, False, "suspect_card_rbh", flags)
    return CardResult(r_bh, rn_wm2, True, "ok", flags)


def count_reasons(results: Iterable[CardResult]) -> dict[str, int]:
    """Count results by reason code, for the training-set report (T4.3)."""
    return dict(Counter(r.reason for r in results))
