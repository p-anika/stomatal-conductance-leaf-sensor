"""Thermal conductance index and CWSI from dry and wet references (T1.6, Appendix A eqs. 12-13).

    I_g  = (T_dry - T_leaf) / (T_leaf - T_wet)        dimensionless, proportional to g_s
    CWSI = (T_leaf - T_wet) / (T_dry - T_wet)          dimensionless, about 0 (wet) to 1 (dry)
    Identity: I_g = 1 / CWSI - 1.

Values outside the expected range (I_g < 0, CWSI < 0 or > 1) are kept exactly as
measured: clipping would bias stressed leaves (T1.6).

Rules (thresholds from config `jones`):
  - any input missing or non-finite -> both invalid, missing_input
  - T_dry - T_wet < min_dry_wet_diff_c -> both invalid, references_not_bracketing
    (owner-approved reason code, 2026-11-08; I_g is invalid too)
  - T_leaf - T_wet < min_leaf_wet_diff_c -> I_g valid, censor above_range, value None
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IndexValue:
    """One index with its status. value is None when invalid or above_range."""
    value: float | None
    valid: bool
    reason: str
    censor: str


@dataclass(frozen=True)
class JonesResult:
    i_g: IndexValue
    cwsi: IndexValue


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def conductance_index(t_dry_c: float | None, t_leaf_c: float | None, t_wet_c: float | None,
                      cfg: dict[str, Any]) -> JonesResult:
    """Return I_g and CWSI (eqs. 12, 13) from temperatures in deg C. Never raises on bad data."""
    if not all(_is_number(x) for x in (t_dry_c, t_leaf_c, t_wet_c)):
        bad = IndexValue(None, False, "missing_input", "none")
        return JonesResult(bad, bad)

    j = cfg["jones"]
    dry_wet = t_dry_c - t_wet_c
    if dry_wet < j["min_dry_wet_diff_c"]:
        # The references are too close together to bracket the leaf.
        bad = IndexValue(None, False, "references_not_bracketing", "none")
        return JonesResult(bad, bad)

    leaf_wet = t_leaf_c - t_wet_c
    cwsi = IndexValue(leaf_wet / dry_wet, True, "ok", "none")
    if leaf_wet < j["min_leaf_wet_diff_c"]:
        # Leaf almost as cool as the wet reference: I_g is unbounded (very open stomata).
        i_g = IndexValue(None, True, "ok", "above_range")
    else:
        i_g = IndexValue((t_dry_c - t_leaf_c) / leaf_wet, True, "ok", "none")
    return JonesResult(i_g, cwsi)
