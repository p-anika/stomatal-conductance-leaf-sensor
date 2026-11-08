"""Load config.yaml and calibration.json, and pick light-source settings (T1.1).

calibration.json mirrors the section and key names of config.yaml. A non-null value
in calibration.json overrides the config value with the same section and key (schema
in docs/decision_log.md, 2026-11-08). The calibrated value is what counts as
"calibrated": config.yaml only ever holds defaults or reference values.

Light (Appendix A eq. 1): E_s = c_position x k x lux, with k in W m-2 per lux.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Spec rule (T1.1), not a physical constant: only these sources may borrow the
# sunlight reference k, and only when physics.allow_uncalibrated_k is true.
FALLBACK_ELIGIBLE_SOURCES = ("window", "sunlight")
REFERENCE_SOURCE = "sunlight"
# Spec rule (A16): an unmeasured position factor is taken as 1.0 and flagged.
DEFAULT_POSITION_FACTOR = 1.0
POSITION_NAMES = ("leaf", "card")


class CalibrationMissing(Exception):
    """A value that must come from calibration is still null."""


@dataclass(frozen=True)
class LightSource:
    """Settings for one light source.

    a_s_leaf: leaf shortwave absorptivity (dimensionless, 0-1).
    k_wm2_per_lux: lux to W m-2 factor.
    k_source: "calibration" or "sunlight_reference".
    flags: non-fatal flags to copy onto every result (k_uncalibrated, a_s_default).
    sigma_lux_fraction: fractional 1-sigma light error to use in uncertainty.py.
    """
    name: str
    a_s_leaf: float
    k_wm2_per_lux: float
    k_source: str
    flags: tuple[str, ...]
    sigma_lux_fraction: float


def load_config(path: str | Path) -> dict[str, Any]:
    """Read config.yaml into a dict. Raises FileNotFoundError if it is missing."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return cfg


def load_calibration(path: str | Path) -> dict[str, Any]:
    """Read calibration.json. If the file is absent, warn and return {}."""
    path = Path(path)
    if not path.is_file():
        logger.warning("calibration file %s not found; using config defaults "
                       "(uncalibrated values will raise or be flagged)", path)
        return {}
    with open(path, encoding="utf-8") as f:
        cal = json.load(f)
    if not isinstance(cal, dict):
        raise ValueError(f"{path} must contain a JSON object at the top level")
    return cal


def _calibrated(cal: dict[str, Any], *keys: str) -> Any:
    """Return cal[k1][k2]... or None if any level is missing."""
    node: Any = cal
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def get_light_source(cfg: dict[str, Any], cal: dict[str, Any], name: str) -> LightSource:
    """Return a_s_leaf, k and k_source for light source `name`.

    k comes from calibration.json. If it is null there:
      - window or sunlight with physics.allow_uncalibrated_k = true: use the sunlight
        reference k from config, flag k_uncalibrated, use sigma_lux_fraction_uncalibrated.
      - anything else (always led_lamp, and any source not in config): raise
        CalibrationMissing. LED luminous efficacy differs from sunlight by about 2-3x.
    """
    listed = cfg["light_sources"].get(name)
    flags: list[str] = []

    k = _calibrated(cal, "light_sources", name, "k_wm2_per_lux")
    if k is not None:
        k_source = "calibration"
        sigma = cfg["uncertainty"]["sigma_lux_fraction"]
    elif (listed is not None and name in FALLBACK_ELIGIBLE_SOURCES
          and cfg["physics"]["allow_uncalibrated_k"]):
        k = cfg["light_sources"][REFERENCE_SOURCE]["k_wm2_per_lux"]
        if k is None:
            raise CalibrationMissing("the sunlight reference k_wm2_per_lux in config.yaml is null")
        k_source = "sunlight_reference"
        flags.append("k_uncalibrated")
        sigma = cfg["uncertainty"]["sigma_lux_fraction_uncalibrated"]
    else:
        raise CalibrationMissing(
            f"k_wm2_per_lux for light source '{name}' is not calibrated. Run "
            f"calibration/light_calibration.py against a reference light meter. "
            f"No fallback exists for this source."
            + ("" if name in FALLBACK_ELIGIBLE_SOURCES else
               " (Only window and sunlight may use the sunlight reference, with "
               "physics.allow_uncalibrated_k: true.)"))

    k = float(k)
    if not math.isfinite(k) or k <= 0:
        raise ValueError(f"k_wm2_per_lux for '{name}' must be a positive number, got {k}")

    a_s = _calibrated(cal, "light_sources", name, "a_s_leaf")
    if a_s is None:
        if listed is None or listed.get("a_s_leaf") is None:
            raise ValueError(f"no a_s_leaf for light source '{name}' in config or calibration")
        a_s = listed["a_s_leaf"]
        flags.append("a_s_default")
    a_s = float(a_s)
    if not 0.0 < a_s <= 1.0:
        raise ValueError(f"a_s_leaf for '{name}' must be in (0, 1], got {a_s}")

    return LightSource(name=name, a_s_leaf=a_s, k_wm2_per_lux=k, k_source=k_source,
                       flags=tuple(flags), sigma_lux_fraction=float(sigma))


def get_position_factor(cfg: dict[str, Any], cal: dict[str, Any], which: str) -> tuple[float, tuple[str, ...]]:
    """Return (c_position, flags) for which = "leaf" or "card" (A16).

    Reads spatial.irradiance_factor_<which> from calibration.json, then config.yaml.
    If both are null: 1.0 with flag c_position_default.
    """
    if which not in POSITION_NAMES:
        raise ValueError(f"position must be one of {POSITION_NAMES}, got '{which}'")
    key = f"irradiance_factor_{which}"
    value = _calibrated(cal, "spatial", key)
    if value is None:
        value = cfg["spatial"].get(key)
    if value is None:
        return DEFAULT_POSITION_FACTOR, ("c_position_default",)
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"spatial.{key} must be a positive number, got {value}")
    return value, ()
