"""Readers for project data files (T2.2: reference-card manifest).

data/reference_cards.csv columns (spec Appendix B). The 11 columns named in T2.2 must be in
the header; the 8 geometry columns may be left out until photos are measured (owner
decision, 2026-11-08). A column being present does not mean it has a value: text columns
card_id, shape_group, source, material and coating must be filled; every number may be
blank until it is measured (blank -> None), but a number that is given must be in range.

A malformed manifest raises ReferenceCardError (a file problem, not a data value).
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

CARD_REQUIRED_COLUMNS = ("card_id", "shape_group", "source", "material", "thickness_mm",
                         "coating", "a_s_card", "emissivity", "area_mm2", "photo_path", "notes")
CARD_OPTIONAL_COLUMNS = ("d_min_mm", "d_eq_mm", "aspect_ratio", "eccentricity", "solidity",
                         "circularity", "edge_density", "geometry_source")
CARD_SOURCES = ("designed", "traced_leaf")
GEOMETRY_SOURCES = ("phone", "camera")
_MUST_FILL = ("card_id", "shape_group", "source", "material", "coating")

# Allowed range per numeric column: (low, high, low_inclusive, high_inclusive).
# These are definitions, not physical constants: a fraction cannot exceed 1, a length
# cannot be negative, and d_max/d_min (aspect ratio) cannot be below 1.
_RANGES: dict[str, tuple[float, float, bool, bool]] = {
    "thickness_mm": (0.0, math.inf, False, False),
    "a_s_card": (0.0, 1.0, False, True),
    "emissivity": (0.0, 1.0, False, True),
    "area_mm2": (0.0, math.inf, False, False),
    "d_min_mm": (0.0, math.inf, False, False),
    "d_eq_mm": (0.0, math.inf, False, False),
    "aspect_ratio": (1.0, math.inf, True, False),
    "eccentricity": (0.0, 1.0, True, False),
    "solidity": (0.0, 1.0, False, True),
    "circularity": (0.0, 1.0, False, True),
    "edge_density": (0.0, 1.0, True, True),
}


class ReferenceCardError(ValueError):
    """The reference-card manifest is malformed (missing column, bad value, duplicate id)."""


@dataclass(frozen=True)
class ReferenceCard:
    """One card. Lengths mm, area mm2; a_s_card, emissivity and shape ratios dimensionless.
    None means "not measured yet"."""
    card_id: str
    shape_group: str
    source: str
    material: str
    thickness_mm: float | None
    coating: str
    a_s_card: float | None
    emissivity: float | None
    area_mm2: float | None
    d_min_mm: float | None
    d_eq_mm: float | None
    aspect_ratio: float | None
    eccentricity: float | None
    solidity: float | None
    circularity: float | None
    edge_density: float | None
    geometry_source: str | None
    photo_path: str | None
    notes: str


def _range_text(lo: float, hi: float, lo_inc: bool, hi_inc: bool) -> str:
    left = "[" if lo_inc else "("
    right = "]" if hi_inc else ")"
    return f"{left}{lo:g}, {'inf' if math.isinf(hi) else f'{hi:g}'}{right}"


def _number(col: str, text: str, where: str) -> float | None:
    if text == "":
        return None
    try:
        value = float(text)
    except ValueError:
        raise ReferenceCardError(f"{where}: {col} must be a number or blank, got {text!r}") from None
    lo, hi, lo_inc, hi_inc = _RANGES[col]
    ok = (math.isfinite(value) and (value >= lo if lo_inc else value > lo)
          and (value <= hi if hi_inc else value < hi))
    if not ok:
        raise ReferenceCardError(f"{where}: {col} = {value} is outside "
                                 f"{_range_text(lo, hi, lo_inc, hi_inc)}")
    return value


def load_reference_cards(path: str | Path) -> dict[str, ReferenceCard]:
    """Read and validate data/reference_cards.csv. Returns {card_id: ReferenceCard} in file
    order (empty for a header-only file). Raises FileNotFoundError or ReferenceCardError."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"reference-card manifest not found: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        missing = [c for c in CARD_REQUIRED_COLUMNS if c not in header]
        if missing:
            raise ReferenceCardError(f"{path}: missing required column(s) {missing}")
        unknown = [c for c in header if c not in CARD_REQUIRED_COLUMNS + CARD_OPTIONAL_COLUMNS]
        if unknown:
            raise ReferenceCardError(f"{path}: unknown column(s) {unknown}; allowed columns are "
                                     "listed in spec Appendix B")
        cards: dict[str, ReferenceCard] = {}
        for row_number, row in enumerate(reader, start=2):   # row 1 is the header
            where = f"{path.name} row {row_number}"
            values = {c: (row.get(c) or "").strip() for c in CARD_REQUIRED_COLUMNS + CARD_OPTIONAL_COLUMNS}
            for col in _MUST_FILL:
                if values[col] == "":
                    raise ReferenceCardError(f"{where}: {col} must not be blank")
            if values["source"] not in CARD_SOURCES:
                raise ReferenceCardError(f"{where}: source must be one of {list(CARD_SOURCES)}, "
                                         f"got {values['source']!r}")
            if values["geometry_source"] not in ("",) + GEOMETRY_SOURCES:
                raise ReferenceCardError(f"{where}: geometry_source must be one of "
                                         f"{list(GEOMETRY_SOURCES)} or blank, "
                                         f"got {values['geometry_source']!r}")
            card_id = values["card_id"]
            if card_id in cards:
                raise ReferenceCardError(f"{where}: duplicate card_id {card_id!r}")
            nums = {c: _number(c, values[c], where) for c in _RANGES}
            cards[card_id] = ReferenceCard(
                card_id=card_id, shape_group=values["shape_group"], source=values["source"],
                material=values["material"], coating=values["coating"],
                geometry_source=values["geometry_source"] or None,
                photo_path=values["photo_path"] or None, notes=values["notes"], **nums)
    return cards
