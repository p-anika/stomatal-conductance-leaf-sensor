"""Readers for project data files (T2.2: reference-card manifest; T3.5: photo_features.csv;
T4.1: session folders and ground truth).

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
import json
import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CARDS_PATH = REPO_ROOT / "data" / "reference_cards.csv"
DEFAULT_GROUND_TRUTH_PATH = REPO_ROOT / "data" / "validation" / "ground_truth_gs.csv"

CARD_REQUIRED_COLUMNS = ("card_id", "shape_group", "source", "material", "thickness_mm",
                         "coating", "a_s_card", "emissivity", "area_mm2", "photo_path", "notes")
CARD_OPTIONAL_COLUMNS = ("d_min_mm", "d_eq_mm", "aspect_ratio", "eccentricity", "solidity",
                         "circularity", "edge_density", "geometry_source")
# Full Appendix B column order of data/reference_cards.csv (used when writing the file).
CARD_COLUMNS_ORDER = ("card_id", "shape_group", "source", "material", "thickness_mm", "coating",
                      "a_s_card", "emissivity", "area_mm2", "d_min_mm", "d_eq_mm", "aspect_ratio",
                      "eccentricity", "solidity", "circularity", "edge_density", "geometry_source",
                      "photo_path", "notes")
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


def load_reference_cards(path: str | Path = DEFAULT_CARDS_PATH) -> dict[str, ReferenceCard]:
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


# ---------------------------------------------------------------- photo_features.csv (T3.5)

# Geometry columns written by processing/extract_photo_features.py (one per T3.1 feature).
PHOTO_GEOMETRY_COLUMNS = ("area_mm2", "area_projected_mm2", "perimeter_mm", "d_min_mm", "d_max_mm",
                          "d_eq_mm", "aspect_ratio", "eccentricity", "solidity", "circularity",
                          "edge_density", "n_components")
PHOTO_REQUIRED_COLUMNS = ("timestamp", "image_file") + PHOTO_GEOMETRY_COLUMNS + ("ok", "reason")
_OK_TEXT = {"true": True, "false": False}


class SchemaError(ValueError):
    """A data file does not match its spec Appendix B schema (missing file, column or bad value)."""


class PhotoFeaturesError(SchemaError):
    """photo_features.csv is malformed (missing column, unreadable ok or timestamp)."""


def load_photo_features(path: str | Path) -> pd.DataFrame:
    """Read data/processed/<session>/photo_features.csv.

    Returns a DataFrame with timestamp as timezone-aware UTC (NaT where blank: photo not in the
    sensor log), ok as bool, geometry columns as float (NaN where the photo failed).
    """
    path = Path(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in PHOTO_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise PhotoFeaturesError(f"{path}: missing column(s) {missing}")
    ok = df["ok"].str.strip().str.lower().map(_OK_TEXT)
    if ok.isna().any():
        bad = df.loc[ok.isna(), "ok"].tolist()
        raise PhotoFeaturesError(f"{path}: column ok must be true or false, got {bad}")
    df["ok"] = ok.astype(bool)
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"].replace("", None), utc=True, format="ISO8601")
    except (ValueError, TypeError) as e:
        raise PhotoFeaturesError(f"{path}: unreadable timestamp ({e})") from e
    for col in PHOTO_GEOMETRY_COLUMNS + ("mm_per_pixel", "incidence_angle_deg"):
        if col in df.columns:
            num = pd.to_numeric(df[col].replace("", None), errors="coerce")
            if (num.isna() & (df[col] != "")).any():
                raise PhotoFeaturesError(f"{path}: column {col} has a value that is not a number")
            df[col] = num.astype(float)
    return df


# ---------------------------------------------------------------- session folders (T4.1)

# Spec Appendix B column lists, in order.
SENSOR_LOG_COLUMNS = (
    "session_id", "timestamp_iso", "timestamp_corrected_iso", "monotonic_s", "air_temp_c",
    "humidity_pct", "pressure_hpa", "lux", "tsl_visible", "tsl_infrared", "tsl_full_spectrum",
    "tsl_gain", "tsl_integration_ms", "wind_ms", "wind_raw", "wind_status", "wind_source",
    "thermal_frame_idx", "thermal_min_c", "thermal_max_c", "thermal_mean_c", "leaf_temp_c",
    "leaf_temp_median_c", "leaf_temp_std_c", "leaf_n_px", "card_temp_c", "card_temp_std_c",
    "card_n_px", "wet_ref_temp_c", "wet_ref_n_px", "image_file", "qc_flags")
# Text columns of the sensor log; every other column must be a number or blank.
SENSOR_LOG_TEXT_COLUMNS = ("session_id", "tsl_gain", "wind_status", "wind_source", "image_file",
                           "qc_flags")
SENSOR_LOG_TIME_COLUMNS = ("timestamp_iso", "timestamp_corrected_iso")
EVENTS_COLUMNS = ("timestamp_iso", "label", "note")
MANIFEST_FIELDS = ("session_id", "start_iso", "end_iso", "clock_status", "clock_offset_s",
                   "n_samples", "git_commit", "config_snapshot", "calibration_sha256",
                   "package_versions", "sensor_settings", "throttled_flags")
CLOCK_STATUSES = ("ntp_synced", "rtc", "manual_offset", "unsynced")       # spec B14
GROUND_TRUTH_COLUMNS = ("plant_id", "timestamp_iso", "instrument", "surface", "g_s_mol",
                        "leaf_position", "notes")
GT_INSTRUMENTS = ("porometer", "gravimetric")
GT_SURFACES = ("adaxial", "abaxial", "both_summed", "whole_leaf")


@dataclass(frozen=True)
class Session:
    """One loaded session. metadata: session.json (Appendix B session_metadata columns).
    sensor: sensor_log.csv with numbers as float (blank -> NaN) and the two timestamp columns
    as UTC datetimes (blank -> NaT). events: events.csv (empty table if the file is absent).
    paths: where each session file is (it may not exist, e.g. thermal_frames.f32).
    synthetic: True when run_manifest sensor_settings.synthetic is true (ml/synthetic_data.py)."""
    session_id: str
    metadata: dict[str, Any]
    sensor: pd.DataFrame
    events: pd.DataFrame
    manifest: dict[str, Any]
    paths: dict[str, Path]
    synthetic: bool


def list_sessions(raw_dir: str | Path = REPO_ROOT / "data" / "raw") -> list[str]:
    """Session ids (folder names) under raw_dir that contain run_manifest.json, sorted."""
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        return []
    return sorted(p.name for p in raw_dir.iterdir() if (p / "run_manifest.json").is_file())


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SchemaError(f"{path.name} not found in {path.parent}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SchemaError(f"{path}: not valid JSON ({e})") from e
    if not isinstance(data, dict):
        raise SchemaError(f"{path}: must hold a JSON object")
    return data


def _require_columns(path: Path, header: list[str], required: tuple[str, ...]) -> None:
    missing = [c for c in required if c not in header]
    if missing:
        raise SchemaError(f"{path}: missing column(s) {missing} (spec Appendix B)")
    extra = [c for c in header if c not in required]
    if extra:
        logger.warning("%s: extra column(s) %s not in spec Appendix B; kept", path.name, extra)


def _parse_time_column(path: Path, df: pd.DataFrame, col: str, required: bool) -> pd.Series:
    """ISO 8601 with a UTC offset -> UTC datetime. Blank -> NaT (an error when required).
    Row numbers in messages count the header as row 1."""
    values: list[pd.Timestamp] = []
    for pos, text in enumerate(df[col].astype(str)):
        text = text.strip()
        if text == "":
            if required:
                raise SchemaError(f"{path}: {col} is blank in row {pos + 2}")
            values.append(pd.NaT)
            continue
        try:
            t = pd.Timestamp(text)
        except ValueError as e:
            raise SchemaError(f"{path}: {col} in row {pos + 2} is not an ISO 8601 time: {text!r}") from e
        if t.tzinfo is None:
            raise SchemaError(f"{path}: {col} in row {pos + 2} has no UTC offset: {text!r}")
        values.append(t.tz_convert("UTC"))
    return pd.Series(pd.to_datetime(values, utc=True), index=df.index)


def _numeric_column(path: Path, df: pd.DataFrame, col: str) -> pd.Series:
    text = df[col].astype(str).str.strip()
    num = pd.to_numeric(text.where(text != "", None), errors="coerce")
    bad = (num.isna() & (text != "")).to_numpy()
    if bad.any():
        pos = int(bad.argmax())
        raise SchemaError(f"{path}: {col} in row {pos + 2} is not a number: {text.iloc[pos]!r}")
    return num.astype(float)


def load_sensor_log(path: str | Path) -> pd.DataFrame:
    """Read and check sensor_log.csv (Appendix B columns; extra columns kept with a warning)."""
    path = Path(path)
    if not path.is_file():
        raise SchemaError(f"sensor_log.csv not found: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    _require_columns(path, list(df.columns), SENSOR_LOG_COLUMNS)
    for col in SENSOR_LOG_COLUMNS:
        if col in SENSOR_LOG_TIME_COLUMNS:
            df[col] = _parse_time_column(path, df, col, required=(col == "timestamp_iso"))
        elif col not in SENSOR_LOG_TEXT_COLUMNS:
            df[col] = _numeric_column(path, df, col)
    return df


def load_events(path: str | Path) -> pd.DataFrame:
    """Read events.csv; an absent file gives an empty table with the Appendix B columns."""
    path = Path(path)
    if not path.is_file():
        logger.warning("%s not found; no events for this session", path)
        return pd.DataFrame({c: pd.Series(dtype=str) for c in EVENTS_COLUMNS})
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    _require_columns(path, list(df.columns), EVENTS_COLUMNS)
    _parse_time_column(path, df, "timestamp_iso", required=True)     # check only; text kept
    return df


def load_session(session_id: str, raw_dir: str | Path = REPO_ROOT / "data" / "raw") -> Session:
    """Load data/raw/<session_id>/: run_manifest.json, session.json, sensor_log.csv, events.csv.
    Raises SchemaError naming the file and the field for any schema problem."""
    from acquisition.session_metadata import FIELDNAMES      # laptop-safe; imported here to keep data_io light

    folder = Path(raw_dir) / session_id
    if not folder.is_dir():
        raise SchemaError(f"no session folder {folder} (session {session_id!r})")
    paths = {"folder": folder, "manifest": folder / "run_manifest.json",
             "metadata": folder / "session.json", "sensor_log": folder / "sensor_log.csv",
             "events": folder / "events.csv", "thermal_frames": folder / "thermal_frames.f32",
             "images": folder / "images", "logger_log": folder / "logger.log"}

    manifest = _read_json(paths["manifest"])
    missing = [k for k in MANIFEST_FIELDS if k not in manifest]
    if missing:
        raise SchemaError(f"{paths['manifest']}: missing field(s) {missing} (spec Appendix B)")
    if manifest["clock_status"] not in CLOCK_STATUSES:
        raise SchemaError(f"{paths['manifest']}: clock_status must be one of {list(CLOCK_STATUSES)}, "
                          f"got {manifest['clock_status']!r}")

    metadata = _read_json(paths["metadata"])
    missing = [k for k in FIELDNAMES if k not in metadata]
    if missing:
        raise SchemaError(f"{paths['metadata']}: missing field(s) {missing} (spec Appendix B)")
    for name, value in (("run_manifest.json", manifest["session_id"]),
                        ("session.json", metadata["session_id"])):
        if value != session_id:
            raise SchemaError(f"{name}: session_id {value!r} does not match the folder {session_id!r}")

    sensor = load_sensor_log(paths["sensor_log"])
    ids = set(sensor["session_id"].astype(str).str.strip()) - {""}
    if ids - {session_id}:
        raise SchemaError(f"{paths['sensor_log']}: session_id column holds {sorted(ids)}, "
                          f"expected {session_id!r}")
    settings = manifest.get("sensor_settings")
    synthetic = isinstance(settings, dict) and settings.get("synthetic") is True
    return Session(session_id, metadata, sensor, load_events(paths["events"]), manifest, paths, synthetic)


def load_ground_truth(path: str | Path = DEFAULT_GROUND_TRUTH_PATH) -> pd.DataFrame:
    """Read data/validation/ground_truth_gs.csv (typed in by hand). g_s_mol in mol m-2 s-1, >= 0.
    Which surface label matches this project's g_s must be settled before real use (A40)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ground-truth file not found: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    _require_columns(path, list(df.columns), GROUND_TRUTH_COLUMNS)
    for pos, row in enumerate(df.itertuples(index=False)):
        where = f"{path.name} row {pos + 2}"
        if row.plant_id.strip() == "":
            raise SchemaError(f"{where}: plant_id must not be blank")
        if row.instrument not in GT_INSTRUMENTS:
            raise SchemaError(f"{where}: instrument must be one of {list(GT_INSTRUMENTS)}, "
                              f"got {row.instrument!r}")
        if row.surface not in GT_SURFACES:
            raise SchemaError(f"{where}: surface must be one of {list(GT_SURFACES)}, got {row.surface!r}")
    df["g_s_mol"] = _numeric_column(path, df, "g_s_mol")
    if df["g_s_mol"].isna().any() or (df["g_s_mol"] < 0).any():
        raise SchemaError(f"{path}: g_s_mol must be a number >= 0 in every row")
    df["timestamp_iso"] = _parse_time_column(path, df, "timestamp_iso", required=True)
    return df
