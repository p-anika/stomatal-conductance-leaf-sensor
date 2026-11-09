"""Session metadata: who, what and how for one measurement session (T2.1, spec C1-C3).

Records everything that is not a sensor reading but is needed to interpret one: plant,
genotype, treatment, growth stage, randomised block and position, light set-up, reference
cards, operator. Columns are spec Appendix B (data/raw/session_metadata.csv); a copy of each
session goes to data/raw/<session_id>/session.json.

Rules:
- Choices (genotype, water_treatment, reproductive_stage, light_source, fan_state,
  background, dry_reference_type) come from config `enumerations`; anything else is
  rejected with a message naming the field, the bad value and the allowed values.
- position and leaf_position are normalised to lowercase_with_underscores
  ("Bench A-3" -> "bench_a_3") so the same place is never spelled two ways.
- Lamp fields are required for light_source led_lamp and must be blank otherwise.
- Raw data is never rewritten (owner decision 2026-11-08, deviation from T2.1): if
  session_metadata.csv exists with any header other than the current one, nothing is
  written and MetadataFileError asks the user to move the file aside. There is no migration.
- dry_reference_type card with a card_id missing from data/reference_cards.csv logs a
  warning only (owner decision 2026-11-08).

Use from the command line (flags for any field; prompts for the rest):
    python -m acquisition.session_metadata --plant-id HK-03 --genotype Houjaku_Kuwazu ...
combined_logger.py calls prompt_for_session() with no arguments.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from processing.config import load_config
from processing.data_io import ReferenceCardError, load_reference_cards

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"
METADATA_RELPATH = Path("data") / "raw" / "session_metadata.csv"
MANIFEST_RELPATH = Path("data") / "reference_cards.csv"

# Spec Appendix B column order.
FIELDNAMES = [
    "session_id", "timestamp_start", "plant_id", "genotype", "water_treatment",
    "reproductive_stage", "block", "position", "leaf_position", "light_source",
    "lamp_distance_mm", "window_blocked", "acclimation_minutes", "fan_state",
    "working_distance_mm", "incidence_angle_deg", "card_angle_deg", "lamp_angle_deg",
    "lamp_to_card_mm", "background", "card_id", "dry_reference_type",
    "wet_reference_present", "operator", "porometer_gs_mol", "notes",
]
# Fields the operator supplies; session_id and timestamp_start are generated.
OPERATOR_FIELDS = FIELDNAMES[2:]

ENUM_FIELDS = ("genotype", "water_treatment", "reproductive_stage", "light_source",
               "fan_state", "background", "dry_reference_type")
LAMP_FIELDS = ("lamp_distance_mm", "lamp_angle_deg", "lamp_to_card_mm")
LAMP_SOURCE = "led_lamp"
OPTIONAL_FIELDS = ("card_id", "card_angle_deg", "porometer_gs_mol", "notes") + LAMP_FIELDS
POSITIVE_FIELDS = ("lamp_distance_mm", "working_distance_mm", "lamp_to_card_mm")
ANGLE_FIELDS = ("incidence_angle_deg", "card_angle_deg", "lamp_angle_deg")
NONNEGATIVE_FIELDS = ("acclimation_minutes", "porometer_gs_mol")
BOOL_FIELDS = ("window_blocked", "wet_reference_present")
LABEL_FIELDS = ("position", "leaf_position")
TEXT_FIELDS = ("operator", "card_id", "notes")

# Definitions, not physical constants: angles are measured from the surface normal, so
# 0-90 degrees covers every possible tilt.
ANGLE_MIN_DEG, ANGLE_MAX_DEG = 0.0, 90.0
_TRUE = ("true", "yes", "y", "1")
_FALSE = ("false", "no", "n", "0")
_PLANT_ID = re.compile(r"^[A-Za-z0-9_-]+$")   # becomes part of a folder name

HELP = {
    "plant_id": "plant label, e.g. HK-03 (letters, digits, - and _)",
    "genotype": "soybean genotype",
    "water_treatment": "watering treatment",
    "reproductive_stage": "growth stage (VE, VC, V1-V15 = trifoliate nodes, R1-R8)",
    "block": "randomised block number (1, 2, ...)",
    "position": "place of the plant in the block, e.g. bench_a_3",
    "leaf_position": "which leaf is measured, e.g. top_trifoliate",
    "light_source": "light source",
    "lamp_distance_mm": "lamp to leaf distance, mm (led_lamp only)",
    "window_blocked": "window covered? true/false",
    "acclimation_minutes": "minutes the plant sat under this light before logging",
    "fan_state": "fan setting",
    "working_distance_mm": "camera lens to leaf distance, mm",
    "incidence_angle_deg": "leaf tilt from facing the camera, degrees 0-90",
    "card_angle_deg": "reference card tilt, degrees 0-90 (optional)",
    "lamp_angle_deg": "lamp tilt from pointing straight at the leaf, degrees 0-90 (led_lamp only)",
    "lamp_to_card_mm": "lamp to reference card distance, mm (led_lamp only)",
    "background": "background behind the leaf",
    "card_id": "reference card ID from data/reference_cards.csv (optional)",
    "dry_reference_type": "dry reference used for the Jones index",
    "wet_reference_present": "wet reference in view? true/false",
    "operator": "initials of the person running the session",
    "porometer_gs_mol": "porometer g_s, mol m-2 s-1 (optional)",
    "notes": "anything unusual (optional)",
}


class MetadataFileError(Exception):
    """A metadata file on disk is in a state that must not be written over."""


@dataclass(frozen=True)
class SessionMetadata:
    """One session (Appendix B columns). Lengths mm, angles degrees, time minutes,
    porometer g_s mol m-2 s-1. None means left blank (optional or not applicable)."""
    session_id: str
    timestamp_start: str            # ISO 8601 with UTC offset
    plant_id: str
    genotype: str
    water_treatment: str
    reproductive_stage: str
    block: int
    position: str
    leaf_position: str
    light_source: str
    lamp_distance_mm: float | None
    window_blocked: bool
    acclimation_minutes: float
    fan_state: str
    working_distance_mm: float
    incidence_angle_deg: float
    card_angle_deg: float | None
    lamp_angle_deg: float | None
    lamp_to_card_mm: float | None
    background: str
    card_id: str | None
    dry_reference_type: str
    wet_reference_present: bool
    operator: str
    porometer_gs_mol: float | None
    notes: str


# ---------------------------------------------------------------- one field

def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _float(name: str, text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {text!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {text!r}")
    return value


def allowed_values(name: str, cfg: dict[str, Any]) -> list[str]:
    return [str(v) for v in cfg["enumerations"][name]]


def parse_field(name: str, value: Any, cfg: dict[str, Any]) -> Any:
    """Check and convert one operator field. Returns the typed value (None when an
    optional field is blank). Raises ValueError with a message that names the field."""
    if name not in OPERATOR_FIELDS:
        raise ValueError(f"unknown field {name!r}")
    text = _text(value)
    if text == "":
        if name in OPTIONAL_FIELDS:
            return "" if name == "notes" else None
        raise ValueError(f"{name} is required")

    if name in ENUM_FIELDS:
        allowed = allowed_values(name, cfg)
        if text not in allowed:
            raise ValueError(f"{name} must be one of {allowed}, got {text!r} "
                             "(spelling and capitals must match exactly)")
        return text
    if name == "plant_id":
        if not _PLANT_ID.match(text):
            raise ValueError(f"plant_id may only contain letters, digits, - and _ "
                             f"(it becomes part of a folder name), got {text!r}")
        return text
    if name == "block":
        if not re.fullmatch(r"\d+", text) or int(text) < 1:
            raise ValueError(f"block must be a whole number of 1 or more, got {text!r}")
        return int(text)
    if name in LABEL_FIELDS:
        label = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
        if not label:
            raise ValueError(f"{name} must contain letters or digits, got {text!r}")
        return label
    if name in BOOL_FIELDS:
        low = text.lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise ValueError(f"{name} must be true or false (yes/no, 1/0 also work), got {text!r}")
    if name in POSITIVE_FIELDS:
        v = _float(name, text)
        if v <= 0:
            raise ValueError(f"{name} must be greater than 0 mm, got {text!r}")
        return v
    if name in ANGLE_FIELDS:
        v = _float(name, text)
        if not ANGLE_MIN_DEG <= v <= ANGLE_MAX_DEG:
            raise ValueError(f"{name} must be between 0 and 90 degrees, got {text!r}")
        return v
    if name in NONNEGATIVE_FIELDS:
        v = _float(name, text)
        if v < 0:
            raise ValueError(f"{name} must be 0 or more, got {text!r}")
        return v
    return text   # operator, card_id, notes: free text


# ---------------------------------------------------------------- whole record

def validate_metadata(raw: Mapping[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Check every operator field and the lamp rule. Returns typed values for all
    OPERATOR_FIELDS. Raises ValueError listing every problem at once (one per line)."""
    unknown = sorted(set(raw) - set(OPERATOR_FIELDS))
    errors = [f"unknown field(s) {unknown}"] if unknown else []
    clean: dict[str, Any] = {}
    for name in OPERATOR_FIELDS:
        try:
            clean[name] = parse_field(name, raw.get(name), cfg)
        except ValueError as e:
            errors.append(str(e))
    if "light_source" in clean:
        errors += _lamp_errors(clean)
    if errors:
        raise ValueError("invalid session metadata:\n  " + "\n  ".join(errors))
    return clean


def _lamp_errors(clean: Mapping[str, Any]) -> list[str]:
    """Lamp geometry only means something for the lamp."""
    errors = []
    for name in LAMP_FIELDS:
        if name not in clean:
            continue                     # already reported as a bad value
        if clean["light_source"] == LAMP_SOURCE and clean[name] is None:
            errors.append(f"{name} is required when light_source is {LAMP_SOURCE}")
        elif clean["light_source"] != LAMP_SOURCE and clean[name] is not None:
            errors.append(f"{name} must be blank unless light_source is {LAMP_SOURCE} "
                          f"(light_source is {clean['light_source']!r})")
    return errors


def make_session(raw: Mapping[str, Any], cfg: dict[str, Any],
                 now: datetime | None = None) -> SessionMetadata:
    """Validate and build a SessionMetadata. now must be time-zone aware (default: the
    computer's local time with its UTC offset). session_id = plant_id + '_' + start time
    with ':' replaced by '-' (safe as a folder name)."""
    clean = validate_metadata(raw, cfg)
    now = now if now is not None else datetime.now().astimezone()
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("timestamp_start needs a time zone (UTC offset); got a naive time")
    timestamp = now.isoformat(timespec="seconds")
    session_id = f"{clean['plant_id']}_{timestamp.replace(':', '-')}"
    return SessionMetadata(session_id=session_id, timestamp_start=timestamp, **clean)


# ---------------------------------------------------------------- files

def check_metadata_file(csv_path: Path) -> None:
    """Refuse (MetadataFileError) if csv_path exists with any header but the current one.
    The file is only read, never changed."""
    if not csv_path.exists():
        return
    with open(csv_path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), None)
    if header != FIELDNAMES:
        raise MetadataFileError(
            f"{csv_path} has an old or unknown header, so nothing was written. Raw data is "
            "never rewritten. Please move it aside (for example rename it to "
            "session_metadata_old.csv) and run again; a new file with the current "
            f"columns will be created.\n  found:    {header}\n  expected: {FIELDNAMES}")


def _warn_if_card_unknown(s: SessionMetadata, manifest_path: Path) -> None:
    """Warning only: the card can be added to the manifest after the session."""
    if s.dry_reference_type != "card":
        return
    try:
        cards = load_reference_cards(manifest_path)
    except (FileNotFoundError, ReferenceCardError) as e:
        logger.warning("could not read the reference-card manifest (%s); card_id %r was not "
                       "checked", e, s.card_id)
        return
    if s.card_id not in cards:
        logger.warning("dry_reference_type is card but card_id %r is not in the reference-card "
                       "manifest %s; add the card there before analysis", s.card_id, manifest_path)


def _csv_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def start_session(raw: Mapping[str, Any], cfg: dict[str, Any], root: str | Path = ".",
                  manifest_path: str | Path | None = None,
                  now: datetime | None = None) -> SessionMetadata:
    """Validate, then write data/raw/<session_id>/session.json and append one row to
    data/raw/session_metadata.csv (both under root). Nothing is written if validation
    fails, the CSV header is not current, or session.json already exists."""
    s = make_session(raw, cfg, now)
    root = Path(root)
    csv_path = root / METADATA_RELPATH
    check_metadata_file(csv_path)
    session_dir = csv_path.parent / s.session_id
    json_path = session_dir / "session.json"
    if json_path.exists():
        raise MetadataFileError(f"{json_path} already exists; it is never overwritten")
    _warn_if_card_unknown(s, Path(manifest_path) if manifest_path else root / MANIFEST_RELPATH)

    session_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, "x", encoding="utf-8") as f:        # "x": fail rather than overwrite
        json.dump(asdict(s), f, indent=2)
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if new_file:
            writer.writeheader()
        writer.writerow({k: _csv_value(v) for k, v in asdict(s).items()})
    logger.info("session %s started; metadata in %s", s.session_id, json_path)
    return s


# ---------------------------------------------------------------- prompts and command line

def _prompt_text(name: str, cfg: dict[str, Any]) -> str:
    hint = f" {allowed_values(name, cfg)}" if name in ENUM_FIELDS else ""
    optional = " (optional, Enter to skip)" if name in OPTIONAL_FIELDS else ""
    return f"{name}: {HELP[name]}{hint}{optional}: "


def collect_answers(given: Mapping[str, Any], cfg: dict[str, Any],
                    input_fn: Callable[[str], str] = input) -> dict[str, Any]:
    """Return a full set of answers: given values, plus a prompt for each missing field.
    A bad typed answer is explained and asked again. Lamp fields are not asked unless
    light_source is led_lamp."""
    answers = dict(given)
    for name in OPERATOR_FIELDS:
        if name in answers:
            continue
        if name in LAMP_FIELDS and _text(answers.get("light_source")) != LAMP_SOURCE:
            answers[name] = ""
            continue
        while True:
            text = input_fn(_prompt_text(name, cfg))
            try:
                parse_field(name, text, cfg)
            except ValueError as e:
                print(f"  {e}. Try again.", file=sys.stderr)   # interactive feedback
                continue
            answers[name] = text
            break
    return answers


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m acquisition.session_metadata",
        description="Start a measurement session. Give any field as a flag; you will be "
                    "asked for the rest.")
    for name in OPERATOR_FIELDS:
        p.add_argument("--" + name.replace("_", "-"), dest=name, default=None, help=HELP[name])
    p.add_argument("--root", default=".", help="repository folder (default: current folder)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to config.yaml")
    return p


def main(argv: list[str] | None = None, input_fn: Callable[[str], str] = input,
         manifest_path: str | Path | None = None, now: datetime | None = None) -> SessionMetadata:
    """Command-line entry point. A bad flag value exits with a message (exit code 2)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    given = {n: getattr(args, n) for n in OPERATOR_FIELDS if getattr(args, n) is not None}
    for name, value in given.items():
        try:
            parse_field(name, value, cfg)
        except ValueError as e:
            parser.error(str(e))
    answers = collect_answers(given, cfg, input_fn)
    try:
        return start_session(answers, cfg, root=args.root, manifest_path=manifest_path, now=now)
    except (ValueError, MetadataFileError) as e:
        parser.error(str(e))
        raise   # unreachable: parser.error exits


def prompt_for_session(cfg: dict[str, Any] | None = None, input_fn: Callable[[str], str] = input,
                       root: str | Path = ".", manifest_path: str | Path | None = None
                       ) -> SessionMetadata:
    """Ask for every field and start the session (used by combined_logger.py)."""
    cfg = cfg if cfg is not None else load_config(DEFAULT_CONFIG)
    print("--- New measurement session ---")
    return start_session(collect_answers({}, cfg, input_fn), cfg, root=root,
                         manifest_path=manifest_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    session = main()
    print(f"Session started: {session.session_id}")
