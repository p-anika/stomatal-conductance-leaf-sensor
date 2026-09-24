"""
Module 8: Session/metadata logger.

Records everything that isn't a raw sensor reading but is needed to
interpret one: plant ID, genotype, water treatment, working distance,
measured incidence angle, and a session_id that ties a block of
sensor_log.csv rows (and any images/reference-card readings taken in
that block) back to a specific plant on a specific day.

Call start_session() (or the interactive prompt_for_session()) once at
the start of a field session, before the logging loop in
combined_logger.py begins. Every row the logger writes during that
session should carry the returned session_id.
"""

import csv
import os
from dataclasses import dataclass, asdict
from datetime import datetime

METADATA_PATH = os.path.join("data", "raw", "session_metadata.csv")

FIELDNAMES = [
    "session_id", "timestamp_start", "plant_id", "genotype",
    "water_treatment", "working_distance_mm", "incidence_angle_deg", "notes",
]


@dataclass
class SessionMetadata:
    session_id: str
    timestamp_start: str
    plant_id: str
    genotype: str            # e.g. "Houjaku_Kuwazu" or "Williams_82"
    water_treatment: str     # "well_watered" or "drought_stressed"
    working_distance_mm: float
    incidence_angle_deg: float
    notes: str = ""


def _ensure_metadata_file():
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    if not os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def start_session(plant_id: str, genotype: str, water_treatment: str,
                   working_distance_mm: float, incidence_angle_deg: float,
                   notes: str = "") -> SessionMetadata:
    """
    Create a new session record, append it to session_metadata.csv, and
    return it so the caller (combined_logger.py) can tag every sensor
    row with its session_id.
    """
    _ensure_metadata_file()

    timestamp_start = datetime.now().isoformat()
    # Filesystem- and CSV-safe session id: plant + start time, colons swapped out.
    session_id = f"{plant_id}_{timestamp_start.replace(':', '-')}"

    metadata = SessionMetadata(
        session_id=session_id,
        timestamp_start=timestamp_start,
        plant_id=plant_id,
        genotype=genotype,
        water_treatment=water_treatment,
        working_distance_mm=working_distance_mm,
        incidence_angle_deg=incidence_angle_deg,
        notes=notes,
    )

    with open(METADATA_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(asdict(metadata))

    return metadata


def prompt_for_session() -> SessionMetadata:
    """
    Interactive helper for starting a session from the command line at
    the Pi. Called by combined_logger.py when run directly in the field.
    """
    print("--- New measurement session ---")
    plant_id = input("Plant ID (e.g. HK-03): ").strip()
    genotype = input("Genotype [Houjaku_Kuwazu / Williams_82]: ").strip()
    water_treatment = input("Water treatment [well_watered / drought_stressed]: ").strip()
    working_distance_mm = float(input("Working distance (mm): ").strip())
    incidence_angle_deg = float(input("Measured incidence angle (deg): ").strip())
    notes = input("Notes (optional): ").strip()

    return start_session(
        plant_id=plant_id,
        genotype=genotype,
        water_treatment=water_treatment,
        working_distance_mm=working_distance_mm,
        incidence_angle_deg=incidence_angle_deg,
        notes=notes,
    )


if __name__ == "__main__":
    # Run directly (`python -m acquisition.session_metadata` from repo root)
    # to test the prompts without starting the full sensor loop.
    session = prompt_for_session()
    print(f"Session started: {session.session_id}")
