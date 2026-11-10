"""Measure the leaf in every photo of a session (T3.5).

    python -m processing.extract_photo_features --session SESSION_ID [--root .] [--mm-per-pixel S]
                                                 [--mode leaf|card] [--incidence-angle DEG]

Reads data/raw/SESSION_ID/images/*.jpg|jpeg|png (never writes there) and writes
data/processed/SESSION_ID/photo_features.csv plus overlays/<image>_overlay.png for review.

Scale, mm px-1 (owner decision 2026-11-09, until camera_scale.py exists): --mm-per-pixel for
every image, else images/scale.csv (image_file, mm_per_pixel, notes) per image. An image with no
scale is kept as a row with ok false and reason missing_input. No scale source at all = error.
Tilt: --incidence-angle, else incidence_angle_deg from data/raw/SESSION_ID/session.json, else none.
Photo time: timestamp_corrected_iso (else timestamp_iso) of the sensor_log.csv row whose
image_file names the photo (matched on the file name); blank if no row names it.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from processing.config import load_config
from processing.data_io import PHOTO_GEOMETRY_COLUMNS
from processing.leaf_segmentation import segment_and_measure
from processing.measurement_windows import row_times

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
SCALE_FILE = "scale.csv"
OUTPUT_COLUMNS = (["timestamp", "image_file"] + list(PHOTO_GEOMETRY_COLUMNS)
                  + ["ok", "reason", "flags", "mm_per_pixel", "incidence_angle_deg"])


def read_image(path: Path) -> np.ndarray | None:
    """BGR uint8 image, or None if unreadable. imdecode copes with any characters in the path."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    return img


def load_scales(images_dir: Path) -> dict[str, float]:
    """image file name -> mm_per_pixel from images/scale.csv ({} if the file is absent)."""
    path = images_dir / SCALE_FILE
    if not path.is_file():
        return {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {"image_file", "mm_per_pixel"} <= set(df.columns):
        raise SystemExit(f"{path} needs columns image_file and mm_per_pixel")
    out: dict[str, float] = {}
    for name, val in zip(df["image_file"], df["mm_per_pixel"]):
        try:
            s = float(val)
        except ValueError:
            raise SystemExit(f"{path}: mm_per_pixel for {name!r} is not a number: {val!r}")
        if not (s > 0 and math.isfinite(s)):
            raise SystemExit(f"{path}: mm_per_pixel for {name!r} must be positive, got {val!r}")
        out[Path(name).name] = s
    return out


def photo_times(session_dir: Path) -> dict[str, str]:
    """image file name -> ISO time of the sensor_log row that names it ({} if no log)."""
    path = session_dir / "sensor_log.csv"
    if not path.is_file():
        return {}
    log = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "image_file" not in log.columns:
        return {}
    times = row_times(log)
    out: dict[str, str] = {}
    for name, t in zip(log["image_file"], times):
        name = Path(str(name).strip().replace("\\", "/")).name
        if name and pd.notna(t) and name not in out:      # first row naming the photo wins
            out[name] = t.isoformat()
    return out


def session_incidence_angle(session_dir: Path) -> float | None:
    path = session_dir / "session.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8")).get("incidence_angle_deg")
    return None if value is None else float(value)


def _row(image_file: str, timestamp: str, s: float | None, angle: float | None) -> dict[str, Any]:
    row: dict[str, Any] = {c: "" for c in OUTPUT_COLUMNS}
    row.update(timestamp=timestamp, image_file=image_file, ok=False,
               mm_per_pixel="" if s is None else s, incidence_angle_deg="" if angle is None else angle)
    return row


def extract_session(root: Path, session_id: str, mm_per_pixel: float | None, mode: str,
                    incidence_angle_deg: float | None, cfg: dict[str, Any]) -> Path:
    """Process every image of one session; return the path of photo_features.csv."""
    session_dir = root / "data" / "raw" / session_id
    images_dir = session_dir / "images"
    if not images_dir.is_dir():
        raise SystemExit(f"no images folder at {images_dir}")
    scales = load_scales(images_dir)
    if mm_per_pixel is None and not scales:
        raise SystemExit(f"no scale: pass --mm-per-pixel or create {images_dir / SCALE_FILE} "
                         "(columns image_file, mm_per_pixel, notes)")
    angle = incidence_angle_deg if incidence_angle_deg is not None else session_incidence_angle(session_dir)
    times = photo_times(session_dir)
    out_dir = root / "data" / "processed" / session_id
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    for path in images:
        s = mm_per_pixel if mm_per_pixel is not None else scales.get(path.name)
        row = _row(path.name, times.get(path.name, ""), s, angle)
        img = read_image(path)
        if s is None or img is None:
            row["reason"] = "missing_input"
            logger.warning("%s: %s", path.name, "no mm_per_pixel" if s is None else "unreadable image")
            rows.append(row)
            continue
        r = segment_and_measure(img, s, mode, incidence_angle_deg=angle, params=cfg["segmentation"])
        row.update(ok=r.valid, reason=r.reason, flags=";".join(r.flags))
        if r.valid:
            row.update({c: getattr(r, c) for c in PHOTO_GEOMETRY_COLUMNS})
        else:
            row["n_components"] = r.n_components
        if r.overlay is not None:
            ok, buf = cv2.imencode(".png", r.overlay)
            if ok:
                (overlay_dir / f"{path.stem}_overlay.png").write_bytes(buf.tobytes())
        logger.info("%s: %s", path.name, r.reason)
        rows.append(row)

    csv_path = out_dir / "photo_features.csv"
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(csv_path, index=False)
    logger.info("wrote %d rows to %s", len(rows), csv_path)
    return csv_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m processing.extract_photo_features",
                                description="Measure leaf geometry in every photo of a session (T3.5).")
    p.add_argument("--session", required=True, help="session id (folder name under data/raw/)")
    p.add_argument("--root", default=".", help="repository root (default: current folder)")
    p.add_argument("--mm-per-pixel", type=float, default=None,
                   help="scale for every image, mm per pixel (else images/scale.csv)")
    p.add_argument("--mode", choices=("leaf", "card"), default="leaf", help="what the photos show")
    p.add_argument("--incidence-angle", type=float, default=None,
                   help="camera tilt in degrees (else session.json incidence_angle_deg)")
    p.add_argument("--config", default=None, help="config.yaml path (default: <root>/config.yaml, "
                   "falling back to the repository's)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = Path(args.root)
    cfg_path = Path(args.config) if args.config else root / "config.yaml"
    if not cfg_path.is_file():
        cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    if args.mm_per_pixel is not None and not (args.mm_per_pixel > 0 and math.isfinite(args.mm_per_pixel)):
        raise SystemExit(f"--mm-per-pixel must be positive, got {args.mm_per_pixel}")
    extract_session(root, args.session, args.mm_per_pixel, args.mode, args.incidence_angle,
                    load_config(cfg_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
