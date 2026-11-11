"""RGB photos for the leaf geometry (spec N1, Section 8). [PI for ArducamCamera only]

    camera.capture(session_dir, timestamp, label="leaf") -> "images/<timestamp>_<label>.jpg" or None

Classes:
    ArducamCamera  the real camera through picamera2 (imported inside the constructor, so this module
                   imports on a laptop). MANUAL focus: AfMode = Manual and
                       LensPosition [dioptres] = 1000 / working_distance_mm
                   unless sensors.arducam.lens_position is set. Resolution from sensors.arducam.resolution.
    FileCamera     hands back images from a folder in name order, cycling (tests and laptop runs).
    NullCamera     takes no photo, returns None (camera disabled or not fitted).
Each saved image gets a JSON sidecar with the same stem: camera, exposure (us), analogue gain,
lens position, working distance (mm) and resolution. ':' in the timestamp becomes '-' so the file
name is valid on every system.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

IMAGE_DIR = "images"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
MM_PER_M = 1000.0      # 1 dioptre = 1 / (distance in m) = 1000 / (distance in mm)


class CameraUnavailable(RuntimeError):
    """The camera library or the camera itself is not available (not a Pi, not connected)."""


def lens_position_dioptres(cfg: dict[str, Any]) -> float:
    """sensors.arducam.lens_position if set, else 1000 / working_distance_mm (dioptres)."""
    a = cfg["sensors"]["arducam"]
    if a.get("lens_position") is not None:
        return float(a["lens_position"])
    wd = a.get("working_distance_mm")
    if wd is None or not float(wd) > 0:
        raise ValueError("set sensors.arducam.working_distance_mm (mm, lens to leaf) or lens_position "
                         "(dioptres) in config.yaml before using the camera")
    return MM_PER_M / float(wd)


def _image_path(session_dir: str | Path, timestamp: str, label: str, suffix: str = ".jpg") -> tuple[Path, str]:
    name = f"{str(timestamp).replace(':', '-')}_{label}{suffix}"
    rel = f"{IMAGE_DIR}/{name}"
    path = Path(session_dir) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    return path, rel


def _write_sidecar(image_path: Path, meta: dict[str, Any]) -> None:
    image_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


class NullCamera:
    """No camera: capture() saves nothing and returns None."""

    def capture(self, session_dir: str | Path, timestamp: str, label: str = "leaf") -> str | None:
        return None


class FileCamera:
    """Copies the next image from source_dir (name order, cycling) into the session."""

    def __init__(self, source_dir: str | Path):
        self.files = sorted(p for p in Path(source_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not self.files:
            raise ValueError(f"no images in {source_dir}")
        self._i = 0

    def capture(self, session_dir: str | Path, timestamp: str, label: str = "leaf") -> str:
        src = self.files[self._i % len(self.files)]
        self._i += 1
        path, rel = _image_path(session_dir, timestamp, label, src.suffix.lower())
        shutil.copyfile(src, path)
        _write_sidecar(path, {"camera": "file", "source_file": src.name, "timestamp": str(timestamp)})
        return rel


class ArducamCamera:
    """Arducam through picamera2 with manual focus. [PI]

    picam / af_manual let tests pass a fake camera object and the manual-focus constant."""

    def __init__(self, cfg: dict[str, Any], picam: Any = None, af_manual: Any = None):
        a = cfg["sensors"]["arducam"]
        self.lens_position = lens_position_dioptres(cfg)
        self.working_distance_mm = a.get("working_distance_mm")
        self.resolution = list(a["resolution"])
        if picam is None:
            try:
                from libcamera import controls                       # comes with picamera2 (apt)
                from picamera2 import Picamera2
            except ImportError as e:
                raise CameraUnavailable("picamera2 is not installed: this needs the Raspberry Pi "
                                        "(sudo apt install python3-picamera2)") from e
            try:
                picam = Picamera2()
            except (RuntimeError, IndexError) as e:
                raise CameraUnavailable(f"no camera found ({e}); check the ribbon cable and run "
                                        "rpicam-hello --list-cameras") from e
            af_manual = controls.AfModeEnum.Manual
        self.picam = picam
        picam.configure(picam.create_still_configuration(main={"size": tuple(self.resolution)}))
        picam.start()
        # manual focus at the fixed working distance: autofocus would refocus on the background
        picam.set_controls({"AfMode": af_manual, "LensPosition": self.lens_position})

    def capture(self, session_dir: str | Path, timestamp: str, label: str = "leaf") -> str:
        path, rel = _image_path(session_dir, timestamp, label)
        self.picam.capture_file(str(path))
        md = self.picam.capture_metadata() or {}
        _write_sidecar(path, {"camera": "arducam", "timestamp": str(timestamp),
                              "exposure_us": md.get("ExposureTime"), "analogue_gain": md.get("AnalogueGain"),
                              "lens_position_dioptres": md.get("LensPosition", self.lens_position),
                              "working_distance_mm": self.working_distance_mm, "resolution": self.resolution})
        return rel
