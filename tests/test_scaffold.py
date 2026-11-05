"""Phase 0 checks (T0.1 to T0.5)."""
import datetime
import subprocess
from pathlib import Path

import numpy as np
import pytest

REQUIRED_TOP_KEYS = {"physics", "convection", "methods", "spatial", "light_sources", "card",
                     "enumerations", "acquisition", "clock", "windows", "sensors",
                     "uncertainty", "ml"}

HARDWARE_PREFIXES = ("adafruit", "smbus2", "rpi", "picamera2")


def test_trivial() -> None:
    assert 1 + 1 == 2


def test_config_loads_with_all_sections(default_config: dict) -> None:
    assert REQUIRED_TOP_KEYS <= set(default_config)


def test_config_spot_values_match_appendix_b(default_config: dict) -> None:
    phys = default_config["physics"]
    assert phys["faces"] in (1, 2)
    assert phys["stefan_boltzmann"] == pytest.approx(5.670374419e-8)
    assert phys["gamma_pa_per_k_per_hpa"] == pytest.approx(0.0665)
    assert default_config["convection"]["enhancement_factor"] == pytest.approx(1.0)
    # led_lamp must have no k: it has to be calibrated (T1.1)
    assert default_config["light_sources"]["led_lamp"]["k_wm2_per_lux"] is None
    # YAML reads an unquoted date as a date object; the logger will rely on that
    assert default_config["clock"]["min_valid_date"] == datetime.date(2026, 10, 1)
    assert default_config["sensors"]["arducam"]["resolution"] == [2304, 1296]


def test_synthetic_frame_shape(synthetic_thermal_frame: np.ndarray) -> None:
    assert synthetic_thermal_frame.shape == (24, 32)
    assert synthetic_thermal_frame.max() > synthetic_thermal_frame.min() + 2.0


def test_fake_session_dir(fake_session_dir: Path) -> None:
    assert (fake_session_dir / "images").is_dir()


def _req_names(path: Path) -> list:
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.append(line.split("==")[0].split(">=")[0].strip().lower())
    return names


def test_requirements_split(repo_root: Path) -> None:
    laptop = _req_names(repo_root / "requirements.txt")
    pi = _req_names(repo_root / "requirements-pi.txt")
    for pkg in ["numpy", "scipy", "pandas", "pyyaml", "scikit-learn", "joblib",
                "opencv-python-headless", "matplotlib", "statsmodels", "pytest"]:
        assert pkg in laptop
    assert not [n for n in laptop if n.startswith(HARDWARE_PREFIXES)]
    for pkg in ["adafruit-circuitpython-bme280", "adafruit-circuitpython-tsl2591",
                "adafruit-circuitpython-mlx90640", "adafruit-blinka", "numpy", "pyyaml"]:
        assert pkg in pi


def test_gitignore_rules(repo_root: Path) -> None:
    for path in ["x/__pycache__/a.pyc", ".venv/a", ".pytest_cache/a", ".DS_Store",
                 "data/raw/s1/images/a.jpg", "data/raw/s1/thermal_frames.f32"]:
        result = subprocess.run(["git", "check-ignore", "-q", path], cwd=repo_root)
        assert result.returncode == 0, f"{path} should be ignored"
    # summary CSVs must stay committable (Section 6, Git)
    result = subprocess.run(["git", "check-ignore", "-q", "data/raw/s1/sensor_log.csv"], cwd=repo_root)
    assert result.returncode == 1


def test_data_raw_has_no_stray_files(repo_root: Path) -> None:
    raw = repo_root / "data" / "raw"
    stray = [p.name for p in raw.iterdir() if p.is_file() and p.name != ".gitkeep"]
    assert stray == [], f"stray files in data/raw: {stray}"
    assert any((raw / "debug_logs").glob("*.csv"))
