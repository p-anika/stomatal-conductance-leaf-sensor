"""Shared pytest fixtures (T0.4).

Fixtures:
- repo_root: path to the repository root.
- default_config: config.yaml parsed into a dict.
- synthetic_thermal_frame: a fake 24 x 32 MLX90640 frame in deg C with a warm "leaf" block.
- fake_session_dir: an empty data/raw/<session_id>/ style folder inside pytest's tmp_path.

Hardware tests use @pytest.mark.hardware and are skipped unless pytest is run with --hardware.
"""
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# MLX90640 array shape (rows, columns) from the sensor's datasheet: 24 x 32 pixels.
THERMAL_SHAPE = (24, 32)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--hardware", action="store_true", default=False,
                     help="run tests marked @pytest.mark.hardware (Raspberry Pi with sensors only)")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "hardware: needs the Raspberry Pi and real sensors")


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    if config.getoption("--hardware"):
        return
    skip_hw = pytest.mark.skip(reason="hardware test: run on the Pi with --hardware")
    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(skip_hw)


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def default_config() -> Dict[str, Any]:
    # Plain yaml.safe_load for now. T1.1 adds processing/config.load_config, which
    # should replace this so every test goes through the real loader.
    with open(REPO_ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def synthetic_thermal_frame() -> np.ndarray:
    """24 x 32 float32 frame, deg C: background 22.0 C, leaf block 25.0 C, small noise.

    Fixed seed so tests are repeatable. Values are test inputs only, not physics.
    """
    rng = np.random.default_rng(seed=0)
    frame = np.full(THERMAL_SHAPE, 22.0, dtype=np.float32)
    frame[8:16, 10:22] = 25.0
    frame += rng.normal(0.0, 0.05, THERMAL_SHAPE).astype(np.float32)
    return frame


@pytest.fixture
def fake_session_dir(tmp_path: Path) -> Path:
    """Empty session folder laid out like data/raw/<session_id>/ (Section 5)."""
    session = tmp_path / "data" / "raw" / "test_session_001"
    (session / "images").mkdir(parents=True)
    return session
