"""Tests for processing/jones_index.py (T1.6, Appendix A eqs. 12-13).

Hand-computed values. Out-of-range values must be kept as measured (never clipped).
"""
import math
from pathlib import Path

import pytest

from processing.config import load_config
from processing.jones_index import conductance_index

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def cfg():
    return load_config(REPO_ROOT / "config.yaml")


def test_hand_values(cfg):
    # dry 30, leaf 26, wet 22: I_g = (30-26)/(26-22) = 1.0; CWSI = (26-22)/(30-22) = 0.5
    r = conductance_index(30.0, 26.0, 22.0, cfg)
    assert r.i_g.value == pytest.approx(1.0) and r.cwsi.value == pytest.approx(0.5)
    for part in (r.i_g, r.cwsi):
        assert part.valid and part.reason == "ok" and part.censor == "none"


def test_hand_values_2(cfg):
    # dry 31.5, leaf 24.0, wet 21.0: I_g = 7.5/3.0 = 2.5; CWSI = 3.0/10.5 = 0.285714
    r = conductance_index(31.5, 24.0, 21.0, cfg)
    assert r.i_g.value == pytest.approx(2.5)
    assert r.cwsi.value == pytest.approx(3.0 / 10.5)


@pytest.mark.parametrize("t_leaf", [22.5, 24.0, 27.0, 29.5])
def test_identity(cfg, t_leaf):
    # I_g = 1 / CWSI - 1 (algebra from eqs. 12 and 13)
    r = conductance_index(30.0, t_leaf, 22.0, cfg)
    assert r.i_g.value == pytest.approx(1.0 / r.cwsi.value - 1.0)


def test_leaf_hotter_than_dry_kept_not_clipped(cfg):
    # leaf 30.4 > dry 30: I_g = -0.4/8.4 = -0.047619; CWSI = 8.4/8 = 1.05
    r = conductance_index(30.0, 30.4, 22.0, cfg)
    assert r.i_g.value == pytest.approx(-0.4 / 8.4)
    assert r.cwsi.value == pytest.approx(1.05)
    assert r.i_g.censor == "none" and r.cwsi.censor == "none"
    assert r.i_g.valid and r.cwsi.valid


def test_leaf_colder_than_wet_cwsi_negative(cfg):
    # leaf 21.9 < wet 22: CWSI = -0.1/8 = -0.0125 (kept); I_g above_range (leaf - wet < 0.2)
    r = conductance_index(30.0, 21.9, 22.0, cfg)
    assert r.cwsi.value == pytest.approx(-0.0125) and r.cwsi.valid
    assert r.i_g.censor == "above_range" and r.i_g.value is None and r.i_g.valid


# 22.2 - 22.0 is not exactly 0.2 in floating point, so the boundary uses 22.25.
@pytest.mark.parametrize("t_leaf, censor", [(22.19, "above_range"), (22.25, "none"),
                                            (22.0, "above_range")])
def test_ig_above_range_threshold(cfg, t_leaf, censor):
    r = conductance_index(30.0, t_leaf, 22.0, cfg)
    assert r.i_g.censor == censor and r.i_g.valid
    if censor == "none":
        assert r.i_g.value == pytest.approx(7.75 / 0.25)


def test_threshold_from_config(cfg):
    cfg["jones"]["min_leaf_wet_diff_c"] = 1.0
    assert conductance_index(30.0, 22.5, 22.0, cfg).i_g.censor == "above_range"


@pytest.mark.parametrize("t_dry", [22.99, 22.0, 21.0])
def test_references_not_bracketing(cfg, t_dry):
    r = conductance_index(t_dry, 22.5, 22.0, cfg)
    for part in (r.i_g, r.cwsi):
        assert part.valid is False and part.reason == "references_not_bracketing"
        assert part.value is None and part.censor == "none"


def test_bracketing_threshold_inclusive(cfg):
    r = conductance_index(23.0, 22.5, 22.0, cfg)   # dry - wet = 1.0 exactly -> allowed
    assert r.cwsi.valid and r.cwsi.value == pytest.approx(0.5)


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
@pytest.mark.parametrize("pos", [0, 1, 2])
def test_missing_input(cfg, bad, pos):
    args = [30.0, 26.0, 22.0]
    args[pos] = bad
    r = conductance_index(*args, cfg)
    for part in (r.i_g, r.cwsi):
        assert part.valid is False and part.reason == "missing_input" and part.value is None


def test_values_are_finite_floats(cfg):
    r = conductance_index(30.0, 26.0, 22.0, cfg)
    assert math.isfinite(r.i_g.value) and math.isfinite(r.cwsi.value)
