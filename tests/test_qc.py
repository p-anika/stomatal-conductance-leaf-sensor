"""Tests for processing/qc.py (T3.3).

Done-when (spec): the first debug log is flagged (lux of about 2 amid readings near 88; the
thermal frame near 37 C). Rows are never deleted or changed. Owner decisions 2026-11-09:
minimum spike sizes (config placeholders); no spike checks on leaf, card or wet-reference
temperatures; humidity row 8 of the debug log is flagged; overflow at 65535 for now.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from processing import qc

REPO_ROOT = Path(__file__).resolve().parent.parent
DEBUG_LOG = REPO_ROOT / "data" / "raw" / "debug_logs" / "first_sensor_log_(old_combined_logger).csv"


def flags_of(out):
    return [set(f.split(";")) - {""} for f in out["qc_flags"]]


def frame(**cols):
    n = len(next(iter(cols.values())))
    base = {"timestamp_iso": [f"2026-11-09T10:00:{i:02d}-05:00" for i in range(n)]}
    base.update(cols)
    return pd.DataFrame(base)


# ---------- the first debug log (spec done-when) ----------

def test_debug_log_flags(default_config):
    before = DEBUG_LOG.read_bytes()
    df = pd.read_csv(DEBUG_LOG)
    out = qc.add_qc_flags(df, default_config)
    assert DEBUG_LOG.read_bytes() == before                     # raw file untouched
    assert len(out) == len(df) == 10                            # no row deleted
    pd.testing.assert_frame_equal(out.drop(columns="qc_flags"), df)   # no value changed
    f = flags_of(out)
    # Data rows are 0-based here; file line = index + 2.
    assert f[7] == {"spike_lux"} and f[8] == {"spike_lux"}                 # lux 1.98, 1.98
    assert f[4] == {"spike_thermal_min_c", "spike_thermal_max_c", "spike_thermal_mean_c"}  # ~37 C frame
    assert f[6] == {"spike_humidity_pct"}                                  # 68.6 % RH (owner: flag)
    assert all(f[i] == set() for i in (0, 1, 2, 3, 5, 9))


def test_input_frame_not_modified(default_config):
    df = pd.read_csv(DEBUG_LOG)
    copy = df.copy()
    qc.add_qc_flags(df, default_config)
    pd.testing.assert_frame_equal(df, copy)


# ---------- range checks ----------

@pytest.mark.parametrize("col, bad, good", [
    ("air_temp_c", [-10.5, 50.5], [-10.0, 50.0]),
    ("humidity_pct", [-0.1, 100.1], [0.0, 100.0]),
    ("pressure_hpa", [899.9, 1100.1], [900.0, 1100.0]),
    ("lux", [-0.01, float("inf")], [0.0, 1e6]),
    ("thermal_mean_c", [-10.1, 80.1], [-10.0, 80.0]),
    ("leaf_temp_c", [-11.0, 81.0], [-10.0, 80.0]),
    ("card_temp_c", [-11.0, 81.0], [-10.0, 80.0]),
])
def test_range_limits_inclusive(default_config, col, bad, good):
    out = qc.add_qc_flags(frame(**{col: bad + good}), default_config)
    f = flags_of(out)
    assert all(f"range_{col}" in f[i] for i in (0, 1))
    assert all(f"range_{col}" not in f[i] for i in (2, 3))


def test_nan_is_not_a_range_failure(default_config):
    out = qc.add_qc_flags(frame(air_temp_c=[20.0, np.nan, 20.0]), default_config)
    assert flags_of(out) == [set(), set(), set()]


def test_missing_columns_are_skipped(default_config):
    out = qc.add_qc_flags(frame(lux=[1.0, 2.0]), default_config)
    assert list(out.columns) == ["timestamp_iso", "lux", "qc_flags"]


# ---------- spike detection (Hampel, window 7, 3 sigma, minimum size) ----------

def test_mad_to_sigma_is_derived(default_config):
    """1 / Phi^-1(3/4): for normal data the median absolute deviation equals 0.6745 sigma."""
    assert default_config["qc"]["mad_to_sigma"] == pytest.approx(1.0 / norm.ppf(0.75), rel=1e-4)


def test_hampel_flags_single_spike():
    x = np.array([10.0, 10.1, 9.9, 10.0, 15.0, 10.1, 9.9, 10.0, 10.1])
    assert qc.hampel_flags(x, 7, 3.0, 1.4826).tolist() == [i == 4 for i in range(9)]


def test_hampel_window_shrinks_at_ends():
    x = np.array([15.0, 10.0, 10.1, 9.9, 10.0, 10.1])     # spike in the first row
    assert qc.hampel_flags(x, 7, 3.0, 1.4826)[0]
    x2 = np.array([10.0, 10.1, 9.9, 10.0, 10.1, 15.0])    # and in the last row
    assert qc.hampel_flags(x2, 7, 3.0, 1.4826)[-1]


def test_hampel_two_adjacent_spikes():
    x = np.array([88.0, 88.5, 88.2, 90.0, 2.0, 2.0, 89.5, 88.8, 88.1])
    assert np.flatnonzero(qc.hampel_flags(x, 7, 3.0, 1.4826)).tolist() == [4, 5]


def test_hampel_step_is_not_a_spike():
    # A real step (lamp switched on) must not be flagged: the median follows the new level.
    x = np.array([22.0] * 6 + [25.0] * 6) + np.tile([0.0, 0.02, -0.02], 4)
    assert not qc.hampel_flags(x, 7, 3.0, 1.4826, min_abs=0.5).any()


def test_minimum_spike_size_stops_quantisation_flags():
    # 0.01 hPa steps: MAD is tiny, so the plain 3-sigma rule flags them.
    x = np.array([994.81, 994.78, 994.83, 994.83, 994.78, 994.79, 994.83, 994.77, 994.77, 994.77])
    assert qc.hampel_flags(x, 7, 3.0, 1.4826).any()
    assert not qc.hampel_flags(x, 7, 3.0, 1.4826, min_abs=1.0).any()


def test_minimum_relative_size():
    x = np.array([88.6, 88.4, 88.5, 90.5, 88.5, 88.6, 88.4])
    assert qc.hampel_flags(x, 7, 3.0, 1.4826)[3]
    assert not qc.hampel_flags(x, 7, 3.0, 1.4826, min_rel=0.2)[3]


def test_hampel_constant_series_and_nan():
    x = np.array([5.0, 5.0, np.nan, 5.0, 5.0, 5.0])
    assert not qc.hampel_flags(x, 7, 3.0, 1.4826).any()


def test_hampel_too_few_points():
    assert not qc.hampel_flags(np.array([1.0, 100.0, 1.0]), 7, 3.0, 1.4826).any()


def test_hampel_bad_window():
    with pytest.raises(ValueError):
        qc.hampel_flags(np.zeros(5), 6, 3.0, 1.4826)


def test_no_spike_checks_on_roi_temperatures(default_config):
    """Owner decision B: flagging on the measured leaf, card or wet-reference temperature
    would select data on the result. Range checks still apply to them."""
    spiky = [25.0, 25.1, 24.9, 25.0, 35.0, 25.1, 24.9, 25.0]
    for col in ("leaf_temp_c", "card_temp_c", "wet_ref_temp_c"):
        assert col not in default_config["qc"]["spike_columns"]
        out = qc.add_qc_flags(frame(**{col: spiky}), default_config)
        assert flags_of(out)[4] == set()


def test_spike_thresholds_from_config(default_config):
    c = {**default_config, "qc": {**default_config["qc"], "spike_min_abs": {
        **default_config["qc"]["spike_min_abs"], "air_temp_c": 100.0}}}
    x = [22.0, 22.1, 21.9, 22.0, 30.0, 22.1, 21.9]
    assert "spike_air_temp_c" in flags_of(qc.add_qc_flags(frame(air_temp_c=x), default_config))[4]
    assert flags_of(qc.add_qc_flags(frame(air_temp_c=x), c))[4] == set()


# ---------- overflow and duplicate timestamps ----------

def test_overflow_at_full_count(default_config):
    out = qc.add_qc_flags(frame(tsl_full_spectrum=[100, 65535, 65534],
                                tsl_infrared=[10, 10, 65535]), default_config)
    f = flags_of(out)
    assert f[0] == set() and "sensor_overflow" in f[1] and "sensor_overflow" in f[2]


def test_infinite_lux_is_overflow(default_config):
    f = flags_of(qc.add_qc_flags(frame(lux=[88.0, float("inf"), 88.0]), default_config))
    assert "sensor_overflow" in f[1] and "range_lux" in f[1] and f[0] == set()


def test_duplicate_timestamps_flag_every_copy(default_config):
    df = pd.DataFrame({"timestamp_iso": ["t1", "t2", "t2", "t3", "t1"], "lux": [1.0] * 5})
    f = flags_of(qc.add_qc_flags(df, default_config))
    assert ["duplicate_timestamp" in x for x in f] == [True, True, True, False, True]


def test_old_log_timestamp_column(default_config):
    df = pd.DataFrame({"timestamp": ["a", "a", "b"], "lux": [1.0, 1.0, 1.0]})
    f = flags_of(qc.add_qc_flags(df, default_config))
    assert ["duplicate_timestamp" in x for x in f] == [True, True, False]


# ---------- flag column ----------

def test_existing_flags_are_kept(default_config):
    df = frame(lux=[-1.0, 1.0])
    df["qc_flags"] = ["logger_retry", ""]
    out = qc.add_qc_flags(df, default_config)
    assert out["qc_flags"].tolist() == ["logger_retry;range_lux", ""]


def test_existing_flags_nan_and_no_duplicates(default_config):
    df = frame(lux=[-1.0, 1.0])
    df["qc_flags"] = ["range_lux", np.nan]
    assert qc.add_qc_flags(df, default_config)["qc_flags"].tolist() == ["range_lux", ""]


def test_several_flags_on_one_row(default_config):
    df = pd.DataFrame({"timestamp_iso": ["t", "t"], "lux": [-1.0, 1.0], "air_temp_c": [60.0, 20.0]})
    f = flags_of(qc.add_qc_flags(df, default_config))
    assert f[0] == {"range_lux", "range_air_temp_c", "duplicate_timestamp"}


def test_empty_frame(default_config):
    out = qc.add_qc_flags(pd.DataFrame({"timestamp_iso": [], "lux": []}), default_config)
    assert len(out) == 0 and "qc_flags" in out.columns
