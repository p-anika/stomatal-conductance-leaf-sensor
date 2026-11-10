"""Tests for processing/measurement_windows.py (T3.4, B15).

Done-when (spec): synthetic drifting and flat series are classified correctly, and windows
with photos at known times receive the right features. Owner decisions 2026-11-09:
acclimation = logged minutes + time since session start, reset by a lamp_on event; one reason
per window (missing_input > not_steady_state > no_photo_in_window) with every other problem
in flags; leaf_moved = (max - min) / median of d_min > windows.leaf_moved_range_frac.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from processing.measurement_windows import attach_photo_features, build_windows, row_times

T0 = pd.Timestamp("2026-11-09T10:00:00+00:00")
DT_S = 5
N_ROWS = 120          # 10 minutes at 5 s


def make_log(leaf_slope_c_per_min: float = 0.0, noise_c: float = 0.02, seed: int = 0) -> pd.DataFrame:
    t = T0 + pd.to_timedelta(np.arange(N_ROWS) * DT_S, unit="s")
    minutes = np.arange(N_ROWS) * DT_S / 60.0
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "session_id": "S1",
        "timestamp_iso": [x.isoformat() for x in t],
        "timestamp_corrected_iso": "",
        "monotonic_s": np.arange(N_ROWS) * float(DT_S),
        "air_temp_c": 22.0,
        "humidity_pct": 50.0,
        "pressure_hpa": 1013.0,
        "lux": 1000.0,
        "leaf_temp_c": 25.0 + leaf_slope_c_per_min * minutes + rng.normal(0, noise_c, N_ROWS),
        "image_file": "",
        "qc_flags": "",
    })


def no_events() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp_iso", "label", "note"])


def iso(minutes: float) -> str:
    return (T0 + pd.Timedelta(minutes=minutes)).isoformat()


def photos(rows: list[tuple[float, float, bool]]) -> pd.DataFrame:
    """rows: (minutes after T0, d_min_mm, ok)."""
    return pd.DataFrame({
        "timestamp": [iso(m) for m, _, _ in rows],
        "image_file": [f"img_{i}.jpg" for i in range(len(rows))],
        "d_min_mm": [d for _, d, _ in rows],
        "area_mm2": [100.0 * d for _, d, _ in rows],
        "ok": [ok for _, _, ok in rows],
        "reason": ["ok" if ok else "segmentation_failed" for _, _, ok in rows],
    })


@pytest.fixture
def cfg(default_config: dict[str, Any]) -> dict[str, Any]:
    return default_config


def flags_of(row: pd.Series) -> set[str]:
    return set(filter(None, str(row["flags"]).split(";")))


# ---------- windowing and averaging ----------

def test_ten_windows_of_twelve_samples(cfg):
    w = build_windows(make_log(), no_events(), {"acclimation_minutes": 40}, cfg)
    assert len(w) == 10
    assert (w["n_samples"] == 12).all()
    assert w["window_start_iso"].iloc[3] == iso(3)
    assert w["window_end_iso"].iloc[3] == iso(4)
    assert w["air_temp_c"].iloc[0] == pytest.approx(22.0)
    assert w["air_temp_c_sd"].iloc[0] == pytest.approx(0.0)
    assert "monotonic_s" not in w.columns            # counters are not averaged


def test_qc_flagged_rows_excluded(cfg):
    log = make_log()
    log.loc[30, "air_temp_c"] = 60.0                 # out of range -> flagged by qc.add_qc_flags
    log.loc[[50, 51, 52], "qc_flags"] = "spike_lux"  # flags already in the file are respected
    w = build_windows(log, no_events(), {"acclimation_minutes": 40}, cfg)
    assert w["n_samples"].iloc[2] == 11 and w["n_excluded"].iloc[2] == 1
    assert w["air_temp_c"].iloc[2] == pytest.approx(22.0)
    assert w["n_samples"].iloc[4] == 9
    assert w["reason"].iloc[4] == "missing_input"    # fewer than min_samples (10) left
    assert not w["valid"].iloc[4]


def test_corrected_timestamp_preferred():
    log = make_log().head(3)
    log.loc[1, "timestamp_corrected_iso"] = "2026-11-09T11:00:05+00:00"
    t = row_times(log)
    assert t.iloc[0] == T0 and t.iloc[1] == pd.Timestamp("2026-11-09T11:00:05+00:00")


# ---------- steady state (done-when) ----------

def test_flat_series_is_steady(cfg):
    w = build_windows(make_log(0.0), no_events(), {"acclimation_minutes": 40}, cfg)
    # windows 0 and 1 end before 3 minutes of data exist
    assert not w["steady_state"].iloc[0] and "stability_window_incomplete" in flags_of(w.iloc[0])
    assert w["steady_state"].iloc[2:].all()
    assert (w["leaf_slope_c_per_min"].iloc[2:].abs() < 0.05).all()


def test_drifting_series_not_steady(cfg):
    w = build_windows(make_log(0.2), no_events(), {"acclimation_minutes": 40}, cfg)
    assert not w["steady_state"].any()
    assert (w["reason"] == "not_steady_state").all()
    assert w["leaf_slope_c_per_min"].iloc[5] == pytest.approx(0.2, abs=0.01)
    assert "leaf_temp_drifting" in flags_of(w.iloc[5])


def test_slow_drift_below_threshold_is_steady(cfg):
    w = build_windows(make_log(0.02), no_events(), {"acclimation_minutes": 40}, cfg)
    assert w["steady_state"].iloc[2:].all()


def test_acclimation_grows_with_elapsed_time(cfg):
    w = build_windows(make_log(), no_events(), {"acclimation_minutes": 25}, cfg)
    assert w["acclimation_minutes"].iloc[2] == pytest.approx(28.0)   # 25 + 3 min at window end
    assert not w["steady_state"].iloc[3]                             # 29 min at window end
    assert "acclimation_too_short" in flags_of(w.iloc[3])
    assert w["steady_state"].iloc[4]                                 # 30 min


def test_lamp_on_event_resets_acclimation(cfg):
    ev = pd.DataFrame({"timestamp_iso": [iso(2)], "label": ["lamp_on"], "note": [""]})
    w = build_windows(make_log(), ev, {"acclimation_minutes": 40}, cfg)
    assert w["acclimation_minutes"].iloc[0] == pytest.approx(41.0)   # window ends before the event
    assert w["acclimation_minutes"].iloc[5] == pytest.approx(4.0)    # 6 - 2 min
    assert not w["steady_state"].iloc[2:].any()


def test_meta_can_be_object(cfg):
    class Meta:
        acclimation_minutes = 40.0
    w = build_windows(make_log(), no_events(), Meta(), cfg)
    assert w["steady_state"].iloc[2:].all()


# ---------- photo pairing (done-when, B15) ----------

PHOTO_ROWS = [(0.33, 99.0, False),   # failed segmentation: ignored
              (3.17, 30.0, True), (3.67, 31.0, True),   # both inside window 3
              (8.0, 30.5, True)]


def test_photos_attached_by_time(cfg):
    w = build_windows(make_log(), no_events(), {"acclimation_minutes": 40}, cfg,
                      photo_features=photos(PHOTO_ROWS))
    # window 3: median of the two photos inside
    assert w["d_min_mm"].iloc[3] == pytest.approx(30.5)
    assert w["n_photos"].iloc[3] == 2
    assert w["photo_gap_sec"].iloc[3] == pytest.approx(10.0, abs=0.5)   # 3:40 vs centre 3:30
    # window 5 (centre 5:30): nearest is 3:40 (110 s), not 8:00 (150 s)
    assert w["d_min_mm"].iloc[5] == pytest.approx(31.0)
    assert w["photo_gap_sec"].iloc[5] == pytest.approx(110.0, abs=0.5)
    # window 6 (centre 6:30): 8:00 is 90 s away
    assert w["d_min_mm"].iloc[6] == pytest.approx(30.5)
    # window 0 (centre 0:30): nearest ok photo is 160 s away -> none
    assert math.isnan(w["d_min_mm"].iloc[0])
    assert w["n_photos"].iloc[0] == 0


def test_no_photo_reason_and_flags(cfg):
    w = build_windows(make_log(), no_events(), {"acclimation_minutes": 40}, cfg,
                      photo_features=photos([(3.17, 30.0, True)]))
    # window 8 (centre 8:30) is steady but has no photo within 120 s
    assert w["reason"].iloc[8] == "no_photo_in_window" and not w["valid"].iloc[8]
    # window 0 is not steady AND has no photo: one reason, the other problem in flags
    assert w["reason"].iloc[0] == "not_steady_state"
    assert "no_photo_in_window" in flags_of(w.iloc[0])
    assert w["valid"].iloc[3] and w["reason"].iloc[3] == "ok"


def test_join_ignores_row_order(cfg):
    log = make_log()
    a = build_windows(log, no_events(), {"acclimation_minutes": 40}, cfg,
                      photo_features=photos(PHOTO_ROWS))
    shuffled = photos(PHOTO_ROWS).sample(frac=1.0, random_state=3).reset_index(drop=True)
    b = build_windows(log, no_events(), {"acclimation_minutes": 40}, cfg, photo_features=shuffled)
    pd.testing.assert_frame_equal(a, b)


def test_leaf_moved_flag(cfg):
    still = build_windows(make_log(), no_events(), {"acclimation_minutes": 40}, cfg,
                          photo_features=photos(PHOTO_ROWS))        # range 1/30.5 = 3.3 %
    assert not any("leaf_moved" in flags_of(r) for _, r in still.iterrows())
    moved_rows = [(3.17, 30.0, True), (5.0, 33.0, True)]             # range 3/31.5 = 9.5 %
    moved = build_windows(make_log(), no_events(), {"acclimation_minutes": 40}, cfg,
                          photo_features=photos(moved_rows))
    assert all("leaf_moved" in flags_of(r) for _, r in moved.iterrows())
    assert moved["valid"].iloc[3]                                    # a flag, not a failure


def test_attach_directly(cfg):
    w = build_windows(make_log(), no_events(), {"acclimation_minutes": 40}, cfg)
    out = attach_photo_features(w, photos(PHOTO_ROWS), max_gap_sec=30.0,
                                leaf_moved_range_frac=0.05)
    assert math.isnan(out["d_min_mm"].iloc[5])                       # 110 s > 30 s
    assert out["d_min_mm"].iloc[3] == pytest.approx(30.5)
    assert "d_min_mm" not in w.columns                               # input not modified


def test_missing_timestamp_column_raises(cfg):
    log = make_log().drop(columns=["timestamp_iso", "timestamp_corrected_iso"])
    with pytest.raises(KeyError, match="timestamp"):
        build_windows(log, no_events(), {"acclimation_minutes": 40}, cfg)
