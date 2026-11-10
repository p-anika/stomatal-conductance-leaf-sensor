"""Tests for processing/thermal_roi.py (T3.2).

Done-when (spec): synthetic frames with a known warm patch return the right mean, and small
patches are rejected. Owner decisions 2026-11-09: MAD trim k and min_coverage from config;
field of view from config (55 x 35 deg) with horizontal and vertical footprints.
"""
import math

import numpy as np
import pytest

from processing import thermal_roi as tr

ROWS, COLS = 24, 32


# ---------- frame_from_list ----------

def test_frame_from_list_row_major():
    f = tr.frame_from_list(list(range(768)))
    assert f.shape == (ROWS, COLS)
    assert f[0, 0] == 0 and f[0, 31] == 31 and f[1, 0] == 32 and f[23, 31] == 767


@pytest.mark.parametrize("n", [0, 767, 769])
def test_frame_from_list_wrong_length(n):
    with pytest.raises(ValueError, match="768"):
        tr.frame_from_list([0.0] * n)


# ---------- masks ----------

def test_rect_mask_half_open():
    m = tr.rect_mask(8, 16, 10, 22)
    assert m.shape == (ROWS, COLS) and m.dtype == bool
    assert m.sum() == 8 * 12
    assert m[8, 10] and m[15, 21] and not m[16, 10] and not m[8, 22]


@pytest.mark.parametrize("args", [(5, 5, 0, 3), (-1, 3, 0, 3), (0, 25, 0, 3), (0, 3, 4, 2),
                                  (0, 3, 0, 33)])
def test_rect_mask_bad_bounds(args):
    with pytest.raises(ValueError):
        tr.rect_mask(*args)


def test_erode_mask_removes_edge_ring():
    m = tr.rect_mask(8, 16, 10, 22)
    assert tr.erode_mask(m, 0).sum() == 96
    assert tr.erode_mask(m, 1).sum() == 6 * 10
    assert tr.erode_mask(m, 2).sum() == 4 * 8
    assert tr.erode_mask(m, 4).sum() == 0


def test_erode_mask_frame_border_counts_as_outside():
    full = np.ones((ROWS, COLS), dtype=bool)
    assert tr.erode_mask(full, 1).sum() == (ROWS - 2) * (COLS - 2)


def test_erode_mask_negative():
    with pytest.raises(ValueError):
        tr.erode_mask(tr.rect_mask(0, 3, 0, 3), -1)


# ---------- roi_stats ----------

def test_roi_stats_known_warm_patch(synthetic_thermal_frame, default_config):
    m = tr.erode_mask(tr.rect_mask(8, 16, 10, 22), 1)
    s = tr.roi_stats(synthetic_thermal_frame, m, default_config)
    assert s.valid and s.reason == "ok"
    assert s.mean_c == pytest.approx(25.0, abs=0.03)
    assert s.median_c == pytest.approx(25.0, abs=0.03)
    assert s.std_c == pytest.approx(0.05, abs=0.02)
    assert s.n_px + s.n_trimmed == 60


def test_roi_stats_hand_values(default_config):
    frame = np.zeros((ROWS, COLS))
    m = tr.rect_mask(0, 2, 0, 4)                    # 8 pixels
    frame[0, :4] = [20.0, 21.0, 22.0, 23.0]
    frame[1, :4] = [20.0, 21.0, 22.0, 23.0]
    s = tr.roi_stats(frame, m, default_config)
    # median 21.5; |dev| = 1.5,0.5,0.5,1.5 (x2) -> MAD 1.0; limit 3 x 1.4826 x 1.0: nothing trimmed
    assert s.n_trimmed == 0 and s.n_px == 8
    assert s.mean_c == pytest.approx(21.5) and s.median_c == pytest.approx(21.5)
    assert s.std_c == pytest.approx(np.std([20, 21, 22, 23] * 2, ddof=1))


def test_roi_stats_trims_mixed_background_pixel(default_config):
    frame = np.full((ROWS, COLS), 22.0)
    rng = np.random.default_rng(1)
    frame[4:8, 4:8] = 25.0 + rng.normal(0, 0.1, (4, 4))
    frame[4, 4] = 22.5                               # a pixel half on the background
    s = tr.roi_stats(frame, tr.rect_mask(4, 8, 4, 8), default_config)
    assert s.n_trimmed == 1 and s.n_px == 15
    assert s.mean_c == pytest.approx(25.0, abs=0.08)


def test_roi_stats_no_trim_when_mad_zero(default_config):
    frame = np.full((ROWS, COLS), 25.0)
    frame[0, 0] = 24.0
    s = tr.roi_stats(frame, tr.rect_mask(0, 3, 0, 3), default_config)
    assert s.n_trimmed == 0 and s.n_px == 9


def test_roi_stats_trim_k_from_config(default_config):
    c = {**default_config, "acquisition": {**default_config["acquisition"], "roi_mad_k": 100.0}}
    frame = np.full((ROWS, COLS), 22.0)
    frame[4:8, 4:8] = 25.0 + np.random.default_rng(1).normal(0, 0.1, (4, 4))
    frame[4, 4] = 22.5
    assert tr.roi_stats(frame, tr.rect_mask(4, 8, 4, 8), c).n_trimmed == 0


def test_small_patch_rejected(synthetic_thermal_frame, default_config):
    s = tr.roi_stats(synthetic_thermal_frame, tr.rect_mask(8, 10, 10, 12), default_config)  # 4 px
    assert not s.valid and s.reason == "roi_too_small"
    assert s.n_px == 4 and s.mean_c is None and s.median_c is None and s.std_c is None


def test_min_pixels_default_from_config_and_override(synthetic_thermal_frame, default_config):
    m = tr.rect_mask(8, 10, 10, 13)                   # 6 px
    assert default_config["acquisition"]["min_roi_pixels"] == 6
    assert tr.roi_stats(synthetic_thermal_frame, m, default_config).valid
    assert not tr.roi_stats(synthetic_thermal_frame, m, default_config, min_pixels=7).valid


def test_too_small_after_trimming(default_config):
    frame = np.full((ROWS, COLS), 25.0)
    frame[0, :6] = [25.0, 25.01, 24.99, 25.0, 25.02, 30.0]   # one hot outlier
    s = tr.roi_stats(frame, tr.rect_mask(0, 1, 0, 6), default_config)
    assert s.n_trimmed == 1 and s.n_px == 5 and s.reason == "roi_too_small"


def test_non_finite_pixels_ignored(default_config):
    frame = np.full((ROWS, COLS), 25.0)
    frame[0, 0] = np.nan
    s = tr.roi_stats(frame, tr.rect_mask(0, 3, 0, 3), default_config)
    assert s.valid and s.n_px == 8 and s.n_nonfinite == 1


def test_roi_stats_shape_mismatch(default_config):
    with pytest.raises(ValueError):
        tr.roi_stats(np.zeros((24, 32)), np.ones((10, 10), bool), default_config)


# ---------- pixel footprint ----------

def test_pixel_footprint_hand_values(default_config):
    """2 d tan(fov/2) / n:  h = 2 x 200 x tan(27.5 deg) / 32, v = 2 x 200 x tan(17.5 deg) / 24."""
    fp = tr.pixel_footprint_mm(200.0, default_config)
    assert fp.horizontal_mm == pytest.approx(400 * math.tan(math.radians(27.5)) / 32)
    assert fp.vertical_mm == pytest.approx(400 * math.tan(math.radians(17.5)) / 24)
    assert fp.horizontal_mm == pytest.approx(6.508, abs=1e-3)
    assert fp.vertical_mm == pytest.approx(5.255, abs=1e-3)
    assert fp.largest_mm == fp.horizontal_mm


def test_fov_in_config(default_config):
    s = default_config["sensors"]["mlx90640"]
    assert s["hfov_deg"] == 55 and s["vfov_deg"] == 35


@pytest.mark.parametrize("d", [0.0, -1.0, float("nan")])
def test_pixel_footprint_bad_distance(default_config, d):
    with pytest.raises(ValueError):
        tr.pixel_footprint_mm(d, default_config)


# ---------- warp_mask_to_thermal (BLOCKED-CAM to validate on real images) ----------

# RGB image 10x the thermal resolution. RGB pixel centre (x, y) maps to thermal
# ((x + 0.5)/10 - 0.5, (y + 0.5)/10 - 0.5), so RGB block [10c, 10c+10) is thermal pixel c.
SCALE = np.array([[0.1, 0.0, -0.45], [0.0, 0.1, -0.45], [0.0, 0.0, 1.0]])


def rgb_mask(r0, r1, c0, c1):
    m = np.zeros((240, 320), dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


def test_warp_aligned_block(default_config):
    out = tr.warp_mask_to_thermal(rgb_mask(50, 150, 100, 200), SCALE, (ROWS, COLS), default_config)
    assert out.dtype == bool and out.shape == (ROWS, COLS)
    assert np.array_equal(out, tr.rect_mask(5, 15, 10, 20))


def test_warp_partial_pixels_dropped_at_full_coverage(default_config):
    # Shift by 5 RGB px: the edge thermal pixels are only half covered.
    out = tr.warp_mask_to_thermal(rgb_mask(55, 155, 105, 205), SCALE, (ROWS, COLS), default_config)
    assert np.array_equal(out, tr.rect_mask(6, 15, 11, 20))


def test_warp_coverage_from_config(default_config):
    c = {**default_config, "acquisition": {**default_config["acquisition"], "min_pixel_coverage": 0.5}}
    out = tr.warp_mask_to_thermal(rgb_mask(55, 155, 105, 205), SCALE, (ROWS, COLS), c)
    # Edge pixels are half covered (8 of 16 sample points) and pass at 0.5; the four
    # corner pixels are a quarter covered (4 of 16) and do not.
    expected = tr.rect_mask(5, 16, 10, 21)
    for r, col in [(5, 10), (5, 20), (15, 10), (15, 20)]:
        expected[r, col] = False
    assert np.array_equal(out, expected)


def test_warp_outside_image_is_not_leaf(default_config):
    shift = SCALE.copy()
    shift[0, 2] += 30.0          # RGB image lands 30 thermal columns to the right
    out = tr.warp_mask_to_thermal(np.ones((240, 320), bool), shift, (ROWS, COLS), default_config)
    assert out[:, :30].sum() == 0 and out[:, 30:].all()


def test_warp_bad_matrix(default_config):
    with pytest.raises(ValueError):
        tr.warp_mask_to_thermal(rgb_mask(0, 10, 0, 10), np.zeros((3, 3)), (ROWS, COLS), default_config)
    with pytest.raises(ValueError):
        tr.warp_mask_to_thermal(rgb_mask(0, 10, 0, 10), np.eye(2), (ROWS, COLS), default_config)
