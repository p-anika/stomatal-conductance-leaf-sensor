"""Tests for processing/leaf_segmentation.py (T3.1).

Done-when (spec, synthetic part): drawn shapes of known size (ellipse, circle, lobed polygon)
return area and d_min within 2 %. The real-photo part (five leaf overlays that look right)
needs phone/Arducam photos and is checked by eye, not here.

Shapes are drawn analytically: a pixel is "inside" when its centre is inside the true shape,
so the true area and d_min are known exactly in mm.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
import pytest
from matplotlib.path import Path as MplPath

from processing.leaf_segmentation import (
    SegmentationResult,
    mm_per_pixel_from_fov,
    segment_and_measure,
)

H, W = 800, 1000
MM_PER_PX = 0.1
LEAF_BGR = (40, 140, 50)         # green leaf
BG_LEAF_BGR = (200, 200, 200)    # light-gray paper
CARD_BGR = (25, 25, 25)          # matte black card
BG_CARD_BGR = (235, 235, 235)    # white paper


@pytest.fixture
def seg_params(default_config: dict[str, Any]) -> dict[str, Any]:
    return default_config["segmentation"]


def _pixel_centres() -> tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))


def ellipse_mask(cx: float, cy: float, a_px: float, b_px: float, angle_deg: float = 0.0) -> np.ndarray:
    x, y = _pixel_centres()
    t = math.radians(angle_deg)
    u = (x - cx) * math.cos(t) + (y - cy) * math.sin(t)
    v = -(x - cx) * math.sin(t) + (y - cy) * math.cos(t)
    return (u / a_px) ** 2 + (v / b_px) ** 2 <= 1.0


def lobed_polygon(cx: float, cy: float, r0: float, amp: float, lobes: int, n: int = 360) -> np.ndarray:
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = r0 * (1 + amp * np.cos(lobes * th))
    return np.column_stack([cx + r * np.cos(th), cy + 0.7 * r * np.sin(th)])


def polygon_mask(verts: np.ndarray) -> np.ndarray:
    x, y = _pixel_centres()
    inside = MplPath(verts).contains_points(np.column_stack([x.ravel(), y.ravel()]))
    return inside.reshape(H, W)


def shoelace_area(verts: np.ndarray) -> float:
    x, y = verts[:, 0], verts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def paint(mask: np.ndarray, fg: tuple[int, int, int], bg: tuple[int, int, int],
          noise: float = 4.0, seed: int = 0) -> np.ndarray:
    img = np.empty((H, W, 3), dtype=float)
    img[:] = bg
    img[mask] = fg
    img += np.random.default_rng(seed).normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------- done-when: known shapes within 2 % ----------

@pytest.mark.parametrize("angle_deg", [0.0, 30.0])
def test_ellipse_area_and_dmin_within_2pct(seg_params, angle_deg):
    a, b = 300.0, 150.0
    img = paint(ellipse_mask(500, 400, a, b, angle_deg), LEAF_BGR, BG_LEAF_BGR)
    r = segment_and_measure(img, MM_PER_PX, "leaf", params=seg_params)
    assert r.valid and r.reason == "ok"
    assert r.area_mm2 == pytest.approx(math.pi * a * b * MM_PER_PX**2, rel=0.02)
    assert r.d_min_mm == pytest.approx(2 * b * MM_PER_PX, rel=0.02)
    assert r.d_max_mm == pytest.approx(2 * a * MM_PER_PX, rel=0.02)
    assert r.eccentricity == pytest.approx(math.sqrt(1 - (b / a) ** 2), abs=0.02)
    assert r.aspect_ratio == pytest.approx(a / b, rel=0.03)
    assert r.solidity == pytest.approx(1.0, abs=0.02)
    assert r.n_components == 1


def test_circle(seg_params):
    rad = 200.0
    img = paint(ellipse_mask(500, 400, rad, rad), LEAF_BGR, BG_LEAF_BGR)
    r = segment_and_measure(img, MM_PER_PX, "leaf", params=seg_params)
    assert r.valid
    assert r.area_mm2 == pytest.approx(math.pi * rad**2 * MM_PER_PX**2, rel=0.02)
    assert r.d_min_mm == pytest.approx(2 * rad * MM_PER_PX, rel=0.02)
    assert r.d_eq_mm == pytest.approx(2 * rad * MM_PER_PX, rel=0.02)
    assert r.eccentricity < 0.1
    # Pixel staircase makes the perimeter a little long, so circularity sits just below 1
    # (documented bias, not corrected; owner decision 2026-11-09).
    assert 0.85 < r.circularity <= 1.02
    assert r.perimeter_mm == pytest.approx(2 * math.pi * rad * MM_PER_PX, rel=0.1)


def test_lobed_polygon(seg_params):
    verts = lobed_polygon(500, 400, 300, 0.25, 5)
    img = paint(polygon_mask(verts), LEAF_BGR, BG_LEAF_BGR)
    r = segment_and_measure(img, MM_PER_PX, "leaf", params=seg_params)
    assert r.valid
    assert r.area_mm2 == pytest.approx(shoelace_area(verts) * MM_PER_PX**2, rel=0.02)
    (_, _), (w, h), _ = cv2.minAreaRect(verts.astype(np.float32))
    assert r.d_min_mm == pytest.approx(min(w, h) * MM_PER_PX, rel=0.02)
    assert r.solidity < 0.97          # lobes make it non-convex


def test_card_mode_dark_object(seg_params):
    a, b = 250.0, 120.0
    img = paint(ellipse_mask(500, 400, a, b, 15), CARD_BGR, BG_CARD_BGR)
    r = segment_and_measure(img, MM_PER_PX, "card", params=seg_params)
    assert r.valid
    assert r.area_mm2 == pytest.approx(math.pi * a * b * MM_PER_PX**2, rel=0.02)
    assert r.d_min_mm == pytest.approx(2 * b * MM_PER_PX, rel=0.02)


def test_holes_filled_and_specks_ignored(seg_params):
    mask = ellipse_mask(500, 400, 300, 150)
    hole = ellipse_mask(500, 400, 20, 20)
    mask_with_hole = mask & ~hole
    speck = ellipse_mask(100, 100, 3, 3)
    other = ellipse_mask(880, 120, 40, 40)   # a second, smaller real object
    img = paint(mask_with_hole | speck | other, LEAF_BGR, BG_LEAF_BGR)
    r = segment_and_measure(img, MM_PER_PX, "leaf", params=seg_params)
    assert r.valid
    assert r.area_mm2 == pytest.approx(math.pi * 300 * 150 * MM_PER_PX**2, rel=0.02)
    assert r.n_components == 2               # speck removed by opening; small object counted
    assert r.mask[400, 500]                  # hole filled


def test_edge_density_higher_for_textured_leaf(seg_params):
    mask = ellipse_mask(500, 400, 300, 150)
    plain = paint(mask, LEAF_BGR, BG_LEAF_BGR)
    veined = plain.copy()
    for x in range(220, 800, 25):            # dark "veins" inside the leaf
        line = np.zeros((H, W), bool)
        line[:, x:x + 3] = True
        veined[line & mask] = (20, 90, 30)
    r_plain = segment_and_measure(plain, MM_PER_PX, "leaf", params=seg_params)
    r_vein = segment_and_measure(veined, MM_PER_PX, "leaf", params=seg_params)
    assert r_plain.valid and r_vein.valid
    assert r_plain.edge_density < 0.01       # the outline itself is not counted
    assert r_vein.edge_density > 5 * max(r_plain.edge_density, 1e-3)


def test_result_carries_mask_and_overlay(seg_params):
    img = paint(ellipse_mask(500, 400, 300, 150), LEAF_BGR, BG_LEAF_BGR)
    r = segment_and_measure(img, MM_PER_PX, "leaf", params=seg_params)
    assert isinstance(r, SegmentationResult)
    assert r.mask.shape == (H, W) and r.mask.dtype == bool
    assert r.overlay.shape == img.shape and r.overlay.dtype == np.uint8
    assert not np.array_equal(r.overlay, img)
    assert r.flags == ()


def test_params_none_loads_repo_config():
    img = paint(ellipse_mask(500, 400, 300, 150), LEAF_BGR, BG_LEAF_BGR)
    assert segment_and_measure(img, MM_PER_PX, "leaf").valid


# ---------- failures ----------

def test_blank_image_fails(seg_params):
    img = paint(np.zeros((H, W), bool), LEAF_BGR, BG_LEAF_BGR)
    r = segment_and_measure(img, MM_PER_PX, "leaf", params=seg_params)
    assert not r.valid and r.reason == "segmentation_failed"
    assert r.area_mm2 is None and r.d_min_mm is None


def test_touching_border_fails(seg_params):
    img = paint(ellipse_mask(150, 400, 300, 150), LEAF_BGR, BG_LEAF_BGR)
    r = segment_and_measure(img, MM_PER_PX, "leaf", params=seg_params)
    assert not r.valid and r.reason == "segmentation_failed"
    assert r.mask is not None and r.overlay is not None    # kept for human review


def test_implausible_area_fails(seg_params):
    img = paint(ellipse_mask(500, 400, 300, 150), LEAF_BGR, BG_LEAF_BGR)
    too_big = segment_and_measure(img, 10.0, "leaf", params=seg_params)
    too_small = segment_and_measure(img, 0.001, "leaf", params=seg_params)
    assert too_big.reason == "segmentation_failed"
    assert too_small.reason == "segmentation_failed"


# ---------- foreshortening (owner decision 2026-11-09: area and d_eq only) ----------

def test_foreshortening_corrects_area_and_deq_only(seg_params):
    img = paint(ellipse_mask(500, 400, 300, 150), LEAF_BGR, BG_LEAF_BGR)
    flat = segment_and_measure(img, MM_PER_PX, "leaf", params=seg_params)
    tilt = segment_and_measure(img, MM_PER_PX, "leaf", incidence_angle_deg=30.0, params=seg_params)
    cos30 = math.cos(math.radians(30))
    assert tilt.area_mm2 == pytest.approx(flat.area_mm2 / cos30, rel=1e-9)
    assert tilt.area_projected_mm2 == pytest.approx(flat.area_mm2, rel=1e-9)
    assert tilt.d_eq_mm == pytest.approx(math.sqrt(4 * tilt.area_mm2 / math.pi), rel=1e-9)
    assert tilt.d_min_mm == flat.d_min_mm and tilt.d_max_mm == flat.d_max_mm
    assert tilt.perimeter_mm == flat.perimeter_mm
    assert "foreshortening_corrected" in tilt.flags
    assert "foreshortening_corrected" not in flat.flags


@pytest.mark.parametrize("angle", [-5.0, 89.0, float("nan")])
def test_bad_incidence_angle_raises(seg_params, angle):
    img = paint(ellipse_mask(500, 400, 300, 150), LEAF_BGR, BG_LEAF_BGR)
    with pytest.raises(ValueError):
        segment_and_measure(img, MM_PER_PX, "leaf", incidence_angle_deg=angle, params=seg_params)


# ---------- input checks (programmer errors) ----------

def test_bad_inputs_raise(seg_params):
    img = paint(ellipse_mask(500, 400, 300, 150), LEAF_BGR, BG_LEAF_BGR)
    with pytest.raises(ValueError, match="mode"):
        segment_and_measure(img, MM_PER_PX, "flower", params=seg_params)
    with pytest.raises(ValueError, match="mm_per_pixel"):
        segment_and_measure(img, 0.0, "leaf", params=seg_params)
    with pytest.raises(ValueError, match="image"):
        segment_and_measure(img[:, :, 0], MM_PER_PX, "leaf", params=seg_params)
    with pytest.raises(KeyError, match="canny_low"):
        p = dict(seg_params)
        del p["canny_low"]
        segment_and_measure(img, MM_PER_PX, "leaf", params=p)


# ---------- mm_per_pixel_from_fov ----------

def test_mm_per_pixel_from_fov():
    # 90 deg field at 100 mm covers 2 x 100 x tan 45 = 200 mm over 1000 px -> 0.2 mm/px
    assert mm_per_pixel_from_fov(100.0, 90.0, 1000) == pytest.approx(0.2)
    # 60 deg at 300 mm over 2304 px: 2 x 300 x tan 30 / 2304
    assert mm_per_pixel_from_fov(300.0, 60.0, 2304) == pytest.approx(600 * math.tan(math.radians(30)) / 2304)


@pytest.mark.parametrize("args", [(0, 60, 100), (100, 0, 100), (100, 180, 100), (100, 60, 0)])
def test_mm_per_pixel_from_fov_rejects_bad(args):
    with pytest.raises(ValueError):
        mm_per_pixel_from_fov(*args)
