"""Regions of interest (ROIs) on the 24 x 32 MLX90640 thermal frame (T3.2).

Frame: 768 values in deg C, row-major, 24 rows x 32 columns (the sensor's pixel layout).
Masks are boolean arrays of the frame's shape; True = pixel belongs to the ROI.

roi_stats trims outlier pixels before averaging (one pass):
    keep |T - median| <= k x s x MAD,   MAD = median(|T - median|),  s = 1.4826 (MAD -> sigma)
so a pixel that is partly background cannot pull the leaf mean. With MAD = 0 nothing is trimmed.

Pixel footprint (pinhole camera, flat target square to the camera):
    footprint [mm] = 2 x distance [mm] x tan(fov / 2) / number of pixels across that fov
horizontal uses hfov and 32 columns, vertical uses vfov and 24 rows (config sensors.mlx90640).

Deviations from the spec signatures (decision_log 2026-11-09): roi_stats, pixel_footprint_mm
and warp_mask_to_thermal take cfg so their settings come from config; pixel_footprint_mm
returns both footprints.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy import ndimage

# Sensor pixel layout (MLX90640 datasheet geometry, not a tunable setting).
MLX_ROWS, MLX_COLS = 24, 32
MLX_SHAPE = (MLX_ROWS, MLX_COLS)


def frame_from_list(values: Sequence[float]) -> np.ndarray:
    """768 values (deg C, row-major) -> float64 array of shape (24, 32)."""
    arr = np.asarray(values, dtype=float)
    if arr.size != MLX_ROWS * MLX_COLS:
        raise ValueError(f"a thermal frame needs {MLX_ROWS * MLX_COLS} values (768), got {arr.size}")
    return arr.reshape(MLX_SHAPE)


def rect_mask(r0: int, r1: int, c0: int, c1: int,
              shape: tuple[int, int] = MLX_SHAPE) -> np.ndarray:
    """Rectangle of rows r0..r1-1 and columns c0..c1-1 (Python-style half-open ranges)."""
    rows, cols = shape
    if not (0 <= r0 < r1 <= rows and 0 <= c0 < c1 <= cols):
        raise ValueError(f"rectangle rows [{r0}, {r1}) and columns [{c0}, {c1}) must be non-empty "
                         f"and inside a {rows} x {cols} frame")
    m = np.zeros(shape, dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


def erode_mask(mask: np.ndarray, n: int) -> np.ndarray:
    """Remove n rings of edge pixels (3 x 3 square neighbourhood). Pixels beyond the frame
    border count as outside, so ROI pixels on the frame edge are removed too."""
    if n < 0:
        raise ValueError(f"erosion steps must be 0 or more, got {n}")
    mask = np.asarray(mask, dtype=bool)
    if n == 0:
        return mask.copy()
    return ndimage.binary_erosion(mask, structure=np.ones((3, 3), bool), iterations=n,
                                  border_value=0)


@dataclass(frozen=True)
class RoiStats:
    """Temperatures deg C (None when invalid). n_px = pixels used after trimming;
    n_trimmed = pixels removed as outliers; n_nonfinite = NaN/inf pixels ignored."""
    mean_c: float | None
    median_c: float | None
    std_c: float | None          # sample standard deviation (ddof = 1)
    n_px: int
    n_trimmed: int
    n_nonfinite: int
    valid: bool
    reason: str


def roi_stats(frame: np.ndarray, mask: np.ndarray, cfg: dict[str, Any],
              min_pixels: int | None = None) -> RoiStats:
    """Mean, median and SD of the ROI after one MAD trimming pass (see module docstring).

    min_pixels defaults to config acquisition.min_roi_pixels. Fewer usable pixels than that,
    before or after trimming -> valid False, reason roi_too_small.
    """
    frame = np.asarray(frame, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if frame.shape != mask.shape:
        raise ValueError(f"frame shape {frame.shape} and mask shape {mask.shape} differ")
    acq = cfg["acquisition"]
    min_px = int(acq["min_roi_pixels"] if min_pixels is None else min_pixels)
    values = frame[mask]
    finite = np.isfinite(values)
    n_nonfinite = int((~finite).sum())
    values = values[finite]

    n_trimmed = 0
    if values.size >= min_px:
        med = np.median(values)
        mad = np.median(np.abs(values - med))
        if mad > 0:
            limit = acq["roi_mad_k"] * cfg["qc"]["mad_to_sigma"] * mad
            keep = np.abs(values - med) <= limit
            n_trimmed = int((~keep).sum())
            values = values[keep]
    if values.size < min_px:
        return RoiStats(None, None, None, int(values.size), n_trimmed, n_nonfinite, False,
                        "roi_too_small")
    return RoiStats(float(values.mean()), float(np.median(values)), float(values.std(ddof=1)),
                    int(values.size), n_trimmed, n_nonfinite, True, "ok")


@dataclass(frozen=True)
class Footprint:
    """Size of one thermal pixel on a flat target at the given distance, mm."""
    horizontal_mm: float
    vertical_mm: float

    @property
    def largest_mm(self) -> float:
        """Use this for pure-pixel decisions (owner decision 2026-11-09): a pixel is only
        surely all-leaf if the leaf is wider than the larger side."""
        return max(self.horizontal_mm, self.vertical_mm)


def pixel_footprint_mm(distance_mm: float, cfg: dict[str, Any]) -> Footprint:
    """footprint = 2 d tan(fov/2) / n, horizontal (hfov, 32 columns) and vertical (vfov, 24 rows).
    fov from config sensors.mlx90640 (degrees)."""
    if not (math.isfinite(distance_mm) and distance_mm > 0):
        raise ValueError(f"distance_mm must be a positive number, got {distance_mm}")
    s = cfg["sensors"]["mlx90640"]
    h = 2.0 * distance_mm * math.tan(math.radians(s["hfov_deg"]) / 2.0) / MLX_COLS
    v = 2.0 * distance_mm * math.tan(math.radians(s["vfov_deg"]) / 2.0) / MLX_ROWS
    return Footprint(h, v)


def warp_mask_to_thermal(mask_rgb: np.ndarray, matrix: np.ndarray, shape: tuple[int, int],
                         cfg: dict[str, Any]) -> np.ndarray:
    """Map an RGB-image leaf mask onto the thermal grid. [BLOCKED-CAM to validate]

    matrix (3 x 3) maps RGB pixel coordinates (x = column, y = row, homogeneous) to thermal
    coordinates; pixel centres are at whole numbers. Each thermal pixel is sampled at
    S x S points (S = acquisition.warp_supersample); each point is mapped back into the RGB
    image and looked up (nearest pixel; outside the image = not leaf). A thermal pixel is
    kept if its covered fraction >= acquisition.min_pixel_coverage (1.0 = pure pixels only).
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"matrix must be 3 x 3, got shape {matrix.shape}")
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        raise ValueError("matrix cannot be inverted") from None
    mask_rgb = np.asarray(mask_rgb, dtype=bool)
    acq = cfg["acquisition"]
    n_sub = int(acq["warp_supersample"])
    rows, cols = shape

    offsets = (np.arange(n_sub) + 0.5) / n_sub - 0.5
    ty = (np.arange(rows)[:, None] + offsets[None, :]).reshape(-1)          # rows*S
    tx = (np.arange(cols)[:, None] + offsets[None, :]).reshape(-1)          # cols*S
    gx, gy = np.meshgrid(tx, ty)                                            # (rows*S, cols*S)
    pts = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)])
    src = inverse @ pts
    with np.errstate(divide="ignore", invalid="ignore"):
        sx = np.rint(src[0] / src[2])
        sy = np.rint(src[1] / src[2])
    h_rgb, w_rgb = mask_rgb.shape
    inside = np.isfinite(sx) & np.isfinite(sy) & (sx >= 0) & (sx < w_rgb) & (sy >= 0) & (sy < h_rgb)
    hit = np.zeros(gx.size, dtype=bool)
    hit[inside] = mask_rgb[sy[inside].astype(int), sx[inside].astype(int)]
    coverage = hit.reshape(rows, n_sub, cols, n_sub).mean(axis=(1, 3))
    # Tiny tolerance so exactly-full pixels are not lost to floating-point rounding.
    return coverage >= acq["min_pixel_coverage"] - 1e-9
