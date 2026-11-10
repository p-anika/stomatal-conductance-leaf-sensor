"""Find the leaf (or reference card) in a top-down colour photo and measure its shape (T3.1).

Steps (settings from config.yaml section `segmentation`):
  1. Contrast image: leaf mode uses excess green ExG = 2G - R - B (leaf bright, grey paper ~0);
     card mode uses inverted grey (black card bright on white paper).
  2. Gaussian blur, then Otsu threshold (picks the cut that best separates two brightness groups).
  3. Morphological open (removes specks) and close (bridges small gaps).
  4. Keep the largest connected component; fill its holes.
  5. Measure on the filled mask, with s = mm_per_pixel [mm px-1]:
       area_mm2      = N_px x s^2                       (pixel count)
       perimeter_mm  = contour length x s               (biased high by the pixel staircase; not corrected)
       d_min_mm      = (short side of min-area rectangle + 1 px) x s
       d_max_mm      = (long side  of min-area rectangle + 1 px) x s
       d_eq_mm       = sqrt(4 A / pi)
       aspect_ratio  = d_max / d_min
       eccentricity  = sqrt(1 - (b/a)^2), a >= b semi-axes of the fitted ellipse
       solidity      = contour area / convex-hull area
       circularity   = 4 pi A / P^2
       edge_density  = Canny edge pixels inside the eroded mask / pixels in the eroded mask
       n_components  = pieces of at least min_component_px after step 3
     The +1 px: contour points sit on pixel centres, so a run of n pixels spans n - 1 between
     centres; adding half a pixel at each end restores the true width.
  6. Optional tilt: A_true = A_projected / cos(incidence angle). Only area and d_eq are corrected
     (the tilt axis is unknown, so lengths cannot be); the result gets flag foreshortening_corrected.

Failure (valid False, reason segmentation_failed): no component, the mask touches the image
border, the mean contrast inside minus outside the mask is below min_contrast (Otsu splits even
pure noise into two groups), or the area is outside [min_area_mm2, max_area_mm2]. Bad arguments raise ValueError.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODES = ("leaf", "card")
REQUIRED_PARAMS = ("blur_ksize_px", "open_kernel_px", "close_kernel_px", "min_component_px",
                   "min_area_mm2", "max_area_mm2", "canny_low", "canny_high", "edge_erode_px",
                   "max_incidence_angle_deg", "min_contrast")
# Display colours for the review overlay only (BGR); not measurement settings.
_TINT_BGR = (0, 0, 255)
_CONTOUR_BGR = (0, 255, 255)
_RECT_BGR = (255, 0, 0)
_ELLIPSE_BGR = (255, 0, 255)
_TINT_ALPHA = 0.35


@dataclass(frozen=True)
class SegmentationResult:
    """Geometry in mm / mm2 (None when invalid). mask: bool (H, W), True = object, or None when
    nothing was found. overlay: BGR uint8 copy of the photo with the mask and fits drawn, or None.
    area_mm2 is the tilt-corrected area when an incidence angle was given; area_projected_mm2
    is always the area as seen in the photo."""
    valid: bool
    reason: str
    flags: tuple[str, ...]
    mode: str
    area_mm2: float | None
    area_projected_mm2: float | None
    perimeter_mm: float | None
    d_min_mm: float | None
    d_max_mm: float | None
    d_eq_mm: float | None
    aspect_ratio: float | None
    eccentricity: float | None
    solidity: float | None
    circularity: float | None
    edge_density: float | None
    n_components: int
    incidence_angle_deg: float | None
    mask: np.ndarray | None
    overlay: np.ndarray | None


def mm_per_pixel_from_fov(working_distance_mm: float, hfov_deg: float, image_width_px: int) -> float:
    """Pinhole-camera scale for a flat target square to the camera (fallback until camera_scale.py).

    Width seen [mm] = 2 x D x tan(hfov / 2);  mm_per_pixel = that width / image_width_px.
    Units: mm / (px) -> mm px-1. Ignores lens distortion.
    """
    if not (working_distance_mm > 0 and math.isfinite(working_distance_mm)):
        raise ValueError(f"working_distance_mm must be a positive number, got {working_distance_mm}")
    if not (0 < hfov_deg < 180):
        raise ValueError(f"hfov_deg must be between 0 and 180 degrees, got {hfov_deg}")
    if not (image_width_px > 0):
        raise ValueError(f"image_width_px must be positive, got {image_width_px}")
    return 2.0 * working_distance_mm * math.tan(math.radians(hfov_deg) / 2.0) / image_width_px


def _default_params() -> dict[str, Any]:
    from processing.config import load_config
    return load_config(Path(__file__).resolve().parent.parent / "config.yaml")["segmentation"]


def _check_params(params: dict[str, Any]) -> None:
    for key in REQUIRED_PARAMS:
        if key not in params:
            raise KeyError(f"config segmentation.{key} is missing (needed by leaf_segmentation)")


def _odd_kernel(size_px: int) -> np.ndarray:
    size = max(1, int(size_px))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def contrast_image(image_bgr: np.ndarray, mode: str) -> np.ndarray:
    """float32 image where the object is bright, in colour levels (not stretched).
    leaf: ExG = 2G - R - B (-510 to 510); card: 255 - grey (0 to 255).
    ExG cancels grey backgrounds (R = G = B gives 0) whatever their brightness."""
    if mode == "leaf":
        b, g, r = (image_bgr[:, :, i].astype(np.float32) for i in range(3))
        return 2.0 * g - r - b
    return 255.0 - cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def threshold_mask(contrast: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Stretch to 0-255, blur, Otsu threshold, open, close. Returns uint8 mask (0 or 255)."""
    contrast = cv2.normalize(contrast, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    k = int(params["blur_ksize_px"])
    if k > 1:
        k += (k % 2 == 0)
        contrast = cv2.GaussianBlur(contrast, (k, k), 0)
    _, mask = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _odd_kernel(params["open_kernel_px"]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _odd_kernel(params["close_kernel_px"]))
    return mask


def largest_component_filled(mask: np.ndarray, min_component_px: int) -> tuple[np.ndarray | None, int]:
    """(filled bool mask of the largest piece or None, number of pieces >= min_component_px)."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]           # label 0 is the background
    n_big = int(np.sum(areas >= min_component_px))
    if areas.size == 0:
        return None, 0
    biggest = 1 + int(np.argmax(areas))
    piece = (labels == biggest).astype(np.uint8)
    contours, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(piece)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)   # filling the outline fills holes
    return filled.astype(bool), n_big


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def edge_density(image_bgr: np.ndarray, mask: np.ndarray, params: dict[str, Any]) -> float:
    """Canny edge pixels inside the eroded mask / pixels in the eroded mask (dimensionless).
    Eroding first keeps the object's own outline out of the count, so this measures texture."""
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grey, float(params["canny_low"]), float(params["canny_high"])) > 0
    n_erode = int(params["edge_erode_px"])
    inner = mask.astype(np.uint8)
    if n_erode > 0:
        inner = cv2.erode(inner, _odd_kernel(2 * n_erode + 1))
    inner = inner.astype(bool)
    n_inner = int(inner.sum())
    if n_inner == 0:
        return 0.0
    return float(np.count_nonzero(edges & inner)) / n_inner


def _overlay(image_bgr: np.ndarray, mask: np.ndarray, contour: np.ndarray | None,
             rect: Any = None, ellipse: Any = None) -> np.ndarray:
    out = image_bgr.copy()
    tint = np.zeros_like(out)
    tint[:] = _TINT_BGR
    blended = cv2.addWeighted(out, 1 - _TINT_ALPHA, tint, _TINT_ALPHA, 0)
    out[mask] = blended[mask]
    if contour is not None:
        cv2.drawContours(out, [contour], -1, _CONTOUR_BGR, 2)
    if rect is not None:
        cv2.drawContours(out, [np.round(cv2.boxPoints(rect)).astype(np.int32)], -1, _RECT_BGR, 2)
    if ellipse is not None:
        cv2.ellipse(out, ellipse, _ELLIPSE_BGR, 2)
    return out


def _invalid(mode: str, angle: float | None, n_components: int, mask: np.ndarray | None,
             overlay: np.ndarray | None, why: str) -> SegmentationResult:
    logger.info("segmentation_failed (%s mode): %s", mode, why)
    return SegmentationResult(False, "segmentation_failed", (), mode, None, None, None, None, None,
                              None, None, None, None, None, None, n_components, angle, mask, overlay)


def segment_and_measure(image_bgr: np.ndarray, mm_per_pixel: float, mode: str,
                        incidence_angle_deg: float | None = None,
                        params: dict[str, Any] | None = None) -> SegmentationResult:
    """Segment one object and return its geometry (equations in the module docstring).

    image_bgr: uint8 (H, W, 3) photo in OpenCV's BGR order. mm_per_pixel: mm px-1 at the object.
    mode: "leaf" or "card". incidence_angle_deg: camera tilt from straight down, degrees, in
    [0, max_incidence_angle_deg]; None = no correction. params: config `segmentation` section
    (None = read config.yaml).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if not (isinstance(mm_per_pixel, (int, float)) and math.isfinite(mm_per_pixel) and mm_per_pixel > 0):
        raise ValueError(f"mm_per_pixel must be a positive number, got {mm_per_pixel}")
    img = np.asarray(image_bgr)
    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError(f"image_bgr must be a uint8 colour image of shape (H, W, 3), "
                         f"got shape {img.shape} dtype {img.dtype}")
    params = _default_params() if params is None else params
    _check_params(params)
    if incidence_angle_deg is not None:
        max_angle = float(params["max_incidence_angle_deg"])
        if not (math.isfinite(incidence_angle_deg) and 0.0 <= incidence_angle_deg <= max_angle):
            raise ValueError(f"incidence_angle_deg must be between 0 and {max_angle} degrees "
                             f"(segmentation.max_incidence_angle_deg), got {incidence_angle_deg}")

    contrast = contrast_image(img, mode)
    raw = threshold_mask(contrast, params)
    mask, n_comp = largest_component_filled(raw, int(params["min_component_px"]))
    if mask is None:
        return _invalid(mode, incidence_angle_deg, 0, None, None, "no object found")

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    if _touches_border(mask):
        return _invalid(mode, incidence_angle_deg, n_comp, mask, _overlay(img, mask, contour),
                        "mask touches the image border (object cut off)")
    if len(contour) < 5:
        return _invalid(mode, incidence_angle_deg, n_comp, mask, _overlay(img, mask, contour),
                        "object too small to fit an ellipse")

    # Otsu always splits an image in two, even pure noise; demand a real object-background gap.
    gap = float(contrast[mask].mean() - contrast[~mask].mean())
    if gap < float(params["min_contrast"]):
        return _invalid(mode, incidence_angle_deg, n_comp, mask, _overlay(img, mask, contour),
                        f"object-background contrast {gap:.1f} below min_contrast "
                        f"{params['min_contrast']}")

    s = float(mm_per_pixel)
    area_proj_mm2 = float(mask.sum()) * s * s
    flags: list[str] = []
    area_mm2 = area_proj_mm2
    if incidence_angle_deg is not None:
        # First-order foreshortening: a tilted flat leaf looks cos(angle) times smaller.
        area_mm2 = area_proj_mm2 / math.cos(math.radians(incidence_angle_deg))
        flags.append("foreshortening_corrected")

    rect = cv2.minAreaRect(contour)
    ellipse = cv2.fitEllipse(contour)
    overlay = _overlay(img, mask, contour, rect, ellipse)
    if not (params["min_area_mm2"] <= area_mm2 <= params["max_area_mm2"]):
        return _invalid(mode, incidence_angle_deg, n_comp, mask, overlay,
                        f"area {area_mm2:.1f} mm2 outside [{params['min_area_mm2']}, "
                        f"{params['max_area_mm2']}] mm2")

    (_, _), (rw, rh), _ = rect
    d_min_mm = (min(rw, rh) + 1.0) * s
    d_max_mm = (max(rw, rh) + 1.0) * s
    (_, _), (ax1, ax2), _ = ellipse
    a, b = max(ax1, ax2) / 2.0, min(ax1, ax2) / 2.0
    eccentricity = math.sqrt(max(0.0, 1.0 - (b / a) ** 2)) if a > 0 else 0.0
    hull_area_px = cv2.contourArea(cv2.convexHull(contour))
    solidity = cv2.contourArea(contour) / hull_area_px if hull_area_px > 0 else 0.0
    perimeter_mm = cv2.arcLength(contour, True) * s

    return SegmentationResult(
        valid=True, reason="ok", flags=tuple(flags), mode=mode,
        area_mm2=area_mm2, area_projected_mm2=area_proj_mm2, perimeter_mm=perimeter_mm,
        d_min_mm=d_min_mm, d_max_mm=d_max_mm, d_eq_mm=math.sqrt(4.0 * area_mm2 / math.pi),
        aspect_ratio=d_max_mm / d_min_mm, eccentricity=eccentricity, solidity=solidity,
        circularity=4.0 * math.pi * area_proj_mm2 / perimeter_mm**2,
        edge_density=edge_density(img, mask, params), n_components=n_comp,
        incidence_angle_deg=incidence_angle_deg, mask=mask, overlay=overlay,
    )
