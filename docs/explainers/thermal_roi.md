# thermal_roi (processing/thermal_roi.py, task T3.2)

## What it does
The thermal camera (MLX90640) gives a small picture of 24 × 32 temperatures. Only some of those pixels are the leaf (or the card, or the wet reference). This module marks which pixels belong to each object (a **mask**, also called the ROI, region of interest), and then works out that object's temperature: mean, median and spread. It also says how big one pixel is on the leaf, and (for later) turns a leaf outline from the colour camera into a mask on the thermal grid.

## How it works
1. **`frame_from_list(values)`**: 768 numbers (°C, as the sensor library gives them, row by row) → a 24 × 32 array. Any other count raises `ValueError`.
2. **`rect_mask(r0, r1, c0, c1)`**: a rectangle mask covering rows r0 to r1 − 1 and columns c0 to c1 − 1 (the end is not included, as in Python). Must be non-empty and inside the frame.
3. **`erode_mask(mask, n)`**: peels n rings of pixels off the edge of a mask. Pixels touching the frame border are also removed, because outside the frame counts as "not leaf". Example: an 8 × 12 rectangle (96 px) becomes 6 × 10 (60 px) after one step.
4. **`roi_stats(frame, mask, cfg, min_pixels=None)`** returns `RoiStats(mean_c, median_c, std_c, n_px, n_trimmed, n_nonfinite, valid, reason)`:
   1. take the frame's values inside the mask, ignore NaN or infinite pixels (counted in `n_nonfinite`);
   2. if there are fewer than `min_pixels` (config `acquisition.min_roi_pixels`, 6) → `valid=False, reason=roi_too_small`;
   3. **trim outliers once:** median m, MAD = median(|T − m|); drop every pixel with |T − m| > k × 1.4826 × MAD, with k = `acquisition.roi_mad_k` (3). If the MAD is 0, nothing is dropped;
   4. if fewer than `min_pixels` are left → `roi_too_small`;
   5. otherwise return the mean, median and sample standard deviation (°C) of the kept pixels.
5. **`pixel_footprint_mm(distance_mm, cfg)`** returns `Footprint(horizontal_mm, vertical_mm)` and `largest_mm`, the size of one pixel on a flat target at that distance (derived below).
6. **`warp_mask_to_thermal(mask_rgb, matrix, shape, cfg)`** \[BLOCKED-CAM: tested only on synthetic masks\]: `matrix` (3 × 3) maps colour-camera pixel positions to thermal pixel positions (from camera registration, later). Each thermal pixel is checked at 4 × 4 = 16 points (`acquisition.warp_supersample`). Each point is mapped back into the colour image to see whether it lands on the leaf. The **covered fraction** is the share of the 16 points that do. The pixel is kept only if that fraction is ≥ `acquisition.min_pixel_coverage` (1.0, so only pixels that are entirely leaf).

## Why it is done this way
- **Mixed pixels.** One thermal pixel covers about 6.5 × 5.3 mm at 200 mm. A pixel on the leaf edge sees partly leaf and partly the background, so its reading is somewhere in between. The leaf's temperature must come from **pure** pixels only. Three tools work together: eroding the mask removes the edge ring, `min_pixel_coverage = 1.0` keeps only fully covered pixels, and MAD trimming removes any mixed pixel that is still left.
- **Median and MAD, not mean and SD, for trimming.** One background pixel 3 °C away inflates the standard deviation so much that it could pass a "mean ± 3 SD" test. The median and MAD are hardly moved by it, so it is caught. The factor 1.4826 turns the MAD into a standard deviation for normal noise (derivation in [qc.md](qc.md)).
- **One trimming pass.** Repeating the trim until nothing changes would keep eating into the real spread of the leaf; one pass removes clear outliers only.
- **Too small → invalid, not a guess.** With fewer than 6 pixels the average is too noisy to use. The window is labelled `roi_too_small` rather than given a number.
- **Field of view from config, both directions** (owner decision, 2026-11-09). The sensor's view is 55° wide and 35° tall (Adafruit product listing), and its pixels are not square on the target, so both footprints are reported. For a pure-pixel decision (is the leaf big enough to have pure pixels?) the **larger** side is used, because a pixel is only surely all-leaf if the leaf is wider than that side.

### Pixel footprint: derivation
The camera sees an angle (field of view, fov) spread over n pixels. At distance d, the view is a strip of width W. Draw a triangle from the lens to the two edges of the strip: half the angle, fov/2, has tan(fov/2) = (W/2)/d, so

  W = 2 d tan(fov / 2),  footprint = W / n.

Units: d in mm, tan has no units, n is a count → mm per pixel ✓.

At d = 200 mm:
- horizontal: 2 × 200 × tan(27.5°) / 32 = 400 × 0.52057 / 32 = **6.51 mm**
- vertical: 2 × 200 × tan(17.5°) / 24 = 400 × 0.31530 / 24 = **5.26 mm**
- largest: **6.51 mm**

This assumes the target is flat and square to the camera, and that the pixels are evenly spread across the view (no lens distortion).

## Worked example
Synthetic frame (test fixture): background 22.0 °C, leaf block rows 8–15, columns 10–21 at 25.0 °C, plus noise of σ 0.05 °C.
1. `rect_mask(8, 16, 10, 22)` → 96 pixels. `erode_mask(..., 1)` → 60 pixels (6 × 10), all inside the leaf.
2. Median = 24.996, MAD = 0.0366, so the trim limit is 3 × 1.4826 × 0.0366 = 0.163 °C. Pure noise at σ 0.05 goes past this limit only rarely; here 1 of the 60 pixels does and is trimmed (`n_trimmed = 1`, `n_px = 59`). Because noise is trimmed equally often on both sides, this barely moves the mean.
3. Result: mean 24.998 °C, SD 0.047 °C, `valid=True`.

Mixed-pixel case (test): a 4 × 4 leaf at 25.0 ± 0.1 °C with one corner pixel at 22.5 °C (half background). Median = 25.03, MAD = 0.038, limit = 0.17 °C. The 22.5 °C pixel is 2.5 °C away and is trimmed (`n_trimmed = 1`), so the mean is 25.01 °C instead of being pulled down to 24.85 °C.

Too-small case: a 2 × 2 patch (4 pixels) → `valid=False, reason=roi_too_small`.

## Methods paragraph (draft)
Leaf, reference-card and wet-reference temperatures were taken from regions of interest on the 24 × 32-pixel thermal image (MLX90640; 55° × 35° field of view, giving 6.5 × 5.3 mm pixels at 200 mm). Regions were eroded by [n] pixel(s) to exclude pixels mixing leaf and background, and [for photo-derived masks, only thermal pixels fully covered by the leaf mask were kept]. Within each region, pixels deviating from the median by more than 3 × 1.4826 × the median absolute deviation were excluded (one pass), and the mean, median and standard deviation of the remaining pixels were recorded; regions with fewer than 6 usable pixels were marked invalid. Software: Python 3.13, NumPy, SciPy [versions].

## Limitations
1. **Field of view not yet measured on this camera.** 55° × 35° is from the product listing; it must be confirmed by imaging a warm object of known size at a known distance (T7.4). If it is off, every footprint scales with it.
2. **The warp is untested on real images** (BLOCKED-CAM). The 3 × 3 matrix must come from registering the colour and thermal cameras; nearest-pixel lookup and 4 × 4 sampling are approximations of the true pixel area.
3. **Flat-target assumption.** A tilted or curled leaf has larger pixels on its surface than the footprint says (see incidence angle in session metadata).
4. **Trimming can be one-sided.** Mixed pixels are all cooler (or all warmer) than the leaf, so trimming them moves the mean towards the true leaf value. That is the purpose, but if real leaf pixels differ a lot (for example a sunlit and a shaded half), trimming could remove real leaf.
5. **Thermal camera optics.** The sensor's own blur spreads a pixel's view onto its neighbours, so a pixel just inside the edge can still be slightly mixed; erosion helps.
6. **Pixel order.** `frame_from_list` assumes the library's row-by-row order and does not flip or mirror the image. Whether the image is mirrored must be checked once on the Pi with a warm object on one side.
