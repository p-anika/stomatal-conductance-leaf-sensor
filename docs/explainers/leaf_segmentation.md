# leaf_segmentation.py (T3.1)

## What it does
It takes a colour photo taken straight down at a leaflet (or at a cut-out reference card), finds which pixels belong to the object, and measures the object's size and shape in millimetres. The most important output is **d_min_mm**, the leaf's width. The energy-balance code uses it as the leaf's "characteristic dimension", the length that sets how thick the still-air layer around the leaf is. The other shape numbers ("features") describe the leaf so that later steps (the ML correction, card-versus-leaf comparisons) can tell leaves and cards of different shapes apart.

It also returns the **mask** (a true/false picture: true = object) and an **overlay** (the photo with the object tinted red, its outline in yellow, the fitted rectangle in blue and the fitted ellipse in magenta) so a person can check every result by eye.

## How it works
Main function: `segment_and_measure(image_bgr, mm_per_pixel, mode, incidence_angle_deg=None, params=None)`.
- In: a photo (OpenCV colour order B, G, R, values 0–255); the scale `mm_per_pixel` (mm px⁻¹); `mode` = `"leaf"` or `"card"`; an optional camera tilt in degrees; and the settings (config.yaml `segmentation`, loaded automatically if `params` is None).
- Out: a `SegmentationResult` with `valid`, `reason`, `flags`, every feature below, `mask` and `overlay`.

Steps:

1. **Make the object bright (`contrast_image`).**
   - *Leaf mode:* **excess green**, ExG = 2G − R − B, for every pixel. A grey or white background has R ≈ G ≈ B, so ExG ≈ 0 however bright or dim the paper is. A green leaf has G much bigger than R and B, so ExG is large. Example: grey paper (200, 200, 200) gives 2·200 − 200 − 200 = 0. A leaf pixel (R 50, G 140, B 40) gives 2·140 − 50 − 40 = 190.
   - *Card mode:* the cards are matte black on white paper, so it uses darkness = 255 − grey. Black card ≈ 230, white paper ≈ 20.
2. **Pick the cut automatically (`threshold_mask`).** The contrast image is stretched to 0–255 and blurred a little (`blur_ksize_px`, so single noisy pixels don't matter). Then **Otsu's method** is applied. Otsu tries every possible cut-off and keeps the one that makes the "object" group and the "background" group each as uniform as possible (it maximises the difference between the two group means, weighted by group size). You don't have to choose a threshold by hand.
3. **Clean the mask.**
   - **Opening** (`open_kernel_px`) shaves the mask and grows it back, which deletes specks smaller than the brush.
   - **Closing** (`close_kernel_px`) grows the mask and shaves it back, which fills small gaps and nicks.
4. **Keep one object (`largest_component_filled`).** Touching true pixels are grouped into "connected components" (pieces). The biggest piece is kept. Its outline is then drawn filled in, which fills any holes (an insect hole, a bright vein). **n_components** counts the pieces of at least `min_component_px` pixels.
5. **Check it is believable.** The result is `valid = False`, `reason = segmentation_failed` when:
   - nothing is found;
   - the mask touches the photo edge (the object is cut off, so every size would be wrong);
   - the contrast gap is too small: the average contrast inside the mask minus outside is below `min_contrast`. Otsu always splits an image in two, even a photo of blank paper. Without this check a noise blob passed as a "leaf" in testing;
   - the area is outside `min_area_mm2`–`max_area_mm2` (usually a wrong mm_per_pixel).

   The mask and overlay are still returned when possible, so you can see what went wrong.
6. **Measure.** Each feature is explained in the next section.
7. **Optional tilt correction.** See "Tilt correction" below.

Helper: `mm_per_pixel_from_fov(working_distance_mm, hfov_deg, image_width_px)` gives a rough scale from the camera geometry until `camera_scale.py` exists. It returns mm px⁻¹.

### Every feature, in plain words
Let s = mm_per_pixel, A = area, P = perimeter.

| Feature | Plain meaning | Formula | Units | Typical values |
|---|---|---|---|---|
| **area_mm2** | How much paper the leaf covers | number of mask pixels × s² (÷ cos θ if tilted) | mm² | soybean leaflet: a few thousand mm² |
| **area_projected_mm2** | Area exactly as seen in the photo (never tilt-corrected) | pixels × s² | mm² | |
| **perimeter_mm** | Length of the outline, like a string laid round the edge | length of the pixel outline × s | mm | |
| **d_min_mm** | The leaf's **width**: the short side of the smallest rectangle that fits round it (the rectangle may be rotated) | (short side + 1 px) × s | mm | the energy-balance length |
| **d_max_mm** | The leaf's **length**: long side of that rectangle | (long side + 1 px) × s | mm | |
| **d_eq_mm** | Diameter of a circle with the same area | √(4A/π) | mm | |
| **aspect_ratio** | How stretched it is | d_max / d_min | none | 1 = as long as wide; 2 = twice as long |
| **eccentricity** | How stretched the *best-fitting ellipse* is | √(1 − (b/a)²), a and b = long and short semi-axes | none | 0 = circle, near 1 = very thin |
| **solidity** | How "filled-in" the shape is: a rubber band stretched round it encloses the convex hull; solidity = area ÷ hull area | contour area / hull area | none | 1 = no dents; lobed or notched shapes < 1 |
| **circularity** | How close to a circle, using the outline length | 4πA / P² | none | circle 1 in theory (≈ 0.91 here, see Limitations); ragged edges much lower |
| **edge_density** | A texture measure: how much fine detail (veins, spots, wrinkles) is inside the leaf | Canny edge pixels inside the shrunken mask ÷ pixels in the shrunken mask | none (0–1) | smooth card ≈ 0; veined leaf larger |
| **n_components** | How many separate pieces were found before keeping the biggest | count of pieces ≥ min_component_px | count | 1 is ideal; > 1 means a second object or debris |

More on the less obvious ones:
- **Why +1 px on d_min and d_max?** The outline OpenCV traces runs through the *centres* of the edge pixels. Something 300 pixels wide has 299 pixel-steps between its first and last centres. Adding half a pixel at each end gives back the true width.
- **Eccentricity vs aspect ratio:** both measure stretch. Aspect ratio comes from the rectangle, so a pointy tip counts fully. Eccentricity comes from an ellipse fitted to the whole outline, so it is smoother.
- **Solidity is the stand-in for cupping.** A top-down photo can't see a leaf curl upward, but curling and folding usually pull the outline in and make dents, which lowers solidity. The incidence angle is the other stand-in (spec T3.1).
- **Edge density:** the Canny edge detector marks pixels where brightness changes sharply (`canny_low` and `canny_high` set how sharp). The mask is first shrunk by `edge_erode_px` pixels so the leaf's own outline doesn't count. Only texture inside counts.

### Tilt correction (owner decision 2026-11-09)
A flat leaf tilted by θ from facing the camera looks shorter in one direction by cos θ, so its area looks cos θ times smaller. When `incidence_angle_deg` is given:
- `area_mm2 = area_projected_mm2 / cos θ`, and `d_eq_mm` is recomputed from it;
- **d_min, d_max, perimeter and circularity are not changed.** We don't know which direction the leaf is tilted in, so we can't tell which length shrank;
- the flag `foreshortening_corrected` is added;
- angles outside 0–`max_incidence_angle_deg` (60°) raise an error. At steep angles a small angle mistake makes a big area mistake (see below).

Units check: (mm²) / (dimensionless cos) = mm². ✓

### Deriving mm_per_pixel_from_fov
A pinhole camera at distance D from a flat target sees a strip of width w with tan(hfov/2) = (w/2)/D, so w = 2·D·tan(hfov/2). Spread over N pixels: s = 2·D·tan(hfov/2)/N.
Units: mm × (dimensionless) / px = mm px⁻¹. ✓
Example: D = 100 mm, hfov = 90°, N = 1000 px → 2·100·1/1000 = 0.2 mm px⁻¹ (this is a test).

## Why it is done this way
- **Excess green, not plain brightness:** leaf and grey paper can have similar brightness, but they always differ in greenness. ExG also ignores how bright the lamp is, because uniform grey gives 0. Rejected: hue thresholds in HSV, which need hand-tuned hue limits.
- **Otsu, not a fixed threshold:** lighting changes between sessions, and Otsu adapts to each photo. The `min_contrast` check covers Otsu's one weakness: it always finds a split, even when there is nothing there.
- **Pixel-count area, not the outline's area:** the outline through pixel centres cuts off half a pixel all round (about 0.3 % low for a 300 px leaf). Counting pixels doesn't have that problem.
- **d_min from the minimum-area rectangle:** this is the spec's definition. For an ellipse it equals the minor axis exactly.
- **Largest component + hole filling:** a leaf is one solid piece. Holes would make the area too small but don't change how heat moves across the leaf's surface.
- **Fail instead of guessing:** a cut-off or noise-only mask gives wrong numbers that look right. Returning `segmentation_failed` keeps them out of every table.

## Worked example
A drawn ellipse (test `test_ellipse_area_and_dmin_within_2pct`), semi-axes a = 300 px and b = 150 px, at s = 0.1 mm px⁻¹, green (R 50, G 140, B 40) on grey (200, 200, 200) paper.

1. **ExG:** leaf 190, paper 0. Otsu cuts between them. Contrast gap ≈ 190 ≥ 30 ✓.
2. **Mask:** 141 345 pixels, one component, not touching the border.
3. **area_mm2** = 141 345 × 0.1² = **1413.4 mm²**. True area π·300·150·0.01 = 1413.7 mm², error −0.02 % ✓ (limit 2 %).
4. **d_min:** rectangle short side 298 px. (298 + 1) × 0.1 = **29.9 mm**. True 2b·s = 30.0 mm, error −0.3 % ✓.
5. **d_max** = 59.9 mm. **d_eq** = √(4 × 1413.4/π) = **42.4 mm**. **aspect_ratio** = 59.9/29.9 = **2.00**.
6. **eccentricity:** b/a = 0.5, so √(1 − 0.25) = **0.866** (measured 0.8667).
7. **solidity** = **0.997**. An ellipse is convex, so ideally 1.
8. **perimeter_mm** = **152.7 mm**. The true ellipse perimeter (Ramanujan's formula, π[3(a+b) − √((3a+b)(a+3b))]) is 145.3 mm, so the staircase adds +5.1 %. As a result **circularity** = 4π·1413.4/152.7² = **0.76** instead of the true 0.84.
9. **edge_density** = **0.0**: a smooth painted shape has no inner texture.
10. **With a 30° tilt:** area = 1413.4 / cos 30° = 1413.4 / 0.866 = **1632.1 mm²**, d_eq = **45.6 mm**, flag `foreshortening_corrected`. d_min stays 29.9 mm.

The other drawn shapes:
- **Circle,** r = 200 px: area 1256.3 mm² (true 1256.6), d_min 39.9 mm (true 40.0), circularity 0.906.
- **Five-lobed polygon:** solidity 0.805, circularity 0.51. Area and d_min are within 2 % of the exact polygon values.

## Methods paragraph (draft)
Leaf and reference-card geometry was measured from top-down colour photographs with a custom Python routine (Python 3.13, OpenCV 5.0.0, NumPy 2.5.4). Leaf pixels were separated from the neutral background using the excess-green index (2G − R − B); dark reference cards were separated using inverted greyscale. The threshold was chosen automatically for each image by Otsu's method after a 5-pixel Gaussian blur. The mask was cleaned by morphological opening and closing, reduced to its largest connected component, and hole-filled. Images were rejected when the mask touched the image border, when the object–background contrast was below a fixed minimum, or when the area was outside 1–200 cm². Area was computed from the pixel count. The characteristic dimension d_min was the short side of the minimum-area bounding rectangle. Shape descriptors were also recorded: equivalent diameter, aspect ratio, ellipse eccentricity, solidity, circularity, and a Canny edge density inside the eroded mask as a texture proxy. When the camera was tilted, projected area was divided by the cosine of the incidence angle. On synthetic shapes of known size, area and d_min were within 2 % of the true values. [Add: number of real images checked, and the scale method: ruler or camera_scale.py.]

## Limitations
1. **Only tested on drawn shapes.** None of the settings (blur, morphology sizes, min_contrast, area limits, Canny thresholds) has been tuned on real photos. They are placeholders until real Arducam photos exist (BLOCKED-CAM). The spec's "five real leaf overlays look correct" check is still to do. Phone photos in data/samples/ test only whether segmentation works.
2. **ML features must come from Arducam photos only (owner decision 2026-11-09).** Edge density (and to a lesser extent perimeter and circularity) depends on resolution, focus, compression and Canny settings. A phone photo and an Arducam photo of the same leaf give different numbers, so mixing them would teach the model camera differences instead of leaf differences.
3. **Perimeter is about 5 % too long and circularity about 10 % too low.** The pixel outline goes in steps (stairs) instead of a smooth curve. This is documented, not corrected (owner decision 2026-11-09). Compare circularity only between photos taken at a similar resolution.
4. **Settings are in pixels, not mm.** A different resolution or distance changes what counts as a "speck". Re-check the settings when the camera setup changes.
5. **Tilt correction is first-order.** It assumes a flat leaf with one tilt and a known angle. Tilt error grows with angle: the relative area error is about tan θ × angle error, so a 5° mistake at 60° gives about 15 %. Lengths are not corrected.
6. **No curvature.** A top-down photo can't see cupping. Solidity and incidence angle are only stand-ins.
7. **Background and lighting matter.** ExG fails on green or brown backgrounds and when the leaf is yellow or very dark. Shadows can join the card in card mode. A black ruler can be picked up as a card.
8. **Scale is only as good as mm_per_pixel.** The ruler must lie at the same height as the leaf. `mm_per_pixel_from_fov` ignores lens distortion.
9. **Characteristic dimension (open item).** The energy balance uses d_min directly. From memory (UNVERIFIED, docs/sources.md), Campbell & Norman scale leaf width by about 0.72. This is logged as an open item, not changed.
