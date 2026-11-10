# measurement_windows.py and extract_photo_features.py (T3.4, T3.5)

## What it does
The logger writes one sensor row every 5 s, and the camera takes one photo every 60 s. A single 5-second reading is too noisy to compute stomatal conductance from, and a leaf that is still warming up or cooling down breaks the energy-balance assumption that the leaf is in **steady state** (heat in = heat out).

So `measurement_windows.py`:
1. groups the rows into **60-second windows** and averages them;
2. decides whether each window is **steady** (the leaf temperature has stopped changing and the leaf has had enough time under the light);
3. gives each window the **leaf geometry** (d_min and the rest) from the photo or photos taken closest in time.

`extract_photo_features.py` is the step before that. It runs the leaf finder (T3.1) on every photo of a session and writes one row per photo to `data/processed/<session>/photo_features.csv`, plus an overlay picture for each photo so you can check the masks.

## How it works

### extract_photo_features (T3.5)
`python -m processing.extract_photo_features --session SESSION_ID [--mm-per-pixel 0.085] [--mode leaf|card]`
1. **Find the images.** Every `.jpg`, `.jpeg` or `.png` in `data/raw/SESSION_ID/images/`, in alphabetical order. That folder is only read, never written.
2. **Scale (mm per pixel).** Use `--mm-per-pixel` if given, otherwise that image's row in `images/scale.csv`. An image with no scale is still listed, with `ok = false` and `reason = missing_input`. If there is no scale at all, the command stops with a message.
3. **Tilt.** Use `--incidence-angle` if given, otherwise `incidence_angle_deg` from `session.json`, otherwise no correction. If an angle is given, even 0°, the T3.1 flag `foreshortening_corrected` is set.
4. **Photo time.** The time of the `sensor_log.csv` row whose `image_file` column names the photo. The corrected clock time (`timestamp_corrected_iso`) is used when it exists, otherwise `timestamp_iso`. A photo no log row names (for example a phone photo) gets a blank time. Its features are still measured, but it can't be paired with a window.
5. **Measure** with `segment_and_measure` (T3.1) and save the overlay as `overlays/<name>_overlay.png`.
6. **Write** `photo_features.csv` with these columns: timestamp, image_file, the 12 geometry columns, ok, reason, flags, mm_per_pixel, incidence_angle_deg. `processing.data_io.load_photo_features` reads it back and checks the columns.

### build_windows (T3.4)
`build_windows(sensor_df, events_df, meta, cfg, photo_features=None)` returns one row per window.
1. **QC first.** `qc.add_qc_flags` is run inside, so it can't be forgotten. Any row with a QC flag is left out of the averages. `n_excluded` counts how many rows were left out.
2. **Windows.** They start at the first logged time and run back to back, 60 s each (`window_sec`). A row belongs to the window whose interval contains it, start included, end excluded.
3. **Averages.** For every numeric column, the window row keeps the column's name for the mean (e.g. `air_temp_c`) and adds `_sd` for the standard deviation (e.g. `air_temp_c_sd`). Counters (`monotonic_s`, `thermal_frame_idx`) are not averaged.
4. **Steady-state rule** (next section).
5. **Photo pairing** (section after that), if `photo_features` is given.
6. **One reason, all problems in flags.** `reason` is the first of these that applies:
   - `missing_input`: fewer than 10 clean rows, or no leaf temperature.
   - `not_steady_state`
   - `no_photo_in_window`
   - otherwise `ok` (and `valid = true`).

   Every other problem is also written in `flags`, so a window that is both drifting and missing a photo says so.

### The steady-state rule
A window is **steady** only if all three are true:

| Test | Setting | Plain meaning |
|---|---|---|
| \|slope\| < 0.05 °C min⁻¹ | `stability_slope_c_per_min` | The leaf temperature has stopped rising or falling |
| acclimation ≥ 30 min | `min_acclimation_minutes` | The leaf has had time to open its stomata under this light |
| data reaches back 3 min | `stability_minutes` | There is enough history to judge the slope |

- **Slope:** take the clean rows from the **3 minutes ending at the window's end**, which is longer than the window itself. Fit the straight line *T = m·t + c* by least squares (`np.polyfit`) with t in minutes. The slope *m* is in °C min⁻¹. Why 3 minutes and not 60 s: a longer stretch makes the slope much less noisy (see "Why").
- **Acclimation time:** the minutes you typed in at session start (`acclimation_minutes`), plus the minutes since the first log row. If `events.csv` has a row labelled `lamp_on` (setting `acclimation_reset_label`) at or before the window's end, the clock instead restarts at the latest such event, because new light means the leaf has to acclimate again.
- **Flags explaining a failure:** `leaf_temp_drifting`, `acclimation_too_short`, `stability_window_incomplete`.

### Photo-to-window pairing (B15)
`attach_photo_features(windows, photo_features, max_gap_sec, leaf_moved_range_frac)`
1. Only photos with `ok = true` and a timestamp are used. Failed photos are ignored.
2. If one or more photos were taken **inside** the window, each feature is the **median** of those photos.
3. Otherwise, use the single photo **nearest the window's centre**, if it is within `max_photo_gap_sec` (120 s).
4. Otherwise the features are blank and the window gets `no_photo_in_window`.
5. `photo_gap_sec` is the time from the window centre to the closest photo used, and `n_photos` is how many photos were used.
6. Matching is **by time only, never by row number**. Shuffling the photo table gives the same answer (there is a test for this).
7. **leaf_moved:** take d_min_mm from every ok photo in the session. If (largest − smallest) ÷ median > 0.05 (`leaf_moved_range_frac`), every window gets the flag `leaf_moved`. It is a warning, not a failure: the leaf may have shifted or curled, so its thermal region and geometry may be out of date.

## Why it is done this way
- **Averaging over 60 s:** 12 samples average out sensor noise. One minute is still short compared with how slowly a leaf's temperature settles.
- **Slope over 3 minutes instead of over 60 s:** the uncertainty of a fitted slope is SE = σ / √Σ(tᵢ − t̄)², where σ is the per-sample noise. For 36 samples 5 s apart, Σ(tᵢ − t̄)² = 27.0 min². For 12 samples it is only 1.0 min², so the slope from a 60 s stretch would be about 5 times noisier.
  Units: °C / √(min²) = °C min⁻¹ ✓.
- **How often a perfectly flat leaf fails the test (false alarm):** if the noise is random and independent from sample to sample, the slope of a flat leaf scatters with standard error SE, and it fails when |slope| ≥ 0.05.

  | Noise per sample σ | SE of slope | False-alarm rate |
  |---|---|---|
  | 0.05 °C | 0.010 °C min⁻¹ | ≈ 0 % |
  | 0.10 °C | 0.019 °C min⁻¹ | ≈ 1 % (a 20 000-run simulation gave 1.06 %) |
  | 0.15 °C | 0.029 °C min⁻¹ | ≈ 8 % |
  | 0.20 °C | 0.039 °C min⁻¹ | ≈ 19 % |
  | 0.30 °C | 0.058 °C min⁻¹ | ≈ 39 % |

  The threshold was **not** changed (owner decision 2026-11-09). The real noise of the leaf ROI temperature must be measured in **T7.4 and T7.5** and checked against this table. Noise that is correlated over time (slow wobbles) gives more false alarms than the table shows.
- **Median of photos, not mean:** one bad mask can't drag the geometry.
- **Nearest photo within 120 s:** leaf shape changes slowly, so a photo 1–2 minutes away is still a fair description. Beyond that it isn't trusted.
- **Range test for leaf_moved (owner decision):** (max − min)/median reacts to a single sudden shift. The rejected alternative, a CV (SD/mean) test, would need a much bigger shift to trip at the same 5 %.
- **One reason plus flags:** tables can be filtered on one reason, and the flags keep every other problem visible.
- **Rejected:** windows aligned to event markers (not needed yet); dropping windows that fail (failed windows stay in the table, with their reason).

## Worked example
Fabricated session (tests/test_measurement_windows.py):
- Log starts 10:00:00 with one row every 5 s for 10 minutes, so 120 rows make 10 windows of 12.
- Logged acclimation is 25 min; leaf temperature is flat at 25.00 °C (noise 0.02 °C).

1. **Window 2** (10:02:00–10:03:00) ends 3 min after the start. The 3-minute stretch has 36 rows and the slope ≈ 0.00 °C min⁻¹ < 0.05 ✓. Acclimation = 25 + 3 = **28 min** < 30 ✗, so `not_steady_state` with flag `acclimation_too_short`.
2. **Window 4** (ends 10:05:00): acclimation = 25 + 5 = 30 min ≥ 30 ✓, so **steady**.
3. **Window 0 and 1:** the data don't reach back 3 min, so they get `stability_window_incomplete` and are not steady.
4. **A drifting leaf** at 0.2 °C min⁻¹ has a fitted slope of 0.20 > 0.05 in every window, so `not_steady_state` with flag `leaf_temp_drifting`.
5. **Photos** (ok) at 10:03:10 (d_min 30.0 mm), 10:03:40 (31.0 mm) and 10:08:00 (30.5 mm):
   - Window 3 (10:03–10:04, centre 10:03:30): two photos inside, so d_min = median(30.0, 31.0) = **30.5 mm**, n_photos = 2, gap = |10:03:40 − 10:03:30| = **10 s**.
   - Window 5 (centre 10:05:30): no photo inside. Nearest is 10:03:40, 110 s away, closer than 10:08:00 (150 s). 110 ≤ 120, so d_min = **31.0 mm**, gap 110 s.
   - Window 0 (centre 10:00:30): nearest ok photo is 160 s away > 120, so `no_photo_in_window`. It is also not steady, so reason = `not_steady_state` and flags include `no_photo_in_window`.
   - leaf_moved: (31.0 − 30.0) / 30.5 = 0.033 ≤ 0.05, so no flag. With photos of 30.0 and 33.0 mm: 3.0 / 31.5 = 0.095 > 0.05, so every window is flagged `leaf_moved`.

## Methods paragraph (draft)
Sensor records (5 s interval) were grouped into consecutive 60 s windows after excluding every record with a quality-control flag. Window means and standard deviations were computed for each variable. A window was considered steady state when three conditions held: (i) the least-squares slope of leaf temperature over the preceding 3 min was below 0.05 °C min⁻¹ in magnitude; (ii) the light-acclimation time, the logged pre-session acclimation plus elapsed time, restarted at any lamp-on event, was at least 30 min; and (iii) a full 3 min of data was available. Windows with fewer than 10 valid records were excluded as missing input. Leaf geometry was extracted from every photograph (Section T3.1) and assigned to windows by timestamp only: the median of photographs taken within the window, otherwise the photograph nearest the window centre within 120 s. A session was flagged when the range of d_min across photographs exceeded 5 % of its median. Every window was kept in the output with a single reason code and a list of all other quality flags. Software: Python 3.13, pandas 3.1.1, NumPy 2.5.4.

## Limitations
1. **The slope threshold has not been checked against real thermal noise.** See the false-alarm table; measure the ROI noise in T7.4/T7.5.
2. **The steady-state test uses leaf temperature only.** A steady leaf under changing air temperature or light still passes. The air and lux SDs are reported so this can be checked.
3. **Acclimation depends on the typed-in minutes and on a `lamp_on` event being logged.** A light change that isn't logged is not detected.
4. **The scale is entered by hand** (`--mm-per-pixel` or scale.csv) until `camera_scale.py` exists, and is only as good as the ruler measurement.
5. **Photos with no sensor-log row have no time** and can't be paired, so phone photos only test segmentation (T3.1 decision).
6. **leaf_moved uses d_min only**, so a shift that keeps the width the same is missed. A single bad-but-"ok" mask can also trip it, so check the overlays.
7. **Windows are cut on the clock**, not on events, so a window can straddle a lamp change (the acclimation test then fails it).
8. **Real photos need the Arducam** (BLOCKED-CAM). Everything here was tested on fabricated logs and drawn images.
