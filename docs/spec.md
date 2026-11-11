# Stomatal Conductance Leaf Sensor: Software Specification and Implementation Plan

*Handoff document for finishing the project’s software. Version of October 30, 2026. Repository: https://github.com/p-anika/stomatal-conductance-leaf-sensor*

## 0. How to use this document

This document assumes you know nothing about the project. Sections 1–3 explain what is being built and where it stands today. Section 4 lists every change needed to make the code logical, easy to use and scientifically rigorous. Sections 5–7 give the target repository layout, coding conventions and a task-by-task implementation plan. Section 8 collects the tasks that cannot be finished until two parts arrive, and says exactly what to do now and what to do when they do. Section 9 lists decisions only the project owner can make (each has a default, so work never has to stop). Section 10 defines “done”. The appendices hold the equations, the configuration file and the data schemas.

Task tags used throughout:

- **\[READY\]** can be done now, on any computer, with no hardware.
- **\[PI\]** needs the Raspberry Pi with the sensors that already work (BME280, TSL2591, MLX90640).
- **\[BLOCKED-CAM\]** needs the CSI-to-HDMI extension cable so the Arducam camera can be connected. Write and unit-test the code now; run it on hardware when the cable arrives.
- **\[BLOCKED-WIND\]** needs the SparkFun FS3000 wind sensor (shipped, in delivery). Same rule.
- **\[OWNER\]** a physical task or a decision only the project owner can make.
- **P0 / P1 / P2**: priority. P0 is required for the core claim, P1 for the machine-learning claim, P2 is optional polish. \[DEFER\] marks work that is worthwhile but must not start until Tier 1 works end to end (a real card\_direct g\_s from a session, checked against ground truth); where a task is marked DEFER, the plan says which minimal version to build meanwhile.

## 1. The project in plain language

**The problem.** Crop breeders must find drought-tolerant varieties. A key measurement is stomatal conductance (g\_s): how open the tiny pores (stomata) on a leaf are, which sets how fast the leaf loses water. The standard tool, a porometer, is expensive and clamps onto the leaf, which disturbs the thin layer of still air the measurement depends on. Satellite and drone thermal imaging is cheap but cannot see individual plant varieties.

**The device.** A roughly $250–350, contact-free sensor head on an adjustable boom, pointed at one leaf from a fixed distance. It reads the leaf’s temperature (thermal camera), air temperature, humidity and pressure, light, and wind, and photographs the leaf. A leaf that loses water cools itself, so an energy balance turns the leaf’s temperature plus the surroundings into g\_s.

**The hard part.** The energy balance needs the boundary-layer resistance r\_b: how strongly the film of still air clinging to the leaf resists heat and vapor flow. It depends on leaf size, shape and wind. Existing low-cost methods use a flat-plate formula. This project adds (1) a dry reference card, which cannot transpire, so its temperature gives r\_b directly by simple algebra; and (2) a small machine-learning model that corrects r\_b for the real leaf’s shape. The project’s specific, narrow novelty is that geometry-aware correction, tested against the flat-plate formula. Method hierarchy: card\_direct (the dry card’s measured r\_b used for the leaf) is the primary method and the basis of all headline results; ml\_corrected is a secondary refinement, reported only if it passes validation; flat\_plate is the baseline both are compared with. Dry-reference thermal g\_s itself is already published (Jones 1999; Leinonen et al. 2006), so the write-up must cite that work.

**The experiment.** Soybean (Glycine max), two cultivars (Houjaku Kuwazu, drought-tolerant; Williams 82, drought-sensitive), two water treatments (well-watered, drought-stressed), 4–5 plants per group. The first cohort is about three months old and already has pods, so it will senesce soon; a second cohort should be started. The target competition is the Plant Sciences category at ISEF (a science fair).

**Hardware and status**

| Part | Role | Status |
| --- | --- | --- |
| Raspberry Pi 4 (2 GB), microSD, STEMMA QT breakout, USB power bank | Base station, runs all code | Working |
| BME280 | Air temperature, humidity, pressure | Wired, mounted, logging |
| TSL2591 | Light (lux, visible and IR channels) | Wired, mounted, logging |
| MLX90640 (32×24 pixels, 55° field of view) | Thermal camera | Wired, mounted, logging |
| Arducam Camera Module 3 (12 MP, autofocus) | Leaf photos for geometry | Delivered, not mounted; waiting for the CSI-to-HDMI extension cable |
| SparkFun FS3000 air-velocity sensor | Wind speed at the leaf | Shipped, in delivery |
| Reference cards (several shapes) | Dry reference | In progress |
| Measurement lamp (LED flood light) | Constant light during measurements | Not bought yet (recommended) |

## 2. Final goal of the code

When finished, one person with the device should be able to do the following.

1. **In the field (on the Pi):** start a session with a few prompts or command-line flags; the logger records sensors, raw thermal frames and photos, labeled with plant, genotype, treatment and conditions.
2. **Back on a laptop:** run one command to get g\_s for every measurement window, computed three ways (primary: direct card value; secondary: ML-corrected; baseline: flat-plate), each with an uncertainty range and a validity flag; get genotype × treatment statistics; and get every figure and table needed for the ISEF board and research paper.
3. **Evidence of accuracy:** scripts that compare the device against independent references (thermometer calibration, a borrowed porometer or weighed-transpiration g\_s, petroleum-jelly-coated leaves for r\_b) and report bias, error, and the lowest g\_s the device can resolve.

Pipeline in one line: session metadata → logger (sensors, thermal frames, photos) → raw data folder → quality control → leaf/card geometry and temperatures → averaged measurement windows → card r\_b (physics) → training set → ML correction → leaf g\_s (three methods) with uncertainty → validation → statistics → figures.

**Claims the software must support**

- **Tier 1 (core):** a calibrated, validated, low-cost contact-free sensor (using the primary card\_direct method) that detects drought responses and genotype differences, with stated error bars and detection limit.
- **Tier 2 (novelty):** the geometry-aware r\_b correction (the secondary method) measurably beats the flat-plate baseline on real leaves and, for atypically shaped leaves, improves on card\_direct. If it does not, the software must report that honestly.

**Design principles**

- Physics, ML and analysis code never imports hardware libraries, so everything but the logger runs on any laptop.
- Raw data is never edited or deleted.
- Every result carries its method, a valid flag, a reason code and an uncertainty.
- No silent defaults: a missing wind speed, light calibration or card value produces a flagged result, never a quiet guess.
- Every constant lives in one config file with its source cited.
- Everything is reproducible from raw data with documented commands.

* Data are never excluded because of their measured value, only because of failed inputs or quality checks. Values below the detection limit are censored and kept.

## 3. Current state of the repository

| Path | State |
| --- | --- |
| hardware\_tests/test\_bme280.py, test\_tsl2591.py, test\_mlx90640.py | Done (one-off sensor checks) |
| acquisition/session\_metadata.py | Works; needs more fields and validation |
| acquisition/combined\_logger.py | Works for BME280, TSL2591, MLX90640 every 5 s; logs only min/max/mean of the whole thermal frame; no photos, no wind, no card/leaf separation |
| processing/energy\_balance\_utils.py, physics\_baseline.py, gs\_calculator.py | Implemented and tested with synthetic numbers, but several constants and assumptions must change (Section 4, group A) |
| ml/, analysis/, docs/ | Empty (.gitkeep only) |
| requirements.txt | Partial; optional packages commented out |
| config.yaml, .gitignore, tests/ | Do not exist |
| data/raw/debug\_logs/ | One early debug CSV; never use it as real data (verify it is not sitting directly in data/raw/) |

Code is edited on a laptop and on the Pi and synced through GitHub, so any module not touching sensors must run on a laptop.

## 4. Master list of changes

IDs are referenced by the implementation plan.

### A. Physics and science corrections (processing/)

- **A1 Absorptivity.** Do not use the 0.23 albedo (it is a FAO-56 value for a whole grass canopy, not a single leaf). Leaf shortwave absorptivity a\_s comes from config per light source (defaults: sunlight or window 0.5; visible-only LED lamp 0.85; both to be refined). The reference card gets its own a\_s\_card (matte black ≈ 0.95; dark-green matte is closer to a leaf). The current code wrongly gives the card and leaf one shared value (0.5).
- **A2 Specific heat of air** cₚ = 1005 J kg⁻¹ K⁻¹ (code has 1010), so ρcₚ = 1210 J m⁻³ K⁻¹ as in the notes.
- **A3 Psychrometric constant** computed from the logged pressure: γ \[Pa/K\] = 0.0665 × P \[hPa\] (the FAO-56 form 0.665×10⁻³ × P in kPa; the handwritten notes mistyped it as 0.00062 on one page). Add pressure\_hpa to the leaf reading. Remove the fixed 66.5.
- **A4 Longwave radiation.** Keep the net longwave term but use an explicit surroundings temperature T\_sur (default: air temperature; optionally measured by pointing the thermal camera at a wall or window). Window glass is opaque to thermal infrared, so a window counts as a surface at its own, usually colder, temperature.
- **A5 Sidedness.** Pick one convention and apply it everywhere. Default: both leaf faces, per unit projected area (soybean has stomata on both surfaces). That means longwave doubles, r\_b uses the two-sided flat-plate coefficient, and g\_s is the total conductance of both surfaces. Config key `faces`.
- **A6 Vapor-to-heat boundary-layer ratio** r\_bw = ratio × r\_bH becomes an explicit config value (default 0.93; verify in Campbell & Norman 1998; effect on g\_s about 2%).
- **A7 Light conversion.** Replace the hard-coded lux ÷ 126 with a calibrated factor k (W m⁻² per lux) per light source, stored in the calibration file. 1/120 is only a reference for outdoor sunlight; window glass, clouds and LED spectra change it. An LED lamp must never fall back to the sunlight value: luminous efficacy differs from sunlight by roughly 2–3× (the sunlight factor overestimates irradiance for white LEDs and underestimates it for red-blue grow lights), so the code raises an error instead. Log the TSL2591 visible and infrared channels and gain.
- **A8 Unit conversion** from m/s to mol m⁻² s⁻¹: g\_mol = g\_ms × P / (R × T\_air), about 41 mol m⁻³ at room conditions.
- **A9 Method labeling.** Rename r\_bh\_corrected to r\_bh and add r\_bh\_method ∈ {flat\_plate, card\_direct, ml\_corrected}. All three feed the same g\_s solver, so methods can be compared directly. Roles (config methods): card\_direct is the primary method and the basis of all headline results (it needs neither the wind sensor nor the camera); ml\_corrected is secondary and reported only if it passes validation; flat\_plate is the baseline.
- **A10 Flat-plate baseline function** built from a Nusselt-number correlation instead of a bare coefficient. Forced convection: Nu\_forced = b × 0.66 × Pr^(1/3) × Re^(1/2), with Re = U d / ν and Pr = ν / D\_H; conductance g\_bH = s × D\_H × Nu / d and r\_bH = 1 / g\_bH, with d in m, U in m/s and s the sidedness factor. The common shortcut r\_bH = C √(d/U) has C in s^0.5 m⁻¹, not s m⁻¹, because √(d/U) is in s^0.5. Worked out at about 20 °C (ν ≈ 1.5×10⁻⁵ m² s⁻¹, D\_H ≈ 2.1×10⁻⁵ m² s⁻¹, Pr ≈ 0.71): C = √ν / (s × b × 0.66 × D\_H × Pr^(1/3)) ≈ 313 / (s × b). So C ≈ 156 for two faces with no enhancement (s = 2, b = 1), ≈ 104 for two faces with b = 1.5, ≈ 208 for one face with b = 1.5 and ≈ 313 for one face with b = 1. An enhancement factor raises conductance and therefore lowers C. The enhancement factor of 1.5 appears in Bonan’s textbook code. These values are unverified until checked against the books, and C must never be copied as a bare number. Sources disagree on the enhancement factor (1.4 to 1.5), on whether it already represents both faces, and on how to define the leaf’s characteristic width (one source uses 0.72 of the leaf width), so check all three in Campbell & Norman (1998), cite them, and do not count sidedness twice. Mixing a one-sided r\_b (s = 1) with two-sided longwave and g\_s (faces = 2) would put r\_b about 2× off, so s is always taken from physics.faces, and the enhancement factor defaults to 1.0 until verified. A baseline scaled wrongly would also make the ML correction look better than it is, so the baseline must follow the published convention exactly. The choice of characteristic dimension d (short side of the minimum bounding rectangle, equivalent diameter, or extent along the flow) is config-driven and must be justified in the write-up.
- **A11 Free and mixed convection.** The forced formula gives infinite r\_b as U → 0, and indoors with the fan off the wind is often at or below the FS3000’s noise floor, where buoyancy-driven (free) convection sets r\_b. Add a free-convection term, Nu\_free = 0.54 × Pr^(1/4) × Gr^(1/4) with Gr = g β |T\_leaf − T\_air| d³ / ν² (β = 1/T\_air in kelvin), and add it to the forced Nusselt number (Nu = Nu\_forced + Nu\_free, the combination in Bonan’s textbook code; config convection.combination can switch to larger-of; verify which Campbell & Norman recommend). Wind readings below wind\_noise\_floor\_ms count as zero forced flow instead of being replaced by a floor value. Label every window forced, mixed or free by the ratio of free to forced conductance (mixed\_convection\_ratio, default 0.3: forced if below it, free if above its inverse, mixed in between). Mixed and free windows get the flag baseline\_unreliable, and the benchmark judges the flat-plate method on forced windows only.
- **A12 Validity and detection limit.** Return reason codes instead of None. Add a Monte Carlo uncertainty module that propagates sensor errors (leaf temperature, air temperature, humidity, light, r\_b) into a g\_s interval, and use it to set a per-window detection limit. Readings below that limit are kept and labeled as censored (A15), never dropped. This matters because a 0.5 °C leaf-temperature error alone changes g\_s by roughly −21% to +27% in a typical case.
- **A13 Jones conductance index (Plan B).** I\_g = (T\_dry − T\_leaf) / (T\_leaf − T\_wet), computed from in-frame dry and wet reference temperatures. It needs no light calibration and cancels camera offsets, so it still supports genotype and treatment comparisons if absolute g\_s disappoints. Also compute the bounded index CWSI = (T\_leaf − T\_wet) / (T\_dry − T\_wet), which stays finite when stomata close. Both need a wet reference (damp filter paper or felt) in the thermal frame.
- **A14 Wind is never silently defaulted.** Missing wind produces reason code no\_wind; a manual wind value is allowed only with an explicit flag recorded in the output.

* **A15 Censoring instead of rejection.** A leaf with nearly closed stomata has a true latent heat flux near zero, so sensor noise makes the computed λE hover around zero and sometimes go negative. Rejecting those windows would delete exactly the drought-stressed data the project needs and would bias the stressed-group mean upward, shrinking the apparent drought effect. So λE ≤ 0 is not an error. The window is kept and labeled censor = below\_detection (g\_s is at or below the detection limit for that window’s conditions); r\_s ≤ 0 is labeled above\_range. The Monte Carlo module counts non-positive-λE draws as g\_s = 0 instead of dropping them. Every analysis reports the share of censored windows per genotype × treatment and uses censoring-aware statistics (E7). Bounded metrics that never truncate (the Jones index and CWSI, A13) are analyzed alongside g\_s.
* **A16 Spatial equivalence of card and leaf.** The card’s r\_b stands in for the leaf’s only if both receive the same irradiance and airflow. A lamp has hot spots and edge falloff, and a backdrop reflects differently onto the card than onto the leaf. Because r\_bH = ρcₚ ΔT / Rn, a 5–10% error in the card’s Rn is a 5–10% error in its r\_b: modest for one reading, but systematic across every training label and comparable to the shape effect the ML must learn. Therefore: measure irradiance at the card, the leaf and the light sensor with the TSL2591 under the real lamp and store position factors c\_card and c\_leaf (T7.8); use E\_s = c\_position × k × lux; hold the card at the same tilt and distance as the leaf, in front of the same neutral matte backdrop, with the lamp in a fixed jig; log card\_angle\_deg, lamp angles, distances and background; carry the remaining Rn uncertainty into the labels (sigma\_rn\_card\_fraction); verify with a position-swap check (T5.8, deferred; the light-field map in T7.8 comes first); and map wind at both positions once the FS3000 arrives.

- **A17 Card-to-leaf buoyancy transfer.** card\_direct assumes the card’s r\_b equals the leaf’s. In forced flow that is reasonable once shape is accounted for, but at low wind buoyancy matters: free convection grows with |T\_surface − T\_air| (Nu\_free ∝ Gr^(1/4)). A card heated several °C above air has a stronger buoyant flow, so a lower r\_b, than a transpiring leaf near air temperature, so card\_direct would underestimate the leaf’s r\_b at low wind. Mitigations: (1) every window is labeled forced, mixed or free (A11); card\_direct windows in mixed or free regimes get flag card\_transfer\_unreliable, and windows with no wind measurement get regime\_unknown; (2) default protocol: run plant and card sessions with the fan on low so flow is forced (decision 21); (3) later option: solve the card’s r\_b, split it into forced and free parts with the Nusselt model, and recompute the free part for the leaf’s own |ΔT|; (4) test it with card\_buoyancy\_check (T5.9): at fixed wind and card, vary irradiance so ΔT varies, and examine solved r\_b against ΔT. A petroleum-jelly leaf cannot test this by itself because a coated leaf warms like the card; it tests shape transfer, not temperature-difference transfer.

### B. Data acquisition (acquisition/)

- **B1 Sensor abstraction.** One class per sensor with real and simulated versions and a null version when disabled in config, so the logger runs on a laptop and with missing parts.
- **B2 Region-of-interest temperatures.** Report leaf, card and wet-reference temperatures from their own pixel regions (mean, median, standard deviation, pixel count), not the whole-frame mean. Start with manual rectangles; later derive the leaf region from the photo mask.
- **B3 Raw thermal frames** (24×32 values) saved per sample so everything can be reprocessed, appended to one binary file per session rather than one file per frame (B13).
- **B4 Target labeling.** Card and leaf are measured in the same frame via separate regions; event markers (for example lamp on, measurement start) are written to an events file.
- **B5 Averaging and steady-state gating.** Measurements are averages over a window; windows are flagged if leaf temperature is still drifting or the light-acclimation time (log it) was too short.
- **B6 TSL2591 gain handling.** Catch the overflow error under bright lamps, step gain down automatically, and log gain and integration time.
- **B7 Robustness.** Retry failed reads, handle Ctrl+C cleanly, use timezone-aware timestamps, handle an unsynchronized clock (B14), and log undervoltage flags.
- **B8 Command-line arguments** with interactive prompts as a fallback.
- **B9 Run manifest** per session: git commit, config snapshot, calibration file version, library versions, sensor settings.
- **B10 Camera capture** (Arducam, locked focus at working distance, timestamped files). \[BLOCKED-CAM\]
- **B11 Wind logging** (FS3000) and fan state. \[BLOCKED-WIND\]
- **B12 Quality-control flags** for anomalies already seen in the first log: lux dropping from about 88 to about 2 for two samples, and one thermal frame near 37 °C (likely an object in front of the camera). Flag, never delete.

* **B13 Thermal frame storage.** Append each 24×32 frame as 768 little-endian float32 values (3,072 bytes) to one file per session, thermal\_frames.f32; record the frame number in the log row (thermal\_frame\_idx); flush and fsync every 12 frames (a default; tuning the interval, power-loss testing and the USB-drive option are DEFER until Tier 1 works); read back with numpy.fromfile. The write rate is only about 0.6 kB/s, so SD-card bandwidth is not the concern. The real risks are hundreds of tiny files per session and corruption when a USB power bank cuts power mid-write. Also use a quality A1/A2-rated card, copy each day’s sessions off the Pi, and optionally write sessions to a USB flash drive.
* **B14 Clock integrity \[DEFER the typed-offset and RTC parts until Tier 1 works; minimum now: record NTP status in the manifest and refuse to start on a clearly wrong time\].** The Pi 4 has no battery clock. Raspberry Pi OS usually restores its last saved time at boot (a stale time rather than 1970; verify on your Pi), but with no network the clock is wrong by however long the Pi was off or unsynchronized. At session start the logger checks NTP synchronization. If unsynchronized it asks the user to type the current date and time from a trusted source such as a phone, stores clock\_offset\_s (typed time minus system time) in run\_manifest.json, writes a corrected timestamp (timestamp\_corrected\_iso) in every row, and records clock\_status (ntp\_synced, rtc, manual\_offset or unsynced). It refuses to start, unless --allow-unsynced-clock is given, if the system time is earlier than the last session in session\_metadata.csv or before a fixed minimum date. Optional hardware fix: an I2C real-time-clock module (DS3231 or PCF8523, roughly $5–15, address 0x68, no clash with the current sensors), enabled in the Pi boot config (decision 15 in Section 9). Offline demonstrations, such as at a fair, are the main case where this matters.
* **B15 Photo-to-window pairing.** Sensor rows arrive every 5 s, measurement windows last 60 s (12 rows) and photos arrive every 60 s. Leaf geometry changes slowly, so features are extracted from every photo (processing/extract\_photo\_features.py, T3.5) and attached to each window by timestamp, never by row index: the median of the photos inside the window, else the nearest photo within max\_photo\_gap\_sec (default 120 s). A window with no photo in range gets reason no\_photo\_in\_window. If d\_min varies by more than 5% between photos in a session, flag leaf\_moved (the leaf shifted or curled, so the thermal region and geometry may be stale).

### C. Metadata and experimental design (session\_metadata.py)

- **C1 Structured fields** (not free text): reproductive\_stage, fan\_state, light\_source, lamp\_distance\_mm, window\_blocked, acclimation\_minutes, block, position, leaf\_position, card\_id, card\_angle\_deg, lamp\_angle\_deg, lamp\_to\_card\_mm, background, dry\_reference\_type, wet\_reference\_present, operator, optional porometer\_gs.
- **C2 Validated choices** so a typo cannot create a new category (genotype, treatment, light source, fan state).
- **C3 Randomization record:** plants are arranged in randomized blocks and rotated, because a window creates a light gradient.
- **C4 Reference-card manifest** (data/reference\_cards.csv): shape, material, coating, measured absorptivity, geometry.

### D. Machine learning (ml/)

- **D1** Predict the ratio f = r\_b,measured / r\_b,flat-plate rather than r\_b itself, so f = 1 is the built-in baseline.
- **D2** Evaluate by leave-one-shape-out cross-validation, never random row splits (readings of the same card are not independent).
- **D3** Compare a ridge regression and a small tree ensemble against the null model (f = 1); give prediction intervals by bootstrap; save a model card (features, metrics, data hash, library versions). If the model does not beat the null by a set margin, predict f = 1 and flag it.
- **D4** Needs 15–20 or more distinct card shapes, including shapes traced from real soybean leaflets, plus petroleum-jelly-coated real leaves as an independent test set. \[OWNER\] for the physical parts.
- **D5** Synthetic data generator with a known true correction so the plumbing can be tested before real data exists.
- **D6** No physics-informed neural network and no two-wind-sensor formula (both were dropped from the design).

### E. Validation and analysis (analysis/, calibration/)

- **E1** Thermal camera and BME280 calibration against a reference thermometer (ice bath, room, warm water), giving offsets and error estimates.
- **E2** Light calibration: fit k per light source against a borrowed reference light meter.
- **E3** r\_b validation with petroleum-jelly-coated real leaves (a non-transpiring leaf of real shape).
- **E4** g\_s validation against a porometer or weighed transpiration, both on a whole-leaf basis (soybean has stomata on both faces and most porometers read only one, so measure the upper and lower faces separately within a few minutes, alternating which face is read first, and add them: g\_total = g\_adaxial + g\_abaxial; weighed transpiration is whole-leaf by nature; never compare a one-sided reading with the device’s two-sided value; record which surface each reading came from): bias, error, Bland–Altman plot.
- **E5** Detection-limit analysis, bracketed by a petroleum-jelly leaf (g\_s ≈ 0) and a wet reference (very high g\_s).
- **E6** Method comparison (flat plate, card direct, ML), split by how much each leaf’s shape deviates from typical.
- **E7** Statistics: the plant is the experimental unit; mixed-effects model (genotype × treatment fixed, plant random), plus an ANOVA on plant means as a check; below-detection windows are kept and handled with censoring-aware methods (A15); effect sizes with confidence intervals; the headline claim is relative differences.
- **E8** One command regenerates every figure and table.

* **E9** Spatial and airflow equivalence of card and leaf: light-field mapping and, later, a card position-swap check (deferred; A16).

- **E10** Card-to-leaf buoyancy transfer: solved r\_b versus surface-to-air temperature difference at fixed wind and shape (T5.9, A17).

### F. Software engineering

- **F1** Automated tests (pytest) for every module, including a round-trip test: simulate a leaf with known g\_s, compute its temperature from the energy balance, then confirm the solver recovers that g\_s; and a truncation test showing that dropping non-positive latent-heat readings biases stressed-leaf means upward.
- **F2** config.yaml for constants and settings; calibration.json for measured calibrations.
- **F3** Shared data-loading code with schema checks.
- **F4** Python logging instead of print.
- **F5** .gitignore, pinned requirements, README, and docs (equations, assumptions and limitations, measurement protocol, Pi setup, design notes).
- **F6** Move the early debug CSV into data/raw/debug\_logs/ and make analysis loaders ignore folders without a run manifest.

## 5. Target repository layout

```
stomatal-conductance-leaf-sensor/
  README.md                        update (run commands, status)
  requirements.txt                 laptop and analysis packages, pinned
  requirements-pi.txt              Pi-only hardware libraries
  config.yaml                      NEW: every constant and setting (Appendix B)
  .gitignore                       NEW
  calibration/
    calibration.json               generated: offsets, light factor k, position factors, card absorptivity, ROIs
    define_thermal_rois.py         [PI]
    thermal_temperature_cal.py     [PI] plus a reference thermometer
    light_calibration.py           [PI] plus a reference light meter
    light_field_map.py             [PI] lamp uniformity; card and leaf position factors
    wind_check.py                  [BLOCKED-WIND]
    focus_sweep.py                 [BLOCKED-CAM]
    camera_scale.py                [BLOCKED-CAM]
    thermal_rgb_alignment.py       [BLOCKED-CAM]
  hardware_tests/
    test_bme280.py, test_tsl2591.py, test_mlx90640.py    done
    test_fs3000.py                 [BLOCKED-WIND]
    test_arducam.py                [BLOCKED-CAM]
  acquisition/
    sensors.py                     NEW: real, simulated and null sensor classes
    session_metadata.py            extend
    combined_logger.py             rewrite
    mark_event.py                  NEW
    camera_capture.py              [BLOCKED-CAM]
  processing/
    config.py  data_io.py  qc.py  measurement_windows.py  thermal_roi.py
    leaf_segmentation.py  extract_photo_features.py  energy_balance_utils.py
    physics_baseline.py  gs_calculator.py  uncertainty.py  jones_index.py
    pipeline_dry_run.py
  ml/
    synthetic_data.py  build_training_set.py
    train_correction_model.py  predict_correction.py
  analysis/
    aggregate_results.py  baseline_benchmark.py  genotype_comparison.py
    detection_limit.py  validate_rb_vaseline.py  position_swap_check.py
    validate_gs_ground_truth.py  make_figures.py
  tests/                           one test file per module, plus conftest.py
  docs/                            equations, assumptions, protocol, Pi setup, design notes
  data/
    reference_cards.csv
    validation/ground_truth_gs.csv           typed in by hand from porometer or scale
    raw/session_metadata.csv
    raw/debug_logs/                          never used in analysis
    raw/<session_id>/                        sensor_log.csv, events.csv, run_manifest.json,
                                             thermal_frames.f32 (one file per session),
                                             images/, logger.log
    processed/                               rb_training_set.csv, leaf_gs_results.csv,
                                             <session_id>/photo_features.csv, ...
    models/                                  correction_model.joblib and its model card
```

## 6. Conventions every file must follow

- **Units.** Temperatures in °C (convert to kelvin only inside radiation terms). Pressure in hPa in files, Pa inside vapor-pressure math. Resistances in s m⁻¹. Conductance in m s⁻¹ and mol m⁻² s⁻¹. Fluxes in W m⁻². Lengths in mm in data files, m inside formulas. Wind in m s⁻¹. Light in lux in raw logs, W m⁻² only after calibration.
- **Column names** are snake\_case with a unit suffix: leaf\_temp\_c, wind\_ms, pressure\_hpa, lux.
- **Hardware isolation.** Only acquisition/sensors.py, acquisition/camera\_capture.py, hardware\_tests/ and calibration scripts marked \[PI\] may import board, adafruit\_\*, or picamera2, and only inside functions or try/except, so every other import works on a laptop.
- **Config over constants.** Physical constants and settings come from processing/config.py. No numeric literals for physical constants inside functions.
- **Result objects.** Frozen dataclasses with valid, reason and flags. Bad data returns an invalid result; it never raises. Exceptions are for programmer errors only (missing config key, malformed file).
- **Reason codes (fixed vocabulary):** ok, missing\_input, no\_wind, rn\_near\_zero, low\_card\_signal, suspect\_card\_rbh, roi\_too\_small, not\_steady\_state, sensor\_overflow, qc\_spike, segmentation\_failed, no\_photo\_in\_window, clock\_unsynced, references\_not\_bracketing (Jones index only: the dry and wet references do not bracket the leaf temperature, or are too close together). Censoring is not a failure: every g\_s result also carries censor = none, below\_detection or above\_range, and censored results stay in every table.
- **Flags (non-fatal):** k\_uncalibrated, a\_s\_default, c\_position\_default (a leaf or card irradiance factor is not yet measured, so 1.0 was used), vpd\_nonpositive (vapor pressure deficit is zero or negative while latent heat flux is positive; the result is censored above\_range and kept, never dropped), low\_light, marginal, baseline\_unreliable (mixed or free convection), card\_transfer\_unreliable (card\_direct under mixed or free convection, A17), regime\_unknown (no wind measurement), foreshortening\_corrected (leaf\_segmentation divided the photo area by cos of the incidence angle; set whenever incidence\_angle\_deg is given, including 0, T3.1), leaf\_moved ((max − min)/median of d\_min over a session’s photos exceeds windows.leaf\_moved\_range\_frac, T3.4), stability\_window\_incomplete (less than stability\_minutes of data before the window end, T3.4), leaf\_temp\_drifting (leaf-temperature slope at or above stability\_slope\_c\_per\_min, T3.4), acclimation\_too\_short (acclimation time below min\_acclimation\_minutes, T3.4), card\_temp\_drifting (card-temperature slope at or above stability\_slope\_c\_per\_min in card windows, T3.4/T4.3), card\_angle\_default (card\_angle\_deg was blank, so 0 was used, T4.3), ml\_no\_improvement (the correction model did not beat f = 1, so f = 1 was returned, T4.5), outside\_training\_range (a predicted feature lies outside the training range, T4.5), ml\_model\_missing (no correction model file, so ml\_corrected is invalid with reason missing\_input, T5.1), jones\_no\_leaf\_dry\_reference (a wet reference is present but the dry reference is not a vaseline\_leaf, so I\_g and CWSI are blank, T5.1), dl\_not\_resolved (detection limit blank: no point on the grid is resolved at this window's conditions, T5.1), dl\_search\_failed (detection limit blank: the forward root search failed or errored at one or more grid points, T5.1).
- **Raw data is append-only.** Processing writes only to data/processed/ and data/models/.
- **Logging** with the logging module (console plus data/raw/\<session\_id>/logger.log), not print.
- **Style.** Type hints; each docstring states the equation and units; syntax compatible with the target Python version recorded in CLAUDE.md (the lower of the laptop and Raspberry Pi versions).
- **Command line.** Every script runs as python -m package.module from the repository root and supports --help. Hardware scripts print a clear message and exit if hardware libraries are missing.
- **Tests.** pytest -q must pass on a laptop with no hardware before any merge.
- **Git.** Small commits that say what and why. Raw images and thermal frames are not committed (back them up separately); summary CSVs may be.

## 7. Implementation plan

Work the phases in order. Tasks within a phase can usually be done in any order unless a dependency is stated. Every task ends with a “done when” test.

### Phase 0: Repository hygiene \[READY\]

- **T0.1 .gitignore (P0).** Ignore `__pycache__/`, `*.pyc`, `.venv/`, `.pytest_cache/`, `.DS_Store`, `data/raw/*/images/` and `data/raw/*/thermal_frames.f32`. Done when git status is clean after running tests.
- **T0.2 config.yaml (P0).** Create it from Appendix B. Done when processing/config.py loads it in a test.
- **T0.3 Requirements (P0).** requirements.txt: numpy, scipy, pandas, pyyaml, scikit-learn, joblib, opencv-python-headless, matplotlib, statsmodels, pytest. requirements-pi.txt: adafruit-circuitpython-bme280, adafruit-circuitpython-tsl2591, adafruit-circuitpython-mlx90640, adafruit-blinka, numpy, pyyaml (FS3000 library added later). On the Pi, picamera2 comes from apt, not pip. Pin versions after the first clean install. Done when a fresh virtual environment installs and pytest runs.
- **T0.4 Tests scaffold (P0).** tests/conftest.py with fixtures (default config, synthetic thermal frame, fake session folder). Done when pytest runs one trivial test.
- **T0.5 Debug log (P0).** Confirm the early debug CSV is in data/raw/debug\_logs/ and nothing stray sits in data/raw/.
- **T0.6 Docs skeleton (P1).** docs/equations.md (copy of Appendix A), assumptions\_and\_limitations.md (a numbered register of every assumption with its equation, the owner’s decision and the planned test, starting with A1 to A17; each coding task adds to it), measurement\_protocol.md, pi\_setup.md, design\_notes.md; README with run commands and a status table.

### Phase 1: Core physics refactor \[READY, P0\]

Implements A1–A14 and F1. Key signatures (all pure functions, no hardware):

```
saturation_vapor_pressure_pa(temp_c) -> float          # Tetens: 610.78*exp(17.27*T/(T+237.3))
psychrometric_constant_pa(pressure_hpa) -> float       # 0.0665 * P_hPa
molar_density_mol_m3(pressure_hpa, air_temp_c) -> float   # P/(R*T)
lux_to_irradiance_wm2(lux, k) -> float
net_radiation_wm2(irradiance_wm2, a_s, surface_temp_c, surroundings_temp_c,
                  emissivity, faces, include_longwave) -> float
flat_plate_rbh(d_m, wind_ms, delta_t_c, air_temp_c, cfg) -> (r_bh, regime)
solve_card_rbh(reading, cfg, cal) -> CardResult
solve_stomatal_conductance(reading, cfg, cal) -> ConductanceResult
```

- **T1.1 processing/config.py.** load\_config(path) and load\_calibration(path) (empty defaults with a warning if the file is absent). get\_light\_source(cfg, cal, name) returns a\_s\_leaf, k and k\_source. If k is still null it raises CalibrationMissing with a message naming light\_calibration.py. This is never overridden for led\_lamp, or for any source not listed in config, because LED luminous efficacy differs from sunlight by roughly 2–3×, in either direction depending on the lamp’s spectrum. For window and sunlight only, physics.allow\_uncalibrated\_k = true permits the sunlight reference value, adds flag k\_uncalibrated and the larger sigma\_lux\_fraction\_uncalibrated to every result, and the analysis scripts refuse such rows unless run with --include-uncalibrated for exploration. Done when tests cover all three paths: calibrated, window or sunlight fallback, and LED refusal.
- **T1.2 energy\_balance\_utils.py.** Rewrite with the signatures above. Net radiation: Rn = a\_s × E\_s + faces × ε × σ × (T\_sur⁴ − T\_s⁴), temperatures in kelvin, longwave omitted if include\_longwave is false. Flat plate: Nusselt-number form of A10 and A11 (forced plus free convection), with U set to zero below wind\_noise\_floor\_ms; returns r\_bH and the regime label (forced, mixed or free). Constants come from config. Irradiance at a surface is E\_s = c\_position × k × lux, where c\_position (card or leaf factor) comes from calibration.json and defaults to 1.0 with a flag until measured (A16). The module’s explainer must derive C from the Nusselt form and print a table of C for each (s, b) pair; never paste a coefficient from memory or from this spec. s is read from physics.faces, never set separately. Done when outputs match the worked example in T1.7.
- **T1.3 physics\_baseline.py.** ReferenceCardReading(card\_id, card\_temp\_c, air\_temp\_c, surroundings\_temp\_c, lux, light\_source, c\_card), where c\_card is the card position’s irradiance factor (A16). solve\_card\_rbh returns CardResult(r\_bh, rn\_wm2, valid, reason, flags). Equation: r\_bH = ρcₚ (T\_card − T\_air) / Rn, using the card’s own absorptivity a\_s\_card. Gate on exogenous conditions, not on the noisy outcome: a reading is unusable if the absorbed shortwave a\_s\_card × E\_s,card is below min\_card\_absorbed\_sw\_wm2 (low\_card\_signal; default 40 W m⁻², about 2 °C of expected card-minus-air difference at r\_b = 70 s m⁻¹, only about twice the placeholder temperature sigma of 1 °C. Treat 40 W m⁻² as a placeholder: after thermal calibration (T7.5), set the threshold so the expected signal is at least about 5 times the calibrated noise of a window mean) or if |Rn| < 1 W m⁻² (rn\_near\_zero). Selecting by the measured temperature difference or by the sign of r\_bH would bias the training labels upward, so a window-mean r\_bH ≤ 0 is kept in the rejects file as suspect\_card\_rbh and counted; a rising count points to a temperature-calibration problem, not noise. Solve on window means, never on single 5-second samples. Done when each gate is tested and the training-set report prints the suspect\_card\_rbh count.
- **T1.4 gs\_calculator.py.** LeafReading(leaf\_temp\_c, air\_temp\_c, surroundings\_temp\_c, relative\_humidity\_pct, pressure\_hpa, lux, light\_source, r\_bh, r\_bh\_method). ConductanceResult(r\_s, g\_s\_ms, g\_s\_mol, rn\_wm2, h\_wm2, latent\_wm2, vpd\_pa, valid, reason, censor, g\_s\_mol\_dl, flags, g\_s\_lo, g\_s\_hi). Here valid means the inputs were usable; censor is none, below\_detection or above\_range; g\_s\_mol\_dl is the detection limit for these conditions. Censored results are valid and are never discarded (A15). Steps: Rn; H = ρcₚ(T\_s − T\_a)/r\_bH; λE = Rn − H (if ≤ 0, do not reject: return censor = below\_detection, meaning the stomata are closed to within the detection limit, with g\_s\_mol\_dl as the upper bound); r\_s + r\_bw = ρcₚ × VPD / (γ × λE) with r\_bw = ratio × r\_bH; r\_s = that − r\_bw (if ≤ 0, return censor = above\_range, not an error); g\_s = 1/r\_s; convert to mol. Done when the round-trip test passes and censored cases come back labeled instead of dropped.
- **T1.5 uncertainty.py.** gs\_uncertainty(reading, sigmas, n=2000, seed=0) draws Gaussian errors for leaf temperature, air temperature, humidity, light (fraction) and r\_b (fraction), re-solves each draw, and returns the median, 2.5th and 97.5th percentiles, the fraction of draws with non-positive latent heat flux, and the detection limit g\_s\_dl. Draws with λE ≤ 0 are counted as g\_s = 0 (closed stomata) and draws with r\_s ≤ 0 are capped at the top of the measurable range; none are dropped, so the interval and its upper bound include the closed-stomata case. g\_s\_dl is the smallest g\_s resolved in at least 90% of draws with a relative interval width under the config limit. A resolved label goes in the table; it is never a filter. A second helper propagates sigma\_rn\_card\_fraction into card r\_b labels. Sigmas come from calibration.json, with the config placeholders until then. Make leaf and air temperature errors optionally correlated, since both are calibrated against one thermometer. Done when the ±0.5 °C test in T1.7 matches.
- **T1.6 jones\_index.py.** conductance\_index(t\_dry, t\_leaf, t\_wet) returns I\_g = (T\_dry − T\_leaf) / (T\_leaf − T\_wet) and the bounded index cwsi = (T\_leaf − T\_wet) / (T\_dry − T\_wet), each with valid, reason and censor. Noise can push values slightly outside the expected range (I\_g below 0, CWSI outside 0–1); keep them as measured, because clipping would bias stressed leaves. I\_g is labeled above\_range when T\_leaf − T\_wet is under 0.2 °C; CWSI is invalid only if T\_dry − T\_wet is under 1 °C, because then the references do not bracket the leaf. Done when tested on hand-computed values.
- **T1.7 tests (tests/test\_energy\_balance.py, test\_physics\_roundtrip.py, test\_uncertainty.py).**
  1. **Worked example.** Leaf 23.0 °C, air 24.0 °C, surroundings 24.0 °C, RH 50%, 995 hPa, incident shortwave 120 W m⁻², a\_s = 0.85, ε = 0.97, faces = 1 (set explicitly in the test, because the config default is 2), r\_bH = 70 s m⁻¹, ratio 0.93. Expected: Rn ≈ 107.7, H ≈ −17.3, λE ≈ 125.0 W m⁻², VPD ≈ 1317 Pa, r\_s ≈ 127.6 s m⁻¹, g\_s ≈ 0.00784 m s⁻¹ ≈ 0.316 mol m⁻² s⁻¹ (tolerance 1%).
  2. **Round trip.** For g\_s from 0.02 to 0.6 mol m⁻² s⁻¹ and several r\_b, wind and light values, find T\_s by root finding (scipy brentq) on the energy balance, run the solver, recover g\_s within 0.1%.
  3. **Sensitivity.** In the worked example, a −0.5 °C or +0.5 °C leaf-temperature error changes g\_s by about +27% and −21% (tolerance ±3 points).
  4. **Card round trip.** Choose r\_b, light and air temperature, solve for T\_card, recover r\_b.
  5. **Edge cases.** Each reason code and both censor labels are produced by a crafted input; non-positive λE returns below\_detection and is never rejected; missing k raises CalibrationMissing. Truncation test: simulate 200 stressed windows with true g\_s near zero and noisy temperatures; the mean of the non-censored windows alone must come out biased high, while the pooled flux-space estimate (processing/uncertainty.py: pooled\_gs\_mol, which averages latent heat flux and VPD over replicate windows that share one true g\_s and one set of conditions, then solves once) recovers the true value within 10% on every one of 20 seeds. Conditions fixed in the test: true g\_s 0.05 mol m-2 s-1, leaf temperature sigma 0.5 C, air temperature sigma 0.25 C, 200 windows per seed. A second test at 0.02 mol (about 5% of windows censored) checks only that censored windows occur and that the truncated mean is high. The 10% target holds only when thermal sensors are calibrated to about this accuracy (the explainer tabulates error at 0.25, 0.5 and 1.0 C). Pooling averages out random noise only; a calibration offset shared by all windows does not cancel. T5.1 and T5.3 should pool windows within a plant in flux space before converting to g\_s (decision A37).
  6. **Helpers.** Saturation vapor pressure at 20 °C ≈ 2338 Pa; molar density at 20 °C and 1013 hPa ≈ 41.6 mol m⁻³. Convection: r\_bH is finite at U = 0, falls smoothly as wind rises, the regime label switches at the configured ratio, the equivalent coefficient C = √ν / (s × b × 0.66 × D\_H × Pr^(1/3)) comes out near 156, 208, 104 and 313 s^0.5 m⁻¹ for (s, b) = (2, 1), (1, 1.5), (2, 1.5) and (1, 1) (tolerance 3%), and the forced and free Nusselt terms reproduce a worked value from Campbell & Norman once the book has been checked (mark UNVERIFIED until then).

### Phase 2: Metadata \[READY\]

- **T2.1 session\_metadata.py (P0).** Extend the dataclass with: session\_id, timestamp\_start (ISO with timezone), plant\_id, genotype, water\_treatment, reproductive\_stage, block, position, leaf\_position, light\_source, lamp\_distance\_mm, window\_blocked, acclimation\_minutes, fan\_state, working\_distance\_mm, incidence\_angle\_deg, card\_angle\_deg, lamp\_angle\_deg, lamp\_to\_card\_mm, background (neutral\_matte or other), card\_id, dry\_reference\_type (none, card, vaseline\_leaf), wet\_reference\_present, operator, porometer\_gs\_mol (optional), notes. Allowed values for genotype (Houjaku\_Kuwazu, Williams\_82), water\_treatment (well\_watered, drought\_stressed), light\_source (window, led\_lamp, sunlight), fan\_state (off, low, high) and reproductive\_stage live in config; reject anything else with a clear message. Support command-line flags (--plant-id, --genotype, ...) and fall back to prompts for missing values. Validation: block is a whole number ≥ 1; position and leaf\_position are normalised to lowercase with underscores; lamp\_distance\_mm, lamp\_angle\_deg and lamp\_to\_card\_mm are required only for led\_lamp (blank otherwise); angles are 0 to 90°; plant\_id contains only letters, digits, - and \_; timestamps carry a UTC offset. When dry\_reference\_type is card and the card\_id is not in data/reference\_cards.csv, warn (do not reject). If data/raw/session\_metadata.csv exists with an old or unknown header, refuse to write and ask the user to move it aside; never rewrite raw data. Also write a session.json copy inside the session folder. Done when tests cover rejection of bad values and refusal of an old-format file.
- **T2.2 Reference-card manifest (P0).** data/reference\_cards.csv with the columns of Appendix B. These 11 are required: card\_id, shape\_group, source (designed, traced\_leaf), material, thickness\_mm, coating, a\_s\_card, emissivity, area\_mm2, photo\_path, notes; d\_min\_mm, d\_eq\_mm, aspect\_ratio, eccentricity, solidity, circularity, edge\_density and geometry\_source are optional until measured. Numbers may be blank until measured. Loader in data\_io.py. Done when the loader validates required columns.
- **T2.3 Tests (P0).** tests/test\_session\_metadata.py.

### Phase 3: Thermal and image processing \[READY with sample images; final tuning BLOCKED-CAM\]

- **T3.1 leaf\_segmentation.py (P0).** segment\_and\_measure(image\_bgr, mm\_per\_pixel, mode, incidence\_angle\_deg=None, params=None) returns geometry, a mask and an overlay image. Steps: excess-green index (2G − R − B) for leaf mode, or a dark-object threshold for card mode; Otsu threshold; morphological open and close; keep the largest connected component and fill holes; fit contour and ellipse. Features: area\_mm2, perimeter\_mm, d\_min\_mm (short side of the minimum-area rectangle), d\_max\_mm, d\_eq\_mm (√(4A/π)), aspect\_ratio, eccentricity (√(1 − (b/a)²)), solidity (area over convex-hull area), circularity (4πA/P²), edge\_density (Canny edge pixels inside the mask divided by mask pixels, a texture proxy), n\_components. Fail with segmentation\_failed if there is no component, the mask touches the image border, or the area is implausible. If incidence\_angle\_deg is given, apply a first-order foreshortening correction and flag it foreshortening\_corrected (set whenever an angle is given, including 0): area\_mm2 = projected area / cos(angle), and d\_eq\_mm is recomputed from that corrected area; d\_min\_mm, d\_max\_mm and perimeter\_mm stay uncorrected because the tilt axis is unknown. Edge density and all ML training features must come from Arducam photos only; phone photos only test segmentation. A top-down photo cannot see leaf cupping, so true curvature is not a feature; incidence angle and solidity stand in for it (optional P2: a side-view photo). Provide mm\_per\_pixel\_from\_fov(working\_distance\_mm, hfov\_deg, image\_width\_px) as a fallback until camera\_scale.py exists; for phone photos, include a ruler. Done when drawn shapes of known size (ellipse, circle, lobed polygon) return area and d\_min within 2%, and at least five real leaf photos give masks that look correct in the overlays.
- **T3.2 thermal\_roi.py (P0).** frame\_from\_list(values) to a 24×32 array; rect\_mask(r0, r1, c0, c1); erode\_mask(mask, n); roi\_stats(frame, mask, min\_pixels=6) returns mean, median, standard deviation, pixel count, valid and reason (roi\_too\_small), after trimming outliers by median absolute deviation (pixels more than k × 1.4826 × MAD from the median, k = acquisition.roi\_mad\_k, 3); pixel\_footprint\_mm(distance\_mm, cfg) reports horizontal and vertical footprints from sensors.mlx90640.hfov\_deg 55 and vfov\_deg 35 (Adafruit 55 × 35° product listing; confirm with a known-size warm object at T7.4), about 6.5 × 5.3 mm per pixel at 200 mm, and the pure-pixel decision uses the larger footprint; warp\_mask\_to\_thermal(mask\_rgb, matrix, shape, cfg) for later use keeps a thermal pixel only if its covered fraction is at least acquisition.min\_pixel\_coverage (1.0), sampling each thermal pixel at acquisition.warp\_supersample × warp\_supersample points (4) \[BLOCKED-CAM to validate\]. Done when synthetic frames with a known warm patch return the right mean and small patches are rejected.
- **T3.3 qc.py (P0).** Range checks (air temperature −10 to 50 °C, humidity 0–100%, pressure 900–1100 hPa, lux ≥ 0, thermal −10 to 80 °C), spike detection (Hampel filter, window 7, 3σ, with a minimum spike size per variable from config qc; placeholders: air temperature 0.5 °C, humidity 3%, pressure 1 hPa, lux 20% of the window median, thermal 1 °C), overflow and duplicate-timestamp detection. Spike checks apply to the environment sensors and the whole-frame thermal\_min\_c, thermal\_max\_c and thermal\_mean\_c, never to leaf\_temp\_c, card\_temp\_c or wet\_ref\_temp\_c (range checks still apply), because flagging on those would select data on the measured result. sensor\_overflow uses 65535 until the installed adafruit\_tsl2591 library's own limit is checked on the Pi (open item). NaN values are not flagged. Adds a qc\_flags column; never deletes rows. Done when the first debug log is flagged (lux of about 2 amid readings near 88; the thermal frame near 37 °C; the humidity jump on row 8).
- **T3.4 measurement\_windows.py (P0).** build\_windows(sensor\_df, events\_df, meta, cfg, photo\_features=None) splits a session into windows (default 60 s, at least 10 samples), excludes qc-flagged rows, and reports mean and standard deviation of every variable. steady\_state is true when the leaf-temperature slope over the last stability\_minutes (default 3) is below stability\_slope\_c\_per\_min (default 0.05) and the logged light-acclimation time meets min\_acclimation\_minutes (default 30, as used in soybean gas-exchange studies). Otherwise reason not\_steady\_state. As implemented (decision\_log 2026-11-09): row time is timestamp\_corrected\_iso when filled, else timestamp\_iso; qc.add\_qc\_flags runs inside and every flagged row is excluded; the slope is a least-squares fit over the clean rows in [window end − stability\_minutes, window end), and the data must reach back that far; acclimation is the logged acclimation\_minutes plus time since the first row, restarted by the latest events.csv row labelled windows.acclimation\_reset\_label (lamp\_on) at or before the window end; each window has one reason, in the order missing\_input (fewer than min\_samples clean rows or no leaf temperature), not\_steady\_state, no\_photo\_in\_window, and every other problem is also listed in flags. build\_windows also attaches photo features: attach\_photo\_features(windows, photo\_features, max\_gap\_sec, leaf\_moved\_range\_frac) joins by timestamp (the median of photos inside the window, else the nearest photo within windows.max\_photo\_gap\_sec), never by row index. Windows with no photo in range get reason no\_photo\_in\_window, and a session whose d\_min range, (max − min)/median over its ok photos, exceeds windows.leaf\_moved\_range\_frac (0.05) gets flag leaf\_moved (B15). build\_windows also takes temp\_column (default leaf\_temp\_c; card windows use card\_temp\_c, giving card\_slope\_c\_per\_min and flag card\_temp\_drifting). Done when synthetic drifting and flat series are classified correctly and windows with photos at known times receive the right features.

* **T3.5 processing/extract\_photo\_features.py (P0).** Command: `python -m processing.extract_photo_features --session SESSION_ID`. Runs T3.1 on every image in the session’s images folder, using the session’s mm\_per\_pixel and incidence angle, and writes data/processed/SESSION\_ID/photo\_features.csv (timestamp, image\_file, all geometry features, ok, reason) plus overlay PNGs for human review. It works now on a folder of phone photos; real photos need the camera \[BLOCKED-CAM\]. Done when it processes a folder of test images and the CSV loads through data\_io.

### Phase 4: Data handling and machine learning \[READY with synthetic data\]

- **T4.1 data\_io.py (P0).** list\_sessions(raw\_dir) (only folders that contain run\_manifest.json), load\_session(session\_id) returning metadata, sensor table, events and file paths, load\_reference\_cards(), load\_ground\_truth(); schema checks with clear errors. As implemented (decision\_log 2026-11-09): load\_session(session\_id, raw\_dir) returns a frozen Session (session\_id, metadata from session.json, sensor table, events, manifest, file paths, synthetic); schema problems raise SchemaError naming the file and column (and row); a missing events.csv gives an empty table with a warning; extra sensor\_log columns are kept with a warning. Open item: the porometer surface label in ground\_truth\_gs.csv (adaxial, abaxial, both\_summed, whole\_leaf) must be settled before real ground truth is used (A40). Done when a fabricated session folder loads.
- **T4.2 ml/synthetic\_data.py (P1).** make\_synthetic\_cards(n\_shapes=20, seed=0) with random geometry features and a hidden true correction f = 1 + a×(1 − solidity) + b×(aspect\_ratio − 1) + c×edge\_density plus noise. simulate\_card\_session and simulate\_leaf\_session run the forward physics (root-find the temperature from a known r\_b or g\_s, add sensor noise) and emit tables in exactly the real schemas. As implemented (decision\_log 2026-11-09): make\_card\_sets gives train, held-out test (another seed) and null (f\_true = 1) card sets; card and leaf sessions come from forward energy-balance root-finding that never calls the inverse solvers; per-sample noise (synthetic.noise) and an optional per-session bias (synthetic.session\_bias, drawn from the uncertainty sigmas) are separate, and in bias mode one thermal offset is shared by leaf and card (one camera); the root-finder search range is synthetic.root\_bracket\_c. Open items: synthetic conditions are perfectly steady, so a drift and warm-up variant is needed before end-to-end testing (T6.1); shape features of synthetic cards are approximate (perimeter and circularity from a rule of thumb). Done when the outputs pass the same schema checks as real data.
- **T4.3 ml/build\_training\_set.py (P1).** Command: python -m ml.build\_training\_set --out data/processed/rb\_training\_set.csv. Uses steady-state card windows selected only by exogenous conditions (steady state, wind present, absorbed shortwave above the threshold; never by the measured temperature difference or by the sign of r\_bH), joins card geometry and wind (rows without valid wind are rejected as no\_wind), solves card r\_bH, computes the flat-plate r\_bH (forced plus free convection) from the card’s d, wind and temperature difference, and stores f = r\_bH / r\_bH\_flat. Writes rejects with reasons to a separate file. Requires at least min\_shapes distinct shape\_group values (default 10) unless --force is given, which prints an overfitting warning. Prints a report: rows kept, rejects by reason, shape count, wind and light ranges, the suspect\_card\_rbh count by shape and wind (it should not depend on either), and each row’s label uncertainty from sigma\_rn\_card\_fraction (used as a sample weight). As implemented (decision\_log 2026-11-09): the gate is only steady state (card\_temp\_c), wind present and absorbed shortwave of at least physics.min\_card\_absorbed\_sw\_wm2 (a placeholder); cards whose geometry\_source is not camera are rejected with reason missing\_input; the target f is on a linear scale and training is unweighted; r\_bh\_card\_sigma is stored and reported, not used as a weight; the report adds the truncation bias of mean f from rejecting r\_bH ≤ 0 and warns above ml.max\_suspect\_fraction; rb\_training\_set.csv has an extra flags column; a blank card\_angle\_deg becomes 0 with flag card\_angle\_default. Open items: (1) the minimum card ΔT, the lamp strength and the absorbed-shortwave gate are chosen from the ml/delta\_t\_sweep.py table together with the real calibration (T7.5); with placeholder sigmas the sweep suggests at least about 4 °C at about 200 W m⁻² and 0.8 m s⁻¹; (2) never gate on the measured ΔT, only on lamp strength or other settings; (3) the light-calibration offset (k) is shared by all sessions in a real study and leaves a bias of about 6–8 % in f that a larger ΔT cannot remove, so the light meter and the k calibration are make-or-break, like the thermal calibration. Done when it runs on synthetic data and the report matches known counts.
- **T4.4 ml/train\_correction\_model.py (P1).** Features: d\_min\_mm, d\_eq\_mm, aspect\_ratio, eccentricity, solidity, circularity, edge\_density, incidence\_angle\_deg, wind\_ms. Target: f (or its logarithm). Candidates: a null model (f = 1), ridge regression, and a small random forest or gradient-boosting model. Evaluate with leave-one-shape-out cross-validation (LeaveOneGroupOut on shape\_group), reporting RMSE and MAE of r\_bH in s m⁻¹ for each model. Pick the best. If it does not beat the null model by at least ml.min\_improvement (default 5%), save the model card with status no\_improvement and make prediction return f = 1 with a flag. Fit the final model on all data, bootstrap (200 resamples) for prediction intervals, and save data/models/correction\_model.joblib plus a JSON model card (features, metrics, training-data hash, number of shapes, library versions, date). Done when the model recovers the hidden synthetic correction and correctly reports no\_improvement when the synthetic correction is set to zero. As implemented (decision\_log 2026-11-09): candidates null, ridge and random forest; the go/no-go decision against null uses nested leave-one-shape-out CV with the 5 % threshold (ml.min\_improvement), and the model card holds both the inner (selection) and outer (honest) scores; status ok or no\_improvement; the bootstrap resamples whole shape groups; f\_lo and f\_hi combine the bootstrap spread and the CV residual SD in quadrature (approximate).
- **T4.5 ml/predict\_correction.py (P1).** predict\_f(features) returns f\_hat, f\_lo, f\_hi and a flag; predict\_rbh(features, wind\_ms, delta\_t\_c, air\_temp\_c, cfg) returns f\_hat × flat-plate r\_bH with an interval. Refuses to run if wind is missing (reason no\_wind). A no\_improvement model returns f = 1 with flag ml\_no\_improvement; a feature outside the training range sets outside\_training\_range. Done when a saved model reloads and predicts.
- **T4.6 tests (P1).** tests/test\_ml\_pipeline.py covering T4.2–T4.5, including that cross-validation never puts the same shape\_group in train and test.

### Phase 5: Analysis scripts \[READY with synthetic data; real validation needs real data\]

- **T5.1 aggregate\_results.py (P0).** Command: python -m analysis.aggregate\_results. For each steady-state leaf window it computes g\_s with each r\_b method: flat\_plate (needs leaf d from a photo and wind), card\_direct (uses a valid card window from the same session within card\_match\_minutes, default 10, as r\_b), and ml\_corrected (needs the trained model, geometry and wind). Each result goes through the uncertainty module. If wet and dry reference temperatures exist, it also computes the Jones index. Output: data/processed/leaf\_gs\_results.csv (Appendix B). A method that cannot run produces an invalid row with a reason code, never a silently missing row. Censored windows (below\_detection, above\_range) are written with their label and detection limit and are never removed. Geometry comes from the photo join in T3.4. Each row carries method\_role (primary, secondary or baseline) from config methods and the convection regime. Until the camera and wind sensor exist, only card\_direct and the Jones index can run on real data; all methods run on synthetic data. As implemented (decision\_log 2026-11-10): every leaf window produces three rows (card\_direct, ml\_corrected, flat\_plate), each with its method\_role from config methods; the roles are fixed before any real data is collected and must not be changed after seeing results. Non-steady windows are written as invalid rows with reason not\_steady\_state; no row is ever dropped. In the Monte Carlo each method uses its own r\_b interval, sigma\_rb = ln(r\_bh\_hi / r\_bh\_lo) / 3.92 (card Monte Carlo for card\_direct, ML interval for ml\_corrected); flat\_plate keeps uncertainty.sigma\_rb\_fraction (0.2), which is too small (up to 3× error, decision 20). The Jones index uses only a vaseline\_leaf as the dry reference; with a card as the dry reference, I\_g and CWSI are blank with flag jones\_no\_leaf\_dry\_reference; references\_not\_bracketing is written into flags; the extra column jones\_ig\_censor carries the censor label for I\_g. Open item: the leaf and the card share one thermal camera, so their errors are correlated; the Monte Carlo treats them as independent (card\_direct intervals probably conservative) until correlated draws are added (A43). Done when a synthetic campaign produces all rows.
- **T5.2 baseline\_benchmark.py (P0 for the comparison to ground truth; P1 for the three-method claim).** Compares each method with the ground truth in data/validation/ground\_truth\_gs.csv, starting with card\_direct, whose error is the headline accuracy result (paired by plant\_id and time within 10 minutes). Reports bias, MAE, RMSE, R², regression slope and intercept, and a Bland–Altman plot. Splits results by shape deviation (absolute distance of solidity and aspect ratio from the median leaf, in tertiles) by g\_s range and by convection regime (forced, mixed, free; the flat-plate baseline is judged on forced windows only), and tests whether ML error is lower than flat-plate error (paired Wilcoxon test on absolute errors). As implemented (decision\_log 2026-11-10): pairing, the whole-leaf rule and the statistics are shared with T5.6. Leaf shape comes from the same photo join as build\_windows (attach\_photo\_features, windows.max\_photo\_gap\_sec); a window with no photo is shape\_unknown. Shape deviation = √(z\_sol² + z\_ar²), where z = (x − median) / (1.4826 × MAD) for solidity and aspect ratio (MAD = median absolute deviation; 1.4826 = 1 / Φ⁻¹(0.75), so the scale equals the SD for normal data; falls back to the SD when the MAD is 0); medians and MADs are taken over the distinct paired windows with known shape. Shape-deviation tertiles (q1 = most typical), g\_s-range tertiles (on the true g\_s) and convection-regime splits (validation.n\_split\_bins) are descriptive only: n and statistics, no p-values. flat\_plate is judged on forced windows in every table except the by-regime table. ML versus flat plate is a two-sided paired Wilcoxon signed-rank test on per-plant mean absolute error over forced-convection windows, pairing the two methods on the same ground-truth values (two-sided is a deliberate deviation from the one-sided wording above, because ML could also be worse); the window-level test is secondary and labelled non-independent. The primary analysis is pre-registered in the decision log (2026-11-10, before any real data) and must not be changed after seeing results: plant-level RMSE per method on forced-convection windows, and the plant-level Wilcoxon. Done when synthetic data shows ML better on atypical shapes and no difference on typical ones.
- **T5.3 genotype\_comparison.py (P0).** The primary analysis uses card\_direct; every other method and the Jones index are run as sensitivity analyses. Each uses: a mixed-effects model (statsmodels MixedLM), g\_s \~ genotype × treatment with a random intercept for plant\_id (block as a covariate if present); an ANOVA on plant means as a check; effect sizes with 95% confidence intervals; headline numbers are the percent drop under drought for each genotype. Keeps censored windows (A15). It reports the share of below-detection windows per genotype × treatment; fits a censoring-aware model on plant-level values (a left-censored Tobit-style maximum-likelihood fit, coded with scipy; DEFER until Tier 1 works, and until then rely on the rank test and sensitivity table below, which already keep every censored window); runs a rank-based test on plant medians with censored values tied at the detection limit; and shows a sensitivity table that substitutes 0, half the detection limit and the detection limit for censored values. The same comparison is run on the Jones index and CWSI, which are never censored. It is also run on forced-regime windows only (once wind data exist), because card\_direct is least reliable at low wind (A17). Only windows with unusable inputs, failed quality checks or no steady state are excluded, and the counts by reason are reported. The plant is the unit of replication; windows from one plant are never treated as independent. Done when synthetic data with a planted effect recovers it, including a case where the stressed group’s true g\_s is near zero.
- **T5.4 detection\_limit.py (P0).** Sweeps true g\_s over a range at representative conditions (taken from real sessions), runs the uncertainty module, and reports the lowest g\_s that is resolved in at least 90% of draws with a relative 95% interval width under the limit. Compares with the empirical brackets: a petroleum-jelly leaf (g\_s near zero, should be unresolved) and a wet reference (very high g\_s). This value is the detection limit stored with censored windows (A15). As implemented (decision\_log 2026-11-10): the resolved band (lowest and highest resolved g\_s) uses the full three-part A33 rule (at least 90% of draws resolved, relative width below uncertainty.max\_relative\_width, p97.5 below g\_s\_cap\_mol); resolution\_at in processing/uncertainty.py is shared with detection\_limit\_mol, so the sweep and the per-window limit agree; the brackets table and the figure are produced; every output is labelled SYNTHETIC when synthetic data is used, and real and synthetic sessions are never mixed. Open items: with the placeholder sigmas no g\_s is resolved, so no detection limit can be stated until the thermal and light calibrations (T7.5); the synthetic target of about 0.2 °C leaf error and 5% light error gives a limit of about 0.09 and a resolved range of 0.09 to 1.2 mol m⁻² s⁻¹; rerun python -m analysis.detection\_limit with the real calibrations. Done when a figure and a one-line summary are produced.
- **T5.5 validate\_rb\_vaseline.py (P1).** For sessions with dry\_reference\_type = vaseline\_leaf: solve the coated leaf’s r\_bH like a card (use the leaf’s a\_s, and note that the coating may change optical properties), then compare it with the flat-plate and ML predictions for that leaf’s geometry and wind. Reports bias and error per method. Done when run on synthetic coated-leaf data.
- **T5.6 validate\_gs\_ground\_truth.py (P0).** Paired device-versus-porometer or weighed-transpiration comparison for each method, on a whole-leaf basis (E4; rows with only one surface are listed separately and never compared directly): regression, Bland–Altman, bias, RMSE, with the detection limit overlaid. As implemented (decision\_log 2026-11-10): the script refuses to run unless physics.faces = 2 (the device g\_s is then both faces combined). Whole-leaf rule (E4): porometer rows with surface whole\_leaf or both\_summed are used as they are; an adaxial and an abaxial row for the same plant\_id and leaf\_position within validation.surface\_pair\_minutes are summed (g\_ad + g\_ab) and timed at their midpoint, closest pairs first, each reading used once; leftover one-surface rows go to a separate table with reason one\_surface\_only and are never compared. Pairing is by plant\_id, leaf\_position (when both are filled) and method: the nearest valid device window whose midpoint is within validation.pair\_max\_minutes. Only valid, uncensored device rows enter the statistics; censored and invalid matches (and truth values with no window) are counted and listed. Statistics (d = device − truth): bias, MAE, RMSE, SD of d, ordinary least-squares slope and intercept of device on truth, R² (n ≥ 3), Bland–Altman limits bias ± validation.loa\_z × SD. Plant-level statistics (one point per plant: mean device and mean truth) are primary; window-level statistics are secondary and labelled non-independent. A detection table reports, among valid device rows with a known g\_s\_mol\_dl, how often a true g\_s below the limit was returned as below\_detection and how often a true g\_s above it was wrongly returned as below\_detection. Gravimetric rows are never in the g\_s error tables: they appear only in a separate per-plant table (device mean within validation.gravimetric\_match\_hours of a weighing) with a Spearman rank correlation across plants (A45). --synthetic reads leaf\_gs\_results\_synthetic.csv and ground\_truth\_gs\_synthetic.csv and stamps every output SYNTHETIC. Open items: the gravimetric weighing protocol and its units (leaf-level or whole-plant, and how weight loss becomes g\_s) must be defined (A45); the porometer surface label must be settled before real use (A40). Done when synthetic pairs with known error are recovered.
- **T5.7 make\_figures.py (P0).** Command: python -m analysis.make\_figures. Regenerates every figure and table into docs/figures/ and docs/tables/ from data/ alone: g\_s by genotype and treatment with confidence intervals, method comparison, Bland–Altman, detection limit, ML cross-validation performance, calibration residuals, fraction of windows below detection by group. Done when one command rebuilds everything.

* **T5.8 analysis/position\_swap\_check.py (P2, DEFER until Tier 1 works).** Compares the r\_b solved for the same card at the leaf position and at the card-arm position under identical conditions. Reports the ratio with a confidence interval and flags it if |ratio − 1| exceeds 0.05 (A16). Done when synthetic data with a planted 8% irradiance difference is detected.

- **T5.9 analysis/card\_buoyancy\_check.py (P1).** From card sessions at one wind setting and several irradiance levels (T10.2), regress window-mean r\_bH on |T\_card − T\_air| for each card shape and wind setting, with confidence intervals from the label uncertainty, and compare with the forced-plus-free Nusselt prediction. Report the fractional change in r\_bH between the card’s typical ΔT and a leaf-like ΔT (about 0 to 1 °C). Where it exceeds card\_transfer\_max\_change, flag card\_direct windows in that regime as card\_transfer\_unreliable. Done when synthetic data generated with the Nusselt model recovers the planted ΔT dependence.

### Phase 6: End-to-end dry run \[READY, P0\]

- **T6.1 processing/pipeline\_dry\_run.py.** Command: python -m processing.pipeline\_dry\_run. Generates synthetic cards and leaves with known correction and known g\_s (T4.2), solves card r\_b, builds the training set, trains, predicts for the synthetic leaves, computes g\_s three ways with uncertainty, runs the statistics, and draws a figure. It asserts that the pipeline completes, that ml\_corrected has lower RMSE than flat\_plate on atypical-shape leaves, that the planted genotype × treatment effect is recovered even when the stressed group’s true g\_s is near zero (and that dropping censored windows would have shrunk it), that runtime is under 2 minutes, and that the exit code is 0. Run it before every merge to the main branch. Open item (from the T4.3 ΔT sweep): apply one shared k error to both the card training and the leaf g\_s calculation, to see whether the errors partly cancel.

### Phase 7: Logger rewrite and calibration scripts \[PI\]

- **T7.1 acquisition/sensors.py (P0).** One class per sensor: Bme280Sensor, Tsl2591Sensor, Mlx90640Sensor, and stubs Fs3000Sensor and ArducamCamera (completed in Section 8). Each has read() returning a small dataclass with values and a status. Each also has a Simulated version (plausible values with noise, honoring the configured light source and fan state) and a Null version (returns blanks). build\_sensors(cfg, simulate=False) picks them from config. TSL2591: catch the overflow RuntimeError, step the gain down (high, medium, low), retry, and report lux, visible, infrared, full spectrum, gain and integration time. MLX90640: retry on ValueError up to 5 times and return a 24×32 array; refresh rate from config (currently 2 Hz). Done when the logger runs end to end with simulation on a laptop.
- **T7.2 acquisition/combined\_logger.py rewrite (P0).** Command: python -m acquisition.combined\_logger \[--simulate\] \[metadata flags\]. Flow: collect metadata (T2.1); run the minimum clock check of B14 (timedatectl show -p NTPSynchronized; record clock\_status in the manifest and refuse to start on a clearly wrong time; the typed-offset logic is deferred); create data/raw/\<session\_id>/; write run\_manifest.json (start time, clock\_status, clock\_offset\_s, git commit hash from git rev-parse HEAD, config snapshot, calibration file hash, package versions, sensor settings, and the output of vcgencmd get\_throttled if available); build the sensors; then every log\_interval\_sec (default 5): read all sensors, append the thermal frame to thermal\_frames.f32 and record thermal\_frame\_idx (flush and fsync every 12 frames), compute leaf, card and wet-reference ROI statistics from the ROIs in calibration.json, apply range checks, write one row in the Appendix B schema (wind and image columns blank while those sensors are disabled), and flush. Call camera.capture(...) every photo\_interval\_sec (a NullCamera until Section 8 is done). On Ctrl+C: flush, record end time and sample count in run\_manifest.json. Done when a 10-minute simulated session produces every expected file and data\_io loads it, and a real 10-minute session on the Pi runs without crashes.
- **T7.3 acquisition/mark\_event.py (P1).** Command: python -m acquisition.mark\_event lamp\_on. Appends a timestamp and label to the current session’s events.csv (the current session is recorded in a pointer file, data/raw/.current\_session). Labels: lamp\_on, window\_covered, measurement\_start, measurement\_end, card\_swap, fan\_change. Done when build\_windows (T3.4) reads the events.
- **T7.4 calibration/define\_thermal\_rois.py (P0).** Shows the live thermal frame (or saves a PNG if there is no display), asks for row and column ranges for the leaf, card and wet reference, checks each has at least min\_pixels pure pixels after erosion, and writes them to calibration.json under roi. Done when overlays look right on a warm-object test.
- **T7.5 calibration/thermal\_temperature\_cal.py (P0, needs a reference thermometer).** Guided steps at three or more known temperatures (an ice-and-water bath near 0 °C, room temperature, warm water around 35–40 °C), each viewed through a high-emissivity surface (the water surface or a black matte-coated container). Records the MLX90640 ROI mean and the BME280 air temperature next to the typed-in thermometer reading, fits an offset (and a gain only if justified), and writes offsets, residual standard deviation (used later as the temperature sigma in uncertainty), date and thermometer description to calibration.json. Done when corrected residuals are printed and stored. Open item (from T3.4): use the leaf ROI noise measured here and with the T7.4 regions to re-check the steady-state slope test; for a flat leaf it gives false alarms about 1% of the time at 0.1 °C sample noise and about 19% at 0.2 °C.
- **T7.6 calibration/light\_calibration.py (P0, needs a reference light meter).** For one light source at a time, records TSL2591 lux at several intensities (vary the lamp distance) next to the reference reading. Accepts W m⁻² from a pyranometer, or PAR in µmol m⁻² s⁻¹ with a configurable conversion (about 4.57 µmol per J for sunlight). For a visible-only LED lamp, PAR in W m⁻² is essentially all of the shortwave E\_s. For window daylight or sunlight, PAR is only about half of total shortwave, so use a pyranometer or divide by the PAR fraction. Fits k through the origin and stores k, residual and date per light source. k must exist before any g\_s can be computed for that source; there is no fallback for an LED lamp. Done when k is stored and the physics code reads it.
- **T7.7 Docs (P1).** docs/pi\_setup.md: enable I2C; install requirements-pi.txt; set the clock (if an RTC module is fitted, enable it in the boot config; note whether the fake-hwclock service is present); if the long sensor cable causes lockups, lower the I2C bus speed with the dtparam i2c\_arm\_baudrate setting in the Pi boot config and confirm the thermal frame rate is still adequate; run the hardware tests. docs/measurement\_protocol.md: cover the window or move away from it; fix the lamp in its jig at the logged distance and angle; hold the card at the same tilt and distance as the leaf, in front of the same neutral matte backdrop; mark lamp\_on; wait the acclimation time; place card and wet reference; start the logger; record several windows; take a photo; take a porometer or scale reading when available; rotate plants weekly within randomized blocks.

* **T7.8 calibration/light\_field\_map.py (P0, PI).** With the real lamp in its jig, move the TSL2591 (on a short cable) over a grid of positions covering the leaf area, the card area and the plate’s own light-sensor position, recording lux at each (five readings, median). Computes c\_leaf = lux at the leaf position ÷ lux at the sensor position, c\_card likewise, and the spread across the card and leaf areas, and writes them to calibration.json with the lamp distance and angle. Warns if irradiance varies by more than 5% across the card or leaf area, or if c\_card and c\_leaf differ by more than 0.05, because r\_b is inversely proportional to Rn and such differences become systematic label errors (A16). Repeat whenever the lamp or jig changes. Done when the factors are stored and the physics code reads them (flag c\_uncalibrated until then).

### Phases 8 and 9: Camera and wind sensor \[BLOCKED\]

See Section 8 for the complete task lists, what to write now, and what to do when each part arrives.

### Phase 10: Real-data integration and final analysis \[OWNER plus code; needs real data\]

- **T10.1** Run the calibrations (T7.4–T7.6) and the light-field map (T7.8) and store them.
- **T10.2 Card campaign.** Each card in the manifest, at three or more wind conditions (after the FS3000 arrives) and three or more light levels, with repeats; sessions marked dry\_reference\_type = card. Include traced soybean-leaflet cards and at least one petroleum-jelly leaf session.
- **T10.3 Wet-reference and coated-leaf sessions** for the detection-limit brackets and the Jones index.
- **T10.4 Plant sessions.** Two genotypes × two treatments × 4–5 plants, on several days, in randomized blocks; porometer or weighed-transpiration pairs on a subset, typed into data/validation/ground\_truth\_gs.csv.
- **T10.5 Run the pipeline:** build windows, build training set, train, aggregate results, benchmark, validate, statistics, figures.
- **T10.6 Review.** Inspect QC flags and exclusion counts, confirm the detection limit, state honestly which tier of claim the data supports, and finalize docs/assumptions\_and\_limitations.md.

## 8. Blocked items: CSI-to-HDMI cable (Arducam) and FS3000 (wind)

### 8.1 Summary

| Item | Blocked by | Write and test now | Do when the part arrives |
| --- | --- | --- | --- |
| acquisition/camera\_capture.py | Cable | Class with file-based fake camera and tests | Run on the real camera |
| hardware\_tests/test\_arducam.py | Cable | Write it | Run it, check the image |
| calibration/focus\_sweep.py, camera\_scale.py, thermal\_rgb\_alignment.py | Cable | Write against saved or phone images | Run on real images, store results |
| Photo hook in the logger | Cable | Already wired to a null or file camera | Enable in config |
| Real card and leaf geometry features | Cable | Segmentation tested on phone photos and drawn shapes | Re-tune on real photos, fill the card manifest |
| Leaf region from the photo mask (roi\_source: rgb\_mask) | Cable | Warp function with tests | Switch on after alignment passes |
| hardware\_tests/test\_fs3000.py | FS3000 | Write it | Run it in still air |
| Wind in the logger, fan sub-dataset | FS3000 | Columns exist, blank; sensor class and simulator written | Enable in config, collect data |
| calibration/wind\_check.py | FS3000 | Write it | Run it, set the low-wind floor |
| Flat-plate baseline and ML features on real data | FS3000 (and camera for d) | Tested on synthetic data | Rebuild training set and rerun benchmark |

### 8.2 Camera: Arducam Camera Module 3 over the CSI-to-HDMI extension

**Write and test now (no camera needed)**

- **N1 acquisition/camera\_capture.py.** Classes ArducamCamera (imports picamera2 inside its constructor, in try/except), FileCamera (hands back images from a folder, for tests) and NullCamera. capture(session\_dir, timestamp, label='leaf') saves images/\<timestamp>\_\<label>.jpg plus a small JSON file with exposure, gain, lens position, working distance and resolution, and returns the relative path. Use manual focus: set AfMode to manual and LensPosition to 1000 divided by working distance in mm (the lens position is in dioptres). Config keys: sensors.arducam.enabled, working\_distance\_mm, lens\_position, resolution, photo\_interval\_sec.
- **N2 hardware\_tests/test\_arducam.py.** Lists cameras, captures one still, prints resolution and metadata, exits with an error if no camera is found.
- **N3 calibration/focus\_sweep.py.** Sweeps LensPosition around 1000 / working\_distance\_mm, captures at each setting, scores sharpness (variance of the Laplacian on the center crop), prints the best value and writes it to calibration.json.
- **N4 calibration/camera\_scale.py.** Takes a photo of a ruler or an object of known width at the working distance, takes two pixel positions and the known length, and computes mm\_per\_pixel. Also prints the field-of-view estimate for comparison. Writes calibration.json.
- **N5 calibration/thermal\_rgb\_alignment.py.** Takes four or more corresponding points seen in both cameras (a small warm object, such as a bottle of warm water on a cold background, moved to several positions: mark its position in the thermal grid as row and column and in the photo as x and y). Fits an affine transform (cv2.estimateAffine2D) or a homography (cv2.findHomography), stores the matrix and the reprojection error in calibration.json, and supplies the matrix used by thermal\_roi.warp\_mask\_to\_thermal.
- **N6 Segmentation tuning on phone photos** with a ruler in frame, and phone or scanner photos of each reference card, so geometry columns in data/reference\_cards.csv can be filled early. Mark them geometry\_source = phone and redo them later with geometry\_source = camera.

**When the cable arrives**

1. Power the Pi off. Connect the extension kit between the Arducam and the Pi’s camera (CSI) connector, not the display connector. Seat the ribbon fully with the contacts the right way round and close the latch. Power on.
2. Check detection: rpicam-hello --list-cameras (libcamera-hello on older Raspberry Pi OS). Expect an imx708 entry. If none appears, reseat both ends of the extension and check the connector.
3. Install: sudo apt install -y python3-picamera2 python3-opencv. If you use a virtual environment on the Pi, create it with --system-site-packages so picamera2 is visible.
4. Run hardware\_tests/test\_arducam.py. Confirm a clean image. If the long cable causes frame errors or dropouts, reseat it, shorten it, or lower the resolution, and record the finding in docs/pi\_setup.md.
5. Set working\_distance\_mm in config, run focus\_sweep.py, and store lens\_position.
6. Run camera\_scale.py and thermal\_rgb\_alignment.py. Check by imaging a known object: its measured size should be within 3%.
7. Set sensors.arducam.enabled: true and run a 10-minute session. Confirm files appear in images/ and the image\_file column is filled.
8. Run segmentation on real leaf and card photos, tune the thresholds, save overlays, and compare d\_min and area with caliper measurements (within 3%).
9. Switch acquisition.roi\_source to rgb\_mask only after the alignment check passes; keep the manual rectangles as a fallback.
10. Photograph every card, fill the geometry columns in reference\_cards.csv (geometry\_source = camera), and rebuild the training set (this also needs wind, see below).
11. Add camera errors to the QC flags.

### 8.3 Wind: SparkFun FS3000 air-velocity sensor

**Write and test now (no sensor needed)**

- **M1 Fs3000Sensor in acquisition/sensors.py.** Use SparkFun’s Qwiic FS3000 Python package (search PyPI for the FS3000 Qwiic library), or raw I2C reads with smbus2 following the datasheet: read the 5-byte response, check the checksum, and convert the 12-bit value with the datasheet table for the variant. Config sensors.fs3000.variant is 1005 (about 0–7.2 m/s) or 1015 (about 0–15 m/s); read the product label and datasheet to confirm. Expected I2C address 0x28 (verify with i2cdetect). read() returns wind\_ms, wind\_raw and status.
- **M2 Simulated sensor.** Wind depends on fan\_state (off: near zero plus noise; low: about 0.5–1 m/s; high: about 1.5–3 m/s) so downstream code can be tested.
- **M3 hardware\_tests/test\_fs3000.py.** Prints wind speed every second; exits with an error if the sensor is not found.
- **M4 calibration/wind\_check.py.** Records 60 s of still air (noise floor and offset), then 60 s at each fan setting; prints mean and standard deviation; suggests wind\_noise\_floor\_ms; writes calibration.json.
- **M5 Logger columns** wind\_ms, wind\_raw, wind\_status and wind\_source (measured, manual\_assumed, none) always exist and are blank until the sensor is enabled. Downstream code already treats a blank as no\_wind (A14). A manual wind is allowed only through an explicit --assumed-wind flag, which sets wind\_source = manual\_assumed and propagates a flag to every result.
- **M6 fan\_state** is already a validated metadata field (T2.1).

**When the sensor arrives**

1. Read the label to see which variant you have and set sensors.fs3000.variant.
2. Power off, connect it into the I2C chain with a Qwiic or STEMMA QT cable, power on, and run i2cdetect -y 1. Expect the existing devices (BME280 at 0x77, TSL2591 at 0x29, MLX90640 at 0x33) plus the FS3000.
3. Run test\_fs3000.py in still air; values should sit near zero. Note the noise.
4. Mount it so air flows through its sensing channel in the direction the datasheet specifies, at the leaf’s position and out of the sensor plate’s wake. Record the orientation in docs/measurement\_protocol.md.
5. Run wind\_check.py and set wind\_noise\_floor\_ms to the sensor’s noise floor; readings below it count as zero forced flow and free convection takes over (A11).
6. Set sensors.fs3000.enabled: true and run a session with the fan off, low and high. Confirm the wind columns fill and wind\_source is measured.
7. Map the wind: with the fan running, measure at the leaf position, the card position and the sensor plate, and store the ratios in calibration.json; the card and leaf must see the same airflow to within about 10% (A16). Optionally also compare with a handheld anemometer or a known fan distance.
8. Collect the fan sub-dataset: every card shape at fan off, low and high, with fan\_state recorded.
9. Rebuild the training set with wind, retrain, recompute flat-plate baselines with measured wind, and rerun the benchmark.
10. Check that no real-data result uses wind\_source other than measured (or that any exception is flagged).

### 8.4 What still works while the parts are missing

All of Phases 0–6 (physics, metadata, segmentation, ROI and window code, ML and analysis on synthetic data, the dry run); Phase 7 on the real Pi with the BME280, TSL2591 and MLX90640; the thermal and light calibrations; real card sessions that give the primary card\_direct method (it needs neither the wind sensor nor the camera) and, with a wet reference, the Jones index. Only the flat-plate baseline on real leaves, the ML correction on real leaves, geometry from the device’s own photos, and the fan sub-dataset wait for the parts.

## 9. Decisions only the project owner can make

Each has a default in the code so work can continue. Revisit when the stated input is available.

| # | Decision | Default used | Resolve by |
| --- | --- | --- | --- |
| 1 | Include longwave radiation | Yes, surroundings = air temperature unless measured | Compare with a measured wall or window temperature |
| 2 | Sidedness convention | faces = 2 (both leaf faces, per projected area) | Verify in Campbell & Norman (1998) and Guilioni et al. (2008) |
| 3 | Vapor-to-heat boundary-layer ratio | 0.93 | Verify the value in the same references |
| 4 | Flat-plate convection parameters | Nusselt form, enhancement 1.0 (unverified), two faces, forced plus free (C ≈ 156 s^0.5 m⁻¹; 104 to 313 across conventions) | Check against Campbell & Norman; do not count sidedness twice |
| 5 | Characteristic dimension d | d\_min | Test sensitivity; justify in write-up |
| 6 | Leaf absorptivity per light source | Window 0.5, LED lamp 0.85 | Literature or a measurement |
| 7 | Card coating and its absorptivity | Dark-green matte, a\_s\_card measured (black would be about 0.95) | Measure or justify before building the full card set |
| 8 | Measurement light | LED lamp with the window covered | Buy a lamp; calibrate k (required: no LED fallback) |
| 9 | Wet reference material | Damp filter paper or felt | Choose and test |
| 10 | Ground-truth instrument | Porometer if loaned, else weighed transpiration | Ask university labs, extension offices, teachers |
| 11 | Wind noise floor | 0.1 m/s placeholder | wind\_check.py |
| 12 | Number of card shapes | 15–20 or more, including traced leaflets | Cut from one plastic sheet |
| 13 | Fair date, forms and timeline limits | Unknown | Confirm with the fair’s review committee and Society for Science rules |
| 14 | Second plant cohort | Start now | Seeds and a light source |

**Added after external review (defaults in the code):** (15) Clock fix \[DEFER until Tier 1 works\]: for now only record the clock status and refuse to start on a clearly wrong clock; the typed-in time offset is the later default; buy a $5–15 RTC module if sessions will run without network, for example at a fair demonstration. (16) Lamp uniformity target: irradiance varies by no more than 5% across the card and leaf areas, and c\_card and c\_leaf differ by no more than 0.05. (17) Censoring analysis: default is a rank test plus a sensitivity table, with a Tobit-style plant-level fit added later (DEFER until Tier 1 works); a Bayesian censored model is an alternative if you are comfortable with one.

**Added after second review (defaults in the code):** (18) Primary reference card: a card cut to the shape and size of a typical soybean leaflet, traced from your own leaves, used for card\_direct (config methods.primary\_card\_id). (19) Convection combination rule: sum of the forced and free Nusselt numbers by default; switch to larger-of if Campbell & Norman says so. (20) Flat-plate parameters: enhancement factor, sidedness and characteristic width are still to be verified against Campbell & Norman (1998) and cited. The plausible range of C is about 104 to 313 s^0.5 m⁻¹ depending on those choices, so r\_b could be off by up to about 3×; treat the baseline as provisional until then.

**Added after third review (defaults in the code):** (21) Measurement airflow: run every plant and card session with the fan on low so convection is forced and card\_direct transfers (A17); use fan off only for the ML sub-dataset and checks. (22) Sidedness: faces = 2 drives r\_b, longwave and the g\_s convention together; the porometer comparison is whole-leaf, so ask whoever lends the porometer whether it can read both faces. (23) Card gate threshold (min\_card\_absorbed\_sw\_wm2) is a placeholder until the thermal calibration gives a real noise figure.

**Owner tasks that are not code:** update the handwritten notes to match Appendix A; confirm the TSL2591 faces the same way as the leaf’s upper surface (toward the light), not toward the leaf; check that the sensor plate does not shade the leaf; obtain a reference thermometer and a light meter; buy petroleum jelly and a lamp; build the cards, a lamp jig and a neutral matte backdrop.

## 10. Definition of done

- pytest passes on a laptop, and the end-to-end dry run passes.
- A real 30-minute session on the Pi produces a complete, loadable session folder with sensible leaf, card and reference temperatures and no crashes.
- Thermal, air-temperature and light calibrations are stored with dates, and the uncertainty sigmas come from them.
- Every result row has method, valid, reason and an uncertainty range, and a search of the code finds no hard-coded physical constants.
- Validation scripts have been run on real paired data and report bias and error; the detection limit is documented.
- The ML model was trained on at least the minimum number of distinct shapes, leave-one-shape-out results are reported, and the model card states whether it beats the null model.
- Genotype and treatment comparisons use plant-level mixed models with confidence intervals.
- One command regenerates all figures and tables from raw data.
- README and docs are current and the handwritten notes match Appendix A.
- The camera and wind sensor are integrated, and real sessions fill the image and wind columns.

* Windows with closed stomata are kept as censored values, the share censored per group is reported, and the genotype and treatment comparison does not depend on dropping them.
* Every session has a trustworthy clock record (synchronized, or recorded as unsynchronized), and a session timestamp matches an external record such as a phone; typed-offset and RTC support are deferred.
* Lamp irradiance and wind at the card and leaf positions have been mapped, and the card position-swap check passes.
* Photos are paired to measurement windows by timestamp, and thermal frames are stored in one file per session.

- card\_direct, the primary method, has been validated against ground truth with its error and detection limit reported; ml\_corrected is reported only if it passes validation, and flat\_plate is judged on forced-convection windows only.
- No g\_s is computed for an LED lamp without a calibrated light factor.

## Appendix A. Final equation set

All temperatures in °C except in the radiation term (kelvin). Symbols: T\_s leaf (or card) surface temperature, T\_a air temperature, T\_sur surroundings temperature, RH relative humidity (%), P pressure (hPa).

```
1  Shortwave on the leaf        E_s = c_pos x k x lux                                   W m-2
2  Net radiation                Rn = a_s x E_s + n x e_L x sigma x (T_sur^4 - T_s^4)
                                n = faces (1 or 2); e_L = 0.97; sigma = 5.670374419e-8
3  Sensible heat                H = rho cp (T_s - T_a) / r_bH                   rho cp = 1.204 x 1005 = 1210
4  Energy balance               Rn = H + lambdaE   so   lambdaE = Rn - H
5  Vapor pressure               e_s(T) = 610.78 exp(17.27 T / (T + 237.3))      Pa
                                e_a = e_s(T_a) x RH / 100
                                VPD = e_s(T_s) - e_a
6  Psychrometric constant       gamma = 0.0665 x P_hPa                          Pa K-1
7  Latent heat                  lambdaE = (rho cp / gamma) x VPD / (r_s + r_bw)
                                r_bw = ratio x r_bH   (ratio = 0.93)
8  Solve for g_s                r_s = rho cp x VPD / (gamma x lambdaE) - r_bw
                                g_s = 1 / r_s                                   m s-1
                                g_s,mol = g_s x P_Pa / (R x T_a_K)              R = 8.314
9  Card (lambdaE = 0)           r_bH,card = rho cp (T_card - T_a) / Rn_card     uses a_s,card
10 Flat plate baseline          r_bH,flat = 1 / g_bH;  g_bH = s x D_H x (Nu_forced + Nu_free) / d                 Nu_forced = b x 0.66 x Pr^(1/3) x Re^(1/2); Nu_free = 0.54 x Pr^(1/4) x Gr^(1/4); s and b to be verified. Shortcut r = C sqrt(d/U) has C in s^0.5 m-1 (C = 313 / (s x b): 156 for s = 2, b = 1; unverified)
11 ML correction                r_bH,leaf = f_hat(geometry, wind) x r_bH,flat
                                training target f = r_bH,card / r_bH,flat
12 Jones conductance index      I_g = (T_dry - T_leaf) / (T_leaf - T_wet)
13 Water stress index           CWSI = (T_leaf - T_wet) / (T_dry - T_wet)
   c_pos = card or leaf position factor from light_field_map.py (default 1)
```

**Validity rules:** g\_s has a point estimate when lambdaE > 0 and r\_s > 0. lambdaE ≤ 0 means g\_s is at or below the detection limit (censored, below\_detection) and r\_s ≤ 0 means it is above the measurable range (above\_range); both are kept, never dropped. Inputs must be usable: steady state reached and wind measured. For cards the gate is exogenous: absorbed shortwave of at least 40 W m⁻² and |Rn| ≥ 1 W m⁻²; the measured temperature difference is never used to select data.

**Detection-limit rule (added in Session 6, decision A33).** A true g\_s counts as resolved only if all three hold: (1) in at least 90% of the Monte Carlo draws, latent heat flux is above zero, r\_s is above zero and VPD is above zero; (2) the relative 95% interval width is below uncertainty.max\_relative\_width (config, 1.0); and (3) the 97.5th percentile is below uncertainty.g\_s\_cap\_mol (config, 2.0 mol m-2 s-1, UNVERIFIED). Draws with r\_s at or below zero are held at the cap, so an interval that reaches the cap is not bounded above and must not count as precise.

**VPD at or below zero (decision from the Session 8 review).** If latent heat flux is positive while VPD is zero or negative, r\_s comes out negative, so the result is censored above\_range and flagged vpd\_nonpositive. It is not a failure and not a reason code. Such windows stay in every table, in pooled flux-space averages and in the Monte Carlo percentiles (their draws are held at the cap like any draw with r\_s at or below zero). Excluding them would remove the high-conductance end and bias means low.

**Worked example (test fixture):** leaf 23.0, air 24.0, surroundings 24.0, RH 50%, 995 hPa, E\_s 120 W m⁻², a\_s 0.85, faces 1 (set explicitly; the config default is 2), r\_bH 70 s m⁻¹. Result: Rn 107.7, H −17.3, λE 125.0 W m⁻², VPD 1317 Pa, r\_s 127.6 s m⁻¹, g\_s 0.00784 m s⁻¹ = 0.316 mol m⁻² s⁻¹. A ±0.5 °C leaf-temperature error gives 0.402 and 0.250 mol m⁻² s⁻¹.

## Appendix B. config.yaml and data schemas

**config.yaml**

```
physics:
  air_density_kg_m3: 1.204
  specific_heat_air_j_kg_k: 1005.0
  gamma_pa_per_k_per_hpa: 0.0665        # FAO-56: 0.665e-3 x P[kPa] kPa/K
  stefan_boltzmann: 5.670374419e-8
  leaf_emissivity_longwave: 0.97
  rbw_to_rbh_ratio: 0.93                # verify; effect on g_s about 2 percent
  faces: 2                              # 1 or 2 (Section 4, A5); the ONE switch for longwave, r_b sidedness and the g_s convention
  include_longwave: true
  characteristic_dimension: d_min       # d_min or d_eq
  min_abs_rn_wm2: 1.0
  min_card_absorbed_sw_wm2: 40.0        # placeholder gate for card readings; set from calibrated noise after T7.5 (T1.3)
  allow_uncalibrated_k: false           # window and sunlight only; never honored for LED (T1.1)
convection:                             # flat-plate baseline (A10, A11); verify every value in Campbell & Norman
  forced_nu_coeff: 0.66                 # laminar plate
  enhancement_factor: 1.0               # UNVERIFIED: 1.0 until checked; sources give 1.4 to 1.5; C is proportional to 1/b
  free_nu_coeff: 0.54
  combination: sum                      # sum or max
  sidedness_factor: from_faces                 # always equals physics.faces; correlation assumed per face (UNVERIFIED); never set separately
  card_transfer_max_change: 0.15        # flag card_direct when r_b changes more than this from card to leaf temperature difference (A17)
  mixed_convection_ratio: 0.3           # free/forced conductance ratio that marks mixed convection
  wind_noise_floor_ms: 0.1              # readings below this count as zero forced flow; set by wind_check.py
methods:
  primary: card_direct                  # basis of all headline results
  secondary: ml_corrected               # reported only if it passes validation
  baseline: flat_plate                  # comparison method
  primary_card_id: null                 # card cut to a typical soybean leaflet shape and size
spatial:
  irradiance_factor_leaf: null          # c_leaf from light_field_map.py; 1.0 plus a flag until measured
  irradiance_factor_card: null          # c_card
  max_uniformity_spread: 0.05
  max_card_leaf_difference: 0.05
light_sources:
  window:   {a_s_leaf: 0.5,  k_wm2_per_lux: null}    # calibrate; fallback only with allow_uncalibrated_k
  led_lamp: {a_s_leaf: 0.85, k_wm2_per_lux: null}    # must be calibrated; no fallback exists
  sunlight: {a_s_leaf: 0.5,  k_wm2_per_lux: 0.00833} # outdoor reference only
card:
  a_s_card: null                        # measure or justify; matte black about 0.95
  emissivity: 0.95
enumerations:
  genotype: [Houjaku_Kuwazu, Williams_82]
  water_treatment: [well_watered, drought_stressed]
  light_source: [window, led_lamp, sunlight]
  fan_state: ["off", low, high]         # quoted: bare off, no and yes become booleans in YAML
  dry_reference_type: [none, card, vaseline_leaf]
  reproductive_stage: [VE, VC, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, R1, R2, R3, R4, R5, R6, R7, R8]
  background: [neutral_matte, other]
acquisition:
  log_interval_sec: 5
  thermal_refresh_hz: 2
  thermal_storage: single_file          # thermal_frames.f32, appended (B13)
  fsync_every_frames: 12
  photo_interval_sec: 60
  roi_source: manual                    # manual or rgb_mask
  min_roi_pixels: 6
  roi_mad_k: 3.0                        # trim ROI pixels more than k x 1.4826 x MAD from the median
  min_pixel_coverage: 1.0               # keep a thermal pixel only if this fraction lies inside the leaf mask
  warp_supersample: 4                   # sample points per thermal pixel side when warping the RGB mask to the thermal grid
qc:
  ranges:                               # [low, high], inclusive; null = no limit
    air_temp_c: [-10.0, 50.0]
    humidity_pct: [0.0, 100.0]
    pressure_hpa: [900.0, 1100.0]
    lux: [0.0, null]
  thermal_range_c: [-10.0, 80.0]        # applied to every column in thermal_columns
  thermal_columns: [thermal_min_c, thermal_max_c, thermal_mean_c, leaf_temp_c, leaf_temp_median_c,
                    card_temp_c, wet_ref_temp_c]
  spike_columns: [air_temp_c, humidity_pct, pressure_hpa, lux, thermal_min_c, thermal_max_c,
                  thermal_mean_c]       # never leaf, card or wet-reference temperatures
  hampel_window: 7
  hampel_n_sigma: 3.0
  mad_to_sigma: 1.4826                  # 1 / Phi^-1(0.75), derived
  spike_min_abs: {air_temp_c: 0.5, humidity_pct: 3.0, pressure_hpa: 1.0,
                  thermal_min_c: 1.0, thermal_max_c: 1.0, thermal_mean_c: 1.0}   # placeholders
  spike_min_rel: {lux: 0.2}             # fraction of the window median; placeholder
  overflow_columns: [tsl_full_spectrum, tsl_infrared]
  overflow_count: 65535                 # until the adafruit_tsl2591 library's own limit is checked on the Pi
  timestamp_columns: [timestamp_iso, timestamp]
clock:
  min_valid_date: 2026-10-01            # refuse to start if the system time is earlier (B14)
  allow_unsynced: false
windows:
  window_sec: 60
  min_samples: 10
  stability_minutes: 3
  stability_slope_c_per_min: 0.05
  min_acclimation_minutes: 30
  card_match_minutes: 10
  max_photo_gap_sec: 120                # photo-to-window join (B15)
  leaf_moved_range_frac: 0.05           # leaf_moved when (max - min) / median of d_min over the session's photos exceeds this (T3.4)
  acclimation_reset_label: lamp_on      # events.csv label that restarts the light-acclimation clock (T3.4)
segmentation:                           # T3.1; ALL PLACEHOLDERS, tune on Arducam photos (BLOCKED-CAM)
  modes: [leaf, card]                   # leaf = excess green (2G - R - B); card = dark object on light background
  blur_ksize_px: 5                      # Gaussian blur (odd) before Otsu
  open_kernel_px: 5                     # morphological opening (odd): removes specks
  close_kernel_px: 9                    # morphological closing (odd): bridges small gaps
  min_component_px: 200                 # smaller pieces are not counted in n_components
  min_area_mm2: 100.0                   # plausible area range; outside -> segmentation_failed
  max_area_mm2: 20000.0
  canny_low: 50                         # Canny thresholds (grey levels) for edge_density
  canny_high: 150
  edge_erode_px: 5                      # erode the mask before counting edges so the outline is not counted
  max_incidence_angle_deg: 60.0         # larger angles raise an error
  min_contrast: 30.0                    # mean ExG (leaf) or darkness (card) inside minus outside the mask; below -> segmentation_failed
sensors:
  fs3000: {enabled: false, variant: 1005}
  arducam: {enabled: false, working_distance_mm: null, lens_position: null, resolution: [2304, 1296]}
  mlx90640: {hfov_deg: 55, vfov_deg: 35}   # Adafruit 55 x 35 deg listing; confirm at T7.4
uncertainty:
  sigma_leaf_temp_c: 1.0                # placeholders; replace after thermal calibration
  sigma_air_temp_c: 0.5
  sigma_rh_pct: 3.0
  sigma_lux_fraction: 0.15
  sigma_lux_fraction_uncalibrated: 0.5  # used when k_uncalibrated is flagged
  sigma_rb_fraction: 0.2
  sigma_rn_card_fraction: 0.1           # card irradiance mismatch (A16)
  n_draws: 2000
  max_relative_width: 1.0
  dl_min_resolved_fraction: 0.9         # detection limit definition (A15)
ml:
  min_shapes: 10
  min_improvement: 0.05
  n_bootstrap: 200
  features: [d_min_mm, d_eq_mm, aspect_ratio, eccentricity, solidity, circularity, edge_density, incidence_angle_deg, wind_ms]
  max_suspect_fraction: 0.02            # T4.3; warn when more gated card windows have r_bH <= 0 (placeholder)
  ridge_alpha: 1.0                      # T4.4 model settings: PLACEHOLDERS, untuned
  forest_n_estimators: 100
  forest_max_depth: 3
  random_state: 0
  interval_level: 0.95                  # f_lo / f_hi coverage (T4.5)
validation:                             # T5.2 / T5.6
  pair_max_minutes: 10                  # ground truth to device window midpoint, same plant (T5.2)
  surface_pair_minutes: 5               # PLACEHOLDER: adaxial + abaxial readings summed only if this close (E4 "a few minutes")
  gravimetric_match_hours: 1.0          # PLACEHOLDER: device windows this close to a weighing enter the gravimetric trend table (A45)
  loa_z: 1.96                           # Bland-Altman limits of agreement = bias +/- loa_z x SD of differences
  n_split_bins: 3                       # tertiles for the shape-deviation and g_s-range splits (T5.2)
synthetic:                              # T4.2; used only by ml/synthetic_data.py. EVERY VALUE IS INVENTED (not a measurement or calibration)
  k_wm2_per_lux: 0.01                   # only in a returned synthetic calibration dict; light_sources.led_lamp k stays null
  a_s_card: 0.95
  train_seed: 0
  test_seed: 1                          # held-out set
  n_shapes: 20
  correction_a: 0.8                     # f = 1 + a (1 - solidity) + b (aspect_ratio - 1) + c edge_density + N(0, noise_sd)
  correction_b: 0.15
  correction_c: 2.0
  correction_noise_sd: 0.05
  solidity_range: [0.7, 1.0]
  aspect_ratio_range: [1.0, 3.0]
  edge_density_range: [0.0, 0.1]
  d_min_mm_range: [20.0, 60.0]
  card_thickness_mm: 1.0
  noise: {leaf_temp_c: 0.1, card_temp_c: 0.1, air_temp_c: 0.05, humidity_pct: 0.5,
          pressure_hpa: 0.1, lux_fraction: 0.01, wind_ms: 0.02}   # per sample, not the uncertainty sigmas
  session_bias: false                   # true: one offset per session from the uncertainty sigmas; thermal offset shared by leaf and card
  conditions: {air_temp_c: 24.0, humidity_pct: 50.0, pressure_hpa: 995.0, irradiance_wm2: 120.0,
               wind_ms: 0.5, duration_min: 15.0, acclimation_minutes: 40.0}
  start_iso: "2026-11-09T10:00:00+00:00"
  roi_n_px: 20
  root_bracket_c: [-20.0, 40.0]         # root-finder search range around air temperature, deg C
  bracket_dry_g_s_mol: 1.0e-6           # analysis/detection_limit.py (T5.4); INVENTED: petroleum-jelly leaf stand-in
  bracket_wet_g_s_mol: 1000.0           # INVENTED: wet reference stand-in (r_s about 0)
  delta_t_sweep:                        # ml/delta_t_sweep.py (T4.3 follow-up); INVENTED settings
    irradiance_wm2: [60.0, 100.0, 200.0, 400.0, 800.0]   # incident on the card
    wind_ms: [0.8, 2.0]
    offset_scales: [0.0, 0.5, 1.0]      # x the uncertainty sigmas; 0 = per-sample noise only
    repeats: 10
    duration_min: 6.0
```

**calibration.json schema (decided in Session 3).** One JSON object; every key is optional. It mirrors the section and key names of config.yaml, and a non-null calibration value overrides the config value with the same section and key. Only calibration.json counts as calibrated.

```json
{
  "light_sources": {
    "<name>": { "k_wm2_per_lux": 0.0, "a_s_leaf": 0.0 }
  },
  "spatial": { "irradiance_factor_leaf": 0.0, "irradiance_factor_card": 0.0 },
  "card": { "a_s_card": 0.0, "emissivity": 0.0 }
}
```

Rules: k\_wm2\_per\_lux is a number above 0 and a\_s\_leaf is a number in (0, 1\]; the spatial factors are above 0. For led\_lamp, k stays null in config.yaml with no default and no fallback anywhere. The sunlight value in config.yaml is a reference value, not a calibration, so uncalibrated sunlight also needs allow\_uncalibrated\_k and carries the k\_uncalibrated flag. Values quoted in chat (for example 0.004) are examples, never defaults. Temperature offsets and ROIs get keys when their tasks (T7.4, T7.5) are written.

**data/raw/\<session\_id>/sensor\_log.csv columns:** session\_id, timestamp\_iso, timestamp\_corrected\_iso, monotonic\_s, air\_temp\_c, humidity\_pct, pressure\_hpa, lux, tsl\_visible, tsl\_infrared, tsl\_full\_spectrum, tsl\_gain, tsl\_integration\_ms, wind\_ms, wind\_raw, wind\_status, wind\_source, thermal\_frame\_idx, thermal\_min\_c, thermal\_max\_c, thermal\_mean\_c, leaf\_temp\_c, leaf\_temp\_median\_c, leaf\_temp\_std\_c, leaf\_n\_px, card\_temp\_c, card\_temp\_std\_c, card\_n\_px, wet\_ref\_temp\_c, wet\_ref\_n\_px, image\_file, qc\_flags.

**events.csv columns:** timestamp\_iso, label, note.

**run\_manifest.json fields:** session\_id, start\_iso, end\_iso, clock\_status, clock\_offset\_s, n\_samples, git\_commit, config\_snapshot, calibration\_sha256, package\_versions, sensor\_settings, throttled\_flags.

**data/raw/session\_metadata.csv columns:** session\_id, timestamp\_start, plant\_id, genotype, water\_treatment, reproductive\_stage, block, position, leaf\_position, light\_source, lamp\_distance\_mm, window\_blocked, acclimation\_minutes, fan\_state, working\_distance\_mm, incidence\_angle\_deg, card\_angle\_deg, lamp\_angle\_deg, lamp\_to\_card\_mm, background, card\_id, dry\_reference\_type, wet\_reference\_present, operator, porometer\_gs\_mol, notes.

**data/reference\_cards.csv columns:** card\_id, shape\_group, source (designed or traced\_leaf), material, thickness\_mm, coating, a\_s\_card, emissivity, area\_mm2, d\_min\_mm, d\_eq\_mm, aspect\_ratio, eccentricity, solidity, circularity, edge\_density, geometry\_source (phone or camera), photo\_path, notes.

**data/processed/rb\_training\_set.csv columns:** row\_id, session\_id, card\_id, shape\_group, window\_start\_iso, window\_end\_iso, n\_samples, card\_temp\_c, air\_temp\_c, surroundings\_temp\_c, lux, light\_source, rn\_wm2, c\_card, wind\_ms, convection\_regime, r\_bh\_card, r\_bh\_card\_sigma, r\_bh\_flat, f\_ratio, the geometry feature columns, photo\_gap\_sec, incidence\_angle\_deg, valid, reason.

**data/processed/leaf\_gs\_results.csv columns:** session\_id, plant\_id, genotype, water\_treatment, block, position, leaf\_position, reproductive\_stage, window\_start\_iso, window\_end\_iso, n\_samples, steady\_state, method, method\_role, convection\_regime, r\_bh, r\_bh\_lo, r\_bh\_hi, leaf\_temp\_c, air\_temp\_c, rh\_pct, pressure\_hpa, lux, rn\_wm2, latent\_wm2, vpd\_pa, r\_s, g\_s\_ms, g\_s\_mol, g\_s\_mol\_lo, g\_s\_mol\_hi, resolved, censor (none, below\_detection, above\_range), g\_s\_mol\_dl, valid, reason, flags, photo\_gap\_sec, jones\_ig, cwsi, light\_source, jones\_ig\_censor (none, above\_range; blank when not computed).

**data/validation/ground\_truth\_gs.csv columns:** plant\_id, timestamp\_iso, instrument (porometer or gravimetric), surface (adaxial, abaxial, both\_summed or whole\_leaf), g\_s\_mol, leaf\_position, notes.

## Appendix C. Glossary

- **Stomata / g\_s:** pores on the leaf surface and how open they are; g\_s is the conductance of water vapor through them.
- **Boundary layer / r\_b:** the thin film of still air on a surface; r\_b is how strongly it resists heat (r\_bH) or vapor (r\_bw) flow.
- **r\_s:** stomatal resistance, the inverse of g\_s.
- **VPD:** vapor pressure deficit, how much drier the air is than the leaf’s interior.
- **Rn, H, λE:** absorbed net radiation, heat carried away by air, and heat carried away by evaporation.
- **ROI:** region of interest, the group of thermal pixels that cover only the leaf, card or wet reference.
- **Dry and wet reference:** surfaces that cannot transpire (dry) or evaporate freely (wet), used to bracket leaf temperature.
- **Petroleum-jelly leaf:** a real leaf coated so it cannot transpire; it keeps its real shape, so its r\_b can be measured by the same algebra as a card.
- **Jones conductance index:** a relative g\_s measure from dry, leaf and wet temperatures.
- **Leave-one-shape-out validation:** testing the model on a card shape it never saw during training.
- **Bland–Altman plot:** a plot of device-minus-reference difference against the average, showing bias and spread.
- **Mixed-effects model:** a statistical model that treats repeated readings from the same plant correctly.
- **Steady state:** leaf temperature has stopped drifting, so the energy-balance equation applies.
- **I2C, STEMMA QT / Qwiic:** the two-wire sensor bus and its plug-in connectors.
- **CSI:** the Raspberry Pi camera connector.

* **Censored value:** a measurement known only to lie above or below a limit (here, g\_s below the detection limit). It is kept and analyzed as such, never deleted.
* **Detection limit:** the smallest g\_s the device can resolve under given conditions, from the uncertainty analysis.

## Appendix D. Revision log: external reviews

Three outside reviews were checked against the physics and this document. Every point is listed with its verdict and where the plan changed.

### Review 1 (five points)

- **λE ≤ 0 truncation: valid and important.** Not fatal as stated, but dropping those windows removes the closed-stomata data and biases stressed means upward, which shrinks the apparent drought effect. Changed: A15; T1.3 to T1.7, T5.1, T5.3, T5.4, T6.1; Appendix A validity rules; truncation test.
- **Pi clock without a network: valid in substance, partly overstated.** Raspberry Pi OS usually restores the last saved time, so expect a stale time rather than 1970 (verify), and sessions at home normally have Wi-Fi. Changed: B14; T7.2, T7.7; decision 15; manifest fields; config. Third review: the typed-offset and RTC parts are deferred.
- **Card and leaf radiative and airflow mismatch: valid.** The error is proportional (5–10% in Rn gives 5–10% in r\_b), not massive, but it is systematic across every training label. Changed: A16, E9, T7.8, T5.8; metadata fields; wind-mapping step; config.
- **Photos every 60 s versus sensor rows every 5 s: valid gap.** It also exposed a missing batch step for photo features. Changed: B15, T3.5; T3.4, T5.1; config.
- **SD-card load from many .npy files: partly valid.** Write bandwidth is only about 0.6 kB/s, so a bottleneck is unlikely, but hundreds of tiny files and power-loss corruption are real. Changed: B13; T0.1, T7.2; layout; schema.

### Review 2 (four points)

- **C is in s^0.5 m⁻¹, not s m⁻¹: correct on units.** My first fix was itself wrong (see review 3). Changed: A10, Appendix A equation 10, config, decisions 4 and 20.
- **The sunlight lux fallback must fail loudly for LED: correct.** Efficacy differs by roughly 2–3×, in either direction depending on the lamp spectrum. Changed: A7, T1.1, T7.6, config, decision 8, definition of done.
- **Add a free-convection term or flag low-wind windows: correct.** The earlier max(U, floor) rule was a crude patch. Changed: A11, T1.2, T1.7, T4.3, T5.2, config, flags, schemas, Section 8.3, decisions 11 and 19.
- **State that card\_direct is primary and ML secondary: correct.** It also fits the schedule, because card\_direct needs neither the wind sensor nor the camera. Changed: Sections 1 and 2, A9, T5.1 to T5.3, config methods, Section 8.4, definition of done, decision 18.

### Review 3 (six points)

- **The convection numbers contradict their own formula: correct.** C = √ν / (s × b × 0.66 × D\_H × Pr^(1/3)) ≈ 313 / (s × b), giving 156 for two faces with no enhancement, 104 for two faces with b = 1.5 and 208 for one face with b = 1.5. The earlier text gave 210 for two faces, and the config mixed s = 1 with faces = 2, which could put r\_b about 2× off. Changed: A10, config (one switch, physics.faces; enhancement factor 1.0 and unverified), T1.2, T1.7, Appendix A, decisions 4 and 20.
- **The card gate was oversold: correct.** At 40 W m⁻² the expected signal is about 2 °C, only about twice the placeholder sigma of 1 °C. It is now a placeholder set from calibrated noise. Changed: T1.3, config, decision 23.
- **Card versus leaf buoyancy: correct, with one change to the suggested test.** A coated leaf warms like the card, so it cannot detect a buoyancy mismatch alone; the test is solved r\_b against temperature difference at fixed wind. Changed: A17, E10, T5.9, T5.3, flags, config, decision 21.
- **Porometer sidedness: correct.** Changed: E4, T5.6, ground-truth schema, decision 22.
- **Deferred items were not marked: correct.** New \[DEFER\] tag on the Tobit fit, the typed-offset clock and RTC logic, fsync tuning and the position-swap check; the cheap versions stay (censored windows are kept, the clock is recorded, the light-field map comes first). Changed: tags in Section 0, T5.3, T5.8, B13, B14, T7.2, decisions 15 and 17, definition of done.
- **Cosmetic: correct on faces = 1, layout not reproducible.** T1.7 and Appendix A now say faces = 1 is set explicitly. Both former appendix tables were intact in the editor (header plus five and four rows), so the split was most likely a page break in an export; the logs are now lists under one heading, which cannot split mid-table.
