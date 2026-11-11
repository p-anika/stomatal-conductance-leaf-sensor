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
