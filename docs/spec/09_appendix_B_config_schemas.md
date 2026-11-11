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
