# Assumptions and limitations register

Every assumption the measurement depends on, in one numbered list. Each coding task adds rows here.
Equation numbers refer to [equations.md](equations.md) (Appendix A). Source text: spec Section 4 (docs/spec/01_change_list.md).

How to read the columns:
- **Assumption**: what we take to be true, in plain words.
- **Eq.**: which equation(s) it affects.
- **Owner's decision**: the choice made. "Spec default" means the plan's default value; the owner has not separately confirmed it here yet.
- **Planned test**: how we will check it, with the task ID where the spec names one. "TBD" means no test is planned yet.
- **Status**: open (not yet checked), tested, or accepted as a limitation.

| ID | Assumption | Eq. | Owner's decision | Planned test | Status |
| --- | --- | --- | --- | --- | --- |
| A1 | A single leaf absorbs a fixed fraction a_s of incoming shortwave light, set per light source. The 0.23 albedo is for a whole grass canopy and is not used. The card has its own absorptivity a_s_card. | 1, 2, 9 | Spec default: leaf a_s 0.5 (window, sunlight), 0.85 (LED lamp). Card value still to measure or justify (matte black about 0.95). | Measure or justify a_s_card before card results are used. Leaf values: TBD. | open |
| A2 | Specific heat of air cp = 1005 J kg⁻¹ K⁻¹ and air density 1.204 kg m⁻³, so ρcp ≈ 1210 J m⁻³ K⁻¹. | 3, 7, 8, 9 | Spec default (config physics). Replaces 1010 in the old code. | Unit test that ρcp from config is 1210 (Phase 1). | open |
| A3 | Psychrometric constant γ [Pa/K] = 0.0665 × P [hPa], from the logged pressure (FAO-56). | 6, 7, 8 | Spec default. Removes the fixed 66.5. | Unit test against a hand calculation (Phase 1). | open |
| A4 | The surroundings radiate as a surface at T_sur, by default equal to air temperature. A window counts as a surface at its own, often colder, temperature. | 2 | Spec default: T_sur = T_air; optional thermal-camera reading of a wall or window. | TBD | open |
| A5 | Both leaf faces exchange heat and water (soybean has stomata on both sides). One switch, `faces`, sets longwave, r_b sidedness and the g_s convention together. | 2, 8, 10 | Spec default: faces = 2. | Compare with porometer readings summed over both faces (E4). | open |
| A6 | Boundary-layer resistance for water vapour is a fixed fraction of that for heat: r_bw = 0.93 × r_bH. | 7, 8 | Spec default 0.93, to verify in Campbell & Norman (1998). Effect on g_s about 2%. | Check the source (docs/sources.md, UNVERIFIED). | open |
| A7 | Lux is converted to W m⁻² with a calibrated factor k per light source. An LED lamp never uses the sunlight value (1/120); the code stops with an error instead. | 1 | Spec default: k must be calibrated. Sunlight fallback only for window or sunlight with allow_uncalibrated_k. | T1.1 tests (calibrated, fallback, LED refusal); light calibration against a reference meter (E2). | open |
| A8 | Ideal-gas conversion: g_mol = g_ms × P / (R × T_air), about 41 mol m⁻³ at room conditions. | 8 | Spec default. | Unit test (Phase 1). | open |
| A9 | The three r_b methods (flat_plate, card_direct, ml_corrected) feed the same g_s solver, so their results can be compared directly. | 9, 10, 11 | Spec default: card_direct primary; ml_corrected secondary, reported only if validated; flat_plate baseline. | Method comparison (E6). | open |
| A10 | Flat-plate baseline: r_b comes from a laminar Nusselt-number correlation, with sidedness s taken from `faces` and an enhancement factor b. | 10 | Spec default: b = 1.0 until verified (sources give 1.4 to 1.5); characteristic dimension d_min. | Check coefficients and sidedness in Campbell & Norman (docs/sources.md). | open |
| A11 | At low wind, free (buoyant) convection adds to forced convection: Nu = Nu_forced + Nu_free. Wind below the sensor's noise floor counts as zero forced flow. Every window is labelled forced, mixed or free. | 10 | Spec default: combination = sum; mixed_convection_ratio 0.3; flag baseline_unreliable for mixed or free windows. | Verify sum vs larger-of in Campbell & Norman; wind_check.py sets the noise floor. | open |
| A12 | Sensor errors can be carried through to a g_s interval by Monte Carlo, which sets a detection limit for each window. | Validity rules | Spec default (uncertainty values in config are placeholders). | Uncertainty module tests; replace the sigmas after thermal calibration (E1). | open |
| A13 | The Jones index and CWSI, from in-frame dry and wet references, can compare treatments without light calibration (Plan B). | 12, 13 | Spec default; needs a wet reference in the thermal frame. | TBD | open |
| A14 | Wind is never filled in silently. Missing wind gives reason no_wind; a manual value needs an explicit flag. | 10 | Spec default. | Unit test of the no_wind path. | open |
| A15 | λE ≤ 0 means g_s is at or below the detection limit, not an error. Such windows are kept and labelled censored, because dropping them would bias drought-stressed means upward. | Validity rules | Spec default: censor = below_detection or above_range; censoring-aware statistics (E7). | Truncation test (F1). | open |
| A16 | The card and the leaf receive the same irradiance and airflow, once position factors c_card and c_leaf are applied. | 1, 9 | Spec default: measure position factors; same tilt, distance and backdrop for card and leaf. | Light-field map (T7.8); position-swap check (T5.8, deferred). | open |
| A17 | The card's r_b equals the leaf's. This can fail at low wind, because a card warmer than the air has stronger buoyant flow than a transpiring leaf. | 9, 10 | Spec default: run sessions with the fan on low so flow is forced (decision 21); flag card_transfer_unreliable in mixed or free windows. | Card buoyancy check (T5.9). | open |
