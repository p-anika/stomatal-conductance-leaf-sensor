# Sources

Every citation or coefficient Claude Code uses. Status stays UNVERIFIED until the owner has opened the source and confirmed the number.

| Item | Value or claim | Source (as given) | Status | Used in |
| --- | --- | --- | --- | --- |
| config: rbw_to_rbh_ratio | 0.93 | Appendix B ("verify"). Rough check: (thermal diffusivity / vapour diffusivity)^(2/3) ≈ (21.5/24.0)^(2/3) ≈ 0.93 at 20 °C; diffusivities recalled, not looked up | UNVERIFIED | config.yaml physics |
| config: sunlight k_wm2_per_lux | 0.00833 W m⁻² per lux (≈120 lux per W m⁻²) | Appendix B, no source given | UNVERIFIED | config.yaml light_sources |
| config: enhancement_factor | 1.0 (sources said to give 1.4–1.5) | Appendix B; Campbell & Norman named in spec | UNVERIFIED | config.yaml convection |
| config: gamma_pa_per_k_per_hpa | 0.0665 Pa K⁻¹ per hPa | FAO-56 (γ = 0.665×10⁻³·P[kPa] kPa/K). Units: 0.665e-3 kPa/K per kPa = 0.665 Pa/K per kPa = 0.0665 Pa/K per hPa ✓ | UNVERIFIED | config.yaml physics |
| config: kinematic_viscosity_m2_s | 1.51×10⁻⁵ m² s⁻¹ (air, 20 °C) | Recalled by Claude from Campbell & Norman (1998) Table A.3; not looked up | UNVERIFIED | config.yaml convection; energy_balance_utils |
| config: thermal_diffusivity_m2_s | 2.15×10⁻⁵ m² s⁻¹ (air, 20 °C) | Recalled by Claude from Campbell & Norman (1998) Table A.3; not looked up | UNVERIFIED | config.yaml convection; energy_balance_utils |
| config: min_grashof_delta_t_c | 0.5 °C | Owner placeholder (2026-11-08); no source | UNVERIFIED | config.yaml convection; flat_plate_rbh |
| config: forced_nu_coeff, free_nu_coeff | 0.66 (laminar forced), 0.54 (free, heated plate facing up) | Appendix B, Campbell & Norman named | UNVERIFIED | flat_plate_rbh |
| claim: lower-face free convection | about half of the upper face for a heated horizontal plate | Recalled by Claude; no source checked | UNVERIFIED | explainer Limitations 3 |
| config: tetens_a_pa, tetens_b, tetens_c_c | 610.78 Pa, 17.27, 237.3 °C | Spec Appendix A eq. 5 (no citation given) | UNVERIFIED | saturation_vapor_pressure_pa |
| config: gas_constant_j_mol_k | 8.314 J mol⁻¹ K⁻¹ | Spec Appendix A eq. 8 | UNVERIFIED | molar_density_mol_m3 |
| config: gravity_m_s2 | 9.81 m s⁻² | Standard gravity, recalled | UNVERIFIED | Grashof number |
| config: stefan_boltzmann | 5.670374419×10⁻⁸ W m⁻² K⁻⁴ | Appendix B; equals 2π⁵k⁴/(15h³c²) with exact 2019 SI constants (recalled, check CODATA) | UNVERIFIED | net_radiation_wm2 |
| explainer: R_d, M_air, R_v, λ(0 °C), D_v | 287.05 J kg⁻¹ K⁻¹, 0.028965 kg mol⁻¹, 461.5 J kg⁻¹ K⁻¹, 2.501 MJ kg⁻¹, 24.0–24.2 mm² s⁻¹ | Recalled by Claude, used only in explainer derivations, not in code | UNVERIFIED | docs/explainers/energy_balance_utils.md |
| explainer: diffusivity temperature scaling | ν, D_H ∝ T^1.75 | Recalled by Claude, used only to size Limitation 4 | UNVERIFIED | docs/explainers/energy_balance_utils.md |
| check: spec C values 156/208/104/313 | code gives 154.04/205.38/102.69/308.08 s^0.5 m⁻¹ (within 3%) | Spec T1.7 item 6 | consistent with spec (inputs UNVERIFIED) | tests/test_energy_balance.py |
| config: card.a_s_card_placeholder | 0.95 (matte black card, shortwave absorptivity) | Appendix B comment "matte black about 0.95"; no source given | UNVERIFIED | config.yaml card; physics_baseline.card_absorptivity (only when allow_default_a_s_card is true) |
| config: uncertainty.g_s_cap_mol | 2.0 mol m⁻² s⁻¹ (top of measurable range) | Owner decision 2026-11-08; chosen above typical soybean maxima (about 1–1.5 mol m⁻² s⁻¹, recalled, not checked) | UNVERIFIED | uncertainty.py |
| config: uncertainty sigmas (all) | leaf 1.0 °C, air 0.5 °C, RH 3 %, lux 0.15 / 0.5, r_b 0.2, card Rn 0.1 | Spec Appendix B placeholders; sensor accuracy classes (MLX90640, BME280) recalled, datasheets not checked | UNVERIFIED (pre-calibration placeholders) | uncertainty.py |
| claim: Jones conductance index and CWSI | I_g = (T_dry − T_leaf)/(T_leaf − T_wet); CWSI = (T_leaf − T_wet)/(T_dry − T_wet) | Jones (1999), named from memory; citation details not checked | UNVERIFIED | jones_index.py, docs/explainers/jones_index.md |
| config: sensors.mlx90640 hfov_deg, vfov_deg | 55° × 35° field of view | Adafruit MLX90640 product listing (55° × 35° version), as given by the owner 2026-11-09 | VERIFIED from product listing (owner); still to be confirmed by imaging a known-size warm object at T7.4 | config.yaml sensors; thermal_roi.pixel_footprint_mm |
| config: qc.overflow_count | 65535 (16-bit full scale of a TSL2591 raw channel) | Definition of a 16-bit counter. The true saturation count depends on integration time and may be lower | OPEN: replace with the limit used by the installed adafruit_tsl2591 library (check on the Pi) | config.yaml qc; qc.py |
| config: qc.mad_to_sigma | 1.4826 = 1 / Φ⁻¹(0.75) | Derived (docs/explainers/qc.md); test checks it against scipy.stats.norm.ppf(0.75) | derived, not a citation | config.yaml qc; qc.py; thermal_roi.py |
| claim: leaf characteristic dimension ≈ 0.72 × leaf width | d = 0.72 × width for the boundary layer of a leaf | Campbell & Norman, recalled by Claude from memory; not checked | UNVERIFIED — OPEN ITEM (owner 2026-11-09): spec uses d_min directly; no change made | physics.characteristic_dimension; leaf_segmentation explainer Limitation 9 |
| method: Otsu threshold, excess-green index | Otsu (1979) between-class-variance threshold; ExG = 2G − R − B (often attributed to Woebbecke et al., 1995) | Named from memory; citation details not checked | UNVERIFIED | leaf_segmentation.py |
| config: segmentation (all keys) | blur 5 px, open 5 px, close 9 px, min_component 200 px, area 100–20000 mm², Canny 50/150, erode 5 px, max angle 60°, min_contrast 30 | Chosen by Claude as placeholders, tested only on drawn shapes | PLACEHOLDER — tune on Arducam photos (BLOCKED-CAM) | leaf_segmentation.py |
| config: synthetic (all keys) | correction a 0.8, b 0.15, c 2.0, noise SD 0.05; shape ranges; per-sample noise leaf/card 0.1 °C, air 0.05 °C, RH 0.5 %, pressure 0.1 hPa, lux 1 %, wind 0.02 m s⁻¹; k 0.01 W m⁻² lux⁻¹; a_s_card 0.95; conditions 24 °C, 50 %, 995 hPa, 120 W m⁻², 0.5 m s⁻¹ | Invented by Claude for software tests (owner approved 2026-11-09) | NOT A SOURCE — synthetic only; never quote as a measurement | ml/synthetic_data.py |
| claim: surface texture (edge density) changes leaf boundary-layer resistance | built into the synthetic f formula | No source; owner and Claude doubt it at leaf scale | UNVERIFIED — synthetic convenience only | ml/synthetic_data.py, docs/explainers/synthetic_data.md |
| method: Brent's root-finding method | scipy.optimize.brentq | SciPy documentation; Brent (1973), named from memory | UNVERIFIED citation details | ml/synthetic_data.py |
