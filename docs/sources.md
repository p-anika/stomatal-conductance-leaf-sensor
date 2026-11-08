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
