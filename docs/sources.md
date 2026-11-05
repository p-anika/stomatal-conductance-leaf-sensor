# Sources

Every citation or coefficient Claude Code uses. Status stays UNVERIFIED until the owner has opened the source and confirmed the number.

| Item | Value or claim | Source (as given) | Status | Used in |
| --- | --- | --- | --- | --- |
| config: rbw_to_rbh_ratio | 0.93 | Appendix B ("verify"). Rough check: (thermal diffusivity / vapour diffusivity)^(2/3) ≈ (21.5/24.0)^(2/3) ≈ 0.93 at 20 °C; diffusivities recalled, not looked up | UNVERIFIED | config.yaml physics |
| config: sunlight k_wm2_per_lux | 0.00833 W m⁻² per lux (≈120 lux per W m⁻²) | Appendix B, no source given | UNVERIFIED | config.yaml light_sources |
| config: enhancement_factor | 1.0 (sources said to give 1.4–1.5) | Appendix B; Campbell & Norman named in spec | UNVERIFIED | config.yaml convection |
| config: gamma_pa_per_k_per_hpa | 0.0665 Pa K⁻¹ per hPa | FAO-56 (γ = 0.665×10⁻³·P[kPa] kPa/K). Units: 0.665e-3 kPa/K per kPa = 0.665 Pa/K per kPa = 0.0665 Pa/K per hPa ✓ | UNVERIFIED | config.yaml physics |
