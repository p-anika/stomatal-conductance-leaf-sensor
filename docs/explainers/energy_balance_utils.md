# energy_balance_utils (and config)

Modules: [processing/energy_balance_utils.py](../../processing/energy_balance_utils.py) and [processing/config.py](../../processing/config.py). Tasks T1.1 and T1.2, Appendix A equations 1–10.

## What it does
A leaf in light heats up. Some of that heat leaves as warm air (sensible heat, H), and the rest leaves as evaporating water (latent heat, λE). This module holds the small formulas used to split that energy:
- how much light energy hits the leaf
- how much radiation the leaf absorbs in total
- how "thirsty" the air is (vapor pressure)
- how easily heat moves from a flat plate into the air (boundary-layer resistance r_bH)

`config.py` loads the settings file and the calibration file. It decides which light conversion factor k may be used, and it refuses to guess one for the LED lamp.

## How it works
**config.py**
1. `load_config(path)` reads config.yaml into a dict. It stops with an error if the file is missing.
2. `load_calibration(path)` reads calibration.json into a dict. If the file does not exist it logs a warning and returns `{}`. calibration.json uses the same section and key names as config.yaml, and a calibrated value always wins over the config value (schema in docs/decision_log.md, 2026-11-08).
3. `get_light_source(cfg, cal, name)` returns a `LightSource`: `a_s_leaf`, `k_wm2_per_lux`, `k_source`, `flags` and `sigma_lux_fraction`.
   - If calibration.json has k for this source, that k is used (`k_source = "calibration"`).
   - Otherwise, window or sunlight with `physics.allow_uncalibrated_k: true` may use the sunlight reference k (0.00833 W m⁻² per lux). In that case the result gets the flag `k_uncalibrated` and the larger light uncertainty `sigma_lux_fraction_uncalibrated` (0.5 instead of 0.15).
   - In every other case, including **led_lamp always**, it raises `CalibrationMissing` and tells you to run `calibration/light_calibration.py`.
   - If a_s_leaf comes from config rather than calibration, the flag `a_s_default` is added.
4. `get_position_factor(cfg, cal, "leaf" or "card")` returns the irradiance position factor c (A16). It is 1.0 with flag `c_position_default` until it has been measured.

**energy_balance_utils.py** (every function takes `cfg`, so no constant is typed inside a function):

| Function | In | Out |
| --- | --- | --- |
| `rho_cp_j_m3_k(cfg)` | — | ρcₚ, J m⁻³ K⁻¹ |
| `saturation_vapor_pressure_pa(temp_c, cfg)` | °C | Pa |
| `psychrometric_constant_pa(pressure_hpa, cfg)` | hPa | γ, Pa K⁻¹ |
| `molar_density_mol_m3(pressure_hpa, air_temp_c, cfg)` | hPa, °C | mol m⁻³ |
| `lux_to_irradiance_wm2(lux, k, c_position=1.0)` | lux, W m⁻² lux⁻¹ | E_s, W m⁻² |
| `net_radiation_wm2(E_s, a_s, T_s, T_sur, ε, faces, include_longwave, cfg)` | W m⁻², °C | Rn, W m⁻² |
| `flat_plate_conductances(d_m, wind_ms, delta_t_c, air_temp_c, cfg)` | m, m s⁻¹, °C | (g_forced, g_free), m s⁻¹ |
| `flat_plate_rbh(d_m, wind_ms, delta_t_c, air_temp_c, cfg)` | same | (r_bH in s m⁻¹, regime) |
| `equivalent_coefficient_c(cfg)` | — | C, s^0.5 m⁻¹ |
| `convection_flags(regime)` | regime | `()` or `("baseline_unreliable",)` |

Flat-plate steps:
1. Wind below `wind_noise_floor_ms` (0.1 m s⁻¹) is treated as 0.
2. Re = U·d/ν.
3. Gr = g·β·d³·ΔT/ν², with β = 1/T_air (K). ΔT here is max(|T_s − T_a|, 0.5 °C), so the free term never drops to zero.
4. Nu_forced = b·0.66·Pr^(1/3)·Re^(1/2) and Nu_free = 0.54·Pr^(1/4)·Gr^(1/4).
5. g = s·D_H·Nu/d for each term, with s = physics.faces. Then r_bH = 1/(g_forced + g_free). The two terms are added, as in the spec; the "max" option is refused.
6. Regime: "forced" if g_free < 0.3·g_forced; "free" if g_forced < 0.3·g_free (this includes no wind); "mixed" otherwise. Mixed and free results should carry the flag `baseline_unreliable`.

## Why it is done this way
Everything comes from config so a constant can only change in one visible, logged place. The LED refusal exists because an LED's lux-to-watts factor can differ from sunlight's by 2–3×, in either direction. A wrong k would bias every g_s, and you would not be able to see it. Wind below the sensor's noise floor is set to zero because a reading of 0.03 m s⁻¹ is noise, not air movement. The rejected alternative was returning an infinite r_bH when wind and ΔT are both zero. Instead, a ΔT floor keeps the free-convection term alive, and the result is flagged as unreliable.

## Derivations and units checks for every constant

### ρcₚ (volumetric heat capacity of air), eq. 3
- Ideal gas for dry air: ρ = P / (R_d·T), with R_d = R / M_air = 8.314 / 0.028965 = 287.05 J kg⁻¹ K⁻¹. (M_air is from memory: UNVERIFIED.)
- At 20 °C (293.15 K) and 101 325 Pa: ρ = 101 325 / (287.05 × 293.15) = **1.204 kg m⁻³**. This matches config.
- ρcₚ = 1.204 × 1005 = **1210.0 J m⁻³ K⁻¹**. This matches eq. 3.
- Units: kg m⁻³ × J kg⁻¹ K⁻¹ = J m⁻³ K⁻¹. In H = ρcₚ·ΔT / r_bH this gives J m⁻³ K⁻¹ × K ÷ s m⁻¹ = J m⁻² s⁻¹ = **W m⁻²** ✓.

### γ (psychrometric constant), eq. 6
- γ = cₚ·P / (ε·λ). Here ε = 0.622 is the ratio of the molar masses of water and air, and λ = 2.45 MJ kg⁻¹ is the latent heat (FAO-56 values with cₚ = 1.013 kJ kg⁻¹ K⁻¹).
- γ/P = 1013 / (0.622 × 2.45×10⁶) = 6.647×10⁻⁴ K⁻¹. Per hPa that is × 100 = **0.0665 Pa K⁻¹ hPa⁻¹**. This matches config.
- Units: (J kg⁻¹ K⁻¹ × Pa) ÷ (J kg⁻¹) = **Pa K⁻¹** ✓.
- Check: γ(1013 hPa) = 67.36 Pa K⁻¹ and γ(995 hPa) = 66.17 Pa K⁻¹.

### Tetens saturation vapor pressure, eq. 5
- Physics: the Clausius–Clapeyron equation, d ln e_s / dT = λ/(R_v·T²). If λ is held constant and you integrate from 0 °C, you get e_s = e_s(0)·exp[(λ/(R_v·T₀))·T_c/(T_c + 273.15)].
- That has the Tetens shape e_s = a·exp(b·T/(T + c)), with a = e_s(0 °C). With λ = 2.501 MJ kg⁻¹ and R_v = 461.5 J kg⁻¹ K⁻¹ (both from memory, UNVERIFIED) this gives b ≈ 19.8 and c = 273.15.
- Because λ really changes with temperature, the spec's coefficients b = 17.27 and c = 237.3 °C are an empirical fit, not a derivation. a = 610.78 Pa is e_s at 0 °C. The three numbers come from the spec and are UNVERIFIED (docs/sources.md).
- Units: b·T/(T + c) is °C/°C, so it has no units. The result has the units of a: **Pa** ✓.
- Checks: e_s(20 °C) = 2338 Pa (spec value ✓), e_s(23 °C) = 2809.3 Pa, e_s(24 °C) = 2983.8 Pa.

### r_bw / r_bH ratio = 0.93, eq. 7
- In forced laminar flow, Nu ∝ Pr^(1/3)·Re^(1/2), so g = D·Nu/d ∝ D·(ν/D)^(1/3)·ν^(−1/2) = D^(2/3)·ν^(−1/6). The same holds for vapor with Sc = ν/D_v in place of Pr.
- So g_v / g_H = (D_v/D_H)^(2/3), and r_bw / r_bH = (D_H/D_v)^(2/3).
- With D_H = 21.5 and D_v ≈ 24.0–24.2 mm² s⁻¹ at 20 °C (from memory, UNVERIFIED): (21.5/24.2)^(2/3) = 0.924 and (21.5/24.0)^(2/3) = 0.929, so **≈ 0.93** ✓.
- For free convection (Nu ∝ Pr^(1/4)) the same steps give the exponent 3/4, so the ratio is ≈ 0.915. The effect on g_s is about 2% (config comment).
- Units: a ratio of two resistances in s m⁻¹, so no units ✓. This module does not use the ratio; gs_calculator (T1.4) will.

### Molar density, eq. 8
- n/V = P/(R·T). At 20 °C and 1013 hPa: 101 300 / (8.314 × 293.15) = **41.57 mol m⁻³** (spec ≈ 41.6 ✓).
- Units: Pa ÷ (J mol⁻¹ K⁻¹ × K) = (N m⁻²) ÷ (N m mol⁻¹) = **mol m⁻³** ✓.

### Stefan–Boltzmann σ, eq. 2
- σ = 2π⁵k_B⁴ / (15·h³·c²) = 5.670374419×10⁻⁸ W m⁻² K⁻⁴. Since the 2019 SI redefinition, k_B, h and c are exact, so σ is too. I recalled this rather than looked it up, so it is UNVERIFIED until checked against CODATA.
- Units: W m⁻² K⁻⁴ × K⁴ = **W m⁻²** ✓.

### Convection: Re, Gr, Pr and the coefficient C, eq. 10
- Re = U·d/ν: (m s⁻¹ × m) ÷ (m² s⁻¹), no units ✓.
- Gr = g·β·d³·ΔT/ν²: (m s⁻² × K⁻¹ × m³ × K) ÷ (m⁴ s⁻²), no units ✓.
- Pr = ν/D_H = 15.1 / 21.5 = **0.7023**, no units. Both values are for air at 20 °C, recalled from Campbell & Norman Table A.3: UNVERIFIED.
- g = s·D_H·Nu/d: (m² s⁻¹) ÷ m = **m s⁻¹** ✓, so r = 1/g is in s m⁻¹ ✓.

**Deriving C** (forced flow only). Put Re = U·d/ν into Nu_forced and then into g:

g_forced = s·D_H·b·0.66·Pr^(1/3)·(U·d/ν)^(1/2) / d = s·b·0.66·D_H·Pr^(1/3)·√U / (√ν·√d)

r = 1/g = [√ν / (s·b·0.66·D_H·Pr^(1/3))] · √(d/U) = **C·√(d/U)**

Units of C: √(m² s⁻¹) ÷ (m² s⁻¹) = m s^−½ × s m⁻² = **s^½ m⁻¹** ✓. Multiplied by √(d/U) (√(m ÷ m s⁻¹) = s^½) this gives **s m⁻¹** ✓.

Numbers: √ν = √(1.51×10⁻⁵) = 3.886×10⁻³; Pr^(1/3) = 0.8889; 0.66 × 2.15×10⁻⁵ × 0.8889 = 1.2613×10⁻⁵. So C(s = 1, b = 1) = 3.886×10⁻³ / 1.2613×10⁻⁵ = 308.08.

Table of C, as computed by `equivalent_coefficient_c` with the config ν and D_H:

| s (faces) | b (enhancement) | C computed, s^0.5 m⁻¹ | spec T1.7 value | difference |
| --- | --- | --- | --- | --- |
| 2 | 1.0 | 154.04 | 156 | −1.3% |
| 1 | 1.5 | 205.38 | 208 | −1.3% |
| 2 | 1.5 | 102.69 | 104 | −1.3% |
| 1 | 1.0 | 308.08 | 313 | −1.6% |

All four are within the spec's 3% tolerance. The 313 in the spec corresponds to slightly cooler air (about 15 °C), where ν and D_H are smaller.

## Worked example
**Net radiation** (Appendix A fixture: E_s = 120 W m⁻², a_s = 0.85, ε = 0.97, faces = 1, leaf 23.0 °C, surroundings 24.0 °C):
- Shortwave: 0.85 × 120 = 102.00 W m⁻².
- Longwave: 297.15⁴ − 296.15⁴ = 1.0442×10⁸ K⁴, and 1 × 0.97 × 5.6704×10⁻⁸ × 1.0442×10⁸ = 5.74 W m⁻².
- Rn = **107.74 W m⁻²** (spec 107.7 ✓). The leaf is cooler than the room, so it gains a little longwave.

**Flat plate** (d = 0.05 m, T_s − T_a = 2 °C, air 22 °C, faces = 2, b = 1):
- At U = 1 m s⁻¹: Re = 0.05/1.51×10⁻⁵ = 3311, √Re = 57.54, Nu_forced = 0.66 × 0.8889 × 57.54 = 33.76, g_forced = 2 × 2.15×10⁻⁵ × 33.76/0.05 = 0.02903 m s⁻¹.
- Gr = 9.81 × (1/295.15) × 0.05³ × 2 / (1.51×10⁻⁵)² = 36 443, Gr^¼ = 13.82, Nu_free = 0.54 × 0.9155 × 13.82 = 6.83, g_free = 0.00587 m s⁻¹.
- r_bH = 1/(0.02903 + 0.00587) = **28.6 s m⁻¹**. Ratio g_free/g_forced = 0.20 < 0.3, so the regime is **forced**.

| wind, m s⁻¹ | r_bH, s m⁻¹ (faces 2) | regime |
| --- | --- | --- |
| 0 | 170.2 | free |
| 0.1 | 66.4 | mixed |
| 0.3 | 45.9 | mixed |
| 1.0 | 28.6 | forced |
| 2.0 | 21.3 | forced |

Note: with the 0.1 m s⁻¹ noise floor, "free" is only ever reached when the wind is read as zero. Any measured wind from 0.1 to about 0.45 m s⁻¹ (at these settings) is "mixed".

## Methods paragraph (draft)
Net radiation on the leaf was calculated as Rn = a_s·E_s + n·ε·σ·(T_sur⁴ − T_s⁴), where E_s = c·k·lux is the shortwave irradiance from the illuminance sensor and k is the lux-to-irradiance factor calibrated for each light source against a reference meter. Data under an LED lamp were never analysed with an uncalibrated k. Saturation vapor pressure followed the Tetens equation and the psychrometric constant was 0.0665·P (P in hPa; FAO-56). As a physics baseline, the heat boundary-layer resistance of a flat plate was computed from laminar forced convection (Nu = 0.66·Pr^(1/3)·Re^(1/2)) plus free convection (Nu = 0.54·Pr^(1/4)·Gr^(1/4)). Wind below the anemometer's noise floor (0.1 m s⁻¹) was treated as still air, and each window was labelled forced, mixed or free. Calculations used Python 3.13 (versions in docs/versions/). [Add the Campbell & Norman citation and package versions once verified.]

## Limitations
1. **ρcₚ is fixed at 20 °C and 1013 hPa.** At the worked example's 24 °C and 995 hPa, the real air density is 1.167 kg m⁻³, 3.1% lower. Using that would raise g_s by about 4% (0.316 to 0.329 mol m⁻² s⁻¹). The card method uses the same ρcₚ to get r_bH, so part of this error cancels there.
2. **cₚ in γ differs from cₚ in ρcₚ.** FAO-56 γ uses cₚ = 1013, while ρcₚ uses 1005. Recomputing γ with 1005 gives 0.06595 (−0.8%), which changes g_s in the worked example by −1.2%.
3. **Free convection is counted the same on both faces.** The 0.54 coefficient is for a heated plate facing up (or a cooled plate facing down). The other face is roughly half as effective (from memory, UNVERIFIED). If so, with s = 2 the model overstates free convection. At d = 0.05 m and ΔT = 2 °C, r_bH would be 33% higher at zero wind, 7% higher at 0.3 m s⁻¹ and 4% higher at 1 m s⁻¹.
4. **ν and D_H are constants at 20 °C.** C ∝ ν^(1/6)·D_H^(−2/3). If both diffusivities scale as about T^1.75 (from memory, UNVERIFIED), C changes by about −2.9% at 30 °C and +3.1% at 10 °C, or roughly 1% per 3–4 °C.
5. The 0.5 °C ΔT floor in the Grashof number is a placeholder. It only matters when the plate is within 0.5 °C of the air.
6. The flat plate is a baseline. Real leaves are not flat, smooth plates, which is the reason the card and ML methods exist.
7. ν, D_H, the Nusselt coefficients, b = 1.0 and the Tetens coefficients are not yet checked in Campbell & Norman (docs/sources.md).
