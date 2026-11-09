# gs_calculator (processing/gs_calculator.py, task T1.4)

## What it does
This is the module that produces the project's main number: **stomatal conductance, g_s**, which measures how open the leaf's pores (stomata) are. It takes the leaf's temperature, the air conditions, the light, and the boundary-layer resistance r_bH (from the card, the flat-plate formula, or the ML model). It then works out how much energy the leaf is spending on evaporating water, and from that, how open the stomata must be. A leaf that is cooler than expected is evaporating a lot, which means its stomata are open.

## How it works
1. **`LeafReading`** holds one window's means. These are the leaf temperature `leaf_temp_c`, air temperature `air_temp_c` and surroundings temperature `surroundings_temp_c` (all °C), `relative_humidity_pct` (%), `pressure_hpa` (hPa), `lux`, `light_source`, `r_bh` (s m⁻¹), and `r_bh_method` (`card_direct`, `ml_corrected` or `flat_plate`).
2. **`solve_gs(reading, cfg, cal)`** takes these steps:
   1. Checks the inputs. A missing or non-finite value, r_bH ≤ 0, pressure ≤ 0, RH outside 0–100, or negative lux gives `valid=False, reason=missing_input`. (Zero lux, darkness, is a real reading and is kept.)
   2. Converts light: E_s = c_leaf × k × lux, in W m⁻² (eq. 1). k and the leaf absorptivity a_s come from `get_light_source`, and c_leaf from `get_position_factor`. Their flags are copied to the result.
   3. Net radiation: Rn = a_s E_s + n ε_L σ (T_sur⁴ − T_s⁴), with temperatures in kelvin (eq. 2).
   4. Sensible heat: H = ρcₚ (T_s − T_a) / r_bH (eq. 3).
   5. Latent heat: λE = Rn − H (eq. 4).
   6. VPD = e_s(T_s) − e_s(T_a) × RH/100 (eq. 5). **If VPD ≤ 0, it adds the flag `vpd_nonpositive`** and carries on. This is not a failure (see "Why").
   7. **If λE ≤ 0, it stops with `censor=below_detection`** (valid).
   8. r_s = ρcₚ VPD / (γ λE) − r_bw, where r_bw = 0.93 × r_bH (eq. 8), γ = 0.0665 × P_hPa (eq. 6). **If r_s ≤ 0, it returns `censor=above_range`** (valid). A window with VPD ≤ 0 and λE > 0 always lands here, because the first term is ≤ 0.
   9. g_s = 1/r_s in m s⁻¹, then converts to mol m⁻² s⁻¹.
3. **`count_censor(results)`** counts the valid results under each censor label.

### What each result contains

| Field | Normal (`censor=none`) | `below_detection` (λE ≤ 0) | `above_range` (r_s ≤ 0) | Bad input (`missing_input`) |
| --- | --- | --- | --- | --- |
| valid | True | **True** | **True** | False |
| reason | ok | ok | ok | missing_input |
| censor | none | below_detection | above_range | none |
| rn_wm2, h_wm2, latent_wm2, vpd_pa | numbers | numbers | numbers | None |
| r_s | number > 0 | None | number ≤ 0 (kept) | None |
| g_s_ms, g_s_mol | numbers | None | None | None |
| g_s_mol_dl, g_s_lo, g_s_hi | None (filled by T1.5) | None (T1.5; dl is the upper bound) | None (T1.5) | None |
| flags | light and position flags | same, plus `vpd_nonpositive` if VPD ≤ 0 | same, plus `vpd_nonpositive` if VPD ≤ 0 | same |

A window with VPD ≤ 0 is never `censor=none`: with λE > 0 it is `above_range`, with λE ≤ 0 it is `below_detection`.

## Why it is done this way

### The idea: a leaf's energy budget has one unknown left
Energy in = energy out: **Rn = H + λE**. We measure Rn (from light and temperatures) and H (from the leaf–air temperature difference and r_bH), so the leftover, **λE = Rn − H**, is the energy spent on evaporation. Evaporation is driven by how dry the air is compared with the inside of the leaf (VPD), and it is held back by two resistances in a row: the stomata (r_s) and the boundary layer for water vapour (r_bw):

  λE = (ρcₚ / γ) × VPD / (r_s + r_bw)

Solving for r_s gives **r_s = ρcₚ VPD / (γ λE) − r_bw**, and **g_s = 1 / r_s**.

**Units check for r_s:** ρcₚ [J m⁻³ K⁻¹] × VPD [Pa] / (γ [Pa K⁻¹] × λE [J s⁻¹ m⁻²]) = J m⁻³ / (J s⁻¹ m⁻²) = **s m⁻¹** ✓.

### Why censoring instead of rejecting (A15)
A drought-stressed leaf with nearly closed stomata evaporates almost nothing, so its true λE is close to 0. Camera noise of a fraction of a degree then pushes the computed λE slightly above or below 0. If we threw away every window with λE ≤ 0, we would delete mostly the **most-stressed** readings. The stressed group's average g_s would come out too high, and the drought effect, which is the main result, would look smaller than it really is.

So λE ≤ 0 is not an error. It means "g_s is at or below what this setup can detect." The window is kept, marked `valid=True` and `censor=below_detection`, and its upper bound will be the detection limit `g_s_mol_dl` from the uncertainty module (T1.5). The statistics (T5.1, T5.3) must treat it as a value known only to be below that limit, never as missing.

r_s ≤ 0 is the opposite end: the leaf is cooler than even fully open stomata with this r_bH could explain. It is labelled `above_range` and also kept.

### Why VPD ≤ 0 is kept (censored above_range), not thrown away
VPD is the "drying power" of the air for this leaf. VPD = e_s(T_leaf) − e_a, so it depends on the **measured leaf temperature**. In humid air, a well-watered leaf that is evaporating hard is cool, and camera noise can push its reading below the dew point, giving VPD ≤ 0 while λE is still clearly positive. With λE > 0, eq. 8 gives r_s = (a number ≤ 0) − r_bw < 0, so the window means "more open than this setup can measure", the same as any r_s ≤ 0.

An earlier version marked these windows invalid (`reason=vpd_nonpositive`). The Session 8 review found that this throws away exactly the **coolest, most-open** windows, so averages came out too low. That is selecting data on the measured result, the same mistake as dropping λE ≤ 0, but at the other end. A simulation shows the size: at RH 97 %, 30 W m⁻² light and a true g_s of 0.3 mol m⁻² s⁻¹ (σ_leaf 0.5 °C), about 5 % of windows have VPD ≤ 0. Dropping them makes the pooled g_s about **15 % too low** on average. Keeping them gives about +6 % (test `test_vpd_nonpositive_windows_kept_and_dropping_them_biases_low`).

So now the spec says: VPD ≤ 0 with λE > 0 is `censor=above_range`, `valid=True`, with the **flag** `vpd_nonpositive` (it is no longer a reason code). These windows stay in every table, in pooled averages and in the Monte Carlo percentiles. The flag lets you count them: many flagged windows in one session point to a humidity or temperature calibration problem, or to real condensation.

When VPD ≤ 0 and λE ≤ 0 together (for example dew in the dark), the spec says nothing. The code applies the usual order, so the window is `below_detection` and still flagged (owner approved, 2026-11-08). Be careful with these windows: when water condenses on the leaf, vapour moves *onto* the surface instead of out through the stomata, so eq. 8 is not strictly valid and the window says little about g_s. The label is a convention, not proof of closed stomata (register A39). Analysis reports must print how many windows carry the `vpd_nonpositive` flag, split into λE > 0 (above_range) and λE ≤ 0 (below_detection), so they can be reviewed by hand. There is no threshold: the flag is set at exactly VPD ≤ 0. A small but positive VPD is solved normally; its large uncertainty is shown by the uncertainty module (T1.5).

### The mol conversion (eq. 8)
g_s in m s⁻¹ is a speed. Plant physiologists usually report it in mol m⁻² s⁻¹, the number of moles of gas passing through each square metre each second. Multiplying by the **molar density of air** (moles per cubic metre) converts one to the other. From the ideal gas law PV = nRT:

  n / V = P / (R T)

**Units:** Pa / (J mol⁻¹ K⁻¹ × K) = (N m⁻²) / (N m mol⁻¹) = **mol m⁻³** ✓. Then m s⁻¹ × mol m⁻³ = **mol m⁻² s⁻¹** ✓.

At 995 hPa and 24.0 °C: n/V = 99500 / (8.314 × 297.15) = **40.28 mol m⁻³**.

The spec uses **air** temperature in this step. Using leaf temperature (23.0 °C) instead would give 40.41 mol m⁻³, which is **0.3 % higher**. We keep the spec version (decision log, 2026-11-08).

## Worked example (spec Appendix A test fixture)
Inputs: leaf 23.0 °C, air 24.0 °C, surroundings 24.0 °C, RH 50 %, 995 hPa, E_s 120 W m⁻² (k = 0.01 × 12000 lux, c_leaf 1.0), a_s 0.85 (LED), ε_L 0.97, **faces 1** (set explicitly; config default is 2), r_bH 70 s m⁻¹.

1. **Rn** = 0.85 × 120 + 1 × 0.97 × σ × (297.15⁴ − 296.15⁴) = 102.0 + 5.74 = **107.7 W m⁻²**. The leaf is cooler than its surroundings, so it gains a little longwave.
2. **H** = 1210.02 × (23.0 − 24.0) / 70 = **−17.3 W m⁻²**. The leaf is cooler than the air, so the air heats the leaf.
3. **λE** = 107.7 − (−17.3) = **125.0 W m⁻²**, which is > 0, so the calculation continues.
4. e_s(23.0) = 2809.3 Pa; e_a = e_s(24.0) × 0.50 = 2983.8 × 0.50 = 1491.9 Pa; **VPD** = **1317 Pa**.
5. **γ** = 0.0665 × 995 = 66.17 Pa K⁻¹.
6. r_s + r_bw = 1210.02 × 1317.4 / (66.17 × 125.03) = **192.7 s m⁻¹**; r_bw = 0.93 × 70 = 65.1; **r_s = 127.6 s m⁻¹**.
7. **g_s** = 1 / 127.6 = **0.00784 m s⁻¹** × 40.28 mol m⁻³ = **0.316 mol m⁻² s⁻¹**.

The code gives r_s 127.59, g_s 0.007837 m s⁻¹, 0.3157 mol m⁻² s⁻¹. A ±0.5 °C leaf-temperature error (22.5 or 23.5 °C) gives 0.402 and 0.250 mol m⁻² s⁻¹. Those are about +27 % and −21 %, which is why the thermal calibration and uncertainty module matter.

**Censored example:** leaf 27.0 °C, lux 1000 (E_s 10 W m⁻²), otherwise the same. Rn = −9.1, H = +51.9, λE = −60.9 W m⁻² ≤ 0. Result: `valid=True`, `censor=below_detection`, with Rn, H, λE and VPD filled in and g_s `None`.

**VPD ≤ 0 example:** the worked example with RH 100 %. e_a = 2983.8 Pa, so VPD = 2809.3 − 2983.8 = −174.5 Pa, while λE = 125.0 W m⁻² (faces 1). r_s = 1210.02 × (−174.5) / (66.17 × 125.0) − 65.1 = −25.5 − 65.1 = −90.6 s m⁻¹ ≤ 0. Result: `valid=True`, `censor=above_range`, flag `vpd_nonpositive`, r_s −90.6 kept, g_s `None`.

**Same window with faces = 2 (independent hand check, test `test_hand_calculation_two_faces`).** Only the longwave term changes: 2 × 0.97 × σ × (297.15⁴ − 296.15⁴) = 11.49, so Rn = 113.49 and λE = 130.77 W m⁻². Then r_s + r_bw = 1210.02 × 1317.44 / (66.17 × 130.77) = 184.23, r_s = 119.13 s m⁻¹, g_s = 0.008394 m s⁻¹ = **0.338 mol m⁻² s⁻¹**. These numbers were worked out by hand and typed into the test, so the test does not depend on the code being checked.

## Methods paragraph (draft)
Stomatal conductance was computed from the leaf energy balance (Appendix A). Net radiation per one-sided leaf area was calculated from calibrated shortwave irradiance (lux × a source-specific calibration factor × a position factor), a leaf shortwave absorptivity of [0.85 / 0.5], and longwave exchange with surroundings (ε = 0.97, both faces). Sensible heat flux was H = ρcₚ (T_leaf − T_air) / r_bH, with r_bH from [the dry reference card]. Latent heat flux was the residual λE = Rn − H. Total stomatal resistance was obtained by inverting λE = (ρcₚ/γ) VPD / (r_s + r_bw), with r_bw = 0.93 r_bH and γ = 0.0665 P (hPa). Conductance was converted to molar units using the ideal-gas molar density at air temperature and measured pressure. Windows with λE ≤ 0 were retained as left-censored (below the per-window detection limit), and windows with r_s ≤ 0, including those where the leaf-to-air VPD was zero or negative while λE was positive, were retained as above the measurable range (the latter flagged); no window was discarded on the basis of the measured leaf temperature or any computed flux. Only windows with missing or physically impossible inputs (non-finite values, non-positive r_bH or pressure, RH outside 0–100 %, negative illuminance) were excluded. Calculations used 60-s window means. Software: Python 3.13 [versions from scripts/snapshot_versions.py].

## Limitations
1. **Two faces lumped into one path (faces = 2).** The formula treats the leaf as one stomatal resistance in series with one boundary-layer resistance. In reality, each face has its own stomata and its own boundary layer, working side by side. Lumping is exact only if both faces have equal g_s. Soybean has more stomata underneath.
   **Worked size of the error.** This is a hypothetical case using the worked example's λE = 125.0 W m⁻² and VPD = 1317 Pa, with an effective r_bH = 70 s m⁻¹ (both faces together), so each face has r_bH = 140 and r_bw = 130.2 s m⁻¹. The lumped formula gives g_s = 0.00784 m s⁻¹ (0.316 mol). If the true upper-to-lower split is 1 : 2, the same λE requires g_ad = 0.00272 and g_ab = 0.00544 m s⁻¹, a total of **0.00815 m s⁻¹ (0.328 mol)**. So the lumped result is **3.9 % too low**. A 1 : 3 split gives 9.0 % too low, and an equal split gives 0 % (exact). The error grows as the boundary-layer resistance becomes larger compared with r_s. It affects every genotype and treatment in the same direction, so relative comparisons are less affected than absolute values. (Register A28.)
2. **Molar conversion uses air temperature** (spec). Leaf temperature would give about 0.3 % more at the worked-example conditions.
3. **No detection limit or interval yet.** g_s_mol_dl, g_s_lo and g_s_hi stay `None` until T1.5. Until then, a `below_detection` result has no numeric upper bound.
4. **Near-saturated air.** VPD ≤ 0 windows give no point value of g_s; they are kept as `above_range` (or `below_detection` when λE ≤ 0) with flag `vpd_nonpositive`. A large share of flagged windows means the humidity or temperature calibration should be checked. Windows with small positive VPD are kept, but their g_s is very uncertain; T1.5's Monte Carlo interval shows that.
5. **Card and leaf share the air thermometer** (register A38). With card_direct, one air-temperature error changes both r_bH and H, so the two errors are correlated. Not modelled yet (planned in T6.1).
6. **Window means and upstream checks.** Steady state, wind and window means are checked upstream (T3.4), not here.
7. **ρcₚ fixed** (A18) and **γ uses a different cₚ** (A19).
