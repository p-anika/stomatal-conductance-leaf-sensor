# gs_calculator (processing/gs_calculator.py, task T1.4)

## What it does
This is the module that produces the project's main number: **stomatal conductance, g_s**, which measures how open the leaf's pores (stomata) are. It takes the leaf's temperature, the air conditions, the light, and the boundary-layer resistance r_bH (from the card, the flat-plate formula, or the ML model). It then works out how much energy the leaf is spending on evaporating water, and from that, how open the stomata must be. A leaf that is cooler than expected is evaporating a lot, which means its stomata are open.

## How it works
1. **`LeafReading`** holds one window's means. These are the leaf temperature `leaf_temp_c`, air temperature `air_temp_c` and surroundings temperature `surroundings_temp_c` (all °C), `relative_humidity_pct` (%), `pressure_hpa` (hPa), `lux`, `light_source`, `r_bh` (s m⁻¹), and `r_bh_method` (`card_direct`, `ml_corrected` or `flat_plate`).
2. **`solve_gs(reading, cfg, cal)`** takes these steps:
   1. Checks the inputs. A missing or non-finite value, r_bH ≤ 0, pressure ≤ 0, or RH outside 0–100 gives `valid=False, reason=missing_input`.
   2. Converts light: E_s = c_leaf × k × lux, in W m⁻² (eq. 1). k and the leaf absorptivity a_s come from `get_light_source`, and c_leaf from `get_position_factor`. Their flags are copied to the result.
   3. Net radiation: Rn = a_s E_s + n ε_L σ (T_sur⁴ − T_s⁴), with temperatures in kelvin (eq. 2).
   4. Sensible heat: H = ρcₚ (T_s − T_a) / r_bH (eq. 3).
   5. Latent heat: λE = Rn − H (eq. 4). **If λE ≤ 0 (and VPD > 0), it stops with `censor=below_detection`.**
   6. VPD = e_s(T_s) − e_s(T_a) × RH/100 (eq. 5). **If VPD ≤ 0, it returns `valid=False, reason=vpd_nonpositive`** (no censor label). This check runs *before* the λE check, so in the code it comes first. γ = 0.0665 × P_hPa (eq. 6).
   7. r_s = ρcₚ VPD / (γ λE) − r_bw, where r_bw = 0.93 × r_bH (eq. 8). **If r_s ≤ 0, it returns `censor=above_range`.**
   8. g_s = 1/r_s in m s⁻¹, then converts to mol m⁻² s⁻¹.
3. **`count_censor(results)`** counts the valid results under each censor label.

### What each result contains

| Field | Normal (`censor=none`) | `below_detection` (λE ≤ 0) | `above_range` (r_s ≤ 0) | `vpd_nonpositive` (VPD ≤ 0) | Bad input (`missing_input`) |
| --- | --- | --- | --- | --- | --- |
| valid | True | **True** | **True** | False | False |
| reason | ok | ok | ok | vpd_nonpositive | missing_input |
| censor | none | below_detection | above_range | none | none |
| rn_wm2, h_wm2, latent_wm2, vpd_pa | numbers | numbers | numbers | numbers (vpd_pa ≤ 0) | None |
| r_s | number > 0 | None | number ≤ 0 (kept) | None | None |
| g_s_ms, g_s_mol | numbers | None | None | None | None |
| g_s_mol_dl, g_s_lo, g_s_hi | None (filled by T1.5) | None (T1.5; dl is the upper bound) | None (T1.5) | None | None |
| flags | light and position flags | same | same | same | same |

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

### Why VPD ≤ 0 is invalid, not censored
VPD is the "drying power" of the air for this leaf. If it is zero or negative, the air is as wet as (or wetter than) the inside of the leaf, so the leaf cannot evaporate at all and water would condense on it instead. Then eq. 8 has no meaning: dividing a zero or negative VPD by λE gives r_s ≤ 0, which would wrongly be read as "stomata extremely open" (`above_range`). The real cause is condensation or a humidity or temperature error. So the window gets `valid=False, reason=vpd_nonpositive` and no censor label. It is checked before λE, so it wins even when λE ≤ 0 too. There is no threshold: the cut is exactly VPD ≤ 0. A small but positive VPD is solved normally; it makes g_s very sensitive to sensor errors, and that is handled by the uncertainty module (T1.5), not by a cut here. `vpd_nonpositive` was added by the owner (2026-11-08) and is not in the spec's original list of reason codes.

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

## Methods paragraph (draft)
Stomatal conductance was computed from the leaf energy balance (Appendix A). Net radiation per one-sided leaf area was calculated from calibrated shortwave irradiance (lux × a source-specific calibration factor × a position factor), a leaf shortwave absorptivity of [0.85 / 0.5], and longwave exchange with surroundings (ε = 0.97, both faces). Sensible heat flux was H = ρcₚ (T_leaf − T_air) / r_bH, with r_bH from [the dry reference card]. Latent heat flux was the residual λE = Rn − H. Total stomatal resistance was obtained by inverting λE = (ρcₚ/γ) VPD / (r_s + r_bw), with r_bw = 0.93 r_bH and γ = 0.0665 P (hPa). Conductance was converted to molar units using the ideal-gas molar density at air temperature and measured pressure. Windows with λE ≤ 0 were retained as left-censored (below the per-window detection limit), and windows with r_s ≤ 0 as above the measurable range; no windows were discarded on the basis of the computed flux. Calculations used 60-s window means. Software: Python 3.13 [versions from scripts/snapshot_versions.py].

## Limitations
1. **Two faces lumped into one path (faces = 2).** The formula treats the leaf as one stomatal resistance in series with one boundary-layer resistance. In reality, each face has its own stomata and its own boundary layer, working side by side. Lumping is exact only if both faces have equal g_s. Soybean has more stomata underneath.
   **Worked size of the error.** This is a hypothetical case using the worked example's λE = 125.0 W m⁻² and VPD = 1317 Pa, with an effective r_bH = 70 s m⁻¹ (both faces together), so each face has r_bH = 140 and r_bw = 130.2 s m⁻¹. The lumped formula gives g_s = 0.00784 m s⁻¹ (0.316 mol). If the true upper-to-lower split is 1 : 2, the same λE requires g_ad = 0.00272 and g_ab = 0.00544 m s⁻¹, a total of **0.00815 m s⁻¹ (0.328 mol)**. So the lumped result is **3.9 % too low**. A 1 : 3 split gives 9.0 % too low, and an equal split gives 0 % (exact). The error grows as the boundary-layer resistance becomes larger compared with r_s. It affects every genotype and treatment in the same direction, so relative comparisons are less affected than absolute values. (Register A28.)
2. **Molar conversion uses air temperature** (spec). Leaf temperature would give about 0.3 % more at the worked-example conditions.
3. **No detection limit or interval yet.** g_s_mol_dl, g_s_lo and g_s_hi stay `None` until T1.5. Until then, a `below_detection` result has no numeric upper bound.
4. **Near-saturated air.** VPD ≤ 0 windows are invalid (`vpd_nonpositive`) and give no g_s. Windows with small positive VPD are kept, but their g_s is very uncertain; T1.5's Monte Carlo interval must show that.
5. **Window means and upstream checks.** Steady state, wind and window means are checked upstream (T3.4), not here.
6. **ρcₚ fixed** (A18) and **γ uses a different cₚ** (A19).
