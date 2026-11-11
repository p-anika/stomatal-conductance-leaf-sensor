# card_buoyancy_check (analysis/card_buoyancy_check.py)

## What it does
The card_direct method borrows the reference card's boundary-layer resistance r_bH and uses it
for the leaf. But a black card under the lamp runs several degrees warmer than the air, while a
leaf runs within about 1 °C. Warm air rises off a hot surface (**free convection, or
buoyancy**), and this extra air movement carries heat away and lowers r_bH. So the card's
r_bH may be lower than the r_bH the same shape would have at a leaf's temperature difference.

This script measures how much the card's r_bH changes with its temperature difference. It flags
any card shape and wind setting where carrying the card's r_bH over to a leaf would be more than
15 % wrong (`convection.card_transfer_max_change`).

Run it with `python -m analysis.card_buoyancy_check` (add `--synthetic` for synthetic data).

**What you must measure:** each card shape at **three or more lamp levels** at **one fixed wind
setting** (T10.2). Each session is one lamp level. A group with fewer is reported as
`too_few_lamp_levels` and gets no flag.

## How it works
Input: the valid card windows in `rb_training_set.csv` (T4.3). Groups are card × wind setting,
where the wind setting is the session's median wind rounded to 0.1 m s⁻¹.

1. **Primary check: r_bH against absorbed shortwave.** SW_abs = a_s_card × c_card × k × lux
   [W m⁻²]. The lamp setting is what we control, and the card's temperature follows from it, so
   absorbed light is the honest (exogenous) x-axis.
   - The fit is a weighted straight line (weights 1/σ², σ = `r_bh_card_sigma`) with a 95 %
     confidence interval on the slope.
   - `sw_rel_change` = slope × (SW range) / mean r_bH is the fractional change across the lamp
     levels used.
   - Buoyancy shows as a **negative** slope: more light, warmer card, lower r_bH.
2. **Secondary: r_bH against |T_card − T_air|.** **Caveat:** r_bH = ρc_p ΔT / Rn is itself
   computed from ΔT. Any temperature error therefore appears on both axes and can create a
   slope by itself (see also A42). This number is reported but does not decide anything.
3. **Nusselt-form fit (this decides).**
   - Derivation (Appendix A eq. 10): the free-convection Nusselt number is
     Nu_free = 0.54 Pr^¼ Gr^¼, and the Grashof number Gr = g β d³ ΔT / ν² is proportional to
     ΔT. So the free-convection conductance g_free ∝ ΔT^¼. Forced convection does not depend on
     ΔT.
   - The conductance is the sum, **g_bH = 1/r_bH = a + b × |ΔT|^¼**. This is a straight line in
     ΔT^¼, fitted by weighted least squares in conductance space (σ_g = σ_r / r_bH²).
   - Units: a in m s⁻¹, b in m s⁻¹ °C^−¼.
   - Fractional change from the card's median ΔT to a leaf-like ΔT:
     **frac_change = g(ΔT_card) / g(ΔT_leaf) − 1** (positive means the leaf's r_bH is larger).
     It is reported at 1.0 °C (this decides) and at 0.5 °C.
   - The 95 % interval comes from 2000 draws of (a, b) from their fitted covariance.
   - **Pre-registered rule:** `card_transfer_unreliable` when |frac_change at 1.0 °C| > 0.15.
4. **Model comparison:** the same fractional change predicted by the flat-plate formula
   (`frac_change_model`). The correction factor f cancels in the ratio.

The output is a per-(card, wind) table. **Open item:** stamping the flag onto card_direct
windows in `leaf_gs_results.csv` (in aggregate_results, T5.1) is not done yet. It must be done
before any real leaf results are analysed.

## Why it is done this way
- **Light as the primary x-axis.** Avoids the false slope that comes from putting ΔT on both
  axes.
- **The ΔT^¼ form.** It is the physics of free convection and is steep near ΔT = 0. A straight
  line in ΔT, fitted at card ΔTs of 3–8 °C, would underestimate how much r_bH changes when you
  extrapolate down to a leaf's 0–1 °C. Both are reported, but the Nusselt form decides.
- **Rejected alternatives:**
  - Deciding from the straight-line fit (wrong shape near 0).
  - Deciding from the ΔT slope (the same noise is on both axes).
  - Changing aggregate_results in this task (owner decision: table first, wiring as an open item).

## Worked example
Synthetic card SYN7_C00 (d_min = 45 mm, wind 0.3 m s⁻¹, four lamp levels 100–800 W m⁻², data
generated from the flat-plate model with f = 1.2):

1. The card's median ΔT is **7.64 °C**.
2. Absorbed-light slope: **−0.0106 s m⁻¹ per W m⁻²**, so r_bH falls 14.7 % across the lamp levels.
3. Nusselt fit, then the change to a 1 °C leaf: frac_change = **+0.154**. The leaf's r_bH would
   be 15.4 % larger than the card's. The flat-plate model gives the same 0.154 (the data were
   made from it).
4. 0.154 > 0.15, so this shape at 0.3 m s⁻¹ is **card_transfer_unreliable**. The second card
   (SYN7_C01, 56 mm, median ΔT 6.71 °C) gives 0.147 and is not flagged.

At 0.5 °C the change is about +0.20 for both cards.

## Methods paragraph (draft)
To test whether the reference card's boundary-layer resistance transfers to leaves, each card
shape was measured at three or more lamp irradiances at one fixed wind setting. Within each card
and wind setting, the card's r_bH was regressed on absorbed shortwave irradiance (weighted least
squares, weights from the r_bH uncertainty). This was the primary check, because irradiance was
set by the experimenter. A regression on the card–air temperature difference was secondary. The
conductance 1/r_bH was fitted to a + b|ΔT|^¼, the form implied by forced plus free convection
over a flat plate. The fractional change in r_bH between the card's median temperature
difference and a leaf-like 1 °C was computed, with a 95 % interval from the parameter
covariance. Card shapes and wind settings with a change above 15 % were flagged as unreliable
for the card-direct method (pre-registered). Analyses used Python 3.13.12, statsmodels 0.15.0
and NumPy 2.5.4.

## Limitations
- **Buoyancy is not symmetric.** The fit uses |T_card − T_air|, but buoyancy depends on
  direction. A plate warmer than the air and facing up has rising air that strengthens
  convection; a leaf cooler than the air has sinking air that is partly trapped and weakens it.
  The check is only a rough guide for leaves, especially transpiring leaves cooler than air.
- **Few lamp levels.** Three or four lamp levels give few degrees of freedom; the interval on
  the change is wide with real noise.
- **ΔT^¼ is the laminar flat-plate form.** Real cards in a small room may not follow it exactly.
  The fit only extrapolates within that assumed form.
- **One wind setting per group.** The result applies only to the wind settings measured.
- **Not yet applied to leaf results** (open item above).
