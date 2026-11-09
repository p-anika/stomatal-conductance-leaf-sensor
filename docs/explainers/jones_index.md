# jones_index (processing/jones_index.py, task T1.6)

## What it does
This is a backup way (Plan B) to compare how open the stomata are, **without** needing calibrated light or r_b. It puts the leaf's temperature on a scale between two reference surfaces in the same thermal image: a **wet** reference (evaporating as fast as possible, so the coolest a leaf could be) and a **dry** reference (no evaporation, so the warmest a leaf could be).

## How it works
`conductance_index(t_dry_c, t_leaf_c, t_wet_c, cfg)` returns `JonesResult(i_g, cwsi)`. Each part is an `IndexValue(value, valid, reason, censor)`.

1. If any temperature is missing or not a number, both parts are invalid with reason `missing_input`.
2. If T_dry − T_wet < 1.0 °C (`jones.min_dry_wet_diff_c`), both parts are invalid with reason `references_not_bracketing`. The two references are too close to make a scale.
3. **CWSI** = (T_leaf − T_wet) / (T_dry − T_wet) (eq. 13). Near 0 means the leaf is like the wet reference (open stomata); near 1 means it is like the dry reference (closed).
4. **I_g** = (T_dry − T_leaf) / (T_leaf − T_wet) (eq. 12). This grows with stomatal conductance. If T_leaf − T_wet < 0.2 °C (`jones.min_leaf_wet_diff_c`), the bottom of the fraction is nearly zero, so I_g is huge and meaningless. It is then returned as `value=None, valid=True, censor=above_range`.

The two are linked: **I_g = 1/CWSI − 1** (divide the top and bottom of eq. 12 by T_dry − T_wet).

## Why it is done this way
- **No clipping.** Sensor noise can make CWSI slightly below 0 or above 1, or I_g slightly below 0. Those values are kept exactly as measured. If we set every CWSI above 1 to exactly 1, stressed leaves (CWSI near 1) would lose their high-side noise but keep their low-side noise. Their average would then be pushed down, making them look less stressed than they are. The same reasoning as A15 applies.
- **Units:** both are a temperature divided by a temperature, so they have no units ✓.

## Worked example
T_dry = 31.5 °C, T_leaf = 24.0 °C, T_wet = 21.0 °C.
- CWSI = (24.0 − 21.0) / (31.5 − 21.0) = 3.0 / 10.5 = **0.286**
- I_g = (31.5 − 24.0) / (24.0 − 21.0) = 7.5 / 3.0 = **2.5**
- Check: 1/0.286 − 1 = 2.5 ✓

Out of range, kept as measured: T_dry 30.0, T_leaf 30.4, T_wet 22.0 gives CWSI = 8.4/8.0 = **1.05** and I_g = −0.4/8.4 = **−0.048**.

## Methods paragraph (draft)
As a calibration-independent comparison, the thermal conductance index I_g = (T_dry − T_leaf)/(T_leaf − T_wet) and the crop water stress index CWSI = (T_leaf − T_wet)/(T_dry − T_wet) were computed from dry and wet reference surfaces in the same thermal frame (Jones 1999 — citation UNVERIFIED). Values outside the nominal range were retained without clipping. Windows in which the references differed by less than 1 °C were excluded as non-bracketing, and I_g was treated as above the measurable range when the leaf was within 0.2 °C of the wet reference.

## Limitations
1. **The dry reference must absorb light like a leaf.** A matte black card (a_s ≈ 0.95, versus a leaf's 0.5–0.85) runs hotter than a non-transpiring leaf would. That makes T_dry too high, CWSI too low and I_g too high. The correct dry reference is the petroleum-jelly-coated leaf (T10.3). There is no code guard; this is recorded as a limitation (register A35).
2. **The wet reference must stay fully wet** and be the same size and shape as the leaf; otherwise T_wet is too high.
3. I_g and CWSI depend on light, wind and humidity, so they compare leaves **within** a session well, but **between** sessions only roughly.
4. The 0.2 °C and 1.0 °C thresholds are from the spec and should be revisited once thermal noise is measured (T7.5).
