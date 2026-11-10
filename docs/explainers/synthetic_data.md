# synthetic_data.py (T4.2) and the session loaders in data_io.py (T4.1)

> **Synthetic results prove the code works. They say nothing about real plants.**
> Every number the generator uses (the correction formula and its coefficients, the shapes, the noise, the lamp calibration k, the weather) is invented and lives in config.yaml under `synthetic`. If the pipeline recovers a planted answer from synthetic data, that shows the code does what it was written to do. It does **not** show that soybean leaves behave this way, that the real sensors are this accurate, or that the ML correction will work on real cards. Never quote a synthetic number as a result about plants.

## What it does
`ml/synthetic_data.py` makes fake data with a **known right answer**:
- **Fake reference cards** (shapes with features like solidity and aspect ratio), each with a hidden boundary-layer correction **f**. This is the number the later ML step (T4.3–T4.5) must learn to predict.
- **Fake sessions:** a card, or a leaf with a chosen stomatal conductance g_s, sitting under a lamp. The generator works out what temperature the card or leaf *would* reach from the physics, adds realistic sensor noise, and writes everything **in exactly the same file formats as a real session**.

Then the real analysis code can be run on it, and its answer compared with the planted truth.

`processing/data_io.py` gained the readers that every later step uses for real data:
- `list_sessions(raw_dir)` lists folders that contain run_manifest.json.
- `load_session(session_id, raw_dir)` reads metadata, the sensor table, events and the manifest, and returns the file paths.
- `load_ground_truth(path)` reads the hand-typed porometer table.
- `load_reference_cards()` now has a default path.

Each loader checks the Appendix B schema and stops with a `SchemaError` that names the file, the column and the row. The synthetic files go through exactly these checks (see "How it is checked").

## How it works
1. **Shapes (`make_synthetic_cards(n_shapes=20, seed=0)`).** For each shape, draw at random:
   - d_min 20–60 mm;
   - aspect ratio 1–3;
   - solidity 0.7–1.0;
   - edge density 0–0.1.

   The other features follow from these so they agree with each other:
   - d_max = aspect × d_min;
   - area = solidity × π/4 × d_min × d_max (an ellipse with dents);
   - d_eq = √(4A/π);
   - eccentricity = √(1 − 1/aspect²);
   - perimeter = ellipse perimeter × (1 + 2(1 − solidity));
   - circularity = 4πA/P².

   The output is two tables:
   - **cards:** exactly the columns of data/reference_cards.csv;
   - **truth:** f, kept separate so it can never leak into the "measured" data.
2. **The hidden correction.**
   f = 1 + a·(1 − solidity) + b·(aspect − 1) + c·edge_density + noise, with a = 0.8, b = 0.15, c = 2.0 and noise SD 0.05. In the default set, f runs from 1.14 to 1.71.
3. **Three card sets (`make_card_sets`):**
   - **train** (seed 0);
   - **held-out test** (seed 1): different shapes and IDs, for testing a model on cards it never saw;
   - **null** (same shapes as train, but f = 1 for every card): a model trained on it should find *no* improvement. If it claims one, it is fitting noise.
4. **Forward physics.** The real code goes *from* temperatures *to* r_b or g_s. The generator goes the other way: it picks r_b or g_s and finds the temperature that balances the energy budget. It uses a root-finder (`scipy.optimize.brentq`), which homes in on the temperature where "energy in − energy out" = 0.
   - **Card** (no evaporation, eq. 9): Rn(T) − ρc_p(T − T_a)/r_bH(T) = 0.
   - **Leaf** (eqs. 2–7): Rn(T) − ρc_p(T − T_a)/r_bH(T) − (ρc_p/γ)·VPD(T)/(r_s + 0.93·r_bH(T)) = 0, with r_s = (molar density)/g_s.
   - **True r_bH** = f × the flat-plate r_bH (eq. 10) for that shape's d_min and the wind. The flat-plate value depends on the surface temperature (warm air rises), so r_b and T are solved together.
   - Only the small helper functions in `energy_balance_utils` are shared with the real code. The inverse solvers (`solve_card_rbh`, `solve_gs`) are never used to make data, so a bug in them can't hide itself.
5. **Sessions (`simulate_card_session`, `simulate_leaf_session`).**
   - Default conditions: 15 minutes at 5 s per row (180 rows), air 24 °C, RH 50 %, 995 hPa, 120 W m⁻² lamp light, wind 0.5 m s⁻¹, 40 minutes of acclimation.
   - **Per-row random noise:** leaf and card 0.1 °C, air 0.05 °C, RH 0.5 %, pressure 0.1 hPa, lux 1 %, wind 0.02 m s⁻¹. This is the jitter of one reading, *not* the calibration uncertainty in the `uncertainty` section.
   - **Optional per-session bias** (`synthetic.session_bias: true`, off by default): one fixed offset for the whole session, drawn from the uncertainty sigmas (thermal 1.0 °C, air 0.5 °C, RH 3 %, lux 15 %). This imitates a calibration error. One thermal offset is shared by leaf and card because one camera sees both. Unlike noise, a bias does not average away; a test checks this.
   - Lux is written as E_s / k with a synthetic k = 0.01 W m⁻² per lux. That k is handed back in a separate calibration dictionary. **The real config.yaml keeps led_lamp k = null**, and a test checks this.
   - **Outputs:** sensor_log, events, metadata (built and checked by the same `make_session` the logger uses), run_manifest (`sensor_settings.synthetic = true`, so a loaded session reports `synthetic = True`), photo_features (one photo per 60 s), and a **truth** dictionary.
6. **`write_session(sim, root)`** writes a real-layout folder: `data/raw/<id>/` plus `data/processed/<id>/photo_features.csv` and `synthetic_truth.json`. The truth is never written inside `raw`. It refuses to overwrite an existing raw folder.

## Why it is done this way
- **A known answer is the only way to check the whole pipeline.** With real leaves nobody knows the true g_s. With synthetic data we do, so any error is the code's.
- **Same schemas, same loaders:** if synthetic files pass the real checks, then later steps can't secretly depend on something only synthetic data has.
- **Noise separate from bias:** random noise tests averaging; a fixed bias tests what calibration errors do. Using the 1.0 °C uncertainty as per-row noise would have made almost every window fail the steady-state test (slope standard error ≈ 0.19 °C min⁻¹).
- **The null and held-out sets** protect against the two classic ML mistakes: finding patterns that aren't there, and grading a model on data it trained on.
- **About edge density in f:** the formula makes surface texture change the boundary layer. Whether fine texture really changes heat transfer at leaf scale is **not verified**. It is in the formula only so the ML code has a texture feature to find. Nothing here says texture matters physically.

## Worked example
Card `SYN0_C00` (seed 0): d_min 45.48 mm, aspect 1.057, solidity 0.871, edge density 0.0405.
1. **f_deterministic** = 1 + 0.8 × (1 − 0.871) + 0.15 × (1.057 − 1) + 2.0 × 0.0405 = 1 + 0.103 + 0.008 + 0.081 = **1.192**. Plus noise, **f_true = 1.195**.
2. **Conditions:** air 24.0 °C, surroundings 24.0 °C, E_s = 120 W m⁻², a_s,card 0.95, wind 0.5 m s⁻¹, faces 2.
3. **Root-find:** T_card = **26.86 °C**. Flat-plate r_bH there is 35.60 s m⁻¹, so the true r_bH = 1.195 × 35.60 = **42.53 s m⁻¹**.
4. **Check by hand.** Absorbed shortwave 0.95 × 120 = 114.0 W m⁻². Longwave = 2 × 0.95 × 5.67×10⁻⁸ × (297.15⁴ − 300.01⁴) ≈ −32.8 W m⁻². So Rn ≈ 81.2 W m⁻². Sensible heat H = 1210 × (26.86 − 24.0) / 42.53 = 81.3 W m⁻² ✓. Units: (J m⁻³ K⁻¹ × K) / (s m⁻¹) = J m⁻² s⁻¹ = W m⁻² ✓.
5. **The same session also holds a leaf** (shape SYN0_C03, f_true 1.437) with g_s = 0.30 mol m⁻² s⁻¹. Its true r_bH is 37.83 s m⁻¹ and the root-find gives T_leaf = **23.05 °C**, about 1 °C below air, because it is evaporating.
6. **Noise off:** `solve_card_rbh` returns r_bH = 42.53 and `solve_gs` returns g_s = 0.300, both matching the truth to better than 1 part in a million.
7. **Noise on:** windows 3–14 are `ok` and get photo geometry. Windows 0–2 lack 3 minutes of history (T3.4 rule).
8. **Bias on** (seed 7): thermal +0.35 °C, air −1.17 °C, RH −1.0 %, lux × 0.90. g_s from the session mean then misses 0.30 by more than 5 %, and a test checks this.

## Methods paragraph (draft)
To verify the analysis software, synthetic datasets with known answers were generated (Python 3.13, NumPy 2.5.4, SciPy 1.18.1, pandas 3.1.1). Synthetic reference cards had randomly drawn shape descriptors and a hidden boundary-layer correction f = 1 + a(1 − solidity) + b(aspect ratio − 1) + c·edge density + ε. The coefficients were chosen arbitrarily (a = 0.8, b = 0.15, c = 2.0, ε ~ N(0, 0.05)) and are not physical estimates. Card and leaf temperatures were obtained by solving the steady-state energy balance with Brent's root-finding method, with r_bH = f × the flat-plate value and, for leaves, a specified stomatal conductance. Gaussian per-sample noise was added to every logged variable, optionally together with a per-session calibration bias. All synthetic files were written in the same schemas as real sessions and read with the same validated loaders. A null set (f = 1) and an independent held-out set (different random seed) were also generated. Synthetic data were used only to verify the software; no conclusion about plants is drawn from them.

## Limitations
1. **Synthetic results say nothing about real plants** (see the top of this page).
2. **The correction formula is invented.** Real cards may follow a different, non-linear rule, or depend on features not included here.
3. **Conditions are perfectly steady** apart from the noise: no lamp warm-up, no drifting air temperature, no leaf movement. Real sessions will be harder.
4. **Noise is independent from row to row.** Real thermal-camera noise can be correlated in time, which makes averages less precise than the synthetic data suggest.
5. **Surroundings = air temperature.** In a real lamp-lit room the walls, window and lamp housing are at other temperatures (A41), which changes longwave exchange.
6. **Geometry is plausible, not exact.** Perimeter and circularity come from a rule of thumb, not a drawn shape.
7. **No porometer rows are generated.** Which porometer surface label matches this project's g_s with faces = 2 is still open (A40).
8. **The per-session bias draws on placeholder sigmas** (`uncertainty` section). Replace them after calibration (T7.5).
