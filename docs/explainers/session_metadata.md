# session_metadata (acquisition/session_metadata.py, T2.1) and the card manifest (processing/data_io.py, T2.2)

## What it does
Sensors record numbers, but a number is useless if you don't know which plant, which treatment and which set-up it came from. At the start of every measurement session this module asks for (or takes as command-line flags) everything that is not a sensor reading: the plant, its genotype and watering treatment, its growth stage, where it sits in the randomised layout, the light and fan set-up, the reference cards and who ran the session. It checks every answer, so a typo can't create a fake new group (for example "Williams 82" next to "Williams_82"). It then saves one row to `data/raw/session_metadata.csv` and a copy as `data/raw/<session_id>/session.json`.

The second part, `load_reference_cards`, reads `data/reference_cards.csv`, the list of reference cards (shape, material, coating, measured absorptivity and geometry), and checks it.

## How it works
1. **Answers come in as text.** From flags (`--plant-id HK-03`), from prompts, or from a dictionary in code.
2. **`parse_field(name, value, cfg)`** checks one field and converts it (text to a number, "yes" to `True`, "Bench A-3" to `bench_a_3`). A bad value raises `ValueError` with a message that names the field, shows what was typed and says what is allowed.
3. **`validate_metadata(raw, cfg)`** checks all fields and the lamp rule (lamp fields are needed for `led_lamp` and must be blank otherwise). It reports **every** problem at once, one per line.
4. **`make_session(raw, cfg, now)`** adds `timestamp_start` (local time **with** its UTC offset) and builds `session_id` = plant_id + "_" + start time, with ":" swapped for "-" so it works as a folder name.
5. **`start_session(raw, cfg, root, manifest_path)`**:
   1. refuses if `session_metadata.csv` exists with any header other than the current one (see "Why");
   2. refuses if `session.json` for this session already exists;
   3. if `dry_reference_type` is `card`, warns (does not stop) when `card_id` is not in the card manifest;
   4. writes `session.json`, then appends the CSV row (creating the file with its header if it is new).
   Nothing is written if any check fails.
6. **Command line:** `python -m acquisition.session_metadata --plant-id HK-03 --genotype Houjaku_Kuwazu ...`. Any field not given is asked for. A bad typed answer is explained and asked again; a bad flag stops the program with the message. `prompt_for_session()` asks for everything and is what `combined_logger.py` calls.
7. **`load_reference_cards(path)`** returns `{card_id: ReferenceCard}`. It stops with `ReferenceCardError` (naming the row and column) if a required column is missing, a column is unknown, a card_id is blank or repeated, a choice is not allowed, or a number is out of range.

## Every session field, in plain words

Allowed choices come from `config.yaml → enumerations`, so the list can't drift between the code and the config.

| Field | What it means | Allowed / units | Example |
| --- | --- | --- | --- |
| session_id | Unique name of this session; also the name of its folder. Made automatically. | plant_id + start time | `HK-03_2026-11-08T14-30-05-05-00` |
| timestamp_start | When the session started, with the time-zone offset so times from different days and clocks compare correctly. Made automatically. | ISO 8601 | `2026-11-08T14:30:05-05:00` |
| plant_id | Label on the pot. | letters, digits, `-`, `_` | `HK-03` |
| genotype | Soybean variety. | `Houjaku_Kuwazu`, `Williams_82` | `Houjaku_Kuwazu` |
| water_treatment | Watering group of the plant. | `well_watered`, `drought_stressed` | `drought_stressed` |
| reproductive_stage | Growth stage (see "Growth stages" below). Vegetative stages are counted by fully developed trifoliate nodes; R1 (first flower) to R8 (full maturity) are the reproductive stages. | `VE`, `VC`, `V1`–`V15`, `R1`–`R8` | `V3` |
| block | Number of the randomised block the plant is in. Blocks spread the window's light gradient evenly over the treatments (spec C3). | whole number ≥ 1 | `2` |
| position | Where the plant sits inside its block. Stored lowercase with underscores. | free label | `bench_a_3` |
| leaf_position | Which leaf on the plant was measured. Stored lowercase with underscores. | free label | `top_trifoliate` |
| light_source | What lit the leaf. | `window`, `led_lamp`, `sunlight` | `led_lamp` |
| lamp_distance_mm | Distance from the lamp to the leaf. Only for `led_lamp`, blank otherwise. | mm, > 0 | `300` |
| window_blocked | Was the window covered, so only the lamp lit the leaf? | true / false | `true` |
| acclimation_minutes | How long the plant sat under this light before logging started. Stomata take time to adjust. | minutes, ≥ 0 | `35` |
| fan_state | Fan setting (it changes the wind, and so the boundary layer). | `off`, `low`, `high` | `low` |
| working_distance_mm | Distance from the camera lens to the leaf. Sets how many mm one pixel covers. | mm, > 0 | `200` |
| incidence_angle_deg | How far the leaf is tilted away from facing the camera (0 = facing it straight on). | degrees, 0–90 | `10` |
| card_angle_deg | Same tilt for the reference card. Optional. | degrees, 0–90 | `10` |
| lamp_angle_deg | How far the lamp is turned away from pointing straight at the leaf. Only for `led_lamp`. | degrees, 0–90 | `0` |
| lamp_to_card_mm | Distance from the lamp to the reference card. Only for `led_lamp`. | mm, > 0 | `310` |
| background | What is behind the leaf in the thermal image. | `neutral_matte`, `other` | `neutral_matte` |
| card_id | Which reference card was used (an ID in `data/reference_cards.csv`). Optional. | text | `card01` |
| dry_reference_type | The "dry" reference for the Jones index (something that does not evaporate). | `none`, `card`, `vaseline_leaf` | `card` |
| wet_reference_present | Was a wet reference (fully evaporating surface) in view? | true / false | `true` |
| operator | Initials of the person running the session. | text | `AP` |
| porometer_gs_mol | Stomatal conductance measured by a porometer on the same leaf, for checking the camera method. Optional. | mol m⁻² s⁻¹, ≥ 0 | `0.31` |
| notes | Anything unusual. Optional. | text | `first test session` |

True/false fields accept `true/false`, `yes/no`, `y/n` and `1/0`, in any capitals.

### Growth stages (reproductive_stage)
Before it flowers, a soybean plant is staged by **counting its nodes that carry a fully developed trifoliate leaf** (a leaf made of three leaflets). A node counts as fully developed once the leaf at the node above it has unrolled far enough that its leaflet edges no longer touch.

| Stage | What you see on the plant |
| --- | --- |
| VE | Emergence: the seedling's cotyledons (seed leaves) are above the soil. |
| VC | The two unifoliate leaves (the first, single-leaflet pair) are unrolled; no trifoliate node is fully developed yet. |
| V1, V2, … V15 | V*n* = *n* nodes with a fully developed trifoliate leaf. Example: three such nodes → `V3`. |
| R1 – R8 | Reproductive stages, from R1 (one open flower anywhere on the plant) to R8 (full maturity). |

Count only the main stem. Write the stage exactly as listed (capital letters, no space): `V3`, not `v3` or `V 3`. A plant with more than 15 trifoliate nodes that has not flowered is not in the list; add `V16` and so on to `config.yaml` if that happens. These definitions follow the standard soybean staging system and should be checked against a printed staging guide before the paper is written.

## Every card-manifest column, in plain words

The first 11 columns (spec T2.2) must be in the file. The 8 geometry columns (spec Appendix B) may be left out until the card photos are measured (T3.1, T3.5). Numbers may be blank until measured; a number that is filled in must be in range.

| Column | What it means | Allowed / units | Example |
| --- | --- | --- | --- |
| card_id | Unique name of the card. Must be filled, never repeated. | text | `card01` |
| shape_group | Shape family, for grouping cards in the ML training set. Must be filled. | text | `leaflet` |
| source | Where the shape came from. | `designed`, `traced_leaf` | `traced_leaf` |
| material | What the card is made of. Must be filled. | text | `card stock` |
| thickness_mm | Card thickness. | mm, > 0 | `0.3` |
| coating | Surface finish. Must be filled. | text | `matte black` |
| a_s_card | Fraction of sunlight-type (shortwave) light the card absorbs. Measured, never guessed. | 0 < a ≤ 1 | `0.94` |
| emissivity | How well the surface gives off heat radiation (1 = perfect). | 0 < ε ≤ 1 | `0.95` |
| area_mm2 | One-sided area of the card. | mm², > 0 | `1500` |
| d_min_mm | Short side of the smallest box around the card (the boundary-layer length). Optional. | mm, > 0 | `30` |
| d_eq_mm | Diameter of a circle with the same area. Optional. | mm, > 0 | `43.7` |
| aspect_ratio | Long side ÷ short side. Optional. | ≥ 1 | `1.8` |
| eccentricity | How stretched the fitted ellipse is (0 = circle). Optional. | 0 ≤ e < 1 | `0.83` |
| solidity | Area ÷ area of the shape with its dents filled in (1 = no dents). Optional. | 0 < s ≤ 1 | `0.97` |
| circularity | 4π × area ÷ perimeter² (1 = circle). Optional. | 0 < c ≤ 1 | `0.71` |
| edge_density | Fraction of pixels inside the card that are edges (a texture measure). Optional. | 0 to 1 | `0.04` |
| geometry_source | Which photo the geometry was measured from. Optional. | `phone`, `camera` | `phone` |
| photo_path | Path to the card photo. May be blank. | text | `data/cards/card01.jpg` |
| notes | Anything unusual. May be blank. | text | `edge slightly bent` |

The repository's `data/reference_cards.csv` holds the header only: no card has been measured yet, and no values are made up.

## Why it is done this way
- **Fixed choices instead of free text (spec C2).** If one session says `Williams_82` and another `williams 82`, the analysis would treat them as two genotypes. Rejecting anything not in the config list stops that at the source. Capitals must match exactly, for the same reason.
- **position and leaf_position are normalised, not restricted.** There is no fixed list of bench places or leaves, so any label is allowed, but "Bench A-3" and "bench a 3" both become `bench_a_3`.
- **The time zone is stored.** Without it, a timestamp can't be compared reliably with the sensor log or a porometer reading taken on a different clock.
- **Raw data is never rewritten.** T2.1 asked to back up an old-format `session_metadata.csv` and rewrite it with the new columns. The project rule is that raw data is never edited, so the owner chose instead (2026-11-08): if the file has an old or unknown header, nothing is written and the message asks you to move the file aside. A new file is then started. This is a deliberate deviation from T2.1.
- **Card not in the manifest is a warning, not an error** (owner decision). A card can be added to the manifest after the session; stopping a session for that would lose a measurement.
- **All problems reported at once.** Fixing one typo at a time is slow; the message lists every field that is wrong.

## Worked example
Typed answers: plant_id `HK-03`, genotype `Houjaku_Kuwazu`, water_treatment `well_watered`, reproductive_stage `R2`, block `1`, position `Bench A-3`, leaf_position `Top Trifoliate`, light_source `led_lamp`, lamp_distance_mm `300`, window_blocked `yes`, acclimation_minutes `35`, fan_state `low`, working_distance_mm `200`, incidence_angle_deg `10`, card_angle_deg `10`, lamp_angle_deg `0`, lamp_to_card_mm `310`, background `neutral_matte`, card_id `card01`, dry_reference_type `card`, wet_reference_present `true`, operator `AP`, porometer blank, notes `first test session`. Start time 2026-11-08 14:30:05 at UTC−5.

1. Conversions: block → 1; distances → 300.0, 200.0, 310.0; `yes` → true; `Bench A-3` → `bench_a_3`; `Top Trifoliate` → `top_trifoliate`; porometer → blank (None).
2. Lamp rule: light_source is `led_lamp`, and all three lamp fields are filled, so it passes.
3. timestamp_start = `2026-11-08T14:30:05-05:00`; session_id = `HK-03_2026-11-08T14-30-05-05-00`.
4. Files: `data/raw/HK-03_2026-11-08T14-30-05-05-00/session.json` and one new row in `data/raw/session_metadata.csv`.

If genotype had been typed `Williams82`, the message would be:
`genotype must be one of ['Houjaku_Kuwazu', 'Williams_82'], got 'Williams82' (spelling and capitals must match exactly)`.

## Methods paragraph (draft)
For each measurement session, metadata were recorded with a validated entry script (Python 3.13 [version]): plant identity, genotype, water treatment, growth stage (VE, VC and V1–V15, counted by fully developed trifoliate nodes on the main stem, or R1–R8), randomised block and within-block position, measured leaf, light source and lamp geometry, acclimation time, fan setting, camera working distance and leaf incidence angle, reference-card identity and angle, background, dry and wet references, and operator. Categorical fields were restricted to predefined values to prevent inconsistent labels, and session start times were stored with their UTC offset. Plants were arranged in randomised blocks and rotated between sessions to distribute the light gradient from the window across treatments. Reference cards were catalogued with their material, coating, measured shortwave absorptivity, longwave emissivity and shape descriptors.

## Limitations
1. **Vegetative stages stop at V15.** A plant past V15 that has not yet flowered can't be entered until `V16` (and later) is added to the config. Staging also depends on the operator judging when a leaf is "fully developed", so two people can be one stage apart.
2. **The randomisation itself is not generated here.** The module records block and position; the randomised layout and rotation plan must be made and kept separately (spec C3).
3. **card_id is checked only when dry_reference_type is card,** and only with a warning. A card used for r_b (card_direct) but not as the dry reference is not checked against the manifest.
4. **Old-format metadata files are not converted.** Any old file must be moved aside by hand; its rows stay in the old 8-column format.
5. **Prompts are tested with simulated typing only.** Running them on the Pi in a real terminal is untested until the output is pasted.
6. **Angles are not measured here.** They are typed in by the operator, so they are only as good as the protractor reading.
