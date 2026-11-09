# Spec changes

One row per change to a file in docs/spec/. docs/spec.md is rebuilt from docs/spec/ after each change.

| Date | Spec file | What changed | Why | Task or session |
| --- | --- | --- | --- | --- |
| 2026-11-08 | 03_phases_0_to_3.md, 09_appendix_B_config_schemas.md | T2.1: old-header backup-and-rewrite replaced by "refuse to write and ask the user to move the file aside; never rewrite raw data"; done-when now says "refusal of an old-format file"; validation rules added (block ≥ 1, position and leaf_position lowercase with underscores, lamp fields only for led_lamp, angles 0–90°, plant_id letters/digits/-/_, timestamps with UTC offset); warning when a card dry reference's card_id is not in the manifest. T2.2: columns are those of Appendix B, the 11 listed required, the 8 geometry columns optional until measured, numbers may be blank until measured. Appendix B config: reproductive_stage [VE, VC, V1–V15, R1–R8]; new enumerations.background [neutral_matte, other]; fan_state written ["off", low, high] with a note that bare off, no and yes become booleans in YAML. | Bring the spec in line with the owner-approved T2.1–T2.3 implementation (never rewrite raw data; Appendix B card columns; vegetative stages by trifoliate nodes; YAML boolean pitfall found by a test). | T2.1–T2.3 (2026-11-08) |
