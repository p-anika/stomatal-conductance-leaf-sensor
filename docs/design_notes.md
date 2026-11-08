# Design notes

**Status: stub.** Short notes on why the project is built the way it is. Each note says what was decided, why, and what else was considered. The detailed record is [decision_log.md](decision_log.md); this file is the readable summary for the paper.

## Code layout
- Settings and physical constants live in `config.yaml`, never inside the code, so every number used is in one place and can be cited.
- Only the Pi-specific files import hardware libraries. Everything else runs and is tested on the laptop with simulated data.
- Raw data are never changed. Processing writes new files to `data/processed/`.

## Measurement method
- The dry reference card (card_direct) is the main way of finding the boundary-layer resistance r_b. The flat-plate formula is a comparison baseline. The ML correction is reported only if it passes validation (assumption A9).
- Low-g_s readings are kept and labelled as censored, not deleted (assumption A15).

## TODO
- Add a note as each phase is finished.
