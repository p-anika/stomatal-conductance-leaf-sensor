# Stomatal Conductance Leaf Sensor

A contact-free way to estimate how open a soybean leaf's stomata are (stomatal conductance, g_s) from its temperature, using a thermal camera, a dry reference card and small environmental sensors on a Raspberry Pi. The full plan is in [docs/spec/](docs/spec/); the equations are in [docs/equations.md](docs/equations.md).

## Folders
- `hardware_tests/`: one script per sensor, for checking it works on the Pi.
- `acquisition/`: field scripts: session metadata and the combined logger.
- `processing/`: physics: reference-card baseline, g_s calculator, shared helpers.
- `ml/`, `analysis/`: not written yet.
- `tests/`: pytest tests (run on the laptop, no hardware needed).
- `data/raw/`: one folder per session (never edited). `data/raw/debug_logs/` holds early test logs that are never used in analysis.
- `data/processed/`, `data/models/`: outputs of processing and ML.
- `docs/`: equations, assumptions, protocol, Pi setup, design notes, decision log.
- `config.yaml`: every constant and setting.

## Setup on the laptop
Run from the repository folder:
```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```
Pi setup is in [docs/pi_setup.md](docs/pi_setup.md).

## Run commands
Always run from the repository folder.

Tests (laptop):
```
.venv\Scripts\python -m pytest -q --tb=short
```

Record software versions for the Methods section (after installing or changing packages):
```
.venv\Scripts\python scripts/snapshot_versions.py --label laptop
```

Older scripts (pre-Phase-1 versions; their physics will be corrected in Phase 1):
```
python -m processing.physics_baseline      # synthetic-data sanity check
python -m processing.gs_calculator         # synthetic-data sanity check
python -m acquisition.combined_logger      # on the Pi only
```

Sensor checks (on the Pi only):
```
python3 hardware_tests/test_bme280.py
python3 hardware_tests/test_tsl2591.py
python3 hardware_tests/test_mlx90640.py
```

## Status
"Untested on hardware" means nobody has yet run that part of this version of the code on the Pi with real sensors and recorded the output.

| Part | Laptop tests | Hardware | Notes |
| --- | --- | --- | --- |
| Repository setup (config.yaml, requirements, tests scaffold, docs) | pass | not needed | Phase 0 done |
| requirements-pi.txt | n/a | Untested on hardware | Not installed or pinned on the Pi yet |
| hardware_tests/ (BME280, TSL2591, MLX90640) | n/a (Pi scripts) | Untested on hardware for this version | The spec lists them as done; an early debug log from 2026-09-01 exists. Re-run and paste output to confirm |
| acquisition/combined_logger.py | none | Untested on hardware for this version | Old version made the 2026-09-01 debug log; rewrite planned in a later phase |
| acquisition/session_metadata.py | none | not needed | To be extended in Phase 2 |
| processing/physics_baseline.py, gs_calculator.py, energy_balance_utils.py | none | not needed | Old physics; to be corrected in Phase 1 (assumptions A1 to A3) |
| Camera (Arducam), wind sensor (FS3000) | n/a | Untested on hardware | Blocked: parts not set up yet |
| ml/, analysis/ | n/a | not needed | Not written yet |
