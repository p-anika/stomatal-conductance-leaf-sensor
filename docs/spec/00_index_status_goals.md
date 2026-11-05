# Stomatal Conductance Leaf Sensor: Software Specification and Implementation Plan

*Handoff document for finishing the project’s software. Version of October 30, 2026. Repository: https://github.com/p-anika/stomatal-conductance-leaf-sensor*

## 0. How to use this document

This document assumes you know nothing about the project. Sections 1–3 explain what is being built and where it stands today. Section 4 lists every change needed to make the code logical, easy to use and scientifically rigorous. Sections 5–7 give the target repository layout, coding conventions and a task-by-task implementation plan. Section 8 collects the tasks that cannot be finished until two parts arrive, and says exactly what to do now and what to do when they do. Section 9 lists decisions only the project owner can make (each has a default, so work never has to stop). Section 10 defines “done”. The appendices hold the equations, the configuration file and the data schemas.

Task tags used throughout:

- **\[READY\]** can be done now, on any computer, with no hardware.
- **\[PI\]** needs the Raspberry Pi with the sensors that already work (BME280, TSL2591, MLX90640).
- **\[BLOCKED-CAM\]** needs the CSI-to-HDMI extension cable so the Arducam camera can be connected. Write and unit-test the code now; run it on hardware when the cable arrives.
- **\[BLOCKED-WIND\]** needs the SparkFun FS3000 wind sensor (shipped, in delivery). Same rule.
- **\[OWNER\]** a physical task or a decision only the project owner can make.
- **P0 / P1 / P2**: priority. P0 is required for the core claim, P1 for the machine-learning claim, P2 is optional polish. \[DEFER\] marks work that is worthwhile but must not start until Tier 1 works end to end (a real card\_direct g\_s from a session, checked against ground truth); where a task is marked DEFER, the plan says which minimal version to build meanwhile.

## 1. The project in plain language

**The problem.** Crop breeders must find drought-tolerant varieties. A key measurement is stomatal conductance (g\_s): how open the tiny pores (stomata) on a leaf are, which sets how fast the leaf loses water. The standard tool, a porometer, is expensive and clamps onto the leaf, which disturbs the thin layer of still air the measurement depends on. Satellite and drone thermal imaging is cheap but cannot see individual plant varieties.

**The device.** A roughly $250–350, contact-free sensor head on an adjustable boom, pointed at one leaf from a fixed distance. It reads the leaf’s temperature (thermal camera), air temperature, humidity and pressure, light, and wind, and photographs the leaf. A leaf that loses water cools itself, so an energy balance turns the leaf’s temperature plus the surroundings into g\_s.

**The hard part.** The energy balance needs the boundary-layer resistance r\_b: how strongly the film of still air clinging to the leaf resists heat and vapor flow. It depends on leaf size, shape and wind. Existing low-cost methods use a flat-plate formula. This project adds (1) a dry reference card, which cannot transpire, so its temperature gives r\_b directly by simple algebra; and (2) a small machine-learning model that corrects r\_b for the real leaf’s shape. The project’s specific, narrow novelty is that geometry-aware correction, tested against the flat-plate formula. Method hierarchy: card\_direct (the dry card’s measured r\_b used for the leaf) is the primary method and the basis of all headline results; ml\_corrected is a secondary refinement, reported only if it passes validation; flat\_plate is the baseline both are compared with. Dry-reference thermal g\_s itself is already published (Jones 1999; Leinonen et al. 2006), so the write-up must cite that work.

**The experiment.** Soybean (Glycine max), two cultivars (Houjaku Kuwazu, drought-tolerant; Williams 82, drought-sensitive), two water treatments (well-watered, drought-stressed), 4–5 plants per group. The first cohort is about three months old and already has pods, so it will senesce soon; a second cohort should be started. The target competition is the Plant Sciences category at ISEF (a science fair).

**Hardware and status**

| Part | Role | Status |
| --- | --- | --- |
| Raspberry Pi 4 (2 GB), microSD, STEMMA QT breakout, USB power bank | Base station, runs all code | Working |
| BME280 | Air temperature, humidity, pressure | Wired, mounted, logging |
| TSL2591 | Light (lux, visible and IR channels) | Wired, mounted, logging |
| MLX90640 (32×24 pixels, 55° field of view) | Thermal camera | Wired, mounted, logging |
| Arducam Camera Module 3 (12 MP, autofocus) | Leaf photos for geometry | Delivered, not mounted; waiting for the CSI-to-HDMI extension cable |
| SparkFun FS3000 air-velocity sensor | Wind speed at the leaf | Shipped, in delivery |
| Reference cards (several shapes) | Dry reference | In progress |
| Measurement lamp (LED flood light) | Constant light during measurements | Not bought yet (recommended) |

## 2. Final goal of the code

When finished, one person with the device should be able to do the following.

1. **In the field (on the Pi):** start a session with a few prompts or command-line flags; the logger records sensors, raw thermal frames and photos, labeled with plant, genotype, treatment and conditions.
2. **Back on a laptop:** run one command to get g\_s for every measurement window, computed three ways (primary: direct card value; secondary: ML-corrected; baseline: flat-plate), each with an uncertainty range and a validity flag; get genotype × treatment statistics; and get every figure and table needed for the ISEF board and research paper.
3. **Evidence of accuracy:** scripts that compare the device against independent references (thermometer calibration, a borrowed porometer or weighed-transpiration g\_s, petroleum-jelly-coated leaves for r\_b) and report bias, error, and the lowest g\_s the device can resolve.

Pipeline in one line: session metadata → logger (sensors, thermal frames, photos) → raw data folder → quality control → leaf/card geometry and temperatures → averaged measurement windows → card r\_b (physics) → training set → ML correction → leaf g\_s (three methods) with uncertainty → validation → statistics → figures.

**Claims the software must support**

- **Tier 1 (core):** a calibrated, validated, low-cost contact-free sensor (using the primary card\_direct method) that detects drought responses and genotype differences, with stated error bars and detection limit.
- **Tier 2 (novelty):** the geometry-aware r\_b correction (the secondary method) measurably beats the flat-plate baseline on real leaves and, for atypically shaped leaves, improves on card\_direct. If it does not, the software must report that honestly.

**Design principles**

- Physics, ML and analysis code never imports hardware libraries, so everything but the logger runs on any laptop.
- Raw data is never edited or deleted.
- Every result carries its method, a valid flag, a reason code and an uncertainty.
- No silent defaults: a missing wind speed, light calibration or card value produces a flagged result, never a quiet guess.
- Every constant lives in one config file with its source cited.
- Everything is reproducible from raw data with documented commands.

* Data are never excluded because of their measured value, only because of failed inputs or quality checks. Values below the detection limit are censored and kept.

## 3. Current state of the repository

| Path | State |
| --- | --- |
| hardware\_tests/test\_bme280.py, test\_tsl2591.py, test\_mlx90640.py | Done (one-off sensor checks) |
| acquisition/session\_metadata.py | Works; needs more fields and validation |
| acquisition/combined\_logger.py | Works for BME280, TSL2591, MLX90640 every 5 s; logs only min/max/mean of the whole thermal frame; no photos, no wind, no card/leaf separation |
| processing/energy\_balance\_utils.py, physics\_baseline.py, gs\_calculator.py | Implemented and tested with synthetic numbers, but several constants and assumptions must change (Section 4, group A) |
| ml/, analysis/, docs/ | Empty (.gitkeep only) |
| requirements.txt | Partial; optional packages commented out |
| config.yaml, .gitignore, tests/ | Do not exist |
| data/raw/debug\_logs/ | One early debug CSV; never use it as real data (verify it is not sitting directly in data/raw/) |

Code is edited on a laptop and on the Pi and synced through GitHub, so any module not touching sensors must run on a laptop.
