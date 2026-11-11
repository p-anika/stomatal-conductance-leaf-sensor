# Sensors (T7.1, M1, N1): hardware guide

Code: `acquisition/sensors.py` (BME280, TSL2591, MLX90640, FS3000) and
`acquisition/camera_capture.py` (Arducam). Everything here is **untested on real hardware** until
you paste the outputs from `docs/pi_first_run_checklist.md`.

## Parts needed
- Raspberry Pi (Debian 13 trixie, Python 3.13.5), power supply, microSD card with the repository.
- BME280 (air temperature, humidity, pressure), TSL2591 (light), MLX90640 (thermal camera): these
  already work with `hardware_tests/test_*.py`.
- FS3000-1005 air-velocity sensor on a breakout board (SparkFun Qwiic, 3.3 V).
- Arducam camera, ribbon cable, and the passive CSI-to-HDMI extension (two adapter boards plus a
  short HDMI cable that carries the camera signal; **it is not a video output**).
- Qwiic / STEMMA QT cables (or jumper wires).
- To check the FS3000: the fan used in the experiment.

## Setup and wiring (with pin names)
All four sensors share **one I2C bus** (bus 1). Power the Pi **off** before connecting anything.

| Pi header pin | Signal | Qwiic / STEMMA QT wire colour | Goes to |
| --- | --- | --- | --- |
| Pin 1 | 3.3 V | red | VIN / 3V3 of every sensor board |
| Pin 3 | GPIO2 = SDA1 | blue | SDA of every sensor board |
| Pin 5 | GPIO3 = SCL1 | yellow | SCL of every sensor board |
| Pin 6 | GND | black | GND of every sensor board |

- The boards can be chained with Qwiic / STEMMA QT cables (any order).
- **Use 3.3 V, not 5 V.** The FS3000's own supply is 3.3 V (datasheet 3.2), and its accuracy
  depends on a steady supply.
- **Keep SDA and SCL wires about the same length.** The datasheet (5.1) says the load on both
  lines should be equal. Long cables can cause lock-ups; see the checklist.
- **Expected I2C addresses:** FS3000 **0x28** (datasheet 5.1.1). TSL2591 0x29, MLX90640 0x33 and
  BME280 0x77 or 0x76 are recalled values, UNVERIFIED: confirm them with `i2cdetect -y 1`.
- **FS3000 direction:** the datasheet drawing (Figure 1, pin assignments) shows the flow arrow.
  Point the arrow along the fan's airflow, at the leaf position.
- **Camera:**
  1. Pi off.
  2. Open the CSI connector latch and insert the ribbon with its metal contacts facing the contacts
     inside the connector. Close the latch.
  3. Do the same on the extension's adapter board.
  4. Connect the two adapter boards with the **shortest** HDMI cable you have.

  Check your Pi model's camera connector in the Raspberry Pi camera documentation before you plug
  anything in: Pi 4 and Pi 5 use different connector sizes.

## Commands to run on the Pi
Follow `docs/pi_first_run_checklist.md`, in order. The key commands are:
```
i2cdetect -y 1
.venv/bin/python -m pytest -q tests/test_sensors.py tests/test_camera_capture.py
.venv/bin/python -m pytest -q tests/test_sensors_hardware.py --hardware
.venv/bin/python hardware_tests/test_fs3000.py
rpicam-hello --list-cameras
```

## What a good result looks like
- `i2cdetect` shows 28, 29, 33 and 76 or 77.
- The hardware tests pass. Typical values:
  - BME280: room temperature about 18–28 °C, humidity 20–80 %, pressure about 950–1050 hPa.
  - MLX90640: a 24 × 32 frame with a median near room temperature.
- **FS3000:**
  - Fan off: about 0 m/s (counts near 409).
  - Fan on: a steady value. Accuracy is **5 % of full scale, about ±0.36 m/s** (datasheet 3.3).
  - The raw decoder and the SparkFun library should agree within about 0.05 m/s.
- **Camera:** `rpicam-hello --list-cameras` lists one camera, and the camera hardware test saves a
  JPEG of more than 10 kB.

## What failure looks like, and the first thing to check
| Symptom | First thing to check |
| --- | --- |
| An address missing in `i2cdetect` | That board's 3.3 V and GND wires; then SDA/SCL swapped |
| `Remote I/O error` or random read errors | Cable length: shorten it, or lower the I2C speed (see the checklist) |
| FS3000 `checksum_error` every time | Wiring and bus speed; then paste the bytes printed by `hardware_tests/test_fs3000.py` |
| FS3000 always `overflow` | Wrong variant: read the label on the module (1005 or 1015); only 1005 has a table |
| TSL2591 `overflow` even at low gain | The sensor is pointed straight into the lamp: angle it or add a diffuser |
| `no cameras available` | Ribbon orientation and the latch at both ends; then the HDMI extension cable |
| Camera works for a while, then errors | The extension cable is too long or bent; try the shortest cable |

## What to paste back to Claude Code (last 20 lines of output)
Paste the output of every checklist step, especially `i2cdetect -y 1`, the hardware test summary,
the 10 lines from `hardware_tests/test_fs3000.py`, and `rpicam-hello --list-cameras`.
