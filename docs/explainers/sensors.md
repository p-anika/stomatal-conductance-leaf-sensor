# sensors (acquisition/sensors.py) and camera_capture (acquisition/camera_capture.py)

## What it does
These are the pieces the logger (T7.2) uses to talk to the hardware. Each sensor has three versions
with the same `read()`:
- **the real one**, for the Pi
- **a Simulated one**, which gives believable made-up numbers on the laptop
- **a Null one**, which returns blanks when a sensor is switched off

So the whole program can be built and tested on the laptop before the hardware arrives.

## How it works
1. **Readings.** Each `read()` returns a small record with values and a status: `ok`, `overflow`,
   `read_error`, `checksum_error` or `unavailable`.
2. **BME280:** air temperature (°C), humidity (%), pressure (hPa).
3. **TSL2591, light.** It starts at high gain. If the light is too bright, the chip reports an
   overflow; the code then lowers the gain (high → medium → low) and tries again. It remembers the
   gain that worked and never raises it again, which is a known limitation. It reports lux, the raw
   counts, and the gain and integration time used.
4. **MLX90640, thermal camera.** It reads a 24 × 32 frame in °C at 2 Hz. A bad frame (ValueError)
   is retried up to 5 times.
5. **FS3000, wind.** It reads 5 bytes from address 0x28 and checks the checksum: all 5 bytes must
   add up to a multiple of 256. It takes the 12-bit count and converts it to m/s with the
   datasheet table, drawing straight lines between the table points.
6. **Simulated sensors.** They follow the session's light source (lamp, window, sunlight) and fan
   setting (off, low, high), so later code sees realistic changes.
7. **Camera.** `capture()` saves a JPEG and a small JSON file with the camera settings. Focus is
   manual: lens position (dioptres) = 1000 / working distance (mm).

## Why it is done this way
- **Hardware libraries are imported only inside the real classes,** so every test runs on the
  laptop (spec Section 6).
- **One shared record per sensor** means the logger never needs to know whether a sensor is
  real, simulated or switched off.
- **FS3000 from the datasheet bytes, not only a library.** Every number can be traced to the
  datasheet. The library is used once on the Pi to cross-check.
- **Rejected alternatives:**
  - Stepping the TSL gain back up (not in the spec; it could flip gains every reading).
  - A 1015 table (the module is a 1005; no table was supplied).

## Worked example
**FS3000.** The bytes `CC 01 99 01 99` (the datasheet's own example):
1. Sum = 0xCC + 0x01 + 0x99 + 0x01 + 0x99 = 0x200, and 0x200 mod 256 = 0, so the reading is valid.
2. Count = (0x01 & 0x0F) × 256 + 0x99 = 409, so the wind is **0 m/s** (the bottom of the table).

**Halfway between two table points.** Count 1218.5 lies halfway between 915 (1.07 m/s) and 1522
(2.01 m/s), so wind = 1.07 + 0.5 × (2.01 − 1.07) = **1.54 m/s**. Units: counts → m/s by
interpolation; no physics involved.

**Lens.** A leaf 300 mm from the lens gives 1000 / 300 = **3.33 dioptres**.

## Methods paragraph (draft)
Air temperature, humidity and pressure were measured with a BME280 sensor, illuminance with a
TSL2591 (gain reduced automatically on saturation), leaf and card temperatures with an MLX90640
thermal array (24 × 32 pixels, 2 Hz), and air velocity with an FS3000-1005 sensor read over I2C
and converted with the manufacturer's count-to-velocity table by linear interpolation (stated
accuracy 5 % of full scale, ±0.36 m s⁻¹). Leaf photographs were taken with an Arducam camera at a
fixed manual focus. Software: Python 3.13 on Raspberry Pi OS (Debian 13).

## Limitations
- **Nothing has run on the real hardware yet;** see `docs/pi_first_run_checklist.md`.
- **The TSL2591 gain never steps back up** within a session.
- **Linear interpolation** between the FS3000 table points is an assumption; the datasheet gives
  only the points.
- **The FS3000 byte layout** on our module and the **SparkFun package name** are UNVERIFIED.
- **Simulated values are invented** and only for testing.
