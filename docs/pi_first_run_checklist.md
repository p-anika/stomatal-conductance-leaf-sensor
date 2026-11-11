# Pi first-run checklist (sensors, FS3000, camera)

Do these in order. After each step, compare with "Expected". If it doesn't match, do "If it
fails", then stop and paste the output to Claude Code. **Nothing here has been run on real
hardware yet.**

**Before you leave the laptop:** commit and push the T7.1 work (the git commands from this task),
so the Pi can `git pull` it.

---

### 1. Power off and wire
- Pi **off** and unplugged.
- Connect the four sensors to the I2C bus as in `docs/hardware/sensors.md`: pin 1 3.3 V, pin 3
  SDA, pin 5 SCL, pin 6 GND.
- **Camera:**
  1. Insert the ribbon with the contacts facing the connector contacts, and close the latch.
  2. Connect the CSI-to-HDMI adapter boards with the **shortest** HDMI cable.
  3. Check the ribbon orientation at **both** ends.
- **Expected:** nothing yet.
- **If it fails:** –

### 2. Power on, check the bus
```
ls /dev/i2c-1
sudo apt install -y i2c-tools
i2cdetect -y 1
```
- **Expected:** `/dev/i2c-1` exists, and the grid shows **28** (FS3000), **29** (TSL2591),
  **33** (MLX90640) and **76** or **77** (BME280).
- **If it fails:**
  - **No `/dev/i2c-1`:** run `sudo raspi-config`, then Interface Options → I2C → Enable, and
    reboot.
  - **One address missing:** power off and check that board's wires.
  - **All missing:** check SDA and SCL are not swapped.

### 3. Check power and the clock
```
vcgencmd get_throttled
timedatectl show -p NTPSynchronized
```
- **Expected:** `throttled=0x0` and `NTPSynchronized=yes`.
- **If it fails:**
  - **Throttled is not 0x0:** use the official power supply.
  - **Not synchronized:** connect to the internet and wait a minute.

### 4. Get the code and the libraries
```
cd ~/stomatal-conductance-leaf-sensor
git pull
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements-pi.txt pytest
.venv/bin/pip install sparkfun-qwiic-fs3000
```
(`--system-site-packages` lets the venv see picamera2, which comes from apt. If `.venv` already
exists, skip the `venv` line.)
- **Expected:** "Successfully installed …" for each package. The last line may fail because the
  SparkFun package name is UNVERIFIED; that is OK.
- **If it fails:**
  - **SparkFun line:** search pypi.org for "qwiic fs3000" and tell Claude Code the right name.
    Step 7 still runs without it.
  - **Any other line:** paste the error.

### 5. Laptop tests, now on the Pi (no hardware used)
```
.venv/bin/python -m pytest -q tests/test_sensors.py tests/test_camera_capture.py
```
- **Expected:** `25 passed`. These use fake drivers, so they should pass exactly as on the laptop.
- **If it fails:** paste the output. This means a library version difference, not a wiring
  problem.

### 6. Real sensors, one at a time
```
.venv/bin/python -m pytest -q tests/test_sensors_hardware.py --hardware -k bme280
.venv/bin/python -m pytest -q tests/test_sensors_hardware.py --hardware -k tsl2591
.venv/bin/python -m pytest -q tests/test_sensors_hardware.py --hardware -k mlx90640
.venv/bin/python -m pytest -q tests/test_sensors_hardware.py --hardware -k fs3000
```
- **Expected:** `1 passed` for each line.
- **If it fails:**
  - **`SensorUnavailable`:** that board does not answer. Go back to step 2.
  - **Anything else:** paste it.

### 7. FS3000: raw decoder against the SparkFun library, same moment
Run it twice: once with the fan **off**, once with the fan **on** (low setting).
```
.venv/bin/python hardware_tests/test_fs3000.py
```
- **Expected:**
  - 10 lines.
  - Fan off: about 0.00 m/s, count near 409, `checksum_ok=True`.
  - Fan on: a steady value.
  - If the library loaded, its m/s is within about 0.05 m/s of the raw decoder.
  - The datasheet accuracy is ±0.36 m/s (5 % of 7.23 m/s).
- **If it fails:**
  - **`checksum_ok=False`:** paste the printed bytes. The byte layout is UNVERIFIED on our module.
  - **Library and raw decoder disagree:** paste both columns. Do not use the wind values until
    this is resolved.
  - **Always `overflow`:** read the label on the module (1005 or 1015).

### 8. Camera: detected?
```
rpicam-hello --list-cameras
```
- **Expected:** one camera listed with its sensor name and modes.
- **If it fails:** `no cameras available` means you should power off, re-seat the ribbon at both
  ends (check orientation and latches), and check the HDMI extension cable.

### 9. Camera: one manual-focus still
```
rpicam-still -n -o ~/focus_test.jpg --autofocus-mode manual --lens-position 3.33
```
3.33 dioptres = 1000 / 300 mm. Use 1000 divided by your real lens-to-leaf distance in mm, and
check the flag names with `rpicam-still --help`.
- **Expected:** `~/focus_test.jpg` exists, and a leaf 300 mm away looks sharp.
- **If it fails:** paste the error. For a blurry image, measure the working distance again.

### 10. Camera: dropped frames over the extension cable
```
rpicam-vid -n -t 20000 -o /dev/null
dmesg | grep -iE "unicam|cfe|csi|dropped|error" | tail -20
```
- **Expected:** `rpicam-vid` runs for 20 s without errors, and `dmesg` shows no new
  CSI or unicam errors.
- **If it fails:** errors or dropped frames mean the extension cable is too long, bent or loose.
  Try the shortest HDMI cable, then the ribbon on its own without the extension.

### 11. Camera through our code
First set `sensors.arducam.working_distance_mm` in `config.yaml` to the measured lens-to-leaf
distance.
```
.venv/bin/python -m pytest -q tests/test_sensors_hardware.py --hardware -k arducam
```
- **Expected:** `1 passed`.
- **If it fails:**
  - **`CameraUnavailable ... picamera2`:** the venv was created without
    `--system-site-packages`, or `python3-picamera2` is missing
    (`sudo apt install -y python3-picamera2`).

### 12. If the I2C bus locks up or gives random errors (long sensor cable)
```
sudo nano /boot/firmware/config.txt
```
Add `dtparam=i2c_arm_baudrate=50000` (lower speed), save, reboot, and repeat steps 2 and 6. A lower
speed also lowers the possible thermal frame rate. Check that the MLX90640 still reads at 2 Hz.

### 13. Paste back
Paste the output of steps 2, 3, 6, 7 (both runs), 8, 10 and 11 to Claude Code.
