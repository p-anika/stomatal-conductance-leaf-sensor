## 8. Blocked items: CSI-to-HDMI cable (Arducam) and FS3000 (wind)

### 8.1 Summary

| Item | Blocked by | Write and test now | Do when the part arrives |
| --- | --- | --- | --- |
| acquisition/camera\_capture.py | Cable | Class with file-based fake camera and tests | Run on the real camera |
| hardware\_tests/test\_arducam.py | Cable | Write it | Run it, check the image |
| calibration/focus\_sweep.py, camera\_scale.py, thermal\_rgb\_alignment.py | Cable | Write against saved or phone images | Run on real images, store results |
| Photo hook in the logger | Cable | Already wired to a null or file camera | Enable in config |
| Real card and leaf geometry features | Cable | Segmentation tested on phone photos and drawn shapes | Re-tune on real photos, fill the card manifest |
| Leaf region from the photo mask (roi\_source: rgb\_mask) | Cable | Warp function with tests | Switch on after alignment passes |
| hardware\_tests/test\_fs3000.py | FS3000 | Write it | Run it in still air |
| Wind in the logger, fan sub-dataset | FS3000 | Columns exist, blank; sensor class and simulator written | Enable in config, collect data |
| calibration/wind\_check.py | FS3000 | Write it | Run it, set the low-wind floor |
| Flat-plate baseline and ML features on real data | FS3000 (and camera for d) | Tested on synthetic data | Rebuild training set and rerun benchmark |

### 8.2 Camera: Arducam Camera Module 3 over the CSI-to-HDMI extension

**Write and test now (no camera needed)**

- **N1 acquisition/camera\_capture.py.** Classes ArducamCamera (imports picamera2 inside its constructor, in try/except), FileCamera (hands back images from a folder, for tests) and NullCamera. capture(session\_dir, timestamp, label='leaf') saves images/\<timestamp>\_\<label>.jpg plus a small JSON file with exposure, gain, lens position, working distance and resolution, and returns the relative path. Use manual focus: set AfMode to manual and LensPosition to 1000 divided by working distance in mm (the lens position is in dioptres). Config keys: sensors.arducam.enabled, working\_distance\_mm, lens\_position, resolution, photo\_interval\_sec.
- **N2 hardware\_tests/test\_arducam.py.** Lists cameras, captures one still, prints resolution and metadata, exits with an error if no camera is found.
- **N3 calibration/focus\_sweep.py.** Sweeps LensPosition around 1000 / working\_distance\_mm, captures at each setting, scores sharpness (variance of the Laplacian on the center crop), prints the best value and writes it to calibration.json.
- **N4 calibration/camera\_scale.py.** Takes a photo of a ruler or an object of known width at the working distance, takes two pixel positions and the known length, and computes mm\_per\_pixel. Also prints the field-of-view estimate for comparison. Writes calibration.json.
- **N5 calibration/thermal\_rgb\_alignment.py.** Takes four or more corresponding points seen in both cameras (a small warm object, such as a bottle of warm water on a cold background, moved to several positions: mark its position in the thermal grid as row and column and in the photo as x and y). Fits an affine transform (cv2.estimateAffine2D) or a homography (cv2.findHomography), stores the matrix and the reprojection error in calibration.json, and supplies the matrix used by thermal\_roi.warp\_mask\_to\_thermal.
- **N6 Segmentation tuning on phone photos** with a ruler in frame, and phone or scanner photos of each reference card, so geometry columns in data/reference\_cards.csv can be filled early. Mark them geometry\_source = phone and redo them later with geometry\_source = camera.

**When the cable arrives**

1. Power the Pi off. Connect the extension kit between the Arducam and the Pi’s camera (CSI) connector, not the display connector. Seat the ribbon fully with the contacts the right way round and close the latch. Power on.
2. Check detection: rpicam-hello --list-cameras (libcamera-hello on older Raspberry Pi OS). Expect an imx708 entry. If none appears, reseat both ends of the extension and check the connector.
3. Install: sudo apt install -y python3-picamera2 python3-opencv. If you use a virtual environment on the Pi, create it with --system-site-packages so picamera2 is visible.
4. Run hardware\_tests/test\_arducam.py. Confirm a clean image. If the long cable causes frame errors or dropouts, reseat it, shorten it, or lower the resolution, and record the finding in docs/pi\_setup.md.
5. Set working\_distance\_mm in config, run focus\_sweep.py, and store lens\_position.
6. Run camera\_scale.py and thermal\_rgb\_alignment.py. Check by imaging a known object: its measured size should be within 3%.
7. Set sensors.arducam.enabled: true and run a 10-minute session. Confirm files appear in images/ and the image\_file column is filled.
8. Run segmentation on real leaf and card photos, tune the thresholds, save overlays, and compare d\_min and area with caliper measurements (within 3%).
9. Switch acquisition.roi\_source to rgb\_mask only after the alignment check passes; keep the manual rectangles as a fallback.
10. Photograph every card, fill the geometry columns in reference\_cards.csv (geometry\_source = camera), and rebuild the training set (this also needs wind, see below).
11. Add camera errors to the QC flags.

### 8.3 Wind: SparkFun FS3000 air-velocity sensor

**Write and test now (no sensor needed)**

- **M1 Fs3000Sensor in acquisition/sensors.py.** Use SparkFun’s Qwiic FS3000 Python package (search PyPI for the FS3000 Qwiic library), or raw I2C reads with smbus2 following the datasheet: read the 5-byte response, check the checksum, and convert the 12-bit value with the datasheet table for the variant. Config sensors.fs3000.variant is 1005 (about 0–7.2 m/s) or 1015 (about 0–15 m/s); read the product label and datasheet to confirm. Expected I2C address 0x28 (verify with i2cdetect). read() returns wind\_ms, wind\_raw and status.
- **M2 Simulated sensor.** Wind depends on fan\_state (off: near zero plus noise; low: about 0.5–1 m/s; high: about 1.5–3 m/s) so downstream code can be tested.
- **M3 hardware\_tests/test\_fs3000.py.** Prints wind speed every second; exits with an error if the sensor is not found.
- **M4 calibration/wind\_check.py.** Records 60 s of still air (noise floor and offset), then 60 s at each fan setting; prints mean and standard deviation; suggests wind\_noise\_floor\_ms; writes calibration.json.
- **M5 Logger columns** wind\_ms, wind\_raw, wind\_status and wind\_source (measured, manual\_assumed, none) always exist and are blank until the sensor is enabled. Downstream code already treats a blank as no\_wind (A14). A manual wind is allowed only through an explicit --assumed-wind flag, which sets wind\_source = manual\_assumed and propagates a flag to every result.
- **M6 fan\_state** is already a validated metadata field (T2.1).

**When the sensor arrives**

1. Read the label to see which variant you have and set sensors.fs3000.variant.
2. Power off, connect it into the I2C chain with a Qwiic or STEMMA QT cable, power on, and run i2cdetect -y 1. Expect the existing devices (BME280 at 0x77, TSL2591 at 0x29, MLX90640 at 0x33) plus the FS3000.
3. Run test\_fs3000.py in still air; values should sit near zero. Note the noise.
4. Mount it so air flows through its sensing channel in the direction the datasheet specifies, at the leaf’s position and out of the sensor plate’s wake. Record the orientation in docs/measurement\_protocol.md.
5. Run wind\_check.py and set wind\_noise\_floor\_ms to the sensor’s noise floor; readings below it count as zero forced flow and free convection takes over (A11).
6. Set sensors.fs3000.enabled: true and run a session with the fan off, low and high. Confirm the wind columns fill and wind\_source is measured.
7. Map the wind: with the fan running, measure at the leaf position, the card position and the sensor plate, and store the ratios in calibration.json; the card and leaf must see the same airflow to within about 10% (A16). Optionally also compare with a handheld anemometer or a known fan distance.
8. Collect the fan sub-dataset: every card shape at fan off, low and high, with fan\_state recorded.
9. Rebuild the training set with wind, retrain, recompute flat-plate baselines with measured wind, and rerun the benchmark.
10. Check that no real-data result uses wind\_source other than measured (or that any exception is flagged).

### 8.4 What still works while the parts are missing

All of Phases 0–6 (physics, metadata, segmentation, ROI and window code, ML and analysis on synthetic data, the dry run); Phase 7 on the real Pi with the BME280, TSL2591 and MLX90640; the thermal and light calibrations; real card sessions that give the primary card\_direct method (it needs neither the wind sensor nor the camera) and, with a wet reference, the Jones index. Only the flat-plate baseline on real leaves, the ML correction on real leaves, geometry from the device’s own photos, and the fan sub-dataset wait for the parts.
