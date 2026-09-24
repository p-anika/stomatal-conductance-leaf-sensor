""" 
Module 2 (+3): Combined sensor logger / field acquisition script.

Run this on the Pi to start a measurement session. It prompts for
session metadata (module 8), then polls BME280, TSL2591, and MLX90640 on
a fixed interval, writing timestamped rows -- tagged with the session_id
-- to data/raw/<session_id>/sensor_log.csv.

RGB capture (Arducam) is left as a commented hook below. Once the
plate/boom are assembled and the camera is wired in, write
acquisition/camera_capture.py with a take_photo(session_dir, timestamp)
function and uncomment the two marked lines -- no other restructuring
needed.

Run from the repo root as:
    python -m acquisition.combined_logger
"""

import time
import csv
import os
import board
import adafruit_bme280.basic as adafruit_bme280
import adafruit_tsl2591
import adafruit_mlx90640
import numpy as np
from datetime import datetime

from acquisition.session_metadata import prompt_for_session
# from acquisition.camera_capture import take_photo  # uncomment once Arducam is mounted

LOG_INTERVAL_SEC = 5


def main():
    session = prompt_for_session()
    session_dir = os.path.join("data", "raw", session.session_id)
    os.makedirs(session_dir, exist_ok=True)
    log_path = os.path.join(session_dir, "sensor_log.csv")

    i2c = board.I2C()
    bme280 = adafruit_bme280.Adafruit_BME280_I2C(i2c)
    tsl = adafruit_tsl2591.TSL2591(i2c)
    mlx = adafruit_mlx90640.MLX90640(i2c)
    mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ

    frame = [0] * 768

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow([
                "session_id", "timestamp", "air_temp_c", "humidity_pct",
                "pressure_hpa", "lux", "thermal_min_c", "thermal_max_c",
                "thermal_mean_c",
            ])

        print(f"Logging session {session.session_id} -- Ctrl+C to stop.")
        while True:
            try:
                mlx.getFrame(frame)
                arr = np.array(frame).reshape((24, 32))
                timestamp = datetime.now().isoformat()
                row = [
                    session.session_id,
                    timestamp,
                    round(bme280.temperature, 2),
                    round(bme280.relative_humidity, 2),
                    round(bme280.pressure, 2),
                    round(tsl.lux, 2),
                    round(float(arr.min()), 2),
                    round(float(arr.max()), 2),
                    round(float(arr.mean()), 2),
                ]
                writer.writerow(row)
                f.flush()
                print(row)

                # Once the Arducam is connected and mounted on the plate:
                # take_photo(session_dir=session_dir, timestamp=timestamp)

            except ValueError:
                pass  # occasional bad thermal frame reads -- skip, don't crash the session
            time.sleep(LOG_INTERVAL_SEC)


if __name__ == "__main__":
    main()
