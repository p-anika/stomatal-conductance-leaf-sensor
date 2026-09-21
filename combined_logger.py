import time
import csv
import board
import adafruit_bme280.basic as adafruit_bme280
import adafruit_tsl2591
import adafruit_mlx90640
import numpy as np
from datetime import datetime

i2c = board.I2C()
bme280 = adafruit_bme280.Adafruit_BME280_I2C(i2c)
tsl = adafruit_tsl2591.TSL2591(i2c)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ

frame = [0] * 768
LOG_INTERVAL_SEC = 5

with open("sensor_log.csv", "a", newline="") as f:
    writer = csv.writer(f)
    if f.tell() == 0:
        writer.writerow(["timestamp", "air_temp_c", "humidity_pct", "pressure_hpa",
                          "lux", "thermal_min_c", "thermal_max_c", "thermal_mean_c"])

    while True:
        try:
            mlx.getFrame(frame)
            arr = np.array(frame).reshape((24, 32))
            row = [
                datetime.now().isoformat(),
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
        except ValueError:
            pass  # skip occasional bad thermal frame reads
        time.sleep(LOG_INTERVAL_SEC)

