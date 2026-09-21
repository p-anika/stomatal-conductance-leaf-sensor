
import time
import board
import adafruit_mlx90640
import numpy as np

i2c = board.I2C()
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ

frame = [0] * 768
while True:
    try:
        mlx.getFrame(frame)
        arr = np.array(frame).reshape((24, 32))
        print(f"Min: {arr.min():.1f} C  Max: {arr.max():.1f} C  Mean: {arr.mean():.1f} C")
        time.sleep(1)
    except ValueError:
        continue  # occasional read errors are normal, just retry
