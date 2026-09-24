import time
import board
import adafruit_tsl2591

i2c = board.I2C()
tsl = adafruit_tsl2591.TSL2591(i2c)

while True:
    print(f"Lux: {tsl.lux:.1f}  Infrared: {tsl.infrared}  Visible: {tsl.visible}  Full spectrum: {tsl.full_spectrum}")
    time.sleep(2) 
