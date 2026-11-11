"""FS3000 check on the Pi: read the same moment with our raw datasheet decoder and with SparkFun's
library, and print both. [PI, BLOCKED-WIND until the sensor is wired]

    .venv/bin/python hardware_tests/test_fs3000.py

The SparkFun package name (pip: sparkfun-qwiic-fs3000) and its method names are UNVERIFIED: this
script looks for them and prints what it finds. If the library is missing it still runs the raw read.
Not a pytest file (hardware_tests/ is not collected on the laptop).
"""
import time

import yaml

from acquisition.sensors import Fs3000Sensor, fs3000_checksum_ok, fs3000_count

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
cfg["sensors"]["fs3000"]["enabled"] = True
raw = Fs3000Sensor(cfg)

lib = None
try:
    import qwiic_fs3000                                   # UNVERIFIED module name
    cls = next(getattr(qwiic_fs3000, n) for n in dir(qwiic_fs3000) if n.lower().startswith("qwiicfs3000"))
    lib = cls()
    if hasattr(lib, "begin"):
        lib.begin()
    for name in ("set_range", "setRange"):
        if hasattr(lib, name):
            rng = next((getattr(lib, a) for a in dir(lib) if "RANGE_7" in a), None)
            if rng is not None:
                getattr(lib, name)(rng)                   # FS3000-1005: 0-7 m/s range
            break
    print("library found:", cls.__name__, "methods:", [m for m in dir(lib) if "read" in m.lower()])
except Exception as e:                                   # noqa: BLE001 - diagnostic script
    print("SparkFun library not usable:", e)

for i in range(10):
    r = raw.read()
    b = bytes(raw.raw_reader())
    line = (f"raw decoder: {r.wind_ms:.2f} m/s (count {r.wind_raw:.0f}, {r.status}) | bytes {b.hex(' ')} "
            f"checksum_ok={fs3000_checksum_ok(b)} count={fs3000_count(b)}")
    if lib is not None:
        for name in ("read_meters_per_second", "readMetersPerSecond", "read_mps"):
            if hasattr(lib, name):
                line += f" | library {name}: {getattr(lib, name)():.2f} m/s"
                break
        for name in ("read_raw", "readRaw"):
            if hasattr(lib, name):
                line += f" raw {getattr(lib, name)()}"
                break
    print(line)
    time.sleep(1.0)
