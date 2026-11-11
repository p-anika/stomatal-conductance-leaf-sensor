"""Tests for acquisition/sensors.py (T7.1, M1, M2) on the laptop: fake drivers stand in for the
hardware. Real-sensor tests are in test_sensors_hardware.py (marked hardware)."""
from __future__ import annotations

import copy
import math
import subprocess
import sys

import numpy as np
import pytest

from acquisition import sensors as S


# ---------------------------------------------------------------- FS3000 decoding (datasheet)

def test_checksum_datasheet_example():
    assert S.fs3000_checksum_ok(bytes([0xCC, 0x01, 0x99, 0x01, 0x99]))      # datasheet section 5.3
    assert not S.fs3000_checksum_ok(bytes([0xCD, 0x01, 0x99, 0x01, 0x99]))


def test_count_uses_lower_nibble_of_high_byte():
    assert S.fs3000_count(bytes([0x00, 0xF1, 0x99, 0x00, 0x00])) == 0x199


def test_counts_to_wind_table_and_edges(default_config):
    t = default_config["sensors"]["fs3000_table_1005"]
    for c, w in zip(t["counts"], t["wind_ms"]):
        assert S.fs3000_counts_to_wind(c, t) == (pytest.approx(w), S.OK)
    assert S.fs3000_counts_to_wind(300, t) == (0.0, S.OK)
    mid, st = S.fs3000_counts_to_wind((915 + 1522) // 2, t)
    assert st == S.OK and 1.07 < mid < 2.01                                   # linear between points
    w, st = S.fs3000_counts_to_wind(3687, t)
    assert st == S.OVERFLOW and math.isnan(w)


def _frame(count: int) -> bytes:
    data = [(count >> 8) & 0x0F, count & 0xFF, (count >> 8) & 0x0F, count & 0xFF]
    return bytes([(-sum(data)) % 256] + data)


def test_fs3000_sensor_raw_reader(default_config):
    s = S.Fs3000Sensor(default_config, raw_reader=lambda: _frame(1522))
    r = s.read()
    assert r.status == S.OK and r.wind_raw == 1522 and r.wind_ms == pytest.approx(2.01)
    bad = S.Fs3000Sensor(default_config, raw_reader=lambda: bytes([0, 5, 0xF2, 0, 0]))
    assert bad.read().status == S.CHECKSUM_ERROR

    def boom():
        raise OSError("bus")
    assert S.Fs3000Sensor(default_config, raw_reader=boom).read().status == S.READ_ERROR


def test_fs3000_variant_without_table_refused(default_config):
    cfg = copy.deepcopy(default_config)
    cfg["sensors"]["fs3000"]["variant"] = 1015
    with pytest.raises(ValueError, match="1015"):
        S.Fs3000Sensor(cfg, raw_reader=lambda: _frame(1000))


# ---------------------------------------------------------------- TSL2591 gain stepping

class FakeTsl:
    def __init__(self, overflow_gains):
        self.overflow_gains = set(overflow_gains)
        self.gain = None
        self.integration_time = "IT100"
        self.set_gains = []

    def __setattr__(self, k, v):
        if k == "gain" and v is not None:
            self.__dict__.setdefault("set_gains", []).append(v)
        object.__setattr__(self, k, v)

    @property
    def lux(self):
        if self.gain in self.overflow_gains:
            raise RuntimeError("Overflow reading light channels!")
        return 1234.0

    visible, infrared, full_spectrum = 100, 20, 120


GAINS = {"low": "GL", "medium": "GM", "high": "GH", "max": "GX"}
IT = {"IT100": 100}


def test_tsl_steps_down_and_remembers(default_config):
    d = FakeTsl(overflow_gains={"GH"})
    s = S.Tsl2591Sensor(default_config, driver=d, gain_constants=GAINS, integration_ms=IT)
    r = s.read()
    assert r.status == S.OK and r.gain == "medium" and r.lux == 1234.0 and r.integration_ms == 100
    d.overflow_gains = set()                       # light dropped: still medium, never steps back up
    assert s.read().gain == "medium"
    assert "GX" not in d.set_gains                 # max never used unless config allows it


def test_tsl_all_gains_overflow(default_config):
    d = FakeTsl(overflow_gains={"GH", "GM", "GL"})
    r = S.Tsl2591Sensor(default_config, driver=d, gain_constants=GAINS, integration_ms=IT).read()
    assert r.status == S.OVERFLOW and r.gain == "low" and math.isnan(r.lux)


# ---------------------------------------------------------------- MLX90640 retries

class FakeMlx:
    def __init__(self, fails):
        self.fails, self.calls = fails, 0

    def getFrame(self, frame):
        self.calls += 1
        if self.calls <= self.fails:
            raise ValueError("Frame data error")
        frame[:] = [25.0] * 768


def test_mlx_retries_then_ok(default_config):
    d = FakeMlx(fails=2)
    r = S.Mlx90640Sensor(default_config, driver=d).read()
    assert r.status == S.OK and r.frame_c.shape == (24, 32) and r.retries == 2


def test_mlx_gives_up_after_max_retries(default_config):
    d = FakeMlx(fails=99)
    r = S.Mlx90640Sensor(default_config, driver=d).read()
    assert r.status == S.READ_ERROR and r.frame_c is None and d.calls == 5


# ---------------------------------------------------------------- BME280

class FakeBme:
    temperature, relative_humidity, pressure = 23.4, 48.0, 996.5


def test_bme_ok_and_error(default_config):
    r = S.Bme280Sensor(default_config, driver=FakeBme()).read()
    assert r.status == S.OK and r.air_temp_c == 23.4 and r.pressure_hpa == 996.5

    class Bad:
        @property
        def temperature(self):
            raise OSError("I2C")
    assert S.Bme280Sensor(default_config, driver=Bad()).read().status == S.READ_ERROR


# ---------------------------------------------------------------- simulated and null (M2)

@pytest.mark.parametrize("fan, lo, hi", [("off", -0.2, 0.25), ("low", 0.3, 1.2), ("high", 1.3, 3.2)])
def test_simulated_wind_follows_fan_state(default_config, fan, lo, hi):
    s = S.SimulatedFs3000Sensor(default_config, fan_state=fan, seed=1)
    vals = [s.read().wind_ms for _ in range(50)]
    assert all(lo <= v <= hi for v in vals) and all(v >= 0 for v in vals)


def test_simulated_light_follows_light_source(default_config):
    a = S.SimulatedTsl2591Sensor(default_config, light_source="led_lamp", seed=0).read()
    b = S.SimulatedTsl2591Sensor(default_config, light_source="sunlight", seed=0).read()
    assert a.status == S.OK and b.lux > 3 * a.lux


def test_simulated_thermal_frame_has_leaf_and_card(default_config):
    sim = default_config["sensors"]["simulated"]
    f = S.SimulatedMlx90640Sensor(default_config, seed=0).read().frame_c
    r0, r1, c0, c1 = sim["leaf_block"]
    k0, k1, l0, l1 = sim["card_block"]
    assert f.shape == (24, 32)
    assert f[r0:r1, c0:c1].mean() < f.mean() < f[k0:k1, l0:l1].mean()


def test_simulated_is_reproducible(default_config):
    a = S.SimulatedBme280Sensor(default_config, seed=3).read()
    b = S.SimulatedBme280Sensor(default_config, seed=3).read()
    assert a == b


def test_null_sensors_blank(default_config):
    for cls in (S.NullBme280Sensor, S.NullTsl2591Sensor, S.NullMlx90640Sensor, S.NullFs3000Sensor):
        r = cls().read()
        assert r.status == S.UNAVAILABLE


# ---------------------------------------------------------------- build_sensors and hardware isolation

def test_build_sensors_simulate(default_config):
    cfg = copy.deepcopy(default_config)
    b = S.build_sensors(cfg, simulate=True, light_source="led_lamp", fan_state="low")
    assert isinstance(b.bme280, S.SimulatedBme280Sensor) and isinstance(b.mlx90640, S.SimulatedMlx90640Sensor)
    assert isinstance(b.fs3000, S.NullFs3000Sensor)          # disabled in config
    cfg["sensors"]["fs3000"]["enabled"] = True
    assert isinstance(S.build_sensors(cfg, simulate=True, fan_state="low").fs3000, S.SimulatedFs3000Sensor)


def test_build_sensors_real_on_laptop_raises_clear_error(default_config):
    with pytest.raises(S.SensorUnavailable, match="Raspberry Pi"):
        S.build_sensors(default_config, simulate=False)


def test_importing_sensors_loads_no_hardware_library():
    code = ("import sys, acquisition.sensors, acquisition.camera_capture; "
            "bad = [m for m in ('board', 'adafruit_bme280', 'adafruit_tsl2591', 'adafruit_mlx90640', 'smbus2', "
            "'picamera2', 'qwiic_fs3000') if m in sys.modules]; print(bad)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(S.REPO_ROOT))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"
