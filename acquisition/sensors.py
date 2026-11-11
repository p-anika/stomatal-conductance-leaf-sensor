"""Sensor classes for the logger (T7.1, spec M1, M2). [PI for the real classes only]

Every sensor has three versions with the same read():
    <Name>Sensor            the real sensor on the Pi's I2C bus. Hardware libraries (board, adafruit_*,
                            smbus2) are imported INSIDE the constructor, so this module imports on a laptop.
                            A missing library or sensor raises SensorUnavailable with a clear message.
    Simulated<Name>Sensor   plausible values with noise (config sensors.simulated, INVENTED), following
                            the session's light_source and fan_state, for laptop runs and tests.
    Null<Name>Sensor        blanks (NaN) with status "unavailable", for a disabled sensor.
read() returns a small frozen record with the values and a status:
    ok, overflow (beyond the sensor's range), read_error (I2C or frame error), checksum_error (FS3000),
    unavailable (Null sensor).

BME280: air_temp_c [deg C], humidity_pct [%], pressure_hpa [hPa].
TSL2591: lux, visible, infrared, full_spectrum (raw counts), gain, integration_ms. Starts at
    sensors.tsl2591.start_gain (high); on the overflow RuntimeError it steps the gain DOWN
    (high -> medium -> low) and retries. The gain that worked is kept for the next read; it never
    steps back up (limitation: after a bright moment, dim readings stay at the lower gain). Max gain
    is used only if sensors.tsl2591.allow_max_gain is true.
MLX90640: frame_c, a 24 x 32 array [deg C]; getFrame is retried on ValueError up to
    sensors.mlx90640_max_retries times; refresh rate acquisition.thermal_refresh_hz.
FS3000 (Renesas datasheet R36DS0009EU0100 Rev.1.00): read 5 bytes from address 0x28:
    byte 1 checksum, byte 2 flow high byte (only the lower 4 bits valid), byte 3 flow low byte,
    bytes 4-5 generic checksum data (byte layout: datasheet Figure 7, UNVERIFIED on our module).
    Valid when (sum of all 5 bytes) mod 256 == 0 (datasheet 5.3: the checksum is the two's complement
    of the 8-bit sum of the data bytes). count = ((byte2 & 0x0F) << 8) | byte3 (12 bit).
    wind_ms = linear interpolation in sensors.fs3000_table_1005 (datasheet Figure 2; linear between
    points is an ASSUMPTION); count <= 409 -> 0 m/s; count > 3686 -> overflow. Accuracy 5 % of full
    scale (about +/- 0.36 m/s for the 1005). Only variant 1005 has a table.
build_sensors(cfg, simulate, light_source, fan_state) picks the versions from config.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from acquisition.camera_capture import ArducamCamera, CameraUnavailable, NullCamera  # noqa: F401 (re-export)

REPO_ROOT = Path(__file__).resolve().parent.parent
OK, OVERFLOW, READ_ERROR, CHECKSUM_ERROR, UNAVAILABLE = "ok", "overflow", "read_error", "checksum_error", "unavailable"
FRAME_SHAPE = (24, 32)                       # MLX90640 pixels (datasheet)
GAIN_ORDER = ("max", "high", "medium", "low")
NAN = float("nan")


class SensorUnavailable(RuntimeError):
    """A real sensor cannot be used: hardware library missing (not a Pi) or the sensor did not answer."""


def _need_pi(name: str, err: Exception) -> SensorUnavailable:
    return SensorUnavailable(f"{name}: hardware library or sensor not available ({err}). This needs the "
                             "Raspberry Pi with the sensor connected; on a laptop use simulate=True")


# ---------------------------------------------------------------- records

@dataclass(frozen=True)
class Bme280Reading:
    air_temp_c: float = NAN
    humidity_pct: float = NAN
    pressure_hpa: float = NAN
    status: str = UNAVAILABLE


@dataclass(frozen=True)
class Tsl2591Reading:
    lux: float = NAN
    visible: float = NAN
    infrared: float = NAN
    full_spectrum: float = NAN
    gain: str = ""
    integration_ms: float = NAN
    status: str = UNAVAILABLE


@dataclass(frozen=True, eq=False)
class ThermalFrame:
    frame_c: np.ndarray | None = None
    status: str = UNAVAILABLE
    retries: int = 0


@dataclass(frozen=True)
class WindReading:
    wind_ms: float = NAN
    wind_raw: float = NAN
    status: str = UNAVAILABLE


# ---------------------------------------------------------------- BME280

class Bme280Sensor:
    """BME280 on I2C (adafruit_bme280). [PI]"""

    def __init__(self, cfg: dict[str, Any], driver: Any = None):
        if driver is None:
            try:
                import adafruit_bme280.basic as adafruit_bme280
                import board
                driver = adafruit_bme280.Adafruit_BME280_I2C(board.I2C())
            except (ImportError, NotImplementedError, RuntimeError, ValueError, OSError) as e:
                raise _need_pi("BME280", e) from e
        self.driver = driver

    def read(self) -> Bme280Reading:
        try:
            return Bme280Reading(float(self.driver.temperature), float(self.driver.relative_humidity),
                                 float(self.driver.pressure), OK)
        except (OSError, RuntimeError, ValueError):
            return Bme280Reading(status=READ_ERROR)


# ---------------------------------------------------------------- TSL2591

def _tsl_gain_order(cfg: dict[str, Any]) -> list[str]:
    t = cfg["sensors"]["tsl2591"]
    names = [g for g in GAIN_ORDER if g != "max" or t.get("allow_max_gain", False)]
    start = str(t.get("start_gain", "high"))
    if start not in names:
        raise ValueError(f"sensors.tsl2591.start_gain must be one of {names}, got {start}")
    return names[names.index(start):]


class Tsl2591Sensor:
    """TSL2591 light sensor (adafruit_tsl2591), gain stepped DOWN on overflow. [PI]

    gain_constants / integration_ms let tests pass a fake driver: {gain name: driver constant} and
    {driver integration constant: milliseconds}."""

    def __init__(self, cfg: dict[str, Any], driver: Any = None, gain_constants: dict[str, Any] | None = None,
                 integration_ms: dict[Any, float] | None = None):
        if driver is None:
            try:
                import adafruit_tsl2591
                import board
                driver = adafruit_tsl2591.TSL2591(board.I2C())
            except (ImportError, NotImplementedError, RuntimeError, ValueError, OSError) as e:
                raise _need_pi("TSL2591", e) from e
            m = adafruit_tsl2591
            gain_constants = {"low": m.GAIN_LOW, "medium": m.GAIN_MED, "high": m.GAIN_HIGH, "max": m.GAIN_MAX}
            integration_ms = {getattr(m, f"INTEGRATIONTIME_{ms}MS"): float(ms) for ms in (100, 200, 300, 400, 500, 600)}
        self.driver = driver
        self.gains = gain_constants or {}
        self.it_ms = integration_ms or {}
        self.order = _tsl_gain_order(cfg)
        self._i = 0                                  # index into order of the gain that last worked

    def read(self) -> Tsl2591Reading:
        it = self.it_ms.get(getattr(self.driver, "integration_time", None), NAN)
        while True:
            name = self.order[self._i]
            try:
                self.driver.gain = self.gains[name]
                return Tsl2591Reading(float(self.driver.lux), float(self.driver.visible), float(self.driver.infrared),
                                      float(self.driver.full_spectrum), name, it, OK)
            except RuntimeError:                     # adafruit_tsl2591 raises this on overflow
                if self._i + 1 < len(self.order):
                    self._i += 1                     # step down; never back up (documented limitation)
                    continue
                return Tsl2591Reading(gain=name, integration_ms=it, status=OVERFLOW)
            except (OSError, ValueError):
                return Tsl2591Reading(gain=name, integration_ms=it, status=READ_ERROR)


# ---------------------------------------------------------------- MLX90640

class Mlx90640Sensor:
    """MLX90640 thermal camera (adafruit_mlx90640), 24 x 32 deg C. [PI]"""

    def __init__(self, cfg: dict[str, Any], driver: Any = None):
        self.max_retries = int(cfg["sensors"]["mlx90640_max_retries"])
        if driver is None:
            try:
                import adafruit_mlx90640
                import board
                driver = adafruit_mlx90640.MLX90640(board.I2C())
                hz = cfg["acquisition"]["thermal_refresh_hz"]
                name = f"REFRESH_{int(hz)}_HZ" if float(hz) >= 1 else "REFRESH_0_5_HZ"
                driver.refresh_rate = getattr(adafruit_mlx90640.RefreshRate, name)
            except (ImportError, NotImplementedError, RuntimeError, ValueError, OSError, AttributeError) as e:
                raise _need_pi("MLX90640", e) from e
        self.driver = driver

    def read(self) -> ThermalFrame:
        frame = [0.0] * (FRAME_SHAPE[0] * FRAME_SHAPE[1])
        for attempt in range(self.max_retries):
            try:
                self.driver.getFrame(frame)
                return ThermalFrame(np.array(frame, dtype=float).reshape(FRAME_SHAPE), OK, attempt)
            except ValueError:                       # occasional bad frame: retry (T7.1)
                continue
            except (OSError, RuntimeError):
                return ThermalFrame(None, READ_ERROR, attempt)
        return ThermalFrame(None, READ_ERROR, self.max_retries)


# ---------------------------------------------------------------- FS3000

def fs3000_checksum_ok(b: bytes) -> bool:
    """Datasheet 5.3: checksum + 8-bit sum of the data bytes == 0 (mod 256)."""
    return len(b) == 5 and sum(b) % 256 == 0


def fs3000_count(b: bytes) -> int:
    """12-bit flow count: lower 4 bits of byte 2, then byte 3 (datasheet Figure 7)."""
    return ((b[1] & 0x0F) << 8) | b[2]


def fs3000_counts_to_wind(count: int, table: dict[str, list[float]]) -> tuple[float, str]:
    """(wind m s-1, status): linear interpolation in the datasheet table; <= first count -> 0;
    above the last count -> (nan, overflow)."""
    counts, speeds = table["counts"], table["wind_ms"]
    if count <= counts[0]:
        return 0.0, OK
    if count > counts[-1]:
        return NAN, OVERFLOW
    return float(np.interp(count, counts, speeds)), OK


class Fs3000Sensor:
    """FS3000 air-velocity sensor, raw I2C read with smbus2 (datasheet decoding). [PI]

    raw_reader: a function returning the 5 bytes (tests pass a fake)."""

    def __init__(self, cfg: dict[str, Any], raw_reader: Callable[[], bytes] | None = None):
        s = cfg["sensors"]
        variant = int(s["fs3000"]["variant"])
        key = f"fs3000_table_{variant}"
        if key not in s:
            raise ValueError(f"no count-to-m/s table for FS3000-{variant} in config (only 1005 is from the "
                             "datasheet so far); add sensors.{key} from the datasheet first")
        self.table = s[key]
        if raw_reader is None:
            address = int(s["fs3000_i2c_address"])
            try:
                from smbus2 import SMBus, i2c_msg
                bus = SMBus(1)
            except (ImportError, OSError, FileNotFoundError, PermissionError) as e:
                raise _need_pi("FS3000", e) from e

            def raw_reader() -> bytes:
                msg = i2c_msg.read(address, 5)
                bus.i2c_rdwr(msg)
                return bytes(list(msg))
        self.raw_reader = raw_reader

    def read(self) -> WindReading:
        try:
            b = bytes(self.raw_reader())
        except (OSError, RuntimeError, ValueError):
            return WindReading(status=READ_ERROR)
        if not fs3000_checksum_ok(b):
            return WindReading(status=CHECKSUM_ERROR)
        count = fs3000_count(b)
        wind, status = fs3000_counts_to_wind(count, self.table)
        return WindReading(wind, float(count), status)


# ---------------------------------------------------------------- simulated (M2) and null

class _Sim:
    def __init__(self, cfg: dict[str, Any], seed: int = 0):
        self.sim = cfg["sensors"]["simulated"]
        self.rng = np.random.default_rng(seed)

    def noise(self, key: str) -> float:
        return float(self.rng.normal(0.0, float(self.sim["noise_sd"][key])))


class SimulatedBme280Sensor(_Sim):
    def read(self) -> Bme280Reading:
        s = self.sim
        return Bme280Reading(s["air_temp_c"] + self.noise("air_temp_c"), s["humidity_pct"] + self.noise("humidity_pct"),
                             s["pressure_hpa"] + self.noise("pressure_hpa"), OK)


class SimulatedTsl2591Sensor(_Sim):
    def __init__(self, cfg: dict[str, Any], light_source: str | None = None, seed: int = 0):
        super().__init__(cfg, seed)
        self.base_lux = float(self.sim["lux_by_light_source"].get(light_source or "led_lamp",
                                                                   self.sim["lux_by_light_source"]["led_lamp"]))

    def read(self) -> Tsl2591Reading:
        lux = max(self.base_lux * (1.0 + self.noise("lux_fraction")), 0.0)
        return Tsl2591Reading(lux, NAN, NAN, NAN, "simulated", NAN, OK)


class SimulatedMlx90640Sensor(_Sim):
    """Air temperature everywhere, a cooler leaf block and a warmer card block, plus pixel noise."""

    def read(self) -> ThermalFrame:
        s = self.sim
        f = np.full(FRAME_SHAPE, float(s["air_temp_c"]))
        r0, r1, c0, c1 = s["leaf_block"]
        f[r0:r1, c0:c1] += float(s["leaf_offset_c"])
        r0, r1, c0, c1 = s["card_block"]
        f[r0:r1, c0:c1] += float(s["card_offset_c"])
        f += self.rng.normal(0.0, float(s["noise_sd"]["pixel_c"]), FRAME_SHAPE)
        return ThermalFrame(f, OK, 0)


class SimulatedFs3000Sensor(_Sim):
    """Wind follows fan_state: midpoint of sensors.simulated.wind_ms_by_fan_state[fan_state] plus noise,
    kept inside that range and never negative (M2)."""

    def __init__(self, cfg: dict[str, Any], fan_state: str | None = None, seed: int = 0):
        super().__init__(cfg, seed)
        lo, hi = self.sim["wind_ms_by_fan_state"][fan_state or "off"]
        self.lo, self.hi = float(lo), float(hi)

    def read(self) -> WindReading:
        w = min(max((self.lo + self.hi) / 2.0 + self.noise("wind_ms"), self.lo, 0.0), self.hi)
        return WindReading(w, NAN, OK)


class NullBme280Sensor:
    def read(self) -> Bme280Reading:
        return Bme280Reading()


class NullTsl2591Sensor:
    def read(self) -> Tsl2591Reading:
        return Tsl2591Reading()


class NullMlx90640Sensor:
    def read(self) -> ThermalFrame:
        return ThermalFrame()


class NullFs3000Sensor:
    def read(self) -> WindReading:
        return WindReading()


# ---------------------------------------------------------------- build

@dataclass
class SensorSet:
    bme280: Any
    tsl2591: Any
    mlx90640: Any
    fs3000: Any
    camera: Any = field(default_factory=NullCamera)


def build_sensors(cfg: dict[str, Any], simulate: bool = False, light_source: str | None = None,
                  fan_state: str | None = None, seed: int = 0) -> SensorSet:
    """Real, Simulated or Null version of each sensor from config: a disabled sensor is Null;
    simulate=True gives Simulated sensors and a NullCamera. Real sensors raise SensorUnavailable
    when their library or hardware is missing."""
    s = cfg["sensors"]
    on = {"bme280": s["bme280"]["enabled"], "tsl2591": s["tsl2591"]["enabled"],
          "mlx90640": s.get("mlx90640", {}).get("enabled", True), "fs3000": s["fs3000"]["enabled"]}
    if simulate:
        return SensorSet(
            SimulatedBme280Sensor(cfg, seed) if on["bme280"] else NullBme280Sensor(),
            SimulatedTsl2591Sensor(cfg, light_source, seed + 1) if on["tsl2591"] else NullTsl2591Sensor(),
            SimulatedMlx90640Sensor(cfg, seed + 2) if on["mlx90640"] else NullMlx90640Sensor(),
            SimulatedFs3000Sensor(cfg, fan_state, seed + 3) if on["fs3000"] else NullFs3000Sensor(),
            NullCamera())
    return SensorSet(
        Bme280Sensor(cfg) if on["bme280"] else NullBme280Sensor(),
        Tsl2591Sensor(cfg) if on["tsl2591"] else NullTsl2591Sensor(),
        Mlx90640Sensor(cfg) if on["mlx90640"] else NullMlx90640Sensor(),
        Fs3000Sensor(cfg) if on["fs3000"] else NullFs3000Sensor(),
        ArducamCamera(cfg) if s["arducam"]["enabled"] else NullCamera())
