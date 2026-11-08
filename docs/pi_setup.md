# Raspberry Pi setup

**Status: stub. None of these steps has been checked by running it on the Pi for this version of the project.** Paste the output of each command into a note here once it has been tried.

## 1. Check the Python version (do this first)
```
python3 --version
```
Checked 2026-11-07: the Pi runs Python 3.13.5 on Debian 13 (trixie), aarch64. The laptop runs 3.13.12. The code targets Python 3.13 on both (see docs/decision_log.md). If `python3 --version` ever shows something other than 3.13, tell Claude before running the code.

## 2. Install the hardware libraries
From the repository folder on the Pi:
```
pip install -r requirements-pi.txt
```
The camera library picamera2 comes from apt, not pip (see the note in requirements-pi.txt). These versions are not pinned yet; pin them after the first clean install.

## 3. Turn on I2C and check the sensors are found
- TODO: steps and the expected sensor addresses.

## 4. Clock
The Pi 4 has no battery-backed clock, so its time can be wrong when it has no internet. TODO: how to check the clock is synced before a session.

## 5. Copying data off the Pi
- TODO.
