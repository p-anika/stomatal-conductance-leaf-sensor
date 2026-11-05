"""Save software and OS versions for the paper's Methods section.

Run from the repo root:  python scripts/snapshot_versions.py --label laptop
(on the Raspberry Pi use --label pi). Writes docs/versions/<label>_<date>.md.
"""
import argparse
import datetime
import platform
import subprocess
import sys
from pathlib import Path

PACKAGES = ["numpy", "scipy", "pandas", "pyyaml", "scikit-learn", "joblib",
            "opencv-python-headless", "matplotlib", "statsmodels", "pytest",
            "adafruit-circuitpython-bme280", "adafruit-circuitpython-tsl2591",
            "adafruit-circuitpython-mlx90640", "adafruit-blinka", "smbus2"]


def package_version(name):
    try:
        from importlib import metadata
        return metadata.version(name)
    except Exception:
        return "not installed"


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unavailable"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True, help="laptop or pi")
    args = ap.parse_args()
    today = datetime.date.today().isoformat()
    lines = [f"# Software versions: {args.label}, {today}", "",
             f"- Python: {sys.version.split()[0]} ({sys.executable})",
             f"- OS: {platform.platform()}",
             f"- Machine: {platform.machine()}",
             f"- Git commit: {run(['git', 'rev-parse', '--short', 'HEAD']) or 'none'}"]
    if Path("/etc/os-release").exists():
        pretty = [l for l in Path("/etc/os-release").read_text().splitlines() if l.startswith("PRETTY_NAME")]
        if pretty:
            lines.append(f"- Distribution: {pretty[0].split('=', 1)[1].strip(chr(34))}")
    lines += ["", "## Packages", "", "| Package | Version |", "| --- | --- |"]
    lines += [f"| {p} | {package_version(p)} |" for p in PACKAGES]
    out = Path("docs/versions")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.label}_{today}.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
