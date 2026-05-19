"""Canonical entrypoint for the 2015-2016 missed drawdown forensic diagnostic."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    runpy.run_path(str(ROOT / "src" / "analysis" / "drawdown_2015_2016_forensic_diagnostic.py"), run_name="__main__")
