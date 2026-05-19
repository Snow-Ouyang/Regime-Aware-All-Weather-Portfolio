"""Canonical entrypoint for monthly SPY timing diagnostics."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    runpy.run_path(str(ROOT / "src" / "analysis" / "monthly_either_crash_brake_diagnostics.py"), run_name="__main__")
