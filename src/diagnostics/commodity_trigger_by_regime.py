"""Canonical entrypoint for commodity crash transmission by regime diagnostics."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    runpy.run_path(str(ROOT / "src" / "analysis" / "commodity_crash_transmission_by_regime.py"), run_name="__main__")
