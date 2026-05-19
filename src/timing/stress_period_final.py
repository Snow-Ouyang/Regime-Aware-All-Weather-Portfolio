"""Canonical entrypoint for final stress-period timeline diagnostics."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    runpy.run_path(str(ROOT / "src" / "strategies" / "backbone_v2_with_steep_commodity_stress.py"), run_name="__main__")
