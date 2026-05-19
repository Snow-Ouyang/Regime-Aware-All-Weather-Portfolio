"""Canonical entrypoint for regime x stress asset behavior diagnostics."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    runpy.run_path(str(ROOT / "src" / "analysis" / "hedge_asset_cross_state_diagnostic_extended.py"), run_name="__main__")
