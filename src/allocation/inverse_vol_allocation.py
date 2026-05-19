"""Canonical entrypoint for inverse-volatility robustness diagnostics."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    runpy.run_path(str(ROOT / "src" / "analysis" / "invvol_window_grid_search.py"), run_name="__main__")
