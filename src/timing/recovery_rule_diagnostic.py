"""Canonical entrypoint for recovery rule diagnostics."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    runpy.run_path(str(ROOT / "src" / "analysis" / "stress_recovery_grid_search.py"), run_name="__main__")
