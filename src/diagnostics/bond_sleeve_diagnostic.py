"""Canonical entrypoint for bond sleeve selection diagnostics."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    runpy.run_path(str(ROOT / "src" / "analysis" / "evaluate_bond_sleeve_candidates.py"), run_name="__main__")
