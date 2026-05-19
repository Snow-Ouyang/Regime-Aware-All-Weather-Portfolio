"""Canonical entrypoint for jump-model regime discovery."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    runpy.run_path(str(ROOT / "src" / "regime" / "fit_simplified_regime_model.py"), run_name="__main__")
