"""Canonical entrypoint for rule-based FLAT/STEEP/INVERTED regime construction."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    runpy.run_path(str(ROOT / "src" / "regime" / "build_regime_dataset.py"), run_name="__main__")
