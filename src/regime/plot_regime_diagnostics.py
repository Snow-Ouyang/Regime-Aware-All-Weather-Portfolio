"""Canonical entrypoint for rule-based regime diagnostic plots."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    runpy.run_path(str(ROOT / "src" / "regime" / "plot_regime_outputs.py"), run_name="__main__")
