"""Canonical entrypoint for MATURE_REGIME_HEDGE_FINAL backtest."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    runpy.run_path(str(ROOT / "src" / "strategies" / "mature_regime_hedge_final.py"), run_name="__main__")
