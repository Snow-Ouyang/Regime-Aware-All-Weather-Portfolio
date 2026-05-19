"""Canonical entrypoint for fixed, inverse-vol, and ERC allocation comparison."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    runpy.run_path(str(ROOT / "src" / "strategies" / "regime_aware_risk_parity_allocation.py"), run_name="__main__")
