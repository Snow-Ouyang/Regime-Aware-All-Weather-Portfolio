"""Canonical entrypoint for FLAT_RISK GOLD vs CASH diagnostics."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    runpy.run_path(str(ROOT / "src" / "analysis" / "flat_risk_gold_cash_diagnostic.py"), run_name="__main__")
