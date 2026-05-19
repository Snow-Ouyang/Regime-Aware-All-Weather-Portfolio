"""Canonical entrypoint for BACKBONE_V2_UPGRADED stress timing."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    runpy.run_path(str(ROOT / "src" / "strategies" / "spy_cash_backbone_upgrade_ablation.py"), run_name="__main__")
