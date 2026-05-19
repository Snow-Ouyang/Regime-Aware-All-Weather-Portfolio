"""Run the final strategy from clean source data only."""

from __future__ import annotations

from pathlib import Path

from final_strategy_source_only_core import FINAL_STRATEGY, ROOT, SPY_BUY_HOLD, SPY_CASH_TIMING, write_source_only_outputs


def main() -> None:
    out_dir = ROOT / "results" / "final_strategy_source_only"
    panel, perf = write_source_only_outputs(out_dir)
    print("Source-only final strategy complete.")
    print(f"output_dir: {out_dir.relative_to(ROOT).as_posix()}")
    print(f"rows: {len(panel)}")
    show = perf[perf["strategy"].isin([SPY_BUY_HOLD, SPY_CASH_TIMING, FINAL_STRATEGY])]
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
