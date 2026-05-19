from pathlib import Path

from final_strategy_source_only_core import ROOT, build_source_panel


def main() -> None:
    out = ROOT / "results" / "main_pipeline_final" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    panel = build_source_panel()
    panel.to_csv(out / "source_canonical_panel.csv", index=False)
    print("PASS source-only data prepare")
    print(f"rows: {len(panel)}")
    print(f"output: {(out / 'source_canonical_panel.csv').relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
