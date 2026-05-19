from final_strategy_source_only_core import ASSETS, ROOT, build_source_panel


def main() -> None:
    out = ROOT / "results" / "main_pipeline_final" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    panel = build_source_panel()
    cols = ["date"] + [f"{asset}_return" for asset in ASSETS]
    panel[cols].to_csv(out / "asset_return_panel.csv", index=False)
    print("PASS source-only asset return panel")
    print(panel[[f"{asset}_return" for asset in ASSETS]].count().to_string())


if __name__ == "__main__":
    main()
