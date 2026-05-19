from final_strategy_source_only_core import ROOT, build_final_source_only_panel


def main() -> None:
    out = ROOT / "results" / "main_pipeline_final" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    panel, perf = build_final_source_only_panel()
    cols = ["date", "FLAT_RATE_REFINED_L50_H30_return", "FLAT_RATE_REFINED_L50_H30_nav", "FLAT_RATE_REFINED_L50_H30_drawdown"]
    panel[cols].to_csv(out / "baseline_strategy_panel.csv", index=False)
    perf.loc[perf["strategy"].eq("FLAT_RATE_REFINED_L50_H30")].to_csv(out / "baseline_strategy_performance.csv", index=False)
    print("PASS source-only baseline strategy")
    print(perf.loc[perf["strategy"].eq("FLAT_RATE_REFINED_L50_H30")].to_string(index=False))


if __name__ == "__main__":
    main()
