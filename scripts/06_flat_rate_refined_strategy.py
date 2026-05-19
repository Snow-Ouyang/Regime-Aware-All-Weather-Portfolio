from final_strategy_source_only_core import ASSETS, ROOT, build_final_source_only_panel


def main() -> None:
    out = ROOT / "results" / "main_pipeline_final" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    panel, perf = build_final_source_only_panel()
    weight_cols = ["date", "flat_refined_state"] + [f"FLAT_RATE_REFINED_L50_H30_weight_{asset}" for asset in ASSETS]
    panel[weight_cols].to_csv(out / "flat_rate_refined_weights.csv", index=False)
    perf.loc[perf["strategy"].eq("FLAT_RATE_REFINED_L50_H30")].to_csv(out / "flat_rate_refined_performance.csv", index=False)
    print("PASS source-only flat-rate refined strategy")
    print(panel["flat_refined_state"].value_counts().to_string())


if __name__ == "__main__":
    main()
