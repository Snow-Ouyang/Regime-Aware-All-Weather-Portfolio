from final_strategy_source_only_core import ROOT, build_source_panel


def main() -> None:
    out = ROOT / "results" / "main_pipeline_final" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    panel = build_source_panel()
    regime = panel[
        ["date", "TERM_SPREAD_10Y_1Y", "GS10", "macro_regime_raw", "macro_regime_confirmed", "refined_regime_raw", "refined_regime_confirmed"]
    ].copy()
    regime.to_csv(out / "regime_summary_source_panel.csv", index=False)
    counts = panel["refined_regime_confirmed"].value_counts().rename_axis("regime").reset_index(name="n_days")
    counts.to_csv(out / "regime_summary.csv", index=False)
    print("PASS source-only rule-based regime")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
