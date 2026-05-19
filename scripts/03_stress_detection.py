from final_strategy_source_only_core import ROOT, build_backbone_and_states, build_source_panel


def main() -> None:
    out = ROOT / "results" / "main_pipeline_final" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    panel = build_backbone_and_states(build_source_panel())
    cols = [
        "date",
        "macro_regime_confirmed",
        "monthly_either_state",
        "VIX_ZSCORE_120D",
        "D_CREDIT_SPREAD_20D",
        "CMDTY_RET60",
        "FLAT_VIX_STRESS",
        "FLAT_CREDIT_DD5_STRESS",
        "STEEP_EITHER_SELL_STRESS",
        "STEEP_CREDIT_DD5_STRESS",
        "STEEP_CMDTY_RET60_NEG10",
        "BACKBONE_V2_ENTRY_SIGNAL",
        "R3_RECOVERY",
        "full_risk_state",
        "steep_slow_growth_overlay_state",
    ]
    panel[cols].to_csv(out / "stress_detection_panel.csv", index=False)
    print("PASS source-only stress detection")
    print(panel[["FLAT_VIX_STRESS", "FLAT_CREDIT_DD5_STRESS", "STEEP_EITHER_SELL_STRESS", "STEEP_CREDIT_DD5_STRESS", "STEEP_CMDTY_RET60_NEG10"]].sum().to_string())


if __name__ == "__main__":
    main()
