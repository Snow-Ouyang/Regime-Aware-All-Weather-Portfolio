from final_strategy_source_only_core import ASSETS, FINAL_STRATEGY, ROOT, build_final_source_only_panel


FINAL = FINAL_STRATEGY


def main() -> None:
    out = ROOT / "results" / "main_pipeline_final" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    panel, perf = build_final_source_only_panel()
    ret_cols = [
        "date",
        f"{FINAL}_return",
        f"{FINAL}_nav",
        f"{FINAL}_drawdown",
        f"{FINAL}_turnover",
        f"{FINAL}_transaction_cost",
        "final_state",
        "final_allocation_state",
        "trigger_lock_active_locks",
        "trigger_lock_entry_signal",
        "trigger_lock_exit_signal",
    ]
    weight_cols = ["date"] + [f"{FINAL}_weight_{asset}" for asset in ASSETS]
    panel[ret_cols].to_csv(out / "final_daily_returns.csv", index=False)
    panel[weight_cols].to_csv(out / "final_daily_weights.csv", index=False)
    display = perf.loc[perf["strategy"].isin(["SPY_BUY_HOLD", "SPY_CASH_TIMING", FINAL])].copy()
    display.to_csv(out / "strategy_performance_comparison.csv", index=False)
    print("PASS source-only trigger-lock final strategy")
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
