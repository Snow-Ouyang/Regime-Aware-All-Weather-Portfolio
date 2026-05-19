# Mainline Independence Check

Status: PASS

The active mainline was rerun after moving old intermediate outputs out of `results/` and `figures/`.

Run order verified:

1. `python scripts/01_data_prepare.py`
2. `python scripts/02_rule_based_regime.py`
3. `python scripts/03_stress_detection.py`
4. `python scripts/04_asset_return_panel.py`
5. `python scripts/05_baseline_strategy.py`
6. `python scripts/06_flat_rate_refined_strategy.py`
7. `python scripts/07_cross_state_asset_behavior.py`
8. `python scripts/08_stress_trigger_diagnostics.py`
9. `python scripts/09_final_strategy_recovery_flat_low_only.py`
10. `python scripts/10_final_report_outputs.py`

Active kept output:

- `results/main_pipeline_final/`

Moved out of active tree:

- old `results/*` folders except `main_pipeline_final`
- old `figures/*` folders
- stale recovery-related files under `results/main_pipeline_final`

Archive location:

- `archive/intermediate_outputs_20260518_182405/`

Final displayed strategies:

- `SPY_BUY_HOLD`
- `SPY_CASH_TIMING`
- `FINAL_REGIME_HEDGE_TRIGGER_LOCK`

Final performance after clean rerun:

| strategy | CAGR | Sharpe | MaxDD | Final Equity | Turnover |
|---|---:|---:|---:|---:|---:|
| SPY_BUY_HOLD | 11.14% | 0.575 | -55.19% | 8.38 | 1.00 |
| SPY_CASH_TIMING | 12.04% | 0.948 | -29.45% | 9.86 | 141.00 |
| FINAL_REGIME_HEDGE_TRIGGER_LOCK | 16.92% | 1.347 | -20.07% | 23.28 | 160.72 |
