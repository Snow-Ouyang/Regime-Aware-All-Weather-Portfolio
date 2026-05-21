# FINAL_CREDIT_Z_UNLOCK_GRID_REPORT

## 1. Purpose

This grid search migrates the most interpretable SPY/CASH credit-lab idea back into the full final strategy: absolute price-confirmed credit widening on entry, plus credit level z-score confirmation on unlock.

## 2. Why this rule

Entry uses SPY drawdown plus absolute credit widening. Unlock requires both short-term credit improvement and credit spread level normalization, to reduce premature exits while spreads remain structurally elevated.

## 3. Grid design

Only credit-related parameters move. VIX, commodity, allocation, regime framework, transaction cost, and execution timing are unchanged.

## 4. Baseline alignment

|   daily_return_correlation_with_main_pipeline_final |   max_abs_daily_return_diff |   mismatched_stress_days |   baseline_CAGR |   baseline_Sharpe |   baseline_MaxDD |   baseline_Final_Equity | alignment_status                               |
|----------------------------------------------------:|----------------------------:|-------------------------:|----------------:|------------------:|-----------------:|------------------------:|:-----------------------------------------------|
|                                            0.951033 |                   0.0725059 |                      159 |         0.20412 |           1.47453 |        -0.223574 |                 42.0998 | MISMATCH_EXPECTED_AFTER_DAILY_CREDIT_MIGRATION |

## 5. Full-sample results

- Top Sharpe candidate: `CW20_ABS15_ZW126_UZ1P5`
- Top composite candidate: `CW20_ABS15_ZW126_UZ1P5`

## 6. Crisis window results

See `grid_crisis_window_comparison.csv` and the case-study plots for 2008, 2022, COVID, and 2025.

## 7. False recovery / missed rebound trade-off

The key question is whether stricter z-score unlock reduces false recovery without paying too much in missed rebound.

## 8. Parameter sensitivity

See `parameter_sensitivity_summary.csv`. If only isolated parameter points work and neighbors do not, the result should be treated as unstable.

## 9. Materially better candidate test

At least one materially better candidate exists.

## 10. Recommendation

A specific z-score unlock challenger is strong enough for the next final-strategy test.

## 11. Limitations

- credit episodes are sparse
- z-score windows may still overfit in-sample
- this is not out-of-sample validation