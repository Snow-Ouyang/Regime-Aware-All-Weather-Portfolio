# Hard Dependency Validation

Overall status: **PASS**

## Rule

The final main pipeline is considered source-only only if numbered main scripts do not read non-main `results/` folders as required inputs, and a source-data rebuild matches the current final reference outputs.

## Dependency Audit

- Main scripts checked: 10
- Forbidden dependency rows: 0

| script                                              | status   | forbidden_reference   | line   | reason                                   |
|:----------------------------------------------------|:---------|:----------------------|:-------|:-----------------------------------------|
| scripts/01_data_prepare.py                          | PASS     |                       |        | no forbidden results dependency detected |
| scripts/02_rule_based_regime.py                     | PASS     |                       |        | no forbidden results dependency detected |
| scripts/03_stress_detection.py                      | PASS     |                       |        | no forbidden results dependency detected |
| scripts/04_asset_return_panel.py                    | PASS     |                       |        | no forbidden results dependency detected |
| scripts/05_baseline_strategy.py                     | PASS     |                       |        | no forbidden results dependency detected |
| scripts/06_flat_rate_refined_strategy.py            | PASS     |                       |        | no forbidden results dependency detected |
| scripts/07_cross_state_asset_behavior.py            | PASS     |                       |        | no forbidden results dependency detected |
| scripts/08_stress_trigger_diagnostics.py            | PASS     |                       |        | no forbidden results dependency detected |
| scripts/09_final_strategy_recovery_flat_low_only.py | PASS     |                       |        | no forbidden results dependency detected |
| scripts/10_final_report_outputs.py                  | PASS     |                       |        | no forbidden results dependency detected |

## Source Rebuild Consistency

- Fields checked: 10
- Failed fields: 0
- Intentional historical-reference differences: 0

| field                  | status   |    n |   match_rate |   mismatch_count |   max_abs_diff |   mean_abs_diff | reason   |
|:-----------------------|:---------|-----:|-------------:|-----------------:|---------------:|----------------:|:---------|
| macro_regime_confirmed | PASS     | 5074 |            1 |                0 |  nan           |   nan           |          |
| SPY_return             | PASS     | 5074 |            1 |                0 |    0           |     0           |          |
| GOLD_return            | PASS     | 5074 |            1 |                0 |    0           |     0           |          |
| CMDTY_FUT_return       | PASS     | 5074 |            1 |                0 |    0           |     0           |          |
| IEF_return             | PASS     | 5074 |            1 |                0 |    0           |     0           |          |
| CASH_return            | PASS     | 5074 |            1 |                0 |    9.98279e-17 |     1.38998e-17 |          |
| VIX_ZSCORE_120D        | PASS     | 5074 |            1 |                0 |    8.88178e-16 |     3.1305e-17  |          |
| CREDIT_SPREAD_BAA_AAA  | PASS     | 5074 |            1 |                0 |    4.44089e-16 |     2.47032e-17 |          |
| D_CREDIT_SPREAD_20D    | PASS     | 5074 |            1 |                0 |    2.22045e-16 |     4.2006e-17  |          |
| CMDTY_RET60            | PASS     | 5074 |            1 |                0 |    9.97466e-17 |     4.18193e-17 |          |

## Conclusion

The mainline passes dependency-level source-only validation. Remaining field differences are documented as intentional differences versus the historical reference panel, not as source-only dependency failures.
