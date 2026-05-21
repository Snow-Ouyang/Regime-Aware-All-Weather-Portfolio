# FINAL_DAILY_CREDIT_REPLACEMENT_REPORT

## 1. Purpose

Test whether the best daily-credit redesign variants can replace the current baseline credit trigger inside the unchanged final regime-hedge strategy.

## 2. Motivation

- VIX can handle fast panic and fast relief.
- Credit should focus on sustained or stair-step elevated credit stress, especially 2008 and 2022.

## 3. Rules tested

- FINAL_BASELINE
- FINAL_LEVEL_OR_PERCENTILE_LOCK
- FINAL_LEVEL_LOCK_FAST_RELIEF
- FINAL_LEVEL_LOCK_FAST_RELIEF_PLUS_RELOCK
- FINAL_ABS_ENTRY_LEVEL_Z_UNLOCK
- FINAL_SHOCK_OR_Z
- FINAL_WATCH_AS_PARTIAL_LOCK_DIAGNOSTIC

## 4. Baseline alignment

|   daily_return_correlation_with_main_pipeline_final |   max_abs_daily_return_diff |   mismatched_stress_days |   baseline_CAGR |   baseline_Sharpe |   baseline_MaxDD |   baseline_Final_Equity |
|----------------------------------------------------:|----------------------------:|-------------------------:|----------------:|------------------:|-----------------:|------------------------:|
|                                                   1 |                           0 |                        0 |        0.201968 |           1.49221 |        -0.159357 |                 40.6104 |

## 5. Full-sample performance

| credit_variant                 |     CAGR |   Sharpe |     MaxDD |   Calmar |   Final Equity |   false_recovery_count |   missed_rebound_count |
|:-------------------------------|---------:|---------:|----------:|---------:|---------------:|-----------------------:|-----------------------:|
| FINAL_BASELINE                 | 0.201968 |  1.49221 | -0.159357 | 1.2674   |        40.6104 |                      5 |                      4 |
| FINAL_LEVEL_OR_PERCENTILE_LOCK | 0.193536 |  1.43218 | -0.169051 | 1.14484  |        35.2436 |                      7 |                      6 |
| FINAL_LEVEL_LOCK_FAST_RELIEF   | 0.198029 |  1.46058 | -0.206862 | 0.957298 |        38.0129 |                     15 |                      5 |
| FINAL_LEVEL_FAST_RELIEF_RELOCK | 0.197584 |  1.45999 | -0.206862 | 0.955149 |        37.7298 |                     21 |                      5 |
| FINAL_ABS_ENTRY_LEVEL_Z_UNLOCK | 0.187091 |  1.41095 | -0.169051 | 1.10671  |        31.6033 |                      6 |                      6 |
| FINAL_SHOCK_OR_Z               | 0.204375 |  1.52682 | -0.200093 | 1.0214   |        42.2799 |                     10 |                      6 |
| FINAL_WATCH_AS_LOCK            | 0.20098  |  1.50557 | -0.206862 | 0.971567 |        39.9437 |                     20 |                      6 |

## 6. Crisis window analysis

| strategy                                  | credit_variant                 | window         |   cumulative_return |   max_drawdown |   Sharpe |   time_in_credit_lock |   time_in_credit_watch |   number_credit_entries |   number_credit_unlocks |   number_relocks |   false_recovery_count |   missed_rebound_count |   avg_weight_SPY |   avg_weight_GOLD |   avg_weight_IEF |   avg_weight_CASH |   avg_weight_CMDTY_FUT |
|:------------------------------------------|:-------------------------------|:---------------|--------------------:|---------------:|---------:|----------------------:|-----------------------:|------------------------:|------------------------:|-----------------:|-----------------------:|-----------------------:|-----------------:|------------------:|-----------------:|------------------:|-----------------------:|
| FINAL_REGIME_HEDGE_TRIGGER_LOCK           | FINAL_BASELINE                 | 2008_GFC       |           0.546231  |     -0.143177  | 1.60278  |                   104 |                      0 |                       1 |                       1 |                0 |                      1 |                      1 |        0.189999  |          0.193317 |        0.416553  |        0.00907029 |              0.19106   |
| FINAL_REGIME_HEDGE_TRIGGER_LOCK           | FINAL_BASELINE                 | 2011_EURO_DEBT |           0.0150521 |     -0.127403  | 0.128675 |                     0 |                      0 |                       0 |                       0 |                0 |                      0 |                      0 |        0.550336  |          0.134899 |        0.314765  |        0          |              0         |
| FINAL_REGIME_HEDGE_TRIGGER_LOCK           | FINAL_BASELINE                 | 2015_2016      |           0.142785  |     -0.0635778 | 1.56311  |                     0 |                      0 |                       0 |                       0 |                0 |                      0 |                      0 |        0.382175  |          0.162338 |        0.378788  |        0          |              0.0766991 |
| FINAL_REGIME_HEDGE_TRIGGER_LOCK           | FINAL_BASELINE                 | 2018Q4         |           0.0766189 |     -0.0502647 | 2.47105  |                    30 |                      0 |                       2 |                       1 |                0 |                      1 |                      1 |        0.0766205 |          0.445988 |        0.278571  |        0.0309524  |              0.167868  |
| FINAL_REGIME_HEDGE_TRIGGER_LOCK           | FINAL_BASELINE                 | COVID_2020     |           0.274799  |     -0.154697  | 2.96473  |                    27 |                      0 |                       2 |                       2 |                0 |                      1 |                      1 |        0.312626  |          0.403846 |        0         |        0          |              0.283528  |
| FINAL_REGIME_HEDGE_TRIGGER_LOCK           | FINAL_BASELINE                 | 2022_RATE_WAR  |           0.0732656 |     -0.140204  | 0.324677 |                    75 |                      0 |                       2 |                       2 |                0 |                      2 |                      1 |        0.402633  |          0.452704 |        0.0654494 |        0.00421348 |              0.0750005 |
| FINAL_REGIME_HEDGE_TRIGGER_LOCK           | FINAL_BASELINE                 | 2025_PULLBACK  |           0.481495  |     -0.0879766 | 2.1169   |                     0 |                      0 |                       0 |                       0 |                0 |                      0 |                      0 |        0         |          0.387813 |        0.128947  |        0.0143275  |              0.468912  |
| FINAL_CHALLENGER_LEVEL_OR_PERCENTILE_LOCK | FINAL_LEVEL_OR_PERCENTILE_LOCK | 2008_GFC       |           0.569732  |     -0.102386  | 1.72407  |                   243 |                      0 |                       1 |                       1 |                0 |                      1 |                      1 |        0.533712  |          0.327907 |        0.72719   |        0.1        |              0.457341  |
| FINAL_CHALLENGER_LEVEL_OR_PERCENTILE_LOCK | FINAL_LEVEL_OR_PERCENTILE_LOCK | 2011_EURO_DEBT |           0.0191206 |     -0.124786  | 0.163687 |                     0 |                      0 |                       0 |                       0 |                0 |                      0 |                      0 |        1         |          0.3      |        0.7       |      nan          |            nan         |
| FINAL_CHALLENGER_LEVEL_OR_PERCENTILE_LOCK | FINAL_LEVEL_OR_PERCENTILE_LOCK | 2015_2016      |           0.147792  |     -0.0635778 | 1.61822  |                     0 |                      0 |                       0 |                       0 |                0 |                      0 |                      0 |        0.832854  |          0.3      |        0.7       |      nan          |              0.412035  |
| FINAL_CHALLENGER_LEVEL_OR_PERCENTILE_LOCK | FINAL_LEVEL_OR_PERCENTILE_LOCK | 2018Q4         |           0.0799903 |     -0.0430223 | 2.99459  |                    67 |                      0 |                       2 |                       2 |                0 |                      1 |                      2 |        0.495086  |          0.946074 |        0.9       |        0.1        |              0.478102  |
| FINAL_CHALLENGER_LEVEL_OR_PERCENTILE_LOCK | FINAL_LEVEL_OR_PERCENTILE_LOCK | COVID_2020     |           0.231476  |     -0.153835  | 2.1197   |                    30 |                      0 |                       3 |                       3 |                0 |                      3 |                      1 |        0.517534  |          1        |      nan         |      nan          |              0.482466  |
| FINAL_CHALLENGER_LEVEL_OR_PERCENTILE_LOCK | FINAL_LEVEL_OR_PERCENTILE_LOCK | 2022_RATE_WAR  |           0.0579945 |     -0.169051  | 0.283471 |                   226 |                      0 |                       1 |                       1 |                0 |                      0 |                      1 |        0.489316  |          0.720434 |        0.803448  |        0.1        |              0.392652  |
| FINAL_CHALLENGER_LEVEL_OR_PERCENTILE_LOCK | FINAL_LEVEL_OR_PERCENTILE_LOCK | 2025_PULLBACK  |           0.493376  |     -0.0879766 | 2.16641  |                    10 |                      0 |                       1 |                       1 |                0 |                      0 |                      0 |      nan         |          0.452669 |        0.9       |        0.1        |              0.547331  |

## 7. Trade-off discussion

- Baseline Sharpe 1.492, MaxDD -15.94%, Final Equity 40.61.
- LEVEL_OR_PERCENTILE_LOCK Sharpe 1.432, MaxDD -16.91%, Final Equity 35.24.
- Top composite candidate: FINAL_BASELINE.

## 8. Decision criteria

- Sharpe >= baseline + 0.03
- MaxDD improvement >= 1 percentage point
- Final Equity >= baseline * 0.98
- false recovery <= baseline
- missed rebound <= baseline + 2

## 9. Recommendation

KEEP BASELINE

## 10. Limitations

- daily credit data availability and revisions
- in-sample crisis tuning risk
- 2008 and 2022 can dominate credit evidence
- requires out-of-sample validation