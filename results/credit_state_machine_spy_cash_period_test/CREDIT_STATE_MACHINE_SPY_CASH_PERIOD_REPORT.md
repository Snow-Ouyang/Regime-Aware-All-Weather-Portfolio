# CREDIT_STATE_MACHINE_SPY_CASH_PERIOD_REPORT

## 1. Purpose

This diagnostic evaluates complete credit stress periods rather than point entries and exits.

## 2. Motivation

The abs_entry_level_z_unlock rule appears to cover large drawdown windows, but the right question is whether the full lock period is a better SPY/CASH timing definition.

## 3. State Machines

- BASELINE: SPY_DD <= -5% and D_CREDIT_15D > 0.10; unlock when D_CREDIT_15D < 0 and SPY > MA20.
- ABS_ENTRY_LEVEL_Z_UNLOCK: same entry, but unlock also requires CREDIT_LEVEL_Z_252D < 1.0.

## 4. Credit-only SPY/CASH Results

| strategy                                      |     CAGR |   Sharpe |   Sortino |     MaxDD |   Calmar |   Final Equity |   annualized_vol |   turnover |   transaction_cost_drag |   time_in_credit_lock |   number_credit_periods |   avg_credit_period_duration |   false_recovery_count |   missed_rebound_count |
|:----------------------------------------------|---------:|---------:|----------:|----------:|---------:|---------------:|-----------------:|-----------:|------------------------:|----------------------:|------------------------:|-----------------------------:|-----------------------:|-----------------------:|
| SPY_BUY_HOLD                                  | 0.111385 | 0.575244 |  0.701605 | -0.551894 | 0.201823 |        8.38451 |         0.193631 |          1 |                  0.0005 |                     0 |                       0 |                     nan      |                      0 |                      0 |
| CREDIT_ONLY_BASELINE_SPY_CASH                 | 0.105963 | 0.593231 |  0.716334 | -0.560116 | 0.189181 |        7.59825 |         0.17862  |         37 |                  0.0185 |                   231 |                       9 |                      25.6667 |                      5 |                      4 |
| CREDIT_ONLY_ABS_ENTRY_LEVEL_Z_UNLOCK_SPY_CASH | 0.135387 | 0.956039 |  1.11736  | -0.261526 | 0.517681 |       12.892   |         0.141613 |         29 |                  0.0145 |                   813 |                       7 |                     116.143  |                      4 |                      6 |

## 5. Combined VIX + CMDTY + CREDIT SPY/CASH Results

| strategy                                    |    CAGR |   Sharpe |   Sortino |     MaxDD |   Calmar |   Final Equity |   annualized_vol |   turnover |   transaction_cost_drag |   time_in_any_lock |   time_in_credit_lock |   time_in_vix_lock |   time_in_cmdty_lock |   number_credit_periods |   false_recovery_count |   missed_rebound_count |
|:--------------------------------------------|--------:|---------:|----------:|----------:|---------:|---------------:|-----------------:|-----------:|------------------------:|-------------------:|----------------------:|-------------------:|---------------------:|------------------------:|-----------------------:|-----------------------:|
| SPY_CASH_BASELINE_ALL_LOCKS                 | 0.11192 | 0.856911 |   1.00374 | -0.368906 | 0.303384 |        8.46618 |         0.130609 |        149 |                  0.0745 |                944 |                   212 |                316 |                  534 |                       9 |                      5 |                      4 |
| SPY_CASH_ABS_ENTRY_LEVEL_Z_UNLOCK_ALL_LOCKS | 0.12278 | 1.07099  |   1.1757  | -0.187834 | 0.653663 |       10.2963  |         0.114642 |        137 |                  0.0685 |               1393 |                   719 |                316 |                  534 |                       7 |                      4 |                      6 |

## 6. Stress-period Quality

| state_machine                          |   number_periods |   total_days_locked |   avg_duration |   median_duration |   avg_SPY_return_during_lock |   avg_SPY_maxDD_during_lock |   avg_CASH_excess_over_SPY |   pct_periods_cash_beats_spy |   false_recovery_count |   missed_rebound_count |   avg_next_21d_SPY_return_after_unlock |   avg_next_21d_SPY_maxDD_after_unlock |   avg_next_63d_SPY_return_after_unlock |   avg_next_63d_SPY_maxDD_after_unlock |
|:---------------------------------------|-----------------:|--------------------:|---------------:|------------------:|-----------------------------:|----------------------------:|---------------------------:|-----------------------------:|-----------------------:|-----------------------:|---------------------------------------:|--------------------------------------:|---------------------------------------:|--------------------------------------:|
| BASELINE_CREDIT_STATE_MACHINE          |                9 |                 231 |        25.6667 |                24 |                    0.0118097 |                  -0.0777444 |                 -0.0100937 |                     0.222222 |                      5 |                      4 |                            -0.00954708 |                            -0.0565126 |                              0.0278299 |                            -0.0879694 |
| ABS_ENTRY_LEVEL_Z_UNLOCK_STATE_MACHINE |                7 |                 813 |       116.143  |                66 |                   -0.0390586 |                  -0.179294  |                  0.0472563 |                     0.428571 |                      4 |                      6 |                             0.00487203 |                            -0.0558184 |                              0.0653373 |                            -0.0826703 |

## 7. Period Overlap Analysis

| category      |   n_days |   SPY_cumulative_return |   CASH_cumulative_return |   CASH_excess_over_SPY |   SPY_maxDD |   avg_CREDIT_LEVEL_Z |   avg_D_CREDIT_15D |
|:--------------|---------:|------------------------:|-------------------------:|-----------------------:|------------:|---------------------:|-------------------:|
| both_lock     |      231 |                0.100619 |                0.0155395 |              -0.085079 |   -0.301799 |             2.69049  |          0.139298  |
| baseline_only |        0 |              nan        |              nan         |             nan        |  nan        |           nan        |        nan         |
| new_only      |      582 |               -0.38318  |                0.042411  |               0.425591 |   -0.479675 |             1.89036  |          0.0344792 |
| neither       |     4261 |               11.3566   |                0.307751  |             -11.0489   |   -0.272695 |            -0.437665 |         -0.0137784 |

## 8. Final Strategy Challenger

| strategy                       |     CAGR |   Sharpe |   Sortino |     MaxDD |   Calmar |   Final Equity |   turnover |   time_in_credit_lock |   false_recovery_count |   missed_rebound_count |
|:-------------------------------|---------:|---------:|----------:|----------:|---------:|---------------:|-----------:|----------------------:|-----------------------:|-----------------------:|
| FINAL_BASELINE                 | 0.201968 |  1.49221 |   2.01336 | -0.159357 |  1.2674  |        40.6104 |   181.757  |                   273 |                      5 |                      4 |
| FINAL_ABS_ENTRY_LEVEL_Z_UNLOCK | 0.187091 |  1.41095 |   1.89978 | -0.169051 |  1.10671 |        31.6033 |    15.2785 |                   719 |                      4 |                      6 |

## 9. Recommendation

ABS_ENTRY_LEVEL_Z_UNLOCK defines better credit stress periods in SPY/CASH. Keep it as a documented defensive extension; final strategy still needs separate evidence.

## 10. Limitations

- daily credit data availability
- crisis sample sparsity
- threshold still in-sample
- out-of-sample validation is needed