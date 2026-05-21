# VIX_CREDIT_STATE_MACHINE_REPORT

This experiment compares VIX + CREDIT state machines without commodity locks.
The credit rule is Z0.9_N1_MA50: entry keeps the existing D_CREDIT_15D > 0.10 rule, while unlock requires D_CREDIT_15D < 0, SPY > MA50, and CREDIT_LEVEL_Z_252D < 0.9.

## Performance

| strategy                   |     CAGR |   Sharpe |   Sortino |     MaxDD |   Calmar |   Final Equity |   annualized_vol |   turnover |   transaction_cost_drag |   time_in_stress |   time_in_vix_lock |   time_in_credit_lock |   number_entries |   number_unlocks | scope       | exit_mode   |
|:---------------------------|---------:|---------:|----------:|----------:|---------:|---------------:|-----------------:|-----------:|------------------------:|-----------------:|-------------------:|----------------------:|-----------------:|-----------------:|:------------|:------------|
| SPY_CASH_TIMING            | 0.120394 |  0.94822 |   1.10062 | -0.294549 | 0.408738 |        9.86445 |         0.126968 |        141 |                  0.0705 |             1029 |                296 |                   273 |               41 |               39 | nan         | nan         |
| VC_AND_EXIT_FINAL_SCOPE    | 0.127223 |  1.06423 |   1.21238 | -0.214653 | 0.592693 |       11.1485  |         0.119545 |        105 |                  0.0525 |             1139 |                316 |                   919 |               31 |               31 | final_scope | independent |
| VC_AND_EXIT_ALL_REGIME     | 0.110083 |  1.06416 |   1.11809 | -0.147863 | 0.744494 |        8.18898 |         0.103446 |        125 |                  0.0625 |             1690 |                365 |                  1501 |               42 |               42 | all_regime  | independent |
| VC_ANCHOR_EXIT_FINAL_SCOPE | 0.130585 |  1.08866 |   1.245   | -0.214653 | 0.608353 |       11.8374  |         0.119951 |        109 |                  0.0545 |             1119 |                502 |                   899 |               31 |               27 | final_scope | anchor      |
| VC_ANCHOR_EXIT_ALL_REGIME  | 0.119458 |  1.1256  |   1.22231 | -0.145991 | 0.818255 |        9.69984 |         0.106127 |        137 |                  0.0685 |             1520 |                675 |                  1331 |               44 |               34 | all_regime  | anchor      |

## Crisis Windows

| strategy                   |   cumulative_return |   max_drawdown |      Sharpe |   time_in_stress |   time_in_vix_lock |   time_in_credit_lock | window         | scope       | exit_mode   |
|:---------------------------|--------------------:|---------------:|------------:|-----------------:|-------------------:|----------------------:|:---------------|:------------|:------------|
| SPY_CASH_TIMING            |         0.0935864   |     -0.192234  |   0.335594  |              251 |                 35 |                   104 | 2008_GFC       | nan         | nan         |
| SPY_CASH_TIMING            |        -0.0860552   |     -0.138757  |  -0.781176  |               67 |                 17 |                     0 | 2011_EURO_DEBT | nan         | nan         |
| SPY_CASH_TIMING            |         0.0686394   |     -0.0400948 |   0.98656   |              125 |                 17 |                     0 | 2015_2016      | nan         | nan         |
| SPY_CASH_TIMING            |        -0.0482215   |     -0.106517  |  -1.35185   |               51 |                 27 |                    30 | 2018Q4         | nan         | nan         |
| SPY_CASH_TIMING            |         0.0942376   |     -0.0698951 |   1.34129   |               42 |                 29 |                    27 | COVID_2020     | nan         | nan         |
| SPY_CASH_TIMING            |        -0.164562    |     -0.294549  |  -0.631955  |               91 |                 12 |                    75 | 2022_RATE_WAR  | nan         | nan         |
| SPY_CASH_TIMING            |         0.240412    |     -0.145991  |   1.41143   |               49 |                 45 |                     0 | 2025_PULLBACK  | nan         | nan         |
| VC_AND_EXIT_FINAL_SCOPE    |         0.0739782   |     -0.0816719 |   0.446787  |              357 |                 35 |                   357 | 2008_GFC       | final_scope | independent |
| VC_AND_EXIT_FINAL_SCOPE    |        -0.0647025   |     -0.192672  |  -0.462516  |               17 |                 17 |                     0 | 2011_EURO_DEBT | final_scope | independent |
| VC_AND_EXIT_FINAL_SCOPE    |         0.0328662   |     -0.128207  |   0.247785  |               17 |                 17 |                     0 | 2015_2016      | final_scope | independent |
| VC_AND_EXIT_FINAL_SCOPE    |         0.00769103  |      0         | 626.63      |               84 |                 28 |                    84 | 2018Q4         | final_scope | independent |
| VC_AND_EXIT_FINAL_SCOPE    |         0.0185839   |     -0.0698951 |   0.312541  |               67 |                 30 |                    60 | COVID_2020     | final_scope | independent |
| VC_AND_EXIT_FINAL_SCOPE    |        -0.0254964   |     -0.126743  |  -0.186577  |              229 |                 14 |                   215 | 2022_RATE_WAR  | final_scope | independent |
| VC_AND_EXIT_FINAL_SCOPE    |         0.167076    |     -0.147863  |   1.04087   |               68 |                 49 |                    29 | 2025_PULLBACK  | final_scope | independent |
| VC_AND_EXIT_ALL_REGIME     |         0.0739782   |     -0.0816719 |   0.446787  |              357 |                 35 |                   357 | 2008_GFC       | all_regime  | independent |
| VC_AND_EXIT_ALL_REGIME     |        -0.00392801  |     -0.0367784 |  -0.0840604 |              111 |                 17 |                   111 | 2011_EURO_DEBT | all_regime  | independent |
| VC_AND_EXIT_ALL_REGIME     |         0.0225956   |     -0.0415058 |   0.351611  |              137 |                 17 |                   136 | 2015_2016      | all_regime  | independent |
| VC_AND_EXIT_ALL_REGIME     |         0.00769103  |      0         | 626.63      |               84 |                 28 |                    84 | 2018Q4         | all_regime  | independent |
| VC_AND_EXIT_ALL_REGIME     |         0.0185839   |     -0.0698951 |   0.312541  |               67 |                 30 |                    60 | COVID_2020     | all_regime  | independent |
| VC_AND_EXIT_ALL_REGIME     |        -0.0564609   |     -0.126743  |  -0.422395  |              233 |                 14 |                   219 | 2022_RATE_WAR  | all_regime  | independent |
| VC_AND_EXIT_ALL_REGIME     |         0.167076    |     -0.147863  |   1.04087   |               68 |                 49 |                    29 | 2025_PULLBACK  | all_regime  | independent |
| VC_ANCHOR_EXIT_FINAL_SCOPE |         0.0739782   |     -0.0816719 |   0.446787  |              357 |                149 |                   357 | 2008_GFC       | final_scope | anchor      |
| VC_ANCHOR_EXIT_FINAL_SCOPE |        -0.0647025   |     -0.192672  |  -0.462516  |               17 |                 17 |                     0 | 2011_EURO_DEBT | final_scope | anchor      |
| VC_ANCHOR_EXIT_FINAL_SCOPE |         0.0328662   |     -0.128207  |   0.247785  |               17 |                 17 |                     0 | 2015_2016      | final_scope | anchor      |
| VC_ANCHOR_EXIT_FINAL_SCOPE |         0.00769103  |      0         | 626.63      |               84 |                 76 |                    84 | 2018Q4         | final_scope | anchor      |
| VC_ANCHOR_EXIT_FINAL_SCOPE |         0.0175793   |     -0.0698951 |   0.295412  |               66 |                 30 |                    59 | COVID_2020     | final_scope | anchor      |
| VC_ANCHOR_EXIT_FINAL_SCOPE |        -0.0254964   |     -0.126743  |  -0.186577  |              229 |                 14 |                   215 | 2022_RATE_WAR  | final_scope | anchor      |
| VC_ANCHOR_EXIT_FINAL_SCOPE |         0.240412    |     -0.145991  |   1.41143   |               49 |                 49 |                    10 | 2025_PULLBACK  | final_scope | anchor      |
| VC_ANCHOR_EXIT_ALL_REGIME  |         0.0739782   |     -0.0816719 |   0.446787  |              357 |                149 |                   357 | 2008_GFC       | all_regime  | anchor      |
| VC_ANCHOR_EXIT_ALL_REGIME  |        -0.00392801  |     -0.0367784 |  -0.0840604 |              111 |                103 |                   111 | 2011_EURO_DEBT | all_regime  | anchor      |
| VC_ANCHOR_EXIT_ALL_REGIME  |         0.000741038 |     -0.0619591 |   0.0108931 |              131 |                 17 |                   130 | 2015_2016      | all_regime  | anchor      |
| VC_ANCHOR_EXIT_ALL_REGIME  |         0.00769103  |      0         | 626.63      |               84 |                 76 |                    84 | 2018Q4         | all_regime  | anchor      |
| VC_ANCHOR_EXIT_ALL_REGIME  |         0.0175793   |     -0.0698951 |   0.295412  |               66 |                 30 |                    59 | COVID_2020     | all_regime  | anchor      |
| VC_ANCHOR_EXIT_ALL_REGIME  |        -0.0564609   |     -0.126743  |  -0.422395  |              233 |                 14 |                   219 | 2022_RATE_WAR  | all_regime  | anchor      |
| VC_ANCHOR_EXIT_ALL_REGIME  |         0.240412    |     -0.145991  |   1.41143   |               49 |                 49 |                    10 | 2025_PULLBACK  | all_regime  | anchor      |

## Recommendation

VC_ANCHOR_EXIT_ALL_REGIME is the strongest VIX+CREDIT challenger in this limited test.