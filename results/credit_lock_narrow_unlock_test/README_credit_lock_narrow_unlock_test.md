# Credit Lock Narrow Unlock Test

This diagnostic isolates the ABS_ENTRY_LEVEL_Z_UNLOCK credit lock in a SPY/CASH credit-only framework.
Credit remains enabled only in the same partial-regime scope as the mainline: FLAT_LOW_RATE and FLAT_HIGH_RATE.

## Search scope

- Unlock z-threshold around 1.0: 0.8 / 0.9 / 1.0 / 1.1 / 1.2
- Consecutive unlock confirmation days: 1 / 2 / 3
- SPY trend confirmation: MA20 / MA50 / MA20_AND_MA50

## Best balanced candidate

- Variant: `Z0.9_N1_MA50`
- Sharpe: 1.073
- MaxDD: -18.76%
- Final Equity: 15.11
- False recovery count: 2
- Missed rebound count: 4

## Window summary

- 2008_GFC: return 7.40%, maxDD -8.17%, false recovery 0, missed rebound 1
- COVID_2020: return -1.19%, maxDD -12.44%, false recovery 1, missed rebound 1
- 2022_RATE_WAR: return 3.79%, maxDD -9.73%, false recovery 0, missed rebound 0
- 2025_PULLBACK: return 23.16%, maxDD -18.76%, false recovery 0, missed rebound 1

## Interpretation

This script is intentionally narrow. It does not search entry rules, regime scope, or hedge allocation.
It only tests whether a small unlock-side adjustment can improve the current ABS_ENTRY_LEVEL_Z_UNLOCK credit lock.