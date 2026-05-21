# CREDIT_LOCK_EXIT_DIAGNOSTIC_REPORT

## 1. Purpose

This diagnostic studies only the credit trigger lock / unlock logic. Allocation, regime framework, VIX lock, and commodity lock are left unchanged.

## 2. Current problem

- Baseline credit episodes: 8
- Baseline false recovery count: 5
- Baseline missed rebound count: 4
- The baseline unlock uses the current final-rule credit logic from the mainline.

## 3. Baseline credit lock episode diagnostics

- 2022-2023 is the main window where early unlock remains a concern.
- 2008 contains both dead-cat bounce risk and delayed recovery trade-offs.

## 4. Candidate unlock rules

- Candidates test stricter confirmation, MA50 trend confirmation, spread-level normalization, z-score normalization, cooldown, drawdown repair, and fast relock.

## 5. Full-sample performance

- Baseline Sharpe: 1.492, MaxDD: -15.94%, Final Equity: 40.61
- Best balanced candidate `CREDIT_UNLOCK_3D_CONFIRM`: Sharpe 1.489, MaxDD -17.09%, Final Equity 39.33

## 6. Crisis window analysis

- Compare `2008_GFC`, `2022_RATE_WAR`, `COVID_2020`, and `2025_PULLBACK` in the crisis comparison table.

## 7. Trade-off discussion

- Stricter unlocks can reduce false recovery, but they can also increase missed rebound count and keep the strategy in hedge mode too long.
- Fast relock variants test whether re-lock is more effective than simply delaying unlock.

## 8. Recommendation

- Best false-recovery candidate: `CREDIT_UNLOCK_3D_CONFIRM`
- Best 2008 candidate: `CREDIT_UNLOCK_LEVEL_CONFIRM`
- Best 2022 candidate: `FINAL_BASELINE`
- Best balanced candidate: `CREDIT_UNLOCK_3D_CONFIRM`
- Final strategy should change: `NO`

## 9. Proposed final credit rule if any

Only adopt a replacement if the best balanced candidate improves false recovery and drawdown without clearly damaging rebound capture.

## 10. Limitations

- Credit stress samples are sparse.
- 2008 and 2022 are not the same failure mode.
- Unlock and relock thresholds still need OOS validation.