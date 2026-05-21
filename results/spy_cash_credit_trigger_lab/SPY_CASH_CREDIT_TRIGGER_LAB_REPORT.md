# SPY_CASH_CREDIT_TRIGGER_LAB_REPORT

## 1. Purpose

This lab isolates credit trigger timing inside a pure SPY/CASH framework so that hedge asset allocation does not contaminate the timing conclusion.

## 2. Framework

Normal = 100% SPY. Any stress lock = 100% CASH. VIX and commodity locks are kept from the current final strategy. Only credit entry/unlock/relock rules change.

## 3. Baselines

- `SPY_BUY_HOLD`
- `SPY_CASH_NO_CREDIT`
- `SPY_CASH_FINAL_LOCKS`
- `SPY_CASH_CREDIT_ONLY`

## 4. Credit Variants

Variants include absolute 15D/20D spread changes, z-score entry, stricter unlock confirmation, MA50-based unlocks, cooldown unlocks, and fast relock rules.

## 5. Main Results

- Best variant by Sharpe: `ABS_ENTRY_LEVEL_Z_UNLOCK`
- Best variant by MaxDD: `HYBRID_BEST_EFFORT`
- Best balanced variant: `ABS_OR_DZ_15D`

## 6. Does Credit Add Value in SPY/CASH?

- `SPY_CASH_FINAL_LOCKS` Sharpe: 0.948
- `SPY_CASH_NO_CREDIT` Sharpe: 0.867
- `SPY_CASH_CREDIT_ONLY` Sharpe: 0.593
- Credit adds value vs no-credit: YES

## 7. Unlock / Relock Diagnostics

Use the episode diagnostics to compare false recovery, missed rebound, and relock counts. This is the key trade-off surface.

## 8. Crisis Windows

Use the crisis comparison table and case-study plots for 2008, 2022, COVID, and 2025.

## 9. Recommendation

No challenger is strong enough yet. Credit timing should not be over-optimized further inside the final strategy until it proves itself in this simpler SPY/CASH lab.

## 10. Final Conclusion

If a credit variant cannot improve robustly even in a SPY/CASH lab, it should not be promoted directly into the final regime-aware hedge strategy.