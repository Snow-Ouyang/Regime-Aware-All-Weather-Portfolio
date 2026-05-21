# SPY_CASH_TIMING_ATTRIBUTION_REPORT

## 1. Purpose

This is an attribution test, not a rollback proposal. The goal is to separate timing value from hedge-allocation value.

## 2. Key Question

We compare whether the final strategy's edge comes mainly from trigger-lock timing, normal regime allocation, stress hedge allocation, or a combination of all three.

## 3. Strategy Variants

- `SPY_BUY_HOLD`: 100% SPY.
- `SPY_CASH_TRIGGER_LOCK`: same trigger-lock timing as final, but stress = 100% CASH.
- `REGIME_NORMAL_CASH_STRESS`: final normal allocation, stress = CASH.
- `CURRENT_FINAL`: current mainline final strategy.
- `FINAL_WITH_CREDIT_STRESS_CASH`: cash only during credit-triggered stress.

## 4. Main Performance Comparison

- SPY buy-and-hold Sharpe: 0.575
- SPY/CASH trigger-lock Sharpe: 0.948
- Regime-normal + cash-stress Sharpe: 1.380
- Current final Sharpe: 1.492

## 5. Layer Attribution

See `strategy_layer_attribution.csv` for the exact deltas.

## 6. Credit Trigger Focus

- Credit episodes where CASH beat final hedge: 1/8
- Credit episodes where final hedge beat CASH: 7/8

## 7. Crisis Window Analysis

Use the crisis comparison table and case-study CSVs to compare 2008, 2015-2016, COVID, 2022, and 2025.

## 8. Should We Return to SPY/CASH?

The answer depends on whether timing-only SPY/CASH keeps enough of the return while materially simplifying the hedge layer.

## 9. Recommendation

If SPY/CASH remains clearly weaker than the final strategy in compounding and risk-adjusted return, keep it as a benchmark rather than a replacement. If the credit-stress cash variant is locally better, treat that as future refinement work, not immediate rollback.