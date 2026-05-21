# Final Source-Only Strategy Outputs

This folder was generated from `data/raw` and `data/processed` only using the
canonical source-only settings.

Final display strategies:

- `SPY_BUY_HOLD`: always 100% SPY.
- `SPY_CASH_TIMING`: SPY in non-risk, CASH in full-risk; uses the same VIX/CREDIT anchor stress state as the final hedge strategy.
- `FINAL_REGIME_HEDGE_TRIGGER_LOCK`: final hedge allocation with inverse-vol normal allocations and regime-specific trigger-lock stress hedges.

Key design choices:
- Credit spread is daily `DBAA - DAAA`, filled to the trading calendar before feature construction.
- Macro regime has no `NEUTRAL`: term spread maps every day to `INVERTED`,
  `FLAT`, or `STEEP`, then uses 3-day confirmation.
- FLAT is refined with GS10 threshold 3.0 into `FLAT_LOW_RATE` and
  `FLAT_HIGH_RATE`.
- STEEP normal is refined with GS1 threshold 0.3 into `STEEP_LOW_RATE`
  and `STEEP_HIGH_RATE`; the low/high switch also uses 3-day confirmation.
- `CASH_return` uses geometric daily DTB3.
- `CMDTY_RET60` uses synthetic commodity price from `CMDTY_FUT_return`.
- `VIX_ZSCORE_120D` uses 120 trading days, current-day inclusive, `ddof=1`.
- Inverse-vol window grid search showed limited sensitivity across reasonable settings; the final mainline uses 90 trading days.
- Transaction cost uses 10 bps one-way.
- Recovery overlay exploration is not part of the final mainline.

Final allocation settings:
- `FLAT_LOW_RATE_NORMAL`: SPY / CMDTY_FUT inverse-vol.
- `FLAT_LOW_RATE_STRESS`: 100% CASH.
- `FLAT_HIGH_RATE_NORMAL`: GOLD / CMDTY_FUT inverse-vol.
- `FLAT_HIGH_RATE_STRESS`: 100% IEF.
- `STEEP_LOW_RATE_NORMAL`: 100% SPY.
- `STEEP_LOW_RATE_STRESS`: 60% SPY / 40% IEF.
- `STEEP_HIGH_RATE_NORMAL`: SPY / GOLD / CMDTY_FUT inverse-vol.
- `STEEP_HIGH_RATE_STRESS`: 10% CASH / 90% IEF.
- `INVERTED_NORMAL`: SPY / GOLD inverse-vol.
- `INVERTED_STRESS`: 10% CASH + 90% (SPY / GOLD inverse-vol).

How to run the current main sequence:

1. `python scripts/01_data_prepare.py`
2. `python scripts/02_rule_based_regime.py`
3. `python scripts/03_stress_detection.py`
4. `python scripts/04_asset_return_panel.py`
5. `python scripts/05_baseline_strategy.py`
6. `python scripts/06_flat_rate_refined_strategy.py`
7. `python scripts/07_cross_state_asset_behavior.py`
8. `python scripts/08_stress_trigger_diagnostics.py`
9. `python scripts/run_final_strategy_source_only.py`
10. `python scripts/10_final_report_outputs.py`


## Stress Trigger and Turnover Diagnostics

The canonical final strategy uses a trigger-lock state machine. This replaced the prior FLAT_LOW recovery overlay because it is lower-turnover, more tradable, and easier to explain.

### Trigger Rules Summary

- `FLAT_LOW_RATE` / `FLAT_HIGH_RATE` / `INVERTED`: VIX trigger is active.
- `FLAT_LOW_RATE` / `FLAT_HIGH_RATE` / `STEEP_LOW_RATE` / `STEEP_HIGH_RATE` / `INVERTED`: credit trigger is active.
- Commodity trigger is not part of the final mainline.
- Monthly SELL is not part of the final state machine.
- Credit entry uses `D_CREDIT_SPREAD_15D > 0.10`, `SPY drawdown <= -5%`, and `SPY <= MA20`.
- Credit unlock uses `D_CREDIT_SPREAD_15D < 0`, `SPY > MA50`, and `CREDIT_LEVEL_Z_252D < 0.9`.
- VIX unlock uses `VIX_ZSCORE_120D < 1.5` with `SPY > MA20`.
- The anchor-exit rule means VIX-led stress exits on VIX unlock, credit-led stress exits on credit unlock, and BOTH entries unlock independently.

### Key Findings

- The final comparison now uses the same trigger-lock stress state for both `SPY_CASH_TIMING` and `FINAL_REGIME_HEDGE_TRIGGER_LOCK`.
- Cross-state asset behavior is also grouped with the trigger-lock stress state, so the asset evidence and final strategy share the same stress definition.
- Recovery overlay diagnostics are kept as exploratory history, but they are not part of the final mainline.

### Implication

Future research can still study state-machine refinements, but the current mainline is intentionally converged around the VIX/CREDIT anchor state machine without recovery overlay or commodity-trigger complexity.
