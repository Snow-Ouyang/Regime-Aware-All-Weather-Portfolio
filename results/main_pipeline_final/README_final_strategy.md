# Final Source-Only Strategy Outputs

This folder is generated from `data/raw` and `data/processed` only using the canonical source-only settings.

Final display strategies:

- `SPY_BUY_HOLD`: always 100% SPY.
- `SPY_CASH_TIMING`: SPY in non-risk, CASH in trigger-lock stress; uses the same VIX/CREDIT anchor state machine as the final hedge strategy.
- `FINAL_REGIME_HEDGE_TRIGGER_LOCK`: final hedge allocation with six-regime classification, buffered regime transitions, and regime-specific stress sleeves.

Key design choices:

- Credit spread is daily `DBAA - DAAA`, filled to the trading calendar before feature construction.
- Macro regime has no `NEUTRAL`: `INVERTED`, `FLAT`, `STEEP`, with `3-day confirm`.
- `FLAT` uses buffered `GS10` low/mid/high bands:
  - `MID -> LOW = 1.1`
  - `LOW -> MID = 1.3`
  - `HIGH -> MID = 3.4`
  - `MID -> HIGH = 3.6`
- `STEEP` uses buffered `GS10` low/mid/high bands:
  - `MID -> LOW = 2.0`
  - `LOW -> MID = 2.3`
  - `HIGH -> MID = 3.0`
  - `MID -> HIGH = 3.2`
- `STEEP_LOW_RATE` does not allow native credit entries.
- If a `FULL_RISK` episode carries into a new regime, the strategy stays on a stress sleeve until unlock. Carry-over may adopt the new regime's stress sleeve, but it cannot revert to normal during `FULL_RISK`.
- `CASH_return` uses geometric daily DTB3.
- Inverse-vol window is 90 trading days.
- Transaction cost uses 10 bps one-way.

Final allocation settings:

- `FLAT_LOW_RATE_NORMAL`: SPY / CMDTY_FUT inverse-vol.
- `FLAT_MID_RATE_NORMAL`: SPY / GOLD inverse-vol.
- `FLAT_LOWMID_RATE_STRESS`: 100% CASH.
- `FLAT_HIGH_RATE_NORMAL`: GOLD / CMDTY_FUT inverse-vol.
- `FLAT_HIGH_RATE_STRESS`: 70% IEF + 30% (GOLD / CMDTY_FUT inverse-vol).
- `STEEP_LOW_RATE_NORMAL`: SPY / CMDTY_FUT inverse-vol.
- `STEEP_LOW_RATE_STRESS`: 100% SPY.
- `STEEP_MID_RATE_NORMAL`: 100% SPY.
- `STEEP_MID_RATE_STRESS`: 100% IEF.
- `STEEP_HIGH_RATE_NORMAL`: SPY / GOLD / CMDTY_FUT inverse-vol.
- `STEEP_HIGH_RATE_STRESS`: 100% IEF.
- `INVERTED_NORMAL`: SPY / GOLD inverse-vol.
- `INVERTED_STRESS`: 10% CASH + 90% (SPY / GOLD inverse-vol).

Main run order:

1. `python scripts/run_final_strategy_source_only.py`
2. `python scripts/08_stress_trigger_diagnostics.py`
3. `python scripts/10_final_report_outputs.py`
4. `python scripts/hard_validate_main_pipeline_source_only.py`


## GS10 Internal Structure and Regime Buffers

The final regime builder now diagnoses `FLAT` and `STEEP` separately with full-sample `GS10` KDE + HMM outputs. This is not a single global rate split. It is two separate internal-structure diagnostics that support low/mid/high classification within each family.

### FLAT GS10 structure

- LOW: mean GS10 `1.02`, sample weight `19.3%`
- MID: mean GS10 `2.59`, sample weight `43.5%`
- HIGH: mean GS10 `4.40`, sample weight `37.2%`

- Hysteresis bands:
  - `MID -> LOW = 1.1`
  - `LOW -> MID = 1.3`
  - `HIGH -> MID = 3.4`
  - `MID -> HIGH = 3.6`

### STEEP GS10 structure

- LOW: mean GS10 `1.79`, sample weight `36.4%`
- MID: mean GS10 `2.51`, sample weight `39.9%`
- HIGH: mean GS10 `3.59`, sample weight `23.7%`

- Hysteresis bands:
  - `MID -> LOW = 2.0`
  - `LOW -> MID = 2.3`
  - `HIGH -> MID = 3.0`
  - `MID -> HIGH = 3.2`

- All regime transitions still require `3-day confirm`.

This does two things:

1. It reflects the internal structure visible in `GS10` inside `FLAT` and `STEEP`, rather than forcing both into one coarse threshold rule.
2. It reduces turnover by using hysteresis bands instead of single-point internal splits.

The corresponding mainline figures are:

- `results/main_pipeline_final/figures/flat_gs10_kde_hmm.png`
- `results/main_pipeline_final/figures/steep_gs10_kde_hmm.png`
