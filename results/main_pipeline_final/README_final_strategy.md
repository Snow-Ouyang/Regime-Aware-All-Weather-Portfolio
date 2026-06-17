# Final Source-Only Strategy Outputs

This folder is generated from `data/raw` and `data/processed` only using the canonical source-only settings.

Sample window:

- Start: `2007-01-08`
- End: `2026-06-12`
- Trading days: `4886`

Final strategy headline metrics:

- CAGR: `22.09%`
- Annualized volatility: `11.13%`
- Sharpe: `1.984`
- Max drawdown: `-9.73%`
- Final equity multiple: `47.91x`

Final display strategies:

- `SPY_BUY_HOLD`: always 100% SPY.
- `SPY_CASH_TIMING`: SPY in non-risk, CASH in trigger-lock stress; uses the same VIX/CREDIT anchor state machine as the final hedge strategy.
- `FINAL_REGIME_HEDGE_TRIGGER_LOCK`: final hedge allocation with buffered term-spread transitions, oil-aware flat sleeves, expanded commodity ETFs, and the original mainline VIX/CREDIT stress trigger.

Key design choices:

- Credit spread is daily `DBAA - DAAA`, filled to the trading calendar before feature construction.
- Macro regime has no `NEUTRAL`: `INVERTED`, `FLAT`, `STEEP`, with outer term-spread hysteresis:
  - `FLAT -> INVERTED = -0.10`
  - `INVERTED -> FLAT = 0.10`
  - `FLAT -> STEEP = 1.20`
  - `STEEP -> FLAT = 1.00`
  - outer and inner regime transitions both use `2-day confirm`
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
- Carry-over stress is shown explicitly in the cross-state heatmap. `STEEP_LOW_RATE_STRESS` has no native trigger, but if an active stress period carries into `STEEP_LOW_RATE`, it remains a stress sleeve and is analyzed separately.
- `CASH_return` uses geometric daily DTB3.
- Oil level uses a `252-day` oil MA with three persistent states:
  - `HIGH`: entry at `+20%`, exit at `+5%`
  - `LOW`: entry at `-20%`, exit at `-10%`
  - confirmation is `10 trading days`
- When oil is `HIGH`, flat sleeves remove `GOLD` and `DBB` from the candidate set.
- Expanded asset set: `SPY`, `GOLD`, `IEF`, `DBC`, `DBB`, `DBA`, `CASH`.
- Inverse-vol window is 90 trading days.
- Transaction cost uses 10 bps one-way.

Final allocation settings:

- `FLAT_LOW_RATE_NORMAL`: SPY / DBC / DBB inverse-vol, but oil `HIGH` removes `DBB`.
- `FLAT_MID_RATE_NORMAL`: SPY / GOLD inverse-vol.
- `FLAT_LOWMID_RATE_STRESS`: 100% CASH.
- `FLAT_HIGH_RATE_NORMAL`: 40% IEF + 60% (GOLD / DBC inverse-vol), but oil `HIGH` removes `GOLD`.
- `FLAT_HIGH_RATE_STRESS`: 10% DBA + 90% GOLD, but oil `HIGH` collapses this sleeve to 100% DBA.
- `STEEP_LOW_RATE_NORMAL`: 100% SPY.
- `STEEP_LOW_RATE_STRESS`: 100% SPY.
- `STEEP_MID_RATE_NORMAL`: 100% SPY.
- `STEEP_MID_RATE_STRESS`: 100% IEF.
- `STEEP_HIGH_RATE_NORMAL`: SPY / GOLD inverse-vol.
- `STEEP_HIGH_RATE_STRESS`: 100% IEF.
- `INVERTED_NORMAL`: SPY / GOLD inverse-vol.
- `INVERTED_STRESS`: 90% CASH + 10% SPY.

Pure regime x stress outputs are also written into the mainline output set:

- `results/main_pipeline_final/figures/pure_regime_stress_asset_behavior_heatmap.png`
- `results/main_pipeline_final/figures/pure_regime_stress_asset_sharpe_heatmap.png`
- `results/main_pipeline_final/tables/pure_cross_state_asset_behavior.csv`
- `results/main_pipeline_final/figures/oil_stress_asset_behavior_heatmap.png`
- `results/main_pipeline_final/figures/oil_nonrisk_rate_asset_behavior_heatmap.png`
- `results/main_pipeline_final/figures/oil_stress_coverage_heatmap.png`
- `results/main_pipeline_final/figures/oil_nonrisk_rate_coverage_heatmap.png`

Main run order:

1. `python scripts/run_final_strategy_source_only.py`
2. `python scripts/08_stress_trigger_diagnostics.py`
3. `python scripts/10_final_report_outputs.py`
4. `python scripts/hard_validate_main_pipeline_source_only.py`

Live dashboard:

- `python scripts/30_generate_live_regime_dashboard.py`
- Output HTML: `results/live_regime_dashboard/live_regime_dashboard.html`
- The dashboard uses the same canonical regime, stress, oil-level, and allocation logic as the main strategy and fetches the latest macro plus asset data with cache fallback.


## GS10 Internal Structure and Regime Buffers

The final regime builder now diagnoses `FLAT` and `STEEP` separately with full-sample `GS10` KDE + HMM outputs. This is not a single global rate split. It is two separate internal-structure diagnostics that support low/mid/high classification within each family.

### FLAT GS10 structure

- LOW: mean GS10 `1.18`, sample weight `27.5%`
- MID: mean GS10 `2.59`, sample weight `39.0%`
- HIGH: mean GS10 `4.36`, sample weight `33.5%`

- Hysteresis bands:
  - `MID -> LOW = 1.1`
  - `LOW -> MID = 1.3`
  - `HIGH -> MID = 3.4`
  - `MID -> HIGH = 3.6`

### STEEP GS10 structure

- LOW: mean GS10 `1.81`, sample weight `36.5%`
- MID: mean GS10 `2.54`, sample weight `38.7%`
- HIGH: mean GS10 `3.59`, sample weight `24.8%`

- Hysteresis bands:
  - `MID -> LOW = 2.0`
  - `LOW -> MID = 2.3`
  - `HIGH -> MID = 3.0`
  - `MID -> HIGH = 3.2`

- All regime transitions use `2-day confirm`.

This does two things:

1. It reflects the internal structure visible in `GS10` inside `FLAT` and `STEEP`, rather than forcing both into one coarse threshold rule.
2. It reduces turnover by using hysteresis bands instead of single-point internal splits.

The corresponding mainline figures are:

- `results/main_pipeline_final/figures/flat_gs10_kde_hmm.png`
- `results/main_pipeline_final/figures/steep_gs10_kde_hmm.png`


## Term Spread Structure and Outer Hysteresis

We also re-ran single-variable HMM + KDE diagnostics directly on `DGS10 - DGS1` to validate the outer `INVERTED / FLAT / STEEP` transition logic.

### Full-sample 3-state term spread HMM

- INVERTED: mean term spread `-0.68`, sample weight `13.5%`
- FLAT_ZONE: mean term spread `0.63`, sample weight `39.5%`
- STEEP_ZONE: mean term spread `2.13`, sample weight `47.0%`

### Non-inverted 2-state term spread HMM

- LOW_POSITIVE: mean term spread `0.52`, sample weight `32.8%`
- HIGH_POSITIVE: mean term spread `1.97`, sample weight `67.2%`

Interpretation:

1. The full-sample HMM confirms that negative term spread is structurally distinct from the positive-rate states.
2. Inside the non-inverted sample, KDE/HMM still shows a low-positive zone and a high-positive zone rather than one single smooth cloud.
3. That internal structure is why the outer rule now uses hysteresis instead of a single cutoff:
   - `FLAT -> INVERTED = -0.10`
   - `INVERTED -> FLAT = 0.10`
   - `FLAT -> STEEP = 1.20`
   - `STEEP -> FLAT = 1.00`

The corresponding mainline figures are:

- `results/main_pipeline_final/figures/term_spread_full_sample_kde_hmm.png`
- `results/main_pipeline_final/figures/term_spread_non_inverted_kde_hmm.png`
