# Dependency Map

This map identifies files needed by the final README, final report, and final strategy implementation.

## 1. Final Report Dependencies

### Markdown image/link dependencies

- `README.md` (exists)
- `config/project_config.yaml` (exists)
- `data/README.md` (exists)
- `figures/09_final_strategy/mature_regime_hedge_final/case_2008_GFC_final.png` (exists)
- `figures/09_final_strategy/mature_regime_hedge_final/case_2015_2016_final.png` (exists)
- `figures/09_final_strategy/mature_regime_hedge_final/case_2022_rate_war_final.png` (exists)
- `figures/09_final_strategy/mature_regime_hedge_final/case_2025_pullback_final.png` (exists)
- `figures/09_final_strategy/mature_regime_hedge_final/final_component_attribution.png` (exists)
- `figures/09_final_strategy/mature_regime_hedge_final/final_drawdown_comparison.png` (exists)
- `figures/09_final_strategy/mature_regime_hedge_final/final_equity_curve_log.png` (exists)
- `figures/09_final_strategy/mature_regime_hedge_final/final_performance_bar_charts.png` (exists)
- `figures/09_final_strategy/mature_regime_hedge_final/final_weight_stack.png` (exists)
- `reports/FINAL_REPORT.md` (exists)
- `reports/OUTPUT_INDEX.md` (exists)
- `requirements.txt` (exists)
- `results/09_final_strategy/mature_regime_hedge_final/crisis_performance.csv` (exists)
- `results/09_final_strategy/mature_regime_hedge_final/daily_backtest_panel.csv` (exists)
- `results/09_final_strategy/mature_regime_hedge_final/final_strategy_component_attribution.csv` (exists)
- `results/09_final_strategy/mature_regime_hedge_final/final_strategy_decision_summary.csv` (exists)
- `results/09_final_strategy/mature_regime_hedge_final/performance_summary.csv` (exists)
- `src/allocation/final_strategy_backtest.py` (exists)
- `src/strategies/mature_regime_hedge_final.py` (exists)

### Final result tables

- `results/09_final_strategy/mature_regime_hedge_final/performance_summary.csv`
- `results/09_final_strategy/mature_regime_hedge_final/crisis_performance.csv`
- `results/09_final_strategy/mature_regime_hedge_final/final_strategy_component_attribution.csv`
- `results/09_final_strategy/mature_regime_hedge_final/final_strategy_decision_summary.csv`

### Final strategy scripts

- `src/allocation/final_strategy_backtest.py`
- `src/strategies/mature_regime_hedge_final.py`

## 2. Final Strategy Dependencies

The final strategy script uses validated project panels and reconstructs missing fields only when required.

### CSV paths detected in final strategy script

- No direct literal `read_csv` path detected.

### Required return and signal fields

- Returns: `SPY_return`, `GOLD_return`, `CMDTY_FUT_return`, `IEF_return`, `CASH_return` or equivalents.
- Stress signals: `VIX_ZSCORE_120D`, `D_CREDIT_SPREAD_20D`, `spy_drawdown_from_previous_high`, `monthly_either_state`, `SPY_CROSS_ABOVE_MA20`.
- Commodity overlay: `CMDTY_RET60` or fields needed to reconstruct it.
- Regime: `macro_regime_confirmed` with only `FLAT`, `STEEP`, `INVERTED` for the final strategy.

## 3. Upstream Dependencies

### 01_regime_discovery_clustering

Scripts:
- `src/regime/jump_model_clustering.py`
- `src/regime/fit_simplified_regime_model.py`

Outputs:
- `results/01_regime_discovery`
- `results/regime`

### 02_rule_based_regime_construction

Scripts:
- `src/regime/build_rule_based_regime.py`
- `src/regime/plot_regime_diagnostics.py`
- `src/regime/build_regime_dataset.py`
- `src/regime/plot_regime_outputs.py`

Outputs:
- `results/02_rule_based_regime`
- `results/rule_diagnostics`

### 03_monthly_timing_backbone

Scripts:
- `src/timing/reproduce_monthly_timing.py`
- `src/strategies/absolute_momentum_spy_cash.py`
- `src/strategies/faber_spy_cash_timing.py`
- `src/analysis/monthly_either_crash_brake_diagnostics.py`

Outputs:
- `results/03_monthly_timing`
- `results/absolute_momentum_spy_cash`
- `results/faber_spy_cash_timing`
- `results/monthly_either_crash_brake_diagnostics`

### 04_high_frequency_stress_triggers

Scripts:
- `src/timing/build_backbone_v2.py`
- `src/strategies/spy_cash_backbone_upgrade_ablation.py`
- `src/analysis/flat_vix_credit_trigger_diagnostic.py`
- `src/analysis/credit_trigger_by_regime_diagnostic.py`

Outputs:
- `results/04_stress_triggers`
- `results/spy_cash_backbone_upgrade_ablation/daily_backtest_panel.csv`
- `results/flat_vix_credit_trigger_diagnostic`
- `results/credit_trigger_by_regime_diagnostic`

### 05_recovery_rule

Scripts:
- `src/timing/recovery_rule_diagnostic.py`
- `src/analysis/stress_recovery_grid_search.py`

Outputs:
- `results/05_recovery`
- `results/stress_recovery_grid_search`

### 06_2015_2016_slow_growth_repair

Scripts:
- `src/diagnostics/drawdown_2015_2016_forensic.py`
- `src/diagnostics/commodity_trigger_by_regime.py`
- `src/strategies/mature_strategy_steep_cmdty_overlay_50spy50ief.py`

Outputs:
- `results/06_2015_2016_repair`
- `results/drawdown_2015_2016_forensic_diagnostic`
- `results/commodity_crash_transmission_by_regime`
- `results/mature_steep_cmdty_overlay_50spy50ief`

### 07_cross_state_asset_behavior

Scripts:
- `src/diagnostics/cross_state_asset_behavior.py`
- `src/diagnostics/bond_sleeve_diagnostic.py`
- `src/analysis/hedge_asset_cross_state_diagnostic_extended.py`
- `src/analysis/evaluate_bond_sleeve_candidates.py`

Outputs:
- `results/07_cross_state_asset_behavior`
- `results/hedge_asset_cross_state_diagnostic_extended`

### 08_allocation

Scripts:
- `src/allocation/inverse_vol_allocation.py`
- `src/allocation/risk_parity_comparison.py`
- `src/analysis/invvol_window_grid_search.py`
- `src/strategies/regime_aware_risk_parity_allocation.py`

Outputs:
- `results/08_allocation`
- `results/regime_aware_risk_parity_allocation`
- `results/invvol_window_grid_search`

### 09_final_strategy

Scripts:
- `src/allocation/final_strategy_backtest.py`
- `src/strategies/mature_regime_hedge_final.py`

Outputs:
- `results/09_final_strategy/mature_regime_hedge_final/performance_summary.csv`
- `results/09_final_strategy/mature_regime_hedge_final/crisis_performance.csv`
- `results/09_final_strategy/mature_regime_hedge_final/daily_backtest_panel.csv`
- `figures/09_final_strategy/mature_regime_hedge_final/final_equity_curve_log.png`
- `figures/09_final_strategy/mature_regime_hedge_final/final_drawdown_comparison.png`
- `figures/09_final_strategy/mature_regime_hedge_final/final_weight_stack.png`

## 4. Exploratory Outputs

Exploratory outputs are candidates for archive only if they are not referenced by README, FINAL_REPORT, or final strategy outputs.