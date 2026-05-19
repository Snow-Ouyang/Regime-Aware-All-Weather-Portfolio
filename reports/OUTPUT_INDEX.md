# Output Index

This index lists the core files used by the cleaned README and final report.

| File path | Type | README | FINAL_REPORT | Description |
|---|---|---:|---:|---|
| `README.md` | report | yes | no | GitHub-facing project overview and reproduction guide. |
| `reports/FINAL_REPORT.md` | report | no | yes | Full research narrative and final strategy analysis. |
| `config/project_config.yaml` | config | yes | yes | Final strategy settings and canonical output paths. |
| `src/regime/jump_model_clustering.py` | script | no | yes | Canonical entrypoint for regime discovery. |
| `src/regime/build_rule_based_regime.py` | script | yes | yes | Canonical entrypoint for rule-based regime construction. |
| `src/regime/plot_regime_diagnostics.py` | script | no | yes | Canonical entrypoint for regime diagnostic plots. |
| `src/timing/reproduce_monthly_timing.py` | script | no | yes | Canonical entrypoint for monthly SPY timing diagnostics. |
| `src/timing/build_backbone_v2.py` | script | yes | yes | Canonical entrypoint for BACKBONE_V2_UPGRADED. |
| `src/timing/recovery_rule_diagnostic.py` | script | no | yes | Canonical entrypoint for recovery rule diagnostics. |
| `src/timing/stress_period_final.py` | script | no | yes | Canonical entrypoint for final stress-period diagnostics. |
| `src/diagnostics/bond_sleeve_diagnostic.py` | script | no | yes | Canonical entrypoint for bond sleeve selection. |
| `src/diagnostics/cross_state_asset_behavior.py` | script | yes | yes | Canonical entrypoint for regime x stress asset behavior. |
| `src/diagnostics/drawdown_2015_2016_forensic.py` | script | no | yes | Canonical entrypoint for 2015-2016 forensic diagnostic. |
| `src/diagnostics/commodity_trigger_by_regime.py` | script | no | yes | Canonical entrypoint for commodity transmission by regime. |
| `src/diagnostics/flat_risk_gold_cash_diagnostic.py` | script | no | yes | Canonical entrypoint for FLAT_RISK GOLD vs CASH diagnostic. |
| `src/allocation/inverse_vol_allocation.py` | script | no | yes | Canonical entrypoint for inverse-vol window robustness. |
| `src/allocation/risk_parity_comparison.py` | script | no | yes | Canonical entrypoint for fixed, inverse-vol, and ERC allocation comparison. |
| `src/allocation/final_strategy_backtest.py` | script | yes | yes | Canonical entrypoint for `MATURE_REGIME_HEDGE_FINAL`. |
| `src/strategies/mature_regime_hedge_final.py` | script | no | yes | Validated final strategy implementation. |
| `results/01_regime_discovery/` | table/report folder | no | yes | Cleaned regime discovery outputs. |
| `results/02_rule_based_regime/` | table/report folder | no | yes | Cleaned rule-based regime outputs. |
| `results/03_monthly_timing/` | table/report folder | no | yes | Cleaned monthly timing outputs. |
| `results/04_stress_triggers/` | table/report folder | no | yes | Cleaned high-frequency stress trigger outputs. |
| `results/05_recovery/` | table/report folder | no | yes | Cleaned recovery rule diagnostics. |
| `results/06_2015_2016_repair/` | table/report folder | no | yes | Cleaned 2015-2016 slow-growth repair diagnostics. |
| `results/07_cross_state_asset_behavior/` | table/report folder | no | yes | Cleaned cross-state asset behavior diagnostics. |
| `results/08_allocation/` | table/report folder | no | yes | Cleaned allocation comparison outputs. |
| `results/09_final_strategy/mature_regime_hedge_final/performance_summary.csv` | table | yes | yes | Full-period strategy performance summary. |
| `results/09_final_strategy/mature_regime_hedge_final/crisis_performance.csv` | table | no | yes | Crisis window performance summary. |
| `results/09_final_strategy/mature_regime_hedge_final/daily_backtest_panel.csv` | table | no | yes | Daily final strategy panel with states, returns, weights, and signals. |
| `results/09_final_strategy/mature_regime_hedge_final/state_event_log.csv` | table | no | yes | State transition event log. |
| `results/09_final_strategy/mature_regime_hedge_final/episodes.csv` | table | no | yes | Risk and overlay episode details. |
| `results/09_final_strategy/mature_regime_hedge_final/final_strategy_component_attribution.csv` | table | no | yes | Component attribution versus mature baseline. |
| `results/09_final_strategy/mature_regime_hedge_final/final_strategy_decision_summary.csv` | table | no | yes | Final module decision summary. |
| `figures/09_final_strategy/mature_regime_hedge_final/final_equity_curve_log.png` | figure | yes | yes | Final equity curve comparison. |
| `figures/09_final_strategy/mature_regime_hedge_final/final_drawdown_comparison.png` | figure | yes | yes | Final drawdown comparison. |
| `figures/09_final_strategy/mature_regime_hedge_final/final_performance_bar_charts.png` | figure | no | yes | Main performance bar charts. |
| `figures/09_final_strategy/mature_regime_hedge_final/final_weight_stack.png` | figure | yes | yes | Final strategy weight stack. |
| `figures/09_final_strategy/mature_regime_hedge_final/final_state_timeline.png` | figure | no | yes | Final strategy regime and state timeline. |
| `figures/09_final_strategy/mature_regime_hedge_final/case_2008_GFC_final.png` | figure | no | yes | 2008 GFC case study. |
| `figures/09_final_strategy/mature_regime_hedge_final/case_2015_2016_final.png` | figure | yes | yes | 2015-2016 slow-growth case study. |
| `figures/09_final_strategy/mature_regime_hedge_final/case_2022_rate_war_final.png` | figure | yes | yes | 2022 rate and war shock case study. |
| `figures/09_final_strategy/mature_regime_hedge_final/case_2025_pullback_final.png` | figure | yes | yes | 2025 pullback case study. |
| `figures/09_final_strategy/mature_regime_hedge_final/final_component_attribution.png` | figure | no | yes | Final strategy component attribution. |
| `archive/exploratory_unused/moved_items.txt` | archive index | no | no | List of archived exploratory result and figure folders. |
