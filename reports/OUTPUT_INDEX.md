# Output Index

This index lists the current source-only mainline files after the STEEP short-rate split upgrade.

| File path | Type | README | FINAL_REPORT | Description |
|---|---|---:|---:|---|
| `README.md` | report | yes | no | GitHub-facing project overview. |
| `reports/FINAL_REPORT.md` | report | no | yes | Full research narrative. |
| `reports/HARD_DEPENDENCY_VALIDATION.md` | report | no | yes | Source-only dependency validation report. |
| `scripts/final_strategy_source_only_core.py` | script | yes | yes | Canonical source-only panel, trigger-lock state machine, and final strategy implementation. |
| `scripts/01_data_prepare.py` | script | no | yes | Data availability check for source-only inputs. |
| `scripts/02_rule_based_regime.py` | script | yes | yes | Rule-based regime construction with `GS10 = 3.0` FLAT split and `GS1 = 0.3` STEEP split. |
| `scripts/03_stress_detection.py` | script | no | yes | Stress trigger source panel output. |
| `scripts/04_asset_return_panel.py` | script | no | yes | Asset return panel output. |
| `scripts/05_baseline_strategy.py` | script | no | yes | Baseline strategy output retained for source-only reproducibility. |
| `scripts/06_flat_rate_refined_strategy.py` | script | no | yes | Flat-rate refined intermediate output retained for source-only reproducibility. |
| `scripts/07_cross_state_asset_behavior.py` | script | yes | yes | Cross-state asset behavior output wrapper. |
| `scripts/08_stress_trigger_diagnostics.py` | script | yes | yes | Trigger-lock and turnover diagnostics. |
| `scripts/09_final_strategy_recovery_flat_low_only.py` | script | yes | yes | Historical filename; now writes trigger-lock final strategy daily returns and weights. |
| `scripts/10_final_report_outputs.py` | script | yes | yes | Final tables, figures, and README output generator. |
| `scripts/hard_validate_main_pipeline_source_only.py` | script | no | yes | Verifies numbered mainline scripts are source-only and rebuild matches current reference outputs. |
| `results/main_pipeline_final/tables/strategy_performance_comparison.csv` | table | yes | yes | Final strategy performance comparison. |
| `results/main_pipeline_final/tables/cross_state_asset_behavior.csv` | table | yes | yes | Asset behavior by final allocation state. |
| `results/main_pipeline_final/tables/flat_low_high_asset_behavior.csv` | table | yes | yes | Asset behavior by final regime. |
| `results/main_pipeline_final/tables/final_daily_weights.csv` | table | yes | yes | Daily final strategy weights. |
| `results/main_pipeline_final/tables/final_daily_returns.csv` | table | yes | yes | Daily final strategy return, NAV, drawdown, state, and locks. |
| `results/main_pipeline_final/tables/daily_trigger_diagnostics_panel.csv` | table | no | yes | Daily trigger-lock diagnostics. |
| `results/main_pipeline_final/tables/trigger_effectiveness_summary.csv` | table | yes | yes | Trigger-to-unlock episode effectiveness summary. |
| `results/main_pipeline_final/figures/final_equity_curve_comparison.png` | figure | yes | yes | Final equity curve comparison. |
| `results/main_pipeline_final/figures/final_drawdown_curve_comparison.png` | figure | yes | yes | Final drawdown curve comparison. |
| `results/main_pipeline_final/figures/final_performance_bar_charts.png` | figure | no | yes | Performance bar charts. |
| `results/main_pipeline_final/figures/final_strategy_weights_timeline.png` | figure | yes | yes | Final strategy weight timeline. |
| `results/main_pipeline_final/figures/cross_state_asset_behavior_heatmap.png` | figure | yes | yes | Asset annualized return heatmap by final allocation state. |
| `results/main_pipeline_final/figures/cross_state_asset_sharpe_heatmap.png` | figure | yes | yes | Asset Sharpe heatmap by final allocation state. |
| `results/main_pipeline_final/figures/flat_low_high_asset_behavior_heatmap.png` | figure | yes | yes | Asset annualized return heatmap by final regime. |
| `results/main_pipeline_final/figures/flat_low_high_asset_sharpe_heatmap.png` | figure | yes | yes | Asset Sharpe heatmap by final regime. |
| `results/main_pipeline_final/figures/trigger_regime_spy_timeline_long.png` | figure | yes | yes | SPY price with final regime backgrounds, trigger entries, and unlock marks. |
