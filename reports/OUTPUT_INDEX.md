# Output Index

This index reflects the current source-only mainline after the buffered `FLAT/STEEP` GS10 classification and the VIX/CREDIT anchor state machine update.

| File path | Type | Description |
|---|---|---|
| `README.md` | report | GitHub-facing overview of the final mainline. |
| `reports/FINAL_REPORT.md` | report | Full strategy summary and research narrative. |
| `reports/HARD_DEPENDENCY_VALIDATION.md` | report | Source-only dependency validation report. |
| `scripts/final_strategy_source_only_core.py` | script | Canonical source-only implementation of regime classification, stress timing, and final allocation. |
| `scripts/08_stress_trigger_diagnostics.py` | script | Trigger-lock, turnover, and regime-level timing diagnostics. |
| `scripts/10_final_report_outputs.py` | script | Final output generator for tables, figures, heatmaps, and mainline README. |
| `scripts/hard_validate_main_pipeline_source_only.py` | script | Rebuild and dependency validation for the mainline. |
| `results/main_pipeline_final/tables/strategy_performance_comparison.csv` | table | Final performance table for SPY, SPY/CASH timing, and final strategy. |
| `results/main_pipeline_final/tables/final_daily_panel.csv` | table | Canonical source-only daily panel with regimes, states, returns, and weights. |
| `results/main_pipeline_final/tables/final_daily_weights.csv` | table | Daily final strategy weights. |
| `results/main_pipeline_final/tables/final_daily_returns.csv` | table | Daily strategy returns, NAV, drawdown, and state labels. |
| `results/main_pipeline_final/tables/cross_state_asset_behavior.csv` | table | Asset behavior by final heatmap bucket. |
| `results/main_pipeline_final/tables/regime_asset_behavior.csv` | table | Asset behavior by final regime. |
| `results/main_pipeline_final/tables/flat_gs10_hmm_summary.csv` | table | FLAT-only GS10 HMM summary supporting low/mid/high split. |
| `results/main_pipeline_final/tables/steep_gs10_hmm_summary.csv` | table | STEEP-only GS10 HMM summary supporting low/mid/high split. |
| `results/main_pipeline_final/tables/daily_trigger_diagnostics_panel.csv` | table | Daily trigger state and event diagnostics. |
| `results/main_pipeline_final/tables/trigger_effectiveness_summary.csv` | table | Trigger-to-unlock episode effectiveness summary. |
| `results/main_pipeline_final/figures/final_equity_curve_comparison.png` | figure | Final equity curve comparison. |
| `results/main_pipeline_final/figures/final_drawdown_curve_comparison.png` | figure | Final drawdown comparison. |
| `results/main_pipeline_final/figures/final_strategy_weights_timeline.png` | figure | Final strategy weight timeline. |
| `results/main_pipeline_final/figures/cross_state_asset_behavior_heatmap.png` | figure | Final heatmap of annualized return by cross-state bucket. |
| `results/main_pipeline_final/figures/cross_state_asset_sharpe_heatmap.png` | figure | Final heatmap of Sharpe by cross-state bucket. |
| `results/main_pipeline_final/figures/regime_asset_behavior_heatmap.png` | figure | Annualized return by regime. |
| `results/main_pipeline_final/figures/regime_asset_sharpe_heatmap.png` | figure | Sharpe by regime. |
| `results/main_pipeline_final/figures/flat_gs10_kde_hmm.png` | figure | FLAT-only GS10 KDE + HMM diagnostic. |
| `results/main_pipeline_final/figures/steep_gs10_kde_hmm.png` | figure | STEEP-only GS10 KDE + HMM diagnostic. |
| `results/main_pipeline_final/figures/trigger_regime_spy_timeline_long.png` | figure | SPY price with trigger entries, unlocks, and regime overlays. |
