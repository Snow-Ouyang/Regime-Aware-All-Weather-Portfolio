# Deletion / Archive Suggestions

No files were deleted or moved.

## Keep in main pipeline
- `scripts/01_data_prepare.py` through `scripts/10_final_report_outputs.py`: used_by_main_pipeline.
- `results/main_pipeline_final/`: final outputs and report.
- `results/09_final_strategy/mature_regime_hedge_final/`: validated mature baseline input.
- `results/flat_rate_refined_L50_H30/`: fixed refined baseline input.
- `results/recovery_20d_flat_low_only_L50_H30/`: final recovery overlay validation.

## Archive but keep for reproducibility
- `results/flat_regime_distribution_experiment/`: exploratory_only.
- `results/flat_gs10_threshold_robustness/`: superseded_by_final_pipeline, threshold evidence.
- `results/flat_stress_ratio_grid_search/`: superseded_by_final_pipeline, L50_H30 evidence.
- `results/recovery_20d_strategy_test_L50_H30/`: exploratory_only, global recovery evidence.
- `results/recovery_20d_equal_weight_attribution/`: exploratory_only, attribution evidence.
- `results/stress_exit_recovery_regime_attribution/`: exploratory_only, regime-filter evidence.

## Safe to delete or move to scratch after manual review
- `**/__pycache__/`: temporary_debug_file.
- `.ipynb_checkpoints/`: temporary_debug_file.
- duplicate old figures not referenced by `README_final_strategy.md`: duplicated_output.
- obsolete ad-hoc debug CSV/PNG files: unknown, manual confirmation required.