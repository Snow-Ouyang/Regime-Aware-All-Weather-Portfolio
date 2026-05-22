# Regime-Aware Stress Timing and Hedge Allocation Strategy

## Final Result

This repo now uses a buffered multi-state macro classification together with a VIX/CREDIT anchor stress state machine and regime-specific hedge sleeves.

![Final equity curve](results/main_pipeline_final/figures/final_equity_curve_comparison.png)

| Strategy | CAGR | Sharpe | Sortino | MaxDD | Calmar | Final Equity |
|---|---:|---:|---:|---:|---:|---:|
| SPY_BUY_HOLD | 11.14% | 0.575 | 0.702 | -55.19% | 0.202 | 8.38 |
| SPY_CASH_TIMING | 14.09% | 1.280 | 1.420 | -14.60% | 0.965 | 14.22 |
| FINAL_REGIME_HEDGE_TRIGGER_LOCK | 22.20% | 1.808 | 2.303 | -12.49% | 1.776 | 56.61 |

Relative to SPY buy-and-hold, the final strategy materially improves return and drawdown. Relative to the matching `SPY_CASH_TIMING` benchmark, it keeps the improved timing state machine and compounds more efficiently through regime-specific allocation.

## Regime Construction

The first layer remains yield-curve shape:

`term_spread = GS10 - GS1`

- `INVERTED`: `term_spread < 0`
- `FLAT`: `0 <= term_spread <= 1`
- `STEEP`: `term_spread > 1`

Inside `FLAT` and `STEEP`, the final mainline now uses **GS10 hysteresis bands** with `3-day confirm`. This replaced the older single-threshold low/high split because the full-sample GS10 distribution showed a stable three-state internal structure.

### FLAT GS10 structure

![FLAT GS10 KDE and HMM](results/main_pipeline_final/figures/flat_gs10_kde_hmm.png)

HMM mean GS10 levels:
- `LOW`: `1.02`
- `MID`: `2.59`
- `HIGH`: `4.40`

Final bands:
- `MID -> LOW = 1.1`
- `LOW -> MID = 1.3`
- `HIGH -> MID = 3.4`
- `MID -> HIGH = 3.6`

### STEEP GS10 structure

![STEEP GS10 KDE and HMM](results/main_pipeline_final/figures/steep_gs10_kde_hmm.png)

HMM mean GS10 levels:
- `LOW`: `1.79`
- `MID`: `2.51`
- `HIGH`: `3.59`

Final bands:
- `MID -> LOW = 2.0`
- `LOW -> MID = 2.3`
- `HIGH -> MID = 3.0`
- `MID -> HIGH = 3.2`

This gives the final regime universe:

- `FLAT_LOW_RATE`
- `FLAT_MID_RATE`
- `FLAT_HIGH_RATE`
- `STEEP_LOW_RATE`
- `STEEP_MID_RATE`
- `STEEP_HIGH_RATE`
- `INVERTED`

The hysteresis bands reduce internal churn. They are not cosmetic. They are there to suppress repeated high/low switching around the boundary.

## Stress Timing

The final timing module is a **VIX/CREDIT anchor state machine**:

- `VIX` is enabled in:
  - `FLAT_LOW_RATE`
  - `FLAT_MID_RATE`
  - `FLAT_HIGH_RATE`
  - `INVERTED`
- `CREDIT` is enabled in:
  - `FLAT_LOW_RATE`
  - `FLAT_MID_RATE`
  - `FLAT_HIGH_RATE`
  - `STEEP_MID_RATE`
  - `STEEP_HIGH_RATE`
  - `INVERTED`
- `STEEP_LOW_RATE` has **no native trigger**. If stress appears there, it is carry-over from another regime and is not treated as a standalone trigger-enabled stress block.
- If a stress period carries into a new regime, the strategy stays on a **stress sleeve** until the locks truly unlock. Regime shift may remap stress to a new regime's stress sleeve, but it never falls back to a normal sleeve while `FULL_RISK` is still active.

### Trigger rules

VIX entry:
- `VIX_ZSCORE_120D >= 3.0`

VIX unlock:
- `VIX_ZSCORE_120D < 1.5`
- `SPY > MA20`

Credit entry:
- `D_CREDIT_SPREAD_15D > 0.10`
- `SPY <= MA20`

Credit unlock:
- `SPY > MA50`
- `CREDIT_LEVEL_Z_252D < 0.9`

The state machine uses anchor exits:
- if stress started from VIX, VIX unlock is sufficient;
- if stress started from credit, credit unlock is sufficient.

This same state machine drives both:
- `SPY_CASH_TIMING`
- the final hedge strategy

So the timing benchmark and the final strategy are aligned.

## Allocation Logic

Final sleeves:

| Regime / state | Allocation |
|---|---|
| `FLAT_LOW_RATE_NORMAL` | `SPY + CMDTY_FUT` inverse-vol |
| `FLAT_MID_RATE_NORMAL` | `SPY + GOLD` inverse-vol |
| `FLAT_LOW/MID_RATE_STRESS` | `100% CASH` |
| `FLAT_HIGH_RATE_NORMAL` | `GOLD + CMDTY_FUT` inverse-vol |
| `FLAT_HIGH_RATE_STRESS` | `70% IEF + 30% (GOLD + CMDTY_FUT inverse-vol)` |
| `STEEP_LOW_RATE_NORMAL` | `SPY + CMDTY_FUT` inverse-vol |
| `STEEP_LOW_RATE_STRESS` | `100% SPY` |
| `STEEP_MID_RATE_NORMAL` | `100% SPY` |
| `STEEP_MID_RATE_STRESS` | `100% IEF` |
| `STEEP_HIGH_RATE_NORMAL` | `70% GOLD + 30% (SPY + CMDTY_FUT inverse-vol)` |
| `STEEP_HIGH_RATE_STRESS` | `100% IEF` |
| `INVERTED_NORMAL` | `SPY + GOLD` inverse-vol |
| `INVERTED_STRESS` | `10% CASH + 90% (SPY + GOLD inverse-vol)` |

Notes:
- `FLAT_LOW_RATE_STRESS` and `FLAT_MID_RATE_STRESS` are merged into `FLAT_LOWMID_RATE_STRESS` in the mainline heatmap because the low block was too small by itself.
- `STEEP_LOW_RATE_STRESS` is carry-over only. It has no native trigger, but once a stress period enters `STEEP_LOW_RATE`, it remains on a stress sleeve rather than reverting to normal before unlock.

## Heatmap Evidence

![Cross-state return heatmap](results/main_pipeline_final/figures/cross_state_asset_behavior_heatmap.png)

![Cross-state Sharpe heatmap](results/main_pipeline_final/figures/cross_state_asset_sharpe_heatmap.png)

![Pure regime-stress Sharpe heatmap](results/main_pipeline_final/figures/pure_regime_stress_asset_sharpe_heatmap.png)

These heatmaps are part of the main thesis:
1. macro data itself has internal structure, so `FLAT` and `STEEP` should not be forced into one coarse low/high split;
2. asset behavior also changes across those refined states, so the extra classification has allocation value.
3. the pure `regime x stress` Sharpe view is now part of the mainline outputs, so carry-over stress blocks can be analyzed directly.

Current heatmap buckets and sample sizes:
- `FLAT_LOW_RATE_NORMAL`: `193`
- `FLAT_MID_RATE_NORMAL`: `361`
- `FLAT_LOWMID_RATE_STRESS`: `398`
- `FLAT_HIGH_RATE_NORMAL`: `443`
- `FLAT_HIGH_RATE_STRESS`: `121`
- `STEEP_LOW_RATE_NORMAL`: `985`
- `STEEP_LOW_RATE_STRESS`: `188`
- `STEEP_MID_RATE_NORMAL`: `678`
- `STEEP_MID_RATE_STRESS`: `339`
- `STEEP_HIGH_RATE_NORMAL`: `385`
- `STEEP_HIGH_RATE_STRESS`: `248`
- `INVERTED_NORMAL`: `751`
- `INVERTED_STRESS`: `172`

## Crisis Windows

| Window | SPY_CASH_TIMING | FINAL_REGIME_HEDGE_TRIGGER_LOCK |
|---|---:|---:|
| 2008_GFC | `+7.40%`, MaxDD `-8.17%` | `+42.56%`, MaxDD `-6.18%` |
| 2011_EURO_DEBT | `-3.74%`, MaxDD `-4.55%` | `+20.05%`, MaxDD `-9.54%` |
| 2015_2016 | `+3.24%`, MaxDD `-3.32%` | `+17.18%`, MaxDD `-5.70%` |
| COVID_2020 | `+17.01%`, MaxDD `-6.99%` | `+20.59%`, MaxDD `-10.41%` |
| 2022_RATE_WAR | `+3.79%`, MaxDD `-9.73%` | `+14.80%`, MaxDD `-10.29%` |
| 2025_PULLBACK | `+11.58%`, MaxDD `-14.60%` | `+21.18%`, MaxDD `-6.18%` |

The final strategy improves compounding on top of the new timing logic without giving up stress control.

Representative crisis-window figures from the mainline outputs:

![2008 GFC](results/main_pipeline_final/figures/case_2008_GFC_final.png)

![2011 Euro Debt](results/main_pipeline_final/figures/case_2011_euro_debt_final.png)

![COVID 2020](results/main_pipeline_final/figures/case_2020_covid_final.png)

![2022 Rate War](results/main_pipeline_final/figures/case_2022_rate_war_final.png)

## Source-Only Mainline

Main run order:

```bash
python scripts/run_final_strategy_source_only.py
python scripts/08_stress_trigger_diagnostics.py
python scripts/10_final_report_outputs.py
python scripts/hard_validate_main_pipeline_source_only.py
```

Mainline outputs:
- [results/main_pipeline_final/README_final_strategy.md](results/main_pipeline_final/README_final_strategy.md)
- [strategy table](results/main_pipeline_final/tables/strategy_performance_comparison.csv)
- [final report](reports/FINAL_REPORT.md)

## Limitations

- This remains in-sample research.
- `FLAT_HIGH_RATE_STRESS` and `INVERTED_STRESS` are still moderate-sample blocks.
- The HMM/KDE diagnostics support the internal structure, but allocation still needs OOS validation.
- Daily credit data needs consistent filling and calendar alignment.
