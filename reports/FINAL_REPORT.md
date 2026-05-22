# Final Report: Buffered Multi-State Regime Classification with VIX/CREDIT Anchor Timing

## 1. Final Mainline

Final strategies:

| Strategy | CAGR | Sharpe | Sortino | MaxDD | Calmar | Final Equity |
|---|---:|---:|---:|---:|---:|---:|
| SPY_BUY_HOLD | 11.14% | 0.575 | 0.702 | -55.19% | 0.202 | 8.38 |
| SPY_CASH_TIMING | 14.09% | 1.280 | 1.420 | -14.60% | 0.965 | 14.22 |
| FINAL_REGIME_HEDGE_TRIGGER_LOCK | 21.41% | 1.745 | 2.227 | -12.49% | 1.714 | 49.73 |

This is now the canonical source-only mainline.

## 2. What Changed

Two structural changes replaced the older mainline:

1. `FLAT` and `STEEP` are no longer split by one coarse threshold.
   They are internally classified into `LOW / MID / HIGH` with hysteresis bands and `3-day confirm`.
2. `SPY_CASH_TIMING` and the final hedge strategy now share the same stress state machine.
   There is no separate timing logic in the benchmark anymore.

The older commodity trigger is gone from the mainline.

## 3. Buffered Regime Classification

Base shape:

- `INVERTED`: `GS10 - GS1 < 0`
- `FLAT`: `0 <= GS10 - GS1 <= 1`
- `STEEP`: `GS10 - GS1 > 1`

Internal splits use `GS10`.

### FLAT

Diagnostic outputs:
- [flat GS10 KDE/HMM](C:/Users/FeixueOuyang/Desktop/research lab/all weather portfolio/results/main_pipeline_final/figures/flat_gs10_kde_hmm.png)
- [flat GS10 HMM summary](C:/Users/FeixueOuyang/Desktop/research lab/all weather portfolio/results/main_pipeline_final/tables/flat_gs10_hmm_summary.csv)

HMM mean levels:
- `LOW = 1.02`
- `MID = 2.59`
- `HIGH = 4.40`

Bands:
- `MID -> LOW = 1.1`
- `LOW -> MID = 1.3`
- `HIGH -> MID = 3.4`
- `MID -> HIGH = 3.6`

### STEEP

Diagnostic outputs:
- [steep GS10 KDE/HMM](C:/Users/FeixueOuyang/Desktop/research lab/all weather portfolio/results/main_pipeline_final/figures/steep_gs10_kde_hmm.png)
- [steep GS10 HMM summary](C:/Users/FeixueOuyang/Desktop/research lab/all weather portfolio/results/main_pipeline_final/tables/steep_gs10_hmm_summary.csv)

HMM mean levels:
- `LOW = 1.79`
- `MID = 2.51`
- `HIGH = 3.59`

Bands:
- `MID -> LOW = 2.0`
- `LOW -> MID = 2.3`
- `HIGH -> MID = 3.0`
- `MID -> HIGH = 3.2`

All internal transitions require `3-day confirm`.

This classification is not justified only by distribution shape. It is also supported by the cross-state asset heatmaps, where different refined states show different asset leadership.

## 4. Final Stress State Machine

The mainline uses a VIX/CREDIT anchor state machine.

### Enabled trigger sets

`VIX`:
- `FLAT_LOW_RATE`
- `FLAT_MID_RATE`
- `FLAT_HIGH_RATE`
- `INVERTED`

`CREDIT`:
- `FLAT_LOW_RATE`
- `FLAT_MID_RATE`
- `FLAT_HIGH_RATE`
- `STEEP_MID_RATE`
- `STEEP_HIGH_RATE`
- `INVERTED`

`STEEP_LOW_RATE` has no native trigger. If the strategy is still in stress while the regime transitions into `STEEP_LOW_RATE`, that is treated as carry-over rather than as a distinct trigger-enabled stress bucket.

### Rules

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

The first trigger that starts stress is the anchor trigger for exit.

## 5. Heatmap Logic

Mainline heatmap outputs:
- [cross-state return heatmap](C:/Users/FeixueOuyang/Desktop/research lab/all weather portfolio/results/main_pipeline_final/figures/cross_state_asset_behavior_heatmap.png)
- [cross-state Sharpe heatmap](C:/Users/FeixueOuyang/Desktop/research lab/all weather portfolio/results/main_pipeline_final/figures/cross_state_asset_sharpe_heatmap.png)
- [cross-state asset table](C:/Users/FeixueOuyang/Desktop/research lab/all weather portfolio/results/main_pipeline_final/tables/cross_state_asset_behavior.csv)

Final heatmap buckets:
- `FLAT_LOW_RATE_NORMAL`
- `FLAT_MID_RATE_NORMAL`
- `FLAT_LOWMID_RATE_STRESS`
- `FLAT_HIGH_RATE_NORMAL`
- `FLAT_HIGH_RATE_STRESS`
- `STEEP_LOW_RATE_NORMAL`
- `STEEP_MID_RATE_NORMAL`
- `STEEP_MID_RATE_STRESS`
- `STEEP_HIGH_RATE_NORMAL`
- `STEEP_HIGH_RATE_STRESS`
- `INVERTED_NORMAL`
- `INVERTED_STRESS`

Important handling:
- `FLAT_LOW_RATE_STRESS` and `FLAT_MID_RATE_STRESS` are merged into `FLAT_LOWMID_RATE_STRESS`.
- `STEEP_LOW_RATE_STRESS` is not drawn as a separate heatmap bucket because there is no native trigger there. Any such days are carry-over and use the same sleeve as `STEEP_LOW_RATE_NORMAL`.

Current sample sizes:
- `FLAT_LOW_RATE_NORMAL`: `193`
- `FLAT_MID_RATE_NORMAL`: `361`
- `FLAT_LOWMID_RATE_STRESS`: `398`
- `FLAT_HIGH_RATE_NORMAL`: `443`
- `FLAT_HIGH_RATE_STRESS`: `121`
- `STEEP_LOW_RATE_NORMAL`: `985`
- `STEEP_MID_RATE_NORMAL`: `678`
- `STEEP_MID_RATE_STRESS`: `339`
- `STEEP_HIGH_RATE_NORMAL`: `385`
- `STEEP_HIGH_RATE_STRESS`: `248`
- `INVERTED_NORMAL`: `751`
- `INVERTED_STRESS`: `172`

## 6. Final Allocation

| Regime / state | Allocation |
|---|---|
| `FLAT_LOW_RATE_NORMAL` | `SPY + CMDTY_FUT` inverse-vol |
| `FLAT_MID_RATE_NORMAL` | `SPY + GOLD` inverse-vol |
| `FLAT_LOWMID_RATE_STRESS` | `100% CASH` |
| `FLAT_HIGH_RATE_NORMAL` | `GOLD + CMDTY_FUT` inverse-vol |
| `FLAT_HIGH_RATE_STRESS` | `70% IEF + 30% (GOLD + CMDTY_FUT inverse-vol)` |
| `STEEP_LOW_RATE_NORMAL` | `SPY + CMDTY_FUT` inverse-vol |
| `STEEP_MID_RATE_NORMAL` | `100% SPY` |
| `STEEP_MID_RATE_STRESS` | `100% IEF` |
| `STEEP_HIGH_RATE_NORMAL` | `SPY + GOLD + CMDTY_FUT` inverse-vol |
| `STEEP_HIGH_RATE_STRESS` | `100% IEF` |
| `INVERTED_NORMAL` | `SPY + GOLD` inverse-vol |
| `INVERTED_STRESS` | `10% CASH + 90% (SPY + GOLD inverse-vol)` |

## 7. Crisis Windows

| Window | SPY_CASH_TIMING | FINAL_REGIME_HEDGE_TRIGGER_LOCK |
|---|---:|---:|
| 2008_GFC | `+7.40%`, MaxDD `-8.17%` | `+44.42%`, MaxDD `-6.18%` |
| 2011_EURO_DEBT | `-3.74%`, MaxDD `-4.55%` | `+10.34%`, MaxDD `-11.00%` |
| 2015_2016 | `+3.24%`, MaxDD `-3.32%` | `+13.33%`, MaxDD `-7.22%` |
| COVID_2020 | `+17.01%`, MaxDD `-6.99%` | `+20.59%`, MaxDD `-10.41%` |
| 2022_RATE_WAR | `+3.79%`, MaxDD `-9.73%` | `+14.80%`, MaxDD `-10.29%` |
| 2025_PULLBACK | `+11.58%`, MaxDD `-14.60%` | `+21.18%`, MaxDD `-6.18%` |

The final strategy is not only safer than SPY buy-and-hold. It also compounds better than the aligned SPY/CASH timing benchmark.

## 8. Source-Only Reproducibility

Run order:

```bash
python scripts/run_final_strategy_source_only.py
python scripts/08_stress_trigger_diagnostics.py
python scripts/10_final_report_outputs.py
python scripts/hard_validate_main_pipeline_source_only.py
```

The final mainline should reproduce from `data/raw` and `data/processed` only.

## 9. Remaining Limits

- This is still an in-sample result.
- `FLAT_HIGH_RATE_STRESS` and `INVERTED_STRESS` are still moderate-sample blocks.
- The HMM diagnostics support the internal structure, but do not by themselves prove out-of-sample alpha.
- Daily credit series still need careful calendar alignment and missing-value treatment.
