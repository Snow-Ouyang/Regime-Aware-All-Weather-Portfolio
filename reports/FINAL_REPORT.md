# Final Report: Regime-Dependent Stress Triggers and Hedge Allocation

**Stress triggers are regime-dependent, and hedge assets are also regime-dependent.**

This project is not a traditional all-weather portfolio and not a generic timing model. It is a **SPY-centered regime-aware index enhancement strategy**. The objective is to keep SPY as the core long-run return engine while reducing major drawdowns through two linked ideas:

1. Stress detection should depend on the macro regime.
2. Hedge allocation should also depend on the macro regime.

The final strategy is:

`FINAL_REGIME_HEDGE_TRIGGER_LOCK`

It is source-only reproducible and does not rely on exploratory intermediate outputs.

## 1. Motivation

Most timing systems try to find a universal risk-off trigger. Most all-weather systems try to hold a static diversified portfolio across macro states. This project takes a different view.

Stress periods are heterogeneous. A VIX panic, a credit widening episode, a commodity-led growth scare, and an inflation/rate shock are not the same event. Likewise, the best hedge asset changes with the macro backdrop. CASH, IEF, GOLD, commodities, and SPY do not have stable roles across all regimes.

The strategy therefore asks:

- Which trigger is valid in which regime?
- Once stress is detected, which hedge asset is actually useful in that regime?
- Can the strategy preserve SPY participation without accepting SPY-like drawdown?

## 2. From ML Discovery to Rule-Based Regimes

The regime framework was inspired by the earlier ML regime project:

[Market-Regime-Clustering](https://github.com/Snow-Ouyang/Market-Regime-Clustering)

The clustering / jump-model research was used as a discovery phase. It helped reveal the natural distribution of macro variables and highlighted that macro states differ across curve shape, inflation pressure, credit conditions, and growth stress.

The final strategy does **not** trade directly on ML states. Instead, the ML evidence was translated into simpler rule-based macro regimes because rule-based regimes are:

- interpretable;
- stable across reruns;
- easier to audit;
- easier to reproduce from source data;
- better suited for a transparent research project.

Local note: the current cleaned repository does not retain the earlier clustering variable-distribution figures. Those can be restored from the upstream ML project if a separate discovery appendix is needed.

## 3. Final Regime Construction

The final macro regime framework is:

- `FLAT_LOW_RATE`
- `FLAT_HIGH_RATE`
- `STEEP_LOW_RATE`
- `STEEP_HIGH_RATE`
- `INVERTED`

The first split uses:

`term_spread = GS10 - GS1`

Then raw `FLAT` is split by absolute rate level using `GS10 = 3.0`. Confirmed `STEEP` regimes are further split by short-rate level using `GS1 = 0.3`.

| Regime | Rule | Threshold Source | Economic Interpretation | Strategy Implication |
|---|---|---|---|---|
| `INVERTED` | `term_spread < 0` | Yield-curve inversion | Tight policy, late-cycle inversion, unreliable stress-trigger quality | No full-risk trigger; SPY / GOLD inverse-vol |
| Raw `FLAT` | `0 <= term_spread <= 1` | Flat curve band | Curve shape is flat but rate level matters | Split by GS10 |
| `FLAT_LOW_RATE` | raw `FLAT` and `GS10 <= 3.0` | Rounded flat-rate split | Low-rate flat state; SPY and commodities can still work | Normal pool uses SPY / CMDTY_FUT |
| `FLAT_HIGH_RATE` | raw `FLAT` and `GS10 > 3.0` | Same GS10 split | High-rate flat state; equity exposure is less attractive | Normal pool uses GOLD / CMDTY_FUT |
| `STEEP_LOW_RATE` | raw `STEEP` and confirmed `GS1 <= 0.3` | Short-rate split from STEEP GS1 diagnostics | Low short-rate steep curve; SPY remains the clean return engine | Normal pool uses 100% SPY |
| `STEEP_HIGH_RATE` | raw `STEEP` and confirmed `GS1 > 0.3` | Same GS1 split with 3-day confirmation | Higher short-rate steep curve; commodities diversify equity exposure | Normal pool uses SPY / CMDTY_FUT inverse-vol |

The `GS10 = 3.0` threshold is the rounded version of the flat-rate diagnostic threshold. The `GS1 = 0.3` threshold is used inside confirmed `STEEP` regimes because GS1 better captures short-rate policy level, cash yield, and financing pressure than GS10. The main new finding is that commodities behave differently in `STEEP_LOW_RATE` and `STEEP_HIGH_RATE`: adding CMDTY_FUT to inverse-vol allocation in `STEEP_HIGH_RATE_NORMAL` improves return and reduces SPY path drawdown. We also ran an inverse-vol window grid search and found the strategy was not materially sensitive across reasonable windows, so the final mainline keeps a 90-day inverse-vol setting.

### Extreme Inflation

The ML discovery work showed oil-shock / policy-driven inflation-like states. In the current source-only tradable sample, there was not enough stable evidence to define a separate final `EXTREME_INFLATION` regime. The strategy therefore avoids forcing a regime that the sample cannot support. That said, this finding is one reason real assets, commodities, and gold are central to the final allocation research.

## 4. Why This Is Not an All-Weather Portfolio

Traditional all-weather portfolios are generally designed as static or semi-static diversification across growth and inflation quadrants.

This project is different:

- SPY remains the primary return engine.
- Macro regime determines which stress triggers are active.
- Stress episodes are managed with trigger locks rather than one-day exits.
- Hedge sleeves are regime-specific rather than universal.

This is a SPY-centered regime-aware index enhancement strategy. The goal is not to minimize volatility at all costs, but to preserve equity participation while reducing major regime-specific drawdowns.

Monthly trend timing was explored early as a benchmark, but it is not part of the final thesis. The final strategy replaces monthly timing with a higher-frequency trigger-lock stress system.

## 5. Regime-Specific Trigger-Lock Stress System

The final stress module is a trigger-lock state machine.

Each trigger has:

- an enabled regime set;
- an entry condition;
- an unlock condition;
- an economic interpretation.

When a trigger fires, it creates an active lock. The strategy stays in full-risk mode until all active locks are unlocked. Locks can be added during an existing stress episode, and each lock can be released independently.

| Trigger | Enabled Regimes | Entry | Unlock | Economic Meaning | Main Purpose |
|---|---|---|---|---|---|
| VIX lock | `STEEP_LOW_RATE`, `STEEP_HIGH_RATE`, `FLAT_LOW_RATE`, `FLAT_HIGH_RATE` | `VIX_ZSCORE_120D >= 3.0` | `VIX_ZSCORE_120D < 1.5` | Fast panic / volatility shock | Catch sudden volatility stress |
| Credit lock | `FLAT_LOW_RATE`, `FLAT_HIGH_RATE` | SPY drawdown <= -5% and `D_CREDIT_SPREAD_15D > 0.10` | `D_CREDIT_SPREAD_15D < 0` and SPY > MA20 | Price-confirmed credit stress | Avoid credit-led drawdowns |
| Commodity lock | `STEEP_LOW_RATE`, `STEEP_HIGH_RATE` | `CMDTY_RET60 < -10%` | `CMDTY_RET60 > -5%` and SPY > MA20 | STEEP slow-growth / commodity-led stress | Repair 2015-2016 style stress |

Important design choices:

- Monthly SELL is not used in the final strategy.
- Credit trigger is not enabled in `INVERTED`.
- Commodity trigger is only used inside confirmed STEEP regimes, across both low-rate and high-rate STEEP normal states.
- If VIX and credit locks are both active, VIX unlock also unlocks credit.

## 6. Trigger-to-Unlock Episode Diagnostics

Forward 20-day return is useful, but it is incomplete for a lock-based strategy. The final strategy should be evaluated from trigger entry to unlock exit.

| Regime | Trigger | Episodes | Avg Lock Duration | Mean Strategy Return During Stress | Mean SPY Return During Stress | Mean SPY MaxDD During Stress | Mean Drawdown Reduction vs SPY |
|---|---:|---:|---:|---:|---:|---:|---:|
| `FLAT_HIGH_RATE` | CREDIT | 2 | 66.0d | 3.99% | -3.25% | -15.50% | 11.97% |
| `FLAT_HIGH_RATE` | VIX | 7 | 12.6d | -0.02% | 1.24% | -3.55% | 2.54% |
| `FLAT_LOW_RATE` | CREDIT | 5 | 26.4d | 3.05% | 2.44% | -5.58% | 2.31% |
| `FLAT_LOW_RATE` | VIX | 4 | 16.0d | 0.94% | -2.92% | -10.32% | 6.18% |
| `STEEP_HIGH_RATE` | CMDTY | 4 | 71.2d | 5.83% | -10.12% | -18.28% | 15.09% |
| `STEEP_HIGH_RATE` | VIX | 3 | 12.0d | 0.00% | 0.80% | -3.65% | 2.28% |
| `STEEP_LOW_RATE` | CMDTY | 4 | 62.2d | 3.54% | 3.82% | -8.75% | 5.37% |
| `STEEP_LOW_RATE` | VIX | 6 | 7.2d | -0.91% | 2.71% | -1.37% | 0.35% |

Interpretation:

- VIX lock mainly catches fast crash conditions; not every VIX spike is a profitable hedge episode.
- Credit lock is strongest in `FLAT_HIGH_RATE`, where price-confirmed credit stress tends to be more damaging.
- Commodity lock is the key fix for STEEP slow-growth stress, with clearly stronger drawdown-reduction evidence in `STEEP_HIGH_RATE` than in `STEEP_LOW_RATE`.
- The same trigger can be effective in one regime and noisy in another.

Core files:

- `results/main_pipeline_final/tables/stress_entry_attribution.csv`
- `results/main_pipeline_final/tables/trigger_effectiveness_summary.csv`
- `results/main_pipeline_final/figures/stress_entry_timeline_by_trigger.png`
- `results/main_pipeline_final/figures/trigger_regime_spy_timeline_long.png`

## 7. Regime x Stress Asset Behavior

The most important empirical evidence is cross-state asset behavior.

The final allocation is not assigned arbitrarily. It is derived from observed asset behavior under each regime-stress cross state.

Summary:

- `FLAT_LOW_RATE_NORMAL`: SPY and commodities are strong. GOLD is removed from the normal pool.
- `FLAT_LOW_RATE_STRESS`: GOLD has the strongest defensive profile.
- `FLAT_HIGH_RATE_NORMAL`: GOLD and commodities dominate SPY. IEF is not included in the final normal pool.
- `FLAT_HIGH_RATE_STRESS`: IEF performs best, with CASH used as a stabilizer.
- `STEEP_LOW_RATE_NORMAL`: SPY is the return engine.
- `STEEP_HIGH_RATE_NORMAL`: SPY remains useful, but CMDTY_FUT improves diversification and lowers the SPY-only drawdown profile.
- `STEEP_FULL_RISK`: 30% GOLD / 70% IEF is used instead of a single hedge asset.
- `INVERTED`: the final strategy keeps SPY / GOLD inverse-vol and does not force a full-risk state.

![Cross-state asset return heatmap](../results/main_pipeline_final/figures/cross_state_asset_behavior_heatmap.png)

![Cross-state asset Sharpe heatmap](../results/main_pipeline_final/figures/cross_state_asset_sharpe_heatmap.png)

## 8. Final Strategy Allocation

| Macro Regime | State | Trigger Condition | Allocation | Rationale |
|---|---|---|---|---|
| `FLAT_LOW_RATE` | Normal | No active VIX or credit lock | SPY / CMDTY_FUT inverse-vol | Low-rate flat state can still reward equity and commodity exposure |
| `FLAT_LOW_RATE` | Stress | VIX or credit lock active | 100% GOLD | Gold is the strongest defensive asset in this cross-state |
| `FLAT_HIGH_RATE` | Normal | No active VIX or credit lock | GOLD / CMDTY_FUT inverse-vol | High-rate flat state favors real assets over SPY |
| `FLAT_HIGH_RATE` | Stress | VIX or credit lock active | 90% IEF / 10% CASH | IEF has the best stress evidence; CASH stabilizes the hedge sleeve |
| `STEEP_LOW_RATE` | Normal | No active VIX or commodity lock | 100% SPY | Low short-rate STEEP remains equity-friendly |
| `STEEP_HIGH_RATE` | Normal | No active VIX or commodity lock | SPY / CMDTY_FUT inverse-vol | Higher short-rate STEEP benefits from commodity diversification |
| `STEEP` | Full risk | VIX or commodity lock active | 30% GOLD / 70% IEF | Stress remains merged; IEF handles duration stress and GOLD diversifies commodity / inflation shock risk |
| `INVERTED` | Normal only | No full-risk trigger enabled | SPY / GOLD inverse-vol | Inversion is not treated as automatic cash risk-off |

## 9. Backtest Results

| Strategy | CAGR | Sharpe | Sortino | MaxDD | Calmar | Final Equity |
|---|---:|---:|---:|---:|---:|---:|
| SPY_BUY_HOLD | 11.14% | 0.575 | 0.702 | -55.19% | 0.202 | 8.38 |
| SPY_CASH_TIMING | 12.04% | 0.948 | 1.101 | -29.45% | 0.409 | 9.86 |
| FINAL_REGIME_HEDGE_TRIGGER_LOCK | 20.20% | 1.492 | 2.013 | -15.94% | 1.267 | 40.61 |

Compared with SPY buy-and-hold:

- CAGR improves from 11.14% to 20.20%.
- Sharpe improves from 0.575 to 1.492.
- MaxDD falls from -55.19% to -15.94%.
- Final equity improves from 8.38 to 40.61.

The improvement is not from a single optimized trigger. It comes from the interaction of regime-specific triggers, trigger-lock stress episodes, regime-specific hedge allocation, and inverse-vol normal allocation.

![Final equity curve](../results/main_pipeline_final/figures/final_equity_curve_comparison.png)

![Final drawdown curve](../results/main_pipeline_final/figures/final_drawdown_curve_comparison.png)

![Final performance bars](../results/main_pipeline_final/figures/final_performance_bar_charts.png)

![Final weight timeline](../results/main_pipeline_final/figures/final_strategy_weights_timeline.png)

## 10. Crisis Window Analysis

| Window | Main Stress Type | Trigger / State | Hedge Behavior | Result |
|---|---|---|---|---|
| 2008 GFC | Credit and broad risk stress | FLAT_HIGH credit, later STEEP commodity lock | IEF / GOLD hedge exposure avoided the largest equity damage | Final +54.73%, SPY -37.16% |
| 2015-2016 | Commodity / growth stress | STEEP commodity lock | GOLD / IEF hedge repaired the original missed slow-growth stress | Final +14.43%, MaxDD -6.29% |
| COVID 2020 | Fast volatility shock | FLAT_LOW VIX / credit locks | GOLD helped, though fast SPY recovery created opportunity cost | Final +27.41%, MaxDD -15.50% |
| 2022 rate / inflation / war shock | Rate shock and inflation stress | STEEP / FLAT locks | Regime-specific hedge sleeves beat SPY and SPY/CASH timing | Final +6.60%, MaxDD -14.09% |
| 2025 pullback | High-rate volatility stress | FLAT_HIGH VIX lock | IEF / CASH stress sleeve reduced drawdown | Final +19.99%, MaxDD -7.79% |

Case studies:

![2008 GFC](../results/main_pipeline_final/figures/case_2008_GFC_final.png)

![2015-2016](../results/main_pipeline_final/figures/case_2015_2016_final.png)

![2022 rate / war shock](../results/main_pipeline_final/figures/case_2022_rate_war_final.png)

![2025 pullback](../results/main_pipeline_final/figures/case_2025_pullback_final.png)

## 11. Trigger and Turnover Diagnostics

The final stress system is not optimized to minimize turnover. Turnover is mainly caused by full-risk entry and unlock events, not by inverse-vol rebalance.

Diagnostic summary:

- Total full-risk entries: 35.
- Total full-risk exits: 35.
- Top turnover events are full-risk unlock exits and VIX entries.
- `STEEP -> STEEP` turnover is mostly internal transition between `STEEP_LOW_RATE_NORMAL`, `STEEP_HIGH_RATE_NORMAL`, and `STEEP_FULL_RISK`.
- Recovery overlay is not part of the final mainline.

Relevant figures:

![Turnover by trigger event](../results/main_pipeline_final/figures/turnover_by_trigger_event.png)

![Allocation-state transition turnover](../results/main_pipeline_final/figures/turnover_by_allocation_state_transition.png)

![STEEP internal transition turnover](../results/main_pipeline_final/figures/steep_internal_transition_turnover.png)

## 12. Methodology Notes

- The mainline is source-only and uses `data/raw` and `data/processed`.
- It does not depend on old validated or exploratory result folders.
- Signal day `t` affects position on day `t+1`.
- Regime confirmation requires 3 consecutive days.
- FLAT low/high uses rounded `GS10 = 3.0`.
- STEEP low/high uses `GS1 = 0.3` with 3-day confirmation.
- There is no `NEUTRAL` regime and no fallback allocation.
- Inverse-volatility uses a 90 trading day window. A light grid search across reasonable windows showed only limited performance sensitivity, so the final mainline keeps 90.
- Transaction cost is 10 bps one-way.
- CASH uses compounded daily DTB3.
- Credit spread uses WBAA - WAAA.
- Final credit-lock logic uses `D_CREDIT_SPREAD_15D`.
- VIX z-score uses a 120 trading day rolling window, current-day inclusive, `ddof=1`.
- No look-ahead data is used for strategy generation.

Run order:

```bash
python scripts/01_data_prepare.py
python scripts/02_rule_based_regime.py
python scripts/03_stress_detection.py
python scripts/04_asset_return_panel.py
python scripts/05_baseline_strategy.py
python scripts/06_flat_rate_refined_strategy.py
python scripts/07_cross_state_asset_behavior.py
python scripts/08_stress_trigger_diagnostics.py
python scripts/09_final_strategy_recovery_flat_low_only.py
python scripts/10_final_report_outputs.py
```

The name `09_final_strategy_recovery_flat_low_only.py` is historical. It now outputs the trigger-lock final strategy.

## 13. Limitations

- Stress events are sparse and heterogeneous.
- Commodity proxy choice affects commodity-trigger timing.
- Macro data may be revised or published with delay.
- Trigger-lock thresholds require out-of-sample validation.
- The strategy remains SPY-centered, not minimum-volatility.
- Regime thresholds are economically motivated but still simplified.
- The strategy is research code and not financial advice.

## 14. Final Interpretation

The main contribution is not a single optimized trigger. The main contribution is a regime-conditioned framework showing that both stress detection and hedge allocation depend on the macro regime.

The final strategy shows that:

- VIX, credit, and commodity stress do not have the same meaning in every regime.
- GOLD, IEF, CASH, commodities, and SPY do not have fixed roles across regimes.
- A regime-aware trigger-lock state machine can preserve equity participation while materially reducing major drawdowns.
