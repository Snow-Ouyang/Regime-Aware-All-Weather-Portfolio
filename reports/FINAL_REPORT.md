# Final Report: Regime-Dependent Stress Triggers and Hedge Allocation

**Stress triggers are regime-dependent, and hedge assets are also regime-dependent.**

This project is not a traditional all-weather portfolio and not a generic timing model. It is a **SPY-centered regime-aware index enhancement strategy**. The objective is to keep SPY as the core long-run return engine while reducing major drawdowns through two linked ideas:

1. Stress detection should depend on the macro regime.
2. Hedge allocation should also depend on the macro regime.

The final strategy is:

`FINAL_REGIME_HEDGE_TRIGGER_LOCK`

It is source-only reproducible and does not depend on exploratory intermediate outputs.

## 1. Motivation

Most timing systems search for a universal risk-off trigger. Most all-weather systems search for a static hedge sleeve. This project takes a different view.

Stress is heterogeneous. A VIX panic, a persistent credit-widening regime, a stair-step rate shock, and a commodity-led growth scare do not have the same structure. Likewise, the best hedge asset changes with the macro backdrop. CASH, IEF, GOLD, commodities, and SPY do not have stable roles across all regimes.

The strategy therefore asks:

- Which trigger is valid in which regime?
- Once stress is detected, which hedge asset is actually useful in that regime?
- Can the strategy preserve SPY participation without accepting SPY-like drawdown?

## 2. From ML Discovery to Rule-Based Regimes

The regime framework was inspired by the earlier ML regime project:

[Market-Regime-Clustering](https://github.com/Snow-Ouyang/Market-Regime-Clustering)

The clustering / jump-model research was used as a discovery phase. It revealed how macro variables naturally cluster and highlighted that stress states differ across curve shape, rate level, and credit pressure.

The final strategy does **not** trade directly on ML labels. Instead, the ML evidence was translated into simpler rule-based macro regimes because rule-based regimes are:

- interpretable,
- stable across reruns,
- easier to audit,
- easier to reproduce from source data,
- and better suited to a transparent research project.

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
| `INVERTED` | `term_spread < 0` | Yield-curve inversion | Tight policy, late-cycle inversion | Keep SPY/GOLD normal exposure but allow explicit stress states |
| Raw `FLAT` | `0 <= term_spread <= 1` | Flat curve band | Curve shape is flat but rate level matters | Split by GS10 |
| `FLAT_LOW_RATE` | raw `FLAT` and `GS10 <= 3.0` | Rounded flat-rate split | Low-rate flat state; equity still has room to work | SPY / CMDTY_FUT normal sleeve |
| `FLAT_HIGH_RATE` | raw `FLAT` and `GS10 > 3.0` | Same GS10 split | High-rate flat state favors real assets over SPY | GOLD / CMDTY_FUT normal sleeve |
| `STEEP_LOW_RATE` | raw `STEEP` and confirmed `GS1 <= 0.3` | Short-rate split within STEEP | Low short-rate steep curve; SPY remains a strong return engine | 100% SPY normal sleeve |
| `STEEP_HIGH_RATE` | raw `STEEP` and confirmed `GS1 > 0.3` | Same GS1 split with 3-day confirmation | Higher short-rate steep curve; broader diversification is useful | SPY / GOLD / CMDTY_FUT inverse-vol normal sleeve |

The `GS10 = 3.0` threshold is the rounded version of the flat-rate diagnostic threshold. The `GS1 = 0.3` threshold is used inside confirmed `STEEP` regimes because GS1 better captures short-rate policy level, cash yield, and financing pressure than GS10.

## 4. Why This Is Not an All-Weather Portfolio

Traditional all-weather portfolios are generally static or semi-static allocations across growth and inflation quadrants.

This project is different:

- SPY remains the primary return engine.
- Stress detection is regime-conditioned.
- Stress episodes are managed with a lock state machine rather than one-day exits.
- Hedge sleeves are regime-specific rather than universal.

This is a SPY-centered regime-aware index enhancement strategy. The goal is not to minimize volatility at all costs, but to preserve equity participation while reducing major regime-specific drawdowns.

Monthly timing was explored early as a benchmark, but it is not part of the final thesis. The final mainline now uses a cleaner VIX/CREDIT anchor state machine.

## 5. Regime-Specific Trigger-Lock Stress System

The final stress module is a trigger-lock state machine.

When a trigger fires first, it becomes the anchor lock. If another lock is added during the same episode, the anchor still determines the main exit. This prevents late secondary signals from distorting the stress window.

| Trigger | Enabled Regimes | Entry | Unlock | Economic Meaning | Main Purpose |
|---|---|---|---|---|---|
| VIX lock | `FLAT_LOW_RATE`, `FLAT_HIGH_RATE`, `INVERTED` | `VIX_ZSCORE_120D >= 3.0` | `VIX_ZSCORE_120D < 1.5` and `SPY > MA20` | Fast panic / volatility shock | Catch fast shocks, especially when inversion already exists |
| Credit lock | `FLAT_LOW_RATE`, `FLAT_HIGH_RATE`, `STEEP_LOW_RATE`, `STEEP_HIGH_RATE`, `INVERTED` | `SPY_DD <= -5%`, `D_CREDIT_SPREAD_15D > 0.10`, and `SPY <= MA20` | `D_CREDIT_SPREAD_15D < 0`, `SPY > MA50`, and `CREDIT_LEVEL_Z_252D < 0.9` | Price-confirmed and still-elevated credit stress | Capture sustained and stair-step stress such as 2008 and 2022 |

Anchor exit logic:

- If stress began with VIX, VIX unlock is sufficient to exit stress.
- If stress began with credit, credit unlock is sufficient to exit stress.
- If both were active at entry, they unlock independently.

Commodity trigger is not part of the final mainline.

## 6. Trigger-to-Unlock Episode Diagnostics

Forward 20-day return is useful, but incomplete for a lock-based strategy. Trigger quality should be evaluated from entry to unlock.

| Regime | Trigger | Episodes | Avg Lock Duration | Mean Strategy Return During Stress |
|---|---:|---:|---:|---:|
| `FLAT_HIGH_RATE` | CREDIT | 1 | 357.0d | 16.03% |
| `FLAT_HIGH_RATE` | VIX | 5 | 12.0d | -0.30% |
| `FLAT_LOW_RATE` | CREDIT | 2 | 229.0d | 4.99% |
| `FLAT_LOW_RATE` | VIX | 4 | 16.0d | 0.34% |
| `INVERTED` | VIX | 4 | 14.2d | 0.73% |
| `STEEP_HIGH_RATE` | CREDIT | 2 | 91.5d | 2.24% |
| `STEEP_LOW_RATE` | CREDIT | 3 | 101.7d | 7.46% |

Interpretation:

- VIX now mainly handles fast panic in `FLAT` and `INVERTED`.
- Credit is the main sustained-stress detector, including the new `STEEP` and `INVERTED` stress states.
- The new state machine is cleaner than the prior commodity-trigger version, but still preserves differentiated hedge behavior through the allocation layer.

## 7. Regime x Stress Asset Behavior

The final allocation is not assigned arbitrarily. It is derived from observed asset behavior under each regime-stress cross state.

Main findings under the new state machine:

- `FLAT_LOW_RATE_STRESS`: CASH dominates GOLD under the new stress sample.
- `FLAT_HIGH_RATE_STRESS`: IEF remains the strongest hedge asset.
- `STEEP_HIGH_RATE_NORMAL`: SPY, GOLD, and CMDTY_FUT are all strong, so a tri-asset inverse-vol sleeve is justified.
- `STEEP_HIGH_RATE_STRESS`: IEF clearly dominates.
- `STEEP_LOW_RATE_STRESS`: SPY still participates positively, but IEF materially improves path control; this motivates a mixed sleeve rather than a pure hedge.
- `INVERTED_STRESS`: SPY and GOLD remain better than IEF, so the final stress sleeve keeps SPY/GOLD exposure and adds only a small CASH buffer.

![Cross-state asset return heatmap](../results/main_pipeline_final/figures/cross_state_asset_behavior_heatmap.png)

![Cross-state asset Sharpe heatmap](../results/main_pipeline_final/figures/cross_state_asset_sharpe_heatmap.png)

## 8. Final Strategy Allocation

| Macro Regime | State | Trigger Condition | Allocation | Rationale |
|---|---|---|---|---|
| `FLAT_LOW_RATE` | Normal | No active VIX or credit lock | SPY / CMDTY_FUT inverse-vol | Low-rate flat state can still reward equity and commodity exposure |
| `FLAT_LOW_RATE` | Stress | VIX or credit lock active | 100% CASH | New stress sample is cleaner for cash than gold |
| `FLAT_HIGH_RATE` | Normal | No active VIX or credit lock | GOLD / CMDTY_FUT inverse-vol | High-rate flat state favors real assets over SPY |
| `FLAT_HIGH_RATE` | Stress | VIX or credit lock active | 100% IEF | IEF is the strongest stress hedge in this block |
| `STEEP_LOW_RATE` | Normal | No active credit lock | 100% SPY | Low short-rate STEEP remains equity-friendly |
| `STEEP_LOW_RATE` | Stress | Credit lock active | 60% SPY / 40% IEF | Stress is real, but SPY still participates positively |
| `STEEP_HIGH_RATE` | Normal | No active credit lock | SPY / GOLD / CMDTY_FUT inverse-vol | Higher short-rate STEEP benefits from broader diversification |
| `STEEP_HIGH_RATE` | Stress | Credit lock active | 10% CASH / 90% IEF | IEF is the dominant hedge, with a small cash stabilizer |
| `INVERTED` | Normal | No active VIX or credit lock | SPY / GOLD inverse-vol | Inversion is not automatically risk-off |
| `INVERTED` | Stress | VIX or credit lock active | 10% CASH + 90% (SPY / GOLD inverse-vol) | Stress still favors SPY/GOLD over IEF |

## 9. Backtest Results

| Strategy | CAGR | Sharpe | Sortino | MaxDD | Calmar | Final Equity |
|---|---:|---:|---:|---:|---:|---:|
| SPY_BUY_HOLD | 11.14% | 0.575 | 0.702 | -55.19% | 0.202 | 8.38 |
| SPY_CASH_TIMING | 13.45% | 1.223 | 1.347 | -14.60% | 0.921 | 12.69 |
| FINAL_REGIME_HEDGE_TRIGGER_LOCK | 19.73% | 1.660 | 2.125 | -11.58% | 1.704 | 37.54 |

Compared with SPY buy-and-hold:

- CAGR improves from 11.14% to 19.73%.
- Sharpe improves from 0.575 to 1.660.
- MaxDD falls from -55.19% to -11.58%.
- Final equity improves from 8.38 to 37.54.

Compared with the new SPY/CASH timing benchmark:

- CAGR improves from 13.45% to 19.73%.
- Sharpe improves from 1.223 to 1.660.
- MaxDD improves from -14.60% to -11.58%.
- Final equity improves from 12.69 to 37.54.

The improvement is not from a universal trigger. It comes from the interaction of regime-specific triggers, anchor-style stress periods, and regime-specific hedge sleeves.

![Final equity curve](../results/main_pipeline_final/figures/final_equity_curve_comparison.png)

![Final drawdown curve](../results/main_pipeline_final/figures/final_drawdown_curve_comparison.png)

![Final performance bars](../results/main_pipeline_final/figures/final_performance_bar_charts.png)

## 10. Crisis Window Analysis

| Window | Main Stress Type | Final Strategy Result | Comment |
|---|---|---:|---|
| 2008 GFC | Sustained credit crisis | +39.08%, MaxDD -5.54% | New credit lock kept the strategy defensive through the deepest phase |
| 2015-2016 | Slow-growth / commodity stress spillover | +4.88%, MaxDD -5.61% | Less upside than older commodity-trigger variants, but still controlled |
| COVID 2020 | Fast volatility shock | +17.92%, MaxDD -11.09% | VIX handled the fast spike while the hedge sleeve preserved most of the rebound |
| 2022 rate / inflation / war shock | Stair-step elevated credit stress | +20.72%, MaxDD -10.29% | This is the clearest beneficiary of the new credit state machine |
| 2025 pullback | High-rate volatility stress | +19.24%, MaxDD -8.35% | Strong drawdown control without major opportunity cost |

Case studies:

![2008 GFC](../results/main_pipeline_final/figures/case_2008_GFC_final.png)

![2015-2016](../results/main_pipeline_final/figures/case_2015_2016_final.png)

![2022 rate / war shock](../results/main_pipeline_final/figures/case_2022_rate_war_final.png)

![2025 pullback](../results/main_pipeline_final/figures/case_2025_pullback_final.png)

## 11. Trigger and Turnover Diagnostics

The new state machine reduces conceptual complexity even though it still trades actively enough to maintain stress control.

Diagnostic summary:

- Total full-risk entries: 21
- Total full-risk exits: 21
- Top turnover events remain full-risk unlock exits and VIX / credit entries.
- `SPY_CASH_TIMING` and the final hedge strategy now share the same anchor stress definition.
- Commodity-trigger turnover is gone from the mainline.

Relevant figures:

![Turnover by trigger event](../results/main_pipeline_final/figures/turnover_by_trigger_event.png)

![Allocation-state transition turnover](../results/main_pipeline_final/figures/turnover_by_allocation_state_transition.png)

![STEEP internal transition turnover](../results/main_pipeline_final/figures/steep_internal_transition_turnover.png)

## 12. Methodology Notes

- The mainline is source-only and uses `data/raw` and `data/processed`.
- Signal day `t` affects position on day `t+1`.
- Regime confirmation requires 3 consecutive days.
- FLAT low/high uses `GS10 = 3.0`.
- STEEP low/high uses `GS1 = 0.3` with 3-day confirmation.
- There is no `NEUTRAL` regime and no fallback allocation.
- Inverse-volatility uses a 90 trading day window.
- Transaction cost is 10 bps one-way.
- CASH uses compounded daily DTB3.
- Credit spread uses daily `DBAA - DAAA`, aligned to the trading calendar before feature construction.
- Credit unlock requires `CREDIT_LEVEL_Z_252D < 0.9`.
- Commodity trigger is not part of the final mainline.

## 13. Limitations

- Stress windows are sparse and heterogeneous.
- Daily credit data still requires careful missing-value handling and publication awareness.
- The anchor-exit state machine is economically motivated, but still tuned in-sample.
- `STEEP_LOW_RATE_STRESS` and `INVERTED_STRESS` are newer states and need more OOS evidence.
- The strategy remains SPY-centered, not minimum-volatility.
- Regime thresholds are economically motivated but still simplified.
- This is not financial advice.

## 14. Final Interpretation

The main contribution is not a single optimized trigger. The main contribution is a regime-conditioned framework showing that both stress detection and hedge allocation depend on the macro regime.

The current mainline reflects that idea more cleanly than the previous version:

- stress timing is handled by a simpler VIX/CREDIT anchor state machine,
- `SPY_CASH_TIMING` is materially stronger than before,
- and the final hedge strategy preserves that timing improvement while lowering drawdown further through regime-specific sleeves.
