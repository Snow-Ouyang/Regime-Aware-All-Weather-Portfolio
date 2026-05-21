# Regime-Aware Stress Timing and Hedge Allocation Strategy

## Final Result

This project builds a SPY-centered regime-aware index enhancement strategy. By combining regime-conditioned stress timing, anchor-style trigger-lock exits, and regime-specific hedge allocation, the final strategy improves both return and drawdown control relative to SPY buy-and-hold and the SPY/CASH timing benchmark.

![Final equity curve](results/main_pipeline_final/figures/final_equity_curve_comparison.png)

| Strategy | CAGR | Sharpe | Sortino | MaxDD | Calmar | Final Equity |
|---|---:|---:|---:|---:|---:|---:|
| SPY_BUY_HOLD | 11.14% | 0.575 | 0.702 | -55.19% | 0.202 | 8.38 |
| SPY_CASH_TIMING | 13.45% | 1.223 | 1.347 | -14.60% | 0.921 | 12.69 |
| FINAL_REGIME_HEDGE_TRIGGER_LOCK | 19.73% | 1.660 | 2.125 | -11.58% | 1.704 | 37.54 |

Compared with SPY buy-and-hold, the final strategy improves CAGR from 11.14% to 19.73%, raises Sharpe from 0.575 to 1.660, and reduces MaxDD from -55.19% to -11.58%. Compared with the new SPY/CASH timing benchmark, it preserves the stronger stress timing while materially lifting compounding through regime-specific hedge allocation.

![Final drawdown curve](results/main_pipeline_final/figures/final_drawdown_curve_comparison.png)

## Core Insight

**Stress triggers are regime-dependent, and hedge assets are also regime-dependent.**

VIX shock and credit widening do not have the same meaning in every regime. SPY, GOLD, IEF, CASH, and commodities do not have fixed roles across regimes either. The final framework therefore conditions both stress detection and hedge allocation on the macro regime. This is why the project is not a traditional all-weather portfolio or a generic timing model.

## Framework

```text
ML regime discovery
        ->
Rule-based macro regimes
        ->
Regime-specific VIX/CREDIT anchor stress detection
        ->
Regime x stress asset behavior
        ->
Regime-specific hedge allocation
        ->
Final SPY-centered index enhancement strategy
```

This project is not a traditional all-weather portfolio and not a generic market-timing system. It is a **SPY-centered regime-aware index enhancement strategy**. The goal is to keep SPY as the long-term return engine while using regime-conditioned stress timing and regime-specific hedge allocation to avoid the major drawdown windows.

## From ML Regime Discovery to Rule-Based Regimes

This project grew out of earlier machine-learning regime research:

[Market-Regime-Clustering](https://github.com/Snow-Ouyang/Market-Regime-Clustering)

The clustering / jump-model work was used as a discovery layer, not as a direct tradable state. It highlighted how macro variables naturally cluster, which regimes repeatedly host stress, and why different hedge assets behave differently across curve shape and rate-level states.

The final strategy converts those observations into rule-based macro regimes because rule-based regimes are easier to:

- interpret,
- audit,
- reproduce from source data,
- and maintain as a transparent research framework.

## Regime Framework

The final macro regimes are:

- `FLAT_LOW_RATE`
- `FLAT_HIGH_RATE`
- `STEEP_LOW_RATE`
- `STEEP_HIGH_RATE`
- `INVERTED`

The first layer uses:

`term_spread = GS10 - GS1`

| Regime | Rule | Threshold Source | Economic Interpretation | Strategy Implication |
|---|---|---|---|---|
| `INVERTED` | `term_spread < 0` | Yield-curve inversion | Tight-policy / late-cycle inversion | Keep a SPY/GOLD normal sleeve, but allow explicit stress states |
| Raw `FLAT` | `0 <= term_spread <= 1` | Flat curve band | Curve shape is flat but rate level still matters | Split by GS10 |
| `FLAT_LOW_RATE` | raw `FLAT` and `GS10 <= 3.0` | Rounded flat-rate split | Low-rate flat environments still allow equity participation | SPY / CMDTY_FUT normal sleeve |
| `FLAT_HIGH_RATE` | raw `FLAT` and `GS10 > 3.0` | Same GS10 split | High-rate flat environments favor real assets over SPY | GOLD / CMDTY_FUT normal sleeve |
| `STEEP_LOW_RATE` | raw `STEEP` and confirmed `GS1 <= 0.3` | Short-rate split within STEEP | Low short-rate steep curves remain equity-friendly | 100% SPY normal sleeve |
| `STEEP_HIGH_RATE` | raw `STEEP` and confirmed `GS1 > 0.3` | Same GS1 split with 3-day confirmation | Higher short-rate steep curves need broader diversification | SPY / GOLD / CMDTY_FUT inverse-vol normal sleeve |

The `GS10 = 3.0` threshold is the rounded flat-regime split from earlier diagnostics. The `GS1 = 0.3` threshold is used only inside confirmed `STEEP` because GS1 is a cleaner short-rate variable than GS10 for policy-rate level, cash yield, and financing pressure.

## Why This Is Not a Traditional All-Weather Portfolio

Traditional all-weather portfolios are usually static or semi-static allocations across macro quadrants.

This project is different:

- SPY remains the primary return engine.
- Stress detection is regime-conditioned.
- Stress is managed with a lock state machine rather than one-day exits.
- Hedge sleeves are regime-specific rather than universal.

This is a **SPY-centered regime-aware index enhancement strategy**. The goal is not to minimize volatility at all costs, but to preserve equity participation while reducing major regime-specific drawdowns.

Monthly timing was explored as an early benchmark, but it is not part of the final thesis. The final mainline now uses a cleaner VIX/CREDIT anchor state machine.

## Regime-Specific Trigger-Lock Stress System

The final strategy uses an anchor-style trigger-lock state machine.

Each trigger has:

- an enabled regime set,
- an entry condition,
- an unlock condition,
- and an economic interpretation.

When a trigger fires first, it becomes the anchor lock. If another lock is added during the same stress episode, the anchor still controls the main exit. This keeps the state machine interpretable and avoids overreacting to later secondary signals.

| Trigger | Enabled Regimes | Entry | Unlock | Economic Meaning | Main Purpose |
|---|---|---|---|---|---|
| VIX lock | `FLAT_LOW_RATE`, `FLAT_HIGH_RATE`, `INVERTED` | `VIX_ZSCORE_120D >= 3.0` | `VIX_ZSCORE_120D < 1.5` and `SPY > MA20` | Fast panic / volatility shock | Catch fast-risk episodes without needing a commodity trigger |
| Credit lock | `FLAT_LOW_RATE`, `FLAT_HIGH_RATE`, `STEEP_LOW_RATE`, `STEEP_HIGH_RATE`, `INVERTED` | `SPY_DD <= -5%`, `D_CREDIT_SPREAD_15D > 0.10`, and `SPY <= MA20` | `D_CREDIT_SPREAD_15D < 0`, `SPY > MA50`, and `CREDIT_LEVEL_Z_252D < 0.9` | Price-confirmed, still-elevated credit stress | Catch sustained and stair-step credit stress such as 2008 and 2022 |

Anchor exit logic:

- If stress started from VIX, VIX unlock is sufficient to exit stress.
- If stress started from credit, credit unlock is sufficient to exit stress.
- If both were active at entry, they unlock independently.

Commodity trigger is not part of the final mainline.

## Trigger-to-Unlock Episode Diagnostics

Because the final strategy is lock-based, trigger quality should be evaluated from trigger entry to unlock exit, not only with fixed forward-return windows.

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
- Removing the commodity trigger makes the state machine cleaner, while the cross-state asset evidence is still used in the hedge sleeve.

Relevant outputs:

- [stress_entry_attribution.csv](results/main_pipeline_final/tables/stress_entry_attribution.csv)
- [trigger_effectiveness_summary.csv](results/main_pipeline_final/tables/trigger_effectiveness_summary.csv)
- [trigger_regime_spy_timeline_long.png](results/main_pipeline_final/figures/trigger_regime_spy_timeline_long.png)

## Regime x Stress Asset Behavior

The final allocation is not assigned arbitrarily. It is derived from observed asset behavior under each regime-stress cross state.

Key findings under the new state machine:

- `FLAT_LOW_RATE_STRESS`: CASH is cleaner than GOLD under the new stress definition.
- `FLAT_HIGH_RATE_STRESS`: IEF remains the strongest stress asset.
- `STEEP_HIGH_RATE_NORMAL`: SPY, GOLD, and CMDTY_FUT are all strong; inverse-vol across the three gives broader participation.
- `STEEP_HIGH_RATE_STRESS`: IEF is clearly the best stress hedge, with a small CASH sleeve for path stability.
- `STEEP_LOW_RATE_STRESS`: SPY still has positive return, but IEF is also positive and materially stabilizes the sleeve, which motivates a mixed SPY/IEF hedge.
- `INVERTED_STRESS`: SPY and GOLD remain positive while IEF is weak, which supports a cash-plus-SPY/GOLD stress sleeve rather than an IEF-heavy hedge.

![Asset return heatmap](results/main_pipeline_final/figures/cross_state_asset_behavior_heatmap.png)

![Asset Sharpe heatmap](results/main_pipeline_final/figures/cross_state_asset_sharpe_heatmap.png)

## Final Strategy Allocation

| Macro Regime | State | Trigger Condition | Allocation | Rationale |
|---|---|---|---|---|
| `FLAT_LOW_RATE` | Normal | No active VIX or credit lock | SPY / CMDTY_FUT inverse-vol | Low-rate flat state still supports equity and commodity participation |
| `FLAT_LOW_RATE` | Stress | VIX or credit lock active | 100% CASH | New stress sample is cleaner for cash than gold |
| `FLAT_HIGH_RATE` | Normal | No active VIX or credit lock | GOLD / CMDTY_FUT inverse-vol | High-rate flat state favors real assets |
| `FLAT_HIGH_RATE` | Stress | VIX or credit lock active | 100% IEF | IEF is the strongest high-rate flat stress hedge |
| `STEEP_LOW_RATE` | Normal | No active credit lock | 100% SPY | Low short-rate steep curves remain equity-friendly |
| `STEEP_LOW_RATE` | Stress | Credit lock active | 60% SPY / 40% IEF | Stress is real, but SPY still participates; IEF lowers path risk |
| `STEEP_HIGH_RATE` | Normal | No active credit lock | SPY / GOLD / CMDTY_FUT inverse-vol | Higher short-rate steep curves benefit from broader diversification |
| `STEEP_HIGH_RATE` | Stress | Credit lock active | 10% CASH / 90% IEF | IEF is the dominant hedge, with a small cash stabilizer |
| `INVERTED` | Normal | No active VIX or credit lock | SPY / GOLD inverse-vol | Inversion is not automatically risk-off |
| `INVERTED` | Stress | VIX or credit lock active | 10% CASH + 90% (SPY / GOLD inverse-vol) | Stress still favors SPY/GOLD over IEF, but cash reduces path risk |

## Full Backtest Results

| Strategy | CAGR | Sharpe | Sortino | MaxDD | Calmar | Final Equity |
|---|---:|---:|---:|---:|---:|---:|
| SPY_BUY_HOLD | 11.14% | 0.575 | 0.702 | -55.19% | 0.202 | 8.38 |
| SPY_CASH_TIMING | 13.45% | 1.223 | 1.347 | -14.60% | 0.921 | 12.69 |
| FINAL_REGIME_HEDGE_TRIGGER_LOCK | 19.73% | 1.660 | 2.125 | -11.58% | 1.704 | 37.54 |

Compared with SPY buy-and-hold:

- CAGR improves from 11.14% to 19.73%.
- Sharpe improves from 0.575 to 1.660.
- MaxDD improves from -55.19% to -11.58%.
- Final equity improves from 8.38 to 37.54.

Compared with the new SPY/CASH timing benchmark:

- CAGR improves from 13.45% to 19.73%.
- Sharpe improves from 1.223 to 1.660.
- MaxDD improves from -14.60% to -11.58%.
- Final equity improves from 12.69 to 37.54.

![Final equity curve](results/main_pipeline_final/figures/final_equity_curve_comparison.png)

![Final drawdown curve](results/main_pipeline_final/figures/final_drawdown_curve_comparison.png)

![Final weight timeline](results/main_pipeline_final/figures/final_strategy_weights_timeline.png)

## Crisis Window Analysis

| Window | Main Stress Type | Final Strategy Result | Comment |
|---|---|---:|---|
| 2008 GFC | Sustained credit crisis | +39.08%, MaxDD -5.54% | New credit lock kept the strategy defensive through the deepest phase |
| 2015-2016 | Slow-growth / commodity stress | +4.88%, MaxDD -5.61% | Lower upside than older commodity-trigger versions, but still controlled |
| COVID 2020 | Fast volatility shock | +17.92%, MaxDD -11.09% | VIX handled the fast spike while the final hedge sleeve kept most of the rebound |
| 2022 rate / inflation / war shock | Stair-step elevated credit stress | +20.72%, MaxDD -10.29% | This is the main beneficiary of the new credit state machine |
| 2025 pullback | High-rate volatility stress | +19.24%, MaxDD -8.35% | Strong drawdown control without giving up much upside |

Case-study figures:

![2008 GFC case](results/main_pipeline_final/figures/case_2008_GFC_final.png)

![2015-2016 case](results/main_pipeline_final/figures/case_2015_2016_final.png)

![2022 case](results/main_pipeline_final/figures/case_2022_rate_war_final.png)

![2025 case](results/main_pipeline_final/figures/case_2025_pullback_final.png)

## Methodology Notes

- The mainline is source-only: it uses `data/raw` and `data/processed`, not exploratory result folders.
- Signals at day `t` affect positions at day `t+1`.
- Regime confirmation uses 3 consecutive days.
- FLAT low/high uses `GS10 = 3.0`.
- STEEP low/high uses `GS1 = 0.3` with 3-day confirmation.
- The final regime universe has no `NEUTRAL`.
- Inverse-volatility uses a 90 trading day window.
- Transaction cost is 10 bps one-way.
- CASH uses compounded daily DTB3.
- Credit spread is daily `DBAA - DAAA`, filled to the trading calendar before feature construction.
- Credit unlock uses `CREDIT_LEVEL_Z_252D < 0.9`, so the spread level must normalize before a credit-led stress exit.
- Commodity trigger is not used in the final mainline.

Main run order:

```bash
python scripts/run_final_strategy_source_only.py
python scripts/08_stress_trigger_diagnostics.py
python scripts/10_final_report_outputs.py
python scripts/hard_validate_main_pipeline_source_only.py
```

## Limitations

- Stress events are sparse and heterogeneous.
- Daily credit data still requires careful handling of missing values and publication differences.
- The anchor-exit state machine is economically motivated, but still tuned in-sample.
- `STEEP_LOW_RATE_STRESS` and `INVERTED_STRESS` are newer states and need more OOS evidence.
- The strategy remains SPY-centered, not minimum-volatility.
- Regime thresholds are economically motivated but still simplified.
- This is not financial advice.

## Final Interpretation

The main contribution is not a single optimized trigger. The main contribution is a regime-conditioned framework showing that both stress detection and hedge allocation depend on the macro regime.

The current mainline now reflects that idea more cleanly than the previous version:

- stress timing is handled by a simpler VIX/CREDIT anchor state machine,
- `SPY_CASH_TIMING` is materially stronger than before,
- and the final hedge strategy preserves that timing improvement while lowering drawdown further through regime-specific sleeves.
