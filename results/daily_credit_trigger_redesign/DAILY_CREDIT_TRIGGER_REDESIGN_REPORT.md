# DAILY_CREDIT_TRIGGER_REDESIGN_REPORT

## 1. Purpose

Redesign daily credit trigger logic after moving from weekly forward-filled credit data to daily DAAA/DBAA credit series.

## 2. Observed problems from visualization

- 2008: early unlock and weak persistence through the sustained credit spike.
- 2020: fast spike followed by fast relief, so overly strict unlock risks missing the rebound.
- 2022: stair-step elevated credit stress with repeated local improvements and relapses.

## 3. Daily credit features

We compare shock changes, level z-score, percentile, moving-average trend, and peak-relief features.

## 4. Credit variants

Variants range from simple daily baseline to watch-state state machines and fast-relock hybrids.

## 5. SPY/CASH laboratory results

- Best SPY/CASH variant: `LEVEL_OR_PERCENTILE_LOCK`

## 6. Case study analysis

Use 2008, 2020, 2022, and 2025 case charts to judge persistence vs fast relief behavior.

## 7. Final strategy challenger results

- Best final challenger by Sharpe: `WATCH_AS_PARTIAL_LOCK_DIAGNOSTIC`

## 8. Recommendation

Keep baseline daily credit. No final challenger is materially better.

## 9. Limitations

- daily credit data availability and revisions
- in-sample parameter risk
- credit behavior differs across crises