# Source-Only After-Fix Comparison

This comparison uses the new clean source-only canonical settings. The old
reference panel is used only as a historical comparison target, not as the
definition of correctness.

## Canonical Intentional Differences

- `CREDIT_SPREAD_BAA_AAA` now comes directly from raw `WBAA - WAAA`, aligned to
  trading days and forward-filled. Old processed/intermediate credit panels under
  `results/` are intentionally not used.
- `macro_regime_confirmed` no longer allows `NEUTRAL`; it is confirmed from the
  term-spread regimes `INVERTED`, `FLAT`, and `STEEP` with 3-day confirmation
  initialized from the first raw regime.
- FLAT can be split into `FLAT_LOW_RATE` and `FLAT_HIGH_RATE` for allocation via
  the canonical GS10 threshold of 2.9.

## Bug Fixes / Canonical Formula Fixes

- `CASH_return` uses geometric daily RF: `(1 + DTB3 / 100) ** (1 / 252) - 1`.
- `CMDTY_RET60` uses synthetic commodity price from `CMDTY_FUT_return`, not raw
  `GD=F` adjusted close pct_change.
- `VIX_ZSCORE_120D` uses 120 trading days, includes the current day, uses
  `ddof=1`, and is calculated from the full VIX history before sample slicing.

## Remaining Mismatches

Remaining mismatches versus the old reference are expected when caused by the
intentional canonical changes above. Review `trigger_diff_after_fix.csv` and
`source_only_vs_reference_metrics_after_fix.csv` to determine whether they change
signals, allocations, or NAV relative to the old panel.
