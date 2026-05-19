# Risk Factors Panel Cleanup

Generated: 2026-05-18T18:35:29

Decision:
- `data/processed/risk_factors` panels are not used by the source-only final mainline.
- References remain only in exploratory factor and sleeve analysis scripts under `src/analysis` or the panel builder itself.

Deleted panel files:
- data/processed/risk_factors/core_risk_factor_panel.csv
- data/processed/risk_factors/extended_risk_factor_panel.csv
- data/processed/risk_factors/long_history_risk_factor_panel.csv

Yahoo downloader:
- Default universe is now restricted to final required tickers: SPY, GLD, GD=F, IEF.
- Optional `YAHOO_TICKERS` override is still supported if explicitly set.
