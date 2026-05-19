# Data Directory

This folder stores raw and processed input data used by the research pipeline.

- `data/raw/` contains source market, ETF, and macro files. These files are not cleaned or removed by the project organization step.
- `data/processed/` can contain aligned panels and derived features.

The final strategy backtest is built from validated project panels under `results/`, with raw data used as a fallback when a required return or macro series is missing.

Important implementation notes:

- ETF returns are aligned to SPY trading dates.
- Cash return uses the available `CASH_return` or `daily_rf` series.
- Regime data is expected to contain only `FLAT`, `STEEP`, and `INVERTED` in the final strategy.
- Low-frequency macro data may be forward-filled to daily trading dates, but returns are not forward-filled.
