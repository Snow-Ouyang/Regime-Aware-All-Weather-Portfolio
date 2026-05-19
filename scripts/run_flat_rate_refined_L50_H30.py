"""Materialize the fixed FLAT_RATE_REFINED_L50_H30 experiment.

This is an independent output copy of the best balanced FLAT stress ratio
candidate. It does not modify the main strategy or grid-search results.
"""

from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GRID_DIR = ROOT / "results" / "flat_stress_ratio_grid_search"
SOURCE_DAILY = GRID_DIR / "daily_candidate_panels" / "L50_H30_daily.csv"
SOURCE_TABLE = GRID_DIR / "tables" / "grid_search_all_results.csv"
OUT = ROOT / "results" / "flat_rate_refined_L50_H30"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"


def ensure_source_exists() -> None:
    if SOURCE_DAILY.exists() and SOURCE_TABLE.exists():
        return
    runpy.run_path(str(ROOT / "scripts" / "grid_search_flat_stress_ratio.py"), run_name="__main__")
    if not SOURCE_DAILY.exists():
        raise FileNotFoundError(f"Could not generate source candidate daily panel: {SOURCE_DAILY}")


def main() -> None:
    ensure_source_exists()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    import matplotlib.pyplot as plt

    daily = pd.read_csv(SOURCE_DAILY, parse_dates=["date"])
    grid = pd.read_csv(SOURCE_TABLE)
    row = grid.loc[grid["candidate"].eq("L50_H30")].iloc[0]
    base_summary = pd.read_csv(GRID_DIR / "tables" / "baseline_summaries.csv")

    # Rename candidate columns to fixed strategy name.
    name = "FLAT_RATE_REFINED_L50_H30"
    daily = daily.rename(columns={c: c.replace("L50_H30", name) for c in daily.columns})
    daily.to_csv(TABLE_DIR / "daily_returns.csv", index=False)
    weight_cols = ["date", "macro_regime_confirmed", "timing_state", "flat_refined_state"] + [
        f"{name}_weight_{a}" for a in ["SPY", "GOLD", "CMDTY_FUT", "IEF", "CASH"]
    ]
    daily[weight_cols].to_csv(TABLE_DIR / "daily_weights.csv", index=False)

    perf = pd.concat(
        [
            base_summary,
            pd.DataFrame(
                [
                    {
                        "strategy": name,
                        "CAGR": row["CAGR"],
                        "annualized_volatility": row["annualized_volatility"],
                        "Sharpe": row["Sharpe"],
                        "Sortino": row["Sortino"],
                        "MaxDD": row["MaxDD"],
                        "Calmar": row["Calmar"],
                        "final_equity": row["final_equity"],
                        "turnover": row["turnover"],
                        "total_transaction_cost": row["total_transaction_cost"],
                        "worst_12m_return": row["worst_12m_return"],
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    perf.to_csv(TABLE_DIR / "performance_summary.csv", index=False)
    comp = perf.set_index("strategy").T
    if name in comp.columns and "MATURE_REGIME_HEDGE_FINAL" in comp.columns:
        comp["delta_L50_H30_minus_final"] = pd.to_numeric(comp[name], errors="coerce") - pd.to_numeric(
            comp["MATURE_REGIME_HEDGE_FINAL"], errors="coerce"
        )
    comp.reset_index(names="metric").to_csv(TABLE_DIR / "comparison_vs_baseline.csv", index=False)

    final_panel = pd.read_csv(ROOT / "results" / "09_final_strategy" / "mature_regime_hedge_final" / "daily_backtest_panel.csv", parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(final_panel["date"], final_panel["MATURE_REGIME_HEDGE_FINAL_nav"], label="MATURE_REGIME_HEDGE_FINAL")
    ax.plot(daily["date"], daily[f"{name}_nav"], label=name)
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title("FLAT_RATE_REFINED_L50_H30 vs baseline")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "equity_curve_vs_baseline.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(final_panel["date"], final_panel["MATURE_REGIME_HEDGE_FINAL_drawdown"], label="MATURE_REGIME_HEDGE_FINAL")
    ax.plot(daily["date"], daily[f"{name}_drawdown"], label=name)
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title("FLAT_RATE_REFINED_L50_H30 drawdown vs baseline")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "drawdown_curve_vs_baseline.png", dpi=170)
    plt.close(fig)

    readme = f"""# FLAT_RATE_REFINED_L50_H30

## Purpose

This standalone output fixes the refined FLAT baseline at the grid-search candidate `L50_H30`. It does not modify the original baseline, the final strategy, or the grid-search outputs.

## Fixed Stress Weights

- `FLAT_LOW_RATE_STRESS`: 50% GOLD / 50% IEF
- `FLAT_HIGH_RATE_STRESS`: 30% GOLD / 70% CASH

## Rules Kept Unchanged

- Non-FLAT regimes follow `MATURE_REGIME_HEDGE_FINAL`.
- `FLAT_LOW_RATE_NORMAL`: SPY / CMDTY_FUT / GOLD inverse volatility.
- `FLAT_HIGH_RATE_NORMAL`: CMDTY_FUT / GOLD inverse volatility.
- GS10 threshold remains 2.9.
- Existing confirmation/smoothing, transaction cost, rebalance behavior, and 120-day inverse-volatility settings are unchanged.

## Performance

| Metric | Value |
|---|---:|
| CAGR | {row['CAGR']:.2%} |
| Annualized volatility | {row['annualized_volatility']:.2%} |
| Sharpe | {row['Sharpe']:.3f} |
| Sortino | {row['Sortino']:.3f} |
| MaxDD | {row['MaxDD']:.2%} |
| Calmar | {row['Calmar']:.3f} |
| Final equity | {row['final_equity']:.3f} |
| Turnover | {row['turnover']:.3f} |
| Total transaction cost | {row['total_transaction_cost']:.3f} |
| Worst 12M return | {row['worst_12m_return']:.2%} |

## Outputs

- `tables/daily_weights.csv`
- `tables/daily_returns.csv`
- `tables/performance_summary.csv`
- `tables/comparison_vs_baseline.csv`
- `figures/equity_curve_vs_baseline.png`
- `figures/drawdown_curve_vs_baseline.png`
"""
    (OUT / "README_flat_rate_refined_L50_H30.md").write_text(readme, encoding="utf-8")
    print("FLAT_RATE_REFINED_L50_H30 materialized.")
    print(f"output_dir: {OUT.relative_to(ROOT).as_posix()}")
    print(perf.tail(1).to_string(index=False))


if __name__ == "__main__":
    main()
