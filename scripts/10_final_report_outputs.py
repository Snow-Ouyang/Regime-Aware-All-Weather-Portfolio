from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from final_strategy_source_only_core import ASSETS, FINAL_STRATEGY, REFINED_BASELINE, ROOT, SPY_BUY_HOLD, SPY_CASH_TIMING, build_final_source_only_panel


OUT = ROOT / "results" / "main_pipeline_final"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
FINAL = FINAL_STRATEGY
BASE = REFINED_BASELINE
SPY = SPY_BUY_HOLD
DISPLAY_STRATEGIES = [SPY_BUY_HOLD, SPY_CASH_TIMING, FINAL_STRATEGY]


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def asset_behavior(panel: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group, sub in panel.groupby(group_col, dropna=False):
        for asset in ASSETS:
            r = sub[f"{asset}_return"].fillna(0.0)
            excess = r - sub["CASH_return"].fillna(0.0)
            nav = (1.0 + r).cumprod()
            ann_ret = nav.iloc[-1] ** (252.0 / len(r)) - 1.0
            vol = r.std() * (252.0 ** 0.5)
            excess_nav = (1.0 + excess).cumprod()
            ann_excess_ret = excess_nav.iloc[-1] ** (252.0 / len(excess)) - 1.0
            sharpe = 0.0 if asset == "CASH" else (ann_excess_ret / vol if vol > 0 else None)
            rows.append(
                {
                    group_col: group,
                    "asset": asset,
                    "n_obs": len(r),
                    "annualized_return": ann_ret,
                    "annualized_volatility": vol,
                    "annualized_excess_return_vs_cash": ann_excess_ret,
                    "Sharpe": sharpe,
                    "max_drawdown": (nav / nav.cummax() - 1.0).min(),
                    "cumulative_return": nav.iloc[-1] - 1.0,
                }
            )
    return pd.DataFrame(rows)


def plot_asset_behavior_heatmap(
    perf: pd.DataFrame,
    group_col: str,
    path_name: str,
    title: str,
    value_col: str = "annualized_return",
    value_label: str = "Annualized return",
    value_format: str = "percent",
) -> None:
    heat = perf.pivot_table(index="asset", columns=group_col, values=value_col, aggfunc="first")
    heat = heat.reindex(index=ASSETS)
    if value_format == "percent":
        labels = heat.map(lambda x: "" if pd.isna(x) else f"{x:.1%}")
    else:
        labels = heat.map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    fig_w = max(11, 1.15 * len(heat.columns))
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    sns.heatmap(
        heat,
        annot=labels,
        fmt="",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": value_label},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(FIG_DIR / path_name, dpi=170)
    plt.close(fig)


def crisis_performance(panel: pd.DataFrame) -> pd.DataFrame:
    windows = {
        "2008_GFC": ("2007-10-01", "2009-06-30"),
        "2015_2016": ("2015-05-01", "2016-03-31"),
        "COVID_2020": ("2020-02-01", "2020-06-30"),
        "2022_RATE_WAR": ("2021-11-01", "2023-03-31"),
        "2025_PULLBACK": ("2025-01-01", "2025-12-31"),
    }
    rows = []
    for name, (start, end) in windows.items():
        sub = panel.loc[panel["date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()
        if sub.empty:
            continue
        for strategy in DISPLAY_STRATEGIES:
            r = sub[f"{strategy}_return"].fillna(0.0)
            nav = (1.0 + r).cumprod()
            rows.append(
                {
                    "window": name,
                    "strategy": strategy,
                    "cumulative_return": nav.iloc[-1] - 1.0,
                    "max_drawdown": (nav / nav.cummax() - 1.0).min(),
                    "volatility": r.std() * (252.0 ** 0.5),
                }
            )
    return pd.DataFrame(rows)


CASE_WINDOWS = {
    "case_2008_GFC_final.png": ("2007-10-01", "2009-06-30"),
    "case_2015_2016_final.png": ("2015-05-01", "2016-03-31"),
    "case_2022_rate_war_final.png": ("2021-11-01", "2023-03-31"),
    "case_2025_pullback_final.png": ("2025-01-01", "2025-12-31"),
}


def plot_outputs(panel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for strategy in DISPLAY_STRATEGIES:
        ax.plot(panel["date"], panel[f"{strategy}_nav"], label=strategy)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "final_equity_curve_comparison.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for strategy in DISPLAY_STRATEGIES:
        ax.plot(panel["date"], panel[f"{strategy}_drawdown"], label=strategy)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "final_drawdown_curve_comparison.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for strategy in DISPLAY_STRATEGIES:
        nav = panel[f"{strategy}_nav"]
        ax.plot(panel["date"], nav / nav.shift(252) - 1.0, label=strategy)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.legend()
    ax.grid(alpha=0.25)
    ax.set_title("Rolling 12M Return Comparison")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rolling_12m_return_comparison.png", dpi=170)
    plt.close(fig)

    weight_cols = [f"{FINAL}_weight_{asset}" for asset in ASSETS]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.stackplot(panel["date"], *[panel[col] for col in weight_cols], labels=ASSETS, alpha=0.9)
    ax.set_ylim(0, 1)
    ax.set_title(f"{FINAL} Weight Timeline")
    ax.legend(ncol=len(ASSETS), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "final_strategy_weights_timeline.png", dpi=170)
    plt.close(fig)


def plot_performance_bars(perf: pd.DataFrame) -> None:
    metrics = [
        ("CAGR", "CAGR"),
        ("Sharpe", "Sharpe"),
        ("Sortino", "Sortino"),
        ("MaxDD", "MaxDD"),
        ("Calmar", "Calmar"),
        ("final_equity", "Final Equity"),
        ("turnover", "Turnover"),
    ]
    plot_df = perf.loc[perf["strategy"].isin(DISPLAY_STRATEGIES)].copy()
    plot_df["strategy"] = pd.Categorical(plot_df["strategy"], categories=DISPLAY_STRATEGIES, ordered=True)
    plot_df = plot_df.sort_values("strategy")
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    axes = axes.flatten()
    for ax, (col, title) in zip(axes, metrics):
        vals = plot_df[col].astype(float)
        colors = ["#4C78A8", "#F58518", "#54A24B"]
        ax.bar(plot_df["strategy"].astype(str), vals, color=colors[: len(plot_df)])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=25)
        if col in {"CAGR", "MaxDD"}:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        for idx, value in enumerate(vals):
            label = f"{value:.1%}" if col in {"CAGR", "MaxDD"} else f"{value:.2f}"
            ax.text(idx, value, label, ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    axes[-1].axis("off")
    fig.suptitle("Final Strategy Performance Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "final_performance_bar_charts.png", dpi=170)
    plt.close(fig)


def plot_case_studies(panel: pd.DataFrame) -> None:
    for filename, (start, end) in CASE_WINDOWS.items():
        sub = panel.loc[panel["date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()
        if sub.empty:
            continue
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        for strategy in DISPLAY_STRATEGIES:
            local_nav = sub[f"{strategy}_nav"] / sub[f"{strategy}_nav"].iloc[0]
            axes[0].plot(sub["date"], local_nav, label=strategy)
        axes[0].set_title(f"{filename.replace('_', ' ').replace('.png', '')}: NAV, normalized to window start")
        axes[0].grid(alpha=0.25)
        axes[0].legend(loc="best")

        cash_nav = sub[f"{SPY_CASH_TIMING}_nav"] / sub[f"{SPY_CASH_TIMING}_nav"].iloc[0]
        for strategy in [SPY_BUY_HOLD, FINAL_STRATEGY]:
            local_nav = sub[f"{strategy}_nav"] / sub[f"{strategy}_nav"].iloc[0]
            axes[1].plot(sub["date"], local_nav / cash_nav - 1.0, label=f"{strategy} vs {SPY_CASH_TIMING}")
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].set_title(f"Relative performance versus {SPY_CASH_TIMING}")
        axes[1].grid(alpha=0.25)
        axes[1].legend(loc="best")

        for strategy in DISPLAY_STRATEGIES:
            local_nav = sub[f"{strategy}_nav"] / sub[f"{strategy}_nav"].iloc[0]
            dd = local_nav / local_nav.cummax() - 1.0
            axes[2].plot(sub["date"], dd, label=strategy)
        axes[2].set_title("Window drawdown")
        axes[2].grid(alpha=0.25)
        axes[2].legend(loc="best")

        fig.tight_layout()
        fig.savefig(FIG_DIR / filename, dpi=170)
        plt.close(fig)


def stress_trigger_readme_section() -> str:
    return f"""

## Stress Trigger and Turnover Diagnostics

The canonical final strategy uses a trigger-lock state machine. This replaced the prior FLAT_LOW recovery overlay because it is lower-turnover, more tradable, and easier to explain.

### Trigger Rules Summary

- `FLAT_LOW_RATE` / `FLAT_HIGH_RATE` / `INVERTED`: VIX trigger is active.
- `FLAT_LOW_RATE` / `FLAT_HIGH_RATE` / `STEEP_LOW_RATE` / `STEEP_HIGH_RATE` / `INVERTED`: credit trigger is active.
- Commodity trigger is not part of the final mainline.
- Monthly SELL is not part of the final state machine.
- Credit entry uses `D_CREDIT_SPREAD_15D > 0.10`, `SPY drawdown <= -5%`, and `SPY <= MA20`.
- Credit unlock uses `D_CREDIT_SPREAD_15D < 0`, `SPY > MA50`, and `CREDIT_LEVEL_Z_252D < 0.9`.
- VIX unlock uses `VIX_ZSCORE_120D < 1.5` with `SPY > MA20`.
- The anchor-exit rule means VIX-led stress exits on VIX unlock, credit-led stress exits on credit unlock, and BOTH entries unlock independently.

### Key Findings

- The final comparison now uses the same trigger-lock stress state for both `SPY_CASH_TIMING` and `FINAL_REGIME_HEDGE_TRIGGER_LOCK`.
- Cross-state asset behavior is also grouped with the trigger-lock stress state, so the asset evidence and final strategy share the same stress definition.
- Recovery overlay diagnostics are kept as exploratory history, but they are not part of the final mainline.

### Implication

Future research can still study state-machine refinements, but the current mainline is intentionally converged around the VIX/CREDIT anchor state machine without recovery overlay or commodity-trigger complexity.
"""


def main() -> None:
    ensure_dirs()
    panel, perf = build_final_source_only_panel()
    stress_flag = panel["final_state"].eq("FULL_RISK")
    panel["allocation_cross_state"] = panel["final_allocation_state"]
    panel["final_regime_cross_state"] = panel["final_regime_confirmed"] + "_" + stress_flag.map(
        {True: "STRESS", False: "NORMAL"}
    )
    panel.to_csv(OUT / "daily_backtest_panel.csv", index=False)
    panel.to_csv(TABLE_DIR / "daily_backtest_panel.csv", index=False)
    display_perf = perf.loc[perf["strategy"].isin(DISPLAY_STRATEGIES)].copy()
    display_perf.to_csv(TABLE_DIR / "strategy_performance_comparison.csv", index=False)
    cross_behavior = asset_behavior(panel, "allocation_cross_state")
    flat_behavior = asset_behavior(panel, "final_regime_confirmed")
    cross_behavior.to_csv(TABLE_DIR / "cross_state_asset_behavior.csv", index=False)
    flat_behavior.to_csv(TABLE_DIR / "flat_low_high_asset_behavior.csv", index=False)
    crisis_performance(panel).to_csv(TABLE_DIR / "crisis_window_performance.csv", index=False)
    panel[["date"] + [f"{FINAL}_weight_{asset}" for asset in ASSETS]].to_csv(TABLE_DIR / "final_daily_weights.csv", index=False)
    panel[
        [
            "date",
            f"{FINAL}_return",
            f"{FINAL}_nav",
            f"{FINAL}_drawdown",
            f"{FINAL}_turnover",
            f"{FINAL}_transaction_cost",
            "final_state",
            "final_allocation_state",
            "trigger_lock_active_locks",
            "final_regime_confirmed",
            "steep_rate_regime_confirmed",
        ]
    ].to_csv(TABLE_DIR / "final_daily_returns.csv", index=False)
    plot_outputs(panel)
    plot_performance_bars(display_perf)
    plot_case_studies(panel)
    plot_asset_behavior_heatmap(
        cross_behavior,
        "allocation_cross_state",
        "cross_state_asset_behavior_heatmap.png",
        "Asset Annualized Return by Final Allocation State",
    )
    plot_asset_behavior_heatmap(
        cross_behavior,
        "allocation_cross_state",
        "cross_state_asset_sharpe_heatmap.png",
        "Asset Sharpe Ratio by Final Allocation State",
        value_col="Sharpe",
        value_label="Sharpe ratio",
        value_format="number",
    )
    plot_asset_behavior_heatmap(
        flat_behavior,
        "final_regime_confirmed",
        "flat_low_high_asset_behavior_heatmap.png",
        "Asset Annualized Return by Final Regime",
    )
    plot_asset_behavior_heatmap(
        flat_behavior,
        "final_regime_confirmed",
        "flat_low_high_asset_sharpe_heatmap.png",
        "Asset Sharpe Ratio by Final Regime",
        value_col="Sharpe",
        value_label="Sharpe ratio",
        value_format="number",
    )
    readme = f"""# Final Source-Only Strategy Outputs

This folder was generated from `data/raw` and `data/processed` only using the
canonical source-only settings.

Final display strategies:

- `SPY_BUY_HOLD`: always 100% SPY.
- `SPY_CASH_TIMING`: SPY in non-risk, CASH in full-risk; uses the same VIX/CREDIT anchor stress state as the final hedge strategy.
- `FINAL_REGIME_HEDGE_TRIGGER_LOCK`: final hedge allocation with inverse-vol normal allocations and regime-specific trigger-lock stress hedges.

Key design choices:
- Credit spread is daily `DBAA - DAAA`, filled to the trading calendar before feature construction.
- Macro regime has no `NEUTRAL`: term spread maps every day to `INVERTED`,
  `FLAT`, or `STEEP`, then uses 3-day confirmation.
- FLAT is refined with GS10 threshold 3.0 into `FLAT_LOW_RATE` and
  `FLAT_HIGH_RATE`.
- STEEP normal is refined with GS1 threshold 0.3 into `STEEP_LOW_RATE`
  and `STEEP_HIGH_RATE`; the low/high switch also uses 3-day confirmation.
- `CASH_return` uses geometric daily DTB3.
- `CMDTY_RET60` uses synthetic commodity price from `CMDTY_FUT_return`.
- `VIX_ZSCORE_120D` uses 120 trading days, current-day inclusive, `ddof=1`.
- Inverse-vol window grid search showed limited sensitivity across reasonable settings; the final mainline uses 90 trading days.
- Transaction cost uses 10 bps one-way.
- Recovery overlay exploration is not part of the final mainline.

Final allocation settings:
- `FLAT_LOW_RATE_NORMAL`: SPY / CMDTY_FUT inverse-vol.
- `FLAT_LOW_RATE_STRESS`: 100% CASH.
- `FLAT_HIGH_RATE_NORMAL`: GOLD / CMDTY_FUT inverse-vol.
- `FLAT_HIGH_RATE_STRESS`: 100% IEF.
- `STEEP_LOW_RATE_NORMAL`: 100% SPY.
- `STEEP_LOW_RATE_STRESS`: 60% SPY / 40% IEF.
- `STEEP_HIGH_RATE_NORMAL`: SPY / GOLD / CMDTY_FUT inverse-vol.
- `STEEP_HIGH_RATE_STRESS`: 10% CASH / 90% IEF.
- `INVERTED_NORMAL`: SPY / GOLD inverse-vol.
- `INVERTED_STRESS`: 10% CASH + 90% (SPY / GOLD inverse-vol).

How to run the current main sequence:

1. `python scripts/01_data_prepare.py`
2. `python scripts/02_rule_based_regime.py`
3. `python scripts/03_stress_detection.py`
4. `python scripts/04_asset_return_panel.py`
5. `python scripts/05_baseline_strategy.py`
6. `python scripts/06_flat_rate_refined_strategy.py`
7. `python scripts/07_cross_state_asset_behavior.py`
8. `python scripts/08_stress_trigger_diagnostics.py`
9. `python scripts/run_final_strategy_source_only.py`
10. `python scripts/10_final_report_outputs.py`
"""
    readme += stress_trigger_readme_section()
    (OUT / "README_final_strategy.md").write_text(readme, encoding="utf-8")
    print("PASS source-only final report outputs")
    print(display_perf.to_string(index=False))


if __name__ == "__main__":
    main()
