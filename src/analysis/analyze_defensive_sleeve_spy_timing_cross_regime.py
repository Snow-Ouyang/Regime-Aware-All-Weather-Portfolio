from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
REGIME_PANEL_PATH = ROOT / "results" / "reconstructed_regime_asset_behavior" / "reconstructed_regime_panel.csv"
TIMING_PANEL_PATH = ROOT / "results" / "spy_cash_timing_frequency_test" / "daily_backtest_panel.csv"

RESULTS_DIR = ROOT / "results" / "defensive_sleeve_spy_timing_cross_regime"
FIGURES_DIR = ROOT / "figures" / "defensive_sleeve_spy_timing_cross_regime"

PANEL_PATH = RESULTS_DIR / "defensive_spy_timing_cross_state_panel.csv"
SAMPLE_SIZE_PATH = RESULTS_DIR / "cross_state_sample_size_table.csv"
PERFORMANCE_PATH = RESULTS_DIR / "asset_performance_by_macro_timing_cross_state.csv"
ANN_RETURN_PATH = RESULTS_DIR / "annualized_return_by_cross_state.csv"
SHARPE_PATH = RESULTS_DIR / "sharpe_by_cross_state.csv"
MAX_DD_PATH = RESULTS_DIR / "max_drawdown_by_cross_state.csv"
POSITIVE_PATH = RESULTS_DIR / "positive_day_ratio_by_cross_state.csv"
SELL_PATH = RESULTS_DIR / "defensive_assets_when_spy_timing_sell_by_macro_regime.csv"
HOLD_PATH = RESULTS_DIR / "defensive_assets_when_spy_timing_hold_by_macro_regime.csv"
RANKING_PATH = RESULTS_DIR / "defensive_asset_ranking_by_cross_state.csv"
SCORE_PATH = RESULTS_DIR / "defensive_asset_composite_score_by_cross_state.csv"
REPORT_PATH = RESULTS_DIR / "DEFENSIVE_SLEEVE_SPY_TIMING_CROSS_REGIME_ANALYSIS.md"

FIG_RETURN = FIGURES_DIR / "asset_return_heatmap_cross_state.png"
FIG_SHARPE = FIGURES_DIR / "asset_sharpe_heatmap_cross_state.png"
FIG_DRAWDOWN = FIGURES_DIR / "asset_drawdown_heatmap_cross_state.png"
FIG_SELL_RETURN = FIGURES_DIR / "sell_only_defensive_return_heatmap.png"
FIG_SELL_DRAWDOWN = FIGURES_DIR / "sell_only_defensive_drawdown_heatmap.png"
FIG_HOLD_RETURN = FIGURES_DIR / "hold_only_defensive_return_heatmap.png"
FIG_HOLD_HEDGE = FIGURES_DIR / "hold_only_defensive_hedge_heatmap.png"
FIG_SCORE = FIGURES_DIR / "defensive_composite_score_heatmap.png"
FIG_SAMPLE = FIGURES_DIR / "cross_state_sample_size_heatmap.png"

MACRO_ORDER = ["HIGH_INFLATION", "INVERTED", "FLAT", "STEEP", "NEUTRAL"]
TIMING_ORDER = ["HOLD", "SELL"]
DEFENSIVE_ASSETS = ["CASH", "IEF", "GOLD", "CMDTY_FUT"]
ALL_ASSETS = ["CASH", "IEF", "GOLD", "CMDTY_FUT", "SPY"]
RETURN_COLS = {
    "CASH": "CASH_RETURN",
    "IEF": "IEF_RETURN",
    "GOLD": "GOLD_RETURN",
    "CMDTY_FUT": "CMDTY_FUT_RETURN",
    "SPY": "SPY_RETURN",
}


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def annualized_return(daily_returns: pd.Series) -> float:
    s = daily_returns.dropna()
    if s.empty:
        return np.nan
    return float((1.0 + s).prod() ** (252.0 / len(s)) - 1.0)


def max_drawdown(daily_returns: pd.Series) -> float:
    s = daily_returns.dropna()
    if s.empty:
        return np.nan
    wealth = (1.0 + s).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def perf_stats(daily_returns: pd.Series, rf_daily: pd.Series, asset: str) -> dict[str, float]:
    s = daily_returns.dropna()
    if s.empty:
        return {
            "n_obs": 0,
            "average_daily_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "Sharpe": np.nan,
            "max_drawdown": np.nan,
            "worst_day": np.nan,
            "best_day": np.nan,
            "positive_day_ratio": np.nan,
            "cumulative_return_within_state": np.nan,
        }
    rf = rf_daily.loc[s.index]
    excess = s - rf
    ann_ret = annualized_return(s)
    ann_vol = float(s.std(ddof=1) * np.sqrt(252.0)) if len(s) > 1 else np.nan
    if asset == "CASH":
        sharpe = 0.0
    else:
        excess_std = excess.std(ddof=1)
        sharpe = float(excess.mean() / excess_std * np.sqrt(252.0)) if pd.notna(excess_std) and excess_std != 0 else np.nan
    return {
        "n_obs": int(len(s)),
        "average_daily_return": float(s.mean()),
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "Sharpe": sharpe,
        "max_drawdown": max_drawdown(s),
        "worst_day": float(s.min()),
        "best_day": float(s.max()),
        "positive_day_ratio": float((s > 0).mean()),
        "cumulative_return_within_state": float((1.0 + s).prod() - 1.0),
    }


def assign_spell_ids(panel: pd.DataFrame, group_cols: list[str], active_mask: pd.Series | None = None) -> pd.DataFrame:
    out = panel.sort_values("date").reset_index(drop=True).copy()
    active = active_mask.reindex(out.index).fillna(False) if active_mask is not None else pd.Series(True, index=out.index)
    spell_ids: list[float] = []
    current_spell = 0
    prev_active = False
    prev_key = None
    for i, row in out.iterrows():
        is_active = bool(active.iloc[i])
        if not is_active:
            spell_ids.append(np.nan)
            prev_active = False
            prev_key = None
            continue
        key = tuple(row[col] for col in group_cols)
        if (not prev_active) or (key != prev_key):
            current_spell += 1
        spell_ids.append(float(current_spell))
        prev_active = True
        prev_key = key
    out["spell_id"] = spell_ids
    return out


def spell_level_asset_metrics(panel: pd.DataFrame, group_cols: list[str], assets: list[str], active_mask: pd.Series | None = None) -> pd.DataFrame:
    spell_panel = assign_spell_ids(panel, group_cols=group_cols, active_mask=active_mask).dropna(subset=["spell_id"]).copy()
    rows = []
    for spell_id, grp in spell_panel.groupby("spell_id", observed=False):
        key_dict = {col: grp.iloc[0][col] for col in group_cols}
        for asset in assets:
            ret_col = RETURN_COLS[asset]
            stats = perf_stats(grp[ret_col], grp["RF_DAILY"], asset)
            spysub = grp[[ret_col, "SPY_RETURN"]].dropna()
            corr = np.nan if asset == "SPY" else (float(spysub.corr().iloc[0, 1]) if len(spysub) >= 3 else np.nan)
            neg_spy = grp.loc[grp["SPY_RETURN"] < 0, ret_col].dropna()
            rows.append(
                {
                    "spell_id": int(spell_id),
                    **key_dict,
                    "asset": asset,
                    "asset_role": "defensive" if asset in DEFENSIVE_ASSETS else "risk asset reference",
                    "spell_n_obs": stats["n_obs"],
                    **stats,
                    "correlation_with_SPY": corr,
                    "average_daily_return_when_SPY_return_negative": float(neg_spy.mean()) if not neg_spy.empty else np.nan,
                    "average_daily_return_when_SPY_drawdown_below_10pct": float(grp.loc[grp["SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"] < -0.10, ret_col].dropna().mean())
                    if "SPY_DRAWDOWN_FROM_PREVIOUS_HIGH" in grp.columns and not grp.loc[grp["SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"] < -0.10, ret_col].dropna().empty
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def aggregate_spell_metrics(spell_metrics: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if spell_metrics.empty:
        return pd.DataFrame()
    metric_cols = [
        "average_daily_return",
        "annualized_return",
        "annualized_volatility",
        "Sharpe",
        "max_drawdown",
        "worst_day",
        "best_day",
        "positive_day_ratio",
        "cumulative_return_within_state",
        "correlation_with_SPY",
        "average_daily_return_when_SPY_return_negative",
        "average_daily_return_when_SPY_drawdown_below_10pct",
    ]
    rows = []
    for keys, grp in spell_metrics.groupby(group_cols + ["asset", "asset_role"], observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols + ["asset", "asset_role"], keys))
        row["spell_count"] = int(grp["spell_id"].nunique())
        row["n_obs"] = int(grp["spell_n_obs"].sum())
        row["avg_spell_duration"] = float(grp["spell_n_obs"].mean())
        row["LOW_SAMPLE"] = row["n_obs"] < 10
        for col in metric_cols:
            valid = grp[[col, "spell_n_obs"]].dropna()
            row[col] = float(np.average(valid[col], weights=valid["spell_n_obs"])) if not valid.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def load_panel() -> pd.DataFrame:
    if not REGIME_PANEL_PATH.exists():
        raise FileNotFoundError(f"Missing regime panel: {REGIME_PANEL_PATH}")
    if not TIMING_PANEL_PATH.exists():
        raise FileNotFoundError(f"Missing timing panel: {TIMING_PANEL_PATH}")
    regime = pd.read_csv(REGIME_PANEL_PATH)
    timing = pd.read_csv(TIMING_PANEL_PATH, usecols=["date", "monthly_either_weight_spy"])
    regime["date"] = pd.to_datetime(regime["date"])
    timing["date"] = pd.to_datetime(timing["date"])
    panel = regime.merge(timing, on="date", how="inner")
    return panel.sort_values("date").reset_index(drop=True)


def assign_states(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    macro_conditions = [
        (out["CREDIT_SPREAD_BAA_AAA"] > 1.5) & (out["DGS1"] > 5),
        out["TERM_SPREAD_10Y_1Y"] < 0,
        (out["TERM_SPREAD_10Y_1Y"] >= 0) & (out["TERM_SPREAD_10Y_1Y"] < 1),
        out["TERM_SPREAD_10Y_1Y"] >= 1,
    ]
    out["macro_regime"] = np.select(macro_conditions, ["HIGH_INFLATION", "INVERTED", "FLAT", "STEEP"], default="NEUTRAL")
    out["spy_timing_state"] = np.where(out["monthly_either_weight_spy"] == 1.0, "HOLD", "SELL")
    out["cross_state"] = out["macro_regime"] + "_" + out["spy_timing_state"]
    return out


def rank_defensive_assets(perf_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    score_rows = []
    for cross_state, grp in perf_df.loc[perf_df["asset"].isin(DEFENSIVE_ASSETS)].groupby("cross_state", observed=False):
        work = grp.copy()
        if work.empty:
            continue
        work["rank_return"] = work["annualized_return"].rank(ascending=False, method="average")
        work["rank_sharpe"] = work["Sharpe"].rank(ascending=False, method="average")
        work["rank_drawdown"] = work["max_drawdown"].rank(ascending=False, method="average")
        work["rank_corr"] = work["correlation_with_SPY"].rank(ascending=True, method="average")
        work["rank_neg_spy"] = work["average_daily_return_when_SPY_return_negative"].rank(ascending=False, method="average")
        n = max(len(work), 1)
        for col in ["rank_return", "rank_sharpe", "rank_drawdown", "rank_corr", "rank_neg_spy"]:
            work[col.replace("rank_", "score_")] = 1.0 if n == 1 else 1.0 - (work[col] - 1.0) / (n - 1.0)
        work["composite_score"] = work[[c for c in work.columns if c.startswith("score_")]].mean(axis=1)
        rows.append(work[["macro_regime", "spy_timing_state", "cross_state", "asset", "annualized_return", "Sharpe", "max_drawdown", "correlation_with_SPY", "average_daily_return_when_SPY_return_negative", "composite_score"]])
        score_rows.append(work[["cross_state", "asset", "composite_score"]])
    return (
        pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(),
        pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame(),
    )


def pivot_and_save(perf_df: pd.DataFrame, value_col: str, path: Path) -> pd.DataFrame:
    pivot = perf_df.pivot_table(index="asset", columns="cross_state", values=value_col, aggfunc="first")
    ordered_cols = [f"{m}_{t}" for m in MACRO_ORDER for t in TIMING_ORDER if f"{m}_{t}" in pivot.columns]
    pivot = pivot.reindex(index=ALL_ASSETS, columns=ordered_cols)
    pivot.to_csv(path)
    return pivot


def plot_heatmap(pivot: pd.DataFrame, path: Path, title: str, center: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(max(10, 0.9 * max(len(pivot.columns), 1)), 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=center, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(perf_df: pd.DataFrame, sell_df: pd.DataFrame, hold_df: pd.DataFrame, score_df: pd.DataFrame) -> None:
    lines = [
        "# Defensive Sleeve by Macro Regime and SPY Timing State",
        "",
        "## Purpose",
        "",
        "This diagnostic does not build final allocations. It studies defensive asset behavior under `macro_regime x SPY timing state`, where SPY timing state is driven by `MONTHLY_EITHER_CONFIRM`.",
        "",
        "Timing states:",
        "- `HOLD`: Monthly Either Confirm wants to hold SPY.",
        "- `SELL`: Monthly Either Confirm wants to sell SPY and move into defensive assets.",
        "",
        "## Method",
        "",
        "- Macro regime follows the existing rule-based classification: `HIGH_INFLATION`, `INVERTED`, `FLAT`, `STEEP`, `NEUTRAL` fallback.",
        "- Cross states are constructed as `macro_regime + '_' + spy_timing_state`.",
        "- Metrics are computed spell-by-spell on contiguous cross-state windows, then aggregated. This avoids stitched-state drawdown distortion.",
        "- Defensive assets compared: `CASH`, `IEF`, `GOLD`, `CMDTY_FUT`.",
        "- `SPY` remains in the tables as the risk reference.",
        "",
        "## SELL-State Findings",
        "",
    ]
    if not sell_df.empty:
        for macro in MACRO_ORDER:
            grp = sell_df.loc[sell_df["macro_regime"] == macro]
            if grp.empty:
                continue
            best_ret = grp.sort_values("annualized_return", ascending=False).iloc[0]
            best_sharpe = grp.sort_values("Sharpe", ascending=False).iloc[0]
            best_dd = grp.sort_values("max_drawdown", ascending=False).iloc[0]
            lines += [
                f"### {macro} + SELL",
                f"- Highest annualized return: `{best_ret['asset']}` ({best_ret['annualized_return']:.2%}, n={int(best_ret['n_obs'])})",
                f"- Best Sharpe: `{best_sharpe['asset']}` ({best_sharpe['Sharpe']:.2f})",
                f"- Lowest drawdown: `{best_dd['asset']}` ({best_dd['max_drawdown']:.2%})",
                "",
            ]
    else:
        lines.append("No SELL-state rows available.")
        lines.append("")
    lines += [
        "## HOLD-State Hedge Findings",
        "",
        "When SPY timing still says HOLD, the question is not where to hide all capital, but which asset has the best hedge characteristics against SPY weakness.",
        "",
    ]
    if not hold_df.empty:
        for macro in MACRO_ORDER:
            grp = hold_df.loc[hold_df["macro_regime"] == macro]
            if grp.empty:
                continue
            hedges = grp.loc[grp["asset"].isin(DEFENSIVE_ASSETS)].copy()
            if hedges.empty:
                continue
            best_corr = hedges.sort_values("correlation_with_SPY", ascending=True).iloc[0]
            best_neg = hedges.sort_values("average_daily_return_when_SPY_return_negative", ascending=False).iloc[0]
            best_sharpe = hedges.sort_values("Sharpe", ascending=False).iloc[0]
            lines += [
                f"### {macro} + HOLD",
                f"- Lowest correlation with SPY: `{best_corr['asset']}` ({best_corr['correlation_with_SPY']:.2f})",
                f"- Best return when SPY is negative: `{best_neg['asset']}` ({best_neg['average_daily_return_when_SPY_return_negative']:.4%} per day)",
                f"- Best standalone Sharpe among defensive assets: `{best_sharpe['asset']}` ({best_sharpe['Sharpe']:.2f})",
                "",
            ]
    else:
        lines.append("No HOLD-state rows available.")
        lines.append("")
    lines += [
        "## Composite Diagnostic",
        "",
        "Composite score ranks defensive assets within each cross state on:",
        "- annualized return",
        "- Sharpe",
        "- drawdown",
        "- correlation with SPY",
        "- return when SPY is negative",
        "",
        "This is a diagnostic ranking only, not a final allocation rule.",
        "",
        "## Caveats",
        "",
        "- This is not a strategy backtest.",
        "- SELL-state results answer where capital historically performed better after SPY timing turned off.",
        "- HOLD-state results answer which defensive assets hedge SPY more effectively while the main equity timing signal remains on.",
        "- Final portfolio decisions still need explicit no-look-ahead execution and turnover-aware backtests.",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = assign_states(load_panel())
    panel.to_csv(PANEL_PATH, index=False)

    cross_spell = spell_level_asset_metrics(panel, ["macro_regime", "spy_timing_state", "cross_state"], ALL_ASSETS)
    cross_perf = aggregate_spell_metrics(cross_spell, ["macro_regime", "spy_timing_state", "cross_state"])
    cross_perf.to_csv(PERFORMANCE_PATH, index=False)

    sample = cross_perf.pivot_table(index="macro_regime", columns="spy_timing_state", values="n_obs", aggfunc="max").reindex(index=MACRO_ORDER, columns=TIMING_ORDER)
    sample.to_csv(SAMPLE_SIZE_PATH)

    ann_pivot = pivot_and_save(cross_perf, "annualized_return", ANN_RETURN_PATH)
    sharpe_pivot = pivot_and_save(cross_perf, "Sharpe", SHARPE_PATH)
    dd_pivot = pivot_and_save(cross_perf, "max_drawdown", MAX_DD_PATH)
    pos_pivot = pivot_and_save(cross_perf, "positive_day_ratio", POSITIVE_PATH)

    sell_mask = panel["spy_timing_state"] == "SELL"
    sell_spell = spell_level_asset_metrics(panel, ["macro_regime"], DEFENSIVE_ASSETS + ["SPY"], active_mask=sell_mask)
    sell_perf = aggregate_spell_metrics(sell_spell, ["macro_regime"])
    sell_perf.to_csv(SELL_PATH, index=False)

    hold_mask = panel["spy_timing_state"] == "HOLD"
    hold_spell = spell_level_asset_metrics(panel, ["macro_regime"], DEFENSIVE_ASSETS + ["SPY"], active_mask=hold_mask)
    hold_perf = aggregate_spell_metrics(hold_spell, ["macro_regime"])
    hold_perf.to_csv(HOLD_PATH, index=False)

    ranking_df, score_df = rank_defensive_assets(cross_perf)
    ranking_df.to_csv(RANKING_PATH, index=False)
    score_df.to_csv(SCORE_PATH, index=False)

    plot_heatmap(ann_pivot, FIG_RETURN, "Annualized Return by Macro Regime and SPY Timing State")
    plot_heatmap(sharpe_pivot, FIG_SHARPE, "Sharpe by Macro Regime and SPY Timing State")
    plot_heatmap(dd_pivot, FIG_DRAWDOWN, "Max Drawdown by Macro Regime and SPY Timing State", center=0.0)

    if not sell_perf.empty:
        sell_return = sell_perf.pivot_table(index="asset", columns="macro_regime", values="annualized_return", aggfunc="first").reindex(index=DEFENSIVE_ASSETS + ["SPY"], columns=MACRO_ORDER)
        sell_dd = sell_perf.pivot_table(index="asset", columns="macro_regime", values="max_drawdown", aggfunc="first").reindex(index=DEFENSIVE_ASSETS + ["SPY"], columns=MACRO_ORDER)
        plot_heatmap(sell_return, FIG_SELL_RETURN, "SELL-State Annualized Return by Macro Regime")
        plot_heatmap(sell_dd, FIG_SELL_DRAWDOWN, "SELL-State Max Drawdown by Macro Regime", center=0.0)
    if not hold_perf.empty:
        hold_return = hold_perf.pivot_table(index="asset", columns="macro_regime", values="annualized_return", aggfunc="first").reindex(index=DEFENSIVE_ASSETS + ["SPY"], columns=MACRO_ORDER)
        hold_hedge = hold_perf.pivot_table(index="asset", columns="macro_regime", values="correlation_with_SPY", aggfunc="first").reindex(index=DEFENSIVE_ASSETS + ["SPY"], columns=MACRO_ORDER)
        plot_heatmap(hold_return, FIG_HOLD_RETURN, "HOLD-State Annualized Return by Macro Regime")
        plot_heatmap(hold_hedge, FIG_HOLD_HEDGE, "HOLD-State Correlation with SPY", center=0.0)
    if not score_df.empty:
        score_pivot = score_df.pivot_table(index="asset", columns="cross_state", values="composite_score", aggfunc="first")
        ordered_cols = [f"{m}_{t}" for m in MACRO_ORDER for t in TIMING_ORDER if f"{m}_{t}" in score_pivot.columns]
        score_pivot = score_pivot.reindex(index=DEFENSIVE_ASSETS, columns=ordered_cols)
        plot_heatmap(score_pivot, FIG_SCORE, "Defensive Composite Score by Cross State")
    plot_heatmap(sample, FIG_SAMPLE, "Cross-State Sample Size", center=None)

    write_report(cross_perf, sell_perf, hold_perf, score_df)

    print("Saved panel:", PANEL_PATH)
    print("Saved performance:", PERFORMANCE_PATH)
    print("Saved sell-state table:", SELL_PATH)
    print("Saved hold-state table:", HOLD_PATH)
    if not sell_perf.empty:
        for macro in MACRO_ORDER:
            grp = sell_perf.loc[(sell_perf["macro_regime"] == macro) & (sell_perf["asset"].isin(DEFENSIVE_ASSETS))]
            if grp.empty:
                continue
            best = grp.sort_values("annualized_return", ascending=False).iloc[0]
            print(f"SELL {macro}: best return {best['asset']} ({best['annualized_return']:.2%})")
    if not hold_perf.empty:
        for macro in MACRO_ORDER:
            grp = hold_perf.loc[(hold_perf["macro_regime"] == macro) & (hold_perf["asset"].isin(DEFENSIVE_ASSETS))]
            if grp.empty:
                continue
            best = grp.sort_values("average_daily_return_when_SPY_return_negative", ascending=False).iloc[0]
            print(f"HOLD {macro}: best hedge-on-down-days {best['asset']} ({best['average_daily_return_when_SPY_return_negative']:.4%})")


if __name__ == "__main__":
    main()
