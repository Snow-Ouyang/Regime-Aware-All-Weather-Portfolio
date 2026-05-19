from __future__ import annotations

import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "results" / "recovery_20d_strategy_test_L50_H30"
SOURCE_TABLE_DIR = SOURCE_DIR / "tables"

OUTPUT_DIR = ROOT / "results" / "recovery_20d_equal_weight_attribution"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
EPISODE_CHART_DIR = OUTPUT_DIR / "episode_charts"

BASELINE = "FLAT_RATE_REFINED_L50_H30"
RECOVERY = "RECOVERY_20D_EQUAL_WEIGHT"
MATURE = "MATURE_REGIME_HEDGE_FINAL"
OPTIONAL_SPY = "RECOVERY_20D_SPY"
ASSETS = ["SPY", "GOLD", "CMDTY_FUT", "IEF", "CASH"]

REQUIRED_INPUTS = [
    "daily_returns_all_strategies.csv",
    "daily_weights_all_strategies.csv",
    "recovery_episode_strategy_performance.csv",
    "recovery_strategy_summary.csv",
    "performance_comparison.csv",
]

FINAL_PANEL_CANDIDATES = [
    ROOT / "results" / "09_final_strategy" / "mature_regime_hedge_final" / "daily_backtest_panel.csv",
    ROOT / "results" / "mature_regime_hedge_final" / "daily_backtest_panel.csv",
]


def ensure_inputs() -> None:
    missing = [f for f in REQUIRED_INPUTS if not (SOURCE_TABLE_DIR / f).exists()]
    if missing:
        runpy.run_path(str(ROOT / "scripts" / "test_recovery_20d_strategies_L50_H30.py"), run_name="__main__")
    missing = [f for f in REQUIRED_INPUTS if not (SOURCE_TABLE_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required recovery test outputs: {missing}")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_inputs()
    returns = pd.read_csv(SOURCE_TABLE_DIR / "daily_returns_all_strategies.csv", parse_dates=["date"])
    weights = pd.read_csv(SOURCE_TABLE_DIR / "daily_weights_all_strategies.csv", parse_dates=["date"])
    episodes = pd.read_csv(SOURCE_TABLE_DIR / "recovery_episode_strategy_performance.csv", parse_dates=["recovery_start_date", "recovery_end_date"])
    summary = pd.read_csv(SOURCE_TABLE_DIR / "recovery_strategy_summary.csv")
    performance = pd.read_csv(SOURCE_TABLE_DIR / "performance_comparison.csv")

    final_path = next((p for p in FINAL_PANEL_CANDIDATES if p.exists()), None)
    if final_path is None:
        raise FileNotFoundError("Missing final strategy panel for asset returns.")
    final = pd.read_csv(final_path, parse_dates=["date"])
    asset_cols = ["date"] + [f"{asset}_return" for asset in ASSETS]
    missing = [c for c in asset_cols if c not in final.columns]
    if missing:
        raise ValueError(f"Final panel missing asset returns: {missing}")
    assets = final[asset_cols].copy()
    return returns, weights, episodes, summary, performance, assets


def product_return(series: pd.Series) -> float:
    return float((1.0 + series.fillna(0.0)).prod() - 1.0)


def max_drawdown(series: pd.Series) -> float:
    nav = (1.0 + series.fillna(0.0)).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def get_strategy_slice(returns_wide: pd.DataFrame, strategy: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return returns_wide[(returns_wide["date"].ge(start)) & (returns_wide["date"].le(end))].copy()


def build_wide_returns(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.pivot(index="date", columns="strategy", values=["daily_return", "equity", "drawdown", "turnover", "transaction_cost"]).sort_index()


def flatten_wide_returns(wide: pd.DataFrame) -> pd.DataFrame:
    flat = pd.DataFrame({"date": wide.index})
    for field in ["daily_return", "equity", "drawdown", "turnover", "transaction_cost"]:
        for strategy in wide[field].columns:
            flat[f"{strategy}_{field}"] = wide[(field, strategy)].values
    return flat.reset_index(drop=True)


def build_weight_wide(weights: pd.DataFrame) -> pd.DataFrame:
    return weights.pivot_table(index="date", columns=["strategy", "asset"], values="weight", aggfunc="first").sort_index()


def avg_weights(weight_wide: pd.DataFrame, strategy: str, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    sub = weight_wide[(weight_wide.index >= start) & (weight_wide.index <= end)]
    out = {}
    for asset in ASSETS:
        col = (strategy, asset)
        out[asset] = float(sub[col].mean()) if col in sub.columns and len(sub) else np.nan
    return out


def classify_market_period(date: pd.Timestamp) -> str:
    if date < pd.Timestamp("2007-10-01"):
        return "pre_2008"
    if date <= pd.Timestamp("2009-06-30"):
        return "GFC_2007_2009"
    if pd.Timestamp("2010-01-01") <= date <= pd.Timestamp("2014-12-31"):
        return "post_GFC_2010_2014"
    if pd.Timestamp("2015-01-01") <= date <= pd.Timestamp("2016-12-31"):
        return "oil_growth_scare_2015_2016"
    if pd.Timestamp("2017-01-01") <= date <= pd.Timestamp("2019-12-31"):
        return "late_cycle_2017_2019"
    if pd.Timestamp("2020-02-01") <= date <= pd.Timestamp("2020-12-31"):
        return "covid_2020"
    if pd.Timestamp("2021-01-01") <= date <= pd.Timestamp("2022-12-31"):
        return "inflation_hiking_2021_2022"
    if date >= pd.Timestamp("2023-01-01"):
        return "post_hiking_2023_present"
    return "other"


def enrich_episode_attribution(
    episodes: pd.DataFrame, returns_flat: pd.DataFrame, weights_wide: pd.DataFrame, asset_returns: pd.DataFrame
) -> pd.DataFrame:
    trading_dates = returns_flat["date"].tolist()
    rows = []
    for _, ep in episodes.iterrows():
        start = ep["recovery_start_date"]
        end = ep["recovery_end_date"]
        sub = returns_flat[(returns_flat["date"].ge(start)) & (returns_flat["date"].le(end))]
        assets_sub = asset_returns[(asset_returns["date"].ge(start)) & (asset_returns["date"].le(end))]

        baseline_r = sub[f"{BASELINE}_daily_return"]
        recovery_r = sub[f"{RECOVERY}_daily_return"]
        base_w = avg_weights(weights_wide, BASELINE, start, end)
        rec_w = avg_weights(weights_wide, RECOVERY, start, end)

        next_stress = ""
        days_until = np.nan
        if ep["exit_type"] == "interrupted_by_new_stress":
            next_idx = trading_dates.index(end) + 1 if end in trading_dates and trading_dates.index(end) + 1 < len(trading_dates) else None
            if next_idx is not None:
                next_stress = trading_dates[next_idx]
                days_until = len(sub)

        baseline_return = product_return(baseline_r)
        recovery_return = product_return(recovery_r)
        reported_excess = ep.get(f"{RECOVERY}_minus_refined", recovery_return - baseline_return)
        row = {
            "episode_id": int(ep["episode_id"]),
            "recovery_start_date": start.date(),
            "recovery_end_date": end.date(),
            "episode_length_days": int(ep["episode_length_days"]),
            "exit_type": ep["exit_type"],
            "start_regime": ep["start_regime"],
            "start_sub_state": ep["start_sub_state"],
            "selected_assets_for_equal_weight": ep["selected_assets_for_equal_weight"],
            "baseline_return": baseline_return,
            "recovery_equal_weight_return": recovery_return,
            "excess_return_vs_baseline": float(reported_excess),
            "baseline_maxdd": max_drawdown(baseline_r),
            "recovery_equal_weight_maxdd": max_drawdown(recovery_r),
            "maxdd_diff_vs_baseline": max_drawdown(recovery_r) - max_drawdown(baseline_r),
            "baseline_volatility": float(baseline_r.std() * np.sqrt(252)) if len(baseline_r) > 1 else np.nan,
            "recovery_volatility": float(recovery_r.std() * np.sqrt(252)) if len(recovery_r) > 1 else np.nan,
            "baseline_turnover": float(sub[f"{BASELINE}_turnover"].sum()),
            "recovery_turnover": float(sub[f"{RECOVERY}_turnover"].sum()),
            "baseline_transaction_cost": float(sub[f"{BASELINE}_transaction_cost"].sum()),
            "recovery_transaction_cost": float(sub[f"{RECOVERY}_transaction_cost"].sum()),
            "next_stress_start_date": next_stress,
            "days_until_next_stress": days_until,
            "interrupted_by_new_stress_flag": ep["exit_type"] == "interrupted_by_new_stress",
        }
        row["volatility_diff"] = row["recovery_volatility"] - row["baseline_volatility"]
        row["turnover_diff"] = row["recovery_turnover"] - row["baseline_turnover"]
        row["transaction_cost_diff"] = row["recovery_transaction_cost"] - row["baseline_transaction_cost"]

        for asset in ASSETS:
            row[f"{asset}_return"] = product_return(assets_sub[f"{asset}_return"]) if f"{asset}_return" in assets_sub else np.nan
            row[f"baseline_avg_weight_{asset}"] = base_w[asset]
            row[f"recovery_avg_weight_{asset}"] = rec_w[asset]
            row[f"weight_difference_{asset}"] = rec_w[asset] - base_w[asset]

        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values("excess_return_vs_baseline", ascending=False).reset_index(drop=True)
    out["contribution_rank_by_excess_return"] = np.arange(1, len(out) + 1)
    out = out.sort_values("excess_return_vs_baseline", ascending=True).reset_index(drop=True)
    out["drag_rank_by_excess_return"] = np.arange(1, len(out) + 1)
    out = out.sort_values("episode_id").reset_index(drop=True)
    top = out.nsmallest(0, "episode_id")
    top1_ids = set(out.nlargest(1, "excess_return_vs_baseline")["episode_id"])
    top3_ids = set(out.nlargest(3, "excess_return_vs_baseline")["episode_id"])
    top5_ids = set(out.nlargest(5, "excess_return_vs_baseline")["episode_id"])
    bottom3_ids = set(out.nsmallest(3, "excess_return_vs_baseline")["episode_id"])
    bottom5_ids = set(out.nsmallest(5, "excess_return_vs_baseline")["episode_id"])
    out["is_top_contributor_top1"] = out["episode_id"].isin(top1_ids)
    out["is_top_contributor_top3"] = out["episode_id"].isin(top3_ids)
    out["is_top_contributor_top5"] = out["episode_id"].isin(top5_ids)
    out["is_bottom_drag_top3"] = out["episode_id"].isin(bottom3_ids)
    out["is_bottom_drag_top5"] = out["episode_id"].isin(bottom5_ids)
    return out


def contribution_share(ep_attr: pd.DataFrame, n: int) -> float:
    positives = ep_attr.loc[ep_attr["excess_return_vs_baseline"] > 0, "excess_return_vs_baseline"]
    denom = positives.sum()
    if denom <= 0:
        return np.nan
    return float(ep_attr.nlargest(n, "excess_return_vs_baseline")["excess_return_vs_baseline"].clip(lower=0).sum() / denom)


def drag_share(ep_attr: pd.DataFrame, n: int) -> float:
    positives = ep_attr.loc[ep_attr["excess_return_vs_baseline"] > 0, "excess_return_vs_baseline"]
    denom = positives.sum()
    if denom <= 0:
        return np.nan
    return float(abs(ep_attr.nsmallest(n, "excess_return_vs_baseline")["excess_return_vs_baseline"].clip(upper=0).sum()) / denom)


def overall_summary(performance: pd.DataFrame, ep_attr: pd.DataFrame) -> pd.DataFrame:
    b = performance.loc[performance["strategy"].eq(BASELINE)].iloc[0]
    r = performance.loc[performance["strategy"].eq(RECOVERY)].iloc[0]
    positive = int((ep_attr["excess_return_vs_baseline"] > 0).sum())
    negative = int((ep_attr["excess_return_vs_baseline"] < 0).sum())
    material_positive = int((ep_attr["excess_return_vs_baseline"] > 1e-10).sum())
    material_negative = int((ep_attr["excess_return_vs_baseline"] < -1e-10).sum())
    row = {
        "baseline_final_equity": b["final_equity"],
        "recovery_final_equity": r["final_equity"],
        "final_equity_diff": r["final_equity"] - b["final_equity"],
        "baseline_CAGR": b["CAGR"],
        "recovery_CAGR": r["CAGR"],
        "CAGR_diff": r["CAGR"] - b["CAGR"],
        "baseline_Sharpe": b["Sharpe"],
        "recovery_Sharpe": r["Sharpe"],
        "Sharpe_diff": r["Sharpe"] - b["Sharpe"],
        "baseline_MaxDD": b["MaxDD"],
        "recovery_MaxDD": r["MaxDD"],
        "MaxDD_diff": r["MaxDD"] - b["MaxDD"],
        "baseline_Calmar": b["Calmar"],
        "recovery_Calmar": r["Calmar"],
        "Calmar_diff": r["Calmar"] - b["Calmar"],
        "total_excess_return_contribution": ep_attr["excess_return_vs_baseline"].sum(),
        "number_of_recovery_episodes": len(ep_attr),
        "number_of_positive_excess_episodes": positive,
        "number_of_negative_excess_episodes": negative,
        "episode_win_rate": positive / len(ep_attr) if len(ep_attr) else np.nan,
        "number_of_material_positive_excess_episodes": material_positive,
        "number_of_material_negative_excess_episodes": material_negative,
        "material_episode_win_rate": material_positive / len(ep_attr) if len(ep_attr) else np.nan,
        "top_1_episode_contribution_share": contribution_share(ep_attr, 1),
        "top_3_episode_contribution_share": contribution_share(ep_attr, 3),
        "top_5_episode_contribution_share": contribution_share(ep_attr, 5),
        "bottom_3_episode_drag_share": drag_share(ep_attr, 3),
        "bottom_5_episode_drag_share": drag_share(ep_attr, 5),
    }
    return pd.DataFrame([row])


def summarize_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for key, sub in df.groupby(group_col, dropna=False):
        best = sub.nlargest(1, "excess_return_vs_baseline")
        worst = sub.nsmallest(1, "excess_return_vs_baseline")
        rows.append(
            {
                group_col: key,
                "number_of_episodes": len(sub),
                "total_excess_return": sub["excess_return_vs_baseline"].sum(),
                "mean_excess_return": sub["excess_return_vs_baseline"].mean(),
                "median_excess_return": sub["excess_return_vs_baseline"].median(),
                "win_rate": (sub["excess_return_vs_baseline"] > 0).mean(),
                "average_episode_length": sub["episode_length_days"].mean(),
                "interrupted_rate": sub["interrupted_by_new_stress_flag"].mean(),
                "top_contributor_episode": int(best["episode_id"].iloc[0]) if not best.empty else np.nan,
                "worst_drag_episode": int(worst["episode_id"].iloc[0]) if not worst.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("total_excess_return", ascending=False)


def attribution_by_year(ep_attr: pd.DataFrame) -> pd.DataFrame:
    df = ep_attr.copy()
    df["year"] = pd.to_datetime(df["recovery_start_date"]).dt.year
    rows = []
    for year, sub in df.groupby("year"):
        rows.append(
            {
                "year": year,
                "number_of_episodes": len(sub),
                "total_excess_return": sub["excess_return_vs_baseline"].sum(),
                "mean_excess_return": sub["excess_return_vs_baseline"].mean(),
                "median_excess_return": sub["excess_return_vs_baseline"].median(),
                "win_rate": (sub["excess_return_vs_baseline"] > 0).mean(),
                "best_episode_excess_return": sub["excess_return_vs_baseline"].max(),
                "worst_episode_excess_return": sub["excess_return_vs_baseline"].min(),
            }
        )
    return pd.DataFrame(rows)


def attribution_by_market_period(ep_attr: pd.DataFrame) -> pd.DataFrame:
    df = ep_attr.copy()
    df["market_period"] = pd.to_datetime(df["recovery_start_date"]).map(classify_market_period)
    return summarize_group(df, "market_period")


def build_daily_excess(returns_flat: pd.DataFrame, ep_attr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, ep in ep_attr.iterrows():
        start = pd.Timestamp(ep["recovery_start_date"])
        end = pd.Timestamp(ep["recovery_end_date"])
        sub = returns_flat[(returns_flat["date"].ge(start)) & (returns_flat["date"].le(end))].copy()
        sub["episode_id"] = ep["episode_id"]
        sub["baseline_return"] = sub[f"{BASELINE}_daily_return"]
        sub["recovery_equal_weight_return"] = sub[f"{RECOVERY}_daily_return"]
        sub["daily_excess_return"] = sub["recovery_equal_weight_return"] - sub["baseline_return"]
        sub["cumulative_excess_return_within_episode"] = (1 + sub["daily_excess_return"]).cumprod() - 1
        sub["baseline_equity"] = sub[f"{BASELINE}_equity"]
        sub["recovery_equity"] = sub[f"{RECOVERY}_equity"]
        sub["start_regime"] = ep["start_regime"]
        sub["start_sub_state"] = ep["start_sub_state"]
        sub["selected_assets"] = ep["selected_assets_for_equal_weight"]
        sub["days_since_recovery_start"] = np.arange(1, len(sub) + 1)
        sub["stress_flag"] = False
        sub["recovery_override_active_flag"] = True
        rows.append(
            sub[
                [
                    "date",
                    "episode_id",
                    "baseline_return",
                    "recovery_equal_weight_return",
                    "daily_excess_return",
                    "cumulative_excess_return_within_episode",
                    "baseline_equity",
                    "recovery_equity",
                    "start_regime",
                    "start_sub_state",
                    "selected_assets",
                    "days_since_recovery_start",
                    "stress_flag",
                    "recovery_override_active_flag",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    colors = np.where(df[y_col] >= 0, "#2ca02c", "#d62728")
    ax.bar(df[x_col].astype(str), df[y_col], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(y_col)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_outputs(
    ep_attr: pd.DataFrame,
    daily_excess: pd.DataFrame,
    year_df: pd.DataFrame,
    period_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    asset_returns: pd.DataFrame,
    returns_flat: pd.DataFrame,
) -> None:
    ordered = ep_attr.sort_values("recovery_start_date").copy()
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = np.where(ordered["excess_return_vs_baseline"] >= 0, "#2ca02c", "#d62728")
    ax.bar(ordered["episode_id"].astype(str), ordered["excess_return_vs_baseline"], color=colors)
    for _, row in ordered.nlargest(5, "excess_return_vs_baseline").iterrows():
        ax.text(str(row["episode_id"]), row["excess_return_vs_baseline"], f"#{int(row['episode_id'])}", ha="center", va="bottom", fontsize=8)
    for _, row in ordered.nsmallest(5, "excess_return_vs_baseline").iterrows():
        ax.text(str(row["episode_id"]), row["excess_return_vs_baseline"], f"#{int(row['episode_id'])}", ha="center", va="top", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Episode excess return: RECOVERY_20D_EQUAL_WEIGHT vs L50_H30")
    ax.set_xlabel("Episode ID")
    ax.set_ylabel("Excess return")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "episode_excess_return_bar.png", dpi=160)
    plt.close(fig)

    if not daily_excess.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        daily = daily_excess.sort_values("date").copy()
        daily["cumulative_excess_over_time"] = (1 + daily["daily_excess_return"]).cumprod() - 1
        ax.plot(daily["date"], daily["cumulative_excess_over_time"])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Cumulative excess return during recovery override days")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "cumulative_excess_return_over_time.png", dpi=160)
        plt.close(fig)

    positives = ep_attr[ep_attr["excess_return_vs_baseline"] > 0]["excess_return_vs_baseline"].sum()
    shares = []
    for n in [1, 3, 5, 10]:
        shares.append({"top_n": n, "share": contribution_share(ep_attr, n)})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    share_df = pd.DataFrame(shares)
    ax.bar(share_df["top_n"].astype(str), share_df["share"])
    ax.set_ylim(0, max(1.0, share_df["share"].max() * 1.1))
    ax.set_title("Top episode contribution share of positive excess")
    ax.set_xlabel("Top N")
    ax.set_ylabel("Share")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "top_contribution_share_pareto.png", dpi=160)
    plt.close(fig)

    plot_bar(year_df, "year", "total_excess_return", "Attribution by year", FIGURE_DIR / "attribution_by_year.png")
    plot_bar(period_df, "market_period", "total_excess_return", "Attribution by market period", FIGURE_DIR / "attribution_by_market_period.png")
    plot_bar(regime_df, "start_regime", "mean_excess_return", "Mean excess return by start regime", FIGURE_DIR / "attribution_by_start_regime.png")
    plot_bar(pool_df, "selected_assets_for_equal_weight", "total_excess_return", "Attribution by selected assets", FIGURE_DIR / "attribution_by_selected_assets.png")

    for prefix, subset in [("top_contributor", ep_attr.nlargest(5, "excess_return_vs_baseline")), ("worst_drag", ep_attr.nsmallest(5, "excess_return_vs_baseline"))]:
        for _, ep in subset.iterrows():
            start = pd.Timestamp(ep["recovery_start_date"])
            end = pd.Timestamp(ep["recovery_end_date"])
            sub_r = returns_flat[(returns_flat["date"].ge(start)) & (returns_flat["date"].le(end))].copy()
            sub_a = asset_returns[(asset_returns["date"].ge(start)) & (asset_returns["date"].le(end))].copy()
            fig, ax1 = plt.subplots(figsize=(10, 5))
            base = (1 + sub_r[f"{BASELINE}_daily_return"].fillna(0)).cumprod() - 1
            rec = (1 + sub_r[f"{RECOVERY}_daily_return"].fillna(0)).cumprod() - 1
            ax1.plot(sub_r["date"], base, label=BASELINE, linewidth=2)
            ax1.plot(sub_r["date"], rec, label=RECOVERY, linewidth=2)
            for asset in str(ep["selected_assets_for_equal_weight"]).split("/"):
                col = f"{asset}_return"
                if col in sub_a.columns:
                    ax1.plot(sub_a["date"], (1 + sub_a[col].fillna(0)).cumprod() - 1, label=asset, alpha=0.8)
            ax1.axhline(0, color="black", linewidth=0.8)
            ax1.set_title(f"{prefix.replace('_', ' ').title()} episode {int(ep['episode_id'])}: {start.date()} to {end.date()}")
            ax1.set_ylabel("Cumulative return")
            ax1.legend(fontsize=8, ncol=2)
            ax1.grid(alpha=0.25)
            ax2 = ax1.twinx()
            ax2.bar(sub_r["date"], sub_r[f"{RECOVERY}_daily_return"] - sub_r[f"{BASELINE}_daily_return"], alpha=0.2, color="gray", label="daily excess")
            ax2.set_ylabel("Daily excess")
            fig.tight_layout()
            fig.savefig(EPISODE_CHART_DIR / f"{prefix}_episode_{int(ep['episode_id'])}.png", dpi=160)
            plt.close(fig)


def write_readme(overall: pd.DataFrame, ep_attr: pd.DataFrame, period_df: pd.DataFrame, regime_df: pd.DataFrame, pool_df: pd.DataFrame) -> str:
    top1 = overall["top_1_episode_contribution_share"].iloc[0]
    top3 = overall["top_3_episode_contribution_share"].iloc[0]
    top5 = overall["top_5_episode_contribution_share"].iloc[0]
    win_rate = overall["episode_win_rate"].iloc[0]
    best_ep = ep_attr.nlargest(1, "excess_return_vs_baseline").iloc[0]
    worst_ep = ep_attr.nsmallest(1, "excess_return_vs_baseline").iloc[0]
    best_period = period_df.nlargest(1, "total_excess_return").iloc[0]
    worst_period = period_df.nsmallest(1, "total_excess_return").iloc[0]

    if top3 > 0.75 or win_rate < 0.35:
        recommendation = "needs filter"
    elif overall["Sharpe_diff"].iloc[0] > 0 and overall["MaxDD_diff"].iloc[0] >= -1e-9:
        recommendation = "keep"
    else:
        recommendation = "reject"

    text = f"""# RECOVERY_20D_EQUAL_WEIGHT Attribution

## Purpose

This independent attribution study explains why `RECOVERY_20D_EQUAL_WEIGHT` improved full-sample results versus `FLAT_RATE_REFINED_L50_H30` despite a low recovery episode win rate.

## Contribution Share Method

Episode contribution share is calculated as each episode's arithmetic excess return versus `FLAT_RATE_REFINED_L50_H30` divided by the sum of all positive episode excess returns. Drag share uses absolute negative episode excess returns divided by the same positive-excess denominator.

## Overall Findings

- Episode win rate: {win_rate:.2%}
- Positive episodes: {int(overall['number_of_positive_excess_episodes'].iloc[0])}
- Negative episodes: {int(overall['number_of_negative_excess_episodes'].iloc[0])}
- Material positive episodes (>1e-10): {int(overall['number_of_material_positive_excess_episodes'].iloc[0])}
- Material negative episodes (<-1e-10): {int(overall['number_of_material_negative_excess_episodes'].iloc[0])}
- Top 1 contribution share: {top1:.2%}
- Top 3 contribution share: {top3:.2%}
- Top 5 contribution share: {top5:.2%}
- Worst episode: #{int(worst_ep['episode_id'])}, excess {worst_ep['excess_return_vs_baseline']:.2%}
- Best episode: #{int(best_ep['episode_id'])}, excess {best_ep['excess_return_vs_baseline']:.2%}

## Economic Interpretation

The result is not broad-based across most recovery episodes. The low win rate means the improvement is concentrated in a minority of periods where equal-weight selected assets captured rebound exposure that inverse-vol was still underweighting.

The largest contributor starts on {best_ep['recovery_start_date']} in `{best_ep['start_regime']}` / `{best_ep['start_sub_state']}` with selected assets `{best_ep['selected_assets_for_equal_weight']}`. This should be checked as a plausible post-stress rebound rather than treated as parameter optimization evidence.

The largest drag starts on {worst_ep['recovery_start_date']} and has selected assets `{worst_ep['selected_assets_for_equal_weight']}`. If drag episodes cluster around quick re-stress or specific sub-states, a future rule needs a filter.

## Group Findings

- Best market period: `{best_period['market_period']}` with total excess {best_period['total_excess_return']:.2%}.
- Worst market period: `{worst_period['market_period']}` with total excess {worst_period['total_excess_return']:.2%}.
- Best start regime by total excess: `{regime_df.iloc[0]['start_regime']}`.
- Best selected asset pool by total excess: `{pool_df.iloc[0]['selected_assets_for_equal_weight']}`.

## Recommendation

Recommendation: **{recommendation}**.

Supportive evidence: full-sample CAGR, Sharpe, Calmar, and final equity improved without worsening MaxDD.

Cautionary evidence: episode win rate is low and contribution concentration is material. The next step should test filters by start regime/sub-state or simple ex-ante health checks such as credit not widening or SPY trend confirmation.
"""
    (OUTPUT_DIR / "README_recovery_20d_equal_weight_attribution.md").write_text(text, encoding="utf-8")
    return recommendation


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    EPISODE_CHART_DIR.mkdir(parents=True, exist_ok=True)

    returns_long, weights_long, episodes, _, performance, asset_returns = load_inputs()
    returns_wide = build_wide_returns(returns_long)
    returns_flat = flatten_wide_returns(returns_wide)
    weights_wide = build_weight_wide(weights_long)

    ep_attr = enrich_episode_attribution(episodes, returns_flat, weights_wide, asset_returns)
    overall = overall_summary(performance, ep_attr)
    top = ep_attr.nlargest(10, "excess_return_vs_baseline")
    worst = ep_attr.nsmallest(10, "excess_return_vs_baseline")
    year_df = attribution_by_year(ep_attr)
    period_df = attribution_by_market_period(ep_attr)
    regime_df = summarize_group(ep_attr, "start_regime")
    sub_state_df = summarize_group(ep_attr, "start_sub_state")
    pool_df = summarize_group(ep_attr, "selected_assets_for_equal_weight")
    daily_excess = build_daily_excess(returns_flat, ep_attr)

    overall.to_csv(TABLE_DIR / "overall_attribution_summary.csv", index=False)
    ep_attr.to_csv(TABLE_DIR / "recovery_episode_attribution.csv", index=False)
    top.to_csv(TABLE_DIR / "top_recovery_contributors.csv", index=False)
    worst.to_csv(TABLE_DIR / "worst_recovery_drags.csv", index=False)
    year_df.to_csv(TABLE_DIR / "attribution_by_year.csv", index=False)
    period_df.to_csv(TABLE_DIR / "attribution_by_market_period.csv", index=False)
    regime_df.to_csv(TABLE_DIR / "attribution_by_start_regime.csv", index=False)
    sub_state_df.to_csv(TABLE_DIR / "attribution_by_start_sub_state.csv", index=False)
    pool_df.to_csv(TABLE_DIR / "attribution_by_selected_assets.csv", index=False)
    daily_excess.to_csv(TABLE_DIR / "daily_excess_return_during_recovery.csv", index=False)

    plot_outputs(ep_attr, daily_excess, year_df, period_df, regime_df, pool_df, asset_returns, returns_flat)
    recommendation = write_readme(overall, ep_attr, period_df, regime_df, pool_df)

    best_period = period_df.nlargest(1, "total_excess_return").iloc[0]
    worst_period = period_df.nsmallest(1, "total_excess_return").iloc[0]
    best_ep = ep_attr.nlargest(1, "excess_return_vs_baseline").iloc[0]
    worst_ep = ep_attr.nsmallest(1, "excess_return_vs_baseline").iloc[0]

    print("Recovery 20D equal-weight attribution complete.")
    print(f"output_dir: {OUTPUT_DIR.relative_to(ROOT)}")
    print(f"total excess return: {overall['total_excess_return_contribution'].iloc[0]:.4f}")
    print(f"top 1 contribution share: {overall['top_1_episode_contribution_share'].iloc[0]:.2%}")
    print(f"top 3 contribution share: {overall['top_3_episode_contribution_share'].iloc[0]:.2%}")
    print(f"top 5 contribution share: {overall['top_5_episode_contribution_share'].iloc[0]:.2%}")
    print(f"positive / negative episodes: {overall['number_of_positive_excess_episodes'].iloc[0]} / {overall['number_of_negative_excess_episodes'].iloc[0]}")
    print(f"best contributor episode: {int(best_ep['episode_id'])} ({best_ep['excess_return_vs_baseline']:.2%})")
    print(f"worst drag episode: {int(worst_ep['episode_id'])} ({worst_ep['excess_return_vs_baseline']:.2%})")
    print(f"best market period: {best_period['market_period']} ({best_period['total_excess_return']:.2%})")
    print(f"worst market period: {worst_period['market_period']} ({worst_period['total_excess_return']:.2%})")
    print(f"recommendation: {recommendation}")


if __name__ == "__main__":
    main()
