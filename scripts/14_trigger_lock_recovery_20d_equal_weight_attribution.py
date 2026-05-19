from __future__ import annotations

import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_13 = ROOT / "scripts" / "13_trigger_lock_state_machine_v1.py"
OUT = ROOT / "results" / "trigger_lock_recovery_20d_equal_weight_attribution"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
EPISODE_DIR = OUT / "episode_charts"

NO_RECOVERY = "TRIGGER_LOCK_NO_RECOVERY"
CURRENT = "TRIGGER_LOCK_STATE_MACHINE_V1"
RECOVERY_WINDOW = 20
ASSETS = ["SPY", "GOLD", "CMDTY_FUT", "IEF", "CASH"]


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    EPISODE_DIR.mkdir(parents=True, exist_ok=True)


def load_namespace() -> dict:
    return runpy.run_path(str(SCRIPT_13))


def build_inputs(ns: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = ns["load_panel"]()
    candidate_weights, daily_state, _ = ns["build_trigger_lock_strategy"](panel)
    normal_w, _ = ns["normal_weights"](panel)
    no_rec_weights = pd.DataFrame(0.0, index=panel.index, columns=ns["ASSETS"])
    for i, row in panel.iterrows():
        if bool(daily_state.loc[i, "full_risk_active"]):
            refined = row["refined_regime"]
            if refined in {"FLAT_LOW_RATE", "FLAT_HIGH_RATE", "STEEP"}:
                w = ns["stress_weight_dict"](refined)
            else:
                w = {"SPY": 1.0}
        else:
            w = normal_w.loc[i].to_dict()
        no_rec_weights.loc[i, ns["ASSETS"]] = pd.Series(ns["normalize_weight_dict"](w))
    return panel, candidate_weights, no_rec_weights, daily_state


def selected_assets_for_regime(row: pd.Series, normal_w_row: pd.Series) -> list[str]:
    refined = str(row["refined_regime"])
    if refined == "FLAT_LOW_RATE":
        assets = ["SPY", "CMDTY_FUT", "GOLD"]
    elif refined == "FLAT_HIGH_RATE":
        assets = ["CMDTY_FUT", "GOLD"]
    elif refined == "STEEP":
        assets = ["SPY"]
    elif refined == "INVERTED":
        assets = ["SPY", "GOLD"]
    else:
        assets = [a for a in ASSETS if float(normal_w_row.get(a, 0.0)) > 1e-8]
    return [a for a in assets if pd.notna(row.get(f"{a}_return", np.nan))]


def full_risk_entries_exits(active: pd.Series) -> tuple[list[int], list[int]]:
    b = active.fillna(False).astype(bool)
    entries = b.index[b & ~b.shift(1, fill_value=False)].tolist()
    exits = b.index[~b & b.shift(1, fill_value=False)].tolist()
    return entries, exits


def identify_recovery_episodes(panel: pd.DataFrame, daily_state: pd.DataFrame, normal_w: pd.DataFrame) -> pd.DataFrame:
    active = daily_state["full_risk_active"].fillna(False).astype(bool)
    _, exits = full_risk_entries_exits(active)
    entries, _ = full_risk_entries_exits(active)
    rows = []
    episode_id = 1
    for exit_idx in exits:
        start = exit_idx
        next_entry = next((e for e in entries if e > start), None)
        max_end = min(start + RECOVERY_WINDOW - 1, len(panel) - 1)
        if next_entry is not None and next_entry <= max_end:
            end = next_entry - 1
            exit_type = "interrupted_by_new_stress"
        elif max_end == len(panel) - 1 and start + RECOVERY_WINDOW - 1 > len(panel) - 1:
            end = len(panel) - 1
            exit_type = "truncated_by_data_end"
        else:
            end = max_end
            exit_type = "completed_20d"
        if end < start:
            continue
        row = panel.loc[start]
        selected = selected_assets_for_regime(row, normal_w.loc[start])
        rows.append(
            {
                "episode_id": episode_id,
                "start_idx": start,
                "end_idx": end,
                "recovery_start_date": panel.loc[start, "date"],
                "recovery_end_date": panel.loc[end, "date"],
                "episode_length_days": end - start + 1,
                "exit_type": exit_type,
                "start_regime": panel.loc[start, "refined_regime"],
                "start_sub_state": str(daily_state.loc[start, "allocation_state"]).replace("_RECOVERY", "_NORMAL"),
                "selected_assets_for_equal_weight": "/".join(selected),
            }
        )
        episode_id += 1
    return pd.DataFrame(rows)


def max_drawdown(returns: pd.Series) -> float:
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    return float((nav / nav.cummax() - 1.0).min()) if not nav.empty else np.nan


def cumulative_return(returns: pd.Series) -> float:
    return float((1.0 + returns.fillna(0.0)).prod() - 1.0) if len(returns) else np.nan


def compute_episode_tables(panel: pd.DataFrame, episodes: pd.DataFrame, no_rec_out: pd.DataFrame, cur_out: pd.DataFrame, normal_w: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    daily_rows = []
    for _, ep in episodes.iterrows():
        start = int(ep["start_idx"])
        end = int(ep["end_idx"])
        idx = slice(start, end + 1)
        baseline_r = no_rec_out.loc[idx, f"{NO_RECOVERY}_return"].fillna(0.0)
        current_r = cur_out.loc[idx, f"{CURRENT}_return"].fillna(0.0)
        row0 = panel.loc[start]
        selected = [a for a in str(ep["selected_assets_for_equal_weight"]).split("/") if a]

        asset_returns = {asset: panel.loc[idx, f"{asset}_return"].fillna(0.0) for asset in selected}
        asset_total = {asset: cumulative_return(ret) for asset, ret in asset_returns.items()}
        best_asset = max(asset_total, key=asset_total.get) if asset_total else ""
        best_asset_return = asset_total.get(best_asset, np.nan) if best_asset else np.nan
        equal_weight_return = np.nan
        if selected:
            eq_daily = pd.concat(asset_returns, axis=1).mean(axis=1)
            equal_weight_return = cumulative_return(eq_daily)
            equal_weight_mdd = max_drawdown(eq_daily)
        else:
            eq_daily = pd.Series(dtype=float)
            equal_weight_mdd = np.nan

        baseline_total = cumulative_return(baseline_r)
        current_total = cumulative_return(current_r)
        baseline_mdd = max_drawdown(baseline_r)
        current_mdd = max_drawdown(current_r)

        rows.append(
            {
                "episode_id": ep["episode_id"],
                "recovery_start_date": ep["recovery_start_date"],
                "recovery_end_date": ep["recovery_end_date"],
                "episode_length_days": ep["episode_length_days"],
                "exit_type": ep["exit_type"],
                "start_regime": ep["start_regime"],
                "start_sub_state": ep["start_sub_state"],
                "selected_assets_for_equal_weight": ep["selected_assets_for_equal_weight"],
                "baseline_return": baseline_total,
                "current_recovery_return": current_total,
                "equal_weight_return": equal_weight_return,
                "best_single_asset": best_asset,
                "best_single_asset_return": best_asset_return,
                "current_excess_vs_baseline": current_total - baseline_total,
                "equal_weight_excess_vs_baseline": equal_weight_return - baseline_total if pd.notna(equal_weight_return) else np.nan,
                "best_single_excess_vs_baseline": best_asset_return - baseline_total if pd.notna(best_asset_return) else np.nan,
                "equal_weight_minus_best_single": equal_weight_return - best_asset_return if pd.notna(equal_weight_return) and pd.notna(best_asset_return) else np.nan,
                "baseline_maxdd": baseline_mdd,
                "current_recovery_maxdd": current_mdd,
                "equal_weight_maxdd": equal_weight_mdd,
                "current_maxdd_diff_vs_baseline": current_mdd - baseline_mdd,
                "equal_weight_maxdd_diff_vs_baseline": equal_weight_mdd - baseline_mdd if pd.notna(equal_weight_mdd) else np.nan,
                **{f"{asset}_return": asset_total.get(asset, np.nan) for asset in ASSETS},
            }
        )

        base_nav = (1.0 + baseline_r).cumprod()
        cur_nav = (1.0 + current_r).cumprod()
        cum_excess = (1.0 + current_r).cumprod() / (1.0 + baseline_r).cumprod() - 1.0
        for pos, j in enumerate(range(start, end + 1), start=1):
            daily_rows.append(
                {
                    "date": panel.loc[j, "date"],
                    "episode_id": ep["episode_id"],
                    "baseline_return": baseline_r.loc[j],
                    "current_recovery_return": current_r.loc[j],
                    "daily_excess_return": current_r.loc[j] - baseline_r.loc[j],
                    "cumulative_excess_return_within_episode": cum_excess.loc[j],
                    "baseline_equity": base_nav.loc[j],
                    "recovery_equity": cur_nav.loc[j],
                    "start_regime": ep["start_regime"],
                    "start_sub_state": ep["start_sub_state"],
                    "selected_assets": ep["selected_assets_for_equal_weight"],
                    "days_since_recovery_start": pos,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(daily_rows)


def summarize_group(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for key, sub in df.groupby(col, dropna=False):
        rows.append(
            {
                col: key,
                "number_of_episodes": len(sub),
                "total_current_excess_return": sub["current_excess_vs_baseline"].sum(),
                "mean_current_excess_return": sub["current_excess_vs_baseline"].mean(),
                "median_current_excess_return": sub["current_excess_vs_baseline"].median(),
                "current_win_rate_vs_baseline": (sub["current_excess_vs_baseline"] > 0).mean(),
                "total_equal_weight_excess": sub["equal_weight_excess_vs_baseline"].sum(),
                "mean_equal_weight_excess": sub["equal_weight_excess_vs_baseline"].mean(),
                "equal_weight_win_rate_vs_baseline": (sub["equal_weight_excess_vs_baseline"] > 0).mean(),
                "total_best_single_excess": sub["best_single_excess_vs_baseline"].sum(),
                "mean_best_single_excess": sub["best_single_excess_vs_baseline"].mean(),
                "best_single_win_rate_vs_baseline": (sub["best_single_excess_vs_baseline"] > 0).mean(),
                "mean_equal_weight_minus_best_single": sub["equal_weight_minus_best_single"].mean(),
                "top_contributor_episode": sub.sort_values("current_excess_vs_baseline", ascending=False).iloc[0]["episode_id"],
                "worst_drag_episode": sub.sort_values("current_excess_vs_baseline").iloc[0]["episode_id"],
            }
        )
    return pd.DataFrame(rows)


def build_overall_summary(ep_attr: pd.DataFrame, perf_df: pd.DataFrame) -> pd.DataFrame:
    baseline_perf = perf_df.loc[perf_df["strategy"].eq(NO_RECOVERY)].iloc[0]
    current_perf = perf_df.loc[perf_df["strategy"].eq(CURRENT)].iloc[0]
    positive = ep_attr[ep_attr["current_excess_vs_baseline"] > 0].sort_values("current_excess_vs_baseline", ascending=False)
    negative = ep_attr[ep_attr["current_excess_vs_baseline"] < 0].sort_values("current_excess_vs_baseline")
    pos_total = positive["current_excess_vs_baseline"].sum()
    neg_total = negative["current_excess_vs_baseline"].sum()

    def share(frame: pd.DataFrame, n: int, denom: float) -> float:
        if frame.empty or denom == 0:
            return np.nan
        return float(frame.head(n)["current_excess_vs_baseline"].sum() / denom)

    return pd.DataFrame(
        [
            {
                "baseline_final_equity": baseline_perf["final_equity"],
                "recovery_final_equity": current_perf["final_equity"],
                "final_equity_diff": current_perf["final_equity"] - baseline_perf["final_equity"],
                "baseline_CAGR": baseline_perf["CAGR"],
                "recovery_CAGR": current_perf["CAGR"],
                "CAGR_diff": current_perf["CAGR"] - baseline_perf["CAGR"],
                "baseline_Sharpe": baseline_perf["Sharpe"],
                "recovery_Sharpe": current_perf["Sharpe"],
                "Sharpe_diff": current_perf["Sharpe"] - baseline_perf["Sharpe"],
                "baseline_MaxDD": baseline_perf["MaxDD"],
                "recovery_MaxDD": current_perf["MaxDD"],
                "MaxDD_diff": current_perf["MaxDD"] - baseline_perf["MaxDD"],
                "baseline_Calmar": baseline_perf["Calmar"],
                "recovery_Calmar": current_perf["Calmar"],
                "Calmar_diff": current_perf["Calmar"] - baseline_perf["Calmar"],
                "total_excess_return_contribution": ep_attr["current_excess_vs_baseline"].sum(),
                "number_of_recovery_episodes": len(ep_attr),
                "number_of_positive_excess_episodes": int((ep_attr["current_excess_vs_baseline"] > 0).sum()),
                "number_of_negative_excess_episodes": int((ep_attr["current_excess_vs_baseline"] <= 0).sum()),
                "episode_win_rate": float((ep_attr["current_excess_vs_baseline"] > 0).mean()),
                "top_1_episode_contribution_share": share(positive, 1, pos_total),
                "top_3_episode_contribution_share": share(positive, 3, pos_total),
                "top_5_episode_contribution_share": share(positive, 5, pos_total),
                "bottom_3_episode_drag_share": share(negative, 3, neg_total) if neg_total != 0 else np.nan,
                "bottom_5_episode_drag_share": share(negative, 5, neg_total) if neg_total != 0 else np.nan,
            }
        ]
    )


def yearly_summary(ep_attr: pd.DataFrame) -> pd.DataFrame:
    tmp = ep_attr.copy()
    tmp["year"] = tmp["recovery_start_date"].dt.year
    return summarize_group(tmp, "year")


def plot_outputs(ep_attr: pd.DataFrame, daily_excess: pd.DataFrame, perf_df: pd.DataFrame, regime_df: pd.DataFrame, asset_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(ep_attr["episode_id"].astype(str), ep_attr["current_excess_vs_baseline"], color=np.where(ep_attr["current_excess_vs_baseline"] >= 0, "#2ca25f", "#de2d26"))
    ax.tick_params(axis="x", labelrotation=60)
    ax.set_title("Current Recovery Overlay Excess Return by Episode")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "recovery_episode_excess_return_bar.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    if not daily_excess.empty:
        tmp = daily_excess.sort_values("date").copy()
        tmp["cum_excess"] = (1.0 + tmp["daily_excess_return"]).cumprod() - 1.0
        ax.plot(tmp["date"], tmp["cum_excess"])
    ax.set_title("Cumulative Excess Return Over Time")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cumulative_excess_return_over_time.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    show = perf_df[perf_df["strategy"].isin([NO_RECOVERY, CURRENT])].copy()
    ax.bar(show["strategy"], show["final_equity"])
    ax.set_title("No-Recovery vs Current Candidate")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "strategy_comparison_no_recovery_vs_current.png", dpi=160)
    plt.close(fig)

    if not regime_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(regime_df))
        ax.bar(x - 0.25, regime_df["mean_current_excess_return"], width=0.25, label="Current candidate")
        ax.bar(x, regime_df["mean_equal_weight_excess"], width=0.25, label="All-regime equal weight")
        ax.bar(x + 0.25, regime_df["mean_best_single_excess"], width=0.25, label="Best single asset")
        ax.set_xticks(x)
        ax.set_xticklabels(regime_df["start_regime"], rotation=25)
        ax.legend(fontsize=8)
        ax.set_title("Mean Excess Return by Start Regime")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "mean_excess_by_start_regime.png", dpi=160)
        plt.close(fig)

    if not asset_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(asset_df["best_single_asset"], asset_df["count"])
        ax.set_title("Best Single Asset Frequency Across Recovery Episodes")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "best_single_asset_frequency.png", dpi=160)
        plt.close(fig)


def best_asset_frequency(ep_attr: pd.DataFrame) -> pd.DataFrame:
    sub = ep_attr[ep_attr["best_single_asset"].ne("")]
    if sub.empty:
        return pd.DataFrame(columns=["best_single_asset", "count"])
    return sub.groupby("best_single_asset").size().reset_index(name="count").sort_values("count", ascending=False)


def recommendation_by_regime(regime_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in regime_df.iterrows():
        if row["number_of_episodes"] < 3:
            rec = "Insufficient_Sample"
            note = "Sample too small to make a regime-specific recovery decision."
        elif row["mean_current_excess_return"] <= 0 and row["mean_equal_weight_excess"] <= 0 and row["mean_best_single_excess"] <= 0:
            rec = "No_Recovery_Needed"
            note = "Neither current overlay nor hypothetical overlays improved mean recovery return."
        elif row["mean_best_single_excess"] > row["mean_equal_weight_excess"] and row["mean_best_single_excess"] > 0:
            rec = "Single_Asset_Candidate"
            note = "A single asset dominates equal weight in this regime; inspect the best-asset table."
        else:
            rec = "Equal_Weight_Candidate"
            note = "Equal weight retains most recovery value without relying on a single asset winner."
        rows.append(
            {
                "start_regime": row["start_regime"],
                "number_of_episodes": row["number_of_episodes"],
                "mean_current_excess_return": row["mean_current_excess_return"],
                "mean_equal_weight_excess": row["mean_equal_weight_excess"],
                "mean_best_single_excess": row["mean_best_single_excess"],
                "recommendation": rec,
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def write_readme(overall: pd.DataFrame, regime_df: pd.DataFrame, rec_df: pd.DataFrame, ep_attr: pd.DataFrame) -> None:
    ov = overall.iloc[0]
    best_regime = regime_df.sort_values("mean_current_excess_return", ascending=False).iloc[0]["start_regime"] if not regime_df.empty else "N/A"
    best_asset = ep_attr["best_single_asset"].value_counts().index[0] if not ep_attr["best_single_asset"].dropna().empty else "N/A"
    text = f"""# Trigger-Lock Recovery 20D Equal-Weight Attribution

## Purpose

This reruns recovery attribution on the current low-turnover trigger-lock candidate, using stress periods defined by the trigger-lock state machine rather than the canonical high-turnover stress state. The question is whether recovery still adds value once FULL_RISK entry/exit and re-entry are already reduced.

## Setup

- Baseline for attribution: `{NO_RECOVERY}` using the same trigger-lock FULL_RISK state machine with recovery disabled.
- Current candidate: `{CURRENT}` with FLAT_LOW-only 20D equal-weight recovery.
- Diagnostic overlays: all-regime equal-weight selected assets and best single selected asset, both evaluated episode-by-episode as attribution tools rather than tradable final rules.

## Key Results

- Full-sample final equity delta, current candidate vs no recovery: `{ov['final_equity_diff']:.4f}`
- CAGR delta: `{ov['CAGR_diff']:.2%}`
- Sharpe delta: `{ov['Sharpe_diff']:.4f}`
- MaxDD delta: `{ov['MaxDD_diff']:.2%}`
- Recovery episode count under trigger-lock stress definition: `{int(ov['number_of_recovery_episodes'])}`
- Episode win rate of current recovery overlay vs no recovery: `{ov['episode_win_rate']:.2%}`

## Regime Takeaway

The strongest regime by mean current excess return is `{best_regime}`.

## Asset Form Takeaway

Most frequent ex-post best single asset across episodes: `{best_asset}`.

Use `tables/recovery_recommendation_by_regime.csv` to decide whether recovery is still needed, in which regime, and whether a single-asset or equal-weight formulation looks more defensible.
"""
    (OUT / "README_trigger_lock_recovery_20d_equal_weight_attribution.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    ns = load_namespace()
    panel, candidate_weights, no_rec_weights, daily_state = build_inputs(ns)

    no_rec_out = ns["compute_strategy"](panel, no_rec_weights[ns["ASSETS"]].astype(float), NO_RECOVERY)
    cur_out = ns["compute_strategy"](panel, candidate_weights[ns["ASSETS"]].astype(float), CURRENT)
    perf_df = pd.DataFrame([ns["performance_metrics"](no_rec_out, NO_RECOVERY), ns["performance_metrics"](cur_out, CURRENT)])

    normal_w, _ = ns["normal_weights"](panel)
    episodes = identify_recovery_episodes(panel, daily_state, normal_w)
    ep_attr, daily_excess = compute_episode_tables(panel, episodes, no_rec_out, cur_out, normal_w)
    overall = build_overall_summary(ep_attr, perf_df)
    year_df = yearly_summary(ep_attr)
    regime_df = summarize_group(ep_attr, "start_regime")
    sub_state_df = summarize_group(ep_attr, "start_sub_state")
    pool_df = summarize_group(ep_attr, "selected_assets_for_equal_weight")
    asset_freq = best_asset_frequency(ep_attr)
    rec_df = recommendation_by_regime(regime_df)

    overall.to_csv(TABLE_DIR / "overall_attribution_summary.csv", index=False)
    ep_attr.to_csv(TABLE_DIR / "recovery_episode_attribution.csv", index=False)
    ep_attr.sort_values("current_excess_vs_baseline", ascending=False).head(10).to_csv(TABLE_DIR / "top_recovery_contributors.csv", index=False)
    ep_attr.sort_values("current_excess_vs_baseline").head(10).to_csv(TABLE_DIR / "worst_recovery_drags.csv", index=False)
    year_df.to_csv(TABLE_DIR / "attribution_by_year.csv", index=False)
    regime_df.to_csv(TABLE_DIR / "attribution_by_start_regime.csv", index=False)
    sub_state_df.to_csv(TABLE_DIR / "attribution_by_start_sub_state.csv", index=False)
    pool_df.to_csv(TABLE_DIR / "attribution_by_selected_assets.csv", index=False)
    asset_freq.to_csv(TABLE_DIR / "best_single_asset_frequency.csv", index=False)
    daily_excess.to_csv(TABLE_DIR / "daily_excess_return_during_recovery.csv", index=False)
    perf_df.to_csv(TABLE_DIR / "strategy_no_recovery_vs_current.csv", index=False)
    rec_df.to_csv(TABLE_DIR / "recovery_recommendation_by_regime.csv", index=False)

    plot_outputs(ep_attr, daily_excess, perf_df, regime_df, asset_freq)
    write_readme(overall, regime_df, rec_df, ep_attr)

    ov = overall.iloc[0]
    print("Trigger-lock recovery attribution complete.")
    print(f"recovery episodes: {int(ov['number_of_recovery_episodes'])}")
    print(f"current vs no-recovery final equity diff: {ov['final_equity_diff']:.4f}")
    print(f"current vs no-recovery CAGR diff: {ov['CAGR_diff']:.2%}")
    print(f"current vs no-recovery Sharpe diff: {ov['Sharpe_diff']:.4f}")
    print(f"current vs no-recovery MaxDD diff: {ov['MaxDD_diff']:.2%}")
    if not rec_df.empty:
        print("\nRegime recommendations:")
        print(rec_df.to_string(index=False))
    print("Output path:", OUT)


if __name__ == "__main__":
    main()
