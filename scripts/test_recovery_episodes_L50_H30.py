from __future__ import annotations

import runpy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

STRATEGY = "FLAT_RATE_REFINED_L50_H30"
ASSETS = ["SPY", "GOLD", "CMDTY_FUT", "IEF", "CASH"]
RECOVERY_WINDOW = 120
VOL_DRAG_THRESHOLD = 0.02

INPUT_DIR = ROOT / "results" / "flat_rate_refined_L50_H30"
INPUT_PANEL = INPUT_DIR / "tables" / "daily_returns.csv"
FINAL_PANEL_CANDIDATES = [
    ROOT / "results" / "09_final_strategy" / "mature_regime_hedge_final" / "daily_backtest_panel.csv",
    ROOT / "results" / "mature_regime_hedge_final" / "daily_backtest_panel.csv",
]

OUTPUT_DIR = ROOT / "results" / "recovery_test_L50_H30"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
EPISODE_CHART_DIR = OUTPUT_DIR / "episode_charts"


@dataclass
class RecoveryEpisode:
    episode_id: int
    start_idx: int
    end_idx: int
    previous_stress_start_idx: int
    previous_stress_end_idx: int
    next_stress_start_idx: int | None
    exit_type: str


def ensure_inputs() -> None:
    if not INPUT_PANEL.exists():
        runpy.run_path(str(ROOT / "scripts" / "run_flat_rate_refined_L50_H30.py"), run_name="__main__")


def load_panels() -> pd.DataFrame:
    ensure_inputs()
    if not INPUT_PANEL.exists():
        raise FileNotFoundError(f"Missing fixed L50_H30 panel: {INPUT_PANEL.relative_to(ROOT)}")

    panel = pd.read_csv(INPUT_PANEL, parse_dates=["date"])
    final_path = next((p for p in FINAL_PANEL_CANDIDATES if p.exists()), None)
    if final_path is None:
        raise FileNotFoundError("Missing final strategy panel with asset returns.")

    final = pd.read_csv(final_path, parse_dates=["date"])
    needed = ["date"] + [f"{asset}_return" for asset in ASSETS]
    missing = [c for c in needed if c not in final.columns]
    if missing:
        raise ValueError(f"Final panel is missing required asset return columns: {missing}")

    add_cols = needed.copy()
    for c in ["final_state", "overlay_state", "MATURE_REGIME_HEDGE_FINAL_state"]:
        if c in final.columns:
            add_cols.append(c)
    panel = panel.merge(final[add_cols], on="date", how="left")
    panel = panel.sort_values("date").reset_index(drop=True)

    for asset in ASSETS:
        c = f"{asset}_return"
        panel[c] = pd.to_numeric(panel[c], errors="coerce")

    return panel


def build_stress_flags(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    flat_state = out.get("flat_refined_state", pd.Series("", index=out.index)).fillna("")
    timing = out.get("timing_state", pd.Series("", index=out.index)).fillna("")
    final_state = out.get("final_state", pd.Series("", index=out.index)).fillna("")
    mature_state = out.get("MATURE_REGIME_HEDGE_FINAL_state", pd.Series("", index=out.index)).fillna("")

    # FLAT target weights in the refined strategy are applied with a one-day lag.
    # Use the executed FLAT state for recovery detection so episode starts match
    # actual portfolio exposure rather than the same-day target signal.
    executed_flat_state = flat_state.shift(1).fillna(flat_state)
    non_flat = flat_state.eq("NON_FLAT_BASELINE")
    out["executed_flat_refined_state"] = np.where(non_flat, flat_state, executed_flat_state)

    flat_stress = pd.Series(out["executed_flat_refined_state"], index=out.index).isin(
        ["FLAT_LOW_RATE_STRESS", "FLAT_HIGH_RATE_STRESS"]
    )
    formal_risk = timing.astype(str).str.upper().eq("RISK")
    final_risk = final_state.astype(str).str.upper().eq("FULL_RISK")
    mature_risk = mature_state.astype(str).str.upper().eq("FULL_RISK")

    out["is_stress"] = flat_stress | formal_risk | final_risk | mature_risk
    return out


def find_boolean_runs(flags: pd.Series) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    values = flags.fillna(False).astype(bool).to_numpy()
    for i, value in enumerate(values):
        if value and not in_run:
            start = i
            in_run = True
        elif not value and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(values) - 1))
    return runs


def identify_recovery_episodes(panel: pd.DataFrame) -> list[RecoveryEpisode]:
    stress_runs = find_boolean_runs(panel["is_stress"])
    episodes: list[RecoveryEpisode] = []
    for episode_id, (stress_start, stress_end) in enumerate(stress_runs, start=1):
        recovery_start = stress_end + 1
        if recovery_start >= len(panel):
            continue

        next_stress = next((s for s, _ in stress_runs if s > recovery_start), None)
        max_end = min(recovery_start + RECOVERY_WINDOW - 1, len(panel) - 1)
        if next_stress is not None and next_stress <= max_end:
            recovery_end = next_stress - 1
            exit_type = "interrupted_by_new_stress"
        elif max_end >= len(panel) - 1 and recovery_start + RECOVERY_WINDOW - 1 > len(panel) - 1:
            recovery_end = len(panel) - 1
            exit_type = "truncated_by_data_end"
        else:
            recovery_end = max_end
            exit_type = "completed_120d"

        if recovery_end < recovery_start:
            continue

        episodes.append(
            RecoveryEpisode(
                episode_id=episode_id,
                start_idx=recovery_start,
                end_idx=recovery_end,
                previous_stress_start_idx=stress_start,
                previous_stress_end_idx=stress_end,
                next_stress_start_idx=next_stress,
                exit_type=exit_type,
            )
        )
    return episodes


def get_selected_assets_for_episode(start_row: pd.Series) -> list[str]:
    flat_state = str(start_row.get("executed_flat_refined_state", start_row.get("flat_refined_state", "")))
    if flat_state == "FLAT_LOW_RATE_NORMAL":
        return ["SPY", "CMDTY_FUT", "GOLD"]
    if flat_state == "FLAT_HIGH_RATE_NORMAL":
        return ["CMDTY_FUT", "GOLD"]

    selected = []
    for asset in ASSETS:
        value = start_row.get(f"{STRATEGY}_weight_{asset}", 0.0)
        if pd.notna(value) and float(value) > 1e-6:
            selected.append(asset)
    return selected


def asset_pool_from_weights(row: pd.Series) -> str:
    assets = []
    for asset in ASSETS:
        value = row.get(f"{STRATEGY}_weight_{asset}", 0.0)
        if pd.notna(value) and float(value) > 1e-6:
            assets.append(asset)
    return "/".join(assets) if assets else "NONE"


def summarize_asset_pool_changes(episode_panel: pd.DataFrame) -> str:
    pools = episode_panel.apply(asset_pool_from_weights, axis=1)
    changes = []
    previous = None
    for date, pool in zip(episode_panel["date"], pools):
        if pool != previous:
            changes.append(f"{date.date()}:{pool}")
            previous = pool
    return "; ".join(changes)


def cumulative_from_returns(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod() - 1.0


def compute_episode_cumulative_returns(
    episode: RecoveryEpisode, panel: pd.DataFrame, selected_assets: list[str]
) -> pd.DataFrame:
    ep = panel.iloc[episode.start_idx : episode.end_idx + 1].copy().reset_index(drop=True)
    out = pd.DataFrame({"day": np.arange(1, len(ep) + 1), "date": ep["date"]})

    for asset in selected_assets:
        out[asset] = cumulative_from_returns(ep[f"{asset}_return"])

    inverse_returns = ep[f"{STRATEGY}_return"]
    equal_returns = ep[[f"{asset}_return" for asset in selected_assets]].mean(axis=1) if selected_assets else pd.Series(0.0, index=ep.index)
    out["inverse_vol_portfolio"] = cumulative_from_returns(inverse_returns)
    out["equal_weight_portfolio"] = cumulative_from_returns(equal_returns)
    return out


def interval_bounds(length: int) -> list[tuple[str, int, int, bool]]:
    specs = [("D1_20", 1, 20), ("D21_60", 21, 60), ("D61_120", 61, 120)]
    bounds = []
    for name, start, end in specs:
        if length < start:
            continue
        actual_end = min(end, length)
        bounds.append((name, start, actual_end, actual_end == end))
    return bounds


def product_return(series: pd.Series) -> float:
    return float((1.0 + series.fillna(0.0)).prod() - 1.0)


def compute_interval_returns(
    episode: RecoveryEpisode, panel: pd.DataFrame, selected_assets: list[str]
) -> list[dict]:
    ep = panel.iloc[episode.start_idx : episode.end_idx + 1].copy().reset_index(drop=True)
    rows: list[dict] = []
    for interval_name, start_day, end_day, is_full in interval_bounds(len(ep)):
        sl = ep.iloc[start_day - 1 : end_day]
        cum_sl = ep.iloc[:end_day]
        for asset in selected_assets:
            rows.append(
                {
                    "episode_id": episode.episode_id,
                    "recovery_start_date": ep.loc[0, "date"].date(),
                    "interval_name": interval_name,
                    "actual_start_day": start_day,
                    "actual_end_day": end_day,
                    "actual_num_days": end_day - start_day + 1,
                    "is_full_window": is_full,
                    "portfolio_type": "selected_asset",
                    "asset": asset,
                    "interval_return": product_return(sl[f"{asset}_return"]),
                    "cumulative_return_from_recovery_start": product_return(cum_sl[f"{asset}_return"]),
                }
            )

        inverse_col = f"{STRATEGY}_return"
        rows.append(
            {
                "episode_id": episode.episode_id,
                "recovery_start_date": ep.loc[0, "date"].date(),
                "interval_name": interval_name,
                "actual_start_day": start_day,
                "actual_end_day": end_day,
                "actual_num_days": end_day - start_day + 1,
                "is_full_window": is_full,
                "portfolio_type": "inverse_vol_portfolio",
                "asset": "",
                "interval_return": product_return(sl[inverse_col]),
                "cumulative_return_from_recovery_start": product_return(cum_sl[inverse_col]),
            }
        )

        if selected_assets:
            equal_sl = sl[[f"{asset}_return" for asset in selected_assets]].mean(axis=1)
            equal_cum = cum_sl[[f"{asset}_return" for asset in selected_assets]].mean(axis=1)
        else:
            equal_sl = pd.Series(0.0, index=sl.index)
            equal_cum = pd.Series(0.0, index=cum_sl.index)
        rows.append(
            {
                "episode_id": episode.episode_id,
                "recovery_start_date": ep.loc[0, "date"].date(),
                "interval_name": interval_name,
                "actual_start_day": start_day,
                "actual_end_day": end_day,
                "actual_num_days": end_day - start_day + 1,
                "is_full_window": is_full,
                "portfolio_type": "equal_weight_portfolio",
                "asset": "",
                "interval_return": product_return(equal_sl),
                "cumulative_return_from_recovery_start": product_return(equal_cum),
            }
        )
    return rows


def plot_episode_cumulative_returns(curves: pd.DataFrame, episode_row: dict) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    skip = {"day", "date"}
    for col in curves.columns:
        if col not in skip:
            lw = 2.4 if col in {"inverse_vol_portfolio", "equal_weight_portfolio"} else 1.5
            ax.plot(curves["day"], curves[col], label=col, linewidth=lw)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(
        f"Recovery episode {episode_row['episode_id']}: {episode_row['recovery_start_date']} to "
        f"{episode_row['recovery_end_date']} ({episode_row['episode_length_days']}d)\n"
        f"previous stress: {episode_row['previous_stress_start_date']} to {episode_row['previous_stress_end_date']} | "
        f"{episode_row['start_regime']} / {episode_row['start_flat_sub_state']}"
    )
    ax.set_xlabel("Trading days since recovery start")
    ax.set_ylabel("Cumulative return")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(EPISODE_CHART_DIR / f"recovery_episode_{episode_row['episode_id']}_cumulative_returns.png", dpi=150)
    plt.close(fig)


def plot_episode_interval_returns(interval_df: pd.DataFrame, episode_id: int) -> None:
    sub = interval_df[interval_df["episode_id"].eq(episode_id)].copy()
    if sub.empty:
        return
    sub["label"] = np.where(sub["portfolio_type"].eq("selected_asset"), sub["asset"], sub["portfolio_type"])
    labels = list(dict.fromkeys(sub["label"].tolist()))
    intervals = ["D1_20", "D21_60", "D61_120"]
    x = np.arange(len(intervals))
    width = 0.8 / max(len(labels), 1)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, label in enumerate(labels):
        vals = []
        for interval in intervals:
            row = sub[(sub["interval_name"].eq(interval)) & (sub["label"].eq(label))]
            vals.append(float(row["interval_return"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + (i - (len(labels) - 1) / 2) * width, vals, width=width, label=label)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(intervals)
    ax.set_ylabel("Interval return")
    ax.set_title(f"Recovery episode {episode_id} interval returns")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(EPISODE_CHART_DIR / f"recovery_episode_{episode_id}_interval_returns.png", dpi=150)
    plt.close(fig)


def summarize_recovery_results(interval_df: pd.DataFrame, under_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for interval in ["D1_20", "D21_60", "D61_120"]:
        sub = interval_df[interval_df["interval_name"].eq(interval)]
        inv = sub[sub["portfolio_type"].eq("inverse_vol_portfolio")].set_index("episode_id")
        eq = sub[sub["portfolio_type"].eq("equal_weight_portfolio")].set_index("episode_id")
        selected = sub[sub["portfolio_type"].eq("selected_asset")]
        best = selected.groupby("episode_id")["interval_return"].max()
        common = inv.index.intersection(eq.index)
        diff = eq.loc[common, "interval_return"] - inv.loc[common, "interval_return"]

        values = {}
        for name, frame in [("inverse_vol_portfolio", inv), ("equal_weight_portfolio", eq)]:
            rets = frame["interval_return"]
            values.update(
                {
                    f"{name}_count": int(rets.count()),
                    f"{name}_mean_return": rets.mean(),
                    f"{name}_median_return": rets.median(),
                    f"{name}_std_return": rets.std(),
                    f"{name}_win_rate_positive": (rets > 0).mean() if len(rets) else np.nan,
                    f"{name}_worst_return": rets.min(),
                    f"{name}_best_return": rets.max(),
                }
            )

        ranks = []
        for episode_id in common:
            candidates = selected[selected["episode_id"].eq(episode_id)]["interval_return"].tolist()
            candidates.append(eq.loc[episode_id, "interval_return"])
            inverse_ret = inv.loc[episode_id, "interval_return"]
            ranks.append(1 + sum(v > inverse_ret for v in candidates))

        rows.append(
            {
                "interval_name": interval,
                **values,
                "equal_weight_minus_inverse_vol_mean": diff.mean() if len(diff) else np.nan,
                "equal_weight_minus_inverse_vol_median": diff.median() if len(diff) else np.nan,
                "equal_weight_win_rate_vs_inverse_vol": (diff > 0).mean() if len(diff) else np.nan,
                "best_selected_asset_win_rate_vs_inverse_vol": (best.loc[common] > inv.loc[common, "interval_return"]).mean()
                if len(common)
                else np.nan,
                "inverse_vol_rank_among_selected_and_equal_weight": np.mean(ranks) if ranks else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_underinvestment_diagnostics(episodes_df: pd.DataFrame, interval_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for episode_id, group in interval_df.groupby("episode_id"):
        total = group.sort_values("actual_end_day").groupby(["portfolio_type", "asset"], dropna=False).tail(1)
        inv = total[total["portfolio_type"].eq("inverse_vol_portfolio")]["cumulative_return_from_recovery_start"]
        eq = total[total["portfolio_type"].eq("equal_weight_portfolio")]["cumulative_return_from_recovery_start"]
        selected = total[total["portfolio_type"].eq("selected_asset")]
        inv_total = float(inv.iloc[0]) if not inv.empty else np.nan
        eq_total = float(eq.iloc[0]) if not eq.empty else np.nan
        best_total = selected["cumulative_return_from_recovery_start"].max() if not selected.empty else np.nan

        interval_diffs = []
        for interval in ["D1_20", "D21_60", "D61_120"]:
            sub = group[group["interval_name"].eq(interval)]
            inv_i = sub[sub["portfolio_type"].eq("inverse_vol_portfolio")]["interval_return"]
            eq_i = sub[sub["portfolio_type"].eq("equal_weight_portfolio")]["interval_return"]
            sel_i = sub[sub["portfolio_type"].eq("selected_asset")]["interval_return"]
            if not inv_i.empty:
                inv_val = float(inv_i.iloc[0])
                best_val = float(sel_i.max()) if not sel_i.empty else np.nan
                eq_val = float(eq_i.iloc[0]) if not eq_i.empty else np.nan
                interval_diffs.append((interval, max(eq_val - inv_val, best_val - inv_val)))
        largest_interval = max(interval_diffs, key=lambda x: x[1])[0] if interval_diffs else ""
        suspected = any(interval in {"D1_20", "D21_60"} and diff > VOL_DRAG_THRESHOLD for interval, diff in interval_diffs)
        episode_row = episodes_df[episodes_df["episode_id"].eq(episode_id)].iloc[0]
        rows.append(
            {
                "episode_id": episode_id,
                "recovery_start_date": episode_row["recovery_start_date"],
                "episode_length_days": episode_row["episode_length_days"],
                "inverse_vol_total_return": inv_total,
                "equal_weight_total_return": eq_total,
                "best_selected_asset_total_return": best_total,
                "equal_weight_minus_inverse_vol": eq_total - inv_total,
                "best_asset_minus_inverse_vol": best_total - inv_total,
                "inverse_vol_underperformed_equal_weight": bool(eq_total > inv_total),
                "inverse_vol_underperformed_best_asset": bool(best_total > inv_total),
                "underperformance_largest_interval": largest_interval,
                "suspected_vol_drag": bool(suspected),
            }
        )
    return pd.DataFrame(rows)


def plot_aggregate_figures(interval_df: pd.DataFrame, episodes_df: pd.DataFrame, under_df: pd.DataFrame) -> None:
    order = ["D1_20", "D21_60", "D61_120"]

    agg_rows = []
    for interval in order:
        sub = interval_df[interval_df["interval_name"].eq(interval)]
        for asset in ASSETS:
            rets = sub.loc[sub["portfolio_type"].eq("selected_asset") & sub["asset"].eq(asset), "interval_return"]
            if rets.empty:
                continue
            agg_rows.append({"interval_name": interval, "portfolio": asset, "mean": rets.mean(), "median": rets.median()})
        for label in ["inverse_vol_portfolio", "equal_weight_portfolio"]:
            rets = sub.loc[sub["portfolio_type"].eq(label), "interval_return"]
            agg_rows.append({"interval_name": interval, "portfolio": label, "mean": rets.mean(), "median": rets.median()})
    agg = pd.DataFrame(agg_rows)

    for metric, fname in [("mean", "recovery_interval_returns_average.png"), ("median", "recovery_interval_returns_median.png")]:
        fig, ax = plt.subplots(figsize=(9, 5))
        pivot = agg.pivot(index="interval_name", columns="portfolio", values=metric).reindex(order)
        columns = [c for c in ASSETS + ["inverse_vol_portfolio", "equal_weight_portfolio"] if c in pivot.columns]
        pivot = pivot[columns]
        pivot.plot(kind="bar", ax=ax)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel(f"{metric.title()} interval return")
        ax.set_title(f"Recovery interval {metric} returns by selected asset / portfolio")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / fname, dpi=150)
        plt.close(fig)

    summary = summarize_recovery_results(interval_df, under_df)
    win_rows = []
    for interval in order:
        sub = interval_df[interval_df["interval_name"].eq(interval)]
        inv = sub[sub["portfolio_type"].eq("inverse_vol_portfolio")].set_index("episode_id")["interval_return"]
        for asset in ASSETS:
            asset_ret = sub[sub["portfolio_type"].eq("selected_asset") & sub["asset"].eq(asset)].set_index("episode_id")[
                "interval_return"
            ]
            common = inv.index.intersection(asset_ret.index)
            if len(common):
                win_rows.append(
                    {
                        "interval_name": interval,
                        "series": asset,
                        "win_rate": (asset_ret.loc[common] > inv.loc[common]).mean(),
                    }
                )
        eq = sub[sub["portfolio_type"].eq("equal_weight_portfolio")].set_index("episode_id")["interval_return"]
        common = inv.index.intersection(eq.index)
        if len(common):
            win_rows.append(
                {
                    "interval_name": interval,
                    "series": "equal_weight_portfolio",
                    "win_rate": (eq.loc[common] > inv.loc[common]).mean(),
                }
            )
        selected = sub[sub["portfolio_type"].eq("selected_asset")]
        best = selected.groupby("episode_id")["interval_return"].max()
        common = inv.index.intersection(best.index)
        if len(common):
            win_rows.append(
                {
                    "interval_name": interval,
                    "series": "best_selected_asset",
                    "win_rate": (best.loc[common] > inv.loc[common]).mean(),
                }
            )

    win = pd.DataFrame(win_rows).pivot(index="interval_name", columns="series", values="win_rate").reindex(order)
    columns = [c for c in ASSETS + ["equal_weight_portfolio", "best_selected_asset"] if c in win.columns]
    win = win[columns]
    fig, ax = plt.subplots(figsize=(10, 5))
    win.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Win rate vs inverse-vol")
    ax.set_title("Recovery interval win rate vs inverse-vol by asset / portfolio")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "recovery_interval_win_rate_vs_inverse_vol.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    episodes_df["episode_length_days"].hist(ax=ax, bins=20)
    ax.set_title("Recovery episode length distribution")
    ax.set_xlabel("Episode length days")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "recovery_episode_length_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(under_df["episode_id"], under_df["equal_weight_minus_inverse_vol"])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Equal-weight minus inverse-vol by recovery episode")
    ax.set_xlabel("Episode ID")
    ax.set_ylabel("Return difference")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "recovery_equal_weight_minus_inverse_vol_by_episode.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(under_df["episode_id"], under_df["best_asset_minus_inverse_vol"])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Best selected asset minus inverse-vol by recovery episode")
    ax.set_xlabel("Episode ID")
    ax.set_ylabel("Return difference")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "recovery_best_asset_minus_inverse_vol_by_episode.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 2.5))
    ax.scatter(pd.to_datetime(episodes_df["recovery_start_date"]), np.ones(len(episodes_df)), s=25)
    ax.set_yticks([])
    ax.set_title("Recovery start dates timeline")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "recovery_start_dates_timeline.png", dpi=150)
    plt.close(fig)


def write_readme(episodes_df: pd.DataFrame, summary_df: pd.DataFrame, under_df: pd.DataFrame) -> None:
    eq_minus = under_df["equal_weight_minus_inverse_vol"].mean()
    suspected_count = int(under_df["suspected_vol_drag"].sum())
    d1 = summary_df.loc[summary_df["interval_name"].eq("D1_20"), "equal_weight_win_rate_vs_inverse_vol"]
    d21 = summary_df.loc[summary_df["interval_name"].eq("D21_60"), "equal_weight_win_rate_vs_inverse_vol"]
    d61 = summary_df.loc[summary_df["interval_name"].eq("D61_120"), "equal_weight_win_rate_vs_inverse_vol"]
    text = f"""# Recovery Test: L50_H30

## Purpose

This independent diagnostic fixes the refined baseline at **L50_H30** and studies recovery episodes after stress exits. The goal is to test whether the 120-day inverse-volatility estimator underinvests in rebound assets because the trailing window still contains crisis-period volatility.

## Fixed Strategy Under Test

- Strategy: `FLAT_RATE_REFINED_L50_H30`
- `FLAT_LOW_RATE_STRESS`: 50% GOLD / 50% IEF
- `FLAT_HIGH_RATE_STRESS`: 30% GOLD / 70% CASH
- Normal FLAT rules, non-FLAT rules, transaction costs, monthly rebalance, and 120-day inverse-volatility logic are unchanged.

## Recovery Episode Definition

A recovery episode starts when the previous trading day is stress and the current day is non-stress. Each episode lasts up to 120 trading days, but is cut short by a new stress entry or by the end of the data.

## Summary

- Recovery episodes: {len(episodes_df)}
- Average episode length: {episodes_df['episode_length_days'].mean():.1f} trading days
- Completed 120-day episodes: {(episodes_df['recovery_exit_type'] == 'completed_120d').sum()}
- Interrupted by new stress: {(episodes_df['recovery_exit_type'] == 'interrupted_by_new_stress').sum()}
- Average equal-weight minus inverse-vol total return: {eq_minus:.2%}
- Suspected vol-drag episodes: {suspected_count}

## Interval Diagnostics

| Interval | Equal-weight win rate vs inverse-vol | Best selected asset win rate vs inverse-vol |
|---|---:|---:|
| D1_20 | {(float(d1.iloc[0]) if not d1.empty else np.nan):.2%} | {summary_df.loc[summary_df['interval_name'].eq('D1_20'), 'best_selected_asset_win_rate_vs_inverse_vol'].iloc[0]:.2%} |
| D21_60 | {(float(d21.iloc[0]) if not d21.empty else np.nan):.2%} | {summary_df.loc[summary_df['interval_name'].eq('D21_60'), 'best_selected_asset_win_rate_vs_inverse_vol'].iloc[0]:.2%} |
| D61_120 | {(float(d61.iloc[0]) if not d61.empty else np.nan):.2%} | {summary_df.loc[summary_df['interval_name'].eq('D61_120'), 'best_selected_asset_win_rate_vs_inverse_vol'].iloc[0]:.2%} |

## Interpretation

This diagnostic should not be read as a new strategy rule. It identifies whether a post-stress recovery window exists and whether the current inverse-volatility portfolio lags equal-weight selected assets during early recovery. If outperformance is concentrated in D1_20 or D21_60 and fades by D61_120, a future recovery overlay would likely need to be short-lived rather than 120 days.

## Outputs

- `tables/recovery_episodes.csv`
- `tables/recovery_interval_returns_long.csv`
- `tables/recovery_interval_summary.csv`
- `tables/recovery_underinvestment_diagnostics.csv`
- `episode_charts/`
- `figures/`
"""
    (OUTPUT_DIR / "README_recovery_test_L50_H30.md").write_text(text, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    EPISODE_CHART_DIR.mkdir(parents=True, exist_ok=True)

    panel = build_stress_flags(load_panels())
    episodes = identify_recovery_episodes(panel)
    if not episodes:
        raise RuntimeError("No recovery episodes were identified.")

    episode_rows: list[dict] = []
    interval_rows: list[dict] = []

    for episode in episodes:
        ep_panel = panel.iloc[episode.start_idx : episode.end_idx + 1].copy()
        start_row = panel.iloc[episode.start_idx]
        selected_assets = get_selected_assets_for_episode(start_row)
        if not selected_assets:
            selected_assets = ["SPY"]

        next_stress_date = (
            panel.iloc[episode.next_stress_start_idx]["date"].date() if episode.next_stress_start_idx is not None else ""
        )
        episode_row = {
            "episode_id": episode.episode_id,
            "recovery_start_date": panel.iloc[episode.start_idx]["date"].date(),
            "recovery_end_date": panel.iloc[episode.end_idx]["date"].date(),
            "episode_length_days": episode.end_idx - episode.start_idx + 1,
            "previous_stress_start_date": panel.iloc[episode.previous_stress_start_idx]["date"].date(),
            "previous_stress_end_date": panel.iloc[episode.previous_stress_end_idx]["date"].date(),
            "next_stress_start_date": next_stress_date,
            "start_regime": start_row.get("macro_regime_confirmed", ""),
            "start_flat_sub_state": start_row.get("executed_flat_refined_state", start_row.get("flat_refined_state", "")),
            "recovery_exit_type": episode.exit_type,
            "selected_assets_at_start": "/".join(selected_assets),
            "selected_assets_during_episode": summarize_asset_pool_changes(ep_panel),
            "inverse_vol_window": RECOVERY_WINDOW,
        }
        episode_rows.append(episode_row)
        curves = compute_episode_cumulative_returns(episode, panel, selected_assets)
        plot_episode_cumulative_returns(curves, episode_row)
        interval_rows.extend(compute_interval_returns(episode, panel, selected_assets))

    episodes_df = pd.DataFrame(episode_rows)
    interval_df = pd.DataFrame(interval_rows)
    under_df = build_underinvestment_diagnostics(episodes_df, interval_df)
    summary_df = summarize_recovery_results(interval_df, under_df)

    for episode_id in episodes_df["episode_id"]:
        plot_episode_interval_returns(interval_df, int(episode_id))

    plot_aggregate_figures(interval_df, episodes_df, under_df)
    write_readme(episodes_df, summary_df, under_df)

    episodes_df.to_csv(TABLE_DIR / "recovery_episodes.csv", index=False)
    interval_df.to_csv(TABLE_DIR / "recovery_interval_returns_long.csv", index=False)
    summary_df.to_csv(TABLE_DIR / "recovery_interval_summary.csv", index=False)
    under_df.to_csv(TABLE_DIR / "recovery_underinvestment_diagnostics.csv", index=False)

    d1 = summary_df.loc[summary_df["interval_name"].eq("D1_20"), "equal_weight_win_rate_vs_inverse_vol"]
    d21 = summary_df.loc[summary_df["interval_name"].eq("D21_60"), "equal_weight_win_rate_vs_inverse_vol"]
    d61 = summary_df.loc[summary_df["interval_name"].eq("D61_120"), "equal_weight_win_rate_vs_inverse_vol"]

    print("Recovery test L50_H30 complete.")
    print(f"output_dir: {OUTPUT_DIR.relative_to(ROOT)}")
    print(f"number of recovery episodes: {len(episodes_df)}")
    print(f"average episode length: {episodes_df['episode_length_days'].mean():.1f}")
    print(f"completed 120d episodes: {(episodes_df['recovery_exit_type'] == 'completed_120d').sum()}")
    print(f"interrupted by new stress: {(episodes_df['recovery_exit_type'] == 'interrupted_by_new_stress').sum()}")
    print(f"average inverse-vol recovery return: {under_df['inverse_vol_total_return'].mean():.4f}")
    print(f"average equal-weight recovery return: {under_df['equal_weight_total_return'].mean():.4f}")
    print(f"average equal_weight_minus_inverse_vol: {under_df['equal_weight_minus_inverse_vol'].mean():.4f}")
    print(f"D1_20 equal-weight win rate vs inverse-vol: {(float(d1.iloc[0]) if not d1.empty else np.nan):.2%}")
    print(f"D21_60 equal-weight win rate vs inverse-vol: {(float(d21.iloc[0]) if not d21.empty else np.nan):.2%}")
    print(f"D61_120 equal-weight win rate vs inverse-vol: {(float(d61.iloc[0]) if not d61.empty else np.nan):.2%}")
    print(f"suspected vol drag episode count: {int(under_df['suspected_vol_drag'].sum())}")


if __name__ == "__main__":
    main()
