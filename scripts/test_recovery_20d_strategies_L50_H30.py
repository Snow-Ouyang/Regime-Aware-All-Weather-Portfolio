from __future__ import annotations

import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ["SPY", "GOLD", "CMDTY_FUT", "IEF", "CASH"]
REFINED = "FLAT_RATE_REFINED_L50_H30"
MATURE = "MATURE_REGIME_HEDGE_FINAL"
REC_SPY = "RECOVERY_20D_SPY"
REC_EW = "RECOVERY_20D_EQUAL_WEIGHT"
STRATEGIES = [MATURE, REFINED, REC_SPY, REC_EW]

RECOVERY_WINDOW = 20
ONE_WAY_COST_BPS = 5.0
UNDERPERFORMANCE_THRESHOLD = -0.02

INPUT_DIR = ROOT / "results" / "flat_rate_refined_L50_H30"
INPUT_PANEL = INPUT_DIR / "tables" / "daily_returns.csv"
FINAL_PANEL_CANDIDATES = [
    ROOT / "results" / "09_final_strategy" / "mature_regime_hedge_final" / "daily_backtest_panel.csv",
    ROOT / "results" / "mature_regime_hedge_final" / "daily_backtest_panel.csv",
]

OUTPUT_DIR = ROOT / "results" / "recovery_20d_strategy_test_L50_H30"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"


def ensure_inputs() -> None:
    if not INPUT_PANEL.exists():
        runpy.run_path(str(ROOT / "scripts" / "run_flat_rate_refined_L50_H30.py"), run_name="__main__")


def load_panel() -> pd.DataFrame:
    ensure_inputs()
    if not INPUT_PANEL.exists():
        raise FileNotFoundError(f"Missing {INPUT_PANEL.relative_to(ROOT)}")
    refined = pd.read_csv(INPUT_PANEL, parse_dates=["date"])

    final_path = next((p for p in FINAL_PANEL_CANDIDATES if p.exists()), None)
    if final_path is None:
        raise FileNotFoundError("Missing final strategy daily panel.")
    final = pd.read_csv(final_path, parse_dates=["date"])

    needed = ["date"] + [f"{a}_return" for a in ASSETS]
    needed += [f"{MATURE}_weight_{a}" for a in ASSETS]
    needed += [f"{MATURE}_{c}" for c in ["return", "nav", "drawdown", "turnover", "transaction_cost"]]
    for c in ["final_state", "overlay_state", f"{MATURE}_state"]:
        if c in final.columns:
            needed.append(c)
    missing = [c for c in needed if c not in final.columns]
    if missing:
        raise ValueError(f"Final panel missing required columns: {missing}")

    panel = refined.merge(final[needed], on="date", how="left").sort_values("date").reset_index(drop=True)
    for asset in ASSETS:
        panel[f"{asset}_return"] = pd.to_numeric(panel[f"{asset}_return"], errors="coerce")
    return panel


def build_stress_flags(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    flat_state = out.get("flat_refined_state", pd.Series("", index=out.index)).fillna("")
    executed_flat = flat_state.shift(1).fillna(flat_state)
    out["executed_flat_refined_state"] = np.where(flat_state.eq("NON_FLAT_BASELINE"), flat_state, executed_flat)

    flat_stress = pd.Series(out["executed_flat_refined_state"], index=out.index).isin(
        ["FLAT_LOW_RATE_STRESS", "FLAT_HIGH_RATE_STRESS"]
    )
    timing = out.get("timing_state", pd.Series("", index=out.index)).fillna("").astype(str).str.upper()
    final_state = out.get("final_state", pd.Series("", index=out.index)).fillna("").astype(str).str.upper()
    mature_state = out.get(f"{MATURE}_state", pd.Series("", index=out.index)).fillna("").astype(str).str.upper()
    out["is_stress"] = flat_stress | timing.eq("RISK") | final_state.eq("FULL_RISK") | mature_state.eq("FULL_RISK")
    return out


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    clean = {a: max(0.0, float(weights.get(a, 0.0))) for a in ASSETS}
    total = sum(clean.values())
    if total <= 0:
        return clean
    return {a: clean[a] / total for a in ASSETS}


def get_base_weights(row: pd.Series) -> dict[str, float]:
    return normalize_weights({a: row.get(f"{REFINED}_weight_{a}", 0.0) for a in ASSETS})


def selected_assets_for_row(row: pd.Series) -> list[str]:
    state = str(row.get("executed_flat_refined_state", row.get("flat_refined_state", "")))
    if state == "FLAT_LOW_RATE_NORMAL":
        assets = ["SPY", "CMDTY_FUT", "GOLD"]
    elif state == "FLAT_HIGH_RATE_NORMAL":
        assets = ["CMDTY_FUT", "GOLD"]
    else:
        weights = get_base_weights(row)
        assets = [a for a, w in weights.items() if w > 1e-6]

    valid = [a for a in assets if pd.notna(row.get(f"{a}_return", np.nan))]
    return valid


def build_recovery_flags(panel: pd.DataFrame) -> pd.Series:
    active = []
    remaining = 0
    was_stress = False
    for _, row in panel.iterrows():
        is_stress = bool(row["is_stress"])
        if is_stress:
            remaining = 0
            active.append(False)
        else:
            if was_stress:
                remaining = RECOVERY_WINDOW
            active.append(remaining > 0)
            if remaining > 0:
                remaining -= 1
        was_stress = is_stress
    return pd.Series(active, index=panel.index)


def build_override_weights(panel: pd.DataFrame, mode: str) -> pd.DataFrame:
    recovery = build_recovery_flags(panel)
    weights = pd.DataFrame(0.0, index=panel.index, columns=ASSETS)
    for i, row in panel.iterrows():
        base = get_base_weights(row)
        if bool(row["is_stress"]) or not bool(recovery.loc[i]):
            alloc = base
        elif mode == REC_SPY:
            alloc = {a: 0.0 for a in ASSETS}
            alloc["SPY"] = 1.0
        elif mode == REC_EW:
            selected = selected_assets_for_row(row)
            if selected:
                alloc = {a: 0.0 for a in ASSETS}
                for asset in selected:
                    alloc[asset] = 1.0 / len(selected)
            else:
                alloc = base
        else:
            raise ValueError(f"Unknown mode: {mode}")
        weights.loc[i, ASSETS] = pd.Series(normalize_weights(alloc))
    return weights


def compute_strategy_from_weights(panel: pd.DataFrame, weights: pd.DataFrame, strategy: str) -> pd.DataFrame:
    returns = panel[[f"{a}_return" for a in ASSETS]].rename(columns={f"{a}_return": a for a in ASSETS}).fillna(0.0)
    gross = (weights[ASSETS] * returns[ASSETS]).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    cost = 0.5 * turnover * ONE_WAY_COST_BPS / 10000.0
    net = gross - cost
    nav = (1.0 + net).cumprod()
    dd = nav / nav.cummax() - 1.0

    out = pd.DataFrame(
        {
            f"{strategy}_return": net,
            f"{strategy}_nav": nav,
            f"{strategy}_drawdown": dd,
            f"{strategy}_turnover": turnover,
            f"{strategy}_transaction_cost": cost,
        }
    )
    for asset in ASSETS:
        out[f"{strategy}_weight_{asset}"] = weights[asset]
    return out


def max_drawdown_from_returns(returns: pd.Series) -> float:
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def worst_12m_return(nav: pd.Series) -> float:
    return float((nav / nav.shift(252) - 1.0).min())


def sortino_ratio(returns: pd.Series, ann_ret: float) -> float:
    downside = returns[returns < 0].std() * np.sqrt(252)
    return float(ann_ret / downside) if downside and pd.notna(downside) else np.nan


def performance_metrics(panel: pd.DataFrame, strategy: str) -> dict:
    r = panel[f"{strategy}_return"].fillna(0.0)
    nav = panel[f"{strategy}_nav"]
    n = len(r)
    ann_ret = float(nav.iloc[-1] ** (252 / n) - 1.0)
    ann_vol = float(r.std() * np.sqrt(252))
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else np.nan
    maxdd = float(panel[f"{strategy}_drawdown"].min())
    calmar = float(ann_ret / abs(maxdd)) if maxdd < 0 else np.nan
    turnover = float(panel.get(f"{strategy}_turnover", pd.Series(0, index=panel.index)).sum())
    cost = float(panel.get(f"{strategy}_transaction_cost", pd.Series(0, index=panel.index)).sum())
    trades = int((panel.get(f"{strategy}_turnover", pd.Series(0, index=panel.index)) > 1e-8).sum())
    return {
        "strategy": strategy,
        "CAGR": ann_ret,
        "annualized_volatility": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino_ratio(r, ann_ret),
        "MaxDD": maxdd,
        "Calmar": calmar,
        "final_equity": float(nav.iloc[-1]),
        "win_rate": float((r > 0).mean()),
        "worst_day_or_month": float(r.min()),
        "worst_12m_return": worst_12m_return(nav),
        "turnover": turnover,
        "total_transaction_cost": cost,
        "number_of_trades": trades,
    }


def identify_recovery_episodes(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    episode_id = 1
    was_stress = False
    i = 0
    while i < len(panel):
        is_stress = bool(panel.loc[i, "is_stress"])
        if (not is_stress) and was_stress:
            start = i
            max_end = min(start + RECOVERY_WINDOW - 1, len(panel) - 1)
            next_stress = None
            for j in range(start, max_end + 1):
                if bool(panel.loc[j, "is_stress"]):
                    next_stress = j
                    break
            if next_stress is not None:
                end = next_stress - 1
                exit_type = "interrupted_by_new_stress"
            elif max_end == len(panel) - 1 and start + RECOVERY_WINDOW - 1 > len(panel) - 1:
                end = len(panel) - 1
                exit_type = "truncated_by_data_end"
            else:
                end = max_end
                exit_type = "completed_20d"
            if end >= start:
                selected = selected_assets_for_row(panel.loc[start])
                rows.append(
                    {
                        "episode_id": episode_id,
                        "start_idx": start,
                        "end_idx": end,
                        "recovery_start_date": panel.loc[start, "date"],
                        "recovery_end_date": panel.loc[end, "date"],
                        "episode_length_days": end - start + 1,
                        "exit_type": exit_type,
                        "start_regime": panel.loc[start, "macro_regime_confirmed"],
                        "start_sub_state": panel.loc[start, "executed_flat_refined_state"],
                        "selected_assets_for_equal_weight": "/".join(selected),
                    }
                )
                episode_id += 1
        was_stress = is_stress
        i += 1
    return pd.DataFrame(rows)


def compute_episode_strategy_performance(panel: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, ep in episodes.iterrows():
        sl = panel.iloc[int(ep["start_idx"]) : int(ep["end_idx"]) + 1]
        row = ep.drop(labels=["start_idx", "end_idx"]).to_dict()
        for strategy in [REFINED, REC_SPY, REC_EW]:
            ret = (1.0 + sl[f"{strategy}_return"].fillna(0.0)).prod() - 1.0
            row[f"{strategy}_return"] = float(ret)
            row[f"{strategy}_maxdd"] = max_drawdown_from_returns(sl[f"{strategy}_return"])
        row[f"{REC_SPY}_minus_refined"] = row[f"{REC_SPY}_return"] - row[f"{REFINED}_return"]
        row[f"{REC_EW}_minus_refined"] = row[f"{REC_EW}_return"] - row[f"{REFINED}_return"]
        rows.append(row)
    return pd.DataFrame(rows)


def recovery_strategy_summary(ep_perf: pd.DataFrame) -> pd.DataFrame:
    rows = {
        "number_of_recovery_episodes": len(ep_perf),
        "average_episode_length": ep_perf["episode_length_days"].mean(),
        "completed_20d_count": int(ep_perf["exit_type"].eq("completed_20d").sum()),
        "interrupted_by_new_stress_count": int(ep_perf["exit_type"].eq("interrupted_by_new_stress").sum()),
        "truncated_by_data_end_count": int(ep_perf["exit_type"].eq("truncated_by_data_end").sum()),
    }
    for strategy in [REC_SPY, REC_EW]:
        diff = ep_perf[f"{strategy}_minus_refined"]
        rows[f"{strategy}_mean_excess_return_vs_refined"] = diff.mean()
        rows[f"{strategy}_median_excess_return_vs_refined"] = diff.median()
        rows[f"{strategy}_win_rate_vs_refined"] = (diff > 0).mean()
    mean_best = max([REC_SPY, REC_EW], key=lambda s: rows[f"{s}_mean_excess_return_vs_refined"])
    median_best = max([REC_SPY, REC_EW], key=lambda s: rows[f"{s}_median_excess_return_vs_refined"])
    win_best = max([REC_SPY, REC_EW], key=lambda s: rows[f"{s}_win_rate_vs_refined"])
    rows["best_recovery_version_by_mean"] = mean_best
    rows["best_recovery_version_by_median"] = median_best
    rows["best_recovery_version_by_win_rate"] = win_best
    return pd.DataFrame([rows])


def underperformance_cases(ep_perf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in ep_perf.iterrows():
        for strategy in [REC_SPY, REC_EW]:
            diff = row[f"{strategy}_minus_refined"]
            if diff < UNDERPERFORMANCE_THRESHOLD:
                rows.append(
                    {
                        "episode_id": row["episode_id"],
                        "recovery_start_date": row["recovery_start_date"],
                        "recovery_end_date": row["recovery_end_date"],
                        "start_regime": row["start_regime"],
                        "start_sub_state": row["start_sub_state"],
                        "strategy": strategy,
                        "excess_return_vs_refined": diff,
                        "selected_assets": row["selected_assets_for_equal_weight"],
                        "notes": "quick re-stress" if row["exit_type"] == "interrupted_by_new_stress" else "",
                    }
                )
    return pd.DataFrame(rows)


def write_long_outputs(panel: pd.DataFrame) -> None:
    weight_rows = []
    return_rows = []
    for strategy in STRATEGIES:
        for asset in ASSETS:
            col = f"{strategy}_weight_{asset}"
            if col in panel.columns:
                weight_rows.append(pd.DataFrame({"date": panel["date"], "strategy": strategy, "asset": asset, "weight": panel[col]}))
        return_rows.append(
            pd.DataFrame(
                {
                    "date": panel["date"],
                    "strategy": strategy,
                    "daily_return": panel[f"{strategy}_return"],
                    "equity": panel[f"{strategy}_nav"],
                    "drawdown": panel[f"{strategy}_drawdown"],
                    "turnover": panel.get(f"{strategy}_turnover", pd.Series(0, index=panel.index)),
                    "transaction_cost": panel.get(f"{strategy}_transaction_cost", pd.Series(0, index=panel.index)),
                }
            )
        )
    pd.concat(weight_rows, ignore_index=True).to_csv(TABLE_DIR / "daily_weights_all_strategies.csv", index=False)
    pd.concat(return_rows, ignore_index=True).to_csv(TABLE_DIR / "daily_returns_all_strategies.csv", index=False)


def plot_outputs(panel: pd.DataFrame, ep_perf: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for strategy in STRATEGIES:
        ax.plot(panel["date"], panel[f"{strategy}_nav"], label=strategy)
    ax.set_yscale("log")
    ax.set_title("Recovery 20D strategy test: equity curve")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "equity_curve_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for strategy in STRATEGIES:
        ax.plot(panel["date"], panel[f"{strategy}_drawdown"], label=strategy)
    ax.set_title("Recovery 20D strategy test: drawdown")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "drawdown_curve_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for strategy in STRATEGIES:
        rolling = panel[f"{strategy}_nav"] / panel[f"{strategy}_nav"].shift(252) - 1.0
        ax.plot(panel["date"], rolling, label=strategy)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Rolling 12M return comparison")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "rolling_12m_return_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(ep_perf))
    width = 0.4
    ax.bar(x - width / 2, ep_perf[f"{REC_SPY}_minus_refined"], width=width, label=REC_SPY)
    ax.bar(x + width / 2, ep_perf[f"{REC_EW}_minus_refined"], width=width, label=REC_EW)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Recovery episode")
    ax.set_ylabel("Excess return vs refined baseline")
    ax.set_title("Recovery episode excess return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "recovery_episode_excess_return_bar.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    values = {
        "mean excess": [
            summary[f"{REC_SPY}_mean_excess_return_vs_refined"].iloc[0],
            summary[f"{REC_EW}_mean_excess_return_vs_refined"].iloc[0],
        ],
        "median excess": [
            summary[f"{REC_SPY}_median_excess_return_vs_refined"].iloc[0],
            summary[f"{REC_EW}_median_excess_return_vs_refined"].iloc[0],
        ],
        "win rate": [
            summary[f"{REC_SPY}_win_rate_vs_refined"].iloc[0],
            summary[f"{REC_EW}_win_rate_vs_refined"].iloc[0],
        ],
    }
    pd.DataFrame(values, index=[REC_SPY, REC_EW]).plot(kind="bar", ax=ax)
    ax.set_title("Recovery strategy summary vs refined baseline")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "recovery_strategy_summary_bar.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(panel["date"], 0, panel["is_stress"].astype(int), step="pre", alpha=0.35, label="stress")
    ax.fill_between(panel["date"], 1.1, 1.1 + panel["recovery_override_active"].astype(int), step="pre", alpha=0.35, label="recovery override")
    ax.set_ylim(-0.1, 2.3)
    ax.set_yticks([0.5, 1.6])
    ax.set_yticklabels(["stress", "recovery"])
    ax.set_title("Stress and recovery override periods")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "recovery_override_weights_timeline.png", dpi=160)
    plt.close(fig)


def write_readme(perf: pd.DataFrame, summary: pd.DataFrame) -> None:
    best_sharpe = perf.sort_values("Sharpe", ascending=False).iloc[0]["strategy"]
    best_calmar = perf.sort_values("Calmar", ascending=False).iloc[0]["strategy"]
    best_final = perf.sort_values("final_equity", ascending=False).iloc[0]["strategy"]
    text = f"""# Recovery 20D Strategy Test: L50_H30

## Background

The prior recovery diagnostic showed that equal-weight selected assets had a high win rate versus inverse-volatility during D1-20 after stress exits, while the edge did not persist through D21-60 or D61-120. This experiment therefore tests only a 20-trading-day recovery override.

## Strategy Definitions

- `FLAT_RATE_REFINED_L50_H30`: fixed refined baseline.
- `RECOVERY_20D_SPY`: after stress exits, hold 100% SPY for up to 20 trading days unless stress returns.
- `RECOVERY_20D_EQUAL_WEIGHT`: after stress exits, hold equal-weight selected assets for up to 20 trading days unless stress returns.

Priority is `stress allocation > recovery override > refined baseline normal allocation`.

## Main Result

| Strategy | CAGR | Sharpe | MaxDD | Calmar | Final Equity |
|---|---:|---:|---:|---:|---:|
"""
    for _, row in perf.iterrows():
        text += f"| {row['strategy']} | {row['CAGR']:.2%} | {row['Sharpe']:.3f} | {row['MaxDD']:.2%} | {row['Calmar']:.3f} | {row['final_equity']:.3f} |\n"
    text += f"""
## Recovery Episode Summary

- Recovery episodes: {int(summary['number_of_recovery_episodes'].iloc[0])}
- Average episode length: {summary['average_episode_length'].iloc[0]:.1f} trading days
- Completed 20D episodes: {int(summary['completed_20d_count'].iloc[0])}
- Interrupted by new stress: {int(summary['interrupted_by_new_stress_count'].iloc[0])}
- `RECOVERY_20D_SPY` win rate vs refined: {summary[f'{REC_SPY}_win_rate_vs_refined'].iloc[0]:.2%}
- `RECOVERY_20D_EQUAL_WEIGHT` win rate vs refined: {summary[f'{REC_EW}_win_rate_vs_refined'].iloc[0]:.2%}

## Interpretation

This is still a diagnostic strategy test, not a change to the main strategy. If a recovery override improves CAGR and final equity without worsening MaxDD or materially degrading Sharpe, it may be a candidate for a more filtered follow-up test. If improvement is concentrated in a few episodes or turnover rises materially, the cleaner conclusion is to keep L50_H30 unchanged.

Best by Sharpe: `{best_sharpe}`. Best by Calmar: `{best_calmar}`. Best by final equity: `{best_final}`.
"""
    (OUTPUT_DIR / "README_recovery_20d_strategy_test.md").write_text(text, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    panel = build_stress_flags(load_panel())
    panel["recovery_override_active"] = build_recovery_flags(panel)

    # Existing strategy columns.
    for asset in ASSETS:
        panel[f"{REFINED}_weight_{asset}"] = panel[f"{REFINED}_weight_{asset}"].astype(float)

    # Build override variants.
    for strategy in [REC_SPY, REC_EW]:
        weights = build_override_weights(panel, strategy)
        computed = compute_strategy_from_weights(panel, weights, strategy)
        panel = pd.concat([panel, computed], axis=1)

    # Recompute refined long fields are already present; mature comes from final panel.
    perf = pd.DataFrame([performance_metrics(panel, s) for s in STRATEGIES])
    perf.to_csv(TABLE_DIR / "performance_comparison.csv", index=False)

    episodes = identify_recovery_episodes(panel)
    ep_perf = compute_episode_strategy_performance(panel, episodes)
    summary = recovery_strategy_summary(ep_perf)
    under = underperformance_cases(ep_perf)

    ep_perf.to_csv(TABLE_DIR / "recovery_episode_strategy_performance.csv", index=False)
    summary.to_csv(TABLE_DIR / "recovery_strategy_summary.csv", index=False)
    under.to_csv(TABLE_DIR / "recovery_underperformance_cases.csv", index=False)
    write_long_outputs(panel)
    plot_outputs(panel, ep_perf, summary)
    write_readme(perf, summary)

    print("Recovery 20D strategy test complete.")
    print(f"output_dir: {OUTPUT_DIR.relative_to(ROOT)}")
    print(perf[["strategy", "CAGR", "Sharpe", "MaxDD", "Calmar", "final_equity", "turnover", "total_transaction_cost"]].to_string(index=False))
    print(f"number of recovery episodes: {len(ep_perf)}")
    print(f"{REC_SPY} win rate vs refined: {summary[f'{REC_SPY}_win_rate_vs_refined'].iloc[0]:.2%}")
    print(f"{REC_EW} win rate vs refined: {summary[f'{REC_EW}_win_rate_vs_refined'].iloc[0]:.2%}")
    print(f"best recovery version by Sharpe: {perf.sort_values('Sharpe', ascending=False).iloc[0]['strategy']}")
    print(f"best recovery version by Calmar: {perf.sort_values('Calmar', ascending=False).iloc[0]['strategy']}")
    print(f"best recovery version by final equity: {perf.sort_values('final_equity', ascending=False).iloc[0]['strategy']}")
    refined_dd = perf.loc[perf["strategy"].eq(REFINED), "MaxDD"].iloc[0]
    for strategy in [REC_SPY, REC_EW]:
        dd = perf.loc[perf["strategy"].eq(strategy), "MaxDD"].iloc[0]
        print(f"{strategy} MaxDD worsened vs refined: {dd < refined_dd}")


if __name__ == "__main__":
    main()
