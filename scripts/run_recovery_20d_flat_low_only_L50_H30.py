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
GLOBAL_RECOVERY = "RECOVERY_20D_EQUAL_WEIGHT"
STRATEGY = "RECOVERY_20D_EQUAL_WEIGHT_FLAT_LOW_ONLY"
SPY_STRATEGY = "RECOVERY_20D_SPY_FLAT_LOW_ONLY"
STRATEGIES = [MATURE, REFINED, GLOBAL_RECOVERY, STRATEGY, SPY_STRATEGY]

RECOVERY_WINDOW = 20
ONE_WAY_COST_BPS = 5.0

INPUT_PANEL = ROOT / "results" / "flat_rate_refined_L50_H30" / "tables" / "daily_returns.csv"
FINAL_PANEL_CANDIDATES = [
    ROOT / "results" / "09_final_strategy" / "mature_regime_hedge_final" / "daily_backtest_panel.csv",
    ROOT / "results" / "mature_regime_hedge_final" / "daily_backtest_panel.csv",
]
GLOBAL_RECOVERY_DIR = ROOT / "results" / "recovery_20d_strategy_test_L50_H30" / "tables"

OUTPUT_DIR = ROOT / "results" / "recovery_20d_flat_low_only_L50_H30"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"


def ensure_inputs() -> None:
    if not INPUT_PANEL.exists():
        runpy.run_path(str(ROOT / "scripts" / "run_flat_rate_refined_L50_H30.py"), run_name="__main__")
    if not (GLOBAL_RECOVERY_DIR / "daily_returns_all_strategies.csv").exists():
        runpy.run_path(str(ROOT / "scripts" / "test_recovery_20d_strategies_L50_H30.py"), run_name="__main__")


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    clean = {asset: max(0.0, float(weights.get(asset, 0.0))) for asset in ASSETS}
    total = sum(clean.values())
    if total <= 0:
        return clean
    return {asset: clean[asset] / total for asset in ASSETS}


def load_panel() -> pd.DataFrame:
    ensure_inputs()
    refined = pd.read_csv(INPUT_PANEL, parse_dates=["date"])
    final_path = next((p for p in FINAL_PANEL_CANDIDATES if p.exists()), None)
    if final_path is None:
        raise FileNotFoundError("Missing mature final panel.")
    final = pd.read_csv(final_path, parse_dates=["date"])

    needed = ["date"] + [f"{a}_return" for a in ASSETS]
    needed += [f"{MATURE}_weight_{a}" for a in ASSETS]
    needed += [f"{MATURE}_{c}" for c in ["return", "nav", "drawdown", "turnover", "transaction_cost"]]
    for c in ["final_state", "overlay_state", f"{MATURE}_state", "GS10"]:
        if c in final.columns:
            needed.append(c)
    missing = [c for c in needed if c not in final.columns]
    if missing:
        raise ValueError(f"Final panel missing columns: {missing}")

    panel = refined.merge(final[needed], on="date", how="left").sort_values("date").reset_index(drop=True)
    for asset in ASSETS:
        panel[f"{asset}_return"] = pd.to_numeric(panel[f"{asset}_return"], errors="coerce")

    global_returns = GLOBAL_RECOVERY_DIR / "daily_returns_all_strategies.csv"
    global_weights = GLOBAL_RECOVERY_DIR / "daily_weights_all_strategies.csv"
    if global_returns.exists() and global_weights.exists():
        r = pd.read_csv(global_returns, parse_dates=["date"])
        w = pd.read_csv(global_weights, parse_dates=["date"])
        r = r[r["strategy"].eq(GLOBAL_RECOVERY)].copy()
        r = r.rename(
            columns={
                "daily_return": f"{GLOBAL_RECOVERY}_return",
                "equity": f"{GLOBAL_RECOVERY}_nav",
                "drawdown": f"{GLOBAL_RECOVERY}_drawdown",
                "turnover": f"{GLOBAL_RECOVERY}_turnover",
                "transaction_cost": f"{GLOBAL_RECOVERY}_transaction_cost",
            }
        )[["date", f"{GLOBAL_RECOVERY}_return", f"{GLOBAL_RECOVERY}_nav", f"{GLOBAL_RECOVERY}_drawdown", f"{GLOBAL_RECOVERY}_turnover", f"{GLOBAL_RECOVERY}_transaction_cost"]]
        panel = panel.merge(r, on="date", how="left")
        wp = w[w["strategy"].eq(GLOBAL_RECOVERY)].pivot(index="date", columns="asset", values="weight").reset_index()
        wp = wp.rename(columns={asset: f"{GLOBAL_RECOVERY}_weight_{asset}" for asset in ASSETS})
        panel = panel.merge(wp, on="date", how="left")
    return panel


def build_stress_flags(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    flat_state = out.get("flat_refined_state", pd.Series("", index=out.index)).fillna("")
    executed_flat = flat_state.shift(1).fillna(flat_state)
    out["executed_flat_refined_state"] = np.where(flat_state.eq("NON_FLAT_BASELINE"), flat_state, executed_flat)
    flat_stress = pd.Series(out["executed_flat_refined_state"], index=out.index).isin(["FLAT_LOW_RATE_STRESS", "FLAT_HIGH_RATE_STRESS"])
    timing = out.get("timing_state", pd.Series("", index=out.index)).fillna("").astype(str).str.upper()
    final_state = out.get("final_state", pd.Series("", index=out.index)).fillna("").astype(str).str.upper()
    mature_state = out.get(f"{MATURE}_state", pd.Series("", index=out.index)).fillna("").astype(str).str.upper()
    out["is_stress"] = flat_stress | timing.eq("RISK") | final_state.eq("FULL_RISK") | mature_state.eq("FULL_RISK")
    out["is_flat_low_rate_normal"] = out["executed_flat_refined_state"].eq("FLAT_LOW_RATE_NORMAL")
    return out


def get_base_weights(row: pd.Series) -> dict[str, float]:
    return normalize_weights({asset: row.get(f"{REFINED}_weight_{asset}", 0.0) for asset in ASSETS})


def flat_low_equal_weight(row: pd.Series) -> dict[str, float]:
    selected = [asset for asset in ["SPY", "CMDTY_FUT", "GOLD"] if pd.notna(row.get(f"{asset}_return", np.nan))]
    if not selected:
        return get_base_weights(row)
    alloc = {asset: 0.0 for asset in ASSETS}
    for asset in selected:
        alloc[asset] = 1.0 / len(selected)
    return alloc


def build_flat_low_recovery_weights(panel: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.Series]:
    weights = pd.DataFrame(0.0, index=panel.index, columns=ASSETS)
    active_flags = []
    remaining = 0
    was_stress = False
    for i, row in panel.iterrows():
        is_stress = bool(row["is_stress"])
        is_flat_low = bool(row["is_flat_low_rate_normal"])
        if is_stress:
            remaining = 0
            active = False
            alloc = get_base_weights(row)
        else:
            if was_stress and is_flat_low:
                remaining = RECOVERY_WINDOW
            if remaining > 0 and is_flat_low:
                active = True
                if mode == "spy":
                    alloc = {asset: 0.0 for asset in ASSETS}
                    alloc["SPY"] = 1.0
                elif mode == "equal_weight":
                    alloc = flat_low_equal_weight(row)
                else:
                    raise ValueError(f"Unknown recovery mode: {mode}")
                remaining -= 1
            else:
                active = False
                remaining = 0
                alloc = get_base_weights(row)
        weights.loc[i, ASSETS] = pd.Series(normalize_weights(alloc))
        active_flags.append(active)
        was_stress = is_stress
    return weights, pd.Series(active_flags, index=panel.index)


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
    maxdd = float(panel[f"{strategy}_drawdown"].min())
    turnover = float(panel.get(f"{strategy}_turnover", pd.Series(0, index=panel.index)).sum())
    cost = float(panel.get(f"{strategy}_transaction_cost", pd.Series(0, index=panel.index)).sum())
    return {
        "strategy": strategy,
        "CAGR": ann_ret,
        "annualized_volatility": ann_vol,
        "Sharpe": float(ann_ret / ann_vol) if ann_vol > 0 else np.nan,
        "Sortino": sortino_ratio(r, ann_ret),
        "MaxDD": maxdd,
        "Calmar": float(ann_ret / abs(maxdd)) if maxdd < 0 else np.nan,
        "final_equity": float(nav.iloc[-1]),
        "win_rate": float((r > 0).mean()),
        "worst_day_or_month": float(r.min()),
        "worst_12m_return": worst_12m_return(nav),
        "turnover": turnover,
        "total_transaction_cost": cost,
        "number_of_trades": int((panel.get(f"{strategy}_turnover", pd.Series(0, index=panel.index)) > 1e-8).sum()),
    }


def identify_triggered_episodes(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    episode_id = 1
    i = 0
    while i < len(panel):
        if not bool(panel.loc[i, "flat_low_recovery_active"]):
            i += 1
            continue
        start = i
        while i + 1 < len(panel) and bool(panel.loc[i + 1, "flat_low_recovery_active"]):
            i += 1
        end = i
        next_idx = end + 1 if end + 1 < len(panel) else None
        if next_idx is None:
            exit_type = "truncated_by_data_end"
        elif bool(panel.loc[next_idx, "is_stress"]):
            exit_type = "interrupted_by_new_stress"
        elif not bool(panel.loc[next_idx, "is_flat_low_rate_normal"]):
            exit_type = "exited_flat_low_rate"
        else:
            exit_type = "completed_20d"
        sl = panel.iloc[start : end + 1]
        refined_ret = float((1 + sl[f"{REFINED}_return"].fillna(0)).prod() - 1)
        ew_ret = float((1 + sl[f"{STRATEGY}_return"].fillna(0)).prod() - 1)
        spy_ret = float((1 + sl[f"{SPY_STRATEGY}_return"].fillna(0)).prod() - 1)
        refined_maxdd = max_drawdown_from_returns(sl[f"{REFINED}_return"])
        ew_maxdd = max_drawdown_from_returns(sl[f"{STRATEGY}_return"])
        spy_maxdd = max_drawdown_from_returns(sl[f"{SPY_STRATEGY}_return"])
        rows.append(
            {
                "episode_id": episode_id,
                "recovery_start_date": panel.loc[start, "date"],
                "recovery_end_date": panel.loc[end, "date"],
                "episode_length_days": end - start + 1,
                "exit_type": exit_type,
                "start_regime": panel.loc[start, "macro_regime_confirmed"],
                "start_sub_state": panel.loc[start, "executed_flat_refined_state"],
                "GS10_at_start": panel.loc[start, "GS10"] if "GS10" in panel.columns else np.nan,
                "selected_assets": "SPY/CMDTY_FUT/GOLD",
                "refined_baseline_return": refined_ret,
                "flat_low_equal_weight_recovery_return": ew_ret,
                "flat_low_spy_recovery_return": spy_ret,
                "equal_weight_excess_return_vs_refined": ew_ret - refined_ret,
                "spy_excess_return_vs_refined": spy_ret - refined_ret,
                "refined_baseline_maxdd": refined_maxdd,
                "flat_low_equal_weight_recovery_maxdd": ew_maxdd,
                "flat_low_spy_recovery_maxdd": spy_maxdd,
                "equal_weight_maxdd_diff": ew_maxdd - refined_maxdd,
                "spy_maxdd_diff": spy_maxdd - refined_maxdd,
                "days_until_next_stress": end - start + 1 if exit_type == "interrupted_by_new_stress" else np.nan,
                "was_false_recovery_20d": exit_type == "interrupted_by_new_stress",
            }
        )
        episode_id += 1
        i += 1
    return pd.DataFrame(rows)


def recovery_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame([{}])
    row = {
        "number_of_flat_low_recovery_episodes": len(episodes),
        "average_episode_length": episodes["episode_length_days"].mean(),
        "completed_20d_count": int(episodes["exit_type"].eq("completed_20d").sum()),
        "interrupted_by_new_stress_count": int(episodes["exit_type"].eq("interrupted_by_new_stress").sum()),
        "exited_flat_low_rate_count": int(episodes["exit_type"].eq("exited_flat_low_rate").sum()),
        "false_recovery_rate": episodes["was_false_recovery_20d"].mean(),
    }
    for label, col in [
        ("equal_weight", "equal_weight_excess_return_vs_refined"),
        ("spy", "spy_excess_return_vs_refined"),
    ]:
        row[f"{label}_mean_excess_return_vs_refined"] = episodes[col].mean()
        row[f"{label}_median_excess_return_vs_refined"] = episodes[col].median()
        row[f"{label}_total_excess_return_vs_refined"] = episodes[col].sum()
        row[f"{label}_win_rate_vs_refined"] = (episodes[col] > 0).mean()
        row[f"{label}_best_episode_excess_return"] = episodes[col].max()
        row[f"{label}_worst_episode_excess_return"] = episodes[col].min()
    return pd.DataFrame([row])


def write_long_outputs(panel: pd.DataFrame, strategies: list[str]) -> None:
    weight_rows, return_rows = [], []
    for strategy in strategies:
        if f"{strategy}_return" not in panel.columns:
            continue
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


def plot_outputs(panel: pd.DataFrame, episodes: pd.DataFrame, strategies: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for strategy in strategies:
        if f"{strategy}_nav" in panel.columns:
            ax.plot(panel["date"], panel[f"{strategy}_nav"], label=strategy)
    ax.set_yscale("log")
    ax.set_title("FLAT_LOW_ONLY recovery equity comparison")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "equity_curve_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for strategy in strategies:
        if f"{strategy}_drawdown" in panel.columns:
            ax.plot(panel["date"], panel[f"{strategy}_drawdown"], label=strategy)
    ax.set_title("FLAT_LOW_ONLY recovery drawdown comparison")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "drawdown_curve_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for strategy in strategies:
        if f"{strategy}_nav" in panel.columns:
            ax.plot(panel["date"], panel[f"{strategy}_nav"] / panel[f"{strategy}_nav"].shift(252) - 1, label=strategy)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Rolling 12M return comparison")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "rolling_12m_return_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    if not episodes.empty:
        x = np.arange(len(episodes))
        width = 0.4
        ax.bar(x - width / 2, episodes["equal_weight_excess_return_vs_refined"], width=width, label=STRATEGY)
        ax.bar(x + width / 2, episodes["spy_excess_return_vs_refined"], width=width, label=SPY_STRATEGY)
        ax.set_xticks(x)
        ax.set_xticklabels(episodes["episode_id"].astype(str))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("FLAT_LOW recovery episode excess return")
    ax.set_xlabel("Episode ID")
    ax.set_ylabel("Excess return vs refined")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "flat_low_recovery_episode_excess_return_bar.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(panel["date"], 0, panel["is_stress"].astype(int), alpha=0.25, step="pre", label="stress")
    ax.fill_between(panel["date"], 1.1, 1.1 + panel["flat_low_recovery_active"].astype(int), alpha=0.35, step="pre", label="FLAT_LOW recovery")
    skipped = (~panel["is_stress"]) & panel["stress_exit_today"] & (~panel["is_flat_low_rate_normal"])
    ax.scatter(panel.loc[skipped, "date"], np.repeat(2.15, skipped.sum()), s=20, label="stress exits not enabled", color="gray")
    ax.set_yticks([0.5, 1.6, 2.15])
    ax.set_yticklabels(["stress", "FLAT_LOW recovery", "skipped exits"])
    ax.set_title("Recovery trigger timeline")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "recovery_trigger_timeline.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for asset in ASSETS:
        ax.plot(panel["date"], panel[f"{STRATEGY}_weight_{asset}"], label=asset)
    ax.set_title("FLAT_LOW_ONLY recovery strategy weights")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "recovery_weights_timeline.png", dpi=160)
    plt.close(fig)


def write_readme(perf: pd.DataFrame, summary: pd.DataFrame, strategies: list[str]) -> None:
    row = summary.iloc[0]
    refined = perf.loc[perf["strategy"].eq(REFINED)].iloc[0]
    new = perf.loc[perf["strategy"].eq(STRATEGY)].iloc[0]
    spy_new = perf.loc[perf["strategy"].eq(SPY_STRATEGY)].iloc[0]
    global_available = GLOBAL_RECOVERY in perf["strategy"].values
    global_note = ""
    if global_available:
        glob = perf.loc[perf["strategy"].eq(GLOBAL_RECOVERY)].iloc[0]
        global_note = f"\n- Compared with global recovery: final equity {new['final_equity']:.3f} vs {glob['final_equity']:.3f}; Sharpe {new['Sharpe']:.3f} vs {glob['Sharpe']:.3f}."
    text = f"""# RECOVERY_20D_EQUAL_WEIGHT_FLAT_LOW_ONLY

## Purpose

This experiment tests the regime-filtered recovery rule suggested by the attribution analysis: enable the 20D equal-weight recovery override only after a stress exit into `FLAT_LOW_RATE`.

## Rule

- Baseline: `FLAT_RATE_REFINED_L50_H30`.
- Recovery trigger: stress exits and the executed refined state is `FLAT_LOW_RATE_NORMAL`.
- Recovery allocation: 1/3 SPY, 1/3 CMDTY_FUT, 1/3 GOLD.
- Recovery stops after 20 trading days, on new stress, or when the regime leaves `FLAT_LOW_RATE`.
- Priority: stress allocation > FLAT_LOW recovery override > refined baseline.

## Results

- Episodes: {int(row['number_of_flat_low_recovery_episodes'])}
- Equal-weight win rate vs refined: {row['equal_weight_win_rate_vs_refined']:.2%}
- SPY-only win rate vs refined: {row['spy_win_rate_vs_refined']:.2%}
- False recovery rate: {row['false_recovery_rate']:.2%}
- Equal-weight total excess during triggered windows: {row['equal_weight_total_excess_return_vs_refined']:.2%}
- SPY-only total excess during triggered windows: {row['spy_total_excess_return_vs_refined']:.2%}
- CAGR: {new['CAGR']:.2%} vs refined {refined['CAGR']:.2%}
- Sharpe: {new['Sharpe']:.3f} vs refined {refined['Sharpe']:.3f}
- MaxDD: {new['MaxDD']:.2%} vs refined {refined['MaxDD']:.2%}
- Calmar: {new['Calmar']:.3f} vs refined {refined['Calmar']:.3f}
- Final equity: {new['final_equity']:.3f} vs refined {refined['final_equity']:.3f}{global_note}
- SPY-only FLAT_LOW recovery final equity: {spy_new['final_equity']:.3f}; Sharpe: {spy_new['Sharpe']:.3f}; MaxDD: {spy_new['MaxDD']:.2%}.

## Interpretation

This version directly tests whether the recovery overlay should be confined to the only refined regime where attribution was supportive. It avoids applying the overlay to STEEP and FLAT_HIGH_RATE exits, where prior analysis showed no benefit or negative contribution.
"""
    (OUTPUT_DIR / "README_recovery_20d_flat_low_only.md").write_text(text, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    panel = build_stress_flags(load_panel())
    panel["stress_exit_today"] = (~panel["is_stress"]) & (panel["is_stress"].shift(1).fillna(False))

    weights, active = build_flat_low_recovery_weights(panel, mode="equal_weight")
    panel["flat_low_recovery_active"] = active
    computed = compute_strategy_from_weights(panel, weights, STRATEGY)
    panel = pd.concat([panel, computed], axis=1)

    spy_weights, spy_active = build_flat_low_recovery_weights(panel, mode="spy")
    panel["flat_low_spy_recovery_active"] = spy_active
    spy_computed = compute_strategy_from_weights(panel, spy_weights, SPY_STRATEGY)
    panel = pd.concat([panel, spy_computed], axis=1)

    strategies = [MATURE, REFINED]
    if f"{GLOBAL_RECOVERY}_return" in panel.columns:
        strategies.append(GLOBAL_RECOVERY)
    strategies.append(STRATEGY)
    strategies.append(SPY_STRATEGY)

    perf = pd.DataFrame([performance_metrics(panel, s) for s in strategies])
    episodes = identify_triggered_episodes(panel)
    summary = recovery_summary(episodes)

    perf.to_csv(TABLE_DIR / "performance_comparison.csv", index=False)
    episodes.to_csv(TABLE_DIR / "flat_low_recovery_episodes.csv", index=False)
    summary.to_csv(TABLE_DIR / "recovery_summary.csv", index=False)
    write_long_outputs(panel, strategies)
    plot_outputs(panel, episodes, strategies)
    write_readme(perf, summary, strategies)

    print("FLAT_LOW_ONLY recovery test complete.")
    print(f"output_dir: {OUTPUT_DIR.relative_to(ROOT)}")
    print(perf[["strategy", "CAGR", "Sharpe", "MaxDD", "Calmar", "final_equity", "turnover", "total_transaction_cost"]].to_string(index=False))
    print(f"number of flat-low recovery episodes: {int(summary['number_of_flat_low_recovery_episodes'].iloc[0])}")
    print(f"equal-weight win rate vs refined baseline: {summary['equal_weight_win_rate_vs_refined'].iloc[0]:.2%}")
    print(f"SPY-only win rate vs refined baseline: {summary['spy_win_rate_vs_refined'].iloc[0]:.2%}")
    print(f"false recovery rate: {summary['false_recovery_rate'].iloc[0]:.2%}")
    if GLOBAL_RECOVERY in perf["strategy"].values:
        new = perf.loc[perf["strategy"].eq(STRATEGY)].iloc[0]
        glob = perf.loc[perf["strategy"].eq(GLOBAL_RECOVERY)].iloc[0]
        better = (new["final_equity"] > glob["final_equity"]) and (new["MaxDD"] >= glob["MaxDD"])
        print(f"better than global {GLOBAL_RECOVERY}: {better}")


if __name__ == "__main__":
    main()
