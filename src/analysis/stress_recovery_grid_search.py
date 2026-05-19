from __future__ import annotations

from itertools import product
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "stress_recovery_grid_search"
FIGURE_DIR = ROOT / "figures" / "stress_recovery_grid_search"

PANEL_CANDIDATES = [
    ROOT / "results" / "spy_cash_stress_recovery_with_commodity" / "daily_backtest_panel.csv",
    ROOT / "results" / "spy_cash_stress_recovery_with_credit" / "daily_backtest_panel.csv",
    ROOT / "results" / "spy_cash_stress_recovery_timing" / "daily_backtest_panel.csv",
]

CONFIG = {
    "vix_windows": [60, 90, 120, 180],
    "vix_thresholds": [2.5, 3.0, 3.5],
    "dd_thresholds": [-0.03, -0.05, -0.08],
    "credit_windows": [10, 20, 40],
    "credit_thresholds": [0.05, 0.10, 0.15],
    "recovery_ma_windows_secondary": [10, 20, 50],
    "baseline_params": {
        "vix_window": 120,
        "vix_threshold": 3.0,
        "dd_threshold": -0.05,
        "credit_window": 20,
        "credit_threshold": 0.10,
        "recovery_ma_window": 20,
    },
    "one_way_cost_bps": 5.0,
    "output_dir": str(OUTPUT_DIR),
    "figure_dir": str(FIGURE_DIR),
}

CASE_WINDOWS = {
    "GFC_2008_2009": ("2008-09-01", "2009-03-31"),
    "CREDIT_COMMODITY_2015_2016": ("2015-05-01", "2016-03-31"),
    "TIGHTENING_2018Q4": ("2018-10-01", "2019-01-31"),
    "COVID_2020": ("2020-02-19", "2020-04-30"),
    "INFLATION_2022": ("2021-11-01", "2023-03-31"),
    "HIGH_RATE_2023": ("2023-07-01", "2023-11-30"),
    "RECENT_2024_2026": ("2024-01-01", "2026-12-31"),
}

GRID_OUT = OUTPUT_DIR / "grid_search_results.csv"
RANK_OUT = OUTPUT_DIR / "grid_search_ranking.csv"
ROBUST_OUT = OUTPUT_DIR / "robustness_summary.csv"
TOP_SHARPE_OUT = OUTPUT_DIR / "top_20_by_sharpe.csv"
TOP_MAXDD_OUT = OUTPUT_DIR / "top_20_by_maxdd.csv"
TOP_COMPOSITE_OUT = OUTPUT_DIR / "top_20_by_composite.csv"
BASELINE_TOP_OUT = OUTPUT_DIR / "baseline_vs_top10.csv"
SECONDARY_OUT = OUTPUT_DIR / "secondary_recovery_grid_top10.csv"
REPORT_OUT = OUTPUT_DIR / "STRESS_RECOVERY_GRID_SEARCH_REPORT.md"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_panel() -> pd.DataFrame:
    for path in PANEL_CANDIDATES:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "date" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"])
        required = ["spy_price", "spy_daily_return", "daily_rf", "macro_regime_confirmed", "monthly_either_state", "VIX_LEVEL", "CREDIT_SPREAD_BAA_AAA"]
        if all(c in df.columns for c in required):
            out = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
            out["spy_price"] = pd.to_numeric(out["spy_price"], errors="coerce")
            out["spy_daily_return"] = pd.to_numeric(out["spy_daily_return"], errors="coerce")
            out["daily_rf"] = pd.to_numeric(out["daily_rf"], errors="coerce")
            out["VIX_LEVEL"] = pd.to_numeric(out["VIX_LEVEL"], errors="coerce")
            out["CREDIT_SPREAD_BAA_AAA"] = pd.to_numeric(out["CREDIT_SPREAD_BAA_AAA"], errors="coerce").ffill()
            return out.dropna(subset=["spy_price", "spy_daily_return", "daily_rf", "VIX_LEVEL", "CREDIT_SPREAD_BAA_AAA"]).reset_index(drop=True)
    raise FileNotFoundError("No usable input panel found for grid search.")


def _zscore(s: pd.Series, window: int) -> pd.Series:
    roll = s.rolling(window, min_periods=window)
    return (s - roll.mean()) / roll.std(ddof=1).replace(0, np.nan)


def build_vix_zscore_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    for w in CONFIG["vix_windows"]:
        out[f"VIX_ZSCORE_{w}D"] = _zscore(out["VIX_LEVEL"], w)
    return out


def build_credit_change_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["previous_high"] = out["spy_price"].cummax()
    out["spy_drawdown_from_previous_high"] = out["spy_price"] / out["previous_high"] - 1.0
    for w in CONFIG["credit_windows"]:
        out[f"D_CREDIT_SPREAD_{w}D"] = out["CREDIT_SPREAD_BAA_AAA"] - out["CREDIT_SPREAD_BAA_AAA"].shift(w)
    return out


def build_recovery_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    for w in set([20] + CONFIG["recovery_ma_windows_secondary"]):
        out[f"SPY_MA{w}"] = out["spy_price"].rolling(w, min_periods=w).mean()
        out[f"SPY_CROSS_ABOVE_MA{w}"] = (out["spy_price"] > out[f"SPY_MA{w}"]) & (out["spy_price"].shift(1) <= out[f"SPY_MA{w}"].shift(1))
    return out


def build_stress_signal_for_params(panel: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series, dict[str, pd.Series]]:
    flat_vix = panel["macro_regime_confirmed"].eq("FLAT") & (panel[f"VIX_ZSCORE_{params['vix_window']}D"] >= params["vix_threshold"])
    steep_either = panel["macro_regime_confirmed"].eq("STEEP") & panel["monthly_either_state"].eq("SELL")
    credit = (panel["spy_drawdown_from_previous_high"] <= params["dd_threshold"]) & (panel[f"D_CREDIT_SPREAD_{params['credit_window']}D"] > params["credit_threshold"])
    signal = flat_vix | steep_either | credit
    reason = pd.Series("", index=panel.index)
    reason = np.where(flat_vix, "FLAT_VIX_STRESS", reason)
    reason = np.where(steep_either & (reason != ""), reason + "+STEEP_EITHER_SELL", np.where(steep_either, "STEEP_EITHER_SELL", reason))
    reason = pd.Series(reason, index=panel.index)
    reason = np.where(credit & (reason != ""), reason + "+CREDIT_STRESS", np.where(credit, "CREDIT_STRESS", reason))
    components = {"flat_vix": flat_vix, "steep_either": steep_either, "credit": credit, "combined": (flat_vix.astype(int) + steep_either.astype(int) + credit.astype(int)) > 1}
    return signal, pd.Series(reason, index=panel.index), components


def run_state_machine_backtest(panel: pd.DataFrame, params: dict, return_series: bool = False) -> tuple[dict, pd.Series | None]:
    signal, reason, components = build_stress_signal_for_params(panel, params)
    recovery = panel[f"SPY_CROSS_ABOVE_MA{params['recovery_ma_window']}"].fillna(False)
    state, pending_state = "NORMAL", "NORMAL"
    nav = 1.0
    cost_rate = CONFIG["one_way_cost_bps"] / 10000.0
    rets, navs, weights, costs, turnovers = [], [], [], [], []
    events = []
    for i, row in panel.iterrows():
        old_state = state
        cost, turnover = 0.0, 0.0
        if i > 0 and pending_state != state:
            state = pending_state
            old_w = 1.0 if old_state == "NORMAL" else 0.0
            new_w = 1.0 if state == "NORMAL" else 0.0
            turnover = abs(new_w - old_w) + abs((1 - new_w) - (1 - old_w))
            cost = 0.5 * turnover * cost_rate
            sig_i = i - 1
            events.append({"idx": i, "type": "ENTER_RISK" if state == "RISK" else "EXIT_RISK", "reason": pending_reason, "signal_idx": sig_i})
        w = 1.0 if state == "NORMAL" else 0.0
        ret = w * row["spy_daily_return"] + (1 - w) * row["daily_rf"] - cost
        nav *= 1 + float(ret)
        rets.append(ret)
        navs.append(nav)
        weights.append(w)
        costs.append(cost)
        turnovers.append(turnover)
        pending_state, pending_reason = state, ""
        if state == "NORMAL" and bool(signal.iloc[i]):
            pending_state, pending_reason = "RISK", reason.iloc[i]
        elif state == "RISK" and bool(recovery.iloc[i]):
            pending_state, pending_reason = "NORMAL", f"R3_CROSS_ABOVE_MA{params['recovery_ma_window']}"
    ret_s = pd.Series(rets, index=panel.index)
    nav_s = pd.Series(navs, index=panel.index)
    weight_s = pd.Series(weights, index=panel.index)
    event_df = pd.DataFrame(events)
    entries = event_df[event_df["type"].eq("ENTER_RISK")] if not event_df.empty else pd.DataFrame()
    durations = []
    if not event_df.empty:
        enter_idxs = event_df.loc[event_df["type"].eq("ENTER_RISK"), "idx"].tolist()
        exit_idxs = event_df.loc[event_df["type"].eq("EXIT_RISK"), "idx"].tolist()
        for e in enter_idxs:
            later = [x for x in exit_idxs if x > e]
            durations.append((later[0] if later else len(panel) - 1) - e)
    metrics = compute_performance_metrics(panel, ret_s, nav_s, weight_s)
    metrics.update(
        {
            "number_of_switches": len(event_df),
            "number_of_risk_entries": int((event_df["type"] == "ENTER_RISK").sum()) if not event_df.empty else 0,
            "number_of_risk_exits": int((event_df["type"] == "EXIT_RISK").sum()) if not event_df.empty else 0,
            "avg_risk_episode_duration": float(np.mean(durations)) if durations else np.nan,
            "median_risk_episode_duration": float(np.median(durations)) if durations else np.nan,
            "time_in_cash": float(1 - weight_s.mean()),
            "total_turnover": float(np.sum(turnovers)),
            "transaction_cost_drag": float(np.sum(costs)),
            "flat_vix_entry_count": int(components["flat_vix"].iloc[entries["signal_idx"]].sum()) if not entries.empty else 0,
            "steep_either_entry_count": int(components["steep_either"].iloc[entries["signal_idx"]].sum()) if not entries.empty else 0,
            "credit_entry_count": int(components["credit"].iloc[entries["signal_idx"]].sum()) if not entries.empty else 0,
            "combined_entry_count": int(components["combined"].iloc[entries["signal_idx"]].sum()) if not entries.empty else 0,
        }
    )
    metrics.update(compute_crisis_performance(panel, ret_s, nav_s))
    return metrics, nav_s if return_series else None


def compute_performance_metrics(panel: pd.DataFrame, ret_s: pd.Series, nav_s: pd.Series, weight_s: pd.Series) -> dict:
    rf = panel.loc[ret_s.index, "daily_rf"]
    ann = (1 + ret_s).prod() ** (252 / len(ret_s)) - 1
    vol = ret_s.std(ddof=1) * np.sqrt(252)
    ex = ret_s - rf
    sharpe = ex.mean() / ex.std(ddof=1) * np.sqrt(252) if ex.std(ddof=1) != 0 else np.nan
    dd = nav_s / nav_s.cummax() - 1
    mdd = dd.min()
    return {
        "annualized_return": ann,
        "annualized_volatility": vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
        "calmar_ratio": ann / abs(mdd) if mdd < 0 else np.nan,
        "final_nav": nav_s.iloc[-1],
        "excess_return_sharpe": sharpe,
        "time_in_spy": float(weight_s.mean()),
    }


def _subperiod_stats(panel: pd.DataFrame, ret_s: pd.Series, nav_s: pd.Series, start: str, end: str) -> tuple[float, float]:
    mask = panel["date"].between(pd.Timestamp(start), pd.Timestamp(end))
    r = ret_s[mask]
    if r.empty:
        return np.nan, np.nan
    wealth = (1 + r).cumprod()
    return float(wealth.iloc[-1] - 1), float((wealth / wealth.cummax() - 1).min())


def compute_crisis_performance(panel: pd.DataFrame, ret_s: pd.Series, nav_s: pd.Series) -> dict:
    out = {}
    for name, (start, end) in CASE_WINDOWS.items():
        ret, dd = _subperiod_stats(panel, ret_s, nav_s, start, end)
        out[f"{name}_return"] = ret
        out[f"{name}_max_drawdown"] = dd
    return out


def run_grid_search(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(CONFIG["vix_windows"]) * len(CONFIG["vix_thresholds"]) * len(CONFIG["dd_thresholds"]) * len(CONFIG["credit_windows"]) * len(CONFIG["credit_thresholds"])
    n = 0
    for vw, vt, dd, cw, ct in product(CONFIG["vix_windows"], CONFIG["vix_thresholds"], CONFIG["dd_thresholds"], CONFIG["credit_windows"], CONFIG["credit_thresholds"]):
        n += 1
        params = {"vix_window": vw, "vix_threshold": vt, "dd_threshold": dd, "credit_window": cw, "credit_threshold": ct, "recovery_ma_window": 20}
        metrics, _ = run_state_machine_backtest(panel, params)
        row = {**params, **metrics}
        row["is_current_baseline"] = all(row[k] == v for k, v in CONFIG["baseline_params"].items())
        row["distance_from_baseline_params"] = sum(row[k] != v for k, v in CONFIG["baseline_params"].items())
        row["near_baseline_flag"] = row["distance_from_baseline_params"] <= 1
        rows.append(row)
    return pd.DataFrame(rows)


def _normalize(s: pd.Series, higher_better: bool = True) -> pd.Series:
    vals = s.astype(float)
    if vals.max() == vals.min():
        out = pd.Series(0.5, index=s.index)
    else:
        out = (vals - vals.min()) / (vals.max() - vals.min())
    return out if higher_better else 1 - out


def rank_parameter_sets(results: pd.DataFrame, spy_bh_ann: float) -> pd.DataFrame:
    df = results.copy()
    df["sharpe_rank"] = df["sharpe_ratio"].rank(ascending=False, method="min")
    df["maxdd_rank"] = df["max_drawdown"].rank(ascending=False, method="min")
    df["annret_rank"] = df["annualized_return"].rank(ascending=False, method="min")
    eligible = (df["time_in_cash"] <= 0.30) & (df["number_of_switches"] <= 250) & (df["max_drawdown"] >= -0.30) & (df["annualized_return"] >= spy_bh_ann - 0.01)
    df["eligible_for_composite"] = eligible
    df["score"] = np.nan
    sub = df[eligible].copy()
    if not sub.empty:
        score = (
            0.35 * _normalize(sub["sharpe_ratio"])
            + 0.25 * _normalize(sub["annualized_return"])
            + 0.25 * _normalize(sub["max_drawdown"])
            - 0.10 * _normalize(sub["number_of_switches"])
            - 0.05 * _normalize(sub["time_in_cash"])
        )
        df.loc[sub.index, "score"] = score
    df["composite_rank"] = df["score"].rank(ascending=False, method="min")
    return df.sort_values("score", ascending=False, na_position="last")


def benchmark_stats(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, ret in [
        ("SPY_BUY_HOLD", panel["spy_daily_return"]),
        ("MONTHLY_EITHER_CONFIRM", panel.get("MONTHLY_EITHER_CONFIRM_return", panel["spy_daily_return"])),
    ]:
        nav = (1 + ret).cumprod()
        w = pd.Series(1.0, index=panel.index) if name == "SPY_BUY_HOLD" else panel.get("monthly_either_weight_spy", pd.Series(np.nan, index=panel.index))
        rows.append({"strategy": name, **compute_performance_metrics(panel, ret, nav, w)})
    return pd.DataFrame(rows)


def compute_robustness_summary(ranked: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = ranked[ranked["is_current_baseline"]].iloc[0]
    near = ranked[ranked["near_baseline_flag"]]
    materially = ranked[
        (ranked["sharpe_ratio"] >= baseline["sharpe_ratio"] + 0.10)
        & (ranked["max_drawdown"] >= baseline["max_drawdown"] - 0.02)
        & (ranked["number_of_switches"] <= baseline["number_of_switches"] * 1.5)
    ]
    summary = pd.DataFrame(
        [
            {
                "baseline_sharpe": baseline["sharpe_ratio"],
                "baseline_maxdd": baseline["max_drawdown"],
                "baseline_annret": baseline["annualized_return"],
                "baseline_rank_by_sharpe": baseline["sharpe_rank"],
                "baseline_rank_by_composite": baseline["composite_rank"],
                "percentile_rank_by_sharpe": 1 - (baseline["sharpe_rank"] - 1) / len(ranked),
                "percentile_rank_by_maxdd": 1 - (baseline["maxdd_rank"] - 1) / len(ranked),
                "percentile_rank_by_composite": 1 - (baseline["composite_rank"] - 1) / ranked["score"].notna().sum(),
                "near_baseline_avg_sharpe": near["sharpe_ratio"].mean(),
                "near_baseline_median_sharpe": near["sharpe_ratio"].median(),
                "near_baseline_avg_maxdd": near["max_drawdown"].mean(),
                "near_baseline_median_maxdd": near["max_drawdown"].median(),
                "near_baseline_sharpe_std": near["sharpe_ratio"].std(ddof=1),
                "near_baseline_maxdd_std": near["max_drawdown"].std(ddof=1),
                "materially_better_count": len(materially),
            }
        ]
    )
    return summary, materially.sort_values("score", ascending=False)


def run_secondary_recovery_grid(panel: pd.DataFrame, ranked: pd.DataFrame) -> pd.DataFrame:
    rows = []
    top = ranked.head(10)
    for _, base in top.iterrows():
        for ma in CONFIG["recovery_ma_windows_secondary"]:
            params = {k: base[k] for k in ["vix_window", "vix_threshold", "dd_threshold", "credit_window", "credit_threshold"]}
            params["recovery_ma_window"] = ma
            metrics, _ = run_state_machine_backtest(panel, params)
            rows.append({**params, **metrics})
    return pd.DataFrame(rows)


def plot_heatmaps(ranked: pd.DataFrame) -> None:
    bp = CONFIG["baseline_params"]
    vsub = ranked[(ranked["dd_threshold"] == bp["dd_threshold"]) & (ranked["credit_window"] == bp["credit_window"]) & (ranked["credit_threshold"] == bp["credit_threshold"])]
    for metric, path in [("sharpe_ratio", "sharpe_heatmap_vix_params.png"), ("max_drawdown", "maxdd_heatmap_vix_params.png")]:
        heat = vsub.pivot_table(index="vix_window", columns="vix_threshold", values=metric)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(heat, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / path, dpi=160)
        plt.close(fig)
    csub = ranked[(ranked["vix_window"] == bp["vix_window"]) & (ranked["vix_threshold"] == bp["vix_threshold"]) & (ranked["dd_threshold"] == bp["dd_threshold"])]
    for metric, path in [("sharpe_ratio", "sharpe_heatmap_credit_params.png"), ("max_drawdown", "maxdd_heatmap_credit_params.png")]:
        heat = csub.pivot_table(index="credit_window", columns="credit_threshold", values=metric)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(heat, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / path, dpi=160)
        plt.close(fig)


def plot_scatter(ranked: pd.DataFrame) -> None:
    baseline = ranked[ranked["is_current_baseline"]].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=ranked, x="max_drawdown", y="sharpe_ratio", size="number_of_switches", hue="annualized_return", sizes=(20, 220), palette="viridis", ax=ax)
    ax.scatter([baseline["max_drawdown"]], [baseline["sharpe_ratio"]], color="red", marker="*", s=250, label="baseline")
    for _, row in ranked.head(5).iterrows():
        ax.text(row["max_drawdown"], row["sharpe_ratio"], f"{int(row['vix_window'])}/{row['vix_threshold']}/{row['dd_threshold']}/{int(row['credit_window'])}/{row['credit_threshold']}", fontsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "scatter_sharpe_vs_maxdd.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=ranked, x="time_in_cash", y="annualized_return", hue="sharpe_ratio", palette="viridis", ax=ax)
    ax.scatter([baseline["time_in_cash"]], [baseline["annualized_return"]], color="red", marker="*", s=250)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "scatter_return_vs_cash_time.png", dpi=160)
    plt.close(fig)


def plot_sensitivity_boxplots(ranked: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    params = ["vix_threshold", "vix_window", "dd_threshold", "credit_threshold"]
    for j, p in enumerate(params):
        sns.boxplot(data=ranked, x=p, y="sharpe_ratio", ax=axes[0, j])
        sns.boxplot(data=ranked, x=p, y="max_drawdown", ax=axes[1, j])
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "parameter_sensitivity_boxplots.png", dpi=160)
    plt.close(fig)


def plot_top_equity_curves(panel: pd.DataFrame, ranked: pd.DataFrame) -> None:
    picks = {
        "current_baseline": ranked[ranked["is_current_baseline"]].iloc[0],
        "top_sharpe": ranked.sort_values("sharpe_ratio", ascending=False).iloc[0],
        "top_maxdd": ranked.sort_values("max_drawdown", ascending=False).iloc[0],
        "top_composite": ranked.sort_values("score", ascending=False).iloc[0],
    }
    fig, ax = plt.subplots(figsize=(12, 6))
    spy_nav = (1 + panel["spy_daily_return"]).cumprod()
    ax.plot(panel["date"], spy_nav, label="SPY_BUY_HOLD", color="black", alpha=0.65)
    if "MONTHLY_EITHER_CONFIRM_nav" in panel.columns:
        ax.plot(panel["date"], panel["MONTHLY_EITHER_CONFIRM_nav"], label="MONTHLY_EITHER_CONFIRM", alpha=0.65)
    navs = {}
    for name, row in picks.items():
        params = {k: row[k] for k in ["vix_window", "vix_threshold", "dd_threshold", "credit_window", "credit_threshold", "recovery_ma_window"]}
        _, nav = run_state_machine_backtest(panel, params, return_series=True)
        navs[name] = nav
        ax.plot(panel["date"], nav, label=name)
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "equity_curve_top_models.png", dpi=160)
    plt.close(fig)

    crisis_rows = []
    for name, nav in navs.items():
        ret = nav.pct_change().fillna(nav.iloc[0] - 1)
        stats = compute_crisis_performance(panel, ret, nav)
        for period in CASE_WINDOWS:
            crisis_rows.append({"model": name, "period": period, "return": stats[f"{period}_return"]})
    cdf = pd.DataFrame(crisis_rows)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=cdf, x="period", y="return", hue="model", ax=ax)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "crisis_performance_top_models.png", dpi=160)
    plt.close(fig)


def write_markdown_report(ranked: pd.DataFrame, robust: pd.DataFrame, materially: pd.DataFrame, bench: pd.DataFrame) -> None:
    baseline = ranked[ranked["is_current_baseline"]].iloc[0]
    lines = [
        "# Stress-Recovery Grid Search Report",
        "",
        "## Purpose",
        "",
        "This analysis tests robustness of the current SPY/CASH stress-recovery baseline. It is not intended to overfit a final parameter set.",
        "",
        "## Current Baseline",
        "",
        "- FLAT + VIX z-score 120D >= 3.0.",
        "- STEEP + Monthly Either SELL.",
        "- SPY drawdown <= -5% and credit spread 20D change > 0.10.",
        "- Recovery = SPY crosses above MA20.",
        "",
        "## Benchmarks",
        "",
        bench.to_markdown(index=False),
        "",
        "## Baseline Result",
        "",
        baseline.to_frame().T.to_markdown(index=False),
        "",
        "## Top Composite",
        "",
        ranked.head(10).to_markdown(index=False),
        "",
        "## Robustness Summary",
        "",
        robust.to_markdown(index=False),
        "",
        "## Materially Better Candidates",
        "",
        materially.head(20).to_markdown(index=False) if not materially.empty else "_No materially better candidates under the stated rule._",
        "",
        "## Interpretation",
        "",
        "- Robustness should be judged by parameter neighborhoods, not just the top row.",
        "- A candidate with higher Sharpe but much higher switches or crisis concentration should be treated as suspicious.",
        "- The next step should only test a small number of economically interpretable alternatives inside the full regime-hedge framework.",
    ]
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = build_recovery_features(build_credit_change_features(build_vix_zscore_features(load_panel())))
    grid = run_grid_search(panel)
    bench = benchmark_stats(panel)
    spy_ann = bench.loc[bench["strategy"].eq("SPY_BUY_HOLD"), "annualized_return"].iloc[0]
    ranked = rank_parameter_sets(grid, spy_ann)
    robust, materially = compute_robustness_summary(ranked)
    secondary = run_secondary_recovery_grid(panel, ranked)

    grid.to_csv(GRID_OUT, index=False)
    ranked.to_csv(RANK_OUT, index=False)
    robust.to_csv(ROBUST_OUT, index=False)
    ranked.sort_values("sharpe_ratio", ascending=False).head(20).to_csv(TOP_SHARPE_OUT, index=False)
    ranked.sort_values("max_drawdown", ascending=False).head(20).to_csv(TOP_MAXDD_OUT, index=False)
    ranked.sort_values("score", ascending=False).head(20).to_csv(TOP_COMPOSITE_OUT, index=False)
    pd.concat(
        [
            ranked[ranked["is_current_baseline"]],
            ranked.sort_values("score", ascending=False).head(10),
            ranked.sort_values("sharpe_ratio", ascending=False).head(10),
            ranked.sort_values("max_drawdown", ascending=False).head(10),
        ],
        ignore_index=True,
    ).drop_duplicates().to_csv(BASELINE_TOP_OUT, index=False)
    secondary.to_csv(SECONDARY_OUT, index=False)

    plot_heatmaps(ranked)
    plot_scatter(ranked)
    plot_sensitivity_boxplots(ranked)
    plot_top_equity_curves(panel, ranked)
    write_markdown_report(ranked, robust, materially, bench)

    baseline = ranked[ranked["is_current_baseline"]].iloc[0]
    top_sharpe = ranked.sort_values("sharpe_ratio", ascending=False).iloc[0]
    top_maxdd = ranked.sort_values("max_drawdown", ascending=False).iloc[0]
    top_comp = ranked.sort_values("score", ascending=False).iloc[0]
    print(f"1. Total parameter combinations: {len(grid)}")
    print(f"2. Baseline Ann/Sharpe/MaxDD/switches/cash: {baseline['annualized_return']:.2%} / {baseline['sharpe_ratio']:.2f} / {baseline['max_drawdown']:.2%} / {int(baseline['number_of_switches'])} / {baseline['time_in_cash']:.1%}")
    print(f"3. Baseline Sharpe rank / composite rank: {int(baseline['sharpe_rank'])} / {int(baseline['composite_rank'])}")
    for label, row in [("top Sharpe", top_sharpe), ("top MaxDD", top_maxdd), ("top composite", top_comp)]:
        print(f"4. {label}: vix {int(row['vix_window'])}/{row['vix_threshold']}, dd {row['dd_threshold']}, credit {int(row['credit_window'])}/{row['credit_threshold']} | Ann {row['annualized_return']:.2%}, Sharpe {row['sharpe_ratio']:.2f}, MaxDD {row['max_drawdown']:.2%}")
    print(f"7. Materially better parameter sets: {int(robust['materially_better_count'].iloc[0])}")
    print(f"8. Near-baseline Sharpe mean/std: {robust['near_baseline_avg_sharpe'].iloc[0]:.2f} / {robust['near_baseline_sharpe_std'].iloc[0]:.2f}")
    print(f"9. Near-baseline MaxDD mean/std: {robust['near_baseline_avg_maxdd'].iloc[0]:.2%} / {robust['near_baseline_maxdd_std'].iloc[0]:.2%}")
    recommendation = "investigate materially better parameter set" if robust["materially_better_count"].iloc[0] > 0 else "keep baseline"
    print(f"10. Final recommendation: {recommendation}")
    print(f"Saved outputs: {OUTPUT_DIR} and {FIGURE_DIR}")


if __name__ == "__main__":
    main()
