"""Diagnostic of FLAT_RISK GOLD vs CASH and simple hedge variants."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import ttest_1samp, wilcoxon
except Exception:  # pragma: no cover
    ttest_1samp = None
    wilcoxon = None


CONFIG = {
    "output_dir": Path("results/flat_risk_gold_cash_diagnostic"),
    "figure_dir": Path("figures/flat_risk_gold_cash_diagnostic"),
    "one_way_cost_bps": 5,
}

PANEL_CANDIDATES = [
    Path("results/backbone_v2_with_steep_commodity_stress/daily_backtest_panel.csv"),
    Path("results/regime_aware_risk_parity_allocation/daily_backtest_panel.csv"),
]

ASSETS = ["SPY", "GOLD", "CMDTY_FUT", "IEF", "CASH"]
CRISIS_WINDOWS = {
    "UKRAINE_INITIAL_SHOCK": ("2022-02-01", "2022-03-31"),
    "FIRST_HALF_2022": ("2022-01-01", "2022-06-30"),
    "FULL_2022_RATE_WAR": ("2021-11-01", "2023-03-31"),
    "2015_2016": ("2015-05-01", "2016-03-31"),
    "COVID_2020": ("2020-02-01", "2020-06-30"),
    "2025_PULLBACK": ("2025-01-01", "2025-12-31"),
}
VARIANTS = {
    "FLAT_RISK_GOLD": {"GOLD": 1.0},
    "FLAT_RISK_CASH": {"CASH": 1.0},
    "FLAT_RISK_50GOLD_50CASH": {"GOLD": 0.5, "CASH": 0.5},
    "FLAT_RISK_70GOLD_30CASH": {"GOLD": 0.7, "CASH": 0.3},
    "FLAT_RISK_30GOLD_70CASH": {"GOLD": 0.3, "CASH": 0.7},
}


def ensure_dirs() -> None:
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["figure_dir"].mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _col(df: pd.DataFrame, names: Iterable[str], fill: float | str | None = None) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    if fill is None:
        return pd.Series(index=df.index, dtype=float)
    return pd.Series(fill, index=df.index)


def load_panel() -> pd.DataFrame:
    path = next((p for p in PANEL_CANDIDATES if p.exists()), None)
    if path is None:
        raise FileNotFoundError("No mature panel found.")
    df = _read_csv(path)
    needed = [
        "macro_regime_confirmed",
        "MATURE_BASELINE_REGIME_HEDGE_INV_VOL_risk_state",
        "MATURE_BASELINE_REGIME_HEDGE_INV_VOL_weight_SPY",
        "MATURE_BASELINE_REGIME_HEDGE_INV_VOL_weight_GOLD",
        "MATURE_BASELINE_REGIME_HEDGE_INV_VOL_weight_CMDTY_FUT",
        "MATURE_BASELINE_REGIME_HEDGE_INV_VOL_weight_IEF",
        "MATURE_BASELINE_REGIME_HEDGE_INV_VOL_weight_CASH",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing mature baseline columns: {missing}")
    for asset, names in {
        "SPY": ["SPY_return", "spy_daily_return"],
        "GOLD": ["GOLD_return", "GLD_return"],
        "IEF": ["IEF_return"],
        "CMDTY_FUT": ["CMDTY_FUT_return"],
        "CASH": ["CASH_return", "daily_rf"],
    }.items():
        df[f"{asset}_return"] = pd.to_numeric(_col(df, names), errors="coerce").fillna(0.0)
    if "spy_price" not in df.columns:
        df["spy_price"] = (1 + df["SPY_return"]).cumprod()
    if "spy_drawdown_from_previous_high" not in df.columns:
        df["spy_drawdown_from_previous_high"] = df["spy_price"] / df["spy_price"].cummax() - 1.0
    if "CMDTY_FUT_price" not in df.columns:
        df["CMDTY_FUT_price"] = (1 + df["CMDTY_FUT_return"]).cumprod()
    if "CMDTY_RET60" not in df.columns:
        df["CMDTY_RET60"] = df["CMDTY_FUT_price"] / df["CMDTY_FUT_price"].shift(60) - 1.0
    for extra in ["VIX_ZSCORE_120D", "D_CREDIT_SPREAD_20D", "GS10", "GS1", "growth_pc1", "inflation_pc1"]:
        if extra not in df.columns:
            df[extra] = np.nan
    print(f"Loaded panel: {path}")
    return df


def _weights_from_baseline(row: pd.Series) -> Dict[str, float]:
    return {a: float(pd.to_numeric(row.get(f"MATURE_BASELINE_REGIME_HEDGE_INV_VOL_weight_{a}", 0.0), errors="coerce") or 0.0) for a in ASSETS}


def _portfolio_return(weights: Dict[str, float], row: pd.Series) -> float:
    return sum(weights[a] * float(row[f"{a}_return"]) for a in ASSETS)


def _segment_metrics(ret: pd.Series, rf: pd.Series) -> Dict[str, float]:
    nav = (1 + ret.fillna(0.0)).cumprod()
    dd = nav / nav.cummax() - 1.0
    excess = ret - rf
    downside = ret[ret < 0]
    ann_ret = nav.iloc[-1] ** (252 / len(ret)) - 1 if len(ret) else np.nan
    ann_vol = ret.std(ddof=0) * math.sqrt(252)
    return {
        "AnnRet": ann_ret,
        "Sharpe": excess.mean() / excess.std(ddof=0) * math.sqrt(252) if excess.std(ddof=0) > 0 else np.nan,
        "Sortino": excess.mean() / downside.std(ddof=0) * math.sqrt(252) if len(downside) and downside.std(ddof=0) > 0 else np.nan,
        "MaxDD": dd.min() if len(dd) else np.nan,
        "FinalNAV": nav.iloc[-1] if len(nav) else np.nan,
    }


def extract_flat_risk_episodes(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["macro_regime_confirmed"].eq("FLAT") & df["MATURE_BASELINE_REGIME_HEDGE_INV_VOL_risk_state"].eq("FULL_RISK")
    ids = (mask & ~mask.shift(1, fill_value=False)).cumsum()
    rows = []
    active_ids = sorted(ids[mask].unique())
    for ep_id in active_ids:
        sub = df[ids.eq(ep_id) & mask].copy()
        if sub.empty:
            continue
        row = {
            "episode_id": int(ep_id),
            "start_date": sub["date"].iloc[0],
            "end_date": sub["date"].iloc[-1],
            "duration_days": len(sub),
            "entry_reason": sub.get("MATURE_BASELINE_REGIME_HEDGE_INV_VOL_entry_reason", pd.Series("", index=sub.index)).iloc[0],
            "VIX_ZSCORE_at_entry": sub["VIX_ZSCORE_120D"].iloc[0],
            "D_CREDIT_SPREAD_20D_at_entry": sub["D_CREDIT_SPREAD_20D"].iloc[0],
            "GS10_at_entry": sub["GS10"].iloc[0],
            "GS1_at_entry": sub["GS1"].iloc[0],
            "inflation_pc1": sub["inflation_pc1"].iloc[0],
            "growth_pc1": sub["growth_pc1"].iloc[0],
        }
        for asset in ASSETS:
            ret = sub[f"{asset}_return"].fillna(0.0)
            nav = (1 + ret).cumprod()
            row[f"{asset}_return"] = nav.iloc[-1] - 1.0
            row[f"{asset}_max_drawdown"] = (nav / nav.cummax() - 1.0).min()
        row["GOLD_minus_CASH_return"] = row["GOLD_return"] - row["CASH_return"]
        row["GOLD_minus_CASH_maxdd"] = row["GOLD_max_drawdown"] - row["CASH_max_drawdown"]
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(CONFIG["output_dir"] / "flat_risk_episode_asset_performance.csv", index=False)
    return out


def summarize_gold_cash(episodes: pd.DataFrame) -> pd.DataFrame:
    full_start = pd.Timestamp("2021-11-01")
    full_end = pd.Timestamp("2023-03-31")
    init_start = pd.Timestamp("2022-02-01")
    init_end = pd.Timestamp("2022-03-31")
    samples = {
        "ALL_FLAT_RISK_EPISODES": episodes,
        "EX_UKRAINE_INITIAL": episodes[~((episodes["start_date"] <= init_end) & (episodes["end_date"] >= init_start))].copy(),
        "EX_FULL_2022_RATE_WAR": episodes[~((episodes["start_date"] <= full_end) & (episodes["end_date"] >= full_start))].copy(),
    }
    rows = []
    for name, sub in samples.items():
        if sub.empty:
            continue
        diff = sub["GOLD_minus_CASH_return"].dropna()
        t_stat = p_val = wilcoxon_p = np.nan
        if len(diff) >= 2 and ttest_1samp is not None:
            try:
                stat = ttest_1samp(diff, 0.0, nan_policy="omit")
                t_stat = stat.statistic
                p_val = stat.pvalue
            except Exception:
                pass
        if len(diff) >= 2 and wilcoxon is not None:
            try:
                wilcoxon_p = wilcoxon(diff).pvalue
            except Exception:
                pass
        rows.append(
            {
                "sample_name": name,
                "episode_count": len(sub),
                "avg_GOLD_return": sub["GOLD_return"].mean(),
                "avg_CASH_return": sub["CASH_return"].mean(),
                "median_GOLD_return": sub["GOLD_return"].median(),
                "median_CASH_return": sub["CASH_return"].median(),
                "avg_GOLD_minus_CASH_return": sub["GOLD_minus_CASH_return"].mean(),
                "median_GOLD_minus_CASH_return": sub["GOLD_minus_CASH_return"].median(),
                "pct_GOLD_outperforms_CASH": (sub["GOLD_minus_CASH_return"] > 0).mean(),
                "avg_GOLD_maxdd": sub["GOLD_max_drawdown"].mean(),
                "avg_CASH_maxdd": sub["CASH_max_drawdown"].mean(),
                "avg_GOLD_minus_CASH_maxdd": sub["GOLD_minus_CASH_maxdd"].mean(),
                "worst_GOLD_episode": sub.loc[sub["GOLD_return"].idxmin(), "episode_id"],
                "worst_CASH_episode": sub.loc[sub["CASH_return"].idxmin(), "episode_id"],
                "t_stat_gold_minus_cash_return": t_stat,
                "p_value_gold_minus_cash_return": p_val,
                "wilcoxon_p_value": wilcoxon_p,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(CONFIG["output_dir"] / "flat_risk_gold_cash_summary.csv", index=False)
    return out


def analyze_ukraine_2022(df: pd.DataFrame) -> pd.DataFrame:
    full = df[(df["date"] >= "2021-11-01") & (df["date"] <= "2023-03-31")].copy()
    peak_date = full.loc[full["spy_price"].idxmax(), "date"]
    trough_date = full[full["date"] >= peak_date].loc[full[full["date"] >= peak_date]["spy_price"].idxmin(), "date"]
    windows = {
        "UKRAINE_INITIAL_SHOCK": ("2022-02-01", "2022-03-31"),
        "FIRST_HALF_2022": ("2022-01-01", "2022-06-30"),
        "FULL_2022_RATE_WAR": ("2021-11-01", "2023-03-31"),
        "SPY_2022_PEAK_TO_TROUGH": (str(peak_date.date()), str(trough_date.date())),
    }
    rows = []
    for name, (start, end) in windows.items():
        sub = df[(df["date"] >= start) & (df["date"] <= end)].copy()
        if sub.empty:
            continue
        row = {"window": name, "start_date": sub["date"].iloc[0], "end_date": sub["date"].iloc[-1]}
        for asset in ASSETS:
            nav = (1 + sub[f"{asset}_return"].fillna(0.0)).cumprod()
            row[f"{asset}_return"] = nav.iloc[-1] - 1.0
            row[f"{asset}_maxDD"] = (nav / nav.cummax() - 1.0).min()
        row["GOLD_minus_CASH_return"] = row["GOLD_return"] - row["CASH_return"]
        row["GOLD_minus_CASH_path_difference"] = row["GOLD_maxDD"] - row["CASH_maxDD"]
        for strat in ["MATURE_BASELINE_REGIME_HEDGE_INV_VOL", "BACKBONE_V2_SPY_CASH"]:
            if f"{strat}_return" in sub.columns:
                nav = (1 + sub[f"{strat}_return"].fillna(0.0)).cumprod()
                row[f"{strat}_return"] = nav.iloc[-1] - 1.0
                row[f"{strat}_maxDD"] = (nav / nav.cummax() - 1.0).min()
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(CONFIG["output_dir"] / "ukraine_2022_gold_cash_analysis.csv", index=False)
    return out


def run_flat_risk_variants(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    base_weights = {asset: out[f"MATURE_BASELINE_REGIME_HEDGE_INV_VOL_weight_{asset}"].fillna(0.0).to_numpy() for asset in ASSETS}
    base_state = out["MATURE_BASELINE_REGIME_HEDGE_INV_VOL_risk_state"].fillna("NON_RISK").astype(str)
    regime = out["macro_regime_confirmed"].fillna("UNKNOWN").astype(str)
    asset_returns = {asset: out[f"{asset}_return"].fillna(0.0).to_numpy() for asset in ASSETS}
    perf_rows = []
    for variant, hedge in VARIANTS.items():
        weights = {asset: base_weights[asset].copy() for asset in ASSETS}
        mask = base_state.eq("FULL_RISK") & regime.eq("FLAT")
        for asset in ASSETS:
            weights[asset][mask.to_numpy()] = hedge.get(asset, 0.0)
        turnover = np.zeros(len(out))
        tcost = np.zeros(len(out))
        ret = np.zeros(len(out))
        for i in range(len(out)):
            if i > 0:
                tw = sum(abs(weights[a][i] - weights[a][i - 1]) for a in ASSETS)
                turnover[i] = tw
                tcost[i] = 0.5 * tw * CONFIG["one_way_cost_bps"] / 10000.0
            ret[i] = sum(weights[a][i] * asset_returns[a][i] for a in ASSETS) - tcost[i]
        out[f"{variant}_return"] = ret
        out[f"{variant}_nav"] = np.cumprod(1 + ret)
        for asset in ASSETS:
            out[f"{variant}_weight_{asset}"] = weights[asset]
        out[f"{variant}_turnover"] = turnover
        out[f"{variant}_transaction_cost"] = tcost
        rf = out["CASH_return"].fillna(0.0)
        full = _segment_metrics(out[f"{variant}_return"], rf)
        row = {
            "strategy": variant,
            "AnnRet": full["AnnRet"],
            "Sharpe": full["Sharpe"],
            "Sortino": full["Sortino"],
            "MaxDD": full["MaxDD"],
            "Calmar": full["AnnRet"] / abs(full["MaxDD"]) if pd.notna(full["MaxDD"]) and full["MaxDD"] < 0 else np.nan,
            "Final NAV": full["FinalNAV"],
            "time_in_FLAT_RISK": mask.mean(),
            "turnover": turnover.sum(),
            "cost_drag": tcost.sum(),
        }
        for name, (start, end) in CRISIS_WINDOWS.items():
            sub = out[(out["date"] >= start) & (out["date"] <= end)]
            met = _segment_metrics(sub[f"{variant}_return"], sub["CASH_return"])
            row[f"{name}_return"] = (1 + sub[f"{variant}_return"]).prod() - 1.0
            row[f"{name}_MaxDD"] = met["MaxDD"]
        ex2022 = out[(out["date"] < "2021-11-01") | (out["date"] > "2023-03-31")]
        met_ex = _segment_metrics(ex2022[f"{variant}_return"], ex2022["CASH_return"])
        row["EX_2022_return"] = (1 + ex2022[f"{variant}_return"]).prod() - 1.0
        row["EX_2022_MaxDD"] = met_ex["MaxDD"]
        perf_rows.append(row)
    perf = pd.DataFrame(perf_rows)
    perf.to_csv(CONFIG["output_dir"] / "flat_risk_hedge_variant_performance.csv", index=False)
    keep = ["date", "macro_regime_confirmed", "MATURE_BASELINE_REGIME_HEDGE_INV_VOL_risk_state"]
    for variant in VARIANTS:
        keep.extend([f"{variant}_return", f"{variant}_nav", f"{variant}_turnover"] + [f"{variant}_weight_{a}" for a in ASSETS])
    out[keep].to_csv(CONFIG["output_dir"] / "flat_risk_variant_daily_panel.csv", index=False)
    return perf, out


def plot_results(episodes: pd.DataFrame, summary: pd.DataFrame, ukraine: pd.DataFrame, perf: pd.DataFrame, df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(episodes["CASH_return"], episodes["GOLD_return"])
    for _, row in episodes.iterrows():
        if row["start_date"] >= pd.Timestamp("2021-11-01") and row["start_date"] <= pd.Timestamp("2023-03-31"):
            ax.annotate(f"{int(row['episode_id'])}", (row["CASH_return"], row["GOLD_return"]))
    ax.axline((0, 0), slope=1, color="gray", linestyle="--")
    ax.set_xlabel("CASH episode return")
    ax.set_ylabel("GOLD episode return")
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "flat_risk_episode_gold_vs_cash_scatter.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(episodes["episode_id"].astype(str), episodes["GOLD_minus_CASH_return"])
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "flat_risk_gold_minus_cash_by_episode.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = summary["sample_name"].tolist()
    vals = summary["avg_GOLD_minus_CASH_return"].tolist()
    ax.bar(labels, vals)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "flat_risk_gold_cash_boxplot_ex_2022.png", dpi=150)
    plt.close(fig)

    sub = df[(df["date"] >= "2021-11-01") & (df["date"] <= "2023-03-31")].copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    for asset in ASSETS:
        nav = (1 + sub[f"{asset}_return"].fillna(0.0)).cumprod()
        ax.plot(sub["date"], nav, label=asset)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "ukraine_2022_asset_navs.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for strategy in ["BACKBONE_V2_SPY_CASH", "MATURE_BASELINE_REGIME_HEDGE_INV_VOL", "FLAT_RISK_CASH"]:
        col = f"{strategy}_nav"
        if col in sub.columns:
            ax.plot(sub["date"], sub[col], label=strategy)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "ukraine_2022_strategy_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for variant in VARIANTS:
        ax.plot(df["date"], df[f"{variant}_nav"], label=variant)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "flat_risk_hedge_variant_equity_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for variant in VARIANTS:
        nav = df[f"{variant}_nav"]
        dd = nav / nav.cummax() - 1.0
        ax.plot(df["date"], dd, label=variant)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "flat_risk_hedge_variant_drawdown.png", dpi=150)
    plt.close(fig)


def write_markdown_report(summary: pd.DataFrame, ukraine: pd.DataFrame, perf: pd.DataFrame) -> None:
    lines = [
        "# FLAT RISK GOLD CASH Diagnostic Report",
        "",
        "## 2022",
        ukraine.to_markdown(index=False),
        "",
        "## Episode Summary",
        summary.to_markdown(index=False),
        "",
        "## Hedge Variants",
        perf.to_markdown(index=False),
        "",
        "## Interpretation",
        "- 2022 can be decomposed into final return versus path smoothness.",
        "- The all-episode and ex-2022 episode summaries show whether GOLD's advantage survives once the rate-war window is excluded.",
        "- Variant results show whether CASH or a GOLD/CASH mix is a more stable FLAT_RISK hedge than pure GOLD.",
        "",
    ]
    (CONFIG["output_dir"] / "FLAT_RISK_GOLD_CASH_DIAGNOSTIC_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = load_panel()
    episodes = extract_flat_risk_episodes(df)
    summary = summarize_gold_cash(episodes)
    ukraine = analyze_ukraine_2022(df)
    perf, variant_df = run_flat_risk_variants(df)
    plot_results(episodes, summary, ukraine, perf, variant_df)
    write_markdown_report(summary, ukraine, perf)

    print("1. 2022 GOLD vs CASH:")
    print(ukraine[["window", "GOLD_return", "CASH_return", "GOLD_minus_CASH_return", "GOLD_maxDD", "CASH_maxDD"]].to_string(index=False))
    print("2. FLAT_RISK all episodes summary:")
    print(summary[summary["sample_name"].eq("ALL_FLAT_RISK_EPISODES")].to_string(index=False))
    print("3. FLAT_RISK excluding 2022 summary:")
    print(summary[summary["sample_name"].eq("EX_FULL_2022_RATE_WAR")].to_string(index=False))
    print("4. Output path:", CONFIG["output_dir"])


if __name__ == "__main__":
    main()
