"""Two-fund separation experiment on top of the current baseline risky fund.

This script treats the current baseline strategy as the risky fund and CASH
as the safe fund. It does not modify the timing backbone, internal regime
allocation, rebalance frequency, or strategy parameters.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIG = {
    "output_dir": Path("results/two_fund_separation_experiment"),
    "figure_dir": Path("figures/two_fund_separation_experiment"),
    "one_way_cost_bps": 5,
    "monthly_rebalance": True,
    "vol_lookback": 63,
    "vol_targets": [0.06, 0.08, 0.10],
    "max_risky_weight": 1.0,
}

PANEL_CANDIDATES = [
    Path("results/steep_risk_duration_and_ijh_experiment/daily_backtest_panel.csv"),
    Path("results/regime_aware_risk_parity_allocation/daily_backtest_panel.csv"),
]

CRISIS_WINDOWS = {
    "2008_GFC": ("2007-10-01", "2009-06-30"),
    "2020_COVID": ("2020-02-01", "2020-06-30"),
    "2022_RATE_HIKE": ("2021-11-01", "2023-03-31"),
    "2025_PULLBACK": ("2025-01-01", "2025-12-31"),
}

STRATEGIES = [
    "BASELINE",
    "BASELINE_80_20",
    "BASELINE_70_30",
    "BASELINE_60_40",
    "VOL_TARGET_6_CAP_1_0",
    "VOL_TARGET_8_CAP_1_0",
    "VOL_TARGET_10_CAP_1_0",
]


def ensure_dirs() -> None:
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["figure_dir"].mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        for alt in ["DATE", "Date", "observation_date"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "date"})
                break
    if "date" not in df.columns:
        raise ValueError(f"No date column in {path}")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def _is_first_trading_day_of_month(dates: pd.Series) -> pd.Series:
    periods = dates.dt.to_period("M")
    out = periods.ne(periods.shift(1))
    out.iloc[0] = True
    return out


def _max_drawdown(ret: pd.Series) -> float:
    nav = (1 + ret.fillna(0.0)).cumprod()
    return float((nav / nav.cummax() - 1).min()) if not nav.empty else np.nan


def _sortino(ret: pd.Series, rf: pd.Series) -> float:
    excess = ret - rf
    downside = excess[excess < 0]
    if downside.empty:
        return np.nan
    dstd = downside.std(ddof=0)
    if dstd <= 0:
        return np.nan
    return float(excess.mean() / dstd * math.sqrt(252))


def _worst_rolling_period(ret: pd.Series, dates: pd.Series, months: int) -> float:
    monthly = (1 + ret.fillna(0.0).set_axis(pd.to_datetime(dates))).resample("ME").prod() - 1
    if monthly.empty:
        return np.nan
    if months == 1:
        return float(monthly.min())
    roll = (1 + monthly).rolling(months).apply(np.prod, raw=True) - 1
    return float(roll.min()) if roll.notna().any() else np.nan


def _annualized_stats(ret: pd.Series, rf: pd.Series, dates: pd.Series) -> Dict[str, float]:
    ret = ret.fillna(0.0)
    rf = rf.reindex(ret.index).fillna(0.0)
    nav = (1 + ret).cumprod()
    years = len(ret) / 252
    ann = nav.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = ret.std(ddof=0) * math.sqrt(252)
    excess = ret - rf
    sharpe = excess.mean() / excess.std(ddof=0) * math.sqrt(252) if excess.std(ddof=0) > 0 else np.nan
    sortino = _sortino(ret, rf)
    maxdd = _max_drawdown(ret)
    calmar = ann / abs(maxdd) if pd.notna(maxdd) and maxdd < 0 else np.nan
    return {
        "AnnRet": ann,
        "AnnVol": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": maxdd,
        "Calmar": calmar,
        "Worst Month": _worst_rolling_period(ret, dates, 1),
        "Worst 3M": _worst_rolling_period(ret, dates, 3),
        "Worst 12M": _worst_rolling_period(ret, dates, 12),
        "Final NAV": nav.iloc[-1],
    }


def load_panel() -> pd.DataFrame:
    for path in PANEL_CANDIDATES:
        if path.exists():
            df = _read_csv(path)
            print(f"Loaded panel: {path}")
            baseline_col = "BASELINE_return" if "BASELINE_return" in df.columns else None
            if baseline_col is None:
                raise ValueError(f"BASELINE_return not found in {path}")
            if "CASH_return" not in df.columns:
                if "daily_rf" in df.columns:
                    df["CASH_return"] = pd.to_numeric(df["daily_rf"], errors="coerce").fillna(0.0)
                else:
                    raise ValueError("CASH_return and daily_rf both missing.")
            df["baseline_source_return"] = pd.to_numeric(df[baseline_col], errors="coerce").fillna(0.0)
            if "BASELINE_nav" in df.columns:
                df["baseline_source_nav"] = pd.to_numeric(df["BASELINE_nav"], errors="coerce")
            else:
                df["baseline_source_nav"] = (1 + df["baseline_source_return"]).cumprod()
            df["CASH_return"] = pd.to_numeric(df["CASH_return"], errors="coerce").fillna(0.0)
            return df
    raise FileNotFoundError("No prior baseline panel found.")


def build_weights(panel: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"date": panel["date"]})
    out["BASELINE_weight_risky"] = 1.0
    out["BASELINE_80_20_weight_risky"] = 0.8
    out["BASELINE_70_30_weight_risky"] = 0.7
    out["BASELINE_60_40_weight_risky"] = 0.6

    realized = panel["baseline_source_return"].rolling(
        CONFIG["vol_lookback"], min_periods=CONFIG["vol_lookback"]
    ).std(ddof=0) * math.sqrt(252)
    realized_lagged = realized.shift(1)
    first_month = _is_first_trading_day_of_month(panel["date"])

    for target in CONFIG["vol_targets"]:
        name = f"VOL_TARGET_{int(target * 100)}_CAP_1_0"
        weights: List[float] = []
        current = 1.0
        for i in range(len(panel)):
            if i == 0:
                weights.append(current)
                continue
            if bool(first_month.iloc[i]):
                vol = realized_lagged.iloc[i]
                if pd.notna(vol) and vol > 0:
                    current = min(CONFIG["max_risky_weight"], max(0.0, float(target / vol)))
            weights.append(current)
        out[f"{name}_weight_risky"] = weights
    return out


def run_two_fund_backtest(panel: pd.DataFrame, weights: pd.DataFrame, strategy: str) -> pd.DataFrame:
    df = panel.copy()
    risky_col = f"{strategy}_weight_risky"
    df[risky_col] = pd.to_numeric(weights[risky_col], errors="coerce").fillna(1.0)
    df[f"{strategy}_weight_cash"] = 1.0 - df[risky_col]
    df[f"{strategy}_turnover"] = 0.0
    df[f"{strategy}_transaction_cost"] = 0.0
    df[f"{strategy}_return"] = 0.0
    df[f"{strategy}_nav"] = 1.0

    current_risky = float(df.loc[0, risky_col])
    current_cash = 1.0 - current_risky
    nav = 1.0
    for i in range(len(df)):
        target_risky = float(df.loc[i, risky_col])
        target_cash = 1.0 - target_risky
        turnover = 0.0
        cost = 0.0
        if i == 0:
            turnover = abs(target_risky) + abs(target_cash)
            cost = 0.5 * turnover * CONFIG["one_way_cost_bps"] / 10000
            current_risky = target_risky
            current_cash = target_cash
        elif abs(target_risky - current_risky) > 1e-12 or abs(target_cash - current_cash) > 1e-12:
            turnover = abs(target_risky - current_risky) + abs(target_cash - current_cash)
            cost = 0.5 * turnover * CONFIG["one_way_cost_bps"] / 10000
            current_risky = target_risky
            current_cash = target_cash
        gross = current_risky * df.loc[i, "baseline_source_return"] + current_cash * df.loc[i, "CASH_return"]
        net = gross - cost
        nav *= 1 + net
        df.loc[i, f"{strategy}_turnover"] = turnover
        df.loc[i, f"{strategy}_transaction_cost"] = cost
        df.loc[i, f"{strategy}_return"] = net
        df.loc[i, f"{strategy}_nav"] = nav
    return df


def compute_performance_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy in STRATEGIES:
        stats = _annualized_stats(panel[f"{strategy}_return"], panel["CASH_return"], panel["date"])
        rows.append({"strategy": strategy, **stats})
    return pd.DataFrame(rows)


def compute_difference_vs_baseline(perf: pd.DataFrame) -> pd.DataFrame:
    base = perf.loc[perf["strategy"].eq("BASELINE")].iloc[0]
    rows = []
    for _, row in perf[perf["strategy"].ne("BASELINE")].iterrows():
        rows.append(
            {
                "strategy": row["strategy"],
                "dAnnRet": row["AnnRet"] - base["AnnRet"],
                "dAnnVol": row["AnnVol"] - base["AnnVol"],
                "dSharpe": row["Sharpe"] - base["Sharpe"],
                "dMaxDD": row["MaxDD"] - base["MaxDD"],
                "dCalmar": row["Calmar"] - base["Calmar"],
                "dFinalNAV": row["Final NAV"] - base["Final NAV"],
            }
        )
    return pd.DataFrame(rows)


def compute_weight_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy in [s for s in STRATEGIES if s.startswith("VOL_TARGET_")]:
        risky_col = f"{strategy}_weight_risky"
        weights = pd.to_numeric(panel[risky_col], errors="coerce")
        rows.append(
            {
                "strategy": strategy,
                "average_weight": weights.mean(),
                "min_weight": weights.min(),
                "max_weight": weights.max(),
                "percent_time_at_cap": weights.eq(CONFIG["max_risky_weight"]).mean(),
            }
        )
    return pd.DataFrame(rows)


def compute_crisis_performance(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, (start, end) in CRISIS_WINDOWS.items():
        sub = panel[(panel["date"] >= pd.Timestamp(start)) & (panel["date"] <= pd.Timestamp(end))]
        if sub.empty:
            continue
        for strategy in STRATEGIES:
            stats = _annualized_stats(sub[f"{strategy}_return"], sub["CASH_return"], sub["date"])
            rows.append({"period": period, "strategy": strategy, **stats})
    return pd.DataFrame(rows)


def plot_nav(panel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    for strategy in STRATEGIES:
        ax.plot(panel["date"], panel[f"{strategy}_nav"], label=strategy)
    ax.set_yscale("log")
    ax.set_title("Two-Fund Separation NAV Comparison")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "nav_comparison.png", dpi=150)
    plt.close(fig)


def plot_drawdown(panel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    for strategy in STRATEGIES:
        nav = panel[f"{strategy}_nav"]
        ax.plot(panel["date"], nav / nav.cummax() - 1, label=strategy)
    ax.set_title("Two-Fund Separation Drawdown Comparison")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "drawdown_comparison.png", dpi=150)
    plt.close(fig)


def plot_risky_weights(panel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    for strategy in [s for s in STRATEGIES if s.startswith("VOL_TARGET_")]:
        ax.plot(panel["date"], panel[f"{strategy}_weight_risky"], label=strategy)
    ax.set_ylim(0, 1.05)
    ax.set_title("Vol Target Risky Weight")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "risky_weight_timeseries.png", dpi=150)
    plt.close(fig)


def write_report(perf: pd.DataFrame, diff: pd.DataFrame, weights: pd.DataFrame, crisis: pd.DataFrame) -> None:
    out = CONFIG["output_dir"] / "TWO_FUND_SEPARATION_EXPERIMENT.md"
    fixed = perf[perf["strategy"].isin(["BASELINE_80_20", "BASELINE_70_30", "BASELINE_60_40"])].sort_values(
        ["Sharpe", "Calmar"], ascending=[False, False]
    )
    vol_target = perf[perf["strategy"].str.startswith("VOL_TARGET_")].sort_values(
        ["Sharpe", "Calmar"], ascending=[False, False]
    )
    best_fixed = fixed.iloc[0]
    best_vol = vol_target.iloc[0]
    content = f"""# two_fund_separation_experiment

## Purpose

This experiment treats the current baseline as the risky fund and CASH as the safe fund. It does not change timing backbone, regime allocation, rebalance frequency, or strategy internals.

## Assumption

- CASH return uses the project's existing safe return series (`CASH_return` / daily rf proxy).
- No leverage is allowed.
- Vol targeting uses lagged 63d realized vol and monthly rebalance.
- Risky fund weight is clipped to `[0, 1.0]`.

## Full-period Performance

{perf.to_markdown(index=False)}

## Difference vs Baseline

{diff.to_markdown(index=False)}

## Vol Target Weight Summary

{weights.to_markdown(index=False)}

## Crisis Performance

{crisis.to_markdown(index=False)}

## Summary

- Best fixed cash buffer by Sharpe: `{best_fixed["strategy"]}`
- Best vol target by Sharpe: `{best_vol["strategy"]}`
- Evaluate whether lower outer risky weight improves Sharpe / Calmar more than it sacrifices return.
"""
    out.write_text(content, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = load_panel()
    weights = build_weights(panel)
    run_panel = panel.copy()
    for strategy in STRATEGIES:
        run_panel = run_two_fund_backtest(run_panel, weights, strategy)
    perf = compute_performance_summary(run_panel)
    diff = compute_difference_vs_baseline(perf)
    weight_summary = compute_weight_summary(run_panel)
    crisis = compute_crisis_performance(run_panel)

    run_panel.to_csv(CONFIG["output_dir"] / "daily_backtest_panel.csv", index=False)
    perf.to_csv(CONFIG["output_dir"] / "performance_summary.csv", index=False)
    diff.to_csv(CONFIG["output_dir"] / "difference_vs_baseline.csv", index=False)
    weight_summary.to_csv(CONFIG["output_dir"] / "vol_target_weight_summary.csv", index=False)
    crisis.to_csv(CONFIG["output_dir"] / "crisis_performance.csv", index=False)
    plot_nav(run_panel)
    plot_drawdown(run_panel)
    plot_risky_weights(run_panel)
    write_report(perf, diff, weight_summary, crisis)


if __name__ == "__main__":
    main()
