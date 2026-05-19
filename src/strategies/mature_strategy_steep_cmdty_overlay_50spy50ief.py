"""Mature strategy diagnostic with STEEP commodity slow-growth overlay."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIG = {
    "output_dir": Path("results/mature_steep_cmdty_overlay_50spy50ief"),
    "figure_dir": Path("figures/mature_steep_cmdty_overlay_50spy50ief"),
    "one_way_cost_bps": 5,
    "case_2015_start": "2015-05-01",
    "case_2015_peak": "2015-07-20",
    "case_2015_trough": "2016-02-11",
    "case_2015_end": "2016-03-31",
}

PANEL_CANDIDATES = [
    Path("results/backbone_v2_with_steep_commodity_stress/daily_backtest_panel.csv"),
    Path("results/regime_aware_risk_parity_allocation/daily_backtest_panel.csv"),
]

ASSETS = ["SPY", "GOLD", "CMDTY_FUT", "IEF", "CASH"]
BENCHMARK_STRATEGIES = [
    "SPY_BUY_HOLD",
    "BACKBONE_V2_SPY_CASH",
    "REGIME_HEDGE_V1_ORIGINAL",
    "MATURE_BASELINE_REGIME_HEDGE_INV_VOL",
    "MATURE_FULL_ONE_RET60",
]
NEW_STRATEGY = "MATURE_STEEP_CMDTY_OVERLAY_50SPY_50IEF"
CRISIS_WINDOWS = {
    "2015_2016": ("2015-05-01", "2016-03-31"),
    "COVID_2020": ("2020-02-01", "2020-06-30"),
    "2022": ("2021-11-01", "2023-03-31"),
    "2025_PULLBACK": ("2025-01-01", "2025-12-31"),
    "2008_GFC": ("2007-10-01", "2009-06-30"),
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


def load_panel() -> tuple[pd.DataFrame, Path]:
    panel_path = next((p for p in PANEL_CANDIDATES if p.exists()), None)
    if panel_path is None:
        raise FileNotFoundError("No mature baseline panel found.")
    df = _read_csv(panel_path)
    required = [
        "date",
        "macro_regime_confirmed",
        "MATURE_BASELINE_REGIME_HEDGE_INV_VOL_risk_state",
        "MATURE_BASELINE_REGIME_HEDGE_INV_VOL_weight_SPY",
        "MATURE_BASELINE_REGIME_HEDGE_INV_VOL_weight_GOLD",
        "MATURE_BASELINE_REGIME_HEDGE_INV_VOL_weight_CMDTY_FUT",
        "MATURE_BASELINE_REGIME_HEDGE_INV_VOL_weight_IEF",
        "MATURE_BASELINE_REGIME_HEDGE_INV_VOL_weight_CASH",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required mature baseline columns: {missing}")
    bad_regime = df.loc[~df["macro_regime_confirmed"].isin(["FLAT", "STEEP", "INVERTED"]), ["date", "macro_regime_confirmed"]]
    if not bad_regime.empty:
        bad_regime.to_csv(CONFIG["output_dir"] / "unexpected_regimes.csv", index=False)
        warnings.warn(f"Unexpected regimes found on {len(bad_regime)} dates.")
    df["SPY_return"] = pd.to_numeric(_col(df, ["SPY_return", "spy_daily_return"]), errors="coerce").fillna(0.0)
    df["GOLD_return"] = pd.to_numeric(_col(df, ["GOLD_return", "GLD_return"]), errors="coerce").fillna(0.0)
    df["IEF_return"] = pd.to_numeric(_col(df, ["IEF_return"]), errors="coerce").fillna(0.0)
    df["CMDTY_FUT_return"] = pd.to_numeric(_col(df, ["CMDTY_FUT_return"]), errors="coerce").fillna(0.0)
    df["CASH_return"] = pd.to_numeric(_col(df, ["CASH_return", "daily_rf"]), errors="coerce").fillna(0.0)
    df["SPY_CROSS_ABOVE_MA20"] = _col(df, ["SPY_CROSS_ABOVE_MA20"], False).fillna(False).astype(bool)
    df["CMDTY_RET60"] = pd.to_numeric(_col(df, ["CMDTY_RET60"]), errors="coerce")
    df["baseline_state"] = _col(df, ["MATURE_BASELINE_REGIME_HEDGE_INV_VOL_risk_state"], "NON_RISK").fillna("NON_RISK").astype(str)
    df["BACKBONE_V2_BASELINE_ENTRY"] = _col(df, ["BACKBONE_V2_BASELINE_ENTRY"], False).fillna(False).astype(bool)
    df["MATURE_BASELINE_REGIME_HEDGE_INV_VOL_entry_reason"] = _col(df, ["MATURE_BASELINE_REGIME_HEDGE_INV_VOL_entry_reason"], "").fillna("")
    print(f"Loaded panel: {panel_path}")
    return df, panel_path


def build_overlay_signal(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["STEEP_CMDTY_OVERLAY_SIGNAL"] = (
        out["macro_regime_confirmed"].eq("STEEP")
        & (out["CMDTY_RET60"] < -0.10)
        & out["baseline_state"].ne("FULL_RISK")
    )
    return out


def _baseline_weights(row: pd.Series) -> Dict[str, float]:
    return {
        asset: float(pd.to_numeric(row.get(f"MATURE_BASELINE_REGIME_HEDGE_INV_VOL_weight_{asset}", 0.0), errors="coerce") or 0.0)
        for asset in ASSETS
    }


def _portfolio_return(weights: Dict[str, float], row: pd.Series) -> float:
    return sum(weights[a] * float(row[f"{a}_return"]) for a in ASSETS)


def run_overlay_backtest(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    overlay_applied = np.zeros(len(out), dtype=bool)
    state_applied: List[str] = []
    entry_reason = [""] * len(out)
    exit_reason = [""] * len(out)
    weights_history = []
    strategy_ret = np.zeros(len(out))
    turnover = np.zeros(len(out))
    tcost = np.zeros(len(out))
    prev_weights = _baseline_weights(out.iloc[0])
    next_overlay = False
    next_reason = ""

    for i, row in out.iterrows():
        baseline_state = str(row["baseline_state"])
        if baseline_state == "FULL_RISK":
            current_state = "FULL_RISK"
            weights = {"SPY": 0.0, "GOLD": 0.0, "CMDTY_FUT": 0.0, "IEF": 1.0, "CASH": 0.0}
        elif overlay_applied[i]:
            current_state = "SLOW_GROWTH_OVERLAY"
            weights = {"SPY": 0.5, "GOLD": 0.0, "CMDTY_FUT": 0.0, "IEF": 0.5, "CASH": 0.0}
        else:
            current_state = "NON_RISK"
            weights = _baseline_weights(row)
        state_applied.append(current_state)
        if i > 0:
            tw = sum(abs(weights[a] - prev_weights[a]) for a in ASSETS)
            turnover[i] = tw
            tcost[i] = 0.5 * tw * CONFIG["one_way_cost_bps"] / 10000.0
        strategy_ret[i] = _portfolio_return(weights, row) - tcost[i]
        prev_weights = weights
        weights_history.append(weights)
        if i + 1 < len(out):
            baseline_next = str(out.iloc[i + 1]["baseline_state"])
            if baseline_next == "FULL_RISK":
                overlay_applied[i + 1] = False
                if current_state == "SLOW_GROWTH_OVERLAY":
                    exit_reason[i + 1] = "UPGRADE_TO_FULL_RISK"
            else:
                if current_state == "SLOW_GROWTH_OVERLAY":
                    overlay_applied[i + 1] = not bool(row["SPY_CROSS_ABOVE_MA20"])
                    if not overlay_applied[i + 1]:
                        exit_reason[i + 1] = "R3_SPY_CROSS_ABOVE_MA20"
                else:
                    overlay_applied[i + 1] = bool(row["STEEP_CMDTY_OVERLAY_SIGNAL"])
                    if overlay_applied[i + 1]:
                        entry_reason[i + 1] = "STEEP_CMDTY_RET60_NEG10_OVERLAY"

    out[f"{NEW_STRATEGY}_state"] = state_applied
    out[f"{NEW_STRATEGY}_entry_reason"] = entry_reason
    out[f"{NEW_STRATEGY}_exit_reason"] = exit_reason
    out[f"{NEW_STRATEGY}_full_risk_state"] = out[f"{NEW_STRATEGY}_state"].eq("FULL_RISK")
    out[f"{NEW_STRATEGY}_overlay_state"] = out[f"{NEW_STRATEGY}_state"].eq("SLOW_GROWTH_OVERLAY")
    for asset in ASSETS:
        out[f"{NEW_STRATEGY}_weight_{asset}"] = [w[asset] for w in weights_history]
    out[f"{NEW_STRATEGY}_return"] = strategy_ret
    out[f"{NEW_STRATEGY}_turnover"] = turnover
    out[f"{NEW_STRATEGY}_transaction_cost"] = tcost
    out[f"{NEW_STRATEGY}_nav"] = (1 + out[f"{NEW_STRATEGY}_return"].fillna(0.0)).cumprod()
    return out


def _perf_from_returns(ret: pd.Series, rf: pd.Series, dates: pd.Series) -> Dict[str, float]:
    nav = (1 + ret.fillna(0.0)).cumprod()
    dd = nav / nav.cummax() - 1.0
    downside = ret[ret < 0]
    month_key = pd.to_datetime(dates).dt.to_period("M")
    monthly = (1 + ret.fillna(0.0)).groupby(month_key).prod() - 1
    roll3 = (1 + monthly).rolling(3).apply(np.prod, raw=True) - 1 if len(monthly) >= 3 else pd.Series(dtype=float)
    roll12 = (1 + monthly).rolling(12).apply(np.prod, raw=True) - 1 if len(monthly) >= 12 else pd.Series(dtype=float)
    ann_ret = nav.iloc[-1] ** (252 / len(ret)) - 1 if len(ret) else np.nan
    ann_vol = ret.std(ddof=0) * math.sqrt(252)
    excess = ret - rf
    sharpe = excess.mean() / excess.std(ddof=0) * math.sqrt(252) if excess.std(ddof=0) > 0 else np.nan
    sortino = excess.mean() / downside.std(ddof=0) * math.sqrt(252) if len(downside) and downside.std(ddof=0) > 0 else np.nan
    maxdd = dd.min()
    return {
        "AnnRet": ann_ret,
        "AnnVol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": maxdd,
        "Calmar": ann_ret / abs(maxdd) if pd.notna(maxdd) and maxdd < 0 else np.nan,
        "Worst Month": monthly.min() if len(monthly) else np.nan,
        "Worst 3M": roll3.min() if len(roll3) else np.nan,
        "Worst 12M": roll12.min() if len(roll12) else np.nan,
        "Final NAV": nav.iloc[-1] if len(nav) else np.nan,
    }


def compute_performance_tables(df: pd.DataFrame, strategies: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    crisis_rows = []
    rf = df["CASH_return"].fillna(0.0)
    for strategy in strategies:
        ret = df[f"{strategy}_return"].fillna(0.0)
        metrics = _perf_from_returns(ret, rf, df["date"])
        row = {"strategy": strategy, **metrics}
        if f"{strategy}_full_risk_state" in df.columns:
            row["time_in_full_risk"] = df[f"{strategy}_full_risk_state"].mean()
        else:
            row["time_in_full_risk"] = df.get(f"{strategy}_risk_state", pd.Series("NON_RISK", index=df.index)).eq("FULL_RISK").mean()
        if f"{strategy}_overlay_state" in df.columns:
            row["time_in_overlay"] = df[f"{strategy}_overlay_state"].mean()
        else:
            row["time_in_overlay"] = 0.0
        row["turnover"] = df.get(f"{strategy}_turnover", pd.Series(0.0, index=df.index)).fillna(0.0).sum()
        row["cost_drag"] = df.get(f"{strategy}_transaction_cost", pd.Series(0.0, index=df.index)).fillna(0.0).sum()
        rows.append(row)
        for name, (start, end) in CRISIS_WINDOWS.items():
            sub = df[(df["date"] >= start) & (df["date"] <= end)]
            if sub.empty:
                continue
            met = _perf_from_returns(sub[f"{strategy}_return"].fillna(0.0), sub["CASH_return"].fillna(0.0), sub["date"])
            crisis_rows.append(
                {
                    "strategy": strategy,
                    "window": name,
                    "cumulative_return": (1 + sub[f"{strategy}_return"].fillna(0.0)).prod() - 1.0,
                    "max_drawdown": met["MaxDD"],
                    "annualized_return": met["AnnRet"],
                }
            )
    perf = pd.DataFrame(rows)
    crisis = pd.DataFrame(crisis_rows)
    perf.to_csv(CONFIG["output_dir"] / "performance_summary.csv", index=False)
    crisis.to_csv(CONFIG["output_dir"] / "crisis_performance.csv", index=False)
    return perf, crisis


def analyze_case_2015(df: pd.DataFrame, strategies: List[str]) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp(CONFIG["case_2015_start"])
    peak = pd.Timestamp(CONFIG["case_2015_peak"])
    trough = pd.Timestamp(CONFIG["case_2015_trough"])
    end = pd.Timestamp(CONFIG["case_2015_end"])
    full = df[(df["date"] >= start) & (df["date"] <= end)]
    peak_trough = df[(df["date"] >= peak) & (df["date"] <= trough)]
    for strategy in strategies:
        first_entry = pd.NaT
        reason = ""
        if strategy == NEW_STRATEGY:
            enters = full[full[f"{strategy}_entry_reason"].astype(str) != ""]
            if not enters.empty:
                first_entry = enters["date"].iloc[0]
                reason = enters[f"{strategy}_entry_reason"].iloc[0]
        elif f"{strategy}_risk_state" in full.columns:
            state = full[f"{strategy}_risk_state"].fillna("NON_RISK").astype(str)
            enters = full[state.ne(state.shift(1, fill_value="NON_RISK")) & state.isin(["FULL_RISK", "RISK"])]
            if not enters.empty:
                first_entry = enters["date"].iloc[0]
                reason = enters.get(f"{strategy}_entry_reason", pd.Series("", index=enters.index)).iloc[0]
        rows.append(
            {
                "strategy": strategy,
                "first_entry_date_in_case": first_entry,
                "entry_reason": reason,
                "days_after_peak": (first_entry - peak).days if pd.notna(first_entry) else np.nan,
                "days_before_trough": (trough - first_entry).days if pd.notna(first_entry) else np.nan,
                "cumulative_return_full_case": (1 + full[f"{strategy}_return"].fillna(0.0)).prod() - 1.0,
                "max_drawdown_full_case": ((1 + full[f"{strategy}_return"].fillna(0.0)).cumprod() / (1 + full[f"{strategy}_return"].fillna(0.0)).cumprod().cummax() - 1).min(),
                "cumulative_return_peak_to_trough": (1 + peak_trough[f"{strategy}_return"].fillna(0.0)).prod() - 1.0,
                "max_drawdown_peak_to_trough": ((1 + peak_trough[f"{strategy}_return"].fillna(0.0)).cumprod() / (1 + peak_trough[f"{strategy}_return"].fillna(0.0)).cumprod().cummax() - 1).min(),
                "time_in_overlay_full_case": full.get(f"{strategy}_overlay_state", pd.Series(False, index=full.index)).mean(),
                "time_in_full_risk_full_case": full.get(f"{strategy}_full_risk_state", pd.Series(False, index=full.index)).mean(),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(CONFIG["output_dir"] / "case_2015_2016_entry_comparison.csv", index=False)
    return out


def plot_results(df: pd.DataFrame, strategies: List[str]) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    for strategy in strategies:
        ax.plot(df["date"], df[f"{strategy}_nav"], label=strategy)
    ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Equity Curve")
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "equity_curve_log.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 7))
    for strategy in strategies:
        nav = df[f"{strategy}_nav"]
        dd = nav / nav.cummax() - 1.0
        ax.plot(df["date"], dd, label=strategy)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Drawdown")
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "drawdown_comparison.png", dpi=150)
    plt.close(fig)


def write_markdown_report(perf: pd.DataFrame, crisis: pd.DataFrame, case_2015: pd.DataFrame) -> None:
    def row(name: str) -> str:
        r = perf.loc[perf["strategy"].eq(name)].iloc[0]
        return f"| {name} | {r['AnnRet']:.2%} | {r['Sharpe']:.3f} | {r['MaxDD']:.2%} | {r['Final NAV']:.2f} |"

    lines = [
        "# STEEP CMDTY OVERLAY 50SPY50IEF Report",
        "",
        "## Purpose",
        "Test whether a STEEP-only commodity slow-growth overlay improves 2015-2016 without changing the mature baseline backbone.",
        "",
        "## Main Performance",
        "| Strategy | AnnRet | Sharpe | MaxDD | Final NAV |",
        "|---|---:|---:|---:|---:|",
        row("MATURE_BASELINE_REGIME_HEDGE_INV_VOL"),
        row("MATURE_FULL_ONE_RET60"),
        row(NEW_STRATEGY),
        "",
        "## 2015-2016",
        case_2015[case_2015["strategy"].isin(["MATURE_BASELINE_REGIME_HEDGE_INV_VOL", "MATURE_FULL_ONE_RET60", NEW_STRATEGY])]
        .to_markdown(index=False),
        "",
        "## Crisis Windows",
        crisis[crisis["strategy"].isin(["MATURE_BASELINE_REGIME_HEDGE_INV_VOL", "MATURE_FULL_ONE_RET60", NEW_STRATEGY])].to_markdown(index=False),
        "",
        "## Interpretation",
        "- Overlay only acts in STEEP non-risk when commodity slow-growth weakness appears.",
        "- Full risk still overrides overlay and remains 100% IEF.",
        "- The comparison to `MATURE_FULL_ONE_RET60` shows whether a 50/50 overlay is a less aggressive repair than direct full-risk entry.",
        "",
    ]
    (CONFIG["output_dir"] / "STEEP_CMDTY_OVERLAY_50SPY50IEF_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df, src = load_panel()
    df = build_overlay_signal(df)
    df = run_overlay_backtest(df)
    strategies = BENCHMARK_STRATEGIES + [NEW_STRATEGY]
    perf, crisis = compute_performance_tables(df, strategies)
    case_2015 = analyze_case_2015(df, strategies)
    keep_cols = ["date", "macro_regime_confirmed", "baseline_state", "CMDTY_RET60", "SPY_CROSS_ABOVE_MA20", "STEEP_CMDTY_OVERLAY_SIGNAL"]
    for strategy in strategies:
        for suffix in ["return", "nav", "weight_SPY", "weight_GOLD", "weight_CMDTY_FUT", "weight_IEF", "weight_CASH", "risk_state", "state", "overlay_state", "full_risk_state", "entry_reason", "turnover", "transaction_cost"]:
            col = f"{strategy}_{suffix}"
            if col in df.columns:
                keep_cols.append(col)
    df[sorted(set(keep_cols), key=keep_cols.index)].to_csv(CONFIG["output_dir"] / "daily_backtest_panel.csv", index=False)
    plot_results(df, strategies)
    write_markdown_report(perf, crisis, case_2015)

    def _show(name: str) -> str:
        r = perf.loc[perf["strategy"].eq(name)].iloc[0]
        return f"{name}: AnnRet {r['AnnRet']:.2%}, Sharpe {r['Sharpe']:.3f}, MaxDD {r['MaxDD']:.2%}, Final NAV {r['Final NAV']:.2f}"

    print("1.", _show("MATURE_BASELINE_REGIME_HEDGE_INV_VOL"))
    print("2.", _show(NEW_STRATEGY))
    print("3.", _show("MATURE_FULL_ONE_RET60"))
    print("4. 2015-2016 comparison:")
    print(case_2015[case_2015["strategy"].isin(["MATURE_BASELINE_REGIME_HEDGE_INV_VOL", "MATURE_FULL_ONE_RET60", NEW_STRATEGY])][["strategy", "cumulative_return_full_case", "max_drawdown_full_case", "cumulative_return_peak_to_trough"]].to_string(index=False))
    print("5. COVID / 2022 / 2025 comparison:")
    print(crisis[crisis["strategy"].isin(["MATURE_BASELINE_REGIME_HEDGE_INV_VOL", "MATURE_FULL_ONE_RET60", NEW_STRATEGY]) & crisis["window"].isin(["COVID_2020", "2022", "2025_PULLBACK"])][["strategy", "window", "cumulative_return", "max_drawdown"]].to_string(index=False))
    print(f"6. Output path: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
