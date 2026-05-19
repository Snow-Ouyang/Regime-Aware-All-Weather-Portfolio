"""SPY/CASH timing backbone upgrade and ablation backtest."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIG = {
    "vix_z_threshold": 3.0,
    "dd_threshold": -0.05,
    "credit_change_threshold": 0.10,
    "recovery_ma_window": 20,
    "one_way_cost_bps": 5,
    "output_dir": Path("results/spy_cash_backbone_upgrade_ablation"),
    "figure_dir": Path("figures/spy_cash_backbone_upgrade_ablation"),
}

PANEL_CANDIDATES = [
    Path("results/flat_vix_credit_trigger_diagnostic/full_backtest_panel.csv"),
    Path("results/spy_cash_stress_recovery_with_credit/daily_backtest_panel.csv"),
    Path("results/spy_cash_stress_recovery_timing/daily_backtest_panel.csv"),
]

CRISIS_WINDOWS = {
    "2008_GFC": ["2007-10-01", "2009-06-30"],
    "2015_2016": ["2015-05-01", "2016-03-31"],
    "2018Q4": ["2018-10-01", "2019-01-31"],
    "COVID_2020": ["2020-02-01", "2020-06-30"],
    "2022": ["2021-11-01", "2023-03-31"],
    "2023": ["2023-01-01", "2023-12-31"],
    "2025_PULLBACK": ["2025-01-01", "2025-12-31"],
    "2024_2026": ["2024-01-01", "2026-12-31"],
}

BACKBONES = [
    "BACKBONE_V1_OLD",
    "BACKBONE_V2_UPGRADED",
    "BACKBONE_V2_NO_EITHER_SELL",
]


def ensure_dirs() -> None:
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["figure_dir"].mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def _merge_missing(base: pd.DataFrame, other: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in base.columns and c in other.columns]
    if not missing:
        return base
    return base.merge(other[["date"] + missing], on="date", how="left")


def load_panel() -> pd.DataFrame:
    frames = []
    for path in PANEL_CANDIDATES:
        if path.exists():
            frames.append((path, _read_csv(path)))
    if not frames:
        raise FileNotFoundError("No input panel found.")
    panel = frames[0][1].copy()
    print(f"Loaded primary panel: {frames[0][0]}")
    needed = [
        "spy_price",
        "spy_daily_return",
        "daily_rf",
        "macro_regime_confirmed",
        "monthly_either_state",
        "VIX_LEVEL",
        "VIX_ZSCORE_120D",
        "CREDIT_SPREAD_BAA_AAA",
        "D_CREDIT_SPREAD_20D",
        "spy_drawdown_from_previous_high",
        "SPY_MA20",
        "SPY_CROSS_ABOVE_MA20",
        "MONTHLY_EITHER_CONFIRM_return",
        "MONTHLY_EITHER_CONFIRM_nav",
        "MONTHLY_EITHER_CONFIRM_weight_spy",
    ]
    for _, df in frames[1:]:
        panel = _merge_missing(panel, df, needed)

    if "spy_drawdown_from_previous_high" not in panel.columns:
        panel["spy_drawdown_from_previous_high"] = panel["spy_price"] / panel["spy_price"].cummax() - 1
    if "VIX_ZSCORE_120D" not in panel.columns:
        roll = panel["VIX_LEVEL"].rolling(120)
        panel["VIX_ZSCORE_120D"] = (panel["VIX_LEVEL"] - roll.mean()) / roll.std(ddof=0)
    if "D_CREDIT_SPREAD_20D" not in panel.columns:
        panel["D_CREDIT_SPREAD_20D"] = panel["CREDIT_SPREAD_BAA_AAA"] - panel["CREDIT_SPREAD_BAA_AAA"].shift(20)
    if "SPY_MA20" not in panel.columns:
        panel["SPY_MA20"] = panel["spy_price"].rolling(CONFIG["recovery_ma_window"]).mean()
    if "SPY_CROSS_ABOVE_MA20" not in panel.columns:
        panel["SPY_CROSS_ABOVE_MA20"] = (panel["spy_price"] > panel["SPY_MA20"]) & (
            panel["spy_price"].shift(1) <= panel["SPY_MA20"].shift(1)
        )
    if "monthly_either_state" not in panel.columns:
        if "monthly_either_weight_spy" in panel.columns:
            panel["monthly_either_state"] = np.where(panel["monthly_either_weight_spy"] >= 0.5, "HOLD", "SELL")
        else:
            panel["monthly_either_state"] = "UNKNOWN"
    panel["spy_daily_return"] = pd.to_numeric(panel["spy_daily_return"], errors="coerce").fillna(0.0)
    panel["daily_rf"] = pd.to_numeric(panel["daily_rf"], errors="coerce").fillna(0.0)
    panel["macro_regime_confirmed"] = panel["macro_regime_confirmed"].fillna("NEUTRAL").astype(str)
    return panel


def build_primitive_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["FLAT_VIX_STRESS"] = out["macro_regime_confirmed"].eq("FLAT") & (out["VIX_ZSCORE_120D"] >= CONFIG["vix_z_threshold"])
    out["FLAT_CREDIT_DD5_STRESS"] = out["macro_regime_confirmed"].eq("FLAT") & (
        (out["spy_drawdown_from_previous_high"] <= CONFIG["dd_threshold"])
        & (out["D_CREDIT_SPREAD_20D"] > CONFIG["credit_change_threshold"])
    )
    out["FLAT_VIX_OR_CREDIT_STRESS"] = out["FLAT_VIX_STRESS"] | out["FLAT_CREDIT_DD5_STRESS"]
    out["STEEP_EITHER_SELL_STRESS"] = out["macro_regime_confirmed"].eq("STEEP") & out["monthly_either_state"].eq("SELL")
    out["STEEP_CREDIT_DD5_STRESS"] = out["macro_regime_confirmed"].eq("STEEP") & (
        (out["spy_drawdown_from_previous_high"] <= CONFIG["dd_threshold"])
        & (out["D_CREDIT_SPREAD_20D"] > CONFIG["credit_change_threshold"])
    )
    out["INVERTED_CREDIT_DD5_SIGNAL"] = out["macro_regime_confirmed"].eq("INVERTED") & (
        (out["spy_drawdown_from_previous_high"] <= CONFIG["dd_threshold"])
        & (out["D_CREDIT_SPREAD_20D"] > CONFIG["credit_change_threshold"])
    )
    out["CREDIT_DD5_ALL_REGIME_SIGNAL"] = (
        (out["spy_drawdown_from_previous_high"] <= CONFIG["dd_threshold"])
        & (out["D_CREDIT_SPREAD_20D"] > CONFIG["credit_change_threshold"])
    )
    return out


def build_backbone_entry_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    v1 = out["FLAT_VIX_STRESS"] | out["STEEP_EITHER_SELL_STRESS"] | out["CREDIT_DD5_ALL_REGIME_SIGNAL"]
    v2 = out["FLAT_VIX_OR_CREDIT_STRESS"] | out["STEEP_EITHER_SELL_STRESS"] | out["STEEP_CREDIT_DD5_STRESS"]
    v2_no = out["FLAT_VIX_OR_CREDIT_STRESS"] | out["STEEP_CREDIT_DD5_STRESS"]
    out["V1_OLD_entry_signal"] = v1
    out["V2_UPGRADED_entry_signal"] = v2
    out["V2_NO_EITHER_entry_signal"] = v2_no

    def reason_row(row: pd.Series, version: str) -> str:
        if version == "V1":
            if row["FLAT_VIX_STRESS"]:
                return "FLAT_VIX_STRESS"
            if row["STEEP_EITHER_SELL_STRESS"]:
                return "STEEP_EITHER_SELL_STRESS"
            if row["CREDIT_DD5_ALL_REGIME_SIGNAL"]:
                return "CREDIT_DD5_ALL_REGIME_SIGNAL"
        if version == "V2":
            if row["FLAT_VIX_OR_CREDIT_STRESS"]:
                return "FLAT_VIX_OR_CREDIT_STRESS"
            if row["STEEP_EITHER_SELL_STRESS"]:
                return "STEEP_EITHER_SELL_STRESS"
            if row["STEEP_CREDIT_DD5_STRESS"]:
                return "STEEP_CREDIT_DD5_STRESS"
        if version == "V2_NO":
            if row["FLAT_VIX_OR_CREDIT_STRESS"]:
                return "FLAT_VIX_OR_CREDIT_STRESS"
            if row["STEEP_CREDIT_DD5_STRESS"]:
                return "STEEP_CREDIT_DD5_STRESS"
        return ""

    out["V1_OLD_entry_reason"] = out.apply(lambda r: reason_row(r, "V1") if r["V1_OLD_entry_signal"] else "", axis=1)
    out["V2_UPGRADED_entry_reason"] = out.apply(lambda r: reason_row(r, "V2") if r["V2_UPGRADED_entry_signal"] else "", axis=1)
    out["V2_NO_EITHER_entry_reason"] = out.apply(lambda r: reason_row(r, "V2_NO") if r["V2_NO_EITHER_entry_signal"] else "", axis=1)
    return out


def _strategy_cols(name: str) -> Tuple[str, str, str, str, str]:
    return (
        f"{name}_risk_state",
        f"{name}_weight_spy",
        f"{name}_weight_cash",
        f"{name}_return",
        f"{name}_nav",
    )


def run_state_machine_backtest(df: pd.DataFrame, entry_signal: pd.Series, entry_reason: pd.Series, strategy_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    recovery = df["SPY_CROSS_ABOVE_MA20"].fillna(False).astype(bool)
    state = "NORMAL"
    pending_state = "NORMAL"
    pending_reason = ""
    rs, ws, costs = [], [], []
    events = []
    prev_weight = 1.0
    for i in range(len(df)):
        if pending_state != state:
            old = state
            state = pending_state
            new_weight = 0.0 if state == "RISK" else 1.0
            turnover = abs(new_weight - prev_weight) + abs((1 - new_weight) - (1 - prev_weight))
            cost = 0.5 * turnover * CONFIG["one_way_cost_bps"] / 10000
            reason = pending_reason if state == "RISK" else "R3_SPY_CROSS_ABOVE_MA20"
            events.append(
                {
                    "strategy": strategy_name,
                    "event_date": df.iloc[i]["date"],
                    "event_type": "ENTER_RISK" if state == "RISK" else "EXIT_RISK",
                    "reason": reason,
                    "macro_regime_confirmed": df.iloc[i]["macro_regime_confirmed"],
                    "monthly_either_state": df.iloc[i]["monthly_either_state"],
                    "VIX_LEVEL": df.iloc[i].get("VIX_LEVEL", np.nan),
                    "VIX_ZSCORE_120D": df.iloc[i]["VIX_ZSCORE_120D"],
                    "CREDIT_SPREAD_BAA_AAA": df.iloc[i]["CREDIT_SPREAD_BAA_AAA"],
                    "D_CREDIT_SPREAD_20D": df.iloc[i]["D_CREDIT_SPREAD_20D"],
                    "spy_drawdown_from_previous_high": df.iloc[i]["spy_drawdown_from_previous_high"],
                    "SPY_price": df.iloc[i]["spy_price"],
                    "SPY_MA20": df.iloc[i]["SPY_MA20"],
                    "previous_state": old,
                    "new_state": state,
                }
            )
            prev_weight = new_weight
            costs.append(cost)
        else:
            costs.append(0.0)
        weight = 0.0 if state == "RISK" else 1.0
        rs.append(state)
        ws.append(weight)
        next_state = state
        next_reason = ""
        if state == "NORMAL" and bool(entry_signal.iloc[i]):
            next_state = "RISK"
            next_reason = str(entry_reason.iloc[i])
        elif state == "RISK" and bool(recovery.iloc[i]):
            next_state = "NORMAL"
        pending_state = next_state
        pending_reason = next_reason

    out = pd.DataFrame(index=df.index)
    rs_col, ws_col, wc_col, ret_col, nav_col = _strategy_cols(strategy_name)
    out[rs_col] = rs
    out[ws_col] = ws
    out[wc_col] = 1 - out[ws_col]
    out[f"{strategy_name}_transaction_cost"] = costs
    out[ret_col] = out[ws_col] * df["spy_daily_return"] + out[wc_col] * df["daily_rf"] - out[f"{strategy_name}_transaction_cost"]
    out[nav_col] = (1 + out[ret_col].fillna(0.0)).cumprod()
    return out, pd.DataFrame(events)


def extract_risk_episodes(df: pd.DataFrame, strategies: List[str], event_log: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in strategies:
        rs_col, _, _, ret_col, _ = _strategy_cols(s)
        if rs_col not in df.columns:
            continue
        is_risk = df[rs_col].eq("RISK")
        starts = df.index[is_risk & ~is_risk.shift(1, fill_value=False)]
        for eid, start in enumerate(starts, 1):
            end = start
            while end + 1 < len(df) and is_risk.iloc[end + 1]:
                end += 1
            sub = df.iloc[start : end + 1]
            ev = event_log[(event_log["strategy"] == s) & (event_log["event_type"] == "ENTER_RISK") & (event_log["event_date"] == sub.iloc[0]["date"])]
            entry_reason = ev["reason"].iloc[0] if not ev.empty else ""
            spy_nav = (1 + sub["spy_daily_return"].fillna(0.0)).cumprod()
            cash_nav = (1 + sub["daily_rf"].fillna(0.0)).cumprod()
            strat_nav = (1 + sub[ret_col].fillna(0.0)).cumprod()
            rows.append(
                {
                    "strategy": s,
                    "episode_id": eid,
                    "risk_start_date": sub.iloc[0]["date"],
                    "risk_end_date": sub.iloc[-1]["date"],
                    "duration_days": len(sub),
                    "entry_reason": entry_reason,
                    "macro_regime_at_entry": sub.iloc[0]["macro_regime_confirmed"],
                    "dominant_macro_regime": sub["macro_regime_confirmed"].mode().iloc[0],
                    "SPY_drawdown_at_entry": sub.iloc[0]["spy_drawdown_from_previous_high"],
                    "VIX_ZSCORE_at_entry": sub.iloc[0]["VIX_ZSCORE_120D"],
                    "D_CREDIT_SPREAD_20D_at_entry": sub.iloc[0]["D_CREDIT_SPREAD_20D"],
                    "monthly_either_state_at_entry": sub.iloc[0]["monthly_either_state"],
                    "SPY_return_during_risk": spy_nav.iloc[-1] - 1,
                    "CASH_return_during_risk": cash_nav.iloc[-1] - 1,
                    "strategy_return_during_risk": strat_nav.iloc[-1] - 1,
                    "SPY_max_drawdown_during_risk": (spy_nav / spy_nav.cummax() - 1).min(),
                    "SPY_max_runup_during_risk": spy_nav.max() - 1,
                    "exited_by_R3_date": sub.iloc[-1]["date"],
                }
            )
    return pd.DataFrame(rows)


def compute_performance_metrics(df: pd.DataFrame, strategies: List[str], episodes: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    rows = []
    years = len(df) / 252
    avg_duration = {}
    if episodes is not None and not episodes.empty:
        avg_duration = episodes.groupby("strategy")["duration_days"].mean().to_dict()
    for s in strategies:
        ret_col = f"{s}_return"
        nav_col = f"{s}_nav"
        if ret_col not in df.columns or nav_col not in df.columns:
            continue
        ret = df[ret_col].fillna(0.0)
        nav = df[nav_col]
        excess = ret - df["daily_rf"].fillna(0.0)
        ws_col = f"{s}_weight_spy"
        wc_col = f"{s}_weight_cash"
        rows.append(
            {
                "strategy": s,
                "annualized_return": nav.iloc[-1] ** (1 / years) - 1,
                "annualized_volatility": ret.std(ddof=0) * math.sqrt(252),
                "sharpe_ratio": excess.mean() / excess.std(ddof=0) * math.sqrt(252) if excess.std(ddof=0) > 0 else np.nan,
                "max_drawdown": (nav / nav.cummax() - 1).min(),
                "calmar_ratio": (nav.iloc[-1] ** (1 / years) - 1) / abs((nav / nav.cummax() - 1).min()),
                "final_nav": nav.iloc[-1],
                "number_of_switches": int((df.get(ws_col, pd.Series(1, index=df.index)).diff().abs() > 0).sum()),
                "number_of_risk_entries": int(
                    (df.get(f"{s}_risk_state", pd.Series("NORMAL", index=df.index)).eq("RISK")
                    & ~df.get(f"{s}_risk_state", pd.Series("NORMAL", index=df.index)).shift(1, fill_value="NORMAL").eq("RISK")).sum()
                ),
                "avg_risk_episode_duration": avg_duration.get(s, np.nan),
                "time_in_cash": df.get(wc_col, pd.Series(0, index=df.index)).mean(),
                "total_turnover": int((df.get(ws_col, pd.Series(1, index=df.index)).diff().abs() > 0).sum()) * 2,
                "transaction_cost_drag": df.get(f"{s}_transaction_cost", pd.Series(0, index=df.index)).sum(),
            }
        )
    return pd.DataFrame(rows)


def compute_crisis_performance(df: pd.DataFrame, strategies: List[str]) -> pd.DataFrame:
    rows = []
    max_date = df["date"].max()
    for period, (start, end) in CRISIS_WINDOWS.items():
        sub = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= min(pd.Timestamp(end), max_date))]
        if len(sub) < 2:
            continue
        for s in strategies:
            ret_col = f"{s}_return"
            if ret_col not in sub.columns:
                continue
            ret = sub[ret_col].fillna(0.0)
            nav = (1 + ret).cumprod()
            excess = ret - sub["daily_rf"].fillna(0.0)
            rows.append(
                {
                    "period": period,
                    "strategy": s,
                    "cumulative_return": nav.iloc[-1] - 1,
                    "max_drawdown": (nav / nav.cummax() - 1).min(),
                    "volatility": ret.std(ddof=0) * math.sqrt(252),
                    "Sharpe": excess.mean() / excess.std(ddof=0) * math.sqrt(252) if excess.std(ddof=0) > 0 else np.nan,
                    "time_in_cash": sub.get(f"{s}_weight_cash", pd.Series(0, index=sub.index)).mean(),
                    "number_of_switches": int((sub.get(f"{s}_weight_spy", pd.Series(1, index=sub.index)).diff().abs() > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def compute_performance_by_regime(df: pd.DataFrame, strategies: List[str]) -> pd.DataFrame:
    rows = []
    for regime, sub in df.groupby("macro_regime_confirmed"):
        for s in strategies:
            ret_col = f"{s}_return"
            if ret_col not in sub.columns:
                continue
            ret = sub[ret_col].fillna(0.0)
            nav = (1 + ret).cumprod()
            excess = ret - sub["daily_rf"].fillna(0.0)
            rows.append(
                {
                    "macro_regime_confirmed": regime,
                    "strategy": s,
                    "n_obs": len(sub),
                    "annualized_return": nav.iloc[-1] ** (252 / len(sub)) - 1 if len(sub) > 0 else np.nan,
                    "volatility": ret.std(ddof=0) * math.sqrt(252),
                    "Sharpe": excess.mean() / excess.std(ddof=0) * math.sqrt(252) if excess.std(ddof=0) > 0 else np.nan,
                    "max_drawdown": (nav / nav.cummax() - 1).min(),
                    "time_in_cash": sub.get(f"{s}_weight_cash", pd.Series(0, index=sub.index)).mean(),
                }
            )
    return pd.DataFrame(rows)


def analyze_signal_overlap(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df[[
        "date",
        "macro_regime_confirmed",
        "STEEP_EITHER_SELL_STRESS",
        "STEEP_CREDIT_DD5_STRESS",
        "FLAT_VIX_STRESS",
        "FLAT_CREDIT_DD5_STRESS",
    ]].copy()
    out["any_high_freq_stress"] = out["FLAT_VIX_STRESS"] | out["FLAT_CREDIT_DD5_STRESS"] | out["STEEP_CREDIT_DD5_STRESS"]
    out["any_either_stress"] = out["STEEP_EITHER_SELL_STRESS"]
    out["overlap_either_and_credit"] = out["STEEP_EITHER_SELL_STRESS"] & out["STEEP_CREDIT_DD5_STRESS"]
    out["overlap_either_and_any_high_freq"] = out["STEEP_EITHER_SELL_STRESS"] & out["any_high_freq_stress"]
    steep_days = out["macro_regime_confirmed"].eq("STEEP")
    either_days = out["STEEP_EITHER_SELL_STRESS"].sum()
    high_freq_days = out["any_high_freq_stress"].sum()
    summary = pd.DataFrame(
        [
            {
                "total_days": len(out),
                "STEEP_days": int(steep_days.sum()),
                "either_sell_days": int(either_days),
                "steep_credit_days": int(out["STEEP_CREDIT_DD5_STRESS"].sum()),
                "flat_vix_days": int(out["FLAT_VIX_STRESS"].sum()),
                "flat_credit_days": int(out["FLAT_CREDIT_DD5_STRESS"].sum()),
                "either_and_steep_credit_overlap_days": int(out["overlap_either_and_credit"].sum()),
                "either_overlap_with_any_high_freq_days": int(out["overlap_either_and_any_high_freq"].sum()),
                "either_overlap_ratio_with_steep_credit": out["overlap_either_and_credit"].sum() / either_days if either_days else np.nan,
                "either_overlap_ratio_with_any_high_freq": out["overlap_either_and_any_high_freq"].sum() / either_days if either_days else np.nan,
                "high_freq_overlap_ratio_with_either": out["overlap_either_and_any_high_freq"].sum() / high_freq_days if high_freq_days else np.nan,
            }
        ]
    )
    return out, summary


def analyze_episode_overlap(df: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    v2 = episodes[episodes["strategy"] == "BACKBONE_V2_UPGRADED"].copy()
    v2_no = episodes[episodes["strategy"] == "BACKBONE_V2_NO_EITHER_SELL"].copy()
    for _, ep in v2.iterrows():
        sub = df[(df["date"] >= ep["risk_start_date"]) & (df["date"] <= ep["risk_end_date"])].copy()
        no_overlap = v2_no[(v2_no["risk_start_date"] <= ep["risk_end_date"]) & (v2_no["risk_end_date"] >= ep["risk_start_date"])]
        contains_either = bool(sub["STEEP_EITHER_SELL_STRESS"].any())
        contains_credit = bool(sub["STEEP_CREDIT_DD5_STRESS"].any())
        contains_flat = bool(sub["FLAT_VIX_OR_CREDIT_STRESS"].any())
        rows.append(
            {
                "episode_id": ep["episode_id"],
                "start": ep["risk_start_date"],
                "end": ep["risk_end_date"],
                "entry_reason": ep["entry_reason"],
                "contains_either_sell": contains_either,
                "contains_credit_stress": contains_credit,
                "contains_flat_vix_or_credit": contains_flat,
                "would_exist_without_either": not no_overlap.empty,
                "overlap_with_no_either_episode": bool(not no_overlap.empty),
                "SPY_return_during_episode": ep["SPY_return_during_risk"],
                "SPY_max_drawdown_during_episode": ep["SPY_max_drawdown_during_risk"],
                "strategy_return_during_episode": ep["strategy_return_during_risk"],
            }
        )
    return pd.DataFrame(rows)


def compute_either_incremental_value(v2_perf: pd.Series, no_perf: pd.Series, overlap: pd.DataFrame) -> pd.DataFrame:
    either_only = overlap[(overlap["contains_either_sell"]) & (~overlap["would_exist_without_either"])]
    return pd.DataFrame(
        [
            {
                "either_only_episode_count": len(either_only),
                "either_only_avg_SPY_return_during_episode": either_only["SPY_return_during_episode"].mean() if not either_only.empty else np.nan,
                "either_only_avg_SPY_max_drawdown_during_episode": either_only["SPY_max_drawdown_during_episode"].mean() if not either_only.empty else np.nan,
                "either_only_crisis_periods_covered": ",".join(sorted({str(x)[:10] for x in either_only["start"]})) if not either_only.empty else "",
                "V2_vs_NO_EITHER_annret_diff": v2_perf["annualized_return"] - no_perf["annualized_return"],
                "V2_vs_NO_EITHER_sharpe_diff": v2_perf["sharpe_ratio"] - no_perf["sharpe_ratio"],
                "V2_vs_NO_EITHER_maxdd_diff": v2_perf["max_drawdown"] - no_perf["max_drawdown"],
                "V2_vs_NO_EITHER_cash_time_diff": v2_perf["time_in_cash"] - no_perf["time_in_cash"],
                "V2_vs_NO_EITHER_switch_diff": v2_perf["number_of_switches"] - no_perf["number_of_switches"],
            }
        ]
    )


def plot_backtest_results(df: pd.DataFrame, perf: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    selected = ["SPY_BUY_HOLD", "MONTHLY_EITHER_CONFIRM", "BACKBONE_V1_OLD", "BACKBONE_V2_UPGRADED", "BACKBONE_V2_NO_EITHER_SELL", "CASH_ONLY"]
    for s in selected:
        if f"{s}_nav" in df.columns:
            ax.plot(df["date"], df[f"{s}_nav"], label=s)
    ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "equity_curve_log.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for s in selected:
        if f"{s}_nav" in df.columns:
            nav = df[f"{s}_nav"]
            ax.plot(df["date"], nav / nav.cummax() - 1, label=s)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "drawdown_comparison.png", dpi=150)
    plt.close(fig)

    show = perf[perf["strategy"].isin(selected)]
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for ax, metric in zip(axes, ["annualized_return", "sharpe_ratio", "max_drawdown", "number_of_switches", "time_in_cash"]):
        ax.bar(show["strategy"], show[metric])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=90, labelsize=7)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "performance_bar_charts.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(df["date"], df["spy_price"] / df["spy_price"].iloc[0], label="SPY")
    for s in ["BACKBONE_V1_OLD", "BACKBONE_V2_UPGRADED", "BACKBONE_V2_NO_EITHER_SELL"]:
        axes[0].plot(df["date"], df[f"{s}_nav"] / df[f"{s}_nav"].iloc[0], label=s)
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].plot(df["date"], df["spy_drawdown_from_previous_high"], label="SPY drawdown")
    axes[1].axhline(-0.05, color="red", linestyle="--", linewidth=0.8)
    axes[2].plot(df["date"], df["VIX_ZSCORE_120D"], label="VIX z120")
    axes[2].axhline(CONFIG["vix_z_threshold"], color="red", linestyle="--", linewidth=0.8)
    axes[3].plot(df["date"], df["D_CREDIT_SPREAD_20D"], label="credit chg20")
    axes[3].axhline(CONFIG["credit_change_threshold"], color="red", linestyle="--", linewidth=0.8)
    regimes = pd.Categorical(df["macro_regime_confirmed"])
    axes[4].imshow([regimes.codes], aspect="auto", extent=[df["date"].iloc[0], df["date"].iloc[-1], 0, 1], cmap="tab20")
    axes[4].set_yticks([])
    axes[4].set_title("macro regime", loc="left", fontsize=9)
    for s, level in zip(["BACKBONE_V1_OLD", "BACKBONE_V2_UPGRADED", "BACKBONE_V2_NO_EITHER_SELL"], [0, 1.2, 2.4]):
        axes[5].fill_between(df["date"], level, level + df[f"{s}_risk_state"].eq("RISK").astype(int), step="post", alpha=0.5, label=s)
    axes[5].legend(fontsize=7)
    axes[5].set_yticks([])
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "risk_state_timeline.png", dpi=150)
    plt.close(fig)


def plot_overlap_results(df: pd.DataFrame, overlap_daily: pd.DataFrame, overlap_episodes: pd.DataFrame) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(df["date"], df["STEEP_EITHER_SELL_STRESS"].astype(int), label="STEEP_EITHER")
    axes[1].plot(df["date"], df["STEEP_CREDIT_DD5_STRESS"].astype(int), label="STEEP_CREDIT")
    axes[2].plot(df["date"], df["FLAT_VIX_STRESS"].astype(int), label="FLAT_VIX")
    axes[3].plot(df["date"], df["FLAT_CREDIT_DD5_STRESS"].astype(int), label="FLAT_CREDIT")
    axes[4].plot(overlap_daily["date"], overlap_daily["any_high_freq_stress"].astype(int), label="ANY_HIGH_FREQ")
    for ax in axes:
        ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "signal_overlap_timeline.png", dpi=150)
    plt.close(fig)

    either_only = overlap_episodes[(overlap_episodes["contains_either_sell"]) & (~overlap_episodes["would_exist_without_either"])]
    if not either_only.empty:
        sel = either_only.head(3)
        fig, axes = plt.subplots(len(sel), 1, figsize=(12, 3 * len(sel)), sharex=False)
        if len(sel) == 1:
            axes = [axes]
        for ax, (_, ep) in zip(axes, sel.iterrows()):
            window = df[(df["date"] >= pd.Timestamp(ep["start"]) - pd.Timedelta(days=40)) & (df["date"] <= pd.Timestamp(ep["end"]) + pd.Timedelta(days=40))]
            ax.plot(window["date"], window["spy_drawdown_from_previous_high"], label="SPY DD")
            ax.plot(window["date"], window["STEEP_EITHER_SELL_STRESS"].astype(int) * -0.02, label="Either")
            ax.plot(window["date"], window["STEEP_CREDIT_DD5_STRESS"].astype(int) * -0.04, label="Credit")
            ax.plot(window["date"], window["BACKBONE_V2_UPGRADED_risk_state"].eq("RISK").astype(int) * -0.06, label="V2")
            ax.plot(window["date"], window["BACKBONE_V2_NO_EITHER_SELL_risk_state"].eq("RISK").astype(int) * -0.08, label="V2_NO")
            ax.legend(fontsize=7, ncol=5)
            ax.set_title(f"Either-only episode {int(ep['episode_id'])}")
        fig.tight_layout()
        fig.savefig(CONFIG["figure_dir"] / "either_only_episode_case_studies.png", dpi=150)
        plt.close(fig)


def _plot_case(df: pd.DataFrame, start: str, end: str, path: Path) -> None:
    sub = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))].copy()
    if len(sub) < 5:
        return
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(sub["date"], sub["spy_price"] / sub["spy_price"].iloc[0], label="SPY")
    for s in ["BACKBONE_V1_OLD", "BACKBONE_V2_UPGRADED", "BACKBONE_V2_NO_EITHER_SELL"]:
        axes[0].plot(sub["date"], sub[f"{s}_nav"] / sub[f"{s}_nav"].iloc[0], label=s)
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].plot(sub["date"], sub["spy_drawdown_from_previous_high"], label="SPY DD")
    axes[1].legend(fontsize=8)
    for s in ["BACKBONE_V1_OLD", "BACKBONE_V2_UPGRADED", "BACKBONE_V2_NO_EITHER_SELL"]:
        axes[2].plot(sub["date"], sub[f"{s}_risk_state"].eq("RISK").astype(int), label=s)
    axes[2].legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_markdown_report(perf: pd.DataFrame, crisis: pd.DataFrame, overlap_summary: pd.DataFrame, overlap_ep: pd.DataFrame, inc: pd.DataFrame) -> None:
    def table(df: pd.DataFrame, n: int = 20) -> str:
        return "_No data._" if df.empty else df.head(n).to_markdown(index=False)

    content = f"""# Backbone Upgrade Ablation Report

## Purpose

This run upgrades the SPY/CASH stress-recovery timing backbone and tests whether `STEEP + Monthly Either SELL` still adds value after adding higher-frequency FLAT and credit stress logic.

## Strategy Definitions

- `BACKBONE_V1_OLD`: FLAT VIX + STEEP Monthly Either SELL + credit DD5 all regime.
- `BACKBONE_V2_UPGRADED`: FLAT VIX OR credit + STEEP Monthly Either SELL + STEEP credit DD5, with inverted credit disabled.
- `BACKBONE_V2_NO_EITHER_SELL`: same as V2, but remove `STEEP + Monthly Either SELL`.

Monthly Either SELL is only used as a stress trigger in `STEEP`.

## Main Performance Comparison

{table(perf)}

## Crisis Period Comparison

{table(crisis, 40)}

## Either SELL Overlap Analysis

Daily overlap summary:

{table(overlap_summary)}

Episode overlap summary:

{table(overlap_ep, 30)}

Incremental value:

{table(inc)}

## Interpretation

Use the V2 vs V2_NO_EITHER comparison to determine whether the low-frequency Monthly Either SELL still protects episodes that are not already covered by the higher-frequency signals.

## Recommendation

Choose the next hedge-allocation backbone by comparing:

- V2 performance vs V1.
- V2 vs V2_NO_EITHER crisis coverage.
- The number and quality of either-only episodes.

## Caveats

- Monthly Either is low frequency and can overlap partially with faster triggers.
- All exits use the same R3 recovery rule.
- This remains a SPY/CASH-only backbone test without hedge assets.
"""
    (CONFIG["output_dir"] / "BACKBONE_UPGRADE_ABLATION_REPORT.md").write_text(content, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = build_backbone_entry_signals(build_primitive_signals(load_panel()))

    out = panel[[
        "date",
        "spy_price",
        "spy_daily_return",
        "daily_rf",
        "macro_regime_confirmed",
        "monthly_either_state",
        "VIX_LEVEL",
        "VIX_ZSCORE_120D",
        "CREDIT_SPREAD_BAA_AAA",
        "D_CREDIT_SPREAD_20D",
        "spy_drawdown_from_previous_high",
        "SPY_MA20",
        "SPY_CROSS_ABOVE_MA20",
        "FLAT_VIX_STRESS",
        "FLAT_CREDIT_DD5_STRESS",
        "FLAT_VIX_OR_CREDIT_STRESS",
        "STEEP_EITHER_SELL_STRESS",
        "STEEP_CREDIT_DD5_STRESS",
        "INVERTED_CREDIT_DD5_SIGNAL",
        "CREDIT_DD5_ALL_REGIME_SIGNAL",
        "V1_OLD_entry_signal",
        "V1_OLD_entry_reason",
        "V2_UPGRADED_entry_signal",
        "V2_UPGRADED_entry_reason",
        "V2_NO_EITHER_entry_signal",
        "V2_NO_EITHER_entry_reason",
    ]].copy()

    # Benchmarks
    out["SPY_BUY_HOLD_weight_spy"] = 1.0
    out["SPY_BUY_HOLD_weight_cash"] = 0.0
    out["SPY_BUY_HOLD_return"] = out["spy_daily_return"]
    out["SPY_BUY_HOLD_nav"] = (1 + out["SPY_BUY_HOLD_return"]).cumprod()
    out["CASH_ONLY_weight_spy"] = 0.0
    out["CASH_ONLY_weight_cash"] = 1.0
    out["CASH_ONLY_return"] = out["daily_rf"]
    out["CASH_ONLY_nav"] = (1 + out["CASH_ONLY_return"]).cumprod()
    if "MONTHLY_EITHER_CONFIRM_return" in panel.columns:
        out["MONTHLY_EITHER_CONFIRM_return"] = panel["MONTHLY_EITHER_CONFIRM_return"]
        out["MONTHLY_EITHER_CONFIRM_nav"] = panel["MONTHLY_EITHER_CONFIRM_nav"]
        if "MONTHLY_EITHER_CONFIRM_weight_spy" in panel.columns:
            out["MONTHLY_EITHER_CONFIRM_weight_spy"] = panel["MONTHLY_EITHER_CONFIRM_weight_spy"]
            out["MONTHLY_EITHER_CONFIRM_weight_cash"] = 1 - panel["MONTHLY_EITHER_CONFIRM_weight_spy"]

    event_logs = []
    mapping = {
        "BACKBONE_V1_OLD": ("V1_OLD_entry_signal", "V1_OLD_entry_reason"),
        "BACKBONE_V2_UPGRADED": ("V2_UPGRADED_entry_signal", "V2_UPGRADED_entry_reason"),
        "BACKBONE_V2_NO_EITHER_SELL": ("V2_NO_EITHER_entry_signal", "V2_NO_EITHER_entry_reason"),
    }
    for strat, (sig, reason) in mapping.items():
        cols, ev = run_state_machine_backtest(panel, panel[sig].fillna(False), panel[reason], strat)
        out = pd.concat([out, cols], axis=1)
        event_logs.append(ev)
    event_log = pd.concat(event_logs, ignore_index=True) if event_logs else pd.DataFrame()

    episodes = extract_risk_episodes(out, BACKBONES, event_log)
    strategies = ["SPY_BUY_HOLD", "CASH_ONLY", "MONTHLY_EITHER_CONFIRM"] + BACKBONES
    perf = compute_performance_metrics(out, strategies, episodes)
    crisis = compute_crisis_performance(out, strategies)
    perf_regime = compute_performance_by_regime(out, strategies)

    overlap_daily, overlap_summary = analyze_signal_overlap(panel)
    overlap_ep = analyze_episode_overlap(out, episodes)
    v2 = perf[perf["strategy"] == "BACKBONE_V2_UPGRADED"].iloc[0]
    no = perf[perf["strategy"] == "BACKBONE_V2_NO_EITHER_SELL"].iloc[0]
    inc = compute_either_incremental_value(v2, no, overlap_ep)

    out.to_csv(CONFIG["output_dir"] / "daily_backtest_panel.csv", index=False)
    event_log.to_csv(CONFIG["output_dir"] / "risk_state_event_log.csv", index=False)
    episodes.to_csv(CONFIG["output_dir"] / "risk_episodes.csv", index=False)
    perf.to_csv(CONFIG["output_dir"] / "performance_summary.csv", index=False)
    crisis.to_csv(CONFIG["output_dir"] / "crisis_performance.csv", index=False)
    perf_regime.to_csv(CONFIG["output_dir"] / "performance_by_regime.csv", index=False)
    overlap_daily.to_csv(CONFIG["output_dir"] / "signal_daily_overlap.csv", index=False)
    overlap_summary.to_csv(CONFIG["output_dir"] / "signal_overlap_summary.csv", index=False)
    overlap_ep.to_csv(CONFIG["output_dir"] / "episode_overlap_summary.csv", index=False)
    inc.to_csv(CONFIG["output_dir"] / "either_incremental_value.csv", index=False)

    plot_backtest_results(out, perf)
    plot_overlap_results(out, overlap_daily, overlap_ep)
    _plot_case(out, *CRISIS_WINDOWS["2008_GFC"], CONFIG["figure_dir"] / "crisis_case_study_2008.png")
    _plot_case(out, *CRISIS_WINDOWS["2018Q4"], CONFIG["figure_dir"] / "crisis_case_study_2018Q4.png")
    _plot_case(out, *CRISIS_WINDOWS["COVID_2020"], CONFIG["figure_dir"] / "crisis_case_study_COVID.png")
    _plot_case(out, *CRISIS_WINDOWS["2022"], CONFIG["figure_dir"] / "crisis_case_study_2022.png")
    _plot_case(out, *CRISIS_WINDOWS["2025_PULLBACK"], CONFIG["figure_dir"] / "crisis_case_study_2025.png")
    write_markdown_report(perf, crisis, overlap_summary, overlap_ep, inc)

    def _row(name: str) -> pd.Series:
        return perf[perf["strategy"] == name].iloc[0]

    v1 = _row("BACKBONE_V1_OLD")
    v2r = _row("BACKBONE_V2_UPGRADED")
    v2n = _row("BACKBONE_V2_NO_EITHER_SELL")
    print(f"1. V1 old Ann/Sharpe/MaxDD/switches/cash: {v1['annualized_return']:.2%} / {v1['sharpe_ratio']:.2f} / {v1['max_drawdown']:.2%} / {int(v1['number_of_switches'])} / {v1['time_in_cash']:.1%}")
    print(f"2. V2 upgraded Ann/Sharpe/MaxDD/switches/cash: {v2r['annualized_return']:.2%} / {v2r['sharpe_ratio']:.2f} / {v2r['max_drawdown']:.2%} / {int(v2r['number_of_switches'])} / {v2r['time_in_cash']:.1%}")
    print(f"3. V2 no either Ann/Sharpe/MaxDD/switches/cash: {v2n['annualized_return']:.2%} / {v2n['sharpe_ratio']:.2f} / {v2n['max_drawdown']:.2%} / {int(v2n['number_of_switches'])} / {v2n['time_in_cash']:.1%}")
    print(f"4. V2 better than V1? {bool((v2r['sharpe_ratio'] >= v1['sharpe_ratio']) and (v2r['max_drawdown'] >= v1['max_drawdown']))}")
    print(f"5. Remove either hurts Sharpe / MaxDD? {v2n['sharpe_ratio'] < v2r['sharpe_ratio']} / {v2n['max_drawdown'] < v2r['max_drawdown']}")
    print(f"6. Either sell days: {int(overlap_summary.iloc[0]['either_sell_days'])}")
    print(f"7. Either vs STEEP credit overlap ratio: {overlap_summary.iloc[0]['either_overlap_ratio_with_steep_credit']:.2%}")
    print(f"8. Either vs any high-frequency overlap ratio: {overlap_summary.iloc[0]['either_overlap_ratio_with_any_high_freq']:.2%}")
    print(f"9. Either-only episodes count / avg SPY drawdown: {int(inc.iloc[0]['either_only_episode_count'])} / {inc.iloc[0]['either_only_avg_SPY_max_drawdown_during_episode']:.2%}")
    best = perf.sort_values(["sharpe_ratio", "max_drawdown"], ascending=[False, False]).iloc[0]["strategy"]
    print(f"10. Recommended next hedge allocation backbone: {best}")
    print(f"Saved outputs: {CONFIG['output_dir'].resolve()} and {CONFIG['figure_dir'].resolve()}")


if __name__ == "__main__":
    main()
