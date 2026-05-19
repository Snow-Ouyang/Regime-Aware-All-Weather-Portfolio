"""Credit DD5 trigger regime diagnostic and SPY/CASH backtest.

This script isolates the trigger:
    SPY drawdown from previous high <= -5%
    and credit spread 20D change > 0.10

It runs event-level diagnostics by macro regime and simple SPY/CASH
state-machine backtests with regime-gated variants. It intentionally excludes
VIX, Monthly Either, commodity, and hedge-asset triggers from the pure credit
tests, except for reading the current full baseline if already available.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIG = {
    "dd_threshold": -0.05,
    "credit_change_window": 20,
    "credit_change_threshold": 0.10,
    "strict_inverted_dd_threshold": -0.08,
    "strict_inverted_credit_threshold": 0.15,
    "recovery_ma_window": 20,
    "cooldown_days": 21,
    "one_way_cost_bps": 5,
    "forward_windows": [5, 10, 21, 42, 63],
    "output_dir": Path("results/credit_trigger_by_regime_diagnostic"),
    "figure_dir": Path("figures/credit_trigger_by_regime_diagnostic"),
}


PANEL_CANDIDATES = [
    Path("results/spy_cash_stress_recovery_with_credit/daily_backtest_panel.csv"),
    Path("results/spy_cash_stress_recovery_with_commodity/daily_backtest_panel.csv"),
    Path("results/spy_cash_stress_recovery_timing/daily_backtest_panel.csv"),
]


CRISIS_WINDOWS = {
    "2006_credit_trigger_tests": ["2006-01-01", "2006-12-31"],
    "2008_GFC": ["2007-10-01", "2009-06-30"],
    "2015_2016": ["2015-05-01", "2016-03-31"],
    "2018Q4": ["2018-10-01", "2019-01-31"],
    "COVID_2020": ["2020-02-01", "2020-06-30"],
    "2022": ["2021-11-01", "2023-03-31"],
    "2023": ["2023-01-01", "2023-12-31"],
    "2024_2026": ["2024-01-01", "2026-12-31"],
}


PURE_STRATEGIES = [
    "CREDIT_DD5_ALL_REGIME",
    "CREDIT_DD5_FLAT_ONLY",
    "CREDIT_DD5_STEEP_ONLY",
    "CREDIT_DD5_INVERTED_ONLY",
    "CREDIT_DD5_EX_INVERTED",
    "CREDIT_DD5_FLAT_STEEP_ONLY",
    "CREDIT_DD5_STRICT_INVERTED",
]


def ensure_dirs() -> None:
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["figure_dir"].mkdir(parents=True, exist_ok=True)


def load_panel() -> pd.DataFrame:
    for path in PANEL_CANDIDATES:
        if path.exists():
            df = pd.read_csv(path)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
            print(f"Loaded panel: {path}")
            break
    else:
        raise FileNotFoundError("No suitable daily panel found.")

    required = ["spy_price", "spy_daily_return", "daily_rf", "macro_regime_confirmed", "CREDIT_SPREAD_BAA_AAA"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "monthly_either_state" not in df.columns:
        if "monthly_either_weight_spy" in df.columns:
            df["monthly_either_state"] = np.where(df["monthly_either_weight_spy"] >= 0.5, "HOLD", "SELL")
        else:
            warnings.warn("monthly_either_state missing; filling UNKNOWN.")
            df["monthly_either_state"] = "UNKNOWN"

    if "spy_drawdown_from_previous_high" not in df.columns:
        df["previous_high"] = df["spy_price"].cummax()
        df["spy_drawdown_from_previous_high"] = df["spy_price"] / df["previous_high"] - 1

    cw = CONFIG["credit_change_window"]
    if "D_CREDIT_SPREAD_20D" not in df.columns:
        df["D_CREDIT_SPREAD_20D"] = df["CREDIT_SPREAD_BAA_AAA"] - df["CREDIT_SPREAD_BAA_AAA"].shift(cw)

    ma = CONFIG["recovery_ma_window"]
    if "SPY_MA20" not in df.columns:
        df["SPY_MA20"] = df["spy_price"].rolling(ma).mean()
    if "SPY_CROSS_ABOVE_MA20" not in df.columns:
        df["SPY_CROSS_ABOVE_MA20"] = (df["spy_price"] > df["SPY_MA20"]) & (
            df["spy_price"].shift(1) <= df["SPY_MA20"].shift(1)
        )

    df["macro_regime_confirmed"] = df["macro_regime_confirmed"].fillna("NEUTRAL").astype(str)
    df["daily_rf"] = pd.to_numeric(df["daily_rf"], errors="coerce").fillna(0.0)
    df["spy_daily_return"] = pd.to_numeric(df["spy_daily_return"], errors="coerce").fillna(0.0)
    return df


def build_credit_trigger(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["CREDIT_DD5_TRIGGER"] = (
        (out["spy_drawdown_from_previous_high"] <= CONFIG["dd_threshold"])
        & (out["D_CREDIT_SPREAD_20D"] > CONFIG["credit_change_threshold"])
    )
    out["CREDIT_DD5_STRICT_INVERTED_TRIGGER"] = np.where(
        out["macro_regime_confirmed"].eq("INVERTED"),
        (out["spy_drawdown_from_previous_high"] <= CONFIG["strict_inverted_dd_threshold"])
        & (out["D_CREDIT_SPREAD_20D"] > CONFIG["strict_inverted_credit_threshold"]),
        out["CREDIT_DD5_TRIGGER"],
    )
    out["credit_trigger_regime"] = np.where(out["CREDIT_DD5_TRIGGER"], out["macro_regime_confirmed"], "")
    return out


def extract_credit_events(df: pd.DataFrame) -> pd.DataFrame:
    events = []
    trigger = df["CREDIT_DD5_TRIGGER"].fillna(False).astype(bool)
    can_fire = True
    cooldown = 0
    prev = False
    for i, flag in enumerate(trigger):
        if cooldown > 0:
            cooldown -= 1
            if not flag:
                can_fire = True
            prev = flag
            continue
        if flag and not prev and can_fire:
            r = df.iloc[i]
            events.append(
                {
                    "event_idx": i,
                    "event_date": r["date"],
                    "macro_regime_confirmed": r["macro_regime_confirmed"],
                    "monthly_either_state": r["monthly_either_state"],
                    "spy_drawdown_at_event": r["spy_drawdown_from_previous_high"],
                    "CREDIT_SPREAD_BAA_AAA_at_event": r["CREDIT_SPREAD_BAA_AAA"],
                    "D_CREDIT_SPREAD_20D_at_event": r["D_CREDIT_SPREAD_20D"],
                    "SPY_MA20_at_event": r["SPY_MA20"],
                    "SPY_above_MA20_at_event": r["spy_price"] > r["SPY_MA20"],
                }
            )
            cooldown = CONFIG["cooldown_days"]
            can_fire = False
        if not flag:
            can_fire = True
        prev = flag
    return pd.DataFrame(events)


def _forward_path_metrics(prices: pd.Series, start_idx: int, horizon: int) -> Dict[str, float]:
    end_idx = min(start_idx + horizon, len(prices) - 1)
    path = prices.iloc[start_idx : end_idx + 1].astype(float)
    if len(path) < 2 or path.iloc[0] <= 0:
        return {"return": np.nan, "max_drawdown": np.nan, "max_runup": np.nan, "days_to_trough": np.nan}
    rel = path / path.iloc[0] - 1
    running_high = path.cummax()
    dd = path / running_high - 1
    trough_pos = int(dd.values.argmin())
    return {
        "return": rel.iloc[-1],
        "max_drawdown": dd.min(),
        "max_runup": rel.max(),
        "days_to_trough": trough_pos,
    }


def compute_forward_outcomes(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    out = events.copy()
    prices = df["spy_price"]
    for j, ev in out.iterrows():
        idx = int(ev["event_idx"])
        for h in CONFIG["forward_windows"]:
            m = _forward_path_metrics(prices, idx, h)
            out.loc[j, f"forward_spy_return_{h}d"] = m["return"]
            out.loc[j, f"forward_spy_max_drawdown_{h}d"] = m["max_drawdown"]
            out.loc[j, f"forward_spy_max_runup_{h}d"] = m["max_runup"]
            out.loc[j, f"days_to_trough_{h}d"] = m["days_to_trough"]
    out["mdd_21d_below_3"] = out["forward_spy_max_drawdown_21d"] <= -0.03
    out["mdd_21d_below_5"] = out["forward_spy_max_drawdown_21d"] <= -0.05
    out["mdd_63d_below_10"] = out["forward_spy_max_drawdown_63d"] <= -0.10
    out["false_alarm_21d"] = out["forward_spy_max_drawdown_21d"] > -0.03
    out["quick_rebound_21d"] = out["forward_spy_return_21d"] > 0.03
    out["strong_rebound_63d"] = out["forward_spy_return_63d"] > 0.08
    return out


def summarize_events_by_regime(events: pd.DataFrame, sample_start: pd.Timestamp, sample_end: pd.Timestamp) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    years = max((sample_end - sample_start).days / 365.25, 1e-9)
    rows = []
    for regime, sub in events.groupby("macro_regime_confirmed"):
        rows.append(
            {
                "macro_regime_confirmed": regime,
                "event_count": len(sub),
                "events_per_year": len(sub) / years,
                "avg_forward_return_21d": sub["forward_spy_return_21d"].mean(),
                "avg_forward_return_63d": sub["forward_spy_return_63d"].mean(),
                "avg_forward_mdd_21d": sub["forward_spy_max_drawdown_21d"].mean(),
                "avg_forward_mdd_63d": sub["forward_spy_max_drawdown_63d"].mean(),
                "median_forward_mdd_21d": sub["forward_spy_max_drawdown_21d"].median(),
                "pct_mdd_21d_below_3": sub["mdd_21d_below_3"].mean(),
                "pct_mdd_21d_below_5": sub["mdd_21d_below_5"].mean(),
                "pct_mdd_63d_below_10": sub["mdd_63d_below_10"].mean(),
                "false_alarm_rate_21d": sub["false_alarm_21d"].mean(),
                "quick_rebound_rate_21d": sub["quick_rebound_21d"].mean(),
                "strong_rebound_rate_63d": sub["strong_rebound_63d"].mean(),
                "avg_days_to_trough_21d": sub["days_to_trough_21d"].mean(),
                "avg_days_to_trough_63d": sub["days_to_trough_63d"].mean(),
                "median_spy_drawdown_at_event": sub["spy_drawdown_at_event"].median(),
                "median_credit_change20_at_event": sub["D_CREDIT_SPREAD_20D_at_event"].median(),
            }
        )
    return pd.DataFrame(rows)


def _entry_signal(df: pd.DataFrame, strategy: str) -> Tuple[pd.Series, pd.Series]:
    base = df["CREDIT_DD5_TRIGGER"].fillna(False).astype(bool)
    strict = df["CREDIT_DD5_STRICT_INVERTED_TRIGGER"].fillna(False).astype(bool)
    regime = df["macro_regime_confirmed"]
    if strategy == "CREDIT_DD5_ALL_REGIME":
        sig = base
    elif strategy == "CREDIT_DD5_FLAT_ONLY":
        sig = base & regime.eq("FLAT")
    elif strategy == "CREDIT_DD5_STEEP_ONLY":
        sig = base & regime.eq("STEEP")
    elif strategy == "CREDIT_DD5_INVERTED_ONLY":
        sig = base & regime.eq("INVERTED")
    elif strategy == "CREDIT_DD5_EX_INVERTED":
        sig = base & ~regime.eq("INVERTED")
    elif strategy == "CREDIT_DD5_FLAT_STEEP_ONLY":
        sig = base & regime.isin(["FLAT", "STEEP"])
    elif strategy == "CREDIT_DD5_STRICT_INVERTED":
        sig = strict
    else:
        raise ValueError(strategy)
    reason = pd.Series("", index=df.index, dtype=object)
    reason.loc[sig] = strategy.replace("CREDIT_DD5_", "") + "_CREDIT_DD5"
    return sig, reason


def run_credit_state_machine(df: pd.DataFrame, strategy: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    entry, reason = _entry_signal(df, strategy)
    recovery = df["SPY_CROSS_ABOVE_MA20"].fillna(False).astype(bool)
    n = len(df)
    state = "NORMAL"
    pending_state = "NORMAL"
    pending_reason = ""
    risk_state = []
    weight_spy = []
    costs = []
    events = []
    prev_weight = 1.0
    current_entry_reason = ""
    for i in range(n):
        if pending_state != state:
            old_state = state
            state = pending_state
            new_weight = 0.0 if state == "RISK" else 1.0
            turnover = abs(new_weight - prev_weight) + abs((1 - new_weight) - (1 - prev_weight))
            cost = 0.5 * turnover * CONFIG["one_way_cost_bps"] / 10000
            event_type = "ENTER_RISK" if state == "RISK" else "EXIT_RISK"
            ev_reason = pending_reason if pending_reason else ("R3_SPY_CROSS_ABOVE_MA20" if event_type == "EXIT_RISK" else "")
            if event_type == "ENTER_RISK":
                current_entry_reason = ev_reason
            events.append(
                {
                    "strategy": strategy,
                    "event_date": df.iloc[i]["date"],
                    "event_type": event_type,
                    "reason": ev_reason,
                    "macro_regime_confirmed": df.iloc[i]["macro_regime_confirmed"],
                    "monthly_either_state": df.iloc[i]["monthly_either_state"],
                    "previous_state": old_state,
                    "new_state": state,
                    "spy_drawdown_from_previous_high": df.iloc[i]["spy_drawdown_from_previous_high"],
                    "D_CREDIT_SPREAD_20D": df.iloc[i]["D_CREDIT_SPREAD_20D"],
                }
            )
            prev_weight = new_weight
            costs.append(cost)
        else:
            costs.append(0.0)
        w = 0.0 if state == "RISK" else 1.0
        risk_state.append(state)
        weight_spy.append(w)

        next_state = state
        next_reason = ""
        if state == "NORMAL" and bool(entry.iloc[i]):
            next_state = "RISK"
            next_reason = reason.iloc[i]
        elif state == "RISK" and bool(recovery.iloc[i]):
            next_state = "NORMAL"
            next_reason = "R3_SPY_CROSS_ABOVE_MA20"
        pending_state = next_state
        pending_reason = next_reason

    out = pd.DataFrame(index=df.index)
    out[f"{strategy}_risk_state"] = risk_state
    out[f"{strategy}_weight_spy"] = weight_spy
    out[f"{strategy}_weight_cash"] = 1 - out[f"{strategy}_weight_spy"]
    out[f"{strategy}_transaction_cost"] = costs
    out[f"{strategy}_return"] = (
        out[f"{strategy}_weight_spy"] * df["spy_daily_return"]
        + out[f"{strategy}_weight_cash"] * df["daily_rf"]
        - out[f"{strategy}_transaction_cost"]
    )
    out[f"{strategy}_nav"] = (1 + out[f"{strategy}_return"].fillna(0.0)).cumprod()
    return out, pd.DataFrame(events)


def _max_drawdown(ret: pd.Series) -> float:
    nav = (1 + ret.fillna(0.0)).cumprod()
    return float((nav / nav.cummax() - 1).min()) if len(nav) else np.nan


def compute_performance_metrics(panel: pd.DataFrame, strategies: List[str]) -> pd.DataFrame:
    rows = []
    years = len(panel) / 252
    for strat in strategies:
        ret_col = f"{strat}_return"
        nav_col = f"{strat}_nav"
        if ret_col not in panel.columns or nav_col not in panel.columns:
            continue
        ret = panel[ret_col].fillna(0.0)
        nav = panel[nav_col]
        excess = ret - panel["daily_rf"].fillna(0.0)
        switches = int((panel.get(f"{strat}_weight_spy", pd.Series(1, index=panel.index)).diff().abs() > 0).sum())
        risk_entries = int((panel.get(f"{strat}_risk_state", pd.Series("", index=panel.index)).eq("RISK") & ~panel.get(f"{strat}_risk_state", pd.Series("", index=panel.index)).shift(1, fill_value="NORMAL").eq("RISK")).sum())
        rows.append(
            {
                "strategy": strat,
                "annualized_return": nav.iloc[-1] ** (1 / years) - 1,
                "annualized_volatility": ret.std(ddof=0) * math.sqrt(252),
                "sharpe_ratio": excess.mean() / excess.std(ddof=0) * math.sqrt(252) if excess.std(ddof=0) > 0 else np.nan,
                "max_drawdown": (nav / nav.cummax() - 1).min(),
                "calmar_ratio": (nav.iloc[-1] ** (1 / years) - 1) / abs((nav / nav.cummax() - 1).min()),
                "final_nav": nav.iloc[-1],
                "number_of_switches": switches,
                "number_of_risk_entries": risk_entries,
                "avg_risk_episode_duration": np.nan,
                "time_in_cash": panel.get(f"{strat}_weight_cash", pd.Series(0, index=panel.index)).mean(),
                "total_turnover": switches * 2,
                "transaction_cost_drag": panel.get(f"{strat}_transaction_cost", pd.Series(0, index=panel.index)).sum(),
            }
        )
    return pd.DataFrame(rows)


def extract_risk_episodes(panel: pd.DataFrame, strategies: List[str], event_log: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strat in strategies:
        state_col = f"{strat}_risk_state"
        if state_col not in panel.columns:
            continue
        is_risk = panel[state_col].eq("RISK")
        starts = panel.index[is_risk & ~is_risk.shift(1, fill_value=False)]
        entries = event_log[(event_log["strategy"] == strat) & (event_log["event_type"] == "ENTER_RISK")]
        for eid, start in enumerate(starts, 1):
            end = start
            while end + 1 < len(panel) and is_risk.iloc[end + 1]:
                end += 1
            sub = panel.iloc[start : end + 1]
            r0 = sub.iloc[0]
            match = entries[entries["event_date"].eq(r0["date"])]
            entry_reason = match["reason"].iloc[0] if not match.empty else "CREDIT_DD5"
            spy_nav = (1 + sub["spy_daily_return"].fillna(0.0)).cumprod()
            cash_nav = (1 + sub["daily_rf"].fillna(0.0)).cumprod()
            strat_nav = (1 + sub[f"{strat}_return"].fillna(0.0)).cumprod()
            rows.append(
                {
                    "strategy": strat,
                    "episode_id": eid,
                    "risk_start_date": r0["date"],
                    "risk_end_date": sub.iloc[-1]["date"],
                    "duration_days": len(sub),
                    "entry_regime": r0["macro_regime_confirmed"],
                    "entry_reason": entry_reason,
                    "SPY_drawdown_at_entry": r0["spy_drawdown_from_previous_high"],
                    "D_CREDIT_SPREAD_20D_at_entry": r0["D_CREDIT_SPREAD_20D"],
                    "SPY_return_during_risk": spy_nav.iloc[-1] - 1,
                    "CASH_return_during_risk": cash_nav.iloc[-1] - 1,
                    "strategy_return_during_risk": strat_nav.iloc[-1] - 1,
                    "SPY_max_drawdown_during_risk": (spy_nav / spy_nav.cummax() - 1).min(),
                    "SPY_max_runup_during_risk": spy_nav.max() - 1,
                    "exited_by_R3_date": sub.iloc[-1]["date"],
                }
            )
    return pd.DataFrame(rows)


def compute_crisis_performance(panel: pd.DataFrame, strategies: List[str]) -> pd.DataFrame:
    rows = []
    max_date = panel["date"].max()
    for name, (start, end) in CRISIS_WINDOWS.items():
        sub = panel[(panel["date"] >= pd.Timestamp(start)) & (panel["date"] <= min(pd.Timestamp(end), max_date))]
        if len(sub) < 2:
            continue
        years = len(sub) / 252
        for strat in strategies:
            ret_col = f"{strat}_return"
            if ret_col not in sub.columns:
                continue
            ret = sub[ret_col].fillna(0.0)
            nav = (1 + ret).cumprod()
            excess = ret - sub["daily_rf"].fillna(0.0)
            rows.append(
                {
                    "period": name,
                    "strategy": strat,
                    "cumulative_return": nav.iloc[-1] - 1,
                    "annualized_return": nav.iloc[-1] ** (1 / years) - 1,
                    "max_drawdown": (nav / nav.cummax() - 1).min(),
                    "volatility": ret.std(ddof=0) * math.sqrt(252),
                    "sharpe": excess.mean() / excess.std(ddof=0) * math.sqrt(252) if excess.std(ddof=0) > 0 else np.nan,
                    "time_in_cash": sub.get(f"{strat}_weight_cash", pd.Series(0, index=sub.index)).mean(),
                    "number_of_switches": int((sub.get(f"{strat}_weight_spy", pd.Series(1, index=sub.index)).diff().abs() > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def build_regime_gating_decision_table(summary: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if summary.empty:
        return pd.DataFrame()
    ep = episodes[episodes["strategy"] == "CREDIT_DD5_ALL_REGIME"] if not episodes.empty else pd.DataFrame()
    for _, r in summary.iterrows():
        regime = r["macro_regime_confirmed"]
        eps = ep[ep["entry_regime"].eq(regime)]
        if r["event_count"] < 8:
            action = "INSUFFICIENT_SAMPLE"
        elif r["false_alarm_rate_21d"] > 0.50 and r["quick_rebound_rate_21d"] > 0.40:
            action = "DISABLE_OR_PARTIAL_DERISK"
        elif r["pct_mdd_21d_below_5"] >= 0.40 and r["avg_forward_mdd_21d"] <= -0.05:
            action = "ENABLE_FULL_RISK"
        else:
            action = "KEEP_DIAGNOSTIC_ONLY"
        rows.append(
            {
                "regime": regime,
                "event_count": r["event_count"],
                "false_alarm_rate_21d": r["false_alarm_rate_21d"],
                "quick_rebound_rate_21d": r["quick_rebound_rate_21d"],
                "pct_mdd_21d_below_5": r["pct_mdd_21d_below_5"],
                "avg_forward_mdd_21d": r["avg_forward_mdd_21d"],
                "avg_forward_return_21d": r["avg_forward_return_21d"],
                "risk_episode_count": len(eps),
                "avg_SPY_return_during_risk": eps["SPY_return_during_risk"].mean() if not eps.empty else np.nan,
                "avg_SPY_max_drawdown_during_risk": eps["SPY_max_drawdown_during_risk"].mean() if not eps.empty else np.nan,
                "recommended_action": action,
            }
        )
    return pd.DataFrame(rows)


def plot_event_diagnostics(events: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig_dir = CONFIG["figure_dir"]
    if not events.empty:
        for col, name, ylabel in [
            ("forward_spy_max_drawdown_21d", "credit_event_forward_mdd_by_regime.png", "21D forward max drawdown"),
            ("forward_spy_return_21d", "credit_event_forward_return_by_regime.png", "21D forward return"),
        ]:
            regimes = list(events["macro_regime_confirmed"].dropna().unique())
            data = [events.loc[events["macro_regime_confirmed"].eq(r), col].dropna().values for r in regimes]
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.boxplot(data, tick_labels=regimes, showfliers=False)
            for i, r in enumerate(regimes, 1):
                y = events.loc[events["macro_regime_confirmed"].eq(r), col].dropna()
                ax.scatter(np.random.normal(i, 0.04, size=len(y)), y, s=18, alpha=0.7)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylabel(ylabel)
            fig.tight_layout()
            fig.savefig(fig_dir / name, dpi=150)
            plt.close(fig)
    if not summary.empty:
        for col, name in [
            ("false_alarm_rate_21d", "false_alarm_by_regime_bar.png"),
            ("quick_rebound_rate_21d", "quick_rebound_by_regime_bar.png"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 4))
            s = summary.sort_values(col)
            ax.bar(s["macro_regime_confirmed"], s[col])
            ax.set_title(col)
            ax.set_ylim(0, 1)
            fig.tight_layout()
            fig.savefig(fig_dir / name, dpi=150)
            plt.close(fig)


def _drawdown(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1


def plot_strategy_results(panel: pd.DataFrame, strategies: List[str]) -> None:
    fig_dir = CONFIG["figure_dir"]
    selected = [
        "SPY_BUY_HOLD",
        "CREDIT_DD5_ALL_REGIME",
        "CREDIT_DD5_EX_INVERTED",
        "CREDIT_DD5_FLAT_STEEP_ONLY",
        "CREDIT_DD5_STRICT_INVERTED",
        "STRESS_RECOVERY_R3_CREDIT_DD5_FULL",
    ]
    selected = [s for s in selected if f"{s}_nav" in panel.columns]
    fig, ax = plt.subplots(figsize=(12, 6))
    for s in selected:
        ax.plot(panel["date"], panel[f"{s}_nav"], label=s)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("Credit trigger strategy equity curves")
    fig.tight_layout()
    fig.savefig(fig_dir / "strategy_equity_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for s in selected:
        ax.plot(panel["date"], _drawdown(panel[f"{s}_nav"]), label=s)
    ax.legend(fontsize=8)
    ax.set_title("Drawdown comparison")
    fig.tight_layout()
    fig.savefig(fig_dir / "drawdown_comparison.png", dpi=150)
    plt.close(fig)

    strip_strats = [s for s in PURE_STRATEGIES if f"{s}_risk_state" in panel.columns]
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1, 0.5, 1]})
    axes[0].plot(panel["date"], panel["spy_drawdown_from_previous_high"], label="SPY drawdown")
    axes[0].axhline(-0.05, color="red", linestyle="--", linewidth=0.8)
    axes[0].legend(fontsize=8)
    axes[1].plot(panel["date"], panel["D_CREDIT_SPREAD_20D"], label="credit 20D change")
    axes[1].axhline(0.10, color="red", linestyle="--", linewidth=0.8)
    axes[1].legend(fontsize=8)
    regimes = pd.Categorical(panel["macro_regime_confirmed"])
    axes[2].imshow([regimes.codes], aspect="auto", extent=[panel["date"].iloc[0], panel["date"].iloc[-1], 0, 1], cmap="tab20")
    axes[2].set_yticks([])
    axes[2].set_title("macro regime", loc="left", fontsize=9)
    offset = 0
    for s in strip_strats:
        risk = panel[f"{s}_risk_state"].eq("RISK").astype(int)
        axes[3].fill_between(panel["date"], offset, offset + risk, step="post", alpha=0.45, label=s)
        offset += 1.15
    axes[3].legend(fontsize=7, ncol=2)
    axes[3].set_yticks([])
    fig.tight_layout()
    fig.savefig(fig_dir / "risk_state_timeline_by_strategy.png", dpi=150)
    plt.close(fig)


def plot_case_studies(panel: pd.DataFrame) -> None:
    cases = {
        "case_study_inverted_credit_false_positive.png": ["2006-06-01", "2006-08-31"],
        "case_study_2018Q4_credit.png": ["2018-10-01", "2019-01-31"],
        "case_study_2022_credit.png": ["2021-11-01", "2023-03-31"],
    }
    for filename, (start, end) in cases.items():
        sub = panel[(panel["date"] >= pd.Timestamp(start)) & (panel["date"] <= pd.Timestamp(end))].copy()
        if len(sub) < 5:
            continue
        fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(sub["date"], sub["spy_price"] / sub["spy_price"].iloc[0], label="SPY")
        axes[0].legend(fontsize=8)
        axes[1].plot(sub["date"], sub["spy_drawdown_from_previous_high"], label="SPY DD")
        axes[1].axhline(-0.05, color="red", linestyle="--", linewidth=0.8)
        axes[1].legend(fontsize=8)
        axes[2].plot(sub["date"], sub["D_CREDIT_SPREAD_20D"], label="credit chg20")
        axes[2].axhline(0.10, color="red", linestyle="--", linewidth=0.8)
        axes[2].legend(fontsize=8)
        for s in ["CREDIT_DD5_ALL_REGIME", "CREDIT_DD5_EX_INVERTED", "CREDIT_DD5_STRICT_INVERTED"]:
            if f"{s}_risk_state" in sub.columns:
                axes[3].plot(sub["date"], sub[f"{s}_risk_state"].eq("RISK").astype(int), label=s)
        axes[3].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(CONFIG["figure_dir"] / filename, dpi=150)
        plt.close(fig)


def write_markdown_report(
    events: pd.DataFrame,
    summary: pd.DataFrame,
    perf: pd.DataFrame,
    crisis: pd.DataFrame,
    decision: pd.DataFrame,
) -> None:
    def table(df: pd.DataFrame, n: int = 20) -> str:
        return "_No data._" if df.empty else df.head(n).to_markdown(index=False)

    content = f"""# Credit Trigger By Regime Diagnostic

## Purpose

This analysis isolates CREDIT_DD5_TRIGGER and tests whether it should be enabled in each macro regime. It intentionally excludes VIX stress, Monthly Either SELL, commodity triggers, and hedge assets from the pure credit-trigger tests.

## Trigger Definition

`CREDIT_DD5_TRIGGER = SPY drawdown from previous high <= -5% and credit spread 20D change > 0.10`.

Event extraction uses a {CONFIG['cooldown_days']}-trading-day cooldown. Backtests use R3 recovery: SPY crosses above MA20.

## Event-Level Findings

By-regime event summary:

{table(summary)}

Decision table:

{table(decision)}

## State-Machine Backtest Findings

Performance summary:

{table(perf)}

## Inverted Regime Diagnosis

The key question is whether INVERTED credit-trigger events are valid stress entries or late signals after a drawdown. Use `credit_dd5_event_table.csv`, `credit_dd5_event_summary_by_regime.csv`, and the inverted case-study plot to inspect whether INVERTED events show high false alarm or quick rebound rates.

## Crisis Period Analysis

{table(crisis, 40)}

## Recommended Gating Rule

Use the `regime_gating_decision_table.csv` rules as an initial screen:

- `ENABLE_FULL_RISK` means the trigger shows enough follow-through drawdown evidence.
- `DISABLE_OR_PARTIAL_DERISK` means false alarms and quick rebounds are high.
- `INSUFFICIENT_SAMPLE` means do not infer a full allocation rule yet.
- `KEEP_DIAGNOSTIC_ONLY` means evidence is mixed.

## Implication for Next Hedge Backtest

If INVERTED shows quick rebound / false-positive behavior, the next timing backbone should test `EX_INVERTED`, `FLAT_STEEP_ONLY`, or a partial-inverted version instead of the all-regime credit trigger.

## Caveats

- Event counts can be small by regime.
- Credit spread data may be weekly and forward-filled in the upstream panel.
- Macro regime labels may lag actual market conditions.
- This remains a simplified SPY/CASH diagnostic.
"""
    (CONFIG["output_dir"] / "CREDIT_TRIGGER_BY_REGIME_DIAGNOSTIC.md").write_text(content, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = build_credit_trigger(load_panel())

    events = compute_forward_outcomes(panel, extract_credit_events(panel))
    events.to_csv(CONFIG["output_dir"] / "credit_dd5_event_table.csv", index=False)
    event_summary = summarize_events_by_regime(events, panel["date"].min(), panel["date"].max())
    event_summary.to_csv(CONFIG["output_dir"] / "credit_dd5_event_summary_by_regime.csv", index=False)

    out = panel[
        [
            "date",
            "spy_price",
            "spy_daily_return",
            "daily_rf",
            "macro_regime_confirmed",
            "monthly_either_state",
            "CREDIT_SPREAD_BAA_AAA",
            "D_CREDIT_SPREAD_20D",
            "spy_drawdown_from_previous_high",
            "SPY_MA20",
            "SPY_CROSS_ABOVE_MA20",
            "CREDIT_DD5_TRIGGER",
            "CREDIT_DD5_STRICT_INVERTED_TRIGGER",
            "credit_trigger_regime",
        ]
    ].copy()

    out["SPY_BUY_HOLD_weight_spy"] = 1.0
    out["SPY_BUY_HOLD_weight_cash"] = 0.0
    out["SPY_BUY_HOLD_return"] = out["spy_daily_return"]
    out["SPY_BUY_HOLD_nav"] = (1 + out["SPY_BUY_HOLD_return"].fillna(0.0)).cumprod()

    event_logs = []
    for strat in PURE_STRATEGIES:
        cols, log = run_credit_state_machine(panel, strat)
        out = pd.concat([out, cols], axis=1)
        event_logs.append(log)
    event_log = pd.concat(event_logs, ignore_index=True) if event_logs else pd.DataFrame()

    if "STRESS_RECOVERY_R3_CREDIT_DD5_nav" in panel.columns:
        out["STRESS_RECOVERY_R3_CREDIT_DD5_FULL_nav"] = panel["STRESS_RECOVERY_R3_CREDIT_DD5_nav"]
        out["STRESS_RECOVERY_R3_CREDIT_DD5_FULL_return"] = panel.get(
            "STRESS_RECOVERY_R3_CREDIT_DD5_return", out["STRESS_RECOVERY_R3_CREDIT_DD5_FULL_nav"].pct_change().fillna(0.0)
        )
        if "CREDIT_DD5_R3_weight_spy" in panel.columns:
            out["STRESS_RECOVERY_R3_CREDIT_DD5_FULL_weight_spy"] = panel["CREDIT_DD5_R3_weight_spy"]
            out["STRESS_RECOVERY_R3_CREDIT_DD5_FULL_weight_cash"] = panel.get(
                "CREDIT_DD5_R3_weight_cash", 1 - panel["CREDIT_DD5_R3_weight_spy"]
            )
        if "CREDIT_DD5_R3_risk_state" in panel.columns:
            out["STRESS_RECOVERY_R3_CREDIT_DD5_FULL_risk_state"] = panel["CREDIT_DD5_R3_risk_state"]
        if "transaction_cost_CREDIT_DD5_R3" in panel.columns:
            out["STRESS_RECOVERY_R3_CREDIT_DD5_FULL_transaction_cost"] = panel["transaction_cost_CREDIT_DD5_R3"]

    strategies = ["SPY_BUY_HOLD"] + PURE_STRATEGIES
    if "STRESS_RECOVERY_R3_CREDIT_DD5_FULL_nav" in out.columns:
        strategies.append("STRESS_RECOVERY_R3_CREDIT_DD5_FULL")

    episodes = extract_risk_episodes(out.join(panel[["macro_regime_confirmed"]], rsuffix="_p"), PURE_STRATEGIES, event_log)
    # out already contains macro_regime_confirmed; join above only protects older local calls.
    if "macro_regime_confirmed_p" in episodes.columns:
        episodes = episodes.drop(columns=["macro_regime_confirmed_p"])

    perf = compute_performance_metrics(out, strategies)
    if not episodes.empty:
        avg_duration = episodes.groupby("strategy")["duration_days"].mean().to_dict()
        perf["avg_risk_episode_duration"] = perf["strategy"].map(avg_duration).fillna(perf["avg_risk_episode_duration"])
    crisis = compute_crisis_performance(out, strategies)
    decision = build_regime_gating_decision_table(event_summary, episodes)

    out.to_csv(CONFIG["output_dir"] / "daily_backtest_panel.csv", index=False)
    perf.to_csv(CONFIG["output_dir"] / "performance_summary.csv", index=False)
    episodes.to_csv(CONFIG["output_dir"] / "risk_episodes.csv", index=False)
    event_log.to_csv(CONFIG["output_dir"] / "risk_state_event_log.csv", index=False)
    crisis.to_csv(CONFIG["output_dir"] / "crisis_performance.csv", index=False)
    decision.to_csv(CONFIG["output_dir"] / "regime_gating_decision_table.csv", index=False)

    plot_event_diagnostics(events, event_summary)
    plot_strategy_results(out, strategies)
    plot_case_studies(out)
    write_markdown_report(events, event_summary, perf, crisis, decision)

    print(f"1. CREDIT_DD5_TRIGGER events: {len(events)}")
    print("2. Events by regime:", event_summary.set_index("macro_regime_confirmed")["event_count"].to_dict() if not event_summary.empty else {})
    print("3. False alarm by regime:", event_summary.set_index("macro_regime_confirmed")["false_alarm_rate_21d"].round(3).to_dict() if not event_summary.empty else {})
    print("4. Quick rebound by regime:", event_summary.set_index("macro_regime_confirmed")["quick_rebound_rate_21d"].round(3).to_dict() if not event_summary.empty else {})
    print("5. P(21D MDD < -5%) by regime:", event_summary.set_index("macro_regime_confirmed")["pct_mdd_21d_below_5"].round(3).to_dict() if not event_summary.empty else {})
    for s in ["CREDIT_DD5_ALL_REGIME", "CREDIT_DD5_EX_INVERTED", "CREDIT_DD5_STRICT_INVERTED", "CREDIT_DD5_INVERTED_ONLY"]:
        row = perf[perf["strategy"].eq(s)]
        if not row.empty:
            r = row.iloc[0]
            print(f"{s}: Ann {r['annualized_return']:.2%}, Sharpe {r['sharpe_ratio']:.2f}, MaxDD {r['max_drawdown']:.2%}, cash {r['time_in_cash']:.1%}")
    inv = decision[decision["regime"].eq("INVERTED")]
    print("9. INVERTED recommendation:", inv["recommended_action"].iloc[0] if not inv.empty else "n/a")
    best = perf[perf["strategy"].isin(PURE_STRATEGIES)].sort_values("sharpe_ratio", ascending=False).head(1)
    print("10. Recommended next timing backbone:", best["strategy"].iloc[0] if not best.empty else "n/a")
    print(f"Saved outputs: {CONFIG['output_dir'].resolve()} and {CONFIG['figure_dir'].resolve()}")


if __name__ == "__main__":
    main()
