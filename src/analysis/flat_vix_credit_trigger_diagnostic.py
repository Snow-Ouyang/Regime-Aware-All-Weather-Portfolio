"""FLAT regime VIX + credit trigger combination diagnostic."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIG = {
    "vix_z_threshold": 3.0,
    "vix_warning_threshold": 2.5,
    "dd_threshold": -0.05,
    "credit_change_threshold": 0.10,
    "credit_weak_threshold": 0.05,
    "confirmation_windows": [10, 21],
    "recovery_ma_window": 20,
    "cooldown_days": 21,
    "one_way_cost_bps": 5,
    "output_dir": Path("results/flat_vix_credit_trigger_diagnostic"),
    "figure_dir": Path("figures/flat_vix_credit_trigger_diagnostic"),
}


PANEL_CANDIDATES = [
    Path("results/credit_trigger_by_regime_diagnostic/daily_backtest_panel.csv"),
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

FLAT_TRIGGER_NAMES = [
    "FLAT_VIX_ONLY",
    "FLAT_CREDIT_ONLY",
    "FLAT_VIX_OR_CREDIT",
    "FLAT_VIX_AND_CREDIT_SAME_DAY",
    "FLAT_VIX_AND_CREDIT_10D",
    "FLAT_VIX_AND_CREDIT_21D",
    "FLAT_VIX_WITH_CREDIT_STRICT_CONFIRM",
    "FLAT_VIX_WITH_CREDIT_WEAK_CONFIRM",
    "FLAT_CREDIT_WITH_VIX_CONFIRM_21D",
    "FLAT_CREDIT_WITH_VIX_WARNING_CONFIRM_21D",
]

FLAT_ONLY_STRATEGIES = [
    "FLAT_VIX_ONLY_R3",
    "FLAT_CREDIT_ONLY_R3",
    "FLAT_VIX_OR_CREDIT_R3",
    "FLAT_VIX_AND_CREDIT_10D_R3",
    "FLAT_VIX_AND_CREDIT_21D_R3",
    "FLAT_VIX_WITH_CREDIT_WEAK_CONFIRM_R3",
    "FLAT_CREDIT_WITH_VIX_WARNING_CONFIRM_R3",
]

FLAT_ONLY_SIGNAL_MAP = {
    "FLAT_VIX_ONLY_R3": "FLAT_VIX_ONLY",
    "FLAT_CREDIT_ONLY_R3": "FLAT_CREDIT_ONLY",
    "FLAT_VIX_OR_CREDIT_R3": "FLAT_VIX_OR_CREDIT",
    "FLAT_VIX_AND_CREDIT_10D_R3": "FLAT_VIX_AND_CREDIT_10D",
    "FLAT_VIX_AND_CREDIT_21D_R3": "FLAT_VIX_AND_CREDIT_21D",
    "FLAT_VIX_WITH_CREDIT_WEAK_CONFIRM_R3": "FLAT_VIX_WITH_CREDIT_WEAK_CONFIRM",
    "FLAT_CREDIT_WITH_VIX_WARNING_CONFIRM_R3": "FLAT_CREDIT_WITH_VIX_WARNING_CONFIRM_21D",
}

FULL_STRATEGIES = [
    "FULL_FLAT_VIX_ONLY",
    "FULL_FLAT_CREDIT_ONLY",
    "FULL_FLAT_VIX_OR_CREDIT",
    "FULL_FLAT_VIX_AND_CREDIT_10D",
    "FULL_FLAT_VIX_AND_CREDIT_21D",
    "FULL_FLAT_VIX_WITH_CREDIT_WEAK_CONFIRM",
    "FULL_FLAT_CREDIT_WITH_VIX_WARNING_CONFIRM",
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
    dfs = []
    for path in PANEL_CANDIDATES:
        if path.exists():
            dfs.append((path, _read_csv(path)))
    if not dfs:
        raise FileNotFoundError("No panel found for FLAT VIX/credit diagnostic.")
    panel = dfs[0][1].copy()
    print(f"Loaded primary panel: {dfs[0][0]}")
    needed = [
        "VIX_LEVEL",
        "VIX_ZSCORE_120D",
        "CREDIT_SPREAD_BAA_AAA",
        "D_CREDIT_SPREAD_20D",
        "spy_drawdown_from_previous_high",
        "SPY_MA20",
        "SPY_CROSS_ABOVE_MA20",
        "monthly_either_state",
        "MONTHLY_EITHER_CONFIRM_nav",
        "STRESS_RECOVERY_R3_CREDIT_DD5_nav",
        "STRESS_RECOVERY_R3_CREDIT_DD5_return",
        "CREDIT_DD5_R3_weight_spy",
        "CREDIT_DD5_R3_weight_cash",
        "CREDIT_DD5_R3_risk_state",
    ]
    for _, df in dfs[1:]:
        panel = _merge_missing(panel, df, needed)

    if "spy_drawdown_from_previous_high" not in panel.columns:
        panel["spy_drawdown_from_previous_high"] = panel["spy_price"] / panel["spy_price"].cummax() - 1
    if "VIX_ZSCORE_120D" not in panel.columns:
        if "VIX_LEVEL" not in panel.columns:
            raise ValueError("VIX_LEVEL missing; cannot rebuild VIX z-score.")
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
    panel["daily_rf"] = pd.to_numeric(panel["daily_rf"], errors="coerce").fillna(0.0)
    panel["spy_daily_return"] = pd.to_numeric(panel["spy_daily_return"], errors="coerce").fillna(0.0)
    panel["macro_regime_confirmed"] = panel["macro_regime_confirmed"].fillna("NEUTRAL").astype(str)
    return panel


def _days_since(flag: pd.Series) -> pd.Series:
    last = -1
    out = []
    for i, f in enumerate(flag.fillna(False).astype(bool)):
        if f:
            last = i
            out.append(0)
        else:
            out.append(np.nan if last < 0 else i - last)
    return pd.Series(out, index=flag.index, dtype=float)


def build_flat_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["FLAT_MASK"] = out["macro_regime_confirmed"].eq("FLAT")
    out["FLAT_VIX_STRESS"] = out["FLAT_MASK"] & (out["VIX_ZSCORE_120D"] >= CONFIG["vix_z_threshold"])
    out["FLAT_CREDIT_STRESS"] = out["FLAT_MASK"] & (
        (out["spy_drawdown_from_previous_high"] <= CONFIG["dd_threshold"])
        & (out["D_CREDIT_SPREAD_20D"] > CONFIG["credit_change_threshold"])
    )
    out["FLAT_VIX_WARNING"] = out["FLAT_MASK"] & (out["VIX_ZSCORE_120D"] >= CONFIG["vix_warning_threshold"])
    out["FLAT_CREDIT_WEAK"] = out["FLAT_MASK"] & (
        (out["spy_drawdown_from_previous_high"] <= CONFIG["dd_threshold"])
        & (out["D_CREDIT_SPREAD_20D"] > CONFIG["credit_weak_threshold"])
    )

    out["flat_vix_seen_10d"] = out["FLAT_VIX_STRESS"].rolling(10, min_periods=1).max().astype(bool)
    out["flat_credit_seen_10d"] = out["FLAT_CREDIT_STRESS"].rolling(10, min_periods=1).max().astype(bool)
    out["flat_vix_seen_21d"] = out["FLAT_VIX_STRESS"].rolling(21, min_periods=1).max().astype(bool)
    out["flat_credit_seen_21d"] = out["FLAT_CREDIT_STRESS"].rolling(21, min_periods=1).max().astype(bool)
    out["flat_credit_weak_seen_21d"] = out["FLAT_CREDIT_WEAK"].rolling(21, min_periods=1).max().astype(bool)
    out["flat_vix_warning_seen_21d"] = out["FLAT_VIX_WARNING"].rolling(21, min_periods=1).max().astype(bool)

    out["FLAT_VIX_ONLY"] = out["FLAT_VIX_STRESS"]
    out["FLAT_CREDIT_ONLY"] = out["FLAT_CREDIT_STRESS"]
    out["FLAT_VIX_OR_CREDIT"] = out["FLAT_VIX_STRESS"] | out["FLAT_CREDIT_STRESS"]
    out["FLAT_VIX_AND_CREDIT_SAME_DAY"] = out["FLAT_VIX_STRESS"] & out["FLAT_CREDIT_STRESS"]
    out["FLAT_VIX_AND_CREDIT_10D"] = out["FLAT_MASK"] & out["flat_vix_seen_10d"] & out["flat_credit_seen_10d"]
    out["FLAT_VIX_AND_CREDIT_21D"] = out["FLAT_MASK"] & out["flat_vix_seen_21d"] & out["flat_credit_seen_21d"]
    out["FLAT_VIX_WITH_CREDIT_STRICT_CONFIRM"] = out["FLAT_VIX_STRESS"] & out["flat_credit_seen_21d"]
    out["FLAT_VIX_WITH_CREDIT_WEAK_CONFIRM"] = out["FLAT_VIX_STRESS"] & out["flat_credit_weak_seen_21d"]
    out["FLAT_CREDIT_WITH_VIX_CONFIRM_21D"] = out["FLAT_CREDIT_STRESS"] & out["flat_vix_seen_21d"]
    out["FLAT_CREDIT_WITH_VIX_WARNING_CONFIRM_21D"] = out["FLAT_CREDIT_STRESS"] & out["flat_vix_warning_seen_21d"]

    intensity = np.where(
        out["FLAT_VIX_AND_CREDIT_21D"],
        "BOTH",
        np.where(out["FLAT_VIX_STRESS"], "VIX_ONLY", np.where(out["FLAT_CREDIT_STRESS"], "CREDIT_ONLY", "NONE")),
    )
    out["FLAT_STRESS_INTENSITY"] = intensity
    out["days_since_last_vix_stress"] = _days_since(out["FLAT_VIX_STRESS"])
    out["days_since_last_credit_stress"] = _days_since(out["FLAT_CREDIT_STRESS"])
    return out


def _forward_metrics(prices: pd.Series, idx: int, horizon: int) -> Dict[str, float]:
    end = min(idx + horizon, len(prices) - 1)
    path = prices.iloc[idx : end + 1].astype(float)
    if len(path) < 2:
        return {"return": np.nan, "mdd": np.nan, "runup": np.nan, "days_to_trough": np.nan}
    rel = path / path.iloc[0] - 1
    dd = path / path.cummax() - 1
    return {
        "return": rel.iloc[-1],
        "mdd": dd.min(),
        "runup": rel.max(),
        "days_to_trough": int(dd.values.argmin()),
    }


def extract_flat_events(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in FLAT_TRIGGER_NAMES:
        flag = df[name].fillna(False).astype(bool)
        prev = False
        cooldown = 0
        can_fire = True
        for i, f in enumerate(flag):
            if cooldown > 0:
                cooldown -= 1
                if not f:
                    can_fire = True
                prev = f
                continue
            if f and not prev and can_fire:
                r = df.iloc[i]
                row = {
                    "trigger_name": name,
                    "event_idx": i,
                    "event_date": r["date"],
                    "macro_regime_confirmed": r["macro_regime_confirmed"],
                    "spy_price": r["spy_price"],
                    "spy_drawdown_from_previous_high": r["spy_drawdown_from_previous_high"],
                    "VIX_LEVEL": r.get("VIX_LEVEL", np.nan),
                    "VIX_ZSCORE_120D": r["VIX_ZSCORE_120D"],
                    "CREDIT_SPREAD_BAA_AAA": r["CREDIT_SPREAD_BAA_AAA"],
                    "D_CREDIT_SPREAD_20D": r["D_CREDIT_SPREAD_20D"],
                    "monthly_either_state": r["monthly_either_state"],
                    "SPY_MA20": r["SPY_MA20"],
                    "SPY_above_MA20": r["spy_price"] > r["SPY_MA20"],
                    "FLAT_VIX_STRESS": r["FLAT_VIX_STRESS"],
                    "FLAT_CREDIT_STRESS": r["FLAT_CREDIT_STRESS"],
                    "FLAT_VIX_WARNING": r["FLAT_VIX_WARNING"],
                    "FLAT_CREDIT_WEAK": r["FLAT_CREDIT_WEAK"],
                    "FLAT_STRESS_INTENSITY": r["FLAT_STRESS_INTENSITY"],
                    "days_since_last_vix_stress": r["days_since_last_vix_stress"],
                    "days_since_last_credit_stress": r["days_since_last_credit_stress"],
                }
                for h in [5, 10, 21, 42, 63]:
                    m = _forward_metrics(df["spy_price"], i, h)
                    row[f"forward_spy_return_{h}d"] = m["return"]
                    row[f"forward_spy_max_drawdown_{h}d"] = m["mdd"]
                    row[f"forward_spy_max_runup_{h}d"] = m["runup"]
                    row[f"days_to_trough_{h}d"] = m["days_to_trough"]
                row["mdd_21d_below_3"] = row["forward_spy_max_drawdown_21d"] <= -0.03
                row["mdd_21d_below_5"] = row["forward_spy_max_drawdown_21d"] <= -0.05
                row["mdd_63d_below_10"] = row["forward_spy_max_drawdown_63d"] <= -0.10
                row["false_alarm_21d"] = row["forward_spy_max_drawdown_21d"] > -0.03
                row["quick_rebound_21d"] = row["forward_spy_return_21d"] > 0.03
                row["strong_rebound_63d"] = row["forward_spy_return_63d"] > 0.08
                rows.append(row)
                cooldown = CONFIG["cooldown_days"]
                can_fire = False
            if not f:
                can_fire = True
            prev = f
    return pd.DataFrame(rows)


def summarize_flat_events(events: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for name, sub in events.groupby("trigger_name"):
        years = max((sub["event_date"].max() - sub["event_date"].min()).days / 365.25, 1e-9) if len(sub) > 1 else 1.0
        rows.append(
            {
                "trigger_name": name,
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
                "median_spy_drawdown_at_event": sub["spy_drawdown_from_previous_high"].median(),
                "median_vix_z_at_event": sub["VIX_ZSCORE_120D"].median(),
                "median_credit_chg20_at_event": sub["D_CREDIT_SPREAD_20D"].median(),
            }
        )
    intensity = (
        events[events["FLAT_STRESS_INTENSITY"].isin(["VIX_ONLY", "CREDIT_ONLY", "BOTH"])]
        .groupby("FLAT_STRESS_INTENSITY")
        .agg(
            event_count=("trigger_name", "count"),
            avg_forward_return_21d=("forward_spy_return_21d", "mean"),
            avg_forward_mdd_21d=("forward_spy_max_drawdown_21d", "mean"),
            pct_mdd_21d_below_5=("mdd_21d_below_5", "mean"),
            false_alarm_rate_21d=("false_alarm_21d", "mean"),
        )
        .reset_index()
    )
    return pd.DataFrame(rows), intensity


def _weight_cols(name: str) -> Tuple[str, str, str, str]:
    return f"{name}_risk_state", f"{name}_weight_spy", f"{name}_weight_cash", f"{name}_transaction_cost"


def run_state_machine_backtest(df: pd.DataFrame, entry_signal: pd.Series, strategy_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    recovery = df["SPY_CROSS_ABOVE_MA20"].fillna(False).astype(bool)
    state = "NORMAL"
    pending_state = "NORMAL"
    risk_states = []
    w_spy = []
    costs = []
    prev_weight = 1.0
    events = []
    for i in range(len(df)):
        if pending_state != state:
            old = state
            state = pending_state
            new_weight = 0.0 if state == "RISK" else 1.0
            turnover = abs(new_weight - prev_weight) + abs((1 - new_weight) - (1 - prev_weight))
            cost = 0.5 * turnover * CONFIG["one_way_cost_bps"] / 10000
            events.append(
                {
                    "strategy": strategy_name,
                    "event_date": df.iloc[i]["date"],
                    "event_type": "ENTER_RISK" if state == "RISK" else "EXIT_RISK",
                    "macro_regime_confirmed": df.iloc[i]["macro_regime_confirmed"],
                    "spy_drawdown_from_previous_high": df.iloc[i]["spy_drawdown_from_previous_high"],
                }
            )
            prev_weight = new_weight
            costs.append(cost)
        else:
            costs.append(0.0)
        weight = 0.0 if state == "RISK" else 1.0
        risk_states.append(state)
        w_spy.append(weight)
        next_state = state
        if state == "NORMAL" and bool(entry_signal.iloc[i]):
            next_state = "RISK"
        elif state == "RISK" and bool(recovery.iloc[i]):
            next_state = "NORMAL"
        pending_state = next_state

    out = pd.DataFrame(index=df.index)
    rs_col, ws_col, wc_col, tc_col = _weight_cols(strategy_name)
    out[rs_col] = risk_states
    out[ws_col] = w_spy
    out[wc_col] = 1 - out[ws_col]
    out[tc_col] = costs
    out[f"{strategy_name}_return"] = out[ws_col] * df["spy_daily_return"] + out[wc_col] * df["daily_rf"] - out[tc_col]
    out[f"{strategy_name}_nav"] = (1 + out[f"{strategy_name}_return"].fillna(0.0)).cumprod()
    return out, pd.DataFrame(events)


def compute_performance_metrics(df: pd.DataFrame, strategies: List[str]) -> pd.DataFrame:
    rows = []
    years = len(df) / 252
    for s in strategies:
        ret_col = f"{s}_return"
        nav_col = f"{s}_nav"
        if ret_col not in df.columns or nav_col not in df.columns:
            continue
        ret = df[ret_col].fillna(0.0)
        nav = df[nav_col]
        vol = ret.std(ddof=0) * math.sqrt(252)
        excess = ret - df["daily_rf"].fillna(0.0)
        rs_col, ws_col, wc_col, tc_col = _weight_cols(s)
        rows.append(
            {
                "strategy": s,
                "annualized_return": nav.iloc[-1] ** (1 / years) - 1,
                "volatility": vol,
                "Sharpe": excess.mean() / excess.std(ddof=0) * math.sqrt(252) if excess.std(ddof=0) > 0 else np.nan,
                "max_drawdown": (nav / nav.cummax() - 1).min(),
                "final_nav": nav.iloc[-1],
                "switches": int((df.get(ws_col, pd.Series(1, index=df.index)).diff().abs() > 0).sum()),
                "risk_entries": int(
                    (df.get(rs_col, pd.Series("NORMAL", index=df.index)).eq("RISK")
                    & ~df.get(rs_col, pd.Series("NORMAL", index=df.index)).shift(1, fill_value="NORMAL").eq("RISK")).sum()
                ),
                "time_in_cash": df.get(wc_col, pd.Series(0, index=df.index)).mean(),
                "transaction_cost_drag": df.get(tc_col, pd.Series(0, index=df.index)).sum(),
            }
        )
    return pd.DataFrame(rows)


def compute_risk_episodes(df: pd.DataFrame, strategies: List[str]) -> pd.DataFrame:
    rows = []
    for s in strategies:
        rs_col = f"{s}_risk_state"
        if rs_col not in df.columns:
            continue
        is_risk = df[rs_col].eq("RISK")
        starts = df.index[is_risk & ~is_risk.shift(1, fill_value=False)]
        for eid, start in enumerate(starts, 1):
            end = start
            while end + 1 < len(df) and is_risk.iloc[end + 1]:
                end += 1
            sub = df.iloc[start : end + 1]
            spy_nav = (1 + sub["spy_daily_return"].fillna(0.0)).cumprod()
            strat_nav = (1 + sub[f"{s}_return"].fillna(0.0)).cumprod()
            rows.append(
                {
                    "strategy": s,
                    "episode_id": eid,
                    "risk_start_date": sub.iloc[0]["date"],
                    "risk_end_date": sub.iloc[-1]["date"],
                    "duration_days": len(sub),
                    "entry_regime": sub.iloc[0]["macro_regime_confirmed"],
                    "SPY_return_during_risk": spy_nav.iloc[-1] - 1,
                    "strategy_return_during_risk": strat_nav.iloc[-1] - 1,
                    "SPY_max_drawdown_during_risk": (spy_nav / spy_nav.cummax() - 1).min(),
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
            nav_col = f"{s}_nav"
            if ret_col not in sub.columns:
                continue
            ret = sub[ret_col].fillna(0.0)
            nav = (1 + ret).cumprod()
            excess = ret - sub["daily_rf"].fillna(0.0)
            wc_col = f"{s}_weight_cash"
            ws_col = f"{s}_weight_spy"
            rows.append(
                {
                    "period": period,
                    "strategy": s,
                    "cumulative_return": nav.iloc[-1] - 1,
                    "max_drawdown": (nav / nav.cummax() - 1).min(),
                    "volatility": ret.std(ddof=0) * math.sqrt(252),
                    "Sharpe": excess.mean() / excess.std(ddof=0) * math.sqrt(252) if excess.std(ddof=0) > 0 else np.nan,
                    "time_in_cash": sub.get(wc_col, pd.Series(0, index=sub.index)).mean(),
                    "number_of_switches": int((sub.get(ws_col, pd.Series(1, index=sub.index)).diff().abs() > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def build_decision_table(event_summary: pd.DataFrame, flat_perf: pd.DataFrame, full_perf: pd.DataFrame, crisis: pd.DataFrame) -> pd.DataFrame:
    rows = []
    flat_map = {
        "FLAT_VIX_ONLY": "FLAT_VIX_ONLY_R3",
        "FLAT_CREDIT_ONLY": "FLAT_CREDIT_ONLY_R3",
        "FLAT_VIX_OR_CREDIT": "FLAT_VIX_OR_CREDIT_R3",
        "FLAT_VIX_AND_CREDIT_10D": "FLAT_VIX_AND_CREDIT_10D_R3",
        "FLAT_VIX_AND_CREDIT_21D": "FLAT_VIX_AND_CREDIT_21D_R3",
        "FLAT_VIX_WITH_CREDIT_WEAK_CONFIRM": "FLAT_VIX_WITH_CREDIT_WEAK_CONFIRM_R3",
        "FLAT_CREDIT_WITH_VIX_WARNING_CONFIRM_21D": "FLAT_CREDIT_WITH_VIX_WARNING_CONFIRM_R3",
    }
    full_map = {
        "FLAT_VIX_ONLY": "FULL_FLAT_VIX_ONLY",
        "FLAT_CREDIT_ONLY": "FULL_FLAT_CREDIT_ONLY",
        "FLAT_VIX_OR_CREDIT": "FULL_FLAT_VIX_OR_CREDIT",
        "FLAT_VIX_AND_CREDIT_10D": "FULL_FLAT_VIX_AND_CREDIT_10D",
        "FLAT_VIX_AND_CREDIT_21D": "FULL_FLAT_VIX_AND_CREDIT_21D",
        "FLAT_VIX_WITH_CREDIT_WEAK_CONFIRM": "FULL_FLAT_VIX_WITH_CREDIT_WEAK_CONFIRM",
        "FLAT_CREDIT_WITH_VIX_WARNING_CONFIRM_21D": "FULL_FLAT_CREDIT_WITH_VIX_WARNING_CONFIRM",
    }
    vix_only_sharpe = full_perf.loc[full_perf["strategy"].eq("FULL_FLAT_VIX_ONLY"), "Sharpe"]
    vix_only_maxdd = full_perf.loc[full_perf["strategy"].eq("FULL_FLAT_VIX_ONLY"), "max_drawdown"]
    vix_only_sharpe = vix_only_sharpe.iloc[0] if not vix_only_sharpe.empty else np.nan
    vix_only_maxdd = vix_only_maxdd.iloc[0] if not vix_only_maxdd.empty else np.nan
    max_full_sharpe = full_perf["Sharpe"].max() if not full_perf.empty else np.nan
    for _, r in event_summary.iterrows():
        trig = r["trigger_name"]
        flat_name = flat_map.get(trig)
        full_name = full_map.get(trig)
        flat_row = flat_perf[flat_perf["strategy"].eq(flat_name)] if flat_name else pd.DataFrame()
        full_row = full_perf[full_perf["strategy"].eq(full_name)]
        covid = crisis[(crisis["strategy"].eq(full_name)) & (crisis["period"].eq("COVID_2020"))]
        p2025 = crisis[(crisis["strategy"].eq(full_name)) & (crisis["period"].eq("2025_PULLBACK"))]
        if r["event_count"] < 3:
            action = "INSUFFICIENT_SAMPLE"
        elif r["false_alarm_rate_21d"] > 0.50:
            action = "DIAGNOSTIC_ONLY"
        elif "AND" in trig and (covid.empty or covid["time_in_cash"].fillna(0).iloc[0] < 0.05):
            action = "TOO_RESTRICTIVE"
        elif (
            not full_row.empty
            and full_row["Sharpe"].iloc[0] >= max_full_sharpe - 0.02
            and (np.isnan(vix_only_maxdd) or full_row["max_drawdown"].iloc[0] >= vix_only_maxdd)
            and full_row["switches"].iloc[0] <= (full_perf["switches"].median() * 1.5)
        ):
            action = "CANDIDATE_FOR_BACKBONE"
        else:
            action = "KEEP_AS_REFERENCE"
        rows.append(
            {
                "trigger_name": trig,
                "event_count": r["event_count"],
                "false_alarm_rate_21d": r["false_alarm_rate_21d"],
                "pct_mdd_21d_below_5": r["pct_mdd_21d_below_5"],
                "avg_forward_mdd_21d": r["avg_forward_mdd_21d"],
                "quick_rebound_rate_21d": r["quick_rebound_rate_21d"],
                "flat_only_sharpe": flat_row["Sharpe"].iloc[0] if not flat_row.empty else np.nan,
                "flat_only_maxdd": flat_row["max_drawdown"].iloc[0] if not flat_row.empty else np.nan,
                "full_strategy_sharpe": full_row["Sharpe"].iloc[0] if not full_row.empty else np.nan,
                "full_strategy_maxdd": full_row["max_drawdown"].iloc[0] if not full_row.empty else np.nan,
                "full_strategy_annret": full_row["annualized_return"].iloc[0] if not full_row.empty else np.nan,
                "full_strategy_switches": full_row["switches"].iloc[0] if not full_row.empty else np.nan,
                "full_strategy_time_in_cash": full_row["time_in_cash"].iloc[0] if not full_row.empty else np.nan,
                "crisis_COVID_return": covid["cumulative_return"].iloc[0] if not covid.empty else np.nan,
                "crisis_2025_return": p2025["cumulative_return"].iloc[0] if not p2025.empty else np.nan,
                "recommended_action": action,
            }
        )
    return pd.DataFrame(rows)


def plot_event_quality(event_summary: pd.DataFrame, intensity: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        event_summary["false_alarm_rate_21d"],
        event_summary["pct_mdd_21d_below_5"],
        s=event_summary["event_count"] * 40,
        c=event_summary["avg_forward_mdd_21d"],
        cmap="viridis_r",
        alpha=0.8,
    )
    for _, r in event_summary.iterrows():
        ax.text(r["false_alarm_rate_21d"], r["pct_mdd_21d_below_5"], r["trigger_name"], fontsize=7)
    ax.set_xlabel("false_alarm_rate_21d")
    ax.set_ylabel("P(21D MDD < -5%)")
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "flat_trigger_event_quality_scatter.png", dpi=150)
    plt.close(fig)

    if not intensity.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(intensity["FLAT_STRESS_INTENSITY"], intensity["avg_forward_mdd_21d"])
        axes[0].set_title("avg forward mdd 21d")
        axes[1].bar(intensity["FLAT_STRESS_INTENSITY"], intensity["avg_forward_return_21d"])
        axes[1].set_title("avg forward return 21d")
        fig.tight_layout()
        fig.savefig(CONFIG["figure_dir"] / "flat_intensity_forward_outcomes.png", dpi=150)
        plt.close(fig)


def plot_backtests(flat_panel: pd.DataFrame, full_panel: pd.DataFrame, flat_perf: pd.DataFrame, full_perf: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for s in ["SPY_BUY_HOLD"] + FLAT_ONLY_STRATEGIES:
        if f"{s}_nav" in flat_panel.columns:
            ax.plot(flat_panel["date"], flat_panel[f"{s}_nav"], label=s)
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "flat_only_equity_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    selected = [
        "SPY_BUY_HOLD",
        "OLD_FULL_BASELINE",
        "EX_INVERTED_FULL_BASELINE",
        "FULL_FLAT_VIX_ONLY",
        "FULL_FLAT_VIX_OR_CREDIT",
        "FULL_FLAT_VIX_AND_CREDIT_21D",
        "FULL_FLAT_VIX_WITH_CREDIT_WEAK_CONFIRM",
    ]
    for s in selected:
        if f"{s}_nav" in full_panel.columns:
            ax.plot(full_panel["date"], full_panel[f"{s}_nav"], label=s)
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "full_backbone_equity_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for s in selected:
        if f"{s}_nav" in full_panel.columns:
            nav = full_panel[f"{s}_nav"]
            ax.plot(full_panel["date"], nav / nav.cummax() - 1, label=s)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "full_backbone_drawdowns.png", dpi=150)
    plt.close(fig)

    show = full_perf[full_perf["strategy"].isin(selected)]
    if not show.empty:
        fig, axes = plt.subplots(1, 5, figsize=(16, 4))
        for ax, metric in zip(axes, ["annualized_return", "Sharpe", "max_drawdown", "switches", "time_in_cash"]):
            ax.bar(show["strategy"], show[metric])
            ax.set_title(metric)
            ax.tick_params(axis="x", rotation=90, labelsize=7)
        fig.tight_layout()
        fig.savefig(CONFIG["figure_dir"] / "full_performance_bar_charts.png", dpi=150)
        plt.close(fig)


def plot_case_studies(df: pd.DataFrame) -> None:
    cases = {
        "case_study_COVID_flat_triggers.png": ["2020-02-01", "2020-06-30"],
        "case_study_2025_flat_triggers.png": ["2025-01-01", "2025-12-31"],
        "case_study_2015_2016_flat_triggers.png": ["2015-05-01", "2016-03-31"],
        "case_study_2018Q4_flat_triggers.png": ["2018-10-01", "2019-01-31"],
    }
    for filename, (start, end) in cases.items():
        sub = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))].copy()
        if len(sub) < 5:
            continue
        fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(sub["date"], sub["spy_drawdown_from_previous_high"], label="SPY drawdown")
        axes[0].axhline(-0.05, color="red", linestyle="--", linewidth=0.8)
        axes[0].legend(fontsize=8)
        axes[1].plot(sub["date"], sub["VIX_ZSCORE_120D"], label="VIX z120")
        axes[1].axhline(CONFIG["vix_z_threshold"], color="red", linestyle="--", linewidth=0.8)
        axes[1].legend(fontsize=8)
        axes[2].plot(sub["date"], sub["D_CREDIT_SPREAD_20D"], label="credit chg20")
        axes[2].axhline(CONFIG["credit_change_threshold"], color="red", linestyle="--", linewidth=0.8)
        axes[2].legend(fontsize=8)
        axes[3].plot(sub["date"], sub["FLAT_VIX_STRESS"].astype(int), label="VIX")
        axes[3].plot(sub["date"], sub["FLAT_CREDIT_STRESS"].astype(int), label="Credit")
        axes[3].plot(sub["date"], sub["FLAT_VIX_OR_CREDIT"].astype(int), label="OR")
        axes[3].plot(sub["date"], sub["FLAT_VIX_AND_CREDIT_21D"].astype(int), label="AND21")
        axes[3].legend(fontsize=8, ncol=4)
        fig.tight_layout()
        fig.savefig(CONFIG["figure_dir"] / filename, dpi=150)
        plt.close(fig)


def write_markdown_report(event_summary: pd.DataFrame, intensity: pd.DataFrame, flat_perf: pd.DataFrame, full_perf: pd.DataFrame, crisis: pd.DataFrame, decision: pd.DataFrame) -> None:
    def table(df: pd.DataFrame, n: int = 20) -> str:
        return "_No data._" if df.empty else df.head(n).to_markdown(index=False)

    content = f"""# FLAT VIX Credit Trigger Diagnostic

## Purpose

This analysis isolates how VIX stress and credit stress should be combined inside the FLAT regime.

## Current Issue

- FLAT + VIX is a high-quality fast-stress trigger.
- FLAT + credit has some information, but weaker quality than STEEP credit stress.
- The open question is whether OR / AND / confirmation logic improves signal quality enough to justify inclusion in the full backbone.

## Trigger Definitions

The study includes:

- VIX only
- Credit only
- OR
- AND same day / 10D / 21D
- VIX with credit confirmation
- Credit with VIX confirmation
- Intensity labels: `NONE`, `VIX_ONLY`, `CREDIT_ONLY`, `BOTH`

## Event-Level Findings

{table(event_summary)}

Intensity summary:

{table(intensity)}

## FLAT-Only Backtest Findings

{table(flat_perf)}

## Full Backbone Findings

{table(full_perf)}

## Crisis Window Analysis

{table(crisis, 40)}

## Recommendation

{table(decision)}

The candidate marked `CANDIDATE_FOR_BACKBONE` should be the first FLAT stress definition tested in the next hedge-allocation backtest. Signals marked `DIAGNOSTIC_ONLY` or `TOO_RESTRICTIVE` should not replace the current FLAT logic without stronger evidence.

## Caveats

- FLAT credit event count is still limited.
- Credit spread is weekly forward-filled upstream.
- This only compares full-risk SPY/CASH switches, not partial-risk intensity.
- VIX and credit may capture different timing within the same stress episode.
"""
    (CONFIG["output_dir"] / "FLAT_VIX_CREDIT_TRIGGER_DIAGNOSTIC.md").write_text(content, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = build_flat_signals(load_panel())
    panel.to_csv(CONFIG["output_dir"] / "signal_panel_debug.csv", index=False)

    events = extract_flat_events(panel)
    event_summary, intensity = summarize_flat_events(events)
    events.to_csv(CONFIG["output_dir"] / "flat_trigger_event_table.csv", index=False)
    event_summary.to_csv(CONFIG["output_dir"] / "flat_trigger_event_summary.csv", index=False)
    intensity.to_csv(CONFIG["output_dir"] / "flat_intensity_forward_outcomes.csv", index=False)

    # Flat-only backtests
    flat_panel = panel[[
        "date", "spy_price", "spy_daily_return", "daily_rf", "macro_regime_confirmed", "monthly_either_state",
        "VIX_LEVEL", "VIX_ZSCORE_120D", "CREDIT_SPREAD_BAA_AAA", "D_CREDIT_SPREAD_20D",
        "spy_drawdown_from_previous_high", "SPY_MA20", "SPY_CROSS_ABOVE_MA20",
        "FLAT_VIX_STRESS", "FLAT_CREDIT_STRESS", "FLAT_VIX_WARNING", "FLAT_CREDIT_WEAK", "FLAT_STRESS_INTENSITY"
    ]].copy()
    flat_panel["SPY_BUY_HOLD_weight_spy"] = 1.0
    flat_panel["SPY_BUY_HOLD_weight_cash"] = 0.0
    flat_panel["SPY_BUY_HOLD_return"] = flat_panel["spy_daily_return"]
    flat_panel["SPY_BUY_HOLD_nav"] = (1 + flat_panel["SPY_BUY_HOLD_return"]).cumprod()
    flat_panel["CASH_ONLY_weight_spy"] = 0.0
    flat_panel["CASH_ONLY_weight_cash"] = 1.0
    flat_panel["CASH_ONLY_return"] = flat_panel["daily_rf"]
    flat_panel["CASH_ONLY_nav"] = (1 + flat_panel["CASH_ONLY_return"]).cumprod()
    flat_event_logs = []
    for s in FLAT_ONLY_STRATEGIES:
        trig = FLAT_ONLY_SIGNAL_MAP[s]
        cols, logs = run_state_machine_backtest(panel, panel[trig].fillna(False), s)
        flat_panel = pd.concat([flat_panel, cols], axis=1)
        flat_event_logs.append(logs)
    flat_perf = compute_performance_metrics(flat_panel, ["SPY_BUY_HOLD", "CASH_ONLY"] + FLAT_ONLY_STRATEGIES)
    flat_panel.to_csv(CONFIG["output_dir"] / "flat_only_backtest_panel.csv", index=False)
    flat_perf.to_csv(CONFIG["output_dir"] / "flat_only_performance_summary.csv", index=False)

    # Full backbone comparison
    full_panel = flat_panel.copy()
    non_flat_stress = (
        (panel["macro_regime_confirmed"].eq("STEEP") & panel["monthly_either_state"].eq("SELL"))
        | (
            panel["macro_regime_confirmed"].eq("STEEP")
            & (panel["spy_drawdown_from_previous_high"] <= CONFIG["dd_threshold"])
            & (panel["D_CREDIT_SPREAD_20D"] > CONFIG["credit_change_threshold"])
        )
    )
    full_logs = []
    mapping = {
        "FULL_FLAT_VIX_ONLY": panel["FLAT_VIX_ONLY"],
        "FULL_FLAT_CREDIT_ONLY": panel["FLAT_CREDIT_ONLY"],
        "FULL_FLAT_VIX_OR_CREDIT": panel["FLAT_VIX_OR_CREDIT"],
        "FULL_FLAT_VIX_AND_CREDIT_10D": panel["FLAT_VIX_AND_CREDIT_10D"],
        "FULL_FLAT_VIX_AND_CREDIT_21D": panel["FLAT_VIX_AND_CREDIT_21D"],
        "FULL_FLAT_VIX_WITH_CREDIT_WEAK_CONFIRM": panel["FLAT_VIX_WITH_CREDIT_WEAK_CONFIRM"],
        "FULL_FLAT_CREDIT_WITH_VIX_WARNING_CONFIRM": panel["FLAT_CREDIT_WITH_VIX_WARNING_CONFIRM_21D"],
    }
    for s, trig in mapping.items():
        cols, logs = run_state_machine_backtest(panel, (non_flat_stress | trig.fillna(False)), s)
        full_panel = pd.concat([full_panel, cols], axis=1)
        full_logs.append(logs)

    # Old and ex-inverted baselines
    if "STRESS_RECOVERY_R3_CREDIT_DD5_nav" in panel.columns:
        full_panel["OLD_FULL_BASELINE_nav"] = panel["STRESS_RECOVERY_R3_CREDIT_DD5_nav"]
        full_panel["OLD_FULL_BASELINE_return"] = panel["STRESS_RECOVERY_R3_CREDIT_DD5_return"]
        full_panel["OLD_FULL_BASELINE_weight_spy"] = panel.get("CREDIT_DD5_R3_weight_spy", np.nan)
        full_panel["OLD_FULL_BASELINE_weight_cash"] = panel.get("CREDIT_DD5_R3_weight_cash", np.nan)
        full_panel["OLD_FULL_BASELINE_risk_state"] = panel.get("CREDIT_DD5_R3_risk_state", "")
    cols, logs = run_state_machine_backtest(panel, non_flat_stress | panel["FLAT_VIX_ONLY"], "EX_INVERTED_FULL_BASELINE")
    full_panel = pd.concat([full_panel, cols], axis=1)
    full_logs.append(logs)

    full_strats = ["SPY_BUY_HOLD", "MONTHLY_EITHER_CONFIRM", "OLD_FULL_BASELINE", "EX_INVERTED_FULL_BASELINE"] + FULL_STRATEGIES
    full_perf = compute_performance_metrics(full_panel, full_strats)
    full_episodes = compute_risk_episodes(full_panel, FULL_STRATEGIES + ["EX_INVERTED_FULL_BASELINE"])
    crisis = compute_crisis_performance(full_panel, full_strats)
    decision = build_decision_table(event_summary, flat_perf, full_perf, crisis)

    full_panel.to_csv(CONFIG["output_dir"] / "full_backtest_panel.csv", index=False)
    full_perf.to_csv(CONFIG["output_dir"] / "full_performance_summary.csv", index=False)
    full_episodes.to_csv(CONFIG["output_dir"] / "full_risk_episodes.csv", index=False)
    crisis.to_csv(CONFIG["output_dir"] / "crisis_performance.csv", index=False)
    decision.to_csv(CONFIG["output_dir"] / "flat_trigger_decision_table.csv", index=False)

    plot_event_quality(event_summary, intensity)
    plot_backtests(flat_panel, full_panel, flat_perf, full_perf)
    plot_case_studies(panel)
    write_markdown_report(event_summary, intensity, flat_perf, full_perf, crisis, decision)

    def _show(name: str) -> str:
        row = event_summary[event_summary["trigger_name"].eq(name)]
        if row.empty:
            return "n/a"
        r = row.iloc[0]
        return f"{int(r['event_count'])} / {r['false_alarm_rate_21d']:.2f} / {r['pct_mdd_21d_below_5']:.2f}"

    print(f"1. FLAT VIX event count / false alarm / P(21D MDD<-5%): {_show('FLAT_VIX_ONLY')}")
    print(f"2. FLAT credit event count / false alarm / P(21D MDD<-5%): {_show('FLAT_CREDIT_ONLY')}")
    print(f"3. FLAT OR event count / false alarm / P(21D MDD<-5%): {_show('FLAT_VIX_OR_CREDIT')}")
    print(f"4. FLAT AND 21D event count / false alarm / P(21D MDD<-5%): {_show('FLAT_VIX_AND_CREDIT_21D')}")
    for s in ["FULL_FLAT_VIX_ONLY", "FULL_FLAT_VIX_OR_CREDIT", "FULL_FLAT_VIX_AND_CREDIT_21D"]:
        row = full_perf[full_perf["strategy"].eq(s)]
        if not row.empty:
            r = row.iloc[0]
            print(f"{s}: Ann {r['annualized_return']:.2%}, Sharpe {r['Sharpe']:.2f}, MaxDD {r['max_drawdown']:.2%}, switches {int(r['switches'])}")
    best_sharpe = full_perf.loc[full_perf["Sharpe"].idxmax()] if not full_perf.empty else None
    best_maxdd = full_perf.loc[full_perf["max_drawdown"].idxmax()] if not full_perf.empty else None
    print(f"8. Best full strategy by Sharpe: {best_sharpe['strategy'] if best_sharpe is not None else 'n/a'}")
    print(f"9. Best full strategy by MaxDD: {best_maxdd['strategy'] if best_maxdd is not None else 'n/a'}")
    cand = decision[decision["recommended_action"].eq("CANDIDATE_FOR_BACKBONE")]
    print(f"10. Recommended FLAT stress rule: {cand.iloc[0]['trigger_name'] if not cand.empty else 'KEEP_VIX_ONLY_OR_REFERENCE'}")
    print(f"Saved outputs: {CONFIG['output_dir'].resolve()} and {CONFIG['figure_dir'].resolve()}")


if __name__ == "__main__":
    main()
