from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "spy_cash_stress_recovery_with_credit"
FIGURE_DIR = ROOT / "figures" / "spy_cash_stress_recovery_with_credit"

CREDIT_PANEL = ROOT / "results" / "credit_spread_stress_trigger_diagnostic" / "credit_trigger_daily_panel.csv"
BASE_PANEL = ROOT / "results" / "spy_cash_stress_recovery_timing" / "daily_backtest_panel.csv"
RULE_PANEL = ROOT / "results" / "rule_diagnostics" / "rule_state_panel.csv"
RAW_WBAA = ROOT / "data" / "raw" / "macro" / "Credit" / "WBAA.csv"
RAW_WAAA = ROOT / "data" / "raw" / "macro" / "Credit" / "WAAA.csv"

CONFIG = {
    "vix_z_window": 120,
    "vix_z_threshold": 3.0,
    "credit_change_window": 20,
    "credit_change_abs_threshold": 0.10,
    "dd_threshold": -0.05,
    "recovery_rule": "R3_SPY_CROSS_ABOVE_MA20",
    "one_way_cost_bps": 5.0,
    "output_dir": str(OUTPUT_DIR),
    "figure_dir": str(FIGURE_DIR),
}

STRATEGY_SPECS = {
    "STRESS_RECOVERY_R3_BASE": ("base_stress_entry_signal", "base_stress_entry_reason", "BASE_R3"),
    "STRESS_RECOVERY_R3_CREDIT_DD5": ("credit_dd5_stress_entry_signal", "credit_dd5_stress_entry_reason", "CREDIT_DD5_R3"),
    "STRESS_RECOVERY_R3_CREDIT_ONLY": ("credit_only_stress_entry_signal", "credit_only_stress_entry_reason", "CREDIT_ONLY_R3"),
}
STRATEGIES = ["SPY_BUY_HOLD", "CASH_ONLY", "MONTHLY_EITHER_CONFIRM"] + list(STRATEGY_SPECS)

CASE_WINDOWS = {
    "GFC_2008_2009": ("2008-09-01", "2009-03-31"),
    "CREDIT_COMMODITY_2015_2016": ("2015-05-01", "2016-03-31"),
    "TIGHTENING_2018Q4": ("2018-10-01", "2019-01-31"),
    "COVID_2020": ("2020-02-19", "2020-04-30"),
    "INFLATION_2022": ("2021-11-01", "2023-03-31"),
    "HIGH_RATE_2023": ("2023-07-01", "2023-11-30"),
    "RECENT_2024_2026": ("2024-01-01", "2026-12-31"),
}

DAILY_OUT = OUTPUT_DIR / "daily_backtest_panel.csv"
EVENT_LOG_OUT = OUTPUT_DIR / "risk_state_event_log.csv"
EPISODES_OUT = OUTPUT_DIR / "risk_episodes.csv"
PERF_OUT = OUTPUT_DIR / "performance_summary.csv"
CRISIS_OUT = OUTPUT_DIR / "crisis_performance.csv"
REGIME_OUT = OUTPUT_DIR / "performance_by_regime.csv"
ENTRY_OUT = OUTPUT_DIR / "entry_reason_summary.csv"
REPORT_OUT = OUTPUT_DIR / "SPY_CASH_STRESS_RECOVERY_WITH_CREDIT_SUMMARY.md"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    date_col = "date" if "date" in df.columns else "observation_date" if "observation_date" in df.columns else "DATE" if "DATE" in df.columns else None
    if date_col is None:
        return None
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date")


def load_credit_data() -> tuple[pd.DataFrame | None, str]:
    rule = _read_csv(RULE_PANEL)
    if rule is not None and "CREDIT_SPREAD_BAA_AAA" in rule.columns:
        out = rule[["date", "CREDIT_SPREAD_BAA_AAA"]].copy()
        out["CREDIT_SPREAD_BAA_AAA"] = pd.to_numeric(out["CREDIT_SPREAD_BAA_AAA"], errors="coerce")
        return out, "results/rule_diagnostics/rule_state_panel.csv:CREDIT_SPREAD_BAA_AAA"
    wbaa, waaa = _read_csv(RAW_WBAA), _read_csv(RAW_WAAA)
    if wbaa is not None and waaa is not None and {"WBAA"}.issubset(wbaa.columns) and {"WAAA"}.issubset(waaa.columns):
        out = wbaa[["date", "WBAA"]].merge(waaa[["date", "WAAA"]], on="date", how="outer").sort_values("date")
        out["WBAA"] = pd.to_numeric(out["WBAA"], errors="coerce")
        out["WAAA"] = pd.to_numeric(out["WAAA"], errors="coerce")
        out[["WBAA", "WAAA"]] = out[["WBAA", "WAAA"]].ffill()
        out["CREDIT_SPREAD_BAA_AAA"] = out["WBAA"] - out["WAAA"]
        return out, "raw WBAA-WAAA, forward-filled"
    return None, "missing"


def load_base_panel() -> tuple[pd.DataFrame, str]:
    panel = _read_csv(CREDIT_PANEL)
    credit_source = "results/credit_spread_stress_trigger_diagnostic/credit_trigger_daily_panel.csv"
    if panel is None:
        panel = _read_csv(BASE_PANEL)
        if panel is None:
            raise FileNotFoundError("Could not load stress-recovery base panel.")
        credit, credit_source = load_credit_data()
        if credit is None:
            raise ValueError("Could not load credit spread data.")
        panel = pd.merge_asof(panel.sort_values("date"), credit.sort_values("date"), on="date", direction="backward")
    required = ["spy_price", "spy_daily_return", "daily_rf", "macro_regime_confirmed", "monthly_either_state", "VIX_LEVEL"]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return panel.sort_values("date").reset_index(drop=True), credit_source


def build_credit_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["spy_price"] = pd.to_numeric(out["spy_price"], errors="coerce")
    out["spy_daily_return"] = pd.to_numeric(out["spy_daily_return"], errors="coerce")
    out["daily_rf"] = pd.to_numeric(out["daily_rf"], errors="coerce")
    out["previous_high"] = out["spy_price"].cummax()
    out["spy_drawdown_from_previous_high"] = out["spy_price"] / out["previous_high"] - 1.0
    if "SPY_MA20" not in out.columns:
        out["SPY_MA20"] = out["spy_price"].rolling(20, min_periods=20).mean()
    out["SPY_CROSS_ABOVE_MA20"] = (out["spy_price"] > out["SPY_MA20"]) & (out["spy_price"].shift(1) <= out["SPY_MA20"].shift(1))
    if "VIX_ZSCORE_120D" not in out.columns:
        roll = out["VIX_LEVEL"].rolling(CONFIG["vix_z_window"], min_periods=CONFIG["vix_z_window"])
        out["VIX_ZSCORE_120D"] = (out["VIX_LEVEL"] - roll.mean()) / roll.std(ddof=1).replace(0, np.nan)
    out["CREDIT_SPREAD_BAA_AAA"] = pd.to_numeric(out["CREDIT_SPREAD_BAA_AAA"], errors="coerce").ffill()
    out["D_CREDIT_SPREAD_20D"] = out["CREDIT_SPREAD_BAA_AAA"].diff(CONFIG["credit_change_window"])
    out["CREDIT_CHG20_GT_0_10"] = out["D_CREDIT_SPREAD_20D"] > CONFIG["credit_change_abs_threshold"]
    out["DD5_AND_CREDIT_CHG20_GT_0_10"] = (out["spy_drawdown_from_previous_high"] <= CONFIG["dd_threshold"]) & out["CREDIT_CHG20_GT_0_10"]
    return out.dropna(subset=["spy_price", "spy_daily_return", "daily_rf"]).reset_index(drop=True)


def _reason_from_flags(flags: list[tuple[bool, str]]) -> str:
    return "+".join(reason for cond, reason in flags if bool(cond))


def build_base_stress_signal(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    flat_vix = out["macro_regime_confirmed"].eq("FLAT") & (out["VIX_ZSCORE_120D"] >= CONFIG["vix_z_threshold"])
    steep_either = out["macro_regime_confirmed"].eq("STEEP") & out["monthly_either_state"].eq("SELL")
    out["base_stress_entry_signal"] = flat_vix | steep_either
    out["base_stress_entry_reason"] = [
        _reason_from_flags([(fv, "FLAT_VIX_STRESS"), (se, "STEEP_EITHER_SELL")])
        for fv, se in zip(flat_vix, steep_either)
    ]
    return out


def build_credit_stress_signals(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    dd5 = out["DD5_AND_CREDIT_CHG20_GT_0_10"]
    pure = out["CREDIT_CHG20_GT_0_10"]
    out["credit_dd5_stress_entry_signal"] = out["base_stress_entry_signal"] | dd5
    out["credit_only_stress_entry_signal"] = out["base_stress_entry_signal"] | pure
    out["credit_dd5_stress_entry_reason"] = [
        _reason_from_flags([(b, br), (c, "DD5_AND_CREDIT_CHG20_GT_0_10")])
        for b, br, c in zip(out["base_stress_entry_signal"], out["base_stress_entry_reason"], dd5)
    ]
    out["credit_only_stress_entry_reason"] = [
        _reason_from_flags([(b, br), (c, "CREDIT_CHG20_GT_0_10")])
        for b, br, c in zip(out["base_stress_entry_signal"], out["base_stress_entry_reason"], pure)
    ]
    return out


def run_state_machine_strategy(panel: pd.DataFrame, strategy: str, signal_col: str, reason_col: str, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = panel.copy()
    state_col = f"{prefix}_risk_state"
    weight_col = f"{prefix}_weight_spy"
    cash_col = f"{prefix}_weight_cash"
    ret_col = f"{strategy}_return"
    nav_col = f"{strategy}_nav"
    cost_col = f"transaction_cost_{prefix}"
    turnover_col = f"{strategy}_turnover"
    out[state_col] = ""
    out[weight_col] = np.nan
    out[cash_col] = np.nan
    out[ret_col] = np.nan
    out[nav_col] = np.nan
    out[cost_col] = 0.0
    out[turnover_col] = 0.0
    state = "NORMAL"
    pending_state = "NORMAL"
    pending_reason = ""
    nav = 1.0
    events = []
    cost_rate = CONFIG["one_way_cost_bps"] / 10000.0
    for i, row in out.iterrows():
        old_state = state
        if i > 0 and pending_state != state:
            state = pending_state
            old_w = 1.0 if old_state == "NORMAL" else 0.0
            new_w = 1.0 if state == "NORMAL" else 0.0
            turnover = abs(new_w - old_w) + abs((1 - new_w) - (1 - old_w))
            cost = 0.5 * turnover * cost_rate
            out.loc[i, cost_col] = cost
            out.loc[i, turnover_col] = turnover
            sig = out.loc[i - 1]
            events.append(
                {
                    "strategy": strategy,
                    "signal_date": sig["date"],
                    "event_date": row["date"],
                    "event_type": "ENTER_RISK" if state == "RISK" else "EXIT_RISK",
                    "reason": pending_reason,
                    "macro_regime_confirmed": sig["macro_regime_confirmed"],
                    "monthly_either_state": sig["monthly_either_state"],
                    "VIX_LEVEL": sig["VIX_LEVEL"],
                    "VIX_ZSCORE_120D": sig["VIX_ZSCORE_120D"],
                    "CREDIT_SPREAD_BAA_AAA": sig["CREDIT_SPREAD_BAA_AAA"],
                    "D_CREDIT_SPREAD_20D": sig["D_CREDIT_SPREAD_20D"],
                    "spy_drawdown_from_previous_high": sig["spy_drawdown_from_previous_high"],
                    "SPY_price": sig["spy_price"],
                    "SPY_MA20": sig["SPY_MA20"],
                    "previous_state": old_state,
                    "new_state": state,
                }
            )
        w_spy = 1.0 if state == "NORMAL" else 0.0
        w_cash = 1.0 - w_spy
        daily_ret = w_spy * row["spy_daily_return"] + w_cash * row["daily_rf"] - out.loc[i, cost_col]
        nav *= 1.0 + float(daily_ret)
        out.loc[i, state_col] = state
        out.loc[i, weight_col] = w_spy
        out.loc[i, cash_col] = w_cash
        out.loc[i, ret_col] = daily_ret
        out.loc[i, nav_col] = nav

        pending_state = state
        pending_reason = ""
        if state == "NORMAL" and bool(row[signal_col]):
            pending_state = "RISK"
            pending_reason = row[reason_col]
        elif state == "RISK" and bool(row["SPY_CROSS_ABOVE_MA20"]):
            pending_state = "NORMAL"
            pending_reason = "R3_SPY_CROSS_ABOVE_MA20"
    return out, pd.DataFrame(events)


def build_benchmarks(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["SPY_BUY_HOLD_return"] = out["spy_daily_return"]
    out["SPY_BUY_HOLD_nav"] = (1 + out["SPY_BUY_HOLD_return"]).cumprod()
    out["CASH_ONLY_return"] = out["daily_rf"]
    out["CASH_ONLY_nav"] = (1 + out["CASH_ONLY_return"]).cumprod()
    if "MONTHLY_EITHER_CONFIRM_return" not in out.columns or "transaction_cost_MONTHLY_EITHER_CONFIRM" not in out.columns:
        w = pd.to_numeric(out.get("monthly_either_weight_spy", pd.Series(1.0, index=out.index)), errors="coerce").fillna(1.0)
        prev = w.shift(1).fillna(w.iloc[0])
        turnover = (w - prev).abs() + ((1 - w) - (1 - prev)).abs()
        out["transaction_cost_MONTHLY_EITHER_CONFIRM"] = 0.5 * turnover * CONFIG["one_way_cost_bps"] / 10000.0
        out["MONTHLY_EITHER_CONFIRM_turnover"] = turnover
        out["MONTHLY_EITHER_CONFIRM_return"] = w * out["spy_daily_return"] + (1 - w) * out["daily_rf"] - out["transaction_cost_MONTHLY_EITHER_CONFIRM"]
    out["MONTHLY_EITHER_CONFIRM_nav"] = (1 + out["MONTHLY_EITHER_CONFIRM_return"]).cumprod()
    return out


def extract_risk_episodes(panel: pd.DataFrame, event_log: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, (_, _, prefix) in STRATEGY_SPECS.items():
        state_col = f"{prefix}_risk_state"
        risk = panel[state_col].eq("RISK").to_numpy()
        eid = 0
        i = 0
        while i < len(panel):
            if not risk[i]:
                i += 1
                continue
            start = i
            while i + 1 < len(panel) and risk[i + 1]:
                i += 1
            end = i
            eid += 1
            sub = panel.iloc[start : end + 1]
            entry = event_log[(event_log["strategy"].eq(strategy)) & (event_log["event_type"].eq("ENTER_RISK")) & (event_log["event_date"].eq(sub["date"].iloc[0]))]
            exit_ev = event_log[(event_log["strategy"].eq(strategy)) & (event_log["event_type"].eq("EXIT_RISK")) & (event_log["event_date"] > sub["date"].iloc[-1])]
            spy_wealth = sub["spy_price"] / sub["spy_price"].iloc[0]
            rows.append(
                {
                    "strategy": strategy,
                    "episode_id": eid,
                    "risk_start_date": sub["date"].iloc[0],
                    "risk_end_date": sub["date"].iloc[-1],
                    "duration_days": len(sub),
                    "entry_reason": entry["reason"].iloc[0] if not entry.empty else "",
                    "exit_reason": exit_ev["reason"].iloc[0] if not exit_ev.empty else "",
                    "macro_regime_at_entry": sub["macro_regime_confirmed"].iloc[0],
                    "SPY_drawdown_at_entry": sub["spy_drawdown_from_previous_high"].iloc[0],
                    "VIX_ZSCORE_at_entry": sub["VIX_ZSCORE_120D"].iloc[0],
                    "CREDIT_SPREAD_at_entry": sub["CREDIT_SPREAD_BAA_AAA"].iloc[0],
                    "D_CREDIT_SPREAD_20D_at_entry": sub["D_CREDIT_SPREAD_20D"].iloc[0],
                    "SPY_return_during_risk_episode": (1 + sub["spy_daily_return"]).prod() - 1,
                    "CASH_return_during_risk_episode": (1 + sub["daily_rf"]).prod() - 1,
                    "strategy_return_during_risk_episode": (1 + sub[f"{strategy}_return"]).prod() - 1,
                    "SPY_max_drawdown_during_risk_episode": (spy_wealth / spy_wealth.cummax() - 1).min(),
                    "SPY_max_runup_during_risk_episode": (sub["spy_price"] / sub["spy_price"].iloc[0] - 1).max(),
                }
            )
            i += 1
    return pd.DataFrame(rows)


def _perf(panel: pd.DataFrame, strategy: str, ret_col: str, nav_col: str, weight_col: str | None = None, cost_col: str | None = None, turnover_col: str | None = None, events: pd.DataFrame | None = None, episodes: pd.DataFrame | None = None) -> dict:
    s = panel[ret_col].dropna()
    rf = panel.loc[s.index, "daily_rf"]
    ann = (1 + s).prod() ** (252 / len(s)) - 1
    vol = s.std(ddof=1) * np.sqrt(252)
    if strategy == "CASH_ONLY":
        sharpe = 0.0
    else:
        ex = s - rf
        sharpe = ex.mean() / ex.std(ddof=1) * np.sqrt(252) if ex.std(ddof=1) != 0 else np.nan
    wealth = (1 + s).cumprod()
    mdd = (wealth / wealth.cummax() - 1).min()
    if weight_col:
        time_spy = panel[weight_col].mean()
    elif strategy == "SPY_BUY_HOLD":
        time_spy = 1.0
    elif strategy == "CASH_ONLY":
        time_spy = 0.0
    elif strategy == "MONTHLY_EITHER_CONFIRM":
        time_spy = panel.get("monthly_either_weight_spy", pd.Series(1.0, index=panel.index)).mean()
    else:
        time_spy = np.nan
    ev = events[events["strategy"].eq(strategy)] if events is not None and not events.empty else pd.DataFrame()
    ep = episodes[episodes["strategy"].eq(strategy)] if episodes is not None and not episodes.empty else pd.DataFrame()
    return {
        "strategy": strategy,
        "start_date": panel["date"].iloc[0].date().isoformat(),
        "end_date": panel["date"].iloc[-1].date().isoformat(),
        "annualized_return": ann,
        "annualized_volatility": vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
        "calmar_ratio": ann / abs(mdd) if mdd < 0 else np.nan,
        "final_nav": panel[nav_col].iloc[-1],
        "number_of_switches": len(ev) if not ev.empty else int(panel.get("MONTHLY_EITHER_CONFIRM_turnover", pd.Series(0)).sum() / 2) if strategy == "MONTHLY_EITHER_CONFIRM" else 0,
        "number_of_risk_entries": int((ev["event_type"] == "ENTER_RISK").sum()) if not ev.empty else 0,
        "number_of_risk_exits": int((ev["event_type"] == "EXIT_RISK").sum()) if not ev.empty else 0,
        "avg_risk_episode_duration": ep["duration_days"].mean() if not ep.empty else np.nan,
        "time_in_spy": time_spy,
        "time_in_cash": 1 - time_spy if pd.notna(time_spy) else np.nan,
        "total_turnover": panel[turnover_col].sum() if turnover_col and turnover_col in panel else np.nan,
        "transaction_cost_drag": panel[cost_col].sum() if cost_col and cost_col in panel else panel.get("transaction_cost_MONTHLY_EITHER_CONFIRM", pd.Series(0)).sum() if strategy == "MONTHLY_EITHER_CONFIRM" else 0.0,
    }


def compute_performance_metrics(panel: pd.DataFrame, events: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    rows = [
        _perf(panel, "SPY_BUY_HOLD", "SPY_BUY_HOLD_return", "SPY_BUY_HOLD_nav"),
        _perf(panel, "CASH_ONLY", "CASH_ONLY_return", "CASH_ONLY_nav"),
        _perf(panel, "MONTHLY_EITHER_CONFIRM", "MONTHLY_EITHER_CONFIRM_return", "MONTHLY_EITHER_CONFIRM_nav"),
    ]
    for strategy, (_, _, prefix) in STRATEGY_SPECS.items():
        rows.append(_perf(panel, strategy, f"{strategy}_return", f"{strategy}_nav", f"{prefix}_weight_spy", f"transaction_cost_{prefix}", f"{strategy}_turnover", events, episodes))
    return pd.DataFrame(rows)


def _group_stats(sub: pd.DataFrame, strategy: str, ret_col: str, weight_col: str | None = None) -> dict:
    s = sub[ret_col].dropna()
    if s.empty:
        return {"n_obs": 0, "annualized_return": np.nan, "max_drawdown": np.nan, "volatility": np.nan, "sharpe": np.nan, "time_in_cash": np.nan}
    rf = sub.loc[s.index, "daily_rf"]
    ann = (1 + s).prod() ** (252 / len(s)) - 1
    vol = s.std(ddof=1) * np.sqrt(252)
    ex = s - rf
    sharpe = 0.0 if strategy == "CASH_ONLY" else ex.mean() / ex.std(ddof=1) * np.sqrt(252) if ex.std(ddof=1) != 0 else np.nan
    wealth = (1 + s).cumprod()
    mdd = (wealth / wealth.cummax() - 1).min()
    if weight_col:
        cash = 1 - sub[weight_col].mean()
    elif strategy == "SPY_BUY_HOLD":
        cash = 0.0
    elif strategy == "CASH_ONLY":
        cash = 1.0
    elif strategy == "MONTHLY_EITHER_CONFIRM":
        cash = 1 - sub.get("monthly_either_weight_spy", pd.Series(1.0, index=sub.index)).mean()
    else:
        cash = np.nan
    return {"n_obs": len(s), "annualized_return": ann, "max_drawdown": mdd, "volatility": vol, "sharpe": sharpe, "time_in_cash": cash}


def compute_crisis_performance(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mapping = [("SPY_BUY_HOLD", "SPY_BUY_HOLD_return", None), ("CASH_ONLY", "CASH_ONLY_return", None), ("MONTHLY_EITHER_CONFIRM", "MONTHLY_EITHER_CONFIRM_return", None)]
    mapping += [(s, f"{s}_return", f"{p}_weight_spy") for s, (_, _, p) in STRATEGY_SPECS.items()]
    for period, (start, end) in CASE_WINDOWS.items():
        sub = panel[panel["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        if sub.empty:
            continue
        for strategy, ret_col, weight_col in mapping:
            st = _group_stats(sub, strategy, ret_col, weight_col)
            sw = events[(events["strategy"].eq(strategy)) & events["event_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
            rows.append({"period": period, "strategy": strategy, "cumulative_return": (1 + sub[ret_col]).prod() - 1, "number_of_switches": len(sw), **st})
    return pd.DataFrame(rows)


def compute_regime_performance(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mapping = [("SPY_BUY_HOLD", "SPY_BUY_HOLD_return", None), ("CASH_ONLY", "CASH_ONLY_return", None), ("MONTHLY_EITHER_CONFIRM", "MONTHLY_EITHER_CONFIRM_return", None)]
    mapping += [(s, f"{s}_return", f"{p}_weight_spy") for s, (_, _, p) in STRATEGY_SPECS.items()]
    for regime, sub in panel.groupby("macro_regime_confirmed"):
        for strategy, ret_col, weight_col in mapping:
            rows.append({"macro_regime_confirmed": regime, "strategy": strategy, **_group_stats(sub, strategy, ret_col, weight_col)})
    return pd.DataFrame(rows)


def entry_reason_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()
    return (
        episodes.groupby(["strategy", "entry_reason"], dropna=False)
        .agg(
            event_count=("episode_id", "count"),
            avg_risk_episode_duration=("duration_days", "mean"),
            avg_SPY_drawdown_at_entry=("SPY_drawdown_at_entry", "mean"),
            avg_SPY_return_during_risk_episode=("SPY_return_during_risk_episode", "mean"),
            avg_SPY_max_drawdown_during_risk_episode=("SPY_max_drawdown_during_risk_episode", "mean"),
            avg_strategy_return_during_risk_episode=("strategy_return_during_risk_episode", "mean"),
        )
        .reset_index()
    )


def plot_results(panel: pd.DataFrame, perf: pd.DataFrame, events: pd.DataFrame) -> None:
    nav_cols = {
        "SPY_BUY_HOLD": "SPY_BUY_HOLD_nav",
        "MONTHLY_EITHER_CONFIRM": "MONTHLY_EITHER_CONFIRM_nav",
        "STRESS_RECOVERY_R3_BASE": "STRESS_RECOVERY_R3_BASE_nav",
        "STRESS_RECOVERY_R3_CREDIT_DD5": "STRESS_RECOVERY_R3_CREDIT_DD5_nav",
        "STRESS_RECOVERY_R3_CREDIT_ONLY": "STRESS_RECOVERY_R3_CREDIT_ONLY_nav",
        "CASH_ONLY": "CASH_ONLY_nav",
    }
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, col in nav_cols.items():
        ax.plot(panel["date"], panel[col], label=label)
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("SPY/CASH stress-recovery with credit")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "equity_curve_log.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for label, col in nav_cols.items():
        if label == "CASH_ONLY":
            continue
        dd = panel[col] / panel[col].cummax() - 1
        ax.plot(panel["date"], dd, label=label)
    ax.legend()
    ax.set_title("Drawdown comparison")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "drawdown_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(5, 1, figsize=(14, 11), sharex=True)
    for label, col in nav_cols.items():
        if label != "CASH_ONLY":
            axes[0].plot(panel["date"], panel[col], label=label)
    axes[0].legend(fontsize=8)
    axes[1].plot(panel["date"], panel["spy_drawdown_from_previous_high"], color="red")
    axes[1].axhline(-0.05, color="black", ls="--", lw=0.8)
    axes[2].plot(panel["date"], panel["VIX_ZSCORE_120D"], color="purple")
    axes[2].axhline(CONFIG["vix_z_threshold"], color="red", ls="--", lw=0.8)
    axes[3].plot(panel["date"], panel["CREDIT_SPREAD_BAA_AAA"], label="credit spread")
    axes[3].plot(panel["date"], panel["D_CREDIT_SPREAD_20D"], label="20d change")
    axes[3].legend(fontsize=8)
    for j, (_, _, prefix) in enumerate(STRATEGY_SPECS.values()):
        axes[4].fill_between(panel["date"], j, j + 0.8, where=panel[f"{prefix}_risk_state"].eq("RISK"), alpha=0.5)
    axes[4].set_yticks([0.4, 1.4, 2.4])
    axes[4].set_yticklabels(["BASE", "CREDIT_DD5", "CREDIT_ONLY"])
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "risk_state_timeline.png", dpi=160)
    plt.close(fig)

    metrics = ["annualized_return", "sharpe_ratio", "max_drawdown", "calmar_ratio", "number_of_switches", "time_in_cash"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, metric in zip(axes.ravel(), metrics):
        sns.barplot(data=perf, x="strategy", y=metric, ax=ax)
        ax.tick_params(axis="x", rotation=70)
        ax.set_title(metric)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "performance_bar_charts.png", dpi=160)
    plt.close(fig)

    enter = events[events["event_type"].eq("ENTER_RISK")]
    if not enter.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(data=enter, y="reason", hue="strategy", ax=ax)
        ax.set_title("Entry reason counts")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "entry_reason_count_bar.png", dpi=160)
        plt.close(fig)

    for name, (start, end) in {
        "2015_2016": CASE_WINDOWS["CREDIT_COMMODITY_2015_2016"],
        "2018Q4": CASE_WINDOWS["TIGHTENING_2018Q4"],
        "2022": CASE_WINDOWS["INFLATION_2022"],
        "2023": CASE_WINDOWS["HIGH_RATE_2023"],
    }.items():
        sub = panel[panel["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        if sub.empty:
            continue
        fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
        axes[0].plot(sub["date"], sub["spy_price"] / sub["spy_price"].iloc[0], color="black", label="SPY")
        axes[0].plot(sub["date"], sub["spy_drawdown_from_previous_high"], color="red", alpha=0.7, label="SPY DD")
        axes[0].legend()
        axes[1].plot(sub["date"], sub["VIX_ZSCORE_120D"], color="purple")
        axes[1].axhline(3, color="red", ls="--", lw=0.8)
        axes[2].plot(sub["date"], sub["CREDIT_SPREAD_BAA_AAA"], label="credit")
        axes[2].plot(sub["date"], sub["D_CREDIT_SPREAD_20D"], label="20d chg")
        axes[2].legend()
        for label, col in nav_cols.items():
            if label in ["SPY_BUY_HOLD", "STRESS_RECOVERY_R3_BASE", "STRESS_RECOVERY_R3_CREDIT_DD5", "STRESS_RECOVERY_R3_CREDIT_ONLY"]:
                axes[3].plot(sub["date"], sub[col] / sub[col].iloc[0], label=label)
        axes[3].legend(fontsize=8)
        for ax in axes:
            ev = events[events["event_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
            for _, row in ev.iterrows():
                ax.axvline(pd.Timestamp(row["event_date"]), color="gray", alpha=0.25, lw=0.8)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"crisis_case_study_{name}.png", dpi=160)
        plt.close(fig)


def write_summary_md(perf: pd.DataFrame, crisis: pd.DataFrame, entry: pd.DataFrame, credit_source: str) -> None:
    lines = [
        "# SPY/CASH Stress-Recovery With Credit Summary",
        "",
        "## Purpose",
        "",
        "This SPY/CASH test keeps recovery fixed at R3 (`SPY crosses above MA20`) and tests whether credit spread triggers improve the current stress-recovery baseline.",
        "",
        "## Strategy Definitions",
        "",
        "- `STRESS_RECOVERY_R3_BASE`: FLAT + VIX z-score >= 3.0, or STEEP + Monthly Either SELL.",
        "- `STRESS_RECOVERY_R3_CREDIT_DD5`: BASE plus `SPY drawdown <= -5% and credit spread 20D change > 0.10`.",
        "- `STRESS_RECOVERY_R3_CREDIT_ONLY`: BASE plus `credit spread 20D change > 0.10`.",
        "",
        "## Credit Data",
        "",
        f"- Source: `{credit_source}`.",
        "- Credit spread is forward-filled onto SPY trading dates where needed.",
        "",
        "## Main Performance Comparison",
        "",
        perf.to_markdown(index=False),
        "",
        "## Crisis Performance",
        "",
        crisis.to_markdown(index=False),
        "",
        "## Entry Reason Summary",
        "",
        entry.to_markdown(index=False) if not entry.empty else "_No entry reason data._",
        "",
        "## Interpretation",
        "",
        "- `CREDIT_DD5` tests credit stress only after equity drawdown confirmation.",
        "- `CREDIT_ONLY` tests whether credit widening is early enough to improve exits, with higher risk of over-defensiveness.",
        "- This remains a simplified SPY/CASH state machine and should be validated inside the full regime-hedge strategy before use.",
    ]
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel, credit_source = load_base_panel()
    panel = build_credit_stress_signals(build_base_stress_signal(build_credit_features(panel)))
    panel = build_benchmarks(panel)

    event_logs = []
    for strategy, (signal_col, reason_col, prefix) in STRATEGY_SPECS.items():
        panel, ev = run_state_machine_strategy(panel, strategy, signal_col, reason_col, prefix)
        event_logs.append(ev)
    events = pd.concat(event_logs, ignore_index=True)
    episodes = extract_risk_episodes(panel, events)
    perf = compute_performance_metrics(panel, events, episodes)
    crisis = compute_crisis_performance(panel, events)
    regime = compute_regime_performance(panel)
    entry = entry_reason_summary(episodes)

    panel.to_csv(DAILY_OUT, index=False)
    events.to_csv(EVENT_LOG_OUT, index=False)
    episodes.to_csv(EPISODES_OUT, index=False)
    perf.to_csv(PERF_OUT, index=False)
    crisis.to_csv(CRISIS_OUT, index=False)
    regime.to_csv(REGIME_OUT, index=False)
    entry.to_csv(ENTRY_OUT, index=False)
    plot_results(panel, perf, events)
    write_summary_md(perf, crisis, entry, credit_source)

    base = perf[perf["strategy"].eq("STRESS_RECOVERY_R3_BASE")].iloc[0]
    dd5 = perf[perf["strategy"].eq("STRESS_RECOVERY_R3_CREDIT_DD5")].iloc[0]
    pure = perf[perf["strategy"].eq("STRESS_RECOVERY_R3_CREDIT_ONLY")].iloc[0]
    piv = crisis.pivot(index="period", columns="strategy", values="cumulative_return")
    def improves(period: str, strategy: str) -> str:
        if period not in piv.index:
            return "n/a"
        return str(bool(piv.loc[period, strategy] > piv.loc[period, "STRESS_RECOVERY_R3_BASE"]))
    print(f"1. BASE_R3 Ann/Sharpe/MaxDD/switches: {base['annualized_return']:.2%} / {base['sharpe_ratio']:.2f} / {base['max_drawdown']:.2%} / {int(base['number_of_switches'])}")
    print(f"2. CREDIT_DD5_R3 Ann/Sharpe/MaxDD/switches: {dd5['annualized_return']:.2%} / {dd5['sharpe_ratio']:.2f} / {dd5['max_drawdown']:.2%} / {int(dd5['number_of_switches'])}")
    print(f"3. CREDIT_ONLY_R3 Ann/Sharpe/MaxDD/switches: {pure['annualized_return']:.2%} / {pure['sharpe_ratio']:.2f} / {pure['max_drawdown']:.2%} / {int(pure['number_of_switches'])}")
    print(f"4. CREDIT_DD5 improves 2015-2016: {improves('CREDIT_COMMODITY_2015_2016', 'STRESS_RECOVERY_R3_CREDIT_DD5')}")
    print(f"5. CREDIT_DD5 improves 2018Q4: {improves('TIGHTENING_2018Q4', 'STRESS_RECOVERY_R3_CREDIT_DD5')}")
    print(f"6. CREDIT_DD5 improves 2022: {improves('INFLATION_2022', 'STRESS_RECOVERY_R3_CREDIT_DD5')}")
    print(f"7. CREDIT_ONLY more defensive: {bool(pure['time_in_cash'] > dd5['time_in_cash'])}; switches {int(pure['number_of_switches'])} vs {int(dd5['number_of_switches'])}")
    best_sharpe = perf.loc[perf["strategy"].isin(list(STRATEGY_SPECS)), :].sort_values("sharpe_ratio", ascending=False).iloc[0]
    best_dd = perf.loc[perf["strategy"].isin(list(STRATEGY_SPECS)), :].sort_values("max_drawdown", ascending=False).iloc[0]
    print(f"8. Highest Sharpe: {best_sharpe['strategy']}")
    print(f"9. Lowest MaxDD: {best_dd['strategy']}")
    print(f"10. Recommended next full regime-hedge test: {best_sharpe['strategy']}")
    print(f"Saved outputs: {OUTPUT_DIR} and {FIGURE_DIR}")


if __name__ == "__main__":
    main()
