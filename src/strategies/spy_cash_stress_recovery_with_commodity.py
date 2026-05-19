from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "spy_cash_stress_recovery_with_commodity"
FIGURE_DIR = ROOT / "figures" / "spy_cash_stress_recovery_with_commodity"

BASE_CANDIDATES = [
    ROOT / "results" / "spy_cash_stress_recovery_with_credit" / "daily_backtest_panel.csv",
    ROOT / "results" / "spy_cash_stress_recovery_timing" / "daily_backtest_panel.csv",
]
COMMODITY_SOURCES = [
    ROOT / "results" / "commodity_stress_trigger_diagnostic" / "commodity_trigger_daily_panel.csv",
    ROOT / "results" / "reconstructed_regime_asset_behavior" / "reconstructed_regime_panel.csv",
    ROOT / "results" / "regime_hedge_steep_sell_ief" / "daily_backtest_panel.csv",
    ROOT / "data" / "raw" / "asset" / "CMDTY_FUT.csv",
    ROOT / "data" / "raw" / "macro" / "commodity" / "CMDTY_FUT.csv",
]

CONFIG = {
    "vix_z_window": 120,
    "vix_z_threshold": 3.0,
    "credit_change_window": 20,
    "credit_change_abs_threshold": 0.10,
    "dd_threshold": -0.05,
    "commodity_ret_window": 60,
    "commodity_ret_threshold": -0.10,
    "recovery_rule": "R3_SPY_CROSS_ABOVE_MA20",
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

STRATEGY_SPECS = {
    "STRESS_RECOVERY_R3_BASE": ("base_stress_entry_signal", "base_stress_entry_reason", "BASE_R3"),
    "STRESS_RECOVERY_R3_CREDIT_DD5": ("credit_dd5_stress_entry_signal", "credit_dd5_stress_entry_reason", "CREDIT_DD5_R3"),
    "STRESS_RECOVERY_R3_CMDTY_GROWTH": ("cmdty_growth_stress_entry_signal", "cmdty_growth_stress_entry_reason", "CMDTY_GROWTH_R3"),
    "STRESS_RECOVERY_R3_CREDIT_AND_CMDTY": ("credit_and_cmdty_stress_entry_signal", "credit_and_cmdty_stress_entry_reason", "CREDIT_AND_CMDTY_R3"),
}

DAILY_OUT = OUTPUT_DIR / "daily_backtest_panel.csv"
EVENT_LOG_OUT = OUTPUT_DIR / "risk_state_event_log.csv"
EPISODES_OUT = OUTPUT_DIR / "risk_episodes.csv"
PERF_OUT = OUTPUT_DIR / "performance_summary.csv"
CRISIS_OUT = OUTPUT_DIR / "crisis_performance.csv"
REGIME_OUT = OUTPUT_DIR / "performance_by_regime.csv"
ENTRY_OUT = OUTPUT_DIR / "entry_reason_summary.csv"
REPORT_OUT = OUTPUT_DIR / "SPY_CASH_STRESS_RECOVERY_WITH_COMMODITY_SUMMARY.md"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        warnings.warn(f"Could not read {path}: {exc}")
        return None
    date_col = "date" if "date" in df.columns else "observation_date" if "observation_date" in df.columns else "DATE" if "DATE" in df.columns else None
    if date_col is None:
        return None
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date")


def _first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def load_base_panel() -> pd.DataFrame:
    for path in BASE_CANDIDATES:
        df = _read_csv(path)
        if df is None:
            continue
        required = ["spy_price", "spy_daily_return", "daily_rf", "macro_regime_confirmed", "monthly_either_state", "VIX_LEVEL"]
        if all(c in df.columns for c in required):
            return df.sort_values("date").reset_index(drop=True)
    raise FileNotFoundError("Could not load a usable SPY/CASH stress-recovery panel.")


def load_commodity_data() -> tuple[pd.DataFrame, str]:
    for path in COMMODITY_SOURCES:
        df = _read_csv(path)
        if df is None:
            continue
        price_col = _first_col(df, ["CMDTY_FUT_price", "CMDTY_FUT", "CMDTY", "commodity_price", "close", "Adj Close"])
        ret_col = _first_col(df, ["CMDTY_FUT_return", "CMDTY_FUT_RETURN", "CMDTY_ret", "commodity_return"])
        if price_col is None and ret_col is None:
            continue
        out = df[["date"]].copy()
        bits = []
        if price_col:
            out["CMDTY_FUT_price"] = pd.to_numeric(df[price_col], errors="coerce")
            bits.append(price_col)
        if ret_col:
            out["CMDTY_FUT_return"] = pd.to_numeric(df[ret_col], errors="coerce")
            bits.append(ret_col)
        return out, f"{path.relative_to(ROOT)}:{'/'.join(bits)}"
    raise FileNotFoundError("Could not locate commodity data.")


def build_commodity_features(panel: pd.DataFrame) -> pd.DataFrame:
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
    if "D_CREDIT_SPREAD_20D" not in out.columns:
        out["D_CREDIT_SPREAD_20D"] = pd.to_numeric(out["CREDIT_SPREAD_BAA_AAA"], errors="coerce").diff(CONFIG["credit_change_window"])

    commodity, source = load_commodity_data()
    out = out.merge(commodity, on="date", how="left")
    out["CMDTY_FUT_return"] = pd.to_numeric(out.get("CMDTY_FUT_return"), errors="coerce")
    if "CMDTY_FUT_price" not in out.columns or out["CMDTY_FUT_price"].isna().all():
        out["CMDTY_FUT_price"] = (1.0 + out["CMDTY_FUT_return"].fillna(0.0)).cumprod()
    else:
        out["CMDTY_FUT_price"] = pd.to_numeric(out["CMDTY_FUT_price"], errors="coerce")
        out["CMDTY_FUT_return"] = out["CMDTY_FUT_return"].combine_first(out["CMDTY_FUT_price"].pct_change())
    out["CMDTY_RET60"] = out["CMDTY_FUT_price"] / out["CMDTY_FUT_price"].shift(CONFIG["commodity_ret_window"]) - 1.0
    out["CMDTY_MA60"] = out["CMDTY_FUT_price"].rolling(60, min_periods=60).mean()
    out["CMDTY_DRAWDOWN_FROM_HIGH"] = out["CMDTY_FUT_price"] / out["CMDTY_FUT_price"].cummax() - 1.0
    out.attrs["commodity_source"] = source
    return out.dropna(subset=["spy_price", "spy_daily_return", "daily_rf"]).reset_index(drop=True)


def _combine_reasons(items: list[tuple[pd.Series, str]]) -> pd.Series:
    reasons = pd.Series("", index=items[0][0].index)
    for mask, reason in items:
        reasons = np.where(mask & (reasons != ""), reasons + "+" + reason, np.where(mask, reason, reasons))
        reasons = pd.Series(reasons, index=items[0][0].index)
    return reasons


def build_base_stress_signal(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["BASE_STRESS"] = (out["macro_regime_confirmed"].eq("FLAT") & (out["VIX_ZSCORE_120D"] >= CONFIG["vix_z_threshold"])) | (
        out["macro_regime_confirmed"].eq("STEEP") & out["monthly_either_state"].eq("SELL")
    )
    flat_vix = out["macro_regime_confirmed"].eq("FLAT") & (out["VIX_ZSCORE_120D"] >= CONFIG["vix_z_threshold"])
    steep_sell = out["macro_regime_confirmed"].eq("STEEP") & out["monthly_either_state"].eq("SELL")
    out["base_stress_entry_signal"] = out["BASE_STRESS"]
    out["base_stress_entry_reason"] = _combine_reasons([(flat_vix, "FLAT_VIX_STRESS"), (steep_sell, "STEEP_EITHER_SELL")])
    return out


def build_credit_stress_signal(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["CREDIT_DD5_STRESS"] = (out["spy_drawdown_from_previous_high"] <= CONFIG["dd_threshold"]) & (
        out["D_CREDIT_SPREAD_20D"] > CONFIG["credit_change_abs_threshold"]
    )
    out["credit_dd5_stress_entry_signal"] = out["BASE_STRESS"] | out["CREDIT_DD5_STRESS"]
    out["credit_dd5_stress_entry_reason"] = _combine_reasons(
        [(out["BASE_STRESS"], out["base_stress_entry_reason"]), (out["CREDIT_DD5_STRESS"], "DD5_AND_CREDIT_CHG20_GT_0_10")]
    )
    return out


def build_commodity_stress_signal(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["CMDTY_GROWTH_STRESS"] = (
        (out["spy_drawdown_from_previous_high"] <= CONFIG["dd_threshold"])
        & (out["CMDTY_RET60"] < CONFIG["commodity_ret_threshold"])
        & (out["D_CREDIT_SPREAD_20D"] > 0)
    )
    out["cmdty_growth_stress_entry_signal"] = out["BASE_STRESS"] | out["CMDTY_GROWTH_STRESS"]
    out["credit_and_cmdty_stress_entry_signal"] = out["BASE_STRESS"] | out["CREDIT_DD5_STRESS"] | out["CMDTY_GROWTH_STRESS"]
    out["cmdty_growth_stress_entry_reason"] = _combine_reasons(
        [(out["BASE_STRESS"], out["base_stress_entry_reason"]), (out["CMDTY_GROWTH_STRESS"], "DD5_CMDTY_RET60_NEG10_AND_CREDIT_WIDEN")]
    )
    out["credit_and_cmdty_stress_entry_reason"] = _combine_reasons(
        [
            (out["BASE_STRESS"], out["base_stress_entry_reason"]),
            (out["CREDIT_DD5_STRESS"], "DD5_AND_CREDIT_CHG20_GT_0_10"),
            (out["CMDTY_GROWTH_STRESS"], "DD5_CMDTY_RET60_NEG10_AND_CREDIT_WIDEN"),
        ]
    )
    return out


def run_state_machine_strategy(panel: pd.DataFrame, strategy: str, signal_col: str, reason_col: str, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = panel.copy()
    state_col, w_col, cash_col = f"{prefix}_risk_state", f"{prefix}_weight_spy", f"{prefix}_weight_cash"
    ret_col, nav_col, cost_col, turnover_col = f"{strategy}_return", f"{strategy}_nav", f"transaction_cost_{prefix}", f"{strategy}_turnover"
    out[state_col], out[w_col], out[cash_col], out[ret_col], out[nav_col] = "", np.nan, np.nan, np.nan, np.nan
    out[cost_col], out[turnover_col] = 0.0, 0.0
    state, pending_state, pending_reason, nav = "NORMAL", "NORMAL", "", 1.0
    events = []
    cost_rate = CONFIG["one_way_cost_bps"] / 10000.0
    for i, row in out.iterrows():
        old_state = state
        if i > 0 and pending_state != state:
            state = pending_state
            old_w, new_w = (1.0 if old_state == "NORMAL" else 0.0), (1.0 if state == "NORMAL" else 0.0)
            turnover = abs(new_w - old_w) + abs((1 - new_w) - (1 - old_w))
            out.loc[i, turnover_col] = turnover
            out.loc[i, cost_col] = 0.5 * turnover * cost_rate
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
                    "CREDIT_SPREAD_BAA_AAA": sig.get("CREDIT_SPREAD_BAA_AAA", np.nan),
                    "D_CREDIT_SPREAD_20D": sig.get("D_CREDIT_SPREAD_20D", np.nan),
                    "CMDTY_RET60": sig["CMDTY_RET60"],
                    "CMDTY_DRAWDOWN_FROM_HIGH": sig["CMDTY_DRAWDOWN_FROM_HIGH"],
                    "spy_drawdown_from_previous_high": sig["spy_drawdown_from_previous_high"],
                    "SPY_price": sig["spy_price"],
                    "SPY_MA20": sig["SPY_MA20"],
                    "previous_state": old_state,
                    "new_state": state,
                }
            )
        w_spy = 1.0 if state == "NORMAL" else 0.0
        daily_ret = w_spy * row["spy_daily_return"] + (1.0 - w_spy) * row["daily_rf"] - out.loc[i, cost_col]
        nav *= 1.0 + float(daily_ret)
        out.loc[i, state_col], out.loc[i, w_col], out.loc[i, cash_col] = state, w_spy, 1.0 - w_spy
        out.loc[i, ret_col], out.loc[i, nav_col] = daily_ret, nav
        pending_state, pending_reason = state, ""
        if state == "NORMAL" and bool(row[signal_col]):
            pending_state, pending_reason = "RISK", row[reason_col]
        elif state == "RISK" and bool(row["SPY_CROSS_ABOVE_MA20"]):
            pending_state, pending_reason = "NORMAL", "R3_SPY_CROSS_ABOVE_MA20"
    return out, pd.DataFrame(events)


def build_benchmarks(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["SPY_BUY_HOLD_return"], out["SPY_BUY_HOLD_nav"] = out["spy_daily_return"], (1.0 + out["spy_daily_return"]).cumprod()
    out["CASH_ONLY_return"], out["CASH_ONLY_nav"] = out["daily_rf"], (1.0 + out["daily_rf"]).cumprod()
    if "MONTHLY_EITHER_CONFIRM_return" not in out.columns or "MONTHLY_EITHER_CONFIRM_nav" not in out.columns:
        w = pd.to_numeric(out.get("monthly_either_weight_spy", pd.Series(1.0, index=out.index)), errors="coerce").fillna(1.0)
        prev = w.shift(1).fillna(w.iloc[0])
        turnover = (w - prev).abs() + ((1 - w) - (1 - prev)).abs()
        out["MONTHLY_EITHER_CONFIRM_turnover"] = turnover
        out["transaction_cost_MONTHLY_EITHER_CONFIRM"] = 0.5 * turnover * CONFIG["one_way_cost_bps"] / 10000.0
        out["MONTHLY_EITHER_CONFIRM_return"] = w * out["spy_daily_return"] + (1 - w) * out["daily_rf"] - out["transaction_cost_MONTHLY_EITHER_CONFIRM"]
        out["MONTHLY_EITHER_CONFIRM_nav"] = (1.0 + out["MONTHLY_EITHER_CONFIRM_return"]).cumprod()
    return out


def extract_risk_episodes(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, (_, _, prefix) in STRATEGY_SPECS.items():
        risk = panel[f"{prefix}_risk_state"].eq("RISK").to_numpy()
        eid, i = 0, 0
        while i < len(panel):
            if not risk[i]:
                i += 1
                continue
            start = i
            while i + 1 < len(panel) and risk[i + 1]:
                i += 1
            end, eid = i, eid + 1
            sub = panel.iloc[start : end + 1]
            entry = events[(events["strategy"].eq(strategy)) & (events["event_type"].eq("ENTER_RISK")) & (events["event_date"].eq(sub["date"].iloc[0]))]
            exit_ev = events[(events["strategy"].eq(strategy)) & (events["event_type"].eq("EXIT_RISK")) & (events["event_date"] > sub["date"].iloc[-1])]
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
                    "CREDIT_SPREAD_at_entry": sub.get("CREDIT_SPREAD_BAA_AAA", pd.Series(np.nan, index=sub.index)).iloc[0],
                    "D_CREDIT_SPREAD_20D_at_entry": sub.get("D_CREDIT_SPREAD_20D", pd.Series(np.nan, index=sub.index)).iloc[0],
                    "CMDTY_RET60_at_entry": sub["CMDTY_RET60"].iloc[0],
                    "CMDTY_DRAWDOWN_at_entry": sub["CMDTY_DRAWDOWN_FROM_HIGH"].iloc[0],
                    "SPY_return_during_risk_episode": (1.0 + sub["spy_daily_return"]).prod() - 1.0,
                    "CASH_return_during_risk_episode": (1.0 + sub["daily_rf"]).prod() - 1.0,
                    "strategy_return_during_risk_episode": (1.0 + sub[f"{strategy}_return"]).prod() - 1.0,
                    "SPY_max_drawdown_during_risk_episode": (spy_wealth / spy_wealth.cummax() - 1.0).min(),
                    "SPY_max_runup_during_risk_episode": (sub["spy_price"] / sub["spy_price"].iloc[0] - 1.0).max(),
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
        return {"n_obs": 0, "annualized_return": np.nan, "volatility": np.nan, "Sharpe": np.nan, "max_drawdown": np.nan, "time_in_cash": np.nan}
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
        cash = 0
    elif strategy == "CASH_ONLY":
        cash = 1
    elif strategy == "MONTHLY_EITHER_CONFIRM":
        cash = 1 - sub.get("monthly_either_weight_spy", pd.Series(1.0, index=sub.index)).mean()
    else:
        cash = np.nan
    return {"n_obs": len(s), "annualized_return": ann, "volatility": vol, "Sharpe": sharpe, "max_drawdown": mdd, "time_in_cash": cash}


def compute_crisis_performance(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mapping = [("SPY_BUY_HOLD", "SPY_BUY_HOLD_return", None), ("CASH_ONLY", "CASH_ONLY_return", None), ("MONTHLY_EITHER_CONFIRM", "MONTHLY_EITHER_CONFIRM_return", None)]
    mapping += [(s, f"{s}_return", f"{p}_weight_spy") for s, (_, _, p) in STRATEGY_SPECS.items()]
    for period, (start, end) in CASE_WINDOWS.items():
        sub = panel[panel["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        if sub.empty:
            continue
        for strategy, ret_col, weight_col in mapping:
            sw = events[events["strategy"].eq(strategy) & events["event_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
            rows.append({"period": period, "strategy": strategy, "cumulative_return": (1 + sub[ret_col]).prod() - 1, "number_of_switches": len(sw), **_group_stats(sub, strategy, ret_col, weight_col)})
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
        "STRESS_RECOVERY_R3_CMDTY_GROWTH": "STRESS_RECOVERY_R3_CMDTY_GROWTH_nav",
        "STRESS_RECOVERY_R3_CREDIT_AND_CMDTY": "STRESS_RECOVERY_R3_CREDIT_AND_CMDTY_nav",
        "CASH_ONLY": "CASH_ONLY_nav",
    }
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, col in nav_cols.items():
        ax.plot(panel["date"], panel[col], label=label)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "equity_curve_log.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for label, col in nav_cols.items():
        if label != "CASH_ONLY":
            ax.plot(panel["date"], panel[col] / panel[col].cummax() - 1, label=label)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "drawdown_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True)
    for label, col in nav_cols.items():
        if label != "CASH_ONLY":
            axes[0].plot(panel["date"], panel[col], label=label)
    axes[0].legend(fontsize=7)
    axes[1].plot(panel["date"], panel["spy_drawdown_from_previous_high"], color="red")
    axes[1].axhline(-0.05, color="black", ls="--", lw=0.8)
    axes[2].plot(panel["date"], panel["VIX_ZSCORE_120D"], color="purple")
    axes[2].axhline(3, color="red", ls="--", lw=0.8)
    axes[3].plot(panel["date"], panel["CREDIT_SPREAD_BAA_AAA"], label="credit")
    axes[3].plot(panel["date"], panel["D_CREDIT_SPREAD_20D"], label="20d chg")
    axes[3].legend(fontsize=7)
    axes[4].plot(panel["date"], panel["CMDTY_RET60"], label="CMDTY_RET60")
    axes[4].axhline(-0.10, color="red", ls="--", lw=0.8)
    axes[4].legend(fontsize=7)
    for j, (_, _, prefix) in enumerate(STRATEGY_SPECS.values()):
        axes[5].fill_between(panel["date"], j, j + 0.8, where=panel[f"{prefix}_risk_state"].eq("RISK"), alpha=0.45)
    axes[5].set_yticks([0.4, 1.4, 2.4, 3.4])
    axes[5].set_yticklabels(["BASE", "CREDIT", "CMDTY", "BOTH"])
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "risk_state_timeline.png", dpi=160)
    plt.close(fig)

    metrics = ["annualized_return", "sharpe_ratio", "max_drawdown", "calmar_ratio", "number_of_switches", "time_in_cash"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, metric in zip(axes.ravel(), metrics):
        sns.barplot(data=perf, x="strategy", y=metric, ax=ax)
        ax.tick_params(axis="x", rotation=70)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "performance_bar_charts.png", dpi=160)
    plt.close(fig)

    enter = events[events["event_type"].eq("ENTER_RISK")]
    if not enter.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(data=enter, y="reason", hue="strategy", ax=ax)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "entry_reason_count_bar.png", dpi=160)
        plt.close(fig)

    for name, key in [("2015_2016", "CREDIT_COMMODITY_2015_2016"), ("2018Q4", "TIGHTENING_2018Q4"), ("2020_COVID", "COVID_2020"), ("2022", "INFLATION_2022"), ("2023", "HIGH_RATE_2023")]:
        start, end = CASE_WINDOWS[key]
        sub = panel[panel["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        if sub.empty:
            continue
        fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
        axes[0].plot(sub["date"], sub["spy_price"] / sub["spy_price"].iloc[0], label="SPY", color="black")
        axes[0].plot(sub["date"], sub["spy_drawdown_from_previous_high"], label="SPY DD", color="red", alpha=0.7)
        axes[0].legend(fontsize=8)
        axes[1].plot(sub["date"], sub["CMDTY_FUT_price"] / sub["CMDTY_FUT_price"].iloc[0], label="CMDTY")
        axes[1].plot(sub["date"], sub["CMDTY_RET60"], label="CMDTY_RET60")
        axes[1].legend(fontsize=8)
        axes[2].plot(sub["date"], sub["D_CREDIT_SPREAD_20D"], label="credit 20d")
        axes[2].plot(sub["date"], sub["VIX_ZSCORE_120D"], label="VIX z")
        axes[2].legend(fontsize=8)
        for label, col in nav_cols.items():
            if label in ["SPY_BUY_HOLD", "STRESS_RECOVERY_R3_BASE", "STRESS_RECOVERY_R3_CREDIT_DD5", "STRESS_RECOVERY_R3_CMDTY_GROWTH", "STRESS_RECOVERY_R3_CREDIT_AND_CMDTY"]:
                axes[3].plot(sub["date"], sub[col] / sub[col].iloc[0], label=label)
        axes[3].legend(fontsize=7)
        ev = events[events["event_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        for ax in axes:
            for _, row in ev.iterrows():
                ax.axvline(pd.Timestamp(row["event_date"]), color="gray", alpha=0.2, lw=0.8)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"crisis_case_study_{name}.png", dpi=160)
        plt.close(fig)


def write_summary_md(perf: pd.DataFrame, crisis: pd.DataFrame, entry: pd.DataFrame, source: str) -> None:
    lines = [
        "# SPY/CASH Stress-Recovery With Commodity Summary",
        "",
        "## Purpose",
        "",
        "This SPY/CASH test fixes recovery at R3 (`SPY crosses above MA20`) and tests whether commodity/global-growth stress improves the existing VIX, Monthly Either, and credit-trigger framework.",
        "",
        "## Strategy Definitions",
        "",
        "- `BASE`: FLAT + VIX z-score >= 3.0, or STEEP + Monthly Either SELL.",
        "- `CREDIT_DD5`: BASE plus `SPY drawdown <= -5% and credit spread 20D change > 0.10`.",
        "- `CMDTY_GROWTH`: BASE plus `SPY drawdown <= -5%, CMDTY_RET60 < -10%, and credit spread 20D change > 0`.",
        "- `CREDIT_AND_CMDTY`: BASE plus both credit and commodity-growth triggers.",
        "",
        "## Commodity Data",
        "",
        f"- Source: `{source}`.",
        "- Commodity weakness is treated as a global growth / demand stress proxy, not as a hedge asset.",
        "",
        "## Main Performance",
        "",
        perf.to_markdown(index=False),
        "",
        "## Crisis Performance",
        "",
        crisis.to_markdown(index=False),
        "",
        "## Entry Reason Summary",
        "",
        entry.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- The commodity trigger should be judged by crisis-specific improvement and total false-defense cost.",
        "- If it only helps 2015-2016 but damages COVID or recent bull markets, it should stay diagnostic.",
        "- This remains a SPY/CASH simplification; hedge sleeves are not tested here.",
    ]
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = build_benchmarks(build_commodity_stress_signal(build_credit_stress_signal(build_base_stress_signal(build_commodity_features(load_base_panel())))))
    source = panel.attrs.get("commodity_source", "unknown")
    events_list = []
    for strategy, (signal_col, reason_col, prefix) in STRATEGY_SPECS.items():
        panel, ev = run_state_machine_strategy(panel, strategy, signal_col, reason_col, prefix)
        events_list.append(ev)
    events = pd.concat(events_list, ignore_index=True)
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
    write_summary_md(perf, crisis, entry, source)

    def row(strategy: str) -> pd.Series:
        return perf[perf["strategy"].eq(strategy)].iloc[0]
    base, credit = row("STRESS_RECOVERY_R3_BASE"), row("STRESS_RECOVERY_R3_CREDIT_DD5")
    cmdty, both = row("STRESS_RECOVERY_R3_CMDTY_GROWTH"), row("STRESS_RECOVERY_R3_CREDIT_AND_CMDTY")
    piv = crisis.pivot(index="period", columns="strategy", values="cumulative_return")
    def improves(period: str, strategy: str) -> bool | str:
        if period not in piv.index:
            return "n/a"
        return bool(piv.loc[period, strategy] > piv.loc[period, "STRESS_RECOVERY_R3_BASE"])
    for i, (name, r) in enumerate([("BASE_R3", base), ("CREDIT_DD5_R3", credit), ("CMDTY_GROWTH_R3", cmdty), ("CREDIT_AND_CMDTY_R3", both)], 1):
        print(f"{i}. {name} Ann/Sharpe/MaxDD/switches: {r['annualized_return']:.2%} / {r['sharpe_ratio']:.2f} / {r['max_drawdown']:.2%} / {int(r['number_of_switches'])}")
    print(f"5. CMDTY_GROWTH improves 2015-2016: {improves('CREDIT_COMMODITY_2015_2016', 'STRESS_RECOVERY_R3_CMDTY_GROWTH')}")
    print(f"6. CMDTY_GROWTH improves 2018Q4: {improves('TIGHTENING_2018Q4', 'STRESS_RECOVERY_R3_CMDTY_GROWTH')}")
    print(f"7. CMDTY_GROWTH hurts COVID: {not improves('COVID_2020', 'STRESS_RECOVERY_R3_CMDTY_GROWTH')}")
    print(f"8. CMDTY_GROWTH improves 2022 / 2023: {improves('INFLATION_2022', 'STRESS_RECOVERY_R3_CMDTY_GROWTH')} / {improves('HIGH_RATE_2023', 'STRESS_RECOVERY_R3_CMDTY_GROWTH')}")
    strat_perf = perf[perf["strategy"].isin(STRATEGY_SPECS)]
    best_sharpe = strat_perf.sort_values("sharpe_ratio", ascending=False).iloc[0]["strategy"]
    best_dd = strat_perf.sort_values("max_drawdown", ascending=False).iloc[0]["strategy"]
    print(f"9. Highest Sharpe: {best_sharpe}")
    print(f"10. Lowest MaxDD: {best_dd}")
    print(f"11. Recommended next full regime-hedge test: {best_sharpe}")
    print(f"Saved outputs: {OUTPUT_DIR} and {FIGURE_DIR}")


if __name__ == "__main__":
    main()
