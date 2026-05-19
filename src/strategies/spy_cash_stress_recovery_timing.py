from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]

INPUT_CANDIDATES = [
    ROOT / "results" / "spy_cash_timing_frequency_test" / "daily_backtest_panel.csv",
    ROOT / "results" / "regime_labeled_sell_lag_diagnostic" / "regime_labeled_daily_panel.csv",
    ROOT / "results" / "high_frequency_regime_diagnostics" / "high_frequency_regime_feature_panel.csv",
    ROOT / "results" / "regime_hedge_steep_sell_ief" / "daily_backtest_panel.csv",
]
RECON_PANEL = ROOT / "results" / "reconstructed_regime_asset_behavior" / "reconstructed_regime_panel.csv"
RULE_PANEL = ROOT / "results" / "rule_diagnostics" / "rule_state_panel.csv"
RAW_VIX = ROOT / "data" / "raw" / "macro" / "volatility" / "VIXCLS.csv"
DTB3_PATH = ROOT / "data" / "raw" / "rate" / "DTB3.csv"
TEST_STEEP_ONLY_PATH = ROOT / "results" / "steep_only_sell_test" / "steep_only_sell_daily_panel.csv"

RESULTS_DIR = ROOT / "results" / "spy_cash_stress_recovery_timing"
FIGURES_DIR = ROOT / "figures" / "spy_cash_stress_recovery_timing"

DAILY_OUT = RESULTS_DIR / "daily_backtest_panel.csv"
EVENT_LOG_OUT = RESULTS_DIR / "risk_state_event_log.csv"
RISK_EPISODES_OUT = RESULTS_DIR / "risk_episodes.csv"
PERF_OUT = RESULTS_DIR / "performance_summary.csv"
CRISIS_OUT = RESULTS_DIR / "crisis_performance.csv"
REGIME_OUT = RESULTS_DIR / "performance_by_regime.csv"
ENTRY_REASON_OUT = RESULTS_DIR / "performance_by_stress_entry_reason.csv"
SUMMARY_OUT = RESULTS_DIR / "SPY_CASH_STRESS_RECOVERY_TIMING_SUMMARY.md"

FIG_EQUITY = FIGURES_DIR / "equity_curve_log.png"
FIG_DD = FIGURES_DIR / "drawdown_comparison.png"
FIG_STATE = FIGURES_DIR / "risk_state_timeline.png"
FIG_COVID = FIGURES_DIR / "covid_case_study.png"
FIG_CRISIS = FIGURES_DIR / "crisis_case_study_2008_2020_2022.png"
FIG_BAR = FIGURES_DIR / "performance_bar_charts.png"
FIG_HIST = FIGURES_DIR / "risk_episode_duration_histogram.png"

CONFIG = {
    "vix_z_window": 120,
    "vix_z_threshold": 3.0,
    "enabled_vix_regimes": ["FLAT"],
    "enabled_either_sell_regimes": ["STEEP"],
    "recovery_rules": ["R8", "R3"],
    "one_way_cost_bps": 5.0,
    "initial_nav": 1.0,
    "output_dir": str(RESULTS_DIR),
    "figure_dir": str(FIGURES_DIR),
}

STRATEGIES = [
    "SPY_BUY_HOLD",
    "CASH_ONLY",
    "MONTHLY_EITHER_CONFIRM",
    "STRESS_RECOVERY_R8",
    "STRESS_RECOVERY_R3",
]
CASE_WINDOWS = {
    "GFC_2008_2009": ("2008-09-01", "2009-03-31"),
    "COVID_2020": ("2020-02-19", "2020-04-30"),
    "INFLATION_2022": ("2022-01-01", "2022-12-31"),
    "RECENT_2024_2026": ("2024-01-01", "2026-12-31"),
}
REGIME_COLORS = {
    "HIGH_INFLATION": "#d95f02",
    "INVERTED": "#7570b3",
    "FLAT": "#1b9e77",
    "STEEP": "#66a61e",
    "NEUTRAL": "#999999",
}


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _read_panel(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        warnings.warn(f"Could not read {path}: {exc}")
        return None
    if "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date")


def _first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _load_raw_vix() -> pd.DataFrame | None:
    if not RAW_VIX.exists():
        return None
    df = pd.read_csv(RAW_VIX)
    date_col = _first_col(df, ["DATE", "date", "observation_date"])
    vix_col = _first_col(df, ["VIXCLS", "VIX_LEVEL"])
    if date_col is None or vix_col is None:
        return None
    out = df[[date_col, vix_col]].rename(columns={date_col: "date", vix_col: "VIX_LEVEL"})
    out["date"] = pd.to_datetime(out["date"])
    out["VIX_LEVEL"] = pd.to_numeric(out["VIX_LEVEL"].replace(".", np.nan), errors="coerce")
    return out.dropna(subset=["date"]).sort_values("date")


def _load_rf() -> pd.DataFrame | None:
    if not DTB3_PATH.exists():
        return None
    rf = pd.read_csv(DTB3_PATH)
    date_col = _first_col(rf, ["DATE", "date"])
    val_col = _first_col(rf, ["DTB3"])
    if date_col is None or val_col is None:
        return None
    out = rf[[date_col, val_col]].rename(columns={date_col: "date", val_col: "DTB3"})
    out["date"] = pd.to_datetime(out["date"])
    out["DTB3"] = pd.to_numeric(out["DTB3"].replace(".", np.nan), errors="coerce")
    out["daily_rf"] = (1.0 + out["DTB3"].ffill() / 100.0) ** (1.0 / 252.0) - 1.0
    return out[["date", "daily_rf"]].sort_values("date")


def _rebuild_macro_regime(df: pd.DataFrame) -> pd.Series:
    credit = pd.to_numeric(df.get("CREDIT_SPREAD_BAA_AAA"), errors="coerce")
    dgs1 = pd.to_numeric(df.get("DGS1"), errors="coerce")
    term = pd.to_numeric(df.get("TERM_SPREAD_10Y_1Y"), errors="coerce")
    return pd.Series(
        np.select(
            [(credit > 1.5) & (dgs1 > 5.0), term < 0.0, (term >= 0.0) & (term < 1.0), term >= 1.0],
            ["HIGH_INFLATION", "INVERTED", "FLAT", "STEEP"],
            default="NEUTRAL",
        ),
        index=df.index,
    )


def build_monthly_either_if_needed(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    if "monthly_either_state" in out.columns and "monthly_either_weight_spy" in out.columns:
        return out
    if "monthly_either_weight_spy" in out.columns:
        out["monthly_either_state"] = np.where(out["monthly_either_weight_spy"] >= 0.5, "HOLD", "SELL")
        return out

    month_end = out.groupby(out["date"].dt.to_period("M")).tail(1).copy()
    month_end["faber_10m_sma"] = month_end["spy_price"].rolling(10, min_periods=10).mean()
    month_end["faber_signal"] = (month_end["spy_price"] > month_end["faber_10m_sma"]).astype(float)
    cash_nav = (1.0 + out["daily_rf"].fillna(0.0)).cumprod()
    out["_cash_nav_tmp"] = cash_nav
    month_end = month_end.merge(out[["date", "_cash_nav_tmp"]], on="date", how="left")
    month_end["spy_12m_return"] = month_end["spy_price"] / month_end["spy_price"].shift(12) - 1.0
    month_end["cash_12m_return"] = month_end["_cash_nav_tmp"] / month_end["_cash_nav_tmp"].shift(12) - 1.0
    month_end["antonacci_signal"] = (month_end["spy_12m_return"] - month_end["cash_12m_return"] > 0).astype(float)
    month_end["monthly_either_weight_spy_signal"] = ((month_end["faber_signal"] == 1) | (month_end["antonacci_signal"] == 1)).astype(float)
    signal = month_end[["date", "monthly_either_weight_spy_signal"]].copy()
    signal["effective_date"] = signal["date"].shift(-1)
    out = out.drop(columns=["_cash_nav_tmp"], errors="ignore")
    out = out.merge(signal[["date", "monthly_either_weight_spy_signal"]], on="date", how="left")
    out["monthly_either_weight_spy"] = out["monthly_either_weight_spy_signal"].ffill().shift(1)
    out["monthly_either_weight_spy"] = out["monthly_either_weight_spy"].fillna(1.0)
    out["monthly_either_state"] = np.where(out["monthly_either_weight_spy"] >= 0.5, "HOLD", "SELL")
    return out.drop(columns=["monthly_either_weight_spy_signal"], errors="ignore")


def load_data() -> pd.DataFrame:
    panel = None
    for path in INPUT_CANDIDATES:
        df = _read_panel(path)
        if df is None or df.empty:
            continue
        if _first_col(df, ["spy_price", "SPY_PRICE"]) is not None or _first_col(df, ["SPY_RETURN", "SPY_ret", "spy_daily_return"]) is not None:
            panel = df
            break
    if panel is None:
        raise FileNotFoundError("No usable daily SPY panel found.")

    price_col = _first_col(panel, ["spy_price", "SPY_PRICE", "SPY"])
    ret_col = _first_col(panel, ["SPY_RETURN", "SPY_ret", "spy_daily_return", "SPY_BUY_HOLD_return"])
    if price_col is not None:
        panel["spy_price"] = pd.to_numeric(panel[price_col], errors="coerce")
        panel["spy_daily_return"] = panel["spy_price"].pct_change()
        if ret_col is not None:
            panel["spy_daily_return"] = pd.to_numeric(panel[ret_col], errors="coerce").combine_first(panel["spy_daily_return"])
    elif ret_col is not None:
        panel["spy_daily_return"] = pd.to_numeric(panel[ret_col], errors="coerce")
        panel["spy_price"] = (1.0 + panel["spy_daily_return"].fillna(0.0)).cumprod()
    else:
        raise ValueError("Missing SPY price or return.")

    rf_col = _first_col(panel, ["daily_rf", "RF_DAILY", "CASH_RETURN"])
    if rf_col is not None:
        panel["daily_rf"] = pd.to_numeric(panel[rf_col], errors="coerce")
    else:
        rf = _load_rf()
        if rf is None:
            raise ValueError("Missing daily_rf/RF_DAILY and could not load DTB3.csv.")
        panel = pd.merge_asof(panel.sort_values("date"), rf.sort_values("date"), on="date", direction="backward")

    vix_col = _first_col(panel, ["VIX_LEVEL", "VIXCLS"])
    if vix_col is not None:
        panel["VIX_LEVEL"] = pd.to_numeric(panel[vix_col], errors="coerce")
    else:
        vix_sources = []
        for path in [RECON_PANEL, RULE_PANEL]:
            vix_panel = _read_panel(path)
            if vix_panel is not None and "VIX_LEVEL" in vix_panel.columns:
                vix_sources.append(vix_panel[["date", "VIX_LEVEL"]])
        raw = _load_raw_vix()
        if raw is not None:
            vix_sources.append(raw)
        if not vix_sources:
            raise ValueError("Missing VIX_LEVEL and could not load VIXCLS.csv.")
        vix = pd.concat(vix_sources, ignore_index=True).sort_values("date").drop_duplicates("date")
        panel = panel.merge(vix, on="date", how="left")

    if "macro_regime_confirmed" not in panel.columns:
        for path in [RECON_PANEL, RULE_PANEL]:
            reg = _read_panel(path)
            if reg is None:
                continue
            keep = ["date"] + [c for c in ["macro_regime_confirmed", "CREDIT_SPREAD_BAA_AAA", "DGS1", "TERM_SPREAD_10Y_1Y"] if c in reg.columns]
            reg = reg[keep].copy()
            if "macro_regime_confirmed" not in reg.columns and {"CREDIT_SPREAD_BAA_AAA", "DGS1", "TERM_SPREAD_10Y_1Y"}.issubset(reg.columns):
                reg["macro_regime_confirmed"] = _rebuild_macro_regime(reg)
            if "macro_regime_confirmed" in reg.columns:
                panel = panel.merge(reg[["date", "macro_regime_confirmed"]], on="date", how="left")
                break
    if "macro_regime_confirmed" not in panel.columns:
        raise ValueError("Missing macro_regime_confirmed and could not rebuild it.")

    panel = build_monthly_either_if_needed(panel)
    panel = panel.dropna(subset=["spy_price", "spy_daily_return", "daily_rf", "VIX_LEVEL", "macro_regime_confirmed"]).sort_values("date").reset_index(drop=True)
    return panel


def build_vix_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    w = CONFIG["vix_z_window"]
    out["VIX_MA120"] = out["VIX_LEVEL"].rolling(w, min_periods=w).mean()
    out["VIX_STD120"] = out["VIX_LEVEL"].rolling(w, min_periods=w).std(ddof=1)
    out["VIX_ZSCORE_120D"] = (out["VIX_LEVEL"] - out["VIX_MA120"]) / out["VIX_STD120"].replace(0.0, np.nan)
    return out


def build_price_recovery_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["SPY_MA20"] = out["spy_price"].rolling(20, min_periods=20).mean()
    out["SPY_5D_RETURN"] = out["spy_price"] / out["spy_price"].shift(5) - 1.0
    out["SPY_CROSS_ABOVE_MA20"] = (out["spy_price"] > out["SPY_MA20"]) & (out["spy_price"].shift(1) <= out["SPY_MA20"].shift(1))
    return out


def build_stress_entry_signal(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    steep_either = out["macro_regime_confirmed"].eq("STEEP") & out["monthly_either_state"].eq("SELL")
    vix_stress = out["macro_regime_confirmed"].isin(CONFIG["enabled_vix_regimes"]) & (out["VIX_ZSCORE_120D"] >= CONFIG["vix_z_threshold"])
    out["stress_entry_signal"] = steep_either | vix_stress
    out["stress_entry_reason"] = ""
    out.loc[steep_either, "stress_entry_reason"] = "STEEP_EITHER_SELL"
    for regime in CONFIG["enabled_vix_regimes"]:
        out.loc[out["macro_regime_confirmed"].eq(regime) & vix_stress & ~steep_either, "stress_entry_reason"] = f"{regime}_VIX_STRESS"
    out.loc[out["macro_regime_confirmed"].eq("STEEP") & vix_stress & steep_either, "stress_entry_reason"] = "STEEP_EITHER_SELL+STEEP_VIX_STRESS"
    return out


def _exit_signal(row: pd.Series, recovery_rule: str) -> tuple[bool, str]:
    if recovery_rule == "R8":
        return bool(row["SPY_5D_RETURN"] > 0.03), "R8_SPY_5D_RETURN_GT_3"
    if recovery_rule == "R3":
        return bool(row["SPY_CROSS_ABOVE_MA20"]), "R3_SPY_CROSS_ABOVE_MA20"
    raise ValueError(f"Unknown recovery rule: {recovery_rule}")


def run_state_machine_strategy(panel: pd.DataFrame, recovery_rule: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = panel.copy()
    prefix = f"STRESS_RECOVERY_{recovery_rule}"
    state_col = f"{recovery_rule}_risk_state"
    out[state_col] = ""
    out[f"{recovery_rule}_weight_spy"] = np.nan
    out[f"{recovery_rule}_weight_cash"] = np.nan
    out[f"{prefix}_return"] = np.nan
    out[f"{prefix}_nav"] = np.nan
    out[f"transaction_cost_{recovery_rule}"] = 0.0
    out[f"{prefix}_turnover"] = 0.0

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
            turnover = abs(new_w - old_w) + abs((1.0 - new_w) - (1.0 - old_w))
            cost = 0.5 * turnover * cost_rate
            out.loc[i, f"transaction_cost_{recovery_rule}"] = cost
            out.loc[i, f"{prefix}_turnover"] = turnover
            events.append(
                {
                    "strategy": prefix,
                    "signal_date": out.loc[i - 1, "date"],
                    "event_date": row["date"],
                    "event_type": "ENTER_RISK" if state == "RISK" else "EXIT_RISK",
                    "reason": pending_reason,
                    "macro_regime_confirmed": out.loc[i - 1, "macro_regime_confirmed"],
                    "monthly_either_state": out.loc[i - 1, "monthly_either_state"],
                    "VIX_LEVEL": out.loc[i - 1, "VIX_LEVEL"],
                    "VIX_ZSCORE_120D": out.loc[i - 1, "VIX_ZSCORE_120D"],
                    "SPY_price": out.loc[i - 1, "spy_price"],
                    "SPY_5D_RETURN": out.loc[i - 1, "SPY_5D_RETURN"],
                    "SPY_MA20": out.loc[i - 1, "SPY_MA20"],
                    "previous_state": old_state,
                    "new_state": state,
                }
            )

        weight_spy = 1.0 if state == "NORMAL" else 0.0
        weight_cash = 1.0 - weight_spy
        daily_ret = weight_spy * row["spy_daily_return"] + weight_cash * row["daily_rf"] - out.loc[i, f"transaction_cost_{recovery_rule}"]
        nav *= 1.0 + float(daily_ret)
        out.loc[i, state_col] = state
        out.loc[i, f"{recovery_rule}_weight_spy"] = weight_spy
        out.loc[i, f"{recovery_rule}_weight_cash"] = weight_cash
        out.loc[i, f"{prefix}_return"] = daily_ret
        out.loc[i, f"{prefix}_nav"] = nav

        pending_state = state
        pending_reason = ""
        if state == "NORMAL" and bool(row["stress_entry_signal"]):
            pending_state = "RISK"
            pending_reason = row["stress_entry_reason"]
        elif state == "RISK":
            exit_now, reason = _exit_signal(row, recovery_rule)
            if exit_now:
                pending_state = "NORMAL"
                pending_reason = reason

    return out, pd.DataFrame(events)


def _perf_stats(panel: pd.DataFrame, strategy: str, ret_col: str, nav_col: str, weight_col: str | None = None, cost_col: str | None = None, turnover_col: str | None = None, events: pd.DataFrame | None = None) -> dict[str, float | str]:
    s = panel[ret_col].dropna()
    rf = panel.loc[s.index, "daily_rf"]
    ann_ret = float((1.0 + s).prod() ** (252.0 / len(s)) - 1.0)
    ann_vol = float(s.std(ddof=1) * np.sqrt(252.0)) if len(s) > 1 else np.nan
    if strategy == "CASH_ONLY":
        sharpe = 0.0
    else:
        ex = s - rf
        ex_std = ex.std(ddof=1)
        sharpe = float(ex.mean() / ex_std * np.sqrt(252.0)) if pd.notna(ex_std) and ex_std != 0 else np.nan
    wealth = (1.0 + s).cumprod()
    mdd = float((wealth / wealth.cummax() - 1.0).min())
    if weight_col is not None and weight_col in panel.columns:
        time_spy = float(panel[weight_col].mean())
    elif strategy == "SPY_BUY_HOLD":
        time_spy = 1.0
    elif strategy == "CASH_ONLY":
        time_spy = 0.0
    elif strategy == "MONTHLY_EITHER_CONFIRM":
        time_spy = float(panel["monthly_either_weight_spy"].mean())
    else:
        time_spy = np.nan
    ev = events.loc[events["strategy"] == strategy] if events is not None and not events.empty else pd.DataFrame()
    return {
        "strategy": strategy,
        "start_date": panel["date"].iloc[0].date().isoformat(),
        "end_date": panel["date"].iloc[-1].date().isoformat(),
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
        "calmar_ratio": float(ann_ret / abs(mdd)) if mdd < 0 else np.nan,
        "final_nav": float(panel[nav_col].iloc[-1]) if nav_col in panel.columns else float(wealth.iloc[-1]),
        "number_of_switches": int(len(ev)) if not ev.empty else int(panel[weight_col].diff().abs().sum()) if weight_col in panel.columns else 0,
        "number_of_risk_entries": int((ev["event_type"] == "ENTER_RISK").sum()) if not ev.empty else 0,
        "number_of_risk_exits": int((ev["event_type"] == "EXIT_RISK").sum()) if not ev.empty else 0,
        "avg_risk_episode_duration": np.nan,
        "time_in_spy": time_spy,
        "time_in_cash": 1.0 - time_spy if pd.notna(time_spy) else np.nan,
        "total_turnover": float(panel[turnover_col].sum()) if turnover_col in panel.columns else np.nan,
        "transaction_cost_drag": float(panel[cost_col].sum()) if cost_col in panel.columns else 0.0,
    }


def extract_risk_episodes(panel: pd.DataFrame, event_log: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, rule in [("STRESS_RECOVERY_R8", "R8"), ("STRESS_RECOVERY_R3", "R3")]:
        state_col = f"{rule}_risk_state"
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
            spy_rel = sub["spy_price"] / sub["spy_price"].iloc[0] - 1.0
            spy_wealth = sub["spy_price"] / sub["spy_price"].iloc[0]
            trough_pos = int(np.argmin(sub["spy_price"].to_numpy()))
            entry = event_log.loc[(event_log["strategy"] == strategy) & (event_log["event_type"] == "ENTER_RISK") & (event_log["event_date"] == sub["date"].iloc[0])]
            exit_ev = event_log.loc[(event_log["strategy"] == strategy) & (event_log["event_type"] == "EXIT_RISK") & (event_log["event_date"] > sub["date"].iloc[-1])]
            rows.append(
                {
                    "strategy": strategy,
                    "episode_id": eid,
                    "risk_start_date": sub["date"].iloc[0],
                    "risk_end_date": sub["date"].iloc[-1],
                    "duration_days": int(len(sub)),
                    "entry_reason": entry["reason"].iloc[0] if not entry.empty else "",
                    "exit_reason": exit_ev["reason"].iloc[0] if not exit_ev.empty else "",
                    "macro_regime_at_entry": sub["macro_regime_confirmed"].iloc[0],
                    "VIX_ZSCORE_at_entry": float(sub["VIX_ZSCORE_120D"].iloc[0]),
                    "SPY_return_during_risk_episode": float((1.0 + sub["spy_daily_return"]).prod() - 1.0),
                    "CASH_return_during_risk_episode": float((1.0 + sub["daily_rf"]).prod() - 1.0),
                    "strategy_return_during_risk_episode": float((1.0 + sub[f"{strategy}_return"]).prod() - 1.0),
                    "SPY_max_drawdown_during_risk_episode": float((spy_wealth / spy_wealth.cummax() - 1.0).min()),
                    "SPY_max_runup_during_risk_episode": float(spy_rel.max()),
                    "days_to_exit": int(len(sub)),
                    "exit_after_trough_days": int((len(sub) - 1) - trough_pos),
                }
            )
            i += 1
    return pd.DataFrame(rows)


def compute_performance_metrics(panel: pd.DataFrame, event_log: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    rows = [
        _perf_stats(panel, "SPY_BUY_HOLD", "SPY_BUY_HOLD_return", "SPY_BUY_HOLD_nav"),
        _perf_stats(panel, "CASH_ONLY", "CASH_ONLY_return", "CASH_ONLY_nav"),
        _perf_stats(
            panel,
            "MONTHLY_EITHER_CONFIRM",
            "MONTHLY_EITHER_CONFIRM_return",
            "MONTHLY_EITHER_CONFIRM_nav",
            "monthly_either_weight_spy",
            "transaction_cost_MONTHLY_EITHER_CONFIRM",
            "MONTHLY_EITHER_CONFIRM_turnover",
        ),
        _perf_stats(panel, "STRESS_RECOVERY_R8", "STRESS_RECOVERY_R8_return", "STRESS_RECOVERY_R8_nav", "R8_weight_spy", "transaction_cost_R8", "STRESS_RECOVERY_R8_turnover", event_log),
        _perf_stats(panel, "STRESS_RECOVERY_R3", "STRESS_RECOVERY_R3_return", "STRESS_RECOVERY_R3_nav", "R3_weight_spy", "transaction_cost_R3", "STRESS_RECOVERY_R3_turnover", event_log),
    ]
    perf = pd.DataFrame(rows)
    if not episodes.empty:
        dur = episodes.groupby("strategy")["duration_days"].mean()
        perf["avg_risk_episode_duration"] = perf["strategy"].map(dur).combine_first(perf["avg_risk_episode_duration"])
    return perf


def _group_stats(sub: pd.DataFrame, strategy: str, ret_col: str, weight_col: str | None = None) -> dict[str, float]:
    s = sub[ret_col].dropna()
    if s.empty:
        return {"n_obs": 0, "annualized_return": np.nan, "volatility": np.nan, "Sharpe": np.nan, "max_drawdown": np.nan, "time_in_cash": np.nan}
    rf = sub.loc[s.index, "daily_rf"]
    ann = float((1.0 + s).prod() ** (252.0 / len(s)) - 1.0)
    vol = float(s.std(ddof=1) * np.sqrt(252.0)) if len(s) > 1 else np.nan
    if strategy == "CASH_ONLY":
        sharpe = 0.0
    else:
        ex = s - rf
        ex_std = ex.std(ddof=1)
        sharpe = float(ex.mean() / ex_std * np.sqrt(252.0)) if pd.notna(ex_std) and ex_std != 0 else np.nan
    wealth = (1.0 + s).cumprod()
    mdd = float((wealth / wealth.cummax() - 1.0).min())
    if weight_col and weight_col in sub.columns:
        cash = float((1.0 - sub[weight_col]).mean())
    elif strategy == "CASH_ONLY":
        cash = 1.0
    elif strategy == "SPY_BUY_HOLD":
        cash = 0.0
    elif strategy == "MONTHLY_EITHER_CONFIRM":
        cash = float((1.0 - sub["monthly_either_weight_spy"]).mean())
    else:
        cash = np.nan
    return {"n_obs": int(len(s)), "annualized_return": ann, "volatility": vol, "Sharpe": sharpe, "max_drawdown": mdd, "time_in_cash": cash}


def compute_crisis_performance(panel: pd.DataFrame, event_log: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mapping = [
        ("SPY_BUY_HOLD", "SPY_BUY_HOLD_return", None),
        ("CASH_ONLY", "CASH_ONLY_return", None),
        ("MONTHLY_EITHER_CONFIRM", "MONTHLY_EITHER_CONFIRM_return", None),
        ("STRESS_RECOVERY_R8", "STRESS_RECOVERY_R8_return", "R8_weight_spy"),
        ("STRESS_RECOVERY_R3", "STRESS_RECOVERY_R3_return", "R3_weight_spy"),
    ]
    for period, (start, end) in CASE_WINDOWS.items():
        sub = panel.loc[panel["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        if sub.empty:
            continue
        for strategy, ret_col, weight_col in mapping:
            stats = _group_stats(sub, strategy, ret_col, weight_col)
            if strategy == "MONTHLY_EITHER_CONFIRM":
                switches = int(sub["monthly_either_weight_spy"].diff().abs().fillna(0.0).sum())
            else:
                switches = len(event_log.loc[(event_log["strategy"] == strategy) & event_log["event_date"].between(pd.Timestamp(start), pd.Timestamp(end))])
            rows.append({"period": period, "strategy": strategy, "cumulative_return": float((1.0 + sub[ret_col]).prod() - 1.0), "number_of_switches": int(switches), **stats})
    return pd.DataFrame(rows)


def compute_regime_performance(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mapping = [
        ("SPY_BUY_HOLD", "SPY_BUY_HOLD_return", None),
        ("CASH_ONLY", "CASH_ONLY_return", None),
        ("MONTHLY_EITHER_CONFIRM", "MONTHLY_EITHER_CONFIRM_return", None),
        ("STRESS_RECOVERY_R8", "STRESS_RECOVERY_R8_return", "R8_weight_spy"),
        ("STRESS_RECOVERY_R3", "STRESS_RECOVERY_R3_return", "R3_weight_spy"),
    ]
    for regime, grp in panel.groupby("macro_regime_confirmed"):
        for strategy, ret_col, weight_col in mapping:
            rows.append({"macro_regime_confirmed": regime, "strategy": strategy, **_group_stats(grp, strategy, ret_col, weight_col)})
    return pd.DataFrame(rows)


def compute_entry_reason_performance(panel: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()
    rows = []
    for reason, grp in episodes.groupby("entry_reason", dropna=False):
        rows.append(
            {
                "stress_entry_reason": reason,
                "event_count": int(len(grp)),
                "avg_forward_SPY_drawdown_after_entry": float(grp["SPY_max_drawdown_during_risk_episode"].mean()),
                "R8_episode_return": float(grp.loc[grp["strategy"] == "STRESS_RECOVERY_R8", "strategy_return_during_risk_episode"].mean()),
                "R3_episode_return": float(grp.loc[grp["strategy"] == "STRESS_RECOVERY_R3", "strategy_return_during_risk_episode"].mean()),
                "average_R8_duration": float(grp.loc[grp["strategy"] == "STRESS_RECOVERY_R8", "duration_days"].mean()),
                "average_R3_duration": float(grp.loc[grp["strategy"] == "STRESS_RECOVERY_R3", "duration_days"].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_results(panel: pd.DataFrame, perf: pd.DataFrame, episodes: pd.DataFrame) -> None:
    nav_cols = {
        "SPY_BUY_HOLD": "SPY_BUY_HOLD_nav",
        "MONTHLY_EITHER_CONFIRM": "MONTHLY_EITHER_CONFIRM_nav",
        "STRESS_RECOVERY_R8": "STRESS_RECOVERY_R8_nav",
        "STRESS_RECOVERY_R3": "STRESS_RECOVERY_R3_nav",
        "CASH_ONLY": "CASH_ONLY_nav",
    }
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, col in nav_cols.items():
        ax.plot(panel["date"], panel[col], label=name)
    ax.set_yscale("log")
    ax.set_title("SPY/CASH Stress-Recovery Timing")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_EQUITY, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for name, ret_col in [
        ("SPY_BUY_HOLD", "SPY_BUY_HOLD_return"),
        ("MONTHLY_EITHER_CONFIRM", "MONTHLY_EITHER_CONFIRM_return"),
        ("STRESS_RECOVERY_R8", "STRESS_RECOVERY_R8_return"),
        ("STRESS_RECOVERY_R3", "STRESS_RECOVERY_R3_return"),
    ]:
        wealth = (1.0 + panel[ret_col]).cumprod()
        ax.plot(panel["date"], wealth / wealth.cummax() - 1.0, label=name)
    ax.legend()
    ax.set_title("Drawdown Comparison")
    fig.tight_layout()
    fig.savefig(FIG_DD, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.5, 1.5]})
    axes[0].plot(panel["date"], panel["SPY_BUY_HOLD_nav"], label="SPY", color="black")
    axes[0].plot(panel["date"], panel["STRESS_RECOVERY_R8_nav"], label="R8", color="tab:blue")
    axes[0].plot(panel["date"], panel["STRESS_RECOVERY_R3_nav"], label="R3", color="tab:orange")
    axes[0].legend()
    axes[1].plot(panel["date"], panel["VIX_ZSCORE_120D"], color="tab:purple")
    axes[1].axhline(CONFIG["vix_z_threshold"], color="red", linestyle="--")
    for regime, color in REGIME_COLORS.items():
        axes[2].fill_between(panel["date"], 0.65, 0.95, where=panel["macro_regime_confirmed"].eq(regime), color=color, alpha=0.5)
    axes[2].fill_between(panel["date"], 0.35, 0.6, where=panel["R8_risk_state"].eq("RISK"), color="tab:blue", alpha=0.7)
    axes[2].fill_between(panel["date"], 0.05, 0.3, where=panel["R3_risk_state"].eq("RISK"), color="tab:orange", alpha=0.7)
    axes[2].set_yticks([0.17, 0.47, 0.8])
    axes[2].set_yticklabels(["R3", "R8", "Regime"])
    fig.tight_layout()
    fig.savefig(FIG_STATE, dpi=180)
    plt.close(fig)

    _plot_case(panel, "COVID_2020", ("2020-02-19", "2020-04-30"), FIG_COVID)
    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    for ax, (name, window) in zip(axes, [("GFC_2008_2009", CASE_WINDOWS["GFC_2008_2009"]), ("COVID_2020", CASE_WINDOWS["COVID_2020"]), ("INFLATION_2022", CASE_WINDOWS["INFLATION_2022"])]):
        _plot_case_panel(panel, name, window, ax)
    fig.tight_layout()
    fig.savefig(FIG_CRISIS, dpi=180)
    plt.close(fig)

    metrics = ["annualized_return", "sharpe_ratio", "max_drawdown", "calmar_ratio", "number_of_switches"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, metric in zip(axes, metrics):
        sns.barplot(data=perf, x="strategy", y=metric, ax=ax)
        ax.tick_params(axis="x", rotation=45)
        ax.set_title(metric)
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(FIG_BAR, dpi=180)
    plt.close(fig)

    if not episodes.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(data=episodes, x="duration_days", hue="strategy", bins=20, alpha=0.5, ax=ax)
        ax.set_title("Risk Episode Duration")
        fig.tight_layout()
        fig.savefig(FIG_HIST, dpi=180)
        plt.close(fig)


def _plot_case_panel(panel: pd.DataFrame, name: str, window: tuple[str, str], ax: plt.Axes) -> None:
    sub = panel.loc[panel["date"].between(pd.Timestamp(window[0]), pd.Timestamp(window[1]))].copy()
    if sub.empty:
        return
    for label, col in [("SPY", "SPY_BUY_HOLD_nav"), ("Monthly Either", "MONTHLY_EITHER_CONFIRM_nav"), ("R8", "STRESS_RECOVERY_R8_nav"), ("R3", "STRESS_RECOVERY_R3_nav")]:
        ax.plot(sub["date"], sub[col] / sub[col].iloc[0], label=label)
    ax.set_title(name)
    ax.legend(fontsize=8, ncol=4)


def _plot_case(panel: pd.DataFrame, name: str, window: tuple[str, str], path: Path) -> None:
    sub = panel.loc[panel["date"].between(pd.Timestamp(window[0]), pd.Timestamp(window[1]))].copy()
    if sub.empty:
        return
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(sub["date"], sub["spy_price"] / sub["spy_price"].iloc[0], label="SPY", color="black")
    axes[0].plot(sub["date"], sub["STRESS_RECOVERY_R8_nav"] / sub["STRESS_RECOVERY_R8_nav"].iloc[0], label="R8", color="tab:blue")
    axes[0].plot(sub["date"], sub["STRESS_RECOVERY_R3_nav"] / sub["STRESS_RECOVERY_R3_nav"].iloc[0], label="R3", color="tab:orange")
    trough = sub.loc[sub["spy_price"].idxmin(), "date"]
    axes[0].axvline(trough, color="black", linestyle=":", label="SPY trough")
    axes[0].legend()
    axes[0].set_title(name)
    axes[1].plot(sub["date"], sub["VIX_ZSCORE_120D"], color="tab:purple")
    axes[1].axhline(CONFIG["vix_z_threshold"], color="red", linestyle="--")
    axes[2].fill_between(sub["date"], 0.55, 0.9, where=sub["R8_risk_state"].eq("RISK"), color="tab:blue", alpha=0.7)
    axes[2].fill_between(sub["date"], 0.1, 0.45, where=sub["R3_risk_state"].eq("RISK"), color="tab:orange", alpha=0.7)
    axes[2].set_yticks([0.28, 0.72])
    axes[2].set_yticklabels(["R3", "R8"])
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary_md(perf: pd.DataFrame, crisis: pd.DataFrame) -> None:
    table = perf.to_markdown(index=False)
    crisis_table = crisis.to_markdown(index=False)
    lines = [
        "# SPY/CASH Stress-Recovery Timing Summary",
        "",
        "## Purpose",
        "",
        "This test uses only SPY and CASH. The state machine is either 100% SPY or 100% CASH. It tests fixed stress entry with price-based recovery exits.",
        "",
        "## Stress Entry Definition",
        "",
        "- `STEEP + Monthly Either SELL` enters risk.",
        "- `FLAT + VIX z-score 120D >= 3.0` enters risk.",
        "- VIX fast-stress is disabled outside FLAT in this test.",
        "",
        "## Recovery Exit Definitions",
        "",
        "- R8: `SPY 5D return > 3%`.",
        "- R3: `SPY crosses above MA20`.",
        "",
        "## Transaction Cost",
        "",
        "- Uses half-turnover convention: `cost = 0.5 * turnover * one_way_cost_bps / 10000`.",
        "- One-way cost is 5 bps.",
        "",
        "## Performance Comparison",
        "",
        table,
        "",
        "## Crisis Performance",
        "",
        crisis_table,
        "",
        "## Interpretation",
        "",
        "- R8 is the aggressive recovery candidate.",
        "- R3 is the conservative recovery candidate.",
        "- This is still a SPY/CASH diagnostic and does not include hedge assets.",
        "",
        "## Next Step",
        "",
        "- Embed the better recovery state machine into `REGIME_HEDGE_STEEP_SELL_IEF` and test FLAT_STRESS / STEEP_STRESS allocation variants.",
    ]
    SUMMARY_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = build_stress_entry_signal(build_price_recovery_features(build_vix_zscore(load_data())))

    panel, r8_events = run_state_machine_strategy(panel, "R8")
    panel, r3_events = run_state_machine_strategy(panel, "R3")
    event_log = pd.concat([r8_events, r3_events], ignore_index=True)

    panel["SPY_BUY_HOLD_return"] = panel["spy_daily_return"]
    panel["SPY_BUY_HOLD_nav"] = (1.0 + panel["SPY_BUY_HOLD_return"]).cumprod()
    panel["CASH_ONLY_return"] = panel["daily_rf"]
    panel["CASH_ONLY_nav"] = (1.0 + panel["CASH_ONLY_return"]).cumprod()

    monthly_prev_weight = panel["monthly_either_weight_spy"].shift(1).fillna(panel["monthly_either_weight_spy"].iloc[0])
    monthly_turnover = (
        (panel["monthly_either_weight_spy"] - monthly_prev_weight).abs()
        + ((1.0 - panel["monthly_either_weight_spy"]) - (1.0 - monthly_prev_weight)).abs()
    )
    panel["MONTHLY_EITHER_CONFIRM_turnover"] = monthly_turnover
    panel["transaction_cost_MONTHLY_EITHER_CONFIRM"] = 0.5 * monthly_turnover * (CONFIG["one_way_cost_bps"] / 10000.0)
    panel["MONTHLY_EITHER_CONFIRM_return"] = (
        panel["monthly_either_weight_spy"] * panel["spy_daily_return"]
        + (1.0 - panel["monthly_either_weight_spy"]) * panel["daily_rf"]
        - panel["transaction_cost_MONTHLY_EITHER_CONFIRM"]
    )
    panel["MONTHLY_EITHER_CONFIRM_nav"] = (1.0 + panel["MONTHLY_EITHER_CONFIRM_return"]).cumprod()

    if TEST_STEEP_ONLY_PATH.exists():
        test = pd.read_csv(TEST_STEEP_ONLY_PATH, usecols=["date", "TEST_STEEP_ONLY_SELL_return", "TEST_STEEP_ONLY_SELL_nav"])
        test["date"] = pd.to_datetime(test["date"])
        panel = panel.merge(test, on="date", how="left")

    risk_episodes = extract_risk_episodes(panel, event_log)
    perf = compute_performance_metrics(panel, event_log, risk_episodes)
    crisis = compute_crisis_performance(panel, event_log)
    regime_perf = compute_regime_performance(panel)
    entry_reason = compute_entry_reason_performance(panel, risk_episodes)

    panel.to_csv(DAILY_OUT, index=False)
    event_log.to_csv(EVENT_LOG_OUT, index=False)
    risk_episodes.to_csv(RISK_EPISODES_OUT, index=False)
    perf.to_csv(PERF_OUT, index=False)
    crisis.to_csv(CRISIS_OUT, index=False)
    regime_perf.to_csv(REGIME_OUT, index=False)
    entry_reason.to_csv(ENTRY_REASON_OUT, index=False)
    plot_results(panel, perf, risk_episodes)
    write_summary_md(perf, crisis)

    r8 = perf.loc[perf["strategy"] == "STRESS_RECOVERY_R8"].iloc[0]
    r3 = perf.loc[perf["strategy"] == "STRESS_RECOVERY_R3"].iloc[0]
    me = perf.loc[perf["strategy"] == "MONTHLY_EITHER_CONFIRM"].iloc[0]
    covid = crisis.loc[crisis["period"] == "COVID_2020"].sort_values("cumulative_return", ascending=False)
    gfc = crisis.loc[crisis["period"] == "GFC_2008_2009"].sort_values("cumulative_return", ascending=False)
    infl = crisis.loc[crisis["period"] == "INFLATION_2022"].sort_values("cumulative_return", ascending=False)
    recommend = "R8" if (r8["sharpe_ratio"] >= r3["sharpe_ratio"] and r8["max_drawdown"] >= r3["max_drawdown"]) else ("R8" if r8["annualized_return"] > r3["annualized_return"] else "R3")
    print(f"1. R8 annualized / Sharpe / MaxDD: {r8['annualized_return']:.2%} / {r8['sharpe_ratio']:.2f} / {r8['max_drawdown']:.2%}")
    print(f"2. R3 annualized / Sharpe / MaxDD: {r3['annualized_return']:.2%} / {r3['sharpe_ratio']:.2f} / {r3['max_drawdown']:.2%}")
    print(f"3. Monthly Either annualized / Sharpe / MaxDD: {me['annualized_return']:.2%} / {me['sharpe_ratio']:.2f} / {me['max_drawdown']:.2%}")
    print(f"4. R8 beats Monthly Either by Sharpe: {bool(r8['sharpe_ratio'] > me['sharpe_ratio'])}")
    print(f"5. R3 beats Monthly Either by Sharpe: {bool(r3['sharpe_ratio'] > me['sharpe_ratio'])}")
    print(f"6. Higher return, R8 vs R3: {'R8' if r8['annualized_return'] > r3['annualized_return'] else 'R3'}")
    print(f"7. Lower drawdown, R8 vs R3: {'R8' if r8['max_drawdown'] > r3['max_drawdown'] else 'R3'}")
    print(f"8. Best COVID strategy: {covid.iloc[0]['strategy'] if not covid.empty else 'n/a'}")
    print(f"9. Best 2008 / 2022 strategy: {gfc.iloc[0]['strategy'] if not gfc.empty else 'n/a'} / {infl.iloc[0]['strategy'] if not infl.empty else 'n/a'}")
    print(f"10. Recommended next recovery rule: {recommend}")
    print(f"Saved outputs: {RESULTS_DIR} and {FIGURES_DIR}")


if __name__ == "__main__":
    main()
