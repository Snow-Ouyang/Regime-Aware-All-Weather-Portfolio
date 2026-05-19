from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "rule_based_allocation"
FIGURES_DIR = ROOT / "figures" / "rule_based_allocation"

DAILY_RETURNS_PATH = ROOT / "data" / "processed" / "assets" / "daily_returns.csv"
DAILY_CLOSE_PATH = ROOT / "data" / "processed" / "assets" / "daily_adjusted_close.csv"
VIX_PATH = ROOT / "data" / "raw" / "macro" / "volatility" / "VIXCLS.csv"
DGS1_PATH = ROOT / "data" / "raw" / "macro" / "rate" / "DGS1.csv"
DGS10_PATH = ROOT / "data" / "raw" / "macro" / "rate" / "DGS10.csv"
WAAA_PATH = ROOT / "data" / "raw" / "macro" / "Credit" / "WAAA.csv"
WBAA_PATH = ROOT / "data" / "raw" / "macro" / "Credit" / "WBAA.csv"
DTB3_CANDIDATES = [
    ROOT / "data" / "raw" / "rate" / "DTB3.csv",
    ROOT / "data" / "raw" / "macro" / "rate" / "DTB3.csv",
]
OLD_RULE_RETURNS_PATH = ROOT / "results" / "rule_based_backtest" / "rule_based_strategy_daily_returns.csv"

PANEL_PATH = RESULTS_DIR / "rule_strategy_daily_panel_v2.csv"
RETURNS_PATH = RESULTS_DIR / "rule_strategy_daily_returns_v2.csv"
EQUITY_PATH = RESULTS_DIR / "rule_strategy_equity_curves_v2.csv"
TARGET_WEIGHTS_PATH = RESULTS_DIR / "daily_target_weights_v2.csv"
ACTUAL_WEIGHTS_PATH = RESULTS_DIR / "daily_actual_weights_v2.csv"
REBALANCE_LOG_PATH = RESULTS_DIR / "rule_strategy_rebalance_log_v2.csv"
PERFORMANCE_PATH = RESULTS_DIR / "rule_strategy_performance_summary_v2.csv"
AVG_MACRO_PATH = RESULTS_DIR / "average_weights_by_macro_regime_v2.csv"
AVG_VIX_PATH = RESULTS_DIR / "average_weights_by_vix_overlay_state_v2.csv"
PERF_MACRO_PATH = RESULTS_DIR / "performance_by_macro_regime_v2.csv"
PERF_CROSS_PATH = RESULTS_DIR / "performance_by_cross_state_v2.csv"
DRAW_PATH = RESULTS_DIR / "drawdown_series_v2.csv"
VALIDATION_PATH = RESULTS_DIR / "backtest_validation_report_v2.csv"
EPISODE_PATH = RESULTS_DIR / "vix_stress_episode_diagnostics_v2.csv"
PERIOD_PATH = RESULTS_DIR / "period_diagnostics_v2.csv"
REPORT_PATH = RESULTS_DIR / "RULE_BASED_BACKTEST_REPORT_V2.md"

ASSETS = ["SPY", "IEF", "GOLD", "CMDTY_FUT", "CASH"]
CORE_STRATEGIES = ["RULE_BASED_NEW", "SPY_ONLY", "STATIC_40_30_15_15", "CASH_ONLY"]
MACRO_ORDER = ["HIGH_INFLATION", "INVERTED", "FLAT", "STEEP", "NEUTRAL"]
VIX_ORDER = ["NORMAL", "WARNING", "STRESS", "RECOVERY"]

CONFIRMATION_DAYS = 3
DRIFT_THRESHOLD = 0.05
PROGRESS_REBALANCE_THRESHOLD = 0.02
TCOST_BPS_PER_TOTAL_TRADED_NOTIONAL = 5.0

NORMAL_ALLOCATIONS = {
    "HIGH_INFLATION": {"SPY": 0.70, "IEF": 0.00, "GOLD": 0.00, "CMDTY_FUT": 0.00, "CASH": 0.30},
    "INVERTED": {"SPY": 0.50, "IEF": 0.10, "GOLD": 0.30, "CMDTY_FUT": 0.00, "CASH": 0.10},
    "FLAT": {"SPY": 0.50, "IEF": 0.00, "GOLD": 0.30, "CMDTY_FUT": 0.20, "CASH": 0.00},
    "STEEP": {"SPY": 0.70, "IEF": 0.00, "GOLD": 0.10, "CMDTY_FUT": 0.20, "CASH": 0.00},
    "NEUTRAL": {"SPY": 0.40, "IEF": 0.20, "GOLD": 0.20, "CMDTY_FUT": 0.00, "CASH": 0.20},
}

STRESS_ALLOCATIONS = {
    "HIGH_INFLATION": {"SPY": 0.30, "IEF": 0.00, "GOLD": 0.00, "CMDTY_FUT": 0.00, "CASH": 0.70},
    "INVERTED": {"SPY": 0.30, "IEF": 0.00, "GOLD": 0.00, "CMDTY_FUT": 0.00, "CASH": 0.70},
    "FLAT": {"SPY": 0.30, "IEF": 0.40, "GOLD": 0.30, "CMDTY_FUT": 0.00, "CASH": 0.00},
    "STEEP": {"SPY": 0.20, "IEF": 0.40, "GOLD": 0.40, "CMDTY_FUT": 0.00, "CASH": 0.00},
    "NEUTRAL": {"SPY": 0.40, "IEF": 0.20, "GOLD": 0.20, "CMDTY_FUT": 0.00, "CASH": 0.20},
}

PERIOD_WINDOWS = {
    "CRISIS_2008": ("2008-09-01", "2009-03-31"),
    "CRASH_2020": ("2020-02-19", "2020-04-30"),
    "INFLATION_2022": ("2022-01-01", "2022-12-31"),
    "RECENT_2024_2026": ("2024-01-01", "2026-12-31"),
}


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def read_fred_csv(path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
    value_col = next((c for c in df.columns if c != date_col), df.columns[-1])
    out = df[[date_col, value_col]].copy()
    out.columns = ["date", value_name]
    out["date"] = pd.to_datetime(out["date"])
    out[value_name] = pd.to_numeric(out[value_name].replace(".", np.nan), errors="coerce")
    return out.sort_values("date")


def load_rf_daily() -> tuple[pd.DataFrame, Path]:
    path = next((p for p in DTB3_CANDIDATES if p.exists()), None)
    if path is None:
        raise FileNotFoundError("DTB3.csv not found in expected paths.")
    rf = read_fred_csv(path, "DTB3")
    rf["DTB3_RATE"] = rf["DTB3"] / 100.0
    rf["RF_DAILY"] = (1.0 + rf["DTB3_RATE"].ffill()) ** (1.0 / 252.0) - 1.0
    return rf[["date", "DTB3", "DTB3_RATE", "RF_DAILY"]], path


def build_daily_panel() -> tuple[pd.DataFrame, dict[str, str]]:
    returns = pd.read_csv(DAILY_RETURNS_PATH)
    closes = pd.read_csv(DAILY_CLOSE_PATH, usecols=["date", "SPY"])
    returns["date"] = pd.to_datetime(returns["date"])
    closes["date"] = pd.to_datetime(closes["date"])

    required_return_cols = ["SPY", "IEF", "GLD", "GD=F"]
    missing = [col for col in required_return_cols if col not in returns.columns]
    if missing:
        raise ValueError(f"Missing required asset return columns: {missing}")

    asset_ret = returns[["date", "SPY", "IEF", "GLD", "GD=F"]].copy().rename(
        columns={"SPY": "SPY_RET", "IEF": "IEF_RET", "GLD": "GOLD_RET", "GD=F": "CMDTY_FUT_RET"}
    )

    vix = read_fred_csv(VIX_PATH, "VIX_LEVEL")
    dgs1 = read_fred_csv(DGS1_PATH, "DGS1")
    dgs10 = read_fred_csv(DGS10_PATH, "DGS10")
    waaa = read_fred_csv(WAAA_PATH, "WAAA")
    wbaa = read_fred_csv(WBAA_PATH, "WBAA")
    credit = waaa.merge(wbaa, on="date", how="outer").sort_values("date")
    credit[["WAAA", "WBAA"]] = credit[["WAAA", "WBAA"]].ffill()
    credit["CREDIT_SPREAD_BAA_AAA"] = credit["WBAA"] - credit["WAAA"]
    rf, rf_path = load_rf_daily()

    panel = asset_ret.copy()
    for frame in [vix, dgs1, dgs10, credit[["date", "CREDIT_SPREAD_BAA_AAA"]], rf]:
        panel = panel.merge(frame, on="date", how="left")

    for col in ["VIX_LEVEL", "DGS1", "DGS10", "CREDIT_SPREAD_BAA_AAA", "DTB3", "DTB3_RATE", "RF_DAILY"]:
        panel[col] = panel[col].ffill()
    panel["TERM_SPREAD_10Y_1Y"] = panel["DGS10"] - panel["DGS1"]
    panel = panel.merge(closes, on="date", how="left")

    required = [
        "SPY_RET",
        "IEF_RET",
        "GOLD_RET",
        "CMDTY_FUT_RET",
        "RF_DAILY",
        "VIX_LEVEL",
        "DGS1",
        "DGS10",
        "CREDIT_SPREAD_BAA_AAA",
        "TERM_SPREAD_10Y_1Y",
        "SPY",
    ]
    before = len(panel)
    panel = panel.dropna(subset=required).copy()
    panel["sample_observations_dropped"] = before - len(panel)
    panel["CASH_RET"] = panel["RF_DAILY"]
    panel["SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"] = panel["SPY"] / panel["SPY"].cummax() - 1.0

    sources = {
        "daily_returns": str(DAILY_RETURNS_PATH),
        "daily_close": str(DAILY_CLOSE_PATH),
        "VIX_LEVEL": str(VIX_PATH),
        "DGS1": str(DGS1_PATH),
        "DGS10": str(DGS10_PATH),
        "CREDIT_SPREAD_BAA_AAA": f"{WBAA_PATH} - {WAAA_PATH}",
        "RF_DAILY": str(rf_path),
    }
    return panel.reset_index(drop=True), sources


def raw_macro_regime(row: pd.Series) -> str:
    if pd.isna(row["CREDIT_SPREAD_BAA_AAA"]) or pd.isna(row["DGS1"]) or pd.isna(row["TERM_SPREAD_10Y_1Y"]):
        return "NEUTRAL"
    if row["CREDIT_SPREAD_BAA_AAA"] > 1.5 and row["DGS1"] > 5:
        return "HIGH_INFLATION"
    if row["TERM_SPREAD_10Y_1Y"] < 0:
        return "INVERTED"
    if 0 <= row["TERM_SPREAD_10Y_1Y"] < 1:
        return "FLAT"
    if row["TERM_SPREAD_10Y_1Y"] >= 1:
        return "STEEP"
    return "NEUTRAL"


def confirm_regime(raw: pd.Series, confirmation_days: int, initial_confirmed: str) -> pd.DataFrame:
    confirmed = initial_confirmed
    candidate = initial_confirmed
    candidate_count = 0
    confirmed_list = []
    candidate_list = []
    candidate_count_list = []
    switch_flags = []

    for value in raw.astype(str):
        switch_flag = False
        if value == confirmed:
            candidate = confirmed
            candidate_count = 0
        elif value == candidate:
            candidate_count += 1
        else:
            candidate = value
            candidate_count = 1

        if candidate != confirmed and candidate_count >= confirmation_days:
            confirmed = candidate
            switch_flag = True
            candidate = confirmed
            candidate_count = 0

        confirmed_list.append(confirmed)
        candidate_list.append(candidate)
        candidate_count_list.append(candidate_count)
        switch_flags.append(switch_flag)

    return pd.DataFrame(
        {
            "candidate": candidate_list,
            "candidate_count": candidate_count_list,
            "confirmed": confirmed_list,
            "switch_flag": switch_flags,
        }
    )


def build_vix_overlay(panel: pd.DataFrame, confirmation_days: int) -> pd.DataFrame:
    raw = np.select(
        [panel["VIX_LEVEL"] >= 25, panel["VIX_LEVEL"] >= 20],
        ["STRESS", "WARNING"],
        default="NORMAL",
    )
    confirmed = "NORMAL"
    candidate = "NORMAL"
    count = 0
    states = []
    candidates = []
    counts = []
    switches = []
    progress_list = []

    for raw_state, vix in zip(raw, panel["VIX_LEVEL"]):
        switch = False
        desired = raw_state
        if confirmed == "STRESS" and raw_state != "STRESS":
            desired = "RECOVERY"
        elif confirmed == "RECOVERY" and raw_state == "STRESS":
            desired = "STRESS"
        elif confirmed == "RECOVERY" and raw_state == "NORMAL":
            desired = "NORMAL"
        elif confirmed == "RECOVERY" and raw_state == "WARNING":
            desired = "RECOVERY"
        elif confirmed == "WARNING" and raw_state == "STRESS":
            desired = "STRESS"
        elif confirmed == "WARNING" and raw_state == "NORMAL":
            desired = "NORMAL"
        elif confirmed == "NORMAL" and raw_state == "WARNING":
            desired = "WARNING"
        elif confirmed == "NORMAL" and raw_state == "STRESS":
            desired = "STRESS"

        if desired == confirmed:
            candidate = confirmed
            count = 0
        elif desired == candidate:
            count += 1
        else:
            candidate = desired
            count = 1

        if candidate != confirmed and count >= confirmation_days:
            confirmed = candidate
            switch = True
            candidate = confirmed
            count = 0

        if confirmed == "NORMAL":
            progress = 0.0
        elif confirmed == "STRESS":
            progress = 1.0
        else:
            progress = float(np.clip((vix - 20.0) / 5.0, 0.0, 1.0))

        states.append(confirmed)
        candidates.append(candidate)
        counts.append(count)
        switches.append(switch)
        progress_list.append(progress)

    return pd.DataFrame(
        {
            "vix_overlay_raw": raw,
            "vix_overlay_candidate": candidates,
            "vix_overlay_candidate_count": counts,
            "vix_overlay_confirmed": states,
            "vix_switch_flag": switches,
            "vix_progress": progress_list,
            "vix_level": panel["VIX_LEVEL"].values,
        }
    )


def interpolate_target(macro: str, progress: float) -> dict[str, float]:
    normal = NORMAL_ALLOCATIONS[macro]
    stress = STRESS_ALLOCATIONS[macro]
    weights = {
        asset: normal[asset] * (1.0 - progress) + stress[asset] * progress
        for asset in ASSETS
    }
    total = sum(weights.values())
    if not np.isclose(total, 1.0):
        weights = {asset: value / total for asset, value in weights.items()}
    return weights


def add_signal_columns(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    targets = []
    for _, row in out.iterrows():
        macro = row["macro_regime_confirmed"]
        progress = row["vix_progress"]
        weights = interpolate_target(macro, progress)
        targets.append(weights)
    targets_df = pd.DataFrame(targets)
    targets_df.columns = [f"target_weight_{asset.lower()}" for asset in targets_df.columns]
    return pd.concat([out.reset_index(drop=True), targets_df], axis=1)


def is_first_trading_day_of_month(dates: pd.Series) -> pd.Series:
    return dates.dt.to_period("M") != dates.dt.to_period("M").shift(1)


def load_old_rule_returns(new_dates: pd.Series) -> pd.Series | None:
    if not OLD_RULE_RETURNS_PATH.exists():
        return None
    old = pd.read_csv(OLD_RULE_RETURNS_PATH, usecols=["date", "RULE_BASED"])
    old["date"] = pd.to_datetime(old["date"])
    old = old.rename(columns={"RULE_BASED": "OLD_RULE_BASED"})
    merged = pd.DataFrame({"date": new_dates}).merge(old, on="date", how="left")
    if merged["OLD_RULE_BASED"].notna().sum() == 0:
        return None
    return merged["OLD_RULE_BASED"]


def compute_backtest(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = panel.copy().reset_index(drop=True)
    monthly_flag = is_first_trading_day_of_month(p["date"])
    ret_cols = {
        "SPY": "SPY_RET",
        "IEF": "IEF_RET",
        "GOLD": "GOLD_RET",
        "CMDTY_FUT": "CMDTY_FUT_RET",
        "CASH": "RF_DAILY",
    }
    tc_rate = TCOST_BPS_PER_TOTAL_TRADED_NOTIONAL / 10000.0

    actual_after_prev = None
    nav = 1.0
    panel_rows = []
    target_rows = []
    actual_rows = []
    rebalance_rows = []
    strategy_returns = []
    equity_rows = []

    prev_target_signal = None
    prev2_target_signal = None

    for i, row in p.iterrows():
        date = row["date"]
        signal_target = {asset: row[f"target_weight_{asset.lower()}"] for asset in ASSETS}
        target_for_execution = signal_target if i == 0 else prev_target_signal.copy()

        if actual_after_prev is None:
            actual_before = target_for_execution.copy()
            pre_weights = target_for_execution.copy()
        else:
            actual_before = actual_after_prev.copy()
            pre_weights = actual_after_prev.copy()

        reasons: list[str] = []
        if i == 0:
            reasons.append("initial")
        elif monthly_flag.iloc[i]:
            reasons.append("monthly_rebalance")
        if i > 0 and bool(p.loc[i - 1, "macro_regime_switch_flag"]):
            reasons.append("macro_regime_change")
        if i > 0 and bool(p.loc[i - 1, "vix_switch_flag"]):
            reasons.append("vix_overlay_change")
        if i > 1 and prev2_target_signal is not None and prev_target_signal is not None:
            target_shift = max(abs(prev_target_signal[a] - prev2_target_signal[a]) for a in ASSETS)
            if p.loc[i - 1, "vix_overlay_confirmed"] in ["WARNING", "RECOVERY"] and target_shift > PROGRESS_REBALANCE_THRESHOLD:
                reasons.append("vix_progress_change")
        drift = max(abs(pre_weights[a] - target_for_execution[a]) for a in ASSETS)
        if drift > DRIFT_THRESHOLD:
            reasons.append("drift_threshold")

        rebalance_flag = len(reasons) > 0
        if rebalance_flag:
            post_rebalance_weights = target_for_execution.copy()
            turnover = float(sum(abs(post_rebalance_weights[a] - pre_weights[a]) for a in ASSETS))
        else:
            post_rebalance_weights = pre_weights.copy()
            turnover = 0.0
        transaction_cost = turnover * tc_rate

        gross_return = float(sum(post_rebalance_weights[a] * row[ret_cols[a]] for a in ASSETS))
        net_return = gross_return - transaction_cost

        sleeve_values = {a: post_rebalance_weights[a] * (1.0 + row[ret_cols[a]]) for a in ASSETS}
        sleeve_total = sum(sleeve_values.values())
        actual_after = {a: sleeve_values[a] / sleeve_total for a in ASSETS}
        nav *= 1.0 + net_return

        panel_rows.append(
            {
                "date": date,
                "SPY_RET": row["SPY_RET"],
                "IEF_RET": row["IEF_RET"],
                "GOLD_RET": row["GOLD_RET"],
                "CMDTY_FUT_RET": row["CMDTY_FUT_RET"],
                "RF_DAILY": row["RF_DAILY"],
                "VIX_LEVEL": row["VIX_LEVEL"],
                "DGS1": row["DGS1"],
                "DGS10": row["DGS10"],
                "CREDIT_SPREAD_BAA_AAA": row["CREDIT_SPREAD_BAA_AAA"],
                "TERM_SPREAD_10Y_1Y": row["TERM_SPREAD_10Y_1Y"],
                "macro_regime_raw": row["macro_regime_raw"],
                "macro_regime_candidate": row["macro_regime_candidate"],
                "macro_regime_candidate_count": row["macro_regime_candidate_count"],
                "macro_regime_confirmed": row["macro_regime_confirmed"],
                "macro_regime_switch_flag": row["macro_regime_switch_flag"],
                "vix_level": row["vix_level"],
                "vix_overlay_raw": row["vix_overlay_raw"],
                "vix_overlay_candidate": row["vix_overlay_candidate"],
                "vix_overlay_candidate_count": row["vix_overlay_candidate_count"],
                "vix_overlay_confirmed": row["vix_overlay_confirmed"],
                "vix_progress": row["vix_progress"],
                "vix_switch_flag": row["vix_switch_flag"],
                **{f"target_weight_{a.lower()}": target_for_execution[a] for a in ASSETS},
                **{f"actual_weight_{a.lower()}_before_return": post_rebalance_weights[a] for a in ASSETS},
                **{f"actual_weight_{a.lower()}_after_return": actual_after[a] for a in ASSETS},
                "portfolio_return_gross": gross_return,
                "transaction_cost": transaction_cost,
                "portfolio_return_net": net_return,
                "nav": nav,
            }
        )
        target_rows.append({"date": date, **{f"target_weight_{a.lower()}": signal_target[a] for a in ASSETS}})
        actual_rows.append(
            {
                "date": date,
                **{f"previous_weight_{a.lower()}": pre_weights[a] for a in ASSETS},
                **{f"actual_weight_{a.lower()}_before_return": post_rebalance_weights[a] for a in ASSETS},
                **{f"actual_weight_{a.lower()}_after_return": actual_after[a] for a in ASSETS},
            }
        )
        rebalance_rows.append(
            {
                "date": date,
                "rebalance_flag": rebalance_flag,
                "rebalance_reason": "|".join(reasons),
                "macro_regime_confirmed": row["macro_regime_confirmed"],
                "vix_overlay_confirmed": row["vix_overlay_confirmed"],
                "vix_level": row["vix_level"],
                "vix_progress": row["vix_progress"],
                **{f"previous_weight_{a.lower()}": pre_weights[a] for a in ASSETS},
                **{f"target_weight_{a.lower()}": target_for_execution[a] for a in ASSETS},
                "turnover": turnover,
                "transaction_cost": transaction_cost,
            }
        )
        strategy_returns.append(net_return)

        prev2_target_signal = prev_target_signal.copy() if prev_target_signal is not None else None
        prev_target_signal = signal_target.copy()
        actual_after_prev = actual_after.copy()

    daily_panel = pd.DataFrame(panel_rows)
    target_df = pd.DataFrame(target_rows)
    actual_df = pd.DataFrame(actual_rows)
    rebalance_df = pd.DataFrame(rebalance_rows)

    returns = daily_panel[["date", "RF_DAILY", "macro_regime_confirmed", "vix_overlay_confirmed"]].copy()
    returns["RULE_BASED_NEW"] = strategy_returns
    returns["SPY_ONLY"] = daily_panel["SPY_RET"]
    returns["STATIC_40_30_15_15"] = (
        0.40 * daily_panel["SPY_RET"]
        + 0.30 * daily_panel["IEF_RET"]
        + 0.15 * daily_panel["GOLD_RET"]
        + 0.15 * daily_panel["RF_DAILY"]
    )
    returns["CASH_ONLY"] = daily_panel["RF_DAILY"]
    old = load_old_rule_returns(returns["date"])
    if old is not None:
        returns["OLD_RULE_BASED"] = old

    strategies = CORE_STRATEGIES + (["OLD_RULE_BASED"] if "OLD_RULE_BASED" in returns.columns else [])
    equity = returns[["date"]].copy()
    for strat in strategies:
        equity[strat] = (1.0 + returns[strat].fillna(0.0)).cumprod()

    return daily_panel, returns, equity, target_df, actual_df, rebalance_df


def drawdown_from_returns(returns: pd.Series) -> pd.Series:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    return wealth / wealth.cummax() - 1.0


def performance_stats(name: str, returns: pd.Series, rf: pd.Series, panel: pd.DataFrame | None = None) -> dict[str, float | str]:
    s = returns.dropna()
    if s.empty:
        return {"strategy": name}
    rf_aligned = rf.loc[s.index]
    excess = s - rf_aligned
    wealth = (1.0 + s).cumprod()
    ann_return = float(wealth.iloc[-1] ** (252.0 / len(s)) - 1.0)
    ann_vol = float(s.std(ddof=1) * np.sqrt(252.0)) if len(s) > 1 else np.nan
    excess_std = excess.std(ddof=1)
    sharpe = 0.0 if name == "CASH_ONLY" else float(excess.mean() / excess_std * np.sqrt(252.0)) if pd.notna(excess_std) and excess_std != 0 else np.nan
    downside = excess[excess < 0].std(ddof=1)
    sortino = 0.0 if name == "CASH_ONLY" else float(excess.mean() / downside * np.sqrt(252.0)) if pd.notna(downside) and downside != 0 else np.nan
    dd = drawdown_from_returns(s)
    mdd = float(dd.min())
    years = len(s) / 252.0
    row = {
        "strategy": name,
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "max_drawdown": mdd,
        "Calmar": float(ann_return / abs(mdd)) if mdd < 0 else np.nan,
        "turnover": np.nan,
        "transaction_cost_drag": np.nan,
        "final_nav": float(wealth.iloc[-1]),
        "positive_day_ratio": float((s > 0).mean()),
        "worst_day": float(s.min()),
        "best_day": float(s.max()),
        "annualized_excess_return": 0.0 if name == "CASH_ONLY" else float((1.0 + excess.mean()) ** 252 - 1.0),
    }
    if panel is not None and name == "RULE_BASED_NEW":
        row["turnover"] = float(panel["turnover"].sum())
        row["transaction_cost_drag"] = float(panel["transaction_cost"].sum())
        row["average_spy_weight"] = float(panel["actual_weight_spy_before_return"].mean())
        row["average_ief_weight"] = float(panel["actual_weight_ief_before_return"].mean())
        row["average_gold_weight"] = float(panel["actual_weight_gold_before_return"].mean())
        row["average_cmdty_fut_weight"] = float(panel["actual_weight_cmdty_fut_before_return"].mean())
        row["average_cash_weight"] = float(panel["actual_weight_cash_before_return"].mean())
        row["annual_transaction_cost_drag"] = float(panel["transaction_cost"].sum() / years) if years > 0 else np.nan
    return row


def summarize_performance(returns: pd.DataFrame, daily_panel: pd.DataFrame, rebalance_log: pd.DataFrame) -> pd.DataFrame:
    strategies = [c for c in returns.columns if c not in ["date", "RF_DAILY", "macro_regime_confirmed", "vix_overlay_confirmed"]]
    rows = []
    rule_panel = daily_panel.copy()
    if "turnover" not in rule_panel.columns and "turnover" in rebalance_log.columns:
        rule_panel = rule_panel.merge(rebalance_log[["date", "turnover"]], on="date", how="left")
    if "transaction_cost" not in rule_panel.columns and "transaction_cost" in rebalance_log.columns:
        rule_panel = rule_panel.merge(rebalance_log[["date", "transaction_cost"]], on="date", how="left")
    for strat in strategies:
        rows.append(performance_stats(strat, returns[strat], returns["RF_DAILY"], rule_panel if strat == "RULE_BASED_NEW" else None))
    return pd.DataFrame(rows)


def conditional_performance(returns: pd.DataFrame, daily_panel: pd.DataFrame, group_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategies = [c for c in returns.columns if c not in ["date", "RF_DAILY", "macro_regime_confirmed", "vix_overlay_confirmed"]]
    merged = returns.merge(
        daily_panel[
            [
                "date",
                "actual_weight_spy_before_return",
                "actual_weight_ief_before_return",
                "actual_weight_gold_before_return",
                "actual_weight_cmdty_fut_before_return",
                "actual_weight_cash_before_return",
            ]
        ],
        on="date",
        how="left",
    )
    perf_rows = []
    avg_rows = []
    for group, grp in merged.groupby(group_col, observed=False):
        for strat in strategies:
            row = performance_stats(strat, grp[strat], grp["RF_DAILY"])
            row[group_col] = group
            perf_rows.append(row)
        avg_rows.append(
            {
                group_col: group,
                "avg_spy_weight": grp["actual_weight_spy_before_return"].mean(),
                "avg_ief_weight": grp["actual_weight_ief_before_return"].mean(),
                "avg_gold_weight": grp["actual_weight_gold_before_return"].mean(),
                "avg_cmdty_fut_weight": grp["actual_weight_cmdty_fut_before_return"].mean(),
                "avg_cash_weight": grp["actual_weight_cash_before_return"].mean(),
            }
        )
    return pd.DataFrame(perf_rows), pd.DataFrame(avg_rows)


def build_cross_state_performance(returns: pd.DataFrame) -> pd.DataFrame:
    temp = returns.copy()
    temp["cross_state"] = temp["macro_regime_confirmed"] + "_" + temp["vix_overlay_confirmed"]
    strategies = [c for c in returns.columns if c not in ["date", "RF_DAILY", "macro_regime_confirmed", "vix_overlay_confirmed"]]
    rows = []
    for cross, grp in temp.groupby("cross_state", observed=False):
        for strat in strategies:
            row = performance_stats(strat, grp[strat], grp["RF_DAILY"])
            row["cross_state"] = cross
            rows.append(row)
    return pd.DataFrame(rows)


def compute_vix_episodes(daily_panel: pd.DataFrame) -> pd.DataFrame:
    data = daily_panel[["date", "VIX_LEVEL", "vix_overlay_confirmed", "SPY_RET", "portfolio_return_net"]].copy()
    stress = data["vix_overlay_confirmed"] == "STRESS"
    grp_id = (stress != stress.shift()).cumsum()
    rows = []
    for _, grp in data.loc[stress].groupby(grp_id):
        start = grp["date"].iloc[0]
        end = grp["date"].iloc[-1]
        rows.append(
            {
                "start_date": start,
                "end_date": end,
                "n_days": int(len(grp)),
                "peak_vix": float(grp["VIX_LEVEL"].max()),
                "spy_return_during_episode": float((1.0 + grp["SPY_RET"]).prod() - 1.0),
                "strategy_return_during_episode": float((1.0 + grp["portfolio_return_net"]).prod() - 1.0),
                "spy_max_drawdown_during_episode": float(drawdown_from_returns(grp["SPY_RET"]).min()),
                "strategy_max_drawdown_during_episode": float(drawdown_from_returns(grp["portfolio_return_net"]).min()),
                "avg_spy_weight": float(daily_panel.loc[grp.index, "actual_weight_spy_before_return"].mean()),
                "avg_ief_weight": float(daily_panel.loc[grp.index, "actual_weight_ief_before_return"].mean()),
                "avg_gold_weight": float(daily_panel.loc[grp.index, "actual_weight_gold_before_return"].mean()),
                "avg_cmdty_fut_weight": float(daily_panel.loc[grp.index, "actual_weight_cmdty_fut_before_return"].mean()),
                "avg_cash_weight": float(daily_panel.loc[grp.index, "actual_weight_cash_before_return"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_period_diagnostics(returns: pd.DataFrame) -> pd.DataFrame:
    strategies = [c for c in returns.columns if c not in ["date", "RF_DAILY", "macro_regime_confirmed", "vix_overlay_confirmed"]]
    rows = []
    for label, (start, end) in PERIOD_WINDOWS.items():
        mask = (returns["date"] >= pd.Timestamp(start)) & (returns["date"] <= pd.Timestamp(end))
        grp = returns.loc[mask]
        for strat in strategies:
            row = performance_stats(strat, grp[strat], grp["RF_DAILY"])
            row["period"] = label
            rows.append(row)
    return pd.DataFrame(rows)


def build_validation_report(daily_panel: pd.DataFrame, rebalance_log: pd.DataFrame, common_start: str) -> pd.DataFrame:
    signal_sum = daily_panel[[f"target_weight_{a.lower()}" for a in ASSETS]].sum(axis=1)
    before_sum = daily_panel[[f"actual_weight_{a.lower()}_before_return" for a in ASSETS]].sum(axis=1)
    after_sum = daily_panel[[f"actual_weight_{a.lower()}_after_return" for a in ASSETS]].sum(axis=1)
    raw_switches = int((daily_panel["macro_regime_raw"] != daily_panel["macro_regime_raw"].shift()).sum() - 1)
    confirmed_switches = int(daily_panel["macro_regime_switch_flag"].sum())
    reduction = float(1.0 - confirmed_switches / raw_switches) if raw_switches > 0 else np.nan
    regime_runs = (daily_panel["macro_regime_confirmed"] != daily_panel["macro_regime_confirmed"].shift()).cumsum()
    avg_duration = float(daily_panel.groupby(regime_runs).size().mean())
    vix_switches = int(daily_panel["vix_switch_flag"].sum())
    total_turnover = float(rebalance_log["turnover"].sum())
    years = len(daily_panel) / 252.0
    annual_tcost = float(rebalance_log["transaction_cost"].sum() / years) if years > 0 else np.nan

    rows = [
        ("weights_sum_to_one_signal", bool(np.allclose(signal_sum, 1.0))),
        ("weights_sum_to_one_before_return", bool(np.allclose(before_sum, 1.0))),
        ("weights_sum_to_one_after_return", bool(np.allclose(after_sum, 1.0))),
        ("no_negative_target_weights", bool((daily_panel[[f"target_weight_{a.lower()}" for a in ASSETS]] >= -1e-12).all().all())),
        ("no_negative_actual_weights", bool((daily_panel[[f"actual_weight_{a.lower()}_before_return" for a in ASSETS]] >= -1e-12).all().all())),
        ("macro_confirmation_days", CONFIRMATION_DAYS),
        ("vix_confirmation_days", CONFIRMATION_DAYS),
        ("raw_macro_regime_switch_count", raw_switches),
        ("confirmed_macro_regime_switch_count", confirmed_switches),
        ("raw_vs_confirmed_switch_reduction_ratio", reduction),
        ("average_confirmed_macro_regime_duration_days", avg_duration),
        ("vix_overlay_switch_count", vix_switches),
        ("total_turnover", total_turnover),
        ("annual_transaction_cost_drag", annual_tcost),
        ("average_cash_weight", float(daily_panel["actual_weight_cash_before_return"].mean())),
        ("average_commodity_weight", float(daily_panel["actual_weight_cmdty_fut_before_return"].mean())),
        ("macro_event_rebalances", int(rebalance_log["rebalance_reason"].fillna("").str.contains("macro_regime_change").sum())),
        ("common_sample_start_date", common_start),
        ("backtest_start_date", str(daily_panel["date"].min().date())),
        ("backtest_end_date", str(daily_panel["date"].max().date())),
        ("trading_days", int(len(daily_panel))),
    ]
    return pd.DataFrame(rows, columns=["check", "value"])


def plot_heatmap(df: pd.DataFrame, title: str, path: Path, fmt: str, center: float | None = None) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(max(10, 0.7 * df.shape[1]), max(4.5, 0.45 * df.shape[0])))
    sns.heatmap(df, annot=True, fmt=fmt, cmap="RdBu_r", center=center, linewidths=0.5, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_outputs(returns: pd.DataFrame, equity: pd.DataFrame, daily_panel: pd.DataFrame, perf: pd.DataFrame, avg_macro: pd.DataFrame, avg_vix: pd.DataFrame, cross_perf: pd.DataFrame) -> None:
    strategies = [c for c in equity.columns if c != "date"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for strat in strategies:
        ax.plot(equity["date"], equity[strat], label=strat)
    ax.set_title("Rule-Based Allocation Equity Curve V2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "equity_curve_full_v2.png", dpi=180)
    ax.set_yscale("log")
    ax.set_title("Rule-Based Allocation Equity Curve V2, Log Scale")
    fig.savefig(FIGURES_DIR / "equity_curve_log_v2.png", dpi=180)
    plt.close(fig)

    draw = pd.DataFrame({"date": returns["date"]})
    dd_strats = [s for s in ["RULE_BASED_NEW", "SPY_ONLY", "STATIC_40_30_15_15", "OLD_RULE_BASED"] if s in returns.columns]
    for strat in dd_strats:
        draw[strat] = drawdown_from_returns(returns[strat])
    draw.to_csv(DRAW_PATH, index=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    for strat in dd_strats:
        ax.plot(draw["date"], draw[strat], label=strat)
    ax.set_title("Drawdown Comparison V2")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "drawdown_comparison_v2.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(
        daily_panel["date"],
        daily_panel["actual_weight_spy_before_return"],
        daily_panel["actual_weight_ief_before_return"],
        daily_panel["actual_weight_gold_before_return"],
        daily_panel["actual_weight_cmdty_fut_before_return"],
        daily_panel["actual_weight_cash_before_return"],
        labels=["SPY", "IEF", "GOLD", "CMDTY_FUT", "CASH"],
    )
    ax.set_title("Actual Weights Over Time V2")
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "actual_weights_over_time_v2.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(
        daily_panel["date"],
        daily_panel["target_weight_spy"],
        daily_panel["target_weight_ief"],
        daily_panel["target_weight_gold"],
        daily_panel["target_weight_cmdty_fut"],
        daily_panel["target_weight_cash"],
        labels=["SPY", "IEF", "GOLD", "CMDTY_FUT", "CASH"],
    )
    ax.set_title("Target Weights Over Time V2")
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "target_weights_over_time_v2.png", dpi=180)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(daily_panel["date"], daily_panel["VIX_LEVEL"], color="tab:red", linewidth=1.0)
    ax1.axhline(20, color="gray", linestyle="--", linewidth=0.8)
    ax1.axhline(25, color="gray", linestyle=":", linewidth=0.8)
    ax2 = ax1.twinx()
    ax2.plot(daily_panel["date"], daily_panel["actual_weight_spy_before_return"], color="tab:blue", linewidth=1.0)
    ax1.set_title("VIX and SPY Weight V2")
    ax1.set_ylabel("VIX")
    ax2.set_ylabel("SPY weight")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "vix_vs_spy_weight_v2.png", dpi=180)
    plt.close(fig)

    regime_colors = {"HIGH_INFLATION": "#ef8a62", "INVERTED": "#2166ac", "FLAT": "#67a9cf", "STEEP": "#1b7837", "NEUTRAL": "#cccccc"}
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily_panel["date"], daily_panel["VIX_LEVEL"], color="black", linewidth=0.7)
    ymax = float(daily_panel["VIX_LEVEL"].max())
    for regime in MACRO_ORDER:
        mask = daily_panel["macro_regime_confirmed"] == regime
        ax.fill_between(daily_panel["date"], 0, ymax, where=mask, alpha=0.12, color=regime_colors[regime], label=regime)
    ax.set_title("Confirmed Macro Regime and VIX Timeline V2")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "macro_regime_vix_timeline_v2.png", dpi=180)
    plt.close(fig)

    metrics = ["annualized_return", "Sharpe", "max_drawdown"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes, metrics):
        sns.barplot(data=perf, x="strategy", y=metric, ax=ax)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "performance_summary_bar_charts_v2.png", dpi=180)
    plt.close(fig)

    macro_heat = avg_macro.set_index("macro_regime_confirmed")[["avg_spy_weight", "avg_ief_weight", "avg_gold_weight", "avg_cmdty_fut_weight", "avg_cash_weight"]]
    vix_heat = avg_vix.set_index("vix_overlay_confirmed")[["avg_spy_weight", "avg_ief_weight", "avg_gold_weight", "avg_cmdty_fut_weight", "avg_cash_weight"]]
    plot_heatmap(macro_heat, "Average Weights by Macro Regime V2", FIGURES_DIR / "average_weights_by_macro_regime_heatmap_v2.png", ".1%")
    plot_heatmap(vix_heat, "Average Weights by VIX Overlay State V2", FIGURES_DIR / "average_weights_by_vix_overlay_state_heatmap_v2.png", ".1%")

    for metric, path, fmt, center in [
        ("annualized_return", FIGURES_DIR / "cross_state_annualized_return_heatmap_v2.png", ".2%", 0),
        ("Sharpe", FIGURES_DIR / "cross_state_sharpe_heatmap_v2.png", ".2f", 0),
        ("max_drawdown", FIGURES_DIR / "cross_state_max_drawdown_heatmap_v2.png", ".2%", None),
    ]:
        pivot = cross_perf.pivot_table(index="strategy", columns="cross_state", values=metric, aggfunc="first")
        plot_heatmap(pivot, f"{metric.replace('_', ' ').title()} by Cross-State V2", path, fmt, center=center)


def write_report(perf: pd.DataFrame, validation: pd.DataFrame, common_start: str, sources: dict[str, str]) -> None:
    rule = perf.loc[perf["strategy"] == "RULE_BASED_NEW"].iloc[0]
    spy = perf.loc[perf["strategy"] == "SPY_ONLY"].iloc[0]
    static = perf.loc[perf["strategy"] == "STATIC_40_30_15_15"].iloc[0]
    lines = [
        "# Rule-Based Backtest Report V2",
        "",
        "## Strategy purpose",
        "",
        "This version extends the rule-based allocation framework with CMDTY_FUT, corrected three-day macro confirmation, three-day VIX overlay confirmation, and regime-specific normal-to-stress interpolation.",
        "",
        "## Data",
        "",
        f"- Asset daily returns: `{sources['daily_returns']}`",
        f"- Asset daily close: `{sources['daily_close']}`",
        f"- VIX: `{sources['VIX_LEVEL']}`",
        f"- DGS1/DGS10: `{sources['DGS1']}`, `{sources['DGS10']}`",
        f"- Credit spread: `{sources['CREDIT_SPREAD_BAA_AAA']}`",
        f"- DTB3 risk-free source: `{sources['RF_DAILY']}`",
        "- Cash uses `daily_rf = (1 + DTB3 / 100) ** (1 / 252) - 1`.",
        f"- Common sample start date: `{common_start}`.",
        "",
        "## Regime rules",
        "",
        "- HIGH_INFLATION: credit spread > 1.5 and DGS1 > 5",
        "- INVERTED: term spread < 0",
        "- FLAT: 0 <= term spread < 1",
        "- STEEP: term spread >= 1",
        "- NEUTRAL: fallback",
        f"- Macro regime uses {CONFIRMATION_DAYS}-trading-day confirmation.",
        "",
        "## VIX overlay",
        "",
        "- NORMAL: VIX < 20",
        "- WARNING: 20 <= VIX < 25",
        "- STRESS: VIX >= 25",
        f"- Overlay changes also require {CONFIRMATION_DAYS}-trading-day confirmation.",
        "- In WARNING and RECOVERY, `progress = clip((VIX - 20) / 5, 0, 1)` and target weights interpolate linearly between normal and stress allocations.",
        "",
        "## Transaction cost",
        "",
        f"- Transaction cost is {TCOST_BPS_PER_TOTAL_TRADED_NOTIONAL:.1f} bps per total traded notional at rebalance.",
        "",
        "## Results",
        "",
        f"- RULE_BASED_NEW annualized return: {rule['annualized_return']:.2%}",
        f"- RULE_BASED_NEW Sharpe: {rule['Sharpe']:.2f}",
        f"- RULE_BASED_NEW max drawdown: {rule['max_drawdown']:.2%}",
        f"- SPY_ONLY annualized return: {spy['annualized_return']:.2%}",
        f"- STATIC_40_30_15_15 annualized return: {static['annualized_return']:.2%}",
        "",
        "See the saved CSVs and figures for crisis-period and cross-state diagnostics.",
        "",
        "## Caveats",
        "",
        "- This is still a fixed-threshold diagnostic strategy, not a fully validated production rule set.",
        "- The sample starts when SPY, IEF, GLD, and GD=F are all available, so long-history inflation regimes remain outside the implementable sample.",
        "- Rebalancing is monthly plus event-driven and progress-driven; alternative execution schedules should be tested later.",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel, sources = build_daily_panel()
    common_start = str(panel["date"].min().date())

    panel["macro_regime_raw"] = panel.apply(raw_macro_regime, axis=1)
    macro_conf = confirm_regime(panel["macro_regime_raw"], confirmation_days=CONFIRMATION_DAYS, initial_confirmed="NEUTRAL")
    panel["macro_regime_candidate"] = macro_conf["candidate"]
    panel["macro_regime_candidate_count"] = macro_conf["candidate_count"]
    panel["macro_regime_confirmed"] = macro_conf["confirmed"]
    panel["macro_regime_switch_flag"] = macro_conf["switch_flag"]

    vix_conf = build_vix_overlay(panel, confirmation_days=CONFIRMATION_DAYS)
    panel = pd.concat([panel.reset_index(drop=True), vix_conf], axis=1)
    panel = add_signal_columns(panel)

    daily_panel, returns, equity, target_df, actual_df, rebalance_df = compute_backtest(panel)
    daily_panel.to_csv(PANEL_PATH, index=False)
    returns.to_csv(RETURNS_PATH, index=False)
    equity.to_csv(EQUITY_PATH, index=False)
    target_df.to_csv(TARGET_WEIGHTS_PATH, index=False)
    actual_df.to_csv(ACTUAL_WEIGHTS_PATH, index=False)
    rebalance_df.to_csv(REBALANCE_LOG_PATH, index=False)

    perf = summarize_performance(returns, daily_panel, rebalance_df)
    perf.to_csv(PERFORMANCE_PATH, index=False)

    perf_macro, avg_macro = conditional_performance(returns, daily_panel, "macro_regime_confirmed")
    perf_vix, avg_vix = conditional_performance(returns, daily_panel, "vix_overlay_confirmed")
    cross_perf = build_cross_state_performance(returns)
    perf_macro.to_csv(PERF_MACRO_PATH, index=False)
    avg_macro.to_csv(AVG_MACRO_PATH, index=False)
    avg_vix.to_csv(AVG_VIX_PATH, index=False)
    cross_perf.to_csv(PERF_CROSS_PATH, index=False)

    episodes = compute_vix_episodes(daily_panel)
    episodes.to_csv(EPISODE_PATH, index=False)
    period_diag = build_period_diagnostics(returns)
    period_diag.to_csv(PERIOD_PATH, index=False)

    validation = build_validation_report(daily_panel, rebalance_df, common_start)
    validation.to_csv(VALIDATION_PATH, index=False)

    plot_outputs(returns, equity, daily_panel, perf, avg_macro, avg_vix, cross_perf)
    write_report(perf, validation, common_start, sources)

    raw_switches = int(validation.loc[validation["check"] == "raw_macro_regime_switch_count", "value"].iloc[0])
    confirmed_switches = int(validation.loc[validation["check"] == "confirmed_macro_regime_switch_count", "value"].iloc[0])
    reduction = float(validation.loc[validation["check"] == "raw_vs_confirmed_switch_reduction_ratio", "value"].iloc[0])
    avg_duration = float(validation.loc[validation["check"] == "average_confirmed_macro_regime_duration_days", "value"].iloc[0])
    vix_switches = int(validation.loc[validation["check"] == "vix_overlay_switch_count", "value"].iloc[0])
    total_turnover = float(validation.loc[validation["check"] == "total_turnover", "value"].iloc[0])
    annual_tcost = float(validation.loc[validation["check"] == "annual_transaction_cost_drag", "value"].iloc[0])
    avg_cash = float(validation.loc[validation["check"] == "average_cash_weight", "value"].iloc[0])
    avg_cmdty = float(validation.loc[validation["check"] == "average_commodity_weight", "value"].iloc[0])

    print(f"Backtest start/end date: {daily_panel['date'].min().date()} to {daily_panel['date'].max().date()}")
    print(f"Common sample start date: {common_start}")
    print(f"Trading days: {len(daily_panel)}")
    print(f"Raw macro regime switch count: {raw_switches}")
    print(f"Confirmed macro regime switch count: {confirmed_switches}")
    print(f"Raw vs confirmed switch reduction ratio: {reduction:.2%}")
    print(f"Average duration of confirmed macro regimes: {avg_duration:.2f} trading days")
    print(f"VIX overlay switch count: {vix_switches}")
    print(f"Total turnover: {total_turnover:.4f}")
    print(f"Annual transaction cost drag: {annual_tcost:.6f}")
    print(f"Average cash weight: {avg_cash:.2%}")
    print(f"Average commodity weight: {avg_cmdty:.2%}")
    print("Performance summary:")
    print(perf[["strategy", "annualized_return", "Sharpe", "max_drawdown", "Calmar", "turnover", "transaction_cost_drag", "final_nav"]].to_string(index=False))
    print("Period diagnostics:")
    print(period_diag[["period", "strategy", "annualized_return", "Sharpe", "max_drawdown", "final_nav"]].to_string(index=False))

    for path in [
        PANEL_PATH,
        RETURNS_PATH,
        EQUITY_PATH,
        TARGET_WEIGHTS_PATH,
        ACTUAL_WEIGHTS_PATH,
        REBALANCE_LOG_PATH,
        PERFORMANCE_PATH,
        AVG_MACRO_PATH,
        AVG_VIX_PATH,
        PERF_MACRO_PATH,
        PERF_CROSS_PATH,
        DRAW_PATH,
        VALIDATION_PATH,
        EPISODE_PATH,
        PERIOD_PATH,
        REPORT_PATH,
        FIGURES_DIR / "equity_curve_full_v2.png",
        FIGURES_DIR / "equity_curve_log_v2.png",
        FIGURES_DIR / "drawdown_comparison_v2.png",
        FIGURES_DIR / "actual_weights_over_time_v2.png",
        FIGURES_DIR / "target_weights_over_time_v2.png",
        FIGURES_DIR / "vix_vs_spy_weight_v2.png",
        FIGURES_DIR / "macro_regime_vix_timeline_v2.png",
        FIGURES_DIR / "average_weights_by_macro_regime_heatmap_v2.png",
        FIGURES_DIR / "average_weights_by_vix_overlay_state_heatmap_v2.png",
        FIGURES_DIR / "performance_summary_bar_charts_v2.png",
        FIGURES_DIR / "cross_state_annualized_return_heatmap_v2.png",
        FIGURES_DIR / "cross_state_sharpe_heatmap_v2.png",
        FIGURES_DIR / "cross_state_max_drawdown_heatmap_v2.png",
    ]:
        if path.exists():
            print(f"Saved: {path}")


if __name__ == "__main__":
    main()
