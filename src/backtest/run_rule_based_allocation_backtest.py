from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "rule_based_backtest"
FIGURES_DIR = ROOT / "figures" / "rule_based_backtest"

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

PANEL_PATH = RESULTS_DIR / "rule_based_daily_backtest_panel.csv"
RETURNS_PATH = RESULTS_DIR / "rule_based_strategy_daily_returns.csv"
EQUITY_PATH = RESULTS_DIR / "rule_based_strategy_equity_curves.csv"
WEIGHTS_PATH = RESULTS_DIR / "rule_based_strategy_daily_weights.csv"
TARGET_WEIGHTS_PATH = RESULTS_DIR / "daily_target_weights.csv"
ACTUAL_WEIGHTS_PATH = RESULTS_DIR / "daily_actual_weights.csv"
REBALANCE_LOG_PATH = RESULTS_DIR / "daily_rebalance_log.csv"
PERFORMANCE_PATH = RESULTS_DIR / "performance_summary.csv"
PERF_MACRO_PATH = RESULTS_DIR / "performance_by_macro_regime.csv"
PERF_VIX_PATH = RESULTS_DIR / "performance_by_vix_overlay_state.csv"
AVG_W_MACRO_PATH = RESULTS_DIR / "average_weights_by_macro_regime.csv"
AVG_W_VIX_PATH = RESULTS_DIR / "average_weights_by_vix_state.csv"
DRAWDOWN_PATH = RESULTS_DIR / "drawdown_series.csv"
EPISODE_PATH = RESULTS_DIR / "vix_stress_episode_diagnostics.csv"
VALIDATION_PATH = RESULTS_DIR / "backtest_validation_report.csv"
REPORT_PATH = RESULTS_DIR / "RULE_BASED_BACKTEST_REPORT.md"

ASSETS = ["SPY", "IEF", "GOLD", "CASH"]
STRATEGIES = ["RULE_BASED", "SPY_ONLY", "STATIC_40_30_15_15", "STATIC_RULE_ASSET_MIX", "CASH_ONLY"]
MACRO_ORDER = ["HIGH_INFLATION", "INVERTED", "FLAT", "STEEP", "NEUTRAL"]
VIX_ORDER = ["NORMAL", "WARNING", "STRESS", "RECOVERY"]

BASE_ALLOCATIONS = {
    "HIGH_INFLATION": {"SPY": 0.30, "IEF": 0.00, "GOLD": 0.00, "CASH": 0.70},
    "INVERTED": {"SPY": 0.30, "IEF": 0.20, "GOLD": 0.30, "CASH": 0.20},
    "FLAT": {"SPY": 0.40, "IEF": 0.00, "GOLD": 0.40, "CASH": 0.20},
    "STEEP": {"SPY": 0.70, "IEF": 0.00, "GOLD": 0.00, "CASH": 0.30},
    "NEUTRAL": {"SPY": 0.40, "IEF": 0.20, "GOLD": 0.20, "CASH": 0.20},
}

STRESS_ALLOCATIONS = {
    "HIGH_INFLATION": {"SPY": 0.20, "IEF": 0.00, "GOLD": 0.00, "CASH": 0.80},
    "INVERTED": {"SPY": 0.20, "IEF": 0.20, "GOLD": 0.30, "CASH": 0.30},
    "FLAT": {"SPY": 0.20, "IEF": 0.00, "GOLD": 0.40, "CASH": 0.40},
    "STEEP": {"SPY": 0.20, "IEF": 0.30, "GOLD": 0.30, "CASH": 0.20},
    "NEUTRAL": {"SPY": 0.20, "IEF": 0.25, "GOLD": 0.25, "CASH": 0.30},
}

CONFIRMATION_DAYS = 2


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


def load_dtb3() -> tuple[pd.DataFrame, Path]:
    path = next((p for p in DTB3_CANDIDATES if p.exists()), None)
    if path is None:
        raise FileNotFoundError("DTB3.csv not found in expected paths.")
    rf = read_fred_csv(path, "DTB3")
    rf["DTB3_RATE"] = rf["DTB3"] / 100.0
    rf["RF_DAILY"] = (1.0 + rf["DTB3_RATE"].ffill()) ** (1.0 / 252.0) - 1.0
    return rf[["date", "DTB3", "DTB3_RATE", "RF_DAILY"]], path


def build_daily_panel() -> tuple[pd.DataFrame, dict[str, str]]:
    prices = pd.read_csv(DAILY_CLOSE_PATH)
    prices["date"] = pd.to_datetime(prices["date"])
    required_assets = ["SPY", "IEF", "GLD"]
    missing_assets = [a for a in required_assets if a not in prices.columns]
    if missing_assets:
        raise ValueError(f"Missing asset price columns: {missing_assets}")
    prices = prices[["date", "SPY", "IEF", "GLD"]].rename(columns={"GLD": "GOLD"})
    returns = prices.copy()
    for asset in ["SPY", "IEF", "GOLD"]:
        returns[f"{asset}_RET"] = returns[asset].pct_change()

    vix = read_fred_csv(VIX_PATH, "VIX_LEVEL")
    dgs1 = read_fred_csv(DGS1_PATH, "DGS1")
    dgs10 = read_fred_csv(DGS10_PATH, "DGS10")
    waaa = read_fred_csv(WAAA_PATH, "WAAA")
    wbaa = read_fred_csv(WBAA_PATH, "WBAA")
    credit = waaa.merge(wbaa, on="date", how="outer").sort_values("date")
    credit[["WAAA", "WBAA"]] = credit[["WAAA", "WBAA"]].ffill()
    credit["CREDIT_SPREAD_BAA_AAA"] = credit["WBAA"] - credit["WAAA"]
    rf, rf_path = load_dtb3()

    panel = returns[["date", "SPY_RET", "IEF_RET", "GOLD_RET"]].copy()
    for frame in [vix, dgs1, dgs10, credit[["date", "CREDIT_SPREAD_BAA_AAA"]], rf]:
        panel = panel.merge(frame, on="date", how="left")
    for col in ["VIX_LEVEL", "DGS1", "DGS10", "CREDIT_SPREAD_BAA_AAA", "DTB3", "DTB3_RATE", "RF_DAILY"]:
        panel[col] = panel[col].ffill()
    panel["TERM_SPREAD_10Y_1Y"] = panel["DGS10"] - panel["DGS1"]

    before = len(panel)
    required = ["SPY_RET", "IEF_RET", "GOLD_RET", "RF_DAILY", "VIX_LEVEL", "DGS1", "DGS10", "CREDIT_SPREAD_BAA_AAA", "TERM_SPREAD_10Y_1Y"]
    panel = panel.dropna(subset=required).copy()
    panel["missing_observations_dropped"] = before - len(panel)
    sources = {
        "SPY/IEF/GOLD": str(DAILY_CLOSE_PATH),
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
    if row["TERM_SPREAD_10Y_1Y"] < 1:
        return "FLAT"
    return "STEEP"


def confirm_macro_regime(raw: pd.Series, confirmation_days: int = 3, default: str = "NEUTRAL") -> pd.DataFrame:
    confirmed_values: list[str] = []
    candidates: list[str] = []
    candidate_counts: list[int] = []
    switch_flags: list[bool] = []
    switch_reasons: list[str] = []
    current = default
    candidate = default
    count = 0
    for value in raw:
        switch_flag = False
        switch_reason = ""
        if value == current:
            candidate = current
            count = 0
        elif value == candidate:
            count += 1
        else:
            candidate = value
            count = 1
        if candidate is not None and count >= confirmation_days:
            previous = current
            current = candidate
            switch_flag = previous != current
            switch_reason = f"{previous}_to_{current}_confirmed_after_{confirmation_days}_days" if switch_flag else ""
            candidate = current
            count = 0
        confirmed_values.append(current)
        candidates.append(candidate)
        candidate_counts.append(count)
        switch_flags.append(switch_flag)
        switch_reasons.append(switch_reason)
    return pd.DataFrame(
        {
            "macro_regime_candidate": candidates,
            "macro_regime_candidate_count": candidate_counts,
            "macro_regime_confirmed": confirmed_values,
            "macro_regime_switch_flag": switch_flags,
            "macro_regime_switch_reason": switch_reasons,
        }
    )


def run_vix_state_machine(panel: pd.DataFrame, confirmation_days: int = 3, min_spy_weight: float = 0.20) -> pd.DataFrame:
    state = "NORMAL"
    episode_start = pd.NaT
    peak = np.nan
    above20 = below20 = above25 = below25 = 0
    rows = []
    for _, row in panel.iterrows():
        vix = row["VIX_LEVEL"]
        date = row["date"]
        above20 = above20 + 1 if vix > 20 else 0
        below20 = below20 + 1 if vix < 20 else 0
        above25 = above25 + 1 if vix >= 25 else 0
        below25 = below25 + 1 if vix < 25 else 0

        if state == "NORMAL":
            if above20 >= confirmation_days:
                state = "WARNING"
                episode_start = date
                peak = vix
        elif state == "WARNING":
            peak = max(peak, vix)
            if above25 >= confirmation_days:
                state = "STRESS"
            elif below20 >= confirmation_days:
                state = "NORMAL"
                episode_start = pd.NaT
                peak = np.nan
        elif state == "STRESS":
            peak = max(peak, vix)
            if below25 >= confirmation_days:
                state = "RECOVERY"
        elif state == "RECOVERY":
            peak = max(peak, vix)
            if above25 >= confirmation_days:
                state = "STRESS"
            elif below20 >= confirmation_days:
                state = "NORMAL"
                episode_start = pd.NaT
                peak = np.nan

        macro = row["macro_regime_confirmed"]
        base_spy = BASE_ALLOCATIONS[macro]["SPY"]
        de_risk_ratio = np.nan
        re_risk_ratio = np.nan
        if state == "NORMAL":
            target_spy = base_spy
        elif state == "WARNING":
            de_risk_ratio = float(np.clip(((peak if pd.notna(peak) else vix) - 20) / 5, 0, 1))
            target_spy = base_spy - (base_spy - min_spy_weight) * de_risk_ratio
        elif state == "STRESS":
            de_risk_ratio = 1.0
            target_spy = min_spy_weight
        else:
            re_risk_ratio = float(np.clip((25 - vix) / 5, 0, 1))
            target_spy = min_spy_weight + re_risk_ratio * (base_spy - min_spy_weight)

        rows.append(
            {
                "vix_overlay_state": state,
                "episode_start_date": episode_start,
                "vix_peak_since_episode_start": peak,
                "vix_warning_confirmed": state in ["WARNING", "STRESS", "RECOVERY"],
                "vix_stress_confirmed": state == "STRESS",
                "vix_crisis_flag": vix >= 30,
                "de_risk_ratio": de_risk_ratio,
                "re_risk_ratio": re_risk_ratio,
                "target_weight_spy_before_flow": target_spy,
            }
        )
    return pd.concat([panel.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def assign_weights(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in panel.iterrows():
        macro = row["macro_regime_confirmed"]
        base = BASE_ALLOCATIONS[macro]
        stress = STRESS_ALLOCATIONS[macro]
        base_spy = base["SPY"]
        target_spy = row["target_weight_spy_before_flow"]
        if base_spy > 0.20:
            intensity = float(np.clip((base_spy - target_spy) / (base_spy - 0.20), 0, 1))
        else:
            intensity = 0.0
        weights = {asset: base[asset] * (1 - intensity) + stress[asset] * intensity for asset in ASSETS}
        total = sum(weights.values())
        if not np.isclose(total, 1.0):
            weights = {k: v / total for k, v in weights.items()}
        rows.append(
            {
                "stress_intensity": intensity,
                **{f"base_weight_{a.lower()}": base[a] for a in ASSETS},
                **{f"stress_weight_{a.lower()}": stress[a] for a in ASSETS},
                **{f"final_weight_{a.lower()}_signal_date": weights[a] for a in ASSETS},
            }
        )
    return pd.concat([panel.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def is_first_trading_day_of_month(dates: pd.Series) -> pd.Series:
    months = dates.dt.to_period("M")
    return months != months.shift(1)


def compute_backtest(
    panel: pd.DataFrame,
    rebalance_frequency: str = "M",
    drift_threshold: float = 0.05,
    transaction_cost_bps: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = panel.copy()
    signal_cols = {asset: f"final_weight_{asset.lower()}_signal_date" for asset in ASSETS}
    used_cols = {asset: f"actual_weight_{asset.lower()}_before_return" for asset in ASSETS}
    after_cols = {asset: f"actual_weight_{asset.lower()}_after_return" for asset in ASSETS}
    ret_cols = {"SPY": "SPY_RET", "IEF": "IEF_RET", "GOLD": "GOLD_RET", "CASH": "RF_DAILY"}
    tc_rate = transaction_cost_bps / 10000.0
    monthly_rebalance = is_first_trading_day_of_month(p["date"])

    portfolio_value = 1.0
    actual_weights_after_prev = None
    actual_records = []
    target_records = []
    rebalance_records = []
    rule_returns = []
    rule_equity = []

    for i, row in p.iterrows():
        date = row["date"]
        if i == 0:
            target = {asset: row[signal_cols[asset]] for asset in ASSETS}
        else:
            target = {asset: p.loc[i - 1, signal_cols[asset]] for asset in ASSETS}

        if actual_weights_after_prev is None:
            pre_rebalance_weights = target.copy()
        else:
            pre_rebalance_weights = actual_weights_after_prev.copy()

        reasons = []
        if i == 0:
            reasons.append("initial")
        if rebalance_frequency.upper() == "D":
            reasons.append("daily_scheduled")
        elif rebalance_frequency.upper() == "M" and bool(monthly_rebalance.iloc[i]) and i > 0:
            reasons.append("monthly_scheduled")
        elif rebalance_frequency.upper() == "W" and i > 0 and date.week != p.loc[i - 1, "date"].week:
            reasons.append("weekly_scheduled")
        if i > 0:
            if i > 1 and p.loc[i - 1, "macro_regime_confirmed"] != p.loc[i - 2, "macro_regime_confirmed"]:
                reasons.append("macro_regime_change")
            if i > 1 and p.loc[i - 1, "vix_overlay_state"] != p.loc[i - 2, "vix_overlay_state"]:
                reasons.append("vix_state_change")
        max_drift = max(abs(pre_rebalance_weights[asset] - target[asset]) for asset in ASSETS)
        if max_drift > drift_threshold:
            reasons.append("drift_threshold")

        do_rebalance = len(reasons) > 0
        if do_rebalance:
            turnover = sum(abs(target[asset] - pre_rebalance_weights[asset]) for asset in ASSETS)
            actual_before = target.copy()
        else:
            turnover = 0.0
            actual_before = pre_rebalance_weights.copy()
        transaction_cost = turnover * tc_rate

        gross_ret = sum(actual_before[asset] * row[ret_cols[asset]] for asset in ASSETS)
        net_ret = gross_ret - transaction_cost
        sleeve_values = {asset: actual_before[asset] * (1 + row[ret_cols[asset]]) for asset in ASSETS}
        total_before_cost = sum(sleeve_values.values())
        actual_after = {asset: sleeve_values[asset] / total_before_cost for asset in ASSETS}
        portfolio_value *= 1 + net_ret

        target_records.append(
            {
                "date": date,
                "target_source_date": p.loc[i - 1, "date"] if i > 0 else date,
                **{f"target_weight_{asset.lower()}": target[asset] for asset in ASSETS},
            }
        )
        actual_records.append(
            {
                "date": date,
                **{used_cols[asset]: actual_before[asset] for asset in ASSETS},
                **{after_cols[asset]: actual_after[asset] for asset in ASSETS},
            }
        )
        rebalance_records.append(
            {
                "date": date,
                "rebalance_flag": do_rebalance,
                "rebalance_reason": "|".join(reasons) if reasons else "",
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "max_abs_drift_before_rebalance": max_drift,
            }
        )
        rule_returns.append(net_ret)
        rule_equity.append(portfolio_value)
        actual_weights_after_prev = actual_after

    actual = pd.DataFrame(actual_records)
    targets = pd.DataFrame(target_records)
    rebalance_log = pd.DataFrame(rebalance_records)
    p["RULE_BASED"] = rule_returns
    p["SPY_ONLY"] = p["SPY_RET"]
    p["STATIC_40_30_15_15"] = 0.40 * p["SPY_RET"] + 0.30 * p["IEF_RET"] + 0.15 * p["GOLD_RET"] + 0.15 * p["RF_DAILY"]
    p["STATIC_RULE_ASSET_MIX"] = 0.40 * p["SPY_RET"] + 0.20 * p["IEF_RET"] + 0.20 * p["GOLD_RET"] + 0.20 * p["RF_DAILY"]
    p["CASH_ONLY"] = p["RF_DAILY"]

    returns = p[["date", "RF_DAILY", "macro_regime_confirmed", "vix_overlay_state", *STRATEGIES]].copy()
    equity = returns[["date", *STRATEGIES]].copy()
    for strat in STRATEGIES:
        equity[strat] = (1 + returns[strat]).cumprod()
    equity["RULE_BASED"] = rule_equity
    weights = p[
        [
            "date",
            "macro_regime_raw",
            "macro_regime_candidate",
            "macro_regime_candidate_count",
            "macro_regime_confirmed",
            "macro_regime_switch_flag",
            "macro_regime_switch_reason",
            "vix_overlay_state",
            "vix_warning_confirmed",
            "vix_stress_confirmed",
            "vix_crisis_flag",
            "vix_peak_since_episode_start",
            "de_risk_ratio",
            "re_risk_ratio",
            "stress_intensity",
            "target_weight_spy_before_flow",
            *[f"final_weight_{a.lower()}_signal_date" for a in ASSETS],
        ]
    ].copy()
    weights = weights.merge(targets, on="date", how="left").merge(actual, on="date", how="left").merge(rebalance_log, on="date", how="left")
    return returns, equity, weights, targets, actual, rebalance_log


def drawdown(series: pd.Series) -> pd.Series:
    wealth = (1 + series.fillna(0)).cumprod()
    return wealth / wealth.cummax() - 1


def performance_stats(name: str, returns: pd.Series, rf: pd.Series, weights: pd.DataFrame | None = None) -> dict[str, float | str]:
    s = returns.dropna()
    rf_aligned = rf.loc[s.index]
    excess = s - rf_aligned
    wealth = (1 + s).cumprod()
    ann_return = float(wealth.iloc[-1] ** (252 / len(s)) - 1)
    ann_vol = float(s.std(ddof=1) * np.sqrt(252))
    ann_excess = float((1 + excess.mean()) ** 252 - 1)
    if name == "CASH_ONLY":
        sharpe = 0.0
        ann_excess = 0.0
    else:
        sharpe = float(excess.mean() / excess.std(ddof=1) * np.sqrt(252)) if excess.std(ddof=1) != 0 else np.nan
    downside = s[s < 0].std(ddof=1) * np.sqrt(252)
    mdd = float((wealth / wealth.cummax() - 1).min())
    row = {
        "strategy": name,
        "total_return": float(wealth.iloc[-1] - 1),
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "annualized_excess_return": ann_excess,
        "Sharpe": sharpe,
        "Sortino": float(ann_return / downside) if pd.notna(downside) and downside != 0 else np.nan,
        "max_drawdown": mdd,
        "Calmar": float(ann_return / abs(mdd)) if mdd < 0 else np.nan,
        "worst_day": float(s.min()),
        "best_day": float(s.max()),
        "positive_day_ratio": float((s > 0).mean()),
    }
    if weights is not None and name == "RULE_BASED":
        row["turnover"] = float(weights["turnover"].mean()) if "turnover" in weights.columns else np.nan
        row["total_turnover"] = float(weights["turnover"].sum()) if "turnover" in weights.columns else np.nan
        row["total_transaction_cost"] = float(weights["transaction_cost"].sum()) if "transaction_cost" in weights.columns else np.nan
        for asset in ASSETS:
            row[f"average_{asset.lower()}_weight"] = float(weights[f"actual_weight_{asset.lower()}_before_return"].mean())
    return row


def summarize_performance(returns: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    rows = [performance_stats(strat, returns[strat], returns["RF_DAILY"], weights if strat == "RULE_BASED" else None) for strat in STRATEGIES]
    return pd.DataFrame(rows)


def conditional_performance(returns: pd.DataFrame, weights: pd.DataFrame, group_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    avg_rows = []
    data = returns.merge(weights[["date", *[f"actual_weight_{a.lower()}_before_return" for a in ASSETS]]], on="date", how="left")
    for group, grp in data.groupby(group_col, observed=False):
        for strat in STRATEGIES:
            row = performance_stats(strat, grp[strat], grp["RF_DAILY"])
            row[group_col] = group
            rows.append(row)
        avg = {group_col: group}
        for asset in ASSETS:
            avg[f"avg_{asset.lower()}_weight"] = grp[f"actual_weight_{asset.lower()}_before_return"].mean()
        avg_rows.append(avg)
    return pd.DataFrame(rows), pd.DataFrame(avg_rows)


def vix_stress_episodes(panel: pd.DataFrame, returns: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    data = panel[["date", "VIX_LEVEL"]].merge(returns[["date", "SPY_ONLY", "RULE_BASED"]], on="date").merge(weights, on="date")
    data["stress"] = data["vix_overlay_state"] == "STRESS"
    groups = (data["stress"] != data["stress"].shift()).cumsum()
    rows = []
    for _, grp in data.loc[data["stress"]].groupby(groups):
        start = grp["date"].iloc[0]
        end = grp["date"].iloc[-1]
        idx_end = data.index[data["date"] == end][0]
        next_21 = data.iloc[idx_end + 1 : idx_end + 22]
        next_63 = data.iloc[idx_end + 1 : idx_end + 64]
        rows.append(
            {
                "start_date": start,
                "end_date": end,
                "n_days": len(grp),
                "peak_vix": grp["VIX_LEVEL"].max(),
                "spy_return_during_episode": (1 + grp["SPY_ONLY"]).prod() - 1,
                "strategy_return_during_episode": (1 + grp["RULE_BASED"]).prod() - 1,
                "spy_max_drawdown_during_episode": drawdown(grp["SPY_ONLY"]).min(),
                "strategy_max_drawdown_during_episode": drawdown(grp["RULE_BASED"]).min(),
                "avg_spy_weight": grp["actual_weight_spy_before_return"].mean(),
                "avg_ief_weight": grp["actual_weight_ief_before_return"].mean(),
                "avg_gold_weight": grp["actual_weight_gold_before_return"].mean(),
                "avg_cash_weight": grp["actual_weight_cash_before_return"].mean(),
                "strategy_return_1m_after_exit": (1 + next_21["RULE_BASED"]).prod() - 1 if not next_21.empty else np.nan,
                "strategy_return_3m_after_exit": (1 + next_63["RULE_BASED"]).prod() - 1 if not next_63.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def validation_report(panel: pd.DataFrame, weights: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    w_signal = weights[[f"final_weight_{a.lower()}_signal_date" for a in ASSETS]]
    w_used = weights[[f"actual_weight_{a.lower()}_before_return" for a in ASSETS]]
    w_after = weights[[f"actual_weight_{a.lower()}_after_return" for a in ASSETS]]
    raw_macro_switches = int((panel["macro_regime_raw"] != panel["macro_regime_raw"].shift()).sum() - 1)
    macro_switches = int(panel["macro_regime_switch_flag"].sum()) if "macro_regime_switch_flag" in panel.columns else int((panel["macro_regime_confirmed"] != panel["macro_regime_confirmed"].shift()).sum() - 1)
    vix_switches = int((weights["vix_overlay_state"] != weights["vix_overlay_state"].shift()).sum() - 1)
    confirmed_runs = (panel["macro_regime_confirmed"] != panel["macro_regime_confirmed"].shift()).cumsum()
    avg_confirmed_duration = float(panel.groupby(confirmed_runs).size().mean())
    macro_event_rebalances = int(weights["rebalance_reason"].fillna("").str.contains("macro_regime_change").sum()) if "rebalance_reason" in weights.columns else 0
    rows = [
        ("weights_sum_to_one_signal", bool(np.allclose(w_signal.sum(axis=1), 1.0))),
        ("actual_weights_sum_to_one_before_return", bool(np.allclose(w_used.sum(axis=1), 1.0))),
        ("actual_weights_sum_to_one_after_return", bool(np.allclose(w_after.sum(axis=1), 1.0))),
        ("no_negative_weights_signal", bool((w_signal >= -1e-12).all().all())),
        ("no_negative_actual_weights_before_return", bool((w_used >= -1e-12).all().all())),
        ("no_negative_actual_weights_after_return", bool((w_after >= -1e-12).all().all())),
        ("weights_shifted_one_day_for_returns", True),
        ("portfolio_drift_enabled", True),
        ("monthly_rebalance_first_trading_day", True),
        ("transaction_cost_bps_per_traded_notional", 5.0),
        ("rf_daily_used_for_cash_growth", True),
        ("rf_daily_used_for_sharpe_excess_return", True),
        ("backtest_start_date", str(panel["date"].min().date())),
        ("backtest_end_date", str(panel["date"].max().date())),
        ("trading_days", int(len(panel))),
        ("missing_observations_dropped", int(panel["missing_observations_dropped"].iloc[0] if "missing_observations_dropped" in panel else 0)),
        ("raw_macro_regime_switches", raw_macro_switches),
        ("macro_regime_switches", macro_switches),
        ("confirmed_macro_switch_reduction_ratio", float(1 - macro_switches / raw_macro_switches) if raw_macro_switches > 0 else np.nan),
        ("average_confirmed_macro_regime_duration_days", avg_confirmed_duration),
        ("event_rebalances_caused_by_macro_regime_change", macro_event_rebalances),
        ("vix_overlay_state_switches", vix_switches),
        ("vix_warning_days", int((weights["vix_overlay_state"] == "WARNING").sum())),
        ("vix_stress_days", int((weights["vix_overlay_state"] == "STRESS").sum())),
        ("vix_recovery_days", int((weights["vix_overlay_state"] == "RECOVERY").sum())),
        ("high_inflation_days", int((panel["macro_regime_confirmed"] == "HIGH_INFLATION").sum())),
    ]
    return pd.DataFrame(rows, columns=["check", "value"])


def plot_outputs(returns: pd.DataFrame, equity: pd.DataFrame, weights: pd.DataFrame, panel: pd.DataFrame, perf: pd.DataFrame, avg_macro: pd.DataFrame, avg_vix: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for strat in ["RULE_BASED", "SPY_ONLY", "STATIC_40_30_15_15", "CASH_ONLY"]:
        ax.plot(equity["date"], equity[strat], label=strat)
    ax.set_title("Rule-Based Allocation Equity Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "equity_curve_full.png", dpi=180)
    ax.set_yscale("log")
    ax.set_title("Rule-Based Allocation Equity Curve, Log Scale")
    fig.savefig(FIGURES_DIR / "equity_curve_log.png", dpi=180)
    plt.close(fig)

    dd = pd.DataFrame({"date": returns["date"]})
    for strat in ["RULE_BASED", "SPY_ONLY", "STATIC_40_30_15_15"]:
        dd[strat] = drawdown(returns[strat])
    dd.to_csv(DRAWDOWN_PATH, index=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    for strat in ["RULE_BASED", "SPY_ONLY", "STATIC_40_30_15_15"]:
        ax.plot(dd["date"], dd[strat], label=strat)
    ax.set_title("Drawdown Comparison")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "drawdown_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(
        weights["date"],
        weights["actual_weight_spy_before_return"],
        weights["actual_weight_ief_before_return"],
        weights["actual_weight_gold_before_return"],
        weights["actual_weight_cash_before_return"],
        labels=["SPY", "IEF", "GOLD", "CASH"],
    )
    ax.set_title("Strategy Actual Weights Before Return Over Time")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "strategy_weights_over_time.png", dpi=180)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(panel["date"], panel["VIX_LEVEL"], color="tab:red", label="VIX")
    ax1.axhline(20, color="gray", linestyle="--", linewidth=1)
    ax1.axhline(25, color="gray", linestyle=":", linewidth=1)
    ax2 = ax1.twinx()
    ax2.plot(weights["date"], weights["actual_weight_spy_before_return"], color="tab:blue", label="Actual SPY weight")
    ax1.set_title("VIX and Actual SPY Weight")
    ax1.set_ylabel("VIX")
    ax2.set_ylabel("SPY weight")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "vix_vs_spy_weight.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(panel["date"], panel["VIX_LEVEL"], color="black", linewidth=0.8)
    for macro in MACRO_ORDER:
        mask = panel["macro_regime_confirmed"] == macro
        ax.fill_between(panel["date"], 0, panel["VIX_LEVEL"].max(), where=mask, alpha=0.12, label=macro)
    ax.set_title("Confirmed Macro Regime and VIX")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "macro_regime_vix_timeline.png", dpi=180)
    plt.close(fig)

    metrics = ["annualized_return", "Sharpe", "max_drawdown"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes, metrics):
        sns.barplot(data=perf, x="strategy", y=metric, ax=ax)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "performance_summary_bar_charts.png", dpi=180)
    plt.close(fig)

    for avg, path, title in [
        (avg_macro, FIGURES_DIR / "average_weights_by_macro_regime_heatmap.png", "Average Weights by Macro Regime"),
        (avg_vix, FIGURES_DIR / "average_weights_by_vix_state_heatmap.png", "Average Weights by VIX Overlay State"),
    ]:
        idx_col = avg.columns[0]
        plot_df = avg.set_index(idx_col)[[f"avg_{a.lower()}_weight" for a in ASSETS]]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.heatmap(plot_df, annot=True, fmt=".1%", cmap="Blues", ax=ax)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)


def write_report(panel: pd.DataFrame, perf: pd.DataFrame, sources: dict[str, str]) -> None:
    best = perf.sort_values("Sharpe", ascending=False).iloc[0]
    strategy = perf.loc[perf["strategy"] == "RULE_BASED"].iloc[0]
    spy = perf.loc[perf["strategy"] == "SPY_ONLY"].iloc[0]
    static = perf.loc[perf["strategy"] == "STATIC_40_30_15_15"].iloc[0]
    lines = [
        "# Rule-Based Backtest Report",
        "",
        "## Strategy Purpose",
        "",
        "This is the first no-look-ahead rule-based allocation backtest based on observable macro/rate regimes and a VIX dynamic risk overlay. It is v0.1 and is not optimized.",
        "",
        "## Data",
        "",
        f"- SPY, IEF, GLD daily adjusted close: `{sources['SPY/IEF/GOLD']}`",
        f"- VIX: `{sources['VIX_LEVEL']}`",
        f"- DGS1/DGS10: `{sources['DGS1']}`, `{sources['DGS10']}`",
        f"- BAA-AAA credit spread: `{sources['CREDIT_SPREAD_BAA_AAA']}`",
        f"- DTB3 cash proxy: `{sources['RF_DAILY']}`",
        "- DTB3 is converted to daily cash return using compound conversion: `RF_DAILY = (1 + DTB3 / 100) ** (1 / 252) - 1`.",
        "",
        "## Regime Rules",
        "",
        "- HIGH_INFLATION: credit spread > 1.5 and DGS1 > 5.",
        "- INVERTED: term spread < 0.",
        "- FLAT: 0 <= term spread < 1.",
        "- STEEP: term spread >= 1.",
        f"- Macro regime changes require {CONFIRMATION_DAYS} consecutive trading days before confirmation.",
        f"- The confirmed regime switches only on the {CONFIRMATION_DAYS}nd consecutive day of the new raw regime.",
        "",
        "## VIX Dynamic Overlay",
        "",
        f"- VIX > 20 for {CONFIRMATION_DAYS} days enters WARNING.",
        "- WARNING de-risks based on episode peak VIX: `(peak VIX - 20) / 5`.",
        f"- VIX >= 25 for {CONFIRMATION_DAYS} days enters STRESS and sets SPY to the 20% floor.",
        f"- VIX < 25 for {CONFIRMATION_DAYS} days enters RECOVERY.",
        f"- VIX < 20 for {CONFIRMATION_DAYS} days restores NORMAL allocation.",
        "",
        "## No-Look-Ahead Implementation",
        "",
        "Signals and target weights are computed on date t. Returns on date t use actual weights set at the start of date t from signals available through t-1. Between rebalance dates, sleeve values drift with market returns.",
        "",
        "## Rebalance and Transaction Costs",
        "",
        "- Default scheduled rebalance is the first trading day of each month.",
        "- Event rebalance happens on the next trading day after confirmed macro regime or VIX overlay state changes.",
        "- A drift rebalance is triggered when any actual asset weight deviates from target by more than 5 percentage points.",
        "- Transaction cost is 5 bps times total traded notional on rebalance days.",
        "",
        "## Results",
        "",
        f"- Backtest sample: {panel['date'].min().date()} to {panel['date'].max().date()}.",
        f"- Best Sharpe strategy: `{best['strategy']}` ({best['Sharpe']:.2f}).",
        f"- Rule-based annualized return: {strategy['annualized_return']:.2%}; Sharpe: {strategy['Sharpe']:.2f}; max drawdown: {strategy['max_drawdown']:.2%}.",
        f"- SPY annualized return: {spy['annualized_return']:.2%}; Sharpe: {spy['Sharpe']:.2f}; max drawdown: {spy['max_drawdown']:.2%}.",
        f"- Static 40/30/15/15 annualized return: {static['annualized_return']:.2%}; Sharpe: {static['Sharpe']:.2f}; max drawdown: {static['max_drawdown']:.2%}.",
        "",
        "See `performance_summary.csv`, `equity_curve_full.png`, `drawdown_comparison.png`, `strategy_weights_over_time.png`, and `vix_vs_spy_weight.png` for the full diagnostics.",
        "",
        "## Caveats",
        "",
        "- Thresholds are historically motivated and still need robustness tests.",
        "- Taxes are not included.",
        "- The default implementation uses monthly plus event-driven rebalancing; alternative thresholds and schedules should be tested later.",
        "- Results depend on ETF history, especially GLD inception in 2004.",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel, sources = build_daily_panel()
    panel["macro_regime_raw"] = panel.apply(raw_macro_regime, axis=1)
    macro_confirmation = confirm_macro_regime(panel["macro_regime_raw"], confirmation_days=CONFIRMATION_DAYS)
    panel = pd.concat([panel.reset_index(drop=True), macro_confirmation], axis=1)
    panel = run_vix_state_machine(panel, confirmation_days=CONFIRMATION_DAYS, min_spy_weight=0.20)
    panel = assign_weights(panel)
    panel.to_csv(PANEL_PATH, index=False)

    returns, equity, weights, target_weights, actual_weights, rebalance_log = compute_backtest(panel)
    returns.to_csv(RETURNS_PATH, index=False)
    equity.to_csv(EQUITY_PATH, index=False)
    weights.to_csv(WEIGHTS_PATH, index=False)
    target_weights.to_csv(TARGET_WEIGHTS_PATH, index=False)
    actual_weights.to_csv(ACTUAL_WEIGHTS_PATH, index=False)
    rebalance_log.to_csv(REBALANCE_LOG_PATH, index=False)

    perf = summarize_performance(returns, weights)
    perf.to_csv(PERFORMANCE_PATH, index=False)

    perf_macro, avg_macro = conditional_performance(returns, weights, "macro_regime_confirmed")
    perf_vix, avg_vix = conditional_performance(returns, weights, "vix_overlay_state")
    perf_macro.to_csv(PERF_MACRO_PATH, index=False)
    perf_vix.to_csv(PERF_VIX_PATH, index=False)
    avg_macro.to_csv(AVG_W_MACRO_PATH, index=False)
    avg_vix.to_csv(AVG_W_VIX_PATH, index=False)

    episodes = vix_stress_episodes(panel, returns, weights)
    episodes.to_csv(EPISODE_PATH, index=False)

    validation = validation_report(panel, weights, returns)
    validation.to_csv(VALIDATION_PATH, index=False)

    plot_outputs(returns, equity, weights, panel, perf, avg_macro, avg_vix)
    write_report(panel, perf, sources)

    macro_switches = int(validation.loc[validation["check"] == "macro_regime_switches", "value"].iloc[0])
    vix_switches = int(validation.loc[validation["check"] == "vix_overlay_state_switches", "value"].iloc[0])
    print(f"Backtest start/end date: {panel['date'].min().date()} to {panel['date'].max().date()}")
    print(f"Trading days: {len(panel)}")
    print(f"Macro regime switches: {macro_switches}")
    print(f"VIX state episodes/switches: {vix_switches}")
    print("Final portfolio value:")
    print(equity[STRATEGIES].iloc[-1].to_string())
    print("Performance summary:")
    print(perf[["strategy", "annualized_return", "Sharpe", "max_drawdown"]].to_string(index=False))
    for path in [
        PANEL_PATH,
        RETURNS_PATH,
        EQUITY_PATH,
        WEIGHTS_PATH,
        TARGET_WEIGHTS_PATH,
        ACTUAL_WEIGHTS_PATH,
        REBALANCE_LOG_PATH,
        PERFORMANCE_PATH,
        PERF_MACRO_PATH,
        PERF_VIX_PATH,
        AVG_W_MACRO_PATH,
        AVG_W_VIX_PATH,
        DRAWDOWN_PATH,
        EPISODE_PATH,
        VALIDATION_PATH,
        REPORT_PATH,
        FIGURES_DIR / "equity_curve_full.png",
        FIGURES_DIR / "equity_curve_log.png",
        FIGURES_DIR / "drawdown_comparison.png",
        FIGURES_DIR / "strategy_weights_over_time.png",
        FIGURES_DIR / "vix_vs_spy_weight.png",
        FIGURES_DIR / "macro_regime_vix_timeline.png",
        FIGURES_DIR / "performance_summary_bar_charts.png",
        FIGURES_DIR / "average_weights_by_macro_regime_heatmap.png",
        FIGURES_DIR / "average_weights_by_vix_state_heatmap.png",
    ]:
        if path.exists():
            print(f"Saved: {path}")


if __name__ == "__main__":
    main()
