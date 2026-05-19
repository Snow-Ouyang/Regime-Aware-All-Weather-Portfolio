from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

TIMING_PANEL_PATH = ROOT / "results" / "spy_cash_timing_frequency_test" / "daily_backtest_panel.csv"
REGIME_PANEL_PATH = ROOT / "results" / "reconstructed_regime_asset_behavior" / "reconstructed_regime_panel.csv"
TEST_STEEP_ONLY_PATH = ROOT / "results" / "steep_only_sell_test" / "steep_only_sell_daily_panel.csv"

RESULTS_DIR = ROOT / "results" / "regime_hedge_steep_sell_ief"
FIGURES_DIR = ROOT / "figures" / "regime_hedge_steep_sell_ief"

DAILY_OUT = RESULTS_DIR / "daily_backtest_panel.csv"
REBALANCE_OUT = RESULTS_DIR / "rebalance_log.csv"
PERF_OUT = RESULTS_DIR / "performance_summary.csv"
REGIME_PERF_OUT = RESULTS_DIR / "performance_by_regime.csv"
CROSS_PERF_OUT = RESULTS_DIR / "performance_by_regime_and_timing_state.csv"
CRISIS_OUT = RESULTS_DIR / "crisis_performance.csv"
SUMMARY_MD_OUT = RESULTS_DIR / "REGIME_HEDGE_STEEP_SELL_IEF_SUMMARY.md"

FIG_LOG = FIGURES_DIR / "equity_curve_log.png"
FIG_LINEAR = FIGURES_DIR / "equity_curve_linear.png"
FIG_DD = FIGURES_DIR / "drawdown_comparison.png"
FIG_W = FIGURES_DIR / "weights_timeline.png"
FIG_REGIME = FIGURES_DIR / "regime_labeled_equity_curve.png"
FIG_BAR = FIGURES_DIR / "performance_bar_charts.png"
FIG_CRISIS = FIGURES_DIR / "crisis_case_study_2008_2020_2022.png"

CONFIG = {
    "rebalance_frequency": "monthly",
    "one_way_cost_bps": 5.0,
    "initial_nav": 1.0,
    "output_dir": str(RESULTS_DIR),
    "figure_dir": str(FIGURES_DIR),
}

ASSETS = ["SPY", "IEF", "GOLD", "CMDTY", "CASH"]
REGIME_ORDER = ["HIGH_INFLATION", "INVERTED", "FLAT", "STEEP", "NEUTRAL"]
CASE_STUDIES = {
    "GFC_2008": ("2007-07-01", "2009-12-31"),
    "COVID_2020": ("2020-01-01", "2020-12-31"),
    "INFLATION_2022": ("2021-11-01", "2023-03-31"),
}


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    if not TIMING_PANEL_PATH.exists():
        raise FileNotFoundError(f"Missing timing panel: {TIMING_PANEL_PATH}")
    if not REGIME_PANEL_PATH.exists():
        raise FileNotFoundError(f"Missing regime panel: {REGIME_PANEL_PATH}")

    timing = pd.read_csv(TIMING_PANEL_PATH)
    timing["date"] = pd.to_datetime(timing["date"])
    regime = pd.read_csv(
        REGIME_PANEL_PATH,
        usecols=[
            "date",
            "RF_DAILY",
            "SPY_RETURN",
            "IEF_RETURN",
            "GOLD_RETURN",
            "CMDTY_FUT_RETURN",
            "CREDIT_SPREAD_BAA_AAA",
            "DGS1",
            "DGS10",
            "TERM_SPREAD_10Y_1Y",
            "SPY_DRAWDOWN_FROM_PREVIOUS_HIGH",
        ],
    )
    regime["date"] = pd.to_datetime(regime["date"])
    regime["macro_regime_confirmed"] = np.select(
        [
            (pd.to_numeric(regime["CREDIT_SPREAD_BAA_AAA"], errors="coerce") > 1.5)
            & (pd.to_numeric(regime["DGS1"], errors="coerce") > 5.0),
            pd.to_numeric(regime["TERM_SPREAD_10Y_1Y"], errors="coerce") < 0.0,
            (pd.to_numeric(regime["TERM_SPREAD_10Y_1Y"], errors="coerce") >= 0.0)
            & (pd.to_numeric(regime["TERM_SPREAD_10Y_1Y"], errors="coerce") < 1.0),
            pd.to_numeric(regime["TERM_SPREAD_10Y_1Y"], errors="coerce") >= 1.0,
        ],
        ["HIGH_INFLATION", "INVERTED", "FLAT", "STEEP"],
        default="NEUTRAL",
    )

    panel = timing.merge(regime, on="date", how="inner")
    required = [
        "date",
        "spy_price",
        "spy_daily_return",
        "daily_rf",
        "monthly_either_weight_spy",
        "MONTHLY_EITHER_CONFIRM_return",
        "MONTHLY_EITHER_CONFIRM_nav",
        "SPY_BUY_HOLD_return",
        "SPY_BUY_HOLD_nav",
        "CASH_ONLY_return",
        "CASH_ONLY_nav",
        "RF_DAILY",
        "SPY_RETURN",
        "IEF_RETURN",
        "GOLD_RETURN",
        "CMDTY_FUT_RETURN",
        "macro_regime_confirmed",
    ]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    panel["monthly_either_state"] = np.where(panel["monthly_either_weight_spy"] >= 0.5, "HOLD", "SELL")
    panel["month"] = panel["date"].dt.to_period("M")
    panel = panel.sort_values("date").reset_index(drop=True)
    return panel


def load_asset_returns(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["SPY_ret"] = out["SPY_RETURN"]
    out["IEF_ret"] = out["IEF_RETURN"]
    out["GOLD_ret"] = out["GOLD_RETURN"]
    out["CMDTY_ret"] = out["CMDTY_FUT_RETURN"]
    out["CASH_ret"] = out["RF_DAILY"]
    # Rebase benchmark NAVs to the common multi-asset sample used in this test.
    out["SPY_BUY_HOLD_nav"] = (1.0 + out["SPY_BUY_HOLD_return"]).cumprod()
    out["CASH_ONLY_nav"] = (1.0 + out["CASH_ONLY_return"]).cumprod()
    out["MONTHLY_EITHER_CONFIRM_nav"] = (1.0 + out["MONTHLY_EITHER_CONFIRM_return"]).cumprod()
    return out


def build_strategy_target_weights(regime: str, timing_state: str) -> dict[str, float]:
    if regime == "INVERTED":
        return {"SPY": 0.80, "IEF": 0.0, "GOLD": 0.20, "CMDTY": 0.0, "CASH": 0.0}
    if regime == "FLAT":
        return {"SPY": 0.70, "IEF": 0.0, "GOLD": 0.0, "CMDTY": 0.30, "CASH": 0.0}
    if regime == "STEEP" and timing_state == "HOLD":
        return {"SPY": 0.80, "IEF": 0.20, "GOLD": 0.0, "CMDTY": 0.0, "CASH": 0.0}
    if regime == "STEEP" and timing_state == "SELL":
        return {"SPY": 0.0, "IEF": 1.0, "GOLD": 0.0, "CMDTY": 0.0, "CASH": 0.0}
    if regime == "HIGH_INFLATION":
        return {"SPY": 0.50, "IEF": 0.0, "GOLD": 0.0, "CMDTY": 0.0, "CASH": 0.50}
    return {"SPY": 0.80, "IEF": 0.0, "GOLD": 0.0, "CMDTY": 0.0, "CASH": 0.20}


def run_monthly_rebalance_backtest(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = panel.copy()
    for col in [f"target_weight_{a}" for a in ASSETS] + [f"actual_weight_{a}" for a in ASSETS]:
        out[col] = np.nan
    out["portfolio_return"] = np.nan
    out["portfolio_nav"] = np.nan
    out["turnover"] = 0.0
    out["transaction_cost"] = 0.0
    out["rebalance_flag"] = False
    out["rebalance_reason"] = ""

    rebalance_rows = []
    current_weights = {a: 0.0 for a in ASSETS}
    current_weights["CASH"] = 1.0
    current_nav = float(CONFIG["initial_nav"])
    asset_values = {a: current_nav * current_weights[a] for a in ASSETS}

    month_start_mask = out["month"] != out["month"].shift(1)
    for i, row in out.iterrows():
        if bool(month_start_mask.iloc[i]):
            signal_row = out.iloc[i - 1] if i > 0 else out.iloc[i]
            target = build_strategy_target_weights(str(signal_row["macro_regime_confirmed"]), str(signal_row["monthly_either_state"]))
            total_value = sum(asset_values.values())
            current_weight_vec = {a: (asset_values[a] / total_value if total_value > 0 else 0.0) for a in ASSETS}
            turnover = sum(abs(target[a] - current_weight_vec[a]) for a in ASSETS)
            if i == 0:
                turnover = 0.0
            cost = turnover * (CONFIG["one_way_cost_bps"] / 10000.0)
            total_after_cost = total_value * (1.0 - cost)
            asset_values = {a: total_after_cost * target[a] for a in ASSETS}
            current_weights = target.copy()
            out.loc[i, "rebalance_flag"] = True
            out.loc[i, "rebalance_reason"] = "month_start"
            out.loc[i, "turnover"] = turnover
            out.loc[i, "transaction_cost"] = cost
            rebalance_rows.append(
                {
                    "rebalance_date": row["date"],
                    "macro_regime_confirmed": signal_row["macro_regime_confirmed"],
                    "monthly_either_state": signal_row["monthly_either_state"],
                    "old_weight_SPY": current_weight_vec["SPY"],
                    "old_weight_IEF": current_weight_vec["IEF"],
                    "old_weight_GOLD": current_weight_vec["GOLD"],
                    "old_weight_CMDTY": current_weight_vec["CMDTY"],
                    "old_weight_CASH": current_weight_vec["CASH"],
                    "new_target_weight_SPY": target["SPY"],
                    "new_target_weight_IEF": target["IEF"],
                    "new_target_weight_GOLD": target["GOLD"],
                    "new_target_weight_CMDTY": target["CMDTY"],
                    "new_target_weight_CASH": target["CASH"],
                    "turnover": turnover,
                    "transaction_cost": cost,
                    "rebalance_reason": "month_start",
                }
            )
        total_value = sum(asset_values.values())
        start_weights = {a: (asset_values[a] / total_value if total_value > 0 else 0.0) for a in ASSETS}
        target_today = current_weights.copy()
        for a in ASSETS:
            out.loc[i, f"target_weight_{a}"] = target_today[a]
            out.loc[i, f"actual_weight_{a}"] = start_weights[a]
        port_ret = (
            start_weights["SPY"] * row["SPY_ret"]
            + start_weights["IEF"] * row["IEF_ret"]
            + start_weights["GOLD"] * row["GOLD_ret"]
            + start_weights["CMDTY"] * row["CMDTY_ret"]
            + start_weights["CASH"] * row["CASH_ret"]
            - out.loc[i, "transaction_cost"]
        )
        out.loc[i, "portfolio_return"] = port_ret
        for a, ret_col in [("SPY", "SPY_ret"), ("IEF", "IEF_ret"), ("GOLD", "GOLD_ret"), ("CMDTY", "CMDTY_ret"), ("CASH", "CASH_ret")]:
            asset_values[a] *= 1.0 + float(row[ret_col])
        current_nav *= 1.0 + float(port_ret)
        out.loc[i, "portfolio_nav"] = current_nav
    return out, pd.DataFrame(rebalance_rows)


def compute_performance_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mapping = [
        ("REGIME_HEDGE_STEEP_SELL_IEF", "portfolio_return", "portfolio_nav", True),
        ("SPY_BUY_HOLD", "SPY_BUY_HOLD_return", "SPY_BUY_HOLD_nav", False),
        ("CASH_ONLY", "CASH_ONLY_return", "CASH_ONLY_nav", False),
        ("MONTHLY_EITHER_CONFIRM", "MONTHLY_EITHER_CONFIRM_return", "MONTHLY_EITHER_CONFIRM_nav", False),
        ("TEST_STEEP_ONLY_SELL", "TEST_STEEP_ONLY_SELL_return", "TEST_STEEP_ONLY_SELL_nav", False),
    ]
    for strategy, ret_col, nav_col, is_new in mapping:
        s = panel[ret_col].dropna()
        rf = panel.loc[s.index, "RF_DAILY"]
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
        calmar = float(ann_ret / abs(mdd)) if mdd < 0 else np.nan
        n_reb = int(panel["rebalance_flag"].sum()) if is_new else int(panel.get(f"{strategy}_turnover_flag", pd.Series(0, index=panel.index)).sum())
        total_turnover = float(panel["turnover"].sum()) if is_new else np.nan
        tc_drag = float(panel["transaction_cost"].sum()) if is_new else float(panel.get(f"{strategy}_transaction_cost", pd.Series(0.0, index=panel.index)).sum())
        if is_new:
            avg_weights = {a: float(panel[f"actual_weight_{a}"].mean()) for a in ASSETS}
        else:
            if strategy == "SPY_BUY_HOLD":
                avg_weights = {"SPY": 1.0, "IEF": 0.0, "GOLD": 0.0, "CMDTY": 0.0, "CASH": 0.0}
            elif strategy == "CASH_ONLY":
                avg_weights = {"SPY": 0.0, "IEF": 0.0, "GOLD": 0.0, "CMDTY": 0.0, "CASH": 1.0}
            elif strategy == "MONTHLY_EITHER_CONFIRM":
                spy_w = float(panel["monthly_either_weight_spy"].mean())
                avg_weights = {"SPY": spy_w, "IEF": 0.0, "GOLD": 0.0, "CMDTY": 0.0, "CASH": 1.0 - spy_w}
            else:
                spy_w = float(panel["TEST_STEEP_ONLY_SELL_weight_spy"].mean())
                avg_weights = {"SPY": spy_w, "IEF": 0.0, "GOLD": 0.0, "CMDTY": 0.0, "CASH": 1.0 - spy_w}
        rows.append(
            {
                "strategy": strategy,
                "start_date": panel["date"].iloc[0].date().isoformat(),
                "end_date": panel["date"].iloc[-1].date().isoformat(),
                "annualized_return": ann_ret,
                "annualized_volatility": ann_vol,
                "sharpe_ratio": sharpe,
                "max_drawdown": mdd,
                "calmar_ratio": calmar,
                "final_nav": float(panel[nav_col].iloc[-1]),
                "number_of_rebalances": n_reb,
                "total_turnover": total_turnover,
                "avg_turnover_per_year": float(total_turnover / (len(s) / 252.0)) if is_new and len(s) > 0 else np.nan,
                "transaction_cost_drag": tc_drag,
                "time_weighted_average_SPY_weight": avg_weights["SPY"],
                "time_weighted_average_IEF_weight": avg_weights["IEF"],
                "time_weighted_average_GOLD_weight": avg_weights["GOLD"],
                "time_weighted_average_CMDTY_weight": avg_weights["CMDTY"],
                "time_weighted_average_CASH_weight": avg_weights["CASH"],
            }
        )
    return pd.DataFrame(rows)


def _group_metrics(sub: pd.DataFrame, strategy: str, ret_col: str, nav_col: str) -> dict[str, float]:
    s = sub[ret_col].dropna()
    rf = sub.loc[s.index, "RF_DAILY"]
    ann_ret = float((1.0 + s).prod() ** (252.0 / len(s)) - 1.0) if len(s) > 0 else np.nan
    vol = float(s.std(ddof=1) * np.sqrt(252.0)) if len(s) > 1 else np.nan
    if strategy == "CASH_ONLY":
        sharpe = 0.0
    else:
        ex = s - rf
        ex_std = ex.std(ddof=1)
        sharpe = float(ex.mean() / ex_std * np.sqrt(252.0)) if pd.notna(ex_std) and ex_std != 0 else np.nan
    wealth = (1.0 + s).cumprod()
    mdd = float((wealth / wealth.cummax() - 1.0).min()) if len(s) > 0 else np.nan
    return {"n_obs": int(len(s)), "annualized_return": ann_ret, "volatility": vol, "Sharpe": sharpe, "max_drawdown": mdd}


def compute_regime_performance(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_regime = []
    rows_cross = []
    strategies = [
        ("REGIME_HEDGE_STEEP_SELL_IEF", "portfolio_return", True),
        ("SPY_BUY_HOLD", "SPY_BUY_HOLD_return", False),
        ("CASH_ONLY", "CASH_ONLY_return", False),
        ("MONTHLY_EITHER_CONFIRM", "MONTHLY_EITHER_CONFIRM_return", False),
        ("TEST_STEEP_ONLY_SELL", "TEST_STEEP_ONLY_SELL_return", False),
    ]
    for regime, grp in panel.groupby("macro_regime_confirmed", observed=False):
        for strategy, ret_col, is_new in strategies:
            stats = _group_metrics(grp, strategy, ret_col, "")
            if is_new:
                avg_w = {a: float(grp[f"actual_weight_{a}"].mean()) for a in ASSETS}
            else:
                if strategy == "SPY_BUY_HOLD":
                    avg_w = {"SPY": 1.0, "IEF": 0.0, "GOLD": 0.0, "CMDTY": 0.0, "CASH": 0.0}
                elif strategy == "CASH_ONLY":
                    avg_w = {"SPY": 0.0, "IEF": 0.0, "GOLD": 0.0, "CMDTY": 0.0, "CASH": 1.0}
                elif strategy == "MONTHLY_EITHER_CONFIRM":
                    spy_w = float(grp["monthly_either_weight_spy"].mean())
                    avg_w = {"SPY": spy_w, "IEF": 0.0, "GOLD": 0.0, "CMDTY": 0.0, "CASH": 1.0 - spy_w}
                else:
                    spy_w = float(grp["TEST_STEEP_ONLY_SELL_weight_spy"].mean())
                    avg_w = {"SPY": spy_w, "IEF": 0.0, "GOLD": 0.0, "CMDTY": 0.0, "CASH": 1.0 - spy_w}
            rows_regime.append({"strategy": strategy, "macro_regime_confirmed": regime, **stats, **{f"average_weight_{a}": avg_w[a] for a in ASSETS}})
    for (regime, state), grp in panel.groupby(["macro_regime_confirmed", "monthly_either_state"], observed=False):
        for strategy, ret_col, is_new in strategies:
            stats = _group_metrics(grp, strategy, ret_col, "")
            if is_new:
                avg_w = {a: float(grp[f"actual_weight_{a}"].mean()) for a in ASSETS}
                strat_ret = float(grp["portfolio_return"].mean())
            else:
                if strategy == "SPY_BUY_HOLD":
                    avg_w = {"SPY": 1.0, "IEF": 0.0, "GOLD": 0.0, "CMDTY": 0.0, "CASH": 0.0}
                elif strategy == "CASH_ONLY":
                    avg_w = {"SPY": 0.0, "IEF": 0.0, "GOLD": 0.0, "CMDTY": 0.0, "CASH": 1.0}
                elif strategy == "MONTHLY_EITHER_CONFIRM":
                    spy_w = float(grp["monthly_either_weight_spy"].mean())
                    avg_w = {"SPY": spy_w, "IEF": 0.0, "GOLD": 0.0, "CMDTY": 0.0, "CASH": 1.0 - spy_w}
                else:
                    spy_w = float(grp["TEST_STEEP_ONLY_SELL_weight_spy"].mean())
                    avg_w = {"SPY": spy_w, "IEF": 0.0, "GOLD": 0.0, "CMDTY": 0.0, "CASH": 1.0 - spy_w}
                strat_ret = float(grp[ret_col].mean())
            rows_cross.append(
                {
                    "strategy": strategy,
                    "macro_regime_confirmed": regime,
                    "monthly_either_state": state,
                    "n_obs": int(len(grp)),
                    "strategy_return": strat_ret,
                    "SPY_return": float(grp["SPY_ret"].mean()),
                    "IEF_return": float(grp["IEF_ret"].mean()),
                    "GOLD_return": float(grp["GOLD_ret"].mean()),
                    "CMDTY_return": float(grp["CMDTY_ret"].mean()),
                    "CASH_return": float(grp["CASH_ret"].mean()),
                    "Sharpe": stats["Sharpe"],
                    "max_drawdown": stats["max_drawdown"],
                    **{f"average_weight_{a}": avg_w[a] for a in ASSETS},
                }
            )
    return pd.DataFrame(rows_regime), pd.DataFrame(rows_cross)


def compute_crisis_performance(panel: pd.DataFrame) -> pd.DataFrame:
    windows = {
        "DOTCOM_2000_2002": ("2000-01-01", "2002-12-31"),
        "GFC_2008_2009": ("2008-09-01", "2009-03-31"),
        "COVID_2020": ("2020-02-19", "2020-04-30"),
        "INFLATION_2022": ("2022-01-01", "2022-12-31"),
        "RECENT_2024_2026": ("2024-01-01", "2026-12-31"),
    }
    rows = []
    strategies = [
        ("REGIME_HEDGE_STEEP_SELL_IEF", "portfolio_return", True),
        ("SPY_BUY_HOLD", "SPY_BUY_HOLD_return", False),
        ("CASH_ONLY", "CASH_ONLY_return", False),
        ("MONTHLY_EITHER_CONFIRM", "MONTHLY_EITHER_CONFIRM_return", False),
        ("TEST_STEEP_ONLY_SELL", "TEST_STEEP_ONLY_SELL_return", False),
    ]
    for period, (start, end) in windows.items():
        sub = panel.loc[(panel["date"] >= pd.Timestamp(start)) & (panel["date"] <= pd.Timestamp(end))].copy()
        if sub.empty:
            continue
        for strategy, ret_col, is_new in strategies:
            stats = _group_metrics(sub, strategy, ret_col, "")
            if is_new:
                avg_w = {a: float(sub[f"actual_weight_{a}"].mean()) for a in ASSETS}
            else:
                if strategy == "SPY_BUY_HOLD":
                    avg_w = {"SPY": 1.0, "IEF": 0.0, "GOLD": 0.0, "CMDTY": 0.0, "CASH": 0.0}
                elif strategy == "CASH_ONLY":
                    avg_w = {"SPY": 0.0, "IEF": 0.0, "GOLD": 0.0, "CMDTY": 0.0, "CASH": 1.0}
                elif strategy == "MONTHLY_EITHER_CONFIRM":
                    spy_w = float(sub["monthly_either_weight_spy"].mean())
                    avg_w = {"SPY": spy_w, "IEF": 0.0, "GOLD": 0.0, "CMDTY": 0.0, "CASH": 1.0 - spy_w}
                else:
                    spy_w = float(sub["TEST_STEEP_ONLY_SELL_weight_spy"].mean())
                    avg_w = {"SPY": spy_w, "IEF": 0.0, "GOLD": 0.0, "CMDTY": 0.0, "CASH": 1.0 - spy_w}
            rows.append({"period": period, "strategy": strategy, "cumulative_return": float((1.0 + sub[ret_col]).prod() - 1.0), **stats, **{f"average_weight_{a}": avg_w[a] for a in ASSETS}})
    return pd.DataFrame(rows)


def plot_results(panel: pd.DataFrame, perf: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, col in [
        ("SPY_BUY_HOLD", "SPY_BUY_HOLD_nav"),
        ("MONTHLY_EITHER_CONFIRM", "MONTHLY_EITHER_CONFIRM_nav"),
        ("TEST_STEEP_ONLY_SELL", "TEST_STEEP_ONLY_SELL_nav"),
        ("REGIME_HEDGE_STEEP_SELL_IEF", "portfolio_nav"),
        ("CASH_ONLY", "CASH_ONLY_nav"),
    ]:
        ax.plot(panel["date"], panel[col], label=name)
    ax.set_yscale("log")
    ax.set_title("Regime Hedge Steep Sell IEF - Log NAV")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_LOG, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, col in [
        ("SPY_BUY_HOLD", "SPY_BUY_HOLD_nav"),
        ("MONTHLY_EITHER_CONFIRM", "MONTHLY_EITHER_CONFIRM_nav"),
        ("TEST_STEEP_ONLY_SELL", "TEST_STEEP_ONLY_SELL_nav"),
        ("REGIME_HEDGE_STEEP_SELL_IEF", "portfolio_nav"),
        ("CASH_ONLY", "CASH_ONLY_nav"),
    ]:
        ax.plot(panel["date"], panel[col], label=name)
    ax.set_title("Regime Hedge Steep Sell IEF - Linear NAV")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_LINEAR, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for name, col in [
        ("SPY_BUY_HOLD", "SPY_BUY_HOLD_return"),
        ("MONTHLY_EITHER_CONFIRM", "MONTHLY_EITHER_CONFIRM_return"),
        ("REGIME_HEDGE_STEEP_SELL_IEF", "portfolio_return"),
    ]:
        wealth = (1.0 + panel[col]).cumprod()
        dd = wealth / wealth.cummax() - 1.0
        ax.plot(panel["date"], dd, label=name)
    ax.set_title("Drawdown Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DD, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    stack = np.vstack([panel[f"actual_weight_{a}"] for a in ASSETS])
    ax.stackplot(panel["date"], stack, labels=ASSETS)
    ax.set_title("New Strategy Weights")
    ax.legend(loc="upper left", ncol=5)
    fig.tight_layout()
    fig.savefig(FIG_W, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1]})
    ax1, ax2, ax3 = axes
    # light regime background
    tmp = panel.reset_index(drop=True)
    starts = tmp["macro_regime_confirmed"] != tmp["macro_regime_confirmed"].shift(1)
    positions = tmp.index[starts].tolist()
    ends = positions[1:] + [len(tmp)]
    regime_colors = {"HIGH_INFLATION": "#d95f02", "INVERTED": "#7570b3", "FLAT": "#1b9e77", "STEEP": "#66a61e", "NEUTRAL": "#999999"}
    for s, e in zip(positions, ends):
        regime = tmp.iloc[s]["macro_regime_confirmed"]
        ax1.axvspan(tmp.iloc[s]["date"], tmp.iloc[e - 1]["date"], color=regime_colors.get(regime, "#cccccc"), alpha=0.08)
        ax2.axvspan(tmp.iloc[s]["date"], tmp.iloc[e - 1]["date"], color=regime_colors.get(regime, "#cccccc"), alpha=0.06)
    ax1.plot(panel["date"], panel["SPY_BUY_HOLD_nav"], label="SPY_BUY_HOLD", color="black")
    ax1.plot(panel["date"], panel["MONTHLY_EITHER_CONFIRM_nav"], label="MONTHLY_EITHER_CONFIRM", color="tab:blue")
    ax1.plot(panel["date"], panel["portfolio_nav"], label="REGIME_HEDGE_STEEP_SELL_IEF", color="tab:orange")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.set_title("Regime-Labeled Equity Curve")
    ax2.plot(panel["date"], panel["SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"], color="tab:red")
    for thr in [-0.05, -0.10, -0.15, -0.20]:
        ax2.axhline(thr, color="gray", linestyle="--", linewidth=0.8)
    steep_sell = (panel["macro_regime_confirmed"] == "STEEP") & (panel["monthly_either_state"] == "SELL")
    ax3.fill_between(panel["date"], 0.55, 0.95, where=panel["monthly_either_state"] == "HOLD", color="green", alpha=0.6)
    ax3.fill_between(panel["date"], 0.55, 0.95, where=panel["monthly_either_state"] == "SELL", color="red", alpha=0.6)
    ax3.fill_between(panel["date"], 0.05, 0.45, where=steep_sell, color="tab:orange", alpha=0.7)
    ax3.set_yticks([0.25, 0.75])
    ax3.set_yticklabels(["STEEP+SELL", "Monthly Either"])
    fig.tight_layout()
    fig.savefig(FIG_REGIME, dpi=180)
    plt.close(fig)

    plot_df = perf.loc[perf["strategy"].isin(["SPY_BUY_HOLD", "MONTHLY_EITHER_CONFIRM", "TEST_STEEP_ONLY_SELL", "REGIME_HEDGE_STEEP_SELL_IEF", "CASH_ONLY"])].copy()
    metrics = ["annualized_return", "sharpe_ratio", "max_drawdown", "calmar_ratio"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric in zip(axes.flatten(), metrics):
        ax.bar(plot_df["strategy"], plot_df[metric])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIG_BAR, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    for ax, (name, (start, end)) in zip(axes, CASE_STUDIES.items()):
        sub = panel.loc[(panel["date"] >= pd.Timestamp(start)) & (panel["date"] <= pd.Timestamp(end))].copy()
        if sub.empty:
            continue
        ax.plot(sub["date"], sub["SPY_BUY_HOLD_nav"] / sub["SPY_BUY_HOLD_nav"].iloc[0], label="SPY", color="black")
        ax.plot(sub["date"], sub["MONTHLY_EITHER_CONFIRM_nav"] / sub["MONTHLY_EITHER_CONFIRM_nav"].iloc[0], label="Monthly Either", color="tab:blue")
        ax.plot(sub["date"], sub["portfolio_nav"] / sub["portfolio_nav"].iloc[0], label="New strategy", color="tab:orange")
        ax2 = ax.twinx()
        ax2.stackplot(sub["date"], sub["actual_weight_SPY"], sub["actual_weight_IEF"], sub["actual_weight_GOLD"], sub["actual_weight_CMDTY"], sub["actual_weight_CASH"], alpha=0.12)
        ax.set_title(name)
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_CRISIS, dpi=180)
    plt.close(fig)


def write_summary_md(perf: pd.DataFrame, crisis: pd.DataFrame) -> None:
    table = perf.to_markdown(index=False)
    crisis_table = crisis.to_markdown(index=False) if not crisis.empty else "_No crisis windows overlap the common sample._"
    lines = [
        "# REGIME_HEDGE_STEEP_SELL_IEF Summary",
        "",
        "## Strategy Idea",
        "",
        "This strategy is not a pure SPY timing overlay. It is a regime-conditioned hedge allocation. Monthly Either SELL only triggers a full SPY exit in STEEP. In other regimes, the portfolio keeps SPY exposure and expresses risk control through a hedge sleeve.",
        "",
        "## Rule Table",
        "",
        "- INVERTED: 80% SPY / 20% GOLD",
        "- FLAT: 70% SPY / 30% CMDTY",
        "- STEEP + HOLD: 80% SPY / 20% IEF",
        "- STEEP + SELL: 100% IEF",
        "- HIGH_INFLATION: 50% SPY / 50% CASH",
        "- NEUTRAL: 80% SPY / 20% CASH",
        "",
        "## Main Performance Comparison",
        "",
        table,
        "",
        "## Data / Sample",
        "",
        f"- Common sample start: {perf['start_date'].iloc[0]}",
        f"- Common sample end: {perf['end_date'].iloc[0]}",
        "- `DOTCOM_2000_2002` is omitted if it falls outside the common sample for SPY / IEF / GOLD / CMDTY / CASH.",
        "",
        "## Crisis Performance",
        "",
        crisis_table,
        "",
        "## Interpretation",
        "",
        "- The main comparison is against SPY buy-and-hold, Monthly Either SPY/CASH timing, and the earlier STEEP-only SELL test.",
        "- This test checks whether adding a hedge sleeve improves the weak drawdown profile of the STEEP-only SELL variant without giving up too much of the timing benefit.",
        "- FLAT uses commodity futures as a real-asset sleeve, which can help in some reflation periods but can also add path risk.",
        "- INVERTED keeps a large SPY allocation and uses GOLD as the hedge sleeve.",
        "- STEEP HOLD adds IEF as a duration hedge, while STEEP SELL rotates fully into IEF rather than cash.",
        "",
        "## Caveats",
        "",
        "- These weights are heuristic, not optimized.",
        "- Commodity futures can materially change path risk.",
        "- IEF can fail in rising-rate episodes.",
        "- Monthly rebalancing may still miss fast crashes.",
        "- Further tests should cover partial SELL rules and hedge weight sensitivity.",
        "",
        "## Next Step",
        "",
        "- Test 10/20/30% hedge sleeve sensitivity.",
        "- Test STEEP SELL routing: IEF vs CASH vs IEF+GOLD.",
        "- Test FLAT CMDTY weight 10/20/30%.",
        "- Test regime-aware partial SPY retention under SELL.",
    ]
    SUMMARY_MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = load_asset_returns(load_data())
    if TEST_STEEP_ONLY_PATH.exists():
        test_panel = pd.read_csv(TEST_STEEP_ONLY_PATH, usecols=["date", "TEST_STEEP_ONLY_SELL_return", "TEST_STEEP_ONLY_SELL_nav", "TEST_STEEP_ONLY_SELL_weight_spy"])
        test_panel["date"] = pd.to_datetime(test_panel["date"])
        panel = panel.merge(test_panel, on="date", how="left")
    else:
        panel["TEST_STEEP_ONLY_SELL_weight_spy"] = np.where((panel["monthly_either_state"] == "SELL") & (panel["macro_regime_confirmed"] == "STEEP"), 0.0, 1.0)
        panel["TEST_STEEP_ONLY_SELL_return"] = panel["TEST_STEEP_ONLY_SELL_weight_spy"] * panel["SPY_RETURN"] + (1.0 - panel["TEST_STEEP_ONLY_SELL_weight_spy"]) * panel["RF_DAILY"]
    panel["TEST_STEEP_ONLY_SELL_nav"] = (1.0 + panel["TEST_STEEP_ONLY_SELL_return"]).cumprod()

    panel, rebalance_log = run_monthly_rebalance_backtest(panel)
    perf = compute_performance_metrics(panel)
    regime_perf, cross_perf = compute_regime_performance(panel)
    crisis = compute_crisis_performance(panel)

    panel.to_csv(DAILY_OUT, index=False)
    rebalance_log.to_csv(REBALANCE_OUT, index=False)
    perf.to_csv(PERF_OUT, index=False)
    regime_perf.to_csv(REGIME_PERF_OUT, index=False)
    cross_perf.to_csv(CROSS_PERF_OUT, index=False)
    crisis.to_csv(CRISIS_OUT, index=False)
    plot_results(panel, perf)
    write_summary_md(perf, crisis)

    new_row = perf.loc[perf["strategy"] == "REGIME_HEDGE_STEEP_SELL_IEF"].iloc[0]
    me_row = perf.loc[perf["strategy"] == "MONTHLY_EITHER_CONFIRM"].iloc[0]
    test_row = perf.loc[perf["strategy"] == "TEST_STEEP_ONLY_SELL"].iloc[0]
    best_crisis = crisis.loc[crisis["strategy"] == "REGIME_HEDGE_STEEP_SELL_IEF"].sort_values("cumulative_return", ascending=False).iloc[0] if not crisis.empty else None
    worst_crisis = crisis.loc[crisis["strategy"] == "REGIME_HEDGE_STEEP_SELL_IEF"].sort_values("cumulative_return", ascending=True).iloc[0] if not crisis.empty else None
    regime_contrib = regime_perf.loc[regime_perf["strategy"] == "REGIME_HEDGE_STEEP_SELL_IEF"].sort_values("annualized_return", ascending=False).iloc[0]

    print(f"1. New strategy annualized return / Sharpe / MaxDD: {new_row['annualized_return']:.2%} / {new_row['sharpe_ratio']:.2f} / {new_row['max_drawdown']:.2%}")
    print(f"2. Beats Monthly Either by Sharpe: {bool(new_row['sharpe_ratio'] > me_row['sharpe_ratio'])}")
    print(f"3. Beats TEST_STEEP_ONLY_SELL by Sharpe: {bool(new_row['sharpe_ratio'] > test_row['sharpe_ratio'])}")
    print(f"4. MaxDD lower than TEST_STEEP_ONLY_SELL: {bool(new_row['max_drawdown'] > test_row['max_drawdown'])}")
    print(f"5. Return higher than Monthly Either: {bool(new_row['annualized_return'] > me_row['annualized_return'])}")
    if best_crisis is not None and worst_crisis is not None:
        print(f"6. Best / worst crisis period for new strategy: {best_crisis['period']} / {worst_crisis['period']}")
    print(f"7. Regime with highest annualized return contribution: {regime_contrib['macro_regime_confirmed']}")
    recommend = bool((new_row["sharpe_ratio"] >= me_row["sharpe_ratio"]) or (new_row["max_drawdown"] > test_row["max_drawdown"]))
    print(f"8. Recommend next weight sensitivity test: {recommend}")
    print(f"Saved outputs: {RESULTS_DIR} and {FIGURES_DIR}")


if __name__ == "__main__":
    main()
