from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "absolute_momentum_spy_cash"

DAILY_CLOSE_PATH = ROOT / "data" / "processed" / "assets" / "daily_adjusted_close.csv"
DAILY_RETURNS_PATH = ROOT / "data" / "processed" / "assets" / "daily_returns.csv"
DTB3_PATH = ROOT / "data" / "raw" / "macro" / "rate" / "DTB3.csv"

DAILY_PANEL_PATH = RESULTS_DIR / "daily_backtest_panel.csv"
MONTHLY_SIGNAL_PATH = RESULTS_DIR / "monthly_signal_panel.csv"
PERF_PATH = RESULTS_DIR / "performance_summary.csv"
YEARLY_PATH = RESULTS_DIR / "yearly_returns.csv"
DD_PATH = RESULTS_DIR / "drawdown_summary.csv"

FIG_LOG_PATH = RESULTS_DIR / "equity_curve_log.png"
FIG_LINEAR_PATH = RESULTS_DIR / "equity_curve_linear.png"
FIG_DD_PATH = RESULTS_DIR / "drawdown_comparison.png"
FIG_SIGNAL_OVERLAY_PATH = RESULTS_DIR / "monthly_signal_overlay.png"
FIG_MOM_PATH = RESULTS_DIR / "excess_momentum_signal.png"
FIG_ROLLING_PATH = RESULTS_DIR / "rolling_3y_performance.png"

DEFAULT_TICKER = "SPY"
TRADING_DAYS = 252
SWITCH_COST_BPS = 20.0


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def read_rate_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
    value_col = next((c for c in df.columns if c != date_col), df.columns[-1])
    out = df[[date_col, value_col]].copy()
    out.columns = ["date", "DTB3"]
    out["date"] = pd.to_datetime(out["date"])
    out["DTB3"] = pd.to_numeric(out["DTB3"].replace(".", np.nan), errors="coerce")
    out = out.sort_values("date")
    out["DTB3"] = out["DTB3"].ffill()
    out["daily_rf"] = (1.0 + out["DTB3"] / 100.0) ** (1.0 / TRADING_DAYS) - 1.0
    return out[["date", "DTB3", "daily_rf"]]


def load_spy_data(ticker: str = DEFAULT_TICKER) -> pd.DataFrame:
    close = pd.read_csv(DAILY_CLOSE_PATH, usecols=["date", ticker])
    ret = pd.read_csv(DAILY_RETURNS_PATH, usecols=["date", ticker])
    close["date"] = pd.to_datetime(close["date"])
    ret["date"] = pd.to_datetime(ret["date"])
    out = close.merge(ret, on="date", how="inner", suffixes=("_price", "_return"))
    out.columns = ["date", "spy_price", "spy_daily_return"]
    out["spy_price"] = pd.to_numeric(out["spy_price"], errors="coerce")
    out["spy_daily_return"] = pd.to_numeric(out["spy_daily_return"], errors="coerce")
    out = out.dropna(subset=["spy_price"]).sort_values("date").reset_index(drop=True)
    return out


def build_base_panel(ticker: str = DEFAULT_TICKER) -> pd.DataFrame:
    spy = load_spy_data(ticker)
    rf = read_rate_csv(DTB3_PATH)
    panel = spy.merge(rf, on="date", how="left")
    panel["daily_rf"] = panel["daily_rf"].ffill()
    panel["DTB3"] = panel["DTB3"].ffill()
    panel = panel.dropna(subset=["spy_price", "daily_rf"]).copy()
    panel["cash_nav"] = (1.0 + panel["daily_rf"]).cumprod()
    panel["month"] = panel["date"].dt.to_period("M")
    return panel.reset_index(drop=True)


def month_end_signal_panel(panel: pd.DataFrame) -> pd.DataFrame:
    monthly = panel.groupby("month", as_index=False).tail(1).copy()
    monthly["spy_12m_return"] = monthly["spy_price"].pct_change(12)
    monthly["rf_12m_return"] = monthly["cash_nav"].pct_change(12)
    monthly["excess_momentum"] = monthly["spy_12m_return"] - monthly["rf_12m_return"]
    monthly["signal"] = np.where(monthly["excess_momentum"] > 0, "SPY", "CASH")
    monthly["next_month_weight_spy"] = np.where(monthly["signal"] == "SPY", 1.0, 0.0)
    monthly["next_month_weight_cash"] = 1.0 - monthly["next_month_weight_spy"]
    monthly["signal_date"] = monthly["date"]
    monthly["effective_date"] = monthly["date"].shift(-1)
    return monthly[
        [
            "month",
            "signal_date",
            "effective_date",
            "spy_price",
            "spy_12m_return",
            "rf_12m_return",
            "excess_momentum",
            "signal",
            "next_month_weight_spy",
            "next_month_weight_cash",
        ]
    ].reset_index(drop=True)


def apply_monthly_signal(panel: pd.DataFrame, monthly_signals: pd.DataFrame, transaction_cost_bps: float) -> pd.DataFrame:
    out = panel.copy()
    out["signal"] = pd.Series(index=out.index, dtype="object")
    out["signal_date"] = pd.NaT
    out["effective_date"] = pd.NaT
    out["spy_12m_return"] = np.nan
    out["rf_12m_return"] = np.nan
    out["excess_momentum"] = np.nan
    out["target_weight_spy"] = np.nan
    out["target_weight_cash"] = np.nan

    signal_map = monthly_signals.dropna(subset=["effective_date"]).copy()
    signal_map["effective_date"] = pd.to_datetime(signal_map["effective_date"])
    for _, row in signal_map.iterrows():
        mask = out["date"] >= row["effective_date"]
        out.loc[mask, "signal"] = row["signal"]
        out.loc[mask, "signal_date"] = row["signal_date"]
        out.loc[mask, "effective_date"] = row["effective_date"]
        out.loc[mask, "spy_12m_return"] = row["spy_12m_return"]
        out.loc[mask, "rf_12m_return"] = row["rf_12m_return"]
        out.loc[mask, "excess_momentum"] = row["excess_momentum"]
        out.loc[mask, "target_weight_spy"] = row["next_month_weight_spy"]
        out.loc[mask, "target_weight_cash"] = row["next_month_weight_cash"]

    out["signal"] = out["signal"].fillna("CASH")
    out["target_weight_spy"] = out["target_weight_spy"].fillna(0.0)
    out["target_weight_cash"] = out["target_weight_cash"].fillna(1.0)

    out["turnover_flag"] = (out["target_weight_spy"] != out["target_weight_spy"].shift(1)).fillna(False)
    out.loc[out.index[0], "turnover_flag"] = False
    out["transaction_cost"] = np.where(out["turnover_flag"], transaction_cost_bps / 10000.0, 0.0)
    out["strategy_return"] = out["target_weight_spy"] * out["spy_daily_return"].fillna(0.0) + out["target_weight_cash"] * out["daily_rf"] - out["transaction_cost"]
    out["strategy_nav"] = (1.0 + out["strategy_return"]).cumprod()
    return out


def add_benchmarks(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["spy_buy_hold_return"] = out["spy_daily_return"].fillna(0.0)
    out["spy_buy_hold_nav"] = (1.0 + out["spy_buy_hold_return"]).cumprod()
    out["cash_only_return"] = out["daily_rf"]
    out["cash_only_nav"] = (1.0 + out["cash_only_return"]).cumprod()
    return out


def build_daily_output(base_panel: pd.DataFrame, monthly_signals: pd.DataFrame) -> pd.DataFrame:
    with_cost = apply_monthly_signal(base_panel, monthly_signals, transaction_cost_bps=SWITCH_COST_BPS)
    no_cost = apply_monthly_signal(base_panel, monthly_signals, transaction_cost_bps=0.0)
    out = add_benchmarks(with_cost)
    out["strategy_return_no_cost"] = no_cost["strategy_return"]
    out["strategy_nav_no_cost"] = no_cost["strategy_nav"]
    valid_signals = monthly_signals.loc[monthly_signals["excess_momentum"].notna() & monthly_signals["effective_date"].notna()].copy()
    first_live_date = valid_signals["effective_date"].min() if not valid_signals.empty else pd.NaT
    if pd.notna(first_live_date):
        out = out.loc[out["date"] >= first_live_date].copy()
        out["cash_nav"] = (1.0 + out["daily_rf"]).cumprod()
        out["strategy_nav"] = (1.0 + out["strategy_return"]).cumprod()
        out["strategy_nav_no_cost"] = (1.0 + out["strategy_return_no_cost"]).cumprod()
        out["spy_buy_hold_nav"] = (1.0 + out["spy_buy_hold_return"]).cumprod()
        out["cash_only_nav"] = (1.0 + out["cash_only_return"]).cumprod()
    return out.reset_index(drop=True)


def annualized_return(returns: pd.Series) -> float:
    s = returns.dropna()
    if s.empty:
        return np.nan
    return float((1.0 + s).prod() ** (TRADING_DAYS / len(s)) - 1.0)


def max_drawdown_from_returns(returns: pd.Series) -> float:
    s = returns.dropna()
    if s.empty:
        return np.nan
    wealth = (1.0 + s).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def perf_row(strategy: str, returns: pd.Series, rf_daily: pd.Series, nav: pd.Series, switches: int | float = np.nan) -> dict[str, object]:
    s = returns.dropna()
    rf = rf_daily.loc[s.index]
    excess = s - rf
    ann_ret = annualized_return(s)
    ann_vol = float(s.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(s) > 1 else np.nan
    if strategy == "CASH_ONLY":
        sharpe = 0.0
    else:
        ex_std = excess.std(ddof=1)
        sharpe = float(excess.mean() / ex_std * np.sqrt(TRADING_DAYS)) if pd.notna(ex_std) and ex_std != 0 else np.nan
    neg = excess[excess < 0]
    sortino = float(excess.mean() / neg.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(neg) > 1 and neg.std(ddof=1) != 0 else np.nan
    mdd = max_drawdown_from_returns(s)
    calmar = float(ann_ret / abs(mdd)) if pd.notna(mdd) and mdd < 0 else np.nan
    monthly = (1.0 + s).groupby(s.index.to_period("M")).prod() - 1.0
    return {
        "strategy": strategy,
        "start_date": s.index.min().date().isoformat(),
        "end_date": s.index.max().date().isoformat(),
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
        "calmar_ratio": calmar,
        "positive_day_ratio": float((s > 0).mean()),
        "positive_month_ratio": float((monthly > 0).mean()) if not monthly.empty else np.nan,
        "final_nav": float(nav.loc[s.index[-1]]),
        "number_of_switches": switches,
        "avg_turnover_per_year": float(switches / ((len(s) / TRADING_DAYS))) if pd.notna(switches) and len(s) > 0 else np.nan,
        "sortino_ratio": sortino,
    }


def performance_summary(panel: pd.DataFrame) -> pd.DataFrame:
    idx = pd.DatetimeIndex(panel["date"])
    rf = pd.Series(panel["daily_rf"].to_numpy(), index=idx)
    rows = [
        perf_row("ABS_MOM_12M_SPY_CASH", pd.Series(panel["strategy_return"].to_numpy(), index=idx), rf, pd.Series(panel["strategy_nav"].to_numpy(), index=idx), int(panel["turnover_flag"].sum())),
        perf_row("ABS_MOM_12M_SPY_CASH_NO_COST", pd.Series(panel["strategy_return_no_cost"].to_numpy(), index=idx), rf, pd.Series(panel["strategy_nav_no_cost"].to_numpy(), index=idx), int(panel["turnover_flag"].sum())),
        perf_row("SPY_BUY_HOLD", pd.Series(panel["spy_buy_hold_return"].to_numpy(), index=idx), rf, pd.Series(panel["spy_buy_hold_nav"].to_numpy(), index=idx), 0),
        perf_row("CASH_ONLY", pd.Series(panel["cash_only_return"].to_numpy(), index=idx), rf, pd.Series(panel["cash_only_nav"].to_numpy(), index=idx), 0),
    ]
    return pd.DataFrame(rows)


def yearly_returns(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["year"] = df["date"].dt.year
    rows = []
    for year, grp in df.groupby("year", observed=False):
        rows.append(
            {
                "year": int(year),
                "ABS_MOM_12M_SPY_CASH": float((1.0 + grp["strategy_return"]).prod() - 1.0),
                "ABS_MOM_12M_SPY_CASH_NO_COST": float((1.0 + grp["strategy_return_no_cost"]).prod() - 1.0),
                "SPY_BUY_HOLD": float((1.0 + grp["spy_buy_hold_return"]).prod() - 1.0),
                "CASH_ONLY": float((1.0 + grp["cash_only_return"]).prod() - 1.0),
            }
        )
    return pd.DataFrame(rows)


def drawdown_episodes(nav: pd.Series, strategy: str, top_n: int = 10) -> pd.DataFrame:
    wealth = nav.dropna()
    dd = wealth / wealth.cummax() - 1.0
    in_dd = dd < 0
    episodes = []
    start = None
    for i, flag in enumerate(in_dd):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            end = i - 1
            segment = dd.iloc[start : end + 1]
            trough_idx = segment.idxmin()
            peak_date = wealth.index[start - 1] if start > 0 else wealth.index[start]
            trough_date = trough_idx
            recovery_date = wealth.index[i]
            episodes.append(
                {
                    "strategy": strategy,
                    "drawdown_start": peak_date,
                    "drawdown_trough": trough_date,
                    "drawdown_recovery": recovery_date,
                    "max_drawdown": float(segment.min()),
                    "duration_days": int((recovery_date - peak_date).days),
                    "recovery_days": int((recovery_date - trough_date).days),
                }
            )
            start = None
    if start is not None:
        segment = dd.iloc[start:]
        trough_idx = segment.idxmin()
        peak_date = wealth.index[start - 1] if start > 0 else wealth.index[start]
        episodes.append(
            {
                "strategy": strategy,
                "drawdown_start": peak_date,
                "drawdown_trough": trough_idx,
                "drawdown_recovery": pd.NaT,
                "max_drawdown": float(segment.min()),
                "duration_days": int((wealth.index[-1] - peak_date).days),
                "recovery_days": np.nan,
            }
        )
    out = pd.DataFrame(episodes)
    if out.empty:
        return out
    return out.sort_values("max_drawdown").head(top_n).reset_index(drop=True)


def drawdown_summary(panel: pd.DataFrame) -> pd.DataFrame:
    idx = pd.DatetimeIndex(panel["date"])
    frames = [
        drawdown_episodes(pd.Series(panel["strategy_nav"].to_numpy(), index=idx), "ABS_MOM_12M_SPY_CASH"),
        drawdown_episodes(pd.Series(panel["strategy_nav_no_cost"].to_numpy(), index=idx), "ABS_MOM_12M_SPY_CASH_NO_COST"),
        drawdown_episodes(pd.Series(panel["spy_buy_hold_nav"].to_numpy(), index=idx), "SPY_BUY_HOLD"),
        drawdown_episodes(pd.Series(panel["cash_only_nav"].to_numpy(), index=idx), "CASH_ONLY"),
    ]
    return pd.concat(frames, ignore_index=True)


def plot_outputs(panel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(panel["date"], panel["strategy_nav"], label="ABS_MOM_12M_SPY_CASH")
    ax.plot(panel["date"], panel["spy_buy_hold_nav"], label="SPY_BUY_HOLD")
    ax.plot(panel["date"], panel["cash_only_nav"], label="CASH_ONLY")
    ax.set_yscale("log")
    ax.set_title("Absolute Momentum SPY vs Cash - Log NAV")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_LOG_PATH, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(panel["date"], panel["strategy_nav"], label="ABS_MOM_12M_SPY_CASH")
    ax.plot(panel["date"], panel["spy_buy_hold_nav"], label="SPY_BUY_HOLD")
    ax.plot(panel["date"], panel["cash_only_nav"], label="CASH_ONLY")
    ax.set_title("Absolute Momentum SPY vs Cash - Linear NAV")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_LINEAR_PATH, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for label, nav_col in [("ABS_MOM_12M_SPY_CASH", "strategy_nav"), ("SPY_BUY_HOLD", "spy_buy_hold_nav")]:
        wealth = panel[nav_col]
        dd = wealth / wealth.cummax() - 1.0
        ax.plot(panel["date"], dd, label=label)
    ax.set_title("Drawdown Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DD_PATH, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(panel["date"], panel["spy_price"] / panel["spy_price"].iloc[0], color="black", linewidth=1.0, label="SPY normalized")
    ax.fill_between(panel["date"], 0, 1, where=panel["target_weight_spy"] > 0.5, color="tab:blue", alpha=0.15, transform=ax.get_xaxis_transform(), label="Hold SPY")
    ax.fill_between(panel["date"], 0, 1, where=panel["target_weight_cash"] > 0.5, color="tab:green", alpha=0.12, transform=ax.get_xaxis_transform(), label="Hold CASH")
    ax.set_title("Monthly Signal Overlay")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_SIGNAL_OVERLAY_PATH, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(panel["date"], panel["excess_momentum"], color="tab:purple")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("12M Excess Momentum Signal")
    fig.tight_layout()
    fig.savefig(FIG_MOM_PATH, dpi=180)
    plt.close(fig)

    roll_window = 756
    out = panel.copy()
    out["roll_abs_mom_ret"] = np.nan
    out["roll_spy_ret"] = np.nan
    out["roll_abs_mom_dd"] = np.nan
    out["roll_spy_dd"] = np.nan
    for i in range(roll_window - 1, len(out)):
        sl = out.iloc[i - roll_window + 1 : i + 1]
        out.loc[out.index[i], "roll_abs_mom_ret"] = annualized_return(sl["strategy_return"])
        out.loc[out.index[i], "roll_spy_ret"] = annualized_return(sl["spy_buy_hold_return"])
        out.loc[out.index[i], "roll_abs_mom_dd"] = max_drawdown_from_returns(sl["strategy_return"])
        out.loc[out.index[i], "roll_spy_dd"] = max_drawdown_from_returns(sl["spy_buy_hold_return"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(out["date"], out["roll_abs_mom_ret"], label="ABS_MOM")
    axes[0].plot(out["date"], out["roll_spy_ret"], label="SPY")
    axes[0].set_title("Rolling 3Y Annualized Return")
    axes[0].legend()
    axes[1].plot(out["date"], out["roll_abs_mom_dd"], label="ABS_MOM")
    axes[1].plot(out["date"], out["roll_spy_dd"], label="SPY")
    axes[1].set_title("Rolling 3Y Max Drawdown")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_ROLLING_PATH, dpi=180)
    plt.close(fig)


def print_diagnostics(panel: pd.DataFrame, perf: pd.DataFrame, yearly: pd.DataFrame) -> None:
    years = len(panel) / TRADING_DAYS
    switches = int(panel["turnover_flag"].sum())
    abs_row = perf.loc[perf["strategy"] == "ABS_MOM_12M_SPY_CASH"].iloc[0]
    spy_row = perf.loc[perf["strategy"] == "SPY_BUY_HOLD"].iloc[0]
    outperform = yearly.loc[(yearly["ABS_MOM_12M_SPY_CASH"] - yearly["SPY_BUY_HOLD"]) > 0.05, "year"].tolist()
    underperform = yearly.loc[(yearly["ABS_MOM_12M_SPY_CASH"] - yearly["SPY_BUY_HOLD"]) < -0.05, "year"].tolist()
    whipsaw_months = panel.loc[panel["turnover_flag"], ["date", "signal"]].copy()
    whipsaw_count = 0
    if len(whipsaw_months) >= 2:
        gaps = whipsaw_months["date"].diff().dt.days
        whipsaw_count = int((gaps <= 40).sum())
    print(f"Sample: {panel['date'].min().date()} to {panel['date'].max().date()}")
    print(f"Total switches: {switches}")
    print(f"Average switches per year: {switches / years:.2f}")
    print(f"ABS_MOM lowers max drawdown vs SPY: {abs_row['max_drawdown'] > spy_row['max_drawdown']}")
    print(f"ABS_MOM improves Sharpe vs SPY: {abs_row['sharpe_ratio'] > spy_row['sharpe_ratio']}")
    print(f"ABS_MOM sacrifices long-term return vs SPY: {abs_row['annualized_return'] < spy_row['annualized_return']}")
    print(f"Years clearly outperforming SPY: {outperform}")
    print(f"Years clearly underperforming SPY: {underperform}")
    print(f"Potential whipsaw count (switches within ~40 days): {whipsaw_count}")


def main() -> None:
    ensure_dirs()
    base = build_base_panel(DEFAULT_TICKER)
    monthly = month_end_signal_panel(base)
    daily = build_daily_output(base, monthly)

    daily_out = daily[
        [
            "date",
            "spy_price",
            "spy_daily_return",
            "daily_rf",
            "cash_nav",
            "spy_12m_return",
            "rf_12m_return",
            "excess_momentum",
            "signal",
            "signal_date",
            "effective_date",
            "target_weight_spy",
            "target_weight_cash",
            "strategy_return",
            "strategy_nav",
            "spy_buy_hold_nav",
            "cash_only_nav",
            "turnover_flag",
            "transaction_cost",
            "strategy_return_no_cost",
            "strategy_nav_no_cost",
        ]
    ].copy()
    daily_out.to_csv(DAILY_PANEL_PATH, index=False)

    monthly_out = monthly[
        [
            "signal_date",
            "effective_date",
            "spy_price",
            "spy_12m_return",
            "rf_12m_return",
            "excess_momentum",
            "signal",
            "next_month_weight_spy",
            "next_month_weight_cash",
        ]
    ].rename(columns={"signal_date": "date"}).copy()
    monthly_out.to_csv(MONTHLY_SIGNAL_PATH, index=False)

    perf = performance_summary(daily)
    perf.to_csv(PERF_PATH, index=False)

    yearly = yearly_returns(daily)
    yearly.to_csv(YEARLY_PATH, index=False)

    dd = drawdown_summary(daily)
    dd.to_csv(DD_PATH, index=False)

    plot_outputs(daily)
    print_diagnostics(daily, perf, yearly)

    for path in [
        DAILY_PANEL_PATH,
        MONTHLY_SIGNAL_PATH,
        PERF_PATH,
        YEARLY_PATH,
        DD_PATH,
        FIG_LOG_PATH,
        FIG_LINEAR_PATH,
        FIG_DD_PATH,
        FIG_SIGNAL_OVERLAY_PATH,
        FIG_MOM_PATH,
        FIG_ROLLING_PATH,
    ]:
        if path.exists():
            print(f"Saved: {path}")


if __name__ == "__main__":
    main()
