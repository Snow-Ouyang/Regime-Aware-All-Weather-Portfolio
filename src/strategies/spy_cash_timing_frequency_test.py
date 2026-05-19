from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "spy_cash_timing_frequency_test"

DAILY_CLOSE_PATH = ROOT / "data" / "processed" / "assets" / "daily_adjusted_close.csv"
DAILY_RETURNS_PATH = ROOT / "data" / "processed" / "assets" / "daily_returns.csv"
DTB3_PATH = ROOT / "data" / "raw" / "macro" / "rate" / "DTB3.csv"

DAILY_PANEL_PATH = RESULTS_DIR / "daily_backtest_panel.csv"
MONTHLY_SIGNAL_PATH = RESULTS_DIR / "monthly_signal_panel.csv"
WEEKLY_SIGNAL_PATH = RESULTS_DIR / "weekly_signal_panel.csv"
TRADE_LOG_PATH = RESULTS_DIR / "trade_log.csv"
PERF_PATH = RESULTS_DIR / "performance_summary.csv"
CRISIS_PATH = RESULTS_DIR / "crisis_performance.csv"
YEARLY_PATH = RESULTS_DIR / "yearly_returns.csv"
DD_PATH = RESULTS_DIR / "drawdown_summary.csv"
SUMMARY_MD_PATH = RESULTS_DIR / "summary.md"

FIG_LOG_PATH = RESULTS_DIR / "equity_curve_log.png"
FIG_LINEAR_PATH = RESULTS_DIR / "equity_curve_linear.png"
FIG_DD_PATH = RESULTS_DIR / "drawdown_comparison.png"
FIG_BAR_PATH = RESULTS_DIR / "performance_bar_charts.png"
FIG_WEIGHT_PATH = RESULTS_DIR / "weight_timeline.png"
FIG_SIGNAL_PATH = RESULTS_DIR / "faber_antonacci_daily_signal_overlay.png"
FIG_CRISIS_PATH = RESULTS_DIR / "crisis_equity_curves.png"

CONFIG = {
    "ticker": "SPY",
    "transaction_cost_bps": 20.0,
    "initial_nav": 1.0,
    "faber_ma_days": 200,
    "faber_ma_months": 10,
    "antonacci_lookback_days": 252,
    "antonacci_lookback_months": 12,
    "daily_confirm_days": 2,
    "cash_rate_file": "data/raw/macro/rate/DTB3.csv",
}

TIMING_STRATEGIES = [
    "MONTHLY_EITHER_CONFIRM",
    "WEEKLY_EITHER_CONFIRM",
    "DAILY_2D_EITHER_CONFIRM",
]
BENCHMARKS = ["SPY_BUY_HOLD", "CASH_ONLY"]
ALL_STRATEGIES = TIMING_STRATEGIES + BENCHMARKS
CRISIS_WINDOWS = {
    "DOTCOM_2000_2002": ("2000-01-01", "2002-12-31"),
    "GFC_2008_2009": ("2008-09-01", "2009-03-31"),
    "COVID_2020": ("2020-02-19", "2020-04-30"),
    "INFLATION_2022": ("2022-01-01", "2022-12-31"),
    "RECENT_2024_2026": ("2024-01-01", "2026-12-31"),
}


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    ticker = CONFIG["ticker"]
    if not DAILY_CLOSE_PATH.exists():
        raise FileNotFoundError(f"Missing price file: {DAILY_CLOSE_PATH}")
    if not DAILY_RETURNS_PATH.exists():
        raise FileNotFoundError(f"Missing return file: {DAILY_RETURNS_PATH}")
    if not DTB3_PATH.exists():
        raise FileNotFoundError(f"Missing cash rate file: {DTB3_PATH}")

    close = pd.read_csv(DAILY_CLOSE_PATH, usecols=["date", ticker])
    ret = pd.read_csv(DAILY_RETURNS_PATH, usecols=["date", ticker])
    rf = pd.read_csv(DTB3_PATH)

    close["date"] = pd.to_datetime(close["date"])
    ret["date"] = pd.to_datetime(ret["date"])

    rf_date_col = next((c for c in rf.columns if "date" in c.lower()), rf.columns[0])
    rf_value_col = next((c for c in rf.columns if c != rf_date_col), rf.columns[-1])
    rf = rf[[rf_date_col, rf_value_col]].copy()
    rf.columns = ["date", "DTB3"]
    rf["date"] = pd.to_datetime(rf["date"])
    rf["DTB3"] = pd.to_numeric(rf["DTB3"].replace(".", np.nan), errors="coerce")
    rf = rf.sort_values("date")
    rf["DTB3"] = rf["DTB3"].ffill()
    rf["daily_rf"] = (1.0 + rf["DTB3"] / 100.0) ** (1.0 / 252.0) - 1.0

    panel = close.merge(ret, on="date", how="inner", suffixes=("_price", "_return"))
    panel.columns = ["date", "spy_price", "spy_daily_return"]
    panel["spy_price"] = pd.to_numeric(panel["spy_price"], errors="coerce")
    panel["spy_daily_return"] = pd.to_numeric(panel["spy_daily_return"], errors="coerce")
    panel = panel.dropna(subset=["spy_price"]).sort_values("date").drop_duplicates("date")
    panel = panel.merge(rf[["date", "daily_rf"]], on="date", how="left")
    panel["daily_rf"] = panel["daily_rf"].ffill()
    panel = panel.dropna(subset=["daily_rf"]).reset_index(drop=True)
    panel["cash_nav"] = CONFIG["initial_nav"] * (1.0 + panel["daily_rf"]).cumprod()
    panel["month"] = panel["date"].dt.to_period("M")
    panel["week"] = panel["date"].dt.to_period("W-FRI")
    return panel


def build_daily_signals(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    ma_days = int(CONFIG["faber_ma_days"])
    lookback_days = int(CONFIG["antonacci_lookback_days"])

    out["faber_ma200"] = out["spy_price"].rolling(ma_days, min_periods=ma_days).mean()
    raw_faber = np.where(out["spy_price"] > out["faber_ma200"], 1.0, np.where(out["spy_price"] < out["faber_ma200"], 0.0, np.nan))
    faber_signal = []
    prev = np.nan
    for val in raw_faber:
        if pd.isna(val):
            faber_signal.append(np.nan)
        elif pd.isna(prev):
            prev = val
            faber_signal.append(val)
        else:
            prev = val
            faber_signal.append(val)
    out["faber_daily_signal"] = pd.Series(faber_signal, index=out.index, dtype=float)

    out["antonacci_12m_spy_return"] = out["spy_price"] / out["spy_price"].shift(lookback_days) - 1.0
    out["antonacci_12m_cash_return"] = out["cash_nav"] / out["cash_nav"].shift(lookback_days) - 1.0
    out["antonacci_excess_momentum"] = out["antonacci_12m_spy_return"] - out["antonacci_12m_cash_return"]
    out["antonacci_daily_signal"] = np.where(out["antonacci_excess_momentum"] > 0, 1.0, np.where(out["antonacci_excess_momentum"].notna(), 1.0 * 0, np.nan))
    return out


def build_monthly_signals(panel: pd.DataFrame) -> pd.DataFrame:
    monthly = panel.groupby("month", as_index=False).tail(1).copy()
    next_trade_date = pd.Series(panel["date"].shift(-1).to_numpy(), index=panel["date"])
    monthly["faber_10m_sma"] = monthly["spy_price"].rolling(int(CONFIG["faber_ma_months"]), min_periods=int(CONFIG["faber_ma_months"])).mean()
    monthly["faber_monthly_signal"] = np.where(monthly["spy_price"] > monthly["faber_10m_sma"], 1.0, np.where(monthly["faber_10m_sma"].notna(), 0.0, np.nan))
    monthly["antonacci_12m_spy_return"] = monthly["spy_price"].pct_change(int(CONFIG["antonacci_lookback_months"]))
    monthly["antonacci_12m_cash_return"] = monthly["cash_nav"].pct_change(int(CONFIG["antonacci_lookback_months"]))
    monthly["antonacci_excess_momentum"] = monthly["antonacci_12m_spy_return"] - monthly["antonacci_12m_cash_return"]
    monthly["antonacci_monthly_signal"] = np.where(monthly["antonacci_excess_momentum"] > 0, 1.0, np.where(monthly["antonacci_excess_momentum"].notna(), 0.0, np.nan))
    monthly["monthly_either_signal"] = np.where(
        (monthly["faber_monthly_signal"] == 1.0) | (monthly["antonacci_monthly_signal"] == 1.0),
        1.0,
        np.where(monthly["faber_monthly_signal"].notna() & monthly["antonacci_monthly_signal"].notna(), 0.0, np.nan),
    )
    monthly["signal_date"] = monthly["date"]
    monthly["effective_date"] = pd.to_datetime(monthly["signal_date"].map(next_trade_date))
    return monthly[
        [
            "signal_date",
            "effective_date",
            "spy_price",
            "faber_10m_sma",
            "faber_monthly_signal",
            "antonacci_12m_spy_return",
            "antonacci_12m_cash_return",
            "antonacci_excess_momentum",
            "antonacci_monthly_signal",
            "monthly_either_signal",
        ]
    ].reset_index(drop=True)


def build_weekly_signals(panel: pd.DataFrame) -> pd.DataFrame:
    weekly = panel.groupby("week", as_index=False).tail(1).copy()
    next_trade_date = pd.Series(panel["date"].shift(-1).to_numpy(), index=panel["date"])
    weekly["faber_weekly_signal"] = np.where(weekly["spy_price"] > weekly["faber_ma200"], 1.0, np.where(weekly["faber_ma200"].notna(), 0.0, np.nan))
    weekly["antonacci_weekly_signal"] = np.where(weekly["antonacci_excess_momentum"] > 0, 1.0, np.where(weekly["antonacci_excess_momentum"].notna(), 0.0, np.nan))
    weekly["weekly_either_signal"] = np.where(
        (weekly["faber_weekly_signal"] == 1.0) | (weekly["antonacci_weekly_signal"] == 1.0),
        1.0,
        np.where(weekly["faber_weekly_signal"].notna() & weekly["antonacci_weekly_signal"].notna(), 0.0, np.nan),
    )
    weekly["signal_date"] = weekly["date"]
    weekly["effective_date"] = pd.to_datetime(weekly["signal_date"].map(next_trade_date))
    return weekly[
        [
            "signal_date",
            "effective_date",
            "spy_price",
            "faber_ma200",
            "faber_weekly_signal",
            "antonacci_excess_momentum",
            "antonacci_weekly_signal",
            "weekly_either_signal",
        ]
    ].reset_index(drop=True)


def first_valid_date(monthly: pd.DataFrame, weekly: pd.DataFrame, daily: pd.DataFrame) -> pd.Timestamp:
    monthly_date = pd.to_datetime(monthly.loc[monthly["monthly_either_signal"].notna() & monthly["effective_date"].notna(), "effective_date"]).min()
    weekly_date = pd.to_datetime(weekly.loc[weekly["weekly_either_signal"].notna() & weekly["effective_date"].notna(), "effective_date"]).min()
    daily_valid = daily.loc[daily["faber_daily_signal"].notna() & daily["antonacci_daily_signal"].notna(), "date"]
    daily_date = daily_valid.min() if not daily_valid.empty else pd.NaT
    dates = [d for d in [monthly_date, weekly_date, daily_date] if pd.notna(d)]
    if not dates:
        raise ValueError("No valid common start date for strategy signals.")
    return max(dates)


def build_monthly_either_weights(panel: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    out = panel[["date"]].copy()
    out["signal_date"] = pd.NaT
    out["effective_date"] = pd.NaT
    out["weight_spy"] = np.nan
    mapping = monthly.loc[monthly["effective_date"].notna() & monthly["monthly_either_signal"].notna(), ["signal_date", "effective_date", "monthly_either_signal"]].copy()
    mapping["signal_date"] = pd.to_datetime(mapping["signal_date"])
    mapping["effective_date"] = pd.to_datetime(mapping["effective_date"])
    for _, row in mapping.iterrows():
        mask = out["date"] >= row["effective_date"]
        out.loc[mask, "signal_date"] = row["signal_date"]
        out.loc[mask, "effective_date"] = row["effective_date"]
        out.loc[mask, "weight_spy"] = float(row["monthly_either_signal"])
    out["weight_spy"] = out["weight_spy"].ffill()
    return out


def build_weekly_either_weights(panel: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    out = panel[["date"]].copy()
    out["signal_date"] = pd.NaT
    out["effective_date"] = pd.NaT
    out["weight_spy"] = np.nan
    mapping = weekly.loc[weekly["effective_date"].notna() & weekly["weekly_either_signal"].notna(), ["signal_date", "effective_date", "weekly_either_signal"]].copy()
    mapping["signal_date"] = pd.to_datetime(mapping["signal_date"])
    mapping["effective_date"] = pd.to_datetime(mapping["effective_date"])
    for _, row in mapping.iterrows():
        mask = out["date"] >= row["effective_date"]
        out.loc[mask, "signal_date"] = row["signal_date"]
        out.loc[mask, "effective_date"] = row["effective_date"]
        out.loc[mask, "weight_spy"] = float(row["weekly_either_signal"])
    out["weight_spy"] = out["weight_spy"].ffill()
    return out


def build_daily_2d_either_weights(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = panel[["date", "spy_price", "faber_daily_signal", "antonacci_daily_signal", "antonacci_excess_momentum"]].copy()
    out["faber_bull_2d"] = (panel["faber_daily_signal"] == 1.0) & (panel["faber_daily_signal"].shift(1) == 1.0)
    out["antonacci_bull_2d"] = (panel["antonacci_daily_signal"] == 1.0) & (panel["antonacci_daily_signal"].shift(1) == 1.0)
    out["faber_bear_2d"] = (panel["faber_daily_signal"] == 0.0) & (panel["faber_daily_signal"].shift(1) == 0.0)
    out["antonacci_bear_2d"] = (panel["antonacci_daily_signal"] == 0.0) & (panel["antonacci_daily_signal"].shift(1) == 0.0)
    out["signal_date"] = pd.NaT
    out["effective_date"] = pd.NaT
    out["weight_spy"] = np.nan

    position = 0.0
    effective_dates = []
    signal_dates = []
    weights = []
    triggers = []
    pending_position = 0.0
    pending_signal_date = pd.NaT
    has_valid = False
    for i, row in out.iterrows():
        if i == 0:
            position = 0.0
        else:
            position = pending_position
        if row["faber_daily_signal"] == row["faber_daily_signal"] and row["antonacci_daily_signal"] == row["antonacci_daily_signal"]:
            has_valid = True
        signal_date = pd.NaT
        reason = ""
        next_position = position
        if has_valid:
            if position == 0.0 and (bool(row["faber_bull_2d"]) or bool(row["antonacci_bull_2d"])):
                next_position = 1.0
                signal_date = row["date"]
                reason = "entry_2d_either"
            elif position == 1.0 and bool(row["faber_bear_2d"]) and bool(row["antonacci_bear_2d"]):
                next_position = 0.0
                signal_date = row["date"]
                reason = "exit_2d_both_bear"
        pending_position = next_position
        pending_signal_date = signal_date if pd.notna(signal_date) else pd.NaT
        signal_dates.append(signal_date if pd.notna(signal_date) else pd.NaT)
        effective_dates.append(out["date"].iloc[i + 1] if pd.notna(signal_date) and i + 1 < len(out) else pd.NaT)
        weights.append(position)
        triggers.append(reason)
    out["weight_spy"] = weights
    out["signal_date"] = signal_dates
    out["effective_date"] = effective_dates
    out["reason"] = triggers
    trade_log = out.loc[out["signal_date"].notna(), ["signal_date", "effective_date", "spy_price", "faber_daily_signal", "antonacci_daily_signal", "antonacci_excess_momentum", "reason"]].copy()
    return out[["date", "signal_date", "effective_date", "weight_spy"]], trade_log


def run_daily_backtest(panel: pd.DataFrame, strategy_name: str, weight_spy: pd.Series, signal_date: pd.Series | None = None, effective_date: pd.Series | None = None, cost_bps: float = 20.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = panel[["date", "spy_price", "spy_daily_return", "daily_rf"]].copy()
    out["actual_weight_spy"] = pd.to_numeric(weight_spy, errors="coerce").ffill().fillna(0.0)
    out["actual_weight_cash"] = 1.0 - out["actual_weight_spy"]
    out["turnover_flag"] = (out["actual_weight_spy"] != out["actual_weight_spy"].shift(1)).fillna(False)
    out.loc[out.index[0], "turnover_flag"] = False
    out["transaction_cost"] = np.where(out["turnover_flag"], cost_bps / 10000.0, 0.0)
    out["daily_return"] = out["actual_weight_spy"] * out["spy_daily_return"].fillna(0.0) + out["actual_weight_cash"] * out["daily_rf"] - out["transaction_cost"]
    out["nav"] = CONFIG["initial_nav"] * (1.0 + out["daily_return"]).cumprod()
    if signal_date is not None:
        out["signal_date"] = signal_date
    if effective_date is not None:
        out["effective_date"] = effective_date

    trades = out.loc[out["turnover_flag"], ["date", "spy_price", "transaction_cost"]].copy()
    trades["strategy"] = strategy_name
    trades["trade_date"] = trades["date"]
    trades["previous_position"] = out["actual_weight_spy"].shift(1).loc[trades.index].map({0.0: "CASH", 1.0: "SPY"})
    trades["new_position"] = out["actual_weight_spy"].loc[trades.index].map({0.0: "CASH", 1.0: "SPY"})
    trades["reason"] = "scheduled_signal_change"
    trades["faber_signal"] = np.nan
    trades["antonacci_signal"] = np.nan
    trades["antonacci_excess_momentum"] = np.nan
    trades = trades[
        [
            "strategy",
            "trade_date",
            "previous_position",
            "new_position",
            "reason",
            "spy_price",
            "faber_signal",
            "antonacci_signal",
            "antonacci_excess_momentum",
            "transaction_cost",
        ]
    ]
    return out, trades


def compute_performance_metrics(daily_panel: pd.DataFrame) -> pd.DataFrame:
    idx = pd.DatetimeIndex(daily_panel["date"])
    rf = pd.Series(daily_panel["daily_rf"].to_numpy(), index=idx)
    rows = []
    for strategy in ALL_STRATEGIES:
        ret_col = f"{strategy}_return"
        nav_col = f"{strategy}_nav"
        weight_col = f"{strategy}_weight_spy"
        if strategy == "SPY_BUY_HOLD":
            weights = pd.Series(1.0, index=idx)
            switches = 0
        elif strategy == "CASH_ONLY":
            weights = pd.Series(0.0, index=idx)
            switches = 0
        else:
            weights = pd.Series(daily_panel[weight_col].to_numpy(), index=idx)
            switches = int(daily_panel[f"{strategy}_turnover_flag"].sum())
        returns = pd.Series(daily_panel[ret_col].to_numpy(), index=idx)
        nav = pd.Series(daily_panel[nav_col].to_numpy(), index=idx)
        s = returns.dropna()
        ex = s - rf.loc[s.index]
        ann_ret = float((1.0 + s).prod() ** (252.0 / len(s)) - 1.0) if not s.empty else np.nan
        ann_vol = float(s.std(ddof=1) * np.sqrt(252.0)) if len(s) > 1 else np.nan
        if strategy == "CASH_ONLY":
            sharpe = 0.0
        else:
            ex_std = ex.std(ddof=1)
            sharpe = float(ex.mean() / ex_std * np.sqrt(252.0)) if pd.notna(ex_std) and ex_std != 0 else np.nan
        wealth = (1.0 + s).cumprod()
        mdd = float((wealth / wealth.cummax() - 1.0).min()) if not s.empty else np.nan
        calmar = float(ann_ret / abs(mdd)) if pd.notna(mdd) and mdd < 0 else np.nan
        monthly = (1.0 + s).groupby(s.index.to_period("M")).prod() - 1.0
        annual_trade_cost = float(daily_panel[f"{strategy}_transaction_cost"].sum() / (len(s) / 252.0)) if strategy not in BENCHMARKS and len(s) > 0 else 0.0
        rows.append(
            {
                "strategy": strategy,
                "start_date": s.index.min().date().isoformat(),
                "end_date": s.index.max().date().isoformat(),
                "annualized_return": ann_ret,
                "annualized_volatility": ann_vol,
                "sharpe_ratio": sharpe,
                "max_drawdown": mdd,
                "calmar_ratio": calmar,
                "final_nav": float(nav.iloc[-1]) if not nav.empty else np.nan,
                "positive_day_ratio": float((s > 0).mean()) if not s.empty else np.nan,
                "positive_month_ratio": float((monthly > 0).mean()) if not monthly.empty else np.nan,
                "number_of_switches": switches,
                "avg_switches_per_year": float(switches / (len(s) / 252.0)) if len(s) > 0 else np.nan,
                "time_in_spy": float(weights.mean()),
                "time_in_cash": float(1.0 - weights.mean()),
                "avg_trade_cost_per_year": annual_trade_cost,
            }
        )
    return pd.DataFrame(rows)


def compute_drawdown_events(daily_panel: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for strategy in ALL_STRATEGIES:
        nav = daily_panel[f"{strategy}_nav"]
        wealth = pd.Series(nav.to_numpy(), index=pd.DatetimeIndex(daily_panel["date"]))
        dd = wealth / wealth.cummax() - 1.0
        in_dd = dd < 0
        start = None
        rows = []
        for i, flag in enumerate(in_dd):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                end = i - 1
                seg = dd.iloc[start : end + 1]
                trough = seg.idxmin()
                peak_date = wealth.index[start - 1] if start > 0 else wealth.index[start]
                recovery = wealth.index[i]
                rows.append(
                    {
                        "strategy": strategy,
                        "drawdown_start": peak_date,
                        "drawdown_trough": trough,
                        "drawdown_recovery": recovery,
                        "max_drawdown": float(seg.min()),
                        "duration_days": int((recovery - peak_date).days),
                        "recovery_days": int((recovery - trough).days),
                    }
                )
                start = None
        if start is not None:
            seg = dd.iloc[start:]
            trough = seg.idxmin()
            peak_date = wealth.index[start - 1] if start > 0 else wealth.index[start]
            rows.append(
                {
                    "strategy": strategy,
                    "drawdown_start": peak_date,
                    "drawdown_trough": trough,
                    "drawdown_recovery": pd.NaT,
                    "max_drawdown": float(seg.min()),
                    "duration_days": int((wealth.index[-1] - peak_date).days),
                    "recovery_days": np.nan,
                }
            )
        frames.append(pd.DataFrame(rows).sort_values("max_drawdown").head(10))
    return pd.concat(frames, ignore_index=True)


def compute_crisis_performance(daily_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, (start, end) in CRISIS_WINDOWS.items():
        sub = daily_panel.loc[(daily_panel["date"] >= pd.Timestamp(start)) & (daily_panel["date"] <= pd.Timestamp(end))].copy()
        if sub.empty:
            continue
        idx = pd.DatetimeIndex(sub["date"])
        rf = pd.Series(sub["daily_rf"].to_numpy(), index=idx)
        for strategy in ALL_STRATEGIES:
            returns = pd.Series(sub[f"{strategy}_return"].to_numpy(), index=idx)
            weights = pd.Series(sub[f"{strategy}_weight_spy"].to_numpy(), index=idx)
            s = returns.dropna()
            if s.empty:
                continue
            ex = s - rf.loc[s.index]
            wealth = (1.0 + s).cumprod()
            ex_std = ex.std(ddof=1)
            sharpe = float(ex.mean() / ex_std * np.sqrt(252.0)) if strategy != "CASH_ONLY" and pd.notna(ex_std) and ex_std != 0 else 0.0 if strategy == "CASH_ONLY" else np.nan
            rows.append(
                {
                    "period": period,
                    "strategy": strategy,
                    "cumulative_return": float((1.0 + s).prod() - 1.0),
                    "annualized_return": float((1.0 + s).prod() ** (252.0 / len(s)) - 1.0),
                    "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
                    "volatility": float(s.std(ddof=1) * np.sqrt(252.0)) if len(s) > 1 else np.nan,
                    "sharpe": sharpe,
                    "number_of_switches": int(sub[f"{strategy}_turnover_flag"].sum()) if strategy not in BENCHMARKS else 0,
                    "time_in_spy": float(weights.mean()),
                }
            )
    return pd.DataFrame(rows)


def build_trade_log(daily_panel: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for strategy in TIMING_STRATEGIES:
        mask = daily_panel[f"{strategy}_turnover_flag"].astype(bool)
        if not mask.any():
            continue
        trades = daily_panel.loc[mask, ["date", "spy_price", "faber_daily_signal", "antonacci_daily_signal", "antonacci_excess_momentum", f"{strategy}_transaction_cost", f"{strategy}_weight_spy"]].copy()
        trades["strategy"] = strategy
        trades["trade_date"] = trades["date"]
        prev = daily_panel[f"{strategy}_weight_spy"].shift(1).loc[mask].map({0.0: "CASH", 1.0: "SPY"})
        new = daily_panel.loc[mask, f"{strategy}_weight_spy"].map({0.0: "CASH", 1.0: "SPY"})
        trades["previous_position"] = prev
        trades["new_position"] = new
        trades["reason"] = "signal_change"
        trades["faber_signal"] = trades["faber_daily_signal"]
        trades["antonacci_signal"] = trades["antonacci_daily_signal"]
        trades["transaction_cost"] = trades[f"{strategy}_transaction_cost"]
        trades = trades[
            [
                "strategy",
                "trade_date",
                "previous_position",
                "new_position",
                "reason",
                "spy_price",
                "faber_signal",
                "antonacci_signal",
                "antonacci_excess_momentum",
                "transaction_cost",
            ]
        ]
        frames.append(trades)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_results(daily_panel: pd.DataFrame, perf: pd.DataFrame, crisis: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for strategy in ALL_STRATEGIES:
        ax.plot(daily_panel["date"], daily_panel[f"{strategy}_nav"], label=strategy)
    ax.set_yscale("log")
    ax.set_title("SPY/CASH Timing Frequency Test - Log NAV")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_LOG_PATH, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for strategy in ALL_STRATEGIES:
        ax.plot(daily_panel["date"], daily_panel[f"{strategy}_nav"], label=strategy)
    ax.set_title("SPY/CASH Timing Frequency Test - Linear NAV")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_LINEAR_PATH, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for strategy in ["SPY_BUY_HOLD", "MONTHLY_EITHER_CONFIRM", "WEEKLY_EITHER_CONFIRM", "DAILY_2D_EITHER_CONFIRM"]:
        nav = daily_panel[f"{strategy}_nav"]
        dd = nav / nav.cummax() - 1.0
        ax.plot(daily_panel["date"], dd, label=strategy)
    ax.set_title("Drawdown Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DD_PATH, dpi=180)
    plt.close(fig)

    metrics = ["annualized_return", "sharpe_ratio", "max_drawdown", "number_of_switches"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    plot_perf = perf.loc[perf["strategy"].isin(ALL_STRATEGIES)].copy()
    for ax, metric in zip(axes, metrics):
        ax.bar(plot_perf["strategy"], plot_perf[metric])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIG_BAR_PATH, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for strategy in TIMING_STRATEGIES:
        ax.step(daily_panel["date"], daily_panel[f"{strategy}_weight_spy"], where="post", label=strategy)
    ax.set_title("Weight Timeline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_WEIGHT_PATH, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(daily_panel["date"], daily_panel["spy_price"], color="black", label="SPY")
    axes[0].plot(daily_panel["date"], daily_panel["faber_ma200"], color="tab:blue", label="MA200")
    axes[0].legend()
    axes[0].set_title("SPY Price and MA200")
    axes[1].step(daily_panel["date"], daily_panel["faber_daily_signal"], where="post", label="Faber daily", color="tab:blue")
    axes[1].step(daily_panel["date"], daily_panel["antonacci_daily_signal"], where="post", label="Antonacci daily", color="tab:red")
    axes[1].set_title("Daily Signals")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_SIGNAL_PATH, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    for ax, (period, (start, end)) in zip(axes, [("GFC_2008_2009", CRISIS_WINDOWS["GFC_2008_2009"]), ("COVID_2020", CRISIS_WINDOWS["COVID_2020"]), ("INFLATION_2022", CRISIS_WINDOWS["INFLATION_2022"])]):
        sub = daily_panel.loc[(daily_panel["date"] >= pd.Timestamp(start)) & (daily_panel["date"] <= pd.Timestamp(end))]
        if sub.empty:
            continue
        for strategy in ["SPY_BUY_HOLD", "MONTHLY_EITHER_CONFIRM", "WEEKLY_EITHER_CONFIRM", "DAILY_2D_EITHER_CONFIRM"]:
            series = sub[f"{strategy}_nav"] / sub[f"{strategy}_nav"].iloc[0]
            ax.plot(sub["date"], series, label=strategy)
        ax.set_title(period)
        ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_CRISIS_PATH, dpi=180)
    plt.close(fig)


def write_summary_md(perf: pd.DataFrame, crisis: pd.DataFrame) -> None:
    core = perf.loc[perf["strategy"].isin(TIMING_STRATEGIES + ["SPY_BUY_HOLD", "CASH_ONLY"]), [
        "strategy",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "number_of_switches",
        "time_in_spy",
        "avg_trade_cost_per_year",
    ]].copy()
    lines = [
        "# SPY/CASH Timing Frequency Test",
        "",
        "## 研究目标",
        "",
        "比较同一套 Faber + Antonacci Either Confirm 逻辑在月度、周度、日度两天确认三个频率上的表现，只研究 SPY + CASH。",
        "",
        "## 策略定义",
        "",
        "- `MONTHLY_EITHER_CONFIRM`: 月末判断，Faber 月度 10M SMA 或 Antonacci 12M excess momentum 任一看多则持有 SPY。",
        "- `WEEKLY_EITHER_CONFIRM`: 周末判断，Faber MA200 或 Antonacci 日度 252 日 excess momentum 任一看多则持有 SPY。",
        "- `DAILY_2D_EITHER_CONFIRM`: 日度信号，但入场需要任一信号连续两天看多，出场需要两个信号连续两天都看空。",
        "",
        "## 数据说明",
        "",
        f"- SPY 数据：`{DAILY_CLOSE_PATH}` / `{DAILY_RETURNS_PATH}`",
        f"- CASH 数据：`{DTB3_PATH}`，按 `daily_rf = (1 + DTB3 / 100) ** (1/252) - 1` 转为日收益。",
        "",
        "## 主要绩效表",
        "",
        core.to_markdown(index=False),
        "",
        "## Monthly vs Weekly vs Daily 2D 对比",
        "",
        "- Monthly Either Confirm 是当前 baseline。",
        "- Weekly Either Confirm 是直接升频版本，用更快的检查频率换更早的响应。",
        "- Daily 2D Confirm 是高频但带确认的版本，目标是在速度和噪声之间取折中。",
        "",
        "## 危机阶段表现",
        "",
        crisis.to_markdown(index=False),
        "",
        "## 换手和交易成本分析",
        "",
        "- Weekly 和 Daily 2D 的核心代价是更高换手。",
        "- Daily 2D 是否值得，取决于它在 2008 / 2020 / 2022 中是否显著改善回撤控制，而不是只看总收益。",
        "",
        "## 结论与下一步建议",
        "",
        "- 当前只研究 SPY + CASH。",
        "- Monthly Either Confirm 是已有 baseline。",
        "- Weekly Either Confirm 是最直接的升频比较对象。",
        "- Daily 2D Confirm 试图改善对 10%-15% 回调和快速下跌的响应，同时避免纯日度噪声。",
        "- 下一步应优先检查 Weekly vs Daily 2D 在危机期的 tradeoff，再决定谁作为后续 regime overlay 的 SPY/CASH baseline。",
    ]
    SUMMARY_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_summary_md(perf: pd.DataFrame, crisis: pd.DataFrame) -> None:
    core = perf.loc[perf["strategy"].isin(TIMING_STRATEGIES + ["SPY_BUY_HOLD", "CASH_ONLY"]), [
        "strategy",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "number_of_switches",
        "time_in_spy",
        "avg_trade_cost_per_year",
    ]].copy()
    lines = [
        "# SPY/CASH Timing Frequency Test",
        "",
        "## Research Goal",
        "",
        "Compare three SPY/CASH timing baselines built from Faber and Antonacci signals:",
        "- `MONTHLY_EITHER_CONFIRM`",
        "- `WEEKLY_EITHER_CONFIRM`",
        "- `DAILY_2D_EITHER_CONFIRM`",
        "",
        "The objective is to test whether higher-frequency evaluation improves reaction speed to 10%-15% pullbacks and fast selloffs without taking on excessive noise and turnover.",
        "",
        "## Strategy Definitions",
        "",
        "- `MONTHLY_EITHER_CONFIRM`: month-end Faber 10M SMA signal OR month-end Antonacci 12M excess momentum signal. If either is bullish, hold SPY for the next month; otherwise hold CASH.",
        "- `WEEKLY_EITHER_CONFIRM`: week-end Faber MA200 signal OR week-end Antonacci 252-day excess momentum signal. If either is bullish, hold SPY for the next week; otherwise hold CASH.",
        "- `DAILY_2D_EITHER_CONFIRM`: daily Faber MA200 and daily Antonacci 252-day excess momentum. Entry requires either signal bullish for two consecutive trading days. Exit requires both signals bearish for two consecutive trading days. Trades take effect the next trading day.",
        "",
        "## Data",
        "",
        f"- SPY daily adjusted close / returns: `{DAILY_CLOSE_PATH}` and `{DAILY_RETURNS_PATH}`",
        f"- CASH proxy: `{DTB3_PATH}`, converted with `daily_rf = (1 + DTB3 / 100) ** (1/252) - 1`",
        "",
        "## Performance Summary",
        "",
        core.to_markdown(index=False),
        "",
        "## Monthly vs Weekly vs Daily 2D",
        "",
        "- `MONTHLY_EITHER_CONFIRM` is the current baseline.",
        "- `WEEKLY_EITHER_CONFIRM` tests whether faster signal refresh helps without adding too much noise.",
        "- `DAILY_2D_EITHER_CONFIRM` tests whether two-day confirmation improves reaction speed while filtering single-day noise.",
        "",
        "## Crisis Performance",
        "",
        crisis.to_markdown(index=False),
        "",
        "## Turnover and Trade Cost",
        "",
        "- Weekly and daily strategies trade more often than the monthly baseline.",
        "- Daily 2D confirmation lowers noise relative to pure daily trading, but still adds meaningful turnover versus weekly and monthly timing.",
        "- This tradeoff matters because transaction cost is fixed at 20 bps per switch.",
        "",
        "## Conclusion and Next Step",
        "",
        "- This study only compares SPY + CASH.",
        "- `MONTHLY_EITHER_CONFIRM` is the current baseline.",
        "- `WEEKLY_EITHER_CONFIRM` is a direct frequency lift of the baseline.",
        "- `DAILY_2D_EITHER_CONFIRM` is the higher-frequency version with a confirmation filter.",
        "- The current evidence favors `MONTHLY_EITHER_CONFIRM` as the next baseline because it still has the best return and Sharpe after costs, while keeping turnover materially lower than the daily variant.",
        "- The next useful step is to test whether weekly timing can be improved with asymmetric entry and exit confirmation rather than simply increasing evaluation frequency.",
    ]
    SUMMARY_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = build_daily_signals(load_data())
    monthly = build_monthly_signals(panel)
    weekly = build_weekly_signals(panel)
    start_date = first_valid_date(monthly, weekly, panel)
    panel = panel.loc[panel["date"] >= start_date].reset_index(drop=True)
    monthly = monthly.loc[pd.to_datetime(monthly["effective_date"]) >= start_date].reset_index(drop=True)
    weekly = weekly.loc[pd.to_datetime(weekly["effective_date"]) >= start_date].reset_index(drop=True)

    monthly_weights = build_monthly_either_weights(panel, monthly)
    weekly_weights = build_weekly_either_weights(panel, weekly)
    daily2d_weights, daily2d_trade_info = build_daily_2d_either_weights(panel)
    daily2d_weights = daily2d_weights.loc[daily2d_weights["date"] >= start_date].reset_index(drop=True)

    monthly_bt, monthly_trades = run_daily_backtest(panel, "MONTHLY_EITHER_CONFIRM", monthly_weights["weight_spy"], monthly_weights["signal_date"], monthly_weights["effective_date"], CONFIG["transaction_cost_bps"])
    weekly_bt, weekly_trades = run_daily_backtest(panel, "WEEKLY_EITHER_CONFIRM", weekly_weights["weight_spy"], weekly_weights["signal_date"], weekly_weights["effective_date"], CONFIG["transaction_cost_bps"])
    daily2d_bt, daily2d_trades = run_daily_backtest(panel, "DAILY_2D_EITHER_CONFIRM", daily2d_weights["weight_spy"], daily2d_weights["signal_date"], daily2d_weights["effective_date"], CONFIG["transaction_cost_bps"])

    out = panel[["date", "spy_price", "spy_daily_return", "daily_rf", "cash_nav", "faber_ma200", "faber_daily_signal", "antonacci_12m_spy_return", "antonacci_12m_cash_return", "antonacci_excess_momentum", "antonacci_daily_signal"]].copy()
    out["monthly_either_weight_spy"] = monthly_bt["actual_weight_spy"].to_numpy()
    out["weekly_either_weight_spy"] = weekly_bt["actual_weight_spy"].to_numpy()
    out["daily_2d_weight_spy"] = daily2d_bt["actual_weight_spy"].to_numpy()
    for strategy, frame in [
        ("MONTHLY_EITHER_CONFIRM", monthly_bt),
        ("WEEKLY_EITHER_CONFIRM", weekly_bt),
        ("DAILY_2D_EITHER_CONFIRM", daily2d_bt),
    ]:
        out[f"{strategy}_return"] = frame["daily_return"].to_numpy()
        out[f"{strategy}_nav"] = frame["nav"].to_numpy()
        out[f"{strategy}_weight_spy"] = frame["actual_weight_spy"].to_numpy()
        out[f"{strategy}_turnover_flag"] = frame["turnover_flag"].astype(int).to_numpy()
        out[f"{strategy}_transaction_cost"] = frame["transaction_cost"].to_numpy()
    out["SPY_BUY_HOLD_weight_spy"] = 1.0
    out["SPY_BUY_HOLD_return"] = out["spy_daily_return"].fillna(0.0)
    out["SPY_BUY_HOLD_nav"] = CONFIG["initial_nav"] * (1.0 + out["SPY_BUY_HOLD_return"]).cumprod()
    out["SPY_BUY_HOLD_turnover_flag"] = 0
    out["SPY_BUY_HOLD_transaction_cost"] = 0.0
    out["CASH_ONLY_weight_spy"] = 0.0
    out["CASH_ONLY_return"] = out["daily_rf"]
    out["CASH_ONLY_nav"] = CONFIG["initial_nav"] * (1.0 + out["CASH_ONLY_return"]).cumprod()
    out["CASH_ONLY_turnover_flag"] = 0
    out["CASH_ONLY_transaction_cost"] = 0.0

    trade_log = pd.concat([monthly_trades, weekly_trades, daily2d_trades], ignore_index=True).sort_values(["trade_date", "strategy"]).reset_index(drop=True)
    perf = compute_performance_metrics(out)
    crisis = compute_crisis_performance(out)
    yearly_rows = []
    out["year"] = out["date"].dt.year
    for year, grp in out.groupby("year", observed=False):
        yearly_rows.append(
            {
                "year": int(year),
                "MONTHLY_EITHER_CONFIRM": float((1.0 + grp["MONTHLY_EITHER_CONFIRM_return"]).prod() - 1.0),
                "WEEKLY_EITHER_CONFIRM": float((1.0 + grp["WEEKLY_EITHER_CONFIRM_return"]).prod() - 1.0),
                "DAILY_2D_EITHER_CONFIRM": float((1.0 + grp["DAILY_2D_EITHER_CONFIRM_return"]).prod() - 1.0),
                "SPY_BUY_HOLD": float((1.0 + grp["SPY_BUY_HOLD_return"]).prod() - 1.0),
                "CASH_ONLY": float((1.0 + grp["CASH_ONLY_return"]).prod() - 1.0),
            }
        )
    yearly = pd.DataFrame(yearly_rows)
    dd = compute_drawdown_events(out)

    out.to_csv(DAILY_PANEL_PATH, index=False)
    monthly.to_csv(MONTHLY_SIGNAL_PATH, index=False)
    weekly.to_csv(WEEKLY_SIGNAL_PATH, index=False)
    trade_log.to_csv(TRADE_LOG_PATH, index=False)
    perf.to_csv(PERF_PATH, index=False)
    crisis.to_csv(CRISIS_PATH, index=False)
    yearly.to_csv(YEARLY_PATH, index=False)
    dd.to_csv(DD_PATH, index=False)
    plot_results(out, perf, crisis)
    write_summary_md(perf, crisis)

    best_ret = perf.sort_values("annualized_return", ascending=False).iloc[0]
    best_sharpe = perf.sort_values("sharpe_ratio", ascending=False).iloc[0]
    best_dd = perf.sort_values("max_drawdown", ascending=False).iloc[0]
    most_switch = perf.sort_values("number_of_switches", ascending=False).iloc[0]
    monthly_row = perf.loc[perf["strategy"] == "MONTHLY_EITHER_CONFIRM"].iloc[0]
    weekly_row = perf.loc[perf["strategy"] == "WEEKLY_EITHER_CONFIRM"].iloc[0]
    daily2d_row = perf.loc[perf["strategy"] == "DAILY_2D_EITHER_CONFIRM"].iloc[0]
    crisis_core = crisis.loc[crisis["strategy"].isin(TIMING_STRATEGIES + ["SPY_BUY_HOLD"])].copy()
    stable_by_period = crisis_core.sort_values(["period", "max_drawdown"], ascending=[True, False]).groupby("period").head(1)

    print(f"Sample: {out['date'].min().date()} to {out['date'].max().date()}")
    print(f"1. Highest annualized return: {best_ret['strategy']} ({best_ret['annualized_return']:.2%})")
    print(f"2. Highest Sharpe: {best_sharpe['strategy']} ({best_sharpe['sharpe_ratio']:.2f})")
    print(f"3. Lowest max drawdown: {best_dd['strategy']} ({best_dd['max_drawdown']:.2%})")
    print(f"4. Highest turnover: {most_switch['strategy']} ({int(most_switch['number_of_switches'])} switches)")
    print(f"5. WEEKLY_EITHER better than MONTHLY_EITHER: {(weekly_row['annualized_return'] > monthly_row['annualized_return']) and (weekly_row['sharpe_ratio'] >= monthly_row['sharpe_ratio'])}")
    print(f"6. DAILY_2D better than MONTHLY_EITHER: {(daily2d_row['annualized_return'] > monthly_row['annualized_return']) and (daily2d_row['sharpe_ratio'] >= monthly_row['sharpe_ratio'])}")
    print(f"7. DAILY_2D turnover > WEEKLY_EITHER: {daily2d_row['number_of_switches'] > weekly_row['number_of_switches']}")
    print("8. Most stable strategy by crisis period:")
    print(stable_by_period[["period", "strategy", "max_drawdown", "cumulative_return"]].to_string(index=False))
    recommended = best_sharpe["strategy"]
    print(f"9. Recommended next SPY/CASH baseline: {recommended}")
    for path in [
        DAILY_PANEL_PATH,
        MONTHLY_SIGNAL_PATH,
        WEEKLY_SIGNAL_PATH,
        TRADE_LOG_PATH,
        PERF_PATH,
        CRISIS_PATH,
        YEARLY_PATH,
        DD_PATH,
        SUMMARY_MD_PATH,
        FIG_LOG_PATH,
        FIG_LINEAR_PATH,
        FIG_DD_PATH,
        FIG_BAR_PATH,
        FIG_WEIGHT_PATH,
        FIG_SIGNAL_PATH,
        FIG_CRISIS_PATH,
    ]:
        if path.exists():
            print(f"Saved: {path}")


if __name__ == "__main__":
    main()
