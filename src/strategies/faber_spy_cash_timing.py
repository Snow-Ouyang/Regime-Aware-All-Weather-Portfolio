from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "faber_spy_cash_timing"

DAILY_CLOSE_PATH = ROOT / "data" / "processed" / "assets" / "daily_adjusted_close.csv"
DAILY_RETURNS_PATH = ROOT / "data" / "processed" / "assets" / "daily_returns.csv"
DTB3_PATH = ROOT / "data" / "raw" / "macro" / "rate" / "DTB3.csv"
ANTONACCI_DAILY_PATH = ROOT / "results" / "absolute_momentum_spy_cash" / "daily_backtest_panel.csv"
ANTONACCI_MONTHLY_PATH = ROOT / "results" / "absolute_momentum_spy_cash" / "monthly_signal_panel.csv"

DAILY_PANEL_PATH = RESULTS_DIR / "daily_backtest_panel.csv"
MONTHLY_SIGNAL_PATH = RESULTS_DIR / "monthly_signal_panel.csv"
PERF_PATH = RESULTS_DIR / "performance_summary.csv"
YEARLY_PATH = RESULTS_DIR / "yearly_returns.csv"
DD_PATH = RESULTS_DIR / "drawdown_summary.csv"
DISAGREE_PATH = RESULTS_DIR / "signal_disagreement_summary.csv"

FIG_LOG_PATH = RESULTS_DIR / "equity_curve_log.png"
FIG_LINEAR_PATH = RESULTS_DIR / "equity_curve_linear.png"
FIG_DD_PATH = RESULTS_DIR / "drawdown_comparison.png"
FIG_SIGNAL_OVERLAY_PATH = RESULTS_DIR / "signal_overlay_faber_vs_antonacci.png"
FIG_FABER_SMA_PATH = RESULTS_DIR / "faber_price_vs_sma.png"
FIG_SIGNAL_COMPARE_PATH = RESULTS_DIR / "antonacci_excess_momentum_vs_faber_sma.png"
FIG_ROLL_SHARPE_PATH = RESULTS_DIR / "rolling_3y_sharpe.png"
FIG_ROLL_DD_PATH = RESULTS_DIR / "rolling_3y_max_drawdown.png"
FIG_SWITCHES_PATH = RESULTS_DIR / "strategy_switches_timeline.png"

DEFAULT_TICKER = "SPY"
TRADING_DAYS = 252
MONTHLY_SMA_WINDOW = 10
DAILY_SMA_WINDOW = 200
LOOKBACK_MONTHS = 12
TCOST_BPS = 20.0

MAIN_PLOT_STRATEGIES = [
    "FABER_10M_SMA_MONTHLY",
    "ANTONACCI_12M_ABS_MOM",
    "SPY_BUY_HOLD",
    "CASH_ONLY",
    "FABER_RECOVERY_OVERLAY",
    "BOTH_CONFIRM",
    "EITHER_CONFIRM",
]
DRAW_PLOT_STRATEGIES = [
    "FABER_10M_SMA_MONTHLY",
    "ANTONACCI_12M_ABS_MOM",
    "FABER_RECOVERY_OVERLAY",
    "SPY_BUY_HOLD",
]
ROLL_PLOT_STRATEGIES = [
    "FABER_10M_SMA_MONTHLY",
    "ANTONACCI_12M_ABS_MOM",
    "FABER_RECOVERY_OVERLAY",
    "SPY_BUY_HOLD",
]
OUTPUT_STRATEGIES = [
    "FABER_10M_SMA_MONTHLY",
    "FABER_10M_SMA_MONTHLY_NO_COST",
    "FABER_200D_SMA_DAILY_SIGNAL_MONTHLY_TRADE",
    "ANTONACCI_12M_ABS_MOM",
    "ANTONACCI_12M_ABS_MOM_NO_COST",
    "BOTH_CONFIRM",
    "EITHER_CONFIRM",
    "FABER_ENTRY_ANTONACCI_EXIT",
    "ANTONACCI_ENTRY_FABER_EXIT",
    "FABER_RECOVERY_OVERLAY",
    "FABER_FULL_RECOVERY_OVERLAY",
]


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
    out = out.dropna(subset=["spy_price"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
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
    panel["ma200"] = panel["spy_price"].rolling(DAILY_SMA_WINDOW, min_periods=DAILY_SMA_WINDOW).mean()
    return panel.reset_index(drop=True)


def compute_antonacci_monthly_signal_from_base(panel: pd.DataFrame) -> pd.DataFrame:
    monthly = panel.groupby("month", as_index=False).tail(1).copy()
    monthly["antonacci_12m_spy_return"] = monthly["spy_price"].pct_change(LOOKBACK_MONTHS)
    monthly["antonacci_12m_cash_return"] = monthly["cash_nav"].pct_change(LOOKBACK_MONTHS)
    monthly["antonacci_excess_momentum"] = monthly["antonacci_12m_spy_return"] - monthly["antonacci_12m_cash_return"]
    monthly["antonacci_signal_num"] = np.where(monthly["antonacci_excess_momentum"] > 0, 1.0, 0.0)
    monthly["signal_date"] = monthly["date"]
    monthly["effective_date"] = monthly["date"].shift(-1)
    return monthly[
        [
            "month",
            "signal_date",
            "effective_date",
            "spy_price",
            "antonacci_12m_spy_return",
            "antonacci_12m_cash_return",
            "antonacci_excess_momentum",
            "antonacci_signal_num",
        ]
    ].reset_index(drop=True)


def load_or_rebuild_antonacci_monthly(panel: pd.DataFrame) -> pd.DataFrame:
    if ANTONACCI_MONTHLY_PATH.exists():
        monthly = pd.read_csv(ANTONACCI_MONTHLY_PATH)
        rename = {
            "date": "signal_date",
            "rf_12m_return": "antonacci_12m_cash_return",
            "spy_12m_return": "antonacci_12m_spy_return",
            "excess_momentum": "antonacci_excess_momentum",
        }
        monthly = monthly.rename(columns={k: v for k, v in rename.items() if k in monthly.columns})
        monthly["signal_date"] = pd.to_datetime(monthly["signal_date"])
        if "effective_date" in monthly.columns:
            monthly["effective_date"] = pd.to_datetime(monthly["effective_date"])
        if "antonacci_signal_num" not in monthly.columns:
            if "signal" in monthly.columns:
                monthly["antonacci_signal_num"] = np.where(monthly["signal"].eq("SPY"), 1.0, 0.0)
            else:
                monthly["antonacci_signal_num"] = np.where(monthly["antonacci_excess_momentum"] > 0, 1.0, 0.0)
        monthly["month"] = monthly["signal_date"].dt.to_period("M")
        needed = {
            "month",
            "signal_date",
            "effective_date",
            "antonacci_12m_spy_return",
            "antonacci_12m_cash_return",
            "antonacci_excess_momentum",
            "antonacci_signal_num",
        }
        if needed.issubset(set(monthly.columns)):
            return monthly[list(needed | {"spy_price"} & set(monthly.columns))].copy()
    return compute_antonacci_monthly_signal_from_base(panel)


def compute_faber_monthly_signal(panel: pd.DataFrame) -> pd.DataFrame:
    monthly = panel.groupby("month", as_index=False).tail(1).copy()
    monthly["faber_10m_sma"] = monthly["spy_price"].rolling(MONTHLY_SMA_WINDOW, min_periods=MONTHLY_SMA_WINDOW).mean()
    monthly["faber_price_to_sma_ratio"] = monthly["spy_price"] / monthly["faber_10m_sma"]
    signal = []
    prev = 0.0
    for _, row in monthly.iterrows():
        price = row["spy_price"]
        sma = row["faber_10m_sma"]
        if pd.isna(sma):
            cur = np.nan
        elif price > sma:
            cur = 1.0
        elif price < sma:
            cur = 0.0
        else:
            cur = prev
        if pd.notna(cur):
            prev = cur
        signal.append(cur)
    monthly["faber_signal_num"] = signal
    monthly["faber_200d_sma"] = monthly["ma200"]
    signal_200d = []
    prev_200d = 0.0
    for _, row in monthly.iterrows():
        price = row["spy_price"]
        sma = row["faber_200d_sma"]
        if pd.isna(sma):
            cur = np.nan
        elif price > sma:
            cur = 1.0
        elif price < sma:
            cur = 0.0
        else:
            cur = prev_200d
        if pd.notna(cur):
            prev_200d = cur
        signal_200d.append(cur)
    monthly["faber_200d_signal_num"] = signal_200d
    monthly["signal_date"] = monthly["date"]
    monthly["effective_date"] = monthly["date"].shift(-1)
    return monthly[
        [
            "month",
            "signal_date",
            "effective_date",
            "spy_price",
            "faber_10m_sma",
            "faber_price_to_sma_ratio",
            "faber_signal_num",
            "faber_200d_sma",
            "faber_200d_signal_num",
        ]
    ].reset_index(drop=True)


def merge_monthly_signals(base: pd.DataFrame) -> pd.DataFrame:
    faber = compute_faber_monthly_signal(base)
    antonacci = load_or_rebuild_antonacci_monthly(base)
    cols = [
        "month",
        "signal_date",
        "effective_date",
        "spy_price",
        "antonacci_12m_spy_return",
        "antonacci_12m_cash_return",
        "antonacci_excess_momentum",
        "antonacci_signal_num",
    ]
    antonacci = antonacci[cols].copy()
    monthly = faber.merge(antonacci, on=["month", "signal_date", "effective_date", "spy_price"], how="outer")
    monthly = monthly.sort_values("signal_date").reset_index(drop=True)
    monthly["faber_signal_num"] = monthly["faber_signal_num"].astype(float)
    monthly["faber_200d_signal_num"] = monthly["faber_200d_signal_num"].astype(float)
    monthly["antonacci_signal_num"] = monthly["antonacci_signal_num"].astype(float)
    return monthly


def stateful_combined_signal(monthly: pd.DataFrame, entry_signal: str, exit_signal: str) -> pd.Series:
    state = 0.0
    out = []
    for _, row in monthly.iterrows():
        entry = row[entry_signal]
        exitv = row[exit_signal]
        if pd.isna(entry) or pd.isna(exitv):
            out.append(np.nan)
            continue
        if state == 0.0:
            if entry == 1.0:
                state = 1.0
        else:
            if exitv == 0.0:
                state = 0.0
        out.append(state)
    return pd.Series(out, index=monthly.index, dtype=float)


def add_combined_signals(monthly: pd.DataFrame) -> pd.DataFrame:
    out = monthly.copy()
    f = out["faber_signal_num"]
    a = out["antonacci_signal_num"]
    out["both_confirm_signal_num"] = np.where((f == 1.0) & (a == 1.0), 1.0, np.where((f == 0.0) | (a == 0.0), 0.0, np.nan))
    out["either_confirm_signal_num"] = np.where((f == 1.0) | (a == 1.0), 1.0, np.where((f == 0.0) & (a == 0.0), 0.0, np.nan))
    out["faber_entry_antonacci_exit_signal_num"] = stateful_combined_signal(out, "faber_signal_num", "antonacci_signal_num")
    out["antonacci_entry_faber_exit_signal_num"] = stateful_combined_signal(out, "antonacci_signal_num", "faber_signal_num")
    out["faber_recovery_overlay_spy_weight"] = np.where(
        a == 1.0,
        1.0,
        np.where((a == 0.0) & (f == 1.0), 0.5, np.where((a == 0.0) & (f == 0.0), 0.0, np.nan)),
    )
    out["faber_full_recovery_overlay_spy_weight"] = np.where(
        a == 1.0,
        1.0,
        np.where((a == 0.0) & (f == 1.0), 1.0, np.where((a == 0.0) & (f == 0.0), 0.0, np.nan)),
    )
    numeric_cols = [
        "faber_signal_num",
        "faber_200d_signal_num",
        "antonacci_signal_num",
        "both_confirm_signal_num",
        "either_confirm_signal_num",
        "faber_entry_antonacci_exit_signal_num",
        "antonacci_entry_faber_exit_signal_num",
        "faber_recovery_overlay_spy_weight",
        "faber_full_recovery_overlay_spy_weight",
    ]
    out[numeric_cols] = out[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return out


def strategy_specs() -> dict[str, dict[str, object]]:
    return {
        "FABER_10M_SMA_MONTHLY": {"col": "faber_signal_num", "cost_bps": TCOST_BPS},
        "FABER_10M_SMA_MONTHLY_NO_COST": {"col": "faber_signal_num", "cost_bps": 0.0},
        "FABER_200D_SMA_DAILY_SIGNAL_MONTHLY_TRADE": {"col": "faber_200d_signal_num", "cost_bps": TCOST_BPS},
        "ANTONACCI_12M_ABS_MOM": {"col": "antonacci_signal_num", "cost_bps": TCOST_BPS},
        "ANTONACCI_12M_ABS_MOM_NO_COST": {"col": "antonacci_signal_num", "cost_bps": 0.0},
        "BOTH_CONFIRM": {"col": "both_confirm_signal_num", "cost_bps": TCOST_BPS},
        "EITHER_CONFIRM": {"col": "either_confirm_signal_num", "cost_bps": TCOST_BPS},
        "FABER_ENTRY_ANTONACCI_EXIT": {"col": "faber_entry_antonacci_exit_signal_num", "cost_bps": TCOST_BPS},
        "ANTONACCI_ENTRY_FABER_EXIT": {"col": "antonacci_entry_faber_exit_signal_num", "cost_bps": TCOST_BPS},
        "FABER_RECOVERY_OVERLAY": {"col": "faber_recovery_overlay_spy_weight", "cost_bps": TCOST_BPS},
        "FABER_FULL_RECOVERY_OVERLAY": {"col": "faber_full_recovery_overlay_spy_weight", "cost_bps": TCOST_BPS},
    }


def expand_monthly_signal_to_daily(base: pd.DataFrame, monthly: pd.DataFrame, signal_col: str, transaction_cost_bps: float) -> pd.DataFrame:
    out = base[["date", "spy_price", "spy_daily_return", "daily_rf", "cash_nav"]].copy()
    out["target_weight_spy"] = pd.Series(np.nan, index=out.index, dtype=float)
    out["target_weight_cash"] = pd.Series(np.nan, index=out.index, dtype=float)
    out["turnover_flag"] = False
    out["transaction_cost"] = 0.0
    mapping = monthly.loc[monthly["effective_date"].notna(), ["signal_date", "effective_date", signal_col]].copy()
    mapping["signal_date"] = pd.to_datetime(mapping["signal_date"])
    mapping["effective_date"] = pd.to_datetime(mapping["effective_date"])
    mapping = mapping.rename(columns={signal_col: "signal_weight_spy"})
    mapping["signal_weight_spy"] = pd.to_numeric(mapping["signal_weight_spy"], errors="coerce")
    mapping = mapping.dropna(subset=["signal_weight_spy"])
    for _, row in mapping.iterrows():
        mask = out["date"] >= row["effective_date"]
        out.loc[mask, "target_weight_spy"] = row["signal_weight_spy"]
    out["target_weight_spy"] = out["target_weight_spy"].fillna(0.0)
    out["target_weight_cash"] = 1.0 - out["target_weight_spy"]
    out["turnover_flag"] = (out["target_weight_spy"] != out["target_weight_spy"].shift(1)).fillna(False)
    out.loc[out.index[0], "turnover_flag"] = False
    out["transaction_cost"] = np.where(out["turnover_flag"], transaction_cost_bps / 10000.0, 0.0)
    out["strategy_return"] = out["target_weight_spy"] * out["spy_daily_return"].fillna(0.0) + out["target_weight_cash"] * out["daily_rf"] - out["transaction_cost"]
    out["strategy_nav"] = (1.0 + out["strategy_return"]).cumprod()
    return out


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


def perf_row(strategy: str, returns: pd.Series, rf_daily: pd.Series, nav: pd.Series, weights: pd.Series | None = None, switches: int | float = np.nan) -> dict[str, object]:
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
    mdd = max_drawdown_from_returns(s)
    calmar = float(ann_ret / abs(mdd)) if pd.notna(mdd) and mdd < 0 else np.nan
    monthly = (1.0 + s).groupby(s.index.to_period("M")).prod() - 1.0
    time_in_spy = float(weights.mean()) if weights is not None else np.nan
    return {
        "strategy": strategy,
        "start_date": s.index.min().date().isoformat(),
        "end_date": s.index.max().date().isoformat(),
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
        "calmar_ratio": calmar,
        "final_nav": float(nav.loc[s.index[-1]]),
        "positive_day_ratio": float((s > 0).mean()),
        "positive_month_ratio": float((monthly > 0).mean()) if not monthly.empty else np.nan,
        "number_of_switches": switches,
        "avg_switches_per_year": float(switches / (len(s) / TRADING_DAYS)) if pd.notna(switches) and len(s) > 0 else np.nan,
        "avg_turnover_per_year": float(switches / (len(s) / TRADING_DAYS)) if pd.notna(switches) and len(s) > 0 else np.nan,
        "time_in_spy": time_in_spy,
        "time_in_cash": 1.0 - time_in_spy if pd.notna(time_in_spy) else np.nan,
    }


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
            seg = dd.iloc[start : end + 1]
            trough = seg.idxmin()
            peak_date = wealth.index[start - 1] if start > 0 else wealth.index[start]
            recovery_date = wealth.index[i]
            episodes.append(
                {
                    "strategy": strategy,
                    "drawdown_start": peak_date,
                    "drawdown_trough": trough,
                    "drawdown_recovery": recovery_date,
                    "max_drawdown": float(seg.min()),
                    "duration_days": int((recovery_date - peak_date).days),
                    "recovery_days": int((recovery_date - trough).days),
                }
            )
            start = None
    if start is not None:
        seg = dd.iloc[start:]
        trough = seg.idxmin()
        peak_date = wealth.index[start - 1] if start > 0 else wealth.index[start]
        episodes.append(
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
    out = pd.DataFrame(episodes)
    return out.sort_values("max_drawdown").head(top_n).reset_index(drop=True) if not out.empty else out


def forward_return(series: pd.Series, horizon: int) -> pd.Series:
    return series.shift(-horizon) / series - 1.0


def forward_max_drawdown(series: pd.Series, horizon: int) -> pd.Series:
    vals = []
    for i in range(len(series)):
        end = min(i + horizon, len(series) - 1)
        window = series.iloc[i : end + 1]
        if len(window) < 2:
            vals.append(np.nan)
            continue
        rel = window / window.iloc[0]
        vals.append(float((rel / rel.cummax() - 1.0).min()))
    return pd.Series(vals, index=series.index)


def build_signal_disagreement_summary(monthly: pd.DataFrame, daily_panel: pd.DataFrame) -> pd.DataFrame:
    daily = daily_panel[["date", "spy_price"]].copy()
    daily["fwd_21d_spy_return"] = forward_return(daily["spy_price"], 21)
    daily["fwd_63d_spy_return"] = forward_return(daily["spy_price"], 63)
    daily["fwd_126d_spy_return"] = forward_return(daily["spy_price"], 126)
    daily["fwd_21d_spy_max_drawdown"] = forward_max_drawdown(daily["spy_price"], 21)
    daily["fwd_63d_spy_max_drawdown"] = forward_max_drawdown(daily["spy_price"], 63)
    monthly_daily = monthly.merge(daily, left_on="effective_date", right_on="date", how="left")
    cond1 = monthly_daily["faber_signal_num"].eq(1.0) & monthly_daily["antonacci_signal_num"].eq(0.0)
    cond2 = monthly_daily["faber_signal_num"].eq(0.0) & monthly_daily["antonacci_signal_num"].eq(1.0)
    rows = []
    for name, mask in [
        ("FABER_SPY_ANTONACCI_CASH", cond1),
        ("FABER_CASH_ANTONACCI_SPY", cond2),
    ]:
        grp = monthly_daily.loc[mask].copy()
        rows.append(
            {
                "case": name,
                "count_days": int(len(daily_panel.loc[(daily_panel["faber_signal"] == 1) & (daily_panel["antonacci_signal"] == 0)]) if name == "FABER_SPY_ANTONACCI_CASH" else len(daily_panel.loc[(daily_panel["faber_signal"] == 0) & (daily_panel["antonacci_signal"] == 1)])),
                "count_months": int(len(grp)),
                "faber_spy_antonacci_cash_days": int(len(daily_panel.loc[(daily_panel["faber_signal"] == 1) & (daily_panel["antonacci_signal"] == 0)])),
                "faber_cash_antonacci_spy_days": int(len(daily_panel.loc[(daily_panel["faber_signal"] == 0) & (daily_panel["antonacci_signal"] == 1)])),
                "avg_next_21d_spy_return": float(grp["fwd_21d_spy_return"].mean()),
                "avg_next_63d_spy_return": float(grp["fwd_63d_spy_return"].mean()),
                "avg_next_126d_spy_return": float(grp["fwd_126d_spy_return"].mean()),
                "avg_next_21d_spy_max_drawdown": float(grp["fwd_21d_spy_max_drawdown"].mean()),
                "avg_next_63d_spy_max_drawdown": float(grp["fwd_63d_spy_max_drawdown"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_daily_panel_and_metrics(base: pd.DataFrame, monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid = monthly.loc[
        monthly["effective_date"].notna()
        & monthly["faber_signal_num"].notna()
        & monthly["antonacci_signal_num"].notna()
    ].copy()
    common_start = pd.to_datetime(valid["effective_date"]).min()
    base_live = base.loc[base["date"] >= common_start].copy().reset_index(drop=True)
    monthly_live = monthly.copy()

    out = base_live[["date", "spy_price", "spy_daily_return", "daily_rf", "cash_nav", "month"]].copy()
    out["faber_10m_sma"] = np.nan
    out["faber_200d_sma"] = np.nan
    out["faber_signal"] = np.nan
    out["faber_200d_signal"] = np.nan
    out["antonacci_12m_spy_return"] = np.nan
    out["antonacci_12m_cash_return"] = np.nan
    out["antonacci_excess_momentum"] = np.nan
    out["antonacci_signal"] = np.nan

    signal_cols = {
        "faber_signal_num": "faber_signal",
        "faber_200d_signal_num": "faber_200d_signal",
        "antonacci_signal_num": "antonacci_signal",
    }
    for _, row in monthly_live.loc[monthly_live["effective_date"].notna()].iterrows():
        eff = pd.to_datetime(row["effective_date"])
        mask = out["date"] >= eff
        out.loc[mask, "faber_10m_sma"] = row["faber_10m_sma"]
        out.loc[mask, "faber_200d_sma"] = row["faber_200d_sma"]
        out.loc[mask, "antonacci_12m_spy_return"] = row["antonacci_12m_spy_return"]
        out.loc[mask, "antonacci_12m_cash_return"] = row["antonacci_12m_cash_return"]
        out.loc[mask, "antonacci_excess_momentum"] = row["antonacci_excess_momentum"]
        for src, dst in signal_cols.items():
            out.loc[mask, dst] = row[src]
        for src, dst in [
            ("both_confirm_signal_num", "both_confirm_signal"),
            ("either_confirm_signal_num", "either_confirm_signal"),
            ("faber_entry_antonacci_exit_signal_num", "faber_entry_antonacci_exit_signal"),
            ("antonacci_entry_faber_exit_signal_num", "antonacci_entry_faber_exit_signal"),
            ("faber_recovery_overlay_spy_weight", "faber_recovery_overlay_spy_weight_signal"),
            ("faber_full_recovery_overlay_spy_weight", "faber_full_recovery_overlay_spy_weight_signal"),
        ]:
            if dst not in out.columns:
                out[dst] = np.nan
            out.loc[mask, dst] = row[src]

    strategy_frames = {}
    for strategy, spec in strategy_specs().items():
        frame = expand_monthly_signal_to_daily(base_live, monthly_live, signal_col=spec["col"], transaction_cost_bps=float(spec["cost_bps"]))
        strategy_frames[strategy] = frame
        prefix = strategy.lower()
        out[f"{prefix}_daily_return"] = frame["strategy_return"].to_numpy()
        out[f"{prefix}_nav"] = frame["strategy_nav"].to_numpy()
        out[f"{prefix}_target_weight_spy"] = frame["target_weight_spy"].to_numpy()
        out[f"{prefix}_target_weight_cash"] = frame["target_weight_cash"].to_numpy()
        out[f"{prefix}_turnover_flag"] = frame["turnover_flag"].astype(int).to_numpy()
        out[f"{prefix}_transaction_cost"] = frame["transaction_cost"].to_numpy()

    out["spy_buy_hold_return"] = out["spy_daily_return"].fillna(0.0)
    out["spy_buy_hold_nav"] = (1.0 + out["spy_buy_hold_return"]).cumprod()
    out["cash_only_return"] = out["daily_rf"]
    out["cash_only_nav"] = (1.0 + out["cash_only_return"]).cumprod()

    idx = pd.DatetimeIndex(out["date"])
    rf = pd.Series(out["daily_rf"].to_numpy(), index=idx)
    perf_rows = []
    for strategy in OUTPUT_STRATEGIES:
        prefix = strategy.lower()
        returns = pd.Series(out[f"{prefix}_daily_return"].to_numpy(), index=idx)
        nav = pd.Series(out[f"{prefix}_nav"].to_numpy(), index=idx)
        weights = pd.Series(out[f"{prefix}_target_weight_spy"].to_numpy(), index=idx)
        switches = int(out[f"{prefix}_turnover_flag"].sum())
        perf_rows.append(perf_row(strategy, returns, rf, nav, weights=weights, switches=switches))
    perf_rows.append(perf_row("SPY_BUY_HOLD", pd.Series(out["spy_buy_hold_return"].to_numpy(), index=idx), rf, pd.Series(out["spy_buy_hold_nav"].to_numpy(), index=idx), weights=pd.Series(1.0, index=idx), switches=0))
    perf_rows.append(perf_row("CASH_ONLY", pd.Series(out["cash_only_return"].to_numpy(), index=idx), rf, pd.Series(out["cash_only_nav"].to_numpy(), index=idx), weights=pd.Series(0.0, index=idx), switches=0))
    perf = pd.DataFrame(perf_rows)

    yearly_rows = []
    out["year"] = out["date"].dt.year
    for year, grp in out.groupby("year", observed=False):
        row = {"year": int(year)}
        for strategy in OUTPUT_STRATEGIES:
            prefix = strategy.lower()
            row[strategy] = float((1.0 + grp[f"{prefix}_daily_return"]).prod() - 1.0)
        row["SPY_BUY_HOLD"] = float((1.0 + grp["spy_buy_hold_return"]).prod() - 1.0)
        row["CASH_ONLY"] = float((1.0 + grp["cash_only_return"]).prod() - 1.0)
        yearly_rows.append(row)
    yearly = pd.DataFrame(yearly_rows)

    dd_frames = []
    for strategy in OUTPUT_STRATEGIES:
        prefix = strategy.lower()
        dd_frames.append(drawdown_episodes(pd.Series(out[f"{prefix}_nav"].to_numpy(), index=idx), strategy))
    dd_frames.append(drawdown_episodes(pd.Series(out["spy_buy_hold_nav"].to_numpy(), index=idx), "SPY_BUY_HOLD"))
    dd_frames.append(drawdown_episodes(pd.Series(out["cash_only_nav"].to_numpy(), index=idx), "CASH_ONLY"))
    dd = pd.concat(dd_frames, ignore_index=True)

    return out.reset_index(drop=True), perf, yearly, dd


def plot_outputs(daily: pd.DataFrame, perf: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for strategy in MAIN_PLOT_STRATEGIES:
        prefix = strategy.lower()
        nav_col = "spy_buy_hold_nav" if strategy == "SPY_BUY_HOLD" else "cash_only_nav" if strategy == "CASH_ONLY" else f"{prefix}_nav"
        ax.plot(daily["date"], daily[nav_col], label=strategy)
    ax.set_yscale("log")
    ax.set_title("SPY/CASH Timing Strategies - Log NAV")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_LOG_PATH, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for strategy in MAIN_PLOT_STRATEGIES:
        prefix = strategy.lower()
        nav_col = "spy_buy_hold_nav" if strategy == "SPY_BUY_HOLD" else "cash_only_nav" if strategy == "CASH_ONLY" else f"{prefix}_nav"
        ax.plot(daily["date"], daily[nav_col], label=strategy)
    ax.set_title("SPY/CASH Timing Strategies - Linear NAV")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_LINEAR_PATH, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for strategy in DRAW_PLOT_STRATEGIES:
        nav_col = "spy_buy_hold_nav" if strategy == "SPY_BUY_HOLD" else f"{strategy.lower()}_nav"
        nav = daily[nav_col]
        dd = nav / nav.cummax() - 1.0
        ax.plot(daily["date"], dd, label=strategy)
    ax.set_title("Drawdown Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DD_PATH, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily["date"], daily["spy_price"] / daily["spy_price"].iloc[0], color="black", linewidth=1.0)
    both_spy = (daily["faber_signal"] == 1) & (daily["antonacci_signal"] == 1)
    both_cash = (daily["faber_signal"] == 0) & (daily["antonacci_signal"] == 0)
    disagree = daily["faber_signal"] != daily["antonacci_signal"]
    ax.fill_between(daily["date"], 0, 1, where=both_spy, color="tab:blue", alpha=0.12, transform=ax.get_xaxis_transform(), label="Both SPY")
    ax.fill_between(daily["date"], 0, 1, where=both_cash, color="tab:green", alpha=0.12, transform=ax.get_xaxis_transform(), label="Both CASH")
    ax.fill_between(daily["date"], 0, 1, where=disagree, color="tab:orange", alpha=0.15, transform=ax.get_xaxis_transform(), label="Disagreement")
    ax.set_title("Faber vs Antonacci Signal Overlay")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_SIGNAL_OVERLAY_PATH, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily["date"], daily["spy_price"], label="SPY price")
    ax.plot(daily["date"], daily["faber_10m_sma"], label="10M SMA")
    flips = daily["faber_signal"].ne(daily["faber_signal"].shift(1)).fillna(False)
    buys = flips & daily["faber_signal"].eq(1)
    sells = flips & daily["faber_signal"].eq(0)
    ax.scatter(daily.loc[buys, "date"], daily.loc[buys, "spy_price"], color="green", s=20, label="Buy")
    ax.scatter(daily.loc[sells, "date"], daily.loc[sells, "spy_price"], color="red", s=20, label="Sell")
    ax.set_title("Faber Price vs 10M SMA")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_FABER_SMA_PATH, dpi=180)
    plt.close(fig)

    ratio = daily["spy_price"] / daily["faber_10m_sma"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(daily["date"], daily["antonacci_excess_momentum"], color="tab:purple")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("Antonacci Excess Momentum")
    axes[1].plot(daily["date"], ratio, color="tab:blue")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title("SPY Price / 10M SMA")
    fig.tight_layout()
    fig.savefig(FIG_SIGNAL_COMPARE_PATH, dpi=180)
    plt.close(fig)

    roll_window = 756
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for strategy in ROLL_PLOT_STRATEGIES:
        if strategy == "SPY_BUY_HOLD":
            ret_col, nav_col = "spy_buy_hold_return", "spy_buy_hold_nav"
        else:
            ret_col, nav_col = f"{strategy.lower()}_daily_return", f"{strategy.lower()}_nav"
        roll_sharpe = []
        roll_dd = []
        for i in range(len(daily)):
            if i < roll_window - 1:
                roll_sharpe.append(np.nan)
                roll_dd.append(np.nan)
                continue
            sl = daily.iloc[i - roll_window + 1 : i + 1]
            r = sl[ret_col]
            ex = r - sl["daily_rf"]
            ex_std = ex.std(ddof=1)
            roll_sharpe.append(float(ex.mean() / ex_std * np.sqrt(TRADING_DAYS)) if pd.notna(ex_std) and ex_std != 0 else np.nan)
            roll_dd.append(max_drawdown_from_returns(r))
        axes[0].plot(daily["date"], roll_sharpe, label=strategy)
        axes[1].plot(daily["date"], roll_dd, label=strategy)
    axes[0].set_title("Rolling 3Y Sharpe")
    axes[1].set_title("Rolling 3Y Max Drawdown")
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_ROLL_SHARPE_PATH, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for strategy in ["FABER_10M_SMA_MONTHLY", "ANTONACCI_12M_ABS_MOM", "FABER_RECOVERY_OVERLAY", "BOTH_CONFIRM", "EITHER_CONFIRM"]:
        ax.step(daily["date"], daily[f"{strategy.lower()}_target_weight_spy"], where="post", label=strategy)
    ax.set_title("Strategy Switches Timeline (SPY Weight)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_SWITCHES_PATH, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for strategy in ROLL_PLOT_STRATEGIES:
        if strategy == "SPY_BUY_HOLD":
            ret_col = "spy_buy_hold_return"
        else:
            ret_col = f"{strategy.lower()}_daily_return"
        roll_dd = []
        for i in range(len(daily)):
            if i < roll_window - 1:
                roll_dd.append(np.nan)
                continue
            sl = daily.iloc[i - roll_window + 1 : i + 1]
            roll_dd.append(max_drawdown_from_returns(sl[ret_col]))
        ax.plot(daily["date"], roll_dd, label=strategy)
    ax.set_title("Rolling 3Y Max Drawdown")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_ROLL_DD_PATH, dpi=180)
    plt.close(fig)


def concise_diagnostics(perf: pd.DataFrame, disagree: pd.DataFrame, monthly: pd.DataFrame) -> list[str]:
    faber = perf.loc[perf["strategy"] == "FABER_10M_SMA_MONTHLY"].iloc[0]
    anton = perf.loc[perf["strategy"] == "ANTONACCI_12M_ABS_MOM"].iloc[0]
    spy = perf.loc[perf["strategy"] == "SPY_BUY_HOLD"].iloc[0]
    recovery = perf.loc[perf["strategy"] == "FABER_RECOVERY_OVERLAY"].iloc[0]
    best = perf.sort_values(["sharpe_ratio", "max_drawdown"], ascending=[False, False]).iloc[0]
    return [
        f"1. Faber 降低 max drawdown vs SPY: {faber['max_drawdown'] > spy['max_drawdown']}",
        f"2. Faber 提高 Sharpe vs SPY: {faber['sharpe_ratio'] > spy['sharpe_ratio']}",
        f"3. Faber vs Antonacci 年化收益更高: {'FABER' if faber['annualized_return'] > anton['annualized_return'] else 'ANTONACCI'}",
        f"4. 最大回撤更小: {'FABER' if faber['max_drawdown'] > anton['max_drawdown'] else 'ANTONACCI'}",
        f"5. 换手更高: {'FABER' if faber['number_of_switches'] > anton['number_of_switches'] else 'ANTONACCI'}",
        f"6. Faber 是否更早入场: 需要看分歧窗口；FABER=SPY, ANTONACCI=CASH 后续21d收益={disagree.loc[disagree['case']=='FABER_SPY_ANTONACCI_CASH','avg_next_21d_spy_return'].iloc[0]:.2%}",
        f"7. Faber 是否更早出场: FABER=CASH, ANTONACCI=SPY 后续21d回撤={disagree.loc[disagree['case']=='FABER_CASH_ANTONACCI_SPY','avg_next_21d_spy_max_drawdown'].iloc[0]:.2%}",
        f"8. Faber=SPY 而 Antonacci=CASH 时，SPY 后续收益是否为正: {disagree.loc[disagree['case']=='FABER_SPY_ANTONACCI_CASH','avg_next_21d_spy_return'].iloc[0] > 0}",
        f"9. Faber=CASH 而 Antonacci=SPY 时，SPY 后续回撤是否更大: {disagree.loc[disagree['case']=='FABER_CASH_ANTONACCI_SPY','avg_next_21d_spy_max_drawdown'].iloc[0] < 0}",
        f"10. 下一轮最值得研究的组合规则: {best['strategy']}；如果只看 overlay，FABER_RECOVERY_OVERLAY 年化={recovery['annualized_return']:.2%}，Sharpe={recovery['sharpe_ratio']:.2f}",
    ]


def main() -> None:
    ensure_dirs()
    base = build_base_panel(DEFAULT_TICKER)
    monthly = add_combined_signals(merge_monthly_signals(base))
    daily, perf, yearly, dd = build_daily_panel_and_metrics(base, monthly)
    disagree = build_signal_disagreement_summary(monthly, daily)

    monthly_out = monthly[
        [
            "signal_date",
            "effective_date",
            "spy_price",
            "faber_10m_sma",
            "faber_signal_num",
            "faber_200d_sma",
            "faber_200d_signal_num",
            "antonacci_12m_spy_return",
            "antonacci_12m_cash_return",
            "antonacci_excess_momentum",
            "antonacci_signal_num",
            "both_confirm_signal_num",
            "either_confirm_signal_num",
            "faber_entry_antonacci_exit_signal_num",
            "antonacci_entry_faber_exit_signal_num",
            "faber_recovery_overlay_spy_weight",
            "faber_full_recovery_overlay_spy_weight",
        ]
    ].copy()
    monthly_out = monthly_out.rename(
        columns={
            "faber_signal_num": "faber_signal",
            "faber_200d_signal_num": "faber_200d_signal",
            "antonacci_signal_num": "antonacci_signal",
            "both_confirm_signal_num": "both_confirm_signal",
            "either_confirm_signal_num": "either_confirm_signal",
            "faber_entry_antonacci_exit_signal_num": "faber_entry_antonacci_exit_signal",
            "antonacci_entry_faber_exit_signal_num": "antonacci_entry_faber_exit_signal",
        }
    )

    daily.to_csv(DAILY_PANEL_PATH, index=False)
    monthly_out.to_csv(MONTHLY_SIGNAL_PATH, index=False)
    perf.to_csv(PERF_PATH, index=False)
    yearly.to_csv(YEARLY_PATH, index=False)
    dd.to_csv(DD_PATH, index=False)
    disagree.to_csv(DISAGREE_PATH, index=False)

    plot_outputs(daily, perf)

    print(f"Sample: {daily['date'].min().date()} to {daily['date'].max().date()}")
    for line in concise_diagnostics(perf, disagree, monthly):
        print(line)
    for path in [
        DAILY_PANEL_PATH,
        MONTHLY_SIGNAL_PATH,
        PERF_PATH,
        YEARLY_PATH,
        DD_PATH,
        DISAGREE_PATH,
        FIG_LOG_PATH,
        FIG_LINEAR_PATH,
        FIG_DD_PATH,
        FIG_SIGNAL_OVERLAY_PATH,
        FIG_FABER_SMA_PATH,
        FIG_SIGNAL_COMPARE_PATH,
        FIG_ROLL_SHARPE_PATH,
        FIG_ROLL_DD_PATH,
        FIG_SWITCHES_PATH,
    ]:
        if path.exists():
            print(f"Saved: {path}")


if __name__ == "__main__":
    main()
