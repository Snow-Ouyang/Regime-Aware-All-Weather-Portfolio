from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "monthly_either_crash_brake_diagnostics"

FREQ_TEST_DAILY_PATH = ROOT / "results" / "spy_cash_timing_frequency_test" / "daily_backtest_panel.csv"
FREQ_TEST_MONTHLY_SIGNAL_PATH = ROOT / "results" / "spy_cash_timing_frequency_test" / "monthly_signal_panel.csv"
DAILY_CLOSE_PATH = ROOT / "data" / "processed" / "assets" / "daily_adjusted_close.csv"
DAILY_RETURNS_PATH = ROOT / "data" / "processed" / "assets" / "daily_returns.csv"
DTB3_PATH = ROOT / "data" / "raw" / "macro" / "rate" / "DTB3.csv"
VIX_PATH = ROOT / "data" / "raw" / "macro" / "volatility" / "VIXCLS.csv"
WAAA_PATH = ROOT / "data" / "raw" / "macro" / "credit" / "WAAA.csv"
WBAA_PATH = ROOT / "data" / "raw" / "macro" / "credit" / "WBAA.csv"

DAILY_PANEL_OUT = RESULTS_DIR / "crash_trigger_daily_panel.csv"
EVENTS_OUT = RESULTS_DIR / "crash_trigger_events.csv"
SUMMARY_OUT = RESULTS_DIR / "trigger_forward_summary.csv"
MISSED_OUT = RESULTS_DIR / "baseline_missed_drawdown_episodes.csv"
PRECISION_RECALL_OUT = RESULTS_DIR / "trigger_precision_recall.csv"
CRISIS_CASE_OUT = RESULTS_DIR / "crisis_case_study.csv"
SUMMARY_MD_OUT = RESULTS_DIR / "summary.md"

FIG_SCATTER = RESULTS_DIR / "trigger_precision_recall_scatter.png"
FIG_MDD_BAR = RESULTS_DIR / "trigger_forward_mdd_bar.png"
FIG_PROB_BAR = RESULTS_DIR / "trigger_future_mdd10_probability.png"
FIG_PRICE_TRIG = RESULTS_DIR / "spy_price_with_drawdown_triggers.png"
FIG_DD = RESULTS_DIR / "drawdown_from_previous_high.png"
FIG_CRISIS = RESULTS_DIR / "crisis_case_study_plots.png"

CONFIG = {
    "ticker": "SPY",
    "cooldown_days": 21,
    "forward_windows": [5, 10, 21, 63],
    "drawdown_thresholds": [-0.05, -0.08, -0.10, -0.15],
    "cash_rate_file": "data/raw/macro/rate/DTB3.csv",
}

CRISIS_WINDOWS = {
    "DOTCOM_2000_2002": ("2000-01-01", "2002-12-31"),
    "GFC_2008_2009": ("2008-09-01", "2009-03-31"),
    "COVID_2020": ("2020-02-19", "2020-04-30"),
    "INFLATION_2022": ("2022-01-01", "2022-12-31"),
    "RECENT_2024_2026": ("2024-01-01", "2026-12-31"),
}


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _read_spy_base() -> pd.DataFrame:
    if not DAILY_CLOSE_PATH.exists() or not DAILY_RETURNS_PATH.exists():
        raise FileNotFoundError("Missing processed SPY asset files.")
    close = pd.read_csv(DAILY_CLOSE_PATH, usecols=["date", CONFIG["ticker"]])
    ret = pd.read_csv(DAILY_RETURNS_PATH, usecols=["date", CONFIG["ticker"]])
    close["date"] = pd.to_datetime(close["date"])
    ret["date"] = pd.to_datetime(ret["date"])
    panel = close.merge(ret, on="date", how="inner", suffixes=("_price", "_return"))
    panel.columns = ["date", "spy_price", "spy_daily_return"]
    panel["spy_price"] = pd.to_numeric(panel["spy_price"], errors="coerce")
    panel["spy_daily_return"] = pd.to_numeric(panel["spy_daily_return"], errors="coerce")
    panel = panel.dropna(subset=["spy_price"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return panel


def _read_rf(dates: pd.Series) -> pd.DataFrame:
    if not DTB3_PATH.exists():
        raise FileNotFoundError(f"Missing cash rate file: {DTB3_PATH}")
    rf = pd.read_csv(DTB3_PATH)
    date_col = next((c for c in rf.columns if "date" in c.lower()), rf.columns[0])
    value_col = next((c for c in rf.columns if c != date_col), rf.columns[-1])
    rf = rf[[date_col, value_col]].copy()
    rf.columns = ["date", "DTB3"]
    rf["date"] = pd.to_datetime(rf["date"])
    rf["DTB3"] = pd.to_numeric(rf["DTB3"].replace(".", np.nan), errors="coerce")
    rf = rf.sort_values("date")
    rf["DTB3"] = rf["DTB3"].ffill()
    rf["daily_rf"] = (1.0 + rf["DTB3"] / 100.0) ** (1.0 / 252.0) - 1.0
    out = pd.DataFrame({"date": pd.to_datetime(dates)})
    out = out.merge(rf[["date", "daily_rf"]], on="date", how="left")
    out["daily_rf"] = out["daily_rf"].ffill()
    return out


def load_data() -> pd.DataFrame:
    if FREQ_TEST_DAILY_PATH.exists():
        panel = pd.read_csv(FREQ_TEST_DAILY_PATH)
        panel["date"] = pd.to_datetime(panel["date"])
        panel = panel.sort_values("date").reset_index(drop=True)
        return panel
    base = _read_spy_base()
    rf = _read_rf(base["date"])
    panel = base.merge(rf, on="date", how="left")
    panel["cash_nav"] = (1.0 + panel["daily_rf"]).cumprod()
    return rebuild_monthly_either_if_needed(panel)


def rebuild_monthly_either_if_needed(panel: pd.DataFrame) -> pd.DataFrame:
    if "monthly_either_weight_spy" in panel.columns:
        return panel

    out = panel.copy()
    out["month"] = out["date"].dt.to_period("M")
    monthly = out.groupby("month", as_index=False).tail(1).copy()
    monthly["faber_10m_sma"] = monthly["spy_price"].rolling(10, min_periods=10).mean()
    monthly["faber_monthly_signal"] = np.where(monthly["spy_price"] > monthly["faber_10m_sma"], 1.0, np.where(monthly["faber_10m_sma"].notna(), 0.0, np.nan))
    monthly["antonacci_12m_spy_return"] = monthly["spy_price"].pct_change(12)
    monthly["antonacci_12m_cash_return"] = monthly["cash_nav"].pct_change(12)
    monthly["antonacci_excess_momentum"] = monthly["antonacci_12m_spy_return"] - monthly["antonacci_12m_cash_return"]
    monthly["antonacci_monthly_signal"] = np.where(monthly["antonacci_excess_momentum"] > 0, 1.0, np.where(monthly["antonacci_excess_momentum"].notna(), 0.0, np.nan))
    monthly["monthly_either_signal"] = np.where(
        (monthly["faber_monthly_signal"] == 1.0) | (monthly["antonacci_monthly_signal"] == 1.0),
        1.0,
        np.where(monthly["faber_monthly_signal"].notna() & monthly["antonacci_monthly_signal"].notna(), 0.0, np.nan),
    )
    next_trade_date = pd.Series(out["date"].shift(-1).to_numpy(), index=out["date"])
    monthly["signal_date"] = monthly["date"]
    monthly["effective_date"] = pd.to_datetime(monthly["signal_date"].map(next_trade_date))
    weight_map = monthly.loc[monthly["effective_date"].notna() & monthly["monthly_either_signal"].notna(), ["signal_date", "effective_date", "monthly_either_signal", "faber_monthly_signal", "antonacci_monthly_signal"]].copy()

    out["monthly_either_weight_spy"] = np.nan
    out["faber_monthly_signal"] = np.nan
    out["antonacci_monthly_signal"] = np.nan
    out["monthly_signal_date"] = pd.NaT
    out["monthly_effective_date"] = pd.NaT
    for _, row in weight_map.iterrows():
        mask = out["date"] >= row["effective_date"]
        out.loc[mask, "monthly_either_weight_spy"] = float(row["monthly_either_signal"])
        out.loc[mask, "faber_monthly_signal"] = float(row["faber_monthly_signal"])
        out.loc[mask, "antonacci_monthly_signal"] = float(row["antonacci_monthly_signal"])
        out.loc[mask, "monthly_signal_date"] = row["signal_date"]
        out.loc[mask, "monthly_effective_date"] = row["effective_date"]
    out["monthly_either_weight_spy"] = out["monthly_either_weight_spy"].ffill().fillna(0.0)
    return out


def _load_vix_series(dates: pd.Series) -> pd.DataFrame:
    if not VIX_PATH.exists():
        return pd.DataFrame({"date": pd.to_datetime(dates), "VIX_LEVEL": np.nan})
    vix = pd.read_csv(VIX_PATH)
    date_col = next((c for c in vix.columns if "date" in c.lower()), vix.columns[0])
    value_col = next((c for c in vix.columns if c != date_col), vix.columns[-1])
    vix = vix[[date_col, value_col]].copy()
    vix.columns = ["date", "VIX_LEVEL"]
    vix["date"] = pd.to_datetime(vix["date"])
    vix["VIX_LEVEL"] = pd.to_numeric(vix["VIX_LEVEL"].replace(".", np.nan), errors="coerce")
    vix = vix.sort_values("date")
    out = pd.DataFrame({"date": pd.to_datetime(dates)}).merge(vix, on="date", how="left")
    out["VIX_LEVEL"] = out["VIX_LEVEL"].ffill()
    return out


def _load_credit_series(dates: pd.Series) -> pd.DataFrame:
    if not (WAAA_PATH.exists() and WBAA_PATH.exists()):
        return pd.DataFrame({"date": pd.to_datetime(dates), "CREDIT_SPREAD_BAA_AAA": np.nan})
    aaa = pd.read_csv(WAAA_PATH)
    baa = pd.read_csv(WBAA_PATH)
    def prep(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
        value_col = next((c for c in df.columns if c != date_col), df.columns[-1])
        df = df[[date_col, value_col]].copy()
        df.columns = ["date", col_name]
        df["date"] = pd.to_datetime(df["date"])
        df[col_name] = pd.to_numeric(df[col_name].replace(".", np.nan), errors="coerce")
        return df.sort_values("date")
    aaa = prep(aaa, "WAAA")
    baa = prep(baa, "WBAA")
    credit = aaa.merge(baa, on="date", how="outer").sort_values("date")
    credit["CREDIT_SPREAD_BAA_AAA"] = credit["WBAA"] - credit["WAAA"]
    out = pd.DataFrame({"date": pd.to_datetime(dates)}).merge(credit[["date", "CREDIT_SPREAD_BAA_AAA"]], on="date", how="left")
    out["CREDIT_SPREAD_BAA_AAA"] = out["CREDIT_SPREAD_BAA_AAA"].ffill()
    return out


def build_drawdown_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["previous_high"] = out["spy_price"].cummax()
    out["spy_drawdown_from_previous_high"] = out["spy_price"] / out["previous_high"] - 1.0
    out["spy_ret_5d"] = out["spy_price"] / out["spy_price"].shift(5) - 1.0
    out["spy_ret_10d"] = out["spy_price"] / out["spy_price"].shift(10) - 1.0
    out["spy_ret_20d"] = out["spy_price"] / out["spy_price"].shift(20) - 1.0
    out["MA50"] = out["spy_price"].rolling(50, min_periods=50).mean()
    out["MA100"] = out["spy_price"].rolling(100, min_periods=100).mean()
    out["MA200"] = out["spy_price"].rolling(200, min_periods=200).mean()
    return out


def build_vix_credit_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    vix = _load_vix_series(out["date"])
    credit = _load_credit_series(out["date"])
    out = out.merge(vix, on="date", how="left")
    out = out.merge(credit, on="date", how="left")

    if out["VIX_LEVEL"].notna().any():
        out["vix_5d_change"] = out["VIX_LEVEL"] / out["VIX_LEVEL"].shift(5) - 1.0
        out["vix_20d_change"] = out["VIX_LEVEL"] / out["VIX_LEVEL"].shift(20) - 1.0
        out["vix_ma60"] = out["VIX_LEVEL"].rolling(60, min_periods=60).mean()
        out["vix_to_ma60"] = out["VIX_LEVEL"] / out["vix_ma60"]
    else:
        out["vix_5d_change"] = np.nan
        out["vix_20d_change"] = np.nan
        out["vix_ma60"] = np.nan
        out["vix_to_ma60"] = np.nan

    if out["CREDIT_SPREAD_BAA_AAA"].notna().any():
        out["credit_spread_5d_change"] = out["CREDIT_SPREAD_BAA_AAA"] - out["CREDIT_SPREAD_BAA_AAA"].shift(5)
        out["credit_spread_20d_change"] = out["CREDIT_SPREAD_BAA_AAA"] - out["CREDIT_SPREAD_BAA_AAA"].shift(20)
    else:
        out["credit_spread_5d_change"] = np.nan
        out["credit_spread_20d_change"] = np.nan
    return out


def build_crash_triggers(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = panel.copy()
    active = out["monthly_either_weight_spy"] == 1.0

    out["CB_DD_5"] = active & (out["spy_drawdown_from_previous_high"] <= -0.05)
    out["CB_DD_8"] = active & (out["spy_drawdown_from_previous_high"] <= -0.08)
    out["CB_DD_10"] = active & (out["spy_drawdown_from_previous_high"] <= -0.10)
    out["CB_DD_15"] = active & (out["spy_drawdown_from_previous_high"] <= -0.15)

    out["CB_RET_10D_NEG5"] = active & (out["spy_ret_10d"] <= -0.05)
    out["CB_RET_20D_NEG8"] = active & (out["spy_ret_20d"] <= -0.08)

    out["CB_PRICE_BELOW_MA50"] = active & (out["spy_price"] < out["MA50"])
    out["CB_PRICE_BELOW_MA100"] = active & (out["spy_price"] < out["MA100"])
    out["CB_PRICE_BELOW_MA200"] = active & (out["spy_price"] < out["MA200"])

    if out["VIX_LEVEL"].notna().any():
        out["CB_VIX_20_AND_BELOW_MA50"] = active & (out["VIX_LEVEL"] > 20.0) & (out["spy_price"] < out["MA50"])
        out["CB_VIX_25"] = active & (out["VIX_LEVEL"] > 25.0)
        out["CB_VIX_SPIKE_AND_RET_NEG"] = active & (out["vix_5d_change"] > 0.30) & (out["spy_ret_10d"] <= -0.03)
        out["CB_VIX_TO_MA60_HIGH"] = active & (out["vix_to_ma60"] > 1.3)
        out["CB_DD8_AND_VIX_RISING"] = active & (out["spy_drawdown_from_previous_high"] <= -0.08) & (out["vix_5d_change"] > 0)
        out["CB_DD8_OR_VIX25"] = active & ((out["spy_drawdown_from_previous_high"] <= -0.08) | (out["VIX_LEVEL"] > 25.0))
    else:
        for col in ["CB_VIX_20_AND_BELOW_MA50", "CB_VIX_25", "CB_VIX_SPIKE_AND_RET_NEG", "CB_VIX_TO_MA60_HIGH", "CB_DD8_AND_VIX_RISING", "CB_DD8_OR_VIX25"]:
            out[col] = False

    if out["CREDIT_SPREAD_BAA_AAA"].notna().any():
        out["CB_CREDIT_WIDEN_20D"] = active & (out["credit_spread_20d_change"] > 0)
        out["CB_DD8_AND_CREDIT_WIDEN"] = active & (out["spy_drawdown_from_previous_high"] <= -0.08) & (out["credit_spread_20d_change"] > 0)
    else:
        out["CB_CREDIT_WIDEN_20D"] = False
        out["CB_DD8_AND_CREDIT_WIDEN"] = False

    out["CB_DD8_AND_BELOW_MA50"] = active & (out["spy_drawdown_from_previous_high"] <= -0.08) & (out["spy_price"] < out["MA50"])

    trigger_cols = [c for c in out.columns if c.startswith("CB_")]
    return out, trigger_cols


def compute_forward_metrics(panel: pd.DataFrame, start_idx: int) -> dict[str, object]:
    price0 = float(panel.at[start_idx, "spy_price"])
    metrics: dict[str, object] = {}
    trough_63_date = pd.NaT
    trough_63_mdd = np.nan
    for w in CONFIG["forward_windows"]:
        end_idx = min(start_idx + w, len(panel) - 1)
        window = panel.iloc[start_idx : end_idx + 1]
        if len(window) < 2:
            metrics[f"forward_return_{w}d"] = np.nan
            metrics[f"forward_max_drawdown_{w}d"] = np.nan
            continue
        end_price = float(window["spy_price"].iloc[-1])
        path = window["spy_price"] / price0
        dd = path / path.cummax() - 1.0
        metrics[f"forward_return_{w}d"] = end_price / price0 - 1.0
        metrics[f"forward_max_drawdown_{w}d"] = float(dd.min())
        if w == 63 and dd.notna().any():
            trough_idx = dd.idxmin()
            trough_63_date = panel.loc[trough_idx, "date"]
            trough_63_mdd = float(dd.min())
    metrics["future_21d_mdd_below_5"] = bool(metrics.get("forward_max_drawdown_21d", np.nan) <= -0.05) if pd.notna(metrics.get("forward_max_drawdown_21d", np.nan)) else False
    metrics["future_63d_mdd_below_10"] = bool(metrics.get("forward_max_drawdown_63d", np.nan) <= -0.10) if pd.notna(metrics.get("forward_max_drawdown_63d", np.nan)) else False
    metrics["future_63d_mdd_below_15"] = bool(metrics.get("forward_max_drawdown_63d", np.nan) <= -0.15) if pd.notna(metrics.get("forward_max_drawdown_63d", np.nan)) else False
    metrics["trough_date_63d"] = trough_63_date
    metrics["days_to_trough_63d"] = int((trough_63_date - panel.at[start_idx, "date"]).days) if pd.notna(trough_63_date) else np.nan
    metrics["trough_mdd_63d"] = trough_63_mdd
    return metrics


def extract_trigger_events(panel: pd.DataFrame, trigger_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cooldown_days = int(CONFIG["cooldown_days"])
    for trigger in trigger_cols:
        cooldown_until = -1
        prev = False
        for idx, row in panel.iterrows():
            current = bool(row[trigger])
            if idx <= cooldown_until:
                prev = current
                continue
            if current and not prev:
                info = {
                    "trigger_name": trigger,
                    "event_date": row["date"],
                    "spy_price": row["spy_price"],
                    "spy_drawdown_from_previous_high": row["spy_drawdown_from_previous_high"],
                    "monthly_either_weight_spy": row["monthly_either_weight_spy"],
                    "VIX_LEVEL": row.get("VIX_LEVEL", np.nan),
                    "CREDIT_SPREAD_BAA_AAA": row.get("CREDIT_SPREAD_BAA_AAA", np.nan),
                }
                info.update(compute_forward_metrics(panel, idx))
                rows.append(info)
                cooldown_until = idx + cooldown_days - 1
            prev = current
    events = pd.DataFrame(rows)
    return events.sort_values(["trigger_name", "event_date"]).reset_index(drop=True) if not events.empty else pd.DataFrame()


def identify_baseline_missed_drawdowns(panel: pd.DataFrame, trigger_cols: list[str], events: pd.DataFrame) -> pd.DataFrame:
    active = panel["monthly_either_weight_spy"] == 1.0
    rows: list[dict[str, object]] = []
    cooldown_until = -1
    for idx, row in panel.loc[active].iterrows():
        if idx <= cooldown_until:
            continue
        fwd = compute_forward_metrics(panel, idx)
        if not bool(fwd["future_63d_mdd_below_10"]):
            continue
        episode_start = row["date"]
        trough_date = fwd["trough_date_63d"]
        near_mask = (events["event_date"] >= episode_start - pd.Timedelta(days=7)) & (events["event_date"] <= episode_start + pd.Timedelta(days=7)) if not events.empty else pd.Series([], dtype=bool)
        fired = sorted(events.loc[near_mask, "trigger_name"].unique().tolist()) if not events.empty else []
        rows.append(
            {
                "episode_id": len(rows) + 1,
                "start_date": episode_start,
                "trough_date": trough_date,
                "forward_max_drawdown_63d": fwd["forward_max_drawdown_63d"],
                "monthly_either_weight_spy": row["monthly_either_weight_spy"],
                "spy_drawdown_from_previous_high_at_start": row["spy_drawdown_from_previous_high"],
                "VIX_LEVEL_at_start": row.get("VIX_LEVEL", np.nan),
                "which_triggers_fired_near_start": "|".join(fired),
            }
        )
        cooldown_until = idx + int(CONFIG["cooldown_days"]) - 1
    return pd.DataFrame(rows)


def compute_precision_recall(events: pd.DataFrame, missed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    summary_rows = []
    sample_years = max((events["event_date"].max() - events["event_date"].min()).days / 365.25, 1.0)
    for trigger, grp in events.groupby("trigger_name", observed=False):
        precision = float(grp["future_63d_mdd_below_10"].mean()) if len(grp) else np.nan
        false_alarm = float((~grp["future_21d_mdd_below_5"]).mean()) if len(grp) else np.nan
        lead = grp["days_to_trough_63d"]
        lead = lead.loc[pd.notna(lead)]
        recall_hits = 0
        if not missed.empty:
            for _, ep in missed.iterrows():
                start = pd.Timestamp(ep["start_date"])
                hit = ((grp["event_date"] >= start - pd.Timedelta(days=7)) & (grp["event_date"] <= start + pd.Timedelta(days=7))).any()
                recall_hits += int(hit)
        recall = float(recall_hits / len(missed)) if len(missed) else np.nan
        rows.append(
            {
                "trigger_name": trigger,
                "precision_63d_mdd10": precision,
                "recall_missed_drawdown": recall,
                "false_alarm_rate_63d": false_alarm,
                "avg_lead_time_to_trough": float(lead.mean()) if not lead.empty else np.nan,
                "event_count": int(len(grp)),
            }
        )
        summary_rows.append(
            {
                "trigger_name": trigger,
                "event_count": int(len(grp)),
                "avg_events_per_year": float(len(grp) / sample_years),
                "avg_forward_return_21d": float(grp["forward_return_21d"].mean()),
                "avg_forward_return_63d": float(grp["forward_return_63d"].mean()),
                "avg_forward_max_drawdown_21d": float(grp["forward_max_drawdown_21d"].mean()),
                "avg_forward_max_drawdown_63d": float(grp["forward_max_drawdown_63d"].mean()),
                "pct_future_21d_mdd_below_5": float(grp["future_21d_mdd_below_5"].mean()),
                "pct_future_63d_mdd_below_10": float(grp["future_63d_mdd_below_10"].mean()),
                "pct_future_63d_mdd_below_15": float(grp["future_63d_mdd_below_15"].mean()),
                "false_alarm_rate_63d": false_alarm,
                "median_days_to_trough_63d": float(grp["days_to_trough_63d"].median()) if grp["days_to_trough_63d"].notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def build_crisis_case_studies(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if events.empty:
        return pd.DataFrame()
    for period, (start, end) in CRISIS_WINDOWS.items():
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        sub_events = events.loc[(events["event_date"] >= start_ts) & (events["event_date"] <= end_ts)]
        for trigger, grp in sub_events.groupby("trigger_name", observed=False):
            first = grp.sort_values("event_date").iloc[0]
            after = panel.loc[panel["date"] > first["event_date"], ["date", "monthly_either_weight_spy"]].copy()
            exit_rows = after.loc[after["monthly_either_weight_spy"] == 0.0, "date"]
            exit_date = exit_rows.iloc[0] if not exit_rows.empty else pd.NaT
            helpful = pd.notna(exit_date) and pd.notna(first["trough_date_63d"]) and exit_date <= first["trough_date_63d"]
            rows.append(
                {
                    "period": period,
                    "trigger_name": trigger,
                    "first_trigger_date": first["event_date"],
                    "SPY_drawdown_at_first_trigger": first["spy_drawdown_from_previous_high"],
                    "Monthly_Either_position_at_first_trigger": first["monthly_either_weight_spy"],
                    "days_before_monthly_either_exit": int((exit_date - first["event_date"]).days) if pd.notna(exit_date) else np.nan,
                    "forward_max_drawdown_after_trigger": first["forward_max_drawdown_63d"],
                    "trigger_helpful_flag": bool(helpful),
                }
            )
    return pd.DataFrame(rows)


def plot_results(panel: pd.DataFrame, summary: pd.DataFrame, precision_recall: pd.DataFrame, events: pd.DataFrame) -> None:
    if not precision_recall.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sizes = 30 + 8 * precision_recall["event_count"].fillna(0)
        ax.scatter(precision_recall["false_alarm_rate_63d"], precision_recall["recall_missed_drawdown"], s=sizes, alpha=0.7)
        for _, row in precision_recall.iterrows():
            ax.annotate(row["trigger_name"], (row["false_alarm_rate_63d"], row["recall_missed_drawdown"]), fontsize=7)
        ax.set_xlabel("False alarm rate 63D")
        ax.set_ylabel("Recall of missed drawdowns")
        ax.set_title("Crash Trigger Precision / Recall")
        fig.tight_layout()
        fig.savefig(FIG_SCATTER, dpi=180)
        plt.close(fig)

    if not summary.empty:
        plot_df = summary.sort_values("avg_forward_max_drawdown_63d")
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.barh(plot_df["trigger_name"], plot_df["avg_forward_max_drawdown_63d"])
        ax.set_title("Average Forward Max Drawdown over 63D")
        fig.tight_layout()
        fig.savefig(FIG_MDD_BAR, dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 7))
        ax.barh(plot_df["trigger_name"], plot_df["pct_future_63d_mdd_below_10"])
        ax.set_title("Probability of Future 63D Drawdown <= -10%")
        fig.tight_layout()
        fig.savefig(FIG_PROB_BAR, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(panel["date"], panel["spy_price"], color="black", label="SPY")
    cash_mask = panel["monthly_either_weight_spy"] == 0.0
    if cash_mask.any():
        ax.fill_between(panel["date"], panel["spy_price"].min(), panel["spy_price"].max(), where=cash_mask, color="lightgray", alpha=0.3, label="Monthly Either = CASH")
    for trig, color in [("CB_DD_8", "tab:red"), ("CB_VIX_20_AND_BELOW_MA50", "tab:orange"), ("CB_VIX_25", "tab:purple"), ("CB_DD8_AND_BELOW_MA50", "tab:blue")]:
        if trig in panel.columns:
            pts = panel.loc[panel[trig]]
            ax.scatter(pts["date"], pts["spy_price"], s=10, label=trig, color=color)
    ax.set_title("SPY Price with Crash Brake Triggers")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_PRICE_TRIG, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(panel["date"], panel["spy_drawdown_from_previous_high"], color="tab:red")
    for thr in [-0.05, -0.08, -0.10, -0.15]:
        ax.axhline(thr, linestyle="--", color="gray", linewidth=0.8)
    ax.fill_between(panel["date"], -0.3, 0.02, where=panel["monthly_either_weight_spy"] == 1.0, color="tab:blue", alpha=0.08, label="Monthly Either = SPY")
    ax.set_ylim(-0.65, 0.02)
    ax.set_title("SPY Drawdown from Previous High")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DD, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    for ax, (period, (start, end)) in zip(axes, [("GFC_2008_2009", CRISIS_WINDOWS["GFC_2008_2009"]), ("COVID_2020", CRISIS_WINDOWS["COVID_2020"]), ("INFLATION_2022", CRISIS_WINDOWS["INFLATION_2022"])]):
        sub = panel.loc[(panel["date"] >= pd.Timestamp(start)) & (panel["date"] <= pd.Timestamp(end))].copy()
        if sub.empty:
            continue
        ax.plot(sub["date"], sub["spy_price"], color="black", label="SPY")
        ax2 = ax.twinx()
        ax2.plot(sub["date"], sub["spy_drawdown_from_previous_high"], color="tab:red", alpha=0.5, label="DD prev high")
        for trig, color in [("CB_DD_8", "tab:red"), ("CB_VIX_25", "tab:purple"), ("CB_DD8_AND_BELOW_MA50", "tab:blue")]:
            if trig in sub.columns:
                pts = sub.loc[sub[trig]]
                ax.scatter(pts["date"], pts["spy_price"], s=14, color=color)
        exits = sub.loc[sub["monthly_either_weight_spy"].diff() == -1.0, "date"]
        for dt in exits:
            ax.axvline(dt, color="green", linestyle="--", linewidth=1)
        ax.set_title(period)
    fig.tight_layout()
    fig.savefig(FIG_CRISIS, dpi=180)
    plt.close(fig)


def write_summary_md(summary: pd.DataFrame, precision_recall: pd.DataFrame, missed: pd.DataFrame, crisis: pd.DataFrame, panel: pd.DataFrame) -> None:
    vix_available = panel["VIX_LEVEL"].notna().any() if "VIX_LEVEL" in panel.columns else False
    credit_available = panel["CREDIT_SPREAD_BAA_AAA"].notna().any() if "CREDIT_SPREAD_BAA_AAA" in panel.columns else False
    top_precision = precision_recall.sort_values("precision_63d_mdd10", ascending=False).head(5) if not precision_recall.empty else pd.DataFrame()
    top_recall = precision_recall.sort_values("recall_missed_drawdown", ascending=False).head(5) if not precision_recall.empty else pd.DataFrame()
    lines = [
        "# Monthly Either Crash Brake Diagnostics",
        "",
        "## Research Goal",
        "",
        "This analysis does not change the baseline strategy. It studies whether short-term crash brake triggers can identify correction risk while `MONTHLY_EITHER_CONFIRM` still holds SPY.",
        "",
        "## Why Evaluate Only When Monthly Either = SPY",
        "",
        "If the baseline already holds CASH, the main timing layer has already de-risked. Crash brake triggers matter only when the baseline remains long SPY and may miss a faster correction.",
        "",
        "## Why Use Drawdown from Previous High",
        "",
        "This study uses drawdown from cumulative previous high rather than n-day rolling highs. That is closer to the investor's actual account state and avoids resetting the risk measure during long declines.",
        "",
        "## Trigger Definitions",
        "",
        "- Drawdown thresholds from previous high",
        "- Short-term return shock triggers",
        "- Moving-average trend breaks",
        "- Optional VIX and credit stress composites when those fields are available",
        "",
        f"VIX features available: `{vix_available}`",
        f"Credit spread features available: `{credit_available}`",
        "",
        "## Main Statistics",
        "",
        summary.to_markdown(index=False) if not summary.empty else "No trigger summary available.",
        "",
        "## Precision / Recall / False Alarm",
        "",
        precision_recall.to_markdown(index=False) if not precision_recall.empty else "No precision/recall table available.",
        "",
        "## Baseline Missed Drawdown Episodes",
        "",
        f"Baseline missed drawdown episode count: `{len(missed)}`",
        "",
        "## Crisis Case Studies",
        "",
        crisis.to_markdown(index=False) if not crisis.empty else "No crisis case study rows available.",
        "",
        "## Recommended Triggers for Next Backtest",
        "",
        "Use the tradeoff among precision, recall, and false-alarm rate rather than any single metric.",
        "",
        "Top precision candidates:",
        top_precision.to_markdown(index=False) if not top_precision.empty else "None",
        "",
        "Top recall candidates:",
        top_recall.to_markdown(index=False) if not top_recall.empty else "None",
        "",
        "## Limitations",
        "",
        "- This is not a full strategy backtest.",
        "- Trigger events are diagnostic labels, not final execution rules.",
        "- Forward windows overlap across events, so event independence is limited.",
        "- Any trigger selected from this step still needs a full out-of-sample or robustness backtest before deployment.",
    ]
    SUMMARY_MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = load_data()
    panel = rebuild_monthly_either_if_needed(panel)
    panel = build_drawdown_features(panel)
    panel = build_vix_credit_features(panel)
    panel, trigger_cols = build_crash_triggers(panel)

    events = extract_trigger_events(panel, trigger_cols)
    missed = identify_baseline_missed_drawdowns(panel, trigger_cols, events)
    precision_recall, summary = compute_precision_recall(events, missed)
    crisis = build_crisis_case_studies(panel, events)

    keep_cols = [
        "date",
        "spy_price",
        "monthly_either_weight_spy",
        "previous_high",
        "spy_drawdown_from_previous_high",
        "spy_ret_5d",
        "spy_ret_10d",
        "spy_ret_20d",
        "MA50",
        "MA100",
        "MA200",
        "VIX_LEVEL",
        "CREDIT_SPREAD_BAA_AAA",
    ] + trigger_cols
    panel[keep_cols].to_csv(DAILY_PANEL_OUT, index=False)
    events.to_csv(EVENTS_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)
    missed.to_csv(MISSED_OUT, index=False)
    precision_recall.to_csv(PRECISION_RECALL_OUT, index=False)
    crisis.to_csv(CRISIS_CASE_OUT, index=False)
    plot_results(panel, summary, precision_recall, events)
    write_summary_md(summary, precision_recall, missed, crisis, panel)

    best_precision = precision_recall.sort_values("precision_63d_mdd10", ascending=False).iloc[0] if not precision_recall.empty else None
    best_recall = precision_recall.sort_values("recall_missed_drawdown", ascending=False).iloc[0] if not precision_recall.empty else None
    best_false_alarm = precision_recall.sort_values("false_alarm_rate_63d", ascending=True).iloc[0] if not precision_recall.empty else None
    covid_help = crisis.loc[crisis["period"] == "COVID_2020"].sort_values("days_before_monthly_either_exit").head(1) if not crisis.empty else pd.DataFrame()
    infl_help = crisis.loc[crisis["period"] == "INFLATION_2022"].sort_values("forward_max_drawdown_after_trigger").head(1) if not crisis.empty else pd.DataFrame()
    dd_vs_combo = precision_recall.loc[precision_recall["trigger_name"].isin(["CB_DD_8", "CB_DD8_AND_BELOW_MA50", "CB_DD8_AND_VIX_RISING", "CB_DD8_OR_VIX25"])] if not precision_recall.empty else pd.DataFrame()
    recommendation = precision_recall.sort_values(["recall_missed_drawdown", "false_alarm_rate_63d", "precision_63d_mdd10"], ascending=[False, True, False]).head(2) if not precision_recall.empty else pd.DataFrame()

    print(f"1. Baseline missed drawdown episodes: {len(missed)}")
    if best_precision is not None:
        print(f"2. Highest precision trigger: {best_precision['trigger_name']} ({best_precision['precision_63d_mdd10']:.2%})")
    if best_recall is not None:
        print(f"3. Highest recall trigger: {best_recall['trigger_name']} ({best_recall['recall_missed_drawdown']:.2%})")
    if best_false_alarm is not None:
        print(f"4. Lowest false alarm trigger: {best_false_alarm['trigger_name']} ({best_false_alarm['false_alarm_rate_63d']:.2%})")
    if not covid_help.empty:
        row = covid_help.iloc[0]
        print(f"5. Earliest helpful trigger in 2020: {row['trigger_name']} on {pd.Timestamp(row['first_trigger_date']).date()}")
    if not infl_help.empty:
        row = infl_help.iloc[0]
        print(f"6. Most helpful trigger in 2022 drawdown: {row['trigger_name']}")
    if not dd_vs_combo.empty:
        pure_dd = dd_vs_combo.loc[dd_vs_combo["trigger_name"] == "CB_DD_8"]
        combo = dd_vs_combo.loc[dd_vs_combo["trigger_name"] != "CB_DD_8"]
        pure_score = pure_dd["recall_missed_drawdown"].max() if not pure_dd.empty else np.nan
        combo_score = combo["recall_missed_drawdown"].max() if not combo.empty else np.nan
        better = "composite" if pd.notna(combo_score) and (pd.isna(pure_score) or combo_score > pure_score) else "pure_drawdown_or_similar"
        print(f"7. Pure previous-high drawdown vs VIX/MA composites: {better}")
    if not recommendation.empty:
        print(f"8. Suitable first crash brake trigger exists: True")
        print("9. Recommended triggers for next backtest: " + ", ".join(recommendation["trigger_name"].tolist()))
    else:
        print("8. Suitable first crash brake trigger exists: False")


if __name__ == "__main__":
    main()
