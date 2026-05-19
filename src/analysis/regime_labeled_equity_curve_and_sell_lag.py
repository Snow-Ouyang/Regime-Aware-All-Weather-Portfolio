from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]

TIMING_PANEL_PATH = ROOT / "results" / "spy_cash_timing_frequency_test" / "daily_backtest_panel.csv"
RULE_STATE_PANEL_PATH = ROOT / "results" / "rule_diagnostics" / "rule_state_panel.csv"
RECON_PANEL_PATH = ROOT / "results" / "reconstructed_regime_asset_behavior" / "reconstructed_regime_panel.csv"
HF_PANEL_PATH = ROOT / "results" / "high_frequency_regime_diagnostics" / "high_frequency_regime_feature_panel.csv"

RESULTS_DIR = ROOT / "results" / "regime_labeled_sell_lag_diagnostic"
FIGURES_DIR = ROOT / "figures" / "regime_labeled_sell_lag_diagnostic"

DAILY_PANEL_OUT = RESULTS_DIR / "regime_labeled_daily_panel.csv"
SPELLS_OUT = RESULTS_DIR / "timing_state_spells.csv"
SELL_EVENTS_OUT = RESULTS_DIR / "sell_lag_event_table.csv"
SELL_SUMMARY_OUT = RESULTS_DIR / "sell_lag_summary_by_regime.csv"
SPY_STATE_PERF_OUT = RESULTS_DIR / "spy_performance_by_regime_and_timing_state.csv"
SPY_SHARPE_PIVOT_OUT = RESULTS_DIR / "spy_sharpe_by_regime_timing_state.csv"
SPY_RETURN_PIVOT_OUT = RESULTS_DIR / "spy_annualized_return_by_regime_timing_state.csv"
SPY_DD_PIVOT_OUT = RESULTS_DIR / "spy_max_drawdown_by_regime_timing_state.csv"
REPORT_OUT = RESULTS_DIR / "REGIME_LABELED_SELL_LAG_DIAGNOSTIC.md"

FIG_EQUITY = FIGURES_DIR / "regime_labeled_equity_curve.png"
FIG_SCATTER = FIGURES_DIR / "sell_lag_scatter.png"
FIG_BAR = FIGURES_DIR / "sell_lag_by_regime_bar.png"
FIG_SHARPE = FIGURES_DIR / "spy_sharpe_heatmap_by_regime_timing.png"
FIG_RETURN = FIGURES_DIR / "spy_return_heatmap_by_regime_timing.png"
FIG_CASE = FIGURES_DIR / "case_study_2008_2020_2022.png"
FIG_EVENTS = FIGURES_DIR / "sell_events_timeline.png"

CONFIG = {
    "drawdown_thresholds": [-0.05, -0.10, -0.15, -0.20],
    "case_study_periods": {
        "GFC_2008": ("2007-07-01", "2009-12-31"),
        "COVID_2020": ("2020-01-01", "2020-12-31"),
        "INFLATION_2022": ("2021-11-01", "2023-03-31"),
    },
    "low_sample_threshold": 60,
}

REGIME_ORDER = ["HIGH_INFLATION", "INVERTED", "FLAT", "STEEP", "NEUTRAL"]
TIMING_ORDER = ["HOLD", "SELL"]
REGIME_COLORS = {
    "HIGH_INFLATION": "#d95f02",
    "INVERTED": "#7570b3",
    "FLAT": "#1b9e77",
    "STEEP": "#66a61e",
    "NEUTRAL": "#999999",
}
TIMING_COLORS = {"HOLD": "#4daf4a", "SELL": "#d62728"}


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_timing_panel() -> pd.DataFrame:
    if not TIMING_PANEL_PATH.exists():
        raise FileNotFoundError(f"Missing timing panel: {TIMING_PANEL_PATH}")
    panel = pd.read_csv(TIMING_PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    required = [
        "date",
        "spy_price",
        "spy_daily_return",
        "daily_rf",
        "MONTHLY_EITHER_CONFIRM_nav",
        "SPY_BUY_HOLD_nav",
        "CASH_ONLY_nav",
        "monthly_either_weight_spy",
    ]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"Missing columns in timing panel: {missing}")
    return panel.sort_values("date").reset_index(drop=True)


def load_macro_regime_panel() -> pd.DataFrame:
    if RECON_PANEL_PATH.exists():
        panel = pd.read_csv(RECON_PANEL_PATH)
        panel["date"] = pd.to_datetime(panel["date"])
        panel["macro_regime_confirmed"] = np.select(
            [
                (pd.to_numeric(panel.get("CREDIT_SPREAD_BAA_AAA"), errors="coerce") > 1.5)
                & (pd.to_numeric(panel.get("DGS1"), errors="coerce") > 5.0),
                pd.to_numeric(panel.get("TERM_SPREAD_10Y_1Y"), errors="coerce") < 0.0,
                (pd.to_numeric(panel.get("TERM_SPREAD_10Y_1Y"), errors="coerce") >= 0.0)
                & (pd.to_numeric(panel.get("TERM_SPREAD_10Y_1Y"), errors="coerce") < 1.0),
                pd.to_numeric(panel.get("TERM_SPREAD_10Y_1Y"), errors="coerce") >= 1.0,
            ],
            ["HIGH_INFLATION", "INVERTED", "FLAT", "STEEP"],
            default="NEUTRAL",
        )
        cols = ["date", "macro_regime_confirmed"]
        extras = [c for c in ["CREDIT_SPREAD_BAA_AAA", "DGS1", "DGS10", "TERM_SPREAD_10Y_1Y", "VIX_LEVEL"] if c in panel.columns]
        return panel[cols + extras].sort_values("date").reset_index(drop=True)
    if RULE_STATE_PANEL_PATH.exists():
        panel = pd.read_csv(RULE_STATE_PANEL_PATH)
        panel["date"] = pd.to_datetime(panel["date"])
        panel["macro_regime_confirmed"] = np.select(
            [
                (pd.to_numeric(panel.get("CREDIT_SPREAD_BAA_AAA"), errors="coerce") > 1.5)
                & (pd.to_numeric(panel.get("DGS1"), errors="coerce") > 5.0),
                pd.to_numeric(panel.get("TERM_SPREAD_10Y_1Y"), errors="coerce") < 0.0,
                (pd.to_numeric(panel.get("TERM_SPREAD_10Y_1Y"), errors="coerce") >= 0.0)
                & (pd.to_numeric(panel.get("TERM_SPREAD_10Y_1Y"), errors="coerce") < 1.0),
                pd.to_numeric(panel.get("TERM_SPREAD_10Y_1Y"), errors="coerce") >= 1.0,
            ],
            ["HIGH_INFLATION", "INVERTED", "FLAT", "STEEP"],
            default="NEUTRAL",
        )
        cols = ["date", "macro_regime_confirmed"]
        extras = [c for c in ["CREDIT_SPREAD_BAA_AAA", "DGS1", "DGS10", "TERM_SPREAD_10Y_1Y", "VIX_LEVEL"] if c in panel.columns]
        daily = panel[cols + extras].copy()
        daily["month"] = daily["date"].dt.to_period("M")
        return daily
    if HF_PANEL_PATH.exists():
        raise ValueError("High-frequency panel is long format and not suitable as primary macro regime input here.")
    raise FileNotFoundError("No suitable macro regime panel found.")


def rebuild_monthly_either_if_needed(timing: pd.DataFrame) -> pd.DataFrame:
    return timing


def merge_daily_panel(timing: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    panel = timing.copy()
    if "month" not in panel.columns:
        panel["month"] = panel["date"].dt.to_period("M")
    if "month" in macro.columns and "macro_regime_confirmed" in macro.columns and macro["date"].dt.normalize().nunique() < panel["date"].dt.normalize().nunique():
        macro_monthly = macro.sort_values("date").groupby("month", as_index=False).tail(1).copy()
        macro_monthly["macro_effective_date"] = macro_monthly["date"]
        panel = panel.merge(macro_monthly.drop(columns=["date"]), on="month", how="left")
    else:
        panel = panel.merge(macro, on="date", how="left")
    panel["macro_regime_confirmed"] = panel["macro_regime_confirmed"].ffill().fillna("NEUTRAL")
    panel["spy_timing_state"] = np.where(panel["monthly_either_weight_spy"] >= 0.5, "HOLD", "SELL")
    panel["cross_state"] = panel["macro_regime_confirmed"] + "_" + panel["spy_timing_state"]
    return panel.sort_values("date").reset_index(drop=True)


def build_drawdown_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["previous_high"] = out["spy_price"].cummax()
    high_dates: list[pd.Timestamp] = []
    current_high = -np.inf
    current_date = pd.NaT
    for _, row in out.iterrows():
        if row["spy_price"] >= current_high:
            current_high = row["spy_price"]
            current_date = row["date"]
        high_dates.append(current_date)
    out["previous_high_date"] = pd.to_datetime(high_dates)
    out["spy_drawdown_from_previous_high"] = out["spy_price"] / out["previous_high"] - 1.0
    out["strategy_cum_high"] = out["MONTHLY_EITHER_CONFIRM_nav"].cummax()
    out["strategy_drawdown"] = out["MONTHLY_EITHER_CONFIRM_nav"] / out["strategy_cum_high"] - 1.0
    out["spy_buy_hold_cum_high"] = out["SPY_BUY_HOLD_nav"].cummax()
    out["spy_buy_hold_drawdown"] = out["SPY_BUY_HOLD_nav"] / out["spy_buy_hold_cum_high"] - 1.0
    return out


def identify_timing_spells(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    state_change = out["spy_timing_state"] != out["spy_timing_state"].shift(1)
    out["spell_id"] = state_change.cumsum().astype(int)
    rows = []
    for spell_id, grp in out.groupby("spell_id", observed=False):
        regime_counts = grp["macro_regime_confirmed"].value_counts()
        dominant = regime_counts.index[0]
        sequence = " -> ".join(pd.Series(grp["macro_regime_confirmed"]).loc[pd.Series(grp["macro_regime_confirmed"]).shift() != pd.Series(grp["macro_regime_confirmed"])].tolist())
        spy_wealth = (1.0 + grp["spy_daily_return"]).cumprod()
        spy_dd = spy_wealth / spy_wealth.cummax() - 1.0
        strat_rets = grp["MONTHLY_EITHER_CONFIRM_return"]
        strat_wealth = (1.0 + strat_rets).cumprod()
        strat_dd = strat_wealth / strat_wealth.cummax() - 1.0
        rows.append(
            {
                "spell_id": int(spell_id),
                "state": grp["spy_timing_state"].iloc[0],
                "start_date": grp["date"].iloc[0],
                "end_date": grp["date"].iloc[-1],
                "n_days": int(len(grp)),
                "start_regime": grp["macro_regime_confirmed"].iloc[0],
                "dominant_regime": dominant,
                "regime_sequence": sequence,
                "start_spy_price": float(grp["spy_price"].iloc[0]),
                "end_spy_price": float(grp["spy_price"].iloc[-1]),
                "spy_return_during_spell": float(spy_wealth.iloc[-1] - 1.0),
                "spy_max_drawdown_during_spell": float(spy_dd.min()),
                "monthly_either_return_during_spell": float(strat_wealth.iloc[-1] - 1.0),
                "monthly_either_max_drawdown_during_spell": float(strat_dd.min()),
                "vix_start": float(grp["VIX_LEVEL"].iloc[0]) if "VIX_LEVEL" in grp.columns and pd.notna(grp["VIX_LEVEL"].iloc[0]) else np.nan,
                "credit_spread_start": float(grp["CREDIT_SPREAD_BAA_AAA"].iloc[0]) if "CREDIT_SPREAD_BAA_AAA" in grp.columns and pd.notna(grp["CREDIT_SPREAD_BAA_AAA"].iloc[0]) else np.nan,
            }
        )
    spells = pd.DataFrame(rows)
    return spells


def _forward_path_metrics(panel: pd.DataFrame, idx: int, window: int) -> dict[str, float]:
    end_idx = min(idx + window, len(panel) - 1)
    sub = panel.iloc[idx : end_idx + 1].copy()
    if len(sub) < 2:
        return {"return": np.nan, "max_drawdown": np.nan, "max_runup": np.nan}
    base = float(sub["spy_price"].iloc[0])
    wealth = sub["spy_price"] / base
    dd = wealth / wealth.cummax() - 1.0
    runup = wealth / wealth.cummin() - 1.0
    return {
        "return": float(wealth.iloc[-1] - 1.0),
        "max_drawdown": float(dd.min()),
        "max_runup": float(runup.max()),
    }


def build_sell_lag_event_table(panel: pd.DataFrame, spells: pd.DataFrame) -> pd.DataFrame:
    sell_mask = (panel["spy_timing_state"] == "SELL") & (panel["spy_timing_state"].shift(1) == "HOLD")
    rows = []
    sell_events = panel.loc[sell_mask].copy()
    for event_id, (idx, row) in enumerate(sell_events.iterrows(), start=1):
        spell = spells.loc[spells["spell_id"] == row["spell_id"]].iloc[0]
        prev_cross = panel.loc[idx - 1, "cross_state"] if idx > 0 else np.nan
        post5 = _forward_path_metrics(panel, idx, 5)
        post10 = _forward_path_metrics(panel, idx, 10)
        post21 = _forward_path_metrics(panel, idx, 21)
        post63 = _forward_path_metrics(panel, idx, 63)
        sell_start = row["date"]
        prev_high_date = row["previous_high_date"]
        rows.append(
            {
                "sell_event_id": event_id,
                "sell_start_date": sell_start,
                "macro_regime_at_sell": row["macro_regime_confirmed"],
                "previous_cross_state": prev_cross,
                "sell_cross_state": row["cross_state"],
                "spy_price_at_sell": row["spy_price"],
                "monthly_either_nav_at_sell": row["MONTHLY_EITHER_CONFIRM_nav"],
                "previous_high_date": prev_high_date,
                "days_from_previous_high_to_sell": int((sell_start - prev_high_date).days) if pd.notna(prev_high_date) else np.nan,
                "spy_drawdown_from_previous_high_at_sell": row["spy_drawdown_from_previous_high"],
                "spy_return_from_previous_high_to_sell": row["spy_price"] / row["previous_high"] - 1.0,
                "spy_return_5d_before_sell": row["spy_price"] / panel["spy_price"].shift(5).iloc[idx] - 1.0 if idx >= 5 else np.nan,
                "spy_return_10d_before_sell": row["spy_price"] / panel["spy_price"].shift(10).iloc[idx] - 1.0 if idx >= 10 else np.nan,
                "spy_return_21d_before_sell": row["spy_price"] / panel["spy_price"].shift(21).iloc[idx] - 1.0 if idx >= 21 else np.nan,
                "spy_return_63d_before_sell": row["spy_price"] / panel["spy_price"].shift(63).iloc[idx] - 1.0 if idx >= 63 else np.nan,
                "sell_spell_end_date": spell["end_date"],
                "sell_spell_length_days": spell["n_days"],
                "spy_return_during_sell_spell": spell["spy_return_during_spell"],
                "spy_max_drawdown_during_sell_spell": spell["spy_max_drawdown_during_spell"],
                "spy_max_runup_during_sell_spell": float((panel.loc[panel["spell_id"] == row["spell_id"], "spy_price"] / row["spy_price"] - 1.0).max()),
                "monthly_either_return_during_sell_spell": spell["monthly_either_return_during_spell"],
                "cash_return_during_sell_spell": float((1.0 + panel.loc[panel["spell_id"] == row["spell_id"], "daily_rf"]).prod() - 1.0),
                "spy_forward_return_5d": post5["return"],
                "spy_forward_return_10d": post10["return"],
                "spy_forward_return_21d": post21["return"],
                "spy_forward_return_63d": post63["return"],
                "spy_forward_max_drawdown_21d": post21["max_drawdown"],
                "spy_forward_max_drawdown_63d": post63["max_drawdown"],
                "spy_forward_max_runup_21d": post21["max_runup"],
                "spy_forward_max_runup_63d": post63["max_runup"],
                "sell_after_5pct_drawdown": bool(row["spy_drawdown_from_previous_high"] <= -0.05),
                "sell_after_10pct_drawdown": bool(row["spy_drawdown_from_previous_high"] <= -0.10),
                "sell_after_15pct_drawdown": bool(row["spy_drawdown_from_previous_high"] <= -0.15),
                "sold_before_further_drawdown": bool(post63["max_drawdown"] <= -0.05) if pd.notna(post63["max_drawdown"]) else False,
                "sold_before_rebound": bool(post63["max_runup"] >= 0.05) if pd.notna(post63["max_runup"]) else False,
                "likely_sold_near_low": bool(
                    (row["spy_drawdown_from_previous_high"] <= -0.08)
                    and (pd.notna(post63["max_runup"]) and post63["max_runup"] >= 0.08)
                    and (pd.notna(post21["max_drawdown"]) and pd.notna(post21["max_runup"]) and abs(post21["max_drawdown"]) < abs(post21["max_runup"]))
                ),
                "VIX_LEVEL_at_sell": row["VIX_LEVEL"] if "VIX_LEVEL" in row.index else np.nan,
                "CREDIT_SPREAD_BAA_AAA_at_sell": row["CREDIT_SPREAD_BAA_AAA"] if "CREDIT_SPREAD_BAA_AAA" in row.index else np.nan,
                "DGS1_at_sell": row["DGS1"] if "DGS1" in row.index else np.nan,
                "DGS10_at_sell": row["DGS10"] if "DGS10" in row.index else np.nan,
                "TERM_SPREAD_10Y_1Y_at_sell": row["TERM_SPREAD_10Y_1Y"] if "TERM_SPREAD_10Y_1Y" in row.index else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_sell_lag_by_regime(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, grp in events.groupby("macro_regime_at_sell", observed=False):
        rows.append(
            {
                "macro_regime_at_sell": regime,
                "n_sell_events": int(len(grp)),
                "avg_days_from_previous_high_to_sell": float(grp["days_from_previous_high_to_sell"].mean()),
                "median_days_from_previous_high_to_sell": float(grp["days_from_previous_high_to_sell"].median()),
                "avg_spy_drawdown_from_previous_high_at_sell": float(grp["spy_drawdown_from_previous_high_at_sell"].mean()),
                "median_spy_drawdown_from_previous_high_at_sell": float(grp["spy_drawdown_from_previous_high_at_sell"].median()),
                "pct_sell_after_5pct_drawdown": float(grp["sell_after_5pct_drawdown"].mean()),
                "pct_sell_after_10pct_drawdown": float(grp["sell_after_10pct_drawdown"].mean()),
                "pct_sell_after_15pct_drawdown": float(grp["sell_after_15pct_drawdown"].mean()),
                "avg_spy_return_21d_before_sell": float(grp["spy_return_21d_before_sell"].mean()),
                "avg_spy_return_63d_before_sell": float(grp["spy_return_63d_before_sell"].mean()),
                "avg_spy_forward_return_21d": float(grp["spy_forward_return_21d"].mean()),
                "avg_spy_forward_return_63d": float(grp["spy_forward_return_63d"].mean()),
                "avg_spy_forward_max_drawdown_63d": float(grp["spy_forward_max_drawdown_63d"].mean()),
                "avg_spy_forward_max_runup_63d": float(grp["spy_forward_max_runup_63d"].mean()),
                "pct_sold_before_further_drawdown": float(grp["sold_before_further_drawdown"].mean()),
                "pct_sold_before_rebound": float(grp["sold_before_rebound"].mean()),
                "pct_likely_sold_near_low": float(grp["likely_sold_near_low"].mean()),
                "avg_sell_spell_length_days": float(grp["sell_spell_length_days"].mean()),
                "avg_spy_return_during_sell_spell": float(grp["spy_return_during_sell_spell"].mean()),
                "avg_monthly_either_return_during_sell_spell": float(grp["monthly_either_return_during_sell_spell"].mean()),
                "LOW_SAMPLE": len(grp) < CONFIG["low_sample_threshold"],
            }
        )
    return pd.DataFrame(rows)


def _annualized_return(s: pd.Series) -> float:
    s = s.dropna()
    if s.empty:
        return np.nan
    return float((1.0 + s).prod() ** (252.0 / len(s)) - 1.0)


def compute_spy_performance_by_regime_timing(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (regime, state), grp in panel.groupby(["macro_regime_confirmed", "spy_timing_state"], observed=False):
        s = grp["spy_daily_return"].dropna()
        rf = grp.loc[s.index, "daily_rf"] if not s.empty else pd.Series(dtype=float)
        ann = _annualized_return(s)
        vol = float(s.std(ddof=1) * np.sqrt(252.0)) if len(s) > 1 else np.nan
        ex = s - rf
        ex_std = ex.std(ddof=1)
        sharpe = float(ex.mean() / ex_std * np.sqrt(252.0)) if pd.notna(ex_std) and ex_std != 0 else np.nan
        # spell-based max drawdown within this regime-timing state
        sub = grp[["date", "spy_daily_return", "macro_regime_confirmed", "spy_timing_state"]].copy()
        spell_change = (sub["macro_regime_confirmed"] != sub["macro_regime_confirmed"].shift(1)) | (sub["spy_timing_state"] != sub["spy_timing_state"].shift(1))
        sub["spell_id"] = spell_change.cumsum()
        spell_dds = []
        spell_weights = []
        for _, spell in sub.groupby("spell_id", observed=False):
            r = spell["spy_daily_return"].dropna()
            if r.empty:
                continue
            wealth = (1.0 + r).cumprod()
            dd = float((wealth / wealth.cummax() - 1.0).min())
            spell_dds.append(dd)
            spell_weights.append(len(r))
        mdd = float(np.average(spell_dds, weights=spell_weights)) if spell_dds else np.nan
        rows.append(
            {
                "macro_regime_confirmed": regime,
                "spy_timing_state": state,
                "n_obs": int(len(s)),
                "annualized_return": ann,
                "annualized_volatility": vol,
                "Sharpe": sharpe,
                "max_drawdown": mdd,
                "positive_day_ratio": float((s > 0).mean()) if not s.empty else np.nan,
                "avg_daily_return": float(s.mean()) if not s.empty else np.nan,
                "worst_day": float(s.min()) if not s.empty else np.nan,
                "best_day": float(s.max()) if not s.empty else np.nan,
                "LOW_SAMPLE": len(s) < CONFIG["low_sample_threshold"],
            }
        )
    return pd.DataFrame(rows)


def _add_regime_background(ax: plt.Axes, panel: pd.DataFrame) -> None:
    work = panel.reset_index(drop=True)
    starts = work["macro_regime_confirmed"] != work["macro_regime_confirmed"].shift(1)
    start_pos = work.index[starts].tolist()
    end_pos = start_pos[1:] + [len(work)]
    for s, e in zip(start_pos, end_pos):
        regime = work.iloc[s]["macro_regime_confirmed"]
        ax.axvspan(work.iloc[s]["date"], work.iloc[e - 1]["date"], color=REGIME_COLORS.get(regime, "#cccccc"), alpha=0.08)


def plot_regime_labeled_equity_curve(panel: pd.DataFrame, sell_events: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1]})
    ax1, ax2, ax3 = axes

    _add_regime_background(ax1, panel)
    _add_regime_background(ax2, panel)

    sell_mask = panel["spy_timing_state"] == "SELL"
    if sell_mask.any():
        change = sell_mask != sell_mask.shift(1)
        starts = panel.index[sell_mask & change.fillna(True)].tolist()
        ends = panel.index[sell_mask & (sell_mask.shift(-1) != sell_mask).fillna(True)].tolist()
        for s, e in zip(starts, ends):
            ax1.axvspan(panel.loc[s, "date"], panel.loc[e, "date"], color="gray", alpha=0.12)
            ax2.axvspan(panel.loc[s, "date"], panel.loc[e, "date"], color="gray", alpha=0.08)

    ax1.plot(panel["date"], panel["SPY_BUY_HOLD_nav"], label="SPY_BUY_HOLD", color="black")
    ax1.plot(panel["date"], panel["MONTHLY_EITHER_CONFIRM_nav"], label="MONTHLY_EITHER_CONFIRM", color="tab:blue")
    ax1.plot(panel["date"], panel["CASH_ONLY_nav"], label="CASH_ONLY", color="tab:green")
    ax1.set_yscale("log")
    ax1.set_title("Regime-Labeled Equity Curve")
    ax1.legend(ncol=3)

    ax2.plot(panel["date"], panel["spy_drawdown_from_previous_high"], color="tab:red", label="SPY drawdown from previous high")
    for thr in CONFIG["drawdown_thresholds"]:
        ax2.axhline(thr, color="gray", linestyle="--", linewidth=0.8)
    for _, row in sell_events.iterrows():
        ax2.axvline(row["sell_start_date"], color=REGIME_COLORS.get(row["macro_regime_at_sell"], "#444444"), alpha=0.7, linewidth=1.0)
    ax2.set_ylim(min(-0.7, panel["spy_drawdown_from_previous_high"].min() * 1.05), 0.02)
    ax2.legend(loc="lower left")

    # timing strip
    y0, y1 = 0.55, 0.95
    for state in TIMING_ORDER:
        mask = panel["spy_timing_state"] == state
        if not mask.any():
            continue
        change = mask != mask.shift(1)
        starts = panel.index[mask & change.fillna(True)].tolist()
        ends = panel.index[mask & (mask.shift(-1) != mask).fillna(True)].tolist()
        for s, e in zip(starts, ends):
            ax3.axvspan(panel.loc[s, "date"], panel.loc[e, "date"], ymin=y0, ymax=y1, color=TIMING_COLORS[state], alpha=0.7)
    # regime strip
    for regime in REGIME_ORDER:
        mask = panel["macro_regime_confirmed"] == regime
        if not mask.any():
            continue
        change = mask != mask.shift(1)
        starts = panel.index[mask & change.fillna(True)].tolist()
        ends = panel.index[mask & (mask.shift(-1) != mask).fillna(True)].tolist()
        for s, e in zip(starts, ends):
            ax3.axvspan(panel.loc[s, "date"], panel.loc[e, "date"], ymin=0.05, ymax=0.45, color=REGIME_COLORS.get(regime, "#cccccc"), alpha=0.7)
    ax3.set_yticks([0.25, 0.75])
    ax3.set_yticklabels(["Macro regime", "Timing state"])
    ax3.set_ylim(0, 1)
    ax3.set_xlabel("Date")
    ax3.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(FIG_EQUITY, dpi=180)
    plt.close(fig)


def plot_sell_lag_scatter(events: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for regime in REGIME_ORDER:
        grp = events.loc[events["macro_regime_at_sell"] == regime]
        if grp.empty:
            continue
        ax.scatter(grp["spy_drawdown_from_previous_high_at_sell"], grp["spy_forward_return_63d"], s=np.clip(grp["sell_spell_length_days"], 20, 300), alpha=0.75, label=regime, color=REGIME_COLORS.get(regime, "#777777"))
    for x in [-0.05, -0.10, -0.15]:
        ax.axvline(x, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("SPY drawdown from previous high at SELL")
    ax.set_ylabel("SPY forward return 63D")
    ax.set_title("SELL Lag Scatter by Regime")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_SCATTER, dpi=180)
    plt.close(fig)


def plot_sell_lag_by_regime(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_df = summary.set_index("macro_regime_at_sell").reindex(REGIME_ORDER).dropna(how="all")
    axes[0].bar(plot_df.index, plot_df["avg_spy_drawdown_from_previous_high_at_sell"], color=[REGIME_COLORS.get(r, "#777777") for r in plot_df.index])
    axes[0].set_title("Avg drawdown at SELL")
    axes[1].bar(plot_df.index, plot_df["avg_spy_forward_return_63d"], color=[REGIME_COLORS.get(r, "#777777") for r in plot_df.index])
    axes[1].set_title("Avg SPY forward return 63D")
    axes[2].bar(plot_df.index, plot_df["pct_likely_sold_near_low"], color=[REGIME_COLORS.get(r, "#777777") for r in plot_df.index])
    axes[2].set_title("Pct likely sold near low")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIG_BAR, dpi=180)
    plt.close(fig)


def plot_heatmaps(perf: pd.DataFrame) -> None:
    sharpe = perf.pivot_table(index="macro_regime_confirmed", columns="spy_timing_state", values="Sharpe", aggfunc="first").reindex(index=REGIME_ORDER, columns=TIMING_ORDER)
    ret = perf.pivot_table(index="macro_regime_confirmed", columns="spy_timing_state", values="annualized_return", aggfunc="first").reindex(index=REGIME_ORDER, columns=TIMING_ORDER)
    dd = perf.pivot_table(index="macro_regime_confirmed", columns="spy_timing_state", values="max_drawdown", aggfunc="first").reindex(index=REGIME_ORDER, columns=TIMING_ORDER)
    sharpe.to_csv(SPY_SHARPE_PIVOT_OUT)
    ret.to_csv(SPY_RETURN_PIVOT_OUT)
    dd.to_csv(SPY_DD_PIVOT_OUT)
    for pivot, path, title, center in [
        (sharpe, FIG_SHARPE, "SPY Sharpe by Regime and Timing State", 0.0),
        (ret, FIG_RETURN, "SPY Annualized Return by Regime and Timing State", 0.0),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=center, ax=ax)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)


def plot_case_studies(panel: pd.DataFrame, sell_events: pd.DataFrame) -> None:
    periods = CONFIG["case_study_periods"]
    fig, axes = plt.subplots(len(periods), 1, figsize=(14, 10), sharex=False)
    if len(periods) == 1:
        axes = [axes]
    for ax, (name, (start, end)) in zip(axes, periods.items()):
        sub = panel.loc[(panel["date"] >= pd.Timestamp(start)) & (panel["date"] <= pd.Timestamp(end))].copy()
        if sub.empty:
            continue
        _add_regime_background(ax, sub)
        ax.plot(sub["date"], sub["spy_price"] / sub["spy_price"].iloc[0], label="SPY normalized", color="black")
        ax.plot(sub["date"], sub["MONTHLY_EITHER_CONFIRM_nav"] / sub["MONTHLY_EITHER_CONFIRM_nav"].iloc[0], label="Monthly Either normalized", color="tab:blue")
        ax2 = ax.twinx()
        ax2.plot(sub["date"], sub["spy_drawdown_from_previous_high"], color="tab:red", alpha=0.5, label="DD prev high")
        sells = sell_events.loc[(sell_events["sell_start_date"] >= pd.Timestamp(start)) & (sell_events["sell_start_date"] <= pd.Timestamp(end))]
        for _, row in sells.iterrows():
            ax.axvline(row["sell_start_date"], color=REGIME_COLORS.get(row["macro_regime_at_sell"], "#444444"), linestyle="--", linewidth=1.0)
        ax.set_title(name)
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_CASE, dpi=180)
    plt.close(fig)


def plot_sell_events_timeline(events: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for regime in REGIME_ORDER:
        grp = events.loc[events["macro_regime_at_sell"] == regime]
        if grp.empty:
            continue
        near_low = grp["likely_sold_near_low"]
        ax.scatter(grp.loc[~near_low, "sell_start_date"], grp.loc[~near_low, "spy_drawdown_from_previous_high_at_sell"], label=regime, color=REGIME_COLORS.get(regime, "#777777"), alpha=0.75)
        ax.scatter(grp.loc[near_low, "sell_start_date"], grp.loc[near_low, "spy_drawdown_from_previous_high_at_sell"], color=REGIME_COLORS.get(regime, "#777777"), marker="x", s=60)
    ax.set_title("SELL Events Timeline")
    ax.set_ylabel("SPY drawdown from previous high at SELL")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_EVENTS, dpi=180)
    plt.close(fig)


def write_markdown_report(panel: pd.DataFrame, spells: pd.DataFrame, sell_events: pd.DataFrame, summary: pd.DataFrame, perf: pd.DataFrame) -> None:
    lines = [
        "# Regime-Labeled Sell-Lag Diagnostic",
        "",
        "## Purpose",
        "",
        "This is a diagnostic layer, not a new strategy. The goal is to evaluate whether `MONTHLY_EITHER_CONFIRM` SELL signals are timely and effective under different macro regimes.",
        "",
        "## Method",
        "",
        "- `HOLD` means Monthly Either wants to hold SPY.",
        "- `SELL` means Monthly Either wants to hold CASH instead of SPY.",
        "- Macro regime comes from the existing daily reconstructed regime panel.",
        "- SPY drawdown is measured from cumulative previous high, not from a rolling n-day high.",
        "- SELL spell performance answers what happens after SELL begins. It does not by itself answer whether SELL was timely.",
        "",
        "## Regime-Labeled Equity Curve",
        "",
        f"Core figure: `{FIG_EQUITY}`",
        "",
        "## Sell-Lag Findings",
        "",
        summary.to_markdown(index=False) if not summary.empty else "No sell events available.",
        "",
        "## Regime-Specific Interpretation",
        "",
        "- `FLAT`: if SELL is followed by strong SPY returns, that points to rebound risk and possible over-selling.",
        "- `INVERTED`: if HOLD and SELL SPY performance are similar, SELL is not a strong risk discriminator.",
        "- `STEEP`: if HOLD strongly dominates SELL, the timing signal is more effective in this regime.",
        "- `HIGH_INFLATION`: interpret cautiously when sample is small.",
        "",
        "## Case Studies",
        "",
        f"Case-study figure: `{FIG_CASE}`",
        "",
        "## Implication for Next Strategy Design",
        "",
        "- Monthly Either should not necessarily map to the same 0/100 equity decision in every regime.",
        "- SELL intensity may need to be regime-conditioned.",
        "- FLAT_SELL may need partial SPY retention or faster recovery logic.",
        "- STEEP_SELL may justify stronger de-risking.",
        "- INVERTED_SELL may justify cash carry with partial SPY rather than a full exit.",
        "- Defensive sleeve design remains important because many corrections still occur while timing state is HOLD.",
        "",
        "## Caveats",
        "",
        "- SELL events can be sparse in some regimes.",
        "- Regime labels may lag market turns.",
        "- Conditional Sharpe does not equal a tradable strategy result.",
        "- A formal backtest is still required for any rule change.",
    ]
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    timing = rebuild_monthly_either_if_needed(load_timing_panel())
    macro = load_macro_regime_panel()
    panel = build_drawdown_features(merge_daily_panel(timing, macro))
    panel["spell_id"] = (panel["spy_timing_state"] != panel["spy_timing_state"].shift(1)).cumsum().astype(int)
    spells = identify_timing_spells(panel)
    sell_events = build_sell_lag_event_table(panel, spells)
    summary = summarize_sell_lag_by_regime(sell_events)
    perf = compute_spy_performance_by_regime_timing(panel)

    panel.to_csv(DAILY_PANEL_OUT, index=False)
    spells.to_csv(SPELLS_OUT, index=False)
    sell_events.to_csv(SELL_EVENTS_OUT, index=False)
    summary.to_csv(SELL_SUMMARY_OUT, index=False)
    perf.to_csv(SPY_STATE_PERF_OUT, index=False)

    plot_regime_labeled_equity_curve(panel, sell_events)
    plot_sell_lag_scatter(sell_events)
    plot_sell_lag_by_regime(summary)
    plot_heatmaps(perf)
    plot_case_studies(panel, sell_events)
    plot_sell_events_timeline(sell_events)
    write_markdown_report(panel, spells, sell_events, summary, perf)

    print(f"1. Total SELL events: {len(sell_events)}")
    print(f"2. Avg SELL drawdown from previous high: {sell_events['spy_drawdown_from_previous_high_at_sell'].mean():.2%}" if not sell_events.empty else "2. Avg SELL drawdown: n/a")
    if not sell_events.empty:
        print(f"3. Pct SELL after -5%/-10%/-15%: {sell_events['sell_after_5pct_drawdown'].mean():.2%} / {sell_events['sell_after_10pct_drawdown'].mean():.2%} / {sell_events['sell_after_15pct_drawdown'].mean():.2%}")
    if not summary.empty:
        laggiest = summary.sort_values("avg_spy_drawdown_from_previous_high_at_sell").iloc[0]
        rebound = summary.sort_values("avg_spy_forward_max_runup_63d", ascending=False).iloc[0]
        further_dd = summary.sort_values("avg_spy_forward_max_drawdown_63d").iloc[0]
        print(f"4. Most lagged SELL regime: {laggiest['macro_regime_at_sell']}")
        print(f"5. Strongest rebound after SELL: {rebound['macro_regime_at_sell']}")
        print(f"6. Most further downside after SELL: {further_dd['macro_regime_at_sell']}")
        flat = summary.loc[summary['macro_regime_at_sell'] == 'FLAT']
        steep = summary.loc[summary['macro_regime_at_sell'] == 'STEEP']
        print(f"7. FLAT_SELL shows rebound risk: {bool((not flat.empty) and (flat['avg_spy_forward_return_63d'].iloc[0] > 0) and (flat['pct_likely_sold_near_low'].iloc[0] >= 0.25))}")
        print(f"8. STEEP_SELL shows timing effectiveness: {bool((not steep.empty) and (steep['avg_spy_forward_return_63d'].iloc[0] <= 0) and (steep['avg_spy_forward_max_drawdown_63d'].iloc[0] < -0.05))}")
    inv_perf = perf.loc[perf["macro_regime_confirmed"] == "INVERTED"]
    if len(inv_perf) >= 2:
        hold = inv_perf.loc[inv_perf["spy_timing_state"] == "HOLD", "annualized_return"]
        sell = inv_perf.loc[inv_perf["spy_timing_state"] == "SELL", "annualized_return"]
        diff_small = bool((not hold.empty) and (not sell.empty) and abs(float(hold.iloc[0] - sell.iloc[0])) < 0.05)
        print(f"9. INVERTED_HOLD vs INVERTED_SELL difference small: {diff_small}")
    print(f"10. Saved outputs: {RESULTS_DIR} and {FIGURES_DIR}")


if __name__ == "__main__":
    main()
