from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]

INPUT_CANDIDATES = [
    ROOT / "results" / "regime_hedge_steep_sell_ief" / "daily_backtest_panel.csv",
    ROOT / "results" / "regime_labeled_sell_lag_diagnostic" / "regime_labeled_daily_panel.csv",
    ROOT / "results" / "high_frequency_regime_diagnostics" / "high_frequency_regime_feature_panel.csv",
    ROOT / "results" / "spy_cash_timing_frequency_test" / "daily_backtest_panel.csv",
]
RECON_PANEL = ROOT / "results" / "reconstructed_regime_asset_behavior" / "reconstructed_regime_panel.csv"
RULE_PANEL = ROOT / "results" / "rule_diagnostics" / "rule_state_panel.csv"
RAW_VIX = ROOT / "data" / "raw" / "macro" / "volatility" / "VIXCLS.csv"

RESULTS_DIR = ROOT / "results" / "vix_zscore_crisis_detector_diagnostic"
FIGURES_DIR = ROOT / "figures" / "vix_zscore_crisis_detector_diagnostic"

EVENT_OUT = RESULTS_DIR / "vix_zscore_event_table.csv"
GRID_OUT = RESULTS_DIR / "vix_zscore_grid_summary.csv"
REGIME_GRID_OUT = RESULTS_DIR / "vix_zscore_grid_summary_by_regime.csv"
COVID_OUT = RESULTS_DIR / "covid_trigger_timing_by_param.csv"
RANK_OUT = RESULTS_DIR / "vix_zscore_param_ranking.csv"
REPORT_OUT = RESULTS_DIR / "VIX_ZSCORE_CRISIS_DETECTOR_DIAGNOSTIC.md"
PANEL_OUT = RESULTS_DIR / "vix_zscore_daily_panel.csv"

FIG_MDD21 = FIGURES_DIR / "vix_zscore_grid_heatmap_mdd21.png"
FIG_PROB = FIGURES_DIR / "vix_zscore_grid_heatmap_mdd_probability.png"
FIG_FALSE = FIGURES_DIR / "vix_zscore_grid_false_alarm_heatmap.png"
FIG_SCATTER = FIGURES_DIR / "vix_zscore_param_scatter.png"
FIG_TIMELINE = FIGURES_DIR / "vix_zscore_timeline_top_params.png"
FIG_COVID = FIGURES_DIR / "covid_vix_zscore_case_study.png"
FIG_REGIME = FIGURES_DIR / "by_regime_top_param_bar.png"

CONFIG = {
    "rolling_windows": [20, 40, 60, 90, 120],
    "z_thresholds": [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
    "cooldown_days_list": [10, 21],
    "forward_windows": [5, 10, 21, 42, 63],
    "covid_start": "2020-02-19",
    "covid_end": "2020-04-30",
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
    vix_col = _first_col(df, ["VIXCLS", "VIX_LEVEL", "vix"])
    if date_col is None or vix_col is None:
        return None
    out = df[[date_col, vix_col]].rename(columns={date_col: "date", vix_col: "VIX_LEVEL"})
    out["date"] = pd.to_datetime(out["date"])
    out["VIX_LEVEL"] = pd.to_numeric(out["VIX_LEVEL"].replace(".", np.nan), errors="coerce")
    return out.dropna(subset=["date"]).sort_values("date")


def _rebuild_macro_regime(df: pd.DataFrame) -> pd.Series:
    credit = pd.to_numeric(df.get("CREDIT_SPREAD_BAA_AAA"), errors="coerce")
    dgs1 = pd.to_numeric(df.get("DGS1"), errors="coerce")
    term = pd.to_numeric(df.get("TERM_SPREAD_10Y_1Y"), errors="coerce")
    return pd.Series(
        np.select(
            [
                (credit > 1.5) & (dgs1 > 5.0),
                term < 0.0,
                (term >= 0.0) & (term < 1.0),
                term >= 1.0,
            ],
            ["HIGH_INFLATION", "INVERTED", "FLAT", "STEEP"],
            default="NEUTRAL",
        ),
        index=df.index,
    )


def load_data() -> tuple[pd.DataFrame, str]:
    source = ""
    panel = None
    for path in INPUT_CANDIDATES:
        df = _read_panel(path)
        if df is None or df.empty:
            continue
        if _first_col(df, ["spy_price", "SPY_PRICE", "SPY_ADJ_CLOSE"]) is not None or _first_col(df, ["SPY_RETURN", "SPY_ret", "spy_daily_return"]) is not None:
            panel = df
            source = str(path.relative_to(ROOT))
            break
    if panel is None:
        raise FileNotFoundError("No usable daily SPY panel found.")

    spy_price_col = _first_col(panel, ["spy_price", "SPY_PRICE", "SPY_ADJ_CLOSE", "SPY"])
    spy_ret_col = _first_col(panel, ["SPY_RETURN", "SPY_ret", "spy_daily_return", "SPY_BUY_HOLD_return"])
    if spy_price_col is not None:
        panel["spy_price"] = pd.to_numeric(panel[spy_price_col], errors="coerce")
        panel["spy_daily_return"] = panel["spy_price"].pct_change()
        if spy_ret_col is not None:
            panel["spy_daily_return"] = pd.to_numeric(panel[spy_ret_col], errors="coerce").combine_first(panel["spy_daily_return"])
    elif spy_ret_col is not None:
        panel["spy_daily_return"] = pd.to_numeric(panel[spy_ret_col], errors="coerce")
        panel["spy_price"] = (1.0 + panel["spy_daily_return"].fillna(0.0)).cumprod()
    else:
        raise ValueError("Missing SPY price or return column.")

    vix_col = _first_col(panel, ["VIX_LEVEL", "VIXCLS", "vix_level"])
    if vix_col is not None:
        panel["VIX_LEVEL"] = pd.to_numeric(panel[vix_col], errors="coerce")
    else:
        vix_sources = []
        for path in [RECON_PANEL, RULE_PANEL]:
            vix_panel = _read_panel(path)
            if vix_panel is not None and "VIX_LEVEL" in vix_panel.columns:
                vix_sources.append(vix_panel[["date", "VIX_LEVEL"]].copy())
        raw = _load_raw_vix()
        if raw is not None:
            vix_sources.append(raw)
        if not vix_sources:
            raise ValueError("Missing VIX_LEVEL and no raw VIXCLS.csv could be loaded.")
        vix = pd.concat(vix_sources, ignore_index=True).sort_values("date").drop_duplicates("date")
        panel = panel.merge(vix, on="date", how="left")

    if "macro_regime_confirmed" not in panel.columns:
        for path in [RECON_PANEL, RULE_PANEL]:
            reg = _read_panel(path)
            if reg is None:
                continue
            keep_cols = ["date"] + [c for c in ["macro_regime_confirmed", "CREDIT_SPREAD_BAA_AAA", "DGS1", "TERM_SPREAD_10Y_1Y"] if c in reg.columns]
            reg = reg[keep_cols].copy()
            if "macro_regime_confirmed" not in reg.columns and {"CREDIT_SPREAD_BAA_AAA", "DGS1", "TERM_SPREAD_10Y_1Y"}.issubset(reg.columns):
                reg["macro_regime_confirmed"] = _rebuild_macro_regime(reg)
            if "macro_regime_confirmed" in reg.columns:
                panel = panel.merge(reg[["date", "macro_regime_confirmed"]], on="date", how="left")
                break
    if "macro_regime_confirmed" not in panel.columns:
        warnings.warn("macro_regime_confirmed unavailable; by-regime outputs will be skipped.")

    if "monthly_either_state" not in panel.columns:
        w_col = _first_col(panel, ["monthly_either_weight_spy", "MONTHLY_EITHER_CONFIRM_weight_spy"])
        if w_col is not None:
            panel["monthly_either_state"] = np.where(pd.to_numeric(panel[w_col], errors="coerce") >= 0.5, "HOLD", "SELL")

    panel = panel.dropna(subset=["spy_price", "VIX_LEVEL"]).sort_values("date").reset_index(drop=True)
    panel["spy_daily_return"] = panel["spy_daily_return"].fillna(panel["spy_price"].pct_change())
    panel["previous_high"] = panel["spy_price"].cummax()
    panel["spy_drawdown_from_previous_high"] = panel["spy_price"] / panel["previous_high"] - 1.0
    for h in [5, 10, 21]:
        panel[f"spy_ret_{h}d"] = panel["spy_price"] / panel["spy_price"].shift(h) - 1.0
    return panel, source


def build_vix_zscores(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    for window in CONFIG["rolling_windows"]:
        mean = out["VIX_LEVEL"].rolling(window, min_periods=window).mean()
        std = out["VIX_LEVEL"].rolling(window, min_periods=window).std(ddof=1)
        out[f"vix_zscore_{window}"] = (out["VIX_LEVEL"] - mean) / std.replace(0.0, np.nan)
    return out


def _forward_path_metrics(prices: pd.Series, idx: int, horizon: int) -> dict[str, float]:
    if idx >= len(prices) - 1:
        return {
            f"forward_return_{horizon}d": np.nan,
            f"forward_max_drawdown_{horizon}d": np.nan,
            f"forward_max_runup_{horizon}d": np.nan,
            f"forward_min_return_from_event_price_{horizon}d": np.nan,
            f"days_to_trough_{horizon}d": np.nan,
            f"days_to_peak_{horizon}d": np.nan,
        }
    end = min(idx + horizon, len(prices) - 1)
    path = prices.iloc[idx : end + 1].astype(float)
    base = float(path.iloc[0])
    if len(path) < 2 or base <= 0:
        ret = mdd = runup = min_ret = trough_days = peak_days = np.nan
    else:
        ret = float(path.iloc[-1] / base - 1.0) if end == idx + horizon else np.nan
        rel = path / base - 1.0
        min_ret = float(rel.min())
        runup = float(rel.max())
        wealth = path / base
        dd = wealth / wealth.cummax() - 1.0
        mdd = float(dd.min())
        trough_days = int(np.argmin(path.to_numpy()))
        peak_days = int(np.argmax(path.to_numpy()))
    return {
        f"forward_return_{horizon}d": ret,
        f"forward_max_drawdown_{horizon}d": mdd,
        f"forward_max_runup_{horizon}d": runup,
        f"forward_min_return_from_event_price_{horizon}d": min_ret,
        f"days_to_trough_{horizon}d": trough_days,
        f"days_to_peak_{horizon}d": peak_days,
    }


def extract_events(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    prices = panel["spy_price"].reset_index(drop=True)
    for window in CONFIG["rolling_windows"]:
        z_col = f"vix_zscore_{window}"
        for threshold in CONFIG["z_thresholds"]:
            trigger = panel[z_col] >= threshold
            for cooldown in CONFIG["cooldown_days_list"]:
                armed = True
                cooldown_until = -1
                for i, is_triggered in enumerate(trigger.fillna(False).to_numpy()):
                    if not is_triggered and i > cooldown_until:
                        armed = True
                    if is_triggered and armed and i > cooldown_until:
                        row = panel.iloc[i]
                        event = {
                            "event_date": row["date"],
                            "row_index": i,
                            "rolling_window": window,
                            "z_threshold": threshold,
                            "cooldown_days": cooldown,
                            "vix_level": float(row["VIX_LEVEL"]),
                            "vix_zscore": float(row[z_col]),
                            "spy_price": float(row["spy_price"]),
                            "spy_drawdown_from_previous_high": float(row["spy_drawdown_from_previous_high"]),
                        }
                        if "macro_regime_confirmed" in panel.columns:
                            event["macro_regime_confirmed"] = row["macro_regime_confirmed"]
                        if "monthly_either_state" in panel.columns:
                            event["monthly_either_state"] = row["monthly_either_state"]
                        for horizon in CONFIG["forward_windows"]:
                            event.update(_forward_path_metrics(prices, i, horizon))
                        rows.append(event)
                        armed = False
                        cooldown_until = i + cooldown
    events = pd.DataFrame(rows)
    if events.empty:
        return events
    events["mdd_10d_below_3"] = events["forward_max_drawdown_10d"] <= -0.03
    events["mdd_21d_below_5"] = events["forward_max_drawdown_21d"] <= -0.05
    events["mdd_21d_below_10"] = events["forward_max_drawdown_21d"] <= -0.10
    events["mdd_63d_below_10"] = events["forward_max_drawdown_63d"] <= -0.10
    events["false_alarm_21d"] = events["forward_max_drawdown_21d"] > -0.03
    return events.sort_values(["rolling_window", "z_threshold", "cooldown_days", "event_date"]).reset_index(drop=True)


def compute_forward_outcomes(events: pd.DataFrame) -> pd.DataFrame:
    return events


def _summarize_group(grp: pd.DataFrame, total_years: float) -> dict[str, float]:
    return {
        "event_count": int(len(grp)),
        "events_per_year": float(len(grp) / total_years) if total_years > 0 else np.nan,
        "avg_forward_return_10d": float(grp["forward_return_10d"].mean()),
        "avg_forward_return_21d": float(grp["forward_return_21d"].mean()),
        "avg_forward_return_63d": float(grp["forward_return_63d"].mean()),
        "avg_forward_mdd_10d": float(grp["forward_max_drawdown_10d"].mean()),
        "avg_forward_mdd_21d": float(grp["forward_max_drawdown_21d"].mean()),
        "avg_forward_mdd_63d": float(grp["forward_max_drawdown_63d"].mean()),
        "median_forward_mdd_21d": float(grp["forward_max_drawdown_21d"].median()),
        "pct_mdd_10d_below_3": float(grp["mdd_10d_below_3"].mean()),
        "pct_mdd_21d_below_5": float(grp["mdd_21d_below_5"].mean()),
        "pct_mdd_21d_below_10": float(grp["mdd_21d_below_10"].mean()),
        "pct_mdd_63d_below_10": float(grp["mdd_63d_below_10"].mean()),
        "false_alarm_rate_21d": float(grp["false_alarm_21d"].mean()),
        "avg_days_to_trough_21d": float(grp["days_to_trough_21d"].mean()),
        "avg_days_to_trough_63d": float(grp["days_to_trough_63d"].mean()),
        "median_vix_level_at_trigger": float(grp["vix_level"].median()),
        "median_vix_zscore_at_trigger": float(grp["vix_zscore"].median()),
        "median_spy_drawdown_at_trigger": float(grp["spy_drawdown_from_previous_high"].median()),
    }


def summarize_grid(events: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    total_years = (panel["date"].iloc[-1] - panel["date"].iloc[0]).days / 365.25
    rows = []
    for keys, grp in events.groupby(["rolling_window", "z_threshold", "cooldown_days"]):
        rows.append({"rolling_window": keys[0], "z_threshold": keys[1], "cooldown_days": keys[2], **_summarize_group(grp, total_years)})
    return pd.DataFrame(rows)


def summarize_by_regime(events: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    if "macro_regime_confirmed" not in events.columns:
        return pd.DataFrame()
    total_years = (panel["date"].iloc[-1] - panel["date"].iloc[0]).days / 365.25
    rows = []
    for keys, grp in events.groupby(["rolling_window", "z_threshold", "cooldown_days", "macro_regime_confirmed"], dropna=False):
        rows.append(
            {
                "rolling_window": keys[0],
                "z_threshold": keys[1],
                "cooldown_days": keys[2],
                "macro_regime_confirmed": keys[3],
                **_summarize_group(grp, total_years),
            }
        )
    return pd.DataFrame(rows)


def analyze_covid_case(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    start = pd.Timestamp(CONFIG["covid_start"])
    end = pd.Timestamp(CONFIG["covid_end"])
    rows = []
    for keys, grp in events.groupby(["rolling_window", "z_threshold", "cooldown_days"]):
        sub = grp.loc[(grp["event_date"] >= start) & (grp["event_date"] <= end)].sort_values("event_date")
        if sub.empty:
            rows.append(
                {
                    "rolling_window": keys[0],
                    "z_threshold": keys[1],
                    "cooldown_days": keys[2],
                    "first_trigger_date_in_covid": pd.NaT,
                }
            )
            continue
        ev = sub.iloc[0]
        rows.append(
            {
                "rolling_window": keys[0],
                "z_threshold": keys[1],
                "cooldown_days": keys[2],
                "first_trigger_date_in_covid": ev["event_date"],
                "days_after_covid_start": int((ev["event_date"] - start).days),
                "SPY_drawdown_at_first_trigger": ev["spy_drawdown_from_previous_high"],
                "VIX_level_at_first_trigger": ev["vix_level"],
                "VIX_zscore_at_first_trigger": ev["vix_zscore"],
                "forward_10d_return": ev["forward_return_10d"],
                "forward_10d_mdd": ev["forward_max_drawdown_10d"],
                "forward_21d_return": ev["forward_return_21d"],
                "forward_21d_mdd": ev["forward_max_drawdown_21d"],
            }
        )
    return pd.DataFrame(rows)


def _minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def rank_parameters(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["eligible"] = (out["event_count"] >= 10) & (out["events_per_year"] <= 6) & (out["false_alarm_rate_21d"] <= 0.70)
    eligible = out.loc[out["eligible"]].copy()
    if eligible.empty:
        return out.assign(composite_score=np.nan).sort_values(["eligible", "event_count"], ascending=[False, False])
    eligible["score_pct_mdd_21d_below_5"] = _minmax(eligible["pct_mdd_21d_below_5"])
    eligible["score_pct_mdd_63d_below_10"] = _minmax(eligible["pct_mdd_63d_below_10"])
    eligible["score_abs_avg_mdd_21d"] = _minmax(-eligible["avg_forward_mdd_21d"])
    eligible["score_abs_avg_mdd_63d"] = _minmax(-eligible["avg_forward_mdd_63d"])
    eligible["score_false_alarm"] = _minmax(eligible["false_alarm_rate_21d"])
    eligible["composite_score"] = (
        0.30 * eligible["score_pct_mdd_21d_below_5"]
        + 0.20 * eligible["score_pct_mdd_63d_below_10"]
        + 0.20 * eligible["score_abs_avg_mdd_21d"]
        + 0.15 * eligible["score_abs_avg_mdd_63d"]
        - 0.15 * eligible["score_false_alarm"]
    )
    out = out.merge(
        eligible[
            [
                "rolling_window",
                "z_threshold",
                "cooldown_days",
                "composite_score",
                "score_pct_mdd_21d_below_5",
                "score_pct_mdd_63d_below_10",
                "score_abs_avg_mdd_21d",
                "score_abs_avg_mdd_63d",
                "score_false_alarm",
            ]
        ],
        on=["rolling_window", "z_threshold", "cooldown_days"],
        how="left",
    )
    return out.sort_values(["eligible", "composite_score"], ascending=[False, False])


def _top_params(ranking: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    top = ranking.loc[ranking["eligible"]].sort_values("composite_score", ascending=False).head(n)
    if top.empty:
        top = ranking.sort_values("event_count", ascending=False).head(n)
    return top


def plot_heatmaps(summary: pd.DataFrame) -> None:
    main = summary.loc[summary["cooldown_days"] == 21].copy()
    for value, path, title, cmap in [
        ("avg_forward_mdd_21d", FIG_MDD21, "Average Forward MDD 21D", "RdYlGn_r"),
        ("pct_mdd_21d_below_5", FIG_PROB, "P(21D MDD <= -5%)", "Reds"),
        ("false_alarm_rate_21d", FIG_FALSE, "False Alarm Rate 21D", "Blues"),
    ]:
        piv = main.pivot(index="rolling_window", columns="z_threshold", values=value)
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.heatmap(piv, annot=True, fmt=".2f", cmap=cmap, ax=ax)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)


def plot_scatter(summary: pd.DataFrame, ranking: pd.DataFrame) -> None:
    main = summary.loc[summary["cooldown_days"] == 21].copy()
    fig, ax = plt.subplots(figsize=(10, 7))
    sizes = 30 + main["event_count"] * 5
    sc = ax.scatter(
        main["false_alarm_rate_21d"],
        main["pct_mdd_21d_below_5"],
        s=sizes,
        c=main["avg_forward_mdd_21d"],
        cmap="RdYlGn_r",
        alpha=0.75,
    )
    top = _top_params(ranking.loc[ranking["cooldown_days"] == 21], 10)
    for _, row in top.iterrows():
        ax.text(row["false_alarm_rate_21d"], row["pct_mdd_21d_below_5"], f"{int(row['rolling_window'])}/{row['z_threshold']}", fontsize=8)
    ax.set_xlabel("False alarm rate 21D")
    ax.set_ylabel("P(21D MDD <= -5%)")
    ax.set_title("VIX Z-score Parameter Tradeoff")
    fig.colorbar(sc, ax=ax, label="Avg forward MDD 21D")
    fig.tight_layout()
    fig.savefig(FIG_SCATTER, dpi=180)
    plt.close(fig)


def _shade_regimes(ax: plt.Axes, panel: pd.DataFrame) -> None:
    if "macro_regime_confirmed" not in panel.columns:
        return
    tmp = panel.reset_index(drop=True)
    starts = tmp["macro_regime_confirmed"] != tmp["macro_regime_confirmed"].shift(1)
    positions = tmp.index[starts].tolist()
    ends = positions[1:] + [len(tmp)]
    for s, e in zip(positions, ends):
        regime = tmp.iloc[s]["macro_regime_confirmed"]
        ax.axvspan(tmp.iloc[s]["date"], tmp.iloc[e - 1]["date"], color=REGIME_COLORS.get(regime, "#cccccc"), alpha=0.08)


def plot_timeline(panel: pd.DataFrame, events: pd.DataFrame, ranking: pd.DataFrame) -> None:
    top = _top_params(ranking.loc[ranking["cooldown_days"] == 21], 3)
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True, gridspec_kw={"height_ratios": [2, 1.5, 1.5, 2]})
    ax1, ax2, ax3, ax4 = axes
    _shade_regimes(ax1, panel)
    ax1.plot(panel["date"], panel["spy_price"] / panel["spy_price"].iloc[0], color="black", label="SPY normalized")
    ax1.legend()
    ax1.set_title("Top VIX Z-score Trigger Timeline")
    ax2.plot(panel["date"], panel["spy_drawdown_from_previous_high"], color="tab:red")
    for thr in [-0.05, -0.10, -0.20]:
        ax2.axhline(thr, color="gray", linestyle="--", linewidth=0.8)
    ax3.plot(panel["date"], panel["VIX_LEVEL"], color="tab:purple", label="VIX")
    ax3.legend()
    for _, row in top.iterrows():
        w = int(row["rolling_window"])
        col = f"vix_zscore_{w}"
        label = f"{w}d z>={row['z_threshold']}"
        ax4.plot(panel["date"], panel[col], label=label)
        ev = events.loc[
            (events["rolling_window"] == row["rolling_window"])
            & (events["z_threshold"] == row["z_threshold"])
            & (events["cooldown_days"] == row["cooldown_days"])
        ]
        for dt in ev["event_date"]:
            ax4.axvline(dt, color="gray", alpha=0.12)
    ax4.axhline(0, color="black", linewidth=0.8)
    ax4.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(FIG_TIMELINE, dpi=180)
    plt.close(fig)


def plot_covid_case(panel: pd.DataFrame, events: pd.DataFrame, ranking: pd.DataFrame) -> None:
    start = pd.Timestamp(CONFIG["covid_start"])
    end = pd.Timestamp(CONFIG["covid_end"])
    sub = panel.loc[panel["date"].between(start, end)].copy()
    if sub.empty:
        return
    top = _top_params(ranking.loc[ranking["cooldown_days"] == 21], 3)
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    ax1, ax2, ax3, ax4 = axes
    ax1.plot(sub["date"], sub["spy_price"] / sub["spy_price"].iloc[0], color="black")
    ax1.set_title("COVID VIX Z-score Case Study")
    ax2.plot(sub["date"], sub["VIX_LEVEL"], color="tab:purple")
    ax2.set_ylabel("VIX")
    ax3.plot(sub["date"], sub["spy_drawdown_from_previous_high"], color="tab:red")
    for _, row in top.iterrows():
        w = int(row["rolling_window"])
        ax4.plot(sub["date"], sub[f"vix_zscore_{w}"], label=f"{w}d z")
        ev = events.loc[
            (events["rolling_window"] == row["rolling_window"])
            & (events["z_threshold"] == row["z_threshold"])
            & (events["cooldown_days"] == row["cooldown_days"])
            & (events["event_date"].between(start, end))
        ]
        for dt in ev["event_date"]:
            for ax in axes:
                ax.axvline(dt, color="tab:orange", alpha=0.25)
    if "macro_regime_confirmed" in sub.columns:
        for regime, color in REGIME_COLORS.items():
            mask = sub["macro_regime_confirmed"].eq(regime).to_numpy()
            ax1.fill_between(sub["date"], 0, 0.03, where=mask, transform=ax1.get_xaxis_transform(), color=color, alpha=0.5)
    ax4.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(FIG_COVID, dpi=180)
    plt.close(fig)


def plot_by_regime(regime_summary: pd.DataFrame, ranking: pd.DataFrame) -> None:
    if regime_summary.empty:
        return
    top = _top_params(ranking.loc[ranking["cooldown_days"] == 21], 3)
    top_keys = set(zip(top["rolling_window"], top["z_threshold"], top["cooldown_days"]))
    sub = regime_summary.loc[
        regime_summary.apply(lambda r: (r["rolling_window"], r["z_threshold"], r["cooldown_days"]) in top_keys, axis=1)
    ].copy()
    if sub.empty:
        return
    sub["param"] = sub["rolling_window"].astype(int).astype(str) + "d z>=" + sub["z_threshold"].astype(str)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=sub, x="macro_regime_confirmed", y="pct_mdd_21d_below_5", hue="param", ax=axes[0])
    axes[0].set_title("P(21D MDD <= -5%) by Regime")
    axes[0].tick_params(axis="x", rotation=25)
    sns.barplot(data=sub, x="macro_regime_confirmed", y="false_alarm_rate_21d", hue="param", ax=axes[1])
    axes[1].set_title("False Alarm Rate by Regime")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIG_REGIME, dpi=180)
    plt.close(fig)


def write_markdown_report(source: str, summary: pd.DataFrame, ranking: pd.DataFrame, regime_summary: pd.DataFrame, covid: pd.DataFrame) -> None:
    top10 = ranking.loc[ranking["eligible"]].head(10)
    if top10.empty:
        top10 = ranking.head(10)
    top_table = top10[
        [
            "rolling_window",
            "z_threshold",
            "cooldown_days",
            "event_count",
            "events_per_year",
            "pct_mdd_21d_below_5",
            "pct_mdd_63d_below_10",
            "false_alarm_rate_21d",
            "avg_forward_mdd_21d",
            "composite_score",
        ]
    ].to_markdown(index=False)
    lines = [
        "# VIX Z-score Crisis Detector Diagnostic",
        "",
        "## Purpose",
        "",
        "This is not a strategy backtest. It tests whether rolling VIX z-score events identify periods with worse future SPY drawdowns.",
        "",
        "## Method",
        "",
        f"- Input source: `{source}`.",
        f"- Rolling windows: {CONFIG['rolling_windows']}.",
        f"- Z thresholds: {CONFIG['z_thresholds']}.",
        f"- Cooldowns: {CONFIG['cooldown_days_list']}.",
        "- Events are first False-to-True threshold crossings with cooldown and re-arming.",
        "- Forward drawdown is measured from the event date path, not from a rolling high.",
        "",
        "## Full-Sample Findings",
        "",
        top_table,
        "",
        f"![MDD heatmap](../../figures/vix_zscore_crisis_detector_diagnostic/{FIG_MDD21.name})",
        "",
        f"![Probability heatmap](../../figures/vix_zscore_crisis_detector_diagnostic/{FIG_PROB.name})",
        "",
        f"![False alarm heatmap](../../figures/vix_zscore_crisis_detector_diagnostic/{FIG_FALSE.name})",
        "",
        f"![Scatter](../../figures/vix_zscore_crisis_detector_diagnostic/{FIG_SCATTER.name})",
        "",
        "## Regime-Conditioned Findings",
        "",
        "- See `vix_zscore_grid_summary_by_regime.csv` for per-regime event quality.",
        "- The key use case is whether FLAT stress episodes can be separated from ordinary FLAT behavior.",
        "",
        f"![By regime](../../figures/vix_zscore_crisis_detector_diagnostic/{FIG_REGIME.name})",
        "",
        "## COVID Case Study",
        "",
        "- See `covid_trigger_timing_by_param.csv` for first trigger timing in COVID.",
        "- This should be compared against Monthly Either SELL timing before converting any trigger into an overlay.",
        "",
        f"![COVID case](../../figures/vix_zscore_crisis_detector_diagnostic/{FIG_COVID.name})",
        "",
        "## Interpretation",
        "",
        "- VIX z-score is a relative fast-stress detector and can catch abrupt changes even when macro curve regimes still look benign.",
        "- It is a candidate for a FLAT_STRESS or GLOBAL_STRESS override, not a complete allocation rule.",
        "- Recovery rules need a separate diagnostic because VIX spikes often occur close to market lows.",
        "",
        "## Caveats",
        "",
        "- VIX spikes do not guarantee further downside.",
        "- Grid search can overfit.",
        "- COVID is one event.",
        "- Follow-up should test 1-2 simple parameters out of sample inside the strategy framework.",
        "",
        "## Next Step",
        "",
        "- Carry the top one or two stable VIX z-score triggers into a crash overlay backtest.",
        "- Test whether they improve FLAT + CMDTY path risk without creating excessive whipsaw.",
    ]
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel, source = load_data()
    panel = build_vix_zscores(panel)
    panel.to_csv(PANEL_OUT, index=False)

    events = compute_forward_outcomes(extract_events(panel))
    events.to_csv(EVENT_OUT, index=False)
    summary = summarize_grid(events, panel)
    summary.to_csv(GRID_OUT, index=False)
    regime_summary = summarize_by_regime(events, panel)
    regime_summary.to_csv(REGIME_GRID_OUT, index=False)
    covid = analyze_covid_case(panel, events)
    covid.to_csv(COVID_OUT, index=False)
    ranking = rank_parameters(summary)
    ranking.to_csv(RANK_OUT, index=False)

    plot_heatmaps(summary)
    plot_scatter(summary, ranking)
    plot_timeline(panel, events, ranking)
    plot_covid_case(panel, events, ranking)
    plot_by_regime(regime_summary, ranking)
    write_markdown_report(source, summary, ranking, regime_summary, covid)

    top = ranking.loc[ranking["eligible"]].head(5)
    if top.empty:
        top = ranking.head(5)
    best = top.iloc[0]
    covid_best = covid.loc[
        (covid["rolling_window"] == best["rolling_window"])
        & (covid["z_threshold"] == best["z_threshold"])
        & (covid["cooldown_days"] == best["cooldown_days"])
    ]
    flat_eff = np.nan
    if not regime_summary.empty:
        flat_row = regime_summary.loc[
            (regime_summary["rolling_window"] == best["rolling_window"])
            & (regime_summary["z_threshold"] == best["z_threshold"])
            & (regime_summary["cooldown_days"] == best["cooldown_days"])
            & (regime_summary["macro_regime_confirmed"] == "FLAT")
        ]
        if not flat_row.empty:
            flat_eff = flat_row["pct_mdd_21d_below_5"].iloc[0]

    print(f"1. Sample: {panel['date'].iloc[0].date()} to {panel['date'].iloc[-1].date()}")
    print("2. Top 5 parameter combinations:")
    print(top[["rolling_window", "z_threshold", "cooldown_days", "event_count", "false_alarm_rate_21d", "pct_mdd_21d_below_5", "composite_score"]].to_string(index=False))
    print(f"3. Top parameter event count / false alarm / pct 21d MDD <= -5%: {int(best['event_count'])} / {best['false_alarm_rate_21d']:.2%} / {best['pct_mdd_21d_below_5']:.2%}")
    if not covid_best.empty and pd.notna(covid_best["first_trigger_date_in_covid"].iloc[0]):
        print(f"4. Top parameter first COVID trigger: {pd.Timestamp(covid_best['first_trigger_date_in_covid'].iloc[0]).date()}")
        print(f"5. SPY drawdown at COVID trigger: {covid_best['SPY_drawdown_at_first_trigger'].iloc[0]:.2%}")
    else:
        print("4. Top parameter first COVID trigger: none")
        print("5. SPY drawdown at COVID trigger: n/a")
    print(f"6. FLAT regime pct 21d MDD <= -5% for top parameter: {flat_eff:.2%}" if pd.notna(flat_eff) else "6. FLAT regime effectiveness unavailable.")
    rec = top.head(2)[["rolling_window", "z_threshold", "cooldown_days"]]
    print("7. Recommended parameters for next overlay test:")
    print(rec.to_string(index=False))
    print(f"8. Saved outputs: {RESULTS_DIR} and {FIGURES_DIR}")


if __name__ == "__main__":
    main()
