from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "commodity_stress_trigger_diagnostic"
FIGURE_DIR = ROOT / "figures" / "commodity_stress_trigger_diagnostic"

BASE_CANDIDATES = [
    ROOT / "results" / "spy_cash_stress_recovery_with_credit" / "daily_backtest_panel.csv",
    ROOT / "results" / "spy_cash_stress_recovery_timing" / "daily_backtest_panel.csv",
]
COMMODITY_SOURCES = [
    ROOT / "data" / "raw" / "asset" / "CMDTY_FUT.csv",
    ROOT / "data" / "raw" / "macro" / "commodity" / "CMDTY_FUT.csv",
    ROOT / "results" / "reconstructed_regime_asset_behavior" / "reconstructed_regime_panel.csv",
    ROOT / "results" / "regime_hedge_steep_sell_ief" / "daily_backtest_panel.csv",
]
MISSED_EPISODES = ROOT / "results" / "missed_drawdown_episode_diagnostic" / "drawdown_episode_table.csv"

CONFIG = {
    "output_dir": str(OUTPUT_DIR),
    "figure_dir": str(FIGURE_DIR),
    "cooldown_days": 21,
    "forward_windows": [5, 10, 21, 42, 63],
    "dd_thresholds": [-0.05, -0.08],
    "commodity_ma_windows": [20, 60, 120],
    "commodity_ret_windows": [20, 60, 120],
    "missed_episode_window": [-5, 10],
    "case_study_windows": {
        "2015_2016": ["2015-05-01", "2016-03-31"],
        "2018Q4": ["2018-10-01", "2019-01-31"],
        "COVID_2020": ["2020-02-01", "2020-06-30"],
        "2022": ["2021-11-01", "2023-03-31"],
        "2023": ["2023-07-01", "2023-11-30"],
    },
}

DAILY_PANEL_OUT = OUTPUT_DIR / "commodity_trigger_daily_panel.csv"
EVENT_TABLE = OUTPUT_DIR / "commodity_trigger_event_table.csv"
SUMMARY_TABLE = OUTPUT_DIR / "commodity_trigger_summary.csv"
SUMMARY_BY_REGIME = OUTPUT_DIR / "commodity_trigger_summary_by_regime.csv"
COVERAGE_TABLE = OUTPUT_DIR / "commodity_trigger_coverage_on_missed_drawdowns.csv"
COVERAGE_SUMMARY = OUTPUT_DIR / "commodity_trigger_missed_episode_coverage_summary.csv"
RANKING_TABLE = OUTPUT_DIR / "commodity_trigger_ranking.csv"
REPORT = OUTPUT_DIR / "COMMODITY_STRESS_TRIGGER_DIAGNOSTIC.md"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        warnings.warn(f"Could not read {path}: {exc}")
        return None
    date_col = "date" if "date" in df.columns else "observation_date" if "observation_date" in df.columns else "DATE" if "DATE" in df.columns else None
    if date_col is None:
        return None
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date")


def _first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def load_base_panel() -> pd.DataFrame:
    for path in BASE_CANDIDATES:
        df = _read_csv(path)
        if df is None:
            continue
        if "spy_price" not in df.columns and "spy_daily_return" not in df.columns:
            continue
        if "spy_price" not in df.columns:
            df["spy_price"] = (1.0 + pd.to_numeric(df["spy_daily_return"], errors="coerce").fillna(0.0)).cumprod()
        df["spy_price"] = pd.to_numeric(df["spy_price"], errors="coerce")
        df["spy_daily_return"] = pd.to_numeric(df.get("spy_daily_return", df["spy_price"].pct_change()), errors="coerce")
        if "base_stress_entry_signal" not in df.columns:
            flat_vix = df["macro_regime_confirmed"].eq("FLAT") & (df["VIX_ZSCORE_120D"] >= 3.0)
            steep_either = df["macro_regime_confirmed"].eq("STEEP") & df["monthly_either_state"].eq("SELL")
            df["base_stress_entry_signal"] = flat_vix | steep_either
        return df.sort_values("date").reset_index(drop=True)
    raise FileNotFoundError("Could not load a usable SPY stress-recovery panel.")


def load_commodity_data() -> tuple[pd.DataFrame, str]:
    for path in COMMODITY_SOURCES:
        df = _read_csv(path)
        if df is None:
            continue
        price_col = _first_col(df, ["CMDTY_FUT_price", "CMDTY_FUT", "CMDTY", "commodity_price", "close", "Adj Close"])
        ret_col = _first_col(df, ["CMDTY_FUT_RETURN", "CMDTY_FUT_return", "CMDTY_ret", "commodity_return"])
        if price_col is None and ret_col is None:
            continue
        out = df[["date"]].copy()
        source_bits = []
        if price_col is not None:
            out["CMDTY_FUT_price"] = pd.to_numeric(df[price_col], errors="coerce")
            source_bits.append(price_col)
        if ret_col is not None:
            out["CMDTY_FUT_return"] = pd.to_numeric(df[ret_col], errors="coerce")
            source_bits.append(ret_col)
        return out, f"{path.relative_to(ROOT)}:{'/'.join(source_bits)}"
    raise FileNotFoundError("Could not locate commodity price or return data.")


def merge_commodity_features(base: pd.DataFrame, commodity: pd.DataFrame) -> pd.DataFrame:
    out = base.merge(commodity, on="date", how="left")
    out["CMDTY_FUT_return"] = pd.to_numeric(out.get("CMDTY_FUT_return"), errors="coerce")
    if "CMDTY_FUT_price" not in out.columns or out["CMDTY_FUT_price"].isna().all():
        out["CMDTY_FUT_price"] = (1.0 + out["CMDTY_FUT_return"].fillna(0.0)).cumprod()
    else:
        out["CMDTY_FUT_price"] = pd.to_numeric(out["CMDTY_FUT_price"], errors="coerce")
        out["CMDTY_FUT_return"] = out["CMDTY_FUT_return"].combine_first(out["CMDTY_FUT_price"].pct_change())
    return out.dropna(subset=["spy_price", "CMDTY_FUT_price"]).reset_index(drop=True)


def build_commodity_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["previous_high"] = out["spy_price"].cummax()
    out["spy_drawdown_from_previous_high"] = out["spy_price"] / out["previous_high"] - 1.0
    out["spy_ret_20d"] = out["spy_price"] / out["spy_price"].shift(20) - 1.0
    out["spy_ret_60d"] = out["spy_price"] / out["spy_price"].shift(60) - 1.0
    for w in CONFIG["commodity_ma_windows"]:
        out[f"CMDTY_MA{w}"] = out["CMDTY_FUT_price"].rolling(w, min_periods=w).mean()
    for w in CONFIG["commodity_ret_windows"]:
        out[f"CMDTY_RET{w}"] = out["CMDTY_FUT_price"] / out["CMDTY_FUT_price"].shift(w) - 1.0
    out["CMDTY_PREVIOUS_HIGH"] = out["CMDTY_FUT_price"].cummax()
    out["CMDTY_DRAWDOWN_FROM_HIGH"] = out["CMDTY_FUT_price"] / out["CMDTY_PREVIOUS_HIGH"] - 1.0
    out["CMDTY_MA60_GAP"] = out["CMDTY_FUT_price"] / out["CMDTY_MA60"] - 1.0
    out["CMDTY_MA120_GAP"] = out["CMDTY_FUT_price"] / out["CMDTY_MA120"] - 1.0
    out["CMDTY_VOL60"] = out["CMDTY_FUT_return"].rolling(60, min_periods=30).std(ddof=1) * np.sqrt(252)
    for h in CONFIG["forward_windows"]:
        out[f"forward_return_{h}d"] = out["spy_price"].shift(-h) / out["spy_price"] - 1.0
    return out


def build_commodity_triggers(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    dd5 = out["spy_drawdown_from_previous_high"] <= -0.05
    dd8 = out["spy_drawdown_from_previous_high"] <= -0.08
    credit_widen = out.get("D_CREDIT_SPREAD_20D", pd.Series(np.nan, index=out.index)) > 0
    out["CMDTY_BELOW_MA60"] = out["CMDTY_FUT_price"] < out["CMDTY_MA60"]
    out["CMDTY_BELOW_MA120"] = out["CMDTY_FUT_price"] < out["CMDTY_MA120"]
    out["CMDTY_RET60_LT_NEG10"] = out["CMDTY_RET60"] < -0.10
    out["CMDTY_DD_LT_NEG10"] = out["CMDTY_DRAWDOWN_FROM_HIGH"] <= -0.10
    out["DD5_AND_CMDTY_BELOW_MA60"] = dd5 & out["CMDTY_BELOW_MA60"]
    out["DD5_AND_CMDTY_BELOW_MA120"] = dd5 & out["CMDTY_BELOW_MA120"]
    out["DD5_AND_CMDTY_RET60_LT_NEG10"] = dd5 & out["CMDTY_RET60_LT_NEG10"]
    out["DD5_AND_CMDTY_DD_LT_NEG10"] = dd5 & out["CMDTY_DD_LT_NEG10"]
    out["DD8_AND_CMDTY_BELOW_MA60"] = dd8 & out["CMDTY_BELOW_MA60"]
    out["DD8_AND_CMDTY_RET60_LT_NEG10"] = dd8 & out["CMDTY_RET60_LT_NEG10"]
    out["DD5_CMDTY_BELOW_MA60_AND_CREDIT_WIDEN"] = dd5 & out["CMDTY_BELOW_MA60"] & credit_widen
    out["DD5_CMDTY_RET60_NEG10_AND_CREDIT_WIDEN"] = dd5 & out["CMDTY_RET60_LT_NEG10"] & credit_widen
    out["DD5_CMDTY_DD10_AND_CREDIT_WIDEN"] = dd5 & out["CMDTY_DD_LT_NEG10"] & credit_widen
    out["BASE_PLUS_DD5_CMDTY_MA60"] = out["base_stress_entry_signal"].fillna(False).astype(bool) | out["DD5_AND_CMDTY_BELOW_MA60"]
    out["BASE_PLUS_DD5_CMDTY_RET60_NEG10"] = out["base_stress_entry_signal"].fillna(False).astype(bool) | out["DD5_AND_CMDTY_RET60_LT_NEG10"]
    return out


def trigger_columns() -> list[str]:
    return [
        "CMDTY_BELOW_MA60",
        "CMDTY_BELOW_MA120",
        "CMDTY_RET60_LT_NEG10",
        "CMDTY_DD_LT_NEG10",
        "DD5_AND_CMDTY_BELOW_MA60",
        "DD5_AND_CMDTY_BELOW_MA120",
        "DD5_AND_CMDTY_RET60_LT_NEG10",
        "DD5_AND_CMDTY_DD_LT_NEG10",
        "DD8_AND_CMDTY_BELOW_MA60",
        "DD8_AND_CMDTY_RET60_LT_NEG10",
        "DD5_CMDTY_BELOW_MA60_AND_CREDIT_WIDEN",
        "DD5_CMDTY_RET60_NEG10_AND_CREDIT_WIDEN",
        "DD5_CMDTY_DD10_AND_CREDIT_WIDEN",
        "BASE_PLUS_DD5_CMDTY_MA60",
        "BASE_PLUS_DD5_CMDTY_RET60_NEG10",
    ]


def _forward_path_metrics(panel: pd.DataFrame, idx: int, window: int) -> dict[str, float]:
    end = min(idx + window, len(panel) - 1)
    path = panel["spy_price"].iloc[idx : end + 1].astype(float)
    if len(path) < 2:
        return {"return": np.nan, "mdd": np.nan, "runup": np.nan, "days_to_trough": np.nan}
    rel = path / path.iloc[0] - 1.0
    wealth = path / path.iloc[0]
    dd = wealth / wealth.cummax() - 1.0
    return {"return": float(rel.iloc[-1]), "mdd": float(dd.min()), "runup": float(rel.max()), "days_to_trough": int(np.argmin(path.to_numpy()))}


def extract_trigger_events(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cooldown = int(CONFIG["cooldown_days"])
    for trig in trigger_columns():
        active = panel[trig].fillna(False).astype(bool).to_numpy()
        last_event = -10**9
        armed = True
        for i, val in enumerate(active):
            if not val:
                armed = True
                continue
            if not armed or i - last_event <= cooldown:
                continue
            if i > 0 and active[i - 1]:
                continue
            armed = False
            last_event = i
            row = {
                "trigger_name": trig,
                "event_idx": i,
                "event_date": panel["date"].iloc[i],
                "macro_regime_confirmed": panel.get("macro_regime_confirmed", pd.Series(index=panel.index)).iloc[i],
                "monthly_either_state": panel.get("monthly_either_state", pd.Series(index=panel.index)).iloc[i],
                "spy_price": panel["spy_price"].iloc[i],
                "spy_drawdown_from_previous_high": panel["spy_drawdown_from_previous_high"].iloc[i],
                "VIX_LEVEL": panel.get("VIX_LEVEL", pd.Series(index=panel.index)).iloc[i],
                "VIX_ZSCORE_120D": panel.get("VIX_ZSCORE_120D", pd.Series(index=panel.index)).iloc[i],
                "CREDIT_SPREAD_BAA_AAA": panel.get("CREDIT_SPREAD_BAA_AAA", pd.Series(index=panel.index)).iloc[i],
                "D_CREDIT_SPREAD_20D": panel.get("D_CREDIT_SPREAD_20D", pd.Series(index=panel.index)).iloc[i],
                "CMDTY_FUT_price": panel["CMDTY_FUT_price"].iloc[i],
                "CMDTY_RET60": panel["CMDTY_RET60"].iloc[i],
                "CMDTY_DRAWDOWN_FROM_HIGH": panel["CMDTY_DRAWDOWN_FROM_HIGH"].iloc[i],
                "CMDTY_MA60_GAP": panel["CMDTY_MA60_GAP"].iloc[i],
                "CMDTY_MA120_GAP": panel["CMDTY_MA120_GAP"].iloc[i],
            }
            for w in CONFIG["forward_windows"]:
                m = _forward_path_metrics(panel, i, w)
                row[f"forward_return_{w}d"] = m["return"]
                row[f"forward_max_drawdown_{w}d"] = m["mdd"]
                if w in [21, 63]:
                    row[f"forward_max_runup_{w}d"] = m["runup"]
                    row[f"days_to_trough_{w}d"] = m["days_to_trough"]
            row["mdd_21d_below_3"] = row["forward_max_drawdown_21d"] <= -0.03
            row["mdd_21d_below_5"] = row["forward_max_drawdown_21d"] <= -0.05
            row["mdd_63d_below_10"] = row["forward_max_drawdown_63d"] <= -0.10
            row["false_alarm_21d"] = row["forward_max_drawdown_21d"] > -0.03
            rows.append(row)
    return pd.DataFrame(rows)


def _summary(sub: pd.DataFrame) -> dict:
    if sub.empty:
        return {}
    years = (sub["event_date"].max() - sub["event_date"].min()).days / 365.25
    return {
        "event_count": len(sub),
        "events_per_year": len(sub) / years if years > 0 else np.nan,
        "avg_forward_return_21d": sub["forward_return_21d"].mean(),
        "avg_forward_return_63d": sub["forward_return_63d"].mean(),
        "avg_forward_mdd_21d": sub["forward_max_drawdown_21d"].mean(),
        "avg_forward_mdd_63d": sub["forward_max_drawdown_63d"].mean(),
        "median_forward_mdd_21d": sub["forward_max_drawdown_21d"].median(),
        "pct_mdd_21d_below_3": sub["mdd_21d_below_3"].mean(),
        "pct_mdd_21d_below_5": sub["mdd_21d_below_5"].mean(),
        "pct_mdd_63d_below_10": sub["mdd_63d_below_10"].mean(),
        "false_alarm_rate_21d": sub["false_alarm_21d"].mean(),
        "avg_days_to_trough_21d": sub["days_to_trough_21d"].mean(),
        "avg_days_to_trough_63d": sub["days_to_trough_63d"].mean(),
        "median_spy_drawdown_at_trigger": sub["spy_drawdown_from_previous_high"].median(),
        "median_cmdty_ret60_at_trigger": sub["CMDTY_RET60"].median(),
        "median_cmdty_drawdown_at_trigger": sub["CMDTY_DRAWDOWN_FROM_HIGH"].median(),
        "median_cmdty_ma60_gap_at_trigger": sub["CMDTY_MA60_GAP"].median(),
    }


def summarize_triggers(events: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"trigger_name": trig, **_summary(sub)} for trig, sub in events.groupby("trigger_name")]).sort_values("event_count", ascending=False)


def summarize_by_regime(events: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"trigger_name": t, "macro_regime_confirmed": r, **_summary(s)} for (t, r), s in events.groupby(["trigger_name", "macro_regime_confirmed"], dropna=False)])


def analyze_missed_episode_coverage(panel: pd.DataFrame, events: pd.DataFrame, summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not MISSED_EPISODES.exists():
        return pd.DataFrame(), pd.DataFrame()
    eps = pd.read_csv(MISSED_EPISODES, parse_dates=["start_date", "trough_date", "end_date"])
    eps = eps[(eps["threshold"].eq(-0.05)) & (eps["missed_by_R3_10d"].astype(bool))]
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(panel["date"])}
    lo_days, hi_days = CONFIG["missed_episode_window"]
    triggers = trigger_columns()
    rows = []
    for _, ep in eps.iterrows():
        start_idx = date_to_idx.get(pd.Timestamp(ep["start_date"]))
        trough_idx = date_to_idx.get(pd.Timestamp(ep["trough_date"]))
        if start_idx is None:
            continue
        row = {
            "episode_id": ep["episode_id"],
            "episode_start": ep["start_date"],
            "episode_trough": ep["trough_date"],
            "macro_regime_at_start": ep.get("macro_regime_at_start"),
            "spy_max_drawdown_episode": ep.get("spy_max_drawdown_episode"),
            "missed_by_R3_10d": ep.get("missed_by_R3_10d"),
        }
        lo_idx, hi_idx = max(0, start_idx + lo_days), min(len(panel) - 1, start_idx + hi_days)
        for trig in triggers:
            cand = events[events["trigger_name"].eq(trig) & events["event_idx"].between(lo_idx, hi_idx)].copy()
            if not cand.empty:
                cand["abs_days"] = (cand["event_idx"] - start_idx).abs()
                ev = cand.sort_values("abs_days").iloc[0]
                row[f"{trig}_first_trigger_date"] = ev["event_date"]
                row[f"{trig}_days_from_episode_start_to_trigger"] = int(ev["event_idx"] - start_idx)
                row[f"{trig}_trigger_before_trough"] = bool(trough_idx is not None and ev["event_idx"] <= trough_idx)
            else:
                row[f"{trig}_first_trigger_date"] = pd.NaT
                row[f"{trig}_days_from_episode_start_to_trigger"] = np.nan
                row[f"{trig}_trigger_before_trough"] = False
        rows.append(row)
    coverage = pd.DataFrame(rows)
    srows = []
    for trig in triggers:
        date_col = f"{trig}_first_trigger_date"
        days_col = f"{trig}_days_from_episode_start_to_trigger"
        before_col = f"{trig}_trigger_before_trough"
        has = coverage[date_col].notna() if not coverage.empty else pd.Series(dtype=bool)
        base = summary.loc[summary["trigger_name"].eq(trig)].iloc[0].to_dict() if trig in set(summary["trigger_name"]) else {}
        srows.append(
            {
                "trigger_name": trig,
                "missed_episode_coverage_rate": has.mean() if len(has) else np.nan,
                "avg_days_from_start_to_trigger": coverage.loc[has, days_col].mean() if len(has) else np.nan,
                "median_days_from_start_to_trigger": coverage.loc[has, days_col].median() if len(has) else np.nan,
                "pct_trigger_before_trough": coverage.loc[has, before_col].mean() if len(has) and has.any() else np.nan,
                "event_count_full_sample": base.get("event_count", np.nan),
                "events_per_year_full_sample": base.get("events_per_year", np.nan),
                "false_alarm_rate_21d": base.get("false_alarm_rate_21d", np.nan),
            }
        )
    return coverage, pd.DataFrame(srows).sort_values(["missed_episode_coverage_rate", "false_alarm_rate_21d"], ascending=[False, True])


def _case_covered(coverage: pd.DataFrame, trigger: str, start: str, end: str) -> bool:
    if coverage.empty:
        return False
    date_col = f"{trigger}_first_trigger_date"
    if date_col not in coverage.columns:
        return False
    sub = coverage[pd.to_datetime(coverage["episode_start"]).between(pd.Timestamp(start), pd.Timestamp(end))]
    return bool(sub[date_col].notna().any()) if not sub.empty else False


def rank_triggers(summary: pd.DataFrame, coverage_summary: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()
    if not coverage_summary.empty:
        df = df.merge(coverage_summary[["trigger_name", "missed_episode_coverage_rate"]], on="trigger_name", how="left")
    else:
        df["missed_episode_coverage_rate"] = np.nan
    eligible = df[(df["event_count"] >= 8) & (df["events_per_year"] <= 8) & (df["false_alarm_rate_21d"] <= 0.70)].copy()
    if eligible.empty:
        ranked = df.assign(score=np.nan)
    else:
        vals = eligible["avg_forward_mdd_21d"].abs()
        eligible["norm_abs_avg_forward_mdd_21d"] = (vals - vals.min()) / (vals.max() - vals.min()) if vals.max() > vals.min() else 0.5
        eligible["score"] = (
            0.25 * eligible["pct_mdd_21d_below_5"].fillna(0)
            + 0.20 * eligible["pct_mdd_63d_below_10"].fillna(0)
            + 0.20 * eligible["norm_abs_avg_forward_mdd_21d"].fillna(0)
            + 0.20 * eligible["missed_episode_coverage_rate"].fillna(0)
            - 0.15 * eligible["false_alarm_rate_21d"].fillna(1)
        )
        ranked = eligible.sort_values("score", ascending=False)
    for label, (start, end) in CONFIG["case_study_windows"].items():
        ranked[f"covers_{label}"] = ranked["trigger_name"].map(lambda t: _case_covered(coverage, t, start, end))
    return ranked


def write_case_studies(panel: pd.DataFrame) -> None:
    cols = [
        "date", "spy_price", "spy_drawdown_from_previous_high", "macro_regime_confirmed", "monthly_either_state",
        "VIX_LEVEL", "VIX_ZSCORE_120D", "CREDIT_SPREAD_BAA_AAA", "D_CREDIT_SPREAD_20D",
        "CMDTY_FUT_price", "CMDTY_RET60", "CMDTY_DRAWDOWN_FROM_HIGH", "CMDTY_MA60", "CMDTY_MA120",
    ] + trigger_columns()
    for name, (start, end) in CONFIG["case_study_windows"].items():
        sub = panel[panel["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        sub[[c for c in cols if c in sub.columns]].to_csv(OUTPUT_DIR / f"case_study_{name}.csv", index=False)


def plot_results(panel: pd.DataFrame, events: pd.DataFrame, summary: pd.DataFrame, coverage_summary: pd.DataFrame, ranking: pd.DataFrame, by_regime: pd.DataFrame) -> None:
    top = ranking.head(10)["trigger_name"].tolist() if not ranking.empty else summary.head(10)["trigger_name"].tolist()
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=summary, x="false_alarm_rate_21d", y="pct_mdd_21d_below_5", size="event_count", hue="avg_forward_mdd_21d", sizes=(50, 350), palette="viridis_r", ax=ax)
    for _, row in summary[summary["trigger_name"].isin(top)].iterrows():
        ax.text(row["false_alarm_rate_21d"], row["pct_mdd_21d_below_5"], row["trigger_name"], fontsize=8)
    ax.set_title("Commodity trigger quality")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "commodity_trigger_scatter.png", dpi=160)
    plt.close(fig)

    tmp = summary.melt(id_vars="trigger_name", value_vars=["avg_forward_mdd_21d", "avg_forward_mdd_63d"], var_name="metric", value_name="value")
    order = summary.sort_values("avg_forward_mdd_21d")["trigger_name"]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=tmp, x="trigger_name", y="value", hue="metric", order=order, ax=ax)
    ax.tick_params(axis="x", rotation=70)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "commodity_trigger_forward_mdd_bar.png", dpi=160)
    plt.close(fig)

    if not coverage_summary.empty:
        cov = coverage_summary.sort_values("missed_episode_coverage_rate", ascending=False)
        fig, ax1 = plt.subplots(figsize=(12, 5))
        sns.barplot(data=cov, x="trigger_name", y="missed_episode_coverage_rate", color="#4c78a8", ax=ax1)
        ax1.tick_params(axis="x", rotation=70)
        ax2 = ax1.twinx()
        ax2.plot(range(len(cov)), cov["events_per_year_full_sample"], color="#e45756", marker="o")
        ax2.set_ylabel("Events/year")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "commodity_trigger_missed_coverage_bar.png", dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    axes[0].plot(panel["date"], panel["spy_price"] / panel["spy_price"].iloc[0], color="black", label="SPY")
    axes[0].plot(panel["date"], panel["spy_drawdown_from_previous_high"], color="red", alpha=0.7, label="SPY DD")
    axes[0].legend()
    axes[1].plot(panel["date"], panel["CMDTY_FUT_price"], label="CMDTY")
    axes[1].plot(panel["date"], panel["CMDTY_MA60"], label="MA60")
    axes[1].plot(panel["date"], panel["CMDTY_MA120"], label="MA120")
    axes[1].legend()
    axes[2].plot(panel["date"], panel["CMDTY_RET60"], label="RET60")
    axes[2].plot(panel["date"], panel["CMDTY_DRAWDOWN_FROM_HIGH"], label="DD from high")
    axes[2].legend()
    for j, trig in enumerate(top[:3]):
        ev = events[events["trigger_name"].eq(trig)]
        axes[3].scatter(ev["event_date"], np.repeat(j, len(ev)), s=18, label=trig)
    axes[3].set_yticks(range(len(top[:3])))
    axes[3].set_yticklabels(top[:3])
    axes[3].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "commodity_spy_timeline_with_triggers.png", dpi=160)
    plt.close(fig)

    for name in ["2015_2016", "2018Q4", "2022"]:
        start, end = CONFIG["case_study_windows"][name]
        sub = panel[panel["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        if sub.empty:
            continue
        fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
        axes[0].plot(sub["date"], sub["spy_drawdown_from_previous_high"], color="red")
        axes[0].axhline(-0.05, color="black", ls="--", lw=0.8)
        axes[1].plot(sub["date"], sub["CMDTY_FUT_price"], label="CMDTY")
        axes[1].plot(sub["date"], sub["CMDTY_MA60"], label="MA60")
        axes[1].plot(sub["date"], sub["CMDTY_MA120"], label="MA120")
        axes[1].legend()
        axes[2].plot(sub["date"], sub["CMDTY_RET60"], label="RET60")
        axes[2].plot(sub["date"], sub["CMDTY_DRAWDOWN_FROM_HIGH"], label="CMDTY DD")
        axes[2].legend()
        if "D_CREDIT_SPREAD_20D" in sub:
            axes[3].plot(sub["date"], sub["D_CREDIT_SPREAD_20D"], label="credit 20d chg")
        axes[3].plot(sub["date"], sub["VIX_ZSCORE_120D"], label="VIX z")
        axes[3].legend()
        for trig in top[:3]:
            ev = events[events["trigger_name"].eq(trig) & events["event_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
            axes[4].scatter(ev["event_date"], np.repeat(trig, len(ev)), label=trig)
        axes[4].legend(loc="best")
        fig.suptitle(f"Commodity case study {name}")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"case_study_{name}_commodity.png", dpi=160)
        plt.close(fig)

    if not by_regime.empty:
        heat = by_regime[by_regime["trigger_name"].isin(top[:10])].pivot_table(index="trigger_name", columns="macro_regime_confirmed", values="pct_mdd_21d_below_5")
        fig, ax = plt.subplots(figsize=(8, max(4, len(heat) * 0.35)))
        sns.heatmap(heat, annot=True, fmt=".2f", cmap="Reds", ax=ax)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "trigger_summary_by_regime_heatmap.png", dpi=160)
        plt.close(fig)


def write_markdown_report(summary: pd.DataFrame, by_regime: pd.DataFrame, coverage_summary: pd.DataFrame, ranking: pd.DataFrame, source: str) -> None:
    lines = [
        "# Commodity Stress Trigger Diagnostic",
        "",
        "## Purpose",
        "",
        "This diagnostic tests whether `SPY already damaged + commodity weakness` helps detect commodity / global-growth stress, especially 2015-2016. It does not change strategy allocation.",
        "",
        "## Data",
        "",
        f"- Commodity source: `{source}`.",
        "- If only returns are available, commodity price is a normalized cumulative-return index.",
        "",
        "## Full-Sample Findings",
        "",
        summary.sort_values(["false_alarm_rate_21d", "pct_mdd_21d_below_5"], ascending=[True, False]).head(12).to_markdown(index=False),
        "",
        "## Missed Drawdown Coverage",
        "",
        coverage_summary.head(12).to_markdown(index=False) if not coverage_summary.empty else "_Missed episode table not available._",
        "",
        "## Regime-Conditioned Findings",
        "",
        by_regime.head(40).to_markdown(index=False) if not by_regime.empty else "_No regime summary._",
        "",
        "## Ranking",
        "",
        ranking.head(10).to_markdown(index=False) if not ranking.empty else "_No trigger passed filters._",
        "",
        "## Interpretation",
        "",
        "- Pure commodity weakness is included only as a benchmark because it can fire often without equity stress.",
        "- Price-confirmed commodity weakness is more relevant to a SPY/CASH stress overlay.",
        "- Any candidate should go through a separate SPY/CASH strategy backtest before it is considered for the full regime-hedge model.",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    base = load_base_panel()
    commodity, source = load_commodity_data()
    panel = build_commodity_triggers(build_commodity_features(merge_commodity_features(base, commodity)))
    events = extract_trigger_events(panel)
    summary = summarize_triggers(events)
    by_regime = summarize_by_regime(events)
    coverage, coverage_summary = analyze_missed_episode_coverage(panel, events, summary)
    ranking = rank_triggers(summary, coverage_summary, coverage)

    panel.to_csv(DAILY_PANEL_OUT, index=False)
    events.drop(columns=["event_idx"], errors="ignore").to_csv(EVENT_TABLE, index=False)
    summary.to_csv(SUMMARY_TABLE, index=False)
    by_regime.to_csv(SUMMARY_BY_REGIME, index=False)
    coverage.to_csv(COVERAGE_TABLE, index=False)
    coverage_summary.to_csv(COVERAGE_SUMMARY, index=False)
    ranking.to_csv(RANKING_TABLE, index=False)
    write_case_studies(panel)
    plot_results(panel, events, summary, coverage_summary, ranking, by_regime)
    write_markdown_report(summary, by_regime, coverage_summary, ranking, source)

    top5 = ranking.head(5)
    print(f"1. Sample range: {panel['date'].iloc[0].date()} to {panel['date'].iloc[-1].date()}")
    print(f"2. Commodity source: {source}")
    print("3. Top 5 commodity triggers by ranking:")
    for _, row in top5.iterrows():
        print(
            f"   {row['trigger_name']}: events {int(row['event_count'])}, false alarm {row['false_alarm_rate_21d']:.1%}, "
            f"pct 21d mdd<-5 {row['pct_mdd_21d_below_5']:.1%}, score {row.get('score', np.nan):.3f}"
        )
    for label in ["2015_2016", "2018Q4", "2022", "2023"]:
        col = f"covers_{label}"
        val = bool(top5[col].any()) if col in top5.columns and not top5.empty else False
        print(f"{label} covered by top 5: {val}")
    recs = ", ".join(top5["trigger_name"].head(2).tolist()) if not top5.empty else "none"
    print(f"9. Recommended for next strategy backtest: {recs}")
    print(f"10. Saved outputs: {OUTPUT_DIR} and {FIGURE_DIR}")


if __name__ == "__main__":
    main()
