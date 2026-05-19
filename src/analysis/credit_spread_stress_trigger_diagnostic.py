from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "credit_spread_stress_trigger_diagnostic"
FIGURE_DIR = ROOT / "figures" / "credit_spread_stress_trigger_diagnostic"

BASE_PANEL = ROOT / "results" / "spy_cash_stress_recovery_timing" / "daily_backtest_panel.csv"
MISSED_EPISODES = ROOT / "results" / "missed_drawdown_episode_diagnostic" / "drawdown_episode_table.csv"
SOURCE_PANELS = [
    ROOT / "results" / "rule_diagnostics" / "rule_state_panel.csv",
    ROOT / "results" / "reconstructed_regime_asset_behavior" / "reconstructed_regime_panel.csv",
    ROOT / "results" / "regime_hedge_steep_sell_ief" / "daily_backtest_panel.csv",
]
RAW_WBAA = ROOT / "data" / "raw" / "macro" / "Credit" / "WBAA.csv"
RAW_WAAA = ROOT / "data" / "raw" / "macro" / "Credit" / "WAAA.csv"

CONFIG = {
    "output_dir": str(OUTPUT_DIR),
    "figure_dir": str(FIGURE_DIR),
    "cooldown_days": 21,
    "forward_windows": [5, 10, 21, 42, 63],
    "credit_z_window": 120,
    "missed_episode_window": [-5, 10],
    "case_study_windows": {
        "2015_2016": ["2015-05-01", "2016-03-31"],
        "2018Q4": ["2018-10-01", "2019-01-31"],
        "2019": ["2019-05-01", "2019-10-31"],
        "2022": ["2021-11-01", "2023-03-31"],
        "2023": ["2023-07-01", "2023-11-30"],
    },
}

EVENT_TABLE = OUTPUT_DIR / "credit_trigger_event_table.csv"
SUMMARY_TABLE = OUTPUT_DIR / "credit_trigger_summary.csv"
SUMMARY_BY_REGIME = OUTPUT_DIR / "credit_trigger_summary_by_regime.csv"
COVERAGE_TABLE = OUTPUT_DIR / "credit_trigger_coverage_on_missed_drawdowns.csv"
COVERAGE_SUMMARY = OUTPUT_DIR / "credit_trigger_missed_episode_coverage_summary.csv"
RANKING_TABLE = OUTPUT_DIR / "credit_trigger_ranking.csv"
REPORT = OUTPUT_DIR / "CREDIT_SPREAD_STRESS_TRIGGER_DIAGNOSTIC.md"
DAILY_PANEL_OUT = OUTPUT_DIR / "credit_trigger_daily_panel.csv"


REGIME_COLORS = {
    "HIGH_INFLATION": "#d95f02",
    "INVERTED": "#7570b3",
    "FLAT": "#1b9e77",
    "STEEP": "#66a61e",
    "NEUTRAL": "#999999",
}


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


def _first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def load_base_panel() -> pd.DataFrame:
    df = _read_csv(BASE_PANEL)
    if df is None:
        raise FileNotFoundError(f"Missing base panel: {BASE_PANEL}")
    price_col = _first_col(df, ["spy_price", "SPY_PRICE"])
    ret_col = _first_col(df, ["spy_daily_return", "SPY_RETURN", "SPY_ret"])
    if price_col is None and ret_col is None:
        raise ValueError("Base panel needs SPY price or return.")
    if price_col is not None:
        df["spy_price"] = pd.to_numeric(df[price_col], errors="coerce")
    else:
        df["spy_daily_return"] = pd.to_numeric(df[ret_col], errors="coerce")
        df["spy_price"] = (1 + df["spy_daily_return"].fillna(0)).cumprod()
    if ret_col is not None:
        df["spy_daily_return"] = pd.to_numeric(df[ret_col], errors="coerce")
    else:
        df["spy_daily_return"] = df["spy_price"].pct_change()
    required = ["macro_regime_confirmed", "monthly_either_state", "VIX_LEVEL", "VIX_ZSCORE_120D", "R3_risk_state", "R3_weight_spy"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        warnings.warn(f"Base panel missing non-fatal columns: {missing}")
    return df.sort_values("date").reset_index(drop=True)


def load_credit_data() -> tuple[pd.DataFrame | None, str]:
    for path in SOURCE_PANELS:
        src = _read_csv(path)
        if src is None:
            continue
        if "CREDIT_SPREAD_BAA_AAA" in src.columns:
            out = src[["date", "CREDIT_SPREAD_BAA_AAA"]].copy()
            out["CREDIT_SPREAD_BAA_AAA"] = pd.to_numeric(out["CREDIT_SPREAD_BAA_AAA"], errors="coerce")
            if out["CREDIT_SPREAD_BAA_AAA"].notna().sum() > 10:
                return out, f"{path.relative_to(ROOT)}:CREDIT_SPREAD_BAA_AAA"
        if {"WBAA", "WAAA"}.issubset(src.columns):
            out = src[["date", "WBAA", "WAAA"]].copy()
            out["WBAA"] = pd.to_numeric(out["WBAA"], errors="coerce")
            out["WAAA"] = pd.to_numeric(out["WAAA"], errors="coerce")
            out["CREDIT_SPREAD_BAA_AAA"] = out["WBAA"] - out["WAAA"]
            return out[["date", "WBAA", "WAAA", "CREDIT_SPREAD_BAA_AAA"]], f"{path.relative_to(ROOT)}:WBAA-WAAA"

    wbaa = _read_csv(RAW_WBAA)
    waaa = _read_csv(RAW_WAAA)
    if wbaa is not None and waaa is not None and "WBAA" in wbaa.columns and "WAAA" in waaa.columns:
        out = wbaa[["date", "WBAA"]].merge(waaa[["date", "WAAA"]], on="date", how="outer").sort_values("date")
        out["WBAA"] = pd.to_numeric(out["WBAA"], errors="coerce")
        out["WAAA"] = pd.to_numeric(out["WAAA"], errors="coerce")
        out[["WBAA", "WAAA"]] = out[["WBAA", "WAAA"]].ffill()
        out["CREDIT_SPREAD_BAA_AAA"] = out["WBAA"] - out["WAAA"]
        return out, "raw weekly WBAA-WAAA forward-filled"
    return None, "missing"


def merge_credit_features(base: pd.DataFrame, credit: pd.DataFrame | None) -> pd.DataFrame:
    out = base.copy()
    if credit is not None:
        out = pd.merge_asof(out.sort_values("date"), credit.sort_values("date"), on="date", direction="backward")
    if "CREDIT_SPREAD_BAA_AAA" not in out.columns:
        raise ValueError("CREDIT_SPREAD_BAA_AAA is required for this diagnostic.")
    return out


def build_credit_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    cs = pd.to_numeric(out["CREDIT_SPREAD_BAA_AAA"], errors="coerce").ffill()
    out["CREDIT_SPREAD_BAA_AAA"] = cs
    out["previous_high"] = out["spy_price"].cummax()
    out["spy_drawdown_from_previous_high"] = out["spy_price"] / out["previous_high"] - 1.0
    out["spy_ret_5d"] = out["spy_price"] / out["spy_price"].shift(5) - 1.0
    out["spy_ret_10d"] = out["spy_price"] / out["spy_price"].shift(10) - 1.0
    out["spy_ret_20d"] = out["spy_price"] / out["spy_price"].shift(20) - 1.0
    out["D_CREDIT_SPREAD_5D"] = cs.diff(5)
    out["D_CREDIT_SPREAD_10D"] = cs.diff(10)
    out["D_CREDIT_SPREAD_20D"] = cs.diff(20)
    w = CONFIG["credit_z_window"]
    roll = cs.rolling(w, min_periods=w)
    out["CREDIT_SPREAD_ZSCORE_120D"] = (cs - roll.mean()) / roll.std(ddof=1).replace(0, np.nan)
    d20 = out["D_CREDIT_SPREAD_20D"]
    roll_d = d20.rolling(w, min_periods=w)
    out["D_CREDIT_SPREAD_20D_ZSCORE_120D"] = (d20 - roll_d.mean()) / roll_d.std(ddof=1).replace(0, np.nan)
    for window in CONFIG["forward_windows"]:
        out[f"spy_forward_return_{window}d"] = out["spy_price"].shift(-window) / out["spy_price"] - 1.0
    return out


def build_credit_triggers(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    dd5 = out["spy_drawdown_from_previous_high"] <= -0.05
    dd8 = out["spy_drawdown_from_previous_high"] <= -0.08
    out["CREDIT_LVL_GT_1_5"] = out["CREDIT_SPREAD_BAA_AAA"] > 1.5
    out["CREDIT_LVL_GT_2_0"] = out["CREDIT_SPREAD_BAA_AAA"] > 2.0
    out["CREDIT_LVL_Z_GT_1_5"] = out["CREDIT_SPREAD_ZSCORE_120D"] > 1.5
    out["CREDIT_LVL_Z_GT_2_0"] = out["CREDIT_SPREAD_ZSCORE_120D"] > 2.0
    out["CREDIT_CHG20_GT_0"] = out["D_CREDIT_SPREAD_20D"] > 0
    out["CREDIT_CHG20_GT_0_05"] = out["D_CREDIT_SPREAD_20D"] > 0.05
    out["CREDIT_CHG20_GT_0_10"] = out["D_CREDIT_SPREAD_20D"] > 0.10
    out["CREDIT_CHG20_Z_GT_1_0"] = out["D_CREDIT_SPREAD_20D_ZSCORE_120D"] > 1.0
    out["CREDIT_CHG20_Z_GT_1_5"] = out["D_CREDIT_SPREAD_20D_ZSCORE_120D"] > 1.5
    out["CREDIT_CHG20_Z_GT_2_0"] = out["D_CREDIT_SPREAD_20D_ZSCORE_120D"] > 2.0
    out["DD5_AND_CREDIT_CHG20_GT_0"] = dd5 & (out["D_CREDIT_SPREAD_20D"] > 0)
    out["DD5_AND_CREDIT_CHG20_GT_0_05"] = dd5 & (out["D_CREDIT_SPREAD_20D"] > 0.05)
    out["DD5_AND_CREDIT_CHG20_GT_0_10"] = dd5 & (out["D_CREDIT_SPREAD_20D"] > 0.10)
    out["DD5_AND_CREDIT_CHG20_Z_GT_1_0"] = dd5 & (out["D_CREDIT_SPREAD_20D_ZSCORE_120D"] > 1.0)
    out["DD5_AND_CREDIT_CHG20_Z_GT_1_5"] = dd5 & (out["D_CREDIT_SPREAD_20D_ZSCORE_120D"] > 1.5)
    out["DD8_AND_CREDIT_CHG20_GT_0"] = dd8 & (out["D_CREDIT_SPREAD_20D"] > 0)
    out["VIX_OR_CREDIT"] = (out["VIX_ZSCORE_120D"] >= 3.0) | (out["D_CREDIT_SPREAD_20D_ZSCORE_120D"] > 1.5)
    out["VIX_AND_CREDIT"] = (out["VIX_ZSCORE_120D"] >= 2.0) & (out["D_CREDIT_SPREAD_20D"] > 0)
    out["DD5_AND_VIX_OR_CREDIT"] = dd5 & ((out["VIX_ZSCORE_120D"] >= 2.0) | (out["D_CREDIT_SPREAD_20D"] > 0))
    return out


def trigger_columns(panel: pd.DataFrame) -> list[str]:
    prefixes = ("CREDIT_", "DD5_", "DD8_", "VIX_")
    return [
        c
        for c in panel.columns
        if c.startswith(prefixes)
        and c
        not in {
            "CREDIT_SPREAD_BAA_AAA",
            "CREDIT_SPREAD_ZSCORE_120D",
            "VIX_LEVEL",
            "VIX_ZSCORE_120D",
        }
        and panel[c].dropna().isin([True, False, 0, 1]).all()
    ]


def _forward_path_metrics(panel: pd.DataFrame, idx: int, window: int) -> dict[str, float]:
    end = min(idx + window, len(panel) - 1)
    path = panel["spy_price"].iloc[idx : end + 1].astype(float)
    if len(path) < 2:
        return {"return": np.nan, "mdd": np.nan, "runup": np.nan, "days_to_trough": np.nan}
    rel = path / path.iloc[0] - 1.0
    wealth = path / path.iloc[0]
    dd = wealth / wealth.cummax() - 1.0
    return {
        "return": float(rel.iloc[-1]),
        "mdd": float(dd.min()),
        "runup": float(rel.max()),
        "days_to_trough": int(np.argmin(path.to_numpy())),
    }


def extract_trigger_events(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cooldown = int(CONFIG["cooldown_days"])
    for trig in trigger_columns(panel):
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
            last_event = i
            armed = False
            row = {
                "trigger_name": trig,
                "event_idx": i,
                "event_date": panel["date"].iloc[i],
                "macro_regime_confirmed": panel.get("macro_regime_confirmed", pd.Series(index=panel.index)).iloc[i],
                "monthly_either_state": panel.get("monthly_either_state", pd.Series(index=panel.index)).iloc[i],
                "R3_risk_state": panel.get("R3_risk_state", pd.Series(index=panel.index)).iloc[i],
                "R3_weight_spy": panel.get("R3_weight_spy", pd.Series(index=panel.index)).iloc[i],
                "spy_price": panel["spy_price"].iloc[i],
                "spy_drawdown_from_previous_high": panel["spy_drawdown_from_previous_high"].iloc[i],
                "VIX_LEVEL": panel.get("VIX_LEVEL", pd.Series(index=panel.index)).iloc[i],
                "VIX_ZSCORE_120D": panel.get("VIX_ZSCORE_120D", pd.Series(index=panel.index)).iloc[i],
                "CREDIT_SPREAD_BAA_AAA": panel["CREDIT_SPREAD_BAA_AAA"].iloc[i],
                "D_CREDIT_SPREAD_20D": panel["D_CREDIT_SPREAD_20D"].iloc[i],
                "CREDIT_SPREAD_ZSCORE_120D": panel["CREDIT_SPREAD_ZSCORE_120D"].iloc[i],
                "D_CREDIT_SPREAD_20D_ZSCORE_120D": panel["D_CREDIT_SPREAD_20D_ZSCORE_120D"].iloc[i],
            }
            for window in CONFIG["forward_windows"]:
                m = _forward_path_metrics(panel, i, window)
                row[f"forward_return_{window}d"] = m["return"]
                row[f"forward_max_drawdown_{window}d"] = m["mdd"]
                if window in (21, 63):
                    row[f"forward_max_runup_{window}d"] = m["runup"]
                    row[f"days_to_trough_{window}d"] = m["days_to_trough"]
            row["mdd_21d_below_3"] = row["forward_max_drawdown_21d"] <= -0.03
            row["mdd_21d_below_5"] = row["forward_max_drawdown_21d"] <= -0.05
            row["mdd_63d_below_10"] = row["forward_max_drawdown_63d"] <= -0.10
            row["false_alarm_21d"] = row["forward_max_drawdown_21d"] > -0.03
            rows.append(row)
    return pd.DataFrame(rows)


def _summary(sub: pd.DataFrame) -> dict[str, float]:
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
        "median_credit_spread_at_trigger": sub["CREDIT_SPREAD_BAA_AAA"].median(),
        "median_credit_change20_at_trigger": sub["D_CREDIT_SPREAD_20D"].median(),
    }


def summarize_triggers(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trig, sub in events.groupby("trigger_name"):
        rows.append({"trigger_name": trig, **_summary(sub)})
    return pd.DataFrame(rows).sort_values("event_count", ascending=False)


def summarize_by_regime(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (trig, regime), sub in events.groupby(["trigger_name", "macro_regime_confirmed"], dropna=False):
        rows.append({"trigger_name": trig, "macro_regime_confirmed": regime, **_summary(sub)})
    return pd.DataFrame(rows)


def analyze_missed_episode_coverage(panel: pd.DataFrame, events: pd.DataFrame, summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not MISSED_EPISODES.exists():
        return pd.DataFrame(), pd.DataFrame()
    eps = pd.read_csv(MISSED_EPISODES, parse_dates=["start_date", "trough_date", "end_date"])
    eps = eps[(eps["threshold"].eq(-0.05)) & (eps["missed_by_R3_10d"].astype(bool))].copy()
    triggers = sorted(events["trigger_name"].unique())
    rows = []
    lo_days, hi_days = CONFIG["missed_episode_window"]
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(panel["date"])}
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
        start = pd.Timestamp(ep["start_date"])
        for trig in triggers:
            lo_idx = max(0, start_idx + lo_days)
            hi_idx = min(len(panel) - 1, start_idx + hi_days)
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


def rank_triggers(summary: pd.DataFrame, coverage_summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()
    if not coverage_summary.empty:
        df = df.merge(coverage_summary[["trigger_name", "missed_episode_coverage_rate"]], on="trigger_name", how="left")
    else:
        df["missed_episode_coverage_rate"] = np.nan
    eligible = df[(df["event_count"] >= 8) & (df["events_per_year"] <= 6) & (df["false_alarm_rate_21d"] <= 0.70)].copy()
    if eligible.empty:
        return df.assign(score=np.nan).sort_values("false_alarm_rate_21d")
    for col in ["avg_forward_mdd_21d"]:
        vals = eligible[col].abs()
        eligible[f"norm_abs_{col}"] = (vals - vals.min()) / (vals.max() - vals.min()) if vals.max() > vals.min() else 0.5
    cov = eligible["missed_episode_coverage_rate"].fillna(0.0)
    eligible["score"] = (
        0.25 * eligible["pct_mdd_21d_below_5"].fillna(0)
        + 0.20 * eligible["pct_mdd_63d_below_10"].fillna(0)
        + 0.20 * eligible["norm_abs_avg_forward_mdd_21d"].fillna(0)
        + 0.15 * cov
        - 0.20 * eligible["false_alarm_rate_21d"].fillna(1)
    )
    return eligible.sort_values("score", ascending=False)


def write_case_studies(panel: pd.DataFrame) -> None:
    cols = [
        "date",
        "spy_price",
        "spy_drawdown_from_previous_high",
        "macro_regime_confirmed",
        "monthly_either_state",
        "R3_risk_state",
        "VIX_LEVEL",
        "VIX_ZSCORE_120D",
        "CREDIT_SPREAD_BAA_AAA",
        "D_CREDIT_SPREAD_20D",
        "D_CREDIT_SPREAD_20D_ZSCORE_120D",
    ] + trigger_columns(panel)
    for name, (start, end) in CONFIG["case_study_windows"].items():
        sub = panel[panel["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        sub[[c for c in cols if c in sub.columns]].to_csv(OUTPUT_DIR / f"case_study_{name}.csv", index=False)


def plot_results(panel: pd.DataFrame, events: pd.DataFrame, summary: pd.DataFrame, coverage_summary: pd.DataFrame, ranking: pd.DataFrame, by_regime: pd.DataFrame) -> None:
    top = ranking.head(10)["trigger_name"].tolist() if not ranking.empty else summary.head(10)["trigger_name"].tolist()
    if not summary.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        s = summary.copy()
        sns.scatterplot(data=s, x="false_alarm_rate_21d", y="pct_mdd_21d_below_5", size="event_count", hue="avg_forward_mdd_21d", sizes=(50, 350), palette="viridis_r", ax=ax)
        for _, row in s[s["trigger_name"].isin(top)].iterrows():
            ax.text(row["false_alarm_rate_21d"], row["pct_mdd_21d_below_5"], row["trigger_name"], fontsize=8)
        ax.set_title("Credit trigger precision / false alarm")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "credit_trigger_scatter.png", dpi=160)
        plt.close(fig)

        order = summary.sort_values("avg_forward_mdd_21d")["trigger_name"].tolist()
        fig, ax = plt.subplots(figsize=(12, 6))
        tmp = summary.melt(id_vars="trigger_name", value_vars=["avg_forward_mdd_21d", "avg_forward_mdd_63d"], var_name="metric", value_name="value")
        sns.barplot(data=tmp, x="trigger_name", y="value", hue="metric", order=order, ax=ax)
        ax.tick_params(axis="x", rotation=70)
        ax.set_title("Average forward drawdown after credit trigger")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "credit_trigger_forward_mdd_bar.png", dpi=160)
        plt.close(fig)

    if not coverage_summary.empty:
        fig, ax1 = plt.subplots(figsize=(12, 5))
        cov = coverage_summary.sort_values("missed_episode_coverage_rate", ascending=False)
        sns.barplot(data=cov, x="trigger_name", y="missed_episode_coverage_rate", color="#4c78a8", ax=ax1)
        ax1.tick_params(axis="x", rotation=70)
        ax2 = ax1.twinx()
        ax2.plot(range(len(cov)), cov["events_per_year_full_sample"], color="#e45756", marker="o")
        ax2.set_ylabel("Events/year")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "credit_trigger_missed_coverage_bar.png", dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    axes[0].plot(panel["date"], panel["spy_price"] / panel["spy_price"].iloc[0], color="black", label="SPY")
    axes[0].plot(panel["date"], panel["spy_drawdown_from_previous_high"], color="red", alpha=0.7, label="SPY DD")
    axes[0].legend(loc="best")
    axes[1].plot(panel["date"], panel["CREDIT_SPREAD_BAA_AAA"], color="#1f77b4")
    axes[1].set_ylabel("Credit spread")
    axes[2].plot(panel["date"], panel["D_CREDIT_SPREAD_20D"], label="20d change", color="#ff7f0e")
    axes[2].plot(panel["date"], panel["D_CREDIT_SPREAD_20D_ZSCORE_120D"], label="20d change z", color="#9467bd", alpha=0.8)
    axes[2].legend(loc="best")
    for j, trig in enumerate(top[:3]):
        ev = events[events["trigger_name"].eq(trig)]
        axes[3].scatter(ev["event_date"], np.repeat(j, len(ev)), label=trig, s=15)
    axes[3].set_yticks(range(len(top[:3])))
    axes[3].set_yticklabels(top[:3])
    axes[3].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "credit_spread_timeline_with_triggers.png", dpi=160)
    plt.close(fig)

    for name in ["2015_2016", "2018Q4", "2022"]:
        start, end = CONFIG["case_study_windows"][name]
        sub = panel[panel["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        if sub.empty:
            continue
        fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
        axes[0].plot(sub["date"], sub["spy_drawdown_from_previous_high"], color="red")
        axes[0].axhline(-0.05, color="black", ls="--", lw=0.8)
        axes[1].plot(sub["date"], sub["CREDIT_SPREAD_BAA_AAA"], color="#1f77b4")
        axes[1].plot(sub["date"], sub["D_CREDIT_SPREAD_20D"], color="#ff7f0e")
        axes[2].plot(sub["date"], sub["VIX_ZSCORE_120D"], color="#9467bd")
        axes[2].axhline(3.0, color="red", ls="--", lw=0.8)
        for trig in top[:3]:
            ev = events[events["trigger_name"].eq(trig) & events["event_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
            axes[3].scatter(ev["event_date"], np.repeat(trig, len(ev)), s=25, label=trig)
        axes[3].legend(loc="best")
        axes[4].fill_between(sub["date"], 0, 1, where=sub["R3_risk_state"].astype(str).eq("RISK"), color="red", alpha=0.35)
        axes[4].set_yticks([0.5])
        axes[4].set_yticklabels(["R3 risk"])
        fig.suptitle(f"Credit trigger case study {name}")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"case_study_{name}_credit.png", dpi=160)
        plt.close(fig)

    if not by_regime.empty:
        candidates = top[:10]
        heat = by_regime[by_regime["trigger_name"].isin(candidates)].pivot_table(index="trigger_name", columns="macro_regime_confirmed", values="pct_mdd_21d_below_5")
        fig, ax = plt.subplots(figsize=(8, max(4, len(heat) * 0.35)))
        sns.heatmap(heat, annot=True, fmt=".2f", cmap="Reds", ax=ax)
        ax.set_title("Pct 21d MDD <= -5% by regime")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "trigger_summary_by_regime_heatmap.png", dpi=160)
        plt.close(fig)


def write_markdown_report(summary: pd.DataFrame, by_regime: pd.DataFrame, coverage_summary: pd.DataFrame, ranking: pd.DataFrame, credit_source: str) -> None:
    top = ranking.head(10)
    lines = [
        "# Credit Spread Stress Trigger Diagnostic",
        "",
        "## Purpose",
        "",
        "This diagnostic tests whether credit-spread triggers can complement VIX z-score and Monthly Either stress signals. It does not modify allocation or add hedge assets.",
        "",
        "## Method",
        "",
        f"- Credit source: `{credit_source}`.",
        "- Credit spread level, 20D changes, rolling 120D z-scores, price-confirmed credit triggers, and VIX+credit combinations are evaluated.",
        "- Events are first False -> True trigger dates with a 21-trading-day cooldown.",
        "- Outcomes are forward SPY return/drawdown over 5/10/21/42/63 trading days.",
        "",
        "## Full-Sample Findings",
        "",
        summary.sort_values(["false_alarm_rate_21d", "pct_mdd_21d_below_5"], ascending=[True, False]).head(12).to_markdown(index=False),
        "",
        "## Regime-Conditioned Findings",
        "",
        by_regime.head(40).to_markdown(index=False) if not by_regime.empty else "_No regime summary available._",
        "",
        "## Missed Drawdown Coverage",
        "",
        coverage_summary.head(12).to_markdown(index=False) if not coverage_summary.empty else "_Missed drawdown episode table not available._",
        "",
        "## Ranking",
        "",
        top.to_markdown(index=False) if not top.empty else "_No trigger passed ranking filters._",
        "",
        "## Interpretation",
        "",
        "- Level-only credit triggers are typically slower and may describe existing stress rather than early warning.",
        "- Change and price-confirmed credit triggers are more relevant candidates for medium-drawdown / funding stress diagnostics.",
        "- Any high-coverage trigger with many events per year needs a separate false-alarm / strategy backtest before implementation.",
        "",
        "## Caveats",
        "",
        "- WBAA/WAAA or credit spread data may be weekly and forward-filled to daily SPY dates.",
        "- Credit spreads can lag equity prices.",
        "- This is a diagnostic grid, not an optimized strategy.",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    base = load_base_panel()
    credit, credit_source = load_credit_data()
    panel = build_credit_triggers(build_credit_features(merge_credit_features(base, credit)))
    events = extract_trigger_events(panel)
    summary = summarize_triggers(events)
    by_regime = summarize_by_regime(events)
    coverage, coverage_summary = analyze_missed_episode_coverage(panel, events, summary)
    ranking = rank_triggers(summary, coverage_summary)

    panel.to_csv(DAILY_PANEL_OUT, index=False)
    events.drop(columns=["event_idx"], errors="ignore").to_csv(EVENT_TABLE, index=False)
    summary.to_csv(SUMMARY_TABLE, index=False)
    by_regime.to_csv(SUMMARY_BY_REGIME, index=False)
    coverage.to_csv(COVERAGE_TABLE, index=False)
    coverage_summary.to_csv(COVERAGE_SUMMARY, index=False)
    ranking.to_csv(RANKING_TABLE, index=False)
    write_case_studies(panel)
    plot_results(panel, events, summary, coverage_summary, ranking, by_regime)
    write_markdown_report(summary, by_regime, coverage_summary, ranking, credit_source)

    top5 = ranking.head(5)
    print(f"1. Sample range: {panel['date'].iloc[0].date()} to {panel['date'].iloc[-1].date()}")
    print(f"2. Credit spread source/frequency: {credit_source}")
    print("3. Top 5 credit triggers by ranking:")
    for _, row in top5.iterrows():
        print(
            f"   {row['trigger_name']}: events {int(row['event_count'])}, false alarm {row['false_alarm_rate_21d']:.1%}, "
            f"pct 21d mdd<-5 {row['pct_mdd_21d_below_5']:.1%}, score {row.get('score', np.nan):.3f}"
        )
    def cov_case(label: str, start: str, end: str) -> str:
        if coverage.empty:
            return "n/a"
        cols = [c for c in coverage.columns if c.endswith("_first_trigger_date")]
        sub = coverage[pd.to_datetime(coverage["episode_start"]).between(pd.Timestamp(start), pd.Timestamp(end))]
        return str(bool(sub[cols].notna().any(axis=1).any())) if not sub.empty else "no missed episode"
    print(f"5. Covers 2015-2016 missed episode: {cov_case('2015', '2015-05-01', '2016-03-31')}")
    print(f"6. Covers 2018Q4: {cov_case('2018Q4', '2018-10-01', '2019-01-31')}")
    print(f"7. Covers 2022: {cov_case('2022', '2021-11-01', '2023-03-31')}")
    print(f"8. Covers 2023: {cov_case('2023', '2023-07-01', '2023-11-30')}")
    recs = ", ".join(top5["trigger_name"].head(2).tolist()) if not top5.empty else "none"
    print(f"9. Recommended for next strategy backtest: {recs}")
    print(f"10. Saved outputs: {OUTPUT_DIR} and {FIGURE_DIR}")


if __name__ == "__main__":
    main()
