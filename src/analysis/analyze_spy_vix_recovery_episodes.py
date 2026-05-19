from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "spy_timing_episode_analysis"

PANEL_CANDIDATES = [
    ROOT / "results" / "rule_based_allocation" / "rule_strategy_daily_panel_v2.csv",
    ROOT / "results" / "rule_based_backtest" / "rule_based_daily_backtest_panel.csv",
    ROOT / "results" / "rule_diagnostics" / "rule_state_panel.csv",
]
DAILY_CLOSE_PATH = ROOT / "data" / "processed" / "assets" / "daily_adjusted_close.csv"
DAILY_RETURNS_PATH = ROOT / "data" / "processed" / "assets" / "daily_returns.csv"
VIX_PATH = ROOT / "data" / "raw" / "macro" / "volatility" / "VIXCLS.csv"
DGS1_PATH = ROOT / "data" / "raw" / "macro" / "rate" / "DGS1.csv"
DGS10_PATH = ROOT / "data" / "raw" / "macro" / "rate" / "DGS10.csv"
WAAA_PATH = ROOT / "data" / "raw" / "macro" / "Credit" / "WAAA.csv"
WBAA_PATH = ROOT / "data" / "raw" / "macro" / "Credit" / "WBAA.csv"

EPISODE_PATH = RESULTS_DIR / "episode_summary.csv"
EPISODE_PATH_SIMPLE = RESULTS_DIR / "episode_summary_simple_exit.csv"
EPISODE_BY_MACRO_PATH = RESULTS_DIR / "episode_summary_by_macro_regime.csv"
EPISODE_BY_MACRO_TYPE_PATH = RESULTS_DIR / "episode_summary_by_macro_and_type.csv"
TRIGGER_EVENTS_PATH = RESULTS_DIR / "recovery_trigger_events.csv"
TRIGGER_SUMMARY_PATH = RESULTS_DIR / "recovery_trigger_summary.csv"
TRIGGER_BY_MACRO_PATH = RESULTS_DIR / "recovery_trigger_by_macro.csv"
WARNING_STRESS_PATH = RESULTS_DIR / "warning_only_vs_stress_by_macro.csv"

FIG_COUNT_PATH = RESULTS_DIR / "episode_count_by_macro_and_type.png"
FIG_MDD_PATH = RESULTS_DIR / "avg_spy_max_drawdown_by_macro_and_type.png"
FIG_RET_PATH = RESULTS_DIR / "avg_spy_return_by_macro_and_type.png"
FIG_PEAK_TO_LOW_PATH = RESULTS_DIR / "days_from_vix_peak_to_spy_low_distribution.png"
FIG_TRIGGER_21D_PATH = RESULTS_DIR / "trigger_quality_heatmap_21d_return.png"
FIG_TRIGGER_EARLY_PATH = RESULTS_DIR / "trigger_early_risk_heatmap.png"
FIG_EVENT_VIX_PEAK_PATH = RESULTS_DIR / "event_study_vix_peak.png"
FIG_EVENT_TRIGGER_PATH = RESULTS_DIR / "event_study_combined_recovery_trigger.png"

REGIME_ORDER = ["FLAT", "INVERTED", "STEEP", "HIGH_INFLATION", "NEUTRAL"]
EPISODE_TYPE_ORDER = ["WARNING_ONLY", "STRESS_EPISODE"]
TRIGGER_ORDER = [
    "VIX_PEAK_PULLBACK_20",
    "SPY_REBOUND_FROM_LOW_3",
    "SPY_ABOVE_MA10",
    "VIX_DOWN_SPY_UP_3D",
    "CREDIT_NOT_WIDENING",
    "COMBINED_RECOVERY",
]
REGIME_COLORS = {
    "FLAT": "#67a9cf",
    "INVERTED": "#2166ac",
    "STEEP": "#1b7837",
    "HIGH_INFLATION": "#ef8a62",
    "NEUTRAL": "#bdbdbd",
}
TIMELINE_YEARS = [2008, 2011, 2020, 2022, 2025]


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def read_fred_csv(path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
    value_col = next((c for c in df.columns if c != date_col), df.columns[-1])
    out = df[[date_col, value_col]].copy()
    out.columns = ["date", value_name]
    out["date"] = pd.to_datetime(out["date"])
    out[value_name] = pd.to_numeric(out[value_name].replace(".", np.nan), errors="coerce")
    return out.sort_values("date")


def raw_macro_regime(row: pd.Series) -> str:
    if pd.isna(row["CREDIT_SPREAD_BAA_AAA"]) or pd.isna(row["DGS1"]) or pd.isna(row["TERM_SPREAD_10Y_1Y"]):
        return "NEUTRAL"
    if row["CREDIT_SPREAD_BAA_AAA"] > 1.5 and row["DGS1"] > 5:
        return "HIGH_INFLATION"
    if row["TERM_SPREAD_10Y_1Y"] < 0:
        return "INVERTED"
    if 0 <= row["TERM_SPREAD_10Y_1Y"] < 1:
        return "FLAT"
    if row["TERM_SPREAD_10Y_1Y"] >= 1:
        return "STEEP"
    return "NEUTRAL"


def confirm_regime(raw: pd.Series, confirmation_days: int = 3, initial_confirmed: str = "NEUTRAL") -> pd.DataFrame:
    confirmed = initial_confirmed
    candidate = initial_confirmed
    candidate_count = 0
    confirmed_list, candidate_list, candidate_count_list, switch_flags = [], [], [], []
    for value in raw.astype(str):
        switch_flag = False
        if value == confirmed:
            candidate = confirmed
            candidate_count = 0
        elif value == candidate:
            candidate_count += 1
        else:
            candidate = value
            candidate_count = 1
        if candidate != confirmed and candidate_count >= confirmation_days:
            confirmed = candidate
            switch_flag = True
            candidate = confirmed
            candidate_count = 0
        confirmed_list.append(confirmed)
        candidate_list.append(candidate)
        candidate_count_list.append(candidate_count)
        switch_flags.append(switch_flag)
    return pd.DataFrame(
        {
            "macro_regime_candidate": candidate_list,
            "macro_regime_candidate_count": candidate_count_list,
            "macro_regime_confirmed": confirmed_list,
            "macro_regime_switch_flag": switch_flags,
        }
    )


def choose_input_panel() -> Path:
    required = {
        "date",
        "VIX_LEVEL",
        "CREDIT_SPREAD_BAA_AAA",
        "DGS1",
        "DGS10",
        "TERM_SPREAD_10Y_1Y",
    }
    best_path = None
    best_score = -1
    for path in PANEL_CANDIDATES:
        if not path.exists():
            continue
        try:
            cols = pd.read_csv(path, nrows=0).columns.tolist()
        except Exception:
            continue
        score = sum(col in cols for col in required)
        score += sum(col in cols for col in ["macro_regime_raw", "macro_regime_confirmed", "vix_overlay_raw", "vix_overlay_confirmed"])
        score += sum(col in cols for col in ["SPY", "SPY_RET", "SPY_RETURN"])
        if score > best_score:
            best_score = score
            best_path = path
    if best_path is None:
        raise FileNotFoundError("No usable daily rule-state panel found.")
    return best_path


def build_base_panel() -> tuple[pd.DataFrame, Path]:
    panel_path = choose_input_panel()
    panel = pd.read_csv(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])

    rename_map = {}
    if "SPY_RET" in panel.columns:
        rename_map["SPY_RET"] = "SPY_RETURN"
    if "vix_level" in panel.columns and "VIX_LEVEL" not in panel.columns:
        rename_map["vix_level"] = "VIX_LEVEL"
    panel = panel.rename(columns=rename_map)

    close = pd.read_csv(DAILY_CLOSE_PATH, usecols=["date", "SPY"])
    close["date"] = pd.to_datetime(close["date"])
    panel = panel.merge(close, on="date", how="left")

    if "SPY_RETURN" not in panel.columns:
        daily = pd.read_csv(DAILY_RETURNS_PATH, usecols=["date", "SPY"])
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.rename(columns={"SPY": "SPY_RETURN"})
        panel = panel.merge(daily, on="date", how="left")

    if "SPY_RETURN" not in panel.columns or panel["SPY_RETURN"].isna().all():
        if "SPY" in panel.columns:
            panel["SPY_RETURN"] = panel["SPY"].pct_change()
        else:
            raise ValueError("Unable to locate SPY return or price series.")

    if "SPY" not in panel.columns or panel["SPY"].isna().all():
        nav = (1.0 + panel["SPY_RETURN"].fillna(0.0)).cumprod()
        panel["SPY"] = nav

    panel["SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"] = panel["SPY"] / panel["SPY"].cummax() - 1.0
    panel["SPY_MA10"] = panel["SPY"].rolling(10, min_periods=10).mean()
    panel["SPY_3D_RETURN"] = panel["SPY"].pct_change(3)
    panel["VIX_3D_CHANGE"] = panel["VIX_LEVEL"].diff(3)
    panel["CREDIT_5D_CHANGE"] = panel["CREDIT_SPREAD_BAA_AAA"].diff(5)

    if "macro_regime_confirmed" not in panel.columns:
        panel["macro_regime_raw"] = panel.apply(raw_macro_regime, axis=1)
        macro_conf = confirm_regime(panel["macro_regime_raw"], confirmation_days=3, initial_confirmed="NEUTRAL")
        panel = pd.concat([panel.reset_index(drop=True), macro_conf], axis=1)
    elif "macro_regime_raw" not in panel.columns:
        panel["macro_regime_raw"] = panel.apply(raw_macro_regime, axis=1)

    if "vix_overlay_raw" not in panel.columns:
        panel["vix_overlay_raw"] = np.select(
            [panel["VIX_LEVEL"] >= 25, panel["VIX_LEVEL"] >= 20],
            ["STRESS", "WARNING"],
            default="NORMAL",
        )
    if "vix_overlay_confirmed" not in panel.columns:
        panel["vix_overlay_confirmed"] = panel["vix_overlay_raw"]

    required = [
        "date",
        "SPY",
        "SPY_RETURN",
        "VIX_LEVEL",
        "CREDIT_SPREAD_BAA_AAA",
        "DGS1",
        "DGS10",
        "TERM_SPREAD_10Y_1Y",
        "macro_regime_raw",
        "macro_regime_confirmed",
        "vix_overlay_raw",
        "vix_overlay_confirmed",
    ]
    panel = panel.dropna(subset=[c for c in required if c in panel.columns]).sort_values("date").reset_index(drop=True)
    return panel, panel_path


def detect_episodes(panel: pd.DataFrame, end_threshold: float) -> list[tuple[int, int]]:
    starts_ends: list[tuple[int, int]] = []
    active = False
    start_idx = None
    prev_vix = np.nan
    for i, vix in enumerate(panel["VIX_LEVEL"]):
        if not active:
            if pd.notna(vix) and vix >= 20 and (pd.isna(prev_vix) or prev_vix < 20):
                active = True
                start_idx = i
        else:
            if pd.notna(vix) and vix < end_threshold:
                starts_ends.append((start_idx, i))
                active = False
                start_idx = None
        prev_vix = vix
    if active and start_idx is not None:
        starts_ends.append((start_idx, len(panel) - 1))
    return starts_ends


def majority_regime(s: pd.Series) -> str:
    if s.empty:
        return "NEUTRAL"
    counts = s.value_counts()
    return str(counts.index[0])


def safe_return(start_price: float, end_price: float) -> float:
    if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
        return np.nan
    return float(end_price / start_price - 1.0)


def forward_return(series: pd.Series, idx: int, days: int) -> float:
    if idx + days >= len(series):
        return np.nan
    return safe_return(series.iloc[idx], series.iloc[idx + days])


def forward_max_drawdown(series: pd.Series, idx: int, days: int) -> float:
    if idx >= len(series):
        return np.nan
    end_idx = min(idx + days, len(series) - 1)
    window = series.iloc[idx : end_idx + 1].dropna()
    if len(window) < 2:
        return np.nan
    rel = window / window.iloc[0]
    return float((rel / rel.cummax() - 1.0).min())


def episode_to_date_flags(ep: pd.DataFrame) -> pd.DataFrame:
    out = ep.copy()
    out["episode_to_date_vix_peak"] = out["VIX_LEVEL"].cummax()
    out["episode_to_date_spy_low"] = out["SPY"].cummin()
    out["trigger_VIX_PEAK_PULLBACK_20"] = out["VIX_LEVEL"] <= 0.8 * out["episode_to_date_vix_peak"]
    out["trigger_SPY_REBOUND_FROM_LOW_3"] = out["SPY"] >= 1.03 * out["episode_to_date_spy_low"]
    out["trigger_SPY_ABOVE_MA10"] = out["SPY"] > out["SPY_MA10"]
    out["trigger_VIX_DOWN_SPY_UP_3D"] = (out["VIX_3D_CHANGE"] < 0) & (out["SPY_3D_RETURN"] > 0)
    out["trigger_CREDIT_NOT_WIDENING"] = out["CREDIT_5D_CHANGE"] <= 0
    trigger_cols = [
        "trigger_VIX_PEAK_PULLBACK_20",
        "trigger_SPY_REBOUND_FROM_LOW_3",
        "trigger_SPY_ABOVE_MA10",
        "trigger_VIX_DOWN_SPY_UP_3D",
        "trigger_CREDIT_NOT_WIDENING",
    ]
    out["trigger_COMBINED_RECOVERY"] = out[trigger_cols].fillna(False).sum(axis=1) >= 2
    return out


def first_true_date(ep: pd.DataFrame, col: str) -> tuple[pd.Timestamp | pd.NaT, int | float]:
    mask = ep[col].fillna(False)
    if not mask.any():
        return pd.NaT, np.nan
    idx = int(np.flatnonzero(mask.to_numpy())[0])
    return ep.iloc[idx]["date"], idx


def build_episode_record(ep: pd.DataFrame, episode_id: int, definition: str) -> dict[str, object]:
    ep = ep.copy().reset_index(drop=True)
    ep = episode_to_date_flags(ep)
    start_date = ep.loc[0, "date"]
    end_date = ep.loc[len(ep) - 1, "date"]
    vix_peak_idx = int(ep["VIX_LEVEL"].idxmax())
    vix_peak_pos = int(np.argmax(ep["VIX_LEVEL"].to_numpy()))
    spy_low_idx = int(ep["SPY"].idxmin())
    spy_low_pos = int(np.argmin(ep["SPY"].to_numpy()))
    stress_mask = ep["VIX_LEVEL"] >= 25
    episode_type = "STRESS_EPISODE" if bool(stress_mask.any()) else "WARNING_ONLY"
    stress_start_date = ep.loc[stress_mask, "date"].iloc[0] if bool(stress_mask.any()) else pd.NaT
    stress_end_date = ep.loc[stress_mask, "date"].iloc[-1] if bool(stress_mask.any()) else pd.NaT
    stress_duration = int(stress_mask.sum()) if bool(stress_mask.any()) else 0
    macro_start = ep.loc[0, "macro_regime_confirmed"]
    macro_peak = ep.loc[vix_peak_pos, "macro_regime_confirmed"]
    macro_low = ep.loc[spy_low_pos, "macro_regime_confirmed"]
    macro_majority = majority_regime(ep["macro_regime_confirmed"])

    rec = {
        "episode_id": episode_id,
        "definition_version": definition,
        "start_date": start_date,
        "end_date": end_date,
        "duration_days": int(len(ep)),
        "episode_type": episode_type,
        "vix_start": float(ep.loc[0, "VIX_LEVEL"]),
        "vix_peak": float(ep["VIX_LEVEL"].max()),
        "vix_peak_date": ep.loc[vix_peak_pos, "date"],
        "vix_end": float(ep.loc[len(ep) - 1, "VIX_LEVEL"]),
        "stress_start_date": stress_start_date,
        "stress_end_date": stress_end_date,
        "stress_duration_days": stress_duration,
        "macro_regime_at_start": macro_start,
        "macro_regime_at_vix_peak": macro_peak,
        "macro_regime_at_spy_low": macro_low,
        "macro_regime_majority": macro_majority,
        "spy_return_start_to_end": safe_return(ep.loc[0, "SPY"], ep.loc[len(ep) - 1, "SPY"]),
        "spy_return_start_to_vix_peak": safe_return(ep.loc[0, "SPY"], ep.loc[vix_peak_pos, "SPY"]),
        "spy_return_vix_peak_to_end": safe_return(ep.loc[vix_peak_pos, "SPY"], ep.loc[len(ep) - 1, "SPY"]),
        "spy_return_start_to_stress_start": safe_return(ep.loc[0, "SPY"], ep.loc[stress_mask.idxmax(), "SPY"]) if bool(stress_mask.any()) else np.nan,
        "spy_return_stress_start_to_end": safe_return(ep.loc[stress_mask.idxmax(), "SPY"], ep.loc[len(ep) - 1, "SPY"]) if bool(stress_mask.any()) else np.nan,
        "spy_max_drawdown_in_episode": float((ep["SPY"] / ep["SPY"].cummax() - 1.0).min()),
        "spy_drawdown_at_start": float(ep.loc[0, "SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"]),
        "spy_drawdown_at_vix_peak": float(ep.loc[vix_peak_pos, "SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"]),
        "spy_low_date": ep.loc[spy_low_pos, "date"],
        "spy_low_level": float(ep.loc[spy_low_pos, "SPY"]),
        "spy_low_before_vix_peak": bool(spy_low_pos < vix_peak_pos),
        "days_from_vix_peak_to_spy_low": int(spy_low_pos - vix_peak_pos),
        "days_from_spy_low_to_vix_peak": int(vix_peak_pos - spy_low_pos),
        "spy_return_low_to_end": safe_return(ep.loc[spy_low_pos, "SPY"], ep.loc[len(ep) - 1, "SPY"]),
        "spy_return_5d_after_vix_peak": forward_return(ep["SPY"], vix_peak_pos, 5),
        "spy_return_10d_after_vix_peak": forward_return(ep["SPY"], vix_peak_pos, 10),
        "spy_return_21d_after_vix_peak": forward_return(ep["SPY"], vix_peak_pos, 21),
        "spy_return_5d_after_spy_low": forward_return(ep["SPY"], spy_low_pos, 5),
        "spy_return_10d_after_spy_low": forward_return(ep["SPY"], spy_low_pos, 10),
        "spy_return_21d_after_spy_low": forward_return(ep["SPY"], spy_low_pos, 21),
    }
    return rec


def build_trigger_records(ep: pd.DataFrame, episode_rec: dict[str, object]) -> list[dict[str, object]]:
    ep = ep.copy().reset_index(drop=True)
    ep = episode_to_date_flags(ep)
    spy_low_pos = int(np.argmin(ep["SPY"].to_numpy()))
    vix_peak_pos = int(np.argmax(ep["VIX_LEVEL"].to_numpy()))
    trigger_map = {
        "VIX_PEAK_PULLBACK_20": "trigger_VIX_PEAK_PULLBACK_20",
        "SPY_REBOUND_FROM_LOW_3": "trigger_SPY_REBOUND_FROM_LOW_3",
        "SPY_ABOVE_MA10": "trigger_SPY_ABOVE_MA10",
        "VIX_DOWN_SPY_UP_3D": "trigger_VIX_DOWN_SPY_UP_3D",
        "CREDIT_NOT_WIDENING": "trigger_CREDIT_NOT_WIDENING",
        "COMBINED_RECOVERY": "trigger_COMBINED_RECOVERY",
    }
    rows: list[dict[str, object]] = []
    for trigger_type, col in trigger_map.items():
        trigger_date, pos = first_true_date(ep, col)
        if pd.isna(trigger_date):
            continue
        pos = int(pos)
        row = {
            "episode_id": episode_rec["episode_id"],
            "episode_type": episode_rec["episode_type"],
            "macro_regime_majority": episode_rec["macro_regime_majority"],
            "trigger_type": trigger_type,
            "trigger_date": trigger_date,
            "days_from_episode_start": pos,
            "days_from_vix_peak": pos - vix_peak_pos,
            "days_from_spy_low": pos - spy_low_pos,
            "vix_at_trigger": float(ep.loc[pos, "VIX_LEVEL"]),
            "spy_drawdown_at_trigger": float(ep.loc[pos, "SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"]),
            "spy_return_from_trigger_to_episode_end": safe_return(ep.loc[pos, "SPY"], ep.loc[len(ep) - 1, "SPY"]),
            "spy_forward_return_5d": forward_return(ep["SPY"], pos, 5),
            "spy_forward_return_10d": forward_return(ep["SPY"], pos, 10),
            "spy_forward_return_21d": forward_return(ep["SPY"], pos, 21),
            "spy_forward_max_drawdown_21d": forward_max_drawdown(ep["SPY"], pos, 21),
            "whether_trigger_before_spy_low": bool(pos < spy_low_pos),
        }
        rows.append(row)
    return rows


def analyze_episodes(panel: pd.DataFrame, end_threshold: float, definition: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_rows: list[dict[str, object]] = []
    trigger_rows: list[dict[str, object]] = []
    for episode_id, (start_idx, end_idx) in enumerate(detect_episodes(panel, end_threshold=end_threshold), start=1):
        ep = panel.iloc[start_idx : end_idx + 1].copy()
        rec = build_episode_record(ep, episode_id=episode_id, definition=definition)
        episode_rows.append(rec)
        trigger_rows.extend(build_trigger_records(ep, rec))
    return pd.DataFrame(episode_rows), pd.DataFrame(trigger_rows)


def summarize_by_macro(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for macro, grp in episodes.groupby("macro_regime_majority", observed=False):
        row = {
            "macro_regime_majority": macro,
            "episode_count": int(len(grp)),
            "warning_only_count": int((grp["episode_type"] == "WARNING_ONLY").sum()),
            "stress_episode_count": int((grp["episode_type"] == "STRESS_EPISODE").sum()),
            "warning_only_ratio": float((grp["episode_type"] == "WARNING_ONLY").mean()),
            "avg_duration": float(grp["duration_days"].mean()),
            "avg_vix_peak": float(grp["vix_peak"].mean()),
            "avg_spy_return_start_to_end": float(grp["spy_return_start_to_end"].mean()),
            "avg_spy_max_drawdown": float(grp["spy_max_drawdown_in_episode"].mean()),
            "median_spy_max_drawdown": float(grp["spy_max_drawdown_in_episode"].median()),
            "avg_days_from_vix_peak_to_spy_low": float(grp["days_from_vix_peak_to_spy_low"].mean()),
            "pct_spy_low_before_or_on_vix_peak": float((grp["days_from_vix_peak_to_spy_low"] <= 0).mean()),
            "avg_return_5d_after_vix_peak": float(grp["spy_return_5d_after_vix_peak"].mean()),
            "avg_return_10d_after_vix_peak": float(grp["spy_return_10d_after_vix_peak"].mean()),
            "avg_return_21d_after_vix_peak": float(grp["spy_return_21d_after_vix_peak"].mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows).set_index("macro_regime_majority").reindex(REGIME_ORDER).reset_index()


def summarize_by_macro_and_type(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (macro, ep_type), grp in episodes.groupby(["macro_regime_majority", "episode_type"], observed=False):
        row = {
            "macro_regime_majority": macro,
            "episode_type": ep_type,
            "episode_count": int(len(grp)),
            "avg_duration": float(grp["duration_days"].mean()),
            "avg_vix_peak": float(grp["vix_peak"].mean()),
            "avg_spy_return_start_to_end": float(grp["spy_return_start_to_end"].mean()),
            "avg_spy_max_drawdown": float(grp["spy_max_drawdown_in_episode"].mean()),
            "median_spy_max_drawdown": float(grp["spy_max_drawdown_in_episode"].median()),
            "avg_days_from_vix_peak_to_spy_low": float(grp["days_from_vix_peak_to_spy_low"].mean()),
            "pct_spy_low_before_or_on_vix_peak": float((grp["days_from_vix_peak_to_spy_low"] <= 0).mean()),
            "avg_return_5d_after_vix_peak": float(grp["spy_return_5d_after_vix_peak"].mean()),
            "avg_return_10d_after_vix_peak": float(grp["spy_return_10d_after_vix_peak"].mean()),
            "avg_return_21d_after_vix_peak": float(grp["spy_return_21d_after_vix_peak"].mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_trigger(triggers: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, grp in triggers.groupby(group_cols, observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "trigger_count": int(len(grp)),
                "avg_days_from_episode_start": float(grp["days_from_episode_start"].mean()),
                "avg_days_from_vix_peak": float(grp["days_from_vix_peak"].mean()),
                "avg_days_from_spy_low": float(grp["days_from_spy_low"].mean()),
                "pct_trigger_before_spy_low": float(grp["whether_trigger_before_spy_low"].mean()),
                "avg_forward_return_5d": float(grp["spy_forward_return_5d"].mean()),
                "avg_forward_return_10d": float(grp["spy_forward_return_10d"].mean()),
                "avg_forward_return_21d": float(grp["spy_forward_return_21d"].mean()),
                "avg_forward_max_drawdown_21d": float(grp["spy_forward_max_drawdown_21d"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def warning_vs_stress_by_macro(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (macro, ep_type), grp in episodes.groupby(["macro_regime_majority", "episode_type"], observed=False):
        rows.append(
            {
                "macro_regime_majority": macro,
                "episode_type": ep_type,
                "count": int(len(grp)),
                "avg_spy_return": float(grp["spy_return_start_to_end"].mean()),
                "avg_max_drawdown": float(grp["spy_max_drawdown_in_episode"].mean()),
                "avg_duration": float(grp["duration_days"].mean()),
                "avg_recovery_return": float(grp["spy_return_low_to_end"].mean()),
                "pct_mdd_worse_than_5pct": float((grp["spy_max_drawdown_in_episode"] <= -0.05).mean()),
                "pct_mdd_worse_than_10pct": float((grp["spy_max_drawdown_in_episode"] <= -0.10).mean()),
                "pct_mdd_worse_than_20pct": float((grp["spy_max_drawdown_in_episode"] <= -0.20).mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_episode_counts(summary_by_macro_type: pd.DataFrame) -> None:
    pivot = summary_by_macro_type.pivot(index="macro_regime_majority", columns="episode_type", values="episode_count").reindex(index=REGIME_ORDER, columns=EPISODE_TYPE_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=["#92c5de", "#ca0020"])
    ax.set_title("Episode Count by Macro Regime and Episode Type")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIG_COUNT_PATH, dpi=180)
    plt.close(fig)


def plot_macro_type_metric(summary_by_macro_type: pd.DataFrame, metric: str, title: str, path: Path, fmt: str = ".2%") -> None:
    pivot = summary_by_macro_type.pivot(index="macro_regime_majority", columns="episode_type", values=metric).reindex(index=REGIME_ORDER, columns=EPISODE_TYPE_ORDER)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap="RdBu_r", center=0 if "return" in metric else None, linewidths=0.5, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_peak_to_low_distribution(episodes: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(episodes["days_from_vix_peak_to_spy_low"].dropna(), bins=20, ax=axes[0], color="#4c78a8")
    axes[0].axvline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("Days From VIX Peak to SPY Low")
    axes[0].set_xlabel("SPY low day minus VIX peak day")
    data = [episodes.loc[episodes["macro_regime_majority"] == regime, "days_from_vix_peak_to_spy_low"].dropna() for regime in REGIME_ORDER]
    axes[1].boxplot(data, tick_labels=REGIME_ORDER, showfliers=False)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title("Days From VIX Peak to SPY Low by Macro Regime")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIG_PEAK_TO_LOW_PATH, dpi=180)
    plt.close(fig)


def plot_trigger_heatmaps(trigger_by_macro: pd.DataFrame) -> None:
    ret_pivot = trigger_by_macro.pivot(index="trigger_type", columns="macro_regime_majority", values="avg_forward_return_21d").reindex(index=TRIGGER_ORDER, columns=REGIME_ORDER)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(ret_pivot, annot=True, fmt=".2%", cmap="RdBu_r", center=0, linewidths=0.5, ax=ax)
    ax.set_title("Trigger Quality: Avg 21d Forward Return")
    fig.tight_layout()
    fig.savefig(FIG_TRIGGER_21D_PATH, dpi=180)
    plt.close(fig)

    early_pivot = trigger_by_macro.pivot(index="trigger_type", columns="macro_regime_majority", values="pct_trigger_before_spy_low").reindex(index=TRIGGER_ORDER, columns=REGIME_ORDER)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(early_pivot, annot=True, fmt=".0%", cmap="Blues", linewidths=0.5, ax=ax)
    ax.set_title("Trigger Early-Risk: Trigger Before SPY Low")
    fig.tight_layout()
    fig.savefig(FIG_TRIGGER_EARLY_PATH, dpi=180)
    plt.close(fig)


def event_study_from_anchor(panel: pd.DataFrame, anchors: pd.DataFrame, anchor_col: str, path: Path, title: str, window_before: int = 20, window_after: int = 40) -> None:
    series_rows = []
    price_map = panel.set_index("date")["SPY"]
    for _, row in anchors.iterrows():
        anchor_date = row[anchor_col]
        regime = row["macro_regime_majority"]
        if pd.isna(anchor_date) or anchor_date not in price_map.index:
            continue
        anchor_idx = price_map.index.get_loc(anchor_date)
        if isinstance(anchor_idx, slice) or isinstance(anchor_idx, np.ndarray):
            continue
        start = max(0, anchor_idx - window_before)
        end = min(len(price_map) - 1, anchor_idx + window_after)
        window = price_map.iloc[start : end + 1]
        rel_days = np.arange(start - anchor_idx, end - anchor_idx + 1)
        norm = window / price_map.iloc[anchor_idx]
        tmp = pd.DataFrame({"rel_day": rel_days, "norm_spy": norm.to_numpy(), "macro_regime_majority": regime})
        series_rows.append(tmp)
    if not series_rows:
        return
    study = pd.concat(series_rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for regime in REGIME_ORDER:
        grp = study.loc[study["macro_regime_majority"] == regime]
        if grp.empty:
            continue
        avg = grp.groupby("rel_day")["norm_spy"].mean()
        ax.plot(avg.index, avg.values, label=regime, color=REGIME_COLORS[regime])
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Trading days relative to anchor")
    ax.set_ylabel("Normalized SPY")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_selected_timelines(panel: pd.DataFrame, episodes: pd.DataFrame, triggers: pd.DataFrame) -> None:
    for year in TIMELINE_YEARS:
        sub = episodes.loc[(episodes["start_date"].dt.year <= year) & (episodes["end_date"].dt.year >= year)]
        if sub.empty:
            continue
        ep = sub.sort_values("start_date").iloc[0]
        start = ep["start_date"] - pd.Timedelta(days=20)
        end = ep["end_date"] + pd.Timedelta(days=30)
        view = panel.loc[(panel["date"] >= start) & (panel["date"] <= end)].copy()
        if view.empty:
            continue
        fig, ax1 = plt.subplots(figsize=(12, 5))
        norm_spy = view["SPY"] / view["SPY"].iloc[0]
        ax1.plot(view["date"], norm_spy, color="tab:blue", label="SPY normalized")
        ax1.set_ylabel("SPY normalized")
        ax2 = ax1.twinx()
        ax2.plot(view["date"], view["VIX_LEVEL"], color="tab:red", alpha=0.7, label="VIX")
        ax2.set_ylabel("VIX")
        ymax = float(view["VIX_LEVEL"].max())
        for regime in REGIME_ORDER:
            mask = view["macro_regime_confirmed"] == regime
            ax2.fill_between(view["date"], 0, ymax, where=mask, alpha=0.10, color=REGIME_COLORS[regime])
        for date_col, color, label in [
            ("start_date", "black", "episode_start"),
            ("vix_peak_date", "red", "vix_peak"),
            ("spy_low_date", "blue", "spy_low"),
        ]:
            ax1.axvline(ep[date_col], color=color, linestyle="--", linewidth=1.0, label=label)
        trg = triggers.loc[(triggers["episode_id"] == ep["episode_id"]) & (triggers["trigger_type"] == "COMBINED_RECOVERY")]
        if not trg.empty:
            ax1.axvline(trg.iloc[0]["trigger_date"], color="green", linestyle=":", linewidth=1.2, label="combined_recovery")
        ax1.set_title(f"SPY/VIX Episode Timeline {year}")
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"selected_episode_timeline_{year}.png", dpi=180)
        plt.close(fig)


def concise_answers(by_macro: pd.DataFrame, by_macro_type: pd.DataFrame, trigger_summary: pd.DataFrame, trigger_by_macro: pd.DataFrame) -> list[str]:
    def get_warning_ratio(regime: str) -> float:
        row = by_macro.loc[by_macro["macro_regime_majority"] == regime]
        return float(row["warning_only_ratio"].iloc[0]) if not row.empty else np.nan

    warning_ratios = ", ".join(
        f"{regime}={get_warning_ratio(regime):.0%}" for regime in ["FLAT", "STEEP", "INVERTED"] if pd.notna(get_warning_ratio(regime))
    )
    high_warning_dd = by_macro_type.loc[
        (by_macro_type["episode_type"] == "WARNING_ONLY") & (by_macro_type["avg_spy_max_drawdown"] <= -0.05),
        "macro_regime_majority",
    ].tolist()
    low_near_peak = by_macro.loc[by_macro["pct_spy_low_before_or_on_vix_peak"] >= 0.5, "macro_regime_majority"].tolist()
    best_trigger = trigger_summary.sort_values(["avg_forward_return_21d", "avg_forward_max_drawdown_21d"], ascending=[False, False]).iloc[0] if not trigger_summary.empty else None
    early_trigger = trigger_summary.sort_values("pct_trigger_before_spy_low", ascending=False).iloc[0] if not trigger_summary.empty else None

    support_flat = by_macro_type.loc[(by_macro_type["macro_regime_majority"] == "FLAT") & (by_macro_type["episode_type"] == "WARNING_ONLY"), "avg_spy_max_drawdown"]
    support_steep = by_macro_type.loc[(by_macro_type["macro_regime_majority"] == "STEEP") & (by_macro_type["episode_type"] == "WARNING_ONLY"), "avg_spy_max_drawdown"]
    support_inverted = by_macro_type.loc[(by_macro_type["macro_regime_majority"] == "INVERTED") & (by_macro_type["episode_type"] == "WARNING_ONLY"), "avg_spy_max_drawdown"]
    lines = [
        f"1. Warning-only 占比：{warning_ratios}",
        f"2. Warning 阶段已出现明显回撤的 regime：{', '.join(high_warning_dd) if high_warning_dd else '没有明显支持'}",
        f"3. SPY 低点通常发生在 VIX peak 之前或附近的 regime：{', '.join(low_near_peak) if low_near_peak else '没有明显集中'}",
        f"4. 21d forward return 最稳定的 trigger：{best_trigger['trigger_type'] if best_trigger is not None else 'N/A'}",
        f"5. 最容易过早触发的 trigger：{early_trigger['trigger_type'] if early_trigger is not None else 'N/A'}",
        "6. 规则含义："
        f" FLAT warning 减仓={'支持' if not support_flat.empty and float(support_flat.iloc[0]) <= -0.05 else '证据弱'}；"
        f" STEEP warning 快速减仓但 recovery 更早加仓={'支持' if not support_steep.empty and float(support_steep.iloc[0]) <= -0.05 else '部分支持/需结合 trigger'}；"
        f" INVERTED 不必过度快速清仓={'支持' if not support_inverted.empty and float(support_inverted.iloc[0]) > -0.05 else '证据弱'}。",
        "7. 关于 exit 逻辑：主分析已使用 hysteresis（<19 退出 warning）；若 trigger 普遍在 SPY 低点附近或之后出现，说明可以研究放松三日确认，或改成 stress <24 / warning <19 的 exit，但这一步仍需放到下一轮策略回测验证。",
    ]
    return lines


def main() -> None:
    ensure_dirs()
    panel, panel_path = build_base_panel()
    episodes, triggers = analyze_episodes(panel, end_threshold=19.0, definition="hysteresis_exit_lt_19")
    episodes_simple, _ = analyze_episodes(panel, end_threshold=20.0, definition="simple_exit_lt_20")

    by_macro = summarize_by_macro(episodes)
    by_macro_type = summarize_by_macro_and_type(episodes)
    trigger_summary = summarize_trigger(triggers, ["trigger_type"]).set_index("trigger_type").reindex(TRIGGER_ORDER).reset_index()
    trigger_by_macro = summarize_trigger(triggers, ["trigger_type", "macro_regime_majority"])
    warning_stress = warning_vs_stress_by_macro(episodes)

    episodes.to_csv(EPISODE_PATH, index=False)
    episodes_simple.to_csv(EPISODE_PATH_SIMPLE, index=False)
    triggers.to_csv(TRIGGER_EVENTS_PATH, index=False)
    by_macro.to_csv(EPISODE_BY_MACRO_PATH, index=False)
    by_macro_type.to_csv(EPISODE_BY_MACRO_TYPE_PATH, index=False)
    trigger_summary.to_csv(TRIGGER_SUMMARY_PATH, index=False)
    trigger_by_macro.to_csv(TRIGGER_BY_MACRO_PATH, index=False)
    warning_stress.to_csv(WARNING_STRESS_PATH, index=False)

    plot_episode_counts(by_macro_type)
    plot_macro_type_metric(by_macro_type, "avg_spy_max_drawdown", "Average SPY Max Drawdown by Macro Regime and Episode Type", FIG_MDD_PATH)
    plot_macro_type_metric(by_macro_type, "avg_spy_return_start_to_end", "Average SPY Return Start-to-End by Macro Regime and Episode Type", FIG_RET_PATH)
    plot_peak_to_low_distribution(episodes)
    plot_trigger_heatmaps(trigger_by_macro)
    event_study_from_anchor(panel, episodes, "vix_peak_date", FIG_EVENT_VIX_PEAK_PATH, "Event Study Around VIX Peak")
    combined = triggers.loc[triggers["trigger_type"] == "COMBINED_RECOVERY"].copy()
    event_study_from_anchor(panel, combined, "trigger_date", FIG_EVENT_TRIGGER_PATH, "Event Study Around Combined Recovery Trigger")
    plot_selected_timelines(panel, episodes, triggers)

    print(f"Input panel: {panel_path}")
    print(f"Episode count (hysteresis): {len(episodes)}")
    print(f"Episode count (simple exit): {len(episodes_simple)}")
    for line in concise_answers(by_macro, by_macro_type, trigger_summary, trigger_by_macro):
        print(line)
    for path in [
        EPISODE_PATH,
        EPISODE_PATH_SIMPLE,
        EPISODE_BY_MACRO_PATH,
        EPISODE_BY_MACRO_TYPE_PATH,
        TRIGGER_EVENTS_PATH,
        TRIGGER_SUMMARY_PATH,
        TRIGGER_BY_MACRO_PATH,
        WARNING_STRESS_PATH,
        FIG_COUNT_PATH,
        FIG_MDD_PATH,
        FIG_RET_PATH,
        FIG_PEAK_TO_LOW_PATH,
        FIG_TRIGGER_21D_PATH,
        FIG_TRIGGER_EARLY_PATH,
        FIG_EVENT_VIX_PEAK_PATH,
        FIG_EVENT_TRIGGER_PATH,
    ]:
        if path.exists():
            print(f"Saved: {path}")
    for year in TIMELINE_YEARS:
        path = RESULTS_DIR / f"selected_episode_timeline_{year}.png"
        if path.exists():
            print(f"Saved: {path}")


if __name__ == "__main__":
    main()
