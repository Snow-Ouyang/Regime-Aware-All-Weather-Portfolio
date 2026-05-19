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

RESULTS_DIR = ROOT / "results" / "stress_recovery_price_trigger_diagnostic"
FIGURES_DIR = ROOT / "figures" / "stress_recovery_price_trigger_diagnostic"

DAILY_PANEL_OUT = RESULTS_DIR / "stress_recovery_daily_panel.csv"
EPISODE_OUT = RESULTS_DIR / "stress_episode_summary.csv"
TRIGGER_EVENT_OUT = RESULTS_DIR / "recovery_trigger_event_table.csv"
TRIGGER_SUMMARY_OUT = RESULTS_DIR / "recovery_trigger_summary.csv"
TRIGGER_STRESS_TYPE_OUT = RESULTS_DIR / "recovery_trigger_summary_by_stress_type.csv"
TRIGGER_REGIME_OUT = RESULTS_DIR / "recovery_trigger_summary_by_regime.csv"
COVID_OUT = RESULTS_DIR / "covid_recovery_trigger_case_study.csv"
RANKING_OUT = RESULTS_DIR / "recovery_trigger_ranking.csv"
REPORT_OUT = RESULTS_DIR / "STRESS_RECOVERY_PRICE_TRIGGER_DIAGNOSTIC.md"

FIG_SCATTER = FIGURES_DIR / "recovery_trigger_quality_scatter.png"
FIG_BAR = FIGURES_DIR / "recovery_trigger_summary_bar.png"
FIG_HEAT = FIGURES_DIR / "recovery_trigger_by_stress_type_heatmap.png"
FIG_COVID = FIGURES_DIR / "covid_recovery_case_study.png"
FIG_TIMELINE = FIGURES_DIR / "stress_episode_timeline.png"
FIG_DAYS = FIGURES_DIR / "days_after_trough_distribution.png"

CONFIG = {
    "vix_z_window": 120,
    "vix_z_threshold": 3.0,
    "stress_cooldown_days": 21,
    "enabled_vix_regimes": ["FLAT", "STEEP"],
    "enabled_either_sell_regimes": ["STEEP"],
    "forward_windows": [5, 10, 21, 42, 63],
    "covid_start": "2020-02-19",
    "covid_end": "2020-04-30",
}

RECOVERY_TRIGGERS = [
    "R1_SPY_GT_MA10",
    "R2_SPY_GT_MA20",
    "R3_SPY_CROSS_ABOVE_MA20",
    "R4_REBOUND_20D_LOW_3",
    "R5_REBOUND_20D_LOW_5",
    "R6_REBOUND_60D_LOW_5",
    "R7_MA20_AND_REBOUND_20D_LOW_3",
    "R8_SPY_5D_RETURN_GT_3",
    "R9_SPY_10D_RETURN_GT_5",
]

REGIME_COLORS = {
    "HIGH_INFLATION": "#d95f02",
    "INVERTED": "#7570b3",
    "FLAT": "#1b9e77",
    "STEEP": "#66a61e",
    "NEUTRAL": "#999999",
}
STRESS_COLORS = {
    "FLAT_VIX_STRESS": "#e41a1c",
    "STEEP_VIX_STRESS": "#ff7f00",
    "STEEP_EITHER_STRESS": "#377eb8",
    "STEEP_COMBINED_STRESS": "#984ea3",
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
    vix_col = _first_col(df, ["VIXCLS", "VIX_LEVEL"])
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
            [(credit > 1.5) & (dgs1 > 5.0), term < 0.0, (term >= 0.0) & (term < 1.0), term >= 1.0],
            ["HIGH_INFLATION", "INVERTED", "FLAT", "STEEP"],
            default="NEUTRAL",
        ),
        index=df.index,
    )


def load_data() -> pd.DataFrame:
    panel = None
    for path in INPUT_CANDIDATES:
        df = _read_panel(path)
        if df is None or df.empty:
            continue
        if _first_col(df, ["spy_price", "SPY_PRICE"]) is not None or _first_col(df, ["SPY_RETURN", "SPY_ret", "spy_daily_return"]) is not None:
            panel = df
            break
    if panel is None:
        raise FileNotFoundError("No usable daily panel with SPY price/returns found.")

    price_col = _first_col(panel, ["spy_price", "SPY_PRICE", "SPY"])
    ret_col = _first_col(panel, ["SPY_RETURN", "SPY_ret", "spy_daily_return", "SPY_BUY_HOLD_return"])
    if price_col is not None:
        panel["spy_price"] = pd.to_numeric(panel[price_col], errors="coerce")
        panel["spy_daily_return"] = panel["spy_price"].pct_change()
        if ret_col is not None:
            panel["spy_daily_return"] = pd.to_numeric(panel[ret_col], errors="coerce").combine_first(panel["spy_daily_return"])
    elif ret_col is not None:
        panel["spy_daily_return"] = pd.to_numeric(panel[ret_col], errors="coerce")
        panel["spy_price"] = (1.0 + panel["spy_daily_return"].fillna(0.0)).cumprod()
    else:
        raise ValueError("Missing SPY price or return column.")

    vix_col = _first_col(panel, ["VIX_LEVEL", "VIXCLS"])
    if vix_col is not None:
        panel["VIX_LEVEL"] = pd.to_numeric(panel[vix_col], errors="coerce")
    else:
        vix_sources = []
        for path in [RECON_PANEL, RULE_PANEL]:
            vix_panel = _read_panel(path)
            if vix_panel is not None and "VIX_LEVEL" in vix_panel.columns:
                vix_sources.append(vix_panel[["date", "VIX_LEVEL"]])
        raw = _load_raw_vix()
        if raw is not None:
            vix_sources.append(raw)
        if not vix_sources:
            raise ValueError("Missing VIX_LEVEL and no raw VIXCLS.csv was found.")
        vix = pd.concat(vix_sources, ignore_index=True).sort_values("date").drop_duplicates("date")
        panel = panel.merge(vix, on="date", how="left")

    if "macro_regime_confirmed" not in panel.columns:
        for path in [RECON_PANEL, RULE_PANEL]:
            reg = _read_panel(path)
            if reg is None:
                continue
            keep = ["date"] + [c for c in ["macro_regime_confirmed", "CREDIT_SPREAD_BAA_AAA", "DGS1", "TERM_SPREAD_10Y_1Y"] if c in reg.columns]
            reg = reg[keep].copy()
            if "macro_regime_confirmed" not in reg.columns and {"CREDIT_SPREAD_BAA_AAA", "DGS1", "TERM_SPREAD_10Y_1Y"}.issubset(reg.columns):
                reg["macro_regime_confirmed"] = _rebuild_macro_regime(reg)
            if "macro_regime_confirmed" in reg.columns:
                panel = panel.merge(reg[["date", "macro_regime_confirmed"]], on="date", how="left")
                break
    if "macro_regime_confirmed" not in panel.columns:
        raise ValueError("Missing macro_regime_confirmed and could not rebuild it.")

    if "monthly_either_state" not in panel.columns:
        w_col = _first_col(panel, ["monthly_either_weight_spy", "MONTHLY_EITHER_CONFIRM_weight_spy"])
        if w_col is None:
            raise ValueError("Missing monthly_either_state and monthly_either_weight_spy.")
        panel["monthly_either_state"] = np.where(pd.to_numeric(panel[w_col], errors="coerce") >= 0.5, "HOLD", "SELL")

    out = panel.dropna(subset=["spy_price", "VIX_LEVEL", "macro_regime_confirmed"]).sort_values("date").reset_index(drop=True)
    out["spy_daily_return"] = out["spy_daily_return"].fillna(out["spy_price"].pct_change())
    out["previous_high"] = out["spy_price"].cummax()
    out["spy_drawdown_from_previous_high"] = out["spy_price"] / out["previous_high"] - 1.0
    return out


def build_vix_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    w = CONFIG["vix_z_window"]
    mean = out["VIX_LEVEL"].rolling(w, min_periods=w).mean()
    std = out["VIX_LEVEL"].rolling(w, min_periods=w).std(ddof=1)
    out["VIX_zscore_120d"] = (out["VIX_LEVEL"] - mean) / std.replace(0.0, np.nan)
    return out


def build_price_recovery_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["SPY_MA10"] = out["spy_price"].rolling(10, min_periods=10).mean()
    out["SPY_MA20"] = out["spy_price"].rolling(20, min_periods=20).mean()
    out["SPY_MA50"] = out["spy_price"].rolling(50, min_periods=50).mean()
    out["SPY_20D_LOW"] = out["spy_price"].rolling(20, min_periods=1).min()
    out["SPY_60D_LOW"] = out["spy_price"].rolling(60, min_periods=1).min()
    out["SPY_5D_RETURN"] = out["spy_price"] / out["spy_price"].shift(5) - 1.0
    out["SPY_10D_RETURN"] = out["spy_price"] / out["spy_price"].shift(10) - 1.0
    out["SPY_REBOUND_FROM_20D_LOW"] = out["spy_price"] / out["SPY_20D_LOW"] - 1.0
    out["SPY_REBOUND_FROM_60D_LOW"] = out["spy_price"] / out["SPY_60D_LOW"] - 1.0
    out["SPY_CROSS_ABOVE_MA20"] = (out["spy_price"] > out["SPY_MA20"]) & (out["spy_price"].shift(1) <= out["SPY_MA20"].shift(1))
    return out


def build_active_stress_signal(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    raw_trigger = out["VIX_zscore_120d"] >= CONFIG["vix_z_threshold"]
    crossing = raw_trigger & ~raw_trigger.shift(1, fill_value=False)
    out["VIX_STRESS_TRIGGER_ANY_REGIME"] = crossing
    enabled_regime = out["macro_regime_confirmed"].isin(CONFIG["enabled_vix_regimes"])
    valid_vix_trigger = crossing & enabled_regime
    valid_vix_level = raw_trigger & enabled_regime
    out["VIX_STRESS_TRIGGER_ACTIVE_REGIME"] = valid_vix_trigger

    active_until = -1
    active = []
    for i, (trig, level_active) in enumerate(zip(valid_vix_trigger.to_numpy(), valid_vix_level.to_numpy())):
        # A crossing starts a stress episode. While the episode is active, any
        # continued z-score breach extends the episode another cooldown window.
        if trig or (level_active and i <= active_until):
            active_until = max(active_until, i + CONFIG["stress_cooldown_days"])
        active.append(i <= active_until)
    out["VIX_STRESS_ACTIVE"] = active
    out["EITHER_STRESS_ACTIVE"] = out["macro_regime_confirmed"].isin(CONFIG["enabled_either_sell_regimes"]) & out["monthly_either_state"].eq("SELL")

    flat = out["macro_regime_confirmed"].eq("FLAT")
    steep = out["macro_regime_confirmed"].eq("STEEP")
    out["active_stress"] = (flat & out["VIX_STRESS_ACTIVE"]) | (steep & (out["VIX_STRESS_ACTIVE"] | out["EITHER_STRESS_ACTIVE"]))

    out["stress_type"] = "NO_ACTIVE_STRESS"
    out.loc[flat & out["VIX_STRESS_ACTIVE"], "stress_type"] = "FLAT_VIX_STRESS"
    out.loc[steep & out["VIX_STRESS_ACTIVE"] & ~out["EITHER_STRESS_ACTIVE"], "stress_type"] = "STEEP_VIX_STRESS"
    out.loc[steep & ~out["VIX_STRESS_ACTIVE"] & out["EITHER_STRESS_ACTIVE"], "stress_type"] = "STEEP_EITHER_STRESS"
    out.loc[steep & out["VIX_STRESS_ACTIVE"] & out["EITHER_STRESS_ACTIVE"], "stress_type"] = "STEEP_COMBINED_STRESS"
    return out


def _forward_metrics(panel: pd.DataFrame, idx: int, prefix: str = "SPY_forward") -> dict[str, float]:
    prices = panel["spy_price"].reset_index(drop=True)
    out: dict[str, float] = {}
    for h in CONFIG["forward_windows"]:
        end = min(idx + h, len(prices) - 1)
        path = prices.iloc[idx : end + 1].astype(float)
        base = float(path.iloc[0]) if len(path) else np.nan
        if len(path) < 2 or not np.isfinite(base) or base <= 0:
            ret = mdd = runup = np.nan
        else:
            ret = float(path.iloc[-1] / base - 1.0) if end == idx + h else np.nan
            wealth = path / base
            mdd = float((wealth / wealth.cummax() - 1.0).min())
            runup = float((path / base - 1.0).max())
        out[f"{prefix}_return_{h}d"] = ret
        out[f"{prefix}_mdd_{h}d"] = mdd
        out[f"{prefix}_max_runup_{h}d"] = runup
    return out


def extract_stress_episodes(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    active = panel["active_stress"].fillna(False).to_numpy()
    i = 0
    eid = 0
    while i < len(panel):
        if not active[i]:
            i += 1
            continue
        start = i
        while i + 1 < len(panel) and active[i + 1]:
            i += 1
        end = i
        sub = panel.iloc[start : end + 1].copy()
        eid += 1
        prices = sub["spy_price"]
        start_price = float(prices.iloc[0])
        rel = prices / start_price - 1.0
        trough_pos = int(np.argmin(prices.to_numpy()))
        vix_pos = int(np.argmax(sub["VIX_LEVEL"].to_numpy()))
        dominant_stress = sub["stress_type"].mode().iloc[0]
        dominant_regime = sub["macro_regime_confirmed"].mode().iloc[0]
        episode = {
            "stress_episode_id": eid,
            "stress_start_date": sub["date"].iloc[0],
            "stress_end_date": sub["date"].iloc[-1],
            "stress_duration_days": int(len(sub)),
            "stress_type_at_start": sub["stress_type"].iloc[0],
            "dominant_stress_type": dominant_stress,
            "macro_regime_at_start": sub["macro_regime_confirmed"].iloc[0],
            "dominant_macro_regime": dominant_regime,
            "monthly_either_state_at_start": sub["monthly_either_state"].iloc[0],
            "VIX_LEVEL_at_start": float(sub["VIX_LEVEL"].iloc[0]),
            "VIX_zscore_120d_at_start": float(sub["VIX_zscore_120d"].iloc[0]),
            "SPY_price_at_start": start_price,
            "SPY_drawdown_from_previous_high_at_start": float(sub["spy_drawdown_from_previous_high"].iloc[0]),
            "SPY_return_start_to_end": float(prices.iloc[-1] / start_price - 1.0),
            "SPY_max_drawdown_in_episode": float(((prices / start_price) / (prices / start_price).cummax() - 1.0).min()),
            "SPY_min_return_from_start": float(rel.min()),
            "SPY_max_runup_in_episode": float(rel.max()),
            "SPY_trough_date": sub["date"].iloc[trough_pos],
            "days_from_stress_start_to_trough": trough_pos,
            "VIX_peak_in_episode": float(sub["VIX_LEVEL"].iloc[vix_pos]),
            "VIX_peak_date": sub["date"].iloc[vix_pos],
            "days_from_stress_start_to_vix_peak": vix_pos,
            "start_index": start,
            "end_index": end,
        }
        episode["SPY_return_21d_after_stress_start"] = _forward_metrics(panel, start)["SPY_forward_return_21d"]
        episode["SPY_return_63d_after_stress_start"] = _forward_metrics(panel, start)["SPY_forward_return_63d"]
        episode["SPY_forward_mdd_21d_after_stress_start"] = _forward_metrics(panel, start)["SPY_forward_mdd_21d"]
        episode["SPY_forward_mdd_63d_after_stress_start"] = _forward_metrics(panel, start)["SPY_forward_mdd_63d"]
        rows.append(episode)
        i += 1
    return pd.DataFrame(rows)


def _trigger_columns(panel: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=panel.index)
    out["R1_SPY_GT_MA10"] = panel["spy_price"] > panel["SPY_MA10"]
    out["R2_SPY_GT_MA20"] = panel["spy_price"] > panel["SPY_MA20"]
    out["R3_SPY_CROSS_ABOVE_MA20"] = panel["SPY_CROSS_ABOVE_MA20"]
    out["R4_REBOUND_20D_LOW_3"] = panel["SPY_REBOUND_FROM_20D_LOW"] >= 0.03
    out["R5_REBOUND_20D_LOW_5"] = panel["SPY_REBOUND_FROM_20D_LOW"] >= 0.05
    out["R6_REBOUND_60D_LOW_5"] = panel["SPY_REBOUND_FROM_60D_LOW"] >= 0.05
    out["R7_MA20_AND_REBOUND_20D_LOW_3"] = (panel["spy_price"] > panel["SPY_MA20"]) & (panel["SPY_REBOUND_FROM_20D_LOW"] >= 0.03)
    out["R8_SPY_5D_RETURN_GT_3"] = panel["SPY_5D_RETURN"] > 0.03
    out["R9_SPY_10D_RETURN_GT_5"] = panel["SPY_10D_RETURN"] > 0.05
    return out.fillna(False)


def evaluate_recovery_triggers(panel: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    trig_df = _trigger_columns(panel)
    rows = []
    for _, ep in episodes.iterrows():
        start = int(ep["start_index"])
        end = int(ep["end_index"])
        sub_idx = list(range(start, end + 1))
        trough_idx = start + int(ep["days_from_stress_start_to_trough"])
        for trigger in RECOVERY_TRIGGERS:
            vals = trig_df.loc[sub_idx, trigger]
            found = bool(vals.any())
            if found:
                trig_idx = int(vals[vals].index[0])
                row = panel.iloc[trig_idx]
                days_after_trough = trig_idx - trough_idx
                event = {
                    "stress_episode_id": ep["stress_episode_id"],
                    "trigger_name": trigger,
                    "trigger_date": row["date"],
                    "trigger_found": True,
                    "trigger_at_stress_start": trig_idx == start,
                    "stress_type": ep["dominant_stress_type"],
                    "macro_regime_at_start": ep["macro_regime_at_start"],
                    "dominant_macro_regime": ep["dominant_macro_regime"],
                    "days_after_stress_start": trig_idx - start,
                    "days_after_trough": days_after_trough,
                    "trigger_before_trough": trig_idx < trough_idx,
                    "SPY_price_at_trigger": float(row["spy_price"]),
                    "SPY_drawdown_from_previous_high_at_trigger": float(row["spy_drawdown_from_previous_high"]),
                    "SPY_return_from_stress_start_to_trigger": float(row["spy_price"] / ep["SPY_price_at_start"] - 1.0),
                    "VIX_LEVEL_at_trigger": float(row["VIX_LEVEL"]),
                    "VIX_zscore_120d_at_trigger": float(row["VIX_zscore_120d"]),
                }
                event.update(_forward_metrics(panel, trig_idx))
            else:
                event = {
                    "stress_episode_id": ep["stress_episode_id"],
                    "trigger_name": trigger,
                    "trigger_date": pd.NaT,
                    "trigger_found": False,
                    "trigger_at_stress_start": False,
                    "stress_type": ep["dominant_stress_type"],
                    "macro_regime_at_start": ep["macro_regime_at_start"],
                    "dominant_macro_regime": ep["dominant_macro_regime"],
                    "days_after_stress_start": np.nan,
                    "days_after_trough": np.nan,
                    "trigger_before_trough": np.nan,
                }
                for h in CONFIG["forward_windows"]:
                    event[f"SPY_forward_return_{h}d"] = np.nan
                    event[f"SPY_forward_mdd_{h}d"] = np.nan
                    event[f"SPY_forward_max_runup_{h}d"] = np.nan
            rows.append(event)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["positive_21d"] = out["SPY_forward_return_21d"] > 0
        out["positive_63d"] = out["SPY_forward_return_63d"] > 0
        out["bad_reentry_21d"] = out["SPY_forward_mdd_21d"] <= -0.05
        out["bad_reentry_63d"] = out["SPY_forward_mdd_63d"] <= -0.10
        out["too_early"] = out["trigger_before_trough"] == True
        out["too_late"] = out["days_after_trough"] > 21
    return out


def _summary(grp: pd.DataFrame) -> dict[str, float]:
    found = grp.loc[grp["trigger_found"]].copy()
    n_ep = int(grp["stress_episode_id"].nunique())
    return {
        "stress_episode_count": n_ep,
        "trigger_found_count": int(len(found)),
        "trigger_coverage_rate": float(len(found) / n_ep) if n_ep else np.nan,
        "avg_days_after_stress_start": float(found["days_after_stress_start"].mean()) if not found.empty else np.nan,
        "median_days_after_stress_start": float(found["days_after_stress_start"].median()) if not found.empty else np.nan,
        "avg_days_after_trough": float(found["days_after_trough"].mean()) if not found.empty else np.nan,
        "median_days_after_trough": float(found["days_after_trough"].median()) if not found.empty else np.nan,
        "pct_trigger_before_trough": float(found["trigger_before_trough"].mean()) if not found.empty else np.nan,
        "pct_trigger_at_stress_start": float(found["trigger_at_stress_start"].mean()) if not found.empty else np.nan,
        "pct_too_late": float(found["too_late"].mean()) if not found.empty else np.nan,
        "avg_forward_return_21d": float(found["SPY_forward_return_21d"].mean()) if not found.empty else np.nan,
        "avg_forward_return_63d": float(found["SPY_forward_return_63d"].mean()) if not found.empty else np.nan,
        "median_forward_return_21d": float(found["SPY_forward_return_21d"].median()) if not found.empty else np.nan,
        "median_forward_return_63d": float(found["SPY_forward_return_63d"].median()) if not found.empty else np.nan,
        "pct_positive_21d": float(found["positive_21d"].mean()) if not found.empty else np.nan,
        "pct_positive_63d": float(found["positive_63d"].mean()) if not found.empty else np.nan,
        "avg_forward_mdd_21d": float(found["SPY_forward_mdd_21d"].mean()) if not found.empty else np.nan,
        "avg_forward_mdd_63d": float(found["SPY_forward_mdd_63d"].mean()) if not found.empty else np.nan,
        "pct_bad_reentry_21d": float(found["bad_reentry_21d"].mean()) if not found.empty else np.nan,
        "pct_bad_reentry_63d": float(found["bad_reentry_63d"].mean()) if not found.empty else np.nan,
    }


def summarize_recovery_triggers(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.DataFrame([{"trigger_name": k, **_summary(g)} for k, g in events.groupby("trigger_name")])
    by_type = pd.DataFrame([{"trigger_name": k[0], "dominant_stress_type": k[1], **_summary(g)} for k, g in events.groupby(["trigger_name", "stress_type"])])
    by_regime = pd.DataFrame([{"trigger_name": k[0], "dominant_macro_regime": k[1], **_summary(g)} for k, g in events.groupby(["trigger_name", "dominant_macro_regime"])])
    return summary, by_type, by_regime


def analyze_covid_case(episodes: pd.DataFrame, trigger_events: pd.DataFrame) -> pd.DataFrame:
    start = pd.Timestamp(CONFIG["covid_start"])
    end = pd.Timestamp(CONFIG["covid_end"])
    covid_eps = episodes.loc[(episodes["stress_start_date"] <= end) & (episodes["stress_end_date"] >= start)]
    rows = []
    for _, ep in covid_eps.iterrows():
        row = {
            "stress_episode_id": ep["stress_episode_id"],
            "stress_start_date": ep["stress_start_date"],
            "stress_end_date": ep["stress_end_date"],
            "stress_type": ep["dominant_stress_type"],
            "SPY_trough_date": ep["SPY_trough_date"],
            "VIX_peak_date": ep["VIX_peak_date"],
        }
        ev = trigger_events.loc[trigger_events["stress_episode_id"] == ep["stress_episode_id"]]
        for trig in RECOVERY_TRIGGERS:
            t = ev.loc[ev["trigger_name"] == trig]
            if t.empty:
                continue
            r = t.iloc[0]
            prefix = trig
            row[f"{prefix}_trigger_date"] = r["trigger_date"]
            row[f"{prefix}_days_after_stress_start"] = r["days_after_stress_start"]
            row[f"{prefix}_days_after_trough"] = r["days_after_trough"]
            row[f"{prefix}_trigger_before_trough"] = r["trigger_before_trough"]
            row[f"{prefix}_SPY_forward_return_21d"] = r["SPY_forward_return_21d"]
            row[f"{prefix}_SPY_forward_mdd_21d"] = r["SPY_forward_mdd_21d"]
            row[f"{prefix}_SPY_forward_return_63d"] = r["SPY_forward_return_63d"]
            row[f"{prefix}_SPY_forward_mdd_63d"] = r["SPY_forward_mdd_63d"]
        rows.append(row)
    return pd.DataFrame(rows)


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def rank_recovery_triggers(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["eligible"] = (out["trigger_coverage_rate"] >= 0.6) & (out["pct_trigger_before_trough"] <= 0.5) & (out["pct_bad_reentry_21d"] <= 0.5)
    elig = out.loc[out["eligible"]].copy()
    if elig.empty:
        out["composite_score"] = np.nan
        return out.sort_values("avg_forward_return_21d", ascending=False)
    elig["score_ret21"] = _minmax(elig["avg_forward_return_21d"])
    elig["score_ret63"] = _minmax(elig["avg_forward_return_63d"])
    elig["days_penalty"] = (elig["median_days_after_trough"] - 7).abs() / 28.0
    elig["composite_score"] = (
        0.25 * elig["pct_positive_21d"]
        + 0.20 * elig["pct_positive_63d"]
        + 0.20 * elig["score_ret21"]
        + 0.15 * elig["score_ret63"]
        - 0.10 * elig["pct_bad_reentry_21d"]
        - 0.10 * elig["days_penalty"]
    )
    out = out.merge(elig[["trigger_name", "composite_score", "score_ret21", "score_ret63", "days_penalty"]], on="trigger_name", how="left")
    return out.sort_values(["eligible", "composite_score"], ascending=[False, False])


def _shade_stress(ax: plt.Axes, episodes: pd.DataFrame) -> None:
    for _, ep in episodes.iterrows():
        ax.axvspan(ep["stress_start_date"], ep["stress_end_date"], color=STRESS_COLORS.get(ep["dominant_stress_type"], "#cccccc"), alpha=0.13)


def plot_results(panel: pd.DataFrame, episodes: pd.DataFrame, events: pd.DataFrame, summary: pd.DataFrame, by_type: pd.DataFrame, ranking: pd.DataFrame) -> None:
    found = events.loc[events["trigger_found"]].copy()
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(found.groupby("trigger_name")["trigger_before_trough"].mean().reindex(summary["trigger_name"]),
                    summary["avg_forward_return_21d"],
                    s=40 + summary["trigger_found_count"] * 12,
                    c=summary["avg_forward_mdd_21d"], cmap="RdYlGn_r", alpha=0.8)
    for _, row in summary.iterrows():
        ax.text(row["pct_trigger_before_trough"], row["avg_forward_return_21d"], row["trigger_name"].split("_")[0], fontsize=8)
    ax.set_xlabel("Pct trigger before trough")
    ax.set_ylabel("Avg 21D forward return")
    ax.set_title("Recovery Trigger Quality")
    fig.colorbar(sc, ax=ax, label="Avg 21D forward MDD")
    fig.tight_layout()
    fig.savefig(FIG_SCATTER, dpi=180)
    plt.close(fig)

    metrics = ["avg_days_after_trough", "pct_positive_21d", "avg_forward_mdd_21d", "pct_bad_reentry_21d"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, metric in zip(axes.flatten(), metrics):
        sns.barplot(data=summary, x="trigger_name", y=metric, ax=ax)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=70)
    fig.tight_layout()
    fig.savefig(FIG_BAR, dpi=180)
    plt.close(fig)

    if not by_type.empty:
        piv = by_type.pivot(index="trigger_name", columns="dominant_stress_type", values="avg_forward_return_21d")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(piv, annot=True, fmt=".2%", cmap="RdYlGn", center=0, ax=ax)
        ax.set_title("Avg 21D Forward Return by Stress Type")
        fig.tight_layout()
        fig.savefig(FIG_HEAT, dpi=180)
        plt.close(fig)

    covid_start = pd.Timestamp(CONFIG["covid_start"])
    covid_end = pd.Timestamp(CONFIG["covid_end"])
    covid = panel.loc[panel["date"].between(covid_start, covid_end)].copy()
    covid_eps = episodes.loc[(episodes["stress_start_date"] <= covid_end) & (episodes["stress_end_date"] >= covid_start)]
    if not covid.empty and not covid_eps.empty:
        fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
        ax1, ax2, ax3, ax4 = axes
        ax1.plot(covid["date"], covid["spy_price"] / covid["spy_price"].iloc[0], color="black", label="SPY")
        for _, ep in covid_eps.iterrows():
            ax1.axvline(ep["stress_start_date"], color="tab:red", linestyle="--", label="stress start")
            ax1.axvline(ep["SPY_trough_date"], color="black", linestyle=":", label="SPY trough")
            ax1.axvline(ep["VIX_peak_date"], color="tab:purple", linestyle=":", label="VIX peak")
        covid_events = events.loc[events["stress_episode_id"].isin(covid_eps["stress_episode_id"]) & events["trigger_found"]]
        for _, ev in covid_events.iterrows():
            ax1.axvline(ev["trigger_date"], alpha=0.15, color="tab:green")
        ax1.set_title("COVID Recovery Case Study")
        ax2.plot(covid["date"], covid["VIX_LEVEL"], color="tab:purple", label="VIX")
        ax2b = ax2.twinx()
        ax2b.plot(covid["date"], covid["VIX_zscore_120d"], color="tab:orange", label="VIX z")
        ax3.plot(covid["date"], covid["spy_drawdown_from_previous_high"], color="tab:red")
        for regime, color in REGIME_COLORS.items():
            ax4.fill_between(covid["date"], 0.55, 0.95, where=covid["macro_regime_confirmed"].eq(regime), color=color, alpha=0.7)
        ax4.fill_between(covid["date"], 0.05, 0.45, where=covid["active_stress"], color="red", alpha=0.7)
        ax4.set_yticks([0.25, 0.75])
        ax4.set_yticklabels(["Stress", "Regime"])
        fig.tight_layout()
        fig.savefig(FIG_COVID, dpi=180)
        plt.close(fig)

    top = ranking.loc[ranking["eligible"]].head(3)
    if top.empty:
        top = ranking.head(3)
    top_names = set(top["trigger_name"])
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(panel["date"], panel["spy_price"] / panel["spy_price"].iloc[0], color="black", label="SPY")
    _shade_stress(ax, episodes)
    for trigger in top_names:
        ev = events.loc[(events["trigger_name"] == trigger) & events["trigger_found"]]
        ax.scatter(ev["trigger_date"], panel.set_index("date").reindex(ev["trigger_date"])["spy_price"] / panel["spy_price"].iloc[0], s=20, label=trigger)
    ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Stress Episodes and Top Recovery Trigger Dates")
    fig.tight_layout()
    fig.savefig(FIG_TIMELINE, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=found, x="trigger_name", y="days_after_trough", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(21, color="gray", linestyle="--", linewidth=0.8)
    ax.tick_params(axis="x", rotation=70)
    ax.set_title("Days After SPY Trough Distribution")
    fig.tight_layout()
    fig.savefig(FIG_DAYS, dpi=180)
    plt.close(fig)


def write_markdown_report(summary: pd.DataFrame, by_type: pd.DataFrame, ranking: pd.DataFrame, covid: pd.DataFrame) -> None:
    top = ranking.head(5).to_markdown(index=False)
    r8 = summary.loc[summary["trigger_name"] == "R8_SPY_5D_RETURN_GT_3"]
    r3 = summary.loc[summary["trigger_name"] == "R3_SPY_CROSS_ABOVE_MA20"]
    recommendation_lines = []
    if not r8.empty:
        rr = r8.iloc[0]
        recommendation_lines.append(
            f"- Aggressive candidate: `R8_SPY_5D_RETURN_GT_3` (avg 21D return {rr['avg_forward_return_21d']:.2%}, before-trough rate {rr['pct_trigger_before_trough']:.1%})."
        )
    if not r3.empty:
        rr = r3.iloc[0]
        recommendation_lines.append(
            f"- Conservative candidate: `R3_SPY_CROSS_ABOVE_MA20` (avg 21D return {rr['avg_forward_return_21d']:.2%}, before-trough rate {rr['pct_trigger_before_trough']:.1%})."
        )
    lines = [
        "# Stress Recovery Price Trigger Diagnostic",
        "",
        "## Purpose",
        "",
        "This diagnostic keeps the stress signal fixed and studies which price-based recovery triggers identify SPY rebound earlier and more reliably.",
        "",
        "## Stress Definition",
        "",
        "- VIX fast stress: `VIX z-score 120d >= 3.0`, cooldown 21 trading days.",
        "- VIX stress is active only in `FLAT` and `STEEP` regimes.",
        "- Monthly Either SELL is active stress only in `STEEP`.",
        "- Final active stress is `FLAT + VIX stress` or `STEEP + (VIX stress or Monthly Either SELL)`.",
        "",
        "## Recovery Trigger Candidates",
        "",
        "- R1: SPY > MA10",
        "- R2: SPY > MA20",
        "- R3: SPY crosses above MA20",
        "- R4: rebound from 20D low >= 3%",
        "- R5: rebound from 20D low >= 5%",
        "- R6: rebound from 60D low >= 5%",
        "- R7: SPY > MA20 and rebound from 20D low >= 3%",
        "- R8: SPY 5D return > 3%",
        "- R9: SPY 10D return > 5%",
        "",
        "## Full-Sample Findings",
        "",
        top,
        "",
        f"![Trigger scatter](../../figures/stress_recovery_price_trigger_diagnostic/{FIG_SCATTER.name})",
        "",
        f"![Trigger bars](../../figures/stress_recovery_price_trigger_diagnostic/{FIG_BAR.name})",
        "",
        "## By Stress Type Findings",
        "",
        f"![By stress type](../../figures/stress_recovery_price_trigger_diagnostic/{FIG_HEAT.name})",
        "",
        "## COVID Case Study",
        "",
        "- See `covid_recovery_trigger_case_study.csv` for trigger-by-trigger COVID timing.",
        "- In COVID, simple rebound-from-low triggers fired before the final trough, while `R8_SPY_5D_RETURN_GT_3` and `R9_SPY_10D_RETURN_GT_5` fired shortly after the trough.",
        "",
        f"![COVID recovery](../../figures/stress_recovery_price_trigger_diagnostic/{FIG_COVID.name})",
        "",
        "## Interpretation",
        "",
        "- VIX z-score is suitable for stress entry; recovery is better handled by price confirmation.",
        "- VIX z-score decline is not used as the main recovery rule because rolling volatility normalization can lag in high-volatility regimes.",
        "- Price rebound and moving-average recovery triggers are more directly tied to re-entry behavior.",
        "",
        "## Recommended Next Step",
        "",
        *recommendation_lines,
        "- Carry one aggressive and one conservative trigger into a formal overlay backtest.",
        "- Compare their effect specifically on FLAT_VIX_STRESS and COVID-like episodes.",
        "",
        "## Caveats",
        "",
        "- Episode count is limited.",
        "- Ranking can overfit.",
        "- This is not a portfolio allocation result.",
    ]
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = build_active_stress_signal(build_price_recovery_features(build_vix_zscore(load_data())))
    episodes = extract_stress_episodes(panel)
    events = evaluate_recovery_triggers(panel, episodes)
    summary, by_type, by_regime = summarize_recovery_triggers(events)
    covid = analyze_covid_case(episodes, events)
    ranking = rank_recovery_triggers(summary)

    panel.to_csv(DAILY_PANEL_OUT, index=False)
    episodes.drop(columns=["start_index", "end_index"], errors="ignore").to_csv(EPISODE_OUT, index=False)
    events.to_csv(TRIGGER_EVENT_OUT, index=False)
    summary.to_csv(TRIGGER_SUMMARY_OUT, index=False)
    by_type.to_csv(TRIGGER_STRESS_TYPE_OUT, index=False)
    by_regime.to_csv(TRIGGER_REGIME_OUT, index=False)
    covid.to_csv(COVID_OUT, index=False)
    ranking.to_csv(RANKING_OUT, index=False)

    plot_results(panel, episodes, events, summary, by_type, ranking)
    write_markdown_report(summary, by_type, ranking, covid)

    stress_dist = episodes["dominant_stress_type"].value_counts() if not episodes.empty else pd.Series(dtype=int)
    covid_ep = episodes.loc[(episodes["stress_start_date"] <= pd.Timestamp(CONFIG["covid_end"])) & (episodes["stress_end_date"] >= pd.Timestamp(CONFIG["covid_start"]))]
    top = ranking.head(5)
    early = summary.sort_values("pct_trigger_before_trough", ascending=False).iloc[0] if not summary.empty else None
    late = summary.sort_values("pct_too_late", ascending=False).iloc[0] if not summary.empty else None
    flat_top = by_regime.loc[by_regime["dominant_macro_regime"] == "FLAT"].sort_values("avg_forward_return_21d", ascending=False).head(1)
    steep_top = by_regime.loc[by_regime["dominant_macro_regime"] == "STEEP"].sort_values("avg_forward_return_21d", ascending=False).head(1)
    eligible = ranking.loc[ranking["eligible"]]
    aggressive_pool = eligible.loc[(eligible["pct_trigger_before_trough"] <= 0.30) & (eligible["median_days_after_trough"] <= 7)]
    aggressive = aggressive_pool.sort_values("avg_forward_return_21d", ascending=False).head(1) if not aggressive_pool.empty else ranking.loc[ranking["trigger_name"] == "R8_SPY_5D_RETURN_GT_3"].head(1)
    conservative = ranking.loc[ranking["trigger_name"] == "R3_SPY_CROSS_ABOVE_MA20"].head(1)
    if conservative.empty:
        conservative_pool = eligible.loc[(eligible["pct_trigger_before_trough"] <= 0.30) & (eligible["median_days_after_trough"].between(3, 15))]
        conservative = conservative_pool.sort_values(["pct_bad_reentry_21d", "avg_forward_mdd_21d"], ascending=[True, False]).head(1) if not conservative_pool.empty else ranking.head(1)

    print(f"1. Active stress episodes: {len(episodes)}")
    print("2. Stress type distribution:")
    print(stress_dist.to_string())
    if not covid_ep.empty:
        ep = covid_ep.iloc[0]
        covid_events = events.loc[(events["stress_episode_id"] == ep["stress_episode_id"]) & events["trigger_found"]].sort_values("trigger_date")
        print(f"3. COVID stress episode: start={pd.Timestamp(ep['stress_start_date']).date()}, trough={pd.Timestamp(ep['SPY_trough_date']).date()}, first recovery={covid_events.iloc[0]['trigger_name']} on {pd.Timestamp(covid_events.iloc[0]['trigger_date']).date() if not covid_events.empty else 'n/a'}")
    print("4. Full-sample top 5 recovery triggers:")
    print(top[["trigger_name", "composite_score", "trigger_coverage_rate", "pct_trigger_before_trough", "avg_forward_return_21d", "pct_bad_reentry_21d"]].to_string(index=False))
    print(f"5. FLAT top trigger: {flat_top.iloc[0]['trigger_name'] if not flat_top.empty else 'n/a'}")
    print(f"6. STEEP top trigger: {steep_top.iloc[0]['trigger_name'] if not steep_top.empty else 'n/a'}")
    print(f"7. Most early trigger: {early['trigger_name'] if early is not None else 'n/a'}")
    print(f"8. Most late trigger: {late['trigger_name'] if late is not None else 'n/a'}")
    print(f"9. Recommended aggressive / conservative: {aggressive.iloc[0]['trigger_name']} / {conservative.iloc[0]['trigger_name']}")
    print(f"10. Saved outputs: {RESULTS_DIR} and {FIGURES_DIR}")


if __name__ == "__main__":
    main()
