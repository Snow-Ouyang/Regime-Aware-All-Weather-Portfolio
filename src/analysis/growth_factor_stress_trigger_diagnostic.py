from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIG = {
    "output_dir": Path("results/growth_factor_stress_trigger_diagnostic"),
    "figure_dir": Path("figures/growth_factor_stress_trigger_diagnostic"),
    "cooldown_days": 21,
    "forward_windows": [21, 42, 63, 126],
    "case_2015_start": "2015-05-01",
    "case_2015_end": "2016-03-31",
    "case_2015_peak": "2015-07-20",
    "case_2015_trough": "2016-02-11",
    "trading_days_per_year": 252,
}

PANEL_CANDIDATES = [
    Path("results/drawdown_2015_2016_forensic_diagnostic/forensic_daily_panel.csv"),
    Path("results/regime_aware_risk_parity_allocation/daily_backtest_panel.csv"),
    Path("results/spy_cash_backbone_upgrade_ablation/daily_backtest_panel.csv"),
]

COMMODITY_SUMMARY = Path("results/commodity_stress_trigger_diagnostic/commodity_trigger_summary.csv")
COMMODITY_SUMMARY_BY_REGIME = Path("results/commodity_stress_trigger_diagnostic/commodity_trigger_summary_by_regime.csv")
COMMODITY_EVENT_TABLE = Path("results/commodity_stress_trigger_diagnostic/commodity_trigger_event_table.csv")

CORE_SIGNALS = [
    "GROWTH_60D_DROP_PCT20",
    "SPY_DD5_AND_GROWTH_60D_DROP_PCT20",
    "SPY_BELOW_MA100_AND_GROWTH_60D_DROP_PCT20",
    "GROWTH_60D_DROP_AND_INFLATION_60D_DROP",
]


def ensure_dirs() -> None:
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["figure_dir"].mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        for alt in ["DATE", "Date", "observation_date"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "date"})
                break
    if "date" not in df.columns:
        raise ValueError(f"No date column in {path}")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def _first_series(df: pd.DataFrame, candidates: List[str], default: Optional[pd.Series] = None) -> pd.Series:
    for name in candidates:
        if name in df.columns:
            obj = df[name]
            if isinstance(obj, pd.DataFrame):
                return obj.iloc[:, 0]
            return obj
    if default is not None:
        return default
    return pd.Series(index=df.index, dtype=float)


def _max_drawdown(ret: pd.Series) -> float:
    nav = (1 + ret.fillna(0.0)).cumprod()
    return float((nav / nav.cummax() - 1).min()) if not nav.empty else np.nan


def _annualize_vol(ret: pd.Series) -> float:
    ret = ret.dropna()
    return float(ret.std(ddof=0) * math.sqrt(CONFIG["trading_days_per_year"])) if len(ret) > 1 else np.nan


def _sharpe(ret: pd.Series, rf: pd.Series) -> float:
    aligned = pd.concat([ret, rf], axis=1).dropna()
    if aligned.empty:
        return np.nan
    excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    std = excess.std(ddof=0)
    return float(excess.mean() / std * math.sqrt(CONFIG["trading_days_per_year"])) if std and std > 0 else np.nan


def _percentile_series(values: pd.Series) -> pd.Series:
    valid = values.dropna()
    out = pd.Series(np.nan, index=values.index, dtype=float)
    if valid.empty:
        return out
    ranks = valid.rank(method="average", pct=True)
    out.loc[valid.index] = ranks
    return out


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    roll = series.rolling(window, min_periods=max(20, window // 2))
    return (series - roll.mean()) / roll.std(ddof=0)


def _normalize(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if x.dropna().nunique() <= 1:
        return pd.Series(0.5, index=series.index)
    mn = x.min()
    mx = x.max()
    return (x - mn) / (mx - mn)


def load_panel() -> pd.DataFrame:
    panel = None
    src = None
    for path in PANEL_CANDIDATES:
        if path.exists():
            panel = _read_csv(path)
            src = path
            break
    if panel is None:
        raise FileNotFoundError("No source panel found.")
    print(f"Loaded panel: {src}")
    panel = panel.loc[:, ~panel.columns.duplicated(keep="first")].copy()
    panel["SPY_return"] = pd.to_numeric(_first_series(panel, ["SPY_return", "spy_daily_return"]), errors="coerce")
    panel["daily_rf"] = pd.to_numeric(_first_series(panel, ["daily_rf"]), errors="coerce").fillna(0.0)
    panel["VIX_LEVEL"] = pd.to_numeric(_first_series(panel, ["VIX_LEVEL"]), errors="coerce")
    panel["VIX_ZSCORE_120D"] = pd.to_numeric(_first_series(panel, ["VIX_ZSCORE_120D"]), errors="coerce")
    panel["CREDIT_SPREAD_BAA_AAA"] = pd.to_numeric(_first_series(panel, ["CREDIT_SPREAD_BAA_AAA"]), errors="coerce")
    panel["D_CREDIT_SPREAD_20D"] = pd.to_numeric(_first_series(panel, ["D_CREDIT_SPREAD_20D"]), errors="coerce")
    panel["spy_drawdown_from_previous_high"] = pd.to_numeric(_first_series(panel, ["spy_drawdown_from_previous_high"]), errors="coerce")
    panel["spy_price"] = pd.to_numeric(_first_series(panel, ["spy_price"]), errors="coerce")
    panel["macro_regime_confirmed"] = panel.get("macro_regime_confirmed", pd.Series("NEUTRAL", index=panel.index)).fillna("NEUTRAL").astype(str)
    panel["monthly_either_state"] = panel.get("monthly_either_state", pd.Series("UNKNOWN", index=panel.index)).fillna("UNKNOWN").astype(str)

    if "SPY_MA20" not in panel.columns:
        panel["SPY_MA20"] = panel["spy_price"].rolling(20, min_periods=20).mean()
    if "SPY_CROSS_ABOVE_MA20" not in panel.columns:
        panel["SPY_CROSS_ABOVE_MA20"] = (panel["spy_price"] > panel["SPY_MA20"]) & (
            panel["spy_price"].shift(1) <= panel["SPY_MA20"].shift(1)
        )
    if "SPY_MA100" not in panel.columns:
        panel["SPY_MA100"] = panel["spy_price"].rolling(100, min_periods=100).mean()
    if "SPY_MA200" not in panel.columns:
        panel["SPY_MA200"] = panel["spy_price"].rolling(200, min_periods=200).mean()
    if "SPY_MA50" not in panel.columns:
        panel["SPY_MA50"] = panel["spy_price"].rolling(50, min_periods=50).mean()
    panel["growth_pc1"] = pd.to_numeric(_first_series(panel, ["growth_pc1"]), errors="coerce")
    panel["inflation_pc1"] = pd.to_numeric(_first_series(panel, ["inflation_pc1"]), errors="coerce")
    panel["term_spread"] = pd.to_numeric(_first_series(panel, ["term_spread"]), errors="coerce")
    panel["GS10"] = pd.to_numeric(_first_series(panel, ["GS10"]), errors="coerce")
    panel["GS1"] = pd.to_numeric(_first_series(panel, ["GS1"]), errors="coerce")

    # rebuild backbone if not present
    panel["FLAT_VIX_STRESS"] = panel["macro_regime_confirmed"].eq("FLAT") & (panel["VIX_ZSCORE_120D"] >= 3.0)
    panel["FLAT_CREDIT_DD5_STRESS"] = panel["macro_regime_confirmed"].eq("FLAT") & (panel["spy_drawdown_from_previous_high"] <= -0.05) & (panel["D_CREDIT_SPREAD_20D"] > 0.10)
    panel["STEEP_EITHER_SELL_STRESS"] = panel["macro_regime_confirmed"].eq("STEEP") & panel["monthly_either_state"].eq("SELL")
    panel["STEEP_CREDIT_DD5_STRESS"] = panel["macro_regime_confirmed"].eq("STEEP") & (panel["spy_drawdown_from_previous_high"] <= -0.05) & (panel["D_CREDIT_SPREAD_20D"] > 0.10)
    panel["BACKBONE_V2_ENTRY_SIGNAL"] = (
        panel["FLAT_VIX_STRESS"] | panel["FLAT_CREDIT_DD5_STRESS"] | panel["STEEP_EITHER_SELL_STRESS"] | panel["STEEP_CREDIT_DD5_STRESS"]
    )
    if "BACKBONE_V2_UPGRADED_risk_state" in panel.columns:
        panel["BACKBONE_V2_RISK_STATE"] = panel["BACKBONE_V2_UPGRADED_risk_state"].astype(str).str.upper().eq("RISK")
    elif "BACKBONE_V2_UPGRADED_weight_spy" in panel.columns:
        panel["BACKBONE_V2_RISK_STATE"] = pd.to_numeric(panel["BACKBONE_V2_UPGRADED_weight_spy"], errors="coerce").fillna(1.0) < 0.5
    elif "timing_state" in panel.columns:
        panel["BACKBONE_V2_RISK_STATE"] = panel["timing_state"].astype(str).eq("RISK")
    else:
        state = []
        current = "NON_RISK"
        for _, row in panel.iterrows():
            state.append(current == "RISK")
            if current == "NON_RISK" and bool(row["BACKBONE_V2_ENTRY_SIGNAL"]):
                current = "RISK"
            elif current == "RISK" and bool(row["SPY_CROSS_ABOVE_MA20"]):
                current = "NON_RISK"
        panel["BACKBONE_V2_RISK_STATE"] = state

    if "CMDTY_RET_60D" in panel.columns and "CMDTY_RET60" not in panel.columns:
        panel["CMDTY_RET60"] = pd.to_numeric(panel["CMDTY_RET_60D"], errors="coerce")
    elif "CMDTY_RET60" in panel.columns:
        panel["CMDTY_RET60"] = pd.to_numeric(panel["CMDTY_RET60"], errors="coerce")
    elif "CMDTY_FUT_return" in panel.columns:
        price = (1 + pd.to_numeric(_first_series(panel, ["CMDTY_FUT_return"]), errors="coerce").fillna(0.0)).cumprod()
        panel["CMDTY_RET60"] = price / price.shift(60) - 1.0
    else:
        panel["CMDTY_RET60"] = np.nan
    return panel


def build_growth_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["growth_pct_full"] = _percentile_series(out["growth_pc1"])
    out["growth_z_252"] = _rolling_z(out["growth_pc1"], 252)
    out["growth_z_120"] = _rolling_z(out["growth_pc1"], 120)
    out["D_GROWTH_20D"] = out["growth_pc1"] - out["growth_pc1"].shift(20)
    out["D_GROWTH_60D"] = out["growth_pc1"] - out["growth_pc1"].shift(60)
    out["D_GROWTH_120D"] = out["growth_pc1"] - out["growth_pc1"].shift(120)
    out["D_GROWTH_20D_pct_full"] = _percentile_series(out["D_GROWTH_20D"])
    out["D_GROWTH_60D_pct_full"] = _percentile_series(out["D_GROWTH_60D"])
    out["D_GROWTH_120D_pct_full"] = _percentile_series(out["D_GROWTH_120D"])
    out["D_GROWTH_20D_z_252"] = _rolling_z(out["D_GROWTH_20D"], 252)
    out["D_GROWTH_60D_z_252"] = _rolling_z(out["D_GROWTH_60D"], 252)
    out["growth_rolling_high_252"] = out["growth_pc1"].rolling(252, min_periods=60).max()
    out["growth_drop_from_252_high"] = out["growth_pc1"] - out["growth_rolling_high_252"]
    out["growth_drop_from_252_high_z"] = _rolling_z(out["growth_drop_from_252_high"], 252)
    out["growth_drop_from_252_high_pct_full"] = _percentile_series(out["growth_drop_from_252_high"])
    if "inflation_pc1" in out.columns:
        out["D_INFLATION_60D"] = out["inflation_pc1"] - out["inflation_pc1"].shift(60)
        out["D_INFLATION_60D_pct_full"] = _percentile_series(out["D_INFLATION_60D"])
        out["inflation_pct_full"] = _percentile_series(out["inflation_pc1"])
    else:
        out["D_INFLATION_60D"] = np.nan
        out["D_INFLATION_60D_pct_full"] = np.nan
        out["inflation_pct_full"] = np.nan
    out["SPY_below_MA100"] = out["spy_price"] < out["SPY_MA100"]
    out["SPY_below_MA200"] = out["spy_price"] < out["SPY_MA200"]
    return out


def build_growth_stress_signals(panel: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = panel.copy()
    threshold = out["growth_drop_from_252_high"].quantile(0.10)
    sigs: Dict[str, pd.Series] = {
        "GROWTH_20D_DROP_PCT10": out["D_GROWTH_20D_pct_full"] <= 0.10,
        "GROWTH_60D_DROP_PCT10": out["D_GROWTH_60D_pct_full"] <= 0.10,
        "GROWTH_60D_DROP_PCT20": out["D_GROWTH_60D_pct_full"] <= 0.20,
        "GROWTH_120D_DROP_PCT10": out["D_GROWTH_120D_pct_full"] <= 0.10,
        "GROWTH_Z_BELOW_NEG1": out["growth_z_252"] <= -1.0,
        "GROWTH_Z_BELOW_NEG1_5": out["growth_z_252"] <= -1.5,
        "GROWTH_LEVEL_PCT20": out["growth_pct_full"] <= 0.20,
        "GROWTH_LEVEL_PCT30": out["growth_pct_full"] <= 0.30,
        "GROWTH_DROP_FROM_HIGH": out["growth_drop_from_252_high"] <= threshold,
        "SPY_DD3_AND_GROWTH_60D_DROP_PCT20": (out["spy_drawdown_from_previous_high"] <= -0.03) & (out["D_GROWTH_60D_pct_full"] <= 0.20),
        "SPY_DD5_AND_GROWTH_60D_DROP_PCT20": (out["spy_drawdown_from_previous_high"] <= -0.05) & (out["D_GROWTH_60D_pct_full"] <= 0.20),
        "SPY_BELOW_MA100_AND_GROWTH_60D_DROP_PCT20": out["SPY_below_MA100"] & (out["D_GROWTH_60D_pct_full"] <= 0.20),
        "SPY_BELOW_MA200_AND_GROWTH_60D_DROP_PCT20": out["SPY_below_MA200"] & (out["D_GROWTH_60D_pct_full"] <= 0.20),
        "GROWTH_60D_DROP_AND_INFLATION_60D_DROP": (out["D_GROWTH_60D_pct_full"] <= 0.20) & (out["D_INFLATION_60D_pct_full"] <= 0.20),
        "SPY_DD5_GROWTH_60D_DROP_AND_INFLATION_60D_DROP": (out["spy_drawdown_from_previous_high"] <= -0.05) & (out["D_GROWTH_60D_pct_full"] <= 0.20) & (out["D_INFLATION_60D_pct_full"] <= 0.20),
        "GROWTH_60D_DROP_VIX_LT3": (out["D_GROWTH_60D_pct_full"] <= 0.20) & (out["VIX_ZSCORE_120D"] < 3.0),
        "SPY_DD5_GROWTH_60D_DROP_VIX_LT3": (out["spy_drawdown_from_previous_high"] <= -0.05) & (out["D_GROWTH_60D_pct_full"] <= 0.20) & (out["VIX_ZSCORE_120D"] < 3.0),
        "GROWTH_60D_DROP_AND_CMDTY_RET60_NEG10": (out["D_GROWTH_60D_pct_full"] <= 0.20) & (out["CMDTY_RET60"] < -0.10),
        "SPY_DD5_GROWTH_60D_DROP_AND_CMDTY_RET60_NEG10": (out["spy_drawdown_from_previous_high"] <= -0.05) & (out["D_GROWTH_60D_pct_full"] <= 0.20) & (out["CMDTY_RET60"] < -0.10),
    }
    for name, ser in sigs.items():
        out[name] = ser.fillna(False)
    return out, list(sigs.keys())


def compute_forward_outcomes(panel: pd.DataFrame, idx: int, windows: List[int]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for h in windows:
        fwd = panel["SPY_return"].iloc[idx + 1 : idx + 1 + h].fillna(0.0)
        if fwd.empty:
            out[f"forward_spy_return_{h}"] = np.nan
            out[f"forward_spy_max_drawdown_{h}"] = np.nan
            out[f"forward_spy_max_runup_{h}"] = np.nan
            out[f"days_to_trough_{h}"] = np.nan
            continue
        nav = (1 + fwd).cumprod()
        dd = nav / nav.cummax() - 1
        runup = nav / nav.cummin() - 1
        out[f"forward_spy_return_{h}"] = float(nav.iloc[-1] - 1)
        out[f"forward_spy_max_drawdown_{h}"] = float(dd.min())
        out[f"forward_spy_max_runup_{h}"] = float(runup.max())
        out[f"days_to_trough_{h}"] = int(dd.idxmin() - fwd.index[0]) if len(dd) else np.nan
    return out


def extract_signal_events(panel: pd.DataFrame, signals: List[str]) -> pd.DataFrame:
    rows = []
    for signal_name in signals:
        signal = panel[signal_name].fillna(False).astype(bool)
        last_event_i: Optional[int] = None
        has_reset = True
        prev = False
        for i, is_true in enumerate(signal):
            if last_event_i is not None and not is_true:
                has_reset = True
            can_fire = last_event_i is None or ((i - last_event_i) > CONFIG["cooldown_days"] and has_reset)
            if is_true and not prev and can_fire:
                row = panel.iloc[i]
                forward = compute_forward_outcomes(panel, i, CONFIG["forward_windows"])
                rows.append(
                    {
                        "signal_name": signal_name,
                        "event_date": row["date"],
                        "macro_regime_confirmed": row["macro_regime_confirmed"],
                        "growth_pc1": row.get("growth_pc1"),
                        "growth_pct_full": row.get("growth_pct_full"),
                        "growth_z_252": row.get("growth_z_252"),
                        "D_GROWTH_20D": row.get("D_GROWTH_20D"),
                        "D_GROWTH_60D": row.get("D_GROWTH_60D"),
                        "D_GROWTH_60D_pct_full": row.get("D_GROWTH_60D_pct_full"),
                        "inflation_pc1": row.get("inflation_pc1"),
                        "D_INFLATION_60D": row.get("D_INFLATION_60D"),
                        "VIX_LEVEL": row.get("VIX_LEVEL"),
                        "VIX_ZSCORE_120D": row.get("VIX_ZSCORE_120D"),
                        "CREDIT_SPREAD_BAA_AAA": row.get("CREDIT_SPREAD_BAA_AAA"),
                        "D_CREDIT_SPREAD_20D": row.get("D_CREDIT_SPREAD_20D"),
                        "term_spread": row.get("term_spread"),
                        "GS10": row.get("GS10"),
                        "GS1": row.get("GS1"),
                        "spy_drawdown_from_previous_high": row.get("spy_drawdown_from_previous_high"),
                        "SPY_MA100": row.get("SPY_MA100"),
                        "SPY_MA200": row.get("SPY_MA200"),
                        "monthly_either_state": row.get("monthly_either_state"),
                        "BACKBONE_V2_RISK_STATE": row.get("BACKBONE_V2_RISK_STATE"),
                        "BACKBONE_V2_ENTRY_SIGNAL": row.get("BACKBONE_V2_ENTRY_SIGNAL"),
                        **forward,
                    }
                )
                last_event_i = i
                has_reset = False
            prev = is_true
    events = pd.DataFrame(rows)
    if events.empty:
        return events
    events["mdd_21d_below_3"] = events["forward_spy_max_drawdown_21"].le(-0.03)
    events["mdd_21d_below_5"] = events["forward_spy_max_drawdown_21"].le(-0.05)
    events["mdd_63d_below_5"] = events["forward_spy_max_drawdown_63"].le(-0.05)
    events["mdd_63d_below_10"] = events["forward_spy_max_drawdown_63"].le(-0.10)
    events["mdd_126d_below_10"] = events["forward_spy_max_drawdown_126"].le(-0.10)
    events["false_alarm_21d"] = events["forward_spy_max_drawdown_21"].gt(-0.03)
    events["false_alarm_63d"] = events["forward_spy_max_drawdown_63"].gt(-0.05)
    events["quick_rebound_21d"] = events["forward_spy_return_21"].gt(0.03)
    events.to_csv(CONFIG["output_dir"] / "growth_stress_event_table.csv", index=False)
    return events


def summarize_events_by_regime(events: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def summarize(df: pd.DataFrame, include_regime: bool) -> pd.DataFrame:
        grp_cols = ["signal_name"] + (["macro_regime_confirmed"] if include_regime else [])
        rows = []
        for keys, sub in df.groupby(grp_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {col: key for col, key in zip(grp_cols, keys)}
            years = max((sub["event_date"].max() - sub["event_date"].min()).days / 365.25, 1 / 365.25)
            row.update(
                {
                    "event_count": len(sub),
                    "events_per_year": len(sub) / years,
                    "avg_forward_return_21d": sub["forward_spy_return_21"].mean(),
                    "avg_forward_return_63d": sub["forward_spy_return_63"].mean(),
                    "avg_forward_return_126d": sub["forward_spy_return_126"].mean(),
                    "avg_forward_mdd_21d": sub["forward_spy_max_drawdown_21"].mean(),
                    "avg_forward_mdd_63d": sub["forward_spy_max_drawdown_63"].mean(),
                    "avg_forward_mdd_126d": sub["forward_spy_max_drawdown_126"].mean(),
                    "median_forward_mdd_63d": sub["forward_spy_max_drawdown_63"].median(),
                    "pct_mdd_21d_below_5": sub["mdd_21d_below_5"].mean(),
                    "pct_mdd_63d_below_5": sub["mdd_63d_below_5"].mean(),
                    "pct_mdd_63d_below_10": sub["mdd_63d_below_10"].mean(),
                    "pct_mdd_126d_below_10": sub["mdd_126d_below_10"].mean(),
                    "false_alarm_rate_21d": sub["false_alarm_21d"].mean(),
                    "false_alarm_rate_63d": sub["false_alarm_63d"].mean(),
                    "quick_rebound_rate_21d": sub["quick_rebound_21d"].mean(),
                    "avg_days_to_trough_63d": sub["days_to_trough_63"].mean(),
                    "median_growth_pct_at_event": sub["growth_pct_full"].median(),
                    "median_growth_change60_at_event": sub["D_GROWTH_60D"].median(),
                    "median_spy_dd_at_event": sub["spy_drawdown_from_previous_high"].median(),
                    "median_vix_z_at_event": sub["VIX_ZSCORE_120D"].median(),
                    "median_credit_chg20_at_event": sub["D_CREDIT_SPREAD_20D"].median(),
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    by_regime = summarize(events, include_regime=True)
    full = summarize(events, include_regime=False)
    by_regime.to_csv(CONFIG["output_dir"] / "growth_stress_summary_by_regime.csv", index=False)
    full.to_csv(CONFIG["output_dir"] / "growth_stress_summary_full_sample.csv", index=False)
    return by_regime, full


def analyze_backbone_overlap(panel: pd.DataFrame, signals: List[str], events: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    overlap_daily = panel[["date", "macro_regime_confirmed", "BACKBONE_V2_ENTRY_SIGNAL", "BACKBONE_V2_RISK_STATE", "FLAT_VIX_STRESS", "FLAT_CREDIT_DD5_STRESS", "STEEP_EITHER_SELL_STRESS", "STEEP_CREDIT_DD5_STRESS"] + signals].copy()
    overlap_daily.to_csv(CONFIG["output_dir"] / "growth_backbone_daily_overlap.csv", index=False)
    backbone_events = panel.loc[panel["BACKBONE_V2_ENTRY_SIGNAL"].fillna(False), "date"]
    rows = []
    for signal_name in signals:
        sig_days = panel[signal_name].fillna(False)
        sig_events = events[events["signal_name"].eq(signal_name)].copy()
        growth_event_dates = list(pd.to_datetime(sig_events["event_date"]))
        overlap_days = (sig_days & panel["BACKBONE_V2_RISK_STATE"].fillna(False)).sum()
        overlap_events = 0
        days_before = []
        growth_only_mdds = []
        growth_only_rets = []
        growth_only_flags = []
        for _, ev in sig_events.iterrows():
            ed = pd.Timestamp(ev["event_date"])
            near = backbone_events[(backbone_events >= ed - pd.Timedelta(days=21)) & (backbone_events <= ed + pd.Timedelta(days=21))]
            if not near.empty:
                overlap_events += 1
                days_before.append((near.min() - ed).days)
            else:
                growth_only_mdds.append(ev["forward_spy_max_drawdown_63"])
                growth_only_rets.append(ev["forward_spy_return_63"])
                growth_only_flags.append(ev["false_alarm_63d"])
        backbone_only = 0
        for bd in backbone_events:
            near = [gd for gd in growth_event_dates if abs((pd.Timestamp(gd) - pd.Timestamp(bd)).days) <= 21]
            if not near:
                backbone_only += 1
        rows.append(
            {
                "signal_name": signal_name,
                "growth_signal_days": int(sig_days.sum()),
                "growth_signal_events": len(sig_events),
                "overlap_days_with_BACKBONE_RISK": int(overlap_days),
                "overlap_ratio_days": float(overlap_days / sig_days.sum()) if sig_days.sum() else np.nan,
                "overlap_events_with_backbone_entry_within_21d": overlap_events,
                "overlap_event_ratio": float(overlap_events / len(sig_events)) if len(sig_events) else np.nan,
                "growth_only_events": len(growth_only_mdds),
                "backbone_only_events": backbone_only,
                "avg_forward_mdd_63d_growth_only": np.mean(growth_only_mdds) if growth_only_mdds else np.nan,
                "avg_forward_return_63d_growth_only": np.mean(growth_only_rets) if growth_only_rets else np.nan,
                "false_alarm_63d_growth_only": np.mean(growth_only_flags) if growth_only_flags else np.nan,
                "pct_mdd_63d_below_5_growth_only": np.mean([x <= -0.05 for x in growth_only_mdds]) if growth_only_mdds else np.nan,
                "avg_days_growth_signal_before_backbone_entry": np.mean(days_before) if days_before else np.nan,
                "median_days_growth_signal_before_backbone_entry": np.median(days_before) if days_before else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(CONFIG["output_dir"] / "growth_backbone_overlap_summary.csv", index=False)
    return overlap_daily, summary


def map_2015_2016_case(panel: pd.DataFrame, signals: List[str], full_summary: pd.DataFrame, by_regime: pd.DataFrame) -> pd.DataFrame:
    start = pd.Timestamp(CONFIG["case_2015_start"])
    end = pd.Timestamp(CONFIG["case_2015_end"])
    peak = pd.Timestamp(CONFIG["case_2015_peak"])
    trough = pd.Timestamp(CONFIG["case_2015_trough"])
    case = panel[(panel["date"] >= start) & (panel["date"] <= end)].copy()
    backbone_dates = case.loc[case["BACKBONE_V2_ENTRY_SIGNAL"], "date"]
    backbone_entry = backbone_dates.min() if not backbone_dates.empty else pd.NaT
    monthly_sell = case.loc[case["monthly_either_state"].eq("SELL"), "date"]
    monthly_sell_date = monthly_sell.min() if not monthly_sell.empty else pd.NaT
    rows = []
    for signal_name in signals:
        hit = case.loc[case[signal_name].fillna(False)]
        first = hit.iloc[0] if not hit.empty else None
        fs = full_summary.loc[full_summary["signal_name"].eq(signal_name)]
        st = by_regime.loc[(by_regime["signal_name"].eq(signal_name)) & (by_regime["macro_regime_confirmed"].eq("STEEP"))]
        rows.append(
            {
                "signal_name": signal_name,
                "first_trigger_date_in_case": first["date"] if first is not None else pd.NaT,
                "macro_regime_at_trigger": first["macro_regime_confirmed"] if first is not None else np.nan,
                "days_after_peak": (pd.Timestamp(first["date"]) - peak).days if first is not None else np.nan,
                "days_before_trough": (trough - pd.Timestamp(first["date"])).days if first is not None else np.nan,
                "SPY_DD_at_trigger": first["spy_drawdown_from_previous_high"] if first is not None else np.nan,
                "growth_pct_at_trigger": first["growth_pct_full"] if first is not None else np.nan,
                "D_GROWTH_60D_at_trigger": first["D_GROWTH_60D"] if first is not None else np.nan,
                "inflation_pct_at_trigger": first["inflation_pct_full"] if first is not None else np.nan,
                "VIX_Z_at_trigger": first["VIX_ZSCORE_120D"] if first is not None else np.nan,
                "credit_chg20_at_trigger": first["D_CREDIT_SPREAD_20D"] if first is not None else np.nan,
                "triggered_before_backbone_entry": pd.notna(backbone_entry) and first is not None and pd.Timestamp(first["date"]) < backbone_entry,
                "triggered_before_monthly_either_sell": pd.notna(monthly_sell_date) and first is not None and pd.Timestamp(first["date"]) < monthly_sell_date,
                "triggered_before_trough": first is not None and pd.Timestamp(first["date"]) <= trough,
                "overlap_with_BACKBONE_RISK": bool(first["BACKBONE_V2_RISK_STATE"]) if first is not None and "BACKBONE_V2_RISK_STATE" in first else np.nan,
                "signal_quality_full_sample": fs["pct_mdd_63d_below_5"].iloc[0] if not fs.empty else np.nan,
                "signal_quality_steep_only": st["pct_mdd_63d_below_5"].iloc[0] if not st.empty else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values(["first_trigger_date_in_case", "signal_name"], na_position="last")
    out.to_csv(CONFIG["output_dir"] / "case_2015_2016_growth_signal_mapping.csv", index=False)
    return out


def build_regime_decision_table(by_regime: pd.DataFrame, overlap_summary: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    merged = by_regime.merge(overlap_summary[["signal_name", "overlap_ratio_days", "growth_only_events"]], on="signal_name", how="left")
    actions = []
    for _, row in merged.iterrows():
        if row["event_count"] < 5:
            action = "INSUFFICIENT_SAMPLE"
        elif row["false_alarm_rate_63d"] <= 0.40 and row["pct_mdd_63d_below_5"] >= 0.50 and row["avg_forward_mdd_63d"] <= -0.05:
            action = "ENABLE_FOR_REGIME"
        elif row["false_alarm_rate_63d"] > 0.60 or row["quick_rebound_rate_21d"] > 0.40:
            action = "DISABLE_FOR_REGIME"
        else:
            action = "DIAGNOSTIC_ONLY"
        actions.append(action)
    merged["recommended_action"] = actions
    merged.to_csv(CONFIG["output_dir"] / "growth_trigger_regime_decision_table.csv", index=False)

    overall_rows = []
    for signal_name, sub in merged.groupby("signal_name"):
        steep = sub[sub["macro_regime_confirmed"].eq("STEEP")]
        others = sub[~sub["macro_regime_confirmed"].eq("STEEP")]
        if not steep.empty and steep["recommended_action"].eq("ENABLE_FOR_REGIME").any():
            if others.empty or others["recommended_action"].isin(["DISABLE_FOR_REGIME", "INSUFFICIENT_SAMPLE", "DIAGNOSTIC_ONLY"]).all():
                rec = "PARTIAL_OVERLAY_STEEP_ONLY"
            else:
                rec = "ENABLE_ALL_REGIME"
        elif sub["growth_only_events"].fillna(0).sum() > 0 and sub["overlap_ratio_days"].mean() < 0.7:
            rec = "DIAGNOSTIC_ONLY"
        else:
            rec = "DISABLE"
        overall_rows.append({"signal_name": signal_name, "overall_recommendation": rec})
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(CONFIG["output_dir"] / "growth_trigger_overall_recommendation.csv", index=False)
    return merged, overall


def compare_with_commodity_triggers(case_map: pd.DataFrame, full_summary: pd.DataFrame, by_regime: pd.DataFrame, overlap_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if COMMODITY_SUMMARY.exists():
        com_full = pd.read_csv(COMMODITY_SUMMARY)
        com_reg = pd.read_csv(COMMODITY_SUMMARY_BY_REGIME) if COMMODITY_SUMMARY_BY_REGIME.exists() else pd.DataFrame()
        com_events = pd.read_csv(COMMODITY_EVENT_TABLE) if COMMODITY_EVENT_TABLE.exists() else pd.DataFrame()
        for _, row in full_summary.iterrows():
            cm = case_map.loc[case_map["signal_name"].eq(row["signal_name"])].iloc[0]
            ov = overlap_summary.loc[overlap_summary["signal_name"].eq(row["signal_name"])].iloc[0]
            rows.append(
                {
                    "signal_family": "growth",
                    "signal_name": row["signal_name"],
                    "event_count": row["event_count"],
                    "false_alarm_rate_63d": row["false_alarm_rate_63d"],
                    "pct_mdd_63d_below_5": row["pct_mdd_63d_below_5"],
                    "avg_forward_mdd_63d": row["avg_forward_mdd_63d"],
                    "overlap_with_backbone_ratio": ov["overlap_ratio_days"],
                    "coverage_2015_2016": pd.notna(cm["first_trigger_date_in_case"]),
                    "triggered_before_backbone_2015_2016": cm["triggered_before_backbone_entry"],
                    "recommended_scope": "growth",
                }
            )
        for _, row in com_full.iterrows():
            cm_case = False
            trig_before = np.nan
            if not com_events.empty:
                case_hits = com_events[
                    (pd.to_datetime(com_events["event_date"]) >= pd.Timestamp(CONFIG["case_2015_start"]))
                    & (pd.to_datetime(com_events["event_date"]) <= pd.Timestamp(CONFIG["case_2015_end"]))
                    & com_events["trigger_name"].eq(row["trigger_name"])
                ]
                cm_case = not case_hits.empty
                if cm_case:
                    trig_before = pd.Timestamp(case_hits["event_date"].iloc[0]) < pd.Timestamp("2015-08-21")
            rows.append(
                {
                    "signal_family": "commodity",
                    "signal_name": row["trigger_name"],
                    "event_count": row["event_count"],
                    "false_alarm_rate_63d": row.get("false_alarm_rate_63d", row.get("false_alarm_rate_21d")),
                    "pct_mdd_63d_below_5": row.get("pct_mdd_63d_below_5", np.nan),
                    "avg_forward_mdd_63d": row.get("avg_forward_mdd_63d"),
                    "overlap_with_backbone_ratio": np.nan,
                    "coverage_2015_2016": cm_case,
                    "triggered_before_backbone_2015_2016": trig_before,
                    "recommended_scope": "commodity",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(CONFIG["output_dir"] / "growth_vs_commodity_trigger_comparison.csv", index=False)
    return out


def plot_growth_timeline(panel: pd.DataFrame) -> None:
    start = pd.Timestamp(CONFIG["case_2015_start"])
    end = pd.Timestamp(CONFIG["case_2015_end"])
    case = panel[(panel["date"] >= start) & (panel["date"] <= end)].copy()
    fig, axes = plt.subplots(8, 1, figsize=(14, 15), sharex=True)
    axes[0].plot(case["date"], case["spy_drawdown_from_previous_high"], color="black")
    axes[0].set_title("SPY Drawdown")
    axes[1].plot(case["date"], case["growth_pct_full"], label="growth pct")
    axes[1].plot(case["date"], case["growth_z_252"], label="growth z252")
    axes[1].legend()
    axes[1].set_title("Growth Level")
    axes[2].plot(case["date"], case["D_GROWTH_60D_pct_full"])
    axes[2].set_title("D_GROWTH_60D percentile")
    axes[3].plot(case["date"], case["inflation_pct_full"])
    axes[3].set_title("Inflation percentile")
    axes[4].plot(case["date"], case["VIX_ZSCORE_120D"])
    axes[4].set_title("VIX z-score")
    axes[5].plot(case["date"], case["D_CREDIT_SPREAD_20D"])
    axes[5].set_title("Credit change 20D")
    axes[6].plot(case["date"], case["macro_regime_confirmed"].astype("category").cat.codes)
    axes[6].set_title("Regime")
    axes[7].plot(case["date"], case["BACKBONE_V2_RISK_STATE"].astype(int), label="backbone risk")
    for sig in CORE_SIGNALS:
        if sig in case.columns:
            axes[7].plot(case["date"], case[sig].astype(int), alpha=0.6, label=sig)
    axes[7].legend(fontsize=7, ncol=2)
    axes[7].set_title("Backbone and Key Growth Triggers")
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "growth_factor_2015_2016_timeline.png", dpi=150)
    plt.close(fig)


def plot_signal_quality(by_regime: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    for regime, sub in by_regime.groupby("macro_regime_confirmed"):
        ax.scatter(sub["false_alarm_rate_63d"], sub["pct_mdd_63d_below_5"], s=sub["event_count"] * 12, alpha=0.7, label=regime)
    for _, row in by_regime.head(20).iterrows():
        ax.text(row["false_alarm_rate_63d"], row["pct_mdd_63d_below_5"], row["signal_name"], fontsize=6)
    ax.set_xlabel("False alarm rate 63d")
    ax.set_ylabel("P(63d MDD < -5%)")
    ax.set_title("Growth Signal Quality by Regime")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "growth_signal_quality_by_regime_scatter.png", dpi=150)
    plt.close(fig)


def plot_overlap(panel: pd.DataFrame) -> None:
    start = pd.Timestamp(CONFIG["case_2015_start"])
    end = pd.Timestamp(CONFIG["case_2015_end"])
    case = panel[(panel["date"] >= start) & (panel["date"] <= end)].copy()
    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(case["date"], case["spy_drawdown_from_previous_high"], color="black")
    axes[0].set_title("SPY Drawdown")
    axes[1].plot(case["date"], case["BACKBONE_V2_RISK_STATE"].astype(int))
    axes[1].set_title("Backbone Risk State")
    for ax, sig in zip(axes[2:], CORE_SIGNALS[:3]):
        ax.plot(case["date"], case[sig].astype(int))
        ax.set_title(sig)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "growth_vs_backbone_overlap_timeline.png", dpi=150)
    plt.close(fig)


def write_markdown_report(
    full_summary: pd.DataFrame,
    by_regime: pd.DataFrame,
    overlap_summary: pd.DataFrame,
    case_map: pd.DataFrame,
    regime_decision: pd.DataFrame,
    overall: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    out = CONFIG["output_dir"] / "GROWTH_FACTOR_STRESS_TRIGGER_REPORT.md"
    content = f"""# GROWTH_FACTOR_STRESS_TRIGGER_REPORT

## Purpose

This report tests whether growth factor deterioration is a cleaner slow-stress trigger than direct commodity price damage, especially for the 2015-2016 missed drawdown.

## Full-sample Findings

{full_summary.to_markdown(index=False)}

## By-regime Findings

{by_regime.to_markdown(index=False)}

## Overlap with Current Backbone

{overlap_summary.to_markdown(index=False)}

## 2015-2016 Mapping

{case_map.to_markdown(index=False)}

## Regime Decision Table

{regime_decision.to_markdown(index=False)}

## Overall Recommendation

{overall.to_markdown(index=False)}

## Growth vs Commodity Comparison

{comparison.to_markdown(index=False)}
"""
    out.write_text(content, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = load_panel()
    panel = build_growth_features(panel)
    panel, signals = build_growth_stress_signals(panel)
    events = extract_signal_events(panel, signals)
    by_regime, full_summary = summarize_events_by_regime(events)
    _, overlap_summary = analyze_backbone_overlap(panel, signals, events)
    case_map = map_2015_2016_case(panel, signals, full_summary, by_regime)
    regime_decision, overall = build_regime_decision_table(by_regime, overlap_summary)
    comparison = compare_with_commodity_triggers(case_map, full_summary, by_regime, overlap_summary)
    plot_growth_timeline(panel)
    plot_signal_quality(by_regime)
    plot_overlap(panel)
    write_markdown_report(full_summary, by_regime, overlap_summary, case_map, regime_decision, overall, comparison)

    case = panel[(panel["date"] >= pd.Timestamp(CONFIG["case_2015_start"])) & (panel["date"] <= pd.Timestamp(CONFIG["case_2015_end"]))]
    peak = pd.Timestamp(CONFIG["case_2015_peak"])
    trough = pd.Timestamp(CONFIG["case_2015_trough"])
    peak_row = case.loc[case["date"].eq(peak)].iloc[0]
    trough_row = case.loc[case["date"].eq(trough)].iloc[0]
    print(f"1. 2015-2016 growth_pc1 peak / trough percentile: {peak_row['growth_pct_full']:.2%} / {trough_row['growth_pct_full']:.2%}")
    covered = case_map.loc[case_map["first_trigger_date_in_case"].notna(), "signal_name"].tolist()
    print(f"2. growth triggers covering 2015-2016: {', '.join(covered[:12])}")
    print("3. earliest growth triggers:")
    print(case_map.sort_values("first_trigger_date_in_case").head(5)[["signal_name", "first_trigger_date_in_case", "days_after_peak"]].to_string(index=False))
    for regime in ["STEEP", "FLAT", "INVERTED"]:
        sub = by_regime[(by_regime["macro_regime_confirmed"].eq(regime)) & (by_regime["signal_name"].isin(CORE_SIGNALS))]
        if not sub.empty:
            print(f"4/5. {regime} core growth signal quality:")
            print(sub[["signal_name", "false_alarm_rate_63d", "pct_mdd_63d_below_5"]].to_string(index=False))
    print("6. overlap with backbone ratio:")
    print(overlap_summary[["signal_name", "overlap_ratio_days", "growth_only_events"]].sort_values("overlap_ratio_days").head(8).to_string(index=False))
    risky_growth_only = overlap_summary.sort_values("pct_mdd_63d_below_5_growth_only", ascending=False).head(5)
    print("7. growth-only events risk:")
    print(risky_growth_only[["signal_name", "growth_only_events", "avg_forward_mdd_63d_growth_only", "false_alarm_63d_growth_only"]].to_string(index=False))
    if not comparison.empty:
        growth_mean = comparison.loc[comparison["signal_family"].eq("growth"), "false_alarm_rate_63d"].mean()
        cmdty_mean = comparison.loc[comparison["signal_family"].eq("commodity"), "false_alarm_rate_63d"].mean()
        print(f"8. growth cleaner than commodity (mean false alarm 63d): growth={growth_mean:.3f}, commodity={cmdty_mean:.3f}")
    top = overall[overall["overall_recommendation"].isin(["PARTIAL_OVERLAY_STEEP_ONLY", "ENABLE_STEEP_ONLY", "ENABLE_ALL_REGIME"])].head(5)
    print(f"9. recommended next partial overlay triggers: {', '.join(top['signal_name'].tolist())}")
    print(f"10. output path: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
