from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIG = {
    "output_dir": Path("results/commodity_crash_transmission_by_regime"),
    "figure_dir": Path("figures/commodity_crash_transmission_by_regime"),
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

CORE_SIGNALS = [
    "CMDTY_RET60_LT_NEG10",
    "SPY_DD5_AND_CMDTY_RET60_NEG10",
    "SPY_BELOW_MA100_AND_CMDTY_RET60_NEG10",
    "SPY_DD5_CMDTY_RET60_NEG10_CREDIT_WIDEN",
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


def _first_series(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    for name in candidates:
        if name in df.columns:
            obj = df[name]
            if isinstance(obj, pd.DataFrame):
                return obj.iloc[:, 0]
            return obj
    return pd.Series(index=df.index, dtype=float)


def _percentile_series(values: pd.Series) -> pd.Series:
    valid = values.dropna()
    out = pd.Series(np.nan, index=values.index, dtype=float)
    if valid.empty:
        return out
    out.loc[valid.index] = valid.rank(method="average", pct=True)
    return out


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
    panel["spy_price"] = pd.to_numeric(_first_series(panel, ["spy_price"]), errors="coerce")
    panel["daily_rf"] = pd.to_numeric(_first_series(panel, ["daily_rf", "CASH_return"]), errors="coerce").fillna(0.0)
    panel["CASH_return"] = pd.to_numeric(_first_series(panel, ["CASH_return", "daily_rf"]), errors="coerce").fillna(0.0)
    panel["GOLD_return"] = pd.to_numeric(_first_series(panel, ["GOLD_return", "GLD_return"]), errors="coerce")
    panel["IEF_return"] = pd.to_numeric(_first_series(panel, ["IEF_return"]), errors="coerce")
    panel["CMDTY_FUT_return"] = pd.to_numeric(_first_series(panel, ["CMDTY_FUT_return"]), errors="coerce")
    panel["VIX_LEVEL"] = pd.to_numeric(_first_series(panel, ["VIX_LEVEL"]), errors="coerce")
    panel["VIX_ZSCORE_120D"] = pd.to_numeric(_first_series(panel, ["VIX_ZSCORE_120D"]), errors="coerce")
    panel["CREDIT_SPREAD_BAA_AAA"] = pd.to_numeric(_first_series(panel, ["CREDIT_SPREAD_BAA_AAA"]), errors="coerce")
    panel["D_CREDIT_SPREAD_20D"] = pd.to_numeric(_first_series(panel, ["D_CREDIT_SPREAD_20D"]), errors="coerce")
    panel["growth_pc1"] = pd.to_numeric(_first_series(panel, ["growth_pc1"]), errors="coerce")
    panel["inflation_pc1"] = pd.to_numeric(_first_series(panel, ["inflation_pc1"]), errors="coerce")
    panel["term_spread"] = pd.to_numeric(_first_series(panel, ["term_spread"]), errors="coerce")
    panel["GS10"] = pd.to_numeric(_first_series(panel, ["GS10"]), errors="coerce")
    panel["GS1"] = pd.to_numeric(_first_series(panel, ["GS1"]), errors="coerce")
    panel["macro_regime_confirmed"] = panel.get("macro_regime_confirmed", pd.Series("NEUTRAL", index=panel.index)).fillna("NEUTRAL").astype(str)
    panel["monthly_either_state"] = panel.get("monthly_either_state", pd.Series("UNKNOWN", index=panel.index)).fillna("UNKNOWN").astype(str)

    if "spy_drawdown_from_previous_high" not in panel.columns:
        panel["spy_drawdown_from_previous_high"] = panel["spy_price"] / panel["spy_price"].cummax() - 1
    else:
        panel["spy_drawdown_from_previous_high"] = pd.to_numeric(panel["spy_drawdown_from_previous_high"], errors="coerce")
    for w in [20, 50, 100, 200]:
        col = f"SPY_MA{w}"
        if col not in panel.columns:
            panel[col] = panel["spy_price"].rolling(w, min_periods=w).mean()
    panel["SPY_below_MA100"] = panel["spy_price"] < panel["SPY_MA100"]
    panel["SPY_below_MA200"] = panel["spy_price"] < panel["SPY_MA200"]
    panel["SPY_RET_20D"] = panel["spy_price"] / panel["spy_price"].shift(20) - 1.0
    panel["SPY_RET_60D"] = panel["spy_price"] / panel["spy_price"].shift(60) - 1.0
    panel["SPY_RET_120D"] = panel["spy_price"] / panel["spy_price"].shift(120) - 1.0

    if "CMDTY_FUT_price" not in panel.columns:
        panel["CMDTY_FUT_price"] = (1 + panel["CMDTY_FUT_return"].fillna(0.0)).cumprod()
    else:
        panel["CMDTY_FUT_price"] = pd.to_numeric(panel["CMDTY_FUT_price"], errors="coerce").combine_first((1 + panel["CMDTY_FUT_return"].fillna(0.0)).cumprod())
    return panel


def build_commodity_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    price = out["CMDTY_FUT_price"]
    out["CMDTY_RET20"] = price / price.shift(20) - 1.0
    out["CMDTY_RET60"] = price / price.shift(60) - 1.0
    out["CMDTY_RET120"] = price / price.shift(120) - 1.0
    out["CMDTY_DD_FROM_HIGH"] = price / price.cummax() - 1.0
    out["CMDTY_MA60"] = price.rolling(60, min_periods=60).mean()
    out["CMDTY_MA120"] = price.rolling(120, min_periods=120).mean()
    out["CMDTY_BELOW_MA60"] = price < out["CMDTY_MA60"]
    out["CMDTY_BELOW_MA120"] = price < out["CMDTY_MA120"]
    out["growth_pct_full"] = _percentile_series(out["growth_pc1"])
    out["inflation_pct_full"] = _percentile_series(out["inflation_pc1"])
    out["VIX_pct_full"] = _percentile_series(out["VIX_LEVEL"])
    out["credit_pct_full"] = _percentile_series(out["CREDIT_SPREAD_BAA_AAA"])
    out["term_spread_pct_full"] = _percentile_series(out["term_spread"])
    out["GS10_pct_full"] = _percentile_series(out["GS10"])
    out["GS1_pct_full"] = _percentile_series(out["GS1"])
    return out


def build_spy_forward_outcomes(panel: pd.DataFrame, idx: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for h in CONFIG["forward_windows"]:
        fwd = panel["SPY_return"].iloc[idx + 1 : idx + 1 + h].fillna(0.0)
        if fwd.empty:
            out[f"forward_spy_return_{h}d"] = np.nan
            out[f"forward_spy_max_drawdown_{h}d"] = np.nan
            out[f"forward_spy_max_runup_{h}d"] = np.nan
            out[f"days_to_trough_{h}d"] = np.nan
            continue
        nav = (1 + fwd).cumprod()
        dd = nav / nav.cummax() - 1
        runup = nav / nav.cummin() - 1
        out[f"forward_spy_return_{h}d"] = float(nav.iloc[-1] - 1)
        out[f"forward_spy_max_drawdown_{h}d"] = float(dd.min())
        out[f"forward_spy_max_runup_{h}d"] = float(runup.max())
        out[f"days_to_trough_{h}d"] = int(dd.idxmin() - fwd.index[0]) if len(dd) else np.nan
    return out


def build_commodity_signals(panel: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = panel.copy()
    sigs = {
        "CMDTY_RET60_LT_NEG10": out["CMDTY_RET60"] < -0.10,
        "CMDTY_RET120_LT_NEG15": out["CMDTY_RET120"] < -0.15,
        "CMDTY_DD_LT_NEG10": out["CMDTY_DD_FROM_HIGH"] <= -0.10,
        "CMDTY_DD_LT_NEG20": out["CMDTY_DD_FROM_HIGH"] <= -0.20,
        "CMDTY_BELOW_MA60": out["CMDTY_BELOW_MA60"],
        "CMDTY_BELOW_MA120": out["CMDTY_BELOW_MA120"],
        "SPY_DD3_AND_CMDTY_RET60_NEG10": (out["spy_drawdown_from_previous_high"] <= -0.03) & (out["CMDTY_RET60"] < -0.10),
        "SPY_DD5_AND_CMDTY_RET60_NEG10": (out["spy_drawdown_from_previous_high"] <= -0.05) & (out["CMDTY_RET60"] < -0.10),
        "SPY_DD5_AND_CMDTY_DD10": (out["spy_drawdown_from_previous_high"] <= -0.05) & (out["CMDTY_DD_FROM_HIGH"] <= -0.10),
        "SPY_BELOW_MA100_AND_CMDTY_RET60_NEG10": out["SPY_below_MA100"] & (out["CMDTY_RET60"] < -0.10),
        "SPY_BELOW_MA200_AND_CMDTY_RET60_NEG10": out["SPY_below_MA200"] & (out["CMDTY_RET60"] < -0.10),
        "CMDTY_RET60_NEG10_AND_CREDIT_WIDEN": (out["CMDTY_RET60"] < -0.10) & (out["D_CREDIT_SPREAD_20D"] > 0),
        "SPY_DD5_CMDTY_RET60_NEG10_CREDIT_WIDEN": (out["spy_drawdown_from_previous_high"] <= -0.05) & (out["CMDTY_RET60"] < -0.10) & (out["D_CREDIT_SPREAD_20D"] > 0),
        "CMDTY_RET60_NEG10_VIX_LT3": (out["CMDTY_RET60"] < -0.10) & (out["VIX_ZSCORE_120D"] < 3.0),
        "SPY_DD5_CMDTY_RET60_NEG10_VIX_LT3": (out["spy_drawdown_from_previous_high"] <= -0.05) & (out["CMDTY_RET60"] < -0.10) & (out["VIX_ZSCORE_120D"] < 3.0),
        "SPY_BELOW_MA100_CMDTY_RET60_NEG10_VIX_LT3": out["SPY_below_MA100"] & (out["CMDTY_RET60"] < -0.10) & (out["VIX_ZSCORE_120D"] < 3.0),
    }
    for name, ser in sigs.items():
        out[name] = ser.fillna(False)
    return out, list(sigs.keys())


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
                rows.append(
                    {
                        "signal_name": signal_name,
                        "event_date": row["date"],
                        "macro_regime_confirmed": row["macro_regime_confirmed"],
                        "SPY_DD_FROM_HIGH": row["spy_drawdown_from_previous_high"],
                        "CMDTY_RET60": row["CMDTY_RET60"],
                        "CMDTY_RET120": row["CMDTY_RET120"],
                        "CMDTY_DD_FROM_HIGH": row["CMDTY_DD_FROM_HIGH"],
                        "VIX_LEVEL": row["VIX_LEVEL"],
                        "VIX_ZSCORE_120D": row["VIX_ZSCORE_120D"],
                        "CREDIT_SPREAD_BAA_AAA": row["CREDIT_SPREAD_BAA_AAA"],
                        "D_CREDIT_SPREAD_20D": row["D_CREDIT_SPREAD_20D"],
                        "growth_pc1": row.get("growth_pc1"),
                        "inflation_pc1": row.get("inflation_pc1"),
                        "term_spread": row.get("term_spread"),
                        "GS10": row.get("GS10"),
                        "GS1": row.get("GS1"),
                        **build_spy_forward_outcomes(panel, i),
                    }
                )
                last_event_i = i
                has_reset = False
            prev = is_true
    events = pd.DataFrame(rows)
    if events.empty:
        return events
    events["mdd_21d_below_3"] = events["forward_spy_max_drawdown_21d"].le(-0.03)
    events["mdd_21d_below_5"] = events["forward_spy_max_drawdown_21d"].le(-0.05)
    events["mdd_63d_below_5"] = events["forward_spy_max_drawdown_63d"].le(-0.05)
    events["mdd_63d_below_10"] = events["forward_spy_max_drawdown_63d"].le(-0.10)
    events["mdd_126d_below_10"] = events["forward_spy_max_drawdown_126d"].le(-0.10)
    events["false_alarm_21d"] = events["forward_spy_max_drawdown_21d"].gt(-0.03)
    events["false_alarm_63d"] = events["forward_spy_max_drawdown_63d"].gt(-0.05)
    events["quick_rebound_21d"] = events["forward_spy_return_21d"].gt(0.03)
    events["strong_forward_return_63d"] = events["forward_spy_return_63d"].gt(0.08)
    events.to_csv(CONFIG["output_dir"] / "commodity_crash_event_table.csv", index=False)
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
                    "avg_forward_return_21d": sub["forward_spy_return_21d"].mean(),
                    "avg_forward_return_63d": sub["forward_spy_return_63d"].mean(),
                    "avg_forward_return_126d": sub["forward_spy_return_126d"].mean(),
                    "avg_forward_mdd_21d": sub["forward_spy_max_drawdown_21d"].mean(),
                    "avg_forward_mdd_63d": sub["forward_spy_max_drawdown_63d"].mean(),
                    "avg_forward_mdd_126d": sub["forward_spy_max_drawdown_126d"].mean(),
                    "median_forward_mdd_63d": sub["forward_spy_max_drawdown_63d"].median(),
                    "pct_mdd_21d_below_5": sub["mdd_21d_below_5"].mean(),
                    "pct_mdd_63d_below_5": sub["mdd_63d_below_5"].mean(),
                    "pct_mdd_63d_below_10": sub["mdd_63d_below_10"].mean(),
                    "pct_mdd_126d_below_10": sub["mdd_126d_below_10"].mean(),
                    "false_alarm_rate_21d": sub["false_alarm_21d"].mean(),
                    "false_alarm_rate_63d": sub["false_alarm_63d"].mean(),
                    "quick_rebound_rate_21d": sub["quick_rebound_21d"].mean(),
                    "avg_days_to_trough_63d": sub["days_to_trough_63d"].mean(),
                    "median_spy_dd_at_event": sub["SPY_DD_FROM_HIGH"].median(),
                    "median_cmdty_ret60_at_event": sub["CMDTY_RET60"].median(),
                    "median_vix_z_at_event": sub["VIX_ZSCORE_120D"].median(),
                    "median_credit_chg20_at_event": sub["D_CREDIT_SPREAD_20D"].median(),
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    by_regime = summarize(events, True)
    full = summarize(events, False)
    by_regime.to_csv(CONFIG["output_dir"] / "commodity_crash_summary_by_regime.csv", index=False)
    full.to_csv(CONFIG["output_dir"] / "commodity_crash_summary_full_sample.csv", index=False)
    return by_regime, full


def build_regime_decision_table(by_regime: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    regime_rows = []
    for _, row in by_regime.iterrows():
        if row["event_count"] < 5:
            action = "INSUFFICIENT_SAMPLE"
        elif row["false_alarm_rate_63d"] <= 0.40 and row["pct_mdd_63d_below_5"] >= 0.50 and row["avg_forward_mdd_63d"] <= -0.05:
            action = "ENABLE_FOR_REGIME"
        elif row["false_alarm_rate_63d"] > 0.60 or row["quick_rebound_rate_21d"] > 0.40:
            action = "DISABLE_FOR_REGIME"
        else:
            action = "DIAGNOSTIC_ONLY"
        x = row.to_dict()
        x["recommended_action"] = action
        regime_rows.append(x)
    regime_table = pd.DataFrame(regime_rows)
    regime_table.to_csv(CONFIG["output_dir"] / "commodity_trigger_regime_decision_table.csv", index=False)

    overall_rows = []
    for signal_name, sub in regime_table.groupby("signal_name"):
        steep_ok = any((sub["macro_regime_confirmed"] == "STEEP") & (sub["recommended_action"] == "ENABLE_FOR_REGIME"))
        all_enable = not sub.empty and (sub["recommended_action"] == "ENABLE_FOR_REGIME").all()
        if steep_ok and sub.loc[sub["macro_regime_confirmed"] != "STEEP", "recommended_action"].isin(["DISABLE_FOR_REGIME", "DIAGNOSTIC_ONLY", "INSUFFICIENT_SAMPLE"]).all():
            rec = "ENABLE_STEEP_ONLY"
        elif all_enable:
            rec = "ENABLE_ALL_REGIME"
        elif (sub["recommended_action"] == "ENABLE_FOR_REGIME").sum() == 0:
            rec = "DISABLE" if (sub["recommended_action"] == "DISABLE_FOR_REGIME").any() else "DIAGNOSTIC_ONLY"
        else:
            rec = "DIAGNOSTIC_ONLY"
        overall_rows.append({"signal_name": signal_name, "overall_recommendation": rec})
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(CONFIG["output_dir"] / "commodity_trigger_overall_recommendation.csv", index=False)
    return regime_table, overall


def map_2015_2016_case(panel: pd.DataFrame, signals: List[str], overall: pd.DataFrame) -> pd.DataFrame:
    start = pd.Timestamp(CONFIG["case_2015_start"])
    end = pd.Timestamp(CONFIG["case_2015_end"])
    peak = pd.Timestamp(CONFIG["case_2015_peak"])
    trough = pd.Timestamp(CONFIG["case_2015_trough"])
    case = panel[(panel["date"] >= start) & (panel["date"] <= end)].copy()
    backbone_entry = case.loc[case.get("BACKBONE_V2_ENTRY_SIGNAL", pd.Series(False, index=case.index)).fillna(False), "date"]
    backbone_entry_date = backbone_entry.min() if not backbone_entry.empty else pd.NaT
    monthly_sell = case.loc[case["monthly_either_state"].eq("SELL"), "date"]
    monthly_sell_date = monthly_sell.min() if not monthly_sell.empty else pd.NaT
    rows = []
    for signal_name in signals:
        hit = case.loc[case[signal_name].fillna(False)]
        first = hit.iloc[0] if not hit.empty else None
        rec = overall.loc[overall["signal_name"].eq(signal_name), "overall_recommendation"]
        rows.append(
            {
                "signal_name": signal_name,
                "first_trigger_date_in_case": first["date"] if first is not None else pd.NaT,
                "macro_regime_at_trigger": first["macro_regime_confirmed"] if first is not None else np.nan,
                "days_after_peak": (pd.Timestamp(first["date"]) - peak).days if first is not None else np.nan,
                "days_before_trough": (trough - pd.Timestamp(first["date"])).days if first is not None else np.nan,
                "SPY_DD_at_trigger": first["spy_drawdown_from_previous_high"] if first is not None else np.nan,
                "CMDTY_RET60_at_trigger": first["CMDTY_RET60"] if first is not None else np.nan,
                "VIX_Z_at_trigger": first["VIX_ZSCORE_120D"] if first is not None else np.nan,
                "credit_chg20_at_trigger": first["D_CREDIT_SPREAD_20D"] if first is not None else np.nan,
                "triggered_before_backbone_entry": pd.notna(backbone_entry_date) and first is not None and pd.Timestamp(first["date"]) < backbone_entry_date,
                "triggered_before_monthly_either_sell": pd.notna(monthly_sell_date) and first is not None and pd.Timestamp(first["date"]) < monthly_sell_date,
                "whether_signal_is_STEEP_enabled_candidate": (rec.iloc[0] == "ENABLE_STEEP_ONLY") if not rec.empty else False,
            }
        )
    out = pd.DataFrame(rows).sort_values(["first_trigger_date_in_case", "signal_name"], na_position="last")
    out.to_csv(CONFIG["output_dir"] / "case_2015_2016_signal_mapping.csv", index=False)
    return out


def analyze_flat_inverted_cases(events: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sub = events[events["macro_regime_confirmed"].isin(["FLAT", "INVERTED"])].copy()
    sub.to_csv(CONFIG["output_dir"] / "flat_inverted_commodity_crash_cases.csv", index=False)
    rows = []
    if not sub.empty:
        for regime in ["FLAT", "INVERTED"]:
            reg = sub[sub["macro_regime_confirmed"].eq(regime)].copy()
            if reg.empty:
                continue
            worst = reg.sort_values("forward_spy_max_drawdown_63d").head(3)
            for _, row in worst.iterrows():
                rows.append({"case_type": f"worst_{regime}_by_mdd63", **row.to_dict()})
            false_alarms = reg[reg["false_alarm_63d"]].sort_values("CMDTY_RET60").head(3)
            for _, row in false_alarms.iterrows():
                rows.append({"case_type": f"false_alarm_{regime}", **row.to_dict()})
    out = pd.DataFrame(rows)
    out.to_csv(CONFIG["output_dir"] / "flat_inverted_case_summary.csv", index=False)
    return sub, out


def compute_macro_context(events: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    p["growth_pct_full"] = _percentile_series(p["growth_pc1"])
    p["inflation_pct_full"] = _percentile_series(p["inflation_pc1"])
    p["VIX_pct_full"] = _percentile_series(p["VIX_LEVEL"])
    p["credit_pct_full"] = _percentile_series(p["CREDIT_SPREAD_BAA_AAA"])
    p["term_spread_pct_full"] = _percentile_series(p["term_spread"])
    p["GS10_pct_full"] = _percentile_series(p["GS10"])
    p["GS1_pct_full"] = _percentile_series(p["GS1"])
    merged = events.merge(
        p[["date", "growth_pct_full", "inflation_pct_full", "VIX_pct_full", "credit_pct_full", "term_spread_pct_full", "GS10_pct_full", "GS1_pct_full"]],
        left_on="event_date",
        right_on="date",
        how="left",
    )
    rows = []
    for (signal_name, regime), sub in merged.groupby(["signal_name", "macro_regime_confirmed"]):
        rows.append(
            {
                "signal_name": signal_name,
                "macro_regime_confirmed": regime,
                "avg_growth_pc1_percentile": sub["growth_pct_full"].mean(),
                "avg_inflation_pc1_percentile": sub["inflation_pct_full"].mean(),
                "avg_VIX_percentile": sub["VIX_pct_full"].mean(),
                "avg_credit_spread_percentile": sub["credit_pct_full"].mean(),
                "avg_term_spread_percentile": sub["term_spread_pct_full"].mean(),
                "avg_GS10_percentile": sub["GS10_pct_full"].mean(),
                "avg_GS1_percentile": sub["GS1_pct_full"].mean(),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(CONFIG["output_dir"] / "commodity_crash_macro_context.csv", index=False)
    return out


def plot_signal_quality(by_regime: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    for regime, sub in by_regime.groupby("macro_regime_confirmed"):
        ax.scatter(sub["false_alarm_rate_63d"], sub["pct_mdd_63d_below_5"], s=sub["event_count"] * 10, alpha=0.7, label=regime)
    for _, row in by_regime[by_regime["signal_name"].isin(CORE_SIGNALS)].iterrows():
        ax.text(row["false_alarm_rate_63d"], row["pct_mdd_63d_below_5"], f"{row['signal_name']}|{row['macro_regime_confirmed']}", fontsize=6)
    ax.set_xlabel("False alarm rate 63d")
    ax.set_ylabel("P(63d MDD < -5%)")
    ax.set_title("Commodity Signal Quality by Regime")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "commodity_signal_quality_by_regime_scatter.png", dpi=150)
    plt.close(fig)


def plot_case_studies(panel: pd.DataFrame, case_map: pd.DataFrame, flat_cases: pd.DataFrame, macro_context: pd.DataFrame, by_regime: pd.DataFrame) -> None:
    start = pd.Timestamp(CONFIG["case_2015_start"])
    end = pd.Timestamp(CONFIG["case_2015_end"])
    case = panel[(panel["date"] >= start) & (panel["date"] <= end)].copy()
    fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True)
    axes[0].plot(case["date"], case["spy_drawdown_from_previous_high"], color="black")
    axes[0].set_title("SPY Drawdown")
    axes[1].plot(case["date"], case["CMDTY_RET60"])
    axes[1].set_title("CMDTY_RET60")
    axes[2].plot(case["date"], case["CMDTY_DD_FROM_HIGH"])
    axes[2].set_title("CMDTY Drawdown")
    axes[3].plot(case["date"], case["VIX_ZSCORE_120D"])
    axes[3].set_title("VIX z-score")
    axes[4].plot(case["date"], case["D_CREDIT_SPREAD_20D"])
    axes[4].set_title("Credit change 20D")
    axes[5].plot(case["date"], case["macro_regime_confirmed"].astype("category").cat.codes)
    axes[5].set_title("Regime")
    for _, row in case_map[case_map["signal_name"].isin(CORE_SIGNALS)].dropna(subset=["first_trigger_date_in_case"]).iterrows():
        for ax in axes:
            ax.axvline(pd.Timestamp(row["first_trigger_date_in_case"]), color="tab:red", alpha=0.2)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "case_2015_2016_commodity_signal_timeline.png", dpi=150)
    plt.close(fig)

    if not flat_cases.empty:
        fig, ax = plt.subplots(figsize=(12, 7))
        for _, row in flat_cases.head(20).iterrows():
            vals = [0, row.get("forward_spy_return_21d", np.nan), row.get("forward_spy_return_63d", np.nan), row.get("forward_spy_return_126d", np.nan)]
            ax.plot([0, 21, 63, 126], vals, alpha=0.35, color="tab:blue" if row["macro_regime_confirmed"] == "FLAT" else "tab:orange")
        ax.set_title("FLAT / INVERTED Commodity Crash Forward Paths")
        ax.set_xlabel("Forward horizon")
        ax.set_ylabel("Cumulative SPY return")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(CONFIG["figure_dir"] / "flat_inverted_commodity_crash_cases.png", dpi=150)
        plt.close(fig)

    pivot = by_regime[by_regime["signal_name"].isin(CORE_SIGNALS)].pivot(index="signal_name", columns="macro_regime_confirmed", values="pct_mdd_63d_below_5")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.fillna(np.nan), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Commodity Transmission Heatmap")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "commodity_transmission_heatmap.png", dpi=150)
    plt.close(fig)

    if not macro_context.empty:
        mc = macro_context[macro_context["signal_name"].isin(CORE_SIGNALS)].copy()
        mat = mc.set_index(["signal_name", "macro_regime_confirmed"])[[
            "avg_growth_pc1_percentile",
            "avg_inflation_pc1_percentile",
            "avg_VIX_percentile",
            "avg_credit_spread_percentile",
            "avg_term_spread_percentile",
        ]]
        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(mat.fillna(np.nan), aspect="auto", cmap="coolwarm")
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels(mat.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels([f"{a}|{b}" for a, b in mat.index])
        ax.set_title("Macro Context Heatmap")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(CONFIG["figure_dir"] / "macro_context_heatmap.png", dpi=150)
        plt.close(fig)


def write_markdown_report(full: pd.DataFrame, by_regime: pd.DataFrame, regime_table: pd.DataFrame, overall: pd.DataFrame, case_map: pd.DataFrame, flat_summary: pd.DataFrame, macro_context: pd.DataFrame) -> None:
    out = CONFIG["output_dir"] / "COMMODITY_CRASH_TRANSMISSION_BY_REGIME_REPORT.md"
    content = f"""# COMMODITY_CRASH_TRANSMISSION_BY_REGIME_REPORT

## Purpose

This report tests whether commodity weakness should be treated as an all-regime trigger or a STEEP-specific slow-stress signal.

## Full-sample Event Quality

{full.to_markdown(index=False)}

## By-regime Transmission

{by_regime.to_markdown(index=False)}

## Regime Decision Table

{regime_table.to_markdown(index=False)}

## Overall Recommendation

{overall.to_markdown(index=False)}

## 2015-2016 Mapping

{case_map.to_markdown(index=False)}

## FLAT / INVERTED Cases

{flat_summary.to_markdown(index=False)}

## Macro Context

{macro_context.to_markdown(index=False)}
"""
    out.write_text(content, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = load_panel()
    panel = build_commodity_features(panel)
    panel, signals = build_commodity_signals(panel)
    events = extract_signal_events(panel, signals)
    by_regime, full = summarize_events_by_regime(events)
    regime_table, overall = build_regime_decision_table(by_regime)
    case_map = map_2015_2016_case(panel, signals, overall)
    flat_cases, flat_summary = analyze_flat_inverted_cases(events)
    macro_context = compute_macro_context(events, panel)
    plot_signal_quality(by_regime)
    plot_case_studies(panel, case_map, flat_cases, macro_context, by_regime)
    write_markdown_report(full, by_regime, regime_table, overall, case_map, flat_summary, macro_context)

    print(f"1. commodity crash signal total events: {len(events)}")
    for regime in ["STEEP", "FLAT", "INVERTED"]:
        sub = by_regime[(by_regime["macro_regime_confirmed"].eq(regime)) & (by_regime["signal_name"].isin(CORE_SIGNALS))]
        if not sub.empty:
            print(f"2/3/4. {regime} core signal quality:")
            print(sub[["signal_name", "false_alarm_rate_63d", "pct_mdd_63d_below_5"]].to_string(index=False))
    earliest = case_map.sort_values("first_trigger_date_in_case").head(1)
    if not earliest.empty:
        print(f"5. earliest 2015-2016 commodity signal: {earliest.iloc[0]['signal_name']} @ {earliest.iloc[0]['first_trigger_date_in_case']}")
    fi_bad = flat_cases.loc[flat_cases["forward_spy_max_drawdown_63d"].le(-0.10)]
    print(f"6. FLAT / INVERTED commodity crash with real SPY damage exists: {not fi_bad.empty}")
    rec_counts = overall["overall_recommendation"].value_counts().to_dict()
    print(f"7. recommendation mix: {rec_counts}")
    top = overall[overall["overall_recommendation"].isin(['ENABLE_STEEP_ONLY','ENABLE_ALL_REGIME'])]["signal_name"].tolist()[:5]
    print(f"8. next partial overlay candidates: {', '.join(top)}")
    print(f"9. output path: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
