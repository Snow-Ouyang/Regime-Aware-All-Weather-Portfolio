"""Mature regime hedge baseline with STEEP-only commodity slow-growth stress tests."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIG = {
    "output_dir": Path("results/backbone_v2_with_steep_commodity_stress"),
    "figure_dir": Path("figures/backbone_v2_with_steep_commodity_stress"),
    "one_way_cost_bps": 5,
    "recovery_ma_window": 20,
    "commodity_ret_window": 60,
    "commodity_ret_threshold": -0.10,
    "dd_threshold": -0.05,
    "credit_widen_threshold": 0.0,
    "vix_z_threshold": 3.0,
    "case_2015_start": "2015-05-01",
    "case_2015_peak": "2015-07-20",
    "case_2015_trough": "2016-02-11",
    "case_2015_end": "2016-03-31",
    "trading_days_per_year": 252,
}

PANEL_CANDIDATES = [
    Path("results/regime_aware_risk_parity_allocation/daily_backtest_panel.csv"),
    Path("results/spy_cash_backbone_upgrade_ablation/daily_backtest_panel.csv"),
    Path("results/drawdown_2015_2016_forensic_diagnostic/forensic_daily_panel.csv"),
    Path("results/flat_vix_credit_trigger_diagnostic/full_backtest_panel.csv"),
]

CRISIS_WINDOWS = {
    "2008_GFC": ("2007-10-01", "2009-06-30"),
    "2015_2016": ("2015-05-01", "2016-03-31"),
    "2018Q4": ("2018-10-01", "2019-01-31"),
    "COVID_2020": ("2020-02-01", "2020-06-30"),
    "2022": ("2021-11-01", "2023-03-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2025_PULLBACK": ("2025-01-01", "2025-12-31"),
    "2024_2026": ("2024-01-01", "2026-12-31"),
}

ASSETS = ["SPY", "GOLD", "CMDTY_FUT", "IEF", "CASH"]
OVERLAY_RULES = {
    "Overlay_20": {"SPY": 0.80, "IEF": 0.10, "GOLD": 0.10, "CMDTY_FUT": 0.0, "CASH": 0.0},
    "Overlay_30": {"SPY": 0.70, "IEF": 0.15, "GOLD": 0.15, "CMDTY_FUT": 0.0, "CASH": 0.0},
    "Overlay_40": {"SPY": 0.60, "IEF": 0.20, "GOLD": 0.20, "CMDTY_FUT": 0.0, "CASH": 0.0},
}


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


def _first_series(df: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    for name in names:
        if name in df.columns:
            obj = df[name]
            if isinstance(obj, pd.DataFrame):
                return obj.iloc[:, 0]
            return obj
    return pd.Series(index=df.index, dtype=float)


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    out = {asset: float(weights.get(asset, 0.0)) for asset in ASSETS}
    total = sum(out.values())
    if total <= 0:
        raise ValueError("Weight sum must be positive.")
    return {k: v / total for k, v in out.items()}


def load_panel() -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    panel["spy_price"] = pd.to_numeric(_first_series(panel, ["spy_price", "SPY_NAV", "SPY_BUY_HOLD_nav"]), errors="coerce")
    panel["spy_daily_return"] = pd.to_numeric(_first_series(panel, ["spy_daily_return", "SPY_return"]), errors="coerce")
    panel["daily_rf"] = pd.to_numeric(_first_series(panel, ["daily_rf", "CASH_return"]), errors="coerce").fillna(0.0)
    panel["macro_regime_confirmed"] = panel.get("macro_regime_confirmed", pd.Series(index=panel.index, dtype=object)).fillna("UNKNOWN").astype(str)
    panel["monthly_either_state"] = panel.get("monthly_either_state", pd.Series(index=panel.index, dtype=object)).fillna("UNKNOWN").astype(str)
    panel["VIX_LEVEL"] = pd.to_numeric(_first_series(panel, ["VIX_LEVEL"]), errors="coerce")
    panel["VIX_ZSCORE_120D"] = pd.to_numeric(_first_series(panel, ["VIX_ZSCORE_120D"]), errors="coerce")
    panel["CREDIT_SPREAD_BAA_AAA"] = pd.to_numeric(_first_series(panel, ["CREDIT_SPREAD_BAA_AAA"]), errors="coerce")
    panel["D_CREDIT_SPREAD_20D"] = pd.to_numeric(_first_series(panel, ["D_CREDIT_SPREAD_20D"]), errors="coerce")
    panel["SPY_MA20"] = pd.to_numeric(_first_series(panel, ["SPY_MA20"]), errors="coerce")
    panel["SPY_CROSS_ABOVE_MA20"] = _first_series(panel, ["SPY_CROSS_ABOVE_MA20"]).astype("boolean")
    panel["CMDTY_FUT_return"] = pd.to_numeric(_first_series(panel, ["CMDTY_FUT_return", "CMDTY_FUT_RETURN", "CMDTY_return", "CMDTY_RETURN", "CMDTY_ret"]), errors="coerce")
    panel["CMDTY_FUT_price"] = pd.to_numeric(_first_series(panel, ["CMDTY_FUT_price", "CMDTY_FUT_NAV"]), errors="coerce")
    panel["GOLD_return"] = pd.to_numeric(_first_series(panel, ["GOLD_return", "GLD_return"]), errors="coerce")
    panel["IEF_return"] = pd.to_numeric(_first_series(panel, ["IEF_return"]), errors="coerce")
    panel["CASH_return"] = pd.to_numeric(_first_series(panel, ["CASH_return", "daily_rf"]), errors="coerce").fillna(0.0)
    panel["FLAT_INV_VOL_weight_SPY"] = pd.to_numeric(_first_series(panel, ["FLAT_INV_VOL_weight_SPY"]), errors="coerce")
    panel["FLAT_INV_VOL_weight_GOLD"] = pd.to_numeric(_first_series(panel, ["FLAT_INV_VOL_weight_GOLD"]), errors="coerce")
    panel["FLAT_INV_VOL_weight_CMDTY_FUT"] = pd.to_numeric(_first_series(panel, ["FLAT_INV_VOL_weight_CMDTY_FUT"]), errors="coerce")
    panel["INVERTED_INV_VOL_weight_SPY"] = pd.to_numeric(_first_series(panel, ["INVERTED_INV_VOL_weight_SPY"]), errors="coerce")
    panel["INVERTED_INV_VOL_weight_GOLD"] = pd.to_numeric(_first_series(panel, ["INVERTED_INV_VOL_weight_GOLD"]), errors="coerce")

    if panel["spy_daily_return"].isna().all():
        panel["spy_daily_return"] = panel["spy_price"].pct_change()
    if panel["spy_price"].isna().all():
        if panel["spy_daily_return"].notna().any():
            panel["spy_price"] = (1 + panel["spy_daily_return"].fillna(0.0)).cumprod()
        else:
            raise ValueError("Missing spy_price and unable to rebuild from SPY returns.")
    if panel["CMDTY_FUT_price"].isna().all():
        panel["CMDTY_FUT_price"] = (1 + panel["CMDTY_FUT_return"].fillna(0.0)).cumprod()
    else:
        rebuild = (1 + panel["CMDTY_FUT_return"].fillna(0.0)).cumprod()
        panel["CMDTY_FUT_price"] = panel["CMDTY_FUT_price"].combine_first(rebuild)
    if panel["CMDTY_FUT_return"].isna().all():
        panel["CMDTY_FUT_return"] = panel["CMDTY_FUT_price"].pct_change()

    if panel["spy_daily_return"].isna().any():
        panel["spy_daily_return"] = panel["spy_daily_return"].fillna(0.0)
    if "spy_drawdown_from_previous_high" not in panel.columns:
        panel["spy_drawdown_from_previous_high"] = panel["spy_price"] / panel["spy_price"].cummax() - 1.0
    else:
        panel["spy_drawdown_from_previous_high"] = pd.to_numeric(panel["spy_drawdown_from_previous_high"], errors="coerce")
        panel["spy_drawdown_from_previous_high"] = panel["spy_drawdown_from_previous_high"].fillna(panel["spy_price"] / panel["spy_price"].cummax() - 1.0)
    if panel["SPY_MA20"].isna().all():
        panel["SPY_MA20"] = panel["spy_price"].rolling(CONFIG["recovery_ma_window"], min_periods=CONFIG["recovery_ma_window"]).mean()
    if panel["SPY_CROSS_ABOVE_MA20"].isna().all():
        panel["SPY_CROSS_ABOVE_MA20"] = (panel["spy_price"] > panel["SPY_MA20"]) & (
            panel["spy_price"].shift(1) <= panel["SPY_MA20"].shift(1)
        )
    if panel["VIX_ZSCORE_120D"].isna().all() and panel["VIX_LEVEL"].notna().any():
        roll = panel["VIX_LEVEL"].rolling(120, min_periods=120)
        panel["VIX_ZSCORE_120D"] = (panel["VIX_LEVEL"] - roll.mean()) / roll.std(ddof=0)
    if panel["D_CREDIT_SPREAD_20D"].isna().all() and panel["CREDIT_SPREAD_BAA_AAA"].notna().any():
        panel["D_CREDIT_SPREAD_20D"] = panel["CREDIT_SPREAD_BAA_AAA"] - panel["CREDIT_SPREAD_BAA_AAA"].shift(20)

    abnormal = panel.loc[~panel["macro_regime_confirmed"].isin(["FLAT", "STEEP", "INVERTED"]), ["date", "macro_regime_confirmed"]].copy()
    if not abnormal.empty:
        warnings.warn(f"Unexpected regimes found: {abnormal['macro_regime_confirmed'].value_counts().to_dict()}")
    return panel, abnormal


def build_spy_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SPY_MA100"] = out["spy_price"].rolling(100, min_periods=100).mean()
    out["SPY_MA200"] = out["spy_price"].rolling(200, min_periods=200).mean()
    out["SPY_below_MA100"] = out["spy_price"] < out["SPY_MA100"]
    out["SPY_below_MA200"] = out["spy_price"] < out["SPY_MA200"]
    out["is_month_start"] = out["date"].dt.to_period("M").ne(out["date"].dt.to_period("M").shift(1, fill_value=out["date"].dt.to_period("M").iloc[0]))
    return out


def build_commodity_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["CMDTY_RET60"] = out["CMDTY_FUT_price"] / out["CMDTY_FUT_price"].shift(CONFIG["commodity_ret_window"]) - 1.0
    return out


def build_baseline_v2_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timing_state" in out.columns:
        out["BACKBONE_V2_BASELINE_RISK_STATE"] = out["timing_state"].fillna("NON_RISK")
    else:
        out["BACKBONE_V2_BASELINE_RISK_STATE"] = "NON_RISK"
    out["FLAT_VIX_STRESS"] = out["macro_regime_confirmed"].eq("FLAT") & (out["VIX_ZSCORE_120D"] >= CONFIG["vix_z_threshold"])
    out["FLAT_CREDIT_DD5_STRESS"] = out["macro_regime_confirmed"].eq("FLAT") & (
        (out["spy_drawdown_from_previous_high"] <= CONFIG["dd_threshold"])
        & (out["D_CREDIT_SPREAD_20D"] > 0.10)
    )
    out["FLAT_VIX_OR_CREDIT_STRESS"] = out["FLAT_VIX_STRESS"] | out["FLAT_CREDIT_DD5_STRESS"]
    out["STEEP_EITHER_SELL_STRESS"] = out["macro_regime_confirmed"].eq("STEEP") & out["monthly_either_state"].eq("SELL")
    out["STEEP_CREDIT_DD5_STRESS"] = out["macro_regime_confirmed"].eq("STEEP") & (
        (out["spy_drawdown_from_previous_high"] <= CONFIG["dd_threshold"])
        & (out["D_CREDIT_SPREAD_20D"] > 0.10)
    )
    out["BACKBONE_V2_BASELINE_ENTRY"] = (
        out["FLAT_VIX_OR_CREDIT_STRESS"] | out["STEEP_EITHER_SELL_STRESS"] | out["STEEP_CREDIT_DD5_STRESS"]
    )
    return out


def build_steep_commodity_triggers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["STEEP_CMDTY_RET60_NEG10"] = out["macro_regime_confirmed"].eq("STEEP") & (out["CMDTY_RET60"] < CONFIG["commodity_ret_threshold"])
    out["STEEP_CMDTY_RET60_NEG10_AND_CREDIT_WIDEN"] = out["macro_regime_confirmed"].eq("STEEP") & (
        (out["CMDTY_RET60"] < CONFIG["commodity_ret_threshold"]) & (out["D_CREDIT_SPREAD_20D"] > CONFIG["credit_widen_threshold"])
    )
    out["STEEP_SPY_DD5_AND_CMDTY_RET60_NEG10"] = out["macro_regime_confirmed"].eq("STEEP") & (
        (out["spy_drawdown_from_previous_high"] <= CONFIG["dd_threshold"]) & (out["CMDTY_RET60"] < CONFIG["commodity_ret_threshold"])
    )
    return out


def _baseline_reason(row: pd.Series) -> str:
    if bool(row["FLAT_VIX_STRESS"]):
        return "FLAT_VIX_STRESS"
    if bool(row["FLAT_CREDIT_DD5_STRESS"]):
        return "FLAT_CREDIT_DD5_STRESS"
    if bool(row["STEEP_EITHER_SELL_STRESS"]):
        return "STEEP_EITHER_SELL_STRESS"
    if bool(row["STEEP_CREDIT_DD5_STRESS"]):
        return "STEEP_CREDIT_DD5_STRESS"
    return ""


def _commodity_reason(row: pd.Series, signal_set: str) -> str:
    if signal_set == "ONE_RET60":
        return "STEEP_CMDTY_RET60_NEG10" if bool(row["STEEP_CMDTY_RET60_NEG10"]) else ""
    if signal_set == "ONE_CREDIT":
        return "STEEP_CMDTY_RET60_NEG10_AND_CREDIT_WIDEN" if bool(row["STEEP_CMDTY_RET60_NEG10_AND_CREDIT_WIDEN"]) else ""
    if signal_set == "ONE_DD5":
        return "STEEP_SPY_DD5_AND_CMDTY_RET60_NEG10" if bool(row["STEEP_SPY_DD5_AND_CMDTY_RET60_NEG10"]) else ""
    if signal_set == "ALL":
        if bool(row["STEEP_SPY_DD5_AND_CMDTY_RET60_NEG10"]):
            return "STEEP_SPY_DD5_AND_CMDTY_RET60_NEG10"
        if bool(row["STEEP_CMDTY_RET60_NEG10_AND_CREDIT_WIDEN"]):
            return "STEEP_CMDTY_RET60_NEG10_AND_CREDIT_WIDEN"
        if bool(row["STEEP_CMDTY_RET60_NEG10"]):
            return "STEEP_CMDTY_RET60_NEG10"
    return ""


def _baseline_weight_dict(row: pd.Series) -> Dict[str, float]:
    cols = {asset: pd.to_numeric(row.get(f"REGIME_HEDGE_INV_VOL_weight_{asset}", np.nan), errors="coerce") for asset in ASSETS}
    if all(pd.notna(v) for v in cols.values()):
        return _normalize_weights(cols)
    regime = row["macro_regime_confirmed"]
    timing_state = row.get("timing_state", "NON_RISK")
    if regime == "FLAT":
        if timing_state == "RISK":
            return _normalize_weights({"GOLD": 1.0})
        return _normalize_weights(
            {
                "SPY": row["FLAT_INV_VOL_weight_SPY"],
                "GOLD": row["FLAT_INV_VOL_weight_GOLD"],
                "CMDTY_FUT": row["FLAT_INV_VOL_weight_CMDTY_FUT"],
            }
        )
    if regime == "STEEP":
        return _normalize_weights({"IEF": 0.80, "GOLD": 0.20}) if timing_state == "RISK" else _normalize_weights({"SPY": 1.0})
    if regime == "INVERTED":
        return _normalize_weights({"SPY": row["INVERTED_INV_VOL_weight_SPY"], "GOLD": row["INVERTED_INV_VOL_weight_GOLD"]})
    return {}


def _target_weights(row: pd.Series, mode: str, overlay_name: Optional[str], warnings_out: List[Dict[str, object]]) -> Dict[str, float]:
    regime = row["macro_regime_confirmed"]
    if regime == "FLAT":
        if mode == "FULL_RISK":
            return _normalize_weights({"GOLD": 1.0})
        return _normalize_weights(
            {
                "SPY": row["FLAT_INV_VOL_weight_SPY"],
                "GOLD": row["FLAT_INV_VOL_weight_GOLD"],
                "CMDTY_FUT": row["FLAT_INV_VOL_weight_CMDTY_FUT"],
            }
        )
    if regime == "STEEP":
        if mode == "FULL_RISK":
            return _normalize_weights({"IEF": 0.80, "GOLD": 0.20})
        if mode == "OVERLAY":
            if overlay_name is None:
                raise ValueError("Overlay mode requires overlay_name.")
            return _normalize_weights(OVERLAY_RULES[overlay_name])
        return _normalize_weights({"SPY": 1.0})
    if regime == "INVERTED":
        return _normalize_weights({"SPY": row["INVERTED_INV_VOL_weight_SPY"], "GOLD": row["INVERTED_INV_VOL_weight_GOLD"]})
    warnings_out.append({"date": row["date"], "macro_regime_confirmed": regime})
    return {}


def run_state_machine_backtest(
    df: pd.DataFrame,
    strategy_name: str,
    commodity_mode: str,
    signal_set: Optional[str] = None,
    overlay_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    current_override = "NONE"
    pending_override = "NONE"
    pending_reason = ""
    prev_weights = {asset: 0.0 for asset in ASSETS}
    event_rows: List[Dict[str, object]] = []
    daily_rows: List[Dict[str, object]] = []
    warnings_out: List[Dict[str, object]] = []

    for i, row in df.iterrows():
        current_override = pending_override
        baseline_full = str(row.get("BACKBONE_V2_BASELINE_RISK_STATE", "NON_RISK")) == "RISK"
        baseline_weights = _baseline_weight_dict(row)
        if baseline_full:
            effective_state = "FULL_RISK"
            target = baseline_weights
        elif current_override == "FULL_RISK":
            effective_state = "FULL_RISK"
            target = _normalize_weights({"IEF": 0.80, "GOLD": 0.20})
        elif current_override == "OVERLAY":
            effective_state = "OVERLAY"
            target = _normalize_weights(OVERLAY_RULES[overlay_name]) if overlay_name is not None else {}
        else:
            effective_state = "NON_RISK"
            target = baseline_weights
        turnover = 0.0
        cost = 0.0
        if target:
            weight_changed = any(abs(target[a] - prev_weights.get(a, 0.0)) > 1e-12 for a in ASSETS)
            if i == 0 or bool(row["is_month_start"]) or weight_changed:
                turnover = sum(abs(target[a] - prev_weights.get(a, 0.0)) for a in ASSETS)
                cost = 0.5 * turnover * CONFIG["one_way_cost_bps"] / 10000.0
                if weight_changed and i > 0:
                    prev_state = daily_rows[-1]["state"]
                    if effective_state != prev_state:
                        event_type = "ENTER_RISK" if effective_state == "FULL_RISK" else "ENTER_OVERLAY" if effective_state == "OVERLAY" else "EXIT_TO_NON_RISK"
                        reason_key = "entry_reason" if event_type != "EXIT_TO_NON_RISK" else "exit_reason"
                        event_rows.append(
                            {
                                "strategy": strategy_name,
                                "event_date": row["date"],
                                "event_type": event_type,
                                reason_key: pending_reason if event_type != "EXIT_TO_NON_RISK" else "R3_SPY_CROSS_ABOVE_MA20",
                                "macro_regime_confirmed": row["macro_regime_confirmed"],
                                "monthly_either_state": row["monthly_either_state"],
                                "VIX_LEVEL": row["VIX_LEVEL"],
                                "VIX_ZSCORE_120D": row["VIX_ZSCORE_120D"],
                                "CREDIT_SPREAD_BAA_AAA": row["CREDIT_SPREAD_BAA_AAA"],
                                "D_CREDIT_SPREAD_20D": row["D_CREDIT_SPREAD_20D"],
                                "CMDTY_RET60": row["CMDTY_RET60"],
                                "spy_drawdown_from_previous_high": row["spy_drawdown_from_previous_high"],
                                "SPY_price": row["spy_price"],
                                "SPY_MA20": row["SPY_MA20"],
                                "previous_state": prev_state,
                                "new_state": effective_state,
                            }
                        )
                prev_weights = target.copy()
        ret = (
            prev_weights["SPY"] * row["spy_daily_return"]
            + prev_weights["GOLD"] * row["GOLD_return"]
            + prev_weights["CMDTY_FUT"] * row["CMDTY_FUT_return"]
            + prev_weights["IEF"] * row["IEF_return"]
            + prev_weights["CASH"] * row["CASH_return"]
            - cost
        )
        daily_rows.append(
            {
                "state": effective_state,
                "weight_SPY": prev_weights["SPY"],
                "weight_GOLD": prev_weights["GOLD"],
                "weight_CMDTY_FUT": prev_weights["CMDTY_FUT"],
                "weight_IEF": prev_weights["IEF"],
                "weight_CASH": prev_weights["CASH"],
                "strategy_return": ret,
                "turnover": turnover,
                "transaction_cost": cost,
            }
        )

        commodity_reason = _commodity_reason(row, signal_set) if signal_set else ""
        commodity_fire = bool(commodity_reason)
        recovery = bool(row["SPY_CROSS_ABOVE_MA20"])
        next_override = current_override
        next_reason = ""
        if current_override == "FULL_RISK":
            if recovery:
                next_override = "NONE"
        elif current_override == "OVERLAY":
            if baseline_full:
                next_override = "OVERLAY"
            elif recovery:
                next_override = "NONE"
        else:
            if commodity_mode == "FULL_RISK" and commodity_fire:
                next_override = "FULL_RISK"
                next_reason = commodity_reason
            elif commodity_mode == "OVERLAY" and commodity_fire:
                next_override = "OVERLAY"
                next_reason = commodity_reason
        pending_override = next_override
        pending_reason = next_reason

    out = pd.DataFrame(daily_rows)
    out[f"{strategy_name}_risk_state"] = out["state"]
    out[f"{strategy_name}_weight_SPY"] = out["weight_SPY"]
    out[f"{strategy_name}_weight_GOLD"] = out["weight_GOLD"]
    out[f"{strategy_name}_weight_CMDTY_FUT"] = out["weight_CMDTY_FUT"]
    out[f"{strategy_name}_weight_IEF"] = out["weight_IEF"]
    out[f"{strategy_name}_weight_CASH"] = out["weight_CASH"]
    out[f"{strategy_name}_return"] = out["strategy_return"]
    out[f"{strategy_name}_nav"] = (1 + out["strategy_return"].fillna(0.0)).cumprod()
    out[f"{strategy_name}_turnover"] = out["turnover"]
    out[f"{strategy_name}_transaction_cost"] = out["transaction_cost"]
    out[f"{strategy_name}_overlay_state"] = out["state"].eq("OVERLAY")
    out[f"{strategy_name}_full_risk_state"] = out["state"].eq("FULL_RISK")
    if warnings_out:
        warn_df = pd.DataFrame(warnings_out).drop_duplicates()
        warn_path = CONFIG["output_dir"] / f"{strategy_name}_unexpected_regimes.csv"
        warn_df.to_csv(warn_path, index=False)
        warnings.warn(f"{strategy_name}: unexpected regime rows saved to {warn_path}")
    return out, pd.DataFrame(event_rows)


def extract_risk_episodes(df: pd.DataFrame, strategies: List[str], event_log: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for strategy in strategies:
        state_col = f"{strategy}_risk_state"
        ret_col = f"{strategy}_return"
        if state_col not in df.columns:
            continue
        full = df[state_col].eq("FULL_RISK")
        starts = df.index[full & ~full.shift(1, fill_value=False)]
        for episode_id, start in enumerate(starts, 1):
            end = start
            while end + 1 < len(df) and full.iloc[end + 1]:
                end += 1
            sub = df.iloc[start : end + 1]
            ev = event_log[
                (event_log["strategy"] == strategy)
                & (event_log["event_type"] == "ENTER_RISK")
                & (event_log["event_date"] == sub.iloc[0]["date"])
            ]
            reason = ev["entry_reason"].iloc[0] if not ev.empty else ""
            spy_nav = (1 + sub["spy_daily_return"].fillna(0.0)).cumprod()
            cash_nav = (1 + sub["CASH_return"].fillna(0.0)).cumprod()
            strat_nav = (1 + sub[ret_col].fillna(0.0)).cumprod()
            rows.append(
                {
                    "strategy": strategy,
                    "episode_id": episode_id,
                    "risk_start_date": sub.iloc[0]["date"],
                    "risk_end_date": sub.iloc[-1]["date"],
                    "duration_days": len(sub),
                    "entry_reason": reason,
                    "macro_regime_at_entry": sub.iloc[0]["macro_regime_confirmed"],
                    "dominant_macro_regime": sub["macro_regime_confirmed"].mode().iloc[0],
                    "SPY_drawdown_at_entry": sub.iloc[0]["spy_drawdown_from_previous_high"],
                    "CMDTY_RET60_at_entry": sub.iloc[0]["CMDTY_RET60"],
                    "VIX_ZSCORE_at_entry": sub.iloc[0]["VIX_ZSCORE_120D"],
                    "D_CREDIT_SPREAD_20D_at_entry": sub.iloc[0]["D_CREDIT_SPREAD_20D"],
                    "monthly_either_state_at_entry": sub.iloc[0]["monthly_either_state"],
                    "SPY_return_during_risk": spy_nav.iloc[-1] - 1,
                    "CASH_return_during_risk": cash_nav.iloc[-1] - 1,
                    "strategy_return_during_risk": strat_nav.iloc[-1] - 1,
                    "SPY_max_drawdown_during_risk": (spy_nav / spy_nav.cummax() - 1).min(),
                    "SPY_max_runup_during_risk": (spy_nav / spy_nav.cummin() - 1).max(),
                    "exited_by_R3_date": sub.iloc[-1]["date"],
                }
            )
    return pd.DataFrame(rows)


def compute_performance_metrics(df: pd.DataFrame, strategies: List[str], episodes: pd.DataFrame) -> pd.DataFrame:
    years = len(df) / CONFIG["trading_days_per_year"]
    rows: List[Dict[str, object]] = []
    for strategy in strategies:
        ret_col = f"{strategy}_return"
        nav_col = f"{strategy}_nav"
        if ret_col not in df.columns:
            continue
        ret = df[ret_col].fillna(0.0)
        nav = df[nav_col]
        excess = ret - df["daily_rf"].fillna(0.0)
        downside = excess.clip(upper=0)
        ep = episodes[episodes["strategy"] == strategy]
        turnover_col = f"{strategy}_turnover"
        if turnover_col in df.columns:
            turnover_ser = pd.to_numeric(df[turnover_col], errors="coerce").fillna(0.0)
        else:
            weight_cols = [f"{strategy}_weight_{asset}" for asset in ASSETS if f"{strategy}_weight_{asset}" in df.columns]
            if weight_cols:
                turnover_ser = df[weight_cols].fillna(0.0).diff().abs().sum(axis=1).fillna(0.0)
            else:
                turnover_ser = pd.Series(0.0, index=df.index)
        tc_col = f"{strategy}_transaction_cost"
        tc_ser = pd.to_numeric(df[tc_col], errors="coerce").fillna(0.0) if tc_col in df.columns else 0.5 * turnover_ser * CONFIG["one_way_cost_bps"] / 10000.0
        full_col = f"{strategy}_full_risk_state"
        overlay_col = f"{strategy}_overlay_state"
        full_ser = df[full_col] if full_col in df.columns else pd.Series(False, index=df.index)
        overlay_ser = df[overlay_col] if overlay_col in df.columns else pd.Series(False, index=df.index)
        state_col = f"{strategy}_risk_state"
        state_ser = df[state_col] if state_col in df.columns else pd.Series("NON_RISK", index=df.index)
        rows.append(
            {
                "strategy": strategy,
                "start_date": df["date"].iloc[0],
                "end_date": df["date"].iloc[-1],
                "annualized_return": nav.iloc[-1] ** (1 / years) - 1,
                "annualized_volatility": ret.std(ddof=0) * math.sqrt(CONFIG["trading_days_per_year"]),
                "sharpe_ratio": excess.mean() / excess.std(ddof=0) * math.sqrt(CONFIG["trading_days_per_year"]) if excess.std(ddof=0) > 0 else np.nan,
                "sortino_ratio": excess.mean() / downside.std(ddof=0) * math.sqrt(CONFIG["trading_days_per_year"]) if downside.std(ddof=0) > 0 else np.nan,
                "max_drawdown": (nav / nav.cummax() - 1).min(),
                "calmar_ratio": (nav.iloc[-1] ** (1 / years) - 1) / abs((nav / nav.cummax() - 1).min()) if (nav / nav.cummax() - 1).min() < 0 else np.nan,
                "final_nav": nav.iloc[-1],
                "number_of_switches": int((turnover_ser > 0).sum()),
                "number_of_full_risk_entries": int((state_ser.eq("FULL_RISK") & ~state_ser.shift(1, fill_value="NON_RISK").eq("FULL_RISK")).sum()),
                "number_of_overlay_entries": int((state_ser.eq("OVERLAY") & ~state_ser.shift(1, fill_value="NON_RISK").eq("OVERLAY")).sum()),
                "avg_risk_episode_duration": ep["duration_days"].mean() if not ep.empty else np.nan,
                "median_risk_episode_duration": ep["duration_days"].median() if not ep.empty else np.nan,
                "time_in_full_risk": full_ser.mean(),
                "time_in_overlay": overlay_ser.mean(),
                "total_turnover": turnover_ser.sum(),
                "transaction_cost_drag": tc_ser.sum(),
            }
        )
    return pd.DataFrame(rows)


def compute_crisis_performance(df: pd.DataFrame, strategies: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    max_date = df["date"].max()
    for period, (start, end) in CRISIS_WINDOWS.items():
        sub = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= min(pd.Timestamp(end), max_date))]
        if len(sub) < 2:
            continue
        for strategy in strategies:
            ret_col = f"{strategy}_return"
            if ret_col not in sub.columns:
                continue
            ret = sub[ret_col].fillna(0.0)
            nav = (1 + ret).cumprod()
            excess = ret - sub["daily_rf"].fillna(0.0)
            turnover_col = f"{strategy}_turnover"
            turnover_ser = pd.to_numeric(sub[turnover_col], errors="coerce").fillna(0.0) if turnover_col in sub.columns else pd.Series(0.0, index=sub.index)
            full_col = f"{strategy}_full_risk_state"
            overlay_col = f"{strategy}_overlay_state"
            full_ser = sub[full_col] if full_col in sub.columns else pd.Series(False, index=sub.index)
            overlay_ser = sub[overlay_col] if overlay_col in sub.columns else pd.Series(False, index=sub.index)
            rows.append(
                {
                    "period": period,
                    "strategy": strategy,
                    "cumulative_return": nav.iloc[-1] - 1,
                    "annualized_return": nav.iloc[-1] ** (CONFIG["trading_days_per_year"] / len(sub)) - 1,
                    "max_drawdown": (nav / nav.cummax() - 1).min(),
                    "volatility": ret.std(ddof=0) * math.sqrt(CONFIG["trading_days_per_year"]),
                    "Sharpe": excess.mean() / excess.std(ddof=0) * math.sqrt(CONFIG["trading_days_per_year"]) if excess.std(ddof=0) > 0 else np.nan,
                    "time_in_full_risk": full_ser.mean(),
                    "time_in_overlay": overlay_ser.mean(),
                    "number_of_switches": int((turnover_ser > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def compute_performance_by_regime(df: pd.DataFrame, strategies: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for regime, sub in df.groupby("macro_regime_confirmed"):
        for strategy in strategies:
            ret_col = f"{strategy}_return"
            if ret_col not in sub.columns:
                continue
            ret = sub[ret_col].fillna(0.0)
            nav = (1 + ret).cumprod()
            excess = ret - sub["daily_rf"].fillna(0.0)
            turnover_col = f"{strategy}_turnover"
            turnover_ser = pd.to_numeric(sub[turnover_col], errors="coerce").fillna(0.0) if turnover_col in sub.columns else pd.Series(0.0, index=sub.index)
            full_col = f"{strategy}_full_risk_state"
            overlay_col = f"{strategy}_overlay_state"
            full_ser = sub[full_col] if full_col in sub.columns else pd.Series(False, index=sub.index)
            overlay_ser = sub[overlay_col] if overlay_col in sub.columns else pd.Series(False, index=sub.index)
            rows.append(
                {
                    "macro_regime_confirmed": regime,
                    "strategy": strategy,
                    "n_obs": len(sub),
                    "annualized_return": nav.iloc[-1] ** (CONFIG["trading_days_per_year"] / len(sub)) - 1,
                    "volatility": ret.std(ddof=0) * math.sqrt(CONFIG["trading_days_per_year"]),
                    "Sharpe": excess.mean() / excess.std(ddof=0) * math.sqrt(CONFIG["trading_days_per_year"]) if excess.std(ddof=0) > 0 else np.nan,
                    "max_drawdown": (nav / nav.cummax() - 1).min(),
                    "time_in_full_risk": full_ser.mean(),
                    "time_in_overlay": overlay_ser.mean(),
                    "number_of_switches": int((turnover_ser > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def analyze_commodity_incremental_value(
    df: pd.DataFrame,
    episodes: pd.DataFrame,
    strategies: List[str],
    mature_strategy: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    mature_full = df[f"{mature_strategy}_full_risk_state"] | df[f"{mature_strategy}_overlay_state"]
    case_start = pd.Timestamp(CONFIG["case_2015_start"])
    case_end = pd.Timestamp(CONFIG["case_2015_end"])
    case_peak = pd.Timestamp(CONFIG["case_2015_peak"])
    case_df = df[(df["date"] >= case_start) & (df["date"] <= case_end)].copy()
    mature_case_nav = (1 + case_df[f"{mature_strategy}_return"].fillna(0.0)).cumprod()
    mature_case_peak = case_df[case_df["date"] >= case_peak].copy()
    mature_peak_nav = (1 + mature_case_peak[f"{mature_strategy}_return"].fillna(0.0)).cumprod() if not mature_case_peak.empty else pd.Series(dtype=float)
    for strategy in strategies:
        if strategy == mature_strategy:
            continue
        active = df[f"{strategy}_full_risk_state"] | df[f"{strategy}_overlay_state"]
        commodity_only = active & ~mature_full
        sub = df[commodity_only].copy()
        commodity_only_episodes = episodes[(episodes["strategy"] == strategy) & (~episodes["risk_start_date"].isin(episodes[episodes["strategy"] == mature_strategy]["risk_start_date"]))]
        case_first = case_df.loc[active.loc[case_df.index]].head(1)
        if not case_first.empty:
            entry_date = case_first["date"].iloc[0]
            days_after_peak = (entry_date - case_peak).days
            days_before_trough = (pd.Timestamp(CONFIG["case_2015_trough"]) - entry_date).days
        else:
            entry_date = pd.NaT
            days_after_peak = np.nan
            days_before_trough = np.nan
        case_nav = (1 + case_df[f"{strategy}_return"].fillna(0.0)).cumprod()
        peak_nav = (1 + mature_case_peak[f"{strategy}_return"].fillna(0.0)).cumprod() if not mature_case_peak.empty else pd.Series(dtype=float)
        overlap_days = int((active & mature_full).sum())
        active_days = int(active.sum())
        lead_deltas = []
        for idx in df.index[df[f"{strategy}_full_risk_state"] & ~df[f"{strategy}_full_risk_state"].shift(1, fill_value=False)]:
            future = df.index[df[f"{mature_strategy}_full_risk_state"] & ~df[f"{mature_strategy}_full_risk_state"].shift(1, fill_value=False)]
            later = future[future >= idx]
            if len(later):
                lead_deltas.append(int(later[0] - idx))
        rows.append(
            {
                "strategy": strategy,
                "added_trigger_name": strategy.replace("MATURE_", ""),
                "added_trigger_event_count": int((df[f"{strategy}_full_risk_state"] & ~df[f"{strategy}_full_risk_state"].shift(1, fill_value=False)).sum() + (df[f"{strategy}_overlay_state"] & ~df[f"{strategy}_overlay_state"].shift(1, fill_value=False)).sum()),
                "commodity_only_risk_episode_count": len(commodity_only_episodes),
                "commodity_only_avg_SPY_return_during_risk": commodity_only_episodes["SPY_return_during_risk"].mean() if not commodity_only_episodes.empty else np.nan,
                "commodity_only_avg_SPY_max_drawdown_during_risk": commodity_only_episodes["SPY_max_drawdown_during_risk"].mean() if not commodity_only_episodes.empty else np.nan,
                "commodity_only_avg_strategy_return_during_risk": commodity_only_episodes["strategy_return_during_risk"].mean() if not commodity_only_episodes.empty else np.nan,
                "avg_days_commodity_entry_before_baseline_entry": np.mean(lead_deltas) if lead_deltas else np.nan,
                "median_days_commodity_entry_before_baseline_entry": np.median(lead_deltas) if lead_deltas else np.nan,
                "overlap_with_baseline_risk_days": overlap_days,
                "overlap_ratio_with_baseline_risk": overlap_days / active_days if active_days else np.nan,
                "baseline_missed_episodes_captured_count": len(commodity_only_episodes),
                "2015_2016_entry_date": entry_date,
                "2015_2016_entry_days_after_peak": days_after_peak,
                "2015_2016_entry_days_before_trough": days_before_trough,
                "2015_2016_improvement_vs_baseline_return": (case_nav.iloc[-1] - 1) - (mature_case_nav.iloc[-1] - 1),
                "2015_2016_improvement_vs_baseline_maxdd": (case_nav / case_nav.cummax() - 1).min() - (mature_case_nav / mature_case_nav.cummax() - 1).min(),
            }
        )
    return pd.DataFrame(rows)


def analyze_2015_case(df: pd.DataFrame, strategies: List[str]) -> pd.DataFrame:
    case_start = pd.Timestamp(CONFIG["case_2015_start"])
    case_end = pd.Timestamp(CONFIG["case_2015_end"])
    peak = pd.Timestamp(CONFIG["case_2015_peak"])
    trough = pd.Timestamp(CONFIG["case_2015_trough"])
    case = df[(df["date"] >= case_start) & (df["date"] <= case_end)].copy()
    peak_to_trough = case[(case["date"] >= peak) & (case["date"] <= trough)].copy()
    rows = []
    for strategy in strategies:
        active = case[f"{strategy}_full_risk_state"] | case[f"{strategy}_overlay_state"]
        first = case.loc[active].head(1)
        entry = first["date"].iloc[0] if not first.empty else pd.NaT
        case_nav = (1 + case[f"{strategy}_return"].fillna(0.0)).cumprod()
        ptt_nav = (1 + peak_to_trough[f"{strategy}_return"].fillna(0.0)).cumprod()
        rows.append(
            {
                "strategy": strategy,
                "first_entry_date_in_case": entry,
                "entry_reason": "" if first.empty else ("OVERLAY" if first[f"{strategy}_overlay_state"].iloc[0] else "FULL_RISK"),
                "days_after_peak": (entry - peak).days if pd.notna(entry) else np.nan,
                "days_before_trough": (trough - entry).days if pd.notna(entry) else np.nan,
                "SPY_DD_at_entry": first["spy_drawdown_from_previous_high"].iloc[0] if not first.empty else np.nan,
                "CMDTY_RET60_at_entry": first["CMDTY_RET60"].iloc[0] if not first.empty else np.nan,
                "VIX_Z_at_entry": first["VIX_ZSCORE_120D"].iloc[0] if not first.empty else np.nan,
                "credit_chg20_at_entry": first["D_CREDIT_SPREAD_20D"].iloc[0] if not first.empty else np.nan,
                "cumulative_return_full_case": case_nav.iloc[-1] - 1,
                "max_drawdown_full_case": (case_nav / case_nav.cummax() - 1).min(),
                "cumulative_return_peak_to_trough": ptt_nav.iloc[-1] - 1,
                "max_drawdown_peak_to_trough": (ptt_nav / ptt_nav.cummax() - 1).min(),
                "time_in_cash_full_case": case[f"{strategy}_weight_CASH"].mean(),
            }
        )
    return pd.DataFrame(rows)


def plot_results(df: pd.DataFrame, perf: pd.DataFrame, plot_strategies: List[str]) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for strategy in plot_strategies:
        ax.plot(df["date"], df[f"{strategy}_nav"], label=strategy)
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "equity_curve_log.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for strategy in plot_strategies:
        nav = df[f"{strategy}_nav"]
        ax.plot(df["date"], nav / nav.cummax() - 1, label=strategy)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "drawdown_comparison.png", dpi=150)
    plt.close(fig)

    show = perf[perf["strategy"].isin(plot_strategies)].copy()
    metrics = ["annualized_return", "sharpe_ratio", "max_drawdown", "calmar_ratio", "final_nav", "number_of_switches", "time_in_full_risk"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(22, 4))
    for ax, metric in zip(axes, metrics):
        ax.bar(show["strategy"], show[metric])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=90, labelsize=7)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "performance_bar_charts.png", dpi=150)
    plt.close(fig)


def plot_case_studies(df: pd.DataFrame, strategies: List[str]) -> None:
    case_specs = {
        "case_2015_2016_strategy_comparison.png": (CONFIG["case_2015_start"], CONFIG["case_2015_end"]),
        "case_2020_COVID_comparison.png": ("2020-02-01", "2020-06-30"),
        "case_2022_comparison.png": ("2021-11-01", "2023-03-31"),
        "case_2025_comparison.png": ("2025-01-01", str(df["date"].max().date())),
    }
    for fname, (start, end) in case_specs.items():
        sub = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))].copy()
        if sub.empty:
            continue
        fig, axes = plt.subplots(7, 1, figsize=(12, 11), sharex=True)
        axes[0].plot(sub["date"], sub["spy_price"], label="SPY")
        axes[0].legend(fontsize=7)
        axes[1].plot(sub["date"], sub["spy_drawdown_from_previous_high"], label="SPY DD")
        axes[2].plot(sub["date"], sub["CMDTY_RET60"], label="CMDTY_RET60")
        axes[3].plot(sub["date"], sub["D_CREDIT_SPREAD_20D"], label="Credit chg20")
        axes[4].plot(sub["date"], sub["VIX_ZSCORE_120D"], label="VIX z")
        regimes = pd.Categorical(sub["macro_regime_confirmed"], categories=["FLAT", "STEEP", "INVERTED"])
        axes[5].imshow([regimes.codes], aspect="auto", extent=[0, len(sub), 0, 1], cmap="tab10")
        axes[5].set_yticks([])
        selected = [s for s in strategies if "MATURE_" in s or s == "MATURE_BASELINE_REGIME_HEDGE_INV_VOL"][:5]
        for strategy in selected:
            axes[6].plot(sub["date"], sub[f"{strategy}_nav"] / sub[f"{strategy}_nav"].iloc[0], label=strategy)
        axes[6].legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(CONFIG["figure_dir"] / fname, dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    sub = df.copy()
    axes[0].plot(sub["date"], sub["spy_drawdown_from_previous_high"], label="SPY DD")
    axes[1].plot(sub["date"], sub["CMDTY_RET60"], label="CMDTY_RET60")
    for col in [
        "STEEP_CMDTY_RET60_NEG10",
        "STEEP_CMDTY_RET60_NEG10_AND_CREDIT_WIDEN",
        "STEEP_SPY_DD5_AND_CMDTY_RET60_NEG10",
        "BACKBONE_V2_BASELINE_ENTRY",
    ]:
        axes[2].plot(sub["date"], sub[col].astype(int), label=col)
    axes[2].legend(fontsize=6, ncol=2)
    axes[3].plot(sub["date"], sub["spy_price"], label="SPY")
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "commodity_trigger_incremental_timeline.png", dpi=150)
    plt.close(fig)


def write_markdown_report(
    perf: pd.DataFrame,
    crisis: pd.DataFrame,
    incremental: pd.DataFrame,
    case_2015: pd.DataFrame,
    output_path: Path,
) -> None:
    def fmt(df: pd.DataFrame, cols: List[str]) -> str:
        if df.empty:
            return "_No rows_"
        return df[cols].to_markdown(index=False)

    report = f"""# STEEP Commodity Stress Backtest Report

## Purpose
Test STEEP-only commodity slow-growth stress triggers on top of the mature baseline `REGIME_HEDGE_INV_VOL`, with both full-risk and partial-overlay usage.

## Mature Baseline
- FLAT_NON_RISK: 120d inverse-vol on SPY / GOLD / CMDTY_FUT
- FLAT_RISK: 100% GOLD
- STEEP_NON_RISK: 100% SPY
- STEEP_RISK: 80% IEF / 20% GOLD
- INVERTED: 120d inverse-vol on SPY / GOLD
- No fallback allocation is used.

## Main Performance
{fmt(perf.sort_values("sharpe_ratio", ascending=False), ["strategy", "annualized_return", "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio", "final_nav", "time_in_full_risk", "time_in_overlay"])}

## 2015-2016 Case
{fmt(case_2015.sort_values("days_after_peak"), ["strategy", "first_entry_date_in_case", "days_after_peak", "days_before_trough", "cumulative_return_full_case", "max_drawdown_full_case", "cumulative_return_peak_to_trough", "max_drawdown_peak_to_trough"])}

## Incremental Value
{fmt(incremental, ["strategy", "commodity_only_risk_episode_count", "avg_days_commodity_entry_before_baseline_entry", "2015_2016_entry_days_after_peak", "2015_2016_improvement_vs_baseline_return", "2015_2016_improvement_vs_baseline_maxdd"])}

## Crisis Windows
{fmt(crisis[crisis["period"].isin(["2008_GFC", "2015_2016", "COVID_2020", "2022", "2025_PULLBACK"])].copy(), ["period", "strategy", "cumulative_return", "max_drawdown", "Sharpe", "time_in_full_risk", "time_in_overlay"])}

## Interpretation
- Full-risk commodity triggers are evaluated as direct STEEP_RISK entry.
- Partial-overlay triggers only apply while the mature strategy remains in STEEP_NON_RISK.
- Full risk overrides overlay.
- This test is diagnostic for whether commodity slow-growth signals belong in the mature backbone or are better treated as a softer overlay.
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel, abnormal = load_panel()
    if not abnormal.empty:
        abnormal.to_csv(CONFIG["output_dir"] / "unexpected_regime_dates.csv", index=False)
    df = build_spy_features(panel)
    df = build_commodity_features(df)
    df = build_baseline_v2_signals(df)
    df = build_steep_commodity_triggers(df)

    strategy_panels: List[pd.DataFrame] = []
    event_logs: List[pd.DataFrame] = []

    # Benchmarks from returns or existing panels.
    bench = pd.DataFrame(index=df.index)
    bench["SPY_BUY_HOLD_risk_state"] = "NON_RISK"
    bench["SPY_BUY_HOLD_weight_SPY"] = 1.0
    bench["SPY_BUY_HOLD_weight_GOLD"] = 0.0
    bench["SPY_BUY_HOLD_weight_CMDTY_FUT"] = 0.0
    bench["SPY_BUY_HOLD_weight_IEF"] = 0.0
    bench["SPY_BUY_HOLD_weight_CASH"] = 0.0
    bench["SPY_BUY_HOLD_return"] = df["spy_daily_return"].fillna(0.0)
    bench["SPY_BUY_HOLD_nav"] = (1 + bench["SPY_BUY_HOLD_return"]).cumprod()
    bench["SPY_BUY_HOLD_turnover"] = 0.0
    bench["SPY_BUY_HOLD_transaction_cost"] = 0.0
    bench["SPY_BUY_HOLD_overlay_state"] = False
    bench["SPY_BUY_HOLD_full_risk_state"] = False

    if "BACKBONE_V2_SPY_CASH_return" in df.columns:
        bench["BACKBONE_V2_SPY_CASH_risk_state"] = np.where(pd.to_numeric(df["BACKBONE_V2_SPY_CASH_weight_SPY"], errors="coerce").fillna(1.0) < 0.5, "FULL_RISK", "NON_RISK")
        bench["BACKBONE_V2_SPY_CASH_weight_SPY"] = pd.to_numeric(df["BACKBONE_V2_SPY_CASH_weight_SPY"], errors="coerce").fillna(1.0)
        bench["BACKBONE_V2_SPY_CASH_weight_GOLD"] = 0.0
        bench["BACKBONE_V2_SPY_CASH_weight_CMDTY_FUT"] = 0.0
        bench["BACKBONE_V2_SPY_CASH_weight_IEF"] = 0.0
        bench["BACKBONE_V2_SPY_CASH_weight_CASH"] = pd.to_numeric(df["BACKBONE_V2_SPY_CASH_weight_CASH"], errors="coerce").fillna(1 - bench["BACKBONE_V2_SPY_CASH_weight_SPY"])
        bench["BACKBONE_V2_SPY_CASH_return"] = pd.to_numeric(df["BACKBONE_V2_SPY_CASH_return"], errors="coerce").fillna(0.0)
        bench["BACKBONE_V2_SPY_CASH_nav"] = pd.to_numeric(df["BACKBONE_V2_SPY_CASH_nav"], errors="coerce").fillna((1 + bench["BACKBONE_V2_SPY_CASH_return"]).cumprod())
    else:
        raise ValueError("Missing BACKBONE_V2_SPY_CASH benchmark columns.")
    bench["BACKBONE_V2_SPY_CASH_turnover"] = (bench["BACKBONE_V2_SPY_CASH_weight_SPY"].diff().abs().fillna(0) * 2)
    bench["BACKBONE_V2_SPY_CASH_transaction_cost"] = 0.5 * bench["BACKBONE_V2_SPY_CASH_turnover"] * CONFIG["one_way_cost_bps"] / 10000
    bench["BACKBONE_V2_SPY_CASH_overlay_state"] = False
    bench["BACKBONE_V2_SPY_CASH_full_risk_state"] = bench["BACKBONE_V2_SPY_CASH_risk_state"].eq("FULL_RISK")

    if "REGIME_HEDGE_V1_ORIGINAL_return" in df.columns:
        for suffix in ["risk_state", "weight_SPY", "weight_GOLD", "weight_CMDTY_FUT", "weight_IEF", "weight_CASH", "return", "nav", "turnover", "transaction_cost", "overlay_state", "full_risk_state"]:
            src_col = f"REGIME_HEDGE_V1_ORIGINAL_{suffix}"
            if src_col in df.columns:
                bench[src_col] = df[src_col]
        if "REGIME_HEDGE_V1_ORIGINAL_risk_state" not in bench.columns:
            bench["REGIME_HEDGE_V1_ORIGINAL_risk_state"] = "NON_RISK"
        if "REGIME_HEDGE_V1_ORIGINAL_overlay_state" not in bench.columns:
            bench["REGIME_HEDGE_V1_ORIGINAL_overlay_state"] = False
        if "REGIME_HEDGE_V1_ORIGINAL_full_risk_state" not in bench.columns:
            bench["REGIME_HEDGE_V1_ORIGINAL_full_risk_state"] = False
    else:
        raise ValueError("Missing REGIME_HEDGE_V1_ORIGINAL benchmark columns.")
    strategy_panels.append(bench)

    baseline_panel, baseline_events = run_state_machine_backtest(df, "MATURE_BASELINE_REGIME_HEDGE_INV_VOL", "NONE", None, None)
    strategy_panels.append(baseline_panel)
    event_logs.append(baseline_events)

    full_risk_variants = [
        ("MATURE_FULL_ONE_RET60", "ONE_RET60"),
        ("MATURE_FULL_ONE_CREDIT", "ONE_CREDIT"),
        ("MATURE_FULL_ONE_DD5", "ONE_DD5"),
        ("MATURE_FULL_ALL_THREE", "ALL"),
    ]
    for name, signal_set in full_risk_variants:
        pnl, ev = run_state_machine_backtest(df, name, "FULL_RISK", signal_set, None)
        strategy_panels.append(pnl)
        event_logs.append(ev)

    overlay_variants = [
        ("MATURE_OVERLAY20_ONE_RET60", "ONE_RET60", "Overlay_20"),
        ("MATURE_OVERLAY30_ONE_RET60", "ONE_RET60", "Overlay_30"),
        ("MATURE_OVERLAY40_ONE_RET60", "ONE_RET60", "Overlay_40"),
        ("MATURE_OVERLAY20_ONE_CREDIT", "ONE_CREDIT", "Overlay_20"),
        ("MATURE_OVERLAY30_ONE_CREDIT", "ONE_CREDIT", "Overlay_30"),
        ("MATURE_OVERLAY40_ONE_CREDIT", "ONE_CREDIT", "Overlay_40"),
        ("MATURE_OVERLAY20_ONE_DD5", "ONE_DD5", "Overlay_20"),
        ("MATURE_OVERLAY30_ONE_DD5", "ONE_DD5", "Overlay_30"),
        ("MATURE_OVERLAY40_ONE_DD5", "ONE_DD5", "Overlay_40"),
        ("MATURE_OVERLAY20_ALL_THREE", "ALL", "Overlay_20"),
        ("MATURE_OVERLAY30_ALL_THREE", "ALL", "Overlay_30"),
        ("MATURE_OVERLAY40_ALL_THREE", "ALL", "Overlay_40"),
    ]
    for name, signal_set, overlay_name in overlay_variants:
        pnl, ev = run_state_machine_backtest(df, name, "OVERLAY", signal_set, overlay_name)
        strategy_panels.append(pnl)
        event_logs.append(ev)

    result = df.copy()
    for panel_piece in strategy_panels:
        result = pd.concat([result, panel_piece], axis=1)
    result = result.loc[:, ~result.columns.duplicated(keep="first")]

    event_log = pd.concat(event_logs, ignore_index=True) if event_logs else pd.DataFrame()
    strategies = [
        "SPY_BUY_HOLD",
        "BACKBONE_V2_SPY_CASH",
        "REGIME_HEDGE_V1_ORIGINAL",
        "MATURE_BASELINE_REGIME_HEDGE_INV_VOL",
        "MATURE_FULL_ONE_RET60",
        "MATURE_FULL_ONE_CREDIT",
        "MATURE_FULL_ONE_DD5",
        "MATURE_FULL_ALL_THREE",
        "MATURE_OVERLAY20_ONE_RET60",
        "MATURE_OVERLAY30_ONE_RET60",
        "MATURE_OVERLAY40_ONE_RET60",
        "MATURE_OVERLAY20_ONE_CREDIT",
        "MATURE_OVERLAY30_ONE_CREDIT",
        "MATURE_OVERLAY40_ONE_CREDIT",
        "MATURE_OVERLAY20_ONE_DD5",
        "MATURE_OVERLAY30_ONE_DD5",
        "MATURE_OVERLAY40_ONE_DD5",
        "MATURE_OVERLAY20_ALL_THREE",
        "MATURE_OVERLAY30_ALL_THREE",
        "MATURE_OVERLAY40_ALL_THREE",
    ]

    episodes = extract_risk_episodes(result, strategies, event_log)
    perf = compute_performance_metrics(result, strategies, episodes)
    crisis = compute_crisis_performance(result, strategies)
    by_regime = compute_performance_by_regime(result, strategies)
    incremental = analyze_commodity_incremental_value(result, episodes, strategies, "MATURE_BASELINE_REGIME_HEDGE_INV_VOL")
    case_2015 = analyze_2015_case(result, strategies)

    result.to_csv(CONFIG["output_dir"] / "daily_backtest_panel.csv", index=False)
    perf.to_csv(CONFIG["output_dir"] / "performance_summary.csv", index=False)
    event_log.to_csv(CONFIG["output_dir"] / "risk_state_event_log.csv", index=False)
    episodes.to_csv(CONFIG["output_dir"] / "risk_episodes.csv", index=False)
    incremental.to_csv(CONFIG["output_dir"] / "commodity_trigger_incremental_value.csv", index=False)
    crisis.to_csv(CONFIG["output_dir"] / "crisis_performance.csv", index=False)
    by_regime.to_csv(CONFIG["output_dir"] / "performance_by_regime.csv", index=False)
    case_2015.to_csv(CONFIG["output_dir"] / "case_2015_2016_entry_comparison.csv", index=False)
    if not abnormal.empty:
        abnormal.to_csv(CONFIG["output_dir"] / "unexpected_regime_dates.csv", index=False)

    plot_list = [
        "SPY_BUY_HOLD",
        "BACKBONE_V2_SPY_CASH",
        "REGIME_HEDGE_V1_ORIGINAL",
        "MATURE_BASELINE_REGIME_HEDGE_INV_VOL",
        "MATURE_FULL_ONE_RET60",
        "MATURE_FULL_ONE_CREDIT",
        "MATURE_FULL_ONE_DD5",
        "MATURE_FULL_ALL_THREE",
    ]
    plot_results(result, perf, plot_list)
    plot_case_studies(
        result,
        [
            "MATURE_BASELINE_REGIME_HEDGE_INV_VOL",
            "MATURE_FULL_ONE_RET60",
            "MATURE_FULL_ONE_CREDIT",
            "MATURE_FULL_ONE_DD5",
            "MATURE_OVERLAY20_ALL_THREE",
        ],
    )
    write_markdown_report(perf, crisis, incremental, case_2015, CONFIG["output_dir"] / "STEEP_COMMODITY_STRESS_BACKTEST_REPORT.md")

    def _pick(name: str) -> pd.Series:
        return perf.loc[perf["strategy"] == name].iloc[0]

    for name in [
        "MATURE_BASELINE_REGIME_HEDGE_INV_VOL",
        "MATURE_FULL_ONE_RET60",
        "MATURE_FULL_ONE_CREDIT",
        "MATURE_FULL_ONE_DD5",
        "MATURE_FULL_ALL_THREE",
    ]:
        row = _pick(name)
        print(
            f"{name}: AnnRet {row['annualized_return']:.2%} / Sharpe {row['sharpe_ratio']:.3f} / "
            f"MaxDD {row['max_drawdown']:.2%} / Final NAV {row['final_nav']:.2f} / "
            f"full risk {row['time_in_full_risk']:.1%} / overlay {row['time_in_overlay']:.1%}"
        )
    case_rank = case_2015.sort_values("cumulative_return_full_case", ascending=False)[["strategy", "cumulative_return_full_case", "max_drawdown_full_case"]]
    print("2015-2016 best full-case:", case_rank.iloc[0].to_dict())
    covid = crisis[crisis["period"] == "COVID_2020"][["strategy", "cumulative_return"]].sort_values("cumulative_return")
    rates = crisis[crisis["period"] == "2022"][["strategy", "cumulative_return"]].sort_values("cumulative_return")
    print("COVID weakest three:", covid.head(3).to_dict("records"))
    print("2022 weakest three:", rates.head(3).to_dict("records"))
    if not incremental.empty:
        print("commodity-only avg SPY max drawdown:")
        print(incremental[["strategy", "commodity_only_avg_SPY_max_drawdown_during_risk"]].to_string(index=False))
    overlay_best = perf[perf["strategy"].str.contains("OVERLAY")].sort_values("sharpe_ratio", ascending=False).head(1)
    print("Recommended next step: partial overlay test" if not overlay_best.empty else "Recommended next step: keep mature baseline")
    print(f"Output path: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
