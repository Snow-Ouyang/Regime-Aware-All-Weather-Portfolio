"""Regime-aware hedge allocation v1 backtest.

This module converts the current cross-state hedge diagnostic conclusions into
an explicit rule-based multi-asset strategy using SPY, GOLD, IEF, and CASH.
No optimization is used.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIG = {
    "output_dir": Path("results/regime_aware_hedge_allocation_v1"),
    "figure_dir": Path("figures/regime_aware_hedge_allocation_v1"),
    "one_way_cost_bps": 5,
    "monthly_rebalance": True,
    "timing_backbone": "BACKBONE_V2_UPGRADED",
    "allocation_rules": {
        "INVERTED": {"SPY": 0.70, "GOLD": 0.20, "IEF": 0.00, "CASH": 0.10},
        "FLAT_NON_RISK": {"SPY": 0.60, "GOLD": 0.40, "IEF": 0.00, "CASH": 0.00},
        "FLAT_RISK": {"SPY": 0.00, "GOLD": 1.00, "IEF": 0.00, "CASH": 0.00},
        "STEEP_NON_RISK": {"SPY": 0.90, "GOLD": 0.00, "IEF": 0.10, "CASH": 0.00},
        "STEEP_RISK": {"SPY": 0.00, "GOLD": 0.00, "IEF": 1.00, "CASH": 0.00},
        "FALLBACK_NON_RISK": {"SPY": 0.80, "GOLD": 0.00, "IEF": 0.00, "CASH": 0.20},
        "FALLBACK_RISK": {"SPY": 0.00, "GOLD": 0.00, "IEF": 0.00, "CASH": 1.00},
    },
}

PANEL_CANDIDATES = [
    Path("results/spy_cash_backbone_upgrade_ablation/daily_backtest_panel.csv"),
    Path("results/flat_vix_credit_trigger_diagnostic/full_backtest_panel.csv"),
    Path("results/spy_cash_stress_recovery_with_credit/daily_backtest_panel.csv"),
]

EVENT_LOG_CANDIDATES = [
    Path("results/spy_cash_backbone_upgrade_ablation/risk_state_event_log.csv"),
    Path("results/spy_cash_stress_recovery_with_credit/risk_state_event_log.csv"),
]

ASSET_PANEL_CANDIDATES = [
    Path("results/regime_hedge_steep_sell_ief/daily_backtest_panel.csv"),
    Path("results/reconstructed_regime_asset_behavior/reconstructed_regime_panel.csv"),
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

ASSETS = ["SPY", "GOLD", "IEF", "CASH"]
STRATEGIES = [
    "SPY_BUY_HOLD",
    "CASH_ONLY",
    "MONTHLY_EITHER_CONFIRM",
    "BACKBONE_V2_SPY_CASH",
    "REGIME_HEDGE_V1",
    "STATIC_60_30_10",
    "STATIC_70_20_10",
    "STATIC_60_20_20",
]


def ensure_dirs() -> None:
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["figure_dir"].mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        for col in ["DATE", "Date", "observation_date"]:
            if col in df.columns:
                df = df.rename(columns={col: "date"})
                break
    if "date" not in df.columns:
        raise ValueError(f"No date column in {path}")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def _merge_missing(base: pd.DataFrame, other: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in base.columns and c in other.columns]
    if not missing:
        return base
    return base.merge(other[["date"] + missing], on="date", how="left")


def _first_existing(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols = list(cols)
    col_set = set(cols)
    for c in candidates:
        if c in col_set:
            return c
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    full = {asset: float(weights.get(asset, 0.0)) for asset in ASSETS}
    total = sum(full.values())
    if total <= 0:
        raise ValueError(f"Invalid zero-sum weights: {weights}")
    return {k: v / total for k, v in full.items()}


def load_base_panel() -> pd.DataFrame:
    frames: List[Tuple[Path, pd.DataFrame]] = []
    for path in PANEL_CANDIDATES:
        if path.exists():
            frames.append((path, _read_csv(path)))
    if not frames:
        raise FileNotFoundError("No base panel found.")
    panel = frames[0][1].copy()
    print(f"Loaded primary panel: {frames[0][0]}")
    needed = [
        "spy_price",
        "spy_daily_return",
        "daily_rf",
        "macro_regime_confirmed",
        "monthly_either_state",
        "VIX_LEVEL",
        "VIX_ZSCORE_120D",
        "CREDIT_SPREAD_BAA_AAA",
        "D_CREDIT_SPREAD_20D",
        "spy_drawdown_from_previous_high",
        "SPY_MA20",
        "SPY_CROSS_ABOVE_MA20",
        "MONTHLY_EITHER_CONFIRM_return",
        "MONTHLY_EITHER_CONFIRM_nav",
        "MONTHLY_EITHER_CONFIRM_weight_spy",
        "MONTHLY_EITHER_CONFIRM_weight_cash",
        "BACKBONE_V2_UPGRADED_risk_state",
        "BACKBONE_V2_UPGRADED_weight_spy",
        "BACKBONE_V2_UPGRADED_weight_cash",
        "BACKBONE_V2_UPGRADED_return",
        "BACKBONE_V2_UPGRADED_nav",
    ]
    for _, df in frames[1:]:
        panel = _merge_missing(panel, df, needed)

    if "spy_drawdown_from_previous_high" not in panel.columns:
        panel["spy_drawdown_from_previous_high"] = panel["spy_price"] / panel["spy_price"].cummax() - 1
    if "VIX_ZSCORE_120D" not in panel.columns:
        roll = panel["VIX_LEVEL"].rolling(120)
        panel["VIX_ZSCORE_120D"] = (panel["VIX_LEVEL"] - roll.mean()) / roll.std(ddof=0)
    if "D_CREDIT_SPREAD_20D" not in panel.columns:
        panel["D_CREDIT_SPREAD_20D"] = panel["CREDIT_SPREAD_BAA_AAA"] - panel["CREDIT_SPREAD_BAA_AAA"].shift(20)
    if "SPY_MA20" not in panel.columns:
        panel["SPY_MA20"] = panel["spy_price"].rolling(20).mean()
    if "SPY_CROSS_ABOVE_MA20" not in panel.columns:
        panel["SPY_CROSS_ABOVE_MA20"] = (panel["spy_price"] > panel["SPY_MA20"]) & (
            panel["spy_price"].shift(1) <= panel["SPY_MA20"].shift(1)
        )
    panel["spy_daily_return"] = pd.to_numeric(panel["spy_daily_return"], errors="coerce")
    panel["daily_rf"] = pd.to_numeric(panel["daily_rf"], errors="coerce").fillna(0.0)
    panel["macro_regime_confirmed"] = panel["macro_regime_confirmed"].fillna("NEUTRAL").astype(str)
    panel["monthly_either_state"] = panel["monthly_either_state"].fillna("UNKNOWN").astype(str)
    return panel


def load_asset_returns(base: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    panel = base.copy()
    panel["SPY_return"] = pd.to_numeric(panel.get("spy_daily_return"), errors="coerce")
    panel["CASH_return"] = pd.to_numeric(panel.get("daily_rf"), errors="coerce").fillna(0.0)
    source_notes: List[str] = ["SPY/CASH from base panel"]

    for path in ASSET_PANEL_CANDIDATES:
        if not path.exists():
            continue
        src = _read_csv(path)
        merge_cols = ["date"]
        rename_map: Dict[str, str] = {}
        ief_col = _first_existing(src.columns, ["IEF_RETURN", "IEF_return", "IEF_ret"])
        gold_col = _first_existing(src.columns, ["GOLD_RETURN", "GOLD_return", "GOLD_ret", "GLD_RETURN", "GLD_return"])
        if ief_col:
            merge_cols.append(ief_col)
            rename_map[ief_col] = "IEF_return"
        if gold_col:
            merge_cols.append(gold_col)
            rename_map[gold_col] = "GOLD_return"
        if len(merge_cols) > 1:
            panel = panel.merge(src[merge_cols].rename(columns=rename_map), on="date", how="left")
            source_notes.append(str(path))
            if "IEF_return" in panel.columns and "GOLD_return" in panel.columns:
                break

    required = ["SPY_return", "GOLD_return", "IEF_return", "CASH_return"]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"Missing required asset returns: {missing}")

    for col in required:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")

    valid_mask = panel[required].notna().all(axis=1)
    if not valid_mask.any():
        raise ValueError("No overlapping sample with SPY/GOLD/IEF/CASH returns.")
    start_idx = valid_mask.idxmax()
    panel = panel.loc[start_idx:].reset_index(drop=True)

    for asset in ASSETS:
        panel[f"{asset}_NAV"] = (1 + panel[f"{asset}_return"].fillna(0.0)).cumprod()

    return panel, "; ".join(source_notes)


def build_backbone_v2_state(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = panel.copy()
    if "BACKBONE_V2_UPGRADED_weight_spy" in df.columns and "BACKBONE_V2_UPGRADED_nav" in df.columns:
        df["timing_state"] = np.where(pd.to_numeric(df["BACKBONE_V2_UPGRADED_weight_spy"], errors="coerce") >= 0.5, "NON_RISK", "RISK")
        df["BACKBONE_V2_SPY_CASH_weight_SPY"] = pd.to_numeric(df["BACKBONE_V2_UPGRADED_weight_spy"], errors="coerce").fillna(1.0)
        if "BACKBONE_V2_UPGRADED_weight_cash" in df.columns:
            df["BACKBONE_V2_SPY_CASH_weight_CASH"] = pd.to_numeric(df["BACKBONE_V2_UPGRADED_weight_cash"], errors="coerce").fillna(
                1 - df["BACKBONE_V2_SPY_CASH_weight_SPY"]
            )
        else:
            df["BACKBONE_V2_SPY_CASH_weight_CASH"] = 1 - df["BACKBONE_V2_SPY_CASH_weight_SPY"]
        df["BACKBONE_V2_SPY_CASH_return"] = pd.to_numeric(df["BACKBONE_V2_UPGRADED_return"], errors="coerce").fillna(0.0)
        df["BACKBONE_V2_SPY_CASH_nav"] = (1 + df["BACKBONE_V2_SPY_CASH_return"]).cumprod()
    else:
        flat_vix = df["macro_regime_confirmed"].eq("FLAT") & (df["VIX_ZSCORE_120D"] >= 3.0)
        flat_credit = df["macro_regime_confirmed"].eq("FLAT") & (
            (df["spy_drawdown_from_previous_high"] <= -0.05) & (df["D_CREDIT_SPREAD_20D"] > 0.10)
        )
        steep_either = df["macro_regime_confirmed"].eq("STEEP") & df["monthly_either_state"].eq("SELL")
        steep_credit = df["macro_regime_confirmed"].eq("STEEP") & (
            (df["spy_drawdown_from_previous_high"] <= -0.05) & (df["D_CREDIT_SPREAD_20D"] > 0.10)
        )
        entry = flat_vix | flat_credit | steep_either | steep_credit
        entry_reason = np.where(
            flat_vix | flat_credit,
            "FLAT_VIX_OR_CREDIT_STRESS",
            np.where(steep_either, "STEEP_EITHER_SELL_STRESS", np.where(steep_credit, "STEEP_CREDIT_DD5_STRESS", "")),
        )
        state = "NON_RISK"
        pending = "NON_RISK"
        pending_reason = ""
        weights = []
        rets = []
        navs = []
        costs = []
        events = []
        nav = 1.0
        prev_w = 1.0
        for i in range(len(df)):
            cost = 0.0
            if pending != state:
                prev_state = state
                state = pending
                new_w = 0.0 if state == "RISK" else 1.0
                turnover = abs(new_w - prev_w) + abs((1 - new_w) - (1 - prev_w))
                cost = 0.5 * turnover * CONFIG["one_way_cost_bps"] / 10000
                prev_w = new_w
                events.append(
                    {
                        "strategy": CONFIG["timing_backbone"],
                        "event_date": df.iloc[i]["date"],
                        "event_type": "ENTER_RISK" if state == "RISK" else "EXIT_RISK",
                        "reason": pending_reason if state == "RISK" else "R3_SPY_CROSS_ABOVE_MA20",
                    }
                )
            w_spy = 0.0 if state == "RISK" else 1.0
            ret = w_spy * df.iloc[i]["SPY_return"] + (1 - w_spy) * df.iloc[i]["CASH_return"] - cost
            nav *= 1 + ret
            weights.append(w_spy)
            rets.append(ret)
            navs.append(nav)
            costs.append(cost)
            next_state = state
            next_reason = ""
            if state == "NON_RISK" and bool(entry.iloc[i]):
                next_state = "RISK"
                next_reason = str(entry_reason[i])
            elif state == "RISK" and bool(df.iloc[i]["SPY_CROSS_ABOVE_MA20"]):
                next_state = "NON_RISK"
            pending = next_state
            pending_reason = next_reason
        df["BACKBONE_V2_SPY_CASH_weight_SPY"] = weights
        df["BACKBONE_V2_SPY_CASH_weight_CASH"] = 1 - df["BACKBONE_V2_SPY_CASH_weight_SPY"]
        df["BACKBONE_V2_SPY_CASH_return"] = rets
        df["BACKBONE_V2_SPY_CASH_nav"] = navs
        df["BACKBONE_V2_SPY_CASH_transaction_cost"] = costs
        df["timing_state"] = np.where(df["BACKBONE_V2_SPY_CASH_weight_SPY"] >= 0.5, "NON_RISK", "RISK")
        event_log = pd.DataFrame(events)
        return df, event_log

    event_log = None
    for path in EVENT_LOG_CANDIDATES:
        if path.exists():
            log = pd.read_csv(path)
            if "event_date" not in log.columns or "strategy" not in log.columns:
                continue
            log["event_date"] = pd.to_datetime(log["event_date"])
            log = log[log["strategy"].astype(str).eq(CONFIG["timing_backbone"])]
            if not log.empty:
                event_log = log.copy()
                break
    if event_log is None:
        event_log = pd.DataFrame(columns=["strategy", "event_date", "event_type", "reason"])

    df["cross_state"] = df["macro_regime_confirmed"] + "_" + df["timing_state"]
    reason_map = (
        event_log[event_log["event_type"].astype(str).eq("ENTER_RISK")]
        .drop_duplicates("event_date")
        .set_index("event_date")["reason"]
        .to_dict()
        if not event_log.empty
        else {}
    )
    df["entry_reason"] = ""
    in_risk = df["timing_state"].eq("RISK")
    starts = df.index[in_risk & ~in_risk.shift(1, fill_value=False)]
    for start in starts:
        end = start
        while end + 1 < len(df) and in_risk.iloc[end + 1]:
            end += 1
        reason = reason_map.get(pd.Timestamp(df.iloc[start]["date"]), "")
        df.loc[start : end, "entry_reason"] = reason
    return df, event_log


def build_allocation_rules(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = panel.copy()
    rule_rows = []
    for regime in ["INVERTED", "FLAT", "STEEP", "NEUTRAL", "HIGH_INFLATION", "OTHER"]:
        for timing_state in ["NON_RISK", "RISK"]:
            if regime == "INVERTED":
                rule_key = "INVERTED"
            elif regime in {"NEUTRAL", "HIGH_INFLATION", "OTHER"}:
                rule_key = "FALLBACK_RISK" if timing_state == "RISK" else "FALLBACK_NON_RISK"
            else:
                rule_key = f"{regime}_{timing_state}"
            weights = _normalize_weights(CONFIG["allocation_rules"].get(rule_key, CONFIG["allocation_rules"]["FALLBACK_NON_RISK"]))
            rule_rows.append(
                {
                    "macro_regime": regime,
                    "timing_state": timing_state,
                    "cross_state": regime if regime == "INVERTED" else f"{regime}_{timing_state}",
                    "target_SPY": weights["SPY"],
                    "target_GOLD": weights["GOLD"],
                    "target_IEF": weights["IEF"],
                    "target_CASH": weights["CASH"],
                }
            )

    fallback_count = 0
    weights_list = []
    for _, row in df.iterrows():
        regime = str(row["macro_regime_confirmed"])
        tstate = str(row["timing_state"])
        if regime == "INVERTED":
            key = "INVERTED"
            cross_state = "INVERTED"
        elif regime == "FLAT":
            key = f"FLAT_{tstate}"
            cross_state = key
        elif regime == "STEEP":
            key = f"STEEP_{tstate}"
            cross_state = key
        else:
            key = "FALLBACK_RISK" if tstate == "RISK" else "FALLBACK_NON_RISK"
            cross_state = f"{regime}_{tstate}"
            fallback_count += 1
        weights = _normalize_weights(CONFIG["allocation_rules"][key])
        weights_list.append(weights)
        df.loc[df.index == row.name, "cross_state"] = cross_state

    wdf = pd.DataFrame(weights_list)
    for asset in ASSETS:
        df[f"REGIME_HEDGE_V1_target_{asset}"] = wdf[asset].values
    df["fallback_flag"] = ~df["macro_regime_confirmed"].isin(["FLAT", "STEEP", "INVERTED"])

    return df, pd.DataFrame(rule_rows)


def _is_first_trading_day_of_month(dates: pd.Series) -> pd.Series:
    period = dates.dt.to_period("M")
    return period.ne(period.shift(1, fill_value=period.iloc[0]))


def _max_drawdown_from_returns(ret: pd.Series) -> float:
    nav = (1 + ret.fillna(0.0)).cumprod()
    if nav.empty:
        return np.nan
    return float((nav / nav.cummax() - 1).min())


def run_multi_asset_backtest(
    panel: pd.DataFrame,
    strategy_name: str,
    target_weights: pd.DataFrame,
    monthly_rebalance: bool,
) -> pd.DataFrame:
    df = panel.copy()
    tw = target_weights[ASSETS].copy()
    tw = tw.div(tw.sum(axis=1), axis=0).fillna(0.0)
    first_month_day = _is_first_trading_day_of_month(df["date"])

    weight_cols = {asset: f"{strategy_name}_weight_{asset}" for asset in ASSETS}
    ret_col = f"{strategy_name}_return"
    nav_col = f"{strategy_name}_nav"
    cost_col = f"transaction_cost_{strategy_name}"
    turn_col = f"turnover_{strategy_name}"

    for col in list(weight_cols.values()) + [ret_col, nav_col, cost_col, turn_col]:
        df[col] = np.nan

    current_weights = tw.iloc[0].to_dict()
    target_prev = tw.iloc[0].to_dict()
    nav = 1.0
    rebalances = 0
    for i in range(len(df)):
        desired = tw.iloc[i].to_dict()
        turnover = 0.0
        cost = 0.0
        target_changed = any(abs(desired[a] - target_prev[a]) > 1e-12 for a in ASSETS)
        rebalance_today = False
        if i > 0 and (target_changed or (monthly_rebalance and bool(first_month_day.iloc[i]))):
            rebalance_today = True
            turnover = float(sum(abs(desired[a] - current_weights[a]) for a in ASSETS))
            cost = 0.5 * turnover * CONFIG["one_way_cost_bps"] / 10000
            current_weights = desired.copy()
            rebalances += 1 if turnover > 0 else 0

        gross = float(sum(current_weights[a] * df.iloc[i][f"{a}_return"] for a in ASSETS))
        net = gross - cost
        nav *= 1 + net
        for asset in ASSETS:
            df.loc[i, weight_cols[asset]] = current_weights[asset]
        df.loc[i, ret_col] = net
        df.loc[i, nav_col] = nav
        df.loc[i, cost_col] = cost
        df.loc[i, turn_col] = turnover

        gross_den = 1 + gross
        if gross_den != 0:
            current_weights = {
                asset: current_weights[asset] * (1 + df.iloc[i][f"{asset}_return"]) / gross_den
                for asset in ASSETS
            }
        target_prev = desired.copy()

    df.attrs[f"{strategy_name}_rebalances"] = rebalances
    return df


def compute_performance_metrics(panel: pd.DataFrame, strategy_name: str) -> Dict[str, object]:
    ret = panel[f"{strategy_name}_return"].fillna(0.0)
    rf = panel["CASH_return"].fillna(0.0)
    nav = (1 + ret).cumprod()
    years = len(ret) / 252
    ann = nav.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = ret.std(ddof=0) * math.sqrt(252)
    excess = ret - rf
    sharpe = excess.mean() / excess.std(ddof=0) * math.sqrt(252) if excess.std(ddof=0) > 0 else np.nan
    maxdd = _max_drawdown_from_returns(ret)
    calmar = ann / abs(maxdd) if pd.notna(maxdd) and maxdd < 0 else np.nan
    avg_w = {asset: panel[f"{strategy_name}_weight_{asset}"].mean() for asset in ASSETS}
    regime_share = panel["macro_regime_confirmed"].value_counts(normalize=True).sort_index()
    return {
        "strategy": strategy_name,
        "start_date": panel["date"].iloc[0],
        "end_date": panel["date"].iloc[-1],
        "annualized_return": ann,
        "annualized_volatility": vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": maxdd,
        "calmar_ratio": calmar,
        "final_nav": nav.iloc[-1],
        "number_of_rebalances": int((panel[f"turnover_{strategy_name}"] > 0).sum()) if f"turnover_{strategy_name}" in panel.columns else 0,
        "total_turnover": panel.get(f"turnover_{strategy_name}", pd.Series(dtype=float)).fillna(0.0).sum() if f"turnover_{strategy_name}" in panel.columns else 0.0,
        "transaction_cost_drag": panel.get(f"transaction_cost_{strategy_name}", pd.Series(dtype=float)).fillna(0.0).sum()
        if f"transaction_cost_{strategy_name}" in panel.columns
        else 0.0,
        "avg_weight_SPY": avg_w["SPY"],
        "avg_weight_GOLD": avg_w["GOLD"],
        "avg_weight_IEF": avg_w["IEF"],
        "avg_weight_CASH": avg_w["CASH"],
        "time_in_risk": panel["timing_state"].eq("RISK").mean(),
        "time_in_each_macro_regime": "; ".join(f"{k}:{v:.2%}" for k, v in regime_share.items()),
    }


def compute_crisis_performance(panel: pd.DataFrame, strategy_names: List[str]) -> pd.DataFrame:
    rows = []
    for period, (start, end) in CRISIS_WINDOWS.items():
        sub = panel[(panel["date"] >= pd.Timestamp(start)) & (panel["date"] <= pd.Timestamp(end))]
        if sub.empty:
            continue
        for strategy in strategy_names:
            ret = sub[f"{strategy}_return"].fillna(0.0)
            rf = sub["CASH_return"].fillna(0.0)
            nav = (1 + ret).cumprod()
            years = len(ret) / 252
            ann = nav.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
            vol = ret.std(ddof=0) * math.sqrt(252)
            sharpe = (ret - rf).mean() / (ret - rf).std(ddof=0) * math.sqrt(252) if (ret - rf).std(ddof=0) > 0 else np.nan
            row = {
                "period": period,
                "strategy": strategy,
                "cumulative_return": nav.iloc[-1] - 1,
                "max_drawdown": _max_drawdown_from_returns(ret),
                "volatility": vol,
                "Sharpe": sharpe,
                "avg_weight_SPY": sub[f"{strategy}_weight_SPY"].mean(),
                "avg_weight_GOLD": sub[f"{strategy}_weight_GOLD"].mean(),
                "avg_weight_IEF": sub[f"{strategy}_weight_IEF"].mean(),
                "avg_weight_CASH": sub[f"{strategy}_weight_CASH"].mean(),
                "turnover": sub.get(f"turnover_{strategy}", pd.Series(dtype=float)).fillna(0.0).sum(),
                "cost_drag": sub.get(f"transaction_cost_{strategy}", pd.Series(dtype=float)).fillna(0.0).sum(),
            }
            if years > 0:
                row["annualized_return"] = ann
            rows.append(row)
    return pd.DataFrame(rows)


def compute_cross_state_performance(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cross_state, sub in panel.groupby("cross_state", dropna=False):
        for name in ["REGIME_HEDGE_V1", "SPY", "GOLD", "IEF", "CASH"]:
            ret_col = f"{name}_return" if name in ["REGIME_HEDGE_V1"] else f"{name}_return"
            ret = sub[ret_col].fillna(0.0)
            rf = sub["CASH_return"].fillna(0.0)
            nav = (1 + ret).cumprod()
            years = len(ret) / 252
            ann = nav.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
            vol = ret.std(ddof=0) * math.sqrt(252)
            sharpe = (ret - rf).mean() / (ret - rf).std(ddof=0) * math.sqrt(252) if (ret - rf).std(ddof=0) > 0 else np.nan
            row = {
                "cross_state": cross_state,
                "name": name,
                "n_obs": len(sub),
                "strategy_return": ann,
                "strategy_vol": vol,
                "strategy_sharpe": sharpe,
                "strategy_maxdd": _max_drawdown_from_returns(ret),
            }
            if name == "REGIME_HEDGE_V1":
                for asset in ASSETS:
                    row[f"avg_weight_{asset}"] = sub[f"REGIME_HEDGE_V1_weight_{asset}"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def plot_equity_curves(panel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    for strategy in ["SPY_BUY_HOLD", "MONTHLY_EITHER_CONFIRM", "BACKBONE_V2_SPY_CASH", "REGIME_HEDGE_V1", "STATIC_60_30_10", "STATIC_70_20_10", "CASH_ONLY"]:
        ax.plot(panel["date"], panel[f"{strategy}_nav"], label=strategy)
    ax.set_yscale("log")
    ax.set_title("Equity Curves (Log Scale)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "equity_curve_log.png", dpi=150)
    plt.close(fig)


def plot_drawdowns(panel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    for strategy in ["SPY_BUY_HOLD", "MONTHLY_EITHER_CONFIRM", "BACKBONE_V2_SPY_CASH", "REGIME_HEDGE_V1", "STATIC_60_30_10", "STATIC_70_20_10"]:
        nav = panel[f"{strategy}_nav"]
        dd = nav / nav.cummax() - 1
        ax.plot(panel["date"], dd, label=strategy)
    ax.set_title("Drawdown Comparison")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "drawdown_comparison.png", dpi=150)
    plt.close(fig)


def plot_weight_stack(panel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.stackplot(
        panel["date"],
        panel["REGIME_HEDGE_V1_weight_SPY"],
        panel["REGIME_HEDGE_V1_weight_GOLD"],
        panel["REGIME_HEDGE_V1_weight_IEF"],
        panel["REGIME_HEDGE_V1_weight_CASH"],
        labels=ASSETS,
    )
    ax.set_title("REGIME_HEDGE_V1 Weight Stack")
    ax.legend(loc="upper left", ncol=4)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "weight_stack_REGIME_HEDGE_V1.png", dpi=150)
    plt.close(fig)


def plot_case_studies(panel: pd.DataFrame) -> None:
    cases = {
        "2015_2016": ("2015-05-01", "2016-03-31"),
        "2018Q4": ("2018-10-01", "2019-01-31"),
        "COVID_2020": ("2020-02-01", "2020-06-30"),
        "2022": ("2021-11-01", "2023-03-31"),
        "2025_PULLBACK": ("2025-01-01", "2025-12-31"),
    }
    for name, (start, end) in cases.items():
        sub = panel[(panel["date"] >= pd.Timestamp(start)) & (panel["date"] <= pd.Timestamp(end))].copy()
        if sub.empty:
            continue
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1.2, 1, 0.8]})
        axes[0].plot(sub["date"], sub["SPY_NAV"] / sub["SPY_NAV"].iloc[0], label="SPY")
        axes[0].plot(sub["date"], sub["BACKBONE_V2_SPY_CASH_nav"] / sub["BACKBONE_V2_SPY_CASH_nav"].iloc[0], label="BACKBONE_V2_SPY_CASH")
        axes[0].plot(sub["date"], sub["REGIME_HEDGE_V1_nav"] / sub["REGIME_HEDGE_V1_nav"].iloc[0], label="REGIME_HEDGE_V1")
        axes[0].legend(fontsize=8)
        axes[0].set_title(name)
        axes[1].plot(sub["date"], sub["spy_drawdown_from_previous_high"], label="SPY drawdown")
        axes[1].legend(fontsize=8)
        axes[2].stackplot(
            sub["date"],
            sub["REGIME_HEDGE_V1_weight_SPY"],
            sub["REGIME_HEDGE_V1_weight_GOLD"],
            sub["REGIME_HEDGE_V1_weight_IEF"],
            sub["REGIME_HEDGE_V1_weight_CASH"],
            labels=ASSETS,
        )
        axes[2].legend(fontsize=7, ncol=4)
        timing_num = sub["timing_state"].map({"NON_RISK": 0, "RISK": 1}).fillna(0)
        axes[3].plot(sub["date"], timing_num, label="timing_state")
        axes[3].set_yticks([0, 1], ["NON_RISK", "RISK"])
        axes[3].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(CONFIG["figure_dir"] / f"crisis_case_study_{name}.png", dpi=150)
        plt.close(fig)


def plot_regime_timeline(panel: pd.DataFrame) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1, 0.8, 0.8]})
    axes[0].plot(panel["date"], panel["SPY_NAV"], label="SPY_NAV")
    axes[0].plot(panel["date"], panel["REGIME_HEDGE_V1_nav"], label="REGIME_HEDGE_V1")
    axes[0].legend(fontsize=8)
    axes[1].plot(panel["date"], panel["spy_drawdown_from_previous_high"], label="SPY drawdown")
    axes[1].legend(fontsize=8)
    axes[2].plot(panel["date"], panel["macro_regime_confirmed"].astype("category").cat.codes, label="macro regime")
    axes[2].legend(fontsize=8)
    axes[3].plot(panel["date"], panel["timing_state"].map({"NON_RISK": 0, "RISK": 1}).fillna(0), label="timing_state")
    axes[3].set_yticks([0, 1], ["NON_RISK", "RISK"])
    axes[3].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "regime_and_timing_state_timeline.png", dpi=150)
    plt.close(fig)


def plot_performance_bars(perf: pd.DataFrame) -> None:
    show = perf[perf["strategy"].isin(["SPY_BUY_HOLD", "BACKBONE_V2_SPY_CASH", "REGIME_HEDGE_V1", "STATIC_60_30_10", "STATIC_70_20_10", "CASH_ONLY"])].copy()
    metrics = ["annualized_return", "sharpe_ratio", "max_drawdown", "calmar_ratio", "final_nav", "total_turnover"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    for ax, metric in zip(axes, metrics):
        ax.bar(show["strategy"], show[metric])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "performance_bar_charts.png", dpi=150)
    plt.close(fig)


def plot_cross_state_heatmap(cross_perf: pd.DataFrame) -> None:
    heat = cross_perf.pivot(index="name", columns="cross_state", values="strategy_sharpe")
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(heat.fillna(np.nan), aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_title("Cross-State Sharpe Heatmap")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "cross_state_return_heatmap.png", dpi=150)
    plt.close(fig)


def write_markdown_report(
    perf: pd.DataFrame,
    crisis: pd.DataFrame,
    regime_counts: pd.DataFrame,
    asset_source: str,
    fallback_days: int,
    sample_days: int,
) -> None:
    p = perf.set_index("strategy")
    def row(name: str) -> pd.Series:
        return p.loc[name]

    r = row("REGIME_HEDGE_V1")
    b = row("BACKBONE_V2_SPY_CASH")
    spy = row("SPY_BUY_HOLD")

    crisis_pivot = crisis.pivot(index="period", columns="strategy", values="cumulative_return") if not crisis.empty else pd.DataFrame()

    lines = [
        "# REGIME_AWARE_HEDGE_ALLOCATION_V1_REPORT",
        "",
        "## 1. Purpose",
        "This round converts the cross-state hedge diagnostic into a simple rule-based multi-asset allocation. No optimization, risk budgeting, or parameter search is used.",
        "",
        "## 2. Timing backbone",
        "BACKBONE_V2_UPGRADED is fixed:",
        "- FLAT = VIX OR credit",
        "- STEEP = Monthly Either SELL OR credit",
        "- INVERTED = no credit full-risk trigger",
        "- Exit = R3, SPY crosses above MA20",
        "",
        "## 3. Allocation rules",
        "- INVERTED = 70 SPY / 20 GOLD / 10 CASH",
        "- FLAT_NON_RISK = 60 SPY / 40 GOLD",
        "- FLAT_RISK = 100 GOLD",
        "- STEEP_NON_RISK = 90 SPY / 10 IEF",
        "- STEEP_RISK = 100 IEF",
        "- fallback NON_RISK = 80 SPY / 20 CASH",
        "- fallback RISK = 100 CASH",
        "",
        "## 4. Main performance",
        f"- REGIME_HEDGE_V1: AnnRet {r['annualized_return']:.2%}, Sharpe {r['sharpe_ratio']:.2f}, MaxDD {r['max_drawdown']:.2%}, Final NAV {r['final_nav']:.2f}",
        f"- BACKBONE_V2_SPY_CASH: AnnRet {b['annualized_return']:.2%}, Sharpe {b['sharpe_ratio']:.2f}, MaxDD {b['max_drawdown']:.2%}, Final NAV {b['final_nav']:.2f}",
        f"- SPY_BUY_HOLD: AnnRet {spy['annualized_return']:.2%}, Sharpe {spy['sharpe_ratio']:.2f}, MaxDD {spy['max_drawdown']:.2%}, Final NAV {spy['final_nav']:.2f}",
        "",
        "## 5. Crisis period analysis",
        "See `crisis_performance.csv` and the case-study figures. Focus points are 2008/GFC, 2015-2016, 2018Q4, COVID, 2022, and 2025 pullback.",
        "",
        "## 6. Allocation behavior",
        f"- Average weights: SPY {r['avg_weight_SPY']:.1%}, GOLD {r['avg_weight_GOLD']:.1%}, IEF {r['avg_weight_IEF']:.1%}, CASH {r['avg_weight_CASH']:.1%}",
        f"- Fallback days: {fallback_days} ({fallback_days / max(sample_days, 1):.1%} of sample, raw share in panel recorded in `regime_state_counts.csv`)",
        f"- Asset return sources: {asset_source}",
        "",
        "## 7. Interpretation",
        "This version directly tests whether replacing cash with Gold in FLAT risk and with IEF in STEEP risk improves the SPY/CASH backbone without adding optimization noise.",
        "",
        "## 8. Recommendation",
        "Use this report to decide whether to keep the regime-aware sleeves as-is, simplify to risk-only replacement, or run a narrow weight sensitivity test around the current hand-set rules.",
        "",
        "## 9. Caveats",
        "- Rules come from in-sample diagnostic work.",
        "- Hedge asset behavior can shift across regimes.",
        "- IEF can fail in rising-rate stress such as 2022.",
        "- Gold can underperform in parts of fast rebounds.",
        "- No weight optimization was attempted.",
    ]
    (CONFIG["output_dir"] / "REGIME_AWARE_HEDGE_ALLOCATION_V1_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = load_base_panel()
    panel, asset_source = load_asset_returns(panel)
    panel, event_log = build_backbone_v2_state(panel)
    panel, allocation_state_table = build_allocation_rules(panel)

    panel["entry_reason"] = panel.get("entry_reason", "").fillna("")
    panel["SPY_return"] = panel["SPY_return"].fillna(0.0)
    panel["GOLD_return"] = panel["GOLD_return"].fillna(0.0)
    panel["IEF_return"] = panel["IEF_return"].fillna(0.0)
    panel["CASH_return"] = panel["CASH_return"].fillna(0.0)

    if "MONTHLY_EITHER_CONFIRM_weight_spy" in panel.columns:
        panel["MONTHLY_EITHER_CONFIRM_weight_SPY"] = pd.to_numeric(panel["MONTHLY_EITHER_CONFIRM_weight_spy"], errors="coerce").fillna(1.0)
    else:
        panel["MONTHLY_EITHER_CONFIRM_weight_SPY"] = np.where(panel["monthly_either_state"].eq("SELL"), 0.0, 1.0)
    panel["MONTHLY_EITHER_CONFIRM_weight_GOLD"] = 0.0
    panel["MONTHLY_EITHER_CONFIRM_weight_IEF"] = 0.0
    panel["MONTHLY_EITHER_CONFIRM_weight_CASH"] = 1 - panel["MONTHLY_EITHER_CONFIRM_weight_SPY"]

    panel["BACKBONE_V2_SPY_CASH_weight_GOLD"] = 0.0
    panel["BACKBONE_V2_SPY_CASH_weight_IEF"] = 0.0

    regime_target = panel[[f"REGIME_HEDGE_V1_target_{asset}" for asset in ASSETS]].rename(
        columns={f"REGIME_HEDGE_V1_target_{asset}": asset for asset in ASSETS}
    )
    spy_target = pd.DataFrame({"SPY": 1.0, "GOLD": 0.0, "IEF": 0.0, "CASH": 0.0}, index=panel.index)
    cash_target = pd.DataFrame({"SPY": 0.0, "GOLD": 0.0, "IEF": 0.0, "CASH": 1.0}, index=panel.index)
    monthly_either_target = panel[[f"MONTHLY_EITHER_CONFIRM_weight_{asset}" for asset in ASSETS]].rename(
        columns={f"MONTHLY_EITHER_CONFIRM_weight_{asset}": asset for asset in ASSETS}
    )
    backbone_target = panel[[f"BACKBONE_V2_SPY_CASH_weight_{asset}" for asset in ASSETS]].rename(
        columns={f"BACKBONE_V2_SPY_CASH_weight_{asset}": asset for asset in ASSETS}
    )
    static_603010 = pd.DataFrame({"SPY": 0.60, "GOLD": 0.30, "IEF": 0.10, "CASH": 0.0}, index=panel.index)
    static_702010 = pd.DataFrame({"SPY": 0.70, "GOLD": 0.20, "IEF": 0.0, "CASH": 0.10}, index=panel.index)
    static_602020 = pd.DataFrame({"SPY": 0.60, "GOLD": 0.20, "IEF": 0.20, "CASH": 0.0}, index=panel.index)

    panel = run_multi_asset_backtest(panel, "SPY_BUY_HOLD", spy_target, monthly_rebalance=False)
    panel = run_multi_asset_backtest(panel, "CASH_ONLY", cash_target, monthly_rebalance=False)
    panel = run_multi_asset_backtest(panel, "MONTHLY_EITHER_CONFIRM", monthly_either_target, monthly_rebalance=False)
    panel = run_multi_asset_backtest(panel, "BACKBONE_V2_SPY_CASH", backbone_target, monthly_rebalance=False)
    panel = run_multi_asset_backtest(panel, "REGIME_HEDGE_V1", regime_target, monthly_rebalance=CONFIG["monthly_rebalance"])
    panel = run_multi_asset_backtest(panel, "STATIC_60_30_10", static_603010, monthly_rebalance=True)
    panel = run_multi_asset_backtest(panel, "STATIC_70_20_10", static_702010, monthly_rebalance=True)
    panel = run_multi_asset_backtest(panel, "STATIC_60_20_20", static_602020, monthly_rebalance=True)

    for asset in ASSETS:
        panel[f"REGIME_HEDGE_V1_weight_{asset}"] = panel[f"REGIME_HEDGE_V1_weight_{asset}"]

    panel["timing_state"] = panel["timing_state"].fillna("NON_RISK")
    panel["cross_state"] = panel["cross_state"].fillna(panel["macro_regime_confirmed"] + "_" + panel["timing_state"])
    panel["SPY_CROSS_ABOVE_MA20"] = panel["SPY_CROSS_ABOVE_MA20"].fillna(False)

    daily_cols = [
        "date",
        "macro_regime_confirmed",
        "timing_state",
        "cross_state",
        "entry_reason",
        "VIX_LEVEL",
        "VIX_ZSCORE_120D",
        "CREDIT_SPREAD_BAA_AAA",
        "D_CREDIT_SPREAD_20D",
        "spy_drawdown_from_previous_high",
        "SPY_CROSS_ABOVE_MA20",
        "SPY_return",
        "GOLD_return",
        "IEF_return",
        "CASH_return",
        "REGIME_HEDGE_V1_weight_SPY",
        "REGIME_HEDGE_V1_weight_GOLD",
        "REGIME_HEDGE_V1_weight_IEF",
        "REGIME_HEDGE_V1_weight_CASH",
        "BACKBONE_V2_SPY_CASH_weight_SPY",
        "BACKBONE_V2_SPY_CASH_weight_CASH",
        "SPY_BUY_HOLD_return",
        "SPY_BUY_HOLD_nav",
        "CASH_ONLY_return",
        "CASH_ONLY_nav",
        "MONTHLY_EITHER_CONFIRM_return",
        "MONTHLY_EITHER_CONFIRM_nav",
        "BACKBONE_V2_SPY_CASH_return",
        "BACKBONE_V2_SPY_CASH_nav",
        "REGIME_HEDGE_V1_return",
        "REGIME_HEDGE_V1_nav",
        "STATIC_60_30_10_return",
        "STATIC_60_30_10_nav",
        "STATIC_70_20_10_return",
        "STATIC_70_20_10_nav",
        "STATIC_60_20_20_return",
        "STATIC_60_20_20_nav",
        "transaction_cost_REGIME_HEDGE_V1",
        "turnover_REGIME_HEDGE_V1",
    ]
    panel["SPY_NAV"] = (1 + panel["SPY_return"]).cumprod()
    panel[daily_cols].to_csv(CONFIG["output_dir"] / "daily_backtest_panel.csv", index=False)

    perf = pd.DataFrame([compute_performance_metrics(panel, s) for s in STRATEGIES])
    perf.to_csv(CONFIG["output_dir"] / "performance_summary.csv", index=False)

    crisis = compute_crisis_performance(panel, STRATEGIES)
    crisis.to_csv(CONFIG["output_dir"] / "crisis_performance.csv", index=False)

    cross_perf = compute_cross_state_performance(panel)
    cross_perf.to_csv(CONFIG["output_dir"] / "performance_by_cross_state.csv", index=False)

    allocation_state_table.to_csv(CONFIG["output_dir"] / "allocation_state_table.csv", index=False)
    regime_counts = (
        panel.groupby(["macro_regime_confirmed", "timing_state", "cross_state"], dropna=False)
        .size()
        .rename("n_days")
        .reset_index()
    )
    regime_counts["percentage_of_sample"] = regime_counts["n_days"] / len(panel)
    regime_counts.to_csv(CONFIG["output_dir"] / "regime_state_counts.csv", index=False)

    perf_regime = []
    for strategy in STRATEGIES:
        for regime, sub in panel.groupby("macro_regime_confirmed", dropna=False):
            ret = sub[f"{strategy}_return"].fillna(0.0)
            rf = sub["CASH_return"].fillna(0.0)
            nav = (1 + ret).cumprod()
            years = len(ret) / 252
            ann = nav.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
            vol = ret.std(ddof=0) * math.sqrt(252)
            sharpe = (ret - rf).mean() / (ret - rf).std(ddof=0) * math.sqrt(252) if (ret - rf).std(ddof=0) > 0 else np.nan
            perf_regime.append(
                {
                    "strategy": strategy,
                    "macro_regime_confirmed": regime,
                    "n_obs": len(sub),
                    "annualized_return": ann,
                    "volatility": vol,
                    "Sharpe": sharpe,
                    "max_drawdown": _max_drawdown_from_returns(ret),
                    "time_in_cash": sub[f"{strategy}_weight_CASH"].mean(),
                }
            )
    pd.DataFrame(perf_regime).to_csv(CONFIG["output_dir"] / "performance_by_regime.csv", index=False)

    plot_equity_curves(panel)
    plot_drawdowns(panel)
    plot_weight_stack(panel)
    plot_regime_timeline(panel)
    plot_case_studies(panel)
    plot_performance_bars(perf)
    plot_cross_state_heatmap(cross_perf)

    fallback_days = int(panel["fallback_flag"].sum())
    write_markdown_report(perf, crisis, regime_counts, asset_source, fallback_days, len(panel))

    p = perf.set_index("strategy")
    reg = p.loc["REGIME_HEDGE_V1"]
    b2 = p.loc["BACKBONE_V2_SPY_CASH"]
    spy = p.loc["SPY_BUY_HOLD"]
    c2015 = crisis[(crisis["period"] == "2015_2016") & (crisis["strategy"].isin(["REGIME_HEDGE_V1", "BACKBONE_V2_SPY_CASH"]))].set_index("strategy")
    c2025 = crisis[(crisis["period"] == "2025_PULLBACK") & (crisis["strategy"].isin(["REGIME_HEDGE_V1", "BACKBONE_V2_SPY_CASH"]))].set_index("strategy")
    ccovid = crisis[(crisis["period"] == "COVID_2020") & (crisis["strategy"].isin(["REGIME_HEDGE_V1", "BACKBONE_V2_SPY_CASH"]))].set_index("strategy")
    c2022 = crisis[(crisis["period"] == "2022") & (crisis["strategy"].isin(["REGIME_HEDGE_V1", "BACKBONE_V2_SPY_CASH"]))].set_index("strategy")

    print(f"1. REGIME_HEDGE_V1 AnnRet / Sharpe / MaxDD / Final NAV: {reg['annualized_return']:.2%} / {reg['sharpe_ratio']:.2f} / {reg['max_drawdown']:.2%} / {reg['final_nav']:.2f}")
    print(f"2. BACKBONE_V2_SPY_CASH AnnRet / Sharpe / MaxDD / Final NAV: {b2['annualized_return']:.2%} / {b2['sharpe_ratio']:.2f} / {b2['max_drawdown']:.2%} / {b2['final_nav']:.2f}")
    print(f"3. SPY_BUY_HOLD AnnRet / Sharpe / MaxDD / Final NAV: {spy['annualized_return']:.2%} / {spy['sharpe_ratio']:.2f} / {spy['max_drawdown']:.2%} / {spy['final_nav']:.2f}")
    print(f"4. REGIME_HEDGE_V1 improves Sharpe vs BACKBONE_V2_SPY_CASH: {reg['sharpe_ratio'] > b2['sharpe_ratio']}")
    print(f"5. REGIME_HEDGE_V1 lowers MaxDD vs BACKBONE_V2_SPY_CASH: {reg['max_drawdown'] > b2['max_drawdown']}")
    if not c2015.empty:
        print(f"6. 2015-2016 improvement: {c2015.loc['REGIME_HEDGE_V1', 'cumulative_return'] > c2015.loc['BACKBONE_V2_SPY_CASH', 'cumulative_return']}")
    if not c2025.empty:
        print(f"7. 2025 pullback improvement: {c2025.loc['REGIME_HEDGE_V1', 'cumulative_return'] > c2025.loc['BACKBONE_V2_SPY_CASH', 'cumulative_return']}")
    if not ccovid.empty:
        print(f"8. COVID dragged vs backbone: {ccovid.loc['REGIME_HEDGE_V1', 'cumulative_return'] < ccovid.loc['BACKBONE_V2_SPY_CASH', 'cumulative_return']}")
    if not c2022.empty:
        print(f"9. 2022 dragged vs backbone: {c2022.loc['REGIME_HEDGE_V1', 'cumulative_return'] < c2022.loc['BACKBONE_V2_SPY_CASH', 'cumulative_return']}")
    print("10. Next suggestion: keep this as a first-pass regime-aware sleeve, then test a narrower risk-only replacement and small weight sensitivity around FLAT Gold / STEEP IEF sleeves.")


if __name__ == "__main__":
    main()
