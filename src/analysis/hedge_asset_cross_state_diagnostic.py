"""Hedge asset cross-state diagnostic.

This module studies hedge asset behavior by macro regime, timing risk state,
entry reason, and selected stress case windows. It does not run a portfolio
allocation or optimize weights.
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
    "output_dir": Path("results/hedge_asset_cross_state_diagnostic"),
    "figure_dir": Path("figures/hedge_asset_cross_state_diagnostic"),
    "timing_backbone": "BACKBONE_V2_UPGRADED",
    "case_windows": {
        "2015_2016": ["2015-05-01", "2016-03-31"],
        "RUSSIA_UKRAINE_WAR": ["2022-02-24", "2022-06-30"],
        "2025_PULLBACK": ["2025-01-01", "2025-12-31"],
        "2023_PULLBACK": ["2023-07-01", "2023-11-30"],
        "2018Q4": ["2018-10-01", "2019-01-31"],
        "COVID_2020": ["2020-02-01", "2020-06-30"],
        "2022": ["2021-11-01", "2023-03-31"],
    },
    "assets": ["SPY", "CASH", "IEF", "GOLD", "CMDTY_FUT", "TLT", "SHY", "TIP"],
    "trading_days_per_year": 252,
}


BASE_PANEL_CANDIDATES = [
    Path("results/spy_cash_backbone_upgrade_ablation/daily_backtest_panel.csv"),
    Path("results/spy_cash_stress_recovery_with_credit/daily_backtest_panel.csv"),
    Path("results/spy_cash_stress_recovery_with_commodity/daily_backtest_panel.csv"),
]

EVENT_LOG_CANDIDATES = [
    Path("results/spy_cash_backbone_upgrade_ablation/risk_state_event_log.csv"),
    Path("results/spy_cash_stress_recovery_with_credit/risk_state_event_log.csv"),
    Path("results/spy_cash_stress_recovery_with_commodity/risk_state_event_log.csv"),
]

ASSET_PANEL_CANDIDATES = [
    Path("results/regime_hedge_steep_sell_ief/daily_backtest_panel.csv"),
    Path("results/reconstructed_regime_asset_behavior/reconstructed_regime_panel.csv"),
]


def ensure_dirs() -> None:
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["figure_dir"].mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        date_cols = [c for c in df.columns if c.lower() in {"date", "datetime", "observation_date"}]
        if not date_cols:
            raise ValueError(f"No date column found in {path}")
        df = df.rename(columns={date_cols[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date")
    return df


def _first_existing(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    col_set = set(cols)
    for c in candidates:
        if c in col_set:
            return c
    lower_map = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _to_bool_state(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    s = series.astype(str).str.upper()
    return s.isin(["RISK", "TRUE", "1", "SELL", "CASH"])


def load_base_panel() -> pd.DataFrame:
    for path in BASE_PANEL_CANDIDATES:
        if path.exists():
            df = _read_csv(path)
            print(f"Loaded base panel: {path}")
            return df
    raise FileNotFoundError("No SPY/CASH stress-recovery base panel found.")


def load_event_log() -> Optional[pd.DataFrame]:
    for path in EVENT_LOG_CANDIDATES:
        if not path.exists():
            continue
        log = pd.read_csv(path)
        if "event_date" in log.columns:
            log["event_date"] = pd.to_datetime(log["event_date"])
        else:
            continue
        if "strategy" in log.columns:
            target = CONFIG["timing_backbone"]
            if target in set(log["strategy"].astype(str)):
                log = log[log["strategy"].astype(str).eq(target)]
            elif "STRESS_RECOVERY_R3_CREDIT_DD5" in set(log["strategy"].astype(str)):
                log = log[log["strategy"].astype(str).eq("STRESS_RECOVERY_R3_CREDIT_DD5")]
        if "event_type" in log.columns:
            log = log[log["event_type"].astype(str).eq("ENTER_RISK")]
        if not log.empty:
            print(f"Loaded risk event log: {path}")
            return log
    warnings.warn("No CREDIT_DD5 risk event log found; entry reasons will be inferred.")
    return None


def _load_optional_raw_asset(asset: str) -> Optional[pd.DataFrame]:
    candidates = [
        Path(f"data/raw/asset/{asset}.csv"),
        Path(f"data/raw/macro/commodity/{asset}.csv"),
        Path(f"data/raw/macro/dollar/{asset}.csv"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        raw = pd.read_csv(path, na_values=[".", "NA", ""])
        date_col = _first_existing(raw.columns, ["date", "DATE", "Date", "observation_date"])
        if not date_col:
            continue
        value_col = _first_existing(raw.columns, [asset, "VALUE", "value", "close", "Close", "adj_close", "Adj Close"])
        if not value_col:
            value_candidates = [c for c in raw.columns if c != date_col and pd.to_numeric(raw[c], errors="coerce").notna().any()]
            if not value_candidates:
                continue
            value_col = value_candidates[0]
        out = raw[[date_col, value_col]].rename(columns={date_col: "date", value_col: f"{asset}_price"})
        out["date"] = pd.to_datetime(out["date"])
        out[f"{asset}_price"] = pd.to_numeric(out[f"{asset}_price"], errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
        out[f"{asset}_return"] = out[f"{asset}_price"].pct_change()
        return out
    return None


def load_asset_returns(base: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Merge available hedge assets into the base panel."""
    panel = base.copy()
    notes: List[str] = []

    if "SPY_return" not in panel.columns:
        panel["SPY_return"] = panel.get("spy_daily_return")
    if "SPY_price" not in panel.columns:
        panel["SPY_price"] = panel.get("spy_price")
    if "CASH_return" not in panel.columns:
        panel["CASH_return"] = panel.get("daily_rf")

    for path in ASSET_PANEL_CANDIDATES:
        if not path.exists():
            continue
        src = _read_csv(path)
        rename_map = {}
        for asset in ["IEF", "GOLD", "CMDTY_FUT", "TLT", "SHY", "TIP"]:
            return_candidates = [
                f"{asset}_return",
                f"{asset}_RETURN",
                f"{asset}_ret",
                f"{asset}_RET",
            ]
            if asset == "CMDTY_FUT":
                return_candidates += ["CMDTY_return", "CMDTY_ret", "CMDTY_FUT_RETURN"]
            col = _first_existing(src.columns, return_candidates)
            if col:
                rename_map[col] = f"{asset}_return"
        keep = ["date"] + list(rename_map.keys())
        if len(keep) > 1:
            add = src[keep].rename(columns=rename_map)
            for col in add.columns:
                if col != "date" and col in panel.columns:
                    panel = panel.drop(columns=[col])
            panel = panel.merge(add, on="date", how="left")
            notes.append(f"merged returns from {path}")

    for asset in ["TLT", "SHY", "TIP"]:
        if f"{asset}_return" in panel.columns:
            continue
        raw = _load_optional_raw_asset(asset)
        if raw is not None:
            panel = panel.merge(raw[["date", f"{asset}_return"]], on="date", how="left")
            notes.append(f"loaded raw asset {asset}")

    if "CMDTY_FUT_return" not in panel.columns and "CMDTY_FUT_price" in panel.columns:
        panel["CMDTY_FUT_return"] = panel["CMDTY_FUT_price"].pct_change()

    available = []
    for asset in CONFIG["assets"]:
        col = f"{asset}_return"
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")
            panel[col] = panel[col].fillna(0.0 if asset == "CASH" else np.nan)
            if panel[col].notna().sum() > 30:
                available.append(asset)
                panel[f"{asset}_NAV"] = (1 + panel[col].fillna(0.0)).cumprod()
        else:
            notes.append(f"missing asset return: {asset}")

    return panel, available, notes


def build_timing_state(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    risk_col = _first_existing(
        df.columns,
        [
            "BACKBONE_V2_UPGRADED_risk_state",
            "CREDIT_DD5_R3_risk_state",
            "STRESS_RECOVERY_R3_CREDIT_DD5_risk_state",
        ],
    )
    weight_col = _first_existing(
        df.columns,
        [
            "BACKBONE_V2_UPGRADED_weight_spy",
            "CREDIT_DD5_R3_weight_spy",
            "STRESS_RECOVERY_R3_CREDIT_DD5_weight_spy",
        ],
    )
    nav_col = _first_existing(
        df.columns,
        [
            "BACKBONE_V2_UPGRADED_nav",
            "STRESS_RECOVERY_R3_CREDIT_DD5_nav",
        ],
    )

    if weight_col:
        df["timing_state"] = np.where(pd.to_numeric(df[weight_col], errors="coerce") >= 0.5, "NON_RISK", "RISK")
        df["best_weight_spy"] = pd.to_numeric(df[weight_col], errors="coerce")
    elif risk_col:
        is_risk = _to_bool_state(df[risk_col])
        df["timing_state"] = np.where(is_risk, "RISK", "NON_RISK")
        df["best_weight_spy"] = np.where(is_risk, 0.0, 1.0)
    else:
        raise ValueError("Cannot find CREDIT_DD5 R3 risk state or weight columns.")

    if nav_col:
        df["best_strategy_nav"] = df[nav_col]
    else:
        ret_col = _first_existing(df.columns, ["BACKBONE_V2_UPGRADED_return", "STRESS_RECOVERY_R3_CREDIT_DD5_return"])
        if ret_col:
            df["best_strategy_nav"] = (1 + df[ret_col].fillna(0.0)).cumprod()

    if "macro_regime_confirmed" not in df.columns:
        raise ValueError("macro_regime_confirmed is required.")
    df["macro_regime_confirmed"] = df["macro_regime_confirmed"].fillna("NEUTRAL").astype(str)
    df["cross_state"] = df["macro_regime_confirmed"] + "_" + df["timing_state"]

    if "spy_drawdown_from_previous_high" not in df.columns:
        df["previous_high"] = df["spy_price"].cummax()
        df["spy_drawdown_from_previous_high"] = df["spy_price"] / df["previous_high"] - 1

    return df


def _infer_entry_reason(row: pd.Series) -> str:
    if bool(row.get("DD5_AND_CREDIT_CHG20_GT_0_10", False)):
        return "DD5_AND_CREDIT_CHG20_GT_0_10"
    if row.get("macro_regime_confirmed") == "FLAT" and row.get("VIX_ZSCORE_120D", -np.inf) >= 3.0:
        return "FLAT_VIX_STRESS"
    if row.get("macro_regime_confirmed") == "STEEP" and row.get("monthly_either_state") == "SELL":
        return "STEEP_EITHER_SELL"
    return "OTHER"


def extract_risk_episodes(panel: pd.DataFrame, event_log: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df = panel.reset_index(drop=True)
    is_risk = df["timing_state"].eq("RISK")
    episodes = []
    reason_map = {}
    if event_log is not None and "event_date" in event_log.columns and "reason" in event_log.columns:
        reason_map = event_log.drop_duplicates("event_date").set_index("event_date")["reason"].to_dict()
    starts = df.index[is_risk & ~is_risk.shift(1, fill_value=False)]
    for eid, start_idx in enumerate(starts, 1):
        end_idx = start_idx
        while end_idx + 1 < len(df) and is_risk.iloc[end_idx + 1]:
            end_idx += 1
        ep = df.iloc[start_idx : end_idx + 1]
        start = ep.iloc[0]
        entry_reason = reason_map.get(pd.Timestamp(start["date"]), _infer_entry_reason(start))
        spy_path = (1 + ep["SPY_return"].fillna(0.0)).cumprod()
        episodes.append(
            {
                "episode_id": eid,
                "risk_start_date": start["date"],
                "risk_end_date": ep.iloc[-1]["date"],
                "duration_days": len(ep),
                "entry_reason": entry_reason,
                "macro_regime_at_entry": start["macro_regime_confirmed"],
                "dominant_macro_regime": ep["macro_regime_confirmed"].mode().iloc[0] if not ep.empty else np.nan,
                "SPY_drawdown_at_entry": start.get("spy_drawdown_from_previous_high", np.nan),
                "VIX_ZSCORE_at_entry": start.get("VIX_ZSCORE_120D", np.nan),
                "CREDIT_SPREAD_at_entry": start.get("CREDIT_SPREAD_BAA_AAA", np.nan),
                "D_CREDIT_SPREAD_20D_at_entry": start.get("D_CREDIT_SPREAD_20D", np.nan),
                "SPY_return_during_episode": spy_path.iloc[-1] - 1 if len(spy_path) else np.nan,
                "SPY_max_drawdown_during_episode": (spy_path / spy_path.cummax() - 1).min() if len(spy_path) else np.nan,
                "SPY_max_runup_during_episode": spy_path.max() - 1 if len(spy_path) else np.nan,
            }
        )
    return pd.DataFrame(episodes)


def attach_episode_reasons(panel: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["risk_episode_id"] = np.nan
    df["entry_reason"] = ""
    if episodes.empty:
        return df
    for _, ep in episodes.iterrows():
        mask = (df["date"] >= ep["risk_start_date"]) & (df["date"] <= ep["risk_end_date"])
        df.loc[mask, "risk_episode_id"] = ep["episode_id"]
        df.loc[mask, "entry_reason"] = ep["entry_reason"]
    return df


def _max_drawdown(ret: pd.Series) -> float:
    nav = (1 + ret.fillna(0.0)).cumprod()
    if nav.empty:
        return np.nan
    return float((nav / nav.cummax() - 1).min())


def _perf_from_returns(ret: pd.Series, rf: Optional[pd.Series] = None, min_obs: int = 2) -> Dict[str, float]:
    ret = pd.to_numeric(ret, errors="coerce").dropna()
    if len(ret) < min_obs:
        return {
            "n_obs": len(ret),
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "Sharpe": np.nan,
            "max_drawdown": np.nan,
            "average_daily_return": np.nan,
            "positive_day_ratio": np.nan,
            "worst_1d_return": np.nan,
            "best_1d_return": np.nan,
            "cumulative_return_within_group": np.nan,
        }
    nav = (1 + ret).cumprod()
    years = len(ret) / CONFIG["trading_days_per_year"]
    ann = nav.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = ret.std(ddof=0) * math.sqrt(CONFIG["trading_days_per_year"])
    if rf is not None:
        rf = pd.to_numeric(rf.reindex(ret.index), errors="coerce").fillna(0.0)
        excess = ret - rf
    else:
        excess = ret
    sharpe = excess.mean() / excess.std(ddof=0) * math.sqrt(CONFIG["trading_days_per_year"]) if excess.std(ddof=0) > 0 else np.nan
    return {
        "n_obs": len(ret),
        "annualized_return": ann,
        "annualized_volatility": vol,
        "Sharpe": sharpe,
        "max_drawdown": _max_drawdown(ret),
        "average_daily_return": ret.mean(),
        "positive_day_ratio": (ret > 0).mean(),
        "worst_1d_return": ret.min(),
        "best_1d_return": ret.max(),
        "cumulative_return_within_group": nav.iloc[-1] - 1,
    }


def _forward_event_metrics(group: pd.DataFrame) -> Tuple[float, float]:
    starts = group.index[group.index.to_series().diff().fillna(2).ne(1)]
    fwd_rets = []
    fwd_mdds = []
    for idx in starts:
        window = group.loc[idx : idx + 20, "SPY_return"] if isinstance(group.index, pd.RangeIndex) else group.iloc[0:0]["SPY_return"]
        if len(window) < 2:
            continue
        nav = (1 + window.fillna(0.0)).cumprod()
        fwd_rets.append(nav.iloc[-1] - 1)
        fwd_mdds.append((nav / nav.cummax() - 1).min())
    return (float(np.nanmean(fwd_rets)) if fwd_rets else np.nan, float(np.nanmean(fwd_mdds)) if fwd_mdds else np.nan)


def compute_asset_performance_by_state(panel: pd.DataFrame, assets: List[str], group_col: str) -> pd.DataFrame:
    records = []
    df = panel.reset_index(drop=True)
    for state, sub in df.groupby(group_col, dropna=False):
        rf = sub["CASH_return"] if "CASH_return" in sub.columns else None
        for asset in assets:
            col = f"{asset}_return"
            if col not in sub.columns:
                continue
            perf = _perf_from_returns(sub[col], rf=rf)
            fwd_ret, fwd_mdd = _forward_event_metrics(sub)
            perf.update(
                {
                    group_col: state,
                    "asset": asset,
                    "avg_forward_return_21d_from_group_start": fwd_ret,
                    "avg_forward_mdd_21d_from_group_start": fwd_mdd,
                }
            )
            records.append(perf)
    return pd.DataFrame(records)


def compute_risk_episode_asset_performance(panel: pd.DataFrame, episodes: pd.DataFrame, assets: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    for _, ep in episodes.iterrows():
        mask = (panel["date"] >= ep["risk_start_date"]) & (panel["date"] <= ep["risk_end_date"])
        sub = panel.loc[mask].copy()
        if sub.empty:
            continue
        rf = sub["CASH_return"] if "CASH_return" in sub.columns else None
        for asset in assets:
            col = f"{asset}_return"
            if col not in sub.columns:
                continue
            perf = _perf_from_returns(sub[col], rf=rf)
            perf.update(ep.to_dict())
            perf["asset"] = asset
            records.append(perf)
    detail = pd.DataFrame(records)
    if detail.empty:
        return detail, pd.DataFrame()
    detail["rank_by_return"] = detail.groupby("episode_id")["cumulative_return_within_group"].rank(ascending=False, method="min")
    detail["rank_by_drawdown"] = detail.groupby("episode_id")["max_drawdown"].rank(ascending=False, method="min")
    detail["rank_by_sharpe"] = detail.groupby("episode_id")["Sharpe"].rank(ascending=False, method="min")
    summary = (
        detail.groupby(["entry_reason", "asset"], dropna=False)
        .agg(
            episode_count=("episode_id", "nunique"),
            avg_episode_return=("cumulative_return_within_group", "mean"),
            median_episode_return=("cumulative_return_within_group", "median"),
            avg_episode_maxdd=("max_drawdown", "mean"),
            avg_episode_sharpe=("Sharpe", "mean"),
            avg_rank_by_return=("rank_by_return", "mean"),
            avg_rank_by_drawdown=("rank_by_drawdown", "mean"),
            avg_rank_by_sharpe=("rank_by_sharpe", "mean"),
        )
        .reset_index()
    )
    return detail, summary


def _local_drawdown_episode(case: pd.DataFrame) -> Dict[str, object]:
    nav = case["SPY_NAV"] / case["SPY_NAV"].iloc[0]
    running_high = nav.cummax()
    dd = nav / running_high - 1
    trough_pos = int(dd.values.argmin())
    trough_date = case.iloc[trough_pos]["date"]
    peak_slice = nav.iloc[: trough_pos + 1]
    peak_pos = int(peak_slice.values.argmax())
    peak_date = case.iloc[peak_pos]["date"]
    recovery_date = pd.NaT
    post = dd.iloc[trough_pos:]
    recovered = post[post >= -0.02]
    if not recovered.empty:
        recovery_date = case.loc[recovered.index[0], "date"]
    return {
        "local_peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
        "max_drawdown": dd.iloc[trough_pos],
        "duration_days": len(case.iloc[peak_pos : trough_pos + 1]),
    }


def define_missed_stress_cases(panel: pd.DataFrame, assets: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    case_rows = []
    asset_rows = []
    daily_panels: Dict[str, pd.DataFrame] = {}
    max_date = panel["date"].max()
    for name, (start, end) in CONFIG["case_windows"].items():
        start_dt = pd.Timestamp(start)
        end_dt = min(pd.Timestamp(end), max_date)
        sub = panel[(panel["date"] >= start_dt) & (panel["date"] <= end_dt)].copy()
        if sub.empty:
            warnings.warn(f"Case window has no data: {name}")
            continue
        for asset in assets:
            col = f"{asset}_return"
            if col in sub.columns:
                sub[f"{asset}_CASE_NAV"] = (1 + sub[col].fillna(0.0)).cumprod()
        daily_panels[name] = sub
        dd_info = _local_drawdown_episode(sub)
        peak_date = dd_info["local_peak_date"]
        trough_date = dd_info["trough_date"]
        peak_row = sub.loc[sub["date"].eq(peak_date)].iloc[0]
        trough_row = sub.loc[sub["date"].eq(trough_date)].iloc[0]
        risk_before_trough = sub[(sub["date"] <= trough_date) & sub["timing_state"].eq("RISK")]
        risk_entry_date = risk_before_trough["date"].iloc[0] if not risk_before_trough.empty else pd.NaT
        entry_reason = risk_before_trough["entry_reason"].replace("", np.nan).dropna().iloc[0] if not risk_before_trough.empty and risk_before_trough["entry_reason"].replace("", np.nan).notna().any() else ""
        case_rows.append(
            {
                "case_name": name,
                "case_start": sub["date"].iloc[0],
                "case_end": sub["date"].iloc[-1],
                **dd_info,
                "macro_regime_at_peak": peak_row["macro_regime_confirmed"],
                "macro_regime_at_trough": trough_row["macro_regime_confirmed"],
                "dominant_macro_regime": sub["macro_regime_confirmed"].mode().iloc[0],
                "timing_state_at_peak": peak_row["timing_state"],
                "timing_state_at_trough": trough_row["timing_state"],
                "entered_RISK_before_trough": not risk_before_trough.empty,
                "risk_entry_date": risk_entry_date,
                "entry_reason": entry_reason,
            }
        )
        peak_to_trough = sub[(sub["date"] >= peak_date) & (sub["date"] <= trough_date)]
        trough_to_end = sub[sub["date"] >= trough_date]
        for asset in assets:
            col = f"{asset}_return"
            if col not in sub.columns:
                continue
            full = _perf_from_returns(sub[col], rf=sub.get("CASH_return"))
            ptt = _perf_from_returns(peak_to_trough[col], rf=peak_to_trough.get("CASH_return"))
            tte = _perf_from_returns(trough_to_end[col], rf=trough_to_end.get("CASH_return"))
            asset_rows.append(
                {
                    "case_name": name,
                    "asset": asset,
                    "peak_to_trough_return": ptt["cumulative_return_within_group"],
                    "peak_to_trough_maxdd": ptt["max_drawdown"],
                    "full_window_return": full["cumulative_return_within_group"],
                    "full_window_maxdd": full["max_drawdown"],
                    "full_window_volatility": full["annualized_volatility"],
                    "full_window_sharpe": full["Sharpe"],
                    "trough_to_end_return": tte["cumulative_return_within_group"],
                    "trough_to_end_maxdd": tte["max_drawdown"],
                }
            )
    case_summary = pd.DataFrame(case_rows)
    asset_perf = pd.DataFrame(asset_rows)
    if not asset_perf.empty:
        asset_perf["rank_by_peak_to_trough_return"] = asset_perf.groupby("case_name")["peak_to_trough_return"].rank(
            ascending=False, method="min"
        )
        asset_perf["rank_by_full_window_return"] = asset_perf.groupby("case_name")["full_window_return"].rank(
            ascending=False, method="min"
        )
    return case_summary, asset_perf, daily_panels


def compute_correlations_by_state(panel: pd.DataFrame, assets: List[str], daily_panels: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    return_cols = [f"{a}_return" for a in assets if f"{a}_return" in panel.columns]
    states = {"RISK": panel[panel["timing_state"].eq("RISK")]}
    for state in ["FLAT_RISK", "STEEP_RISK"]:
        states[state] = panel[panel["cross_state"].eq(state)]
    for state, sub in panel.groupby("cross_state"):
        if state not in states and len(sub) >= 30:
            states[state] = sub
    for case in ["2015_2016", "2025_PULLBACK"]:
        if case in daily_panels:
            states[case] = daily_panels[case]
    for name, sub in states.items():
        if len(sub) < 10:
            continue
        corr = sub[return_cols].corr()
        for r in corr.index:
            for c in corr.columns:
                records.append(
                    {
                        "state": name,
                        "asset_1": r.replace("_return", ""),
                        "asset_2": c.replace("_return", ""),
                        "correlation": corr.loc[r, c],
                    }
                )
        out_path = CONFIG["output_dir"] / f"correlation_matrix_{name}.csv"
        corr.to_csv(out_path)
    return pd.DataFrame(records)


def _normalize(s: pd.Series, inverse: bool = False) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if inverse:
        x = -x
    rng = x.max() - x.min()
    if not np.isfinite(rng) or rng == 0:
        return pd.Series(0.5, index=s.index)
    return (x - x.min()) / rng


def rank_hedge_assets(perf: pd.DataFrame, state_col: str, corr_panel: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if perf.empty:
        return pd.DataFrame()
    rows = []
    for state, sub in perf.groupby(state_col, dropna=False):
        sub = sub.copy()
        if len(sub) == 0:
            continue
        spy_corr = pd.Series(np.nan, index=sub.index)
        if corr_panel is not None and not corr_panel.empty:
            cm = corr_panel[(corr_panel["state"] == state) & (corr_panel["asset_2"] == "SPY")]
            corr_map = cm.set_index("asset_1")["correlation"].to_dict()
            spy_corr = sub["asset"].map(corr_map)
        ann = sub["annualized_return"].fillna(0.0)
        sharpe = sub["Sharpe"].fillna(0.0)
        maxdd = sub["max_drawdown"].fillna(0.0)
        sub["score"] = (
            0.35 * _normalize(ann)
            + 0.25 * _normalize(sharpe)
            + 0.25 * _normalize(maxdd)
            + 0.15 * _normalize(-spy_corr.fillna(0.0))
        )
        sub["rank"] = sub["score"].rank(ascending=False, method="min")
        rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def rank_by_missed_case(asset_perf: pd.DataFrame) -> pd.DataFrame:
    if asset_perf.empty:
        return pd.DataFrame()
    rows = []
    for case, sub in asset_perf.groupby("case_name"):
        sub = sub.copy()
        sharpe = sub["full_window_sharpe"].fillna(0.0)
        sub["score"] = (
            0.45 * _normalize(sub["peak_to_trough_return"])
            + 0.25 * _normalize(sub["full_window_return"])
            + 0.20 * _normalize(sub["peak_to_trough_maxdd"])
            + 0.10 * _normalize(sharpe)
        )
        sub["rank"] = sub["score"].rank(ascending=False, method="min")
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def _save_heatmap(pivot: pd.DataFrame, title: str, path: Path, fmt: str = ".2f") -> None:
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 0.7), max(5, len(pivot.index) * 0.4)))
    data = pivot.astype(float)
    im = ax.imshow(data.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_heatmaps(cross_perf: pd.DataFrame, risk_rank: pd.DataFrame) -> None:
    fig_dir = CONFIG["figure_dir"]
    for metric, name, fmt in [
        ("annualized_return", "asset_return_by_cross_state_heatmap.png", ".1%"),
        ("Sharpe", "asset_sharpe_by_cross_state_heatmap.png", ".2f"),
        ("max_drawdown", "asset_maxdd_by_cross_state_heatmap.png", ".1%"),
    ]:
        pivot = cross_perf.pivot_table(index="asset", columns="cross_state", values=metric)
        _save_heatmap(pivot, metric, fig_dir / name, fmt=fmt)

    if not risk_rank.empty:
        sub = risk_rank[risk_rank["timing_state"].eq("RISK")].sort_values("score", ascending=False).head(10)
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.bar(sub["asset"], sub["score"])
            ax.set_title("Top hedge assets in RISK state")
            ax.set_ylabel("diagnostic score")
            fig.tight_layout()
            fig.savefig(fig_dir / "hedge_asset_ranking_by_risk_state.png", dpi=150)
            plt.close(fig)


def plot_risk_episode_boxplot(detail: pd.DataFrame) -> None:
    if detail.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [g["cumulative_return_within_group"].dropna().values for _, g in detail.groupby("asset")]
    labels = [k for k, _ in detail.groupby("asset")]
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Risk episode asset cumulative returns")
    ax.set_ylabel("episode return")
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "risk_episode_asset_performance_boxplot.png", dpi=150)
    plt.close(fig)


def _plot_state_strip(ax, dates: pd.Series, values: pd.Series, title: str) -> None:
    codes = pd.Categorical(values).codes
    ax.imshow([codes], aspect="auto", extent=[dates.iloc[0], dates.iloc[-1], 0, 1], cmap="tab20")
    ax.set_yticks([])
    ax.set_title(title, loc="left", fontsize=9)


def plot_case_studies(daily_panels: Dict[str, pd.DataFrame], assets: List[str]) -> None:
    selected = {
        "2015_2016": "case_study_2015_2016_hedge_assets.png",
        "2025_PULLBACK": "case_study_2025_pullback_hedge_assets.png",
        "2018Q4": "case_study_2018Q4_hedge_assets.png",
        "2022": "case_study_2022_hedge_assets.png",
    }
    for name, filename in selected.items():
        if name not in daily_panels:
            continue
        sub = daily_panels[name].copy()
        if len(sub) < 5:
            continue
        fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [2, 2, 0.5, 1.5]})
        axes[0].plot(sub["date"], sub["SPY_NAV"] / sub["SPY_NAV"].iloc[0], label="SPY")
        axes[0].plot(sub["date"], 1 + sub["spy_drawdown_from_previous_high"], label="SPY dd index", alpha=0.7)
        axes[0].set_title(name)
        axes[0].legend(loc="best", fontsize=8)
        for asset in assets:
            col = f"{asset}_CASE_NAV"
            if col in sub.columns:
                axes[1].plot(sub["date"], sub[col] / sub[col].iloc[0], label=asset)
        axes[1].legend(loc="best", ncol=4, fontsize=8)
        _plot_state_strip(axes[2], sub["date"], sub["cross_state"], "cross state")
        if "VIX_ZSCORE_120D" in sub.columns:
            axes[3].plot(sub["date"], sub["VIX_ZSCORE_120D"], label="VIX z120")
        if "D_CREDIT_SPREAD_20D" in sub.columns:
            axes[3].plot(sub["date"], sub["D_CREDIT_SPREAD_20D"], label="credit chg20")
        axes[3].axhline(0, color="black", linewidth=0.7)
        axes[3].legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(CONFIG["figure_dir"] / filename, dpi=150)
        plt.close(fig)


def plot_correlation_heatmaps(corr_panel: pd.DataFrame) -> None:
    if corr_panel.empty:
        return
    mapping = {
        "RISK": "correlation_heatmap_RISK.png",
        "2015_2016": "correlation_heatmap_2015_2016.png",
        "2025_PULLBACK": "correlation_heatmap_2025_pullback.png",
    }
    for state, filename in mapping.items():
        sub = corr_panel[corr_panel["state"].eq(state)]
        if sub.empty:
            continue
        pivot = sub.pivot(index="asset_1", columns="asset_2", values="correlation")
        _save_heatmap(pivot, f"Correlation: {state}", CONFIG["figure_dir"] / filename, fmt=".2f")


def write_markdown_report(
    panel: pd.DataFrame,
    assets: List[str],
    notes: List[str],
    macro_perf: pd.DataFrame,
    timing_perf: pd.DataFrame,
    cross_perf: pd.DataFrame,
    risk_summary: pd.DataFrame,
    case_summary: pd.DataFrame,
    case_asset_perf: pd.DataFrame,
    risk_rank: pd.DataFrame,
    case_rank: pd.DataFrame,
) -> None:
    out = CONFIG["output_dir"] / "HEDGE_ASSET_CROSS_STATE_DIAGNOSTIC.md"

    def table(df: pd.DataFrame, n: int = 10) -> str:
        if df.empty:
            return "_No data._"
        return df.head(n).to_markdown(index=False)

    risk_top = risk_rank[risk_rank.get("timing_state", pd.Series(dtype=str)).eq("RISK")].sort_values("score", ascending=False)
    flat_risk = cross_perf[cross_perf["cross_state"].eq("FLAT_RISK")].sort_values("Sharpe", ascending=False)
    steep_risk = cross_perf[cross_perf["cross_state"].eq("STEEP_RISK")].sort_values("Sharpe", ascending=False)
    credit_top = risk_summary[risk_summary["entry_reason"].eq("DD5_AND_CREDIT_CHG20_GT_0_10")].sort_values(
        "avg_episode_return", ascending=False
    )
    case_top = case_rank.sort_values(["case_name", "rank"]).groupby("case_name").head(3)

    content = f"""# Hedge Asset Cross-State Diagnostic

## Purpose

This diagnostic studies hedge asset behavior by macro regime, current timing risk state, risk entry reason, and selected stress windows. It does not set portfolio weights or optimize an allocation.

## Current Timing Backbone

- FLAT + VIX OR credit stress.
- STEEP + Monthly Either SELL.
- STEEP + credit DD5 stress.
- INVERTED credit full-risk trigger disabled.
- Recovery = R3, SPY crosses above MA20.
- NON_RISK = 100% SPY in the timing backbone; RISK = 100% CASH.

## Asset Universe

Available assets: {", ".join(assets)}.

Data notes:
{chr(10).join("- " + n for n in notes) if notes else "- No missing optional asset notes."}

## Cross-State Results

Top assets in RISK by diagnostic score:

{table(risk_top[["timing_state", "asset", "annualized_return", "Sharpe", "max_drawdown", "score", "rank"]], 8)}

FLAT_RISK by Sharpe:

{table(flat_risk[["cross_state", "asset", "annualized_return", "Sharpe", "max_drawdown", "cumulative_return_within_group"]], 8)}

STEEP_RISK by Sharpe:

{table(steep_risk[["cross_state", "asset", "annualized_return", "Sharpe", "max_drawdown", "cumulative_return_within_group"]], 8)}

## Risk Episode Results

Risk episode asset performance by entry reason:

{table(risk_summary.sort_values(["entry_reason", "avg_rank_by_return"])[["entry_reason", "asset", "episode_count", "avg_episode_return", "avg_episode_maxdd", "avg_rank_by_return"]], 20)}

Credit stress risk episodes:

{table(credit_top[["entry_reason", "asset", "episode_count", "avg_episode_return", "avg_episode_maxdd", "avg_rank_by_return"]], 8)}

## Missed Stress Case Studies

Case summary:

{table(case_summary, 20)}

Top hedge assets by missed/stress case:

{table(case_top[["case_name", "asset", "peak_to_trough_return", "full_window_return", "peak_to_trough_maxdd", "score", "rank"]], 30)}

## Correlation Findings

Correlation heatmaps are saved in the figure directory for RISK, 2015-2016, and 2025 pullback when enough observations exist. These should be treated as conditional diagnostics, not stable long-run covariance estimates.

## Interpretation

- CASH remains the cleanest low-volatility baseline in many RISK states, but it is not always the highest-return hedge during stress episodes.
- IEF often needs to be judged by stress type: it can help in disinflationary or growth-scare selloffs, but can fail when rates are rising.
- GOLD can diversify some risk states, especially where equity stress coincides with rate or policy uncertainty.
- CMDTY_FUT should be handled cautiously in RISK states; it can be useful in inflationary regimes but is not a generic hedge for global-growth stress.
- 2015-2016 and recent pullbacks should be evaluated as distinct stress types before assigning any single defensive sleeve.

## Recommended Next Step

Use this diagnostic to choose a small number of explicit hedge-allocation candidates for formal backtest:

- RISK = 100% CASH baseline.
- RISK = 50% CASH / 50% IEF.
- Entry-reason-conditioned hedge basket.
- Keep NORMAL = 100% SPY first, then separately test a normal hedge sleeve.

## Caveats

- Conditional state samples can be short.
- ETF/proxy histories differ by asset.
- Commodity futures proxies are not identical to investable ETF implementation.
- This is diagnostic evidence only; it is not a final portfolio allocation.
"""
    out.write_text(content, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    base = load_base_panel()
    event_log = load_event_log()
    panel, assets, notes = load_asset_returns(base)
    panel = build_timing_state(panel)

    episodes = extract_risk_episodes(panel, event_log=event_log)
    panel = attach_episode_reasons(panel, episodes)
    panel.to_csv(CONFIG["output_dir"] / "hedge_asset_cross_state_daily_panel.csv", index=False)
    episodes.to_csv(CONFIG["output_dir"] / "risk_episodes.csv", index=False)

    macro_perf = compute_asset_performance_by_state(panel, assets, "macro_regime_confirmed")
    timing_perf = compute_asset_performance_by_state(panel, assets, "timing_state")
    cross_perf = compute_asset_performance_by_state(panel, assets, "cross_state")
    macro_perf.to_csv(CONFIG["output_dir"] / "asset_performance_by_macro_regime.csv", index=False)
    timing_perf.to_csv(CONFIG["output_dir"] / "asset_performance_by_timing_state.csv", index=False)
    cross_perf.to_csv(CONFIG["output_dir"] / "asset_performance_by_cross_state.csv", index=False)

    risk_detail, risk_summary = compute_risk_episode_asset_performance(panel, episodes, assets)
    risk_detail.to_csv(CONFIG["output_dir"] / "risk_episode_asset_performance.csv", index=False)
    risk_summary.to_csv(CONFIG["output_dir"] / "risk_episode_asset_summary_by_entry_reason.csv", index=False)
    risk_summary.rename(columns={"entry_reason": "entry_reason"}).to_csv(
        CONFIG["output_dir"] / "asset_performance_by_entry_reason.csv", index=False
    )

    case_summary, case_asset_perf, daily_panels = define_missed_stress_cases(panel, assets)
    case_summary.to_csv(CONFIG["output_dir"] / "missed_stress_case_summary.csv", index=False)
    case_asset_perf.to_csv(CONFIG["output_dir"] / "missed_stress_asset_performance.csv", index=False)
    for name, sub in daily_panels.items():
        sub.to_csv(CONFIG["output_dir"] / f"case_study_{name}.csv", index=False)

    corr_panel = compute_correlations_by_state(panel, assets, daily_panels)
    corr_panel.to_csv(CONFIG["output_dir"] / "asset_correlation_by_cross_state.csv", index=False)

    macro_rank = rank_hedge_assets(macro_perf, "macro_regime_confirmed")
    timing_rank = rank_hedge_assets(timing_perf, "timing_state")
    cross_rank = rank_hedge_assets(cross_perf, "cross_state", corr_panel)
    entry_rank = pd.DataFrame()
    if not risk_summary.empty:
        tmp = risk_summary.rename(
            columns={
                "avg_episode_return": "annualized_return",
                "avg_episode_sharpe": "Sharpe",
                "avg_episode_maxdd": "max_drawdown",
            }
        )
        entry_rank = rank_hedge_assets(tmp, "entry_reason")
    case_rank = rank_by_missed_case(case_asset_perf)
    cross_rank.to_csv(CONFIG["output_dir"] / "hedge_asset_ranking_by_cross_state.csv", index=False)
    entry_rank.to_csv(CONFIG["output_dir"] / "hedge_asset_ranking_by_entry_reason.csv", index=False)
    case_rank.to_csv(CONFIG["output_dir"] / "hedge_asset_ranking_by_missed_case.csv", index=False)

    plot_heatmaps(cross_perf, timing_rank)
    plot_risk_episode_boxplot(risk_detail)
    plot_case_studies(daily_panels, assets)
    plot_correlation_heatmaps(corr_panel)
    write_markdown_report(
        panel,
        assets,
        notes,
        macro_perf,
        timing_perf,
        cross_perf,
        risk_summary,
        case_summary,
        case_asset_perf,
        timing_rank,
        case_rank,
    )

    sample_range = f"{panel['date'].min().date()} to {panel['date'].max().date()}"
    risk_top = timing_rank[timing_rank["timing_state"].eq("RISK")].sort_values("score", ascending=False).head(3)
    flat_top = cross_rank[cross_rank["cross_state"].eq("FLAT_RISK")].sort_values("score", ascending=False).head(3)
    steep_top = cross_rank[cross_rank["cross_state"].eq("STEEP_RISK")].sort_values("score", ascending=False).head(3)
    credit_top = entry_rank[entry_rank.get("entry_reason", pd.Series(dtype=str)).eq("DD5_AND_CREDIT_CHG20_GT_0_10")].sort_values(
        "score", ascending=False
    ).head(3)
    case_2015 = case_summary[case_summary["case_name"].eq("2015_2016")]
    case_2025 = case_summary[case_summary["case_name"].eq("2025_PULLBACK")]
    case_top = case_rank.sort_values(["case_name", "rank"]).groupby("case_name").head(3)

    print(f"1. Sample range: {sample_range}")
    print(f"2. Available assets: {', '.join(assets)}")
    print(f"3. RISK episodes: {len(episodes)}")
    print("4. RISK top 3 hedge assets:", ", ".join(risk_top["asset"].tolist()) if not risk_top.empty else "n/a")
    print("5. FLAT_RISK top 3 hedge assets:", ", ".join(flat_top["asset"].tolist()) if not flat_top.empty else "n/a")
    print("6. STEEP_RISK top 3 hedge assets:", ", ".join(steep_top["asset"].tolist()) if not steep_top.empty else "n/a")
    print("7. CREDIT_DD5_STRESS top 3 hedge assets:", ", ".join(credit_top["asset"].tolist()) if not credit_top.empty else "n/a")
    if not case_2015.empty:
        top = case_top[case_top["case_name"].eq("2015_2016")]["asset"].head(3).tolist()
        print(f"8. 2015-2016 regime/top hedges: {case_2015.iloc[0]['dominant_macro_regime']} / {', '.join(top)}")
    else:
        print("8. 2015-2016 regime/top hedges: n/a")
    if not case_2025.empty:
        top = case_top[case_top["case_name"].eq("2025_PULLBACK")]["asset"].head(3).tolist()
        print(f"9. 2025 pullback regime/top hedges: {case_2025.iloc[0]['dominant_macro_regime']} / {', '.join(top)}")
    else:
        print("9. 2025 pullback regime/top hedges: n/a")
    print("10. Next allocation candidates: CASH baseline, CASH/IEF split, entry-reason-conditioned hedge basket.")
    print(f"Saved outputs: {CONFIG['output_dir'].resolve()} and {CONFIG['figure_dir'].resolve()}")


if __name__ == "__main__":
    main()
