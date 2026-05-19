"""Controlled marginal test for STEEP duration hedge and IJH tilt.

This script keeps the existing timing backbone and rebalance schedule intact,
then runs a controlled experiment:

1. Baseline:
   - STEEP_RISK = 80% IEF + 20% GOLD
2. Add IJH only in STEEP_NON_RISK inverse-vol pool
3. Replace STEEP_RISK IEF with TLT
4. Replace STEEP_RISK IEF with EDV
5. Add IJH and replace with TLT
6. Add IJH and replace with EDV

Assumption made explicit:
- FLAT_NON_RISK uses inverse-vol on SPY / GOLD / CMDTY_FUT.
- INVERTED uses inverse-vol on SPY / GOLD.
- STEEP_NON_RISK baseline is 100% SPY.
- Variants with IJH switch STEEP_NON_RISK to inverse-vol on SPY / IJH.
- FLAT_RISK remains 100% GOLD.
- STEEP_RISK is the only duration-hedge replacement sleeve under test.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIG = {
    "output_dir": Path("results/steep_risk_duration_and_ijh_experiment"),
    "figure_dir": Path("figures/steep_risk_duration_and_ijh_experiment"),
    "one_way_cost_bps": 5,
    "monthly_rebalance": True,
    "risk_window": 120,
    "flat_pool": ["SPY", "GOLD", "CMDTY_FUT"],
    "inverted_pool": ["SPY", "GOLD"],
    "steep_nonrisk_pool_with_ijh": ["SPY", "IJH"],
    "fallback_non_risk": {"SPY": 0.80, "GOLD": 0.0, "CMDTY_FUT": 0.0, "IEF": 0.0, "TLT": 0.0, "EDV": 0.0, "IJH": 0.0, "CASH": 0.20},
    "fallback_risk": {"SPY": 0.0, "GOLD": 0.0, "CMDTY_FUT": 0.0, "IEF": 0.0, "TLT": 0.0, "EDV": 0.0, "IJH": 0.0, "CASH": 1.0},
}

PANEL_CANDIDATES = [
    Path("results/regime_aware_risk_parity_allocation/daily_backtest_panel.csv"),
    Path("results/regime_aware_hedge_allocation_steep_mix/daily_backtest_panel.csv"),
    Path("results/regime_aware_hedge_allocation_steep_test/daily_backtest_panel.csv"),
    Path("results/regime_aware_hedge_allocation_v1/daily_backtest_panel.csv"),
    Path("results/spy_cash_backbone_upgrade_ablation/daily_backtest_panel.csv"),
]

ASSET_PANEL_CANDIDATES = [
    Path("results/regime_aware_risk_parity_allocation/daily_backtest_panel.csv"),
    Path("results/regime_aware_hedge_allocation_steep_mix/daily_backtest_panel.csv"),
    Path("results/regime_aware_hedge_allocation_steep_test/daily_backtest_panel.csv"),
    Path("results/regime_aware_hedge_allocation_v1/daily_backtest_panel.csv"),
    Path("results/reconstructed_regime_asset_behavior/reconstructed_regime_panel.csv"),
]

RAW_ASSET_CANDIDATES = {
    "IJH": [Path("data/raw/asset/IJH.csv")],
    "TLT": [Path("data/raw/asset/TLT.csv")],
    "EDV": [Path("data/raw/asset/EDV.csv")],
}

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

ASSETS = ["SPY", "GOLD", "CMDTY_FUT", "IEF", "TLT", "EDV", "IJH", "CASH"]
EXPERIMENTS = [
    "BASELINE",
    "ADD_IJH_STEEP_NON_RISK",
    "REPLACE_STEEP_RISK_IEF_WITH_TLT",
    "REPLACE_STEEP_RISK_IEF_WITH_EDV",
    "ADD_IJH_AND_REPLACE_WITH_TLT",
    "ADD_IJH_AND_REPLACE_WITH_EDV",
]
ALL_STRATEGIES = [
    "SPY_BUY_HOLD",
    "BACKBONE_V2_SPY_CASH",
    "BASELINE",
    "ADD_IJH_STEEP_NON_RISK",
    "REPLACE_STEEP_RISK_IEF_WITH_TLT",
    "REPLACE_STEEP_RISK_IEF_WITH_EDV",
    "ADD_IJH_AND_REPLACE_WITH_TLT",
    "ADD_IJH_AND_REPLACE_WITH_EDV",
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


def _first_existing(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols = list(cols)
    lower_map = {str(c).lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
    for c in candidates:
        if str(c).lower() in lower_map:
            return lower_map[str(c).lower()]
    return None


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    full = {asset: float(weights.get(asset, 0.0)) for asset in ASSETS}
    total = sum(full.values())
    if total <= 0:
        raise ValueError("Zero-sum weight set.")
    return {asset: weight / total for asset, weight in full.items()}


def _is_first_trading_day_of_month(dates: pd.Series) -> pd.Series:
    periods = dates.dt.to_period("M")
    out = periods.ne(periods.shift(1))
    out.iloc[0] = True
    return out


def _merge_missing(base: pd.DataFrame, other: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in base.columns and c in other.columns]
    if not missing:
        return base
    return base.merge(other[["date"] + missing], on="date", how="left")


def _max_drawdown(ret: pd.Series) -> float:
    nav = (1 + ret.fillna(0.0)).cumprod()
    if nav.empty:
        return np.nan
    return float((nav / nav.cummax() - 1).min())


def _sortino(ret: pd.Series, rf: pd.Series) -> float:
    excess = ret - rf
    downside = excess[excess < 0]
    if downside.empty:
        return np.nan
    downside_std = downside.std(ddof=0)
    if downside_std <= 0:
        return np.nan
    return float(excess.mean() / downside_std * math.sqrt(252))


def _worst_rolling_period(ret: pd.Series, dates: pd.Series, months: int) -> float:
    monthly = (1 + ret.fillna(0.0).set_axis(pd.to_datetime(dates))).resample("ME").prod() - 1
    if monthly.empty:
        return np.nan
    if months == 1:
        return float(monthly.min())
    roll = (1 + monthly).rolling(months).apply(np.prod, raw=True) - 1
    return float(roll.min()) if roll.notna().any() else np.nan


def _annualized_stats(ret: pd.Series, rf: pd.Series, dates: pd.Series) -> Dict[str, float]:
    ret = ret.fillna(0.0)
    rf = rf.reindex(ret.index).fillna(0.0)
    nav = (1 + ret).cumprod()
    years = len(ret) / 252
    ann = nav.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = ret.std(ddof=0) * math.sqrt(252)
    excess = ret - rf
    sharpe = excess.mean() / excess.std(ddof=0) * math.sqrt(252) if excess.std(ddof=0) > 0 else np.nan
    sortino = _sortino(ret, rf)
    maxdd = _max_drawdown(ret)
    calmar = ann / abs(maxdd) if pd.notna(maxdd) and maxdd < 0 else np.nan
    return {
        "AnnRet": ann,
        "AnnVol": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": maxdd,
        "Calmar": calmar,
        "Worst Month": _worst_rolling_period(ret, dates, 1),
        "Worst 3M": _worst_rolling_period(ret, dates, 3),
        "Worst 12M": _worst_rolling_period(ret, dates, 12),
        "Final NAV": nav.iloc[-1],
    }


def load_base_panel() -> pd.DataFrame:
    frames = []
    for path in PANEL_CANDIDATES:
        if path.exists():
            frames.append((path, _read_csv(path)))
    if not frames:
        raise FileNotFoundError("No candidate base panel found.")
    panel = frames[0][1].copy()
    print(f"Loaded base panel: {frames[0][0]}")
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
        "timing_state",
        "cross_state",
        "entry_reason",
        "BACKBONE_V2_SPY_CASH_weight_SPY",
        "BACKBONE_V2_SPY_CASH_weight_CASH",
        "BACKBONE_V2_SPY_CASH_return",
        "BACKBONE_V2_SPY_CASH_nav",
        "SPY_return",
        "GOLD_return",
        "IEF_return",
        "CASH_return",
        "CMDTY_FUT_return",
    ]
    for _, df in frames[1:]:
        panel = _merge_missing(panel, df, needed)
    if "spy_drawdown_from_previous_high" not in panel.columns:
        panel["spy_drawdown_from_previous_high"] = panel["spy_price"] / panel["spy_price"].cummax() - 1
    if "VIX_ZSCORE_120D" not in panel.columns:
        roll = panel["VIX_LEVEL"].rolling(CONFIG["risk_window"], min_periods=CONFIG["risk_window"])
        panel["VIX_ZSCORE_120D"] = (panel["VIX_LEVEL"] - roll.mean()) / roll.std(ddof=0)
    if "D_CREDIT_SPREAD_20D" not in panel.columns:
        panel["D_CREDIT_SPREAD_20D"] = panel["CREDIT_SPREAD_BAA_AAA"] - panel["CREDIT_SPREAD_BAA_AAA"].shift(20)
    if "SPY_MA20" not in panel.columns:
        panel["SPY_MA20"] = panel["spy_price"].rolling(20, min_periods=20).mean()
    if "SPY_CROSS_ABOVE_MA20" not in panel.columns:
        panel["SPY_CROSS_ABOVE_MA20"] = (panel["spy_price"] > panel["SPY_MA20"]) & (
            panel["spy_price"].shift(1) <= panel["SPY_MA20"].shift(1)
        )
    panel["daily_rf"] = pd.to_numeric(panel["daily_rf"], errors="coerce").fillna(0.0)
    panel["macro_regime_confirmed"] = panel["macro_regime_confirmed"].fillna("NEUTRAL").astype(str)
    panel["monthly_either_state"] = panel["monthly_either_state"].fillna("UNKNOWN").astype(str)
    return panel


def _load_raw_asset_return(path: Path, asset: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    raw = pd.read_csv(path, na_values=[".", "NA", ""])
    date_col = _first_existing(raw.columns, ["date", "DATE", "Date", "observation_date"])
    if not date_col:
        return None
    price_col = _first_existing(raw.columns, [asset, f"{asset}_price", "Adj Close", "adj_close", "Close", "close", "VALUE", "value"])
    if not price_col:
        numeric_cols = [c for c in raw.columns if c != date_col and pd.to_numeric(raw[c], errors="coerce").notna().any()]
        if not numeric_cols:
            return None
        price_col = numeric_cols[0]
    out = raw[[date_col, price_col]].rename(columns={date_col: "date", price_col: f"{asset}_price"})
    out["date"] = pd.to_datetime(out["date"])
    out[f"{asset}_price"] = pd.to_numeric(out[f"{asset}_price"], errors="coerce")
    out[f"{asset}_return"] = out[f"{asset}_price"].pct_change(fill_method=None)
    return out[["date", f"{asset}_return"]].sort_values("date").drop_duplicates("date")


def load_asset_returns(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    if "SPY_return" not in out.columns:
        out["SPY_return"] = pd.to_numeric(out["spy_daily_return"], errors="coerce")
    if "CASH_return" not in out.columns:
        out["CASH_return"] = pd.to_numeric(out["daily_rf"], errors="coerce").fillna(0.0)

    mapping = {
        "GOLD_return": ["GOLD_return", "GOLD_RETURN", "GLD_return", "GLD_RETURN"],
        "IEF_return": ["IEF_return", "IEF_RETURN"],
        "CMDTY_FUT_return": ["CMDTY_FUT_return", "CMDTY_FUT_RETURN", "CMDTY_return", "CMDTY_RETURN", "CMDTY_ret"],
        "IJH_return": ["IJH_return", "IJH_RETURN"],
        "TLT_return": ["TLT_return", "TLT_RETURN"],
        "EDV_return": ["EDV_return", "EDV_RETURN"],
    }
    for path in ASSET_PANEL_CANDIDATES:
        if not path.exists():
            continue
        src = _read_csv(path)
        rename_map = {}
        for final, candidates in mapping.items():
            if final in out.columns:
                continue
            col = _first_existing(src.columns, candidates)
            if col:
                rename_map[col] = final
        if rename_map:
            out = out.merge(src[["date"] + list(rename_map.keys())].rename(columns=rename_map), on="date", how="left")

    for asset, paths in RAW_ASSET_CANDIDATES.items():
        final = f"{asset}_return"
        if final in out.columns:
            continue
        loaded = None
        for path in paths:
            loaded = _load_raw_asset_return(path, asset)
            if loaded is not None:
                break
        if loaded is not None:
            out = out.merge(loaded, on="date", how="left")

    required = ["SPY_return", "GOLD_return", "IEF_return", "CASH_return", "CMDTY_FUT_return", "IJH_return", "TLT_return", "EDV_return"]
    for col in required:
        if col not in out.columns:
            raise ValueError(f"Missing required asset return column: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce")

    valid = out[required].notna().all(axis=1)
    if not valid.any():
        raise ValueError("No overlapping sample across required assets.")
    start_idx = int(valid.idxmax())
    out = out.loc[start_idx:].reset_index(drop=True)

    for asset in ASSETS:
        out[f"{asset}_nav"] = (1 + out[f"{asset}_return"].fillna(0.0)).cumprod()
    return out


def build_backbone_v2_state(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    if "BACKBONE_V2_SPY_CASH_weight_SPY" in df.columns:
        state = np.where(pd.to_numeric(df["BACKBONE_V2_SPY_CASH_weight_SPY"], errors="coerce").fillna(1.0) >= 0.5, "NON_RISK", "RISK")
        df["timing_state"] = df.get("timing_state", pd.Series(state)).fillna(pd.Series(state))
        df["BACKBONE_V2_SPY_CASH_weight_CASH"] = pd.to_numeric(df.get("BACKBONE_V2_SPY_CASH_weight_CASH"), errors="coerce").fillna(1 - pd.to_numeric(df["BACKBONE_V2_SPY_CASH_weight_SPY"], errors="coerce").fillna(1.0))
    elif "BACKBONE_V2_UPGRADED_weight_spy" in df.columns:
        spy_w = pd.to_numeric(df["BACKBONE_V2_UPGRADED_weight_spy"], errors="coerce").fillna(1.0)
        df["BACKBONE_V2_SPY_CASH_weight_SPY"] = spy_w
        df["BACKBONE_V2_SPY_CASH_weight_CASH"] = 1 - spy_w
        df["timing_state"] = np.where(spy_w >= 0.5, "NON_RISK", "RISK")
        if "BACKBONE_V2_SPY_CASH_return" not in df.columns and "BACKBONE_V2_UPGRADED_return" in df.columns:
            df["BACKBONE_V2_SPY_CASH_return"] = pd.to_numeric(df["BACKBONE_V2_UPGRADED_return"], errors="coerce").fillna(0.0)
        if "BACKBONE_V2_SPY_CASH_nav" not in df.columns and "BACKBONE_V2_SPY_CASH_return" in df.columns:
            df["BACKBONE_V2_SPY_CASH_nav"] = (1 + df["BACKBONE_V2_SPY_CASH_return"]).cumprod()
    else:
        raise ValueError("Missing BACKBONE_V2 weights.")
    df["cross_state"] = df["macro_regime_confirmed"] + "_" + df["timing_state"]
    df["entry_reason"] = df.get("entry_reason", "").fillna("")
    return df


def compute_inverse_vol_weights(window_df: pd.DataFrame, pool: List[str]) -> Dict[str, float]:
    ret_cols = [f"{asset}_return" for asset in pool]
    subset = window_df[ret_cols].copy()
    subset.columns = pool
    vols = subset.std(ddof=0) * math.sqrt(252)
    valid = vols.replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) != len(pool) or (valid <= 0).any():
        return _normalize_weights({pool[0]: 1.0})
    raw = 1.0 / valid
    raw = raw / raw.sum()
    return _normalize_weights({asset: raw.get(asset, 0.0) for asset in pool})


def build_target_weights(panel: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    first_month = _is_first_trading_day_of_month(panel["date"])
    strategy_rows = {strategy: [] for strategy in ALL_STRATEGIES}

    last_weights = {
        "flat_inv": _normalize_weights({"SPY": 0.60, "GOLD": 0.30, "CMDTY_FUT": 0.10}),
        "inverted_inv": _normalize_weights({"SPY": 0.75, "GOLD": 0.25}),
        "steep_nonrisk_ijh": _normalize_weights({"SPY": 0.80, "IJH": 0.20}),
    }

    def weights_for_strategy(regime: str, timing_state: str, strategy: str) -> Dict[str, float]:
        if strategy == "SPY_BUY_HOLD":
            return _normalize_weights({"SPY": 1.0})
        if strategy == "BACKBONE_V2_SPY_CASH":
            return _normalize_weights({"SPY": 1.0} if timing_state == "NON_RISK" else {"CASH": 1.0})

        steep_nonrisk_with_ijh = strategy in {
            "ADD_IJH_STEEP_NON_RISK",
            "ADD_IJH_AND_REPLACE_WITH_TLT",
            "ADD_IJH_AND_REPLACE_WITH_EDV",
        }
        steep_risk_asset = "IEF"
        if strategy in {"REPLACE_STEEP_RISK_IEF_WITH_TLT", "ADD_IJH_AND_REPLACE_WITH_TLT"}:
            steep_risk_asset = "TLT"
        elif strategy in {"REPLACE_STEEP_RISK_IEF_WITH_EDV", "ADD_IJH_AND_REPLACE_WITH_EDV"}:
            steep_risk_asset = "EDV"

        if regime == "FLAT":
            if timing_state == "RISK":
                return _normalize_weights({"GOLD": 1.0})
            return last_weights["flat_inv"]
        if regime == "INVERTED":
            return last_weights["inverted_inv"]
        if regime == "STEEP":
            if timing_state == "RISK":
                return _normalize_weights({steep_risk_asset: 0.80, "GOLD": 0.20})
            if steep_nonrisk_with_ijh:
                return last_weights["steep_nonrisk_ijh"]
            return _normalize_weights({"SPY": 1.0})
        return _normalize_weights(CONFIG["fallback_risk"] if timing_state == "RISK" else CONFIG["fallback_non_risk"])

    prev_cross_state = None
    for i, row in panel.iterrows():
        regime = str(row["macro_regime_confirmed"])
        timing_state = str(row["timing_state"])
        cross_state = f"{regime}_{timing_state}"
        recalc = bool(first_month.iloc[i]) or (i > 0 and cross_state != prev_cross_state)
        prev_cross_state = cross_state
        if recalc and i >= CONFIG["risk_window"] - 1:
            flat_window = panel.loc[i - CONFIG["risk_window"] + 1 : i]
            last_weights["flat_inv"] = compute_inverse_vol_weights(flat_window, CONFIG["flat_pool"])
            last_weights["inverted_inv"] = compute_inverse_vol_weights(flat_window, CONFIG["inverted_pool"])
            last_weights["steep_nonrisk_ijh"] = compute_inverse_vol_weights(flat_window, CONFIG["steep_nonrisk_pool_with_ijh"])
        for strategy in ALL_STRATEGIES:
            strategy_rows[strategy].append(weights_for_strategy(regime, timing_state, strategy))
    return {strategy: pd.DataFrame(rows) for strategy, rows in strategy_rows.items()}


def run_multi_asset_backtest(panel: pd.DataFrame, strategy: str, target_weights: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    tw = target_weights[ASSETS].copy().fillna(0.0)
    tw = tw.div(tw.sum(axis=1), axis=0).fillna(0.0)
    first_month = _is_first_trading_day_of_month(df["date"])
    current = tw.iloc[0].to_dict()
    prev_target = tw.iloc[0].to_dict()
    nav = 1.0
    for asset in ASSETS:
        df[f"{strategy}_weight_{asset}"] = np.nan
    df[f"{strategy}_return"] = np.nan
    df[f"{strategy}_nav"] = np.nan
    df[f"{strategy}_turnover"] = np.nan
    df[f"{strategy}_transaction_cost"] = np.nan
    for i in range(len(df)):
        desired = tw.iloc[i].to_dict()
        changed = any(abs(desired[a] - prev_target[a]) > 1e-12 for a in ASSETS)
        turnover = 0.0
        cost = 0.0
        if i > 0 and (changed or bool(first_month.iloc[i])):
            turnover = float(sum(abs(desired[a] - current[a]) for a in ASSETS))
            cost = 0.5 * turnover * CONFIG["one_way_cost_bps"] / 10000
            current = desired.copy()
        gross = float(sum(current[a] * df.iloc[i][f"{a}_return"] for a in ASSETS))
        net = gross - cost
        nav *= 1 + net
        for asset in ASSETS:
            df.loc[i, f"{strategy}_weight_{asset}"] = current[asset]
        df.loc[i, f"{strategy}_return"] = net
        df.loc[i, f"{strategy}_nav"] = nav
        df.loc[i, f"{strategy}_turnover"] = turnover
        df.loc[i, f"{strategy}_transaction_cost"] = cost
        denom = 1 + gross
        if denom != 0:
            current = {a: current[a] * (1 + df.iloc[i][f"{a}_return"]) / denom for a in ASSETS}
        prev_target = desired.copy()
    return df


def compute_performance_metrics(panel: pd.DataFrame, strategy: str) -> Dict[str, object]:
    ret = panel[f"{strategy}_return"].fillna(0.0)
    rf = panel["CASH_return"].fillna(0.0)
    stats = _annualized_stats(ret, rf, panel["date"])
    return {
        "strategy": strategy,
        **stats,
    }


def compute_regime_allocation_summary(panel: pd.DataFrame, strategies: List[str]) -> pd.DataFrame:
    rows = []
    for strategy in strategies:
        for cross_state, sub in panel.groupby("cross_state", dropna=False):
            row = {"strategy": strategy, "cross_state": cross_state, "n_obs": len(sub)}
            for asset in ASSETS:
                row[f"avg_weight_{asset}"] = sub[f"{strategy}_weight_{asset}"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def compute_regime_contribution_summary(panel: pd.DataFrame, strategies: List[str]) -> pd.DataFrame:
    rows = []
    for strategy in strategies:
        for cross_state, sub in panel.groupby("cross_state", dropna=False):
            row = {"strategy": strategy, "cross_state": cross_state, "n_obs": len(sub)}
            port_ret = sub[f"{strategy}_return"].fillna(0.0)
            cov = sub[[f"{asset}_return" for asset in ASSETS]].cov() * 252
            avg_w = np.array([sub[f"{strategy}_weight_{asset}"].mean() for asset in ASSETS], dtype=float)
            sigma2 = float(avg_w @ cov.to_numpy() @ avg_w) if len(sub) > 1 else np.nan
            for asset in ASSETS:
                contrib = (sub[f"{strategy}_weight_{asset}"] * sub[f"{asset}_return"]).fillna(0.0)
                row[f"return_contribution_{asset}"] = contrib.sum()
            if np.isfinite(sigma2) and sigma2 > 0:
                m = cov.to_numpy() @ avg_w
                rc = avg_w * m / sigma2
                for asset, val in zip(ASSETS, rc):
                    row[f"risk_contribution_{asset}"] = float(val)
            else:
                for asset in ASSETS:
                    row[f"risk_contribution_{asset}"] = np.nan
            row["portfolio_cumulative_return"] = (1 + port_ret).prod() - 1
            rows.append(row)
    return pd.DataFrame(rows)


def compute_difference_vs_baseline(perf: pd.DataFrame) -> pd.DataFrame:
    base = perf.loc[perf["strategy"].eq("BASELINE")].iloc[0]
    variants = perf[perf["strategy"].isin(EXPERIMENTS[1:])].copy()
    variants["ΔAnnRet"] = variants["AnnRet"] - base["AnnRet"]
    variants["ΔSharpe"] = variants["Sharpe"] - base["Sharpe"]
    variants["ΔMaxDD"] = variants["MaxDD"] - base["MaxDD"]
    variants["ΔCalmar"] = variants["Calmar"] - base["Calmar"]
    variants["ΔFinal NAV"] = variants["Final NAV"] - base["Final NAV"]
    return variants[["strategy", "ΔAnnRet", "ΔSharpe", "ΔMaxDD", "ΔCalmar", "ΔFinal NAV"]]


def compute_crisis_performance(panel: pd.DataFrame, strategies: List[str]) -> pd.DataFrame:
    rows = []
    for period, (start, end) in CRISIS_WINDOWS.items():
        sub = panel[(panel["date"] >= pd.Timestamp(start)) & (panel["date"] <= pd.Timestamp(end))].copy()
        if sub.empty:
            continue
        for strategy in strategies:
            ret = sub[f"{strategy}_return"].fillna(0.0)
            rf = sub["CASH_return"].fillna(0.0)
            stats = _annualized_stats(ret, rf, sub["date"])
            rows.append({"period": period, "strategy": strategy, **stats})
    return pd.DataFrame(rows)


def plot_navs(panel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    for strategy in ["BASELINE"] + EXPERIMENTS[1:]:
        ax.plot(panel["date"], panel[f"{strategy}_nav"], label=strategy)
    ax.set_yscale("log")
    ax.set_title("NAV Comparison")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "nav_comparison.png", dpi=150)
    plt.close(fig)


def plot_drawdowns(panel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    for strategy in ["BASELINE"] + EXPERIMENTS[1:]:
        nav = panel[f"{strategy}_nav"]
        ax.plot(panel["date"], nav / nav.cummax() - 1, label=strategy)
    ax.set_title("Drawdown Comparison")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "drawdown_comparison.png", dpi=150)
    plt.close(fig)


def write_report(perf: pd.DataFrame, diff: pd.DataFrame, alloc: pd.DataFrame, contrib: pd.DataFrame) -> None:
    out = CONFIG["output_dir"] / "STEEP_RISK_DURATION_AND_IJH_EXPERIMENT.md"
    content = f"""# steep_risk_duration_and_ijh_experiment

## Purpose

Controlled marginal test only. This does not change regime definition, sample window, rebalance frequency, or parameter tuning.

## Baseline

- FLAT_NON_RISK: inverse-vol on SPY / GOLD / CMDTY_FUT
- FLAT_RISK: 100% GOLD
- INVERTED: inverse-vol on SPY / GOLD
- STEEP_NON_RISK baseline inverse-vol pool: SPY / GOLD
- STEEP_RISK baseline: 80% IEF / 20% GOLD
- Fallback NON_RISK: 80% SPY / 20% CASH
- Fallback RISK: 100% CASH

## Full-period Performance

{perf.to_markdown(index=False)}

## Difference vs Baseline

{diff.to_markdown(index=False)}

## Regime Allocation Summary (sample)

{alloc.head(24).to_markdown(index=False)}

## Regime Contribution Summary (sample)

{contrib.head(24).to_markdown(index=False)}
"""
    out.write_text(content, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = load_base_panel()
    panel = load_asset_returns(panel)
    panel = build_backbone_v2_state(panel)
    targets = build_target_weights(panel)
    for strategy in ALL_STRATEGIES:
        panel = run_multi_asset_backtest(panel, strategy, targets[strategy])

    perf = pd.DataFrame([compute_performance_metrics(panel, strategy) for strategy in ALL_STRATEGIES])
    diff = compute_difference_vs_baseline(perf)
    crisis = compute_crisis_performance(panel, ALL_STRATEGIES)
    alloc = compute_regime_allocation_summary(panel, ["BASELINE"] + EXPERIMENTS[1:])
    contrib = compute_regime_contribution_summary(panel, ["BASELINE"] + EXPERIMENTS[1:])

    panel.to_csv(CONFIG["output_dir"] / "daily_backtest_panel.csv", index=False)
    perf.to_csv(CONFIG["output_dir"] / "performance_summary.csv", index=False)
    diff.to_csv(CONFIG["output_dir"] / "difference_vs_baseline.csv", index=False)
    alloc.to_csv(CONFIG["output_dir"] / "regime_allocation_summary.csv", index=False)
    contrib.to_csv(CONFIG["output_dir"] / "regime_contribution_summary.csv", index=False)
    crisis.to_csv(CONFIG["output_dir"] / "crisis_performance.csv", index=False)

    plot_navs(panel)
    plot_drawdowns(panel)
    write_report(perf, diff, alloc, contrib)


if __name__ == "__main__":
    main()
