"""Final mature regime hedge strategy and full-project backtest summary."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIG = {
    "output_dir": Path("results/mature_regime_hedge_final"),
    "figure_dir": Path("figures/mature_regime_hedge_final"),
    "one_way_cost_bps": 5,
    "risk_window": 120,
}

BASE_CANDIDATES = [
    Path("results/regime_aware_risk_parity_allocation/daily_backtest_panel.csv"),
    Path("results/mature_steep_cmdty_overlay_50spy50ief/daily_backtest_panel.csv"),
    Path("results/mature_strategy_with_steep_commodity_overlay/daily_backtest_panel.csv"),
    Path("results/spy_cash_backbone_upgrade_ablation/daily_backtest_panel.csv"),
]

ENRICH_CANDIDATES = [
    Path("results/backbone_v2_with_steep_commodity_stress/daily_backtest_panel.csv"),
    Path("results/mature_steep_cmdty_overlay_50spy50ief/daily_backtest_panel.csv"),
    Path("results/flat_risk_gold_cash_diagnostic/flat_risk_variant_daily_panel.csv"),
]

ASSETS = ["SPY", "GOLD", "CMDTY_FUT", "IEF", "CASH"]
FINAL_STRATEGY = "MATURE_REGIME_HEDGE_FINAL"
BENCHMARKS = [
    "SPY_BUY_HOLD",
    "BACKBONE_V2_SPY_CASH",
    "REGIME_HEDGE_V1_ORIGINAL",
    "MATURE_BASELINE_REGIME_HEDGE_INV_VOL",
    "MATURE_FULL_ONE_RET60",
    "MATURE_STEEP_CMDTY_OVERLAY_50SPY_50IEF",
    "FLAT_RISK_50GOLD_50CASH",
    FINAL_STRATEGY,
    "FLAT_RISK_30GOLD_70CASH",
]
CRISIS_WINDOWS = {
    "2008_GFC": ("2007-10-01", "2009-06-30"),
    "2011_EURO_DEBT": ("2011-06-01", "2011-12-31"),
    "2015_2016": ("2015-05-01", "2016-03-31"),
    "2018Q4": ("2018-10-01", "2019-01-31"),
    "COVID_2020": ("2020-02-01", "2020-06-30"),
    "2022_RATE_WAR": ("2021-11-01", "2023-03-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2025_PULLBACK": ("2025-01-01", "2025-12-31"),
    "2024_2026": ("2024-01-01", "2026-12-31"),
}


def ensure_dirs() -> None:
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["figure_dir"].mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"Missing date in {path}")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def _first_existing(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def load_validated_panel() -> tuple[pd.DataFrame, Path]:
    base_path = next((p for p in BASE_CANDIDATES if p.exists()), None)
    if base_path is None:
        raise FileNotFoundError("No validated mature baseline panel found.")
    df = _read_csv(base_path)

    # Enrich with richer strategy/output panels without overriding validated base values.
    for enrich_path in ENRICH_CANDIDATES:
        if not enrich_path.exists():
            continue
        enrich = _read_csv(enrich_path).set_index("date")
        base_index = df.set_index("date")
        for col in enrich.columns:
            if col == "date":
                continue
            if col not in base_index.columns:
                base_index[col] = enrich[col]
            else:
                if base_index[col].isna().all():
                    base_index[col] = enrich[col]
        df = base_index.reset_index()

    # Alias mature baseline from validated risk-parity naming if needed.
    alias_pairs = {
        "MATURE_BASELINE_REGIME_HEDGE_INV_VOL_weight_SPY": "REGIME_HEDGE_INV_VOL_weight_SPY",
        "MATURE_BASELINE_REGIME_HEDGE_INV_VOL_weight_GOLD": "REGIME_HEDGE_INV_VOL_weight_GOLD",
        "MATURE_BASELINE_REGIME_HEDGE_INV_VOL_weight_CMDTY_FUT": "REGIME_HEDGE_INV_VOL_weight_CMDTY_FUT",
        "MATURE_BASELINE_REGIME_HEDGE_INV_VOL_weight_IEF": "REGIME_HEDGE_INV_VOL_weight_IEF",
        "MATURE_BASELINE_REGIME_HEDGE_INV_VOL_weight_CASH": "REGIME_HEDGE_INV_VOL_weight_CASH",
        "MATURE_BASELINE_REGIME_HEDGE_INV_VOL_return": "REGIME_HEDGE_INV_VOL_return",
        "MATURE_BASELINE_REGIME_HEDGE_INV_VOL_nav": "REGIME_HEDGE_INV_VOL_nav",
        "MATURE_BASELINE_REGIME_HEDGE_INV_VOL_turnover": "turnover_REGIME_HEDGE_INV_VOL",
        "MATURE_BASELINE_REGIME_HEDGE_INV_VOL_transaction_cost": "transaction_cost_REGIME_HEDGE_INV_VOL",
    }
    for new, old in alias_pairs.items():
        if new not in df.columns and old in df.columns:
            df[new] = df[old]
    if "MATURE_BASELINE_REGIME_HEDGE_INV_VOL_risk_state" not in df.columns:
        if "timing_state" in df.columns:
            df["MATURE_BASELINE_REGIME_HEDGE_INV_VOL_risk_state"] = np.where(
                df["timing_state"].astype(str).eq("RISK"),
                "FULL_RISK",
                "NON_RISK",
            )
        else:
            raise ValueError("Missing validated mature baseline risk state.")
    if "MATURE_BASELINE_REGIME_HEDGE_INV_VOL_overlay_state" not in df.columns:
        df["MATURE_BASELINE_REGIME_HEDGE_INV_VOL_overlay_state"] = False
    if "MATURE_BASELINE_REGIME_HEDGE_INV_VOL_full_risk_state" not in df.columns:
        df["MATURE_BASELINE_REGIME_HEDGE_INV_VOL_full_risk_state"] = df["MATURE_BASELINE_REGIME_HEDGE_INV_VOL_risk_state"].eq("FULL_RISK")

    # Required core fields
    reqs = {
        "SPY_return": ["SPY_return", "spy_daily_return"],
        "GOLD_return": ["GOLD_return", "GLD_return"],
        "CMDTY_FUT_return": ["CMDTY_FUT_return"],
        "IEF_return": ["IEF_return"],
        "CASH_return": ["CASH_return", "daily_rf"],
        "daily_rf": ["daily_rf", "CASH_return"],
        "macro_regime_confirmed": ["macro_regime_confirmed"],
        "monthly_either_state": ["monthly_either_state"],
        "VIX_LEVEL": ["VIX_LEVEL"],
        "VIX_ZSCORE_120D": ["VIX_ZSCORE_120D"],
        "CREDIT_SPREAD_BAA_AAA": ["CREDIT_SPREAD_BAA_AAA"],
        "D_CREDIT_SPREAD_20D": ["D_CREDIT_SPREAD_20D"],
        "spy_drawdown_from_previous_high": ["spy_drawdown_from_previous_high"],
        "SPY_MA20": ["SPY_MA20"],
        "SPY_CROSS_ABOVE_MA20": ["SPY_CROSS_ABOVE_MA20"],
        "CMDTY_RET60": ["CMDTY_RET60"],
        "spy_price": ["spy_price"],
    }
    for out_name, candidates in reqs.items():
        col = _first_existing(df, candidates)
        if col is None:
            raise ValueError(f"Missing required field for {out_name}: {candidates}")
        if out_name != col:
            df[out_name] = df[col]
    # Coerce numerics
    for col in [
        "SPY_return", "GOLD_return", "CMDTY_FUT_return", "IEF_return", "CASH_return", "daily_rf",
        "VIX_LEVEL", "VIX_ZSCORE_120D", "CREDIT_SPREAD_BAA_AAA", "D_CREDIT_SPREAD_20D",
        "spy_drawdown_from_previous_high", "SPY_MA20", "CMDTY_RET60", "spy_price",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["SPY_CROSS_ABOVE_MA20"] = df["SPY_CROSS_ABOVE_MA20"].fillna(False).astype(bool)
    df["SPY_return"] = df["SPY_return"].fillna(0.0)
    df["GOLD_return"] = df["GOLD_return"].fillna(0.0)
    df["CMDTY_FUT_return"] = df["CMDTY_FUT_return"].fillna(0.0)
    df["IEF_return"] = df["IEF_return"].fillna(0.0)
    df["CASH_return"] = df["CASH_return"].fillna(0.0)
    df["daily_rf"] = df["daily_rf"].fillna(df["CASH_return"])
    print(f"Loaded validated panel: {base_path}")
    return df, base_path


def validate_regime_universe(df: pd.DataFrame) -> pd.DataFrame:
    abnormal = df.loc[~df["macro_regime_confirmed"].isin(["FLAT", "STEEP", "INVERTED"]), ["date", "macro_regime_confirmed"]].copy()
    if not abnormal.empty:
        abnormal.to_csv(CONFIG["output_dir"] / "unexpected_regime_dates.csv", index=False)
        warnings.warn(f"Unexpected regime values found on {len(abnormal)} dates.")
    return abnormal


def compute_inverse_vol_weights(df: pd.DataFrame) -> pd.DataFrame:
    # Use already validated weights; just verify they exist and look sane.
    checks = []
    for prefix, assets in {
        "FLAT": ["SPY", "GOLD", "CMDTY_FUT"],
        "INVERTED": ["SPY", "GOLD"],
    }.items():
        cols = [f"{prefix}_INV_VOL_weight_{a}" for a in assets]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            warnings.warn(f"Missing inverse-vol weight columns: {missing}")
            continue
        sums = df[cols].sum(axis=1)
        checks.append(pd.DataFrame({"date": df["date"], "module": prefix, "weight_sum": sums}))
    if checks:
        pd.concat(checks, ignore_index=True).to_csv(CONFIG["output_dir"] / "inverse_vol_weight_checks.csv", index=False)
    return df


def build_or_load_backbone_state(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timing_state"] = np.where(out["MATURE_BASELINE_REGIME_HEDGE_INV_VOL_risk_state"].eq("FULL_RISK"), "RISK", "NON_RISK")
    out["FLAT_VIX_STRESS"] = out["macro_regime_confirmed"].eq("FLAT") & (out["VIX_ZSCORE_120D"] >= 3.0)
    out["FLAT_CREDIT_DD5_STRESS"] = out["macro_regime_confirmed"].eq("FLAT") & (
        (out["spy_drawdown_from_previous_high"] <= -0.05) & (out["D_CREDIT_SPREAD_20D"] > 0.10)
    )
    out["STEEP_EITHER_SELL_STRESS"] = out["macro_regime_confirmed"].eq("STEEP") & out["monthly_either_state"].eq("SELL")
    out["STEEP_CREDIT_DD5_STRESS"] = out["macro_regime_confirmed"].eq("STEEP") & (
        (out["spy_drawdown_from_previous_high"] <= -0.05) & (out["D_CREDIT_SPREAD_20D"] > 0.10)
    )
    out["STEEP_CMDTY_RET60_NEG10"] = out["macro_regime_confirmed"].eq("STEEP") & (out["CMDTY_RET60"] < -0.10)
    out["BACKBONE_V2_ENTRY_SIGNAL"] = (
        out["FLAT_VIX_STRESS"] | out["FLAT_CREDIT_DD5_STRESS"] | out["STEEP_EITHER_SELL_STRESS"] | out["STEEP_CREDIT_DD5_STRESS"]
    )
    out["R3_RECOVERY"] = out["SPY_CROSS_ABOVE_MA20"]
    return out


def _baseline_weights(row: pd.Series) -> Dict[str, float]:
    return {
        a: float(pd.to_numeric(row.get(f"MATURE_BASELINE_REGIME_HEDGE_INV_VOL_weight_{a}", 0.0), errors="coerce") or 0.0)
        for a in ASSETS
    }


def build_final_strategy_states(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    overlay_signal = (
        out["macro_regime_confirmed"].eq("STEEP")
        & (out["CMDTY_RET60"] < -0.10)
        & out["MATURE_BASELINE_REGIME_HEDGE_INV_VOL_risk_state"].ne("FULL_RISK")
    )
    next_overlay = np.zeros(len(out), dtype=bool)
    state = []
    entry_reason = [""] * len(out)
    weights = []
    turnover = np.zeros(len(out))
    tcost = np.zeros(len(out))
    strategy_ret = np.zeros(len(out))
    prev_weights = _baseline_weights(out.iloc[0])
    for i, row in out.iterrows():
        base_state = str(row["MATURE_BASELINE_REGIME_HEDGE_INV_VOL_risk_state"])
        regime = str(row["macro_regime_confirmed"])
        if base_state == "FULL_RISK":
            if regime == "FLAT":
                cur_state = "FULL_RISK"
                cur_weights = {"SPY": 0.0, "GOLD": 0.5, "CMDTY_FUT": 0.0, "IEF": 0.0, "CASH": 0.5}
            elif regime == "STEEP":
                cur_state = "FULL_RISK"
                cur_weights = {"SPY": 0.0, "GOLD": 0.0, "CMDTY_FUT": 0.0, "IEF": 1.0, "CASH": 0.0}
            else:  # INVERTED should not happen under validated backbone
                cur_state = "NON_RISK"
                cur_weights = _baseline_weights(row)
        elif next_overlay[i]:
            cur_state = "SLOW_GROWTH_OVERLAY"
            cur_weights = {"SPY": 0.5, "GOLD": 0.0, "CMDTY_FUT": 0.0, "IEF": 0.5, "CASH": 0.0}
        else:
            cur_state = "NON_RISK"
            cur_weights = _baseline_weights(row)
        state.append(cur_state)
        weights.append(cur_weights)
        if i > 0:
            tw = sum(abs(cur_weights[a] - prev_weights[a]) for a in ASSETS)
            turnover[i] = tw
            tcost[i] = 0.5 * tw * CONFIG["one_way_cost_bps"] / 10000.0
        prev_weights = cur_weights
        strategy_ret[i] = sum(cur_weights[a] * float(row[f"{a}_return"]) for a in ASSETS) - tcost[i]
        if i + 1 < len(out):
            next_base_state = str(out.iloc[i + 1]["MATURE_BASELINE_REGIME_HEDGE_INV_VOL_risk_state"])
            if next_base_state == "FULL_RISK":
                next_overlay[i + 1] = False
            else:
                if cur_state == "SLOW_GROWTH_OVERLAY":
                    next_overlay[i + 1] = not bool(row["R3_RECOVERY"])
                else:
                    next_overlay[i + 1] = bool(overlay_signal.iloc[i])
                    if next_overlay[i + 1]:
                        entry_reason[i + 1] = "STEEP_CMDTY_RET60_NEG10"

    out["overlay_state"] = np.array(state) == "SLOW_GROWTH_OVERLAY"
    out["final_state"] = state
    out[f"{FINAL_STRATEGY}_state"] = state
    out[f"{FINAL_STRATEGY}_overlay_state"] = out["overlay_state"]
    out[f"{FINAL_STRATEGY}_full_risk_state"] = np.array(state) == "FULL_RISK"
    out[f"{FINAL_STRATEGY}_entry_reason"] = entry_reason
    out[f"{FINAL_STRATEGY}_return"] = strategy_ret
    out[f"{FINAL_STRATEGY}_nav"] = (1 + out[f"{FINAL_STRATEGY}_return"].fillna(0.0)).cumprod()
    out[f"{FINAL_STRATEGY}_turnover"] = turnover
    out[f"{FINAL_STRATEGY}_transaction_cost"] = tcost
    for a in ASSETS:
        out[f"{FINAL_STRATEGY}_weight_{a}"] = [w[a] for w in weights]

    # Bring in flat-risk 50/50 reference if available; else build from baseline on the fly.
    if "FLAT_RISK_50GOLD_50CASH_return" not in out.columns:
        mask = out["MATURE_BASELINE_REGIME_HEDGE_INV_VOL_risk_state"].eq("FULL_RISK") & out["macro_regime_confirmed"].eq("FLAT")
        w_hist = []
        prev = _baseline_weights(out.iloc[0])
        ret = np.zeros(len(out))
        tw = np.zeros(len(out))
        tc = np.zeros(len(out))
        for i, row in out.iterrows():
            w = _baseline_weights(row)
            if mask.iloc[i]:
                w = {"SPY": 0.0, "GOLD": 0.5, "CMDTY_FUT": 0.0, "IEF": 0.0, "CASH": 0.5}
            w_hist.append(w)
            if i > 0:
                tw[i] = sum(abs(w[a] - prev[a]) for a in ASSETS)
                tc[i] = 0.5 * tw[i] * CONFIG["one_way_cost_bps"] / 10000.0
            ret[i] = sum(w[a] * float(row[f"{a}_return"]) for a in ASSETS) - tc[i]
            prev = w
        out["FLAT_RISK_50GOLD_50CASH_return"] = ret
        out["FLAT_RISK_50GOLD_50CASH_nav"] = np.cumprod(1 + ret)
        out["FLAT_RISK_50GOLD_50CASH_turnover"] = tw
        out["FLAT_RISK_50GOLD_50CASH_transaction_cost"] = tc
        for a in ASSETS:
            out[f"FLAT_RISK_50GOLD_50CASH_weight_{a}"] = [w[a] for w in w_hist]
    if "FLAT_RISK_30GOLD_70CASH_return" not in out.columns:
        pass
    return out


def _state_series(df: pd.DataFrame, strategy: str) -> pd.Series:
    if strategy == FINAL_STRATEGY:
        return df[f"{FINAL_STRATEGY}_state"].astype(str)
    if f"{strategy}_state" in df.columns:
        return df[f"{strategy}_state"].fillna("NON_RISK").astype(str)
    if f"{strategy}_risk_state" in df.columns:
        raw = df[f"{strategy}_risk_state"].fillna("NON_RISK").astype(str)
        return raw.replace({"RISK": "FULL_RISK"})
    if strategy == "FLAT_RISK_50GOLD_50CASH" or strategy == "FLAT_RISK_30GOLD_70CASH":
        return df["MATURE_BASELINE_REGIME_HEDGE_INV_VOL_risk_state"].fillna("NON_RISK").astype(str).replace({"RISK": "FULL_RISK"})
    return pd.Series("NON_RISK", index=df.index)


def run_multi_asset_backtest(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # drawdown columns for all strategies
    strategies_present = [s for s in BENCHMARKS if f"{s}_return" in out.columns]
    for strategy in strategies_present:
        nav = (1 + out[f"{strategy}_return"].fillna(0.0)).cumprod() if f"{strategy}_nav" not in out.columns else out[f"{strategy}_nav"].ffill().fillna(1.0)
        out[f"{strategy}_nav"] = nav
        out[f"{strategy}_drawdown"] = nav / nav.cummax() - 1.0
    return out


def _perf(ret: pd.Series, rf: pd.Series, dates: pd.Series) -> Dict[str, float]:
    nav = (1 + ret.fillna(0.0)).cumprod()
    dd = nav / nav.cummax() - 1.0
    excess = ret - rf
    downside = ret[ret < 0]
    month_key = pd.to_datetime(dates).dt.to_period("M")
    monthly = (1 + ret.fillna(0.0)).groupby(month_key).prod() - 1.0
    ann_ret = nav.iloc[-1] ** (252 / len(ret)) - 1 if len(ret) else np.nan
    ann_vol = ret.std(ddof=0) * math.sqrt(252)
    sharpe = excess.mean() / excess.std(ddof=0) * math.sqrt(252) if excess.std(ddof=0) > 0 else np.nan
    sortino = excess.mean() / downside.std(ddof=0) * math.sqrt(252) if len(downside) and downside.std(ddof=0) > 0 else np.nan
    maxdd = dd.min()
    return {
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": maxdd,
        "calmar_ratio": ann_ret / abs(maxdd) if pd.notna(maxdd) and maxdd < 0 else np.nan,
        "final_nav": nav.iloc[-1] if len(nav) else np.nan,
    }


def compute_performance_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rf = df["CASH_return"].fillna(df["daily_rf"]).fillna(0.0)
    strategies = [s for s in BENCHMARKS if f"{s}_return" in df.columns]
    for strategy in strategies:
        ret = df[f"{strategy}_return"].fillna(0.0)
        met = _perf(ret, rf, df["date"])
        state = _state_series(df, strategy)
        rows.append(
            {
                "strategy": strategy,
                "start_date": df["date"].iloc[0],
                "end_date": df["date"].iloc[-1],
                **met,
                "number_of_switches": max(int((state != state.shift(1, fill_value=state.iloc[0])).sum() - 1), 0),
                "number_of_full_risk_entries": int(((state == "FULL_RISK") & (state.shift(1, fill_value="NON_RISK") != "FULL_RISK")).sum()),
                "number_of_overlay_entries": int(((state == "SLOW_GROWTH_OVERLAY") & (state.shift(1, fill_value="NON_RISK") != "SLOW_GROWTH_OVERLAY")).sum()),
                "time_in_full_risk": (state == "FULL_RISK").mean(),
                "time_in_overlay": (state == "SLOW_GROWTH_OVERLAY").mean(),
                "avg_weight_SPY": df.get(f"{strategy}_weight_SPY", pd.Series(np.nan, index=df.index)).mean(),
                "avg_weight_GOLD": df.get(f"{strategy}_weight_GOLD", pd.Series(np.nan, index=df.index)).mean(),
                "avg_weight_CMDTY_FUT": df.get(f"{strategy}_weight_CMDTY_FUT", pd.Series(np.nan, index=df.index)).mean(),
                "avg_weight_IEF": df.get(f"{strategy}_weight_IEF", pd.Series(np.nan, index=df.index)).mean(),
                "avg_weight_CASH": df.get(f"{strategy}_weight_CASH", pd.Series(np.nan, index=df.index)).mean(),
                "total_turnover": df.get(f"{strategy}_turnover", pd.Series(0.0, index=df.index)).fillna(0.0).sum(),
                "transaction_cost_drag": df.get(f"{strategy}_transaction_cost", pd.Series(0.0, index=df.index)).fillna(0.0).sum(),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(CONFIG["output_dir"] / "performance_summary.csv", index=False)
    return out


def compute_crisis_performance(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rf = df["CASH_return"].fillna(df["daily_rf"]).fillna(0.0)
    strategies = [s for s in BENCHMARKS if f"{s}_return" in df.columns]
    for wname, (start, end) in CRISIS_WINDOWS.items():
        sub = df[(df["date"] >= start) & (df["date"] <= end)]
        if sub.empty:
            continue
        for strategy in strategies:
            ret = sub[f"{strategy}_return"].fillna(0.0)
            met = _perf(ret, rf.loc[sub.index], sub["date"])
            state = _state_series(sub, strategy)
            rows.append(
                {
                    "strategy": strategy,
                    "window": wname,
                    "cumulative_return": (1 + ret).prod() - 1.0,
                    "annualized_return": met["annualized_return"],
                    "max_drawdown": met["max_drawdown"],
                    "volatility": met["annualized_volatility"],
                    "Sharpe": met["sharpe_ratio"],
                    "time_in_full_risk": (state == "FULL_RISK").mean(),
                    "time_in_overlay": (state == "SLOW_GROWTH_OVERLAY").mean(),
                    "avg_weight_SPY": sub.get(f"{strategy}_weight_SPY", pd.Series(np.nan, index=sub.index)).mean(),
                    "avg_weight_GOLD": sub.get(f"{strategy}_weight_GOLD", pd.Series(np.nan, index=sub.index)).mean(),
                    "avg_weight_CMDTY_FUT": sub.get(f"{strategy}_weight_CMDTY_FUT", pd.Series(np.nan, index=sub.index)).mean(),
                    "avg_weight_IEF": sub.get(f"{strategy}_weight_IEF", pd.Series(np.nan, index=sub.index)).mean(),
                    "avg_weight_CASH": sub.get(f"{strategy}_weight_CASH", pd.Series(np.nan, index=sub.index)).mean(),
                    "number_of_switches": max(int((state != state.shift(1, fill_value=state.iloc[0])).sum() - 1), 0),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(CONFIG["output_dir"] / "crisis_performance.csv", index=False)
    return out


def extract_state_events(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    strategies = [s for s in BENCHMARKS if f"{s}_return" in df.columns]
    for strategy in strategies:
        state = _state_series(df, strategy)
        reason_col = f"{strategy}_entry_reason"
        for i in range(1, len(df)):
            if state.iloc[i] == state.iloc[i - 1]:
                continue
            reason = df[reason_col].iloc[i] if reason_col in df.columns else ""
            prev_idx = max(i - 1, 0)
            rows.append(
                {
                    "strategy": strategy,
                    "event_date": df["date"].iloc[i],
                    "event_type": "ENTER" if state.iloc[i] != "NON_RISK" else "EXIT",
                    "reason": reason,
                    "macro_regime_confirmed": df["macro_regime_confirmed"].iloc[i],
                    "monthly_either_state": df["monthly_either_state"].iloc[i],
                    "VIX_ZSCORE_120D": df["VIX_ZSCORE_120D"].iloc[i],
                    "D_CREDIT_SPREAD_20D": df["D_CREDIT_SPREAD_20D"].iloc[i],
                    "CMDTY_RET60": df["CMDTY_RET60"].iloc[i],
                    "spy_drawdown_from_previous_high": df["spy_drawdown_from_previous_high"].iloc[i],
                    "previous_state": state.iloc[i - 1],
                    "new_state": state.iloc[i],
                    "next_21d_SPY_return": (1 + df["SPY_return"].iloc[i + 1 : i + 22].fillna(0.0)).prod() - 1.0,
                    "next_63d_SPY_return": (1 + df["SPY_return"].iloc[i + 1 : i + 64].fillna(0.0)).prod() - 1.0,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(CONFIG["output_dir"] / "state_event_log.csv", index=False)
    return out


def extract_episodes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    strategies = [s for s in BENCHMARKS if f"{s}_return" in df.columns]
    for strategy in strategies:
        state = _state_series(df, strategy)
        active = state.ne("NON_RISK")
        start_mask = active & ~active.shift(1, fill_value=False)
        episode_ids = start_mask.cumsum()
        for ep_id in sorted(episode_ids[active].unique()):
            sub = df[episode_ids.eq(ep_id) & active].copy()
            if sub.empty:
                continue
            st = state.loc[sub.index].iloc[0]
            ret = sub[f"{strategy}_return"].fillna(0.0)
            nav = (1 + ret).cumprod()
            rows.append(
                {
                    "strategy": strategy,
                    "episode_id": int(ep_id),
                    "episode_type": st,
                    "start_date": sub["date"].iloc[0],
                    "end_date": sub["date"].iloc[-1],
                    "duration_days": len(sub),
                    "entry_reason": sub.get(f"{strategy}_entry_reason", pd.Series("", index=sub.index)).iloc[0],
                    "macro_regime_at_entry": sub["macro_regime_confirmed"].iloc[0],
                    "dominant_macro_regime": sub["macro_regime_confirmed"].mode().iloc[0],
                    "SPY_return_during_episode": (1 + sub["SPY_return"].fillna(0.0)).prod() - 1.0,
                    "strategy_return_during_episode": (1 + ret).prod() - 1.0,
                    "SPY_max_drawdown_during_episode": ((1 + sub["SPY_return"].fillna(0.0)).cumprod() / (1 + sub["SPY_return"].fillna(0.0)).cumprod().cummax() - 1.0).min(),
                    "strategy_max_drawdown_during_episode": (nav / nav.cummax() - 1.0).min(),
                    "GOLD_return_during_episode": (1 + sub["GOLD_return"].fillna(0.0)).prod() - 1.0,
                    "IEF_return_during_episode": (1 + sub["IEF_return"].fillna(0.0)).prod() - 1.0,
                    "CASH_return_during_episode": (1 + sub["CASH_return"].fillna(0.0)).prod() - 1.0,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(CONFIG["output_dir"] / "episodes.csv", index=False)
    return out


def compute_component_attribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    windows = {"FULL_SAMPLE": (df["date"].iloc[0], df["date"].iloc[-1]), **CRISIS_WINDOWS}
    for name, (start, end) in windows.items():
        sub = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
        if sub.empty:
            continue
        baseline_ret = sub["MATURE_BASELINE_REGIME_HEDGE_INV_VOL_return"].fillna(0.0)
        flat50_ret = sub["FLAT_RISK_50GOLD_50CASH_return"].fillna(0.0)
        final_ret = sub[f"{FINAL_STRATEGY}_return"].fillna(0.0)
        rows.append(
            {
                "window": name,
                "baseline_mature_allocation_contribution": baseline_ret.sum(),
                "FLAT_RISK_50_50_change_contribution": (flat50_ret - baseline_ret).sum(),
                "STEEP_slow_growth_overlay_contribution": (final_ret - flat50_ret).sum(),
                "transaction_cost_drag_final": sub[f"{FINAL_STRATEGY}_transaction_cost"].fillna(0.0).sum(),
                "total_difference_vs_mature_baseline": (final_ret - baseline_ret).sum(),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(CONFIG["output_dir"] / "final_strategy_component_attribution.csv", index=False)
    return out


def plot_equity_curves(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    for s in ["SPY_BUY_HOLD", "BACKBONE_V2_SPY_CASH", "REGIME_HEDGE_V1_ORIGINAL", "MATURE_BASELINE_REGIME_HEDGE_INV_VOL", FINAL_STRATEGY]:
        if f"{s}_nav" in df.columns:
            ax.plot(df["date"], df[f"{s}_nav"], label=s)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "final_equity_curve_log.png", dpi=150)
    plt.close(fig)


def plot_drawdowns(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    for s in ["SPY_BUY_HOLD", "BACKBONE_V2_SPY_CASH", "REGIME_HEDGE_V1_ORIGINAL", "MATURE_BASELINE_REGIME_HEDGE_INV_VOL", FINAL_STRATEGY]:
        if f"{s}_drawdown" in df.columns:
            ax.plot(df["date"], df[f"{s}_drawdown"], label=s)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "final_drawdown_comparison.png", dpi=150)
    plt.close(fig)


def plot_weights(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    stack = np.vstack([df[f"{FINAL_STRATEGY}_weight_{a}"].fillna(0.0) for a in ASSETS])
    ax.stackplot(df["date"], stack, labels=ASSETS)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "final_weight_stack.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(df["date"], df["spy_drawdown_from_previous_high"], label="SPY DD")
    axes[1].plot(df["date"], df["CMDTY_RET60"], label="CMDTY_RET60")
    axes[2].plot(df["date"], np.where(df["final_state"].eq("FULL_RISK"), 2, np.where(df["final_state"].eq("SLOW_GROWTH_OVERLAY"), 1, 0)), label="Final State")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "final_state_timeline.png", dpi=150)
    plt.close(fig)


def plot_case_studies(df: pd.DataFrame) -> None:
    cases = {
        "case_2015_2016_final.png": ("2015-05-01", "2016-03-31"),
        "case_2022_rate_war_final.png": ("2021-11-01", "2023-03-31"),
        "case_2025_pullback_final.png": ("2025-01-01", "2025-12-31"),
        "case_2008_GFC_final.png": ("2007-10-01", "2009-06-30"),
    }
    for fname, (start, end) in cases.items():
        sub = df[(df["date"] >= start) & (df["date"] <= end)]
        if sub.empty:
            continue
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        axes[0].plot(sub["date"], sub["spy_drawdown_from_previous_high"], label="SPY DD")
        axes[1].plot(sub["date"], sub["CMDTY_RET60"], label="CMDTY_RET60")
        axes[2].plot(sub["date"], sub[f"MATURE_BASELINE_REGIME_HEDGE_INV_VOL_nav"] / sub[f"MATURE_BASELINE_REGIME_HEDGE_INV_VOL_nav"].iloc[0], label="Mature Baseline")
        axes[2].plot(sub["date"], sub[f"{FINAL_STRATEGY}_nav"] / sub[f"{FINAL_STRATEGY}_nav"].iloc[0], label="Final")
        for a in ASSETS:
            axes[3].plot(sub["date"], sub[f"{FINAL_STRATEGY}_weight_{a}"], label=a)
        for ax in axes:
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(CONFIG["figure_dir"] / fname, dpi=150)
        plt.close(fig)


def plot_performance_bar_charts(perf: pd.DataFrame) -> None:
    use = perf[perf["strategy"].isin(["SPY_BUY_HOLD", "BACKBONE_V2_SPY_CASH", "REGIME_HEDGE_V1_ORIGINAL", "MATURE_BASELINE_REGIME_HEDGE_INV_VOL", FINAL_STRATEGY])].copy()
    metrics = ["annualized_return", "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio", "final_nav", "total_turnover"]
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    axes = axes.flatten()
    for ax, m in zip(axes, metrics):
        ax.bar(use["strategy"], use[m])
        ax.set_title(m)
        ax.tick_params(axis="x", rotation=20)
    fig.delaxes(axes[-1])
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "final_performance_bar_charts.png", dpi=150)
    plt.close(fig)

    attr = pd.read_csv(CONFIG["output_dir"] / "final_strategy_component_attribution.csv")
    fig, ax = plt.subplots(figsize=(10, 6))
    cols = [
        "FLAT_RISK_50_50_change_contribution",
        "STEEP_slow_growth_overlay_contribution",
        "total_difference_vs_mature_baseline",
    ]
    for c in cols:
        ax.plot(attr["window"], attr[c], marker="o", label=c)
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "final_component_attribution.png", dpi=150)
    plt.close(fig)


def write_final_report(df: pd.DataFrame, perf: pd.DataFrame, crisis: pd.DataFrame, attr: pd.DataFrame) -> None:
    final_row = perf.loc[perf["strategy"].eq(FINAL_STRATEGY)].iloc[0]
    base_row = perf.loc[perf["strategy"].eq("MATURE_BASELINE_REGIME_HEDGE_INV_VOL")].iloc[0]
    lines = [
        "# MATURE_REGIME_HEDGE_FINAL Report",
        "",
        "## Executive Summary",
        f"The final strategy keeps the validated mature baseline backbone, adds a STEEP slow-growth partial overlay, and replaces FLAT_RISK 100% GOLD with 50% GOLD / 50% CASH.",
        "",
        "## Strategy Philosophy",
        "- stress is rare and heterogeneous;",
        "- avoid over-conditioned hedge selection;",
        "- use inverse-vol in normal regimes;",
        "- use simple conservative hedges in risk regimes;",
        "- use slow-growth partial de-risk rather than immediate full-risk.",
        "",
        "## Final Strategy Definition",
        "- FLAT_NON_RISK: SPY / GOLD / CMDTY_FUT inverse-vol (120d)",
        "- FLAT_RISK: 50% GOLD / 50% CASH",
        "- STEEP_NON_RISK: 100% SPY",
        "- STEEP_SLOW_GROWTH_OVERLAY: 50% SPY / 50% IEF",
        "- STEEP_RISK: 100% IEF",
        "- INVERTED: SPY / GOLD inverse-vol (120d)",
        "",
        "## Why this final design?",
        "- FLAT_RISK 50/50 reduces 2022 path risk without building a complex rate filter.",
        "- STEEP commodity trigger stays partial to reduce overfitting to 2015-2016.",
        "- no fallback regime is used.",
        "",
        "## Main Performance",
        perf[perf["strategy"].isin(["SPY_BUY_HOLD", "BACKBONE_V2_SPY_CASH", "REGIME_HEDGE_V1_ORIGINAL", "MATURE_BASELINE_REGIME_HEDGE_INV_VOL", FINAL_STRATEGY])].to_markdown(index=False),
        "",
        "## Crisis Analysis",
        crisis[crisis["strategy"].isin(["MATURE_BASELINE_REGIME_HEDGE_INV_VOL", FINAL_STRATEGY, "BACKBONE_V2_SPY_CASH"])].to_markdown(index=False),
        "",
        "## Component Attribution",
        attr.to_markdown(index=False),
        "",
        "## Robustness and Overfitting Discussion",
        "- the overlay is sample-dependent, so it remains partial rather than full-risk;",
        "- FLAT_RISK uses a fixed 50/50 GOLD/CASH mix instead of a conditional switch;",
        "- no extra rate/inflation conditional hedge is added at this stage.",
        "",
        "## Final Recommendation",
        "Use `MATURE_REGIME_HEDGE_FINAL` as the current project-end strategy version.",
        "",
        "## Next Research Directions",
        "- out-of-sample validation",
        "- commodity proxy robustness",
        "- implementation and live risk monitoring",
    ]
    (CONFIG["output_dir"] / "MATURE_REGIME_HEDGE_FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    decision = pd.DataFrame(
        [
            {"module": "inverse-vol normal allocation", "rule": "FLAT and INVERTED inverse-vol pools", "rationale": "robust diversification", "main benefit": "stable normal-regime allocation", "main cost": "parameterized by historical vol", "included_in_final_strategy": True},
            {"module": "FLAT_RISK 50% GOLD / 50% CASH", "rule": "replace 100% GOLD", "rationale": "reduce 2022 path dependence", "main benefit": "smoother FLAT risk episodes", "main cost": "lower upside in some risk windows", "included_in_final_strategy": True},
            {"module": "STEEP commodity slow-growth overlay", "rule": "STEEP and CMDTY_RET60 < -10%", "rationale": "partial de-risk for slow-growth stress", "main benefit": "helps 2015-2016 without fully exiting risk", "main cost": "can reduce upside during rebounds", "included_in_final_strategy": True},
            {"module": "STEEP full-risk 100% IEF", "rule": "validated backbone full-risk", "rationale": "keep simple defensive hedge", "main benefit": "strong recession/stress defense", "main cost": "rate-shock sensitivity", "included_in_final_strategy": True},
            {"module": "INVERTED inverse-vol SPY/GOLD", "rule": "no full-risk trigger", "rationale": "maintain exposure with diversification", "main benefit": "simple regime-consistent rule", "main cost": "may not fully hedge non-recession shocks", "included_in_final_strategy": True},
        ]
    )
    decision.to_csv(CONFIG["output_dir"] / "final_strategy_decision_summary.csv", index=False)


def main() -> None:
    ensure_dirs()
    df, src = load_validated_panel()
    abnormal = validate_regime_universe(df)
    df = compute_inverse_vol_weights(df)
    df = build_or_load_backbone_state(df)
    df = build_final_strategy_states(df)
    df = run_multi_asset_backtest(df)

    # top-level final columns
    out_cols = [
        "date", "macro_regime_confirmed", "final_state", "timing_state", "overlay_state", "monthly_either_state",
        "VIX_ZSCORE_120D", "CREDIT_SPREAD_BAA_AAA", "D_CREDIT_SPREAD_20D", "CMDTY_RET60",
        "spy_drawdown_from_previous_high", "SPY_CROSS_ABOVE_MA20",
        "SPY_return", "GOLD_return", "CMDTY_FUT_return", "IEF_return", "CASH_return",
        "FLAT_VIX_STRESS", "FLAT_CREDIT_DD5_STRESS", "STEEP_EITHER_SELL_STRESS", "STEEP_CREDIT_DD5_STRESS",
        "STEEP_CMDTY_RET60_NEG10", "BACKBONE_V2_ENTRY_SIGNAL", "R3_RECOVERY",
    ]
    strategies = [s for s in BENCHMARKS if f"{s}_return" in df.columns]
    for s in strategies:
        for suffix in ["weight_SPY", "weight_GOLD", "weight_CMDTY_FUT", "weight_IEF", "weight_CASH", "return", "nav", "drawdown", "turnover", "transaction_cost", "state", "risk_state", "entry_reason"]:
            c = f"{s}_{suffix}"
            if c in df.columns:
                out_cols.append(c)
    df.loc[:, [c for c in out_cols if c in df.columns]].to_csv(CONFIG["output_dir"] / "daily_backtest_panel.csv", index=False)

    perf = compute_performance_summary(df)
    crisis = compute_crisis_performance(df)
    events = extract_state_events(df)
    episodes = extract_episodes(df)
    attr = compute_component_attribution(df)
    plot_equity_curves(df)
    plot_drawdowns(df)
    plot_weights(df)
    plot_case_studies(df)
    plot_performance_bar_charts(perf)
    write_final_report(df, perf, crisis, attr)

    final_row = perf.loc[perf["strategy"].eq(FINAL_STRATEGY)].iloc[0]
    base_row = perf.loc[perf["strategy"].eq("MATURE_BASELINE_REGIME_HEDGE_INV_VOL")].iloc[0]
    spy_row = perf.loc[perf["strategy"].eq("SPY_BUY_HOLD")].iloc[0]
    print("1. Final strategy performance")
    print(final_row.to_string())
    print("2. Mature baseline performance")
    print(base_row.to_string())
    print("3. SPY buy-hold performance")
    print(spy_row.to_string())
    print("4. Final vs baseline delta:")
    print(
        f"AnnRet {final_row['annualized_return']-base_row['annualized_return']:+.2%}, "
        f"Sharpe {final_row['sharpe_ratio']-base_row['sharpe_ratio']:+.3f}, "
        f"MaxDD {final_row['max_drawdown']-base_row['max_drawdown']:+.2%}, "
        f"Calmar {final_row['calmar_ratio']-base_row['calmar_ratio']:+.3f}"
    )
    print("5. Crisis window key results:")
    print(crisis[crisis["strategy"].isin([FINAL_STRATEGY, "MATURE_BASELINE_REGIME_HEDGE_INV_VOL"]) & crisis["window"].isin(["2015_2016", "2022_RATE_WAR", "2025_PULLBACK"])][["strategy", "window", "cumulative_return", "max_drawdown"]].to_string(index=False))
    print("6. Time in overlay / full risk")
    print(final_row[["time_in_full_risk", "time_in_overlay"]].to_string())
    print("7. Average weights")
    print(final_row[["avg_weight_SPY", "avg_weight_GOLD", "avg_weight_CMDTY_FUT", "avg_weight_IEF", "avg_weight_CASH"]].to_string())
    print(f"8. Output paths: {CONFIG['output_dir']} and {CONFIG['figure_dir']}")
    print("9. Final recommendation summary")
    print("Recommend MATURE_REGIME_HEDGE_FINAL as the current project-end version: simpler than conditional hedge trees, structurally robust, and built from validated baseline modules.")


if __name__ == "__main__":
    main()
