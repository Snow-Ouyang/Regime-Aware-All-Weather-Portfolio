"""Inverse-vol window sensitivity for the current baseline allocation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIG = {
    "output_dir": Path("results/invvol_window_grid_search"),
    "figure_dir": Path("figures/invvol_window_grid_search"),
    "one_way_cost_bps": 5,
    "monthly_rebalance": True,
    "windows": [60, 90, 120, 150, 180],
    "flat_pool": ["SPY", "GOLD", "CMDTY_FUT"],
    "inverted_pool": ["SPY", "GOLD"],
    "fallback_non_risk": {"SPY": 0.80, "GOLD": 0.0, "CMDTY_FUT": 0.0, "IEF": 0.0, "CASH": 0.20},
    "fallback_risk": {"SPY": 0.0, "GOLD": 0.0, "CMDTY_FUT": 0.0, "IEF": 0.0, "CASH": 1.0},
}

PANEL_CANDIDATES = [
    Path("results/regime_aware_risk_parity_allocation/daily_backtest_panel.csv"),
    Path("results/spy_cash_backbone_upgrade_ablation/daily_backtest_panel.csv"),
    Path("results/regime_aware_hedge_allocation_v1/daily_backtest_panel.csv"),
]

ASSET_PANEL_CANDIDATES = [
    Path("results/regime_aware_risk_parity_allocation/daily_backtest_panel.csv"),
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
}

ASSETS = ["SPY", "GOLD", "CMDTY_FUT", "IEF", "CASH"]


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


def _merge_missing(base: pd.DataFrame, other: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in base.columns and c in other.columns]
    if not missing:
        return base
    return base.merge(other[["date"] + missing], on="date", how="left")


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    full = {asset: float(weights.get(asset, 0.0)) for asset in ASSETS}
    total = sum(full.values())
    if total <= 0:
        raise ValueError("Zero-sum weights.")
    return {asset: weight / total for asset, weight in full.items()}


def _is_first_trading_day_of_month(dates: pd.Series) -> pd.Series:
    periods = dates.dt.to_period("M")
    out = periods.ne(periods.shift(1))
    out.iloc[0] = True
    return out


def _max_drawdown(ret: pd.Series) -> float:
    nav = (1 + ret.fillna(0.0)).cumprod()
    return float((nav / nav.cummax() - 1).min()) if not nav.empty else np.nan


def _sortino(ret: pd.Series, rf: pd.Series) -> float:
    excess = ret - rf
    downside = excess[excess < 0]
    if downside.empty:
        return np.nan
    dstd = downside.std(ddof=0)
    if dstd <= 0:
        return np.nan
    return float(excess.mean() / dstd * math.sqrt(252))


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
        raise FileNotFoundError("No base panel found.")
    panel = frames[0][1].copy()
    print(f"Loaded base panel: {frames[0][0]}")
    needed = [
        "date", "spy_price", "spy_daily_return", "daily_rf", "macro_regime_confirmed", "monthly_either_state",
        "VIX_LEVEL", "VIX_ZSCORE_120D", "CREDIT_SPREAD_BAA_AAA", "D_CREDIT_SPREAD_20D",
        "spy_drawdown_from_previous_high", "SPY_MA20", "SPY_CROSS_ABOVE_MA20",
        "timing_state", "cross_state", "BACKBONE_V2_SPY_CASH_weight_SPY", "BACKBONE_V2_SPY_CASH_weight_CASH",
        "BACKBONE_V2_SPY_CASH_return", "BACKBONE_V2_SPY_CASH_nav", "SPY_return", "GOLD_return", "IEF_return",
        "CMDTY_FUT_return", "CASH_return",
    ]
    for _, df in frames[1:]:
        panel = _merge_missing(panel, df, needed)
    if "spy_drawdown_from_previous_high" not in panel.columns:
        panel["spy_drawdown_from_previous_high"] = panel["spy_price"] / panel["spy_price"].cummax() - 1
    if "D_CREDIT_SPREAD_20D" not in panel.columns:
        panel["D_CREDIT_SPREAD_20D"] = panel["CREDIT_SPREAD_BAA_AAA"] - panel["CREDIT_SPREAD_BAA_AAA"].shift(20)
    if "SPY_MA20" not in panel.columns:
        panel["SPY_MA20"] = panel["spy_price"].rolling(20, min_periods=20).mean()
    if "SPY_CROSS_ABOVE_MA20" not in panel.columns:
        panel["SPY_CROSS_ABOVE_MA20"] = (panel["spy_price"] > panel["SPY_MA20"]) & (panel["spy_price"].shift(1) <= panel["SPY_MA20"].shift(1))
    panel["daily_rf"] = pd.to_numeric(panel["daily_rf"], errors="coerce").fillna(0.0)
    panel["macro_regime_confirmed"] = panel["macro_regime_confirmed"].fillna("NEUTRAL").astype(str)
    return panel


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
    }
    for path in ASSET_PANEL_CANDIDATES:
        if not path.exists():
            continue
        src = _read_csv(path)
        rename_map = {}
        for final, candidates in mapping.items():
            if final in out.columns:
                continue
            for c in candidates:
                if c in src.columns:
                    rename_map[c] = final
                    break
        if rename_map:
            out = out.merge(src[["date"] + list(rename_map.keys())].rename(columns=rename_map), on="date", how="left")
    required = ["SPY_return", "GOLD_return", "IEF_return", "CMDTY_FUT_return", "CASH_return"]
    for col in required:
        if col not in out.columns:
            raise ValueError(f"Missing asset column: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce")
    valid = out[required].notna().all(axis=1)
    if not valid.any():
        raise ValueError("No overlapping asset sample.")
    out = out.loc[int(valid.idxmax()):].reset_index(drop=True)
    for asset in ASSETS:
        out[f"{asset}_nav"] = (1 + out[f"{asset}_return"].fillna(0.0)).cumprod()
    return out


def build_backbone_v2_state(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    if "BACKBONE_V2_SPY_CASH_weight_SPY" in df.columns:
        spy_w = pd.to_numeric(df["BACKBONE_V2_SPY_CASH_weight_SPY"], errors="coerce").fillna(1.0)
    elif "BACKBONE_V2_UPGRADED_weight_spy" in df.columns:
        spy_w = pd.to_numeric(df["BACKBONE_V2_UPGRADED_weight_spy"], errors="coerce").fillna(1.0)
        df["BACKBONE_V2_SPY_CASH_weight_SPY"] = spy_w
        df["BACKBONE_V2_SPY_CASH_weight_CASH"] = 1 - spy_w
        if "BACKBONE_V2_SPY_CASH_return" not in df.columns and "BACKBONE_V2_UPGRADED_return" in df.columns:
            df["BACKBONE_V2_SPY_CASH_return"] = pd.to_numeric(df["BACKBONE_V2_UPGRADED_return"], errors="coerce").fillna(0.0)
        if "BACKBONE_V2_SPY_CASH_nav" not in df.columns and "BACKBONE_V2_SPY_CASH_return" in df.columns:
            df["BACKBONE_V2_SPY_CASH_nav"] = (1 + df["BACKBONE_V2_SPY_CASH_return"]).cumprod()
    else:
        raise ValueError("Missing BACKBONE_V2 weights.")
    df["timing_state"] = np.where(spy_w >= 0.5, "NON_RISK", "RISK")
    df["cross_state"] = df["macro_regime_confirmed"] + "_" + df["timing_state"]
    return df


def compute_inverse_vol_weights(window_df: pd.DataFrame, pool: List[str]) -> Dict[str, float]:
    cols = [f"{a}_return" for a in pool]
    subset = window_df[cols].copy()
    subset.columns = pool
    vols = subset.std(ddof=0) * math.sqrt(252)
    valid = vols.replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) != len(pool) or (valid <= 0).any():
        return _normalize_weights({pool[0]: 1.0})
    raw = 1.0 / valid
    raw = raw / raw.sum()
    return _normalize_weights({asset: raw.get(asset, 0.0) for asset in pool})


def build_target_weights(panel: pd.DataFrame, risk_window: int) -> pd.DataFrame:
    first_month = _is_first_trading_day_of_month(panel["date"])
    rows = []
    last_flat = _normalize_weights({"SPY": 0.60, "GOLD": 0.30, "CMDTY_FUT": 0.10})
    last_inverted = _normalize_weights({"SPY": 0.75, "GOLD": 0.25})
    prev_cross_state = None
    for i, row in panel.iterrows():
        regime = str(row["macro_regime_confirmed"])
        timing_state = str(row["timing_state"])
        cross_state = f"{regime}_{timing_state}"
        recalc = bool(first_month.iloc[i]) or (i > 0 and cross_state != prev_cross_state)
        prev_cross_state = cross_state
        if recalc and i >= risk_window - 1:
            window_df = panel.loc[i - risk_window + 1 : i]
            last_flat = compute_inverse_vol_weights(window_df, CONFIG["flat_pool"])
            last_inverted = compute_inverse_vol_weights(window_df, CONFIG["inverted_pool"])
        if regime == "FLAT":
            weights = _normalize_weights({"GOLD": 1.0}) if timing_state == "RISK" else last_flat
        elif regime == "INVERTED":
            weights = last_inverted
        elif regime == "STEEP":
            weights = _normalize_weights({"SPY": 1.0}) if timing_state == "NON_RISK" else _normalize_weights({"IEF": 0.80, "GOLD": 0.20})
        else:
            weights = _normalize_weights(CONFIG["fallback_risk"] if timing_state == "RISK" else CONFIG["fallback_non_risk"])
        rows.append(weights)
    return pd.DataFrame(rows)


def run_backtest(panel: pd.DataFrame, strategy: str, target_weights: pd.DataFrame) -> pd.DataFrame:
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


def compute_performance(panel: pd.DataFrame, strategy: str, risk_window: int) -> Dict[str, object]:
    ret = panel[f"{strategy}_return"].fillna(0.0)
    rf = panel["CASH_return"].fillna(0.0)
    stats = _annualized_stats(ret, rf, panel["date"])
    return {"strategy": strategy, "risk_window": risk_window, **stats}


def compute_crisis(panel: pd.DataFrame, strategy: str, risk_window: int) -> pd.DataFrame:
    rows = []
    for period, (start, end) in CRISIS_WINDOWS.items():
        sub = panel[(panel["date"] >= pd.Timestamp(start)) & (panel["date"] <= pd.Timestamp(end))]
        if sub.empty:
            continue
        ret = sub[f"{strategy}_return"].fillna(0.0)
        rf = sub["CASH_return"].fillna(0.0)
        stats = _annualized_stats(ret, rf, sub["date"])
        rows.append({"period": period, "risk_window": risk_window, **stats})
    return pd.DataFrame(rows)


def plot_navs(panel: pd.DataFrame, strategies: List[str]) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    for strategy in strategies:
        ax.plot(panel["date"], panel[f"{strategy}_nav"], label=strategy)
    ax.set_yscale("log")
    ax.set_title("Inverse-Vol Window Grid Search NAV")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "nav_comparison.png", dpi=150)
    plt.close(fig)


def plot_drawdowns(panel: pd.DataFrame, strategies: List[str]) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    for strategy in strategies:
        nav = panel[f"{strategy}_nav"]
        ax.plot(panel["date"], nav / nav.cummax() - 1, label=strategy)
    ax.set_title("Inverse-Vol Window Grid Search Drawdown")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "drawdown_comparison.png", dpi=150)
    plt.close(fig)


def plot_sensitivity(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = ["AnnRet", "Sharpe", "MaxDD", "Final NAV"]
    for ax, metric in zip(axes.flatten(), metrics):
        ax.plot(summary["risk_window"], summary[metric], marker="o")
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "window_sensitivity.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    panel = load_base_panel()
    panel = load_asset_returns(panel)
    panel = build_backbone_v2_state(panel)

    run_panels = {}
    summary_rows = []
    crisis_frames = []
    strategies = []
    base_view = panel.copy()
    # benchmark
    spy_target = pd.DataFrame([_normalize_weights({"SPY": 1.0}) for _ in range(len(panel))])
    cash_target = pd.DataFrame([_normalize_weights({"SPY": 1.0} if s == "NON_RISK" else {"CASH": 1.0}) for s in panel["timing_state"]])
    base_view = run_backtest(base_view, "SPY_BUY_HOLD", spy_target)
    base_view = run_backtest(base_view, "BACKBONE_V2_SPY_CASH", cash_target)
    summary_rows.append(compute_performance(base_view, "SPY_BUY_HOLD", -1))
    summary_rows.append(compute_performance(base_view, "BACKBONE_V2_SPY_CASH", -1))
    for window in CONFIG["windows"]:
        strategy = f"BASELINE_INVVOL_{window}"
        strategies.append(strategy)
        target = build_target_weights(panel, window)
        base_view = run_backtest(base_view, strategy, target)
        summary_rows.append(compute_performance(base_view, strategy, window))
        crisis_frames.append(compute_crisis(base_view, strategy, window))
    summary = pd.DataFrame(summary_rows)
    baseline120 = summary[summary["risk_window"].eq(120)].iloc[0]
    grid_only = summary[summary["risk_window"].isin(CONFIG["windows"])].copy()
    diff = grid_only.copy()
    for metric in ["AnnRet", "Sharpe", "MaxDD", "Calmar", "Final NAV"]:
        diff[f"Δ{metric}"] = diff[metric] - baseline120[metric]

    base_view.to_csv(CONFIG["output_dir"] / "daily_backtest_panel.csv", index=False)
    summary.to_csv(CONFIG["output_dir"] / "performance_summary.csv", index=False)
    diff.to_csv(CONFIG["output_dir"] / "difference_vs_120.csv", index=False)
    pd.concat(crisis_frames, ignore_index=True).to_csv(CONFIG["output_dir"] / "crisis_performance.csv", index=False)
    plot_navs(base_view, ["SPY_BUY_HOLD", "BACKBONE_V2_SPY_CASH"] + strategies)
    plot_drawdowns(base_view, ["SPY_BUY_HOLD", "BACKBONE_V2_SPY_CASH"] + strategies)
    plot_sensitivity(grid_only)


if __name__ == "__main__":
    main()
