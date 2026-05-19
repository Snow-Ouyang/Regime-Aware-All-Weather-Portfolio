"""Regime-aware hedge allocation structure test focused on STEEP regime."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIG = {
    "output_dir": Path("results/regime_aware_hedge_allocation_steep_test"),
    "figure_dir": Path("figures/regime_aware_hedge_allocation_steep_test"),
    "one_way_cost_bps": 5,
    "monthly_rebalance": True,
    "timing_backbone": "BACKBONE_V2_UPGRADED",
    "strategy_rules": {
        "REGIME_HEDGE_V1_ORIGINAL": {
            "INVERTED": {"SPY": 0.70, "GOLD": 0.20, "IEF": 0.00, "CASH": 0.10},
            "FLAT_NON_RISK": {"SPY": 0.60, "GOLD": 0.40, "IEF": 0.00, "CASH": 0.00},
            "FLAT_RISK": {"SPY": 0.00, "GOLD": 1.00, "IEF": 0.00, "CASH": 0.00},
            "STEEP_NON_RISK": {"SPY": 0.90, "GOLD": 0.00, "IEF": 0.10, "CASH": 0.00},
            "STEEP_RISK": {"SPY": 0.00, "GOLD": 0.00, "IEF": 1.00, "CASH": 0.00},
            "FALLBACK_NON_RISK": {"SPY": 0.80, "GOLD": 0.00, "IEF": 0.00, "CASH": 0.20},
            "FALLBACK_RISK": {"SPY": 0.00, "GOLD": 0.00, "IEF": 0.00, "CASH": 1.00},
        },
        "REGIME_HEDGE_V2_STEEP_CASH": {
            "INVERTED": {"SPY": 0.70, "GOLD": 0.20, "IEF": 0.00, "CASH": 0.10},
            "FLAT_NON_RISK": {"SPY": 0.60, "GOLD": 0.40, "IEF": 0.00, "CASH": 0.00},
            "FLAT_RISK": {"SPY": 0.00, "GOLD": 1.00, "IEF": 0.00, "CASH": 0.00},
            "STEEP_NON_RISK": {"SPY": 1.00, "GOLD": 0.00, "IEF": 0.00, "CASH": 0.00},
            "STEEP_RISK": {"SPY": 0.00, "GOLD": 0.00, "IEF": 0.00, "CASH": 1.00},
            "FALLBACK_NON_RISK": {"SPY": 0.80, "GOLD": 0.00, "IEF": 0.00, "CASH": 0.20},
            "FALLBACK_RISK": {"SPY": 0.00, "GOLD": 0.00, "IEF": 0.00, "CASH": 1.00},
        },
        "REGIME_HEDGE_V2_STEEP_GOLD": {
            "INVERTED": {"SPY": 0.70, "GOLD": 0.20, "IEF": 0.00, "CASH": 0.10},
            "FLAT_NON_RISK": {"SPY": 0.60, "GOLD": 0.40, "IEF": 0.00, "CASH": 0.00},
            "FLAT_RISK": {"SPY": 0.00, "GOLD": 1.00, "IEF": 0.00, "CASH": 0.00},
            "STEEP_NON_RISK": {"SPY": 1.00, "GOLD": 0.00, "IEF": 0.00, "CASH": 0.00},
            "STEEP_RISK": {"SPY": 0.00, "GOLD": 1.00, "IEF": 0.00, "CASH": 0.00},
            "FALLBACK_NON_RISK": {"SPY": 0.80, "GOLD": 0.00, "IEF": 0.00, "CASH": 0.20},
            "FALLBACK_RISK": {"SPY": 0.00, "GOLD": 0.00, "IEF": 0.00, "CASH": 1.00},
        },
    },
}

PANEL_CANDIDATES = [
    Path("results/regime_aware_hedge_allocation_v1/daily_backtest_panel.csv"),
    Path("results/spy_cash_backbone_upgrade_ablation/daily_backtest_panel.csv"),
    Path("results/flat_vix_credit_trigger_diagnostic/full_backtest_panel.csv"),
]

ASSET_PANEL_CANDIDATES = [
    Path("results/regime_aware_hedge_allocation_v1/daily_backtest_panel.csv"),
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
    "BACKBONE_V2_SPY_CASH",
    "REGIME_HEDGE_V1_ORIGINAL",
    "REGIME_HEDGE_V2_STEEP_CASH",
    "REGIME_HEDGE_V2_STEEP_GOLD",
    "STATIC_60_30_10",
    "STATIC_70_20_10",
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
        raise ValueError(f"No date column found in {path}")
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
        raise ValueError("Zero-sum weight set.")
    return {k: v / total for k, v in full.items()}


def _is_first_trading_day_of_month(dates: pd.Series) -> pd.Series:
    period = dates.dt.to_period("M")
    out = np.zeros(len(dates), dtype=bool)
    prev = None
    for i, p in enumerate(period):
        if i == 0 or p != prev:
            out[i] = True
        prev = p
    return pd.Series(out, index=dates.index)


def _max_drawdown(ret: pd.Series) -> float:
    nav = (1 + ret.fillna(0.0)).cumprod()
    if nav.empty:
        return np.nan
    return float((nav / nav.cummax() - 1).min())


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
        "timing_state",
        "cross_state",
        "entry_reason",
        "BACKBONE_V2_SPY_CASH_return",
        "BACKBONE_V2_SPY_CASH_nav",
        "BACKBONE_V2_SPY_CASH_weight_SPY",
        "BACKBONE_V2_SPY_CASH_weight_CASH",
        "SPY_return",
        "GOLD_return",
        "IEF_return",
        "CASH_return",
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


def load_asset_returns(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    if "SPY_return" not in out.columns:
        out["SPY_return"] = pd.to_numeric(out["spy_daily_return"], errors="coerce")
    if "CASH_return" not in out.columns:
        out["CASH_return"] = pd.to_numeric(out["daily_rf"], errors="coerce").fillna(0.0)

    for path in ASSET_PANEL_CANDIDATES:
        if not path.exists():
            continue
        src = _read_csv(path)
        merge_cols = ["date"]
        rename_map: Dict[str, str] = {}
        for candidate, final in [
            (["GOLD_return", "GOLD_RETURN", "GLD_return", "GLD_RETURN"], "GOLD_return"),
            (["IEF_return", "IEF_RETURN"], "IEF_return"),
        ]:
            col = next((c for c in candidate if c in src.columns), None)
            if col and final not in out.columns:
                merge_cols.append(col)
                rename_map[col] = final
        if len(merge_cols) > 1:
            out = out.merge(src[merge_cols].rename(columns=rename_map), on="date", how="left")

    required = ["SPY_return", "GOLD_return", "IEF_return", "CASH_return"]
    for col in required:
        if col not in out.columns:
            raise ValueError(f"Missing required asset return column: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce")
    valid = out[required].notna().all(axis=1)
    if not valid.any():
        raise ValueError("No overlapping SPY/GOLD/IEF/CASH sample.")
    start_idx = valid.idxmax()
    out = out.loc[start_idx:].reset_index(drop=True)
    for asset in ASSETS:
        out[f"{asset}_NAV"] = (1 + out[f"{asset}_return"].fillna(0.0)).cumprod()
    return out


def build_backbone_v2_state(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    if "BACKBONE_V2_SPY_CASH_weight_SPY" in df.columns and "timing_state" in df.columns:
        inferred = pd.Series(
            np.where(pd.to_numeric(df["BACKBONE_V2_SPY_CASH_weight_SPY"], errors="coerce") >= 0.5, "NON_RISK", "RISK"),
            index=df.index,
        )
        df["timing_state"] = df["timing_state"].fillna(inferred)
        return df

    if "BACKBONE_V2_UPGRADED_weight_spy" in df.columns:
        df["BACKBONE_V2_SPY_CASH_weight_SPY"] = pd.to_numeric(df["BACKBONE_V2_UPGRADED_weight_spy"], errors="coerce").fillna(1.0)
        if "BACKBONE_V2_UPGRADED_weight_cash" in df.columns:
            df["BACKBONE_V2_SPY_CASH_weight_CASH"] = pd.to_numeric(df["BACKBONE_V2_UPGRADED_weight_cash"], errors="coerce").fillna(1 - df["BACKBONE_V2_SPY_CASH_weight_SPY"])
        else:
            df["BACKBONE_V2_SPY_CASH_weight_CASH"] = 1 - df["BACKBONE_V2_SPY_CASH_weight_SPY"]
        df["BACKBONE_V2_SPY_CASH_return"] = pd.to_numeric(df["BACKBONE_V2_UPGRADED_return"], errors="coerce").fillna(0.0)
        df["BACKBONE_V2_SPY_CASH_nav"] = (1 + df["BACKBONE_V2_SPY_CASH_return"]).cumprod()
    df["timing_state"] = np.where(pd.to_numeric(df["BACKBONE_V2_SPY_CASH_weight_SPY"], errors="coerce") >= 0.5, "NON_RISK", "RISK")
    if "cross_state" not in df.columns:
        df["cross_state"] = df["macro_regime_confirmed"] + "_" + df["timing_state"]
    if "entry_reason" not in df.columns:
        df["entry_reason"] = ""
    return df


def build_allocation_rules(panel: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    rules = CONFIG["strategy_rules"][strategy_name]
    df = panel.copy()
    weights = []
    for _, row in df.iterrows():
        regime = str(row["macro_regime_confirmed"])
        timing_state = str(row["timing_state"])
        if regime == "INVERTED":
            key = "INVERTED"
            cross_state = "INVERTED"
        elif regime == "FLAT":
            key = f"FLAT_{timing_state}"
            cross_state = key
        elif regime == "STEEP":
            key = f"STEEP_{timing_state}"
            cross_state = key
        else:
            key = "FALLBACK_RISK" if timing_state == "RISK" else "FALLBACK_NON_RISK"
            cross_state = f"{regime}_{timing_state}"
        w = _normalize_weights(rules[key])
        weights.append(w)
        df.loc[df.index == row.name, "cross_state"] = cross_state
    wdf = pd.DataFrame(weights)
    for asset in ASSETS:
        df[f"{strategy_name}_target_{asset}"] = wdf[asset].values
    return df


def run_multi_asset_backtest(panel: pd.DataFrame, strategy_name: str, target_weights: pd.DataFrame, monthly_rebalance: bool) -> pd.DataFrame:
    df = panel.copy()
    tw = target_weights[ASSETS].copy().div(target_weights[ASSETS].sum(axis=1), axis=0).fillna(0.0)
    first_month_day = _is_first_trading_day_of_month(df["date"])

    for asset in ASSETS:
        df[f"{strategy_name}_weight_{asset}"] = np.nan
    df[f"{strategy_name}_return"] = np.nan
    df[f"{strategy_name}_nav"] = np.nan
    df[f"turnover_{strategy_name}"] = np.nan
    df[f"transaction_cost_{strategy_name}"] = np.nan

    current = tw.iloc[0].to_dict()
    target_prev = tw.iloc[0].to_dict()
    nav = 1.0
    for i in range(len(df)):
        desired = tw.iloc[i].to_dict()
        target_changed = any(abs(desired[a] - target_prev[a]) > 1e-12 for a in ASSETS)
        turnover = 0.0
        cost = 0.0
        if i > 0 and (target_changed or (monthly_rebalance and bool(first_month_day.iloc[i]))):
            turnover = float(sum(abs(desired[a] - current[a]) for a in ASSETS))
            cost = 0.5 * turnover * CONFIG["one_way_cost_bps"] / 10000
            current = desired.copy()

        gross = float(sum(current[a] * df.iloc[i][f"{a}_return"] for a in ASSETS))
        net = gross - cost
        nav *= 1 + net
        for asset in ASSETS:
            df.loc[i, f"{strategy_name}_weight_{asset}"] = current[asset]
        df.loc[i, f"{strategy_name}_return"] = net
        df.loc[i, f"{strategy_name}_nav"] = nav
        df.loc[i, f"turnover_{strategy_name}"] = turnover
        df.loc[i, f"transaction_cost_{strategy_name}"] = cost

        denom = 1 + gross
        if denom != 0:
            current = {a: current[a] * (1 + df.iloc[i][f"{a}_return"]) / denom for a in ASSETS}
        target_prev = desired.copy()
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
    maxdd = _max_drawdown(ret)
    calmar = ann / abs(maxdd) if pd.notna(maxdd) and maxdd < 0 else np.nan
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
        "number_of_rebalances": int((panel[f"turnover_{strategy_name}"] > 0).sum()),
        "total_turnover": panel[f"turnover_{strategy_name}"].fillna(0.0).sum(),
        "transaction_cost_drag": panel[f"transaction_cost_{strategy_name}"].fillna(0.0).sum(),
        "avg_weight_SPY": panel[f"{strategy_name}_weight_SPY"].mean(),
        "avg_weight_GOLD": panel[f"{strategy_name}_weight_GOLD"].mean(),
        "avg_weight_IEF": panel[f"{strategy_name}_weight_IEF"].mean(),
        "avg_weight_CASH": panel[f"{strategy_name}_weight_CASH"].mean(),
        "time_in_risk": panel["timing_state"].eq("RISK").mean(),
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
            rows.append(
                {
                    "period": period,
                    "strategy": strategy,
                    "cumulative_return": nav.iloc[-1] - 1,
                    "max_drawdown": _max_drawdown(ret),
                    "volatility": vol,
                    "Sharpe": sharpe,
                    "avg_weight_SPY": sub[f"{strategy}_weight_SPY"].mean(),
                    "avg_weight_GOLD": sub[f"{strategy}_weight_GOLD"].mean(),
                    "avg_weight_IEF": sub[f"{strategy}_weight_IEF"].mean(),
                    "avg_weight_CASH": sub[f"{strategy}_weight_CASH"].mean(),
                    "turnover": sub[f"turnover_{strategy}"].fillna(0.0).sum(),
                    "cost_drag": sub[f"transaction_cost_{strategy}"].fillna(0.0).sum(),
                    "annualized_return": ann,
                }
            )
    return pd.DataFrame(rows)


def extract_steep_risk_episodes(panel: pd.DataFrame) -> pd.DataFrame:
    mask = panel["cross_state"].eq("STEEP_RISK")
    rows = []
    starts = panel.index[mask & ~mask.shift(1, fill_value=False)]
    for eid, start_idx in enumerate(starts, 1):
        end_idx = start_idx
        while end_idx + 1 < len(panel) and mask.iloc[end_idx + 1]:
            end_idx += 1
        sub = panel.iloc[start_idx : end_idx + 1]
        rows.append(
            {
                "episode_id": eid,
                "start_date": sub["date"].iloc[0],
                "end_date": sub["date"].iloc[-1],
                "duration_days": len(sub),
                "entry_reason": sub["entry_reason"].replace("", np.nan).dropna().iloc[0] if sub["entry_reason"].replace("", np.nan).notna().any() else "",
            }
        )
    return pd.DataFrame(rows)


def compare_steep_risk_assets(panel: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, ep in episodes.iterrows():
        sub = panel[(panel["date"] >= ep["start_date"]) & (panel["date"] <= ep["end_date"])].copy()
        row = dict(ep)
        for asset in ASSETS:
            ret = sub[f"{asset}_return"].fillna(0.0)
            nav = (1 + ret).cumprod()
            row[f"{asset}_return"] = nav.iloc[-1] - 1
            row[f"{asset}_max_drawdown"] = _max_drawdown(ret)
        for strategy in ["REGIME_HEDGE_V1_ORIGINAL", "REGIME_HEDGE_V2_STEEP_CASH", "REGIME_HEDGE_V2_STEEP_GOLD"]:
            ret = sub[f"{strategy}_return"].fillna(0.0)
            row[f"{strategy}_return"] = (1 + ret).cumprod().iloc[-1] - 1
        rows.append(row)
    return pd.DataFrame(rows)


def plot_equity_curves(panel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    for strategy in ["SPY_BUY_HOLD", "BACKBONE_V2_SPY_CASH", "REGIME_HEDGE_V1_ORIGINAL", "REGIME_HEDGE_V2_STEEP_CASH", "REGIME_HEDGE_V2_STEEP_GOLD", "STATIC_70_20_10"]:
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
    for strategy in ["SPY_BUY_HOLD", "BACKBONE_V2_SPY_CASH", "REGIME_HEDGE_V1_ORIGINAL", "REGIME_HEDGE_V2_STEEP_CASH", "REGIME_HEDGE_V2_STEEP_GOLD"]:
        nav = panel[f"{strategy}_nav"]
        ax.plot(panel["date"], nav / nav.cummax() - 1, label=strategy)
    ax.legend(ncol=2, fontsize=8)
    ax.set_title("Drawdown Comparison")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "drawdown_comparison.png", dpi=150)
    plt.close(fig)


def plot_weight_stacks(panel: pd.DataFrame, strategy_name: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.stackplot(
        panel["date"],
        panel[f"{strategy_name}_weight_SPY"],
        panel[f"{strategy_name}_weight_GOLD"],
        panel[f"{strategy_name}_weight_IEF"],
        panel[f"{strategy_name}_weight_CASH"],
        labels=ASSETS,
    )
    ax.legend(loc="upper left", ncol=4)
    ax.set_ylim(0, 1)
    ax.set_title(strategy_name)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / filename, dpi=150)
    plt.close(fig)


def plot_case_studies(panel: pd.DataFrame) -> None:
    cases = {
        "2022": ("2021-11-01", "2023-03-31"),
        "2008": ("2007-10-01", "2009-06-30"),
        "2015_2016": ("2015-05-01", "2016-03-31"),
        "2018Q4": ("2018-10-01", "2019-01-31"),
        "2025_PULLBACK": ("2025-01-01", "2025-12-31"),
    }
    for name, (start, end) in cases.items():
        sub = panel[(panel["date"] >= pd.Timestamp(start)) & (panel["date"] <= pd.Timestamp(end))].copy()
        if sub.empty:
            continue
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1.2, 1.2, 0.8]})
        for asset in ASSETS:
            axes[0].plot(sub["date"], sub[f"{asset}_NAV"] / sub[f"{asset}_NAV"].iloc[0], label=asset)
        axes[0].legend(fontsize=8, ncol=4)
        for strategy in ["REGIME_HEDGE_V1_ORIGINAL", "REGIME_HEDGE_V2_STEEP_CASH", "REGIME_HEDGE_V2_STEEP_GOLD"]:
            axes[1].plot(sub["date"], sub[f"{strategy}_nav"] / sub[f"{strategy}_nav"].iloc[0], label=strategy)
        axes[1].legend(fontsize=8)
        axes[2].stackplot(
            sub["date"],
            sub["REGIME_HEDGE_V2_STEEP_CASH_weight_SPY"],
            sub["REGIME_HEDGE_V2_STEEP_CASH_weight_GOLD"],
            sub["REGIME_HEDGE_V2_STEEP_CASH_weight_IEF"],
            sub["REGIME_HEDGE_V2_STEEP_CASH_weight_CASH"],
            labels=ASSETS,
        )
        axes[2].legend(fontsize=7, ncol=4)
        axes[3].plot(sub["date"], sub["timing_state"].map({"NON_RISK": 0, "RISK": 1}), label="timing_state")
        axes[3].legend(fontsize=8)
        axes[3].set_yticks([0, 1], ["NON_RISK", "RISK"])
        fig.tight_layout()
        fig.savefig(CONFIG["figure_dir"] / f"crisis_case_study_{name}.png", dpi=150)
        plt.close(fig)


def plot_performance_bars(perf: pd.DataFrame) -> None:
    show = perf[perf["strategy"].isin(STRATEGIES)].copy()
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


def plot_steep_risk_boxplot(compare_df: pd.DataFrame) -> None:
    if compare_df.empty:
        return
    rows = []
    for _, row in compare_df.iterrows():
        for asset in ASSETS:
            rows.append({"asset": asset, "episode_return": row[f"{asset}_return"], "episode_max_drawdown": row[f"{asset}_max_drawdown"]})
    plot_df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_df.boxplot(column="episode_return", by="asset", ax=axes[0], grid=False)
    plot_df.boxplot(column="episode_max_drawdown", by="asset", ax=axes[1], grid=False)
    axes[0].set_title("Episode Return")
    axes[1].set_title("Episode Max Drawdown")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "steep_risk_asset_performance_boxplot.png", dpi=150)
    plt.close(fig)


def write_markdown_report(perf: pd.DataFrame, crisis: pd.DataFrame, steep_compare: pd.DataFrame) -> None:
    p = perf.set_index("strategy")
    v1 = p.loc["REGIME_HEDGE_V1_ORIGINAL"]
    cash = p.loc["REGIME_HEDGE_V2_STEEP_CASH"]
    gold = p.loc["REGIME_HEDGE_V2_STEEP_GOLD"]
    backbone = p.loc["BACKBONE_V2_SPY_CASH"]
    lines = [
        "# STEEP_RISK_HEDGE_TEST_REPORT",
        "",
        "## 1. Purpose",
        "This round tests whether the STEEP sleeve should drop IEF in NON_RISK and replace STEEP_RISK = 100% IEF with either 100% CASH or 100% GOLD.",
        "",
        "## 2. Strategy definitions",
        "- V1 original: STEEP_NON_RISK = 90 SPY / 10 IEF, STEEP_RISK = 100 IEF",
        "- V2 steep cash: STEEP_NON_RISK = 100 SPY, STEEP_RISK = 100 CASH",
        "- V2 steep gold: STEEP_NON_RISK = 100 SPY, STEEP_RISK = 100 GOLD",
        "- BACKBONE_V2_SPY_CASH: 100 SPY in NON_RISK, 100 CASH in RISK",
        "",
        "## 3. Main performance comparison",
        f"- V1 original: AnnRet {v1['annualized_return']:.2%}, Sharpe {v1['sharpe_ratio']:.2f}, MaxDD {v1['max_drawdown']:.2%}, Final NAV {v1['final_nav']:.2f}",
        f"- V2 steep cash: AnnRet {cash['annualized_return']:.2%}, Sharpe {cash['sharpe_ratio']:.2f}, MaxDD {cash['max_drawdown']:.2%}, Final NAV {cash['final_nav']:.2f}",
        f"- V2 steep gold: AnnRet {gold['annualized_return']:.2%}, Sharpe {gold['sharpe_ratio']:.2f}, MaxDD {gold['max_drawdown']:.2%}, Final NAV {gold['final_nav']:.2f}",
        f"- Backbone SPY/CASH: AnnRet {backbone['annualized_return']:.2%}, Sharpe {backbone['sharpe_ratio']:.2f}, MaxDD {backbone['max_drawdown']:.2%}, Final NAV {backbone['final_nav']:.2f}",
        "",
        "## 4. STEEP-specific findings",
        "See `steep_risk_episode_comparison.csv` for episode-by-episode comparison across SPY, CASH, GOLD, IEF and the three tested structures.",
        "",
        "## 5. Crisis period analysis",
        "Focus on 2022 first, then check whether 2008/GFC defense is damaged, and whether 2015-2016, 2018Q4, COVID, and 2025 pullback stay acceptable.",
        "",
        "## 6. Interpretation",
        "The key question is whether IEF in STEEP_RISK was a 2008-specific advantage but a 2022-specific drag, and whether CASH or GOLD gives a more robust replacement.",
        "",
        "## 7. Recommendation",
        "Use the best full-sample and crisis-balanced version from this report as the next allocation candidate.",
        "",
        "## 8. Caveats",
        "- STEEP_RISK sample size is limited.",
        "- 2022 is a special rising-rate shock.",
        "- Gold can be volatile inside STEEP risk.",
        "- No risk budgeting or optimization was used.",
    ]
    (CONFIG["output_dir"] / "STEEP_RISK_HEDGE_TEST_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = load_base_panel()
    panel = load_asset_returns(panel)
    panel = build_backbone_v2_state(panel)

    if "entry_reason" not in panel.columns:
        panel["entry_reason"] = ""
    if "cross_state" not in panel.columns:
        panel["cross_state"] = panel["macro_regime_confirmed"] + "_" + panel["timing_state"]

    panel["SPY_BUY_HOLD_weight_SPY"] = 1.0
    panel["SPY_BUY_HOLD_weight_GOLD"] = 0.0
    panel["SPY_BUY_HOLD_weight_IEF"] = 0.0
    panel["SPY_BUY_HOLD_weight_CASH"] = 0.0
    panel["SPY_BUY_HOLD_return"] = panel["SPY_return"]
    panel["SPY_BUY_HOLD_nav"] = (1 + panel["SPY_BUY_HOLD_return"]).cumprod()
    panel["turnover_SPY_BUY_HOLD"] = 0.0
    panel["transaction_cost_SPY_BUY_HOLD"] = 0.0

    if "BACKBONE_V2_SPY_CASH_weight_SPY" not in panel.columns:
        raise ValueError("BACKBONE_V2_SPY_CASH weights missing.")
    panel["BACKBONE_V2_SPY_CASH_weight_GOLD"] = 0.0
    panel["BACKBONE_V2_SPY_CASH_weight_IEF"] = 0.0
    if "BACKBONE_V2_SPY_CASH_return" not in panel.columns:
        panel["BACKBONE_V2_SPY_CASH_return"] = (
            panel["BACKBONE_V2_SPY_CASH_weight_SPY"] * panel["SPY_return"]
            + panel["BACKBONE_V2_SPY_CASH_weight_CASH"] * panel["CASH_return"]
        )
    if "BACKBONE_V2_SPY_CASH_nav" not in panel.columns:
        panel["BACKBONE_V2_SPY_CASH_nav"] = (1 + panel["BACKBONE_V2_SPY_CASH_return"]).cumprod()
    if "turnover_BACKBONE_V2_SPY_CASH" not in panel.columns:
        panel["turnover_BACKBONE_V2_SPY_CASH"] = 0.0
        panel["transaction_cost_BACKBONE_V2_SPY_CASH"] = 0.0

    static_603010 = pd.DataFrame({"SPY": 0.60, "GOLD": 0.30, "IEF": 0.10, "CASH": 0.0}, index=panel.index)
    static_702010 = pd.DataFrame({"SPY": 0.70, "GOLD": 0.20, "IEF": 0.0, "CASH": 0.10}, index=panel.index)

    for strategy_name in ["REGIME_HEDGE_V1_ORIGINAL", "REGIME_HEDGE_V2_STEEP_CASH", "REGIME_HEDGE_V2_STEEP_GOLD"]:
        rules_df = build_allocation_rules(panel, strategy_name)
        target = rules_df[[f"{strategy_name}_target_{asset}" for asset in ASSETS]].rename(columns={f"{strategy_name}_target_{asset}": asset for asset in ASSETS})
        panel = run_multi_asset_backtest(panel, strategy_name, target, monthly_rebalance=CONFIG["monthly_rebalance"])

    panel = run_multi_asset_backtest(panel, "STATIC_60_30_10", static_603010, monthly_rebalance=True)
    panel = run_multi_asset_backtest(panel, "STATIC_70_20_10", static_702010, monthly_rebalance=True)

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
    ]
    for strategy in STRATEGIES:
        daily_cols += [
            f"{strategy}_weight_SPY",
            f"{strategy}_weight_GOLD",
            f"{strategy}_weight_IEF",
            f"{strategy}_weight_CASH",
            f"{strategy}_return",
            f"{strategy}_nav",
            f"turnover_{strategy}",
            f"transaction_cost_{strategy}",
        ]
    panel[daily_cols].to_csv(CONFIG["output_dir"] / "daily_backtest_panel.csv", index=False)

    perf = pd.DataFrame([compute_performance_metrics(panel, strategy) for strategy in STRATEGIES])
    perf.to_csv(CONFIG["output_dir"] / "performance_summary.csv", index=False)

    crisis = compute_crisis_performance(panel, STRATEGIES)
    crisis.to_csv(CONFIG["output_dir"] / "crisis_performance.csv", index=False)

    cross_rows = []
    for cross_state, sub in panel.groupby("cross_state", dropna=False):
        for strategy in STRATEGIES:
            ret = sub[f"{strategy}_return"].fillna(0.0)
            rf = sub["CASH_return"].fillna(0.0)
            nav = (1 + ret).cumprod()
            years = len(ret) / 252
            ann = nav.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
            vol = ret.std(ddof=0) * math.sqrt(252)
            sharpe = (ret - rf).mean() / (ret - rf).std(ddof=0) * math.sqrt(252) if (ret - rf).std(ddof=0) > 0 else np.nan
            cross_rows.append(
                {
                    "cross_state": cross_state,
                    "strategy": strategy,
                    "n_obs": len(sub),
                    "annualized_return": ann,
                    "volatility": vol,
                    "Sharpe": sharpe,
                    "max_drawdown": _max_drawdown(ret),
                    "avg_weight_SPY": sub[f"{strategy}_weight_SPY"].mean(),
                    "avg_weight_GOLD": sub[f"{strategy}_weight_GOLD"].mean(),
                    "avg_weight_IEF": sub[f"{strategy}_weight_IEF"].mean(),
                    "avg_weight_CASH": sub[f"{strategy}_weight_CASH"].mean(),
                }
            )
    pd.DataFrame(cross_rows).to_csv(CONFIG["output_dir"] / "performance_by_cross_state.csv", index=False)

    steep_episodes = extract_steep_risk_episodes(panel)
    steep_compare = compare_steep_risk_assets(panel, steep_episodes)
    steep_compare.to_csv(CONFIG["output_dir"] / "steep_risk_episode_comparison.csv", index=False)

    plot_equity_curves(panel)
    plot_drawdowns(panel)
    plot_performance_bars(perf)
    plot_weight_stacks(panel, "REGIME_HEDGE_V2_STEEP_CASH", "weight_stack_V2_STEEP_CASH.png")
    plot_weight_stacks(panel, "REGIME_HEDGE_V2_STEEP_GOLD", "weight_stack_V2_STEEP_GOLD.png")
    plot_case_studies(panel)
    plot_steep_risk_boxplot(steep_compare)
    write_markdown_report(perf, crisis, steep_compare)

    p = perf.set_index("strategy")
    v1 = p.loc["REGIME_HEDGE_V1_ORIGINAL"]
    cash = p.loc["REGIME_HEDGE_V2_STEEP_CASH"]
    gold = p.loc["REGIME_HEDGE_V2_STEEP_GOLD"]
    backbone = p.loc["BACKBONE_V2_SPY_CASH"]
    crisis_pivot = crisis.pivot(index="period", columns="strategy", values="cumulative_return") if not crisis.empty else pd.DataFrame()

    print(f"1. V1_ORIGINAL AnnRet / Sharpe / MaxDD / Final NAV: {v1['annualized_return']:.2%} / {v1['sharpe_ratio']:.2f} / {v1['max_drawdown']:.2%} / {v1['final_nav']:.2f}")
    print(f"2. V2_STEEP_CASH AnnRet / Sharpe / MaxDD / Final NAV: {cash['annualized_return']:.2%} / {cash['sharpe_ratio']:.2f} / {cash['max_drawdown']:.2%} / {cash['final_nav']:.2f}")
    print(f"3. V2_STEEP_GOLD AnnRet / Sharpe / MaxDD / Final NAV: {gold['annualized_return']:.2%} / {gold['sharpe_ratio']:.2f} / {gold['max_drawdown']:.2%} / {gold['final_nav']:.2f}")
    print(f"4. BACKBONE_V2_SPY_CASH AnnRet / Sharpe / MaxDD / Final NAV: {backbone['annualized_return']:.2%} / {backbone['sharpe_ratio']:.2f} / {backbone['max_drawdown']:.2%} / {backbone['final_nav']:.2f}")
    for idx, label in [("2022", "5. 2022"), ("2008_GFC", "6. 2008"), ("2015_2016", "7. 2015-2016"), ("2025_PULLBACK", "8. 2025 pullback")]:
        if idx in crisis_pivot.index:
            vals = crisis_pivot.loc[idx, ["REGIME_HEDGE_V1_ORIGINAL", "REGIME_HEDGE_V2_STEEP_CASH", "REGIME_HEDGE_V2_STEEP_GOLD", "BACKBONE_V2_SPY_CASH"]].to_dict()
            print(f"{label} cumulative return comparison: {vals}")
    if not steep_compare.empty:
        avg_assets = {
            asset: steep_compare[f"{asset}_return"].mean()
            for asset in ASSETS
        }
        best_asset = max(avg_assets, key=avg_assets.get)
        print(f"9. STEEP_RISK average best asset: {best_asset} {avg_assets}")
    else:
        print("9. STEEP_RISK average best asset: insufficient sample")
    best = max(
        ["REGIME_HEDGE_V1_ORIGINAL", "REGIME_HEDGE_V2_STEEP_CASH", "REGIME_HEDGE_V2_STEEP_GOLD"],
        key=lambda s: (p.loc[s, "sharpe_ratio"], p.loc[s, "annualized_return"]),
    )
    print(f"10. Recommended next allocation: {best}")


if __name__ == "__main__":
    main()
