"""STEEP risk IEF/GOLD mix backtest."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIG = {
    "output_dir": Path("results/regime_aware_hedge_allocation_steep_mix"),
    "figure_dir": Path("figures/regime_aware_hedge_allocation_steep_mix"),
    "one_way_cost_bps": 5,
    "monthly_rebalance": True,
    "timing_backbone": "BACKBONE_V2_UPGRADED",
    "steep_risk_gold_weights": [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
    "fixed_rules": {
        "INVERTED": {"SPY": 0.70, "GOLD": 0.20, "IEF": 0.00, "CASH": 0.10},
        "FLAT_NON_RISK": {"SPY": 0.60, "GOLD": 0.40, "IEF": 0.00, "CASH": 0.00},
        "FLAT_RISK": {"SPY": 0.00, "GOLD": 1.00, "IEF": 0.00, "CASH": 0.00},
        "STEEP_NON_RISK": {"SPY": 1.00, "GOLD": 0.00, "IEF": 0.00, "CASH": 0.00},
        "FALLBACK_NON_RISK": {"SPY": 0.80, "GOLD": 0.00, "IEF": 0.00, "CASH": 0.20},
        "FALLBACK_RISK": {"SPY": 0.00, "GOLD": 0.00, "IEF": 0.00, "CASH": 1.00},
    },
}

PANEL_CANDIDATES = [
    Path("results/regime_aware_hedge_allocation_steep_test/daily_backtest_panel.csv"),
    Path("results/regime_aware_hedge_allocation_v1/daily_backtest_panel.csv"),
    Path("results/spy_cash_backbone_upgrade_ablation/daily_backtest_panel.csv"),
]

ASSET_PANEL_CANDIDATES = [
    Path("results/regime_aware_hedge_allocation_steep_test/daily_backtest_panel.csv"),
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
MIX_STRATEGIES = []
for gw in CONFIG["steep_risk_gold_weights"]:
    iw = 1.0 - gw
    MIX_STRATEGIES.append(f"STEEP_MIX_{int(round(iw * 100))}_IEF_{int(round(gw * 100))}_GOLD")

BENCHMARK_STRATEGIES = [
    "SPY_BUY_HOLD",
    "BACKBONE_V2_SPY_CASH",
    "REGIME_HEDGE_V1_ORIGINAL",
    "REGIME_HEDGE_STEEP_CASH",
    "STATIC_70_20_10",
    "STATIC_60_30_10",
]
ALL_STRATEGIES = BENCHMARK_STRATEGIES + MIX_STRATEGIES


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
    return period.ne(period.shift(1, fill_value=period.iloc[0]))


def _max_drawdown(ret: pd.Series) -> float:
    nav = (1 + ret.fillna(0.0)).cumprod()
    if nav.empty:
        return np.nan
    return float((nav / nav.cummax() - 1).min())


def _build_mix_rules() -> Dict[str, Dict[str, Dict[str, float]]]:
    rules = {
        "REGIME_HEDGE_V1_ORIGINAL": {
            "INVERTED": {"SPY": 0.70, "GOLD": 0.20, "IEF": 0.00, "CASH": 0.10},
            "FLAT_NON_RISK": {"SPY": 0.60, "GOLD": 0.40, "IEF": 0.00, "CASH": 0.00},
            "FLAT_RISK": {"SPY": 0.00, "GOLD": 1.00, "IEF": 0.00, "CASH": 0.00},
            "STEEP_NON_RISK": {"SPY": 0.90, "GOLD": 0.00, "IEF": 0.10, "CASH": 0.00},
            "STEEP_RISK": {"SPY": 0.00, "GOLD": 0.00, "IEF": 1.00, "CASH": 0.00},
            "FALLBACK_NON_RISK": {"SPY": 0.80, "GOLD": 0.00, "IEF": 0.00, "CASH": 0.20},
            "FALLBACK_RISK": {"SPY": 0.00, "GOLD": 0.00, "IEF": 0.00, "CASH": 1.00},
        },
        "REGIME_HEDGE_STEEP_CASH": {
            "INVERTED": CONFIG["fixed_rules"]["INVERTED"],
            "FLAT_NON_RISK": CONFIG["fixed_rules"]["FLAT_NON_RISK"],
            "FLAT_RISK": CONFIG["fixed_rules"]["FLAT_RISK"],
            "STEEP_NON_RISK": CONFIG["fixed_rules"]["STEEP_NON_RISK"],
            "STEEP_RISK": {"SPY": 0.00, "GOLD": 0.00, "IEF": 0.00, "CASH": 1.00},
            "FALLBACK_NON_RISK": CONFIG["fixed_rules"]["FALLBACK_NON_RISK"],
            "FALLBACK_RISK": CONFIG["fixed_rules"]["FALLBACK_RISK"],
        },
    }
    for gw in CONFIG["steep_risk_gold_weights"]:
        iw = 1.0 - gw
        name = f"STEEP_MIX_{int(round(iw * 100))}_IEF_{int(round(gw * 100))}_GOLD"
        rules[name] = {
            "INVERTED": CONFIG["fixed_rules"]["INVERTED"],
            "FLAT_NON_RISK": CONFIG["fixed_rules"]["FLAT_NON_RISK"],
            "FLAT_RISK": CONFIG["fixed_rules"]["FLAT_RISK"],
            "STEEP_NON_RISK": CONFIG["fixed_rules"]["STEEP_NON_RISK"],
            "STEEP_RISK": {"SPY": 0.00, "GOLD": gw, "IEF": iw, "CASH": 0.00},
            "FALLBACK_NON_RISK": CONFIG["fixed_rules"]["FALLBACK_NON_RISK"],
            "FALLBACK_RISK": CONFIG["fixed_rules"]["FALLBACK_RISK"],
        }
    return rules


STRATEGY_RULES = _build_mix_rules()


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
        for candidates, final in [
            (["GOLD_return", "GOLD_RETURN", "GLD_return", "GLD_RETURN"], "GOLD_return"),
            (["IEF_return", "IEF_RETURN"], "IEF_return"),
        ]:
            col = next((c for c in candidates if c in src.columns), None)
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
    if "BACKBONE_V2_SPY_CASH_weight_SPY" in df.columns:
        if "timing_state" not in df.columns:
            df["timing_state"] = np.where(pd.to_numeric(df["BACKBONE_V2_SPY_CASH_weight_SPY"], errors="coerce") >= 0.5, "NON_RISK", "RISK")
        else:
            inferred = pd.Series(
                np.where(pd.to_numeric(df["BACKBONE_V2_SPY_CASH_weight_SPY"], errors="coerce") >= 0.5, "NON_RISK", "RISK"),
                index=df.index,
            )
            df["timing_state"] = df["timing_state"].fillna(inferred)
    elif "BACKBONE_V2_UPGRADED_weight_spy" in df.columns:
        df["BACKBONE_V2_SPY_CASH_weight_SPY"] = pd.to_numeric(df["BACKBONE_V2_UPGRADED_weight_spy"], errors="coerce").fillna(1.0)
        df["BACKBONE_V2_SPY_CASH_weight_CASH"] = pd.to_numeric(
            df.get("BACKBONE_V2_UPGRADED_weight_cash", 1 - df["BACKBONE_V2_SPY_CASH_weight_SPY"]),
            errors="coerce",
        ).fillna(1 - df["BACKBONE_V2_SPY_CASH_weight_SPY"])
        df["BACKBONE_V2_SPY_CASH_return"] = pd.to_numeric(df["BACKBONE_V2_UPGRADED_return"], errors="coerce").fillna(0.0)
        df["BACKBONE_V2_SPY_CASH_nav"] = (1 + df["BACKBONE_V2_SPY_CASH_return"]).cumprod()
        df["timing_state"] = np.where(df["BACKBONE_V2_SPY_CASH_weight_SPY"] >= 0.5, "NON_RISK", "RISK")
    else:
        raise ValueError("Missing BACKBONE_V2 timing state/weights.")
    if "cross_state" not in df.columns:
        df["cross_state"] = df["macro_regime_confirmed"] + "_" + df["timing_state"]
    if "entry_reason" not in df.columns:
        df["entry_reason"] = ""
    return df


def build_allocation_rules_for_steep_mix(panel: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    rules = STRATEGY_RULES[strategy_name]
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
    first_month = _is_first_trading_day_of_month(df["date"])
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
        if i > 0 and (target_changed or (monthly_rebalance and bool(first_month.iloc[i]))):
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


def compare_steep_risk_assets_and_mixes(panel: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    compare_strats = MIX_STRATEGIES + ["REGIME_HEDGE_V1_ORIGINAL", "REGIME_HEDGE_STEEP_CASH"]
    for _, ep in episodes.iterrows():
        sub = panel[(panel["date"] >= ep["start_date"]) & (panel["date"] <= ep["end_date"])].copy()
        row = dict(ep)
        for asset in ASSETS:
            ret = sub[f"{asset}_return"].fillna(0.0)
            nav = (1 + ret).cumprod()
            row[f"{asset}_return"] = nav.iloc[-1] - 1
            row[f"{asset}_max_drawdown"] = _max_drawdown(ret)
        for strategy in compare_strats:
            ret = sub[f"{strategy}_return"].fillna(0.0)
            nav = (1 + ret).cumprod()
            row[f"{strategy}_return"] = nav.iloc[-1] - 1
            row[f"{strategy}_max_drawdown"] = _max_drawdown(ret)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_equity_curves(panel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    show = ["SPY_BUY_HOLD", "BACKBONE_V2_SPY_CASH", "REGIME_HEDGE_V1_ORIGINAL", "STEEP_MIX_100_IEF_0_GOLD", "STEEP_MIX_80_IEF_20_GOLD", "STEEP_MIX_60_IEF_40_GOLD", "STEEP_MIX_50_IEF_50_GOLD", "STEEP_MIX_0_IEF_100_GOLD"]
    for strategy in show:
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
    show = ["SPY_BUY_HOLD", "BACKBONE_V2_SPY_CASH", "REGIME_HEDGE_V1_ORIGINAL", "STEEP_MIX_100_IEF_0_GOLD", "STEEP_MIX_80_IEF_20_GOLD", "STEEP_MIX_60_IEF_40_GOLD", "STEEP_MIX_50_IEF_50_GOLD", "STEEP_MIX_0_IEF_100_GOLD"]
    for strategy in show:
        nav = panel[f"{strategy}_nav"]
        ax.plot(panel["date"], nav / nav.cummax() - 1, label=strategy)
    ax.legend(ncol=2, fontsize=8)
    ax.set_title("Drawdown Comparison")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "drawdown_comparison.png", dpi=150)
    plt.close(fig)


def plot_steep_mix_sensitivity(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    metrics = [
        ("annualized_return", "annualized_return"),
        ("sharpe_ratio", "sharpe_ratio"),
        ("max_drawdown", "max_drawdown"),
        ("final_nav", "final_nav"),
        ("2022_return", "2022_return"),
        ("2008_return", "2008_return"),
    ]
    axes = axes.ravel()
    for ax, (col, title) in zip(axes, metrics):
        ax.plot(summary["gold_weight"], summary[col], marker="o")
        ax.set_title(title)
        ax.set_xlabel("gold_weight")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "steep_mix_performance_line.png", dpi=150)
    plt.close(fig)


def plot_performance_bars(perf: pd.DataFrame) -> None:
    show = perf[perf["strategy"].isin(["SPY_BUY_HOLD", "BACKBONE_V2_SPY_CASH", "REGIME_HEDGE_V1_ORIGINAL", "REGIME_HEDGE_STEEP_CASH", "STEEP_MIX_100_IEF_0_GOLD", "STEEP_MIX_60_IEF_40_GOLD", "STEEP_MIX_50_IEF_50_GOLD", "STEEP_MIX_0_IEF_100_GOLD"])].copy()
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


def plot_case_studies(panel: pd.DataFrame) -> None:
    cases = {
        "2022": ("2021-11-01", "2023-03-31"),
        "2008": ("2007-10-01", "2009-06-30"),
        "2015_2016": ("2015-05-01", "2016-03-31"),
        "2018Q4": ("2018-10-01", "2019-01-31"),
        "2025_PULLBACK": ("2025-01-01", "2025-12-31"),
    }
    show_strats = ["REGIME_HEDGE_V1_ORIGINAL", "STEEP_MIX_100_IEF_0_GOLD", "STEEP_MIX_60_IEF_40_GOLD", "STEEP_MIX_50_IEF_50_GOLD", "STEEP_MIX_0_IEF_100_GOLD"]
    for name, (start, end) in cases.items():
        sub = panel[(panel["date"] >= pd.Timestamp(start)) & (panel["date"] <= pd.Timestamp(end))].copy()
        if sub.empty:
            continue
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1.2, 1.2, 0.8]})
        for asset in ASSETS:
            axes[0].plot(sub["date"], sub[f"{asset}_NAV"] / sub[f"{asset}_NAV"].iloc[0], label=asset)
        axes[0].legend(fontsize=8, ncol=4)
        for strategy in show_strats:
            axes[1].plot(sub["date"], sub[f"{strategy}_nav"] / sub[f"{strategy}_nav"].iloc[0], label=strategy)
        axes[1].legend(fontsize=8, ncol=2)
        axes[2].stackplot(
            sub["date"],
            sub["STEEP_MIX_60_IEF_40_GOLD_weight_SPY"],
            sub["STEEP_MIX_60_IEF_40_GOLD_weight_GOLD"],
            sub["STEEP_MIX_60_IEF_40_GOLD_weight_IEF"],
            sub["STEEP_MIX_60_IEF_40_GOLD_weight_CASH"],
            labels=ASSETS,
        )
        axes[2].legend(fontsize=7, ncol=4)
        axes[3].plot(sub["date"], sub["timing_state"].map({"NON_RISK": 0, "RISK": 1}), label="timing_state")
        axes[3].legend(fontsize=8)
        axes[3].set_yticks([0, 1], ["NON_RISK", "RISK"])
        fig.tight_layout()
        fig.savefig(CONFIG["figure_dir"] / f"crisis_case_study_{name}.png", dpi=150)
        plt.close(fig)


def plot_steep_risk_boxplot(compare_df: pd.DataFrame) -> None:
    if compare_df.empty:
        return
    rows = []
    show = ["SPY", "CASH", "GOLD", "IEF"] + MIX_STRATEGIES
    for _, row in compare_df.iterrows():
        for name in show:
            rows.append(
                {
                    "name": name,
                    "episode_return": row[f"{name}_return"],
                    "episode_max_drawdown": row[f"{name}_max_drawdown"],
                }
            )
    plot_df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    plot_df.boxplot(column="episode_return", by="name", ax=axes[0], grid=False)
    plot_df.boxplot(column="episode_max_drawdown", by="name", ax=axes[1], grid=False)
    axes[0].set_title("Episode Return")
    axes[1].set_title("Episode Max Drawdown")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "steep_risk_asset_performance_boxplot.png", dpi=150)
    plt.close(fig)


def write_markdown_report(perf: pd.DataFrame, steep_mix_summary: pd.DataFrame) -> None:
    p = perf.set_index("strategy")
    v1 = p.loc["REGIME_HEDGE_V1_ORIGINAL"]
    backbone = p.loc["BACKBONE_V2_SPY_CASH"]
    lines = [
        "# STEEP_RISK_IEF_GOLD_MIX_REPORT",
        "",
        "## 1. Purpose",
        "This round fixes STEEP_NON_RISK = 100% SPY and tests whether mixing GOLD into STEEP_RISK improves full-sample performance and 2022 behavior without giving up too much 2008 defense.",
        "",
        "## 2. Strategy definitions",
        "- V1 original",
        "- REGIME_HEDGE_STEEP_CASH",
        "- STEEP mix versions from 100/0 to 0/100 IEF/GOLD",
        "- SPY/CASH backbone",
        "",
        "## 3. Main performance comparison",
        f"- REGIME_HEDGE_V1_ORIGINAL: AnnRet {v1['annualized_return']:.2%}, Sharpe {v1['sharpe_ratio']:.2f}, MaxDD {v1['max_drawdown']:.2%}, Final NAV {v1['final_nav']:.2f}",
        f"- BACKBONE_V2_SPY_CASH: AnnRet {backbone['annualized_return']:.2%}, Sharpe {backbone['sharpe_ratio']:.2f}, MaxDD {backbone['max_drawdown']:.2%}, Final NAV {backbone['final_nav']:.2f}",
        "",
        "## 4. STEEP_RISK mix sensitivity",
        "See `steep_mix_summary.csv` and `steep_mix_performance_line.png` for the IEF/GOLD frontier across full-sample metrics and key crisis windows.",
        "",
        "## 5. Crisis period analysis",
        "The main tradeoff is 2008/GFC defense versus 2022 rising-rate resilience.",
        "",
        "## 6. Episode-level STEEP_RISK analysis",
        "Use `steep_risk_episode_comparison.csv` to see whether mix improvements are broad or dominated by a few episodes.",
        "",
        "## 7. Interpretation",
        "If more GOLD raises return but materially worsens MaxDD, the mix is behaving like a return enhancer rather than a robust hedge.",
        "",
        "## 8. Recommendation",
        "Choose between keeping pure IEF, adding a small GOLD sleeve such as 80/20 or 60/40, or addressing 2022 through a separate rule rather than through STEEP_RISK mix changes.",
        "",
        "## 9. Caveats",
        "- STEEP_RISK sample size is limited.",
        "- 2022 is a special rising-rate shock.",
        "- Gold hedge has path risk.",
        "- No risk budgeting or volatility targeting was used.",
    ]
    (CONFIG["output_dir"] / "STEEP_RISK_IEF_GOLD_MIX_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


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
        panel["BACKBONE_V2_SPY_CASH_return"] = panel["BACKBONE_V2_SPY_CASH_weight_SPY"] * panel["SPY_return"] + panel["BACKBONE_V2_SPY_CASH_weight_CASH"] * panel["CASH_return"]
    if "BACKBONE_V2_SPY_CASH_nav" not in panel.columns:
        panel["BACKBONE_V2_SPY_CASH_nav"] = (1 + panel["BACKBONE_V2_SPY_CASH_return"]).cumprod()
    if "turnover_BACKBONE_V2_SPY_CASH" not in panel.columns:
        panel["turnover_BACKBONE_V2_SPY_CASH"] = 0.0
        panel["transaction_cost_BACKBONE_V2_SPY_CASH"] = 0.0

    static_702010 = pd.DataFrame({"SPY": 0.70, "GOLD": 0.20, "IEF": 0.00, "CASH": 0.10}, index=panel.index)
    static_603010 = pd.DataFrame({"SPY": 0.60, "GOLD": 0.30, "IEF": 0.10, "CASH": 0.00}, index=panel.index)

    for strategy_name in ["REGIME_HEDGE_V1_ORIGINAL", "REGIME_HEDGE_STEEP_CASH"] + MIX_STRATEGIES:
        rules_df = build_allocation_rules_for_steep_mix(panel, strategy_name)
        target = rules_df[[f"{strategy_name}_target_{asset}" for asset in ASSETS]].rename(columns={f"{strategy_name}_target_{asset}": asset for asset in ASSETS})
        panel = run_multi_asset_backtest(panel, strategy_name, target, monthly_rebalance=CONFIG["monthly_rebalance"])

    panel = run_multi_asset_backtest(panel, "STATIC_70_20_10", static_702010, monthly_rebalance=True)
    panel = run_multi_asset_backtest(panel, "STATIC_60_30_10", static_603010, monthly_rebalance=True)

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
    for strategy in ALL_STRATEGIES:
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

    perf = pd.DataFrame([compute_performance_metrics(panel, strategy) for strategy in ALL_STRATEGIES])
    perf.to_csv(CONFIG["output_dir"] / "performance_summary.csv", index=False)

    crisis = compute_crisis_performance(panel, ALL_STRATEGIES)
    crisis.to_csv(CONFIG["output_dir"] / "crisis_performance.csv", index=False)

    cross_rows = []
    for cross_state, sub in panel.groupby("cross_state", dropna=False):
        for strategy in ALL_STRATEGIES:
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
    steep_compare = compare_steep_risk_assets_and_mixes(panel, steep_episodes)
    steep_compare.to_csv(CONFIG["output_dir"] / "steep_risk_episode_comparison.csv", index=False)

    summary_rows = []
    crisis_pivot = crisis.pivot(index="strategy", columns="period", values="cumulative_return") if not crisis.empty else pd.DataFrame()
    for gw in CONFIG["steep_risk_gold_weights"]:
        iw = 1.0 - gw
        strategy = f"STEEP_MIX_{int(round(iw * 100))}_IEF_{int(round(gw * 100))}_GOLD"
        row = perf.set_index("strategy").loc[strategy]
        ep_ret = steep_compare[f"{strategy}_return"].mean() if not steep_compare.empty else np.nan
        ep_mdd = steep_compare[f"{strategy}_max_drawdown"].mean() if not steep_compare.empty else np.nan
        summary_rows.append(
            {
                "strategy": strategy,
                "ief_weight": iw,
                "gold_weight": gw,
                "annualized_return": row["annualized_return"],
                "sharpe_ratio": row["sharpe_ratio"],
                "max_drawdown": row["max_drawdown"],
                "final_nav": row["final_nav"],
                "2008_return": crisis_pivot.loc[strategy, "2008_GFC"] if "2008_GFC" in crisis_pivot.columns else np.nan,
                "2008_maxdd": crisis[(crisis["strategy"] == strategy) & (crisis["period"] == "2008_GFC")]["max_drawdown"].iloc[0] if ((crisis["strategy"] == strategy) & (crisis["period"] == "2008_GFC")).any() else np.nan,
                "2022_return": crisis_pivot.loc[strategy, "2022"] if "2022" in crisis_pivot.columns else np.nan,
                "2022_maxdd": crisis[(crisis["strategy"] == strategy) & (crisis["period"] == "2022")]["max_drawdown"].iloc[0] if ((crisis["strategy"] == strategy) & (crisis["period"] == "2022")).any() else np.nan,
                "2015_2016_return": crisis_pivot.loc[strategy, "2015_2016"] if "2015_2016" in crisis_pivot.columns else np.nan,
                "2025_return": crisis_pivot.loc[strategy, "2025_PULLBACK"] if "2025_PULLBACK" in crisis_pivot.columns else np.nan,
                "avg_steep_risk_episode_return": ep_ret,
                "avg_steep_risk_episode_maxdd": ep_mdd,
                "total_turnover": row["total_turnover"],
            }
        )
    steep_mix_summary = pd.DataFrame(summary_rows)
    steep_mix_summary.to_csv(CONFIG["output_dir"] / "steep_mix_summary.csv", index=False)

    plot_equity_curves(panel)
    plot_drawdowns(panel)
    plot_steep_mix_sensitivity(steep_mix_summary)
    plot_performance_bars(perf)
    plot_case_studies(panel)
    plot_steep_risk_boxplot(steep_compare)
    write_markdown_report(perf, steep_mix_summary)

    p = perf.set_index("strategy")
    v1 = p.loc["REGIME_HEDGE_V1_ORIGINAL"]
    print(f"1. V1_ORIGINAL AnnRet / Sharpe / MaxDD / Final NAV: {v1['annualized_return']:.2%} / {v1['sharpe_ratio']:.2f} / {v1['max_drawdown']:.2%} / {v1['final_nav']:.2f}")
    for idx, strategy in enumerate(["STEEP_MIX_100_IEF_0_GOLD", "STEEP_MIX_80_IEF_20_GOLD", "STEEP_MIX_60_IEF_40_GOLD", "STEEP_MIX_50_IEF_50_GOLD", "STEEP_MIX_0_IEF_100_GOLD"], start=2):
        row = p.loc[strategy]
        print(f"{idx}. {strategy} AnnRet / Sharpe / MaxDD / Final NAV: {row['annualized_return']:.2%} / {row['sharpe_ratio']:.2f} / {row['max_drawdown']:.2%} / {row['final_nav']:.2f}")
    best_2022 = steep_mix_summary.sort_values("2022_return", ascending=False).iloc[0]
    best_2008 = steep_mix_summary.sort_values("2008_return", ascending=False).iloc[0]
    best_sharpe = steep_mix_summary.sort_values("sharpe_ratio", ascending=False).iloc[0]
    best_maxdd = steep_mix_summary.sort_values("max_drawdown", ascending=False).iloc[0]
    print(f"7. 2022 best version: {best_2022['strategy']} ({best_2022['2022_return']:.2%})")
    print(f"8. 2008 best version: {best_2008['strategy']} ({best_2008['2008_return']:.2%})")
    print(f"9. Full-sample highest Sharpe: {best_sharpe['strategy']} ({best_sharpe['sharpe_ratio']:.2f})")
    print(f"10. Lowest MaxDD: {best_maxdd['strategy']} ({best_maxdd['max_drawdown']:.2%})")
    rec = best_sharpe["strategy"]
    if best_2022["gold_weight"] > best_sharpe["gold_weight"] and best_sharpe["max_drawdown"] < -0.22:
        rec = "keep REGIME_HEDGE_V1_ORIGINAL or use a small GOLD sleeve like 80/20 only if 2022 resilience is prioritized"
    print(f"11. Recommended next allocation: {rec}")


if __name__ == "__main__":
    main()
