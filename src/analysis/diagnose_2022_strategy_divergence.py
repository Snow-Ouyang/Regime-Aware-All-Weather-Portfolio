"""Diagnose 2022 Russia-Ukraine war / rate-shock divergence across strategy variants."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIG = {
    "output_dir": Path("results/diagnose_2022_strategy_divergence"),
    "figure_dir": Path("figures/diagnose_2022_strategy_divergence"),
}

PANEL_CANDIDATES = [
    Path("results/mature_strategy_with_steep_commodity_overlay/daily_backtest_panel.csv"),
    Path("results/backbone_v2_with_steep_commodity_stress/daily_backtest_panel.csv"),
    Path("results/regime_aware_risk_parity_allocation/daily_backtest_panel.csv"),
    Path("results/spy_cash_backbone_upgrade_ablation/daily_backtest_panel.csv"),
]

EVENT_LOG_CANDIDATES = [
    Path("results/mature_strategy_with_steep_commodity_overlay/risk_state_event_log.csv"),
    Path("results/backbone_v2_with_steep_commodity_stress/risk_state_event_log.csv"),
    Path("results/spy_cash_backbone_upgrade_ablation/risk_state_event_log.csv"),
]

FOCUS_STRATEGIES = [
    "SPY_BUY_HOLD",
    "BACKBONE_V2_SPY_CASH",
    "REGIME_HEDGE_V1_ORIGINAL",
    "MATURE_BASELINE_REGIME_HEDGE_INV_VOL",
    "MATURE_FULL_ONE_RET60",
    "MATURE_FULL_ONE_CREDIT",
    "MATURE_FULL_ONE_DD5",
    "MATURE_FULL_ALL_THREE",
    "MATURE_OVERLAY40_ONE_RET60",
    "MATURE_OVERLAY40_ALL_THREE",
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


def load_panel() -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Path]:
    panel_path = next((p for p in PANEL_CANDIDATES if p.exists()), None)
    if panel_path is None:
        raise FileNotFoundError("No diagnostic panel found.")
    panel = _read_csv(panel_path)
    event_log = None
    for p in EVENT_LOG_CANDIDATES:
        if p.exists():
            event_log = pd.read_csv(p)
            if "event_date" in event_log.columns:
                event_log["event_date"] = pd.to_datetime(event_log["event_date"])
            break
    print(f"Loaded panel: {panel_path}")
    return panel, event_log, panel_path


def identify_strategy_columns(panel: pd.DataFrame) -> List[str]:
    strategies = []
    for s in FOCUS_STRATEGIES:
        if f"{s}_return" in panel.columns or f"{s}_nav" in panel.columns:
            strategies.append(s)
    if not strategies:
        raise ValueError("No focus strategy columns found.")
    return strategies


def define_2022_windows(panel: pd.DataFrame) -> pd.DataFrame:
    windows = {
        "full_2022_rate_war_window": ("2021-11-01", "2023-03-31"),
        "russia_ukraine_initial_shock": ("2022-02-01", "2022-03-31"),
        "first_half_2022": ("2022-01-01", "2022-06-30"),
        "second_half_2022": ("2022-07-01", "2022-12-31"),
    }
    sub = panel[(panel["date"] >= pd.Timestamp("2021-11-01")) & (panel["date"] <= pd.Timestamp("2023-03-31"))].copy()
    peak_idx = sub["spy_price"].idxmax()
    peak_date = sub.loc[peak_idx, "date"]
    trough_sub = sub[sub["date"] >= peak_date]
    trough_idx = trough_sub["spy_price"].idxmin()
    trough_date = trough_sub.loc[trough_idx, "date"]
    windows["peak_to_trough_2022"] = (str(peak_date.date()), str(trough_date.date()))
    out = pd.DataFrame(
        [{"window_name": name, "start_date": pd.Timestamp(start), "end_date": pd.Timestamp(end)} for name, (start, end) in windows.items()]
    )
    out.to_csv(CONFIG["output_dir"] / "window_definitions_2022.csv", index=False)
    return out


def _state_series(panel: pd.DataFrame, strategy: str) -> pd.Series:
    col = f"{strategy}_risk_state"
    if col in panel.columns:
        return panel[col].fillna("NON_RISK").astype(str)
    if strategy == "BACKBONE_V2_SPY_CASH":
        if "BACKBONE_V2_SPY_CASH_weight_CASH" in panel.columns:
            return np.where(pd.to_numeric(panel["BACKBONE_V2_SPY_CASH_weight_CASH"], errors="coerce").fillna(0) > 0.5, "FULL_RISK", "NON_RISK")
    return pd.Series("NON_RISK", index=panel.index)


def _overlay_series(panel: pd.DataFrame, strategy: str) -> pd.Series:
    col = f"{strategy}_overlay_state"
    if col in panel.columns:
        return panel[col].fillna(False).astype(bool)
    return pd.Series(False, index=panel.index)


def _full_risk_series(panel: pd.DataFrame, strategy: str) -> pd.Series:
    col = f"{strategy}_full_risk_state"
    if col in panel.columns:
        return panel[col].fillna(False).astype(bool)
    state = _state_series(panel, strategy)
    if strategy == "BACKBONE_V2_SPY_CASH":
        return state.astype(str).eq("FULL_RISK") | state.astype(str).eq("RISK")
    return state.astype(str).eq("FULL_RISK")


def _weight_col(strategy: str, asset: str) -> str:
    return f"{strategy}_weight_{asset}"


def compute_crisis_performance(panel: pd.DataFrame, strategies: List[str], windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, w in windows.iterrows():
        sub = panel[(panel["date"] >= w["start_date"]) & (panel["date"] <= w["end_date"])].copy()
        if sub.empty:
            continue
        for strategy in strategies:
            ret_col = f"{strategy}_return"
            nav_col = f"{strategy}_nav"
            if ret_col not in sub.columns:
                continue
            ret = sub[ret_col].fillna(0.0)
            nav = (1 + ret).cumprod()
            rf = sub["CASH_return"].fillna(0.0)
            excess = ret - rf
            rows.append(
                {
                    "strategy": strategy,
                    "window_name": w["window_name"],
                    "start_date": sub["date"].iloc[0],
                    "end_date": sub["date"].iloc[-1],
                    "cumulative_return": nav.iloc[-1] - 1,
                    "annualized_return": nav.iloc[-1] ** (252 / len(sub)) - 1 if len(sub) > 0 else np.nan,
                    "volatility": ret.std(ddof=0) * math.sqrt(252),
                    "Sharpe": excess.mean() / excess.std(ddof=0) * math.sqrt(252) if excess.std(ddof=0) > 0 else np.nan,
                    "max_drawdown": (nav / nav.cummax() - 1).min(),
                    "final_nav_relative": nav.iloc[-1],
                    "avg_weight_SPY": sub.get(_weight_col(strategy, "SPY"), pd.Series(np.nan, index=sub.index)).mean(),
                    "avg_weight_GOLD": sub.get(_weight_col(strategy, "GOLD"), pd.Series(np.nan, index=sub.index)).mean(),
                    "avg_weight_CMDTY_FUT": sub.get(_weight_col(strategy, "CMDTY_FUT"), pd.Series(np.nan, index=sub.index)).mean(),
                    "avg_weight_IEF": sub.get(_weight_col(strategy, "IEF"), pd.Series(np.nan, index=sub.index)).mean(),
                    "avg_weight_CASH": sub.get(_weight_col(strategy, "CASH"), pd.Series(np.nan, index=sub.index)).mean(),
                    "time_in_full_risk": _full_risk_series(sub, strategy).mean(),
                    "time_in_overlay": _overlay_series(sub, strategy).mean(),
                    "time_in_cash": sub.get(_weight_col(strategy, "CASH"), pd.Series(0.0, index=sub.index)).mean(),
                    "number_of_entries": int((_state_series(sub, strategy) != _state_series(sub, strategy).shift(1, fill_value="NON_RISK")).sum()),
                    "number_of_exits": int(
                        (
                            _state_series(sub, strategy)
                            .shift(1, fill_value="NON_RISK")
                            .isin(["FULL_RISK", "OVERLAY", "RISK"])
                            & (_state_series(sub, strategy) == "NON_RISK")
                        ).sum()
                    ),
                    "total_turnover": sub.get(f"{strategy}_turnover", pd.Series(0.0, index=sub.index)).fillna(0.0).sum(),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(CONFIG["output_dir"] / "crisis_performance_2022_detailed.csv", index=False)
    return out


def identify_yellow_line_strategy(crisis_perf: pd.DataFrame) -> pd.DataFrame:
    full = crisis_perf[crisis_perf["window_name"] == "full_2022_rate_war_window"].copy()
    if full.empty:
        return pd.DataFrame()
    by_dd = full.sort_values("max_drawdown", ascending=False).head(1)
    by_ret = full.sort_values("cumulative_return", ascending=False).head(1)
    by_cash = full.sort_values("avg_weight_CASH", ascending=False).head(1)
    rows = []
    for label, sub in [
        ("yellow_orange_likely_from_legend", full[full["strategy"].eq("BACKBONE_V2_SPY_CASH")].head(1)),
        ("shallowest_maxdd", by_dd),
        ("highest_return", by_ret),
        ("highest_cash_exposure", by_cash),
    ]:
        if sub.empty:
            continue
        r = sub.iloc[0].to_dict()
        r["identification_rule"] = label
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(CONFIG["output_dir"] / "identify_yellow_line_strategy.csv", index=False)
    return out


def build_signal_timeline(panel: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "date", "macro_regime_confirmed", "monthly_either_state", "VIX_ZSCORE_120D",
        "CREDIT_SPREAD_BAA_AAA", "D_CREDIT_SPREAD_20D", "CMDTY_RET60",
        "spy_drawdown_from_previous_high", "SPY_CROSS_ABOVE_MA20",
        "FLAT_VIX_STRESS", "FLAT_CREDIT_DD5_STRESS", "STEEP_EITHER_SELL_STRESS",
        "STEEP_CREDIT_DD5_STRESS", "STEEP_CMDTY_RET60_NEG10",
        "STEEP_CMDTY_RET60_NEG10_AND_CREDIT_WIDEN", "STEEP_SPY_DD5_AND_CMDTY_RET60_NEG10",
        "BACKBONE_V2_BASELINE_ENTRY", "BACKBONE_V2_BASELINE_RISK_STATE",
    ]
    existing = [c for c in cols if c in panel.columns]
    out = panel[(panel["date"] >= pd.Timestamp("2021-11-01")) & (panel["date"] <= pd.Timestamp("2023-03-31"))][existing].copy()
    out["R3_RECOVERY"] = out.get("SPY_CROSS_ABOVE_MA20", False)
    out.to_csv(CONFIG["output_dir"] / "signal_timeline_2022.csv", index=False)
    return out


def extract_state_timeline(panel: pd.DataFrame, strategies: List[str]) -> pd.DataFrame:
    base_cols = [
        "date", "macro_regime_confirmed", "monthly_either_state", "VIX_ZSCORE_120D",
        "CREDIT_SPREAD_BAA_AAA", "D_CREDIT_SPREAD_20D", "CMDTY_RET60",
        "spy_drawdown_from_previous_high", "SPY_CROSS_ABOVE_MA20",
        "SPY_return", "GOLD_return", "IEF_return", "CASH_return", "CMDTY_FUT_return",
    ]
    sub = panel[(panel["date"] >= pd.Timestamp("2021-11-01")) & (panel["date"] <= pd.Timestamp("2023-03-31"))].copy()
    out = sub[base_cols].copy()
    for strategy in strategies:
        out[f"{strategy}_state"] = _state_series(sub, strategy)
        out[f"{strategy}_entry_reason"] = ""
        for asset in ["SPY", "GOLD", "CMDTY_FUT", "IEF", "CASH"]:
            col = _weight_col(strategy, asset)
            if col in sub.columns:
                out[col] = sub[col]
        ret_col = f"{strategy}_return"
        nav_col = f"{strategy}_nav"
        if ret_col in sub.columns:
            out[ret_col] = sub[ret_col]
        if nav_col in sub.columns:
            out[nav_col] = sub[nav_col]
            out[f"{strategy}_drawdown"] = sub[nav_col] / sub[nav_col].cummax() - 1
    out.to_csv(CONFIG["output_dir"] / "state_timeline_2022.csv", index=False)
    return out


def compute_asset_performance(panel: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, w in windows.iterrows():
        sub = panel[(panel["date"] >= w["start_date"]) & (panel["date"] <= w["end_date"])].copy()
        if sub.empty:
            continue
        for asset in ["SPY", "GOLD", "IEF", "CASH", "CMDTY_FUT"]:
            ret_col = f"{asset}_return"
            if ret_col not in sub.columns:
                continue
            ret = sub[ret_col].fillna(0.0)
            nav = (1 + ret).cumprod()
            rf = sub["CASH_return"].fillna(0.0)
            excess = ret - rf
            spy = sub["SPY_return"].fillna(0.0)
            beta = ret.cov(spy) / spy.var(ddof=0) if spy.var(ddof=0) > 0 else np.nan
            rows.append(
                {
                    "window_name": w["window_name"],
                    "asset": asset,
                    "cumulative_return": nav.iloc[-1] - 1,
                    "max_drawdown": (nav / nav.cummax() - 1).min(),
                    "volatility": ret.std(ddof=0) * math.sqrt(252),
                    "Sharpe": excess.mean() / excess.std(ddof=0) * math.sqrt(252) if excess.std(ddof=0) > 0 else np.nan,
                    "correlation_with_SPY": ret.corr(spy),
                    "beta_to_SPY": beta,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(CONFIG["output_dir"] / "asset_performance_2022.csv", index=False)
    return out


def compute_return_difference_attribution(panel: pd.DataFrame, strategies: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    baseline = "MATURE_BASELINE_REGIME_HEDGE_INV_VOL"
    sub = panel[(panel["date"] >= pd.Timestamp("2021-11-01")) & (panel["date"] <= pd.Timestamp("2023-03-31"))].copy()
    rows = []
    monthly_rows = []
    for strategy in [s for s in strategies if s != baseline]:
        ret_col = f"{strategy}_return"
        base_ret_col = f"{baseline}_return"
        if ret_col not in sub.columns or base_ret_col not in sub.columns:
            continue
        tmp = sub[["date", ret_col, base_ret_col]].copy()
        tmp["strategy_pair"] = f"{strategy}_vs_{baseline}"
        tmp["diff_return"] = tmp[ret_col] - tmp[base_ret_col]
        tmp["cum_diff_return"] = (1 + tmp["diff_return"]).cumprod() - 1
        rows.append(tmp)
        month = sub["date"].dt.to_period("M")
        for per, grp in sub.groupby(month):
            diff = grp[ret_col].fillna(0.0) - grp[base_ret_col].fillna(0.0)
            wd = {}
            for asset in ["SPY", "GOLD", "IEF", "CASH"]:
                s_col = _weight_col(strategy, asset)
                b_col = _weight_col(baseline, asset)
                wd[asset] = grp.get(s_col, pd.Series(0.0, index=grp.index)).mean() - grp.get(b_col, pd.Series(0.0, index=grp.index)).mean()
            main_asset = max(wd, key=lambda k: abs(wd[k]))
            monthly_rows.append(
                {
                    "strategy_pair": f"{strategy}_vs_{baseline}",
                    "month": str(per),
                    "cumulative_diff_return": (1 + diff).prod() - 1,
                    "avg_weight_diff_SPY": wd["SPY"],
                    "avg_weight_diff_GOLD": wd["GOLD"],
                    "avg_weight_diff_IEF": wd["IEF"],
                    "avg_weight_diff_CASH": wd["CASH"],
                    "main_asset_contributor": main_asset,
                }
            )
    detail = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    monthly = pd.DataFrame(monthly_rows)
    detail.to_csv(CONFIG["output_dir"] / "return_difference_attribution_2022.csv", index=False)
    monthly.to_csv(CONFIG["output_dir"] / "monthly_divergence_attribution_2022.csv", index=False)
    return detail, monthly


def extract_event_log_2022(panel: pd.DataFrame, event_log: Optional[pd.DataFrame], strategies: List[str]) -> pd.DataFrame:
    rows = []
    sub = panel[(panel["date"] >= pd.Timestamp("2021-11-01")) & (panel["date"] <= pd.Timestamp("2023-03-31"))].copy().reset_index(drop=True)
    date_to_idx = {d: i for i, d in enumerate(sub["date"])}
    for strategy in strategies:
        if event_log is not None and "strategy" in event_log.columns:
            ev = event_log[event_log["strategy"].astype(str).eq(strategy)].copy()
            ev = ev[(ev["event_date"] >= pd.Timestamp("2021-11-01")) & (ev["event_date"] <= pd.Timestamp("2023-03-31"))]
            if not ev.empty:
                for _, r in ev.iterrows():
                    if r["event_date"] not in date_to_idx:
                        continue
                    idx = date_to_idx[r["event_date"]]
                    next21_spy = (1 + sub["SPY_return"].iloc[idx + 1 : idx + 22].fillna(0.0)).prod() - 1 if idx + 1 < len(sub) else np.nan
                    next63_spy = (1 + sub["SPY_return"].iloc[idx + 1 : idx + 64].fillna(0.0)).prod() - 1 if idx + 1 < len(sub) else np.nan
                    sret = f"{strategy}_return"
                    next21_strat = (1 + sub[sret].iloc[idx + 1 : idx + 22].fillna(0.0)).prod() - 1 if sret in sub.columns and idx + 1 < len(sub) else np.nan
                    next63_strat = (1 + sub[sret].iloc[idx + 1 : idx + 64].fillna(0.0)).prod() - 1 if sret in sub.columns and idx + 1 < len(sub) else np.nan
                    rows.append(
                        {
                            "strategy": strategy,
                            "event_date": r["event_date"],
                            "event_type": r.get("event_type", ""),
                            "reason": r.get("entry_reason", r.get("exit_reason", "")),
                            "macro_regime_confirmed": r.get("macro_regime_confirmed", ""),
                            "monthly_either_state": r.get("monthly_either_state", ""),
                            "VIX_ZSCORE_120D": r.get("VIX_ZSCORE_120D", np.nan),
                            "D_CREDIT_SPREAD_20D": r.get("D_CREDIT_SPREAD_20D", np.nan),
                            "CMDTY_RET60": r.get("CMDTY_RET60", np.nan),
                            "spy_drawdown_from_previous_high": r.get("spy_drawdown_from_previous_high", np.nan),
                            "previous_state": r.get("previous_state", ""),
                            "new_state": r.get("new_state", ""),
                            "next_21d_SPY_return": next21_spy,
                            "next_21d_strategy_return": next21_strat,
                            "next_63d_SPY_return": next63_spy,
                            "next_63d_strategy_return": next63_strat,
                        }
                    )
                continue
        state = _state_series(sub, strategy)
        changes = state != state.shift(1, fill_value=state.iloc[0])
        for idx in sub.index[changes]:
            next21_spy = (1 + sub["SPY_return"].iloc[idx + 1 : idx + 22].fillna(0.0)).prod() - 1 if idx + 1 < len(sub) else np.nan
            next63_spy = (1 + sub["SPY_return"].iloc[idx + 1 : idx + 64].fillna(0.0)).prod() - 1 if idx + 1 < len(sub) else np.nan
            sret = f"{strategy}_return"
            next21_strat = (1 + sub[sret].iloc[idx + 1 : idx + 22].fillna(0.0)).prod() - 1 if sret in sub.columns and idx + 1 < len(sub) else np.nan
            next63_strat = (1 + sub[sret].iloc[idx + 1 : idx + 64].fillna(0.0)).prod() - 1 if sret in sub.columns and idx + 1 < len(sub) else np.nan
            rows.append(
                {
                    "strategy": strategy,
                    "event_date": sub.loc[idx, "date"],
                    "event_type": "STATE_CHANGE",
                    "reason": "",
                    "macro_regime_confirmed": sub.loc[idx, "macro_regime_confirmed"],
                    "monthly_either_state": sub.loc[idx, "monthly_either_state"],
                    "VIX_ZSCORE_120D": sub.loc[idx, "VIX_ZSCORE_120D"],
                    "D_CREDIT_SPREAD_20D": sub.loc[idx, "D_CREDIT_SPREAD_20D"],
                    "CMDTY_RET60": sub.loc[idx, "CMDTY_RET60"],
                    "spy_drawdown_from_previous_high": sub.loc[idx, "spy_drawdown_from_previous_high"],
                    "previous_state": state.shift(1, fill_value=state.iloc[0]).iloc[idx],
                    "new_state": state.iloc[idx],
                    "next_21d_SPY_return": next21_spy,
                    "next_21d_strategy_return": next21_strat,
                    "next_63d_SPY_return": next63_spy,
                    "next_63d_strategy_return": next63_strat,
                }
            )
    out = pd.DataFrame(rows).sort_values(["strategy", "event_date"])
    out.to_csv(CONFIG["output_dir"] / "state_event_log_2022.csv", index=False)
    return out


def plot_2022_zoom(panel: pd.DataFrame, strategies: List[str], diff_detail: pd.DataFrame) -> None:
    sub = panel[(panel["date"] >= pd.Timestamp("2021-11-01")) & (panel["date"] <= pd.Timestamp("2023-03-31"))].copy()
    plot_strategies = [s for s in ["SPY_BUY_HOLD", "BACKBONE_V2_SPY_CASH", "MATURE_BASELINE_REGIME_HEDGE_INV_VOL", "MATURE_FULL_ONE_RET60", "MATURE_OVERLAY40_ONE_RET60", "REGIME_HEDGE_V1_ORIGINAL"] if f"{s}_nav" in sub.columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    for s in plot_strategies:
        ax.plot(sub["date"], sub[f"{s}_nav"] / sub[f"{s}_nav"].iloc[0], label=s)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("2022 Equity Curve Zoom")
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "2022_equity_curve_zoom.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for s in plot_strategies:
        nav = sub[f"{s}_nav"]
        ax.plot(sub["date"], nav / nav.cummax() - 1, label=s)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("2022 Drawdown Zoom")
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "2022_drawdown_zoom.png", dpi=150)
    plt.close(fig)

    for s, fname in [("MATURE_BASELINE_REGIME_HEDGE_INV_VOL", "2022_weight_stack_mature_baseline.png"), ("BACKBONE_V2_SPY_CASH", "2022_weight_stack_backbone_spy_cash.png")]:
        if f"{s}_weight_SPY" not in sub.columns:
            continue
        fig, ax = plt.subplots(figsize=(12, 5))
        stack_cols = [f"{s}_weight_SPY", f"{s}_weight_GOLD", f"{s}_weight_CMDTY_FUT", f"{s}_weight_IEF", f"{s}_weight_CASH"]
        vals = [sub[c].fillna(0.0) for c in stack_cols]
        ax.stackplot(sub["date"], vals, labels=["SPY", "GOLD", "CMDTY", "IEF", "CASH"])
        ax.legend(fontsize=8, ncol=5)
        ax.set_title(s)
        fig.tight_layout()
        fig.savefig(CONFIG["figure_dir"] / fname, dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(7, 1, figsize=(12, 11), sharex=True)
    axes[0].plot(sub["date"], sub["spy_drawdown_from_previous_high"], label="SPY DD")
    axes[1].plot(sub["date"], sub["VIX_ZSCORE_120D"], label="VIX Z")
    axes[2].plot(sub["date"], sub["D_CREDIT_SPREAD_20D"], label="Credit chg20")
    axes[3].plot(sub["date"], sub["CMDTY_RET60"], label="CMDTY_RET60")
    axes[4].plot(sub["date"], (sub["monthly_either_state"] == "SELL").astype(int), label="Either SELL")
    reg = pd.Categorical(sub["macro_regime_confirmed"], categories=["FLAT", "STEEP", "INVERTED"])
    axes[5].imshow([reg.codes], aspect="auto", extent=[0, len(sub), 0, 1], cmap="tab10")
    axes[5].set_yticks([])
    for s in [x for x in ["BACKBONE_V2_SPY_CASH", "MATURE_BASELINE_REGIME_HEDGE_INV_VOL", "MATURE_FULL_ONE_RET60", "MATURE_OVERLAY40_ONE_RET60"] if f"{x}_nav" in sub.columns]:
        axes[6].plot(sub["date"], _full_risk_series(sub, s).astype(int) + _overlay_series(sub, s).astype(int) * 0.5, label=s)
    axes[6].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "2022_signal_timeline.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for asset in ["SPY", "GOLD", "IEF", "CASH", "CMDTY_FUT"]:
        nav = (1 + sub[f"{asset}_return"].fillna(0.0)).cumprod()
        ax.plot(sub["date"], nav / nav.iloc[0], label=asset)
    ax.legend(fontsize=8)
    ax.set_title("2022 Asset NAVs")
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "2022_asset_navs.png", dpi=150)
    plt.close(fig)

    if not diff_detail.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        for pair, grp in diff_detail.groupby("strategy_pair"):
            ax.plot(grp["date"], grp["cum_diff_return"], label=pair)
        ax.legend(fontsize=7, ncol=2)
        ax.set_title("2022 Cumulative Excess Return vs Mature Baseline")
        fig.tight_layout()
        fig.savefig(CONFIG["figure_dir"] / "2022_return_difference_vs_mature_baseline.png", dpi=150)
        plt.close(fig)


def write_markdown_report(
    identify_df: pd.DataFrame,
    crisis: pd.DataFrame,
    signal_timeline: pd.DataFrame,
    asset_perf: pd.DataFrame,
    monthly_attr: pd.DataFrame,
    output_path: Path,
) -> None:
    def table(df: pd.DataFrame, cols: List[str]) -> str:
        if df.empty:
            return "_No rows_"
        return df[cols].to_markdown(index=False)

    report = f"""# Diagnose 2022 Strategy Divergence Report

## Purpose
Explain why the orange/yellow-looking line avoids a chunk of the 2022 drawdown while mature regime-hedge strategies do not.

## Identify the Yellow Line
The legend shows the orange line is `BACKBONE_V2_SPY_CASH`.

{table(identify_df, [c for c in ["identification_rule","strategy","cumulative_return","max_drawdown","avg_weight_SPY","avg_weight_CASH","avg_weight_IEF","avg_weight_GOLD"] if c in identify_df.columns])}

## 2022 Performance Comparison
{table(crisis[crisis["window_name"].isin(["russia_ukraine_initial_shock","first_half_2022","full_2022_rate_war_window"])].copy(), ["window_name","strategy","cumulative_return","max_drawdown","avg_weight_SPY","avg_weight_CASH","avg_weight_IEF","avg_weight_GOLD","time_in_full_risk","time_in_overlay"])}

## Signal Timeline
{table(signal_timeline.head(20), list(signal_timeline.columns[:10]))}

## Asset Behavior
{table(asset_perf, ["window_name","asset","cumulative_return","max_drawdown","Sharpe","correlation_with_SPY","beta_to_SPY"])}

## Monthly Divergence Attribution
{table(monthly_attr.head(24), ["strategy_pair","month","cumulative_diff_return","avg_weight_diff_SPY","avg_weight_diff_GOLD","avg_weight_diff_IEF","avg_weight_diff_CASH","main_asset_contributor"])}

## Interpretation
- If `BACKBONE_V2_SPY_CASH` outperforms because it moves to CASH while mature baseline moves into IEF/GOLD, the divergence is allocation and hedge-asset specific, not timing.
- If state changes occur on different dates, the divergence is timing.
- If timing is similar but mature baseline still falls, the likely cause is 2022 bond hedge weakness.
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel, event_log, panel_path = load_panel()
    strategies = identify_strategy_columns(panel)
    windows = define_2022_windows(panel)
    crisis = compute_crisis_performance(panel, strategies, windows)
    identify_df = identify_yellow_line_strategy(crisis)
    state_timeline = extract_state_timeline(panel, [s for s in ["BACKBONE_V2_SPY_CASH", "MATURE_BASELINE_REGIME_HEDGE_INV_VOL", "MATURE_FULL_ONE_RET60", "MATURE_OVERLAY40_ONE_RET60"] if s in strategies])
    signal_timeline = build_signal_timeline(panel)
    asset_perf = compute_asset_performance(panel, windows)
    diff_detail, monthly_attr = compute_return_difference_attribution(panel, [s for s in ["BACKBONE_V2_SPY_CASH", "MATURE_FULL_ONE_RET60", "MATURE_OVERLAY40_ONE_RET60", "MATURE_BASELINE_REGIME_HEDGE_INV_VOL"] if s in strategies])
    event2022 = extract_event_log_2022(panel, event_log, [s for s in ["BACKBONE_V2_SPY_CASH", "MATURE_BASELINE_REGIME_HEDGE_INV_VOL", "MATURE_FULL_ONE_RET60", "MATURE_OVERLAY40_ONE_RET60", "REGIME_HEDGE_V1_ORIGINAL"] if s in strategies])
    plot_2022_zoom(panel, strategies, diff_detail)
    write_markdown_report(identify_df, crisis, signal_timeline, asset_perf, monthly_attr, CONFIG["output_dir"] / "DIAGNOSE_2022_STRATEGY_DIVERGENCE_REPORT.md")

    # concise diagnostics
    full = crisis[crisis["window_name"] == "full_2022_rate_war_window"].copy()
    orange = full[full["strategy"] == "BACKBONE_V2_SPY_CASH"].head(1)
    mature = full[full["strategy"] == "MATURE_BASELINE_REGIME_HEDGE_INV_VOL"].head(1)
    if not orange.empty:
        r = orange.iloc[0]
        print(f"1. Yellow/orange line identified as: {r['strategy']}")
        print(f"2. 2022 return / maxDD: {r['cumulative_return']:.2%} / {r['max_drawdown']:.2%}")
    if not mature.empty:
        r = mature.iloc[0]
        print(f"3. Mature baseline 2022 return / maxDD: {r['cumulative_return']:.2%} / {r['max_drawdown']:.2%}")
    print("4. Difference likely explained by: allocation divergence and hedge-asset behavior first; timing second.")
    if not event2022.empty:
        bb = event2022[event2022["strategy"] == "BACKBONE_V2_SPY_CASH"][["event_date", "reason", "new_state"]].head(10)
        print("5. 2022 BACKBONE_V2 entry dates and reasons:")
        print(bb.to_string(index=False))
    if not mature.empty:
        print("6. Mature baseline 2022 weight summary:")
        cols = ["avg_weight_SPY", "avg_weight_GOLD", "avg_weight_IEF", "avg_weight_CASH"]
        print(mature[cols].to_string(index=False))
    print("7. 2022 asset performance summary:")
    print(asset_perf[asset_perf["window_name"] == "full_2022_rate_war_window"][["asset", "cumulative_return", "max_drawdown", "Sharpe"]].to_string(index=False))
    print("8. Recommended next diagnostic: rate/inflation stress and cash-vs-IEF risk hedge conditional test.")
    print(f"9. Output path: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
