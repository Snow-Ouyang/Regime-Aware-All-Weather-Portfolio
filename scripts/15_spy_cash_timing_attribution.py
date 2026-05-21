from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from final_strategy_source_only_core import (
    ASSETS,
    FINAL_STRATEGY,
    INV_VOL_WINDOW,
    ROOT,
    compute_strategy,
    monthly_hold_weights,
    normalize_weight_dict,
    performance_metrics,
)


OUT = ROOT / "results" / "spy_cash_timing_attribution"
FIG = OUT / "figures"
MAIN = ROOT / "results" / "main_pipeline_final" / "tables"

SPY_BUY_HOLD = "SPY_BUY_HOLD"
CURRENT_FINAL = "CURRENT_FINAL"
SPY_CASH_TRIGGER_LOCK = "SPY_CASH_TRIGGER_LOCK"
SPY_CASH_CREDIT_ONLY = "SPY_CASH_CREDIT_ONLY"
SPY_CASH_VIX_ONLY = "SPY_CASH_VIX_ONLY"
SPY_CASH_CMDTY_ONLY = "SPY_CASH_CMDTY_ONLY"
REGIME_NORMAL_CASH_STRESS = "REGIME_NORMAL_CASH_STRESS"
SPY_NORMAL_REGIME_HEDGE_STRESS = "SPY_NORMAL_REGIME_HEDGE_STRESS"
FINAL_WITH_CASH_STRESS_ONLY = "FINAL_WITH_CASH_STRESS_ONLY"
FINAL_WITH_CREDIT_STRESS_CASH = "FINAL_WITH_CREDIT_STRESS_CASH"
FINAL_WITH_NON_CREDIT_STRESS_CASH = "FINAL_WITH_NON_CREDIT_STRESS_CASH"
REGIME_NORMAL_ONLY_NO_STRESS_HEDGE = "REGIME_NORMAL_ONLY_NO_STRESS_HEDGE"

PRIMARY_COMPARE = [
    SPY_BUY_HOLD,
    SPY_CASH_TRIGGER_LOCK,
    REGIME_NORMAL_CASH_STRESS,
    FINAL_WITH_CASH_STRESS_ONLY,
    CURRENT_FINAL,
]
WINDOWS = {
    "2008_GFC": ("2007-10-01", "2009-06-30"),
    "2015_2016": ("2015-05-01", "2016-03-31"),
    "2018Q4": ("2018-10-01", "2019-01-31"),
    "COVID_2020": ("2020-02-01", "2020-06-30"),
    "2022_RATE_WAR": ("2021-11-01", "2023-03-31"),
    "2025_PULLBACK": ("2025-01-01", None),
}


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def load_panel() -> pd.DataFrame:
    path = MAIN / "daily_backtest_panel.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing main pipeline panel: {path}")
    panel = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    panel["active_locks"] = panel["trigger_lock_active_locks"].fillna("").astype(str)
    panel["credit_active"] = panel["active_locks"].str.contains("CREDIT")
    panel["vix_active"] = panel["active_locks"].str.contains("VIX")
    panel["cmdty_active"] = panel["active_locks"].str.contains("CMDTY")
    panel["stress_active"] = panel["trigger_lock_full_risk_state"].eq("FULL_RISK")
    panel["SPY_MA50"] = panel["spy_price"].rolling(50, min_periods=50).mean()
    panel["final_regime"] = panel["final_regime_confirmed"].fillna("OTHER")
    return panel


def weight_columns(prefix: str) -> list[str]:
    return [f"{prefix}_weight_{asset}" for asset in ASSETS]


def to_weight_frame(panel: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = weight_columns(prefix)
    renamed = {f"{prefix}_weight_{asset}": asset for asset in ASSETS}
    return panel[cols].rename(columns=renamed).fillna(0.0).copy()


def weight_string(row: pd.Series, prefix: str = "") -> str:
    vals = []
    for asset in ASSETS:
        key = f"{prefix}{asset}"
        val = float(row.get(key, 0.0))
        if abs(val) > 1e-8:
            vals.append(f"{asset}:{val:.0%}")
    return "; ".join(vals) if vals else "NONE"


def period_return(ret: pd.Series) -> float:
    if ret.empty:
        return np.nan
    return float((1.0 + ret.fillna(0.0)).prod() - 1.0)


def period_mdd(ret: pd.Series) -> float:
    if ret.empty:
        return np.nan
    nav = (1.0 + ret.fillna(0.0)).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def forward_return(ret: pd.Series, window: int) -> pd.Series:
    return (1.0 + ret.fillna(0.0)).rolling(window).apply(np.prod, raw=True).shift(-window) - 1.0


def forward_mdd(ret: pd.Series, window: int) -> pd.Series:
    vals = []
    arr = ret.fillna(0.0).to_numpy()
    for i in range(len(arr)):
        sub = arr[i + 1 : i + 1 + window]
        if len(sub) == 0:
            vals.append(np.nan)
        else:
            nav = np.cumprod(1.0 + sub)
            vals.append(float((nav / np.maximum.accumulate(nav) - 1.0).min()))
    return pd.Series(vals, index=ret.index)


def build_normal_templates(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "flat_low": monthly_hold_weights(panel, ["SPY", "CMDTY_FUT"], window=INV_VOL_WINDOW),
        "flat_high": monthly_hold_weights(panel, ["GOLD", "CMDTY_FUT"], window=INV_VOL_WINDOW),
        "steep_high": monthly_hold_weights(panel, ["SPY", "CMDTY_FUT"], window=INV_VOL_WINDOW),
        "inverted": monthly_hold_weights(panel, ["SPY", "GOLD"], window=INV_VOL_WINDOW),
    }


def normal_allocation(panel: pd.DataFrame, templates: dict[str, pd.DataFrame]) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=panel.index, columns=ASSETS)
    for i, row in panel.iterrows():
        refined = row["refined_regime_confirmed"]
        if refined == "FLAT_LOW_RATE":
            w = templates["flat_low"].loc[i].to_dict()
        elif refined == "FLAT_HIGH_RATE":
            w = templates["flat_high"].loc[i].to_dict()
        elif refined == "STEEP":
            if row["steep_rate_regime_confirmed"] == "STEEP_HIGH_RATE":
                w = templates["steep_high"].loc[i].to_dict()
            else:
                w = {"SPY": 1.0}
        elif refined == "INVERTED":
            w = templates["inverted"].loc[i].to_dict()
        else:
            w = {"SPY": 1.0}
        weights.loc[i, ASSETS] = pd.Series(normalize_weight_dict(w))
    return weights


def simulate_single_trigger_state(panel: pd.DataFrame, trigger: str) -> pd.DataFrame:
    pending = False
    active = []
    entry_signal = []
    exit_signal = []
    for _, row in panel.iterrows():
        state = pending
        active.append(state)
        entry = False
        exit_ = False
        if trigger == "CREDIT":
            entry_cond = bool(
                row["refined_regime_confirmed"] in {"FLAT_LOW_RATE", "FLAT_HIGH_RATE"}
                and row["spy_drawdown_from_previous_high"] <= -0.05
                and row["D_CREDIT_SPREAD_15D"] > 0.10
            )
            unlock_cond = bool((row["D_CREDIT_SPREAD_15D"] < 0) and (row["spy_price"] > row["SPY_MA20"]))
        elif trigger == "VIX":
            entry_cond = bool(
                row["refined_regime_confirmed"] in {"FLAT_LOW_RATE", "FLAT_HIGH_RATE", "STEEP"}
                and row["VIX_ZSCORE_120D"] >= 3.0
            )
            unlock_cond = bool((row["VIX_ZSCORE_120D"] < 1.5) and (row["spy_price"] > row["SPY_MA20"]))
        elif trigger == "CMDTY":
            entry_cond = bool(row["refined_regime_confirmed"] == "STEEP" and row["CMDTY_RET60"] < -0.10)
            unlock_cond = bool((row["CMDTY_RET60"] > -0.05) and (row["spy_price"] > row["SPY_MA20"]))
        else:
            raise ValueError(trigger)

        if not state and entry_cond:
            pending = True
            entry = True
        elif state and unlock_cond:
            pending = False
            exit_ = True
        else:
            pending = state
        entry_signal.append(entry)
        exit_signal.append(exit_)
    return pd.DataFrame(
        {
            "active": pd.Series(active, index=panel.index),
            "entry_signal": pd.Series(entry_signal, index=panel.index),
            "exit_signal": pd.Series(exit_signal, index=panel.index),
        }
    )


def strategy_from_weights(panel: pd.DataFrame, weights: pd.DataFrame, strategy: str) -> pd.DataFrame:
    result = compute_strategy(panel, weights, strategy)
    return pd.concat([panel[["date"]], weights.add_prefix("weight_"), result], axis=1)


def current_final_from_panel(panel: pd.DataFrame) -> pd.DataFrame:
    cols = {
        f"{FINAL_STRATEGY}_weight_SPY": "weight_SPY",
        f"{FINAL_STRATEGY}_weight_GOLD": "weight_GOLD",
        f"{FINAL_STRATEGY}_weight_CMDTY_FUT": "weight_CMDTY_FUT",
        f"{FINAL_STRATEGY}_weight_IEF": "weight_IEF",
        f"{FINAL_STRATEGY}_weight_CASH": "weight_CASH",
        f"{FINAL_STRATEGY}_return": f"{CURRENT_FINAL}_return",
        f"{FINAL_STRATEGY}_nav": f"{CURRENT_FINAL}_nav",
        f"{FINAL_STRATEGY}_drawdown": f"{CURRENT_FINAL}_drawdown",
        f"{FINAL_STRATEGY}_turnover": f"{CURRENT_FINAL}_turnover",
        f"{FINAL_STRATEGY}_transaction_cost": f"{CURRENT_FINAL}_transaction_cost",
    }
    out = panel[["date", *cols.keys()]].rename(columns=cols).copy()
    out["stress_active"] = panel["stress_active"].to_numpy()
    out["credit_active"] = panel["credit_active"].to_numpy()
    out["vix_active"] = panel["vix_active"].to_numpy()
    out["cmdty_active"] = panel["cmdty_active"].to_numpy()
    out["active_trigger"] = stress_mask_label(panel).to_numpy()
    return out


def current_final_alignment_check(panel: pd.DataFrame, rebuilt: pd.DataFrame) -> pd.DataFrame:
    ref_ret = panel[f"{FINAL_STRATEGY}_return"].fillna(0.0)
    new_ret = rebuilt[f"{CURRENT_FINAL}_return"].fillna(0.0)
    ref_stress = panel["stress_active"].astype(bool)
    new_stress = rebuilt["stress_active"].astype(bool)
    return pd.DataFrame(
        [
            {
                "metric": "daily_return_correlation",
                "value": float(ref_ret.corr(new_ret)),
            },
            {
                "metric": "max_abs_daily_return_difference",
                "value": float((ref_ret - new_ret).abs().max()),
            },
            {
                "metric": "mean_abs_daily_return_difference",
                "value": float((ref_ret - new_ret).abs().mean()),
            },
            {
                "metric": "mismatched_stress_days",
                "value": int((ref_stress != new_stress).sum()),
            },
        ]
    )


def clone_strategy_frame(df: pd.DataFrame, old: str, new: str) -> pd.DataFrame:
    renamed = {}
    for suffix in ["return", "nav", "drawdown", "turnover", "transaction_cost"]:
        renamed[f"{old}_{suffix}"] = f"{new}_{suffix}"
    return df.rename(columns=renamed).copy()


def make_cash_or_spy_weights(panel: pd.DataFrame, active: pd.Series) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=panel.index, columns=ASSETS)
    weights["SPY"] = np.where(active, 0.0, 1.0)
    weights["CASH"] = 1.0 - weights["SPY"]
    return weights


def stress_mask_label(panel: pd.DataFrame) -> pd.Series:
    out = np.where(panel["credit_active"], "CREDIT", np.where(panel["vix_active"], "VIX", np.where(panel["cmdty_active"], "CMDTY", "NONE")))
    return pd.Series(out, index=panel.index)


def build_variants(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    variants: dict[str, pd.DataFrame] = {}
    templates = build_normal_templates(panel)
    normal_weights = normal_allocation(panel, templates)
    final_weights = to_weight_frame(panel, FINAL_STRATEGY)

    buy_hold = pd.DataFrame(0.0, index=panel.index, columns=ASSETS)
    buy_hold["SPY"] = 1.0
    variants[SPY_BUY_HOLD] = strategy_from_weights(panel, buy_hold, SPY_BUY_HOLD)

    current = current_final_from_panel(panel)
    variants[CURRENT_FINAL] = current

    variants[SPY_CASH_TRIGGER_LOCK] = strategy_from_weights(panel, make_cash_or_spy_weights(panel, panel["stress_active"]), SPY_CASH_TRIGGER_LOCK)

    for trigger, name in [("CREDIT", SPY_CASH_CREDIT_ONLY), ("VIX", SPY_CASH_VIX_ONLY), ("CMDTY", SPY_CASH_CMDTY_ONLY)]:
        state = simulate_single_trigger_state(panel, trigger)
        res = strategy_from_weights(panel, make_cash_or_spy_weights(panel, state["active"]), name)
        res["stress_active"] = state["active"].to_numpy()
        res["credit_active"] = (trigger == "CREDIT") & state["active"]
        res["vix_active"] = (trigger == "VIX") & state["active"]
        res["cmdty_active"] = (trigger == "CMDTY") & state["active"]
        res["active_trigger"] = np.where(state["active"], trigger, "NONE")
        variants[name] = res

    weights = final_weights.copy()
    weights.loc[panel["stress_active"], :] = 0.0
    weights.loc[panel["stress_active"], "CASH"] = 1.0
    reg_cash = strategy_from_weights(panel, weights, REGIME_NORMAL_CASH_STRESS)
    reg_cash["stress_active"] = panel["stress_active"].to_numpy()
    reg_cash["credit_active"] = panel["credit_active"].to_numpy()
    reg_cash["vix_active"] = panel["vix_active"].to_numpy()
    reg_cash["cmdty_active"] = panel["cmdty_active"].to_numpy()
    reg_cash["active_trigger"] = stress_mask_label(panel).to_numpy()
    variants[REGIME_NORMAL_CASH_STRESS] = reg_cash
    variants[FINAL_WITH_CASH_STRESS_ONLY] = clone_strategy_frame(
        reg_cash, REGIME_NORMAL_CASH_STRESS, FINAL_WITH_CASH_STRESS_ONLY
    )

    spy_norm = pd.DataFrame(0.0, index=panel.index, columns=ASSETS)
    spy_norm["SPY"] = 1.0
    weights = spy_norm.copy()
    weights.loc[panel["stress_active"], ASSETS] = final_weights.loc[panel["stress_active"], ASSETS].to_numpy()
    spy_norm_hedge = strategy_from_weights(panel, weights, SPY_NORMAL_REGIME_HEDGE_STRESS)
    spy_norm_hedge["stress_active"] = panel["stress_active"].to_numpy()
    spy_norm_hedge["credit_active"] = panel["credit_active"].to_numpy()
    spy_norm_hedge["vix_active"] = panel["vix_active"].to_numpy()
    spy_norm_hedge["cmdty_active"] = panel["cmdty_active"].to_numpy()
    spy_norm_hedge["active_trigger"] = stress_mask_label(panel).to_numpy()
    variants[SPY_NORMAL_REGIME_HEDGE_STRESS] = spy_norm_hedge

    weights = final_weights.copy()
    credit_mask = panel["stress_active"] & panel["credit_active"]
    weights.loc[credit_mask, ASSETS] = 0.0
    weights.loc[credit_mask, "CASH"] = 1.0
    credit_cash = strategy_from_weights(panel, weights, FINAL_WITH_CREDIT_STRESS_CASH)
    credit_cash["stress_active"] = panel["stress_active"].to_numpy()
    credit_cash["credit_active"] = panel["credit_active"].to_numpy()
    credit_cash["vix_active"] = panel["vix_active"].to_numpy()
    credit_cash["cmdty_active"] = panel["cmdty_active"].to_numpy()
    credit_cash["active_trigger"] = stress_mask_label(panel).to_numpy()
    variants[FINAL_WITH_CREDIT_STRESS_CASH] = credit_cash

    weights = final_weights.copy()
    non_credit_mask = panel["stress_active"] & ~panel["credit_active"]
    weights.loc[non_credit_mask, ASSETS] = 0.0
    weights.loc[non_credit_mask, "CASH"] = 1.0
    non_credit_cash = strategy_from_weights(panel, weights, FINAL_WITH_NON_CREDIT_STRESS_CASH)
    non_credit_cash["stress_active"] = panel["stress_active"].to_numpy()
    non_credit_cash["credit_active"] = panel["credit_active"].to_numpy()
    non_credit_cash["vix_active"] = panel["vix_active"].to_numpy()
    non_credit_cash["cmdty_active"] = panel["cmdty_active"].to_numpy()
    non_credit_cash["active_trigger"] = stress_mask_label(panel).to_numpy()
    variants[FINAL_WITH_NON_CREDIT_STRESS_CASH] = non_credit_cash

    no_stress_hedge = strategy_from_weights(panel, normal_weights, REGIME_NORMAL_ONLY_NO_STRESS_HEDGE)
    no_stress_hedge["stress_active"] = panel["stress_active"].to_numpy()
    no_stress_hedge["credit_active"] = panel["credit_active"].to_numpy()
    no_stress_hedge["vix_active"] = panel["vix_active"].to_numpy()
    no_stress_hedge["cmdty_active"] = panel["cmdty_active"].to_numpy()
    no_stress_hedge["active_trigger"] = stress_mask_label(panel).to_numpy()
    variants[REGIME_NORMAL_ONLY_NO_STRESS_HEDGE] = no_stress_hedge

    return variants


def performance_row(strategy: str, df: pd.DataFrame) -> dict[str, object]:
    perf = performance_metrics(df, strategy)
    return {
        "strategy": strategy,
        **perf,
        "annualized_vol": perf["annualized_volatility"],
        "transaction_cost_drag": perf["transaction_cost"],
        "time_in_stress": int(df.get("stress_active", pd.Series(False, index=df.index)).sum()),
        "time_in_credit_stress": int(df.get("credit_active", pd.Series(False, index=df.index)).sum()),
        "time_in_vix_stress": int(df.get("vix_active", pd.Series(False, index=df.index)).sum()),
        "time_in_cmdty_stress": int(df.get("cmdty_active", pd.Series(False, index=df.index)).sum()),
        "avg_weight_SPY": float(df["weight_SPY"].mean()),
        "avg_weight_GOLD": float(df["weight_GOLD"].mean()),
        "avg_weight_IEF": float(df["weight_IEF"].mean()),
        "avg_weight_CASH": float(df["weight_CASH"].mean()),
        "avg_weight_CMDTY_FUT": float(df["weight_CMDTY_FUT"].mean()),
    }


def crisis_row(strategy: str, df: pd.DataFrame, window: str, start: str | None, end: str | None) -> dict[str, object]:
    sub = df.copy()
    if start is not None:
        sub = sub.loc[sub["date"] >= pd.Timestamp(start)]
    if end is not None:
        sub = sub.loc[sub["date"] <= pd.Timestamp(end)]
    ret = sub[f"{strategy}_return"]
    nav = (1.0 + ret.fillna(0.0)).cumprod()
    ann_vol = ret.std(ddof=1) * np.sqrt(252.0)
    ann_ret = nav.iloc[-1] ** (252.0 / len(sub)) - 1.0 if len(sub) else np.nan
    return {
        "strategy": strategy,
        "window": window,
        "cumulative_return": float(nav.iloc[-1] - 1.0) if len(sub) else np.nan,
        "max_drawdown": float((nav / nav.cummax() - 1.0).min()) if len(sub) else np.nan,
        "Sharpe": float(ann_ret / ann_vol) if len(sub) and ann_vol > 0 else np.nan,
        "time_in_stress": int(sub.get("stress_active", pd.Series(False, index=sub.index)).sum()),
        "time_in_credit_stress": int(sub.get("credit_active", pd.Series(False, index=sub.index)).sum()),
        "time_in_vix_stress": int(sub.get("vix_active", pd.Series(False, index=sub.index)).sum()),
        "time_in_cmdty_stress": int(sub.get("cmdty_active", pd.Series(False, index=sub.index)).sum()),
        "avg_weight_SPY": float(sub["weight_SPY"].mean()) if len(sub) else np.nan,
        "avg_weight_GOLD": float(sub["weight_GOLD"].mean()) if len(sub) else np.nan,
        "avg_weight_IEF": float(sub["weight_IEF"].mean()) if len(sub) else np.nan,
        "avg_weight_CASH": float(sub["weight_CASH"].mean()) if len(sub) else np.nan,
        "avg_weight_CMDTY_FUT": float(sub["weight_CMDTY_FUT"].mean()) if len(sub) else np.nan,
    }


def layer_attribution(perf: pd.DataFrame) -> pd.DataFrame:
    idx = perf.set_index("strategy")
    rows = []
    comps = [
        ("Timing contribution", SPY_CASH_TRIGGER_LOCK, SPY_BUY_HOLD, "Trigger-lock timing relative to buy-and-hold."),
        ("Normal regime allocation contribution", REGIME_NORMAL_CASH_STRESS, SPY_CASH_TRIGGER_LOCK, "Normal regime allocation relative to normal=SPY."),
        ("Stress hedge allocation contribution", CURRENT_FINAL, REGIME_NORMAL_CASH_STRESS, "Stress hedge sleeve relative to stress=CASH."),
        ("Stress hedge vs cash contribution", CURRENT_FINAL, FINAL_WITH_CASH_STRESS_ONLY, "All stress hedge sleeves relative to cash."),
        ("Credit stress hedge contribution", CURRENT_FINAL, FINAL_WITH_CREDIT_STRESS_CASH, "Credit stress hedge relative to cash in credit episodes."),
        ("Non-credit stress hedge contribution", CURRENT_FINAL, FINAL_WITH_NON_CREDIT_STRESS_CASH, "VIX/CMDTY hedge relative to cash in non-credit episodes."),
    ]
    for component, a, b, interp in comps:
        ra, rb = idx.loc[a], idx.loc[b]
        rows.append(
            {
                "component": component,
                "strategy_a": a,
                "strategy_b": b,
                "delta_CAGR": ra["CAGR"] - rb["CAGR"],
                "delta_Sharpe": ra["Sharpe"] - rb["Sharpe"],
                "delta_MaxDD": ra["MaxDD"] - rb["MaxDD"],
                "delta_Calmar": ra["Calmar"] - rb["Calmar"],
                "delta_Final_Equity": ra["final_equity"] - rb["final_equity"],
                "interpretation": interp,
            }
        )
    return pd.DataFrame(rows)


def find_episodes(active: pd.Series) -> list[tuple[int, int]]:
    start = active & ~active.shift(1, fill_value=False)
    end = active & ~active.shift(-1, fill_value=False)
    return list(zip(start[start].index.tolist(), end[end].index.tolist()))


def trigger_specific_value(panel: pd.DataFrame, variants: dict[str, pd.DataFrame]) -> pd.DataFrame:
    current = variants[CURRENT_FINAL]
    rows = []
    for trigger, mask in {
        "VIX": current["vix_active"].astype(bool),
        "Credit": current["credit_active"].astype(bool),
        "Commodity": current["cmdty_active"].astype(bool),
        "Combined": current["stress_active"].astype(bool),
    }.items():
        eps = find_episodes(mask)
        spy_ep = []
        cash_ep = []
        hedge_ep = []
        for s, e in eps:
            spy_ret = period_return(panel.loc[s:e, "SPY_return"])
            cash_ret = period_return(panel.loc[s:e, "CASH_return"])
            hedge_ret = period_return(current.loc[s:e, f"{CURRENT_FINAL}_return"])
            spy_ep.append(spy_ret)
            cash_ep.append(cash_ret)
            hedge_ep.append(hedge_ret)
        rows.append(
            {
                "trigger": trigger,
                "n_episodes": len(eps),
                "time_in_trigger": int(mask.sum()),
                "SPY_return_during_trigger": period_return(panel.loc[mask, "SPY_return"]),
                "SPY_maxDD_during_trigger": period_mdd(panel.loc[mask, "SPY_return"]),
                "CASH_return_during_trigger": period_return(panel.loc[mask, "CASH_return"]),
                "final_hedge_return_during_trigger": period_return(current.loc[mask, f"{CURRENT_FINAL}_return"]),
                "final_hedge_maxDD_during_trigger": period_mdd(current.loc[mask, f"{CURRENT_FINAL}_return"]),
                "drawdown_avoided_vs_SPY": period_mdd(current.loc[mask, f"{CURRENT_FINAL}_return"]) - period_mdd(panel.loc[mask, "SPY_return"]),
                "final_hedge_excess_vs_CASH": period_return(current.loc[mask, f"{CURRENT_FINAL}_return"]) - period_return(panel.loc[mask, "CASH_return"]),
                "cash_better_episode_ratio": float(np.mean([c > h for c, h in zip(cash_ep, hedge_ep)])) if eps else np.nan,
                "final_hedge_better_episode_ratio": float(np.mean([h > c for c, h in zip(cash_ep, hedge_ep)])) if eps else np.nan,
            }
        )
    return pd.DataFrame(rows)


def credit_episode_comparison(panel: pd.DataFrame, variants: dict[str, pd.DataFrame]) -> pd.DataFrame:
    current = variants[CURRENT_FINAL]
    cash_var = variants[FINAL_WITH_CREDIT_STRESS_CASH]
    rows = []
    for n, (s, e) in enumerate(find_episodes(current["credit_active"].astype(bool)), start=1):
        next21_ret = forward_return(panel["SPY_return"], 21).iloc[e]
        next21_mdd = forward_mdd(panel["SPY_return"], 21).iloc[e]
        rows.append(
            {
                "episode_id": n,
                "entry_date": panel.loc[s, "date"],
                "unlock_date": panel.loc[min(e + 1, len(panel) - 1), "date"],
                "macro_regime": panel.loc[s:e, "final_regime_confirmed"].mode().iloc[0],
                "duration": int(e - s + 1),
                "SPY_return": period_return(panel.loc[s:e, "SPY_return"]),
                "SPY_maxDD": period_mdd(panel.loc[s:e, "SPY_return"]),
                "CASH_return": period_return(panel.loc[s:e, "CASH_return"]),
                "final_hedge_return": period_return(current.loc[s:e, f"{CURRENT_FINAL}_return"]),
                "final_hedge_maxDD": period_mdd(current.loc[s:e, f"{CURRENT_FINAL}_return"]),
                "final_strategy_return": period_return(current.loc[s:e, f"{CURRENT_FINAL}_return"]),
                "next_21d_SPY_return_after_unlock": next21_ret,
                "next_21d_SPY_maxDD_after_unlock": next21_mdd,
                "cash_better_than_final_hedge": period_return(panel.loc[s:e, "CASH_return"]) > period_return(current.loc[s:e, f"{CURRENT_FINAL}_return"]),
                "final_hedge_better_than_cash": period_return(current.loc[s:e, f"{CURRENT_FINAL}_return"]) > period_return(panel.loc[s:e, "CASH_return"]),
                "notes": "Credit stress episode under current final trigger-lock state.",
            }
        )
    return pd.DataFrame(rows)


def build_case_csv(name: str, start: str, end: str | None, panel: pd.DataFrame, variants: dict[str, pd.DataFrame]) -> None:
    mask = panel["date"] >= pd.Timestamp(start)
    if end is not None:
        mask &= panel["date"] <= pd.Timestamp(end)
    sub = panel.loc[mask, ["date", "final_regime_confirmed", "trigger_lock_active_locks", "SPY_return", "CASH_return"]].copy()
    sub.rename(columns={"final_regime_confirmed": "macro_regime", "trigger_lock_active_locks": "active_trigger"}, inplace=True)
    current = variants[CURRENT_FINAL].loc[mask]
    cash = variants[SPY_CASH_TRIGGER_LOCK].loc[mask]
    reg_cash = variants[REGIME_NORMAL_CASH_STRESS].loc[mask]
    credit_cash = variants[FINAL_WITH_CREDIT_STRESS_CASH].loc[mask]
    out = sub.copy()
    out["final_state"] = current["stress_active"].map({True: "STRESS", False: "NORMAL"}).to_numpy()
    for src, col in [
        (current, "CURRENT_FINAL"),
        (cash, "SPY_CASH_TRIGGER_LOCK"),
        (reg_cash, "REGIME_NORMAL_CASH_STRESS"),
        (credit_cash, "FINAL_WITH_CREDIT_STRESS_CASH"),
    ]:
        out[f"{col}_return"] = src[f"{col}_return"].to_numpy()
        out[f"{col}_NAV"] = src[f"{col}_nav"].to_numpy()
        out[f"{col}_drawdown"] = src[f"{col}_drawdown"].to_numpy()
    out["weights_CURRENT_FINAL"] = current.apply(lambda r: weight_string(r, "weight_"), axis=1).to_numpy()
    out["weights_CASH_STRESS_variant"] = credit_cash.apply(lambda r: weight_string(r, "weight_"), axis=1).to_numpy()
    out.to_csv(OUT / f"case_{name}_spy_cash_vs_final.csv", index=False)


def plot_equity(variants: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    for strategy in PRIMARY_COMPARE:
        df = variants[strategy]
        ax.plot(df["date"], df[f"{strategy}_nav"], label=strategy, linewidth=1.0)
    ax.set_yscale("log")
    ax.set_title("Equity Curve Comparison")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "equity_curve_comparison_spy_cash_vs_final.png", dpi=160)
    plt.close(fig)


def plot_drawdown(variants: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    for strategy in PRIMARY_COMPARE:
        df = variants[strategy]
        ax.plot(df["date"], df[f"{strategy}_drawdown"], label=strategy, linewidth=1.0)
    ax.set_title("Drawdown Comparison")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "drawdown_comparison_spy_cash_vs_final.png", dpi=160)
    plt.close(fig)


def plot_crisis_heatmap(crisis: pd.DataFrame) -> None:
    focus = crisis.loc[crisis["strategy"].isin([CURRENT_FINAL, SPY_CASH_TRIGGER_LOCK, REGIME_NORMAL_CASH_STRESS, SPY_NORMAL_REGIME_HEDGE_STRESS, FINAL_WITH_CREDIT_STRESS_CASH, FINAL_WITH_CASH_STRESS_ONLY])]
    heat = focus.pivot(index="strategy", columns="window", values="cumulative_return")
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(heat, annot=True, fmt=".1%", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title("Crisis Window Comparison")
    fig.tight_layout()
    fig.savefig(FIG / "crisis_window_heatmap_spy_cash_vs_final.png", dpi=160)
    plt.close(fig)


def plot_layer_bar(layer: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(layer["component"], layer["delta_Final_Equity"])
    ax.tick_params(axis="x", labelrotation=35)
    ax.set_title("Layer Attribution: Final Equity Delta")
    fig.tight_layout()
    fig.savefig(FIG / "layer_attribution_bar.png", dpi=160)
    plt.close(fig)


def plot_credit_episode_bars(credit_ep: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(credit_ep))
    ax.bar(x - 0.2, credit_ep["CASH_return"], width=0.2, label="CASH")
    ax.bar(x, credit_ep["final_hedge_return"], width=0.2, label="Final hedge")
    ax.bar(x + 0.2, credit_ep["SPY_return"], width=0.2, label="SPY")
    ax.set_xticks(x)
    ax.set_xticklabels(credit_ep["episode_id"].astype(str))
    ax.set_title("Credit Episode Returns: SPY vs CASH vs Final Hedge")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "credit_cash_vs_hedge_episode_returns.png", dpi=160)
    plt.close(fig)


def plot_stress_scatter(trigger_value: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    d = trigger_value.loc[trigger_value["trigger"] != "Combined"]
    ax.scatter(d["CASH_return_during_trigger"], d["final_hedge_return_during_trigger"], s=80)
    for _, row in d.iterrows():
        ax.text(row["CASH_return_during_trigger"], row["final_hedge_return_during_trigger"], row["trigger"])
    ax.axline((0, 0), slope=1, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("CASH return during trigger")
    ax.set_ylabel("Final hedge return during trigger")
    ax.set_title("Stress Hedge vs CASH")
    fig.tight_layout()
    fig.savefig(FIG / "stress_hedge_vs_cash_scatter.png", dpi=160)
    plt.close(fig)


def plot_case(name: str, start: str, end: str | None, panel: pd.DataFrame, variants: dict[str, pd.DataFrame]) -> None:
    mask = panel["date"] >= pd.Timestamp(start)
    if end is not None:
        mask &= panel["date"] <= pd.Timestamp(end)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(panel.loc[mask, "date"], panel.loc[mask, "spy_price"], color="black", label="SPY price")
    for strategy in [CURRENT_FINAL, SPY_CASH_TRIGGER_LOCK, REGIME_NORMAL_CASH_STRESS, FINAL_WITH_CREDIT_STRESS_CASH]:
        df = variants[strategy].loc[mask]
        axes[1].plot(df["date"], df[f"{strategy}_nav"], label=strategy)
    axes[0].set_title(f"{name}: SPY Price")
    axes[1].set_title(f"{name}: Strategy NAV")
    axes[1].legend(frameon=False, ncol=2)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / f"case_{name}_spy_cash_vs_final.png", dpi=160)
    plt.close(fig)


def build_report(perf: pd.DataFrame, layer: pd.DataFrame, trigger_value: pd.DataFrame, credit_ep: pd.DataFrame) -> None:
    idx = perf.set_index("strategy")
    lines = [
        "# SPY_CASH_TIMING_ATTRIBUTION_REPORT",
        "",
        "## 1. Purpose",
        "",
        "This is an attribution test, not a rollback proposal. The goal is to separate timing value from hedge-allocation value.",
        "",
        "## 2. Key Question",
        "",
        "We compare whether the final strategy's edge comes mainly from trigger-lock timing, normal regime allocation, stress hedge allocation, or a combination of all three.",
        "",
        "## 3. Strategy Variants",
        "",
        "- `SPY_BUY_HOLD`: 100% SPY.",
        "- `SPY_CASH_TRIGGER_LOCK`: same trigger-lock timing as final, but stress = 100% CASH.",
        "- `REGIME_NORMAL_CASH_STRESS`: final normal allocation, stress = CASH.",
        "- `CURRENT_FINAL`: current mainline final strategy.",
        "- `FINAL_WITH_CREDIT_STRESS_CASH`: cash only during credit-triggered stress.",
        "",
        "## 4. Main Performance Comparison",
        "",
        f"- SPY buy-and-hold Sharpe: {idx.loc[SPY_BUY_HOLD, 'Sharpe']:.3f}",
        f"- SPY/CASH trigger-lock Sharpe: {idx.loc[SPY_CASH_TRIGGER_LOCK, 'Sharpe']:.3f}",
        f"- Regime-normal + cash-stress Sharpe: {idx.loc[REGIME_NORMAL_CASH_STRESS, 'Sharpe']:.3f}",
        f"- Current final Sharpe: {idx.loc[CURRENT_FINAL, 'Sharpe']:.3f}",
        "",
        "## 5. Layer Attribution",
        "",
        "See `strategy_layer_attribution.csv` for the exact deltas.",
        "",
        "## 6. Credit Trigger Focus",
        "",
        f"- Credit episodes where CASH beat final hedge: {int(credit_ep['cash_better_than_final_hedge'].sum())}/{len(credit_ep)}",
        f"- Credit episodes where final hedge beat CASH: {int(credit_ep['final_hedge_better_than_cash'].sum())}/{len(credit_ep)}",
        "",
        "## 7. Crisis Window Analysis",
        "",
        "Use the crisis comparison table and case-study CSVs to compare 2008, 2015-2016, COVID, 2022, and 2025.",
        "",
        "## 8. Should We Return to SPY/CASH?",
        "",
        "The answer depends on whether timing-only SPY/CASH keeps enough of the return while materially simplifying the hedge layer.",
        "",
        "## 9. Recommendation",
        "",
        "If SPY/CASH remains clearly weaker than the final strategy in compounding and risk-adjusted return, keep it as a benchmark rather than a replacement. If the credit-stress cash variant is locally better, treat that as future refinement work, not immediate rollback.",
    ]
    (OUT / "SPY_CASH_TIMING_ATTRIBUTION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def build_patch(perf: pd.DataFrame) -> None:
    idx = perf.set_index("strategy")
    if idx.loc[CURRENT_FINAL, "final_equity"] > idx.loc[SPY_CASH_TRIGGER_LOCK, "final_equity"] and idx.loc[CURRENT_FINAL, "Sharpe"] >= idx.loc[SPY_CASH_TRIGGER_LOCK, "Sharpe"]:
        text = "SPY/CASH trigger-lock is an important timing benchmark, but the final regime-aware hedge strategy retains higher compounding and better overall risk-adjusted performance."
    elif idx.loc[SPY_CASH_TRIGGER_LOCK, "MaxDD"] > idx.loc[CURRENT_FINAL, "MaxDD"]:
        text = "SPY/CASH can be viewed as a conservative version, while the final strategy is the higher-compounding regime-aware enhancement version."
    else:
        text = "Credit stress may be better treated with cash than regime hedge assets, and this should be a future strategy refinement."
    (OUT / "README_PATCH_SUGGESTION.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = load_panel()
    variants = build_variants(panel)
    rebuilt_final = strategy_from_weights(panel, to_weight_frame(panel, FINAL_STRATEGY), CURRENT_FINAL)
    rebuilt_final["stress_active"] = panel["stress_active"].to_numpy()
    alignment = current_final_alignment_check(panel, rebuilt_final)
    alignment.to_csv(OUT / "current_final_alignment_check.csv", index=False)

    perf_rows = [performance_row(strategy, df) for strategy, df in variants.items()]
    perf = pd.DataFrame(perf_rows).sort_values("strategy").reset_index(drop=True)
    perf.to_csv(OUT / "strategy_performance_comparison.csv", index=False)

    crisis_rows = []
    for strategy, df in variants.items():
        for name, (start, end) in WINDOWS.items():
            crisis_rows.append(crisis_row(strategy, df, name, start, end))
        crisis_rows.append(crisis_row(strategy, df, "FULL_SAMPLE", None, None))
    crisis = pd.DataFrame(crisis_rows)
    crisis.to_csv(OUT / "crisis_window_comparison.csv", index=False)

    layer = layer_attribution(perf)
    layer.to_csv(OUT / "strategy_layer_attribution.csv", index=False)

    trigger_value = trigger_specific_value(panel, variants)
    trigger_value.to_csv(OUT / "trigger_specific_timing_value.csv", index=False)

    credit_ep = credit_episode_comparison(panel, variants)
    credit_ep.to_csv(OUT / "credit_cash_vs_hedge_episode_comparison.csv", index=False)

    for name, (start, end) in WINDOWS.items():
        build_case_csv(name, start, end, panel, variants)

    daily_frames = []
    weight_frames = []
    for strategy, df in variants.items():
        daily = df[["date", f"{strategy}_return", f"{strategy}_nav", f"{strategy}_drawdown", f"{strategy}_turnover", f"{strategy}_transaction_cost"]].copy()
        daily.columns = ["date", "daily_return", "equity", "drawdown", "turnover", "transaction_cost"]
        daily["strategy"] = strategy
        daily_frames.append(daily)
        w = df[["date", "weight_SPY", "weight_GOLD", "weight_CMDTY_FUT", "weight_IEF", "weight_CASH"]].copy()
        w["strategy"] = strategy
        w = w.melt(
            id_vars=["date", "strategy"],
            value_vars=["weight_SPY", "weight_GOLD", "weight_CMDTY_FUT", "weight_IEF", "weight_CASH"],
            var_name="asset",
            value_name="weight",
        )
        w["asset"] = w["asset"].str.replace("weight_", "", regex=False)
        weight_frames.append(w)
    pd.concat(daily_frames, ignore_index=True).to_csv(OUT / "daily_returns_all_strategies.csv", index=False)
    pd.concat(weight_frames, ignore_index=True).to_csv(OUT / "daily_weights_all_strategies.csv", index=False)

    plot_equity(variants)
    plot_drawdown(variants)
    plot_crisis_heatmap(crisis)
    plot_layer_bar(layer)
    plot_credit_episode_bars(credit_ep)
    plot_stress_scatter(trigger_value)
    for name, (start, end) in WINDOWS.items():
        plot_case(name, start, end, panel, variants)

    build_report(perf, layer, trigger_value, credit_ep)
    build_patch(perf)

    idx = perf.set_index("strategy")
    print("SPY_BUY_HOLD performance")
    print(idx.loc[SPY_BUY_HOLD, ["CAGR", "Sharpe", "MaxDD", "Calmar", "final_equity"]].to_string())
    print("SPY_CASH_TRIGGER_LOCK performance")
    print(idx.loc[SPY_CASH_TRIGGER_LOCK, ["CAGR", "Sharpe", "MaxDD", "Calmar", "final_equity"]].to_string())
    print("CURRENT_FINAL performance")
    print(idx.loc[CURRENT_FINAL, ["CAGR", "Sharpe", "MaxDD", "Calmar", "final_equity"]].to_string())
    print("REGIME_NORMAL_CASH_STRESS performance")
    print(idx.loc[REGIME_NORMAL_CASH_STRESS, ["CAGR", "Sharpe", "MaxDD", "Calmar", "final_equity"]].to_string())
    print("FINAL_WITH_CASH_STRESS_ONLY performance")
    print(idx.loc[FINAL_WITH_CASH_STRESS_ONLY, ["CAGR", "Sharpe", "MaxDD", "Calmar", "final_equity"]].to_string())
    print("FINAL_WITH_CREDIT_STRESS_CASH performance")
    print(idx.loc[FINAL_WITH_CREDIT_STRESS_CASH, ["CAGR", "Sharpe", "MaxDD", "Calmar", "final_equity"]].to_string())
    print("layer attribution summary")
    print(layer[["component", "delta_CAGR", "delta_Sharpe", "delta_MaxDD", "delta_Final_Equity"]].to_string(index=False))
    print("credit stress cash vs final hedge summary")
    print(credit_ep[["cash_better_than_final_hedge", "final_hedge_better_than_cash"]].sum().to_string())
    supports_spy_cash = bool(
        idx.loc[SPY_CASH_TRIGGER_LOCK, "Sharpe"] >= idx.loc[CURRENT_FINAL, "Sharpe"]
        and idx.loc[SPY_CASH_TRIGGER_LOCK, "MaxDD"] >= idx.loc[CURRENT_FINAL, "MaxDD"]
    )
    print("whether evidence supports returning to SPY/CASH")
    print("YES" if supports_spy_cash else "NO")
    print("output paths")
    print(str(OUT))


if __name__ == "__main__":
    main()
