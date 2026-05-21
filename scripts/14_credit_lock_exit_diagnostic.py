from __future__ import annotations

from dataclasses import dataclass
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


OUT = ROOT / "results" / "credit_lock_exit_diagnostic"
FIG = OUT / "figures"
MAIN_TABLES = ROOT / "results" / "main_pipeline_final" / "tables"

BASELINE = "FINAL_BASELINE"
WINDOWS = {
    "2008_GFC": ("2007-10-01", "2009-06-30"),
    "2015_2016": ("2015-01-01", "2016-12-31"),
    "2018Q4": ("2018-10-01", "2019-01-31"),
    "COVID_2020": ("2020-02-01", "2020-06-30"),
    "2022_RATE_WAR": ("2021-11-01", "2023-03-31"),
    "2025_PULLBACK": ("2025-01-01", None),
}
STRATEGIES = [
    BASELINE,
    "CREDIT_UNLOCK_2D_CONFIRM",
    "CREDIT_UNLOCK_3D_CONFIRM",
    "CREDIT_UNLOCK_MA50",
    "CREDIT_UNLOCK_MA20_MA50",
    "CREDIT_UNLOCK_LEVEL_CONFIRM",
    "CREDIT_UNLOCK_ZSCORE_CONFIRM",
    "CREDIT_UNLOCK_COOLDOWN_10D",
    "CREDIT_UNLOCK_COOLDOWN_21D",
    "CREDIT_UNLOCK_TRAILING_DD_REPAIR",
    "CREDIT_UNLOCK_HYBRID_STRICT",
    "CREDIT_UNLOCK_RATE_SHOCK_STRICT",
    "CREDIT_RELOCK_FAST",
    "CREDIT_RELOCK_FAST_WITH_VIX",
    "CREDIT_UNLOCK_MA50_PLUS_RELOCK_FAST",
    "CREDIT_UNLOCK_HYBRID_PLUS_RELOCK_FAST",
]


@dataclass
class StrategyConfig:
    name: str
    credit_unlock_rule: str
    relock_rule: str = "NONE"


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def load_panel() -> pd.DataFrame:
    path = MAIN_TABLES / "daily_backtest_panel.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing main pipeline panel: {path}")
    panel = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return panel


def prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["CREDIT_SPREAD"] = out["WBAA"] - out["WAAA"]
    if "D_CREDIT_SPREAD_15D" not in out:
        out["D_CREDIT_SPREAD_15D"] = out["CREDIT_SPREAD"] - out["CREDIT_SPREAD"].shift(15)
    if "D_CREDIT_SPREAD_20D" not in out:
        out["D_CREDIT_SPREAD_20D"] = out["CREDIT_SPREAD"] - out["CREDIT_SPREAD"].shift(20)
    out["D_CREDIT_SPREAD_60D"] = out["CREDIT_SPREAD"] - out["CREDIT_SPREAD"].shift(60)
    out["SPY_MA50"] = out["spy_price"].rolling(50, min_periods=50).mean()
    out["SPY_MA100"] = out["spy_price"].rolling(100, min_periods=100).mean()
    out["SPY_vs_MA20"] = out["spy_price"] / out["SPY_MA20"] - 1.0
    out["SPY_vs_MA50"] = out["spy_price"] / out["SPY_MA50"] - 1.0
    out["SPY_above_MA20"] = out["spy_price"] > out["SPY_MA20"]
    out["SPY_above_MA50"] = out["spy_price"] > out["SPY_MA50"]
    roll = out["CREDIT_SPREAD"].rolling(252, min_periods=126)
    out["CREDIT_SPREAD_Z_252D"] = (out["CREDIT_SPREAD"] - roll.mean()) / roll.std(ddof=1).replace(0, np.nan)
    out["CREDIT_SPREAD_Q75_252D"] = roll.quantile(0.75)
    out["final_regime"] = out["final_regime_confirmed"].fillna("OTHER")
    out["base_regime"] = out["macro_regime_confirmed"].fillna("OTHER")
    return out


def build_weight_templates(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "flat_low": monthly_hold_weights(df, ["SPY", "CMDTY_FUT"], window=INV_VOL_WINDOW),
        "flat_high": monthly_hold_weights(df, ["GOLD", "CMDTY_FUT"], window=INV_VOL_WINDOW),
        "steep_high": monthly_hold_weights(df, ["SPY", "CMDTY_FUT"], window=INV_VOL_WINDOW),
        "inverted": monthly_hold_weights(df, ["SPY", "GOLD"], window=INV_VOL_WINDOW),
    }


def normal_allocation(row: pd.Series, i: int, w: dict[str, pd.DataFrame]) -> tuple[dict[str, float], str]:
    refined = row["refined_regime_confirmed"]
    if refined == "FLAT_LOW_RATE":
        return w["flat_low"].loc[i].to_dict(), "FLAT_LOW_RATE_NORMAL"
    if refined == "FLAT_HIGH_RATE":
        return w["flat_high"].loc[i].to_dict(), "FLAT_HIGH_RATE_NORMAL"
    if refined == "STEEP":
        if row["steep_rate_regime_confirmed"] == "STEEP_HIGH_RATE":
            return w["steep_high"].loc[i].to_dict(), "STEEP_HIGH_RATE_NORMAL"
        return {"SPY": 1.0}, "STEEP_LOW_RATE_NORMAL"
    if refined == "INVERTED":
        return w["inverted"].loc[i].to_dict(), "INVERTED"
    raise ValueError(f"Unexpected refined regime: {refined}")


def stress_allocation(refined: str) -> tuple[dict[str, float], str]:
    if refined == "FLAT_LOW_RATE":
        return {"GOLD": 1.0}, "FLAT_LOW_RATE_STRESS"
    if refined == "FLAT_HIGH_RATE":
        return {"IEF": 0.90, "CASH": 0.10}, "FLAT_HIGH_RATE_STRESS"
    if refined == "STEEP":
        return {"GOLD": 0.30, "IEF": 0.70}, "STEEP_FULL_RISK"
    if refined == "INVERTED":
        return {"SPY": 0.0, "GOLD": 0.0, "CMDTY_FUT": 0.0, "IEF": 0.0, "CASH": 0.0}, "INVERTED"
    raise ValueError(f"Unexpected refined regime: {refined}")


def allowed_non_credit_entries(row: pd.Series) -> set[str]:
    refined = row["refined_regime_confirmed"]
    locks: set[str] = set()
    if refined in {"FLAT_LOW_RATE", "FLAT_HIGH_RATE", "STEEP"} and row["VIX_ZSCORE_120D"] >= 3.0:
        locks.add("VIX")
    if refined == "STEEP" and row["CMDTY_RET60"] < -0.10:
        locks.add("CMDTY")
    return locks


def credit_entry_baseline(row: pd.Series) -> bool:
    return bool(
        row["refined_regime_confirmed"] in {"FLAT_LOW_RATE", "FLAT_HIGH_RATE"}
        and row["spy_drawdown_from_previous_high"] <= -0.05
        and row["D_CREDIT_SPREAD_15D"] > 0.10
    )


def vix_unlock(row: pd.Series) -> bool:
    return bool((row["VIX_ZSCORE_120D"] < 1.5) and row["SPY_above_MA20"])


def cmdty_unlock(row: pd.Series) -> bool:
    return bool((row["CMDTY_RET60"] > -0.05) and row["SPY_above_MA20"])


def build_configs() -> list[StrategyConfig]:
    return [
        StrategyConfig(BASELINE, "BASE"),
        StrategyConfig("CREDIT_UNLOCK_2D_CONFIRM", "NEG_2D"),
        StrategyConfig("CREDIT_UNLOCK_3D_CONFIRM", "NEG_3D"),
        StrategyConfig("CREDIT_UNLOCK_MA50", "MA50"),
        StrategyConfig("CREDIT_UNLOCK_MA20_MA50", "MA20_MA50"),
        StrategyConfig("CREDIT_UNLOCK_LEVEL_CONFIRM", "LEVEL_CONFIRM"),
        StrategyConfig("CREDIT_UNLOCK_ZSCORE_CONFIRM", "ZSCORE_CONFIRM"),
        StrategyConfig("CREDIT_UNLOCK_COOLDOWN_10D", "COOLDOWN_10D"),
        StrategyConfig("CREDIT_UNLOCK_COOLDOWN_21D", "COOLDOWN_21D"),
        StrategyConfig("CREDIT_UNLOCK_TRAILING_DD_REPAIR", "DD_REPAIR"),
        StrategyConfig("CREDIT_UNLOCK_HYBRID_STRICT", "HYBRID_STRICT"),
        StrategyConfig("CREDIT_UNLOCK_RATE_SHOCK_STRICT", "RATE_SHOCK_STRICT"),
        StrategyConfig("CREDIT_RELOCK_FAST", "BASE", "FAST"),
        StrategyConfig("CREDIT_RELOCK_FAST_WITH_VIX", "BASE", "FAST_WITH_VIX"),
        StrategyConfig("CREDIT_UNLOCK_MA50_PLUS_RELOCK_FAST", "MA50", "FAST"),
        StrategyConfig("CREDIT_UNLOCK_HYBRID_PLUS_RELOCK_FAST", "HYBRID_STRICT", "FAST"),
    ]


def credit_unlock_signal(rule: str, row: pd.Series, state: dict[str, object]) -> bool:
    d15 = float(row["D_CREDIT_SPREAD_15D"]) if pd.notna(row["D_CREDIT_SPREAD_15D"]) else np.nan
    credit_spread = float(row["CREDIT_SPREAD"]) if pd.notna(row["CREDIT_SPREAD"]) else np.nan
    z252 = float(row["CREDIT_SPREAD_Z_252D"]) if pd.notna(row["CREDIT_SPREAD_Z_252D"]) else np.nan
    q75 = float(row["CREDIT_SPREAD_Q75_252D"]) if pd.notna(row["CREDIT_SPREAD_Q75_252D"]) else np.nan
    spy_ma20 = bool(row["SPY_above_MA20"])
    spy_ma50 = bool(row["SPY_above_MA50"])
    dd = float(row["spy_drawdown_from_previous_high"]) if pd.notna(row["spy_drawdown_from_previous_high"]) else np.nan
    gs1 = float(row["GS1"]) if pd.notna(row["GS1"]) else np.nan
    final_regime = str(row["final_regime"])
    neg_today = bool(pd.notna(d15) and d15 < 0)
    if neg_today:
        state["credit_neg_run"] = int(state.get("credit_neg_run", 0)) + 1
    else:
        state["credit_neg_run"] = 0

    baseline_ok = neg_today and spy_ma20

    if rule == "BASE":
        state["credit_cooldown_wait"] = None
        return baseline_ok
    if rule == "NEG_2D":
        return int(state["credit_neg_run"]) >= 2 and spy_ma20
    if rule == "NEG_3D":
        return int(state["credit_neg_run"]) >= 3 and spy_ma20
    if rule == "MA50":
        return neg_today and spy_ma50
    if rule == "MA20_MA50":
        return neg_today and spy_ma20 and spy_ma50
    if rule == "LEVEL_CONFIRM":
        return neg_today and spy_ma20 and pd.notna(q75) and pd.notna(credit_spread) and credit_spread < q75
    if rule == "ZSCORE_CONFIRM":
        return neg_today and spy_ma20 and pd.notna(z252) and z252 < 1.0
    if rule == "DD_REPAIR":
        return neg_today and spy_ma20 and pd.notna(dd) and dd > -0.08
    if rule == "HYBRID_STRICT":
        return int(state["credit_neg_run"]) >= 3 and spy_ma50 and pd.notna(z252) and z252 < 1.0
    if rule == "RATE_SHOCK_STRICT":
        high_rate = final_regime == "FLAT_HIGH_RATE" or (pd.notna(gs1) and gs1 > 0.3)
        if high_rate:
            return int(state["credit_neg_run"]) >= 3 and spy_ma50 and pd.notna(z252) and z252 < 1.0
        return baseline_ok
    if rule in {"COOLDOWN_10D", "COOLDOWN_21D"}:
        wait_target = 10 if rule.endswith("10D") else 21
        wait = state.get("credit_cooldown_wait")
        if wait is None:
            if baseline_ok:
                state["credit_cooldown_wait"] = wait_target
            return False
        if not baseline_ok:
            state["credit_cooldown_wait"] = None
            return False
        if int(wait) <= 1:
            state["credit_cooldown_wait"] = None
            return True
        state["credit_cooldown_wait"] = int(wait) - 1
        return False
    raise ValueError(f"Unknown credit unlock rule: {rule}")


def relock_fast_signal(rule: str, row: pd.Series, days_since_unlock: int | None) -> bool:
    if days_since_unlock is None or days_since_unlock > 21:
        return False
    if row["refined_regime_confirmed"] not in {"FLAT_LOW_RATE", "FLAT_HIGH_RATE"}:
        return False
    dd = bool(row["spy_drawdown_from_previous_high"] <= -0.05)
    d15 = float(row["D_CREDIT_SPREAD_15D"]) if pd.notna(row["D_CREDIT_SPREAD_15D"]) else np.nan
    if rule == "FAST":
        return bool(dd and pd.notna(d15) and d15 > 0)
    if rule == "FAST_WITH_VIX":
        return bool((not row["SPY_above_MA20"]) and ((pd.notna(d15) and d15 > 0) or row["VIX_ZSCORE_120D"] > 1.5))
    return False


def simulate_strategy(panel: pd.DataFrame, templates: dict[str, pd.DataFrame], config: StrategyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = pd.DataFrame(0.0, index=panel.index, columns=ASSETS)
    records: list[dict[str, object]] = []
    pending_full_risk = False
    pending_locks: set[str] = set()
    last_credit_unlock_idx: int | None = None
    prev_credit_active = False
    prev_any_active = False
    state: dict[str, object] = {"credit_neg_run": 0, "credit_cooldown_wait": None}

    for i, row in panel.iterrows():
        current_full_risk = pending_full_risk
        current_locks = set(pending_locks)
        active_credit = current_full_risk and ("CREDIT" in current_locks)
        active_vix = current_full_risk and ("VIX" in current_locks)
        active_cmdty = current_full_risk and ("CMDTY" in current_locks)

        if current_full_risk and row["refined_regime_confirmed"] != "INVERTED":
            alloc, alloc_state = stress_allocation(str(row["refined_regime_confirmed"]))
        else:
            alloc, alloc_state = normal_allocation(row, i, templates)
        weights.loc[i, ASSETS] = pd.Series(normalize_weight_dict(alloc))

        signal_locks = allowed_non_credit_entries(row)
        days_since_credit_unlock = None if last_credit_unlock_idx is None else i - last_credit_unlock_idx
        credit_entry = credit_entry_baseline(row)
        if config.relock_rule != "NONE" and relock_fast_signal(config.relock_rule, row, days_since_credit_unlock):
            credit_entry = True
        if credit_entry:
            signal_locks.add("CREDIT")

        lock_added_today: set[str] = set()
        lock_unlocked_today: set[str] = set()
        credit_entry_marker = False
        credit_unlock_marker = False
        relock_marker = False

        if current_full_risk:
            new_locks = signal_locks - current_locks
            current_locks |= new_locks
            lock_added_today |= new_locks
            if "CREDIT" in new_locks and last_credit_unlock_idx is not None and days_since_credit_unlock is not None and days_since_credit_unlock <= 21:
                relock_marker = True

            unlocked: set[str] = set()
            if "VIX" in current_locks and vix_unlock(row):
                unlocked.add("VIX")
                if "CREDIT" in current_locks:
                    unlocked.add("CREDIT")
            if "CMDTY" in current_locks and cmdty_unlock(row):
                unlocked.add("CMDTY")
            if "CREDIT" in current_locks and "CREDIT" not in unlocked:
                if credit_unlock_signal(config.credit_unlock_rule, row, state):
                    unlocked.add("CREDIT")
            current_locks -= unlocked
            lock_unlocked_today |= unlocked
            pending_locks = set(current_locks)
            pending_full_risk = bool(pending_locks)
        else:
            pending_locks = set(signal_locks)
            pending_full_risk = bool(pending_locks)
            lock_added_today |= pending_locks
            if "CREDIT" in pending_locks and last_credit_unlock_idx is not None and days_since_credit_unlock is not None and days_since_credit_unlock <= 21:
                relock_marker = True

        next_credit_active = pending_full_risk and ("CREDIT" in pending_locks)
        any_active = current_full_risk
        next_any_active = pending_full_risk
        credit_entry_marker = next_credit_active and not active_credit
        credit_unlock_marker = active_credit and not next_credit_active

        if credit_unlock_marker:
            last_credit_unlock_idx = i + 1 if i + 1 < len(panel) else i
            state["credit_neg_run"] = 0
            state["credit_cooldown_wait"] = None
        elif not active_credit:
            state["credit_neg_run"] = 0
            if config.credit_unlock_rule.startswith("COOLDOWN"):
                state["credit_cooldown_wait"] = None

        records.append(
            {
                "date": row["date"],
                "strategy": config.name,
                "final_regime": row["final_regime"],
                "base_regime": row["base_regime"],
                "credit_lock_active": active_credit,
                "vix_lock_active": active_vix,
                "cmdty_lock_active": active_cmdty,
                "any_lock_active": any_active,
                "active_locks": "+".join(sorted(current_locks)),
                "allocation_state": alloc_state,
                "credit_entry_marker": credit_entry_marker,
                "credit_unlock_marker": credit_unlock_marker,
                "relock_marker": relock_marker,
                "lock_added_today": "+".join(sorted(lock_added_today)),
                "lock_unlocked_today": "+".join(sorted(lock_unlocked_today)),
                "SPY_return": row["SPY_return"],
                "SPY_MA20": row["SPY_MA20"],
                "SPY_MA50": row["SPY_MA50"],
                "SPY_MA100": row["SPY_MA100"],
                "SPY_drawdown": row["spy_drawdown_from_previous_high"],
                "CREDIT_SPREAD": row["CREDIT_SPREAD"],
                "D_CREDIT_15D": row["D_CREDIT_SPREAD_15D"],
                "D_CREDIT_20D": row["D_CREDIT_SPREAD_20D"],
                "D_CREDIT_60D": row["D_CREDIT_SPREAD_60D"],
                "CREDIT_SPREAD_Z_252D": row["CREDIT_SPREAD_Z_252D"],
                "CREDIT_SPREAD_Q75_252D": row["CREDIT_SPREAD_Q75_252D"],
                "VIX_Z": row["VIX_ZSCORE_120D"],
                "CMDTY_RET60": row["CMDTY_RET60"],
                "spy_price": row["spy_price"],
                "GS1": row["GS1"],
            }
        )
        prev_credit_active = active_credit
        prev_any_active = any_active

    state_df = pd.DataFrame(records)
    result = compute_strategy(panel, weights, config.name)
    state_df = pd.concat([state_df, weights.add_prefix("weight_").reset_index(drop=True), result.reset_index(drop=True)], axis=1)
    return state_df, result


def period_return(ret: pd.Series) -> float:
    if ret.empty:
        return np.nan
    return float((1.0 + ret.fillna(0.0)).prod() - 1.0)


def period_mdd(ret: pd.Series) -> float:
    if ret.empty:
        return np.nan
    nav = (1.0 + ret.fillna(0.0)).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def forward_stats(panel: pd.DataFrame, idx: int, col: str, horizon: int) -> tuple[float, float]:
    sub = panel.iloc[idx + 1 : idx + 1 + horizon]
    if sub.empty:
        return np.nan, np.nan
    return period_return(sub[col]), period_mdd(sub[col])


def extract_credit_episodes(state_df: pd.DataFrame, panel: pd.DataFrame, strategy: str) -> pd.DataFrame:
    active = state_df["credit_lock_active"].fillna(False).astype(bool)
    start_idx = state_df.index[active & ~active.shift(1, fill_value=False)].tolist()
    end_idx = state_df.index[active & ~active.shift(-1, fill_value=False)].tolist()
    rows = []
    for n, (s, e) in enumerate(zip(start_idx, end_idx), start=1):
        unlock_idx = min(e + 1, len(state_df) - 1)
        sub_state = state_df.loc[s:e]
        sub_panel = panel.loc[s:e]
        trough_price = float(sub_panel["spy_price"].min())
        unlock_price = float(panel.loc[unlock_idx, "spy_price"])
        spy_rebound = unlock_price / trough_price - 1.0 if trough_price > 0 else np.nan
        f21_spy_ret, f21_spy_mdd = forward_stats(panel, unlock_idx, "SPY_return", 21)
        f63_spy_ret, f63_spy_mdd = forward_stats(panel, unlock_idx, "SPY_return", 63)
        f21_strat_ret, _ = forward_stats(state_df, unlock_idx, f"{strategy}_return", 21)
        f63_strat_ret, _ = forward_stats(state_df, unlock_idx, f"{strategy}_return", 63)
        future_lock_63 = bool(state_df.iloc[unlock_idx + 1 : unlock_idx + 64]["any_lock_active"].any()) if unlock_idx + 1 < len(state_df) else False
        false_recovery = bool(
            (pd.notna(f21_spy_mdd) and f21_spy_mdd <= -0.05)
            or (pd.notna(f63_spy_mdd) and f63_spy_mdd <= -0.08)
            or future_lock_63
        )
        missed_rebound = bool(pd.notna(spy_rebound) and spy_rebound > 0.08)
        relock_21 = bool(state_df.iloc[unlock_idx + 1 : unlock_idx + 22]["credit_lock_active"].any()) if unlock_idx + 1 < len(state_df) else False
        relock_63 = bool(state_df.iloc[unlock_idx + 1 : unlock_idx + 64]["credit_lock_active"].any()) if unlock_idx + 1 < len(state_df) else False
        dominant = sub_panel["final_regime_confirmed"].mode()
        rows.append(
            {
                "strategy": strategy,
                "episode_id": n,
                "entry_date": state_df.loc[s, "date"],
                "unlock_date": state_df.loc[unlock_idx, "date"],
                "duration_days": int(e - s + 1),
                "macro_regime_at_entry": sub_panel.iloc[0]["macro_regime_confirmed"],
                "macro_regime_dominant": dominant.iloc[0] if not dominant.empty else sub_panel.iloc[0]["final_regime_confirmed"],
                "final_regime_at_entry": sub_panel.iloc[0]["final_regime_confirmed"],
                "entry_SPY_drawdown": sub_panel.iloc[0]["spy_drawdown_from_previous_high"],
                "entry_credit_spread": sub_panel.iloc[0]["CREDIT_SPREAD"],
                "entry_D_CREDIT_15D": sub_panel.iloc[0]["D_CREDIT_SPREAD_15D"],
                "entry_D_CREDIT_20D": sub_panel.iloc[0]["D_CREDIT_SPREAD_20D"],
                "entry_VIX_Z": sub_panel.iloc[0]["VIX_ZSCORE_120D"],
                "entry_SPY_vs_MA20": sub_panel.iloc[0]["spy_price"] / sub_panel.iloc[0]["SPY_MA20"] - 1.0 if pd.notna(sub_panel.iloc[0]["SPY_MA20"]) else np.nan,
                "unlock_credit_spread": panel.loc[unlock_idx, "CREDIT_SPREAD"],
                "unlock_D_CREDIT_15D": panel.loc[unlock_idx, "D_CREDIT_SPREAD_15D"],
                "unlock_D_CREDIT_20D": panel.loc[unlock_idx, "D_CREDIT_SPREAD_20D"],
                "unlock_SPY_vs_MA20": panel.loc[unlock_idx, "SPY_vs_MA20"],
                "unlock_SPY_vs_MA50": panel.loc[unlock_idx, "SPY_vs_MA50"],
                "unlock_VIX_Z": panel.loc[unlock_idx, "VIX_ZSCORE_120D"],
                "SPY_return_during_lock": period_return(sub_panel["SPY_return"]),
                "SPY_maxDD_during_lock": period_mdd(sub_panel["SPY_return"]),
                "hedge_return_during_lock": period_return(state_df.loc[s:e, [f"weight_{a}" for a in ASSETS]].mul(sub_panel[[f"{a}_return" for a in ASSETS]].to_numpy()).sum(axis=1)),
                "strategy_return_during_lock": period_return(state_df.loc[s:e, f"{strategy}_return"]),
                "next_21d_SPY_return_after_unlock": f21_spy_ret,
                "next_21d_SPY_maxDD_after_unlock": f21_spy_mdd,
                "next_63d_SPY_return_after_unlock": f63_spy_ret,
                "next_63d_SPY_maxDD_after_unlock": f63_spy_mdd,
                "next_21d_strategy_return_after_unlock": f21_strat_ret,
                "next_63d_strategy_return_after_unlock": f63_strat_ret,
                "relock_within_21d": relock_21,
                "relock_within_63d": relock_63,
                "false_recovery_flag": false_recovery,
                "missed_rebound_flag": missed_rebound,
            }
        )
    return pd.DataFrame(rows)


def summarize_strategy(strategy: str, state_df: pd.DataFrame, episodes: pd.DataFrame) -> dict[str, object]:
    perf = performance_metrics(state_df, strategy)
    return {
        "strategy": strategy,
        **perf,
        "annualized_vol": perf["annualized_volatility"],
        "transaction_cost_drag": perf["transaction_cost"],
        "time_in_credit_lock": int(state_df["credit_lock_active"].sum()),
        "number_credit_entries": int(state_df["credit_entry_marker"].sum()),
        "number_credit_unlocks": int(state_df["credit_unlock_marker"].sum()),
        "avg_credit_lock_duration": float(episodes["duration_days"].mean()) if not episodes.empty else np.nan,
        "false_recovery_count": int(episodes["false_recovery_flag"].sum()) if not episodes.empty else 0,
        "relock_within_21d_count": int(episodes["relock_within_21d"].sum()) if not episodes.empty else 0,
        "missed_rebound_count": int(episodes["missed_rebound_flag"].sum()) if not episodes.empty else 0,
    }


def summarize_window(name: str, start: str, end: str | None, strategy: str, state_df: pd.DataFrame, episodes: pd.DataFrame) -> dict[str, object]:
    sub = state_df.loc[state_df["date"] >= pd.Timestamp(start)].copy()
    if end is not None:
        sub = sub.loc[sub["date"] <= pd.Timestamp(end)].copy()
    if sub.empty:
        return {
            "strategy": strategy,
            "window": name,
            "cumulative_return": np.nan,
            "max_drawdown": np.nan,
            "Sharpe": np.nan,
            "time_in_credit_lock": 0,
            "number_credit_unlocks": 0,
            "false_recovery_count": 0,
            "avg_weight_SPY": np.nan,
            "avg_weight_GOLD": np.nan,
            "avg_weight_IEF": np.nan,
            "avg_weight_CASH": np.nan,
            "avg_weight_CMDTY": np.nan,
        }
    ret_col = f"{strategy}_return"
    nav = (1.0 + sub[ret_col].fillna(0.0)).cumprod()
    ann_vol = sub[ret_col].std(ddof=1) * np.sqrt(252.0)
    ann_ret = nav.iloc[-1] ** (252.0 / len(sub)) - 1.0
    window_eps = episodes.loc[(episodes["entry_date"] >= pd.Timestamp(start)) & ((episodes["entry_date"] <= pd.Timestamp(end)) if end else True)]
    return {
        "strategy": strategy,
        "window": name,
        "cumulative_return": float(nav.iloc[-1] - 1.0),
        "max_drawdown": float((nav / nav.cummax() - 1.0).min()),
        "Sharpe": float(ann_ret / ann_vol) if ann_vol > 0 else np.nan,
        "time_in_credit_lock": int(sub["credit_lock_active"].sum()),
        "number_credit_unlocks": int(sub["credit_unlock_marker"].sum()),
        "false_recovery_count": int(window_eps["false_recovery_flag"].sum()) if not window_eps.empty else 0,
        "avg_weight_SPY": float(sub["weight_SPY"].mean()),
        "avg_weight_GOLD": float(sub["weight_GOLD"].mean()),
        "avg_weight_IEF": float(sub["weight_IEF"].mean()),
        "avg_weight_CASH": float(sub["weight_CASH"].mean()),
        "avg_weight_CMDTY": float(sub["weight_CMDTY_FUT"].mean()),
    }


def baseline_case_window_csv(name: str, start: str, end: str | None, panel: pd.DataFrame, state_df: pd.DataFrame, episodes: pd.DataFrame) -> None:
    sub = panel.loc[panel["date"] >= pd.Timestamp(start)].copy()
    if end is not None:
        sub = sub.loc[sub["date"] <= pd.Timestamp(end)].copy()
    state_sub = state_df.loc[state_df["date"].between(sub["date"].min(), sub["date"].max())].copy()
    merged = sub.merge(
        state_sub[["date", "credit_lock_active", "vix_lock_active", "cmdty_lock_active", "any_lock_active", f"{BASELINE}_nav", f"{BASELINE}_drawdown", "credit_entry_marker", "credit_unlock_marker", "relock_marker"]],
        on="date",
        how="left",
    )
    false_dates = set(episodes.loc[episodes["false_recovery_flag"], "unlock_date"])
    merged["false_recovery_marker"] = merged["date"].isin(false_dates)
    merged.rename(
        columns={
            "spy_price": "SPY_price",
            "spy_drawdown_from_previous_high": "SPY_drawdown",
            "CREDIT_SPREAD": "CREDIT_SPREAD",
            "D_CREDIT_SPREAD_15D": "D_CREDIT_15D",
            "D_CREDIT_SPREAD_20D": "D_CREDIT_20D",
            "VIX_ZSCORE_120D": "VIX_Z",
            "credit_lock_active": "credit_lock_state_baseline",
            "vix_lock_active": "vix_lock_state",
            "cmdty_lock_active": "cmdty_lock_state",
            "any_lock_active": "final_state",
            f"{BASELINE}_nav": "baseline_strategy_NAV",
            f"{BASELINE}_drawdown": "baseline_strategy_drawdown",
        },
        inplace=True,
    )
    merged.to_csv(OUT / f"case_{name}_credit_lock.csv", index=False)


def match_baseline_episode(base_row: pd.Series, episodes: pd.DataFrame) -> pd.Series | None:
    if episodes.empty:
        return None
    base_entry = pd.Timestamp(base_row["entry_date"])
    base_unlock = pd.Timestamp(base_row["unlock_date"])
    overlap = episodes.loc[(episodes["entry_date"] <= base_unlock) & (episodes["unlock_date"] >= base_entry)]
    if not overlap.empty:
        return overlap.sort_values("entry_date").iloc[0]
    near = episodes.loc[(episodes["entry_date"] >= base_entry - pd.Timedelta(days=10)) & (episodes["entry_date"] <= base_entry + pd.Timedelta(days=21))]
    if not near.empty:
        return near.sort_values((near["entry_date"] - base_entry).abs()).iloc[0]
    return None


def build_episode_comparison(all_episodes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    baseline = all_episodes[BASELINE]
    rows = []
    for _, base_row in baseline.iterrows():
        for strategy, episodes in all_episodes.items():
            match = base_row if strategy == BASELINE else match_baseline_episode(base_row, episodes)
            rows.append(
                {
                    "episode_id": int(base_row["episode_id"]),
                    "strategy": strategy,
                    "entry_date": match["entry_date"] if match is not None else pd.NaT,
                    "unlock_date": match["unlock_date"] if match is not None else pd.NaT,
                    "duration_days": match["duration_days"] if match is not None else np.nan,
                    "SPY_return_during_lock": match["SPY_return_during_lock"] if match is not None else np.nan,
                    "strategy_return_during_lock": match["strategy_return_during_lock"] if match is not None else np.nan,
                    "next_21d_SPY_return_after_unlock": match["next_21d_SPY_return_after_unlock"] if match is not None else np.nan,
                    "next_21d_SPY_maxDD_after_unlock": match["next_21d_SPY_maxDD_after_unlock"] if match is not None else np.nan,
                    "false_recovery_flag": match["false_recovery_flag"] if match is not None else np.nan,
                    "missed_rebound_flag": match["missed_rebound_flag"] if match is not None else np.nan,
                }
            )
    return pd.DataFrame(rows)


def rank_candidates(perf_df: pd.DataFrame, crisis_df: pd.DataFrame) -> dict[str, str]:
    base = perf_df.loc[perf_df["strategy"] == BASELINE].iloc[0]
    cands = perf_df.loc[perf_df["strategy"] != BASELINE].copy()
    cands["false_recovery_improvement"] = base["false_recovery_count"] - cands["false_recovery_count"]
    cands["missed_rebound_penalty"] = cands["missed_rebound_count"] - base["missed_rebound_count"]
    best_false = cands.sort_values(["false_recovery_count", "Sharpe", "MaxDD"], ascending=[True, False, False]).iloc[0]["strategy"]
    c2008 = crisis_df.loc[crisis_df["window"] == "2008_GFC"].sort_values(["max_drawdown", "cumulative_return"], ascending=[False, False]).iloc[0]["strategy"]
    c2022 = crisis_df.loc[crisis_df["window"] == "2022_RATE_WAR"].sort_values(["max_drawdown", "cumulative_return"], ascending=[False, False]).iloc[0]["strategy"]
    ranks = cands[["strategy"]].copy()
    ranks["r_false"] = cands["false_recovery_count"].rank(method="min", ascending=True)
    ranks["r_missed"] = cands["missed_rebound_count"].rank(method="min", ascending=True)
    ranks["r_sharpe"] = cands["Sharpe"].rank(method="min", ascending=False)
    ranks["r_maxdd"] = cands["MaxDD"].rank(method="min", ascending=False)
    ranks["score"] = ranks[["r_false", "r_missed", "r_sharpe", "r_maxdd"]].sum(axis=1)
    best_balanced = ranks.sort_values("score").iloc[0]["strategy"]
    best_bal_row = cands.loc[cands["strategy"] == best_balanced].iloc[0]
    should_change = bool(
        best_bal_row["false_recovery_count"] < base["false_recovery_count"]
        and best_bal_row["MaxDD"] >= base["MaxDD"]
        and best_bal_row["Sharpe"] >= base["Sharpe"] - 0.02
    )
    return {
        "best_false_recovery": best_false,
        "best_2008_maxdd": c2008,
        "best_2022_maxdd": c2022,
        "best_balanced": best_balanced,
        "should_change": "YES" if should_change else "NO",
    }


def plot_episode_timeline(panel: pd.DataFrame, episodes: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(panel["date"], panel["spy_price"], color="black", linewidth=1.0)
    for _, row in episodes.iterrows():
        ax.axvspan(row["entry_date"], row["unlock_date"], color="#d95f0e", alpha=0.18)
        ax.axvline(row["entry_date"], color="#d7301f", linestyle="-", linewidth=0.8)
        ax.axvline(row["unlock_date"], color="#3182bd", linestyle="--", linewidth=0.8)
        if bool(row["false_recovery_flag"]):
            ax.scatter(row["unlock_date"], panel.loc[panel["date"] == row["unlock_date"], "spy_price"].iloc[0], color="red", s=18, zorder=5)
    ax.set_title("Baseline Credit Lock Episodes")
    ax.set_yscale("log")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "credit_lock_episodes_timeline.png", dpi=160)
    plt.close(fig)


def plot_case_comparison(name: str, start: str, end: str | None, all_states: dict[str, pd.DataFrame], picks: dict[str, str]) -> None:
    selected = [BASELINE, picks["best_2008_maxdd"], picks["best_2022_maxdd"], picks["best_balanced"]]
    selected = list(dict.fromkeys(selected))
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    for strat in selected:
        sub = all_states[strat].loc[all_states[strat]["date"] >= pd.Timestamp(start)].copy()
        if end is not None:
            sub = sub.loc[sub["date"] <= pd.Timestamp(end)]
        axes[0].plot(sub["date"], sub["spy_price"], label="SPY" if strat == selected[0] else None, color="black", linewidth=1.0, alpha=0.7)
        axes[1].plot(sub["date"], sub["CREDIT_SPREAD"], label=f"{strat} credit", linewidth=0.9)
        axes[2].plot(sub["date"], sub[f"{strat}_nav"], label=strat, linewidth=1.0)
        lock = sub["credit_lock_active"].astype(bool)
        for ax in axes:
            ax.fill_between(sub["date"], 0, 1, where=lock, transform=ax.get_xaxis_transform(), color="#fdae6b", alpha=0.08)
    axes[0].set_title(f"{name}: Credit Unlock Comparison")
    axes[1].set_title("Credit Spread")
    axes[2].set_title("Strategy NAV")
    axes[2].legend(frameon=False, ncol=2)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / f"case_{name}_credit_unlock_comparison.png", dpi=160)
    plt.close(fig)


def plot_simple_bars(perf_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    d = perf_df.sort_values("false_recovery_count")
    ax.bar(d["strategy"], d["false_recovery_count"])
    ax.tick_params(axis="x", labelrotation=60)
    ax.set_title("False Recovery Count by Strategy")
    fig.tight_layout()
    fig.savefig(FIG / "false_recovery_by_strategy.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    d = perf_df.sort_values("missed_rebound_count")
    ax.bar(d["strategy"], d["missed_rebound_count"])
    ax.tick_params(axis="x", labelrotation=60)
    ax.set_title("Missed Rebound Count by Strategy")
    fig.tight_layout()
    fig.savefig(FIG / "missed_rebound_by_strategy.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    sc = ax.scatter(
        perf_df["false_recovery_count"],
        perf_df["missed_rebound_count"],
        s=60 + 120 * perf_df["Sharpe"].fillna(0),
        c=perf_df["MaxDD"],
        cmap="viridis",
    )
    for _, row in perf_df.iterrows():
        ax.text(row["false_recovery_count"] + 0.05, row["missed_rebound_count"] + 0.05, row["strategy"], fontsize=8)
    ax.set_xlabel("False recovery count")
    ax.set_ylabel("Missed rebound count")
    ax.set_title("Credit Unlock Trade-off")
    fig.colorbar(sc, ax=ax, label="MaxDD")
    fig.tight_layout()
    fig.savefig(FIG / "performance_tradeoff_credit_unlock.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    dd = perf_df[["strategy", "avg_credit_lock_duration"]].sort_values("avg_credit_lock_duration")
    ax.bar(dd["strategy"], dd["avg_credit_lock_duration"])
    ax.tick_params(axis="x", labelrotation=60)
    ax.set_title("Average Credit Lock Duration by Strategy")
    fig.tight_layout()
    fig.savefig(FIG / "credit_unlock_duration_distribution.png", dpi=160)
    plt.close(fig)


def plot_crisis_heatmap(crisis_df: pd.DataFrame) -> None:
    heat = crisis_df.pivot(index="strategy", columns="window", values="cumulative_return")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(heat, annot=True, fmt=".1%", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title("Crisis Window Cumulative Return by Strategy")
    fig.tight_layout()
    fig.savefig(FIG / "crisis_window_heatmap_credit_unlock.png", dpi=160)
    plt.close(fig)


def build_readme_patch(should_change: str, picks: dict[str, str]) -> None:
    if should_change == "YES":
        text = "\n".join(
            [
                "The credit unlock rule was updated after dedicated lock/unlock diagnostics.",
                "",
                f"Recommended replacement: `{picks['best_balanced']}`.",
                "Update the final trigger-lock table in the README to reflect the new credit unlock rule and relock logic.",
            ]
        )
    else:
        text = (
            "We also tested stricter credit unlock and fast relock rules. Although some variants improved 2008 or 2022 locally, "
            "the trade-off with missed rebounds and reduced upside was not strong enough to replace the simpler baseline credit lock."
        )
    (OUT / "README_PATCH_SUGGESTION.md").write_text(text, encoding="utf-8")


def build_report(perf_df: pd.DataFrame, crisis_df: pd.DataFrame, baseline_eps: pd.DataFrame, picks: dict[str, str]) -> None:
    base = perf_df.loc[perf_df["strategy"] == BASELINE].iloc[0]
    best_bal = perf_df.loc[perf_df["strategy"] == picks["best_balanced"]].iloc[0]
    lines = [
        "# CREDIT_LOCK_EXIT_DIAGNOSTIC_REPORT",
        "",
        "## 1. Purpose",
        "",
        "This diagnostic studies only the credit trigger lock / unlock logic. Allocation, regime framework, VIX lock, and commodity lock are left unchanged.",
        "",
        "## 2. Current problem",
        "",
        f"- Baseline credit episodes: {len(baseline_eps)}",
        f"- Baseline false recovery count: {int(base['false_recovery_count'])}",
        f"- Baseline missed rebound count: {int(base['missed_rebound_count'])}",
        "- The baseline unlock uses the current final-rule credit logic from the mainline.",
        "",
        "## 3. Baseline credit lock episode diagnostics",
        "",
        "- 2022-2023 is the main window where early unlock remains a concern.",
        "- 2008 contains both dead-cat bounce risk and delayed recovery trade-offs.",
        "",
        "## 4. Candidate unlock rules",
        "",
        "- Candidates test stricter confirmation, MA50 trend confirmation, spread-level normalization, z-score normalization, cooldown, drawdown repair, and fast relock.",
        "",
        "## 5. Full-sample performance",
        "",
        f"- Baseline Sharpe: {base['Sharpe']:.3f}, MaxDD: {base['MaxDD']:.2%}, Final Equity: {base['final_equity']:.2f}",
        f"- Best balanced candidate `{picks['best_balanced']}`: Sharpe {best_bal['Sharpe']:.3f}, MaxDD {best_bal['MaxDD']:.2%}, Final Equity {best_bal['final_equity']:.2f}",
        "",
        "## 6. Crisis window analysis",
        "",
        "- Compare `2008_GFC`, `2022_RATE_WAR`, `COVID_2020`, and `2025_PULLBACK` in the crisis comparison table.",
        "",
        "## 7. Trade-off discussion",
        "",
        "- Stricter unlocks can reduce false recovery, but they can also increase missed rebound count and keep the strategy in hedge mode too long.",
        "- Fast relock variants test whether re-lock is more effective than simply delaying unlock.",
        "",
        "## 8. Recommendation",
        "",
        f"- Best false-recovery candidate: `{picks['best_false_recovery']}`",
        f"- Best 2008 candidate: `{picks['best_2008_maxdd']}`",
        f"- Best 2022 candidate: `{picks['best_2022_maxdd']}`",
        f"- Best balanced candidate: `{picks['best_balanced']}`",
        f"- Final strategy should change: `{picks['should_change']}`",
        "",
        "## 9. Proposed final credit rule if any",
        "",
        "Only adopt a replacement if the best balanced candidate improves false recovery and drawdown without clearly damaging rebound capture.",
        "",
        "## 10. Limitations",
        "",
        "- Credit stress samples are sparse.",
        "- 2008 and 2022 are not the same failure mode.",
        "- Unlock and relock thresholds still need OOS validation.",
    ]
    (OUT / "CREDIT_LOCK_EXIT_DIAGNOSTIC_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = prepare_panel(load_panel())
    templates = build_weight_templates(panel)
    configs = build_configs()

    all_states: dict[str, pd.DataFrame] = {}
    all_episodes: dict[str, pd.DataFrame] = {}
    perf_rows: list[dict[str, object]] = []
    crisis_rows: list[dict[str, object]] = []

    for cfg in configs:
        state_df, _ = simulate_strategy(panel, templates, cfg)
        episodes = extract_credit_episodes(state_df, panel, cfg.name)
        all_states[cfg.name] = state_df
        all_episodes[cfg.name] = episodes
        perf_rows.append(summarize_strategy(cfg.name, state_df, episodes))
        for window_name, (start, end) in WINDOWS.items():
            crisis_rows.append(summarize_window(window_name, start, end, cfg.name, state_df, episodes))

    perf_df = pd.DataFrame(perf_rows).sort_values("strategy").reset_index(drop=True)
    crisis_df = pd.DataFrame(crisis_rows)
    baseline_eps = all_episodes[BASELINE].copy()
    baseline_eps.to_csv(OUT / "credit_lock_episodes.csv", index=False)
    for window_name, (start, end) in WINDOWS.items():
        baseline_case_window_csv(window_name, start, end, panel, all_states[BASELINE], baseline_eps)

    episode_cmp = build_episode_comparison(all_episodes)
    episode_cmp.to_csv(OUT / "credit_unlock_episode_comparison.csv", index=False)
    perf_df.to_csv(OUT / "credit_unlock_strategy_performance.csv", index=False)
    crisis_df.to_csv(OUT / "credit_unlock_crisis_comparison.csv", index=False)

    picks = rank_candidates(perf_df, crisis_df)
    picks_df = pd.DataFrame([picks])
    picks_df.to_csv(OUT / "credit_unlock_recommendation_summary.csv", index=False)

    plot_episode_timeline(panel, baseline_eps)
    for case_name, (start, end) in WINDOWS.items():
        plot_case_comparison(case_name, start, end, all_states, picks)
    plot_simple_bars(perf_df)
    plot_crisis_heatmap(crisis_df)

    build_report(perf_df, crisis_df, baseline_eps, picks)
    build_readme_patch(picks["should_change"], picks)

    baseline = perf_df.loc[perf_df["strategy"] == BASELINE].iloc[0]
    best_false = perf_df.loc[perf_df["strategy"] == picks["best_false_recovery"]].iloc[0]
    best_2008 = crisis_df.loc[crisis_df["window"] == "2008_GFC"].sort_values(["max_drawdown", "cumulative_return"], ascending=[False, False]).iloc[0]
    best_2022 = crisis_df.loc[crisis_df["window"] == "2022_RATE_WAR"].sort_values(["max_drawdown", "cumulative_return"], ascending=[False, False]).iloc[0]
    best_bal = perf_df.loc[perf_df["strategy"] == picks["best_balanced"]].iloc[0]

    print("baseline credit episode count:", len(baseline_eps))
    print("baseline false recovery count:", int(baseline["false_recovery_count"]))
    print("baseline missed rebound count:", int(baseline["missed_rebound_count"]))
    print("best candidate by false recovery reduction:", picks["best_false_recovery"])
    print("best candidate by 2022 MaxDD:", best_2022["strategy"], best_2022["max_drawdown"])
    print("best candidate by 2008 MaxDD:", best_2008["strategy"], best_2008["max_drawdown"])
    print(
        "best balanced candidate by Sharpe / MaxDD / false recovery / missed rebound:",
        picks["best_balanced"],
        f"Sharpe={best_bal['Sharpe']:.3f}",
        f"MaxDD={best_bal['MaxDD']:.2%}",
        f"false_recovery={int(best_bal['false_recovery_count'])}",
        f"missed_rebound={int(best_bal['missed_rebound_count'])}",
    )
    print("whether final strategy should change:", picks["should_change"])
    print("output paths:")
    print(str(OUT))


if __name__ == "__main__":
    main()
