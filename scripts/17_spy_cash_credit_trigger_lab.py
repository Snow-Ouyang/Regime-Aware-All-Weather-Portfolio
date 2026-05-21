from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from final_strategy_source_only_core import ROOT, compute_strategy, performance_metrics


OUT = ROOT / "results" / "spy_cash_credit_trigger_lab"
FIG = OUT / "figures"
MAIN = ROOT / "results" / "main_pipeline_final" / "tables"
ASSETS = ["SPY", "GOLD", "CMDTY_FUT", "IEF", "CASH"]
ONE_WAY_COST_BPS = 10.0

SPY_BUY_HOLD = "SPY_BUY_HOLD"
SPY_CASH_FINAL_LOCKS = "SPY_CASH_FINAL_LOCKS"
SPY_CASH_NO_CREDIT = "SPY_CASH_NO_CREDIT"
SPY_CASH_CREDIT_ONLY = "SPY_CASH_CREDIT_ONLY"

WINDOWS = {
    "2008_GFC": ("2007-10-01", "2009-06-30"),
    "2015_2016": ("2015-05-01", "2016-03-31"),
    "2018Q4": ("2018-10-01", "2019-01-31"),
    "COVID_2020": ("2020-02-01", "2020-06-30"),
    "2022_RATE_WAR": ("2021-11-01", "2023-03-31"),
    "2025_PULLBACK": ("2025-01-01", None),
}

VARIANTS = [
    "BASELINE_ABS",
    "ABS_20D",
    "ABS_OR_DZ_15D",
    "DZ_ONLY_15D_1P5",
    "DZ_ONLY_15D_2P0",
    "LEVEL_Z_ENTRY",
    "ABS_ENTRY_LEVEL_Z_UNLOCK",
    "ABS_ENTRY_PERCENTILE_UNLOCK",
    "ABS_ENTRY_MA50_UNLOCK",
    "ABS_ENTRY_MA20_MA50_UNLOCK",
    "ABS_ENTRY_3D_CONFIRM_UNLOCK",
    "ABS_ENTRY_COOLDOWN_10D_UNLOCK",
    "ABS_ENTRY_FAST_RELOCK",
    "ABS_ENTRY_FAST_RELOCK_WITH_VIX",
    "HYBRID_BEST_EFFORT",
]


@dataclass
class CreditVariant:
    name: str
    entry_rule: str
    unlock_rule: str
    relock_rule: str = "NONE"


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def load_panel() -> pd.DataFrame:
    path = MAIN / "daily_backtest_panel.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing main pipeline panel: {path}")
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return df


def prepare_panel(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["CREDIT_SPREAD"] = out["WBAA"] - out["WAAA"]
    out["D_CREDIT_15D"] = out["CREDIT_SPREAD"] - out["CREDIT_SPREAD"].shift(15)
    out["D_CREDIT_20D"] = out["CREDIT_SPREAD"] - out["CREDIT_SPREAD"].shift(20)
    out["D_CREDIT_60D"] = out["CREDIT_SPREAD"] - out["CREDIT_SPREAD"].shift(60)
    out["SPY_MA20"] = out["SPY_MA20"] if "SPY_MA20" in out.columns else out["spy_price"].rolling(20, min_periods=20).mean()
    out["SPY_MA50"] = out["spy_price"].rolling(50, min_periods=50).mean()
    out["SPY_MA100"] = out["spy_price"].rolling(100, min_periods=100).mean()
    out["SPY_above_MA20"] = out["spy_price"] > out["SPY_MA20"]
    out["SPY_above_MA50"] = out["spy_price"] > out["SPY_MA50"]
    out["SPY_above_MA100"] = out["spy_price"] > out["SPY_MA100"]
    out["SPY_DD"] = out["spy_drawdown_from_previous_high"]

    roll_level_252 = out["CREDIT_SPREAD"].rolling(252, min_periods=126)
    roll_level_504 = out["CREDIT_SPREAD"].rolling(504, min_periods=252)
    roll_d15_252 = out["D_CREDIT_15D"].rolling(252, min_periods=126)
    roll_d20_252 = out["D_CREDIT_20D"].rolling(252, min_periods=126)
    roll_d60_252 = out["D_CREDIT_60D"].rolling(252, min_periods=126)

    out["CREDIT_LEVEL_Z_252D"] = (out["CREDIT_SPREAD"] - roll_level_252.mean()) / roll_level_252.std(ddof=1).replace(0, np.nan)
    out["CREDIT_LEVEL_Z_504D"] = (out["CREDIT_SPREAD"] - roll_level_504.mean()) / roll_level_504.std(ddof=1).replace(0, np.nan)
    out["D_CREDIT_15D_Z_252D"] = (out["D_CREDIT_15D"] - roll_d15_252.mean()) / roll_d15_252.std(ddof=1).replace(0, np.nan)
    out["D_CREDIT_20D_Z_252D"] = (out["D_CREDIT_20D"] - roll_d20_252.mean()) / roll_d20_252.std(ddof=1).replace(0, np.nan)
    out["D_CREDIT_60D_Z_252D"] = (out["D_CREDIT_60D"] - roll_d60_252.mean()) / roll_d60_252.std(ddof=1).replace(0, np.nan)
    out["CREDIT_PERCENTILE_252D"] = roll_level_252.rank(pct=True)
    out["CREDIT_PERCENTILE_504D"] = roll_level_504.rank(pct=True)
    out["CREDIT_SPREAD_MA20"] = out["CREDIT_SPREAD"].rolling(20, min_periods=20).mean()
    out["CREDIT_SPREAD_MA60"] = out["CREDIT_SPREAD"].rolling(60, min_periods=60).mean()
    out["CREDIT_ABOVE_MA20"] = out["CREDIT_SPREAD"] > out["CREDIT_SPREAD_MA20"]
    out["CREDIT_ABOVE_MA60"] = out["CREDIT_SPREAD"] > out["CREDIT_SPREAD_MA60"]

    out["final_credit_active"] = out["trigger_lock_active_locks"].fillna("").astype(str).str.contains("CREDIT")
    out["final_vix_active"] = out["trigger_lock_active_locks"].fillna("").astype(str).str.contains("VIX")
    out["final_cmdty_active"] = out["trigger_lock_active_locks"].fillna("").astype(str).str.contains("CMDTY")
    out["final_any_stress"] = out["trigger_lock_full_risk_state"].eq("FULL_RISK")
    return out


def save_feature_panel(df: pd.DataFrame) -> None:
    cols = [
        "date",
        "spy_price",
        "SPY_return",
        "CASH_return",
        "macro_regime_confirmed",
        "final_regime_confirmed",
        "VIX_LEVEL",
        "VIX_ZSCORE_120D",
        "CMDTY_RET60",
        "WBAA",
        "WAAA",
        "CREDIT_SPREAD",
        "D_CREDIT_15D",
        "D_CREDIT_20D",
        "D_CREDIT_60D",
        "CREDIT_LEVEL_Z_252D",
        "CREDIT_LEVEL_Z_504D",
        "D_CREDIT_15D_Z_252D",
        "D_CREDIT_20D_Z_252D",
        "D_CREDIT_60D_Z_252D",
        "CREDIT_PERCENTILE_252D",
        "CREDIT_PERCENTILE_504D",
        "CREDIT_SPREAD_MA20",
        "CREDIT_SPREAD_MA60",
        "CREDIT_ABOVE_MA20",
        "CREDIT_ABOVE_MA60",
        "SPY_MA20",
        "SPY_MA50",
        "SPY_MA100",
        "SPY_DD",
    ]
    df[cols].to_csv(OUT / "credit_feature_panel.csv", index=False)


def build_configs() -> list[CreditVariant]:
    return [
        CreditVariant("BASELINE_ABS", "ABS_15D", "BASE"),
        CreditVariant("ABS_20D", "ABS_20D", "ABS_20D"),
        CreditVariant("ABS_OR_DZ_15D", "ABS_OR_DZ_15D", "BASE"),
        CreditVariant("DZ_ONLY_15D_1P5", "DZ_15D_1P5", "DZ_15D_ZERO"),
        CreditVariant("DZ_ONLY_15D_2P0", "DZ_15D_2P0", "DZ_15D_ZERO"),
        CreditVariant("LEVEL_Z_ENTRY", "LEVEL_Z_ENTRY", "LEVEL_Z_UNLOCK"),
        CreditVariant("ABS_ENTRY_LEVEL_Z_UNLOCK", "ABS_15D", "BASE_PLUS_LEVEL_Z"),
        CreditVariant("ABS_ENTRY_PERCENTILE_UNLOCK", "ABS_15D", "BASE_PLUS_PERCENTILE"),
        CreditVariant("ABS_ENTRY_MA50_UNLOCK", "ABS_15D", "MA50"),
        CreditVariant("ABS_ENTRY_MA20_MA50_UNLOCK", "ABS_15D", "MA20_MA50"),
        CreditVariant("ABS_ENTRY_3D_CONFIRM_UNLOCK", "ABS_15D", "NEG_3D"),
        CreditVariant("ABS_ENTRY_COOLDOWN_10D_UNLOCK", "ABS_15D", "COOLDOWN_10D"),
        CreditVariant("ABS_ENTRY_FAST_RELOCK", "ABS_15D", "BASE", "FAST"),
        CreditVariant("ABS_ENTRY_FAST_RELOCK_WITH_VIX", "ABS_15D", "BASE", "FAST_WITH_VIX"),
        CreditVariant("HYBRID_BEST_EFFORT", "ABS_OR_DZ_15D", "HYBRID_BEST", "FAST_WITH_VIX"),
    ]


def perf_with_alias(df: pd.DataFrame, name: str) -> dict[str, float]:
    p = performance_metrics(df, name)
    return {
        "CAGR": p["CAGR"],
        "Sharpe": p["Sharpe"],
        "Sortino": p["Sortino"],
        "MaxDD": p["MaxDD"],
        "Calmar": p["Calmar"],
        "Final Equity": p["final_equity"],
        "annualized_vol": p["annualized_volatility"],
        "turnover": p["turnover"],
        "transaction_cost_drag": p["transaction_cost"],
    }


def period_return(ret: pd.Series) -> float:
    return float((1.0 + ret.fillna(0.0)).prod() - 1.0) if len(ret) else np.nan


def period_mdd(ret: pd.Series) -> float:
    if len(ret) == 0:
        return np.nan
    nav = (1.0 + ret.fillna(0.0)).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def forward_return(ret: pd.Series, window: int) -> pd.Series:
    vals = []
    arr = ret.fillna(0.0).to_numpy()
    for i in range(len(arr)):
        sub = arr[i + 1 : i + 1 + window]
        vals.append(float(np.prod(1.0 + sub) - 1.0) if len(sub) else np.nan)
    return pd.Series(vals, index=ret.index)


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


def find_episodes(active: pd.Series) -> list[tuple[int, int]]:
    start = active & ~active.shift(1, fill_value=False)
    end = active & ~active.shift(-1, fill_value=False)
    return list(zip(start[start].index.tolist(), end[end].index.tolist()))


def credit_entry_signal(row: pd.Series, rule: str) -> bool:
    if row["refined_regime_confirmed"] not in {"FLAT_LOW_RATE", "FLAT_HIGH_RATE"}:
        return False
    dd_gate = bool(row["SPY_DD"] <= -0.05)
    if not dd_gate:
        return False
    d15 = row["D_CREDIT_15D"]
    d20 = row["D_CREDIT_20D"]
    d15z = row["D_CREDIT_15D_Z_252D"]
    levelz = row["CREDIT_LEVEL_Z_252D"]
    if rule == "ABS_15D":
        return bool(pd.notna(d15) and d15 > 0.10)
    if rule == "ABS_20D":
        return bool(pd.notna(d20) and d20 > 0.10)
    if rule == "ABS_OR_DZ_15D":
        return bool((pd.notna(d15) and d15 > 0.10) or (pd.notna(d15z) and d15z > 1.5))
    if rule == "DZ_15D_1P5":
        return bool(pd.notna(d15z) and d15z > 1.5)
    if rule == "DZ_15D_2P0":
        return bool(pd.notna(d15z) and d15z > 2.0)
    if rule == "LEVEL_Z_ENTRY":
        return bool(pd.notna(levelz) and levelz > 1.0)
    raise ValueError(rule)


def vix_entry_signal(row: pd.Series) -> bool:
    refined = row["refined_regime_confirmed"]
    return bool(refined in {"FLAT_LOW_RATE", "FLAT_HIGH_RATE", "STEEP"} and row["VIX_ZSCORE_120D"] >= 3.0)


def cmdty_entry_signal(row: pd.Series) -> bool:
    return bool(row["refined_regime_confirmed"] == "STEEP" and row["CMDTY_RET60"] < -0.10)


def vix_unlock(row: pd.Series) -> bool:
    return bool((row["VIX_ZSCORE_120D"] < 1.5) and row["SPY_above_MA20"])


def cmdty_unlock(row: pd.Series) -> bool:
    return bool((row["CMDTY_RET60"] > -0.05) and row["SPY_above_MA20"])


def credit_unlock_signal(row: pd.Series, rule: str, state: dict[str, object]) -> bool:
    d15 = row["D_CREDIT_15D"]
    d20 = row["D_CREDIT_20D"]
    d15z = row["D_CREDIT_15D_Z_252D"]
    levelz = row["CREDIT_LEVEL_Z_252D"]
    pct252 = row["CREDIT_PERCENTILE_252D"]
    ma20 = bool(row["SPY_above_MA20"])
    ma50 = bool(row["SPY_above_MA50"])

    neg_d15 = bool(pd.notna(d15) and d15 < 0)
    neg_d20 = bool(pd.notna(d20) and d20 < 0)
    if neg_d15:
        state["neg_run"] = int(state.get("neg_run", 0)) + 1
    else:
        state["neg_run"] = 0

    baseline_ok = neg_d15 and ma20

    if rule == "BASE":
        state["cooldown_wait"] = None
        return baseline_ok
    if rule == "ABS_20D":
        return neg_d20 and ma20
    if rule == "DZ_15D_ZERO":
        return bool(pd.notna(d15z) and d15z < 0 and ma20)
    if rule == "LEVEL_Z_UNLOCK":
        return bool(pd.notna(levelz) and levelz < 0.5 and ma20)
    if rule == "BASE_PLUS_LEVEL_Z":
        return bool(baseline_ok and pd.notna(levelz) and levelz < 1.0)
    if rule == "BASE_PLUS_PERCENTILE":
        return bool(baseline_ok and pd.notna(pct252) and pct252 < 0.75)
    if rule == "MA50":
        return bool(neg_d15 and ma50)
    if rule == "MA20_MA50":
        return bool(neg_d15 and ma20 and ma50)
    if rule == "NEG_3D":
        return bool(int(state["neg_run"]) >= 3 and ma20)
    if rule == "COOLDOWN_10D":
        wait = state.get("cooldown_wait")
        if wait is None:
            if baseline_ok:
                state["cooldown_wait"] = 10
            return False
        if (not ma20) or (pd.notna(d15) and d15 > 0):
            state["cooldown_wait"] = None
            return False
        if int(wait) <= 1:
            state["cooldown_wait"] = None
            return True
        state["cooldown_wait"] = int(wait) - 1
        return False
    if rule == "HYBRID_BEST":
        return bool(int(state["neg_run"]) >= 3 and ma20 and pd.notna(levelz) and levelz < 1.0)
    raise ValueError(rule)


def credit_relock_signal(row: pd.Series, rule: str, days_since_unlock: int | None) -> bool:
    if rule == "NONE" or days_since_unlock is None or days_since_unlock > 21:
        return False
    d15 = row["D_CREDIT_15D"]
    if rule == "FAST":
        return bool((row["SPY_DD"] <= -0.05) and pd.notna(d15) and d15 > 0)
    if rule == "FAST_WITH_VIX":
        return bool((not row["SPY_above_MA20"]) and ((pd.notna(d15) and d15 > 0) or row["VIX_ZSCORE_120D"] > 1.5))
    raise ValueError(rule)


def simulate_variant(panel: pd.DataFrame, cfg: CreditVariant, include_vix: bool, include_cmdty: bool, include_credit: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    current_locks: set[str] = set()
    pending_locks: set[str] = set()
    credit_state: dict[str, object] = {"neg_run": 0, "cooldown_wait": None}
    last_credit_unlock_idx: int | None = None
    rows: list[dict[str, object]] = []
    weights = pd.DataFrame(0.0, index=panel.index, columns=ASSETS)

    for i, row in panel.iterrows():
        current_locks = set(pending_locks)
        entry_signal = False
        exit_signal = False
        added_today: set[str] = set()
        unlocked_today: set[str] = set()
        days_since_unlock = None if last_credit_unlock_idx is None else i - last_credit_unlock_idx

        # states active today drive today's position
        if current_locks:
            weights.loc[i, "CASH"] = 1.0
        else:
            weights.loc[i, "SPY"] = 1.0

        # add locks for next day
        if include_vix and ("VIX" not in current_locks) and vix_entry_signal(row):
            pending_locks.add("VIX")
            added_today.add("VIX")
        if include_cmdty and ("CMDTY" not in current_locks) and cmdty_entry_signal(row):
            pending_locks.add("CMDTY")
            added_today.add("CMDTY")

        if include_credit and ("CREDIT" not in current_locks):
            if credit_entry_signal(row, cfg.entry_rule) or credit_relock_signal(row, cfg.relock_rule, days_since_unlock):
                pending_locks.add("CREDIT")
                added_today.add("CREDIT")

        # unlock today affects next day
        if "VIX" in current_locks and vix_unlock(row):
            pending_locks.discard("VIX")
            unlocked_today.add("VIX")
            if "CREDIT" in current_locks:
                pending_locks.discard("CREDIT")
                unlocked_today.add("CREDIT")
                last_credit_unlock_idx = i
                credit_state["neg_run"] = 0
                credit_state["cooldown_wait"] = None

        if "CMDTY" in current_locks and cmdty_unlock(row):
            pending_locks.discard("CMDTY")
            unlocked_today.add("CMDTY")

        if "CREDIT" in current_locks and "CREDIT" not in unlocked_today and credit_unlock_signal(row, cfg.unlock_rule, credit_state):
            pending_locks.discard("CREDIT")
            unlocked_today.add("CREDIT")
            last_credit_unlock_idx = i
            credit_state["neg_run"] = 0
            credit_state["cooldown_wait"] = None

        entry_signal = bool((not current_locks) and pending_locks)
        exit_signal = bool(current_locks and (not pending_locks))
        rows.append(
            {
                "date": row["date"],
                "stress_active": bool(current_locks),
                "active_locks": "+".join(sorted(current_locks)),
                "credit_active": "CREDIT" in current_locks,
                "vix_active": "VIX" in current_locks,
                "cmdty_active": "CMDTY" in current_locks,
                "entry_signal": entry_signal,
                "exit_signal": exit_signal,
                "locks_added_today": "+".join(sorted(added_today)),
                "locks_unlocked_today": "+".join(sorted(unlocked_today)),
            }
        )
    state = pd.DataFrame(rows)
    return weights, state


def build_strategy_frame(panel: pd.DataFrame, weights: pd.DataFrame, strategy: str, state: pd.DataFrame) -> pd.DataFrame:
    df = compute_strategy(panel, weights, strategy)
    out = pd.concat([panel[["date"]], weights.add_prefix("weight_"), df, state.drop(columns=["date"])], axis=1)
    return out


def count_reloc(ep_df: pd.DataFrame, variant: str) -> tuple[int, int]:
    if len(ep_df) == 0:
        return 0, 0
    return int(ep_df["relock_within_21d"].sum()), int(ep_df["relock_within_63d"].sum())


def summarize_episode_metrics(panel: pd.DataFrame, strategy_df: pd.DataFrame, variant_name: str) -> pd.DataFrame:
    ret21 = forward_return(panel["SPY_return"], 21)
    mdd21 = forward_mdd(panel["SPY_return"], 21)
    ret63 = forward_return(panel["SPY_return"], 63)
    mdd63 = forward_mdd(panel["SPY_return"], 63)
    episodes = []
    credit_mask = strategy_df["credit_active"].astype(bool)
    eps = find_episodes(credit_mask)
    for ep_id, (s, e) in enumerate(eps, start=1):
        sub = panel.loc[s:e]
        unlock_idx = min(e + 1, len(panel) - 1)
        trough_price = float(sub["spy_price"].min())
        unlock_price = float(panel.loc[unlock_idx, "spy_price"])
        trough_to_unlock = unlock_price / trough_price - 1.0 if trough_price > 0 else np.nan
        false_recovery = bool(
            (pd.notna(mdd21.iloc[unlock_idx]) and mdd21.iloc[unlock_idx] <= -0.05)
            or (pd.notna(mdd63.iloc[unlock_idx]) and mdd63.iloc[unlock_idx] <= -0.08)
            or strategy_df.loc[unlock_idx + 1 : min(unlock_idx + 63, len(strategy_df) - 1), "stress_active"].astype(bool).any()
        )
        missed_rebound = bool(pd.notna(trough_to_unlock) and trough_to_unlock > 0.08)
        relock21 = bool(strategy_df.loc[unlock_idx + 1 : min(unlock_idx + 21, len(strategy_df) - 1), "credit_active"].astype(bool).any())
        relock63 = bool(strategy_df.loc[unlock_idx + 1 : min(unlock_idx + 63, len(strategy_df) - 1), "credit_active"].astype(bool).any())
        episodes.append(
            {
                "credit_variant": variant_name,
                "episode_id": ep_id,
                "entry_date": panel.loc[s, "date"],
                "unlock_date": panel.loc[unlock_idx, "date"],
                "duration_days": int(e - s + 1),
                "entry_SPY_DD": panel.loc[s, "SPY_DD"],
                "entry_CREDIT_SPREAD": panel.loc[s, "CREDIT_SPREAD"],
                "entry_D_CREDIT_15D": panel.loc[s, "D_CREDIT_15D"],
                "entry_D_CREDIT_Z": panel.loc[s, "D_CREDIT_15D_Z_252D"],
                "entry_CREDIT_LEVEL_Z": panel.loc[s, "CREDIT_LEVEL_Z_252D"],
                "unlock_D_CREDIT_15D": panel.loc[unlock_idx, "D_CREDIT_15D"],
                "unlock_CREDIT_LEVEL_Z": panel.loc[unlock_idx, "CREDIT_LEVEL_Z_252D"],
                "unlock_SPY_vs_MA20": panel.loc[unlock_idx, "spy_price"] / panel.loc[unlock_idx, "SPY_MA20"] - 1.0 if pd.notna(panel.loc[unlock_idx, "SPY_MA20"]) else np.nan,
                "unlock_SPY_vs_MA50": panel.loc[unlock_idx, "spy_price"] / panel.loc[unlock_idx, "SPY_MA50"] - 1.0 if pd.notna(panel.loc[unlock_idx, "SPY_MA50"]) else np.nan,
                "SPY_return_during_lock": period_return(panel.loc[s:e, "SPY_return"]),
                "SPY_maxDD_during_lock": period_mdd(panel.loc[s:e, "SPY_return"]),
                "CASH_return_during_lock": period_return(panel.loc[s:e, "CASH_return"]),
                "next_21d_SPY_return_after_unlock": ret21.iloc[unlock_idx],
                "next_21d_SPY_maxDD_after_unlock": mdd21.iloc[unlock_idx],
                "next_63d_SPY_return_after_unlock": ret63.iloc[unlock_idx],
                "next_63d_SPY_maxDD_after_unlock": mdd63.iloc[unlock_idx],
                "false_recovery_flag": false_recovery,
                "missed_rebound_flag": missed_rebound,
                "relock_within_21d": relock21,
                "relock_within_63d": relock63,
            }
        )
    return pd.DataFrame(episodes)


def performance_row(strategy: str, credit_variant: str, df: pd.DataFrame, ep_df: pd.DataFrame) -> dict[str, object]:
    perf = perf_with_alias(df, strategy)
    return {
        "strategy": strategy,
        "credit_variant": credit_variant,
        **perf,
        "total_time_in_stress": int(df["stress_active"].sum()),
        "time_in_credit_lock": int(df["credit_active"].sum()),
        "time_in_vix_lock": int(df["vix_active"].sum()),
        "time_in_cmdty_lock": int(df["cmdty_active"].sum()),
        "number_credit_entries": int((df["locks_added_today"] == "CREDIT").sum() + df["locks_added_today"].astype(str).str.contains("CREDIT\\+|\\+CREDIT|CREDIT").sum() - int((df["locks_added_today"] == "CREDIT").sum())),
        "number_credit_unlocks": int(df["locks_unlocked_today"].astype(str).str.contains("CREDIT").sum()),
        "avg_credit_lock_duration": float(ep_df["duration_days"].mean()) if len(ep_df) else np.nan,
        "false_recovery_count": int(ep_df["false_recovery_flag"].sum()) if len(ep_df) else 0,
        "missed_rebound_count": int(ep_df["missed_rebound_flag"].sum()) if len(ep_df) else 0,
        "relock_count": int(ep_df["relock_within_21d"].sum()) if len(ep_df) else 0,
    }


def crisis_row(name: str, credit_variant: str, df: pd.DataFrame, ep_df: pd.DataFrame, window: str, start: str, end: str | None) -> dict[str, object]:
    mask = df["date"] >= pd.Timestamp(start)
    if end is not None:
        mask &= df["date"] <= pd.Timestamp(end)
    sub = df.loc[mask]
    ret = sub[f"{name}_return"]
    nav = (1.0 + ret.fillna(0.0)).cumprod()
    ann_vol = ret.std(ddof=1) * np.sqrt(252.0)
    ann_ret = nav.iloc[-1] ** (252.0 / len(sub)) - 1.0 if len(sub) else np.nan
    if len(ep_df) and "entry_date" in ep_df.columns:
        ep_mask = ep_df["entry_date"].between(pd.Timestamp(start), pd.Timestamp(end) if end is not None else df["date"].max())
        ep_sub = ep_df.loc[ep_mask]
    else:
        ep_sub = pd.DataFrame()
    return {
        "credit_variant": credit_variant,
        "window": window,
        "cumulative_return": float(nav.iloc[-1] - 1.0) if len(sub) else np.nan,
        "max_drawdown": float((nav / nav.cummax() - 1.0).min()) if len(sub) else np.nan,
        "Sharpe": float(ann_ret / ann_vol) if len(sub) and ann_vol > 0 else np.nan,
        "time_in_credit_lock": int(sub["credit_active"].sum()) if len(sub) else 0,
        "time_in_any_stress": int(sub["stress_active"].sum()) if len(sub) else 0,
        "number_credit_entries": int(sub["locks_added_today"].astype(str).str.contains("CREDIT").sum()) if len(sub) else 0,
        "number_credit_unlocks": int(sub["locks_unlocked_today"].astype(str).str.contains("CREDIT").sum()) if len(sub) else 0,
        "false_recovery_count": int(ep_sub["false_recovery_flag"].sum()) if len(ep_sub) else 0,
        "missed_rebound_count": int(ep_sub["missed_rebound_flag"].sum()) if len(ep_sub) else 0,
    }


def build_baselines(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    w = pd.DataFrame(0.0, index=panel.index, columns=ASSETS)
    w["SPY"] = 1.0
    state = pd.DataFrame(
        {
            "date": panel["date"],
            "stress_active": False,
            "active_locks": "",
            "credit_active": False,
            "vix_active": False,
            "cmdty_active": False,
            "entry_signal": False,
            "exit_signal": False,
            "locks_added_today": "",
            "locks_unlocked_today": "",
        }
    )
    out[SPY_BUY_HOLD] = build_strategy_frame(panel, w, SPY_BUY_HOLD, state)

    final_state = pd.DataFrame(
        {
            "date": panel["date"],
            "stress_active": panel["final_any_stress"],
            "active_locks": panel["trigger_lock_active_locks"].fillna(""),
            "credit_active": panel["final_credit_active"],
            "vix_active": panel["final_vix_active"],
            "cmdty_active": panel["final_cmdty_active"],
            "entry_signal": panel["trigger_lock_entry_signal"].fillna(False),
            "exit_signal": panel["trigger_lock_exit_signal"].fillna(False),
            "locks_added_today": panel["trigger_lock_locks_added_today"].fillna(""),
            "locks_unlocked_today": panel["trigger_lock_locks_unlocked_today"].fillna(""),
        }
    )
    w = pd.DataFrame(0.0, index=panel.index, columns=ASSETS)
    w["SPY"] = (~panel["final_any_stress"]).astype(float)
    w["CASH"] = panel["final_any_stress"].astype(float)
    out[SPY_CASH_FINAL_LOCKS] = build_strategy_frame(panel, w, SPY_CASH_FINAL_LOCKS, final_state)

    cfg_none = CreditVariant("NO_CREDIT", "ABS_15D", "BASE")
    w, s = simulate_variant(panel, cfg_none, include_vix=True, include_cmdty=True, include_credit=False)
    out[SPY_CASH_NO_CREDIT] = build_strategy_frame(panel, w, SPY_CASH_NO_CREDIT, s)

    w, s = simulate_variant(panel, cfg_none, include_vix=False, include_cmdty=False, include_credit=True)
    out[SPY_CASH_CREDIT_ONLY] = build_strategy_frame(panel, w, SPY_CASH_CREDIT_ONLY, s)
    return out


def build_variants(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = build_baselines(panel)
    for cfg in build_configs():
        w, s = simulate_variant(panel, cfg, include_vix=True, include_cmdty=True, include_credit=True)
        out[cfg.name] = build_strategy_frame(panel, w, cfg.name, s)
    return out


def credit_incremental_value(perf: pd.DataFrame) -> pd.DataFrame:
    idx = perf.set_index("strategy")
    rows = []
    pairs = [
        ("SPY_CASH_FINAL_LOCKS vs SPY_CASH_NO_CREDIT", SPY_CASH_FINAL_LOCKS, SPY_CASH_NO_CREDIT),
        ("SPY_CASH_CREDIT_ONLY vs SPY_BUY_HOLD", SPY_CASH_CREDIT_ONLY, SPY_BUY_HOLD),
    ]
    for v in VARIANTS:
        pairs.append((f"{v} vs SPY_CASH_NO_CREDIT", v, SPY_CASH_NO_CREDIT))
    for label, a, b in pairs:
        rows.append(
            {
                "comparison": label,
                "delta_CAGR": idx.loc[a, "CAGR"] - idx.loc[b, "CAGR"],
                "delta_Sharpe": idx.loc[a, "Sharpe"] - idx.loc[b, "Sharpe"],
                "delta_MaxDD": idx.loc[a, "MaxDD"] - idx.loc[b, "MaxDD"],
                "delta_Calmar": idx.loc[a, "Calmar"] - idx.loc[b, "Calmar"],
                "delta_Final_Equity": idx.loc[a, "Final Equity"] - idx.loc[b, "Final Equity"],
                "interpretation": "Positive delta means the credit trigger layer added value relative to the comparison baseline.",
            }
        )
    return pd.DataFrame(rows)


def plot_equity_curves(frames: dict[str, pd.DataFrame], best: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    for name in [SPY_BUY_HOLD, SPY_CASH_NO_CREDIT, SPY_CASH_FINAL_LOCKS, best]:
        ax.plot(frames[name]["date"], frames[name][f"{name}_nav"], label=name, linewidth=1.0)
    ax.set_yscale("log")
    ax.set_title("SPY/CASH Credit Variant Equity Curves")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "spy_cash_credit_variant_equity_curve.png", dpi=160)
    plt.close(fig)


def plot_drawdowns(frames: dict[str, pd.DataFrame], best: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    for name in [SPY_BUY_HOLD, SPY_CASH_NO_CREDIT, SPY_CASH_FINAL_LOCKS, best]:
        ax.plot(frames[name]["date"], frames[name][f"{name}_drawdown"], label=name, linewidth=1.0)
    ax.set_title("SPY/CASH Credit Variant Drawdowns")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "spy_cash_credit_variant_drawdown.png", dpi=160)
    plt.close(fig)


def plot_tradeoff(perf: pd.DataFrame) -> None:
    d = perf.loc[perf["strategy"].isin(VARIANTS)].copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(d["false_recovery_count"], d["missed_rebound_count"], s=(d["Sharpe"] * 120).clip(lower=20), c=d["MaxDD"], cmap="viridis_r")
    for _, row in d.iterrows():
        ax.text(row["false_recovery_count"], row["missed_rebound_count"], row["strategy"], fontsize=8)
    ax.set_xlabel("False recovery count")
    ax.set_ylabel("Missed rebound count")
    ax.set_title("Credit Variant Trade-off")
    fig.colorbar(sc, ax=ax, label="MaxDD")
    fig.tight_layout()
    fig.savefig(FIG / "credit_variant_tradeoff_scatter.png", dpi=160)
    plt.close(fig)


def plot_perf_bar(perf: pd.DataFrame) -> None:
    d = perf.loc[perf["strategy"].isin(VARIANTS)].sort_values("Sharpe", ascending=False)
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(d["strategy"], d["Sharpe"])
    ax.tick_params(axis="x", labelrotation=70)
    ax.set_title("Credit Variant Sharpe")
    fig.tight_layout()
    fig.savefig(FIG / "credit_variant_performance_bar.png", dpi=160)
    plt.close(fig)


def plot_case(panel: pd.DataFrame, frames: dict[str, pd.DataFrame], variants: list[str], name: str, start: str, end: str | None) -> None:
    mask = panel["date"] >= pd.Timestamp(start)
    if end is not None:
        mask &= panel["date"] <= pd.Timestamp(end)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(panel.loc[mask, "date"], panel.loc[mask, "spy_price"], color="black", label="SPY")
    axes[0].set_title(f"{name}: SPY Price")
    axes[1].plot(panel.loc[mask, "date"], panel.loc[mask, "CREDIT_SPREAD"], label="Credit spread", color="firebrick")
    axes[1].set_title("Credit Spread")
    for v in variants:
        axes[2].plot(frames[v].loc[mask, "date"], frames[v].loc[mask, f"{v}_nav"], label=v, linewidth=1.0)
    axes[2].legend(frameon=False, ncol=2)
    axes[2].set_title("Strategy NAV")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / f"case_{name}_credit_variant_comparison.png", dpi=160)
    plt.close(fig)


def plot_credit_lock_timeline(frames: dict[str, pd.DataFrame], best: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for name in [SPY_CASH_FINAL_LOCKS, best]:
        axes[0].plot(frames[name]["date"], frames[name]["credit_active"].astype(int), label=name)
        axes[1].plot(frames[name]["date"], frames[name]["stress_active"].astype(int), label=name)
    axes[0].set_title("Credit Lock Timeline")
    axes[1].set_title("Any Stress Timeline")
    for ax in axes:
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "credit_lock_timeline_by_variant.png", dpi=160)
    plt.close(fig)


def plot_incremental(inc: pd.DataFrame) -> None:
    d = inc.loc[inc["comparison"].str.contains("SPY_CASH_NO_CREDIT")].copy()
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(d["comparison"], d["delta_Sharpe"])
    ax.tick_params(axis="x", labelrotation=75)
    ax.set_title("Credit Incremental Value vs No-Credit")
    fig.tight_layout()
    fig.savefig(FIG / "credit_incremental_value_bar.png", dpi=160)
    plt.close(fig)


def build_report(best_sharpe: str, best_mdd: str, best_balanced: str, perf: pd.DataFrame, inc: pd.DataFrame) -> None:
    idx = perf.set_index("strategy")
    adds_value = bool(idx.loc[SPY_CASH_FINAL_LOCKS, "Sharpe"] > idx.loc[SPY_CASH_NO_CREDIT, "Sharpe"])
    migrate = best_balanced != SPY_CASH_FINAL_LOCKS and idx.loc[best_balanced, "Sharpe"] >= idx.loc[SPY_CASH_FINAL_LOCKS, "Sharpe"] and idx.loc[best_balanced, "MaxDD"] >= idx.loc[SPY_CASH_FINAL_LOCKS, "MaxDD"]
    lines = [
        "# SPY_CASH_CREDIT_TRIGGER_LAB_REPORT",
        "",
        "## 1. Purpose",
        "",
        "This lab isolates credit trigger timing inside a pure SPY/CASH framework so that hedge asset allocation does not contaminate the timing conclusion.",
        "",
        "## 2. Framework",
        "",
        "Normal = 100% SPY. Any stress lock = 100% CASH. VIX and commodity locks are kept from the current final strategy. Only credit entry/unlock/relock rules change.",
        "",
        "## 3. Baselines",
        "",
        "- `SPY_BUY_HOLD`",
        "- `SPY_CASH_NO_CREDIT`",
        "- `SPY_CASH_FINAL_LOCKS`",
        "- `SPY_CASH_CREDIT_ONLY`",
        "",
        "## 4. Credit Variants",
        "",
        "Variants include absolute 15D/20D spread changes, z-score entry, stricter unlock confirmation, MA50-based unlocks, cooldown unlocks, and fast relock rules.",
        "",
        "## 5. Main Results",
        "",
        f"- Best variant by Sharpe: `{best_sharpe}`",
        f"- Best variant by MaxDD: `{best_mdd}`",
        f"- Best balanced variant: `{best_balanced}`",
        "",
        "## 6. Does Credit Add Value in SPY/CASH?",
        "",
        f"- `SPY_CASH_FINAL_LOCKS` Sharpe: {idx.loc[SPY_CASH_FINAL_LOCKS, 'Sharpe']:.3f}",
        f"- `SPY_CASH_NO_CREDIT` Sharpe: {idx.loc[SPY_CASH_NO_CREDIT, 'Sharpe']:.3f}",
        f"- `SPY_CASH_CREDIT_ONLY` Sharpe: {idx.loc[SPY_CASH_CREDIT_ONLY, 'Sharpe']:.3f}",
        f"- Credit adds value vs no-credit: {'YES' if adds_value else 'NO'}",
        "",
        "## 7. Unlock / Relock Diagnostics",
        "",
        "Use the episode diagnostics to compare false recovery, missed rebound, and relock counts. This is the key trade-off surface.",
        "",
        "## 8. Crisis Windows",
        "",
        "Use the crisis comparison table and case-study plots for 2008, 2022, COVID, and 2025.",
        "",
        "## 9. Recommendation",
        "",
        ("A challenger should be migrated back into the final regime-hedge framework." if migrate else "No challenger is strong enough yet. Credit timing should not be over-optimized further inside the final strategy until it proves itself in this simpler SPY/CASH lab."),
        "",
        "## 10. Final Conclusion",
        "",
        "If a credit variant cannot improve robustly even in a SPY/CASH lab, it should not be promoted directly into the final regime-aware hedge strategy.",
    ]
    (OUT / "SPY_CASH_CREDIT_TRIGGER_LAB_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    patch = (
        "The SPY/CASH credit lab suggests that "
        + best_balanced
        + " improves credit timing before considering hedge allocation. It should be tested as a final strategy challenger."
        if migrate
        else "We isolated credit trigger timing in a SPY/CASH framework. Although several stricter unlock and z-score variants improved individual episodes, none produced a robust full-sample improvement over the simpler baseline credit lock. Therefore, the final strategy keeps the simpler credit logic."
    )
    (OUT / "README_PATCH_SUGGESTION.md").write_text(patch, encoding="utf-8")


def choose_best_balanced(perf: pd.DataFrame) -> str:
    d = perf.loc[perf["strategy"].isin(VARIANTS)].copy()
    d["score"] = (
        d["Sharpe"].rank(ascending=False, pct=True)
        + d["MaxDD"].rank(ascending=False, pct=True)
        + (-d["false_recovery_count"]).rank(ascending=False, pct=True)
        + (-d["missed_rebound_count"]).rank(ascending=False, pct=True)
    )
    return str(d.sort_values(["score", "Sharpe"], ascending=[False, False]).iloc[0]["strategy"])


def main() -> None:
    ensure_dirs()
    panel = prepare_panel(load_panel())
    save_feature_panel(panel)

    frames = build_variants(panel)

    episode_rows = []
    perf_rows = []
    crisis_rows = []
    for name, df in frames.items():
        variant_label = name if name in VARIANTS else name
        ep = summarize_episode_metrics(panel, df, variant_label)
        if len(ep):
            episode_rows.append(ep)
        perf_rows.append(performance_row(name, variant_label, df, ep))
        for window, (start, end) in WINDOWS.items():
            crisis_rows.append(crisis_row(name, variant_label, df, ep, window, start, end))

    episode_diag = pd.concat(episode_rows, ignore_index=True) if episode_rows else pd.DataFrame()
    perf = pd.DataFrame(perf_rows)
    crisis = pd.DataFrame(crisis_rows)
    inc = credit_incremental_value(perf)

    perf.to_csv(OUT / "spy_cash_credit_variant_performance.csv", index=False)
    episode_diag.to_csv(OUT / "credit_episode_diagnostics_by_variant.csv", index=False)
    crisis.to_csv(OUT / "spy_cash_credit_variant_crisis_comparison.csv", index=False)
    inc.to_csv(OUT / "credit_incremental_value.csv", index=False)

    best_sharpe = str(perf.loc[perf["strategy"].isin(VARIANTS)].sort_values("Sharpe", ascending=False).iloc[0]["strategy"])
    best_mdd = str(perf.loc[perf["strategy"].isin(VARIANTS)].sort_values("MaxDD", ascending=False).iloc[0]["strategy"])
    best_balanced = choose_best_balanced(perf)

    plot_equity_curves(frames, best_sharpe)
    plot_drawdowns(frames, best_sharpe)
    plot_tradeoff(perf)
    plot_perf_bar(perf)
    plot_case(panel, frames, [SPY_CASH_NO_CREDIT, SPY_CASH_FINAL_LOCKS, best_balanced], "2008", *WINDOWS["2008_GFC"])
    plot_case(panel, frames, [SPY_CASH_NO_CREDIT, SPY_CASH_FINAL_LOCKS, best_balanced], "2022", *WINDOWS["2022_RATE_WAR"])
    plot_case(panel, frames, [SPY_CASH_NO_CREDIT, SPY_CASH_FINAL_LOCKS, best_balanced], "COVID", *WINDOWS["COVID_2020"])
    plot_credit_lock_timeline(frames, best_balanced)
    plot_incremental(inc)

    build_report(best_sharpe, best_mdd, best_balanced, perf, inc)

    idx = perf.set_index("strategy")
    adds_value = bool(idx.loc[SPY_CASH_FINAL_LOCKS, "Sharpe"] > idx.loc[SPY_CASH_NO_CREDIT, "Sharpe"])
    migrate = bool(
        best_balanced != SPY_CASH_FINAL_LOCKS
        and idx.loc[best_balanced, "Sharpe"] >= idx.loc[SPY_CASH_FINAL_LOCKS, "Sharpe"]
        and idx.loc[best_balanced, "MaxDD"] >= idx.loc[SPY_CASH_FINAL_LOCKS, "MaxDD"]
    )

    print("SPY_BUY_HOLD performance")
    print(idx.loc[SPY_BUY_HOLD, ["CAGR", "Sharpe", "MaxDD", "Calmar", "Final Equity"]].to_string())
    print("SPY_CASH_NO_CREDIT performance")
    print(idx.loc[SPY_CASH_NO_CREDIT, ["CAGR", "Sharpe", "MaxDD", "Calmar", "Final Equity"]].to_string())
    print("SPY_CASH_FINAL_LOCKS performance")
    print(idx.loc[SPY_CASH_FINAL_LOCKS, ["CAGR", "Sharpe", "MaxDD", "Calmar", "Final Equity"]].to_string())
    print("SPY_CASH_CREDIT_ONLY performance")
    print(idx.loc[SPY_CASH_CREDIT_ONLY, ["CAGR", "Sharpe", "MaxDD", "Calmar", "Final Equity"]].to_string())
    print("best credit variant by Sharpe")
    print(best_sharpe)
    print("best credit variant by MaxDD")
    print(best_mdd)
    print("best balanced variant by Sharpe / MaxDD / false recovery / missed rebound")
    print(best_balanced)
    print("whether credit trigger adds value vs no-credit")
    print("YES" if adds_value else "NO")
    print("whether any variant should be migrated to final strategy")
    print("YES" if migrate else "NO")
    print("output paths")
    print(str(OUT))


if __name__ == "__main__":
    main()
