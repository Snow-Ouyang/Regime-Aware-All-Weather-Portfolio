from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from final_strategy_source_only_core import ASSETS, FINAL_STRATEGY, ROOT, build_final_source_only_panel


OUT = ROOT / "results" / "main_pipeline_final"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
TRIGGERS = [
    "VIX_FULL_RISK_TRIGGER",
    "CREDIT_FULL_RISK_TRIGGER",
]
FULL_RISK_TRIGGERS = [
    "VIX_FULL_RISK_TRIGGER",
    "CREDIT_FULL_RISK_TRIGGER",
]


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_panel() -> pd.DataFrame:
    panel, _ = build_final_source_only_panel()
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    return panel


def allocation_state(row: pd.Series) -> str:
    state = str(row.get("final_allocation_state", row.get("flat_refined_state", "OTHER")))
    valid = {
        "FLAT_LOW_RATE_NORMAL",
        "FLAT_MID_RATE_NORMAL",
        "FLAT_LOWMID_RATE_STRESS",
        "FLAT_HIGH_RATE_NORMAL",
        "FLAT_HIGH_RATE_STRESS",
        "STEEP_LOW_RATE_NORMAL",
        "STEEP_MID_RATE_NORMAL",
        "STEEP_MID_RATE_STRESS",
        "STEEP_HIGH_RATE_NORMAL",
        "STEEP_HIGH_RATE_STRESS",
        "INVERTED_NORMAL",
        "INVERTED_STRESS",
    }
    return state if state in valid else "OTHER"


def trigger_combination(row: pd.Series) -> str:
    parts = []
    if bool(row.get("trigger_vix", False) or row.get("VIX_FULL_RISK_TRIGGER", False)):
        parts.append("VIX")
    if bool(row.get("trigger_credit_drawdown", False) or row.get("EFFECTIVE_CREDIT_DRAWDOWN_TRIGGER", False)):
        parts.append("CREDIT")
    return "+".join(parts) if parts else "UNKNOWN"


def weight_string(row: pd.Series) -> str:
    vals = []
    for asset in ASSETS:
        value = float(row.get(f"{FINAL_STRATEGY}_weight_{asset}", row.get(asset, 0.0)))
        if abs(value) > 1e-6:
            vals.append(f"{asset}:{value:.2%}")
    return "; ".join(vals) if vals else "NONE"


def forward_return(ret: pd.Series, window: int) -> pd.Series:
    return (1.0 + ret.fillna(0.0)).rolling(window).apply(np.prod, raw=True).shift(-window) - 1.0


def forward_mdd(ret: pd.Series, window: int) -> pd.Series:
    vals = []
    arr = ret.fillna(0.0).to_numpy()
    for i in range(len(arr)):
        sub = arr[i : min(len(arr), i + window)]
        if len(sub) == 0:
            vals.append(np.nan)
        else:
            nav = np.cumprod(1.0 + sub)
            vals.append(float((nav / np.maximum.accumulate(nav) - 1.0).min()))
    return pd.Series(vals, index=ret.index)


def add_trigger_flags(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["base_regime"] = out["macro_regime_confirmed"].where(out["macro_regime_confirmed"].isin(["FLAT", "STEEP", "INVERTED"]), "OTHER")
    out["refined_regime"] = out["refined_regime_confirmed"].where(
        out["refined_regime_confirmed"].isin(
            ["FLAT_LOW_RATE", "FLAT_MID_RATE", "FLAT_HIGH_RATE", "STEEP_LOW_RATE", "STEEP_MID_RATE", "STEEP_HIGH_RATE", "INVERTED"]
        ),
        "OTHER",
    )
    out["final_regime"] = out.get("final_regime_confirmed", out["refined_regime"]).where(
        out.get("final_regime_confirmed", out["refined_regime"]).isin(
            ["FLAT_LOW_RATE", "FLAT_MID_RATE", "FLAT_HIGH_RATE", "STEEP_LOW_RATE", "STEEP_MID_RATE", "STEEP_HIGH_RATE", "INVERTED"]
        ),
        "OTHER",
    )
    out["allocation_state"] = out.apply(allocation_state, axis=1)
    out["VIX_FULL_RISK_TRIGGER"] = out["final_regime"].isin(
        ["FLAT_LOW_RATE", "FLAT_MID_RATE", "FLAT_HIGH_RATE", "INVERTED"]
    ) & (
        out["VIX_ZSCORE_120D"] >= 3.0
    )
    out["RAW_CREDIT_DRAWDOWN_TRIGGER"] = (out["D_CREDIT_SPREAD_15D"] > 0.10) & (~out["SPY_above_MA20"])
    out["EFFECTIVE_CREDIT_DRAWDOWN_TRIGGER"] = out["RAW_CREDIT_DRAWDOWN_TRIGGER"] & out["final_regime"].isin(
        ["FLAT_LOW_RATE", "FLAT_MID_RATE", "FLAT_HIGH_RATE", "STEEP_MID_RATE", "STEEP_HIGH_RATE", "INVERTED"]
    )
    out["CREDIT_FULL_RISK_TRIGGER"] = out["EFFECTIVE_CREDIT_DRAWDOWN_TRIGGER"]
    out["MONTHLY_SELL_FULL_RISK_TRIGGER"] = False
    out["CMDTY_FULL_RISK_TRIGGER"] = False
    out["STEEP_SLOW_GROWTH_OVERLAY_TRIGGER"] = False
    out["FULL_RISK_TRIGGER_ANY"] = out["VIX_FULL_RISK_TRIGGER"] | out["CREDIT_FULL_RISK_TRIGGER"]
    out["FULL_RISK_ACTIVE"] = out["trigger_lock_full_risk_state"].eq("FULL_RISK")
    out["SLOW_GROWTH_OVERLAY_ACTIVE"] = False
    out["RECOVERY_ACTIVE"] = False
    weights = out[[f"{FINAL_STRATEGY}_weight_{asset}" for asset in ASSETS]]
    out["turnover"] = 0.5 * weights.diff().abs().sum(axis=1)
    out.loc[out.index[0], "turnover"] = 0.5 * weights.iloc[0].abs().sum()
    out["SPY_return_5d"] = out["SPY_return"].rolling(5).sum()
    out["SPY_return_20d"] = out["SPY_return"].rolling(20).sum()
    out["strategy_return_next_5d"] = forward_return(out[f"{FINAL_STRATEGY}_return"], 5)
    out["strategy_return_next_20d"] = forward_return(out[f"{FINAL_STRATEGY}_return"], 20)
    out["SPY_return_next_5d"] = forward_return(out["SPY_return"], 5)
    out["SPY_return_next_20d"] = forward_return(out["SPY_return"], 20)
    out["max_drawdown_next_20d"] = forward_mdd(out[f"{FINAL_STRATEGY}_return"], 20)
    out["SPY_max_drawdown_next_20d"] = forward_mdd(out["SPY_return"], 20)
    return out


def trigger_frequency_by_regime(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, sub in panel.groupby("final_regime", dropna=False):
        row = {"final_regime": regime, "total_days": len(sub)}
        for col in [
            "VIX_FULL_RISK_TRIGGER",
            "RAW_CREDIT_DRAWDOWN_TRIGGER",
            "EFFECTIVE_CREDIT_DRAWDOWN_TRIGGER",
            "MONTHLY_SELL_FULL_RISK_TRIGGER",
            "STEEP_SLOW_GROWTH_OVERLAY_TRIGGER",
            "FULL_RISK_TRIGGER_ANY",
            "FULL_RISK_ACTIVE",
            "SLOW_GROWTH_OVERLAY_ACTIVE",
            "RECOVERY_ACTIVE",
        ]:
            count = int(sub[col].sum())
            row[f"{col}_count"] = count
            row[f"{col}_rate"] = count / len(sub) if len(sub) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def trigger_overlap_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(panel)
    for a in TRIGGERS:
        for b in TRIGGERS:
            ac = int(panel[a].sum())
            bc = int(panel[b].sum())
            ov = int((panel[a] & panel[b]).sum())
            rows.append(
                {
                    "trigger_A": a,
                    "trigger_B": b,
                    "overlap_count": ov,
                    "overlap_rate_over_all_days": ov / total,
                    "A_count": ac,
                    "B_count": bc,
                    "conditional_prob_A_given_B": ov / bc if bc else np.nan,
                    "conditional_prob_B_given_A": ov / ac if ac else np.nan,
                }
            )
    return pd.DataFrame(rows)


def multi_trigger_by_regime(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, sub in panel.groupby("final_regime", dropna=False):
        count = sub[FULL_RISK_TRIGGERS].sum(axis=1)
        combos = sub[FULL_RISK_TRIGGERS].apply(lambda r: "+".join([c.replace("_FULL_RISK_TRIGGER", "").replace("EFFECTIVE_", "") for c, v in r.items() if v]) or "NONE", axis=1)
        common = combos.value_counts()
        row = {
            "final_regime": regime,
            "total_days": len(sub),
            "days_with_0_full_risk_trigger": int((count == 0).sum()),
            "days_with_1_full_risk_trigger": int((count == 1).sum()),
            "days_with_2_full_risk_triggers": int((count == 2).sum()),
            "days_with_3_full_risk_triggers": int((count == 3).sum()),
            "multi_full_risk_trigger_rate": float((count >= 2).mean()),
            "most_common_trigger_combination": common.index[0] if len(common) else "NONE",
            "most_common_trigger_combination_count": int(common.iloc[0]) if len(common) else 0,
            "days_with_overlay_and_full_risk_trigger": 0,
            "overlay_full_risk_overlap_rate": 0.0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def stress_entries_exits(panel: pd.DataFrame) -> tuple[list[int], list[int]]:
    active = panel["FULL_RISK_ACTIVE"].astype(bool)
    entries = panel.index[active & ~active.shift(1, fill_value=False)].tolist()
    exits = panel.index[~active & active.shift(1, fill_value=False)].tolist()
    return entries, exits


def period_return(sub: pd.DataFrame, col: str) -> float:
    if sub.empty:
        return np.nan
    return float((1.0 + sub[col].fillna(0.0)).prod() - 1.0)


def period_mdd(sub: pd.DataFrame, col: str) -> float:
    if sub.empty:
        return np.nan
    nav = (1.0 + sub[col].fillna(0.0)).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def build_entry_attribution(panel: pd.DataFrame) -> pd.DataFrame:
    entries, exits = stress_entries_exits(panel)
    rows = []
    for n, entry in enumerate(entries, start=1):
        next_exit = next((x for x in exits if x > entry), None)
        signal_idx = max(entry - 1, 0)
        sig = panel.loc[signal_idx]
        cause_vix = bool(sig["VIX_FULL_RISK_TRIGGER"])
        cause_credit = bool(sig["CREDIT_FULL_RISK_TRIGGER"])
        cause_cmdty = bool(sig["CMDTY_FULL_RISK_TRIGGER"])
        cause_monthly = False
        sub = panel.loc[entry : next_exit - 1 if next_exit is not None else panel.index[-1]]
        exit_date = panel.loc[next_exit, "date"] if next_exit is not None else pd.NaT
        next_entry_after_exit = next((e for e in entries if next_exit is not None and e > next_exit), None)
        combo = trigger_combination(
            pd.Series(
                {
                    "trigger_vix": cause_vix,
                    "trigger_credit_drawdown": cause_credit,
                    "trigger_cmdty": cause_cmdty,
                    "trigger_monthly_sell": cause_monthly,
                }
            )
        )
        rows.append(
            {
                "entry_id": n,
                "entry_date": panel.loc[entry, "date"],
                "final_regime_at_entry": panel.loc[entry, "final_regime"],
                "allocation_state_at_entry": panel.loc[entry, "allocation_state"],
                "base_regime_at_entry": panel.loc[entry, "base_regime"],
                "trigger_vix": cause_vix,
                "trigger_credit_drawdown": cause_credit,
                "trigger_monthly_sell": cause_monthly,
                "trigger_cmdty": cause_cmdty,
                "trigger_combination": combo,
                "raw_credit_drawdown_trigger": bool(sig["RAW_CREDIT_DRAWDOWN_TRIGGER"]),
                "effective_credit_drawdown_trigger": bool(sig["EFFECTIVE_CREDIT_DRAWDOWN_TRIGGER"]),
                "slow_growth_overlay_trigger_at_entry": bool(sig["STEEP_SLOW_GROWTH_OVERLAY_TRIGGER"]),
                "previous_final_regime": panel.loc[entry - 1, "final_regime"] if entry > 0 else None,
                "previous_allocation_state": panel.loc[entry - 1, "allocation_state"] if entry > 0 else None,
                "previous_full_risk_active": bool(panel.loc[entry - 1, "FULL_RISK_ACTIVE"]) if entry > 0 else False,
                "previous_recovery_active": bool(panel.loc[entry - 1, "RECOVERY_ACTIVE"]) if entry > 0 else False,
                "next_exit_date": exit_date,
                "stress_duration_days": len(sub),
                "strategy_return_during_stress": period_return(sub, f"{FINAL_STRATEGY}_return"),
                "SPY_return_during_stress": period_return(sub, "SPY_return"),
                "max_drawdown_during_stress": period_mdd(sub, f"{FINAL_STRATEGY}_return"),
                "SPY_max_drawdown_during_stress": period_mdd(sub, "SPY_return"),
                "turnover_on_entry_date": float(panel.loc[entry, "turnover"]),
                "turnover_on_exit_date": float(panel.loc[next_exit, "turnover"]) if next_exit is not None else np.nan,
                "reentered_within_5d_after_exit": bool(next_entry_after_exit is not None and next_exit is not None and next_entry_after_exit - next_exit <= 5),
                "reentered_within_10d_after_exit": bool(next_entry_after_exit is not None and next_exit is not None and next_entry_after_exit - next_exit <= 10),
                "reentered_within_20d_after_exit": bool(next_entry_after_exit is not None and next_exit is not None and next_entry_after_exit - next_exit <= 20),
            }
        )
    return pd.DataFrame(rows)


def build_exit_reentry(panel: pd.DataFrame, entries_df: pd.DataFrame) -> pd.DataFrame:
    entries, exits = stress_entries_exits(panel)
    entry_lookup = entries_df.set_index("entry_date")["trigger_combination"].to_dict() if not entries_df.empty else {}
    rows = []
    for n, exit_idx in enumerate(exits, start=1):
        signal_idx = max(exit_idx - 1, 0)
        next_entry = next((e for e in entries if e > exit_idx), None)
        next_entry_date = panel.loc[next_entry, "date"] if next_entry is not None else pd.NaT
        combo = entry_lookup.get(next_entry_date, "NONE")
        rows.append(
            {
                "exit_id": n,
                "exit_date": panel.loc[exit_idx, "date"],
                "final_regime_at_exit": panel.loc[exit_idx, "final_regime"],
                "allocation_state_at_exit": panel.loc[exit_idx, "allocation_state"],
                "base_regime_at_exit": panel.loc[exit_idx, "base_regime"],
                "exit_reason": "TRIGGER_LOCKS_UNLOCKED",
                "R3_RECOVERY_flag": bool(panel.loc[signal_idx, "trigger_lock_exit_signal"]),
                "SPY_cross_above_MA20_flag": bool(panel.loc[signal_idx, "SPY_CROSS_ABOVE_MA20"]),
                "days_until_next_full_risk_entry": int(next_entry - exit_idx) if next_entry is not None else np.nan,
                "reentered_within_5d": bool(next_entry is not None and next_entry - exit_idx <= 5),
                "reentered_within_10d": bool(next_entry is not None and next_entry - exit_idx <= 10),
                "reentered_within_20d": bool(next_entry is not None and next_entry - exit_idx <= 20),
                "next_entry_date": next_entry_date,
                "next_entry_final_regime": panel.loc[next_entry, "final_regime"] if next_entry is not None else None,
                "next_entry_allocation_state": panel.loc[next_entry, "allocation_state"] if next_entry is not None else None,
                "next_entry_trigger_combination": combo,
                "recovery_active_after_exit": bool(panel.loc[exit_idx, "RECOVERY_ACTIVE"]),
                "flat_low_recovery_enabled_after_exit": bool(panel.loc[exit_idx, "allocation_state"] == "FLAT_LOW_RATE_RECOVERY"),
                "strategy_return_next_5d": panel.loc[exit_idx, "strategy_return_next_5d"],
                "strategy_return_next_20d": panel.loc[exit_idx, "strategy_return_next_20d"],
                "SPY_return_next_5d": panel.loc[exit_idx, "SPY_return_next_5d"],
                "SPY_return_next_20d": panel.loc[exit_idx, "SPY_return_next_20d"],
                "max_drawdown_next_20d": panel.loc[exit_idx, "max_drawdown_next_20d"],
                "SPY_max_drawdown_next_20d": panel.loc[exit_idx, "SPY_max_drawdown_next_20d"],
                "turnover_on_exit_date": panel.loc[exit_idx, "turnover"],
                "turnover_on_next_entry_date": panel.loc[next_entry, "turnover"] if next_entry is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)


def stress_reentry_by_regime(exit_df: pd.DataFrame) -> pd.DataFrame:
    if exit_df.empty:
        return pd.DataFrame()
    rows = []
    for regime, sub in exit_df.groupby("final_regime_at_exit", dropna=False):
        common = sub["next_entry_trigger_combination"].dropna().value_counts()
        rows.append(
            {
                "final_regime_at_exit": regime,
                "exit_count": len(sub),
                "reentry_5d_count": int(sub["reentered_within_5d"].sum()),
                "reentry_5d_rate": float(sub["reentered_within_5d"].mean()),
                "reentry_10d_count": int(sub["reentered_within_10d"].sum()),
                "reentry_10d_rate": float(sub["reentered_within_10d"].mean()),
                "reentry_20d_count": int(sub["reentered_within_20d"].sum()),
                "reentry_20d_rate": float(sub["reentered_within_20d"].mean()),
                "mean_days_until_next_entry": float(sub["days_until_next_full_risk_entry"].mean()),
                "most_common_next_entry_trigger": common.index[0] if len(common) else "NONE",
                "mean_strategy_return_next_20d": float(sub["strategy_return_next_20d"].mean()),
                "mean_SPY_return_next_20d": float(sub["SPY_return_next_20d"].mean()),
            }
        )
    return pd.DataFrame(rows)


def event_reason(row: pd.Series) -> str:
    if row["is_full_risk_entry"]:
        combo = row["entry_trigger_combination"]
        if "+" in combo:
            return "FULL_RISK_ENTRY_MULTI_TRIGGER"
        if combo == "VIX":
            return "FULL_RISK_ENTRY_VIX"
        if combo == "CREDIT":
            return "FULL_RISK_ENTRY_CREDIT"
        if combo == "CMDTY":
            return "FULL_RISK_ENTRY_CMDTY"
        return "FULL_RISK_ENTRY_UNKNOWN"
    if row["is_full_risk_exit"]:
        return "FULL_RISK_EXIT_TRIGGER_UNLOCK"
    if row["is_flat_low_high_switch"]:
        return "FLAT_LOW_HIGH_SWITCH"
    if row["turnover"] > 1e-8 and row["previous_allocation_state"] == row["allocation_state"] and str(row["allocation_state"]).endswith("_NORMAL"):
        return "INVERSE_VOL_REBALANCE"
    return "OTHER"


def add_event_reasons(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["previous_allocation_state"] = out["allocation_state"].shift(1)
    out["previous_refined_regime"] = out["final_regime"].shift(1)
    out["is_full_risk_entry"] = out["FULL_RISK_ACTIVE"] & ~out["FULL_RISK_ACTIVE"].shift(1, fill_value=False)
    out["is_full_risk_exit"] = ~out["FULL_RISK_ACTIVE"] & out["FULL_RISK_ACTIVE"].shift(1, fill_value=False)
    out["is_recovery_entry"] = out["RECOVERY_ACTIVE"] & ~out["RECOVERY_ACTIVE"].shift(1, fill_value=False)
    out["is_recovery_exit"] = ~out["RECOVERY_ACTIVE"] & out["RECOVERY_ACTIVE"].shift(1, fill_value=False)
    out["is_steep_overlay_entry"] = out["SLOW_GROWTH_OVERLAY_ACTIVE"] & ~out["SLOW_GROWTH_OVERLAY_ACTIVE"].shift(1, fill_value=False)
    out["is_steep_overlay_exit"] = ~out["SLOW_GROWTH_OVERLAY_ACTIVE"] & out["SLOW_GROWTH_OVERLAY_ACTIVE"].shift(1, fill_value=False)
    out["is_flat_low_high_switch"] = (
        out["previous_refined_regime"].isin(["FLAT_LOW_RATE", "FLAT_HIGH_RATE"])
        & out["final_regime"].isin(["FLAT_LOW_RATE", "FLAT_HIGH_RATE"])
        & out["previous_refined_regime"].ne(out["final_regime"])
    )
    combos = []
    for i in out.index:
        sig_i = max(i - 1, 0)
        sig = out.loc[sig_i]
        combos.append(
            trigger_combination(
                pd.Series(
                    {
                        "trigger_vix": bool(sig["VIX_FULL_RISK_TRIGGER"]),
                        "trigger_credit_drawdown": sig["CREDIT_FULL_RISK_TRIGGER"],
                        "trigger_cmdty": sig["CMDTY_FULL_RISK_TRIGGER"],
                        "trigger_monthly_sell": False,
                    }
                )
            )
        )
    out["entry_trigger_combination"] = combos
    out["event_reason"] = out.apply(event_reason, axis=1)
    return out


def turnover_by_event(panel: pd.DataFrame) -> pd.DataFrame:
    total = panel["turnover"].sum()
    out = (
        panel.groupby("event_reason", dropna=False)
        .agg(
            count=("turnover", "size"),
            total_turnover=("turnover", "sum"),
            average_turnover=("turnover", "mean"),
            median_turnover=("turnover", "median"),
            average_strategy_return_next_5d=("strategy_return_next_5d", "mean"),
            average_strategy_return_next_20d=("strategy_return_next_20d", "mean"),
            average_SPY_return_next_5d=("SPY_return_next_5d", "mean"),
            average_SPY_return_next_20d=("SPY_return_next_20d", "mean"),
        )
        .reset_index()
    )
    out["share_of_total_turnover"] = out["total_turnover"] / total if total else 0.0
    return out.sort_values("total_turnover", ascending=False)


def transition_turnover(panel: pd.DataFrame, steep_only: bool = False) -> pd.DataFrame:
    df = panel.copy()
    df["from_allocation_state"] = df["allocation_state"].shift(1).fillna("START")
    df["to_allocation_state"] = df["allocation_state"]
    if steep_only:
        df = df.loc[
            df["from_allocation_state"].astype(str).str.startswith("STEEP")
            & df["to_allocation_state"].astype(str).str.startswith("STEEP")
            & df["refined_regime"].eq("STEEP")
        ].copy()
        denom = df["turnover"].sum()
    else:
        denom = df["turnover"].sum()
    out = (
        df.groupby(["from_allocation_state", "to_allocation_state"], dropna=False)
        .agg(
            transition_count=("turnover", "size"),
            total_turnover=("turnover", "sum"),
            average_turnover=("turnover", "mean"),
            median_turnover=("turnover", "median"),
            mean_strategy_return_next_5d=("strategy_return_next_5d", "mean"),
            mean_strategy_return_next_20d=("strategy_return_next_20d", "mean"),
            mean_SPY_return_next_5d=("SPY_return_next_5d", "mean"),
            mean_SPY_return_next_20d=("SPY_return_next_20d", "mean"),
        )
        .reset_index()
        .sort_values("total_turnover", ascending=False)
    )
    share_col = "share_of_steep_turnover" if steep_only else "share_of_total_turnover"
    out[share_col] = out["total_turnover"] / denom if denom else 0.0
    return out


def trigger_effectiveness(entries: pd.DataFrame) -> pd.DataFrame:
    if entries.empty:
        return pd.DataFrame()
    out = (
        entries.groupby(["final_regime_at_entry", "trigger_combination"], dropna=False)
        .agg(
            entry_count=("entry_id", "size"),
            avg_stress_duration_days=("stress_duration_days", "mean"),
            median_stress_duration_days=("stress_duration_days", "median"),
            mean_strategy_return_during_stress=("strategy_return_during_stress", "mean"),
            mean_SPY_return_during_stress=("SPY_return_during_stress", "mean"),
            mean_max_drawdown_during_stress=("max_drawdown_during_stress", "mean"),
            mean_SPY_max_drawdown_during_stress=("SPY_max_drawdown_during_stress", "mean"),
            reentry_rate_5d_after_exit=("reentered_within_5d_after_exit", "mean"),
            reentry_rate_10d_after_exit=("reentered_within_10d_after_exit", "mean"),
            reentry_rate_20d_after_exit=("reentered_within_20d_after_exit", "mean"),
            avg_turnover_entry=("turnover_on_entry_date", "mean"),
            avg_turnover_exit=("turnover_on_exit_date", "mean"),
        )
        .reset_index()
    )
    out["mean_strategy_minus_SPY_during_stress"] = out["mean_strategy_return_during_stress"] - out["mean_SPY_return_during_stress"]
    out["mean_drawdown_reduction_vs_SPY"] = out["mean_max_drawdown_during_stress"] - out["mean_SPY_max_drawdown_during_stress"]
    out["avg_turnover_entry_exit"] = out["avg_turnover_entry"] + out["avg_turnover_exit"]
    return out


def top_turnover_dates(panel: pd.DataFrame) -> pd.DataFrame:
    prev_weight_cols = {asset: panel[f"{FINAL_STRATEGY}_weight_{asset}"].shift(1).fillna(0.0) for asset in ASSETS}
    rows = []
    for _, row in panel.nlargest(50, "turnover").iterrows():
        prev = pd.Series({f"{FINAL_STRATEGY}_weight_{asset}": prev_weight_cols[asset].loc[row.name] for asset in ASSETS})
        rows.append(
            {
                "date": row["date"],
                "turnover": row["turnover"],
                "event_reason": row["event_reason"],
                "previous_refined_regime": row["previous_refined_regime"],
                "current_refined_regime": row["final_regime"],
                "previous_allocation_state": row["previous_allocation_state"],
                "current_allocation_state": row["allocation_state"],
                "previous_weights_string": weight_string(prev),
                "current_weights_string": weight_string(row),
                "VIX_ZSCORE_120D": row["VIX_ZSCORE_120D"],
                "D_CREDIT_SPREAD_20D": row["D_CREDIT_SPREAD_20D"],
                "CMDTY_RET60": row["CMDTY_RET60"],
                "GS10": row["GS10"],
                "term_spread": row["TERM_SPREAD_10Y_1Y"],
                "monthly_either_state": row["monthly_either_state"],
                "SPY_return_1d": row["SPY_return"],
                "SPY_return_5d": row["SPY_return_5d"],
                "SPY_return_20d": row["SPY_return_20d"],
                "likely_reason": row["event_reason"],
            }
        )
    return pd.DataFrame(rows)


def diagnostics_summary(panel: pd.DataFrame, freq: pd.DataFrame, multi: pd.DataFrame, entries: pd.DataFrame, exits: pd.DataFrame, turnover_event: pd.DataFrame, alloc_trans: pd.DataFrame, steep_trans: pd.DataFrame, effectiveness: pd.DataFrame) -> pd.DataFrame:
    top_trigger = entries["trigger_combination"].value_counts().index[0] if not entries.empty else "NONE"
    top_trigger_regime = entries.groupby("final_regime_at_entry").size().sort_values(ascending=False).index[0] if not entries.empty else "NONE"
    highest_trigger_regime = freq.sort_values("FULL_RISK_TRIGGER_ANY_rate", ascending=False)["final_regime"].iloc[0] if not freq.empty else "NONE"
    highest_multi_regime = multi.sort_values("multi_full_risk_trigger_rate", ascending=False)["final_regime"].iloc[0] if not multi.empty else "NONE"
    reentry = stress_reentry_by_regime(exits)
    highest_reentry = reentry.sort_values("reentry_20d_rate", ascending=False)["final_regime_at_exit"].iloc[0] if not reentry.empty else "NONE"
    best_dd = effectiveness.sort_values("mean_drawdown_reduction_vs_SPY", ascending=False).head(1)
    worst_reentry = effectiveness.sort_values("reentry_rate_20d_after_exit", ascending=False).head(1)
    notes = [
        "Turnover is dominated by trigger-lock FULL_RISK entry/exit rather than inverse-vol rebalance.",
        "SPY_CASH_TIMING and final hedge allocation now share the same trigger-lock stress state.",
        "FLAT_LOW/HIGH switching contributes little turnover relative to stress state changes.",
        "Recovery overlay is not part of the final mainline.",
    ]
    return pd.DataFrame(
        [
            {
                "total_days": len(panel),
                "total_full_risk_entries": len(entries),
                "total_full_risk_exits": len(exits),
                "total_recovery_entries": int(panel["is_recovery_entry"].sum()),
                "total_slow_growth_overlay_entries": int(panel["is_steep_overlay_entry"].sum()),
                "most_frequent_trigger": top_trigger,
                "most_frequent_trigger_regime": top_trigger_regime,
                "regime_with_highest_full_risk_trigger_rate": highest_trigger_regime,
                "regime_with_highest_multi_trigger_rate": highest_multi_regime,
                "regime_with_highest_reentry_rate_20d": highest_reentry,
                "top_turnover_event_reason": turnover_event.iloc[0]["event_reason"] if not turnover_event.empty else "NONE",
                "top_turnover_allocation_transition": f"{alloc_trans.iloc[0]['from_allocation_state']} -> {alloc_trans.iloc[0]['to_allocation_state']}" if not alloc_trans.empty else "NONE",
                "top_steep_internal_transition": f"{steep_trans.iloc[0]['from_allocation_state']} -> {steep_trans.iloc[0]['to_allocation_state']}" if not steep_trans.empty else "NONE",
                "trigger_with_best_drawdown_reduction": f"{best_dd.iloc[0]['final_regime_at_entry']}:{best_dd.iloc[0]['trigger_combination']}" if not best_dd.empty else "NONE",
                "trigger_with_worst_reentry_rate": f"{worst_reentry.iloc[0]['final_regime_at_entry']}:{worst_reentry.iloc[0]['trigger_combination']}" if not worst_reentry.empty else "NONE",
                "notes": " ".join(notes),
            }
        ]
    )


def plot_outputs(freq: pd.DataFrame, overlap: pd.DataFrame, multi: pd.DataFrame, entries: pd.DataFrame, exits_by_regime: pd.DataFrame, steep_trans: pd.DataFrame, turnover_event: pd.DataFrame, alloc_trans: pd.DataFrame, effectiveness: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    cols = ["FULL_RISK_TRIGGER_ANY_rate", "FULL_RISK_ACTIVE_rate", "SLOW_GROWTH_OVERLAY_ACTIVE_rate", "RECOVERY_ACTIVE_rate"]
    freq.set_index("final_regime")[cols].plot(kind="bar", ax=ax)
    ax.set_title("Trigger Frequency by Regime")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "trigger_frequency_by_regime.png", dpi=160)
    plt.close(fig)

    heat = overlap.pivot(index="trigger_A", columns="trigger_B", values="conditional_prob_A_given_B")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(heat, annot=True, fmt=".1%", cmap="Blues", ax=ax)
    ax.set_title("Trigger Overlap: P(A | B)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "trigger_overlap_heatmap.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(multi["final_regime"], multi["multi_full_risk_trigger_rate"])
    ax.set_title("Multi Full-Risk Trigger Rate by Regime")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "multi_trigger_rate_by_regime.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    if not entries.empty:
        for combo, sub in entries.groupby("trigger_combination"):
            ax.scatter(sub["entry_date"], [combo] * len(sub), label=combo, s=18)
    ax.set_title("Stress Entry Timeline by Trigger")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stress_entry_timeline_by_trigger.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    if not exits_by_regime.empty:
        ax.bar(exits_by_regime["final_regime_at_exit"], exits_by_regime["reentry_20d_rate"])
    ax.set_title("20D Full-Risk Re-entry Rate by Exit Regime")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stress_reentry_rate_by_regime.png", dpi=160)
    plt.close(fig)

    for name, col, path in [
        ("STEEP Internal Transition Turnover", "total_turnover", "steep_internal_transition_turnover.png"),
        ("STEEP Internal Transition Count", "transition_count", "steep_internal_transition_count.png"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 5))
        d = steep_trans.head(12).copy()
        labels = d["from_allocation_state"] + " -> " + d["to_allocation_state"]
        ax.bar(labels, d[col])
        ax.tick_params(axis="x", labelrotation=55)
        ax.set_title(name)
        fig.tight_layout()
        fig.savefig(FIG_DIR / path, dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    d = turnover_event.head(12).copy()
    ax.bar(d["event_reason"], d["total_turnover"])
    ax.tick_params(axis="x", labelrotation=50)
    ax.set_title("Turnover by Trigger Event")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "turnover_by_trigger_event.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    d = alloc_trans.head(12).copy()
    labels = d["from_allocation_state"] + " -> " + d["to_allocation_state"]
    ax.bar(labels, d["total_turnover"])
    ax.tick_params(axis="x", labelrotation=55)
    ax.set_title("Turnover by Allocation-State Transition")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "turnover_by_allocation_state_transition.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    if not effectiveness.empty:
        d = effectiveness.copy()
        labels = d["final_regime_at_entry"].astype(str) + ":" + d["trigger_combination"].astype(str)
        ax.bar(labels, d["mean_drawdown_reduction_vs_SPY"])
        ax.tick_params(axis="x", labelrotation=55)
    ax.set_title("Trigger Effectiveness by Regime: Drawdown Reduction vs SPY")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "trigger_effectiveness_by_regime.png", dpi=160)
    plt.close(fig)


def plot_long_trigger_timeline(panel: pd.DataFrame) -> None:
    regime_colors = {
        "FLAT_LOW_RATE": "#8bc34a",
        "FLAT_HIGH_RATE": "#f4b400",
        "STEEP_LOW_RATE": "#4fc3f7",
        "STEEP_HIGH_RATE": "#1976d2",
        "INVERTED": "#ff8a65",
    }
    trigger_specs = [
        ("VIX_FULL_RISK_TRIGGER", "VIX", "o", "#7b3294"),
        ("CREDIT_FULL_RISK_TRIGGER", "Credit15+DD", "s", "#d7301f"),
    ]
    fig, ax = plt.subplots(figsize=(22, 7))
    dates = panel["date"]
    prices = panel["spy_price"]

    start = 0
    regimes = panel["final_regime"].fillna("OTHER").astype(str).tolist()
    for i in range(1, len(panel) + 1):
        if i == len(panel) or regimes[i] != regimes[start]:
            ax.axvspan(
                dates.iloc[start],
                dates.iloc[i - 1],
                color=regime_colors.get(regimes[start], "#bdbdbd"),
                alpha=0.42,
                linewidth=0,
            )
            start = i

    ax.plot(dates, prices, color="black", linewidth=1.1, label="SPY price", zorder=3)

    for col, label, marker, color in trigger_specs:
        event_mask = panel[col].fillna(False).astype(bool) & ~panel[col].fillna(False).astype(bool).shift(1, fill_value=False)
        sub = panel.loc[event_mask, ["date", "spy_price"]]
        if not sub.empty:
            ax.scatter(sub["date"], sub["spy_price"], marker=marker, s=34, color=color, edgecolor="white", linewidth=0.5, label=label, zorder=4)

    unlock_mask = (
        panel.get("trigger_lock_locks_unlocked_today", pd.Series(index=panel.index, dtype=object))
        .fillna("")
        .astype(str)
        .ne("")
        & panel.get("trigger_lock_locks_unlocked_today", pd.Series(index=panel.index, dtype=object))
        .fillna("")
        .astype(str)
        .ne("NONE")
    )
    unlock_dates = panel.loc[unlock_mask, "date"]
    for x in unlock_dates:
        ax.axvline(
            x=x,
            color="#4d4d4d",
            linestyle="dashed",
            linewidth=0.9,
            alpha=0.65,
            zorder=1,
        )

    handles = [plt.Line2D([0], [0], color=color, lw=8, alpha=0.28) for color in regime_colors.values()]
    labels = list(regime_colors.keys())
    leg1 = ax.legend(handles, labels, title="Regime", loc="upper left", ncol=5, framealpha=0.95)
    ax.add_artist(leg1)
    trigger_legend = ax.legend(loc="upper right", title="Trigger Events", framealpha=0.95)
    ax.add_artist(trigger_legend)
    unlock_handle = plt.Line2D([0], [0], color="#4d4d4d", linestyle="dashed", linewidth=1.0)
    ax.legend([unlock_handle], ["Trigger unlock"], loc="lower left", framealpha=0.95)

    ax.set_title("SPY Price with Refined Regime Backgrounds, Trigger Events, and Trigger Unlock Marks")
    ax.set_xlabel("")
    ax.set_ylabel("SPY price")
    ax.set_yscale("log")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "trigger_regime_spy_timeline_long.png", dpi=170)
    plt.close(fig)


def update_readme(summary: pd.DataFrame) -> None:
    path = OUT / "README_final_strategy.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Final Source-Only Strategy Outputs\n"
    marker = "\n## Stress Trigger and Turnover Diagnostics\n"
    existing = existing.split(marker)[0].rstrip()
    notes = summary.iloc[0]["notes"] if not summary.empty else ""
    section = f"""
## Stress Trigger and Turnover Diagnostics

This module is diagnostic only. It does not change the canonical final strategy rules or weights.

### Trigger Rules Summary

- `FLAT_LOW_RATE` / `FLAT_MID_RATE` / `FLAT_HIGH_RATE` / `INVERTED`: VIX lock is active.
- `FLAT_LOW_RATE` / `FLAT_MID_RATE` / `FLAT_HIGH_RATE` / `STEEP_MID_RATE` / `STEEP_HIGH_RATE` / `INVERTED`: credit lock is active.
- `STEEP_LOW_RATE` has no native VIX or credit entry. Any stress days there are carry-over days from another regime and should not be interpreted as a separate trigger-enabled stress block.
- Commodity lock is disabled in the final mainline.
- Credit entry uses `D_CREDIT_SPREAD_15D > 0.10` and `SPY <= MA20`.
- Credit unlock uses `SPY > MA50` and `CREDIT_LEVEL_Z_252D < 0.9`.
- VIX unlock uses `VIX_ZSCORE_120D < 1.5` with `SPY > MA20`.
- The state machine uses anchor exits: if stress began with VIX, VIX unlock is sufficient; if stress began with CREDIT, credit unlock is sufficient; if both were active at entry, each unlocks independently.
- Monthly SELL and recovery overlay are not part of the final state machine.

### Key Findings

- FULL_RISK entries and trigger-lock exits explain the main turnover.
- `SPY_CASH_TIMING`, cross-state asset behavior, and the final hedge strategy now use the same VIX/CREDIT anchor stress definition.
- FLAT and STEEP internal buffered regime switches are secondary contributors relative to trigger-lock entries and exits.
- {notes}

### Implication

The final mainline keeps the simplified daily credit trigger with MA50 and credit-level normalization unlock, uses buffered six-regime classification, and does not include recovery overlay or a commodity trigger.
"""
    path.write_text(existing + "\n" + section.strip() + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = add_event_reasons(add_trigger_flags(load_panel()))
    daily_cols = [
        "date",
        "base_regime",
        "refined_regime",
        "final_regime",
        "allocation_state",
        "VIX_FULL_RISK_TRIGGER",
        "RAW_CREDIT_DRAWDOWN_TRIGGER",
        "EFFECTIVE_CREDIT_DRAWDOWN_TRIGGER",
        "CREDIT_FULL_RISK_TRIGGER",
        "CMDTY_FULL_RISK_TRIGGER",
        "FULL_RISK_TRIGGER_ANY",
        "FULL_RISK_ACTIVE",
        "event_reason",
        "turnover",
        "VIX_ZSCORE_120D",
        "D_CREDIT_SPREAD_15D",
        "CMDTY_RET60",
        "GS10",
        "TERM_SPREAD_10Y_1Y",
        "monthly_either_state",
        "SPY_return",
        "SPY_return_5d",
        "SPY_return_20d",
        f"{FINAL_STRATEGY}_return",
        f"{FINAL_STRATEGY}_nav",
        f"{FINAL_STRATEGY}_drawdown",
    ] + [f"{FINAL_STRATEGY}_weight_{asset}" for asset in ASSETS]
    panel[daily_cols].to_csv(TABLE_DIR / "daily_trigger_diagnostics_panel.csv", index=False)

    freq = trigger_frequency_by_regime(panel)
    overlap = trigger_overlap_matrix(panel)
    multi = multi_trigger_by_regime(panel)
    entries = build_entry_attribution(panel)
    exits = build_exit_reentry(panel, entries)
    exits_by_regime = stress_reentry_by_regime(exits)
    steep_trans = transition_turnover(panel, steep_only=True)
    turnover_event = turnover_by_event(panel)
    alloc_trans = transition_turnover(panel, steep_only=False)
    effectiveness = trigger_effectiveness(entries)
    top_dates = top_turnover_dates(panel)
    summary = diagnostics_summary(panel, freq, multi, entries, exits, turnover_event, alloc_trans, steep_trans, effectiveness)

    freq.to_csv(TABLE_DIR / "trigger_frequency_by_regime.csv", index=False)
    overlap.to_csv(TABLE_DIR / "trigger_overlap_matrix.csv", index=False)
    multi.to_csv(TABLE_DIR / "multi_trigger_by_regime.csv", index=False)
    entries.to_csv(TABLE_DIR / "stress_entry_attribution.csv", index=False)
    exits.to_csv(TABLE_DIR / "stress_exit_reentry_diagnostics.csv", index=False)
    exits_by_regime.to_csv(TABLE_DIR / "stress_reentry_by_regime.csv", index=False)
    steep_trans.to_csv(TABLE_DIR / "steep_internal_transition_matrix.csv", index=False)
    turnover_event.to_csv(TABLE_DIR / "turnover_by_trigger_event.csv", index=False)
    alloc_trans.to_csv(TABLE_DIR / "turnover_by_allocation_state_transition.csv", index=False)
    effectiveness.to_csv(TABLE_DIR / "trigger_effectiveness_summary.csv", index=False)
    top_dates.to_csv(TABLE_DIR / "top_turnover_dates.csv", index=False)
    summary.to_csv(TABLE_DIR / "trigger_diagnostics_summary.csv", index=False)

    plot_outputs(freq, overlap, multi, entries, exits_by_regime, steep_trans, turnover_event, alloc_trans, effectiveness)
    plot_long_trigger_timeline(panel)
    update_readme(summary)

    print("Trigger frequency by regime:")
    print(freq[["final_regime", "total_days", "FULL_RISK_TRIGGER_ANY_rate", "FULL_RISK_ACTIVE_rate"]].to_string(index=False))
    print("Total FULL_RISK entries / exits:", len(entries), "/", len(exits))
    print("Top 5 turnover event reasons:")
    print(turnover_event.head(5)[["event_reason", "total_turnover", "share_of_total_turnover"]].to_string(index=False))
    print("Top 5 allocation-state transitions by turnover:")
    print(alloc_trans.head(5)[["from_allocation_state", "to_allocation_state", "total_turnover"]].to_string(index=False))
    print("STEEP internal top transition:")
    if not steep_trans.empty:
        top = steep_trans.iloc[0]
        print(f"{top['from_allocation_state']} -> {top['to_allocation_state']} turnover={top['total_turnover']:.4f}")
    highest = exits_by_regime.sort_values("reentry_20d_rate", ascending=False).head(1)
    print("Highest re-entry regime:", highest["final_regime_at_exit"].iloc[0] if not highest.empty else "NONE")
    print("Output path:", OUT)


if __name__ == "__main__":
    main()
