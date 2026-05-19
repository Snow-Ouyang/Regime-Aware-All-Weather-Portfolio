from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_strategy_source_only_core import (
    ASSETS,
    FINAL_STRATEGY,
    RECOVERY_WINDOW,
    ROOT,
    SPY_BUY_HOLD,
    SPY_CASH_TIMING,
    compute_strategy,
    monthly_hold_weights,
    normalize_weight_dict,
    performance_metrics,
)


OUT = ROOT / "results" / "trigger_lock_state_machine_v1"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
MAIN_PANEL = ROOT / "results" / "main_pipeline_final" / "daily_backtest_panel.csv"
NEW_STRATEGY = "TRIGGER_LOCK_STATE_MACHINE_V1"
CANONICAL = "RECOVERY_20D_EQUAL_WEIGHT_FLAT_LOW_ONLY"
DISPLAY = [SPY_BUY_HOLD, SPY_CASH_TIMING, CANONICAL, NEW_STRATEGY]
LOCKS = ["VIX", "CREDIT", "CMDTY"]


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_panel() -> pd.DataFrame:
    if not MAIN_PANEL.exists():
        raise FileNotFoundError(f"Missing canonical panel: {MAIN_PANEL}")
    panel = pd.read_csv(MAIN_PANEL, parse_dates=["date"])
    required = [
        "date",
        "refined_regime_confirmed",
        "SPY_return",
        "GOLD_return",
        "CMDTY_FUT_return",
        "IEF_return",
        "CASH_return",
        "VIX_ZSCORE_120D",
        "D_CREDIT_SPREAD_20D",
        "spy_drawdown_from_previous_high",
        "CMDTY_RET60",
        "spy_price",
        "SPY_MA20",
        "full_risk_state",
        "recovery_flat_low_active",
    ]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"Missing required columns in {MAIN_PANEL}: {missing}")
    if "CMDTY_FUT_price" not in panel.columns:
        panel["CMDTY_FUT_price"] = (1.0 + panel["CMDTY_FUT_return"].fillna(0.0)).cumprod()
    panel["CMDTY_RET20"] = panel["CMDTY_FUT_price"] / panel["CMDTY_FUT_price"].shift(20) - 1.0
    if "CREDIT_SPREAD_BAA_AAA" not in panel.columns:
        raise ValueError(f"Missing CREDIT_SPREAD_BAA_AAA in {MAIN_PANEL}")
    panel["D_CREDIT_SPREAD_15D"] = panel["CREDIT_SPREAD_BAA_AAA"] - panel["CREDIT_SPREAD_BAA_AAA"].shift(15)
    panel["SPY_close"] = panel["spy_price"]
    panel["refined_regime"] = panel["refined_regime_confirmed"].where(
        panel["refined_regime_confirmed"].isin(["FLAT_LOW_RATE", "FLAT_HIGH_RATE", "STEEP", "INVERTED"]),
        "OTHER",
    )
    return panel


def trigger_conditions(row: pd.Series) -> dict[str, bool]:
    refined = row["refined_regime"]
    vix = bool(row["VIX_ZSCORE_120D"] >= 3.0)
    credit = bool((row["spy_drawdown_from_previous_high"] <= -0.05) and (row["D_CREDIT_SPREAD_15D"] > 0.10))
    cmdty = bool(row["CMDTY_RET60"] < -0.10)
    if refined == "STEEP":
        return {"VIX": vix, "CREDIT": False, "CMDTY": cmdty}
    if refined in {"FLAT_LOW_RATE", "FLAT_HIGH_RATE"}:
        return {"VIX": vix, "CREDIT": credit, "CMDTY": False}
    return {"VIX": False, "CREDIT": False, "CMDTY": False}


def unlock_conditions(row: pd.Series) -> dict[str, bool]:
    spy_above = bool(row["SPY_close"] > row["SPY_MA20"]) if pd.notna(row["SPY_MA20"]) else False
    return {
        "VIX": bool((row["VIX_ZSCORE_120D"] < 1.5) and spy_above),
        "CREDIT": bool((row["D_CREDIT_SPREAD_15D"] < 0.0) and spy_above),
        "CMDTY": bool((row["CMDTY_RET60"] > -0.05) and spy_above),
    }


def normal_weights(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    flat_low = monthly_hold_weights(panel, ["SPY", "CMDTY_FUT", "GOLD"])
    flat_high = monthly_hold_weights(panel, ["CMDTY_FUT", "GOLD"])
    inverted = monthly_hold_weights(panel, ["SPY", "GOLD"])
    weights = pd.DataFrame(0.0, index=panel.index, columns=ASSETS)
    states: list[str] = []
    for i, row in panel.iterrows():
        refined = row["refined_regime"]
        if refined == "FLAT_LOW_RATE":
            w = flat_low.loc[i].to_dict()
            state = "FLAT_LOW_RATE_NORMAL"
        elif refined == "FLAT_HIGH_RATE":
            w = flat_high.loc[i].to_dict()
            state = "FLAT_HIGH_RATE_NORMAL"
        elif refined == "STEEP":
            w = {"SPY": 1.0}
            state = "STEEP_NON_RISK"
        elif refined == "INVERTED":
            w = inverted.loc[i].to_dict()
            state = "INVERTED"
        else:
            w = {"SPY": 1.0}
            state = "OTHER"
        weights.loc[i, ASSETS] = pd.Series(normalize_weight_dict(w))
        states.append(state)
    panel["base_allocation_state"] = states
    return weights, {"FLAT_LOW": flat_low, "FLAT_HIGH": flat_high, "INVERTED": inverted}


def stress_weight_dict(refined_regime: str) -> dict[str, float]:
    if refined_regime == "FLAT_LOW_RATE":
        return {"GOLD": 0.50, "IEF": 0.50}
    if refined_regime == "FLAT_HIGH_RATE":
        return {"GOLD": 0.30, "CASH": 0.70}
    if refined_regime == "STEEP":
        return {"IEF": 1.0}
    raise ValueError(f"No FULL_RISK allocation for regime {refined_regime}")


def allocation_state(refined_regime: str, full_risk: bool, recovery: bool) -> str:
    if full_risk:
        if refined_regime == "FLAT_LOW_RATE":
            return "FLAT_LOW_RATE_STRESS"
        if refined_regime == "FLAT_HIGH_RATE":
            return "FLAT_HIGH_RATE_STRESS"
        if refined_regime == "STEEP":
            return "STEEP_FULL_RISK"
        return "OTHER"
    if recovery:
        return "FLAT_LOW_RATE_RECOVERY"
    if refined_regime == "FLAT_LOW_RATE":
        return "FLAT_LOW_RATE_NORMAL"
    if refined_regime == "FLAT_HIGH_RATE":
        return "FLAT_HIGH_RATE_NORMAL"
    if refined_regime == "STEEP":
        return "STEEP_NON_RISK"
    if refined_regime == "INVERTED":
        return "INVERTED"
    return "OTHER"


def canonical_weights(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[[f"{FINAL_STRATEGY}_weight_{a}" for a in ASSETS]].rename(
        columns={f"{FINAL_STRATEGY}_weight_{a}": a for a in ASSETS}
    )


def spy_weights(panel: pd.DataFrame) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=panel.index, columns=ASSETS)
    w["SPY"] = 1.0
    return w


def spy_cash_weights(panel: pd.DataFrame) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=panel.index, columns=ASSETS)
    risk = panel["full_risk_state"].eq("FULL_RISK")
    w["SPY"] = np.where(risk, 0.0, 1.0)
    w["CASH"] = 1.0 - w["SPY"]
    return w


def weights_to_string(series: pd.Series) -> str:
    keep = [f"{asset}:{series.get(asset, 0.0):.2f}" for asset in ASSETS if abs(float(series.get(asset, 0.0))) > 1e-10]
    return "|".join(keep) if keep else "NONE"


def build_trigger_lock_strategy(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    normal_w, _ = normal_weights(panel)
    weights = pd.DataFrame(0.0, index=panel.index, columns=ASSETS)
    daily_rows: list[dict] = []
    lock_events: list[dict] = []
    pending_full_risk = False
    pending_locks: set[str] = set()
    pending_recovery_seed = False
    pending_recovery_remaining = 0
    current_episode_id = 0
    open_lock_add_idx: dict[str, int] = {}

    for i, row in panel.iterrows():
        refined = row["refined_regime"]
        current_full_risk = pending_full_risk
        current_locks = set(pending_locks)
        current_recovery_remaining = pending_recovery_remaining
        if not current_full_risk:
            if pending_recovery_seed and refined == "FLAT_LOW_RATE":
                current_recovery_remaining = RECOVERY_WINDOW
            elif refined != "FLAT_LOW_RATE":
                current_recovery_remaining = 0
        else:
            current_recovery_remaining = 0
        current_recovery_active = (not current_full_risk) and (current_recovery_remaining > 0) and (refined == "FLAT_LOW_RATE")

        if current_full_risk:
            current_episode_id = current_episode_id or 1
            w = stress_weight_dict(refined) if refined in {"FLAT_LOW_RATE", "FLAT_HIGH_RATE", "STEEP"} else {"SPY": 1.0}
        elif current_recovery_active:
            selected = [asset for asset in ["SPY", "CMDTY_FUT", "GOLD"] if pd.notna(row.get(f"{asset}_return", np.nan))]
            w = {asset: 1.0 / len(selected) for asset in selected} if selected else normal_w.loc[i].to_dict()
        else:
            w = normal_w.loc[i].to_dict()
        weights.loc[i, ASSETS] = pd.Series(normalize_weight_dict(w))
        alloc_state = allocation_state(refined, current_full_risk, current_recovery_active)

        trig = trigger_conditions(row)
        unlock = unlock_conditions(row)
        allowed_trigger_set = {name for name, active in trig.items() if active}
        lock_added_today: list[str] = []
        lock_unlocked_today: list[str] = []
        full_risk_entry_signal = False
        full_risk_exit_signal = False

        next_full_risk = current_full_risk
        next_locks = set(current_locks)
        next_recovery_seed = False
        next_recovery_remaining = current_recovery_remaining

        if current_full_risk:
            for name in sorted(allowed_trigger_set):
                if name not in next_locks:
                    next_locks.add(name)
                    lock_added_today.append(name)
                    open_lock_add_idx[name] = i
                    lock_events.append(
                        {
                            "date": row["date"],
                            "episode_id": current_episode_id,
                            "event_type": "LOCK_ADD",
                            "trigger_name": name,
                            "refined_regime": refined,
                            "condition_values": f"VIX={row['VIX_ZSCORE_120D']:.2f}|DCREDIT={row['D_CREDIT_SPREAD_20D']:.2f}|CMDTY60={row['CMDTY_RET60']:.2%}",
                            "active_locks_after_event": "|".join(sorted(next_locks)),
                        }
                    )
            for name in sorted(list(next_locks)):
                if unlock[name]:
                    next_locks.remove(name)
                    lock_unlocked_today.append(name)
                    lock_events.append(
                        {
                            "date": row["date"],
                            "episode_id": current_episode_id,
                            "event_type": "LOCK_UNLOCK",
                            "trigger_name": name,
                            "refined_regime": refined,
                            "condition_values": f"VIX={row['VIX_ZSCORE_120D']:.2f}|DCREDIT={row['D_CREDIT_SPREAD_20D']:.2f}|CMDTY20={row['CMDTY_RET20']:.2%}|SPY>MA20={row['SPY_close'] > row['SPY_MA20'] if pd.notna(row['SPY_MA20']) else False}",
                            "active_locks_after_event": "|".join(sorted(next_locks)) if next_locks else "NONE",
                        }
                    )
                    open_lock_add_idx.pop(name, None)
            if ("VIX" in current_locks or "VIX" in lock_unlocked_today) and ("CREDIT" in next_locks) and unlock["VIX"]:
                next_locks.remove("CREDIT")
                lock_unlocked_today.append("CREDIT")
                lock_events.append(
                    {
                        "date": row["date"],
                        "episode_id": current_episode_id,
                        "event_type": "LOCK_UNLOCK",
                        "trigger_name": "CREDIT",
                        "refined_regime": refined,
                        "condition_values": f"PriorityUnlockByVIX=True|VIX={row['VIX_ZSCORE_120D']:.2f}|DCREDIT={row['D_CREDIT_SPREAD_20D']:.2f}",
                        "active_locks_after_event": "|".join(sorted(next_locks)) if next_locks else "NONE",
                    }
                )
                open_lock_add_idx.pop("CREDIT", None)
            if not next_locks:
                full_risk_exit_signal = True
                next_full_risk = False
                next_recovery_seed = True
                next_recovery_remaining = 0
                current_episode_id = 0
        else:
            if allowed_trigger_set:
                full_risk_entry_signal = True
                next_full_risk = True
                next_locks = set(sorted(allowed_trigger_set))
                next_recovery_seed = False
                next_recovery_remaining = 0
                current_episode_id = current_episode_id + 1
                for name in sorted(next_locks):
                    open_lock_add_idx[name] = i
                    lock_added_today.append(name)
                    lock_events.append(
                        {
                            "date": row["date"],
                            "episode_id": current_episode_id,
                            "event_type": "LOCK_ADD",
                            "trigger_name": name,
                            "refined_regime": refined,
                            "condition_values": f"VIX={row['VIX_ZSCORE_120D']:.2f}|DCREDIT={row['D_CREDIT_SPREAD_20D']:.2f}|CMDTY60={row['CMDTY_RET60']:.2%}",
                            "active_locks_after_event": "|".join(sorted(next_locks)),
                        }
                    )
            else:
                if current_recovery_active and refined == "FLAT_LOW_RATE" and current_recovery_remaining > 1:
                    next_recovery_remaining = current_recovery_remaining - 1
                else:
                    next_recovery_remaining = 0

        daily_rows.append(
            {
                "date": row["date"],
                "refined_regime": refined,
                "full_risk_active": current_full_risk,
                "active_locks": "|".join(sorted(current_locks)) if current_locks else "NONE",
                "vix_trigger_active": "VIX" in current_locks,
                "credit_trigger_active": "CREDIT" in current_locks,
                "cmdty_trigger_active": "CMDTY" in current_locks,
                "vix_unlock_condition": unlock["VIX"],
                "credit_unlock_condition": unlock["CREDIT"],
                "cmdty_unlock_condition": unlock["CMDTY"],
                "lock_added_today": "|".join(lock_added_today) if lock_added_today else "NONE",
                "lock_unlocked_today": "|".join(lock_unlocked_today) if lock_unlocked_today else "NONE",
                "full_risk_entry_signal": full_risk_entry_signal,
                "full_risk_exit_signal": full_risk_exit_signal,
                "allocation_state": alloc_state,
                "recovery_active": current_recovery_active,
                "VIX_ZSCORE_120D": row["VIX_ZSCORE_120D"],
                "D_CREDIT_SPREAD_20D": row["D_CREDIT_SPREAD_20D"],
                "D_CREDIT_SPREAD_15D": row["D_CREDIT_SPREAD_15D"],
                "CMDTY_RET60": row["CMDTY_RET60"],
                "CMDTY_RET20": row["CMDTY_RET20"],
                "SPY_close": row["SPY_close"],
                "SPY_MA20": row["SPY_MA20"],
            }
        )

        pending_full_risk = next_full_risk
        pending_locks = set(next_locks)
        pending_recovery_seed = next_recovery_seed
        pending_recovery_remaining = next_recovery_remaining

    daily_state = pd.DataFrame(daily_rows)
    lock_event_log = pd.DataFrame(lock_events).sort_values(["date", "event_type", "trigger_name"]).reset_index(drop=True)
    return weights, daily_state, lock_event_log


def full_risk_entries_exits(active: pd.Series) -> tuple[list[int], list[int]]:
    b = active.fillna(False).astype(bool)
    entries = b.index[b & ~b.shift(1, fill_value=False)].tolist()
    exits = b.index[~b & b.shift(1, fill_value=False)].tolist()
    return entries, exits


def reentry_counts(entries: list[int], exits: list[int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for days in [5, 10, 20]:
        count = 0
        for x in exits:
            nxt = next((e for e in entries if e > x), None)
            if nxt is not None and nxt - x <= days:
                count += 1
        out[f"number_of_reentries_within_{days}d"] = count
    return out


def long_weights(panel: pd.DataFrame, strategies: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, w in strategies.items():
        tmp = w.copy()
        tmp["date"] = panel["date"]
        long = tmp.melt(id_vars="date", value_vars=ASSETS, var_name="asset", value_name="weight")
        long.insert(1, "strategy", name)
        rows.append(long)
    return pd.concat(rows, ignore_index=True)


def long_returns(outputs: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    rows = []
    for name in strategies:
        rows.append(
            pd.DataFrame(
                {
                    "date": outputs["date"],
                    "strategy": name,
                    "daily_return": outputs[f"{name}_return"],
                    "equity": outputs[f"{name}_nav"],
                    "drawdown": outputs[f"{name}_drawdown"],
                    "turnover": outputs[f"{name}_turnover"],
                    "transaction_cost": outputs[f"{name}_transaction_cost"],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def combine_strategy_outputs(panel: pd.DataFrame, strategies: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = [panel[["date"]].copy()]
    perf_rows = []
    for name, weights in strategies.items():
        out = compute_strategy(panel, weights[ASSETS].astype(float), name)
        parts.append(out)
        perf_rows.append(performance_metrics(out, name))
    return pd.concat(parts, axis=1), pd.DataFrame(perf_rows)


def augment_performance(perf: pd.DataFrame, states: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for _, row in perf.iterrows():
        name = row["strategy"]
        s = states.get(name, pd.Series(False, index=range(0)))
        entries, exits = full_risk_entries_exits(s.eq("FULL_RISK"))
        r = row.to_dict()
        r["number_of_full_risk_entries"] = len(entries)
        r["number_of_full_risk_exits"] = len(exits)
        r.update(reentry_counts(entries, exits))
        rows.append(r)
    return pd.DataFrame(rows)


def event_turnover(weights: pd.DataFrame, idxs: list[int]) -> float:
    turnover = 0.5 * weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = 0.5 * weights.iloc[0].abs().sum()
    return float(turnover.loc[idxs].sum()) if idxs else 0.0


def build_episode_log(panel: pd.DataFrame, outputs: pd.DataFrame, daily_state: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    entries, exits = full_risk_entries_exits(daily_state["full_risk_active"])
    active = daily_state["full_risk_active"].fillna(False).astype(bool)
    episode_ids = []
    lock_panel = daily_state["active_locks"]
    for ep_id, start in enumerate(entries, start=1):
        end_candidates = [idx for idx in exits if idx > start]
        end = end_candidates[0] - 1 if end_candidates else active.index[active].max()
        sub = outputs.loc[start : end]
        spy_sub = panel.loc[start : end, "SPY_return"].fillna(0.0)
        next_entry = next((e for e in entries if e > end), None)
        episode_ids.append(
            {
                "episode_id": ep_id,
                "entry_date": panel.loc[start, "date"],
                "exit_date": panel.loc[end, "date"],
                "entry_refined_regime": panel.loc[start, "refined_regime"],
                "exit_refined_regime": panel.loc[end, "refined_regime"],
                "initial_locks": daily_state.loc[start, "active_locks"],
                "all_locks_ever_active": "|".join(sorted({x for s in lock_panel.loc[start:end] for x in s.split("|") if x and x != "NONE"})) or "NONE",
                "stress_duration_days": int(end - start + 1),
                "strategy_return_during_stress": float((1.0 + sub[f"{NEW_STRATEGY}_return"]).prod() - 1.0),
                "SPY_return_during_stress": float((1.0 + spy_sub).prod() - 1.0),
                "max_drawdown_during_stress": float(sub[f"{NEW_STRATEGY}_drawdown"].min()),
                "SPY_max_drawdown_during_stress": float(((1.0 + spy_sub).cumprod() / (1.0 + spy_sub).cumprod().cummax() - 1.0).min()),
                "turnover_on_entry": float(0.5 * weights.diff().abs().sum(axis=1).fillna(0.0).iloc[start]),
                "turnover_on_exit": float(0.5 * weights.diff().abs().sum(axis=1).fillna(0.0).iloc[end]),
                "reentered_within_5d": bool(next_entry is not None and next_entry - end <= 5),
                "reentered_within_10d": bool(next_entry is not None and next_entry - end <= 10),
                "reentered_within_20d": bool(next_entry is not None and next_entry - end <= 20),
            }
        )
    return pd.DataFrame(episode_ids)


def build_trigger_summary(lock_event_log: pd.DataFrame, outputs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if lock_event_log.empty:
        return pd.DataFrame(columns=["trigger_name", "lock_add_count", "unlock_count", "average_days_locked", "median_days_locked", "max_days_locked", "average_return_while_locked", "reentry_rate_after_unlock_20d"])
    for trig in LOCKS:
        adds = lock_event_log[(lock_event_log["trigger_name"] == trig) & (lock_event_log["event_type"] == "LOCK_ADD")]
        unlocks = lock_event_log[(lock_event_log["trigger_name"] == trig) & (lock_event_log["event_type"] == "LOCK_UNLOCK")]
        durations = []
        locked_returns = []
        reentries = []
        for _, add in adds.iterrows():
            same = unlocks[(unlocks["episode_id"] == add["episode_id"]) & (unlocks["date"] >= add["date"])]
            if same.empty:
                continue
            unlock = same.iloc[0]
            durations.append(int((unlock["date"] - add["date"]).days))
            # approximate locked return over calendar rows
            rowsel = outputs[(outputs["date"] >= add["date"]) & (outputs["date"] <= unlock["date"])]
            if not rowsel.empty:
                locked_returns.append(float((1.0 + rowsel[f"{NEW_STRATEGY}_return"]).prod() - 1.0))
            next_add = adds[adds["date"] > unlock["date"]]
            if not next_add.empty:
                reentries.append(bool((next_add.iloc[0]["date"] - unlock["date"]).days <= 28))
        rows.append(
            {
                "trigger_name": trig,
                "lock_add_count": int(len(adds)),
                "unlock_count": int(len(unlocks)),
                "average_days_locked": float(np.mean(durations)) if durations else np.nan,
                "median_days_locked": float(np.median(durations)) if durations else np.nan,
                "max_days_locked": float(np.max(durations)) if durations else np.nan,
                "average_return_while_locked": float(np.mean(locked_returns)) if locked_returns else np.nan,
                "reentry_rate_after_unlock_20d": float(np.mean(reentries)) if reentries else np.nan,
            }
        )
    return pd.DataFrame(rows)


def comparison_vs_canonical(panel: pd.DataFrame, daily_state: pd.DataFrame, new_weights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    canonical_state = panel["full_risk_state"]
    for name, state, weights in [
        (CANONICAL, canonical_state, canonical_weights(panel)),
        (NEW_STRATEGY, daily_state["full_risk_active"].map({True: "FULL_RISK", False: "NON_RISK"}), new_weights),
    ]:
        entries, exits = full_risk_entries_exits(state.eq("FULL_RISK"))
        row = {
            "strategy": name,
            "full_risk_entry_count": len(entries),
            "full_risk_exit_count": len(exits),
            "reentry_5d_count": reentry_counts(entries, exits)["number_of_reentries_within_5d"],
            "reentry_10d_count": reentry_counts(entries, exits)["number_of_reentries_within_10d"],
            "reentry_20d_count": reentry_counts(entries, exits)["number_of_reentries_within_20d"],
            "turnover_from_full_risk_entry": event_turnover(weights, entries),
            "turnover_from_full_risk_exit": event_turnover(weights, exits),
            "total_turnover": float((0.5 * weights.diff().abs().sum(axis=1)).fillna(0.0).sum()),
            "STEEP_full_risk_entry_count": int(sum(panel.loc[i, "refined_regime"] == "STEEP" for i in entries)),
            "FLAT_LOW_full_risk_entry_count": int(sum(panel.loc[i, "refined_regime"] == "FLAT_LOW_RATE" for i in entries)),
            "FLAT_HIGH_full_risk_entry_count": int(sum(panel.loc[i, "refined_regime"] == "FLAT_HIGH_RATE" for i in entries)),
            "INVERTED_full_risk_entry_count": int(sum(panel.loc[i, "refined_regime"] == "INVERTED" for i in entries)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def rolling_12m(nav: pd.Series) -> pd.Series:
    return nav / nav.shift(252) - 1.0


def plot_trigger_lock_timeline(panel: pd.DataFrame, lock_event_log: pd.DataFrame) -> None:
    regime_colors = {
        "FLAT_LOW_RATE": "#8bc34a",
        "FLAT_HIGH_RATE": "#f4b400",
        "STEEP": "#4fc3f7",
        "INVERTED": "#ff8a65",
    }
    trigger_specs = {
        "VIX": ("VIX lock add", "o", "#7b3294"),
        "CREDIT": ("Credit lock add", "s", "#d7301f"),
        "CMDTY": ("Cmdty lock add", "D", "#2b8cbe"),
    }
    unlock_colors = {
        "VIX": "#7b3294",
        "CREDIT": "#d7301f",
        "CMDTY": "#2b8cbe",
    }

    fig, ax = plt.subplots(figsize=(34, 8))
    dates = panel["date"]
    prices = panel["spy_price"]
    regimes = panel["refined_regime"].astype(str).tolist()

    start = 0
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

    for trig, (label, marker, color) in trigger_specs.items():
        sub = lock_event_log[(lock_event_log["event_type"] == "LOCK_ADD") & (lock_event_log["trigger_name"] == trig)]
        if not sub.empty:
            spy_at = pd.merge(sub[["date"]], panel[["date", "spy_price"]], on="date", how="left")
            ax.scatter(
                spy_at["date"],
                spy_at["spy_price"],
                marker=marker,
                s=34,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                label=label,
                zorder=4,
            )

    first_unlock_seen: set[tuple[pd.Timestamp, str]] = set()
    for _, row in lock_event_log[lock_event_log["event_type"] == "LOCK_UNLOCK"].iterrows():
        key = (pd.Timestamp(row["date"]), str(row["trigger_name"]))
        if key in first_unlock_seen:
            continue
        first_unlock_seen.add(key)
        ax.axvline(
            x=row["date"],
            color=unlock_colors.get(str(row["trigger_name"]), "#4d4d4d"),
            linestyle="dashed",
            linewidth=0.9,
            alpha=0.65,
            zorder=1,
        )

    handles = [plt.Line2D([0], [0], color=color, lw=8, alpha=0.28) for color in regime_colors.values()]
    labels = list(regime_colors.keys())
    leg1 = ax.legend(handles, labels, title="Regime", loc="upper left", ncol=4, framealpha=0.95)
    ax.add_artist(leg1)

    trigger_leg = ax.legend(loc="upper right", title="Trigger Lock Add Events", framealpha=0.95)
    ax.add_artist(trigger_leg)

    unlock_handles = [
        plt.Line2D([0], [0], color=unlock_colors["VIX"], linestyle="dashed", linewidth=1.0),
        plt.Line2D([0], [0], color=unlock_colors["CREDIT"], linestyle="dashed", linewidth=1.0),
        plt.Line2D([0], [0], color=unlock_colors["CMDTY"], linestyle="dashed", linewidth=1.0),
    ]
    unlock_labels = ["VIX unlock", "Credit unlock", "Cmdty unlock"]
    ax.legend(unlock_handles, unlock_labels, loc="lower left", framealpha=0.95)

    ax.set_title("SPY Price with Refined Regime Backgrounds, Trigger Lock Adds, and Unlock Marks")
    ax.set_ylabel("SPY price")
    ax.set_xlabel("")
    ax.set_yscale("log")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "trigger_lock_regime_spy_timeline_long.png", dpi=170)
    plt.close(fig)


def save_plots(panel: pd.DataFrame, outputs: pd.DataFrame, strategies: dict[str, pd.DataFrame], daily_state: pd.DataFrame, trigger_summary: pd.DataFrame, comp: pd.DataFrame, lock_event_log: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for s in DISPLAY:
        ax.plot(outputs["date"], outputs[f"{s}_nav"], label=s)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "equity_curve_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for s in DISPLAY:
        ax.plot(outputs["date"], outputs[f"{s}_drawdown"], label=s)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "drawdown_curve_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for s in DISPLAY:
        ax.plot(outputs["date"], rolling_12m(outputs[f"{s}_nav"]), label=s)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rolling_12m_return_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    perf_turn = [
        float(outputs[f"{s}_turnover"].sum()) if f"{s}_turnover" in outputs.columns else 0.0
        for s in DISPLAY
    ]
    ax.bar(DISPLAY, perf_turn)
    ax.tick_params(axis="x", labelrotation=25)
    ax.set_title("Total Turnover")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "turnover_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    can_entries, can_exits = full_risk_entries_exits(panel["full_risk_state"].eq("FULL_RISK"))
    new_entries, new_exits = full_risk_entries_exits(daily_state["full_risk_active"])
    ax.scatter(panel.loc[can_entries, "date"], ["canonical_entry"] * len(can_entries), s=12, label=f"{CANONICAL} entry")
    ax.scatter(panel.loc[new_entries, "date"], ["lock_entry"] * len(new_entries), s=12, label=f"{NEW_STRATEGY} entry")
    ax.scatter(panel.loc[can_exits, "date"], ["canonical_exit"] * len(can_exits), s=12, label=f"{CANONICAL} exit")
    ax.scatter(panel.loc[new_exits, "date"], ["lock_exit"] * len(new_exits), s=12, label=f"{NEW_STRATEGY} exit")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "full_risk_entry_exit_timeline_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(panel["date"], daily_state["vix_trigger_active"].astype(int), label="VIX", linewidth=1.0)
    ax.plot(panel["date"], daily_state["credit_trigger_active"].astype(int), label="CREDIT", linewidth=1.0)
    ax.plot(panel["date"], daily_state["cmdty_trigger_active"].astype(int), label="CMDTY", linewidth=1.0)
    ax.plot(panel["date"], daily_state["full_risk_active"].astype(int), label="FULL_RISK", linewidth=1.5, alpha=0.8)
    ax.legend(fontsize=8)
    ax.set_title("Active Locks Timeline")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "active_locks_timeline.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(trigger_summary["trigger_name"], trigger_summary["average_days_locked"])
    ax.set_title("Average Lock Duration")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lock_duration_by_trigger.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(comp["strategy"]))
    ax.bar(x - 0.25, comp["reentry_5d_count"], width=0.25, label="5D")
    ax.bar(x, comp["reentry_10d_count"], width=0.25, label="10D")
    ax.bar(x + 0.25, comp["reentry_20d_count"], width=0.25, label="20D")
    ax.set_xticks(x)
    ax.set_xticklabels(comp["strategy"], rotation=25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reentry_count_comparison.png", dpi=160)
    plt.close(fig)

    new_w = strategies[NEW_STRATEGY]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.stackplot(panel["date"], *[new_w[a] for a in ASSETS], labels=ASSETS, alpha=0.9)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Trigger Lock Strategy Weights")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "weights_timeline_trigger_lock.png", dpi=160)
    plt.close(fig)

    plot_trigger_lock_timeline(panel, lock_event_log)


def write_readme(perf: pd.DataFrame, comp: pd.DataFrame, trigger_summary: pd.DataFrame) -> None:
    lines = [
        "# Trigger Lock State Machine V1",
        "",
        "## Purpose",
        "",
        "This candidate test replaces the single MA20-based FULL_RISK exit with trigger-specific locks. Each active trigger must unlock on its own recovery condition, and FULL_RISK exits only after all active locks are gone.",
        "",
        "## Credit Series Handling",
        "",
        "Canonical source-only credit data uses raw weekly `WBAA` and `WAAA`, forward-filled to trading days first. This experiment derives `D_CREDIT_SPREAD_15D = CREDIT_SPREAD_BAA_AAA - CREDIT_SPREAD_BAA_AAA.shift(15)`, i.e. a 15-trading-day change on the forward-filled daily series, not a weekly delta.",
        "",
        "## Regime-Specific Triggers",
        "",
        "- `STEEP`: `VIX` and `CMDTY` locks only.",
        "- `FLAT_LOW_RATE`: `VIX` and `CREDIT` locks.",
        "- `FLAT_HIGH_RATE`: `VIX` and `CREDIT` locks.",
        "- `INVERTED`: no FULL_RISK trigger.",
        "- `monthly SELL` is discarded entirely.",
        "",
        "## Trigger Unlock Conditions",
        "",
        "- `VIX`: `VIX_ZSCORE_120D < 1.5` and `SPY > MA20`.",
        "- `CREDIT`: `D_CREDIT_SPREAD_15D < 0` and `SPY > MA20`.",
        "- `CMDTY`: `CMDTY_RET60 > -5%` and `SPY > MA20`.",
        "- Priority rule: if `VIX` and `CREDIT` are both active and `VIX` unlocks, then `CREDIT` also unlocks on that date.",
        "",
        "## Multi-Lock Logic",
        "",
        "- New triggers during FULL_RISK add new active locks.",
        "- Locks unlock independently.",
        "- FULL_RISK exits only when all active locks are removed.",
        "- After FULL_RISK exit, FLAT_LOW_RATE-only 20D recovery overlay remains available.",
        "",
        "## Performance Comparison",
        "",
        perf.to_markdown(index=False),
        "",
        "## Trigger Metrics Comparison",
        "",
        comp.to_markdown(index=False),
        "",
        "## Lock Summary",
        "",
        trigger_summary.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "Compare turnover, re-entry counts, MaxDD, and Sharpe against the canonical final strategy. If turnover falls without damaging drawdown control, the trigger-lock state machine is a viable next candidate. If not, the existing R3-based state machine remains the better trade-off.",
    ]
    (OUT / "README_trigger_lock_state_machine_v1.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = load_panel()
    lock_weights, daily_state, lock_event_log = build_trigger_lock_strategy(panel)

    strategies = {
        SPY_BUY_HOLD: spy_weights(panel),
        SPY_CASH_TIMING: spy_cash_weights(panel),
        CANONICAL: canonical_weights(panel),
        NEW_STRATEGY: lock_weights,
    }
    outputs, perf = combine_strategy_outputs(panel, strategies)
    states = {
        SPY_BUY_HOLD: pd.Series("NON_RISK", index=panel.index),
        SPY_CASH_TIMING: panel["full_risk_state"],
        CANONICAL: panel["full_risk_state"],
        NEW_STRATEGY: daily_state["full_risk_active"].map({True: "FULL_RISK", False: "NON_RISK"}),
    }
    perf = augment_performance(perf, states)
    episode_log = build_episode_log(panel, outputs, daily_state, lock_weights)
    trigger_summary = build_trigger_summary(lock_event_log, outputs)
    comp = comparison_vs_canonical(panel, daily_state, lock_weights)

    perf.to_csv(TABLE_DIR / "performance_comparison.csv", index=False)
    long_weights(panel, strategies).to_csv(TABLE_DIR / "daily_weights_all_strategies.csv", index=False)
    long_returns(outputs, DISPLAY).to_csv(TABLE_DIR / "daily_returns_all_strategies.csv", index=False)
    daily_state.to_csv(TABLE_DIR / "trigger_lock_daily_state.csv", index=False)
    episode_log.to_csv(TABLE_DIR / "full_risk_lock_episode_log.csv", index=False)
    lock_event_log.to_csv(TABLE_DIR / "lock_event_log.csv", index=False)
    trigger_summary.to_csv(TABLE_DIR / "trigger_lock_summary.csv", index=False)
    comp.to_csv(TABLE_DIR / "comparison_vs_canonical_trigger_metrics.csv", index=False)

    save_plots(panel, outputs, strategies, daily_state, trigger_summary, comp, lock_event_log)
    write_readme(perf, comp, trigger_summary)

    show = perf[["strategy", "CAGR", "Sharpe", "MaxDD", "Calmar", "final_equity", "turnover", "number_of_full_risk_entries", "number_of_full_risk_exits"]]
    print("Performance comparison:")
    print(show.to_string(index=False))
    print("\nTurnover comparison:")
    print(comp[["strategy", "turnover_from_full_risk_entry", "turnover_from_full_risk_exit", "total_turnover"]].to_string(index=False))
    print("\nFull-risk entry / exit comparison:")
    print(comp[["strategy", "full_risk_entry_count", "full_risk_exit_count"]].to_string(index=False))
    print("\nReentry 5/10/20d comparison:")
    print(comp[["strategy", "reentry_5d_count", "reentry_10d_count", "reentry_20d_count"]].to_string(index=False))
    print("\nLock duration summary:")
    print(trigger_summary.to_string(index=False))
    base = perf.loc[perf["strategy"].eq(CANONICAL)].iloc[0]
    new = perf.loc[perf["strategy"].eq(NEW_STRATEGY)].iloc[0]
    print(
        "\nImproves turnover without worsening MaxDD:",
        bool(new["turnover"] < base["turnover"] and new["MaxDD"] >= base["MaxDD"]),
    )
    print("Output path:", OUT)


if __name__ == "__main__":
    main()
