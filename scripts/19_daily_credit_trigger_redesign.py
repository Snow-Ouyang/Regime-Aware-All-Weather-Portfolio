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
    normal_allocation_by_regime,
    performance_metrics,
    stress_allocation_by_regime,
)


OUT = ROOT / "results" / "daily_credit_trigger_redesign"
FIG = OUT / "figures"
MAIN = ROOT / "results" / "main_pipeline_final" / "tables" / "daily_backtest_panel.csv"

SPY_CASH_BASELINE = "BASELINE_15D_ABS"

WINDOWS = {
    "2008_GFC": ("2007-10-01", "2009-06-30"),
    "2011_EURO_DEBT": ("2011-06-01", "2011-12-31"),
    "2015_2016": ("2015-05-01", "2016-03-31"),
    "2018Q4": ("2018-10-01", "2019-01-31"),
    "COVID_2020": ("2020-02-01", "2020-06-30"),
    "2022_RATE_WAR": ("2021-11-01", "2023-03-31"),
    "2025_PULLBACK": ("2025-01-01", None),
}


@dataclass(frozen=True)
class CreditVariant:
    name: str
    kind: str


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def load_panel() -> pd.DataFrame:
    if not MAIN.exists():
        raise FileNotFoundError(f"Missing main pipeline panel: {MAIN}")
    df = pd.read_csv(MAIN, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return df


def load_daily_credit() -> pd.DataFrame:
    aaa = pd.read_csv(ROOT / "data" / "raw" / "macro" / "Credit" / "DAAA.csv", parse_dates=["observation_date"])
    baa = pd.read_csv(ROOT / "data" / "raw" / "macro" / "Credit" / "DBAA.csv", parse_dates=["observation_date"])
    aaa = aaa.rename(columns={"observation_date": "date", "DAAA": "DAAA"})[["date", "DAAA"]]
    baa = baa.rename(columns={"observation_date": "date", "DBAA": "DBAA"})[["date", "DBAA"]]
    out = aaa.merge(baa, on="date", how="outer").sort_values("date").drop_duplicates("date")
    out["DAAA"] = pd.to_numeric(out["DAAA"], errors="coerce")
    out["DBAA"] = pd.to_numeric(out["DBAA"], errors="coerce")
    out["DAAA"] = out["DAAA"].ffill().bfill()
    out["DBAA"] = out["DBAA"].ffill().bfill()
    return out


def prepare_panel(df: pd.DataFrame) -> pd.DataFrame:
    credit = load_daily_credit()
    out = df.drop(columns=[c for c in ["WAAA", "WBAA", "CREDIT_SPREAD_BAA_AAA", "D_CREDIT_SPREAD_15D", "D_CREDIT_SPREAD_20D"] if c in df.columns]).copy()
    out = out.merge(credit, on="date", how="left")
    out["WAAA"] = out["DAAA"]
    out["WBAA"] = out["DBAA"]
    out["CREDIT_SPREAD"] = out["DBAA"] - out["DAAA"]
    out["CREDIT_SPREAD_BAA_AAA"] = out["CREDIT_SPREAD"]
    out["SPY_MA20"] = out["SPY_MA20"] if "SPY_MA20" in out.columns else out["spy_price"].rolling(20, min_periods=20).mean()
    out["SPY_MA50"] = out["spy_price"].rolling(50, min_periods=50).mean()
    out["SPY_MA100"] = out["spy_price"].rolling(100, min_periods=100).mean()
    out["SPY_above_MA20"] = out["spy_price"] > out["SPY_MA20"]
    out["SPY_above_MA50"] = out["spy_price"] > out["SPY_MA50"]
    out["SPY_above_MA100"] = out["spy_price"] > out["SPY_MA100"]
    out["SPY_DD"] = out["spy_drawdown_from_previous_high"]
    out["final_main_return"] = out[f"{FINAL_STRATEGY}_return"]
    out["final_main_nav"] = out[f"{FINAL_STRATEGY}_nav"]
    out["final_main_drawdown"] = out[f"{FINAL_STRATEGY}_drawdown"]
    return out


def add_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for n in [5, 10, 15, 20, 30, 60]:
        out[f"D_CREDIT_{n}D"] = out["CREDIT_SPREAD"] - out["CREDIT_SPREAD"].shift(n)
    for zwin in [126, 252, 504]:
        roll = out["CREDIT_SPREAD"].rolling(zwin, min_periods=126)
        out[f"CREDIT_LEVEL_Z_{zwin}D"] = (out["CREDIT_SPREAD"] - roll.mean()) / roll.std(ddof=1).replace(0, np.nan)
        out[f"CREDIT_PERCENTILE_{zwin}D"] = roll.rank(pct=True)
    for n in [20, 60, 120]:
        out[f"CREDIT_MA{n}"] = out["CREDIT_SPREAD"].rolling(n, min_periods=min(20, n)).mean()
        out[f"CREDIT_SPREAD_ABOVE_MA{n}"] = out["CREDIT_SPREAD"] > out[f"CREDIT_MA{n}"]
    out["CREDIT_PEAK_20D"] = out["CREDIT_SPREAD"].rolling(20, min_periods=20).max()
    out["CREDIT_PEAK_60D"] = out["CREDIT_SPREAD"].rolling(60, min_periods=60).max()
    out["CREDIT_DD_FROM_20D_PEAK"] = out["CREDIT_SPREAD"] / out["CREDIT_PEAK_20D"] - 1.0
    out["CREDIT_DD_FROM_60D_PEAK"] = out["CREDIT_SPREAD"] / out["CREDIT_PEAK_60D"] - 1.0
    out["CREDIT_ACCEL_5_20"] = out["D_CREDIT_5D"] - out["D_CREDIT_20D"] / 4.0
    out["CREDIT_ACCEL_10_60"] = out["D_CREDIT_10D"] - out["D_CREDIT_60D"] / 6.0
    for n in [10, 20, 30]:
        roll = out[f"D_CREDIT_{n}D"].rolling(252, min_periods=126)
        out[f"D_CREDIT_{n}D_Z_252D"] = (out[f"D_CREDIT_{n}D"] - roll.mean()) / roll.std(ddof=1).replace(0, np.nan)
    return out


def save_feature_panel(df: pd.DataFrame) -> None:
    keep = [
        "date",
        "spy_price",
        "CREDIT_SPREAD",
        "D_CREDIT_5D",
        "D_CREDIT_10D",
        "D_CREDIT_15D",
        "D_CREDIT_20D",
        "D_CREDIT_30D",
        "D_CREDIT_60D",
        "CREDIT_LEVEL_Z_126D",
        "CREDIT_LEVEL_Z_252D",
        "CREDIT_LEVEL_Z_504D",
        "CREDIT_PERCENTILE_126D",
        "CREDIT_PERCENTILE_252D",
        "CREDIT_PERCENTILE_504D",
        "CREDIT_MA20",
        "CREDIT_MA60",
        "CREDIT_MA120",
        "CREDIT_SPREAD_ABOVE_MA20",
        "CREDIT_SPREAD_ABOVE_MA60",
        "CREDIT_SPREAD_ABOVE_MA120",
        "CREDIT_PEAK_20D",
        "CREDIT_PEAK_60D",
        "CREDIT_DD_FROM_20D_PEAK",
        "CREDIT_DD_FROM_60D_PEAK",
        "CREDIT_ACCEL_5_20",
        "CREDIT_ACCEL_10_60",
        "D_CREDIT_10D_Z_252D",
        "D_CREDIT_20D_Z_252D",
        "D_CREDIT_30D_Z_252D",
    ]
    df[keep].to_csv(OUT / "daily_credit_feature_panel.csv", index=False)


def perf_bundle(df: pd.DataFrame, name: str) -> dict[str, float]:
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


def build_variants() -> list[CreditVariant]:
    return [
        CreditVariant("BASELINE_15D_ABS", "BASELINE"),
        CreditVariant("SHOCK_10D_20D", "SHOCK_10D_20D"),
        CreditVariant("SHOCK_OR_Z", "SHOCK_OR_Z"),
        CreditVariant("LEVEL_CONFIRMED_LOCK", "LEVEL_CONFIRMED_LOCK"),
        CreditVariant("LEVEL_OR_PERCENTILE_LOCK", "LEVEL_OR_PERCENTILE_LOCK"),
        CreditVariant("PEAK_RELIEF_UNLOCK", "PEAK_RELIEF_UNLOCK"),
        CreditVariant("LEVEL_LOCK_FAST_RELIEF", "LEVEL_LOCK_FAST_RELIEF"),
        CreditVariant("WATCH_LOCK_STATE_MACHINE", "WATCH_LOCK"),
        CreditVariant("WATCH_AS_PARTIAL_LOCK_DIAGNOSTIC", "WATCH_AS_LOCK"),
        CreditVariant("FAST_RELOCK_AFTER_UNLOCK", "FAST_RELOCK"),
        CreditVariant("LEVEL_LOCK_FAST_RELIEF_PLUS_RELOCK", "LEVEL_FAST_RELIEF_RELOCK"),
    ]


def vix_entry(row: pd.Series) -> bool:
    return bool(row["refined_regime_confirmed"] in {"FLAT_LOW_RATE", "FLAT_HIGH_RATE", "STEEP"} and row["VIX_ZSCORE_120D"] >= 3.0)


def cmdty_entry(row: pd.Series) -> bool:
    return bool(row["refined_regime_confirmed"] == "STEEP" and row["CMDTY_RET60"] < -0.10)


def vix_unlock(row: pd.Series) -> bool:
    return bool((row["VIX_ZSCORE_120D"] < 1.5) and row["SPY_above_MA20"])


def cmdty_unlock(row: pd.Series) -> bool:
    return bool((row["CMDTY_RET60"] > -0.05) and row["SPY_above_MA20"])


def credit_entry(row: pd.Series, kind: str) -> tuple[bool, str]:
    if row["refined_regime_confirmed"] not in {"FLAT_LOW_RATE", "FLAT_HIGH_RATE"}:
        return False, "NO_ENTRY_REGIME"
    dd = bool(row["SPY_DD"] <= -0.05)
    if kind == "BASELINE":
        sig = dd and pd.notna(row["D_CREDIT_15D"]) and row["D_CREDIT_15D"] > 0.10
        return bool(sig), "ABS_15D"
    if kind == "SHOCK_10D_20D":
        sig = dd and ((pd.notna(row["D_CREDIT_10D"]) and row["D_CREDIT_10D"] > 0.08) or (pd.notna(row["D_CREDIT_20D"]) and row["D_CREDIT_20D"] > 0.12))
        return bool(sig), "SHOCK_10D_20D"
    if kind == "SHOCK_OR_Z":
        sig = dd and (
            (pd.notna(row["D_CREDIT_10D"]) and row["D_CREDIT_10D"] > 0.08)
            or (pd.notna(row["D_CREDIT_20D"]) and row["D_CREDIT_20D"] > 0.12)
            or (pd.notna(row["D_CREDIT_10D_Z_252D"]) and row["D_CREDIT_10D_Z_252D"] > 1.5)
        )
        return bool(sig), "SHOCK_OR_Z"
    if kind == "LEVEL_CONFIRMED_LOCK":
        sig = dd and (
            (pd.notna(row["D_CREDIT_15D"]) and row["D_CREDIT_15D"] > 0.10)
            or (pd.notna(row["CREDIT_LEVEL_Z_252D"]) and row["CREDIT_LEVEL_Z_252D"] > 1.5 and not row["SPY_above_MA20"])
        )
        return bool(sig), "LEVEL_CONFIRMED_LOCK"
    if kind == "LEVEL_OR_PERCENTILE_LOCK":
        sig = dd and (
            (pd.notna(row["D_CREDIT_15D"]) and row["D_CREDIT_15D"] > 0.10)
            or (pd.notna(row["CREDIT_PERCENTILE_252D"]) and row["CREDIT_PERCENTILE_252D"] > 0.80 and not row["SPY_above_MA20"])
        )
        return bool(sig), "LEVEL_OR_PERCENTILE_LOCK"
    if kind == "PEAK_RELIEF_UNLOCK":
        sig = dd and pd.notna(row["D_CREDIT_15D"]) and row["D_CREDIT_15D"] > 0.10
        return bool(sig), "PEAK_RELIEF_UNLOCK"
    if kind == "LEVEL_LOCK_FAST_RELIEF":
        sig = dd and (
            (pd.notna(row["D_CREDIT_15D"]) and row["D_CREDIT_15D"] > 0.10)
            or (pd.notna(row["CREDIT_LEVEL_Z_252D"]) and row["CREDIT_LEVEL_Z_252D"] > 1.5 and not row["SPY_above_MA20"])
        )
        return bool(sig), "LEVEL_LOCK_FAST_RELIEF"
    if kind in {"WATCH_LOCK", "WATCH_AS_LOCK"}:
        sig = dd and (
            (pd.notna(row["D_CREDIT_10D"]) and row["D_CREDIT_10D"] > 0.08)
            or (pd.notna(row["D_CREDIT_20D"]) and row["D_CREDIT_20D"] > 0.12)
            or (pd.notna(row["D_CREDIT_10D_Z_252D"]) and row["D_CREDIT_10D_Z_252D"] > 1.5)
            or (pd.notna(row["CREDIT_LEVEL_Z_252D"]) and row["CREDIT_LEVEL_Z_252D"] > 1.5 and not row["SPY_above_MA20"])
        )
        return bool(sig), "WATCH_LOCK"
    if kind == "FAST_RELOCK":
        sig = dd and pd.notna(row["D_CREDIT_15D"]) and row["D_CREDIT_15D"] > 0.10
        return bool(sig), "FAST_RELOCK"
    if kind == "LEVEL_FAST_RELIEF_RELOCK":
        sig = dd and (
            (pd.notna(row["D_CREDIT_15D"]) and row["D_CREDIT_15D"] > 0.10)
            or (pd.notna(row["CREDIT_LEVEL_Z_252D"]) and row["CREDIT_LEVEL_Z_252D"] > 1.5 and not row["SPY_above_MA20"])
        )
        return bool(sig), "LEVEL_FAST_RELIEF_RELOCK"
    raise ValueError(kind)


def credit_unlock(row: pd.Series, kind: str, state: dict[str, object]) -> tuple[bool, str]:
    ma20 = bool(row["SPY_above_MA20"])
    if kind == "BASELINE":
        return bool(ma20 and pd.notna(row["D_CREDIT_15D"]) and row["D_CREDIT_15D"] < 0), "BASELINE_UNLOCK"
    if kind in {"SHOCK_10D_20D", "SHOCK_OR_Z"}:
        return bool(ma20 and pd.notna(row["D_CREDIT_10D"]) and row["D_CREDIT_10D"] < 0 and pd.notna(row["D_CREDIT_20D"]) and row["D_CREDIT_20D"] < 0), "SHOCK_UNLOCK"
    if kind == "LEVEL_CONFIRMED_LOCK":
        return bool(ma20 and pd.notna(row["D_CREDIT_15D"]) and row["D_CREDIT_15D"] < 0 and pd.notna(row["CREDIT_LEVEL_Z_252D"]) and row["CREDIT_LEVEL_Z_252D"] < 1.0), "LEVEL_Z_UNLOCK"
    if kind == "LEVEL_OR_PERCENTILE_LOCK":
        return bool(ma20 and ((pd.notna(row["CREDIT_PERCENTILE_252D"]) and row["CREDIT_PERCENTILE_252D"] < 0.70) or (pd.notna(row["CREDIT_DD_FROM_60D_PEAK"]) and row["CREDIT_DD_FROM_60D_PEAK"] <= -0.20))), "PERCENTILE_OR_RELIEF_UNLOCK"
    if kind == "PEAK_RELIEF_UNLOCK":
        return bool(ma20 and ((pd.notna(row["CREDIT_DD_FROM_20D_PEAK"]) and row["CREDIT_DD_FROM_20D_PEAK"] <= -0.25) or (pd.notna(row["CREDIT_DD_FROM_60D_PEAK"]) and row["CREDIT_DD_FROM_60D_PEAK"] <= -0.20) or (pd.notna(row["CREDIT_MA20"]) and row["CREDIT_SPREAD"] < row["CREDIT_MA20"]))), "PEAK_RELIEF_UNLOCK"
    if kind == "LEVEL_LOCK_FAST_RELIEF":
        return bool(ma20 and ((pd.notna(row["CREDIT_LEVEL_Z_252D"]) and row["CREDIT_LEVEL_Z_252D"] < 1.0) or (pd.notna(row["CREDIT_DD_FROM_20D_PEAK"]) and row["CREDIT_DD_FROM_20D_PEAK"] <= -0.25) or (pd.notna(row["CREDIT_MA20"]) and row["CREDIT_SPREAD"] < row["CREDIT_MA20"]))), "LEVEL_FAST_RELIEF_UNLOCK"
    if kind in {"WATCH_LOCK", "WATCH_AS_LOCK"}:
        unlock = bool(ma20 and ((pd.notna(row["CREDIT_LEVEL_Z_252D"]) and row["CREDIT_LEVEL_Z_252D"] < 1.0) or (pd.notna(row["CREDIT_DD_FROM_20D_PEAK"]) and row["CREDIT_DD_FROM_20D_PEAK"] <= -0.25) or (pd.notna(row["CREDIT_MA20"]) and row["CREDIT_SPREAD"] < row["CREDIT_MA20"])))
        return unlock, "WATCH_UNLOCK"
    if kind == "FAST_RELOCK":
        return bool(ma20 and pd.notna(row["D_CREDIT_15D"]) and row["D_CREDIT_15D"] < 0), "FAST_RELOCK_UNLOCK"
    if kind == "LEVEL_FAST_RELIEF_RELOCK":
        return bool(ma20 and ((pd.notna(row["CREDIT_LEVEL_Z_252D"]) and row["CREDIT_LEVEL_Z_252D"] < 1.0) or (pd.notna(row["CREDIT_DD_FROM_20D_PEAK"]) and row["CREDIT_DD_FROM_20D_PEAK"] <= -0.25) or (pd.notna(row["CREDIT_MA20"]) and row["CREDIT_SPREAD"] < row["CREDIT_MA20"]))), "LEVEL_FAST_RELIEF_UNLOCK"
    raise ValueError(kind)


def credit_watch(row: pd.Series) -> bool:
    return bool(
        (pd.notna(row["CREDIT_LEVEL_Z_252D"]) and row["CREDIT_LEVEL_Z_252D"] > 1.0)
        or (pd.notna(row["CREDIT_PERCENTILE_252D"]) and row["CREDIT_PERCENTILE_252D"] > 0.75)
        or (pd.notna(row["CREDIT_MA60"]) and row["CREDIT_SPREAD"] > row["CREDIT_MA60"])
    )


def credit_watch_off(row: pd.Series) -> bool:
    return bool(
        (pd.notna(row["CREDIT_LEVEL_Z_252D"]) and row["CREDIT_LEVEL_Z_252D"] < 0.5)
        or (pd.notna(row["CREDIT_PERCENTILE_252D"]) and row["CREDIT_PERCENTILE_252D"] < 0.60)
    )


def credit_relock(row: pd.Series, kind: str, days_since_unlock: int | None) -> tuple[bool, str]:
    if days_since_unlock is None or days_since_unlock > 21:
        return False, "NO_RELOCK_WINDOW"
    if kind == "FAST_RELOCK":
        sig = (not row["SPY_above_MA20"]) and (
            (pd.notna(row["D_CREDIT_10D"]) and row["D_CREDIT_10D"] > 0)
            or (pd.notna(row["D_CREDIT_20D"]) and row["D_CREDIT_20D"] > 0)
            or row["VIX_ZSCORE_120D"] > 1.5
        )
        return bool(sig), "FAST_RELOCK"
    if kind == "LEVEL_FAST_RELIEF_RELOCK":
        sig = (not row["SPY_above_MA20"]) and (
            (pd.notna(row["D_CREDIT_10D"]) and row["D_CREDIT_10D"] > 0)
            or (pd.notna(row["CREDIT_LEVEL_Z_252D"]) and row["CREDIT_LEVEL_Z_252D"] > 1.0)
            or row["VIX_ZSCORE_120D"] > 1.5
        )
        return bool(sig), "LEVEL_FAST_RELIEF_RELOCK"
    return False, "NO_RELOCK_RULE"


def simulate_spy_cash(panel: pd.DataFrame, variant: CreditVariant) -> tuple[pd.DataFrame, pd.DataFrame]:
    current_locks: set[str] = set()
    pending_locks: set[str] = set()
    last_credit_unlock_idx: int | None = None
    watch_active = False
    pending_watch = False
    rows = []
    weights = pd.DataFrame(0.0, index=panel.index, columns=ASSETS)

    for i, row in panel.iterrows():
        current_locks = set(pending_locks)
        watch_active = pending_watch
        days_since_unlock = None if last_credit_unlock_idx is None else i - last_credit_unlock_idx

        is_lock = bool(current_locks)
        if variant.kind == "WATCH_AS_LOCK" and watch_active:
            is_lock = True

        if is_lock:
            weights.loc[i, "CASH"] = 1.0
        else:
            weights.loc[i, "SPY"] = 1.0

        added_today: list[str] = []
        unlocked_today: list[str] = []
        entry_types: list[str] = []
        unlock_types: list[str] = []

        # watch state transitions
        if variant.kind in {"WATCH_LOCK", "WATCH_AS_LOCK"}:
            if credit_watch(row):
                pending_watch = True
            elif credit_watch_off(row):
                pending_watch = False

        # add standard locks
        if "VIX" not in current_locks and vix_entry(row):
            pending_locks.add("VIX")
            added_today.append("VIX")
            entry_types.append("VIX")
        if "CMDTY" not in current_locks and cmdty_entry(row):
            pending_locks.add("CMDTY")
            added_today.append("CMDTY")
            entry_types.append("CMDTY")

        credit_ent, credit_ent_type = credit_entry(row, variant.kind)
        credit_rel, credit_rel_type = credit_relock(row, variant.kind, days_since_unlock)
        if "CREDIT" not in current_locks and (credit_ent or credit_rel):
            pending_locks.add("CREDIT")
            added_today.append("CREDIT")
            entry_types.append(credit_rel_type if credit_rel else credit_ent_type)

        # unlocks
        if "VIX" in current_locks and vix_unlock(row):
            pending_locks.discard("VIX")
            unlocked_today.append("VIX")
            unlock_types.append("VIX_UNLOCK")
            if "CREDIT" in current_locks:
                pending_locks.discard("CREDIT")
                unlocked_today.append("CREDIT")
                unlock_types.append("CREDIT_BY_VIX")
                last_credit_unlock_idx = i
        if "CMDTY" in current_locks and cmdty_unlock(row):
            pending_locks.discard("CMDTY")
            unlocked_today.append("CMDTY")
            unlock_types.append("CMDTY_UNLOCK")
        cred_unl, cred_unl_type = credit_unlock(row, variant.kind, {})
        if "CREDIT" in current_locks and cred_unl and "CREDIT" not in unlocked_today:
            pending_locks.discard("CREDIT")
            unlocked_today.append("CREDIT")
            unlock_types.append(cred_unl_type)
            last_credit_unlock_idx = i

        rows.append(
            {
                "date": row["date"],
                "credit_variant": variant.name,
                "stress_active": is_lock,
                "credit_lock_active": "CREDIT" in current_locks,
                "credit_watch_active": watch_active,
                "vix_lock_active": "VIX" in current_locks,
                "cmdty_lock_active": "CMDTY" in current_locks,
                "active_locks": "+".join(sorted(current_locks)),
                "lock_add_types": "+".join(entry_types),
                "lock_unlock_types": "+".join(unlock_types),
                "locks_added_today": "+".join(added_today),
                "locks_unlocked_today": "+".join(unlocked_today),
            }
        )
    state = pd.DataFrame(rows)
    return weights, state


def build_final_challenger(panel: pd.DataFrame, variant: CreditVariant) -> pd.DataFrame:
    flat_low_normal = monthly_hold_weights(panel, ["SPY", "CMDTY_FUT"], window=INV_VOL_WINDOW)
    flat_high_normal = monthly_hold_weights(panel, ["GOLD", "CMDTY_FUT"], window=INV_VOL_WINDOW)
    steep_high_normal = monthly_hold_weights(panel, ["SPY", "CMDTY_FUT"], window=INV_VOL_WINDOW)
    inverted_normal = monthly_hold_weights(panel, ["SPY", "GOLD"], window=INV_VOL_WINDOW)
    current_locks: set[str] = set()
    pending_locks: set[str] = set()
    last_credit_unlock_idx: int | None = None
    watch_active = False
    pending_watch = False
    weights = pd.DataFrame(0.0, index=panel.index, columns=ASSETS)
    rows = []
    for i, row in panel.iterrows():
        current_locks = set(pending_locks)
        watch_active = pending_watch
        refined = row["refined_regime_confirmed"]
        days_since_unlock = None if last_credit_unlock_idx is None else i - last_credit_unlock_idx
        if current_locks:
            if refined == "INVERTED":
                w, allocation_state = normal_allocation_by_regime(
                    i, refined, row["steep_rate_regime_confirmed"], flat_low_normal, flat_high_normal, steep_high_normal, inverted_normal
                )
            else:
                w, allocation_state = stress_allocation_by_regime(refined)
        else:
            w, allocation_state = normal_allocation_by_regime(
                i, refined, row["steep_rate_regime_confirmed"], flat_low_normal, flat_high_normal, steep_high_normal, inverted_normal
            )
        weights.loc[i, ASSETS] = pd.Series(w)

        added_today: list[str] = []
        unlocked_today: list[str] = []
        entry_types: list[str] = []
        unlock_types: list[str] = []

        if variant.kind in {"WATCH_LOCK", "WATCH_AS_LOCK"}:
            if credit_watch(row):
                pending_watch = True
            elif credit_watch_off(row):
                pending_watch = False

        if "VIX" not in current_locks and vix_entry(row):
            pending_locks.add("VIX")
            added_today.append("VIX")
            entry_types.append("VIX")
        if "CMDTY" not in current_locks and cmdty_entry(row):
            pending_locks.add("CMDTY")
            added_today.append("CMDTY")
            entry_types.append("CMDTY")
        cent, cent_t = credit_entry(row, variant.kind)
        crel, crel_t = credit_relock(row, variant.kind, days_since_unlock)
        if "CREDIT" not in current_locks and (cent or crel):
            pending_locks.add("CREDIT")
            added_today.append("CREDIT")
            entry_types.append(crel_t if crel else cent_t)

        if "VIX" in current_locks and vix_unlock(row):
            pending_locks.discard("VIX")
            unlocked_today.append("VIX")
            unlock_types.append("VIX_UNLOCK")
            if "CREDIT" in current_locks:
                pending_locks.discard("CREDIT")
                unlocked_today.append("CREDIT")
                unlock_types.append("CREDIT_BY_VIX")
                last_credit_unlock_idx = i
        if "CMDTY" in current_locks and cmdty_unlock(row):
            pending_locks.discard("CMDTY")
            unlocked_today.append("CMDTY")
            unlock_types.append("CMDTY_UNLOCK")
        cunl, cunl_t = credit_unlock(row, variant.kind, {})
        if "CREDIT" in current_locks and cunl and "CREDIT" not in unlocked_today:
            pending_locks.discard("CREDIT")
            unlocked_today.append("CREDIT")
            unlock_types.append(cunl_t)
            last_credit_unlock_idx = i

        rows.append(
            {
                "date": row["date"],
                "credit_variant": variant.name,
                "stress_active": bool(current_locks),
                "credit_lock_active": "CREDIT" in current_locks,
                "credit_watch_active": watch_active,
                "vix_lock_active": "VIX" in current_locks,
                "cmdty_lock_active": "CMDTY" in current_locks,
                "active_locks": "+".join(sorted(current_locks)),
                "lock_add_types": "+".join(entry_types),
                "lock_unlock_types": "+".join(unlock_types),
                "locks_added_today": "+".join(added_today),
                "locks_unlocked_today": "+".join(unlocked_today),
            }
        )
    state = pd.DataFrame(rows)
    name = f"FINAL_CHALLENGER_{variant.name}"
    strat = compute_strategy(panel, weights, name)
    return pd.concat([panel[["date"]], weights.add_prefix("weight_"), strat, state.drop(columns=["date"])], axis=1)


def strategy_frame(panel: pd.DataFrame, weights: pd.DataFrame, state: pd.DataFrame, name: str) -> pd.DataFrame:
    strat = compute_strategy(panel, weights, name)
    return pd.concat([panel[["date"]], weights.add_prefix("weight_"), strat, state.drop(columns=["date"])], axis=1)


def perf_row(frame: pd.DataFrame, name: str) -> dict[str, object]:
    p = perf_bundle(frame, name)
    return {
        "credit_variant": name,
        **p,
        "time_in_credit_lock": int(frame["credit_lock_active"].sum()),
        "time_in_credit_watch": int(frame["credit_watch_active"].sum()),
        "number_credit_entries": int(frame["locks_added_today"].astype(str).str.contains("CREDIT").sum()),
        "number_credit_unlocks": int(frame["locks_unlocked_today"].astype(str).str.contains("CREDIT").sum()),
        "number_relocks": int(frame["lock_add_types"].astype(str).str.contains("RELOCK").sum()),
    }


def episode_diag(panel: pd.DataFrame, frame: pd.DataFrame, name: str) -> pd.DataFrame:
    r21 = forward_return(panel["SPY_return"], 21)
    m21 = forward_mdd(panel["SPY_return"], 21)
    r63 = forward_return(panel["SPY_return"], 63)
    m63 = forward_mdd(panel["SPY_return"], 63)
    rows = []
    for ep_id, (s, e) in enumerate(find_episodes(frame["credit_lock_active"].astype(bool)), start=1):
        unlock_idx = min(e + 1, len(panel) - 1)
        trough = float(panel.loc[s:e, "spy_price"].min())
        unlock_price = float(panel.loc[unlock_idx, "spy_price"])
        trough_to_unlock = unlock_price / trough - 1.0 if trough > 0 else np.nan
        relock_ep = bool(frame.loc[unlock_idx + 1 : min(unlock_idx + 21, len(frame) - 1), "credit_lock_active"].astype(bool).any())
        rows.append(
            {
                "credit_variant": name,
                "episode_id": ep_id,
                "entry_date": panel.loc[s, "date"],
                "unlock_date": panel.loc[unlock_idx, "date"],
                "duration_days": int(e - s + 1),
                "relock_episode": relock_ep,
                "entry_type": frame.loc[s, "lock_add_types"],
                "unlock_type": frame.loc[unlock_idx, "lock_unlock_types"],
                "macro_regime_at_entry": panel.loc[s, "macro_regime_confirmed"],
                "entry_SPY_DD": panel.loc[s, "SPY_DD"],
                "entry_CREDIT_SPREAD": panel.loc[s, "CREDIT_SPREAD"],
                "entry_D_CREDIT_10D": panel.loc[s, "D_CREDIT_10D"],
                "entry_D_CREDIT_15D": panel.loc[s, "D_CREDIT_15D"],
                "entry_D_CREDIT_20D": panel.loc[s, "D_CREDIT_20D"],
                "entry_CREDIT_LEVEL_Z": panel.loc[s, "CREDIT_LEVEL_Z_252D"],
                "entry_CREDIT_PERCENTILE": panel.loc[s, "CREDIT_PERCENTILE_252D"],
                "unlock_CREDIT_SPREAD": panel.loc[unlock_idx, "CREDIT_SPREAD"],
                "unlock_D_CREDIT_10D": panel.loc[unlock_idx, "D_CREDIT_10D"],
                "unlock_D_CREDIT_20D": panel.loc[unlock_idx, "D_CREDIT_20D"],
                "unlock_CREDIT_LEVEL_Z": panel.loc[unlock_idx, "CREDIT_LEVEL_Z_252D"],
                "unlock_CREDIT_DD_FROM_20D_PEAK": panel.loc[unlock_idx, "CREDIT_DD_FROM_20D_PEAK"],
                "unlock_CREDIT_DD_FROM_60D_PEAK": panel.loc[unlock_idx, "CREDIT_DD_FROM_60D_PEAK"],
                "SPY_return_during_lock": period_return(panel.loc[s:e, "SPY_return"]),
                "SPY_maxDD_during_lock": period_mdd(panel.loc[s:e, "SPY_return"]),
                "CASH_return_during_lock": period_return(panel.loc[s:e, "CASH_return"]),
                "next_21d_SPY_return_after_unlock": r21.iloc[unlock_idx],
                "next_21d_SPY_maxDD_after_unlock": m21.iloc[unlock_idx],
                "next_63d_SPY_return_after_unlock": r63.iloc[unlock_idx],
                "next_63d_SPY_maxDD_after_unlock": m63.iloc[unlock_idx],
                "false_recovery_flag": bool(
                    (pd.notna(m21.iloc[unlock_idx]) and m21.iloc[unlock_idx] <= -0.05)
                    or (pd.notna(m63.iloc[unlock_idx]) and m63.iloc[unlock_idx] <= -0.08)
                    or frame.loc[unlock_idx + 1 : min(unlock_idx + 63, len(frame) - 1), "stress_active"].astype(bool).any()
                ),
                "missed_rebound_flag": bool(pd.notna(trough_to_unlock) and trough_to_unlock > 0.08),
            }
        )
    return pd.DataFrame(rows)


def crisis_row(frame: pd.DataFrame, ep: pd.DataFrame, name: str, start: str, end: str | None, label: str) -> dict[str, object]:
    mask = frame["date"] >= pd.Timestamp(start)
    if end is not None:
        mask &= frame["date"] <= pd.Timestamp(end)
    sub = frame.loc[mask]
    ret = sub[f"{name}_return"]
    nav = (1.0 + ret.fillna(0.0)).cumprod()
    ann_vol = ret.std(ddof=1) * np.sqrt(252.0)
    ann_ret = nav.iloc[-1] ** (252.0 / len(sub)) - 1.0 if len(sub) else np.nan
    ep_sub = ep.loc[ep["entry_date"].between(pd.Timestamp(start), pd.Timestamp(end) if end is not None else frame["date"].max())] if len(ep) else pd.DataFrame()
    return {
        "credit_variant": label,
        "window": label if False else label,  # placeholder overwritten below
        "cumulative_return": float(nav.iloc[-1] - 1.0) if len(sub) else np.nan,
        "max_drawdown": float((nav / nav.cummax() - 1.0).min()) if len(sub) else np.nan,
        "Sharpe": float(ann_ret / ann_vol) if len(sub) and ann_vol > 0 else np.nan,
        "time_in_credit_lock": int(sub["credit_lock_active"].sum()) if len(sub) else 0,
        "time_in_credit_watch": int(sub["credit_watch_active"].sum()) if len(sub) else 0,
        "number_credit_entries": int(sub["locks_added_today"].astype(str).str.contains("CREDIT").sum()) if len(sub) else 0,
        "number_credit_unlocks": int(sub["locks_unlocked_today"].astype(str).str.contains("CREDIT").sum()) if len(sub) else 0,
        "number_relocks": int(sub["lock_add_types"].astype(str).str.contains("RELOCK").sum()) if len(sub) else 0,
        "false_recovery_count": int(ep_sub["false_recovery_flag"].sum()) if len(ep_sub) else 0,
        "missed_rebound_count": int(ep_sub["missed_rebound_flag"].sum()) if len(ep_sub) else 0,
    }


def rank_variants(perf: pd.DataFrame) -> pd.DataFrame:
    d = perf.copy()
    d["rank_sharpe"] = d["Sharpe"].rank(ascending=False, method="min")
    d["rank_maxdd"] = d["MaxDD"].rank(ascending=False, method="min")
    d["rank_final_equity"] = d["Final Equity"].rank(ascending=False, method="min")
    d["rank_false_recovery"] = d["false_recovery_count"].rank(ascending=True, method="min")
    d["rank_missed_rebound"] = d["missed_rebound_count"].rank(ascending=True, method="min")
    d["balanced_composite"] = (
        0.30 * d["rank_sharpe"]
        + 0.25 * d["rank_maxdd"]
        + 0.20 * d["rank_final_equity"]
        + 0.15 * d["rank_false_recovery"]
        + 0.10 * d["rank_missed_rebound"]
    )
    return d.sort_values(["balanced_composite", "Sharpe"], ascending=[True, False]).reset_index(drop=True)


def plot_feature_timeline(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
    axes[0].plot(df["date"], df["CREDIT_SPREAD"], color="firebrick")
    axes[0].set_title("Daily Credit Spread")
    axes[1].plot(df["date"], df["D_CREDIT_10D"], label="D10")
    axes[1].plot(df["date"], df["D_CREDIT_20D"], label="D20")
    axes[1].plot(df["date"], df["D_CREDIT_15D"], label="D15")
    axes[1].legend(frameon=False)
    axes[1].set_title("Credit Change")
    axes[2].plot(df["date"], df["CREDIT_LEVEL_Z_252D"], label="Z252")
    axes[2].plot(df["date"], df["CREDIT_PERCENTILE_252D"], label="Pct252")
    axes[2].legend(frameon=False)
    axes[2].set_title("Level Z / Percentile")
    axes[3].plot(df["date"], df["CREDIT_DD_FROM_20D_PEAK"], label="DD from 20D peak")
    axes[3].plot(df["date"], df["CREDIT_DD_FROM_60D_PEAK"], label="DD from 60D peak")
    axes[3].legend(frameon=False)
    axes[3].set_title("Peak Relief")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "daily_credit_feature_timeline.png", dpi=180)
    plt.close(fig)


def plot_entry_unlock(panel: pd.DataFrame, frames: dict[str, pd.DataFrame], names: list[str]) -> None:
    fig, axes = plt.subplots(len(names), 1, figsize=(16, 3 * len(names)), sharex=True)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        df = frames[name]
        ax.plot(panel["date"], panel["CREDIT_SPREAD"], color="firebrick", linewidth=0.9)
        ax.scatter(df.loc[df["locks_added_today"].astype(str).str.contains("CREDIT"), "date"], panel.loc[df["locks_added_today"].astype(str).str.contains("CREDIT"), "CREDIT_SPREAD"], marker="^", color="black", s=18)
        ax.scatter(df.loc[df["locks_unlocked_today"].astype(str).str.contains("CREDIT"), "date"], panel.loc[df["locks_unlocked_today"].astype(str).str.contains("CREDIT"), "CREDIT_SPREAD"], marker="v", color="royalblue", s=18)
        ax.set_title(name)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "credit_entry_unlock_timeline_by_variant.png", dpi=180)
    plt.close(fig)


def plot_spy_cash_curves(frames: dict[str, pd.DataFrame], baseline: str, best: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    for name in [baseline, best]:
        ax.plot(frames[name]["date"], frames[name][f"{name}_nav"], label=name)
    ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.set_title("SPY/CASH Daily Credit Variants")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "spy_cash_credit_variant_equity_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6))
    for name in [baseline, best]:
        ax.plot(frames[name]["date"], frames[name][f"{name}_drawdown"], label=name)
    ax.legend(frameon=False)
    ax.set_title("SPY/CASH Daily Credit Variant Drawdown")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "spy_cash_credit_variant_drawdown.png", dpi=160)
    plt.close(fig)


def plot_tradeoff(perf: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(perf["false_recovery_count"], perf["missed_rebound_count"], s=(perf["Sharpe"] * 120).clip(lower=30), c=perf["MaxDD"], cmap="viridis_r")
    for _, row in perf.iterrows():
        ax.text(row["false_recovery_count"], row["missed_rebound_count"], row["credit_variant"], fontsize=8)
    ax.set_xlabel("False recovery")
    ax.set_ylabel("Missed rebound")
    ax.set_title("Daily Credit Variant Trade-off")
    fig.colorbar(sc, ax=ax, label="MaxDD")
    fig.tight_layout()
    fig.savefig(FIG / "credit_variant_tradeoff_scatter.png", dpi=160)
    plt.close(fig)


def plot_crisis_heatmap(crisis: pd.DataFrame, names: list[str], path: Path) -> None:
    sub = crisis.loc[crisis["credit_variant"].isin(names)]
    heat = sub.pivot(index="credit_variant", columns="window", values="cumulative_return")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(heat, annot=True, fmt=".1%", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title("Crisis Window Heatmap")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_case(panel: pd.DataFrame, frames: dict[str, pd.DataFrame], names: list[str], tag: str, start: str, end: str | None, out_name: str) -> None:
    mask = panel["date"] >= pd.Timestamp(start)
    if end is not None:
        mask &= panel["date"] <= pd.Timestamp(end)
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    axes[0].plot(panel.loc[mask, "date"], panel.loc[mask, "spy_price"], color="black")
    axes[0].plot(panel.loc[mask, "date"], panel.loc[mask, "SPY_MA20"], color="orange", linewidth=0.9)
    axes[0].set_yscale("log")
    axes[0].set_title(f"{tag}: SPY")
    axes[1].plot(panel.loc[mask, "date"], panel.loc[mask, "CREDIT_SPREAD"], color="firebrick")
    axes[1].set_title("Credit Spread")
    for name in names:
        axes[2].plot(frames[name].loc[mask, "date"], frames[name][f"{name}_nav"].loc[mask], label=name)
    axes[2].legend(frameon=False)
    axes[2].set_title("Strategy NAV")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / out_name, dpi=170)
    plt.close(fig)


def build_report(perf: pd.DataFrame, final_perf: pd.DataFrame, materially_better: pd.DataFrame) -> None:
    ranked = rank_variants(perf)
    best_spy_cash = ranked.iloc[0]
    best_final = final_perf.sort_values("Sharpe", ascending=False).iloc[0]
    has_better = len(materially_better) > 0
    lines = [
        "# DAILY_CREDIT_TRIGGER_REDESIGN_REPORT",
        "",
        "## 1. Purpose",
        "",
        "Redesign daily credit trigger logic after moving from weekly forward-filled credit data to daily DAAA/DBAA credit series.",
        "",
        "## 2. Observed problems from visualization",
        "",
        "- 2008: early unlock and weak persistence through the sustained credit spike.",
        "- 2020: fast spike followed by fast relief, so overly strict unlock risks missing the rebound.",
        "- 2022: stair-step elevated credit stress with repeated local improvements and relapses.",
        "",
        "## 3. Daily credit features",
        "",
        "We compare shock changes, level z-score, percentile, moving-average trend, and peak-relief features.",
        "",
        "## 4. Credit variants",
        "",
        "Variants range from simple daily baseline to watch-state state machines and fast-relock hybrids.",
        "",
        "## 5. SPY/CASH laboratory results",
        "",
        f"- Best SPY/CASH variant: `{best_spy_cash['credit_variant']}`",
        "",
        "## 6. Case study analysis",
        "",
        "Use 2008, 2020, 2022, and 2025 case charts to judge persistence vs fast relief behavior.",
        "",
        "## 7. Final strategy challenger results",
        "",
        f"- Best final challenger by Sharpe: `{best_final['credit_variant']}`",
        "",
        "## 8. Recommendation",
        "",
        ("Adopt a redesigned daily credit rule." if has_better else "Keep baseline daily credit. No final challenger is materially better."),
        "",
        "## 9. Limitations",
        "",
        "- daily credit data availability and revisions",
        "- in-sample parameter risk",
        "- credit behavior differs across crises",
    ]
    (OUT / "DAILY_CREDIT_TRIGGER_REDESIGN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    patch = (
        f"The daily credit redesign suggests replacing the baseline credit trigger with {materially_better.iloc[0]['credit_variant']}."
        if has_better
        else "Daily credit trigger redesign found useful diagnostics but no robust replacement. The baseline credit trigger remains in the final strategy."
    )
    (OUT / "README_PATCH_SUGGESTION.md").write_text(patch, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = add_credit_features(prepare_panel(load_panel()))
    save_feature_panel(panel)

    spy_cash_frames: dict[str, pd.DataFrame] = {}
    spy_cash_perf_rows = []
    spy_cash_crisis_rows = []
    ep_rows = []

    for variant in build_variants():
        w, state = simulate_spy_cash(panel, variant)
        frame = strategy_frame(panel, w, state, variant.name)
        spy_cash_frames[variant.name] = frame
        ep = episode_diag(panel, frame, variant.name)
        perf = perf_row(frame, variant.name)
        perf["avg_credit_lock_duration"] = float(ep["duration_days"].mean()) if len(ep) else np.nan
        perf["false_recovery_count"] = int(ep["false_recovery_flag"].sum()) if len(ep) else 0
        perf["missed_rebound_count"] = int(ep["missed_rebound_flag"].sum()) if len(ep) else 0
        spy_cash_perf_rows.append(perf)
        if len(ep):
            ep_rows.append(ep)
        for window, (start, end) in WINDOWS.items():
            row = crisis_row(frame, ep, variant.name, start, end, variant.name)
            row["window"] = window
            spy_cash_crisis_rows.append(row)

    spy_cash_perf = pd.DataFrame(spy_cash_perf_rows)
    spy_cash_ranked = rank_variants(spy_cash_perf)
    best_spy_cash = str(spy_cash_ranked.iloc[0]["credit_variant"])
    best_sharpe = str(spy_cash_perf.sort_values("Sharpe", ascending=False).iloc[0]["credit_variant"])
    best_mdd = str(spy_cash_perf.sort_values("MaxDD", ascending=False).iloc[0]["credit_variant"])
    spy_cash_crisis = pd.DataFrame(spy_cash_crisis_rows)
    episodes = pd.concat(ep_rows, ignore_index=True) if ep_rows else pd.DataFrame()

    spy_cash_perf.to_csv(OUT / "spy_cash_daily_credit_performance.csv", index=False)
    spy_cash_crisis.to_csv(OUT / "spy_cash_daily_credit_crisis_comparison.csv", index=False)
    episodes.to_csv(OUT / "daily_credit_episode_diagnostics.csv", index=False)

    plot_feature_timeline(panel)
    plot_entry_unlock(panel, spy_cash_frames, [SPY_CASH_BASELINE, best_sharpe, best_mdd, best_spy_cash])
    plot_spy_cash_curves(spy_cash_frames, SPY_CASH_BASELINE, best_spy_cash)
    plot_tradeoff(spy_cash_perf)
    plot_crisis_heatmap(spy_cash_crisis, [SPY_CASH_BASELINE, best_spy_cash, best_sharpe, best_mdd], FIG / "crisis_window_heatmap_daily_credit.png")
    plot_case(panel, spy_cash_frames, [SPY_CASH_BASELINE, best_spy_cash], "2008", *WINDOWS["2008_GFC"], "case_2008_credit_variants.png")
    plot_case(panel, spy_cash_frames, [SPY_CASH_BASELINE, best_spy_cash], "2020", *WINDOWS["COVID_2020"], "case_2020_credit_variants.png")
    plot_case(panel, spy_cash_frames, [SPY_CASH_BASELINE, best_spy_cash], "2022", *WINDOWS["2022_RATE_WAR"], "case_2022_credit_variants.png")
    plot_case(panel, spy_cash_frames, [SPY_CASH_BASELINE, best_spy_cash], "2025", *WINDOWS["2025_PULLBACK"], "case_2025_credit_variants.png")

    # Final challengers
    challenger_names = list(dict.fromkeys([best_sharpe, best_mdd, best_spy_cash]))[:3]
    final_frames = {}
    final_perf_rows = []
    final_crisis_rows = []
    current_final = pd.DataFrame(
        {
            "date": panel["date"],
            f"{FINAL_STRATEGY}_return": panel[f"{FINAL_STRATEGY}_return"],
            f"{FINAL_STRATEGY}_nav": panel[f"{FINAL_STRATEGY}_nav"],
            f"{FINAL_STRATEGY}_drawdown": panel[f"{FINAL_STRATEGY}_drawdown"],
        }
    )
    base_perf = performance_metrics(panel, FINAL_STRATEGY)
    baseline_false = int(episodes.loc[episodes["credit_variant"] == SPY_CASH_BASELINE, "false_recovery_flag"].sum()) if len(episodes) else 0
    baseline_missed = int(episodes.loc[episodes["credit_variant"] == SPY_CASH_BASELINE, "missed_rebound_flag"].sum()) if len(episodes) else 0
    final_perf_rows.append(
        {
            "strategy": FINAL_STRATEGY,
            "credit_variant": "CURRENT_FINAL",
            "CAGR": base_perf["CAGR"],
            "Sharpe": base_perf["Sharpe"],
            "Sortino": base_perf["Sortino"],
            "MaxDD": base_perf["MaxDD"],
            "Calmar": base_perf["Calmar"],
            "Final Equity": base_perf["final_equity"],
            "turnover": base_perf["turnover"],
            "transaction_cost_drag": base_perf["transaction_cost"],
            "time_in_credit_lock": int(panel["trigger_lock_active_locks"].fillna("").astype(str).str.contains("CREDIT").sum()),
            "false_recovery_count": baseline_false,
            "missed_rebound_count": baseline_missed,
        }
    )

    for variant in build_variants():
        if variant.name not in challenger_names:
            continue
        frame = build_final_challenger(panel, variant)
        final_frames[variant.name] = frame
        ep = episode_diag(panel, frame, variant.name)
        perf = perf_bundle(frame, f"FINAL_CHALLENGER_{variant.name}")
        final_perf_rows.append(
            {
                "strategy": f"FINAL_CHALLENGER_{variant.name}",
                "credit_variant": variant.name,
                **perf,
                "time_in_credit_lock": int(frame["credit_lock_active"].sum()),
                "false_recovery_count": int(ep["false_recovery_flag"].sum()) if len(ep) else 0,
                "missed_rebound_count": int(ep["missed_rebound_flag"].sum()) if len(ep) else 0,
            }
        )
        for window, (start, end) in WINDOWS.items():
            row = crisis_row(frame, ep, f"FINAL_CHALLENGER_{variant.name}", start, end, variant.name)
            row["window"] = window
            final_crisis_rows.append(row)

    final_perf = pd.DataFrame(final_perf_rows)
    final_crisis = pd.DataFrame(final_crisis_rows)
    final_ranked = rank_variants(
        final_perf.rename(columns={"strategy": "credit_variant", "Final Equity": "Final Equity", "false_recovery_count": "false_recovery_count", "missed_rebound_count": "missed_rebound_count"})
        [["credit_variant", "Sharpe", "MaxDD", "Final Equity", "false_recovery_count", "missed_rebound_count"]]
    )

    current = final_perf.loc[final_perf["credit_variant"] == "CURRENT_FINAL"].iloc[0]
    materially = final_perf.loc[
        (final_perf["credit_variant"] != "CURRENT_FINAL")
        & (final_perf["Sharpe"] >= current["Sharpe"] + 0.03)
        & (final_perf["MaxDD"] >= current["MaxDD"] + 0.01)
        & (final_perf["Final Equity"] >= current["Final Equity"] * 0.98)
        & (final_perf["false_recovery_count"] <= current["false_recovery_count"])
        & (final_perf["missed_rebound_count"] <= current["missed_rebound_count"] + 2)
    ].sort_values(["Sharpe", "MaxDD"], ascending=[False, False])

    final_perf.to_csv(OUT / "final_strategy_daily_credit_challenger_performance.csv", index=False)
    final_crisis.to_csv(OUT / "final_strategy_daily_credit_challenger_crisis_comparison.csv", index=False)
    spy_cash_ranked.to_csv(OUT / "daily_credit_variant_ranking.csv", index=False)

    # final plots
    if len(final_frames):
        best_final = final_perf.loc[final_perf["credit_variant"] != "CURRENT_FINAL"].sort_values("Sharpe", ascending=False).iloc[0]["credit_variant"]
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(panel["date"], panel[f"{FINAL_STRATEGY}_nav"], label=FINAL_STRATEGY)
        for name, frame in final_frames.items():
            ax.plot(frame["date"], frame[f"FINAL_CHALLENGER_{name}_nav"], label=f"FINAL_CHALLENGER_{name}")
        ax.set_yscale("log")
        ax.legend(frameon=False)
        ax.set_title("Final Challenger Equity Curve")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(FIG / "final_challenger_equity_curve.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(panel["date"], panel[f"{FINAL_STRATEGY}_drawdown"], label=FINAL_STRATEGY)
        for name, frame in final_frames.items():
            ax.plot(frame["date"], frame[f"FINAL_CHALLENGER_{name}_drawdown"], label=f"FINAL_CHALLENGER_{name}")
        ax.legend(frameon=False)
        ax.set_title("Final Challenger Drawdown")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(FIG / "final_challenger_drawdown.png", dpi=160)
        plt.close(fig)

    build_report(spy_cash_perf, final_perf, materially)

    print("SPY/CASH baseline performance")
    print(spy_cash_perf.loc[spy_cash_perf["credit_variant"] == SPY_CASH_BASELINE, ["CAGR", "Sharpe", "MaxDD", "Calmar", "Final Equity"]].to_string(index=False))
    print("best daily credit variant in SPY/CASH")
    print(best_spy_cash)
    print("2008 / 2020 / 2022 case findings")
    print(
        spy_cash_crisis.loc[
            spy_cash_crisis["credit_variant"].isin([SPY_CASH_BASELINE, best_spy_cash])
            & spy_cash_crisis["window"].isin(["2008_GFC", "COVID_2020", "2022_RATE_WAR"]),
            ["credit_variant", "window", "cumulative_return", "max_drawdown", "false_recovery_count", "missed_rebound_count"],
        ].to_string(index=False)
    )
    print("best final challenger")
    if len(final_perf) > 1:
        print(final_perf.loc[final_perf["credit_variant"] != "CURRENT_FINAL"].sort_values("Sharpe", ascending=False).iloc[0][["credit_variant", "Sharpe", "MaxDD", "Final Equity"]].to_string())
    else:
        print("NONE")
    print("materially better candidate count")
    print(len(materially))
    print("recommendation")
    print("ADOPT REDESIGNED DAILY CREDIT RULE" if len(materially) else "KEEP BASELINE DAILY CREDIT")
    print("output paths")
    print(str(OUT))


if __name__ == "__main__":
    main()
