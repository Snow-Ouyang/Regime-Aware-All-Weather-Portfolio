from __future__ import annotations

from dataclasses import dataclass
from itertools import product
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
    build_trigger_lock_final_weights,
    compute_strategy,
    monthly_hold_weights,
    normal_allocation_by_regime,
    performance_metrics,
    stress_allocation_by_regime,
)


OUT = ROOT / "results" / "final_credit_z_unlock_grid"
FIG = OUT / "figures"
MAIN = ROOT / "results" / "main_pipeline_final" / "tables"

WINDOWS = {
    "2008_GFC": ("2007-10-01", "2009-06-30"),
    "2011_EURO_DEBT": ("2011-06-01", "2011-12-31"),
    "2015_2016": ("2015-05-01", "2016-03-31"),
    "2018Q4": ("2018-10-01", "2019-01-31"),
    "COVID_2020": ("2020-02-01", "2020-06-30"),
    "2022_RATE_WAR": ("2021-11-01", "2023-03-31"),
    "2025_PULLBACK": ("2025-01-01", None),
}

CHANGE_WINDOWS = [15, 20, 30]
ABS_THRESHOLDS = [0.05, 0.10, 0.15]
Z_WINDOWS = [126, 252, 504]
UNLOCK_Z_THRESHOLDS = [0.5, 1.0, 1.5]


@dataclass(frozen=True)
class CreditConfig:
    strategy_id: str
    credit_change_window: int
    credit_abs_threshold: float
    credit_level_z_window: int
    unlock_z_threshold: float
    baseline: bool = False


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def load_panel() -> pd.DataFrame:
    path = MAIN / "daily_backtest_panel.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing main pipeline panel: {path}")
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return df


def load_daily_credit() -> pd.DataFrame:
    aaa = pd.read_csv(ROOT / "data" / "raw" / "macro" / "Credit" / "DAAA.csv", parse_dates=["observation_date"])
    baa = pd.read_csv(ROOT / "data" / "raw" / "macro" / "Credit" / "DBAA.csv", parse_dates=["observation_date"])
    aaa = aaa.rename(columns={"observation_date": "date", "DAAA": "DAAA"})[["date", "DAAA"]]
    baa = baa.rename(columns={"observation_date": "date", "DBAA": "DBAA"})[["date", "DBAA"]]
    out = aaa.merge(baa, on="date", how="outer").sort_values("date").drop_duplicates("date")
    out["DAAA"] = pd.to_numeric(out["DAAA"], errors="coerce")
    out["DBAA"] = pd.to_numeric(out["DBAA"], errors="coerce")
    return out


def prepare_panel(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    credit = load_daily_credit()
    out = out.drop(columns=[c for c in ["WAAA", "WBAA", "CREDIT_SPREAD_BAA_AAA", "D_CREDIT_SPREAD_15D", "D_CREDIT_SPREAD_20D"] if c in out.columns])
    out = out.merge(credit, on="date", how="left")
    out["WAAA"] = out["DAAA"]
    out["WBAA"] = out["DBAA"]
    out["CREDIT_SPREAD"] = out["DBAA"] - out["DAAA"]
    out["CREDIT_SPREAD_BAA_AAA"] = out["CREDIT_SPREAD"]
    out["D_CREDIT_SPREAD_15D"] = out["CREDIT_SPREAD"] - out["CREDIT_SPREAD"].shift(15)
    out["D_CREDIT_SPREAD_20D"] = out["CREDIT_SPREAD"] - out["CREDIT_SPREAD"].shift(20)
    out["SPY_MA20"] = out["SPY_MA20"] if "SPY_MA20" in out.columns else out["spy_price"].rolling(20, min_periods=20).mean()
    out["SPY_MA50"] = out["spy_price"].rolling(50, min_periods=50).mean()
    out["SPY_above_MA20"] = out["spy_price"] > out["SPY_MA20"]
    out["SPY_above_MA50"] = out["spy_price"] > out["SPY_MA50"]
    out["SPY_DD"] = out["spy_drawdown_from_previous_high"]
    out["final_main_return"] = out[f"{FINAL_STRATEGY}_return"]
    out["final_main_nav"] = out[f"{FINAL_STRATEGY}_nav"]
    out["final_main_drawdown"] = out[f"{FINAL_STRATEGY}_drawdown"]
    out["final_main_stress"] = out["trigger_lock_full_risk_state"].eq("FULL_RISK")
    out["final_main_credit_active"] = out["trigger_lock_active_locks"].fillna("").astype(str).str.contains("CREDIT")
    out["final_main_vix_active"] = out["trigger_lock_active_locks"].fillna("").astype(str).str.contains("VIX")
    out["final_main_cmdty_active"] = out["trigger_lock_active_locks"].fillna("").astype(str).str.contains("CMDTY")
    return out


def build_credit_features(df: pd.DataFrame, change_window: int, z_window: int) -> pd.DataFrame:
    out = df.copy()
    out["D_CREDIT"] = out["CREDIT_SPREAD"] - out["CREDIT_SPREAD"].shift(change_window)
    roll = out["CREDIT_SPREAD"].rolling(z_window, min_periods=max(63, z_window // 2))
    out["CREDIT_LEVEL_Z"] = (out["CREDIT_SPREAD"] - roll.mean()) / roll.std(ddof=1).replace(0, np.nan)
    return out


def baseline_alignment(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weights, state = build_trigger_lock_final_weights(panel, inv_vol_window=INV_VOL_WINDOW)
    strat = compute_strategy(panel, weights, "BASELINE_REBUILD")
    ret = strat["BASELINE_REBUILD_return"].fillna(0.0)
    ref = panel["final_main_return"].fillna(0.0)
    stress = state["trigger_lock_full_risk_state"].eq("FULL_RISK")
    ref_stress = panel["final_main_stress"].astype(bool)
    check = pd.DataFrame(
        [
            {
                "daily_return_correlation_with_main_pipeline_final": float(ret.corr(ref)),
                "max_abs_daily_return_diff": float((ret - ref).abs().max()),
                "mismatched_stress_days": int((stress != ref_stress).sum()),
                "baseline_CAGR": float(performance_metrics(strat, "BASELINE_REBUILD")["CAGR"]),
                "baseline_Sharpe": float(performance_metrics(strat, "BASELINE_REBUILD")["Sharpe"]),
                "baseline_MaxDD": float(performance_metrics(strat, "BASELINE_REBUILD")["MaxDD"]),
                "baseline_Final_Equity": float(performance_metrics(strat, "BASELINE_REBUILD")["final_equity"]),
            }
        ]
    )
    check.to_csv(OUT / "baseline_alignment_check.csv", index=False)
    corr = float(check.iloc[0]["daily_return_correlation_with_main_pipeline_final"])
    max_diff = float(check.iloc[0]["max_abs_daily_return_diff"])
    mismatch = int(check.iloc[0]["mismatched_stress_days"])
    check["alignment_status"] = np.where(
        (corr < 0.999999999) | (max_diff > 1e-10) | (mismatch != 0),
        "MISMATCH_EXPECTED_AFTER_DAILY_CREDIT_MIGRATION",
        "PASS",
    )
    check.to_csv(OUT / "baseline_alignment_check.csv", index=False)
    return weights, state, strat


def build_configs() -> list[CreditConfig]:
    cfgs = [CreditConfig("BASELINE_ABS", 15, 0.10, 252, np.nan, baseline=True)]
    for cw, ab, zw, uz in product(CHANGE_WINDOWS, ABS_THRESHOLDS, Z_WINDOWS, UNLOCK_Z_THRESHOLDS):
        sid = f"CW{cw}_ABS{int(ab*100):02d}_ZW{zw}_UZ{str(uz).replace('.','P')}"
        cfgs.append(CreditConfig(sid, cw, ab, zw, uz, baseline=False))
    return cfgs


def vix_entry(row: pd.Series) -> bool:
    return bool(row["refined_regime_confirmed"] in {"FLAT_LOW_RATE", "FLAT_HIGH_RATE", "STEEP"} and row["VIX_ZSCORE_120D"] >= 3.0)


def cmdty_entry(row: pd.Series) -> bool:
    return bool(row["refined_regime_confirmed"] == "STEEP" and row["CMDTY_RET60"] < -0.10)


def credit_entry(row: pd.Series, cfg: CreditConfig) -> bool:
    return bool(
        row["refined_regime_confirmed"] in {"FLAT_LOW_RATE", "FLAT_HIGH_RATE"}
        and row["SPY_DD"] <= -0.05
        and pd.notna(row["D_CREDIT"])
        and row["D_CREDIT"] > cfg.credit_abs_threshold
    )


def vix_unlock(row: pd.Series) -> bool:
    return bool((row["VIX_ZSCORE_120D"] < 1.5) and row["SPY_above_MA20"])


def cmdty_unlock(row: pd.Series) -> bool:
    return bool((row["CMDTY_RET60"] > -0.05) and row["SPY_above_MA20"])


def credit_unlock(row: pd.Series, cfg: CreditConfig) -> bool:
    base = bool(pd.notna(row["D_CREDIT"]) and row["D_CREDIT"] < 0 and row["SPY_above_MA20"])
    if cfg.baseline:
        return base
    return bool(base and pd.notna(row["CREDIT_LEVEL_Z"]) and row["CREDIT_LEVEL_Z"] < cfg.unlock_z_threshold)


def build_variant_weights(panel: pd.DataFrame, cfg: CreditConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = build_credit_features(panel, cfg.credit_change_window, cfg.credit_level_z_window)
    flat_low_normal = monthly_hold_weights(df, ["SPY", "CMDTY_FUT"], window=INV_VOL_WINDOW)
    flat_high_normal = monthly_hold_weights(df, ["GOLD", "CMDTY_FUT"], window=INV_VOL_WINDOW)
    steep_high_normal = monthly_hold_weights(df, ["SPY", "CMDTY_FUT"], window=INV_VOL_WINDOW)
    inverted_normal = monthly_hold_weights(df, ["SPY", "GOLD"], window=INV_VOL_WINDOW)
    weights = pd.DataFrame(0.0, index=df.index, columns=ASSETS)

    current_full_risk = False
    current_locks: set[str] = set()
    pending_full_risk = False
    pending_locks: set[str] = set()
    rows: list[dict[str, object]] = []

    for i, row in df.iterrows():
        current_full_risk = pending_full_risk
        current_locks = set(pending_locks)
        refined = row["refined_regime_confirmed"]

        lock_added_today: set[str] = set()
        lock_unlocked_today: set[str] = set()
        entry_signal = False
        exit_signal = False

        if current_full_risk:
            if refined == "INVERTED":
                w, allocation_state = normal_allocation_by_regime(
                    i,
                    refined,
                    row["steep_rate_regime_confirmed"],
                    flat_low_normal,
                    flat_high_normal,
                    steep_high_normal,
                    inverted_normal,
                )
            else:
                w, allocation_state = stress_allocation_by_regime(refined)

            new_locks: set[str] = set()
            if "VIX" not in current_locks and vix_entry(row):
                new_locks.add("VIX")
            if "CMDTY" not in current_locks and cmdty_entry(row):
                new_locks.add("CMDTY")
            if "CREDIT" not in current_locks and credit_entry(row, cfg):
                new_locks.add("CREDIT")

            current_locks |= new_locks
            lock_added_today = set(new_locks)

            unlocks: set[str] = set()
            if "VIX" in current_locks and vix_unlock(row):
                unlocks.add("VIX")
                if "CREDIT" in current_locks:
                    unlocks.add("CREDIT")
            if "CREDIT" in current_locks and credit_unlock(row, cfg):
                unlocks.add("CREDIT")
            if "CMDTY" in current_locks and cmdty_unlock(row):
                unlocks.add("CMDTY")

            current_locks -= unlocks
            lock_unlocked_today = set(unlocks)
            if not current_locks:
                exit_signal = True
                pending_full_risk = False
                pending_locks = set()
            else:
                pending_full_risk = True
                pending_locks = set(current_locks)
        else:
            w, allocation_state = normal_allocation_by_regime(
                i,
                refined,
                row["steep_rate_regime_confirmed"],
                flat_low_normal,
                flat_high_normal,
                steep_high_normal,
                inverted_normal,
            )
            entry_locks: set[str] = set()
            if vix_entry(row):
                entry_locks.add("VIX")
            if cmdty_entry(row):
                entry_locks.add("CMDTY")
            if credit_entry(row, cfg):
                entry_locks.add("CREDIT")
            if entry_locks:
                entry_signal = True
                pending_full_risk = True
                pending_locks = set(entry_locks)
                lock_added_today = set(entry_locks)
            else:
                pending_full_risk = False
                pending_locks = set()

        weights.loc[i, ASSETS] = pd.Series(w)
        rows.append(
            {
                "date": row["date"],
                "strategy_id": cfg.strategy_id,
                "trigger_lock_full_risk_state": "FULL_RISK" if current_full_risk else "NON_RISK",
                "trigger_lock_active_locks": "+".join(sorted(current_locks)),
                "trigger_lock_locks_added_today": "+".join(sorted(lock_added_today)),
                "trigger_lock_locks_unlocked_today": "+".join(sorted(lock_unlocked_today)),
                "trigger_lock_entry_signal": entry_signal,
                "trigger_lock_exit_signal": exit_signal,
                "final_allocation_state": allocation_state,
                "credit_change_window": cfg.credit_change_window,
                "credit_abs_threshold": cfg.credit_abs_threshold,
                "credit_level_z_window": cfg.credit_level_z_window,
                "unlock_z_threshold": cfg.unlock_z_threshold,
                "D_CREDIT": row["D_CREDIT"],
                "CREDIT_LEVEL_Z": row["CREDIT_LEVEL_Z"],
            }
        )

    return weights, pd.DataFrame(rows, index=df.index)


def strategy_frame(panel: pd.DataFrame, weights: pd.DataFrame, state: pd.DataFrame, strategy_id: str) -> pd.DataFrame:
    strat = compute_strategy(panel, weights, strategy_id)
    out = pd.concat([panel[["date"]], weights.add_prefix("weight_"), strat, state.drop(columns=["date"])], axis=1)
    out["stress_active"] = out["trigger_lock_full_risk_state"].eq("FULL_RISK")
    out["credit_active"] = out["trigger_lock_active_locks"].fillna("").astype(str).str.contains("CREDIT")
    out["vix_active"] = out["trigger_lock_active_locks"].fillna("").astype(str).str.contains("VIX")
    out["cmdty_active"] = out["trigger_lock_active_locks"].fillna("").astype(str).str.contains("CMDTY")
    return out


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


def episode_diagnostics(panel: pd.DataFrame, df: pd.DataFrame, cfg: CreditConfig) -> pd.DataFrame:
    r21 = forward_return(panel["SPY_return"], 21)
    m21 = forward_mdd(panel["SPY_return"], 21)
    r63 = forward_return(panel["SPY_return"], 63)
    m63 = forward_mdd(panel["SPY_return"], 63)
    rows = []
    for ep_id, (s, e) in enumerate(find_episodes(df["credit_active"].astype(bool)), start=1):
        unlock_idx = min(e + 1, len(panel) - 1)
        dom_regime = panel.loc[s:e, "final_regime_confirmed"].mode().iloc[0]
        trough = float(panel.loc[s:e, "spy_price"].min())
        unlock_price = float(panel.loc[unlock_idx, "spy_price"])
        trough_to_unlock = unlock_price / trough - 1.0 if trough > 0 else np.nan
        false_recovery = bool(
            (pd.notna(m21.iloc[unlock_idx]) and m21.iloc[unlock_idx] <= -0.05)
            or (pd.notna(m63.iloc[unlock_idx]) and m63.iloc[unlock_idx] <= -0.08)
            or df.loc[unlock_idx + 1 : min(unlock_idx + 63, len(df) - 1), "stress_active"].astype(bool).any()
        )
        relock21 = bool(df.loc[unlock_idx + 1 : min(unlock_idx + 21, len(df) - 1), "credit_active"].astype(bool).any())
        relock63 = bool(df.loc[unlock_idx + 1 : min(unlock_idx + 63, len(df) - 1), "credit_active"].astype(bool).any())
        rows.append(
            {
                "strategy_id": cfg.strategy_id,
                "episode_id": ep_id,
                "entry_date": panel.loc[s, "date"],
                "unlock_date": panel.loc[unlock_idx, "date"],
                "duration_days": int(e - s + 1),
                "macro_regime_at_entry": panel.loc[s, "macro_regime_confirmed"],
                "dominant_regime": dom_regime,
                "credit_change_window": cfg.credit_change_window,
                "credit_abs_threshold": cfg.credit_abs_threshold,
                "credit_level_z_window": cfg.credit_level_z_window,
                "unlock_z_threshold": cfg.unlock_z_threshold,
                "entry_SPY_DD": panel.loc[s, "SPY_DD"],
                "entry_CREDIT_SPREAD": panel.loc[s, "CREDIT_SPREAD"],
                "entry_D_CREDIT": df.loc[s, "D_CREDIT"],
                "entry_CREDIT_LEVEL_Z": df.loc[s, "CREDIT_LEVEL_Z"],
                "unlock_CREDIT_SPREAD": panel.loc[unlock_idx, "CREDIT_SPREAD"],
                "unlock_D_CREDIT": df.loc[unlock_idx, "D_CREDIT"],
                "unlock_CREDIT_LEVEL_Z": df.loc[unlock_idx, "CREDIT_LEVEL_Z"],
                "unlock_SPY_vs_MA20": panel.loc[unlock_idx, "spy_price"] / panel.loc[unlock_idx, "SPY_MA20"] - 1.0 if pd.notna(panel.loc[unlock_idx, "SPY_MA20"]) else np.nan,
                "unlock_SPY_vs_MA50": panel.loc[unlock_idx, "spy_price"] / panel.loc[unlock_idx, "SPY_MA50"] - 1.0 if pd.notna(panel.loc[unlock_idx, "SPY_MA50"]) else np.nan,
                "SPY_return_during_lock": period_return(panel.loc[s:e, "SPY_return"]),
                "SPY_maxDD_during_lock": period_mdd(panel.loc[s:e, "SPY_return"]),
                "final_strategy_return_during_lock": period_return(df.loc[s:e, f"{cfg.strategy_id}_return"]),
                "next_21d_SPY_return_after_unlock": r21.iloc[unlock_idx],
                "next_21d_SPY_maxDD_after_unlock": m21.iloc[unlock_idx],
                "next_63d_SPY_return_after_unlock": r63.iloc[unlock_idx],
                "next_63d_SPY_maxDD_after_unlock": m63.iloc[unlock_idx],
                "false_recovery_flag": false_recovery,
                "missed_rebound_flag": bool(pd.notna(trough_to_unlock) and trough_to_unlock > 0.08),
                "relock_within_21d": relock21,
                "relock_within_63d": relock63,
            }
        )
    return pd.DataFrame(rows)


def performance_row(cfg: CreditConfig, df: pd.DataFrame, ep: pd.DataFrame) -> dict[str, object]:
    p = performance_metrics(df, cfg.strategy_id)
    return {
        "strategy_id": cfg.strategy_id,
        "credit_change_window": cfg.credit_change_window,
        "credit_abs_threshold": cfg.credit_abs_threshold,
        "credit_level_z_window": cfg.credit_level_z_window,
        "unlock_z_threshold": cfg.unlock_z_threshold,
        "CAGR": p["CAGR"],
        "Sharpe": p["Sharpe"],
        "Sortino": p["Sortino"],
        "MaxDD": p["MaxDD"],
        "Calmar": p["Calmar"],
        "Final_Equity": p["final_equity"],
        "annualized_vol": p["annualized_volatility"],
        "turnover": p["turnover"],
        "transaction_cost_drag": p["transaction_cost"],
        "time_in_credit_lock": int(df["credit_active"].sum()),
        "number_credit_entries": int(df["trigger_lock_locks_added_today"].astype(str).str.contains("CREDIT").sum()),
        "number_credit_unlocks": int(df["trigger_lock_locks_unlocked_today"].astype(str).str.contains("CREDIT").sum()),
        "avg_credit_lock_duration": float(ep["duration_days"].mean()) if len(ep) else np.nan,
        "false_recovery_count": int(ep["false_recovery_flag"].sum()) if len(ep) else 0,
        "missed_rebound_count": int(ep["missed_rebound_flag"].sum()) if len(ep) else 0,
        "relock_within_21d_count": int(ep["relock_within_21d"].sum()) if len(ep) else 0,
        "relock_within_63d_count": int(ep["relock_within_63d"].sum()) if len(ep) else 0,
        "avg_weight_SPY": float(df["weight_SPY"].mean()),
        "avg_weight_GOLD": float(df["weight_GOLD"].mean()),
        "avg_weight_IEF": float(df["weight_IEF"].mean()),
        "avg_weight_CASH": float(df["weight_CASH"].mean()),
        "avg_weight_CMDTY_FUT": float(df["weight_CMDTY_FUT"].mean()),
    }


def crisis_row(cfg: CreditConfig, df: pd.DataFrame, ep: pd.DataFrame, window: str, start: str, end: str | None) -> dict[str, object]:
    mask = df["date"] >= pd.Timestamp(start)
    if end is not None:
        mask &= df["date"] <= pd.Timestamp(end)
    sub = df.loc[mask]
    ret = sub[f"{cfg.strategy_id}_return"]
    nav = (1.0 + ret.fillna(0.0)).cumprod()
    ann_vol = ret.std(ddof=1) * np.sqrt(252.0)
    ann_ret = nav.iloc[-1] ** (252.0 / len(sub)) - 1.0 if len(sub) else np.nan
    ep_sub = ep.loc[ep["entry_date"].between(pd.Timestamp(start), pd.Timestamp(end) if end is not None else df["date"].max())] if len(ep) else pd.DataFrame()
    return {
        "strategy_id": cfg.strategy_id,
        "window": window,
        "cumulative_return": float(nav.iloc[-1] - 1.0) if len(sub) else np.nan,
        "max_drawdown": float((nav / nav.cummax() - 1.0).min()) if len(sub) else np.nan,
        "Sharpe": float(ann_ret / ann_vol) if len(sub) and ann_vol > 0 else np.nan,
        "time_in_credit_lock": int(sub["credit_active"].sum()) if len(sub) else 0,
        "number_credit_entries": int(sub["trigger_lock_locks_added_today"].astype(str).str.contains("CREDIT").sum()) if len(sub) else 0,
        "number_credit_unlocks": int(sub["trigger_lock_locks_unlocked_today"].astype(str).str.contains("CREDIT").sum()) if len(sub) else 0,
        "false_recovery_count": int(ep_sub["false_recovery_flag"].sum()) if len(ep_sub) else 0,
        "missed_rebound_count": int(ep_sub["missed_rebound_flag"].sum()) if len(ep_sub) else 0,
        "avg_weight_SPY": float(sub["weight_SPY"].mean()) if len(sub) else np.nan,
        "avg_weight_GOLD": float(sub["weight_GOLD"].mean()) if len(sub) else np.nan,
        "avg_weight_IEF": float(sub["weight_IEF"].mean()) if len(sub) else np.nan,
        "avg_weight_CASH": float(sub["weight_CASH"].mean()) if len(sub) else np.nan,
        "avg_weight_CMDTY_FUT": float(sub["weight_CMDTY_FUT"].mean()) if len(sub) else np.nan,
    }


def add_rankings(grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = grid.copy()
    df["rank_sharpe"] = df["Sharpe"].rank(ascending=False, method="min")
    df["rank_calmar"] = df["Calmar"].rank(ascending=False, method="min")
    df["rank_maxdd"] = df["MaxDD"].rank(ascending=False, method="min")
    df["rank_final_equity"] = df["Final_Equity"].rank(ascending=False, method="min")
    df["rank_false_recovery"] = df["false_recovery_count"].rank(ascending=True, method="min")
    df["rank_missed_rebound"] = df["missed_rebound_count"].rank(ascending=True, method="min")
    df["composite_score"] = (
        0.30 * df["rank_sharpe"].rank(pct=True, ascending=True)
        + 0.25 * df["rank_maxdd"].rank(pct=True, ascending=True)
        + 0.20 * df["rank_final_equity"].rank(pct=True, ascending=True)
        + 0.15 * df["rank_false_recovery"].rank(pct=True, ascending=True)
        + 0.10 * df["rank_missed_rebound"].rank(pct=True, ascending=True)
    )
    df["rank_composite"] = df["composite_score"].rank(ascending=True, method="min")
    baseline = df.loc[df["strategy_id"] == "BASELINE_ABS"].iloc[0]
    df["materially_better"] = (
        (df["Sharpe"] >= baseline["Sharpe"] + 0.03)
        & (df["MaxDD"] >= baseline["MaxDD"] + 0.01)
        & (df["Final_Equity"] >= baseline["Final_Equity"] * 0.98)
        & (df["false_recovery_count"] <= baseline["false_recovery_count"])
        & (df["missed_rebound_count"] <= baseline["missed_rebound_count"] + 2)
    )
    rankings = []
    for label, col, asc in [
        ("top_sharpe", "Sharpe", False),
        ("top_calmar", "Calmar", False),
        ("lowest_maxdd", "MaxDD", False),
        ("highest_final_equity", "Final_Equity", False),
        ("balanced_composite", "composite_score", True),
    ]:
        top = df.sort_values(col, ascending=asc).reset_index(drop=True).head(10).copy()
        top.insert(0, "ranking", label)
        rankings.append(top)
    rankings_df = pd.concat(rankings, ignore_index=True)
    better = df.loc[df["materially_better"]].sort_values(["rank_composite", "Sharpe"], ascending=[True, False])
    if len(better) == 0:
        better = pd.DataFrame([{"message": "No materially better candidate found."}])
    return df, rankings_df, better


def sensitivity_summary(grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for param in ["credit_change_window", "credit_abs_threshold", "credit_level_z_window", "unlock_z_threshold"]:
        grouped = grid.groupby(param).agg(
            mean_Sharpe=("Sharpe", "mean"),
            median_Sharpe=("Sharpe", "median"),
            mean_MaxDD=("MaxDD", "mean"),
            mean_Final_Equity=("Final_Equity", "mean"),
            mean_false_recovery_count=("false_recovery_count", "mean"),
            mean_missed_rebound_count=("missed_rebound_count", "mean"),
        ).reset_index()
        grouped.insert(0, "parameter", param)
        grouped.rename(columns={param: "parameter_value"}, inplace=True)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def heatmap_plot(grid: pd.DataFrame, value: str, filename: str) -> None:
    for z_window in Z_WINDOWS:
        sub = grid.loc[(grid["credit_level_z_window"] == z_window) & (grid["unlock_z_threshold"] == 1.0)]
        if sub.empty:
            continue
        heat = sub.pivot(index="credit_change_window", columns="credit_abs_threshold", values=value)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(heat, annot=True, fmt=".3f" if value not in {"false_recovery_count", "missed_rebound_count"} else ".0f", cmap="RdYlGn", ax=ax)
        ax.set_title(f"{value} | z_window={z_window} | unlock_z=1.0")
        fig.tight_layout()
        stem = Path(filename).stem
        fig.savefig(FIG / f"{stem}_zw{z_window}.png", dpi=160)
        plt.close(fig)


def plot_top_curves(frames: dict[str, pd.DataFrame], rankings: pd.DataFrame, materially_better: pd.DataFrame) -> tuple[str, str, str | None]:
    top_sharpe = str(rankings.loc[rankings["ranking"] == "top_sharpe"].iloc[0]["strategy_id"])
    top_comp = str(rankings.loc[rankings["ranking"] == "balanced_composite"].iloc[0]["strategy_id"])
    best_mat = None
    if "strategy_id" in materially_better.columns and len(materially_better):
        best_mat = str(materially_better.iloc[0]["strategy_id"])
    fig, ax = plt.subplots(figsize=(13, 6))
    names = ["BASELINE_ABS", top_sharpe, top_comp] + ([best_mat] if best_mat else [])
    seen = []
    for name in names:
        if name and name not in seen:
            seen.append(name)
            ax.plot(frames[name]["date"], frames[name][f"{name}_nav"], label=name, linewidth=1.0)
    ax.set_yscale("log")
    ax.set_title("Top Candidates Equity Curve")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "top_candidates_equity_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6))
    for name in seen:
        ax.plot(frames[name]["date"], frames[name][f"{name}_drawdown"], label=name, linewidth=1.0)
    ax.set_title("Top Candidates Drawdown")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "top_candidates_drawdown_curve.png", dpi=160)
    plt.close(fig)
    return top_sharpe, top_comp, best_mat


def plot_crisis_heatmap(crisis: pd.DataFrame, names: list[str]) -> None:
    sub = crisis.loc[crisis["strategy_id"].isin(names)]
    heat = sub.pivot(index="strategy_id", columns="window", values="cumulative_return")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(heat, annot=True, fmt=".1%", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title("Crisis Window Heatmap | Top Candidates")
    fig.tight_layout()
    fig.savefig(FIG / "crisis_window_heatmap_top_candidates.png", dpi=160)
    plt.close(fig)


def plot_param_sensitivity(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = ["mean_Sharpe", "mean_MaxDD", "mean_Final_Equity", "mean_false_recovery_count"]
    for ax, metric in zip(axes.flatten(), metrics):
        sub = summary.loc[summary["parameter"].isin(["credit_change_window", "credit_abs_threshold", "credit_level_z_window", "unlock_z_threshold"])]
        for param, grp in sub.groupby("parameter"):
            ax.plot(grp["parameter_value"].astype(str), grp[metric], marker="o", label=param)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=45)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "parameter_sensitivity_bar.png", dpi=160)
    plt.close(fig)


def plot_case(panel: pd.DataFrame, frames: dict[str, pd.DataFrame], names: list[str], tag: str, start: str, end: str | None) -> None:
    mask = panel["date"] >= pd.Timestamp(start)
    if end is not None:
        mask &= panel["date"] <= pd.Timestamp(end)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(panel.loc[mask, "date"], panel.loc[mask, "spy_price"], color="black", label="SPY")
    axes[0].plot(panel.loc[mask, "date"], panel.loc[mask, "CREDIT_SPREAD"], color="firebrick", label="Credit Spread")
    axes[0].legend(frameon=False)
    axes[0].set_title(f"{tag} | SPY and Credit Spread")
    for name in names:
        axes[1].plot(frames[name].loc[mask, "date"], frames[name].loc[mask, f"{name}_nav"], label=name)
    axes[1].legend(frameon=False)
    axes[1].set_title(f"{tag} | Strategy NAV")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / f"case_{tag}_top_candidates.png", dpi=160)
    plt.close(fig)


def build_report(alignment: pd.DataFrame, rankings: pd.DataFrame, materially_better: pd.DataFrame, summary: pd.DataFrame) -> None:
    top_sharpe = rankings.loc[rankings["ranking"] == "top_sharpe"].iloc[0]
    top_comp = rankings.loc[rankings["ranking"] == "balanced_composite"].iloc[0]
    has_better = "strategy_id" in materially_better.columns
    lines = [
        "# FINAL_CREDIT_Z_UNLOCK_GRID_REPORT",
        "",
        "## 1. Purpose",
        "",
        "This grid search migrates the most interpretable SPY/CASH credit-lab idea back into the full final strategy: absolute price-confirmed credit widening on entry, plus credit level z-score confirmation on unlock.",
        "",
        "## 2. Why this rule",
        "",
        "Entry uses SPY drawdown plus absolute credit widening. Unlock requires both short-term credit improvement and credit spread level normalization, to reduce premature exits while spreads remain structurally elevated.",
        "",
        "## 3. Grid design",
        "",
        "Only credit-related parameters move. VIX, commodity, allocation, regime framework, transaction cost, and execution timing are unchanged.",
        "",
        "## 4. Baseline alignment",
        "",
        alignment.to_markdown(index=False),
        "",
        "## 5. Full-sample results",
        "",
        f"- Top Sharpe candidate: `{top_sharpe['strategy_id']}`",
        f"- Top composite candidate: `{top_comp['strategy_id']}`",
        "",
        "## 6. Crisis window results",
        "",
        "See `grid_crisis_window_comparison.csv` and the case-study plots for 2008, 2022, COVID, and 2025.",
        "",
        "## 7. False recovery / missed rebound trade-off",
        "",
        "The key question is whether stricter z-score unlock reduces false recovery without paying too much in missed rebound.",
        "",
        "## 8. Parameter sensitivity",
        "",
        "See `parameter_sensitivity_summary.csv`. If only isolated parameter points work and neighbors do not, the result should be treated as unstable.",
        "",
        "## 9. Materially better candidate test",
        "",
        ("At least one materially better candidate exists." if has_better else "No materially better candidate found."),
        "",
        "## 10. Recommendation",
        "",
        ("A specific z-score unlock challenger is strong enough for the next final-strategy test." if has_better else "Keep baseline credit logic. Credit level z-score unlock remains a future research direction rather than a mainline replacement."),
        "",
        "## 11. Limitations",
        "",
        "- credit episodes are sparse",
        "- z-score windows may still overfit in-sample",
        "- this is not out-of-sample validation",
    ]
    (OUT / "FINAL_CREDIT_Z_UNLOCK_GRID_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    patch = (
        f"The SPY/CASH credit lab suggests that {materially_better.iloc[0]['strategy_id']} improves credit timing before considering hedge allocation. It should be tested as a final strategy challenger."
        if has_better
        else "We migrated the most promising SPY/CASH credit-lab variant back to the full final strategy and tested a controlled grid over credit change windows, absolute widening thresholds, credit level z-score windows, and unlock z-score thresholds. No candidate robustly dominated the simpler baseline across Sharpe, MaxDD, final equity, false recovery, and missed rebound. Therefore, the final strategy keeps the baseline credit lock, while credit level z-score unlock remains a future research direction."
    )
    (OUT / "README_PATCH_SUGGESTION.md").write_text(patch, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = prepare_panel(load_panel())
    _, _, _ = baseline_alignment(panel)

    frames: dict[str, pd.DataFrame] = {}
    perf_rows = []
    crisis_rows = []
    episode_rows = []

    for cfg in build_configs():
        weights, state = build_variant_weights(panel, cfg)
        df = strategy_frame(panel, weights, state, cfg.strategy_id)
        ep = episode_diagnostics(panel, df, cfg)
        frames[cfg.strategy_id] = df
        perf_rows.append(performance_row(cfg, df, ep))
        episode_rows.append(ep)
        for window, (start, end) in WINDOWS.items():
            crisis_rows.append(crisis_row(cfg, df, ep, window, start, end))

    grid = pd.DataFrame(perf_rows)
    crisis = pd.DataFrame(crisis_rows)
    episodes = pd.concat(episode_rows, ignore_index=True) if episode_rows else pd.DataFrame()
    ranked, rankings, materially_better = add_rankings(grid)
    sensitivity = sensitivity_summary(ranked.loc[ranked["strategy_id"] != "BASELINE_ABS"])

    ranked.to_csv(OUT / "grid_performance.csv", index=False)
    crisis.to_csv(OUT / "grid_crisis_window_comparison.csv", index=False)
    episodes.to_csv(OUT / "grid_credit_episode_diagnostics.csv", index=False)
    rankings.to_csv(OUT / "grid_rankings.csv", index=False)
    materially_better.to_csv(OUT / "materially_better_candidates.csv", index=False)
    sensitivity.to_csv(OUT / "parameter_sensitivity_summary.csv", index=False)

    heatmap_plot(ranked.loc[ranked["strategy_id"] != "BASELINE_ABS"], "Sharpe", "grid_sharpe_heatmap_by_window_threshold.png")
    heatmap_plot(ranked.loc[ranked["strategy_id"] != "BASELINE_ABS"], "MaxDD", "grid_maxdd_heatmap_by_window_threshold.png")
    heatmap_plot(ranked.loc[ranked["strategy_id"] != "BASELINE_ABS"], "Final_Equity", "grid_final_equity_heatmap.png")
    heatmap_plot(ranked.loc[ranked["strategy_id"] != "BASELINE_ABS"], "false_recovery_count", "grid_false_recovery_heatmap.png")
    heatmap_plot(ranked.loc[ranked["strategy_id"] != "BASELINE_ABS"], "missed_rebound_count", "grid_missed_rebound_heatmap.png")
    top_sharpe, top_comp, best_mat = plot_top_curves(frames, rankings, materially_better)
    top_names = ["BASELINE_ABS", top_sharpe, top_comp] + ([best_mat] if best_mat else [])
    top_names = [x for i, x in enumerate(top_names) if x and x not in top_names[:i]]
    plot_crisis_heatmap(crisis, top_names)
    plot_param_sensitivity(sensitivity)
    plot_case(panel, frames, top_names, "2008", *WINDOWS["2008_GFC"])
    plot_case(panel, frames, top_names, "2022", *WINDOWS["2022_RATE_WAR"])
    plot_case(panel, frames, top_names, "COVID", *WINDOWS["COVID_2020"])
    plot_case(panel, frames, top_names, "2025", *WINDOWS["2025_PULLBACK"])

    alignment = pd.read_csv(OUT / "baseline_alignment_check.csv")
    build_report(alignment, rankings, materially_better, sensitivity)

    baseline = ranked.loc[ranked["strategy_id"] == "BASELINE_ABS"].iloc[0]
    top_sharpe_row = ranked.sort_values("Sharpe", ascending=False).iloc[0]
    top_mdd_row = ranked.sort_values("MaxDD", ascending=False).iloc[0]
    top_eq_row = ranked.sort_values("Final_Equity", ascending=False).iloc[0]
    top_comp_row = ranked.sort_values("rank_composite", ascending=True).iloc[0]
    has_better = "strategy_id" in materially_better.columns

    print("baseline performance and alignment check")
    print(alignment.to_string(index=False))
    print("number of grid combinations")
    print(len(build_configs()) - 1)
    print("top Sharpe candidate")
    print(top_sharpe_row[["strategy_id", "Sharpe", "MaxDD", "Final_Equity"]].to_string())
    print("top MaxDD candidate")
    print(top_mdd_row[["strategy_id", "Sharpe", "MaxDD", "Final_Equity"]].to_string())
    print("top Final Equity candidate")
    print(top_eq_row[["strategy_id", "Sharpe", "MaxDD", "Final_Equity"]].to_string())
    print("top composite candidate")
    print(top_comp_row[["strategy_id", "Sharpe", "MaxDD", "Final_Equity", "false_recovery_count", "missed_rebound_count"]].to_string())
    print("number of materially better candidates")
    print(0 if not has_better else len(materially_better))
    print("best materially better candidate if any")
    print("NONE" if not has_better else materially_better.iloc[0].to_string())
    print("2008 / 2022 / COVID / 2025 summary for baseline vs top composite")
    print(
        crisis.loc[
            crisis["strategy_id"].isin(["BASELINE_ABS", top_comp_row["strategy_id"]])
            & crisis["window"].isin(["2008_GFC", "2022_RATE_WAR", "COVID_2020", "2025_PULLBACK"]),
            ["strategy_id", "window", "cumulative_return", "max_drawdown", "Sharpe", "false_recovery_count", "missed_rebound_count"],
        ].to_string(index=False)
    )
    print("recommendation: keep baseline or update credit rule")
    print("UPDATE CREDIT RULE" if has_better else "KEEP BASELINE")
    print("output paths")
    print(str(OUT))


if __name__ == "__main__":
    main()
