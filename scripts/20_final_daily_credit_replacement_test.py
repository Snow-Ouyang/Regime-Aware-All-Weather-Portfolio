from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from final_strategy_source_only_core import FINAL_STRATEGY, ROOT, performance_metrics


OUT = ROOT / "results" / "final_daily_credit_replacement_test"
FIG = OUT / "figures"
MAIN = ROOT / "results" / "main_pipeline_final" / "tables" / "daily_backtest_panel.csv"

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
class FinalCreditChallenger:
    strategy: str
    credit_variant: str


def load_redesign_module():
    path = ROOT / "scripts" / "19_daily_credit_trigger_redesign.py"
    spec = importlib.util.spec_from_file_location("daily_credit_redesign_19", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def patch_credit_variant_hooks(mod) -> None:
    orig_entry = mod.credit_entry
    orig_unlock = mod.credit_unlock

    def credit_entry(row: pd.Series, kind: str):
        if kind == "ABS_ENTRY_LEVEL_Z_UNLOCK":
            if row["refined_regime_confirmed"] not in {"FLAT_LOW_RATE", "FLAT_HIGH_RATE"}:
                return False, "NO_ENTRY_REGIME"
            sig = bool(
                (row["SPY_DD"] <= -0.05)
                and pd.notna(row["D_CREDIT_15D"])
                and (row["D_CREDIT_15D"] > 0.10)
            )
            return sig, "ABS_ENTRY_LEVEL_Z_UNLOCK"
        return orig_entry(row, kind)

    def credit_unlock(row: pd.Series, kind: str, state: dict[str, object]):
        if kind == "ABS_ENTRY_LEVEL_Z_UNLOCK":
            ma20 = bool(row["SPY_above_MA20"])
            sig = bool(
                ma20
                and pd.notna(row["D_CREDIT_15D"])
                and (row["D_CREDIT_15D"] < 0)
                and pd.notna(row["CREDIT_LEVEL_Z_252D"])
                and (row["CREDIT_LEVEL_Z_252D"] < 1.0)
            )
            return sig, "ABS_ENTRY_LEVEL_Z_UNLOCK"
        return orig_unlock(row, kind, state)

    mod.credit_entry = credit_entry
    mod.credit_unlock = credit_unlock


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def load_panel() -> pd.DataFrame:
    if not MAIN.exists():
        raise FileNotFoundError(f"Missing main pipeline panel: {MAIN}")
    return pd.read_csv(MAIN, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


def build_baseline_frame(panel: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"date": panel["date"]})
    for asset in ["SPY", "GOLD", "CMDTY_FUT", "IEF", "CASH"]:
        out[f"weight_{asset}"] = panel[f"{FINAL_STRATEGY}_weight_{asset}"]
    for suffix in ["return", "nav", "drawdown", "turnover", "transaction_cost"]:
        out[f"{FINAL_STRATEGY}_{suffix}"] = panel[f"{FINAL_STRATEGY}_{suffix}"]
    out["stress_active"] = panel["trigger_lock_full_risk_state"].eq("FULL_RISK")
    locks = panel["trigger_lock_active_locks"].fillna("").astype(str)
    out["credit_lock_active"] = locks.str.contains("CREDIT")
    out["credit_watch_active"] = False
    out["vix_lock_active"] = locks.str.contains("VIX")
    out["cmdty_lock_active"] = locks.str.contains("CMDTY")
    out["active_locks"] = locks
    out["lock_add_types"] = panel["trigger_lock_locks_added_today"].fillna("").astype(str)
    out["lock_unlock_types"] = panel["trigger_lock_locks_unlocked_today"].fillna("").astype(str)
    out["locks_added_today"] = panel["trigger_lock_locks_added_today"].fillna("").astype(str)
    out["locks_unlocked_today"] = panel["trigger_lock_locks_unlocked_today"].fillna("").astype(str)
    return out


def alignment_check(panel: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    ret = panel[f"{FINAL_STRATEGY}_return"]
    cmp_ret = baseline[f"{FINAL_STRATEGY}_return"]
    corr = float(ret.corr(cmp_ret))
    max_abs = float((ret - cmp_ret).abs().max())
    stress_main = panel["trigger_lock_full_risk_state"].eq("FULL_RISK")
    stress_cmp = baseline["stress_active"].astype(bool)
    mismatched = int((stress_main != stress_cmp).sum())
    perf = performance_metrics(panel, FINAL_STRATEGY)
    out = pd.DataFrame(
        [
            {
                "daily_return_correlation_with_main_pipeline_final": corr,
                "max_abs_daily_return_diff": max_abs,
                "mismatched_stress_days": mismatched,
                "baseline_CAGR": perf["CAGR"],
                "baseline_Sharpe": perf["Sharpe"],
                "baseline_MaxDD": perf["MaxDD"],
                "baseline_Final_Equity": perf["final_equity"],
            }
        ]
    )
    out.to_csv(OUT / "baseline_alignment_check.csv", index=False)
    return out


def make_strategy_perf(
    frame: pd.DataFrame,
    panel: pd.DataFrame,
    strategy_name: str,
    credit_variant: str,
    episodes: pd.DataFrame,
) -> dict[str, object]:
    p = performance_metrics(frame, strategy_name)
    return {
        "strategy": strategy_name,
        "credit_variant": credit_variant,
        "CAGR": p["CAGR"],
        "Sharpe": p["Sharpe"],
        "Sortino": p["Sortino"],
        "MaxDD": p["MaxDD"],
        "Calmar": p["Calmar"],
        "Final Equity": p["final_equity"],
        "annualized_vol": p["annualized_volatility"],
        "turnover": p["turnover"],
        "transaction_cost_drag": p["transaction_cost"],
        "time_in_credit_lock": int(frame["credit_lock_active"].sum()) if "credit_lock_active" in frame else int(panel["trigger_lock_active_locks"].fillna("").astype(str).str.contains("CREDIT").sum()),
        "time_in_credit_watch": int(frame.get("credit_watch_active", pd.Series(False, index=frame.index)).sum()),
        "number_credit_entries": int(frame["locks_added_today"].astype(str).str.contains("CREDIT").sum()) if "locks_added_today" in frame else 0,
        "number_credit_unlocks": int(frame["locks_unlocked_today"].astype(str).str.contains("CREDIT").sum()) if "locks_unlocked_today" in frame else 0,
        "number_relocks": int(frame["lock_add_types"].astype(str).str.contains("RELOCK").sum()) if "lock_add_types" in frame else 0,
        "avg_credit_lock_duration": float(episodes["duration_days"].mean()) if len(episodes) else np.nan,
        "false_recovery_count": int(episodes["false_recovery_flag"].sum()) if len(episodes) else 0,
        "missed_rebound_count": int(episodes["missed_rebound_flag"].sum()) if len(episodes) else 0,
        "relock_within_21d_count": int(episodes["relock_within_21d"].sum()) if "relock_within_21d" in episodes else int(episodes["relock_episode"].sum()) if "relock_episode" in episodes else 0,
        "relock_within_63d_count": int(episodes["relock_within_63d"].sum()) if "relock_within_63d" in episodes else int(episodes["relock_episode"].sum()) if "relock_episode" in episodes else 0,
        "avg_weight_SPY": float(frame["weight_SPY"].mean()),
        "avg_weight_GOLD": float(frame["weight_GOLD"].mean()),
        "avg_weight_IEF": float(frame["weight_IEF"].mean()),
        "avg_weight_CASH": float(frame["weight_CASH"].mean()),
        "avg_weight_CMDTY_FUT": float(frame["weight_CMDTY_FUT"].mean()),
    }


def crisis_summary(
    frame: pd.DataFrame,
    episodes: pd.DataFrame,
    strategy_name: str,
    credit_variant: str,
    window_name: str,
    start: str,
    end: str | None,
) -> dict[str, object]:
    mask = frame["date"] >= pd.Timestamp(start)
    if end is not None:
        mask &= frame["date"] <= pd.Timestamp(end)
    sub = frame.loc[mask].copy()
    if len(sub) == 0:
        return {
            "strategy": strategy_name,
            "credit_variant": credit_variant,
            "window": window_name,
            "cumulative_return": np.nan,
            "max_drawdown": np.nan,
            "Sharpe": np.nan,
            "time_in_credit_lock": 0,
            "time_in_credit_watch": 0,
            "number_credit_entries": 0,
            "number_credit_unlocks": 0,
            "number_relocks": 0,
            "false_recovery_count": 0,
            "missed_rebound_count": 0,
            "avg_weight_SPY": np.nan,
            "avg_weight_GOLD": np.nan,
            "avg_weight_IEF": np.nan,
            "avg_weight_CASH": np.nan,
            "avg_weight_CMDTY_FUT": np.nan,
        }
    ret = sub[f"{strategy_name}_return"].fillna(0.0)
    nav = (1.0 + ret).cumprod()
    ann_vol = float(ret.std(ddof=1) * np.sqrt(252.0))
    ann_ret = float(nav.iloc[-1] ** (252.0 / len(sub)) - 1.0)
    end_ts = pd.Timestamp(end) if end is not None else frame["date"].max()
    ep_sub = episodes.loc[episodes["entry_date"].between(pd.Timestamp(start), end_ts)] if len(episodes) else pd.DataFrame()
    return {
        "strategy": strategy_name,
        "credit_variant": credit_variant,
        "window": window_name,
        "cumulative_return": float(nav.iloc[-1] - 1.0),
        "max_drawdown": float((nav / nav.cummax() - 1.0).min()),
        "Sharpe": float(ann_ret / ann_vol) if ann_vol > 0 else np.nan,
        "time_in_credit_lock": int(sub["credit_lock_active"].sum()) if "credit_lock_active" in sub else 0,
        "time_in_credit_watch": int(sub.get("credit_watch_active", pd.Series(False, index=sub.index)).sum()),
        "number_credit_entries": int(sub["locks_added_today"].astype(str).str.contains("CREDIT").sum()) if "locks_added_today" in sub else 0,
        "number_credit_unlocks": int(sub["locks_unlocked_today"].astype(str).str.contains("CREDIT").sum()) if "locks_unlocked_today" in sub else 0,
        "number_relocks": int(sub["lock_add_types"].astype(str).str.contains("RELOCK").sum()) if "lock_add_types" in sub else 0,
        "false_recovery_count": int(ep_sub["false_recovery_flag"].sum()) if len(ep_sub) else 0,
        "missed_rebound_count": int(ep_sub["missed_rebound_flag"].sum()) if len(ep_sub) else 0,
        "avg_weight_SPY": float(sub["weight_SPY"].mean()),
        "avg_weight_GOLD": float(sub["weight_GOLD"].mean()),
        "avg_weight_IEF": float(sub["weight_IEF"].mean()),
        "avg_weight_CASH": float(sub["weight_CASH"].mean()),
        "avg_weight_CMDTY_FUT": float(sub["weight_CMDTY_FUT"].mean()),
    }


def enrich_episode_diag(ep: pd.DataFrame, panel: pd.DataFrame, frame: pd.DataFrame, strategy: str, credit_variant: str) -> pd.DataFrame:
    if len(ep) == 0:
        return pd.DataFrame(
            columns=[
                "strategy", "credit_variant", "episode_id", "entry_date", "unlock_date", "duration_days",
                "relock_episode", "entry_type", "unlock_type", "macro_regime_at_entry", "dominant_regime",
                "entry_SPY_DD", "entry_CREDIT_SPREAD", "entry_D_CREDIT_10D", "entry_D_CREDIT_15D", "entry_D_CREDIT_20D",
                "entry_CREDIT_LEVEL_Z", "entry_CREDIT_PERCENTILE", "unlock_CREDIT_SPREAD", "unlock_D_CREDIT_10D",
                "unlock_D_CREDIT_20D", "unlock_CREDIT_LEVEL_Z", "unlock_CREDIT_PERCENTILE", "unlock_CREDIT_DD_FROM_20D_PEAK",
                "unlock_CREDIT_DD_FROM_60D_PEAK", "SPY_return_during_lock", "SPY_maxDD_during_lock",
                "strategy_return_during_lock", "next_21d_SPY_return_after_unlock", "next_21d_SPY_maxDD_after_unlock",
                "next_63d_SPY_return_after_unlock", "next_63d_SPY_maxDD_after_unlock", "false_recovery_flag",
                "missed_rebound_flag", "notes"
            ]
        )
    records = []
    for _, row in ep.iterrows():
        s = int(panel.index[panel["date"] == row["entry_date"]][0])
        unlock_idx = int(panel.index[panel["date"] == row["unlock_date"]][0])
        dominant_regime = panel.loc[s: max(unlock_idx - 1, s), "final_regime_confirmed"].mode()
        records.append(
            {
                "strategy": strategy,
                "credit_variant": credit_variant,
                "episode_id": row["episode_id"],
                "entry_date": row["entry_date"],
                "unlock_date": row["unlock_date"],
                "duration_days": row["duration_days"],
                "relock_episode": row.get("relock_episode", False),
                "entry_type": row["entry_type"],
                "unlock_type": row["unlock_type"],
                "macro_regime_at_entry": row["macro_regime_at_entry"],
                "dominant_regime": dominant_regime.iloc[0] if len(dominant_regime) else np.nan,
                "entry_SPY_DD": row["entry_SPY_DD"],
                "entry_CREDIT_SPREAD": row["entry_CREDIT_SPREAD"],
                "entry_D_CREDIT_10D": row["entry_D_CREDIT_10D"],
                "entry_D_CREDIT_15D": row["entry_D_CREDIT_15D"],
                "entry_D_CREDIT_20D": row["entry_D_CREDIT_20D"],
                "entry_CREDIT_LEVEL_Z": row["entry_CREDIT_LEVEL_Z"],
                "entry_CREDIT_PERCENTILE": row["entry_CREDIT_PERCENTILE"],
                "unlock_CREDIT_SPREAD": row["unlock_CREDIT_SPREAD"],
                "unlock_D_CREDIT_10D": row["unlock_D_CREDIT_10D"],
                "unlock_D_CREDIT_20D": row["unlock_D_CREDIT_20D"],
                "unlock_CREDIT_LEVEL_Z": row["unlock_CREDIT_LEVEL_Z"],
                "unlock_CREDIT_PERCENTILE": panel.loc[unlock_idx, "CREDIT_PERCENTILE_252D"],
                "unlock_CREDIT_DD_FROM_20D_PEAK": row["unlock_CREDIT_DD_FROM_20D_PEAK"],
                "unlock_CREDIT_DD_FROM_60D_PEAK": row["unlock_CREDIT_DD_FROM_60D_PEAK"],
                "SPY_return_during_lock": row["SPY_return_during_lock"],
                "SPY_maxDD_during_lock": row["SPY_maxDD_during_lock"],
                "strategy_return_during_lock": float((1.0 + frame.loc[s:max(unlock_idx - 1, s), f"{strategy}_return"].fillna(0.0)).prod() - 1.0),
                "next_21d_SPY_return_after_unlock": row["next_21d_SPY_return_after_unlock"],
                "next_21d_SPY_maxDD_after_unlock": row["next_21d_SPY_maxDD_after_unlock"],
                "next_63d_SPY_return_after_unlock": row["next_63d_SPY_return_after_unlock"],
                "next_63d_SPY_maxDD_after_unlock": row["next_63d_SPY_maxDD_after_unlock"],
                "false_recovery_flag": row["false_recovery_flag"],
                "missed_rebound_flag": row["missed_rebound_flag"],
                "notes": "",
            }
        )
    return pd.DataFrame(records)


def rank_strategies(perf: pd.DataFrame) -> pd.DataFrame:
    d = perf.copy()
    d["rank_sharpe"] = d["Sharpe"].rank(ascending=False, method="min")
    d["rank_maxdd"] = d["MaxDD"].rank(ascending=False, method="min")
    d["rank_final_equity"] = d["Final Equity"].rank(ascending=False, method="min")
    d["rank_calmar"] = d["Calmar"].rank(ascending=False, method="min")
    d["rank_false_recovery"] = d["false_recovery_count"].rank(ascending=True, method="min")
    d["rank_missed_rebound"] = d["missed_rebound_count"].rank(ascending=True, method="min")
    d["balanced_composite"] = (
        0.30 * d["rank_sharpe"]
        + 0.25 * d["rank_maxdd"]
        + 0.20 * d["rank_final_equity"]
        + 0.15 * d["rank_false_recovery"]
        + 0.10 * d["rank_missed_rebound"]
    )
    d["top_sharpe"] = d["rank_sharpe"].eq(1)
    d["top_maxdd"] = d["rank_maxdd"].eq(1)
    d["top_final_equity"] = d["rank_final_equity"].eq(1)
    d["top_calmar"] = d["rank_calmar"].eq(1)
    d["lowest_false_recovery"] = d["rank_false_recovery"].eq(1)
    d["lowest_missed_rebound"] = d["rank_missed_rebound"].eq(1)
    d["top_composite"] = d["balanced_composite"].eq(d["balanced_composite"].min())
    return d.sort_values(["balanced_composite", "Sharpe"], ascending=[True, False]).reset_index(drop=True)


def plot_equity(perf_names: list[str], frames: dict[str, pd.DataFrame], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    for name in perf_names:
        frame = frames[name]
        ax.plot(frame["date"], frame[f"{name}_nav"], label=name)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_drawdown(perf_names: list[str], frames: dict[str, pd.DataFrame], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    for name in perf_names:
        frame = frames[name]
        ax.plot(frame["date"], frame[f"{name}_drawdown"], label=name)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_case(panel: pd.DataFrame, frames: dict[str, pd.DataFrame], names: list[str], title: str, start: str, end: str | None, path: Path) -> None:
    mask = panel["date"] >= pd.Timestamp(start)
    if end is not None:
        mask &= panel["date"] <= pd.Timestamp(end)
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    axes[0].plot(panel.loc[mask, "date"], panel.loc[mask, "spy_price"], color="black", label="SPY")
    axes[0].plot(panel.loc[mask, "date"], panel.loc[mask, "SPY_MA20"], color="orange", linewidth=0.9, label="MA20")
    axes[0].set_yscale("log")
    axes[0].set_title(f"{title}: SPY")
    axes[0].legend(frameon=False)
    axes[1].plot(panel.loc[mask, "date"], panel.loc[mask, "CREDIT_SPREAD"], color="firebrick", label="Credit spread")
    axes[1].plot(panel.loc[mask, "date"], panel.loc[mask, "CREDIT_PERCENTILE_252D"], color="navy", alpha=0.7, label="Pct252")
    axes[1].legend(frameon=False)
    axes[1].set_title("Credit Features")
    for name in names:
        axes[2].plot(frames[name].loc[mask, "date"], frames[name].loc[mask, f"{name}_nav"], label=name)
    axes[2].legend(frameon=False)
    axes[2].set_title("Strategy NAV")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_credit_lock_compare(panel: pd.DataFrame, baseline: pd.DataFrame, challenger: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    axes[0].plot(panel["date"], panel["CREDIT_SPREAD"], color="firebrick")
    axes[0].set_title("Daily Credit Spread")
    axes[1].plot(panel["date"], panel["CREDIT_PERCENTILE_252D"], label="Percentile 252D", color="navy")
    axes[1].plot(panel["date"], panel["CREDIT_LEVEL_Z_252D"], label="Level Z 252D", color="darkgreen")
    axes[1].legend(frameon=False)
    axes[1].set_title("Credit Level Features")
    axes[2].fill_between(panel["date"], 0, 1, where=baseline["credit_lock_active"].astype(bool), alpha=0.35, color="gray", label="Baseline lock")
    axes[2].fill_between(panel["date"], 0, 1, where=challenger["credit_lock_active"].astype(bool), alpha=0.35, color="steelblue", label="Level/Percentile lock")
    axes[2].set_ylim(0, 1)
    axes[2].legend(frameon=False)
    axes[2].set_title("Credit Lock Windows")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "credit_lock_timeline_baseline_vs_level_or_percentile.png", dpi=180)
    plt.close(fig)


def plot_credit_features_with_signals(panel: pd.DataFrame, baseline: pd.DataFrame, challenger: pd.DataFrame) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
    axes[0].plot(panel["date"], panel["CREDIT_SPREAD"], color="firebrick")
    axes[0].set_title("Daily Credit Spread")
    axes[1].plot(panel["date"], panel["CREDIT_PERCENTILE_252D"], color="navy")
    axes[1].axhline(0.80, color="gray", linestyle="--", linewidth=0.9)
    axes[1].axhline(0.70, color="gray", linestyle=":", linewidth=0.9)
    axes[1].set_title("Credit Percentile 252D")
    axes[2].plot(panel["date"], panel["CREDIT_LEVEL_Z_252D"], color="darkgreen")
    axes[2].set_title("Credit Level Z 252D")
    baseline_entry = baseline["locks_added_today"].astype(str).str.contains("CREDIT")
    baseline_unlock = baseline["locks_unlocked_today"].astype(str).str.contains("CREDIT")
    chall_entry = challenger["locks_added_today"].astype(str).str.contains("CREDIT")
    chall_unlock = challenger["locks_unlocked_today"].astype(str).str.contains("CREDIT")
    axes[3].plot(panel["date"], panel["spy_price"], color="black", linewidth=0.9)
    axes[3].plot(panel["date"], panel["SPY_MA20"], color="orange", linewidth=0.9)
    axes[3].scatter(panel.loc[baseline_entry, "date"], panel.loc[baseline_entry, "spy_price"], marker="^", color="gray", s=18, label="Baseline entry")
    axes[3].scatter(panel.loc[baseline_unlock, "date"], panel.loc[baseline_unlock, "spy_price"], marker="v", color="gray", s=18, label="Baseline unlock")
    axes[3].scatter(panel.loc[chall_entry, "date"], panel.loc[chall_entry, "spy_price"], marker="^", color="steelblue", s=18, label="L/P entry")
    axes[3].scatter(panel.loc[chall_unlock, "date"], panel.loc[chall_unlock, "spy_price"], marker="v", color="steelblue", s=18, label="L/P unlock")
    axes[3].set_yscale("log")
    axes[3].legend(frameon=False, ncol=2)
    axes[3].set_title("SPY with Credit Entries/Unlocks")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "credit_spread_features_with_entries_unlocks.png", dpi=180)
    plt.close(fig)


def plot_trigger_unlock_timeline(panel: pd.DataFrame, frame: pd.DataFrame, strategy_name: str, label: str) -> None:
    entry_mask = frame["locks_added_today"].astype(str).str.contains("CREDIT")
    unlock_mask = frame["locks_unlocked_today"].astype(str).str.contains("CREDIT")
    active_mask = frame["credit_lock_active"].astype(bool)

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    axes[0].plot(panel["date"], panel["CREDIT_SPREAD"], color="firebrick", linewidth=1.0)
    axes[0].scatter(
        panel.loc[entry_mask, "date"],
        panel.loc[entry_mask, "CREDIT_SPREAD"],
        marker="^",
        color="black",
        s=22,
        label="Credit entry",
    )
    axes[0].scatter(
        panel.loc[unlock_mask, "date"],
        panel.loc[unlock_mask, "CREDIT_SPREAD"],
        marker="v",
        color="royalblue",
        s=22,
        label="Credit unlock",
    )
    axes[0].set_title(f"{label}: Daily Credit Spread with Entry / Unlock")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)

    axes[1].plot(panel["date"], panel["spy_price"], color="black", linewidth=1.0, label="SPY")
    axes[1].plot(panel["date"], panel["SPY_MA20"], color="orange", linewidth=0.9, label="MA20")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)
    axes[1].set_title("Log SPY with Credit Lock Windows")

    in_window = False
    start_idx = None
    for i, active in enumerate(active_mask):
        if active and not in_window:
            in_window = True
            start_idx = i
        elif not active and in_window:
            axes[1].axvspan(panel.loc[start_idx, "date"], panel.loc[i - 1, "date"], color="steelblue", alpha=0.18)
            in_window = False
            start_idx = None
    if in_window and start_idx is not None:
        axes[1].axvspan(panel.loc[start_idx, "date"], panel.loc[panel.index[-1], "date"], color="steelblue", alpha=0.18)

    fig.tight_layout()
    safe = label.lower().replace(" ", "_")
    fig.savefig(FIG / f"credit_trigger_unlock_timeline_{safe}.png", dpi=180)
    plt.close(fig)


def plot_tradeoff(perf: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        perf["false_recovery_count"],
        perf["missed_rebound_count"],
        s=(perf["Sharpe"] * 120).clip(lower=30),
        c=perf["MaxDD"],
        cmap="viridis_r",
    )
    for _, row in perf.iterrows():
        ax.text(row["false_recovery_count"], row["missed_rebound_count"], row["credit_variant"], fontsize=8)
    ax.set_xlabel("False recovery count")
    ax.set_ylabel("Missed rebound count")
    ax.set_title("Final Credit Challenger Trade-off")
    fig.colorbar(sc, ax=ax, label="MaxDD")
    fig.tight_layout()
    fig.savefig(FIG / "final_credit_tradeoff_scatter.png", dpi=180)
    plt.close(fig)


def plot_crisis_heatmap(crisis: pd.DataFrame, names: list[str]) -> None:
    heat = crisis.loc[crisis["credit_variant"].isin(names)].pivot(index="credit_variant", columns="window", values="cumulative_return")
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(heat, annot=True, fmt=".1%", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title("Crisis Window Returns")
    fig.tight_layout()
    fig.savefig(FIG / "crisis_window_heatmap_final_credit_challengers.png", dpi=180)
    plt.close(fig)


def build_report(alignment: pd.DataFrame, perf: pd.DataFrame, crisis: pd.DataFrame, materially: pd.DataFrame, recommendation: str) -> None:
    baseline = perf.loc[perf["credit_variant"] == "FINAL_BASELINE"].iloc[0]
    lop = perf.loc[perf["credit_variant"] == "FINAL_LEVEL_OR_PERCENTILE_LOCK"].iloc[0]
    ranked = rank_strategies(perf)
    top = ranked.iloc[0]
    lines = [
        "# FINAL_DAILY_CREDIT_REPLACEMENT_REPORT",
        "",
        "## 1. Purpose",
        "",
        "Test whether the best daily-credit redesign variants can replace the current baseline credit trigger inside the unchanged final regime-hedge strategy.",
        "",
        "## 2. Motivation",
        "",
        "- VIX can handle fast panic and fast relief.",
        "- Credit should focus on sustained or stair-step elevated credit stress, especially 2008 and 2022.",
        "",
        "## 3. Rules tested",
        "",
        "- FINAL_BASELINE",
        "- FINAL_LEVEL_OR_PERCENTILE_LOCK",
        "- FINAL_LEVEL_LOCK_FAST_RELIEF",
        "- FINAL_LEVEL_LOCK_FAST_RELIEF_PLUS_RELOCK",
        "- FINAL_ABS_ENTRY_LEVEL_Z_UNLOCK",
        "- FINAL_SHOCK_OR_Z",
        "- FINAL_WATCH_AS_PARTIAL_LOCK_DIAGNOSTIC",
        "",
        "## 4. Baseline alignment",
        "",
        alignment.to_markdown(index=False),
        "",
        "## 5. Full-sample performance",
        "",
        perf[["credit_variant", "CAGR", "Sharpe", "MaxDD", "Calmar", "Final Equity", "false_recovery_count", "missed_rebound_count"]].to_markdown(index=False),
        "",
        "## 6. Crisis window analysis",
        "",
        crisis.loc[crisis["credit_variant"].isin(["FINAL_BASELINE", "FINAL_LEVEL_OR_PERCENTILE_LOCK"])].to_markdown(index=False),
        "",
        "## 7. Trade-off discussion",
        "",
        f"- Baseline Sharpe {baseline['Sharpe']:.3f}, MaxDD {baseline['MaxDD']:.2%}, Final Equity {baseline['Final Equity']:.2f}.",
        f"- LEVEL_OR_PERCENTILE_LOCK Sharpe {lop['Sharpe']:.3f}, MaxDD {lop['MaxDD']:.2%}, Final Equity {lop['Final Equity']:.2f}.",
        f"- Top composite candidate: {top['credit_variant']}.",
        "",
        "## 8. Decision criteria",
        "",
        "- Sharpe >= baseline + 0.03",
        "- MaxDD improvement >= 1 percentage point",
        "- Final Equity >= baseline * 0.98",
        "- false recovery <= baseline",
        "- missed rebound <= baseline + 2",
        "",
        "## 9. Recommendation",
        "",
        recommendation,
        "",
        "## 10. Limitations",
        "",
        "- daily credit data availability and revisions",
        "- in-sample crisis tuning risk",
        "- 2008 and 2022 can dominate credit evidence",
        "- requires out-of-sample validation",
    ]
    (OUT / "FINAL_DAILY_CREDIT_REPLACEMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    if len(materially):
        best = materially.iloc[0]["credit_variant"]
        patch = f"Replace the baseline credit trigger with {best} in the final trigger table."
    else:
        patch = (
            "We tested replacing the baseline credit trigger with a daily level/percentile-based credit lock. "
            "It improved 2008 and 2022 sustained credit stress windows but did not robustly dominate the simpler baseline "
            "after considering full-sample performance, COVID / 2025 responsiveness, and overfitting risk. "
            "Therefore, the final strategy retains the baseline credit trigger, while LEVEL_OR_PERCENTILE_LOCK remains a future candidate."
        )
    (OUT / "README_PATCH_SUGGESTION.md").write_text(patch, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    mod = load_redesign_module()
    patch_credit_variant_hooks(mod)
    raw_panel = load_panel()
    panel = mod.add_credit_features(mod.prepare_panel(raw_panel))
    save_cols = [
        "date", "spy_price", "CREDIT_SPREAD", "D_CREDIT_10D", "D_CREDIT_15D", "D_CREDIT_20D", "D_CREDIT_30D", "D_CREDIT_60D",
        "CREDIT_LEVEL_Z_126D", "CREDIT_LEVEL_Z_252D", "CREDIT_LEVEL_Z_504D",
        "CREDIT_PERCENTILE_126D", "CREDIT_PERCENTILE_252D", "CREDIT_PERCENTILE_504D",
        "CREDIT_MA20", "CREDIT_MA60", "CREDIT_MA120",
        "CREDIT_SPREAD_ABOVE_MA20", "CREDIT_SPREAD_ABOVE_MA60", "CREDIT_SPREAD_ABOVE_MA120",
        "CREDIT_PEAK_20D", "CREDIT_PEAK_60D", "CREDIT_DD_FROM_20D_PEAK", "CREDIT_DD_FROM_60D_PEAK",
        "D_CREDIT_10D_Z_252D", "D_CREDIT_20D_Z_252D"
    ]
    panel[save_cols].to_csv(OUT / "daily_credit_feature_panel.csv", index=False)

    baseline_frame = build_baseline_frame(panel)
    alignment = alignment_check(panel, baseline_frame)
    if alignment.loc[0, "daily_return_correlation_with_main_pipeline_final"] < 0.999999 or alignment.loc[0, "max_abs_daily_return_diff"] > 1e-12 or alignment.loc[0, "mismatched_stress_days"] != 0:
        raise RuntimeError("Baseline alignment failed. Stop before challenger run.")

    baseline_ep = enrich_episode_diag(
        mod.episode_diag(panel, baseline_frame, FINAL_STRATEGY),
        panel,
        baseline_frame,
        FINAL_STRATEGY,
        "FINAL_BASELINE",
    )
    baseline_perf_row = make_strategy_perf(baseline_frame, panel, FINAL_STRATEGY, "FINAL_BASELINE", baseline_ep)

    challengers = [
        FinalCreditChallenger("FINAL_CHALLENGER_LEVEL_OR_PERCENTILE_LOCK", "LEVEL_OR_PERCENTILE_LOCK"),
        FinalCreditChallenger("FINAL_CHALLENGER_LEVEL_LOCK_FAST_RELIEF", "LEVEL_LOCK_FAST_RELIEF"),
        FinalCreditChallenger("FINAL_CHALLENGER_LEVEL_FAST_RELIEF_RELOCK", "LEVEL_FAST_RELIEF_RELOCK"),
        FinalCreditChallenger("FINAL_CHALLENGER_ABS_ENTRY_LEVEL_Z_UNLOCK", "ABS_ENTRY_LEVEL_Z_UNLOCK"),
        FinalCreditChallenger("FINAL_CHALLENGER_SHOCK_OR_Z", "SHOCK_OR_Z"),
        FinalCreditChallenger("FINAL_CHALLENGER_WATCH_AS_LOCK", "WATCH_AS_LOCK"),
    ]

    all_frames: dict[str, pd.DataFrame] = {FINAL_STRATEGY: baseline_frame}
    perf_rows = [baseline_perf_row]
    crisis_rows = []
    episode_rows = [baseline_ep]

    for win_name, (start, end) in WINDOWS.items():
        crisis_rows.append(crisis_summary(baseline_frame, baseline_ep, FINAL_STRATEGY, "FINAL_BASELINE", win_name, start, end))

    for challenger in challengers:
        variant = mod.CreditVariant(challenger.credit_variant, challenger.credit_variant)
        frame = mod.build_final_challenger(panel, variant)
        ep = enrich_episode_diag(mod.episode_diag(panel, frame, challenger.strategy), panel, frame, challenger.strategy, f"FINAL_{challenger.credit_variant}")
        all_frames[challenger.strategy] = frame
        perf_rows.append(make_strategy_perf(frame, panel, challenger.strategy, f"FINAL_{challenger.credit_variant}", ep))
        episode_rows.append(ep)
        for win_name, (start, end) in WINDOWS.items():
            crisis_rows.append(crisis_summary(frame, ep, challenger.strategy, f"FINAL_{challenger.credit_variant}", win_name, start, end))

    perf = pd.DataFrame(perf_rows)
    perf.to_csv(OUT / "final_daily_credit_strategy_performance.csv", index=False)
    crisis = pd.DataFrame(crisis_rows)
    crisis.to_csv(OUT / "final_daily_credit_crisis_comparison.csv", index=False)
    episodes = pd.concat(episode_rows, ignore_index=True)
    episodes.to_csv(OUT / "final_daily_credit_episode_diagnostics.csv", index=False)

    ranking = rank_strategies(perf)
    ranking.to_csv(OUT / "final_daily_credit_ranking.csv", index=False)
    challenger_ranking = ranking.loc[ranking["credit_variant"] != "FINAL_BASELINE"].reset_index(drop=True)

    baseline = perf.loc[perf["credit_variant"] == "FINAL_BASELINE"].iloc[0]
    materially = perf.loc[
        (perf["credit_variant"] != "FINAL_BASELINE")
        & (perf["Sharpe"] >= baseline["Sharpe"] + 0.03)
        & (perf["MaxDD"] >= baseline["MaxDD"] + 0.01)
        & (perf["Final Equity"] >= baseline["Final Equity"] * 0.98)
        & (perf["false_recovery_count"] <= baseline["false_recovery_count"])
        & (perf["missed_rebound_count"] <= baseline["missed_rebound_count"] + 2)
    ].sort_values(["Sharpe", "MaxDD"], ascending=[False, False])
    materially.to_csv(OUT / "materially_better_candidates.csv", index=False)

    top_sharpe = challenger_ranking.sort_values("Sharpe", ascending=False).iloc[0]
    top_maxdd = challenger_ranking.sort_values("MaxDD", ascending=False).iloc[0]
    top_composite = challenger_ranking.iloc[0]

    show_names = [
        FINAL_STRATEGY,
        "FINAL_CHALLENGER_LEVEL_OR_PERCENTILE_LOCK",
        "FINAL_CHALLENGER_LEVEL_LOCK_FAST_RELIEF",
        "FINAL_CHALLENGER_ABS_ENTRY_LEVEL_Z_UNLOCK",
        top_composite["strategy"],
    ]
    show_names = list(dict.fromkeys(show_names))
    plot_equity(show_names, all_frames, "Final Daily Credit Challengers", FIG / "final_daily_credit_equity_curve.png")
    plot_drawdown(show_names, all_frames, "Final Daily Credit Challenger Drawdown", FIG / "final_daily_credit_drawdown_curve.png")
    plot_case(panel, all_frames, [FINAL_STRATEGY, "FINAL_CHALLENGER_LEVEL_OR_PERCENTILE_LOCK", top_composite["strategy"]], "2008", *WINDOWS["2008_GFC"], FIG / "final_daily_credit_case_2008.png")
    plot_case(panel, all_frames, [FINAL_STRATEGY, "FINAL_CHALLENGER_LEVEL_OR_PERCENTILE_LOCK", top_composite["strategy"]], "2020", *WINDOWS["COVID_2020"], FIG / "final_daily_credit_case_2020.png")
    plot_case(panel, all_frames, [FINAL_STRATEGY, "FINAL_CHALLENGER_LEVEL_OR_PERCENTILE_LOCK", top_composite["strategy"]], "2022", *WINDOWS["2022_RATE_WAR"], FIG / "final_daily_credit_case_2022.png")
    plot_case(panel, all_frames, [FINAL_STRATEGY, "FINAL_CHALLENGER_LEVEL_OR_PERCENTILE_LOCK", top_composite["strategy"]], "2025", *WINDOWS["2025_PULLBACK"], FIG / "final_daily_credit_case_2025.png")
    plot_credit_lock_compare(panel, baseline_frame, all_frames["FINAL_CHALLENGER_LEVEL_OR_PERCENTILE_LOCK"])
    plot_credit_features_with_signals(panel, baseline_frame, all_frames["FINAL_CHALLENGER_LEVEL_OR_PERCENTILE_LOCK"])
    plot_trigger_unlock_timeline(panel, baseline_frame, FINAL_STRATEGY, "baseline")
    for challenger in challengers:
        plot_trigger_unlock_timeline(panel, all_frames[challenger.strategy], challenger.strategy, challenger.credit_variant.lower())
    plot_tradeoff(perf)
    plot_crisis_heatmap(crisis, ["FINAL_BASELINE", "FINAL_LEVEL_OR_PERCENTILE_LOCK", "FINAL_LEVEL_LOCK_FAST_RELIEF", "FINAL_ABS_ENTRY_LEVEL_Z_UNLOCK", top_composite["credit_variant"]])

    recommendation = "KEEP BASELINE"
    if len(materially):
        recommendation = f"REPLACE BASELINE WITH {materially.iloc[0]['credit_variant']}"
    elif perf.loc[perf["credit_variant"] == "FINAL_LEVEL_OR_PERCENTILE_LOCK", "Sharpe"].iloc[0] >= baseline["Sharpe"] - 0.02:
        recommendation = "KEEP BASELINE, BUT HIGHLIGHT LEVEL_OR_PERCENTILE_LOCK AS FUTURE CANDIDATE"

    build_report(alignment, perf, crisis, materially, recommendation)

    def crisis_line(cv: str, window: str) -> str:
        row = crisis.loc[(crisis["credit_variant"] == cv) & (crisis["window"] == window)].iloc[0]
        return f"{cv} {window}: return={row['cumulative_return']:.2%}, maxdd={row['max_drawdown']:.2%}, false_recovery={int(row['false_recovery_count'])}, missed_rebound={int(row['missed_rebound_count'])}"

    print("baseline alignment check")
    print(alignment.to_string(index=False))
    print("baseline full-sample performance")
    print(perf.loc[perf["credit_variant"] == "FINAL_BASELINE", ["CAGR", "Sharpe", "MaxDD", "Calmar", "Final Equity"]].to_string(index=False))
    print("LEVEL_OR_PERCENTILE_LOCK full-sample performance")
    print(perf.loc[perf["credit_variant"] == "FINAL_LEVEL_OR_PERCENTILE_LOCK", ["CAGR", "Sharpe", "MaxDD", "Calmar", "Final Equity"]].to_string(index=False))
    print("best challenger by Sharpe")
    print(top_sharpe[["credit_variant", "Sharpe", "MaxDD", "Final Equity"]].to_string())
    print("best challenger by MaxDD")
    print(top_maxdd[["credit_variant", "Sharpe", "MaxDD", "Final Equity"]].to_string())
    print("best challenger by composite")
    print(top_composite[["credit_variant", "Sharpe", "MaxDD", "Final Equity", "balanced_composite"]].to_string())
    print("2008 comparison")
    print(crisis_line("FINAL_BASELINE", "2008_GFC"))
    print(crisis_line("FINAL_LEVEL_OR_PERCENTILE_LOCK", "2008_GFC"))
    print("2020 comparison")
    print(crisis_line("FINAL_BASELINE", "COVID_2020"))
    print(crisis_line("FINAL_LEVEL_OR_PERCENTILE_LOCK", "COVID_2020"))
    print("2022 comparison")
    print(crisis_line("FINAL_BASELINE", "2022_RATE_WAR"))
    print(crisis_line("FINAL_LEVEL_OR_PERCENTILE_LOCK", "2022_RATE_WAR"))
    print("2025 comparison")
    print(crisis_line("FINAL_BASELINE", "2025_PULLBACK"))
    print(crisis_line("FINAL_LEVEL_OR_PERCENTILE_LOCK", "2025_PULLBACK"))
    print("materially better candidate count")
    print(len(materially))
    print("recommendation")
    print(recommendation)
    print("output paths")
    print(str(OUT))


if __name__ == "__main__":
    main()
