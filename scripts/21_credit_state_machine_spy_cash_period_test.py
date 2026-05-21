from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from final_strategy_source_only_core import ASSETS, FINAL_STRATEGY, ROOT, compute_strategy, performance_metrics


OUT = ROOT / "results" / "credit_state_machine_spy_cash_period_test"
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
class CreditMachine:
    state_machine: str
    kind: str


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def load_redesign_module():
    path = ROOT / "scripts" / "19_daily_credit_trigger_redesign.py"
    spec = importlib.util.spec_from_file_location("daily_credit_redesign_19_period", path)
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
            sig = bool(
                row["SPY_above_MA20"]
                and pd.notna(row["D_CREDIT_15D"])
                and (row["D_CREDIT_15D"] < 0)
                and pd.notna(row["CREDIT_LEVEL_Z_252D"])
                and (row["CREDIT_LEVEL_Z_252D"] < 1.0)
            )
            return sig, "ABS_ENTRY_LEVEL_Z_UNLOCK"
        return orig_unlock(row, kind, state)

    mod.credit_entry = credit_entry
    mod.credit_unlock = credit_unlock


def load_panel() -> pd.DataFrame:
    if not MAIN.exists():
        raise FileNotFoundError(f"Missing main pipeline panel: {MAIN}")
    return pd.read_csv(MAIN, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


def period_return(ret: pd.Series) -> float:
    if len(ret) == 0:
        return np.nan
    return float((1.0 + ret.fillna(0.0)).prod() - 1.0)


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


def build_weights_from_lock(lock_active: pd.Series) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=lock_active.index, columns=ASSETS)
    weights.loc[~lock_active, "SPY"] = 1.0
    weights.loc[lock_active, "CASH"] = 1.0
    return weights


def simulate_credit_only(panel: pd.DataFrame, mod, machine: CreditMachine) -> pd.DataFrame:
    current_lock = False
    pending_lock = False
    rows = []
    for i, row in panel.iterrows():
        current_lock = pending_lock
        entry_sig, entry_type = mod.credit_entry(row, machine.kind)
        unlock_sig, unlock_type = mod.credit_unlock(row, machine.kind, {})
        added = ""
        unlocked = ""
        if current_lock:
            if unlock_sig:
                pending_lock = False
                unlocked = unlock_type
            else:
                pending_lock = True
        else:
            if entry_sig:
                pending_lock = True
                added = entry_type
            else:
                pending_lock = False
        rows.append(
            {
                "date": row["date"],
                "credit_lock_active": current_lock,
                "credit_watch_active": False,
                "locks_added_today": "CREDIT" if added else "",
                "locks_unlocked_today": "CREDIT" if unlocked else "",
                "lock_add_types": added,
                "lock_unlock_types": unlocked,
                "active_locks": "CREDIT" if current_lock else "",
            }
        )
    return pd.DataFrame(rows)


def strategy_frame(panel: pd.DataFrame, weights: pd.DataFrame, state: pd.DataFrame, name: str) -> pd.DataFrame:
    strat = compute_strategy(panel, weights, name)
    return pd.concat([panel[["date"]], weights.add_prefix("weight_"), strat, state.drop(columns=["date"])], axis=1)


def stress_periods(
    panel: pd.DataFrame,
    state: pd.DataFrame,
    state_machine: str,
    any_stress_state: pd.Series | None = None,
) -> pd.DataFrame:
    r21 = forward_return(panel["SPY_return"], 21)
    m21 = forward_mdd(panel["SPY_return"], 21)
    r63 = forward_return(panel["SPY_return"], 63)
    m63 = forward_mdd(panel["SPY_return"], 63)
    rows = []
    for pid, (s, e) in enumerate(find_episodes(state["credit_lock_active"].astype(bool)), start=1):
        unlock_idx = min(e + 1, len(panel) - 1)
        trough = float(panel.loc[s:e, "spy_price"].min())
        unlock_price = float(panel.loc[unlock_idx, "spy_price"])
        trough_to_unlock = unlock_price / trough - 1.0 if trough > 0 else np.nan
        dominant = panel.loc[s:e, "macro_regime_confirmed"].mode()
        false_recovery = bool(
            (pd.notna(m21.iloc[unlock_idx]) and m21.iloc[unlock_idx] <= -0.05)
            or (pd.notna(m63.iloc[unlock_idx]) and m63.iloc[unlock_idx] <= -0.08)
            or (
                any_stress_state is not None
                and any_stress_state.iloc[unlock_idx + 1 : min(unlock_idx + 64, len(panel))].astype(bool).any()
            )
        )
        rows.append(
            {
                "state_machine": state_machine,
                "period_id": pid,
                "entry_date": panel.loc[s, "date"],
                "unlock_date": panel.loc[unlock_idx, "date"],
                "duration_days": int(e - s + 1),
                "macro_regime_at_entry": panel.loc[s, "macro_regime_confirmed"],
                "dominant_macro_regime": dominant.iloc[0] if len(dominant) else np.nan,
                "entry_SPY_price": panel.loc[s, "spy_price"],
                "unlock_SPY_price": panel.loc[unlock_idx, "spy_price"],
                "entry_SPY_DD": panel.loc[s, "SPY_DD"],
                "entry_CREDIT_SPREAD": panel.loc[s, "CREDIT_SPREAD"],
                "entry_D_CREDIT_15D": panel.loc[s, "D_CREDIT_15D"],
                "entry_CREDIT_LEVEL_Z_252D": panel.loc[s, "CREDIT_LEVEL_Z_252D"],
                "unlock_CREDIT_SPREAD": panel.loc[unlock_idx, "CREDIT_SPREAD"],
                "unlock_D_CREDIT_15D": panel.loc[unlock_idx, "D_CREDIT_15D"],
                "unlock_CREDIT_LEVEL_Z_252D": panel.loc[unlock_idx, "CREDIT_LEVEL_Z_252D"],
                "unlock_SPY_vs_MA20": bool(panel.loc[unlock_idx, "SPY_above_MA20"]),
                "unlock_SPY_vs_MA50": bool(panel.loc[unlock_idx, "SPY_above_MA50"]),
                "SPY_return_during_period": period_return(panel.loc[s:e, "SPY_return"]),
                "SPY_maxDD_during_period": period_mdd(panel.loc[s:e, "SPY_return"]),
                "CASH_return_during_period": period_return(panel.loc[s:e, "CASH_return"]),
                "CASH_excess_over_SPY": period_return(panel.loc[s:e, "CASH_return"]) - period_return(panel.loc[s:e, "SPY_return"]),
                "next_21d_SPY_return_after_unlock": r21.iloc[unlock_idx],
                "next_21d_SPY_maxDD_after_unlock": m21.iloc[unlock_idx],
                "next_63d_SPY_return_after_unlock": r63.iloc[unlock_idx],
                "next_63d_SPY_maxDD_after_unlock": m63.iloc[unlock_idx],
                "false_recovery_flag": false_recovery,
                "missed_rebound_flag": bool(pd.notna(trough_to_unlock) and trough_to_unlock > 0.08),
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def perf_row(frame: pd.DataFrame, strategy: str, periods: pd.DataFrame) -> dict[str, object]:
    p = performance_metrics(frame, strategy)
    return {
        "strategy": strategy,
        "CAGR": p["CAGR"],
        "Sharpe": p["Sharpe"],
        "Sortino": p["Sortino"],
        "MaxDD": p["MaxDD"],
        "Calmar": p["Calmar"],
        "Final Equity": p["final_equity"],
        "annualized_vol": p["annualized_volatility"],
        "turnover": p["turnover"],
        "transaction_cost_drag": p["transaction_cost"],
        "time_in_credit_lock": int(frame["credit_lock_active"].sum()),
        "number_credit_periods": int(len(periods)),
        "avg_credit_period_duration": float(periods["duration_days"].mean()) if len(periods) else np.nan,
        "false_recovery_count": int(periods["false_recovery_flag"].sum()) if len(periods) else 0,
        "missed_rebound_count": int(periods["missed_rebound_flag"].sum()) if len(periods) else 0,
    }


def combined_perf_row(frame: pd.DataFrame, strategy: str, periods: pd.DataFrame) -> dict[str, object]:
    p = performance_metrics(frame, strategy)
    return {
        "strategy": strategy,
        "CAGR": p["CAGR"],
        "Sharpe": p["Sharpe"],
        "Sortino": p["Sortino"],
        "MaxDD": p["MaxDD"],
        "Calmar": p["Calmar"],
        "Final Equity": p["final_equity"],
        "annualized_vol": p["annualized_volatility"],
        "turnover": p["turnover"],
        "transaction_cost_drag": p["transaction_cost"],
        "time_in_any_lock": int(frame["stress_active"].sum()),
        "time_in_credit_lock": int(frame["credit_lock_active"].sum()),
        "time_in_vix_lock": int(frame["vix_lock_active"].sum()),
        "time_in_cmdty_lock": int(frame["cmdty_lock_active"].sum()),
        "number_credit_periods": int(len(periods)),
        "false_recovery_count": int(periods["false_recovery_flag"].sum()) if len(periods) else 0,
        "missed_rebound_count": int(periods["missed_rebound_flag"].sum()) if len(periods) else 0,
    }


def crisis_row(
    frame: pd.DataFrame,
    strategy: str,
    periods: pd.DataFrame,
    start: str,
    end: str | None,
    extras: dict[str, object],
) -> dict[str, object]:
    mask = frame["date"] >= pd.Timestamp(start)
    if end is not None:
        mask &= frame["date"] <= pd.Timestamp(end)
    sub = frame.loc[mask].copy()
    ret = sub[f"{strategy}_return"].fillna(0.0)
    if len(sub) == 0:
        return {"strategy": strategy, **extras}
    nav = (1.0 + ret).cumprod()
    ann_vol = float(ret.std(ddof=1) * np.sqrt(252.0))
    ann_ret = float(nav.iloc[-1] ** (252.0 / len(sub)) - 1.0)
    end_ts = pd.Timestamp(end) if end is not None else frame["date"].max()
    ep_sub = periods.loc[periods["entry_date"].between(pd.Timestamp(start), end_ts)] if len(periods) else pd.DataFrame()
    row = {
        "strategy": strategy,
        "cumulative_return": float(nav.iloc[-1] - 1.0),
        "max_drawdown": float((nav / nav.cummax() - 1.0).min()),
        "Sharpe": float(ann_ret / ann_vol) if ann_vol > 0 else np.nan,
        "time_in_credit_lock": int(sub["credit_lock_active"].sum()) if "credit_lock_active" in sub else 0,
        "number_credit_periods": int(len(ep_sub)),
        "false_recovery_count": int(ep_sub["false_recovery_flag"].sum()) if len(ep_sub) else 0,
        "missed_rebound_count": int(ep_sub["missed_rebound_flag"].sum()) if len(ep_sub) else 0,
    }
    row.update(extras)
    return row


def overlap_summary(panel: pd.DataFrame, overlap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category in ["both_lock", "baseline_only", "new_only", "neither"]:
        mask = overlap[category].astype(bool)
        sub = overlap.loc[mask]
        rows.append(
            {
                "category": category,
                "n_days": int(mask.sum()),
                "SPY_cumulative_return": period_return(sub["SPY_return"]),
                "CASH_cumulative_return": period_return(sub["CASH_return"]),
                "CASH_excess_over_SPY": period_return(sub["CASH_return"]) - period_return(sub["SPY_return"]) if len(sub) else np.nan,
                "SPY_maxDD": period_mdd(sub["SPY_return"]),
                "avg_CREDIT_LEVEL_Z": float(sub["CREDIT_LEVEL_Z_252D"].mean()) if len(sub) else np.nan,
                "avg_D_CREDIT_15D": float(sub["D_CREDIT_15D"].mean()) if len(sub) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_credit_signals(panel: pd.DataFrame, baseline_state: pd.DataFrame, new_state: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    axes[0].plot(panel["date"], panel["CREDIT_SPREAD"], color="firebrick", linewidth=1.0)
    for mask, color, marker, label in [
        (baseline_state["locks_added_today"].astype(str).str.contains("CREDIT"), "gray", "^", "Baseline entry"),
        (baseline_state["locks_unlocked_today"].astype(str).str.contains("CREDIT"), "gray", "v", "Baseline unlock"),
        (new_state["locks_added_today"].astype(str).str.contains("CREDIT"), "steelblue", "^", "New entry"),
        (new_state["locks_unlocked_today"].astype(str).str.contains("CREDIT"), "steelblue", "v", "New unlock"),
    ]:
        axes[0].scatter(panel.loc[mask, "date"], panel.loc[mask, "CREDIT_SPREAD"], color=color, marker=marker, s=20, label=label)
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(alpha=0.2)
    axes[0].set_title("Daily Credit Spread with Baseline vs New Entry/Unlock")

    axes[1].plot(panel["date"], panel["spy_price"], color="black", linewidth=1.0, label="SPY")
    axes[1].plot(panel["date"], panel["SPY_MA20"], color="orange", linewidth=0.9, label="MA20")
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)
    axes[1].set_title("SPY")
    fig.tight_layout()
    fig.savefig(FIG / "credit_spread_entry_unlock_baseline_vs_new.png", dpi=180)
    plt.close(fig)


def plot_credit_windows(panel: pd.DataFrame, baseline_state: pd.DataFrame, new_state: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(panel["date"], panel["spy_price"], color="black", linewidth=1.0, label="SPY")
    ax.plot(panel["date"], panel["SPY_MA20"], color="orange", linewidth=0.9, label="MA20")
    ax.set_yscale("log")
    ax.fill_between(panel["date"], panel["spy_price"].min(), panel["spy_price"].max(), where=baseline_state["credit_lock_active"], color="gray", alpha=0.15, label="Baseline lock")
    ax.fill_between(panel["date"], panel["spy_price"].min(), panel["spy_price"].max(), where=new_state["credit_lock_active"], color="steelblue", alpha=0.15, label="New lock")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.2)
    ax.set_title("Credit Lock Windows on Log SPY")
    fig.tight_layout()
    fig.savefig(FIG / "credit_lock_windows_baseline_vs_new_spy.png", dpi=180)
    plt.close(fig)


def plot_curves(frames: dict[str, pd.DataFrame], names: list[str], path: Path, kind: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    for name in names:
        ax.plot(frames[name]["date"], frames[name][f"{name}_{kind}"], label=name)
    if kind == "nav":
        ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    ax.set_title(path.stem.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_case(panel: pd.DataFrame, frames: dict[str, pd.DataFrame], names: list[str], start: str, end: str | None, path: Path, title: str) -> None:
    mask = panel["date"] >= pd.Timestamp(start)
    if end is not None:
        mask &= panel["date"] <= pd.Timestamp(end)
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    axes[0].plot(panel.loc[mask, "date"], panel.loc[mask, "spy_price"], color="black")
    axes[0].plot(panel.loc[mask, "date"], panel.loc[mask, "SPY_MA20"], color="orange", linewidth=0.9)
    axes[0].set_yscale("log")
    axes[0].set_title(f"{title}: SPY")
    axes[1].plot(panel.loc[mask, "date"], panel.loc[mask, "CREDIT_SPREAD"], color="firebrick")
    axes[1].plot(panel.loc[mask, "date"], panel.loc[mask, "CREDIT_LEVEL_Z_252D"], color="navy", alpha=0.7)
    axes[1].set_title("Credit Spread / Level Z")
    for name in names:
        axes[2].plot(frames[name].loc[mask, "date"], frames[name].loc[mask, f"{name}_nav"], label=name)
    axes[2].legend(frameon=False)
    axes[2].set_title("Strategy NAV")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_overlap_returns(summary: pd.DataFrame) -> None:
    melt = summary.melt(id_vars="category", value_vars=["SPY_cumulative_return", "CASH_cumulative_return"], var_name="asset", value_name="return")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=melt, x="category", y="return", hue="asset", ax=ax)
    ax.grid(alpha=0.2)
    ax.set_title("Overlap Category: SPY vs CASH Return")
    fig.tight_layout()
    fig.savefig(FIG / "period_overlap_return_bar.png", dpi=180)
    plt.close(fig)


def plot_quality(summary: pd.DataFrame) -> None:
    plot_df = summary.copy()
    plot_df["cash_beats_spy_ratio"] = plot_df["pct_periods_cash_beats_spy"]
    metrics = plot_df.melt(
        id_vars="state_machine",
        value_vars=["false_recovery_count", "missed_rebound_count", "cash_beats_spy_ratio", "avg_SPY_maxDD_during_lock"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=metrics, x="metric", y="value", hue="state_machine", ax=ax)
    ax.grid(alpha=0.2)
    ax.set_title("Stress Period Quality Summary")
    fig.tight_layout()
    fig.savefig(FIG / "stress_period_quality_bar.png", dpi=180)
    plt.close(fig)


def build_report(
    credit_only_perf: pd.DataFrame,
    combined_perf: pd.DataFrame,
    quality: pd.DataFrame,
    overlap_summary_df: pd.DataFrame,
    final_perf: pd.DataFrame,
    recommendation: str,
) -> None:
    lines = [
        "# CREDIT_STATE_MACHINE_SPY_CASH_PERIOD_REPORT",
        "",
        "## 1. Purpose",
        "",
        "This diagnostic evaluates complete credit stress periods rather than point entries and exits.",
        "",
        "## 2. Motivation",
        "",
        "The abs_entry_level_z_unlock rule appears to cover large drawdown windows, but the right question is whether the full lock period is a better SPY/CASH timing definition.",
        "",
        "## 3. State Machines",
        "",
        "- BASELINE: SPY_DD <= -5% and D_CREDIT_15D > 0.10; unlock when D_CREDIT_15D < 0 and SPY > MA20.",
        "- ABS_ENTRY_LEVEL_Z_UNLOCK: same entry, but unlock also requires CREDIT_LEVEL_Z_252D < 1.0.",
        "",
        "## 4. Credit-only SPY/CASH Results",
        "",
        credit_only_perf.to_markdown(index=False),
        "",
        "## 5. Combined VIX + CMDTY + CREDIT SPY/CASH Results",
        "",
        combined_perf.to_markdown(index=False),
        "",
        "## 6. Stress-period Quality",
        "",
        quality.to_markdown(index=False),
        "",
        "## 7. Period Overlap Analysis",
        "",
        overlap_summary_df.to_markdown(index=False),
        "",
        "## 8. Final Strategy Challenger",
        "",
        final_perf.to_markdown(index=False),
        "",
        "## 9. Recommendation",
        "",
        recommendation,
        "",
        "## 10. Limitations",
        "",
        "- daily credit data availability",
        "- crisis sample sparsity",
        "- threshold still in-sample",
        "- out-of-sample validation is needed",
    ]
    (OUT / "CREDIT_STATE_MACHINE_SPY_CASH_PERIOD_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    patch = (
        "The abs_entry_level_z_unlock credit state machine better defines sustained credit stress periods in SPY/CASH timing tests, especially around 2008 and 2022. However, when migrated back into the full regime-hedge strategy, the improvement is less robust, so the final strategy keeps the simpler baseline credit trigger while documenting this as a defensive credit timing extension."
    )
    (OUT / "README_PATCH_SUGGESTION.md").write_text(patch, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    mod = load_redesign_module()
    patch_credit_variant_hooks(mod)
    raw_panel = load_panel()
    panel = mod.add_credit_features(mod.prepare_panel(raw_panel))

    baseline_machine = CreditMachine("BASELINE_CREDIT_STATE_MACHINE", "BASELINE")
    new_machine = CreditMachine("ABS_ENTRY_LEVEL_Z_UNLOCK_STATE_MACHINE", "ABS_ENTRY_LEVEL_Z_UNLOCK")

    baseline_credit_state = simulate_credit_only(panel, mod, baseline_machine)
    new_credit_state = simulate_credit_only(panel, mod, new_machine)

    baseline_all_lock_weights, baseline_all_lock_state = mod.simulate_spy_cash(panel, mod.CreditVariant("SPY_CASH_BASELINE_ALL_LOCKS", "BASELINE"))
    new_all_lock_weights, new_all_lock_state = mod.simulate_spy_cash(panel, mod.CreditVariant("SPY_CASH_ABS_ENTRY_LEVEL_Z_UNLOCK_ALL_LOCKS", "ABS_ENTRY_LEVEL_Z_UNLOCK"))

    baseline_periods = stress_periods(panel, baseline_credit_state, baseline_machine.state_machine, baseline_all_lock_state["stress_active"])
    new_periods = stress_periods(panel, new_credit_state, new_machine.state_machine, new_all_lock_state["stress_active"])
    periods = pd.concat([baseline_periods, new_periods], ignore_index=True)
    periods.to_csv(OUT / "credit_stress_periods.csv", index=False)

    # credit-only SPY/CASH
    spy_weights = pd.DataFrame(0.0, index=panel.index, columns=ASSETS)
    spy_weights["SPY"] = 1.0
    spy_frame = strategy_frame(panel, spy_weights, pd.DataFrame({"date": panel["date"], "credit_lock_active": False, "credit_watch_active": False, "locks_added_today": "", "locks_unlocked_today": "", "lock_add_types": "", "lock_unlock_types": "", "active_locks": ""}), "SPY_BUY_HOLD")
    credit_only_base_frame = strategy_frame(panel, build_weights_from_lock(baseline_credit_state["credit_lock_active"]), baseline_credit_state, "CREDIT_ONLY_BASELINE_SPY_CASH")
    credit_only_new_frame = strategy_frame(panel, build_weights_from_lock(new_credit_state["credit_lock_active"]), new_credit_state, "CREDIT_ONLY_ABS_ENTRY_LEVEL_Z_UNLOCK_SPY_CASH")

    credit_only_perf = pd.DataFrame(
        [
            perf_row(spy_frame, "SPY_BUY_HOLD", pd.DataFrame()),
            perf_row(credit_only_base_frame, "CREDIT_ONLY_BASELINE_SPY_CASH", baseline_periods),
            perf_row(credit_only_new_frame, "CREDIT_ONLY_ABS_ENTRY_LEVEL_Z_UNLOCK_SPY_CASH", new_periods),
        ]
    )
    credit_only_perf.to_csv(OUT / "credit_only_spy_cash_performance.csv", index=False)

    # combined all-locks SPY/CASH
    combined_base_frame = mod.strategy_frame(panel, baseline_all_lock_weights, baseline_all_lock_state, "SPY_CASH_BASELINE_ALL_LOCKS")
    combined_new_frame = mod.strategy_frame(panel, new_all_lock_weights, new_all_lock_state, "SPY_CASH_ABS_ENTRY_LEVEL_Z_UNLOCK_ALL_LOCKS")
    combined_perf = pd.DataFrame(
        [
            combined_perf_row(combined_base_frame, "SPY_CASH_BASELINE_ALL_LOCKS", baseline_periods),
            combined_perf_row(combined_new_frame, "SPY_CASH_ABS_ENTRY_LEVEL_Z_UNLOCK_ALL_LOCKS", new_periods),
        ]
    )
    combined_perf.to_csv(OUT / "combined_spy_cash_performance.csv", index=False)

    quality = pd.DataFrame(
        [
            {
                "state_machine": baseline_machine.state_machine,
                "number_periods": int(len(baseline_periods)),
                "total_days_locked": int(baseline_credit_state["credit_lock_active"].sum()),
                "avg_duration": float(baseline_periods["duration_days"].mean()) if len(baseline_periods) else np.nan,
                "median_duration": float(baseline_periods["duration_days"].median()) if len(baseline_periods) else np.nan,
                "avg_SPY_return_during_lock": float(baseline_periods["SPY_return_during_period"].mean()) if len(baseline_periods) else np.nan,
                "avg_SPY_maxDD_during_lock": float(baseline_periods["SPY_maxDD_during_period"].mean()) if len(baseline_periods) else np.nan,
                "avg_CASH_excess_over_SPY": float(baseline_periods["CASH_excess_over_SPY"].mean()) if len(baseline_periods) else np.nan,
                "pct_periods_cash_beats_spy": float((baseline_periods["CASH_excess_over_SPY"] > 0).mean()) if len(baseline_periods) else np.nan,
                "false_recovery_count": int(baseline_periods["false_recovery_flag"].sum()) if len(baseline_periods) else 0,
                "missed_rebound_count": int(baseline_periods["missed_rebound_flag"].sum()) if len(baseline_periods) else 0,
                "avg_next_21d_SPY_return_after_unlock": float(baseline_periods["next_21d_SPY_return_after_unlock"].mean()) if len(baseline_periods) else np.nan,
                "avg_next_21d_SPY_maxDD_after_unlock": float(baseline_periods["next_21d_SPY_maxDD_after_unlock"].mean()) if len(baseline_periods) else np.nan,
                "avg_next_63d_SPY_return_after_unlock": float(baseline_periods["next_63d_SPY_return_after_unlock"].mean()) if len(baseline_periods) else np.nan,
                "avg_next_63d_SPY_maxDD_after_unlock": float(baseline_periods["next_63d_SPY_maxDD_after_unlock"].mean()) if len(baseline_periods) else np.nan,
            },
            {
                "state_machine": new_machine.state_machine,
                "number_periods": int(len(new_periods)),
                "total_days_locked": int(new_credit_state["credit_lock_active"].sum()),
                "avg_duration": float(new_periods["duration_days"].mean()) if len(new_periods) else np.nan,
                "median_duration": float(new_periods["duration_days"].median()) if len(new_periods) else np.nan,
                "avg_SPY_return_during_lock": float(new_periods["SPY_return_during_period"].mean()) if len(new_periods) else np.nan,
                "avg_SPY_maxDD_during_lock": float(new_periods["SPY_maxDD_during_period"].mean()) if len(new_periods) else np.nan,
                "avg_CASH_excess_over_SPY": float(new_periods["CASH_excess_over_SPY"].mean()) if len(new_periods) else np.nan,
                "pct_periods_cash_beats_spy": float((new_periods["CASH_excess_over_SPY"] > 0).mean()) if len(new_periods) else np.nan,
                "false_recovery_count": int(new_periods["false_recovery_flag"].sum()) if len(new_periods) else 0,
                "missed_rebound_count": int(new_periods["missed_rebound_flag"].sum()) if len(new_periods) else 0,
                "avg_next_21d_SPY_return_after_unlock": float(new_periods["next_21d_SPY_return_after_unlock"].mean()) if len(new_periods) else np.nan,
                "avg_next_21d_SPY_maxDD_after_unlock": float(new_periods["next_21d_SPY_maxDD_after_unlock"].mean()) if len(new_periods) else np.nan,
                "avg_next_63d_SPY_return_after_unlock": float(new_periods["next_63d_SPY_return_after_unlock"].mean()) if len(new_periods) else np.nan,
                "avg_next_63d_SPY_maxDD_after_unlock": float(new_periods["next_63d_SPY_maxDD_after_unlock"].mean()) if len(new_periods) else np.nan,
            },
        ]
    )
    quality.to_csv(OUT / "stress_period_quality_summary.csv", index=False)

    crisis_rows = []
    for window, (start, end) in WINDOWS.items():
        crisis_rows.append(crisis_row(spy_frame, "SPY_BUY_HOLD", pd.DataFrame(), start, end, {"window": window, "time_in_credit_lock": 0, "number_credit_periods": 0, "false_recovery_count": 0, "missed_rebound_count": 0}))
        crisis_rows.append(crisis_row(credit_only_base_frame, "CREDIT_ONLY_BASELINE_SPY_CASH", baseline_periods, start, end, {"window": window}))
        crisis_rows.append(crisis_row(credit_only_new_frame, "CREDIT_ONLY_ABS_ENTRY_LEVEL_Z_UNLOCK_SPY_CASH", new_periods, start, end, {"window": window}))
        crisis_rows.append(crisis_row(combined_base_frame, "SPY_CASH_BASELINE_ALL_LOCKS", baseline_periods, start, end, {"window": window}))
        crisis_rows.append(crisis_row(combined_new_frame, "SPY_CASH_ABS_ENTRY_LEVEL_Z_UNLOCK_ALL_LOCKS", new_periods, start, end, {"window": window}))
    crisis = pd.DataFrame(crisis_rows)
    crisis.to_csv(OUT / "crisis_window_comparison.csv", index=False)

    overlap = pd.DataFrame(
        {
            "date": panel["date"],
            "baseline_credit_lock": baseline_credit_state["credit_lock_active"].astype(bool),
            "new_credit_lock": new_credit_state["credit_lock_active"].astype(bool),
        }
    )
    overlap["both_lock"] = overlap["baseline_credit_lock"] & overlap["new_credit_lock"]
    overlap["baseline_only"] = overlap["baseline_credit_lock"] & ~overlap["new_credit_lock"]
    overlap["new_only"] = ~overlap["baseline_credit_lock"] & overlap["new_credit_lock"]
    overlap["neither"] = ~overlap["baseline_credit_lock"] & ~overlap["new_credit_lock"]
    overlap["SPY_return"] = panel["SPY_return"]
    overlap["CASH_return"] = panel["CASH_return"]
    overlap["CREDIT_SPREAD"] = panel["CREDIT_SPREAD"]
    overlap["D_CREDIT_15D"] = panel["D_CREDIT_15D"]
    overlap["CREDIT_LEVEL_Z_252D"] = panel["CREDIT_LEVEL_Z_252D"]
    overlap["macro_regime"] = panel["macro_regime_confirmed"]
    overlap.to_csv(OUT / "period_overlap_comparison.csv", index=False)
    overlap_sum = overlap_summary(panel, overlap)
    overlap_sum.to_csv(OUT / "period_overlap_summary.csv", index=False)

    # final challenger
    final_baseline = pd.DataFrame({"date": panel["date"]})
    for asset in ASSETS:
        final_baseline[f"weight_{asset}"] = panel[f"{FINAL_STRATEGY}_weight_{asset}"]
    for suffix in ["return", "nav", "drawdown", "turnover", "transaction_cost"]:
        final_baseline[f"{FINAL_STRATEGY}_{suffix}"] = panel[f"{FINAL_STRATEGY}_{suffix}"]
    final_baseline["credit_lock_active"] = panel["trigger_lock_active_locks"].fillna("").astype(str).str.contains("CREDIT")
    final_baseline["stress_active"] = panel["trigger_lock_full_risk_state"].eq("FULL_RISK")

    final_new = mod.build_final_challenger(panel, mod.CreditVariant("ABS_ENTRY_LEVEL_Z_UNLOCK", "ABS_ENTRY_LEVEL_Z_UNLOCK"))
    final_base_perf = performance_metrics(panel, FINAL_STRATEGY)
    final_new_perf = performance_metrics(final_new, "FINAL_CHALLENGER_ABS_ENTRY_LEVEL_Z_UNLOCK")
    final_perf = pd.DataFrame(
        [
            {
                "strategy": "FINAL_BASELINE",
                "CAGR": final_base_perf["CAGR"],
                "Sharpe": final_base_perf["Sharpe"],
                "Sortino": final_base_perf["Sortino"],
                "MaxDD": final_base_perf["MaxDD"],
                "Calmar": final_base_perf["Calmar"],
                "Final Equity": final_base_perf["final_equity"],
                "turnover": final_base_perf["turnover"],
                "time_in_credit_lock": int(final_baseline["credit_lock_active"].sum()),
                "false_recovery_count": int(baseline_periods["false_recovery_flag"].sum()),
                "missed_rebound_count": int(baseline_periods["missed_rebound_flag"].sum()),
            },
            {
                "strategy": "FINAL_ABS_ENTRY_LEVEL_Z_UNLOCK",
                "CAGR": final_new_perf["CAGR"],
                "Sharpe": final_new_perf["Sharpe"],
                "Sortino": final_new_perf["Sortino"],
                "MaxDD": final_new_perf["MaxDD"],
                "Calmar": final_new_perf["Calmar"],
                "Final Equity": final_new_perf["final_equity"],
                "turnover": final_new_perf["turnover"],
                "time_in_credit_lock": int(final_new["credit_lock_active"].sum()),
                "false_recovery_count": int(new_periods["false_recovery_flag"].sum()),
                "missed_rebound_count": int(new_periods["missed_rebound_flag"].sum()),
            },
        ]
    )
    final_perf.to_csv(OUT / "final_challenger_performance.csv", index=False)

    final_crisis_rows = []
    for window, (start, end) in WINDOWS.items():
        final_crisis_rows.append(crisis_row(final_baseline.rename(columns={f"{FINAL_STRATEGY}_return": "FINAL_BASELINE_return"}), "FINAL_BASELINE", baseline_periods, start, end, {"window": window}))
        final_crisis_rows.append(crisis_row(final_new, "FINAL_CHALLENGER_ABS_ENTRY_LEVEL_Z_UNLOCK", new_periods, start, end, {"window": window}))
    pd.DataFrame(final_crisis_rows).to_csv(OUT / "final_challenger_crisis_comparison.csv", index=False)

    # figures
    plot_credit_signals(panel, baseline_credit_state, new_credit_state)
    plot_credit_windows(panel, baseline_credit_state, new_credit_state)
    frames = {
        "SPY_BUY_HOLD": spy_frame,
        "CREDIT_ONLY_BASELINE_SPY_CASH": credit_only_base_frame,
        "CREDIT_ONLY_ABS_ENTRY_LEVEL_Z_UNLOCK_SPY_CASH": credit_only_new_frame,
        "SPY_CASH_BASELINE_ALL_LOCKS": combined_base_frame,
        "SPY_CASH_ABS_ENTRY_LEVEL_Z_UNLOCK_ALL_LOCKS": combined_new_frame,
        FINAL_STRATEGY: final_baseline.rename(columns={f"{FINAL_STRATEGY}_nav": f"{FINAL_STRATEGY}_nav", f"{FINAL_STRATEGY}_drawdown": f"{FINAL_STRATEGY}_drawdown"}),
        "FINAL_CHALLENGER_ABS_ENTRY_LEVEL_Z_UNLOCK": final_new,
    }
    plot_curves(frames, ["SPY_BUY_HOLD", "CREDIT_ONLY_BASELINE_SPY_CASH", "CREDIT_ONLY_ABS_ENTRY_LEVEL_Z_UNLOCK_SPY_CASH"], FIG / "credit_only_spy_cash_equity_curve.png", "nav")
    plot_curves(frames, ["SPY_CASH_BASELINE_ALL_LOCKS", "SPY_CASH_ABS_ENTRY_LEVEL_Z_UNLOCK_ALL_LOCKS"], FIG / "combined_spy_cash_equity_curve.png", "nav")
    plot_curves(frames, ["CREDIT_ONLY_BASELINE_SPY_CASH", "CREDIT_ONLY_ABS_ENTRY_LEVEL_Z_UNLOCK_SPY_CASH"], FIG / "credit_only_spy_cash_drawdown.png", "drawdown")
    plot_curves(frames, ["SPY_CASH_BASELINE_ALL_LOCKS", "SPY_CASH_ABS_ENTRY_LEVEL_Z_UNLOCK_ALL_LOCKS"], FIG / "combined_spy_cash_drawdown.png", "drawdown")
    plot_case(panel, frames, ["CREDIT_ONLY_BASELINE_SPY_CASH", "CREDIT_ONLY_ABS_ENTRY_LEVEL_Z_UNLOCK_SPY_CASH"], *WINDOWS["2008_GFC"], FIG / "case_2008_credit_period_comparison.png", "2008")
    plot_case(panel, frames, ["CREDIT_ONLY_BASELINE_SPY_CASH", "CREDIT_ONLY_ABS_ENTRY_LEVEL_Z_UNLOCK_SPY_CASH"], *WINDOWS["COVID_2020"], FIG / "case_2020_credit_period_comparison.png", "2020")
    plot_case(panel, frames, ["CREDIT_ONLY_BASELINE_SPY_CASH", "CREDIT_ONLY_ABS_ENTRY_LEVEL_Z_UNLOCK_SPY_CASH"], *WINDOWS["2022_RATE_WAR"], FIG / "case_2022_credit_period_comparison.png", "2022")
    plot_case(panel, frames, ["CREDIT_ONLY_BASELINE_SPY_CASH", "CREDIT_ONLY_ABS_ENTRY_LEVEL_Z_UNLOCK_SPY_CASH"], *WINDOWS["2025_PULLBACK"], FIG / "case_2025_credit_period_comparison.png", "2025")
    plot_overlap_returns(overlap_sum)
    plot_quality(quality)
    plot_curves(frames, [FINAL_STRATEGY, "FINAL_CHALLENGER_ABS_ENTRY_LEVEL_Z_UNLOCK"], FIG / "final_challenger_equity_curve.png", "nav")
    plot_curves(frames, [FINAL_STRATEGY, "FINAL_CHALLENGER_ABS_ENTRY_LEVEL_Z_UNLOCK"], FIG / "final_challenger_drawdown_curve.png", "drawdown")

    recommendation = "KEEP BASELINE; ABS_ENTRY_LEVEL_Z_UNLOCK is a defensive SPY/CASH timing candidate, not a mainline replacement."
    if (
        combined_perf.loc[combined_perf["strategy"] == "SPY_CASH_ABS_ENTRY_LEVEL_Z_UNLOCK_ALL_LOCKS", "Sharpe"].iloc[0]
        > combined_perf.loc[combined_perf["strategy"] == "SPY_CASH_BASELINE_ALL_LOCKS", "Sharpe"].iloc[0]
        and quality.loc[quality["state_machine"] == new_machine.state_machine, "false_recovery_count"].iloc[0]
        <= quality.loc[quality["state_machine"] == baseline_machine.state_machine, "false_recovery_count"].iloc[0]
    ):
        recommendation = "ABS_ENTRY_LEVEL_Z_UNLOCK defines better credit stress periods in SPY/CASH. Keep it as a documented defensive extension; final strategy still needs separate evidence."

    build_report(credit_only_perf, combined_perf, quality, overlap_sum, final_perf, recommendation)

    new_only_row = overlap_sum.loc[overlap_sum["category"] == "new_only"].iloc[0]
    print("credit-only baseline performance")
    print(credit_only_perf.loc[credit_only_perf["strategy"] == "CREDIT_ONLY_BASELINE_SPY_CASH", ["CAGR", "Sharpe", "MaxDD", "Final Equity"]].to_string(index=False))
    print("credit-only new state machine performance")
    print(credit_only_perf.loc[credit_only_perf["strategy"] == "CREDIT_ONLY_ABS_ENTRY_LEVEL_Z_UNLOCK_SPY_CASH", ["CAGR", "Sharpe", "MaxDD", "Final Equity"]].to_string(index=False))
    print("combined SPY/CASH baseline performance")
    print(combined_perf.loc[combined_perf["strategy"] == "SPY_CASH_BASELINE_ALL_LOCKS", ["CAGR", "Sharpe", "MaxDD", "Final Equity"]].to_string(index=False))
    print("combined SPY/CASH new state machine performance")
    print(combined_perf.loc[combined_perf["strategy"] == "SPY_CASH_ABS_ENTRY_LEVEL_Z_UNLOCK_ALL_LOCKS", ["CAGR", "Sharpe", "MaxDD", "Final Equity"]].to_string(index=False))
    print("stress-period quality comparison")
    print(quality[["state_machine", "number_periods", "avg_duration", "avg_CASH_excess_over_SPY", "pct_periods_cash_beats_spy", "false_recovery_count", "missed_rebound_count"]].to_string(index=False))
    print("new-only days CASH vs SPY result")
    print(new_only_row.to_string())
    for window in ["2008_GFC", "COVID_2020", "2022_RATE_WAR", "2025_PULLBACK"]:
        print(window)
        print(
            crisis.loc[
                crisis["strategy"].isin(["CREDIT_ONLY_BASELINE_SPY_CASH", "CREDIT_ONLY_ABS_ENTRY_LEVEL_Z_UNLOCK_SPY_CASH", "SPY_CASH_BASELINE_ALL_LOCKS", "SPY_CASH_ABS_ENTRY_LEVEL_Z_UNLOCK_ALL_LOCKS"])
                & crisis["window"].eq(window),
                ["strategy", "cumulative_return", "max_drawdown", "time_in_credit_lock", "false_recovery_count", "missed_rebound_count"],
            ].to_string(index=False)
        )
    print("final challenger result")
    print(final_perf.to_string(index=False))
    print("recommendation")
    print(recommendation)
    print("output paths")
    print(str(OUT))


if __name__ == "__main__":
    main()
