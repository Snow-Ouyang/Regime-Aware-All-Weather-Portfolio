from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import importlib.util
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from final_strategy_source_only_core import ASSETS, ROOT, compute_strategy, performance_metrics


OUT = ROOT / "results" / "credit_lock_narrow_unlock_test"
FIG = OUT / "figures"
MAIN = ROOT / "results" / "main_pipeline_final" / "tables" / "daily_backtest_panel.csv"

WINDOWS = {
    "2008_GFC": ("2007-10-01", "2009-06-30"),
    "COVID_2020": ("2020-02-01", "2020-06-30"),
    "2022_RATE_WAR": ("2021-11-01", "2023-03-31"),
    "2025_PULLBACK": ("2025-01-01", None),
}


@dataclass(frozen=True)
class CreditVariant:
    name: str
    unlock_z_threshold: float
    confirm_days: int
    trend_rule: str


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def load_redesign_module():
    path = ROOT / "scripts" / "19_daily_credit_trigger_redesign.py"
    spec = importlib.util.spec_from_file_location("daily_credit_redesign_24", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_panel(mod) -> pd.DataFrame:
    if not MAIN.exists():
        raise FileNotFoundError(f"Missing main pipeline panel: {MAIN}")
    raw = pd.read_csv(MAIN, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    panel = mod.add_credit_features(mod.prepare_panel(raw))
    panel["SPY_MA30"] = panel["spy_price"].rolling(30, min_periods=30).mean()
    panel["SPY_MA40"] = panel["spy_price"].rolling(40, min_periods=40).mean()
    panel["SPY_above_MA30"] = panel["spy_price"] > panel["SPY_MA30"]
    panel["SPY_above_MA40"] = panel["spy_price"] > panel["SPY_MA40"]
    return panel


def build_variants() -> list[CreditVariant]:
    z_grid = [0.9, 1.0]
    confirm_grid = [1, 2]
    trend_grid = ["MA20", "MA30", "MA40", "MA50"]
    variants: list[CreditVariant] = []
    for z_thr in z_grid:
        for confirm_days in confirm_grid:
            for trend_rule in trend_grid:
                variants.append(
                    CreditVariant(
                        name=f"Z{z_thr:.1f}_N{confirm_days}_{trend_rule}",
                        unlock_z_threshold=z_thr,
                        confirm_days=confirm_days,
                        trend_rule=trend_rule,
                    )
                )
    return variants


def trend_ok(row: pd.Series, trend_rule: str) -> bool:
    if trend_rule == "MA20":
        return bool(row["SPY_above_MA20"])
    if trend_rule == "MA30":
        return bool(row["SPY_above_MA30"])
    if trend_rule == "MA40":
        return bool(row["SPY_above_MA40"])
    if trend_rule == "MA50":
        return bool(row["SPY_above_MA50"])
    raise ValueError(f"Unknown trend_rule: {trend_rule}")


def entry_ok(row: pd.Series) -> bool:
    return bool(
        row["refined_regime_confirmed"] in {"FLAT_LOW_RATE", "FLAT_HIGH_RATE"}
        and pd.notna(row["SPY_DD"])
        and row["SPY_DD"] <= -0.05
        and pd.notna(row["D_CREDIT_15D"])
        and row["D_CREDIT_15D"] > 0.10
    )


def simulate_credit_only(panel: pd.DataFrame, variant: CreditVariant) -> pd.DataFrame:
    current_lock = False
    pending_lock = False
    neg_run = 0
    rows: list[dict[str, object]] = []

    for _, row in panel.iterrows():
        current_lock = pending_lock
        unlock_cond = bool(
            pd.notna(row["D_CREDIT_15D"])
            and row["D_CREDIT_15D"] < 0
            and pd.notna(row["CREDIT_LEVEL_Z_252D"])
            and row["CREDIT_LEVEL_Z_252D"] < variant.unlock_z_threshold
            and trend_ok(row, variant.trend_rule)
        )
        if unlock_cond:
            neg_run += 1
        else:
            neg_run = 0

        entry = entry_ok(row)
        add = False
        unl = False
        if current_lock:
            if neg_run >= variant.confirm_days:
                pending_lock = False
                unl = True
                neg_run = 0
            else:
                pending_lock = True
        else:
            if entry:
                pending_lock = True
                add = True
            else:
                pending_lock = False
                neg_run = 0

        rows.append(
            {
                "date": row["date"],
                "credit_lock_active": current_lock,
                "credit_entry_marker": add,
                "credit_unlock_marker": unl,
                "unlock_condition_today": unlock_cond,
                "unlock_confirm_run": neg_run,
                "trend_rule": variant.trend_rule,
            }
        )
    return pd.DataFrame(rows)


def build_weights(lock_active: pd.Series) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=lock_active.index, columns=ASSETS)
    weights.loc[~lock_active.astype(bool), "SPY"] = 1.0
    weights.loc[lock_active.astype(bool), "CASH"] = 1.0
    return weights


def strategy_frame(panel: pd.DataFrame, state: pd.DataFrame, strategy: str) -> pd.DataFrame:
    weights = build_weights(state["credit_lock_active"])
    strat = compute_strategy(panel, weights, strategy)
    return pd.concat([panel[["date"]], strat, weights.add_prefix("weight_"), state.drop(columns=["date"])], axis=1)


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


def extract_periods(panel: pd.DataFrame, state: pd.DataFrame, variant: CreditVariant) -> pd.DataFrame:
    r21 = forward_return(panel["SPY_return"], 21)
    m21 = forward_mdd(panel["SPY_return"], 21)
    r63 = forward_return(panel["SPY_return"], 63)
    m63 = forward_mdd(panel["SPY_return"], 63)
    rows: list[dict[str, object]] = []
    for pid, (s, e) in enumerate(find_episodes(state["credit_lock_active"].astype(bool)), start=1):
        unlock_idx = min(e + 1, len(panel) - 1)
        trough = float(panel.loc[s:e, "spy_price"].min())
        unlock_price = float(panel.loc[unlock_idx, "spy_price"])
        trough_to_unlock = unlock_price / trough - 1.0 if trough > 0 else np.nan
        dominant = panel.loc[s:e, "macro_regime_confirmed"].mode()
        false_recovery = bool(
            (pd.notna(m21.iloc[unlock_idx]) and m21.iloc[unlock_idx] <= -0.05)
            or (pd.notna(m63.iloc[unlock_idx]) and m63.iloc[unlock_idx] <= -0.08)
            or state.iloc[unlock_idx + 1 : min(unlock_idx + 64, len(state))]["credit_lock_active"].astype(bool).any()
        )
        rows.append(
            {
                "variant": variant.name,
                "period_id": pid,
                "entry_date": panel.loc[s, "date"],
                "unlock_date": panel.loc[unlock_idx, "date"],
                "duration_days": int(e - s + 1),
                "macro_regime_at_entry": panel.loc[s, "macro_regime_confirmed"],
                "dominant_macro_regime": dominant.iloc[0] if len(dominant) else np.nan,
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
                "next_21d_SPY_return_after_unlock": r21.iloc[unlock_idx],
                "next_21d_SPY_maxDD_after_unlock": m21.iloc[unlock_idx],
                "next_63d_SPY_return_after_unlock": r63.iloc[unlock_idx],
                "next_63d_SPY_maxDD_after_unlock": m63.iloc[unlock_idx],
                "false_recovery_flag": false_recovery,
                "missed_rebound_flag": bool(pd.notna(trough_to_unlock) and trough_to_unlock > 0.08),
            }
        )
    return pd.DataFrame(rows)


def perf_row(frame: pd.DataFrame, periods: pd.DataFrame, variant: CreditVariant) -> dict[str, object]:
    p = performance_metrics(frame, variant.name)
    return {
        "variant": variant.name,
        "unlock_z_threshold": variant.unlock_z_threshold,
        "confirm_days": variant.confirm_days,
        "trend_rule": variant.trend_rule,
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


def crisis_rows(frame: pd.DataFrame, periods: pd.DataFrame, variant: CreditVariant) -> list[dict[str, object]]:
    rows = []
    for window, (start, end) in WINDOWS.items():
        sub = frame.loc[frame["date"] >= pd.Timestamp(start)].copy()
        if end is not None:
            sub = sub.loc[sub["date"] <= pd.Timestamp(end)]
        sub_periods = periods.loc[
            (periods["entry_date"] <= (pd.Timestamp(end) if end is not None else frame["date"].max()))
            & (periods["unlock_date"] >= pd.Timestamp(start))
        ]
        rows.append(
            {
                "variant": variant.name,
                "window": window,
                "cumulative_return": period_return(sub[f"{variant.name}_return"]),
                "max_drawdown": period_mdd(sub[f"{variant.name}_return"]),
                "Sharpe": performance_metrics(sub, variant.name)["Sharpe"] if len(sub) else np.nan,
                "time_in_credit_lock": int(sub["credit_lock_active"].sum()),
                "number_credit_periods": int(len(sub_periods)),
                "false_recovery_count": int(sub_periods["false_recovery_flag"].sum()) if len(sub_periods) else 0,
                "missed_rebound_count": int(sub_periods["missed_rebound_flag"].sum()) if len(sub_periods) else 0,
            }
        )
    return rows


def plot_heatmap(perf: pd.DataFrame, metric: str, title: str, out_path: Path) -> None:
    pivot = perf.pivot_table(index="confirm_days", columns="unlock_z_threshold", values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("unlock z threshold")
    ax.set_ylabel("confirm days")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_trend_bar(perf: pd.DataFrame, metric: str, out_path: Path) -> None:
    summary = perf.groupby("trend_rule")[metric].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(summary.index, summary.values)
    ax.set_title(f"Mean {metric} by trend rule")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_top_curves(frames: dict[str, pd.DataFrame], top_names: list[str]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for name in top_names:
        df = frames[name]
        axes[0].plot(df["date"], df[f"{name}_nav"], label=name, linewidth=1.0)
        axes[1].plot(df["date"], df[f"{name}_drawdown"], label=name, linewidth=1.0)
    axes[0].set_title("Top credit-only challengers NAV")
    axes[1].set_title("Top credit-only challengers drawdown")
    axes[1].legend(loc="lower left", ncol=2, fontsize=8)
    axes[0].legend(loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "top_credit_unlock_challengers.png", dpi=160)
    plt.close(fig)


def build_report(perf: pd.DataFrame, crisis: pd.DataFrame, best: pd.Series) -> None:
    lines = [
        "# Credit Lock Narrow Unlock Test",
        "",
        "This diagnostic isolates the ABS_ENTRY_LEVEL_Z_UNLOCK credit lock in a SPY/CASH credit-only framework.",
        "Credit remains enabled only in the same partial-regime scope as the mainline: FLAT_LOW_RATE and FLAT_HIGH_RATE.",
        "",
        "## Search scope",
        "",
        "- Unlock z-threshold around 1.0: 0.8 / 0.9 / 1.0 / 1.1 / 1.2",
        "- Consecutive unlock confirmation days: 1 / 2 / 3",
        "- SPY trend confirmation: MA20 / MA50 / MA20_AND_MA50",
        "",
        "## Best balanced candidate",
        "",
        f"- Variant: `{best['variant']}`",
        f"- Sharpe: {best['Sharpe']:.3f}",
        f"- MaxDD: {best['MaxDD']:.2%}",
        f"- Final Equity: {best['Final Equity']:.2f}",
        f"- False recovery count: {int(best['false_recovery_count'])}",
        f"- Missed rebound count: {int(best['missed_rebound_count'])}",
        "",
        "## Window summary",
        "",
    ]
    for window in WINDOWS:
        sub = crisis.loc[(crisis["variant"] == best["variant"]) & (crisis["window"] == window)]
        if sub.empty:
            continue
        row = sub.iloc[0]
        lines.append(
            f"- {window}: return {row['cumulative_return']:.2%}, maxDD {row['max_drawdown']:.2%}, "
            f"false recovery {int(row['false_recovery_count'])}, missed rebound {int(row['missed_rebound_count'])}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This script is intentionally narrow. It does not search entry rules, regime scope, or hedge allocation.",
            "It only tests whether a small unlock-side adjustment can improve the current ABS_ENTRY_LEVEL_Z_UNLOCK credit lock.",
        ]
    )
    (OUT / "README_credit_lock_narrow_unlock_test.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    mod = load_redesign_module()
    panel = load_panel(mod)
    variants = build_variants()

    perf_rows: list[dict[str, object]] = []
    crisis_rows_all: list[dict[str, object]] = []
    period_frames: list[pd.DataFrame] = []
    frames: dict[str, pd.DataFrame] = {}

    for variant in variants:
        state = simulate_credit_only(panel, variant)
        frame = strategy_frame(panel, state, variant.name)
        periods = extract_periods(panel, state, variant)
        perf_rows.append(perf_row(frame, periods, variant))
        crisis_rows_all.extend(crisis_rows(frame, periods, variant))
        if not periods.empty:
            period_frames.append(periods)
        frames[variant.name] = frame

    perf = pd.DataFrame(perf_rows).sort_values(["Sharpe", "Final Equity"], ascending=[False, False]).reset_index(drop=True)
    crisis = pd.DataFrame(crisis_rows_all)
    periods_all = pd.concat(period_frames, ignore_index=True) if period_frames else pd.DataFrame()

    perf["Sharpe_rank"] = perf["Sharpe"].rank(ascending=False, method="min")
    perf["MaxDD_rank"] = perf["MaxDD"].rank(ascending=False, method="min")
    perf["FinalEquity_rank"] = perf["Final Equity"].rank(ascending=False, method="min")
    perf["FalseRecovery_rank"] = perf["false_recovery_count"].rank(ascending=True, method="min")
    perf["MissedRebound_rank"] = perf["missed_rebound_count"].rank(ascending=True, method="min")
    perf["composite_score"] = (
        0.30 * perf["Sharpe_rank"]
        + 0.25 * perf["MaxDD_rank"]
        + 0.20 * perf["FinalEquity_rank"]
        + 0.15 * perf["FalseRecovery_rank"]
        + 0.10 * perf["MissedRebound_rank"]
    )
    perf = perf.sort_values(["composite_score", "Sharpe"], ascending=[True, False]).reset_index(drop=True)

    perf.to_csv(OUT / "credit_unlock_narrow_performance.csv", index=False)
    crisis.to_csv(OUT / "credit_unlock_narrow_crisis.csv", index=False)
    periods_all.to_csv(OUT / "credit_unlock_narrow_periods.csv", index=False)

    plot_heatmap(perf, "Sharpe", "Sharpe by unlock z-threshold and confirm days", FIG / "credit_unlock_sharpe_heatmap.png")
    plot_heatmap(perf, "MaxDD", "MaxDD by unlock z-threshold and confirm days", FIG / "credit_unlock_maxdd_heatmap.png")
    plot_trend_bar(perf, "Sharpe", FIG / "credit_unlock_trend_rule_bar.png")

    top_names = perf.head(5)["variant"].tolist()
    plot_top_curves(frames, top_names)

    best = perf.iloc[0]
    build_report(perf, crisis, best)

    print("top 10 narrow credit unlock challengers")
    print(
        perf[
            [
                "variant",
                "unlock_z_threshold",
                "confirm_days",
                "trend_rule",
                "Sharpe",
                "MaxDD",
                "Final Equity",
                "false_recovery_count",
                "missed_rebound_count",
                "composite_score",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    print("output path:", OUT)


if __name__ == "__main__":
    main()
