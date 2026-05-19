from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_strategy_source_only_core import (
    ASSETS,
    FINAL_STRATEGY,
    GS10_THRESHOLD,
    ONE_WAY_COST_BPS,
    RECOVERY_WINDOW,
    ROOT,
    apply_flat_low_recovery,
    base_refined_weights,
    build_backbone_and_states,
    build_source_panel,
    compute_strategy,
    confirm_state,
    performance_metrics,
)


OUT = ROOT / "results" / "turnover_diagnostics_and_reduction"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
MAIN_TABLE_DIR = ROOT / "results" / "main_pipeline_final" / "tables"
ORIGINAL = "FINAL_ORIGINAL"


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_canonical_panel() -> pd.DataFrame:
    required = [
        MAIN_TABLE_DIR / "final_daily_weights.csv",
        MAIN_TABLE_DIR / "final_daily_returns.csv",
        MAIN_TABLE_DIR / "source_canonical_panel.csv",
    ]
    if not all(path.exists() for path in required):
        panel = build_source_panel()
        panel = build_backbone_and_states(panel)
        base_weights, base_states = base_refined_weights(panel)
        final_weights, recovery_active = apply_flat_low_recovery(panel, base_weights, base_states)
        final = compute_strategy(panel, final_weights, FINAL_STRATEGY)
        panel = pd.concat([panel, final], axis=1)
        panel["flat_refined_state"] = base_states
        panel["recovery_flat_low_active"] = recovery_active
        panel["final_state"] = np.where(
            panel["full_risk_state"].eq("FULL_RISK"),
            "FULL_RISK",
            np.where(panel["recovery_flat_low_active"], "RECOVERY_20D_FLAT_LOW", "NON_RISK"),
        )
        return panel

    # Still rebuild full state fields from source-only core; main outputs are used
    # as the canonical source-only checkpoint but do not contain all diagnostic fields.
    panel = build_source_panel()
    panel = build_backbone_and_states(panel)
    base_weights, base_states = base_refined_weights(panel)
    final_weights, recovery_active = apply_flat_low_recovery(panel, base_weights, base_states)
    final = compute_strategy(panel, final_weights, FINAL_STRATEGY)
    panel = pd.concat([panel, final], axis=1)
    panel["flat_refined_state"] = base_states
    panel["recovery_flat_low_active"] = recovery_active
    panel["final_state"] = np.where(
        panel["full_risk_state"].eq("FULL_RISK"),
        "FULL_RISK",
        np.where(panel["recovery_flat_low_active"], "RECOVERY_20D_FLAT_LOW", "NON_RISK"),
    )
    return panel


def final_weight_frame(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[[f"{FINAL_STRATEGY}_weight_{asset}" for asset in ASSETS]].rename(
        columns={f"{FINAL_STRATEGY}_weight_{asset}": asset for asset in ASSETS}
    )


def normalize_weights(row: pd.Series) -> pd.Series:
    clean = row.reindex(ASSETS).fillna(0.0).clip(lower=0.0)
    total = clean.sum()
    if total <= 0:
        return pd.Series({asset: 1.0 / len(ASSETS) for asset in ASSETS})
    return clean / total


def label_allocation(row: pd.Series) -> str:
    if bool(row.get("recovery_flat_low_active", False)):
        return "RECOVERY_20D_FLAT_LOW"
    return str(row.get("flat_refined_state", row.get("final_state", "UNKNOWN")))


def likely_reason(row: pd.Series) -> str:
    if row["is_stress_entry"]:
        return "FULL_RISK_ENTRY"
    if row["is_stress_exit"]:
        return "FULL_RISK_EXIT_R3"
    if row["is_recovery_entry"]:
        return "RECOVERY_ENTRY"
    if row["is_recovery_exit"]:
        return "RECOVERY_EXIT"
    if row["is_flat_low_high_switch"]:
        return "FLAT_LOW_HIGH_SWITCH"
    if row["is_steep_overlay_entry"]:
        return "STEEP_SLOW_GROWTH_OVERLAY_ENTRY"
    if row["is_steep_overlay_exit"]:
        return "STEEP_SLOW_GROWTH_OVERLAY_EXIT"
    if (
        row["turnover"] > 1e-8
        and row["previous_allocation_label"] == row["current_allocation_label"]
        and str(row["current_allocation_label"]).endswith("_NORMAL")
    ):
        return "INVERSE_VOL_REBALANCE"
    return "OTHER"


def build_turnover_diagnostics(panel: pd.DataFrame) -> pd.DataFrame:
    weights = final_weight_frame(panel)
    turnover = 0.5 * weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = 0.5 * weights.iloc[0].abs().sum()
    out = pd.DataFrame(
        {
            "date": panel["date"],
            "turnover": turnover,
            "strategy_return": panel[f"{FINAL_STRATEGY}_return"],
            "equity": panel[f"{FINAL_STRATEGY}_nav"],
            "drawdown": panel[f"{FINAL_STRATEGY}_drawdown"],
            "previous_refined_regime": panel["refined_regime_confirmed"].shift(1),
            "current_refined_regime": panel["refined_regime_confirmed"],
            "previous_stress_state": panel["full_risk_state"].shift(1),
            "current_stress_state": panel["full_risk_state"],
            "previous_allocation_label": panel.apply(label_allocation, axis=1).shift(1),
            "current_allocation_label": panel.apply(label_allocation, axis=1),
            "VIX_ZSCORE_120D": panel["VIX_ZSCORE_120D"],
            "D_CREDIT_SPREAD_20D": panel["D_CREDIT_SPREAD_20D"],
            "CMDTY_RET60": panel["CMDTY_RET60"],
            "GS10": panel["GS10"],
            "term_spread": panel["TERM_SPREAD_10Y_1Y"],
            "SPY_return_1d": panel["SPY_return"],
            "SPY_return_5d": panel["SPY_return"].rolling(5).sum(),
            "SPY_return_20d": panel["SPY_return"].rolling(20).sum(),
        }
    )
    out["is_stress_entry"] = out["current_stress_state"].eq("FULL_RISK") & ~out["previous_stress_state"].eq("FULL_RISK")
    out["is_stress_exit"] = ~out["current_stress_state"].eq("FULL_RISK") & out["previous_stress_state"].eq("FULL_RISK")
    recovery = panel["recovery_flat_low_active"].fillna(False).astype(bool)
    out["is_recovery_entry"] = recovery & ~recovery.shift(1, fill_value=False)
    out["is_recovery_exit"] = ~recovery & recovery.shift(1, fill_value=False)
    out["is_flat_low_high_switch"] = (
        out["previous_refined_regime"].isin(["FLAT_LOW_RATE", "FLAT_HIGH_RATE"])
        & out["current_refined_regime"].isin(["FLAT_LOW_RATE", "FLAT_HIGH_RATE"])
        & out["previous_refined_regime"].ne(out["current_refined_regime"])
    )
    overlay = panel["steep_slow_growth_overlay_state"].fillna(False).astype(bool)
    out["is_steep_overlay_entry"] = overlay & ~overlay.shift(1, fill_value=False)
    out["is_steep_overlay_exit"] = ~overlay & overlay.shift(1, fill_value=False)
    out["trigger_reason"] = out.apply(likely_reason, axis=1)
    return out


def weight_string(row: pd.Series) -> str:
    vals = {asset: float(row[asset]) for asset in ASSETS if abs(float(row[asset])) > 1e-6}
    return "; ".join(f"{k}:{v:.2%}" for k, v in vals.items()) if vals else "NONE"


def top_turnover_dates(diag: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    before = weights.shift(1).fillna(0.0)
    rows = []
    for idx, row in diag.nlargest(50, "turnover").iterrows():
        rows.append(
            {
                "date": row["date"],
                "turnover": row["turnover"],
                "before_weights": weight_string(before.loc[idx]),
                "after_weights": weight_string(weights.loc[idx]),
                "regime_transition": f"{row['previous_refined_regime']} -> {row['current_refined_regime']}",
                "stress_transition": f"{row['previous_stress_state']} -> {row['current_stress_state']}",
                "likely_reason": row["trigger_reason"],
            }
        )
    return pd.DataFrame(rows)


def forward_return(series: pd.Series, window: int) -> pd.Series:
    return (1.0 + series).rolling(window).apply(np.prod, raw=True).shift(-window) - 1.0


def summarize_turnover(diag: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tmp = diag.copy()
    tmp["return_next_5d"] = forward_return(tmp["strategy_return"], 5)
    tmp["return_next_20d"] = forward_return(tmp["strategy_return"], 20)
    total = tmp["turnover"].sum()
    by_reason = (
        tmp.groupby("trigger_reason", dropna=False)
        .agg(
            count=("turnover", "size"),
            total_turnover=("turnover", "sum"),
            average_turnover=("turnover", "mean"),
            median_turnover=("turnover", "median"),
            average_return_next_5d=("return_next_5d", "mean"),
            average_return_next_20d=("return_next_20d", "mean"),
        )
        .reset_index()
        .rename(columns={"trigger_reason": "reason"})
    )
    by_reason["share_of_total_turnover"] = by_reason["total_turnover"] / total if total else 0.0
    by_reason = by_reason.sort_values("total_turnover", ascending=False)

    tmp["previous_regime"] = tmp["previous_refined_regime"].fillna("START")
    tmp["current_regime"] = tmp["current_refined_regime"].fillna("UNKNOWN")
    tmp["drawdown_next_20d"] = tmp["drawdown"].rolling(20).min().shift(-20)
    by_transition = (
        tmp.groupby(["previous_regime", "current_regime"], dropna=False)
        .agg(
            count=("turnover", "size"),
            total_turnover=("turnover", "sum"),
            avg_turnover=("turnover", "mean"),
            mean_strategy_return_next_20d=("return_next_20d", "mean"),
            mean_drawdown_next_20d=("drawdown_next_20d", "mean"),
        )
        .reset_index()
        .sort_values("total_turnover", ascending=False)
    )
    return by_reason, by_transition


def apply_buffer(target: pd.DataFrame, stress_entry: pd.Series, threshold: float) -> pd.DataFrame:
    out = pd.DataFrame(index=target.index, columns=ASSETS, dtype=float)
    current = normalize_weights(target.iloc[0])
    out.iloc[0] = current
    for i in range(1, len(target)):
        tgt = normalize_weights(target.iloc[i])
        if bool(stress_entry.iloc[i]):
            current = tgt
        else:
            proposed = current.copy()
            for asset in ASSETS:
                if abs(float(tgt[asset]) - float(current[asset])) >= threshold:
                    proposed[asset] = tgt[asset]
            current = normalize_weights(proposed)
        out.iloc[i] = current
    return out.astype(float)


def apply_min_hold(target: pd.DataFrame, panel: pd.DataFrame, hold_days: int) -> pd.DataFrame:
    out = pd.DataFrame(index=target.index, columns=ASSETS, dtype=float)
    current = normalize_weights(target.iloc[0])
    out.iloc[0] = current
    days_since_trade = hold_days
    full_risk = panel["full_risk_state"].eq("FULL_RISK")
    stress_entry = full_risk & ~full_risk.shift(1, fill_value=False)
    stress_state = full_risk | panel["flat_refined_state"].astype(str).str.endswith("_STRESS")
    overlay = panel["steep_slow_growth_overlay_state"].fillna(False).astype(bool)
    overlay_entry = overlay & ~overlay.shift(1, fill_value=False)
    for i in range(1, len(target)):
        tgt = normalize_weights(target.iloc[i])
        immediate = bool(stress_entry.iloc[i] or stress_state.iloc[i] or overlay_entry.iloc[i])
        changed = (tgt - current).abs().sum() > 1e-8
        if immediate or (changed and days_since_trade >= hold_days):
            current = tgt
            days_since_trade = 0
        else:
            days_since_trade += 1
        out.iloc[i] = current
    return out.astype(float)


def apply_recovery_with_cooldown(df: pd.DataFrame, base_weights: pd.DataFrame, base_states: pd.Series, cooldown_days: int) -> pd.DataFrame:
    weights = base_weights.copy()
    remaining = 0
    cooldown = 0
    was_stress = False
    for i, row in df.iterrows():
        is_stress = bool(base_states.iloc[i].endswith("_STRESS") or row["full_risk_state"] == "FULL_RISK")
        is_flat_low_normal = base_states.iloc[i] == "FLAT_LOW_RATE_NORMAL"
        if is_stress:
            remaining = 0
        else:
            if was_stress and is_flat_low_normal and cooldown <= 0:
                remaining = RECOVERY_WINDOW
            if remaining > 0 and is_flat_low_normal:
                selected = [asset for asset in ["SPY", "CMDTY_FUT", "GOLD"] if pd.notna(row.get(f"{asset}_return", np.nan))]
                if selected:
                    weights.loc[i, ASSETS] = 0.0
                    for asset in selected:
                        weights.loc[i, asset] = 1.0 / len(selected)
                remaining -= 1
                if remaining == 0:
                    cooldown = cooldown_days
            else:
                remaining = 0
                if cooldown > 0:
                    cooldown -= 1
        was_stress = is_stress
    return weights.astype(float)


def build_hysteresis_final_weights() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = build_source_panel()
    refined = []
    state = None
    for _, row in panel.iterrows():
        macro = row["macro_regime_raw"]
        if macro != "FLAT":
            raw = macro
        else:
            if state == "FLAT_HIGH_RATE":
                raw = "FLAT_LOW_RATE" if row["GS10"] < 2.8 else "FLAT_HIGH_RATE"
            elif state == "FLAT_LOW_RATE":
                raw = "FLAT_HIGH_RATE" if row["GS10"] > 3.0 else "FLAT_LOW_RATE"
            else:
                raw = "FLAT_HIGH_RATE" if row["GS10"] > GS10_THRESHOLD else "FLAT_LOW_RATE"
        refined.append(raw)
        state = raw
    panel["refined_regime_raw"] = refined
    panel["refined_regime_confirmed"] = confirm_state(refined, initial=str(refined[0]))
    panel = build_backbone_and_states(panel)
    base_weights, base_states = base_refined_weights(panel)
    final_weights, recovery_active = apply_flat_low_recovery(panel, base_weights, base_states)
    panel["flat_refined_state"] = base_states
    panel["recovery_flat_low_active"] = recovery_active
    return panel, final_weights


def compute_experiment_panel(panel: pd.DataFrame, strategies: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_parts = [panel[["date"]].copy()]
    rows = []
    for name, weights in strategies.items():
        strat = compute_strategy(panel, weights[ASSETS].astype(float), name)
        all_parts.append(strat)
        metrics = performance_metrics(strat, name)
        # The canonical core stores full two-sided turnover for cost math.
        # This diagnostic reports one-way portfolio turnover as requested:
        # 0.5 * sum(abs(weight_t - weight_{t-1})).
        metrics["turnover"] = float(strat[f"{name}_turnover"].sum() * 0.5)
        rows.append(metrics)
    return pd.concat(all_parts, axis=1), pd.DataFrame(rows)


def run_turnover_experiments(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = final_weight_frame(panel)
    full_risk = panel["full_risk_state"].eq("FULL_RISK")
    stress_entry = full_risk & ~full_risk.shift(1, fill_value=False)
    base_weights, base_states = base_refined_weights(panel)
    strategies: dict[str, pd.DataFrame] = {ORIGINAL: target}
    for label, threshold in [("TURNOVER_BUFFER_2P5", 0.025), ("TURNOVER_BUFFER_5", 0.05), ("TURNOVER_BUFFER_10", 0.10)]:
        strategies[label] = apply_buffer(target, stress_entry, threshold)
    for label, days in [("MIN_HOLD_NORMAL_5D", 5), ("MIN_HOLD_NORMAL_10D", 10)]:
        strategies[label] = apply_min_hold(target, panel, days)
    for label, days in [("RECOVERY_COOLDOWN_10D", 10), ("RECOVERY_COOLDOWN_20D", 20)]:
        strategies[label] = apply_recovery_with_cooldown(panel, base_weights, base_states, days)

    hyst_panel, hyst_weights = build_hysteresis_final_weights()
    # Hysteresis changes only the execution rule. Use the same source date span.
    strategies["FLAT_RATE_HYSTERESIS_2P8_3P0"] = hyst_weights.reindex(panel.index)[ASSETS].astype(float)

    all_returns, perf = compute_experiment_panel(panel, strategies)
    original = perf.loc[perf["strategy"].eq(ORIGINAL)].iloc[0]
    perf["turnover_reduction_pct_vs_original"] = (original["turnover"] - perf["turnover"]) / original["turnover"]
    for col in ["CAGR", "Sharpe", "MaxDD", "final_equity"]:
        perf[f"{col}_diff_vs_original"] = perf[col] - original[col]
    return all_returns, perf


def recommendation(perf: pd.DataFrame) -> pd.DataFrame:
    original = perf.loc[perf["strategy"].eq(ORIGINAL)].iloc[0]
    candidates = perf.loc[~perf["strategy"].eq(ORIGINAL)].copy()
    ok = candidates[
        (candidates["turnover_reduction_pct_vs_original"] >= 0.10)
        & (candidates["MaxDD"] >= original["MaxDD"] - 0.0025)
        & (candidates["Sharpe"] >= original["Sharpe"] - 0.03)
        & (candidates["CAGR"] >= original["CAGR"] - 0.0030)
    ].copy()
    if ok.empty:
        return pd.DataFrame(
            [
                {
                    "recommendation": "No recommended replacement",
                    "reason": "No tested execution rule reduced turnover by at least 10% while preserving MaxDD, Sharpe, and CAGR within the requested tolerances. Keep canonical final strategy.",
                }
            ]
        )
    ok = ok.sort_values(["turnover_reduction_pct_vs_original", "Sharpe", "Calmar", "final_equity"], ascending=[False, False, False, False])
    ok.insert(0, "recommendation", "Recommended execution-enhanced variant")
    return ok.head(5)


def save_plots(panel: pd.DataFrame, diag: pd.DataFrame, top: pd.DataFrame, by_reason: pd.DataFrame, by_transition: pd.DataFrame, exp_panel: pd.DataFrame, perf: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(diag["date"], diag["turnover"], lw=0.8)
    ax.set_title("Daily Turnover")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "daily_turnover_timeline.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    plot_top = top.head(25).copy()
    ax.bar(plot_top["date"].astype(str), plot_top["turnover"])
    ax.tick_params(axis="x", labelrotation=70)
    ax.set_title("Top Turnover Dates")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "top_turnover_dates_bar.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(by_reason["reason"], by_reason["total_turnover"])
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_title("Turnover by Reason")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "turnover_by_reason.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    top_trans = by_transition.head(15).copy()
    labels = top_trans["previous_regime"].astype(str) + " -> " + top_trans["current_regime"].astype(str)
    ax.bar(labels, top_trans["total_turnover"])
    ax.tick_params(axis="x", labelrotation=55)
    ax.set_title("Turnover by Regime Transition")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "turnover_by_regime_transition.png", dpi=160)
    plt.close(fig)

    weights = final_weight_frame(panel)
    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax1.stackplot(panel["date"], *[weights[a] for a in ASSETS], labels=ASSETS, alpha=0.8)
    ax2 = ax1.twinx()
    spike = diag["turnover"] >= diag["turnover"].quantile(0.99)
    ax2.scatter(diag.loc[spike, "date"], diag.loc[spike, "turnover"], color="black", s=10, label="Top 1% turnover")
    ax1.set_title("Weights Timeline with Turnover Spikes")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "weights_timeline_with_turnover_spikes.png", dpi=160)
    plt.close(fig)

    plot_strats = perf.sort_values("final_equity", ascending=False)["strategy"].tolist()
    fig, ax = plt.subplots(figsize=(12, 5))
    for strategy in plot_strats:
        ax.plot(exp_panel["date"], exp_panel[f"{strategy}_nav"], label=strategy, lw=1.0)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_title("Equity Curve: Turnover Reduction Experiments")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "equity_curve_turnover_reduction_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for strategy in plot_strats:
        ax.plot(exp_panel["date"], exp_panel[f"{strategy}_drawdown"], label=strategy, lw=1.0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_title("Drawdown: Turnover Reduction Experiments")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "drawdown_curve_turnover_reduction_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(perf["turnover"], perf["Sharpe"])
    for _, row in perf.iterrows():
        ax.annotate(row["strategy"], (row["turnover"], row["Sharpe"]), fontsize=7)
    ax.set_xlabel("Turnover")
    ax.set_ylabel("Sharpe")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "turnover_vs_sharpe_scatter.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(perf["turnover"], perf["CAGR"])
    for _, row in perf.iterrows():
        ax.annotate(row["strategy"], (row["turnover"], row["CAGR"]), fontsize=7)
    ax.set_xlabel("Turnover")
    ax.set_ylabel("CAGR")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "turnover_vs_cagr_scatter.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    red = perf.loc[~perf["strategy"].eq(ORIGINAL)].sort_values("turnover_reduction_pct_vs_original", ascending=False)
    ax.bar(red["strategy"], red["turnover_reduction_pct_vs_original"])
    ax.tick_params(axis="x", labelrotation=45)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("Turnover Reduction vs Original")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "turnover_reduction_bar.png", dpi=160)
    plt.close(fig)


def write_readme(original_turnover: float, by_reason: pd.DataFrame, perf: pd.DataFrame, rec: pd.DataFrame) -> None:
    top_reasons = by_reason.head(5).copy()
    best = perf.loc[~perf["strategy"].eq(ORIGINAL)].sort_values("turnover_reduction_pct_vs_original", ascending=False).head(1)
    recommendation_exists = not rec["recommendation"].astype(str).str.contains("No recommended replacement").any()
    lines = [
        "# Turnover Diagnostics and Reduction",
        "",
        "This independent diagnostic uses the source-only canonical final pipeline output and does not modify the canonical final strategy.",
        "",
        "## Why This Diagnostic Exists",
        "",
        "The final strategy performs well, but the weight timeline shows periods of rapid allocation changes. This diagnostic decomposes turnover by trigger and tests simple post-processing execution rules.",
        "",
        "## Main Turnover Sources",
        "",
        f"Original final strategy total turnover: `{original_turnover:.4f}`.",
        "",
        top_reasons[["reason", "count", "total_turnover", "share_of_total_turnover"]].to_markdown(index=False),
        "",
        "## Turnover Reduction Tests",
        "",
        "Tests include small weight-change buffers, normal-state minimum holding periods, a FLAT rate hysteresis band, and recovery cooldowns. Stress allocations remain priority and are not delayed by the buffer rules.",
        "",
        perf[["strategy", "CAGR", "Sharpe", "MaxDD", "Calmar", "final_equity", "turnover", "turnover_reduction_pct_vs_original"]].to_markdown(index=False),
        "",
        "## Recommendation",
        "",
    ]
    if recommendation_exists:
        lines.append("At least one execution-layer variant met the requested turnover, MaxDD, Sharpe, and CAGR constraints. Treat it as an execution-enhanced candidate, not a change to the underlying signal logic.")
        lines.append("")
        lines.append(rec.head(5).to_markdown(index=False))
    else:
        lines.append("No tested execution-layer rule met all requested constraints. The canonical final strategy should remain unchanged.")
    if not best.empty:
        row = best.iloc[0]
        lines.extend(
            [
                "",
                "## Best Raw Turnover Reducer",
                "",
                f"`{row['strategy']}` reduced turnover by `{row['turnover_reduction_pct_vs_original']:.1%}` with Sharpe `{row['Sharpe']:.3f}` and MaxDD `{row['MaxDD']:.2%}`.",
            ]
        )
    (OUT / "README_turnover_diagnostics_and_reduction.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = load_canonical_panel()
    panel["date"] = pd.to_datetime(panel["date"])
    diag = build_turnover_diagnostics(panel)
    weights = final_weight_frame(panel)
    top = top_turnover_dates(diag, weights)
    by_reason, by_transition = summarize_turnover(diag)
    exp_panel, perf = run_turnover_experiments(panel)
    rec = recommendation(perf)

    diag.to_csv(TABLE_DIR / "daily_turnover_diagnostics.csv", index=False)
    top.to_csv(TABLE_DIR / "top_turnover_dates.csv", index=False)
    by_reason.to_csv(TABLE_DIR / "turnover_by_reason.csv", index=False)
    by_transition.to_csv(TABLE_DIR / "turnover_by_regime_transition.csv", index=False)
    perf.to_csv(TABLE_DIR / "turnover_reduction_performance_comparison.csv", index=False)
    rec.to_csv(TABLE_DIR / "turnover_reduction_recommendation.csv", index=False)
    exp_panel.to_csv(TABLE_DIR / "turnover_reduction_daily_returns.csv", index=False)

    save_plots(panel, diag, top, by_reason, by_transition, exp_panel, perf)
    original_turnover = float(perf.loc[perf["strategy"].eq(ORIGINAL), "turnover"].iloc[0])
    write_readme(original_turnover, by_reason, perf, rec)

    best_candidate = perf.loc[~perf["strategy"].eq(ORIGINAL)].sort_values(
        ["turnover_reduction_pct_vs_original", "Sharpe"], ascending=[False, False]
    ).iloc[0]
    print("Original final strategy turnover:", f"{original_turnover:.4f}")
    print("Top 5 turnover reasons:")
    print(by_reason.head(5)[["reason", "total_turnover", "share_of_total_turnover"]].to_string(index=False))
    print("Top 10 turnover dates:")
    print(top.head(10)[["date", "turnover", "likely_reason"]].to_string(index=False))
    print("Best turnover reduction candidate:", best_candidate["strategy"])
    print("Recommendation exists:", not rec["recommendation"].astype(str).str.contains("No recommended replacement").any())
    print("Output directory:", OUT)


if __name__ == "__main__":
    main()
