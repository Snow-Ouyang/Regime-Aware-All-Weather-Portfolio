from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_strategy_source_only_core import (
    ASSETS,
    FINAL_STRATEGY,
    ROOT,
    SPY_BUY_HOLD,
    SPY_CASH_TIMING,
    apply_flat_low_recovery,
    base_refined_weights,
    compute_strategy,
    monthly_hold_weights,
    normalize_weight_dict,
    performance_metrics,
)


OUT = ROOT / "results" / "regime_single_trigger_r3_2d_confirm"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
MAIN_PANEL = ROOT / "results" / "main_pipeline_final" / "daily_backtest_panel.csv"
NEW_STRATEGY = "REGIME_SINGLE_TRIGGER_R3_2D_CONFIRM"
CANONICAL = "RECOVERY_20D_EQUAL_WEIGHT_FLAT_LOW_ONLY"
DISPLAY = [SPY_BUY_HOLD, SPY_CASH_TIMING, CANONICAL, NEW_STRATEGY]


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
    return panel


def refined_regime(panel: pd.DataFrame) -> pd.Series:
    if "refined_regime_confirmed" in panel.columns:
        return panel["refined_regime_confirmed"].where(
            panel["refined_regime_confirmed"].isin(["FLAT_LOW_RATE", "FLAT_HIGH_RATE", "STEEP", "INVERTED"]),
            "OTHER",
        )
    base = panel["macro_regime_confirmed"].fillna("OTHER")
    return pd.Series(
        np.select(
            [
                base.eq("FLAT") & (panel["GS10"] <= 2.9),
                base.eq("FLAT") & (panel["GS10"] > 2.9),
                base.eq("STEEP"),
                base.eq("INVERTED"),
            ],
            ["FLAT_LOW_RATE", "FLAT_HIGH_RATE", "STEEP", "INVERTED"],
            default="OTHER",
        ),
        index=panel.index,
    )


def build_single_trigger_signals(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["refined_regime"] = refined_regime(out)
    out["SINGLE_FLAT_LOW_CMDTY_ENTRY"] = out["refined_regime"].eq("FLAT_LOW_RATE") & (out["CMDTY_RET60"] < -0.10)
    out["SINGLE_FLAT_HIGH_CREDIT_ENTRY"] = out["refined_regime"].eq("FLAT_HIGH_RATE") & (
        (out["spy_drawdown_from_previous_high"] <= -0.05) & (out["D_CREDIT_SPREAD_20D"] > 0.10)
    )
    out["SINGLE_STEEP_VIX_ENTRY"] = out["refined_regime"].eq("STEEP") & (out["VIX_ZSCORE_120D"] >= 3.0)
    out["SINGLE_ENTRY_SIGNAL"] = (
        out["SINGLE_FLAT_LOW_CMDTY_ENTRY"] | out["SINGLE_FLAT_HIGH_CREDIT_ENTRY"] | out["SINGLE_STEEP_VIX_ENTRY"]
    )
    above = out["spy_price"] > out["SPY_MA20"]
    out["R3_2D_CONFIRM_SIGNAL"] = above & above.shift(1, fill_value=False)
    return out


def trigger_used(row: pd.Series) -> str:
    if bool(row.get("SINGLE_FLAT_LOW_CMDTY_ENTRY", False)):
        return "FLAT_LOW_CMDTY"
    if bool(row.get("SINGLE_FLAT_HIGH_CREDIT_ENTRY", False)):
        return "FLAT_HIGH_CREDIT"
    if bool(row.get("SINGLE_STEEP_VIX_ENTRY", False)):
        return "STEEP_VIX"
    return "NONE"


def build_single_full_risk_state(panel: pd.DataFrame) -> tuple[pd.Series, list[dict]]:
    states: list[str] = []
    events: list[dict] = []
    state = "NON_RISK"
    pending = "NON_RISK"
    prev_state_for_event = "NON_RISK"
    event_id = 1
    for i, row in panel.iterrows():
        state = pending
        states.append(state)
        if state != prev_state_for_event:
            event_type = "FULL_RISK_ENTRY" if state == "FULL_RISK" else "FULL_RISK_EXIT_R3_2D_CONFIRM"
            signal_row = panel.loc[max(i - 1, 0)]
            events.append(
                {
                    "event_id": event_id,
                    "date": row["date"],
                    "event_type": event_type,
                    "refined_regime": row["refined_regime"],
                    "trigger_used": trigger_used(signal_row) if event_type == "FULL_RISK_ENTRY" else "NONE",
                    "VIX_ZSCORE_120D": row["VIX_ZSCORE_120D"],
                    "D_CREDIT_SPREAD_20D": row["D_CREDIT_SPREAD_20D"],
                    "spy_drawdown_from_previous_high": row["spy_drawdown_from_previous_high"],
                    "CMDTY_RET60": row["CMDTY_RET60"],
                    "SPY_close": row["spy_price"],
                    "SPY_MA20": row["SPY_MA20"],
                    "allocation_state_before": prev_state_for_event,
                    "allocation_state_after": state,
                }
            )
            event_id += 1
            prev_state_for_event = state

        pending = state
        if state != "FULL_RISK" and bool(row["SINGLE_ENTRY_SIGNAL"]):
            pending = "FULL_RISK"
        elif state == "FULL_RISK" and bool(row["R3_2D_CONFIRM_SIGNAL"]):
            pending = "NON_RISK"
    return pd.Series(states, index=panel.index, name="single_full_risk_state"), events


def build_single_strategy_weights(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, list[dict]]:
    work = build_single_trigger_signals(panel)
    single_state, events = build_single_full_risk_state(work)
    work["full_risk_state"] = single_state
    work["steep_slow_growth_overlay_state"] = False
    base_weights, base_states = base_refined_weights(work)
    final_weights, recovery_active = apply_flat_low_recovery(work, base_weights, base_states)

    # Append recovery events after weights are built.
    active = recovery_active.fillna(False).astype(bool)
    next_id = len(events) + 1
    for i in work.index[active & ~active.shift(1, fill_value=False)]:
        events.append(event_row(next_id, work.loc[i], "RECOVERY_ENTRY", "NONE", "NON_RISK", "RECOVERY"))
        next_id += 1
    for i in work.index[~active & active.shift(1, fill_value=False)]:
        events.append(event_row(next_id, work.loc[i], "RECOVERY_EXIT", "NONE", "RECOVERY", "NON_RISK"))
        next_id += 1
    return final_weights, base_states, recovery_active, single_state, events


def event_row(event_id: int, row: pd.Series, event_type: str, trigger: str, before: str, after: str) -> dict:
    return {
        "event_id": event_id,
        "date": row["date"],
        "event_type": event_type,
        "refined_regime": row["refined_regime"],
        "trigger_used": trigger,
        "VIX_ZSCORE_120D": row["VIX_ZSCORE_120D"],
        "D_CREDIT_SPREAD_20D": row["D_CREDIT_SPREAD_20D"],
        "spy_drawdown_from_previous_high": row["spy_drawdown_from_previous_high"],
        "CMDTY_RET60": row["CMDTY_RET60"],
        "SPY_close": row["spy_price"],
        "SPY_MA20": row["SPY_MA20"],
        "allocation_state_before": before,
        "allocation_state_after": after,
    }


def canonical_weights(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[[f"{FINAL_STRATEGY}_weight_{a}" for a in ASSETS]].rename(columns={f"{FINAL_STRATEGY}_weight_{a}": a for a in ASSETS})


def spy_weights(panel: pd.DataFrame) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=panel.index, columns=ASSETS)
    w["SPY"] = 1.0
    return w


def spy_cash_weights(panel: pd.DataFrame) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=panel.index, columns=ASSETS)
    risk = panel["full_risk_state"].eq("FULL_RISK") if "full_risk_state" in panel.columns else False
    w["SPY"] = np.where(risk, 0.0, 1.0)
    w["CASH"] = 1.0 - w["SPY"]
    return w


def combine_strategy_outputs(panel: pd.DataFrame, strategies: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = [panel[["date"]].copy()]
    perf_rows = []
    for name, weights in strategies.items():
        out = compute_strategy(panel, weights[ASSETS].astype(float), name)
        parts.append(out)
        perf_rows.append(performance_metrics(out, name))
    return pd.concat(parts, axis=1), pd.DataFrame(perf_rows)


def full_risk_entries_exits(active: pd.Series) -> tuple[list[int], list[int]]:
    b = active.fillna(False).astype(bool)
    entries = b.index[b & ~b.shift(1, fill_value=False)].tolist()
    exits = b.index[~b & b.shift(1, fill_value=False)].tolist()
    return entries, exits


def reentry_counts(entries: list[int], exits: list[int]) -> dict[str, int]:
    out = {}
    for days in [5, 10, 20]:
        count = 0
        for x in exits:
            nxt = next((e for e in entries if e > x), None)
            if nxt is not None and nxt - x <= days:
                count += 1
        out[f"number_of_reentries_within_{days}d"] = count
    return out


def augment_performance(perf: pd.DataFrame, states: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for _, row in perf.iterrows():
        name = row["strategy"]
        s = states.get(name)
        if s is None:
            entries, exits = [], []
        else:
            entries, exits = full_risk_entries_exits(s.eq("FULL_RISK"))
        r = row.to_dict()
        r["number_of_full_risk_entries"] = len(entries)
        r["number_of_full_risk_exits"] = len(exits)
        r.update(reentry_counts(entries, exits))
        rows.append(r)
    return pd.DataFrame(rows)


def event_turnover(panel: pd.DataFrame, weights: pd.DataFrame, idxs: list[int]) -> float:
    turnover = 0.5 * weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = 0.5 * weights.iloc[0].abs().sum()
    return float(turnover.loc[idxs].sum()) if idxs else 0.0


def trigger_comparison(panel: pd.DataFrame, single_state: pd.Series, single_weights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    canonical_state = panel["full_risk_state"]
    for name, state, weights in [
        (CANONICAL, canonical_state, canonical_weights(panel)),
        (NEW_STRATEGY, single_state, single_weights),
    ]:
        entries, exits = full_risk_entries_exits(state.eq("FULL_RISK"))
        row = {
            "strategy": name,
            "full_risk_entry_count": len(entries),
            "full_risk_exit_count": len(exits),
            "turnover_from_full_risk_entry": event_turnover(panel, weights, entries),
            "turnover_from_full_risk_exit": event_turnover(panel, weights, exits),
            "STEEP_full_risk_entry_count": int(sum(panel.loc[i, "refined_regime_confirmed"] == "STEEP" for i in entries)),
            "FLAT_LOW_full_risk_entry_count": int(sum(panel.loc[i, "refined_regime_confirmed"] == "FLAT_LOW_RATE" for i in entries)),
            "FLAT_HIGH_full_risk_entry_count": int(sum(panel.loc[i, "refined_regime_confirmed"] == "FLAT_HIGH_RATE" for i in entries)),
            "INVERTED_full_risk_entry_count": int(sum(panel.loc[i, "refined_regime_confirmed"] == "INVERTED" for i in entries)),
        }
        row.update(reentry_counts(entries, exits))
        rows.append(row)
    return pd.DataFrame(rows)


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


def rolling_12m(nav: pd.Series) -> pd.Series:
    return nav / nav.shift(252) - 1.0


def save_plots(panel: pd.DataFrame, outputs: pd.DataFrame, single_weights: pd.DataFrame, comp: pd.DataFrame) -> None:
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
    ax.bar(comp["strategy"], comp["turnover_from_full_risk_entry"] + comp["turnover_from_full_risk_exit"])
    ax.tick_params(axis="x", labelrotation=25)
    ax.set_title("Full-Risk Entry/Exit Turnover")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "turnover_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    canonical_entries, _ = full_risk_entries_exits(panel["full_risk_state"].eq("FULL_RISK"))
    ax.scatter(panel.loc[canonical_entries, "date"], ["canonical"] * len(canonical_entries), s=14, label=CANONICAL)
    ax.set_title("Canonical Full-Risk Entry Timeline")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "full_risk_entry_timeline_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(panel["date"], *[single_weights[a] for a in ASSETS], labels=ASSETS, alpha=0.85)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Single Trigger Strategy Weights")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "weights_timeline_single_trigger.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(comp["strategy"]))
    ax.bar(x - 0.25, comp["number_of_reentries_within_5d"], width=0.25, label="5D")
    ax.bar(x, comp["number_of_reentries_within_10d"], width=0.25, label="10D")
    ax.bar(x + 0.25, comp["number_of_reentries_within_20d"], width=0.25, label="20D")
    ax.set_xticks(x)
    ax.set_xticklabels(comp["strategy"], rotation=25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reentry_count_comparison.png", dpi=160)
    plt.close(fig)


def plot_entry_timeline(panel: pd.DataFrame, canonical_state: pd.Series, single_state: pd.Series) -> None:
    can_entries, _ = full_risk_entries_exits(canonical_state.eq("FULL_RISK"))
    single_entries, _ = full_risk_entries_exits(single_state.eq("FULL_RISK"))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(panel.loc[can_entries, "date"], ["canonical"] * len(can_entries), s=14, label=CANONICAL)
    ax.scatter(panel.loc[single_entries, "date"], ["single_trigger"] * len(single_entries), s=14, label=NEW_STRATEGY)
    ax.set_title("Full-Risk Entry Timeline Comparison")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "full_risk_entry_timeline_comparison.png", dpi=160)
    plt.close(fig)


def write_readme(perf: pd.DataFrame, comp: pd.DataFrame) -> None:
    lines = [
        "# Regime-Specific Single Trigger + R3 2D Confirmation Test",
        "",
        "## Purpose",
        "",
        "This light candidate test evaluates whether a simpler regime-specific single-trigger stress rule can reduce trigger conflicts and FULL_RISK whipsaw. It does not replace the canonical final strategy.",
        "",
        "## New Trigger Rules",
        "",
        "- `FLAT_LOW_RATE`: FULL_RISK only from `CMDTY_RET60 < -10%`.",
        "- `FLAT_HIGH_RATE`: FULL_RISK only from credit + SPY drawdown.",
        "- `STEEP`: FULL_RISK only from `VIX_ZSCORE_120D >= 3.0`.",
        "- `INVERTED`: no FULL_RISK trigger.",
        "",
        "## Exit Rule",
        "",
        "R3 requires two consecutive closes above MA20. The second confirmation day generates the exit signal and the next trading day executes the allocation change.",
        "",
        "## Differences vs Canonical Final Strategy",
        "",
        "- Removes STEEP Monthly SELL full-risk trigger.",
        "- Removes STEEP credit full-risk trigger.",
        "- Removes STEEP commodity slow-growth overlay.",
        "- Uses commodity weakness as a FLAT_LOW_RATE full-risk trigger.",
        "- Keeps FLAT_HIGH_RATE credit trigger.",
        "- Keeps FLAT_LOW_RATE 20D recovery overlay, now based on the new stress state.",
        "",
        "## Performance Comparison",
        "",
        perf.to_markdown(index=False),
        "",
        "## Trigger Comparison",
        "",
        comp.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "Compare turnover, re-entry counts, Sharpe, MaxDD, and CAGR. If performance deteriorates, it suggests the original multi-trigger design, despite complexity, is still capturing useful stress information.",
    ]
    (OUT / "README_regime_single_trigger_r3_2d_confirm.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = build_single_trigger_signals(load_panel())
    single_weights, single_base_states, single_recovery, single_state, events = build_single_strategy_weights(panel)

    strategies = {
        SPY_BUY_HOLD: spy_weights(panel),
        SPY_CASH_TIMING: spy_cash_weights(panel),
        CANONICAL: canonical_weights(panel),
        NEW_STRATEGY: single_weights,
    }
    outputs, perf = combine_strategy_outputs(panel, strategies)
    states = {
        SPY_BUY_HOLD: pd.Series("NON_RISK", index=panel.index),
        SPY_CASH_TIMING: panel["full_risk_state"],
        CANONICAL: panel["full_risk_state"],
        NEW_STRATEGY: single_state,
    }
    perf = augment_performance(perf, states)
    comp = trigger_comparison(panel, single_state, single_weights)
    event_log = pd.DataFrame(events).sort_values(["date", "event_id"])

    perf.to_csv(TABLE_DIR / "performance_comparison.csv", index=False)
    long_weights(panel, strategies).to_csv(TABLE_DIR / "daily_weights_all_strategies.csv", index=False)
    long_returns(outputs, DISPLAY).to_csv(TABLE_DIR / "daily_returns_all_strategies.csv", index=False)
    event_log.to_csv(TABLE_DIR / "single_trigger_event_log.csv", index=False)
    comp.to_csv(TABLE_DIR / "trigger_comparison_vs_canonical.csv", index=False)

    save_plots(panel, outputs, single_weights, comp)
    plot_entry_timeline(panel, panel["full_risk_state"], single_state)
    write_readme(perf, comp)

    print("Performance comparison:")
    print(perf[["strategy", "CAGR", "Sharpe", "MaxDD", "Calmar", "final_equity", "turnover", "number_of_full_risk_entries", "number_of_full_risk_exits"]].to_string(index=False))
    print("\nTurnover / trigger comparison:")
    print(comp.to_string(index=False))
    original = perf.loc[perf["strategy"].eq(CANONICAL)].iloc[0]
    new = perf.loc[perf["strategy"].eq(NEW_STRATEGY)].iloc[0]
    print("\nImproves turnover without worsening MaxDD:", bool(new["turnover"] < original["turnover"] and new["MaxDD"] >= original["MaxDD"]))
    print("Output path:", OUT)


if __name__ == "__main__":
    main()
