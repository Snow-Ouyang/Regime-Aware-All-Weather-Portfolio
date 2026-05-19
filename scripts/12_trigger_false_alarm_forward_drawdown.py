from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
INPUT_PANEL = ROOT / "results" / "main_pipeline_final" / "daily_backtest_panel.csv"
OUT = ROOT / "results" / "trigger_false_alarm_forward_drawdown"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
HORIZONS = [5, 10, 20, 60]
TRIGGERS = {
    "VIX_TRIGGER": "VIX_TRIGGER",
    "RAW_CREDIT_DRAWDOWN_TRIGGER": "RAW_CREDIT_DRAWDOWN_TRIGGER",
    "EFFECTIVE_CREDIT_DRAWDOWN_TRIGGER": "EFFECTIVE_CREDIT_DRAWDOWN_TRIGGER",
    "MONTHLY_SELL_TRIGGER": "MONTHLY_SELL_TRIGGER",
    "CMDTY_SLOW_GROWTH_TRIGGER": "CMDTY_SLOW_GROWTH_TRIGGER",
}
REGIME_ORDER = ["FLAT_LOW_RATE", "FLAT_HIGH_RATE", "STEEP", "INVERTED", "OTHER"]


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {INPUT_PANEL}: {missing}")


def load_panel() -> pd.DataFrame:
    if not INPUT_PANEL.exists():
        raise FileNotFoundError(f"Missing canonical main panel: {INPUT_PANEL}")
    df = pd.read_csv(INPUT_PANEL, parse_dates=["date"])
    require_columns(
        df,
        [
            "date",
            "SPY_return",
            "macro_regime_confirmed",
            "refined_regime_confirmed",
            "GS10",
            "VIX_ZSCORE_120D",
            "D_CREDIT_SPREAD_20D",
            "spy_drawdown_from_previous_high",
            "CMDTY_RET60",
            "monthly_either_state",
        ],
    )
    if "FINAL_REGIME_HEDGE_RECOVERY_return" not in df.columns:
        df["FINAL_REGIME_HEDGE_RECOVERY_return"] = np.nan
    return df


def canonical_refined_regime(df: pd.DataFrame) -> pd.Series:
    if "refined_regime_confirmed" in df.columns:
        s = df["refined_regime_confirmed"].where(
            df["refined_regime_confirmed"].isin(REGIME_ORDER[:-1]),
            "OTHER",
        )
        return s.fillna("OTHER")
    base = df["macro_regime_confirmed"].fillna("OTHER")
    return np.select(
        [
            base.eq("FLAT") & (df["GS10"] <= 2.9),
            base.eq("FLAT") & (df["GS10"] > 2.9),
            base.eq("STEEP"),
            base.eq("INVERTED"),
        ],
        ["FLAT_LOW_RATE", "FLAT_HIGH_RATE", "STEEP", "INVERTED"],
        default="OTHER",
    )


def add_triggers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["refined_regime"] = canonical_refined_regime(out)
    out["base_regime"] = out["macro_regime_confirmed"].where(out["macro_regime_confirmed"].isin(["FLAT", "STEEP", "INVERTED"]), "OTHER")
    out["VIX_TRIGGER"] = out["VIX_ZSCORE_120D"] >= 3.0
    out["RAW_CREDIT_DRAWDOWN_TRIGGER"] = (out["spy_drawdown_from_previous_high"] <= -0.05) & (out["D_CREDIT_SPREAD_20D"] > 0.10)
    out["EFFECTIVE_CREDIT_DRAWDOWN_TRIGGER"] = out["RAW_CREDIT_DRAWDOWN_TRIGGER"] & ~out["refined_regime"].eq("INVERTED")
    # This diagnostic intentionally evaluates raw trigger quality across all
    # refined regimes. Strategy-specific gating is handled in the final
    # strategy/backbone modules, not here.
    out["MONTHLY_SELL_TRIGGER"] = out["monthly_either_state"].eq("SELL")
    out["CMDTY_SLOW_GROWTH_TRIGGER"] = out["CMDTY_RET60"] < -0.10
    for col in TRIGGERS:
        out[f"{col}_EVENT"] = out[col].fillna(False).astype(bool) & ~out[col].fillna(False).astype(bool).shift(1, fill_value=False)
    return out


def forward_return(ret: pd.Series, start: int, horizon: int) -> float:
    sub = ret.iloc[start + 1 : start + 1 + horizon].fillna(0.0)
    if sub.empty:
        return np.nan
    return float((1.0 + sub).prod() - 1.0)


def forward_mdd(ret: pd.Series, start: int, horizon: int) -> float:
    sub = ret.iloc[start + 1 : start + 1 + horizon].fillna(0.0)
    if sub.empty:
        return np.nan
    nav = (1.0 + sub).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def extract_events(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    event_id = 1
    for trigger_name in TRIGGERS:
        event_idx = df.index[df[f"{trigger_name}_EVENT"]].tolist()
        for idx in event_idx:
            row = df.loc[idx]
            out = {
                "event_id": event_id,
                "date": row["date"],
                "trigger_name": trigger_name,
                "refined_regime": row["refined_regime"],
                "base_regime": row["base_regime"],
                "VIX_ZSCORE_120D": row["VIX_ZSCORE_120D"],
                "D_CREDIT_SPREAD_20D": row["D_CREDIT_SPREAD_20D"],
                "spy_drawdown_from_previous_high": row["spy_drawdown_from_previous_high"],
                "CMDTY_RET60": row["CMDTY_RET60"],
                "monthly_either_state": row["monthly_either_state"],
            }
            for h in HORIZONS:
                out[f"SPY_return_next_{h}d"] = forward_return(df["SPY_return"], idx, h)
                out[f"SPY_forward_max_drawdown_{h}d"] = forward_mdd(df["SPY_return"], idx, h)
            if "FINAL_REGIME_HEDGE_RECOVERY_return" in df.columns:
                out["strategy_return_next_5d"] = forward_return(df["FINAL_REGIME_HEDGE_RECOVERY_return"], idx, 5)
                out["strategy_return_next_20d"] = forward_return(df["FINAL_REGIME_HEDGE_RECOVERY_return"], idx, 20)
                out["strategy_forward_max_drawdown_20d"] = forward_mdd(df["FINAL_REGIME_HEDGE_RECOVERY_return"], idx, 20)
            rows.append(out)
            event_id += 1
    events = pd.DataFrame(rows)
    if events.empty:
        return events
    events["false_alarm_return_20d"] = events["SPY_return_next_20d"] >= 0
    events["false_alarm_drawdown_20d"] = events["SPY_forward_max_drawdown_20d"] > -0.03
    events["false_alarm_return_60d"] = events["SPY_return_next_60d"] >= 0
    events["false_alarm_drawdown_60d"] = events["SPY_forward_max_drawdown_60d"] > -0.05
    return events


def summarize(events: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    out = (
        events.groupby(group_cols, dropna=False)
        .agg(
            event_count=("event_id", "size"),
            mean_SPY_return_next_5d=("SPY_return_next_5d", "mean"),
            mean_SPY_return_next_10d=("SPY_return_next_10d", "mean"),
            mean_SPY_return_next_20d=("SPY_return_next_20d", "mean"),
            mean_SPY_return_next_60d=("SPY_return_next_60d", "mean"),
            median_SPY_return_next_20d=("SPY_return_next_20d", "median"),
            median_SPY_return_next_60d=("SPY_return_next_60d", "median"),
            mean_SPY_forward_max_drawdown_20d=("SPY_forward_max_drawdown_20d", "mean"),
            mean_SPY_forward_max_drawdown_60d=("SPY_forward_max_drawdown_60d", "mean"),
            median_SPY_forward_max_drawdown_20d=("SPY_forward_max_drawdown_20d", "median"),
            median_SPY_forward_max_drawdown_60d=("SPY_forward_max_drawdown_60d", "median"),
            false_alarm_return_20d_rate=("false_alarm_return_20d", "mean"),
            false_alarm_drawdown_20d_rate=("false_alarm_drawdown_20d", "mean"),
            false_alarm_return_60d_rate=("false_alarm_return_60d", "mean"),
            false_alarm_drawdown_60d_rate=("false_alarm_drawdown_60d", "mean"),
        )
        .reset_index()
    )
    out["hit_rate_drawdown_20d"] = 1.0 - out["false_alarm_drawdown_20d_rate"]
    out["hit_rate_drawdown_60d"] = 1.0 - out["false_alarm_drawdown_60d_rate"]
    return out


def comparison_pivot(summary_by_regime: pd.DataFrame) -> pd.DataFrame:
    if summary_by_regime.empty:
        return pd.DataFrame()
    metrics = [
        "event_count",
        "false_alarm_drawdown_20d_rate",
        "mean_SPY_forward_max_drawdown_20d",
        "mean_SPY_return_next_20d",
    ]
    wide = summary_by_regime.pivot_table(index="trigger_name", columns="refined_regime", values=metrics, aggfunc="first")
    return wide.sort_index(axis=1)


def diagnose(row: pd.Series) -> tuple[str, str]:
    n = int(row["event_count"])
    fa = row["false_alarm_drawdown_20d_rate"]
    mdd = row["mean_SPY_forward_max_drawdown_20d"]
    trig = row["trigger_name"]
    regime = row["refined_regime"]
    if n < 5:
        return "Insufficient_Sample", "Insufficient events to conclude."
    if fa <= 0.40 and mdd <= -0.04:
        return "Effective", f"Trigger is effective in {regime} with low false alarm and large forward drawdown."
    if fa > 0.65 or mdd > -0.02:
        return "Noisy", f"Trigger appears noisy in {regime} with high false alarm rate or shallow forward drawdown."
    if (fa > 0.40 and fa <= 0.65) or (-0.04 < mdd <= -0.02):
        return "Weak", f"Trigger has mixed evidence in {regime}; forward drawdown exists but false alarms are material."
    return "Weak", f"Trigger evidence is mixed for {trig} in {regime}."


def recommendation(summary_by_regime: pd.DataFrame) -> pd.DataFrame:
    if summary_by_regime.empty:
        return pd.DataFrame()
    rows = []
    for _, row in summary_by_regime.iterrows():
        diagnosis, comment = diagnose(row)
        rows.append(
            {
                "trigger_name": row["trigger_name"],
                "refined_regime": row["refined_regime"],
                "event_count": row["event_count"],
                "false_alarm_drawdown_20d_rate": row["false_alarm_drawdown_20d_rate"],
                "mean_SPY_forward_max_drawdown_20d": row["mean_SPY_forward_max_drawdown_20d"],
                "mean_SPY_return_next_20d": row["mean_SPY_return_next_20d"],
                "diagnosis": diagnosis,
                "comment": comment,
            }
        )
    return pd.DataFrame(rows)


def plot_grouped(summary: pd.DataFrame, value: str, filename: str, title: str, percent: bool = False) -> None:
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_df = summary.copy()
    plot_df["refined_regime"] = pd.Categorical(plot_df["refined_regime"], REGIME_ORDER, ordered=True)
    plot_df = plot_df.sort_values(["refined_regime", "trigger_name"])
    sns.barplot(data=plot_df, x="refined_regime", y=value, hue="trigger_name", ax=ax)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=25)
    if percent:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=160)
    plt.close(fig)


def plot_heatmap(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    heat = summary.pivot_table(index="trigger_name", columns="refined_regime", values="mean_SPY_forward_max_drawdown_20d", aggfunc="first")
    labels_src = summary.pivot_table(index="trigger_name", columns="refined_regime", values="false_alarm_drawdown_20d_rate", aggfunc="first")
    count_src = summary.pivot_table(index="trigger_name", columns="refined_regime", values="event_count", aggfunc="first")
    labels = heat.copy().astype(object)
    for r in heat.index:
        for c in heat.columns:
            if pd.isna(heat.loc[r, c]):
                labels.loc[r, c] = ""
            else:
                labels.loc[r, c] = f"{heat.loc[r, c]:.1%}\nFA {labels_src.loc[r, c]:.0%}\nn={int(count_src.loc[r, c])}"
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(heat, annot=labels, fmt="", cmap="RdYlGn_r", center=-0.03, linewidths=0.5, linecolor="white", ax=ax)
    ax.set_title("Trigger Effectiveness Heatmap: 20D Forward Max Drawdown")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "trigger_effectiveness_heatmap.png", dpi=160)
    plt.close(fig)


def save_figures(summary: pd.DataFrame) -> None:
    plot_grouped(
        summary,
        "false_alarm_drawdown_20d_rate",
        "false_alarm_drawdown_20d_by_trigger_regime.png",
        "20D Drawdown-Based False Alarm Rate by Trigger and Regime",
        percent=True,
    )
    plot_grouped(
        summary,
        "mean_SPY_forward_max_drawdown_20d",
        "mean_forward_drawdown_20d_by_trigger_regime.png",
        "Mean SPY 20D Forward Max Drawdown by Trigger and Regime",
        percent=True,
    )
    plot_grouped(
        summary,
        "mean_SPY_return_next_20d",
        "mean_forward_return_20d_by_trigger_regime.png",
        "Mean SPY 20D Forward Return by Trigger and Regime",
        percent=True,
    )
    plot_grouped(
        summary,
        "event_count",
        "trigger_event_count_by_regime.png",
        "Trigger Event Count by Regime",
        percent=False,
    )
    plot_heatmap(summary)


def write_readme(overall: pd.DataFrame, rec: pd.DataFrame) -> None:
    effective = rec.loc[rec["diagnosis"].eq("Effective"), ["trigger_name", "refined_regime", "event_count"]]
    noisy = rec.loc[rec["diagnosis"].eq("Noisy"), ["trigger_name", "refined_regime", "event_count"]]
    lines = [
        "# Trigger False Alarm and Forward Drawdown Diagnostic",
        "",
        "## Purpose",
        "",
        "This light diagnostic reassesses stress triggers after replacing the old single FLAT regime with the canonical `FLAT_LOW_RATE` / `FLAT_HIGH_RATE` framework. It does not change final strategy rules.",
        "",
        "## Trigger Definitions",
        "",
        "- `VIX_TRIGGER`: `VIX_ZSCORE_120D >= 3.0`.",
        "- `RAW_CREDIT_DRAWDOWN_TRIGGER`: SPY drawdown <= -5% and `D_CREDIT_SPREAD_20D > 0.10`.",
        "- `EFFECTIVE_CREDIT_DRAWDOWN_TRIGGER`: raw credit trigger excluding `INVERTED`.",
        "- `MONTHLY_SELL_TRIGGER`: Monthly Either SELL, evaluated across all refined regimes for diagnostic purposes.",
        "- `CMDTY_SLOW_GROWTH_TRIGGER`: `CMDTY_RET60 < -10%`, evaluated across all refined regimes for diagnostic purposes.",
        "",
        "## False Alarm Definitions",
        "",
        "- 20D return false alarm: forward 20D SPY return >= 0.",
        "- 20D drawdown false alarm: forward 20D SPY max drawdown > -3%.",
        "- 60D return false alarm: forward 60D SPY return >= 0.",
        "- 60D drawdown false alarm: forward 60D SPY max drawdown > -5%.",
        "",
        "## Overall Summary",
        "",
        overall.to_markdown(index=False) if not overall.empty else "No trigger events found.",
        "",
        "## Effective Trigger / Regime Combinations",
        "",
        effective.to_markdown(index=False) if not effective.empty else "No combinations met the simple Effective rule.",
        "",
        "## Noisy Trigger / Regime Combinations",
        "",
        noisy.to_markdown(index=False) if not noisy.empty else "No combinations met the simple Noisy rule.",
        "",
        "## Interpretation",
        "",
        "Use this output to identify which trigger/regime pairs deserve future cooldown or state-machine refinement. The trigger flags here are raw all-regime diagnostics unless explicitly named `EFFECTIVE`; this experiment intentionally stops at diagnosis and does not alter the canonical strategy.",
    ]
    (OUT / "README_trigger_false_alarm_forward_drawdown.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = add_triggers(load_panel())
    events = extract_events(panel)
    summary_by_regime = summarize(events, ["trigger_name", "refined_regime"])
    overall = summarize(events, ["trigger_name"])
    pivot = comparison_pivot(summary_by_regime)
    rec = recommendation(summary_by_regime)

    events.to_csv(TABLE_DIR / "trigger_events.csv", index=False)
    summary_by_regime.to_csv(TABLE_DIR / "trigger_false_alarm_summary_by_regime.csv", index=False)
    overall.to_csv(TABLE_DIR / "trigger_false_alarm_summary_overall.csv", index=False)
    pivot.to_csv(TABLE_DIR / "trigger_regime_comparison_pivot.csv")
    rec.to_csv(TABLE_DIR / "trigger_diagnostic_recommendation.csv", index=False)
    save_figures(summary_by_regime)
    write_readme(overall, rec)

    print("Overall summary:")
    print(overall.to_string(index=False))
    print("\nTrigger x regime false alarm summary:")
    print(summary_by_regime[["trigger_name", "refined_regime", "event_count", "false_alarm_drawdown_20d_rate", "mean_SPY_forward_max_drawdown_20d"]].to_string(index=False))
    print("\nNoisy trigger / regime combinations:")
    noisy = rec.loc[rec["diagnosis"].eq("Noisy")]
    print(noisy.to_string(index=False) if not noisy.empty else "None")
    print("\nEffective trigger / regime combinations:")
    effective = rec.loc[rec["diagnosis"].eq("Effective")]
    print(effective.to_string(index=False) if not effective.empty else "None")
    print("Output path:", OUT)


if __name__ == "__main__":
    main()
