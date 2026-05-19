from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ATTR = ROOT / "results" / "recovery_20d_equal_weight_attribution" / "tables" / "recovery_episode_attribution.csv"
SOURCE_EPISODE = ROOT / "results" / "recovery_20d_strategy_test_L50_H30" / "tables" / "recovery_episode_strategy_performance.csv"
OUTPUT_DIR = ROOT / "results" / "stress_exit_recovery_regime_attribution"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

ASSETS = ["SPY", "GOLD", "CMDTY_FUT", "IEF", "CASH"]


def load_episode_data() -> pd.DataFrame:
    if SOURCE_ATTR.exists():
        df = pd.read_csv(SOURCE_ATTR, parse_dates=["recovery_start_date", "recovery_end_date"])
    elif SOURCE_EPISODE.exists():
        df = pd.read_csv(SOURCE_EPISODE, parse_dates=["recovery_start_date", "recovery_end_date"])
        df = df.rename(
            columns={
                "FLAT_RATE_REFINED_L50_H30_return": "baseline_return",
                "RECOVERY_20D_EQUAL_WEIGHT_return": "recovery_equal_weight_return",
                "RECOVERY_20D_EQUAL_WEIGHT_minus_refined": "excess_return_vs_baseline",
                "FLAT_RATE_REFINED_L50_H30_maxdd": "baseline_maxdd",
                "RECOVERY_20D_EQUAL_WEIGHT_maxdd": "recovery_equal_weight_maxdd",
            }
        )
    else:
        raise FileNotFoundError(
            "Missing recovery attribution inputs. Run scripts/test_recovery_20d_strategies_L50_H30.py "
            "and scripts/attribution_recovery_20d_equal_weight.py first."
        )
    return df


def refined_regime(row: pd.Series) -> str:
    sub = str(row.get("start_sub_state", ""))
    regime = str(row.get("start_regime", ""))
    if sub.startswith("FLAT_LOW_RATE"):
        return "FLAT_LOW_RATE"
    if sub.startswith("FLAT_HIGH_RATE"):
        return "FLAT_HIGH_RATE"
    if regime == "INVERTED":
        return "INVERTED"
    if regime == "STEEP":
        return "STEEP"
    if regime == "FLAT":
        gs10 = row.get("GS10", np.nan)
        if pd.notna(gs10):
            return "FLAT_LOW_RATE" if float(gs10) <= 2.9 else "FLAT_HIGH_RATE"
    return "OTHER"


def rate_level(row: pd.Series) -> str:
    r = row["refined_regime_at_exit"]
    if r == "FLAT_LOW_RATE":
        return "LOW_RATE"
    if r == "FLAT_HIGH_RATE":
        return "HIGH_RATE"
    return ""


def prepare_episode_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["refined_regime_at_exit"] = out.apply(refined_regime, axis=1)
    out["original_regime"] = out.get("start_regime", "")
    out["flat_sub_state"] = out.get("start_sub_state", "")
    if "GS10" not in out.columns:
        out["GS10"] = np.nan
    out["rate_level"] = out.apply(rate_level, axis=1)

    out["false_recovery_20d"] = out["exit_type"].eq("interrupted_by_new_stress")
    out["true_recovery_20d"] = ~out["false_recovery_20d"]
    out["days_until_next_stress"] = np.where(out["false_recovery_20d"], out["episode_length_days"], np.nan)
    out["profitable_exit"] = out["excess_return_vs_baseline"] > 0
    out["losing_exit"] = out["excess_return_vs_baseline"] <= 0
    out["material_profitable_exit"] = out["excess_return_vs_baseline"] > 0.005
    out["material_losing_exit"] = out["excess_return_vs_baseline"] < -0.005
    out["maxdd_diff"] = out["recovery_equal_weight_maxdd"] - out["baseline_maxdd"]

    rename = {
        "baseline_avg_weight_SPY": "refined_avg_weight_SPY",
        "baseline_avg_weight_GOLD": "refined_avg_weight_GOLD",
        "baseline_avg_weight_CMDTY_FUT": "refined_avg_weight_CMDTY_FUT",
        "baseline_avg_weight_IEF": "refined_avg_weight_IEF",
        "baseline_avg_weight_CASH": "refined_avg_weight_CASH",
    }
    out = out.rename(columns=rename)
    for asset in ASSETS:
        for prefix in ["refined_avg_weight", "recovery_avg_weight"]:
            col = f"{prefix}_{asset}"
            if col not in out.columns:
                out[col] = np.nan

    cols = [
        "episode_id",
        "recovery_start_date",
        "recovery_end_date",
        "episode_length_days",
        "refined_regime_at_exit",
        "original_regime",
        "flat_sub_state",
        "GS10",
        "rate_level",
        "false_recovery_20d",
        "true_recovery_20d",
        "days_until_next_stress",
        "exit_type",
        "selected_assets_for_equal_weight",
        "baseline_return",
        "recovery_equal_weight_return",
        "excess_return_vs_baseline",
        "profitable_exit",
        "losing_exit",
        "material_profitable_exit",
        "material_losing_exit",
        "baseline_maxdd",
        "recovery_equal_weight_maxdd",
        "maxdd_diff",
    ]
    for asset in ASSETS:
        cols.append(f"refined_avg_weight_{asset}")
    for asset in ASSETS:
        cols.append(f"recovery_avg_weight_{asset}")
    return out[cols].sort_values("episode_id").reset_index(drop=True)


def summarize_by_regime(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, sub in df.groupby("refined_regime_at_exit", dropna=False):
        best = sub.nlargest(1, "excess_return_vs_baseline")
        worst = sub.nsmallest(1, "excess_return_vs_baseline")
        rows.append(
            {
                "refined_regime_at_exit": regime,
                "number_of_episodes": len(sub),
                "false_recovery_count": int(sub["false_recovery_20d"].sum()),
                "false_recovery_rate": sub["false_recovery_20d"].mean(),
                "true_recovery_count": int(sub["true_recovery_20d"].sum()),
                "profitable_exit_count": int(sub["profitable_exit"].sum()),
                "losing_exit_count": int(sub["losing_exit"].sum()),
                "profitable_exit_rate": sub["profitable_exit"].mean(),
                "material_profitable_count": int(sub["material_profitable_exit"].sum()),
                "material_losing_count": int(sub["material_losing_exit"].sum()),
                "mean_excess_return": sub["excess_return_vs_baseline"].mean(),
                "median_excess_return": sub["excess_return_vs_baseline"].median(),
                "total_excess_return": sub["excess_return_vs_baseline"].sum(),
                "std_excess_return": sub["excess_return_vs_baseline"].std(),
                "best_excess_return": sub["excess_return_vs_baseline"].max(),
                "worst_excess_return": sub["excess_return_vs_baseline"].min(),
                "mean_refined_return": sub["baseline_return"].mean(),
                "mean_recovery_return": sub["recovery_equal_weight_return"].mean(),
                "mean_maxdd_diff": sub["maxdd_diff"].mean(),
                "average_episode_length": sub["episode_length_days"].mean(),
                "interrupted_rate": sub["false_recovery_20d"].mean(),
                "top_contributor_episode": int(best["episode_id"].iloc[0]) if not best.empty else np.nan,
                "worst_drag_episode": int(worst["episode_id"].iloc[0]) if not worst.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("total_excess_return", ascending=False)


def cross_table(df: pd.DataFrame, flag_col: str) -> pd.DataFrame:
    rows = []
    for (regime, flag), sub in df.groupby(["refined_regime_at_exit", flag_col], dropna=False):
        rows.append(
            {
                "refined_regime_at_exit": regime,
                flag_col: flag,
                "count": len(sub),
                "mean_excess_return": sub["excess_return_vs_baseline"].mean(),
                "median_excess_return": sub["excess_return_vs_baseline"].median(),
                "total_excess_return": sub["excess_return_vs_baseline"].sum(),
                "profitable_exit_rate": sub["profitable_exit"].mean(),
                "material_losing_count": int(sub["material_losing_exit"].sum()),
                "average_days_until_next_stress": sub["days_until_next_stress"].mean(),
                "average_episode_length": sub["episode_length_days"].mean(),
                "false_recovery_rate": sub["false_recovery_20d"].mean(),
            }
        )
    return pd.DataFrame(rows)


def recommendation_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        reasons = []
        total = row["total_excess_return"]
        mean = row["mean_excess_return"]
        false_rate = row["false_recovery_rate"]
        material_loss = row["material_losing_count"]
        profitable = row["profitable_exit_count"]
        losing = row["losing_exit_count"]
        if total > 0 and mean > 0 and false_rate <= 0.6 and material_loss <= 1 and profitable >= losing:
            action = "True"
            reasons.append("positive total/mean excess with acceptable false recovery and loss profile")
        elif total <= 0 or mean <= 0:
            action = "False"
            reasons.append("non-positive total or mean excess")
        elif false_rate > 0.7 or losing > profitable:
            action = "False"
            reasons.append("false recovery or losing exits dominate")
        else:
            action = "Needs_Filter"
            reasons.append("positive contribution but concentration/false-recovery/loss profile requires filtering")
        rows.append(
            {
                "refined_regime_at_exit": row["refined_regime_at_exit"],
                "enable_recovery_overlay": action,
                "reason": "; ".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def plot_simple_bar(df: pd.DataFrame, x: str, y: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = np.where(df[y] >= 0, "#2ca02c", "#d62728")
    ax.bar(df[x].astype(str), df[y], color=colors)
    ax.set_title(title)
    ax.set_ylabel(y)
    ax.tick_params(axis="x", rotation=30)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_outputs(df: pd.DataFrame, summary: pd.DataFrame, tf: pd.DataFrame, pl: pd.DataFrame) -> None:
    plot_simple_bar(summary, "refined_regime_at_exit", "false_recovery_rate", "False recovery rate by regime", FIGURE_DIR / "false_recovery_rate_by_regime.png")
    plot_simple_bar(summary, "refined_regime_at_exit", "mean_excess_return", "Mean excess return by regime", FIGURE_DIR / "mean_excess_return_by_regime.png")
    plot_simple_bar(summary, "refined_regime_at_exit", "total_excess_return", "Total excess return by regime", FIGURE_DIR / "total_excess_return_by_regime.png")
    plot_simple_bar(summary, "refined_regime_at_exit", "profitable_exit_rate", "Profitable exit rate by regime", FIGURE_DIR / "profitable_exit_rate_by_regime.png")

    pivot = tf.pivot(index="refined_regime_at_exit", columns="false_recovery_20d", values="mean_excess_return").fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Mean excess return by regime and true/false recovery")
    ax.set_ylabel("mean_excess_return")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "regime_true_false_excess_return_bar.png", dpi=160)
    plt.close(fig)

    counts = pl.pivot(index="refined_regime_at_exit", columns="profitable_exit", values="count").fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Profitable / losing exit counts by regime")
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "regime_profitable_losing_counts.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    markers = {True: "x", False: "o"}
    for (regime, false_flag), sub in df.groupby(["refined_regime_at_exit", "false_recovery_20d"]):
        ax.scatter(sub["recovery_start_date"], sub["excess_return_vs_baseline"], label=f"{regime} false={false_flag}", marker=markers[bool(false_flag)], alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Recovery excess return by start date and refined regime")
    ax.set_ylabel("excess_return_vs_refined")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "excess_return_scatter_by_regime.png", dpi=160)
    plt.close(fig)

    top = df.sort_values(["refined_regime_at_exit", "excess_return_vs_baseline"], ascending=[True, False]).groupby("refined_regime_at_exit").head(3)
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = top["refined_regime_at_exit"] + "#" + top["episode_id"].astype(str)
    ax.bar(labels, top["excess_return_vs_baseline"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Top contributors by regime")
    ax.set_ylabel("excess_return_vs_refined")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "top_contributors_by_regime.png", dpi=160)
    plt.close(fig)


def write_readme(summary: pd.DataFrame, recs: pd.DataFrame) -> None:
    best = summary.nlargest(1, "total_excess_return").iloc[0]
    worst = summary.nsmallest(1, "total_excess_return").iloc[0]
    text = f"""# Stress Exit Recovery Attribution By Refined Regime

## Purpose

This independent experiment classifies each stress-exit recovery episode into a refined regime and checks whether `RECOVERY_20D_EQUAL_WEIGHT` works only in selected environments.

## Refined Regime Classification

The script first uses `start_sub_state` when available:

- `FLAT_LOW_RATE_NORMAL` -> `FLAT_LOW_RATE`
- `FLAT_HIGH_RATE_NORMAL` -> `FLAT_HIGH_RATE`

Non-FLAT exits are mapped by `start_regime`: `STEEP`, `INVERTED`, or `OTHER`. GS10 fallback is supported but was not needed when sub-state labels were present.

## Definitions

- `false_recovery_20d`: the episode was interrupted by a new stress event within the 20D recovery window.
- `profitable_exit`: recovery equal-weight excess return versus `FLAT_RATE_REFINED_L50_H30` is positive.
- `material_profitable_exit`: excess return > 0.5%.
- `material_losing_exit`: excess return < -0.5%.

## Main Findings

- Best total excess regime: `{best['refined_regime_at_exit']}` with total excess {best['total_excess_return']:.2%}.
- Worst total excess regime: `{worst['refined_regime_at_exit']}` with total excess {worst['total_excess_return']:.2%}.
- High false-recovery regimes should not automatically receive the recovery overlay unless they also show robust positive contribution.

## Recommendation By Regime

{recs.to_markdown(index=False)}

## Next Step

Use this table to test focused variants such as recovery only in FLAT, recovery only in FLAT_LOW_RATE, or recovery in regimes with positive mean excess and acceptable false-recovery rate. Do not use `days_until_next_stress` as a trading rule because it is only known ex post.
"""
    (OUTPUT_DIR / "README_stress_exit_recovery_regime_attribution.md").write_text(text, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    episodes = prepare_episode_table(load_episode_data())
    summary = summarize_by_regime(episodes)
    tf = cross_table(episodes, "false_recovery_20d")[
        [
            "refined_regime_at_exit",
            "false_recovery_20d",
            "count",
            "mean_excess_return",
            "median_excess_return",
            "total_excess_return",
            "profitable_exit_rate",
            "material_losing_count",
            "average_days_until_next_stress",
        ]
    ]
    pl = cross_table(episodes, "profitable_exit")[
        [
            "refined_regime_at_exit",
            "profitable_exit",
            "count",
            "mean_excess_return",
            "median_excess_return",
            "total_excess_return",
            "average_episode_length",
            "false_recovery_rate",
        ]
    ]
    recs = recommendation_table(summary)

    episodes.to_csv(TABLE_DIR / "stress_exit_episode_regime_attribution.csv", index=False)
    summary.to_csv(TABLE_DIR / "stress_exit_regime_summary.csv", index=False)
    tf.to_csv(TABLE_DIR / "regime_true_false_recovery_cross_table.csv", index=False)
    pl.to_csv(TABLE_DIR / "regime_profitable_losing_exit_cross_table.csv", index=False)
    recs.to_csv(TABLE_DIR / "recovery_regime_recommendation.csv", index=False)

    plot_outputs(episodes, summary, tf, pl)
    write_readme(summary, recs)

    print("Stress exit recovery regime attribution complete.")
    print(f"output_dir: {OUTPUT_DIR.relative_to(ROOT)}")
    print("\nEpisodes by regime:")
    print(summary[["refined_regime_at_exit", "number_of_episodes"]].to_string(index=False))
    print("\nFalse recovery rate by regime:")
    print(summary[["refined_regime_at_exit", "false_recovery_rate"]].to_string(index=False))
    print("\nMean excess return by regime:")
    print(summary[["refined_regime_at_exit", "mean_excess_return"]].to_string(index=False))
    print("\nTotal excess return by regime:")
    print(summary[["refined_regime_at_exit", "total_excess_return"]].to_string(index=False))
    print("\nProfitable exit rate by regime:")
    print(summary[["refined_regime_at_exit", "profitable_exit_rate"]].to_string(index=False))
    print("\nRecommendation by regime:")
    print(recs.to_string(index=False))


if __name__ == "__main__":
    main()
