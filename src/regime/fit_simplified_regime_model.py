from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from regime.utils import (
    DISPLAY_NAMES,
    MODEL_FEATURES,
    N_STATES,
    PENALTY,
    REGIME_ORDER,
    RESULTS,
    SEEDS,
    assign_regime_names,
    build_profile_sentence,
    contiguous_run_lengths,
    ensure_jumpmodels_importable,
    ensure_project_dirs,
    percentile_rank,
)


ensure_jumpmodels_importable()
from jumpmodels.jump import JumpModel


INPUT_PATH = ROOT / "data" / "processed" / "regime_inputs_simplified.csv"
LABELS_PATH = RESULTS / "simplified_regime_labels.csv"
SUMMARY_PATH = RESULTS / "simplified_regime_summary.csv"
PERCENTILES_PATH = RESULTS / "regime_feature_percentiles.csv"
MARKDOWN_PATH = RESULTS / "SIMPLIFIED_REGIME_BASELINE.md"


def fit_one_jump_model(X: np.ndarray, seed: int) -> JumpModel:
    model = JumpModel(
        n_components=N_STATES,
        jump_penalty=PENALTY,
        cont=False,
        random_state=seed,
        max_iter=1000,
        tol=1e-8,
        n_init=10,
        verbose=0,
    )
    model.fit(X, sort_by="freq")
    return model


def select_best_model(X: np.ndarray) -> tuple[JumpModel, int, float]:
    best_model = None
    best_seed = None
    best_objective = np.inf
    for seed in SEEDS:
        model = fit_one_jump_model(X, seed)
        objective = float(model.val_)
        if objective < best_objective:
            best_model = model
            best_seed = seed
            best_objective = objective
    if best_model is None or best_seed is None:
        raise RuntimeError("JumpModel fit did not produce a valid baseline model.")
    return best_model, best_seed, best_objective


def build_percentile_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime_name in REGIME_ORDER:
        regime_panel = panel.loc[panel["regime_name"] == regime_name]
        for feature in MODEL_FEATURES:
            full_sample = panel[feature]
            mean_val = float(regime_panel[feature].mean()) if not regime_panel.empty else np.nan
            median_val = float(regime_panel[feature].median()) if not regime_panel.empty else np.nan
            rows.append(
                {
                    "regime_name": regime_name,
                    "variable": feature,
                    "display_name": DISPLAY_NAMES[feature],
                    "regime_mean": mean_val,
                    "regime_median": median_val,
                    "full_sample_mean_percentile": percentile_rank(full_sample, mean_val),
                    "full_sample_median_percentile": percentile_rank(full_sample, median_val),
                    "count": int(regime_panel[feature].count()),
                }
            )
    return pd.DataFrame(rows)


def build_summary_table(panel: pd.DataFrame, percentiles: pd.DataFrame) -> pd.DataFrame:
    total_transitions = int((panel["state_raw"] != panel["state_raw"].shift()).sum() - 1)
    total_transitions = max(total_transitions, 0)
    years = max((panel["date"].max() - panel["date"].min()).days / 365.25, 1e-9)
    percentile_wide = (
        percentiles[["regime_name", "variable", "full_sample_mean_percentile"]]
        .pivot(index="regime_name", columns="variable", values="full_sample_mean_percentile")
        .rename(columns=lambda col: f"{col}_mean_percentile")
    )

    rows = []
    for regime_name in REGIME_ORDER:
        subset = panel.loc[panel["regime_name"] == regime_name].copy()
        runs = contiguous_run_lengths(subset["state_raw"])
        state_raw = int(subset["state_raw"].mode().iloc[0]) if not subset.empty else np.nan
        row = {
            "state_raw": state_raw,
            "regime_name": regime_name,
            "number_of_months": int(len(subset)),
            "share_of_sample": float(len(subset) / len(panel)) if len(panel) else np.nan,
            "average_duration_months": float(runs["duration"].mean()) if not runs.empty else np.nan,
            "median_duration_months": float(runs["duration"].median()) if not runs.empty else np.nan,
            "number_of_regime_episodes": int(len(runs)),
            "transition_count": total_transitions,
            "annualized_transition_frequency": float(total_transitions / years),
        }
        for feature in MODEL_FEATURES:
            row[f"{feature}_mean"] = float(subset[feature].mean()) if not subset.empty else np.nan
            row[f"{feature}_median"] = float(subset[feature].median()) if not subset.empty else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("regime_name")
    summary = summary.join(percentile_wide, how="left")
    summary = summary.reset_index()
    summary["profile_text"] = summary.apply(build_profile_sentence, axis=1)
    return summary


def print_regime_profiles(summary: pd.DataFrame) -> None:
    for _, row in summary.iterrows():
        print(
            f"{row['regime_name']} | share={row['share_of_sample']:.1%} | "
            f"growth={row['growth_pc1_mean']:.2f} | inflation={row['inflation_pc1_mean']:.2f} | "
            f"gs10={row['gs10_mean']:.2f} | slope={row['term_spread_10y_1y_mean']:.2f} | "
            f"credit={row['credit_spread_mean']:.2f} | {row['profile_text']}"
        )


def write_markdown(summary: pd.DataFrame, state_map: pd.DataFrame) -> None:
    files = [
        "data/processed/regime_inputs_simplified.csv",
        "results/regime/simplified_regime_labels.csv",
        "results/regime/simplified_regime_summary.csv",
        "results/regime/factor_construction_summary.csv",
        "results/regime/regime_feature_percentiles.csv",
        "figures/regime/simplified_regime_timeline.png",
        "figures/regime/distribution_growth_pc1.png",
        "figures/regime/distribution_inflation_pc1.png",
        "figures/regime/distribution_gs10.png",
        "figures/regime/distribution_term_spread_10y_1y.png",
        "figures/regime/distribution_credit_spread.png",
        "figures/regime/regime_feature_percentile_heatmap.png",
        "figures/regime/regime_profile_percentile_bars.png",
    ]
    lines = [
        "# Simplified Regime Baseline",
        "",
        "## Migrated data and variables",
        "",
        "- Raw macro inputs migrated from the Market-Regime-Clustering project: CFNAI, GDP, industrial production, CPI, PPI, GS10, GS1, AAA, and BAA.",
        "- Simplified growth factor uses only `cfnai`, `gdp_amom`, and `ipgr_amom`.",
        "- Simplified inflation factor uses only `cpi_amom` and `ppi_amom`.",
        "- Final regime input features are `growth_pc1`, `inflation_pc1`, `gs10`, `term_spread_10y_1y`, and `credit_spread`.",
        "",
        "## Fixed model definition",
        "",
        "- Model: simplified 4-state Jump Model baseline.",
        "- Penalty: `1.0`.",
        "- ISM and sentiment are intentionally excluded and not reintroduced.",
        "",
        "## Interpretation frame",
        "",
        "- This simplified regime model is best described as a **rate-curve-credit regime framework**.",
        "- `gs10`, `term_spread_10y_1y`, and `credit_spread` do most of the work in separating the three major non-crisis regimes.",
        "- `growth_pc1` and `inflation_pc1` overlap more across normal regimes and mainly help identify the deflationary stress tail.",
        "",
        "## Why penalty is fixed",
        "",
        "- This module is the baseline regime engine for the All Weather extension.",
        "- The goal is a reproducible and interpretable macro regime layer, not another round of penalty or state-count optimization.",
        "",
        "## Produced files",
        "",
    ]
    lines.extend([f"- `{path}`" for path in files])
    lines.extend(
        [
            "",
            "## Regime mapping",
            "",
            state_map.to_markdown(index=False),
            "",
            "## Short regime interpretation",
            "",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(f"- **{row['regime_name']}**: {row['profile_text']}")
    lines.extend(
        [
            "",
            "## Role in the All Weather extension",
            "",
            "- This regime module provides the macro classification layer that future asset selection, signal conditioning, and allocation rules can consume without revisiting the baseline model definition.",
            "- In practice, the three major non-crisis regimes are primarily distinguished by rate level, curve shape, and credit conditions, while the PCs help isolate the stress tail.",
        ]
    )
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_project_dirs()
    panel = pd.read_csv(INPUT_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    X = StandardScaler().fit_transform(panel[MODEL_FEATURES])
    model, best_seed, objective = select_best_model(X)

    labeled = panel.copy()
    labeled["state_raw"] = np.asarray(model.labels_).astype(int)
    probabilities = np.asarray(model.proba_)
    for state in range(N_STATES):
        labeled[f"state_prob_{state}"] = probabilities[:, state]
    labeled["penalty"] = PENALTY
    labeled["best_seed"] = best_seed
    labeled["objective_value"] = objective
    labeled, state_map = assign_regime_names(labeled)
    labeled = labeled.sort_values(["date", "regime_order"]).reset_index(drop=True)

    percentiles = build_percentile_table(labeled)
    summary = build_summary_table(labeled, percentiles)

    labeled.to_csv(LABELS_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    percentiles.to_csv(PERCENTILES_PATH, index=False)
    write_markdown(summary, state_map)
    print_regime_profiles(summary)
    print(f"Saved labels to {LABELS_PATH}")
    print(f"Saved summary to {SUMMARY_PATH}")
    print(f"Saved percentiles to {PERCENTILES_PATH}")


if __name__ == "__main__":
    main()
