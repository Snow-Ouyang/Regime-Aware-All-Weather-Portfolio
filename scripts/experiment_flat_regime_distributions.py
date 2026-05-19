"""Independent diagnostic: distribution heterogeneity inside FLAT regime.

This experiment does not modify any main strategy code or existing results.
It compares all FLAT observations against FLAT stress observations across
selected macro variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "flat_regime_distribution_experiment"


# Edit this map if future panels use different field names.
FIELD_MAP = {
    "date": ["date"],
    "regime": ["macro_regime_confirmed", "market_regime", "regime", "regime_label"],
    # Existing flat stress indicator. The default interpretation is RISK in timing_state.
    "stress_state": [
        "timing_state",
        "risk_state",
        "BACKBONE_V2_UPGRADED_risk_state",
        "MATURE_BASELINE_REGIME_HEDGE_INV_VOL_risk_state",
    ],
    # Optional direct boolean fields. Used before stress_state if available.
    "flat_stress_bool": [
        "flat_stress",
        "FLAT_STRESS",
        "is_flat_stress",
        "FLAT_VIX_OR_CREDIT_STRESS",
    ],
    "term_spread": ["GS10_minus_GS1", "term_spread", "TERM_SPREAD", "T10Y2Y", "T10Y3M"],
    "GS10": ["GS10", "DGS10", "GS10_simple"],
    "credit_spread": [
        "CREDIT_SPREAD_BAA_AAA",
        "credit_spread",
        "credit_spread_ext",
        "credit_spread_simple",
        "BAA_AAA",
    ],
    "growth": ["growth_pc1", "growth", "GROWTH_PC1"],
    "inflation": ["inflation_pc1", "inflation", "INFLATION_PC1"],
}


INPUT_CANDIDATES = [
    "results/drawdown_2015_2016_forensic_diagnostic/forensic_daily_panel.csv",
    "results/06_2015_2016_repair/drawdown_2015_2016_forensic_diagnostic/forensic_daily_panel.csv",
    "results/09_final_strategy/mature_regime_hedge_final/daily_backtest_panel.csv",
    "results/mature_regime_hedge_final/daily_backtest_panel.csv",
    "results/08_allocation/regime_aware_risk_parity_allocation/daily_backtest_panel.csv",
    "results/regime_aware_risk_parity_allocation/daily_backtest_panel.csv",
    "results/04_stress_triggers/spy_cash_backbone_upgrade_ablation/daily_backtest_panel.csv",
    "results/spy_cash_backbone_upgrade_ablation/daily_backtest_panel.csv",
]

VARIABLES = ["term_spread", "GS10", "credit_spread", "growth", "inflation"]
STRESS_VALUES = {"RISK", "FULL_RISK", "STRESS", "1", "TRUE", "YES"}


@dataclass
class ResolvedFields:
    date: str
    regime: str
    stress_source: str
    stress_is_direct_bool: bool
    variables: dict[str, str]


def find_first_existing(columns: Iterable[str], candidates: list[str]) -> str | None:
    cols = set(columns)
    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in cols:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def load_panel() -> tuple[pd.DataFrame, Path]:
    for rel_path in INPUT_CANDIDATES:
        path = ROOT / rel_path
        if path.exists():
            df = pd.read_csv(path)
            if not df.empty:
                return df, path
    raise FileNotFoundError(
        "No candidate master panel found. Checked:\n"
        + "\n".join(f"- {p}" for p in INPUT_CANDIDATES)
    )


def resolve_fields(df: pd.DataFrame) -> ResolvedFields:
    date_col = find_first_existing(df.columns, FIELD_MAP["date"])
    regime_col = find_first_existing(df.columns, FIELD_MAP["regime"])
    direct_stress = find_first_existing(df.columns, FIELD_MAP["flat_stress_bool"])
    stress_state = find_first_existing(df.columns, FIELD_MAP["stress_state"])

    missing_core = []
    if date_col is None:
        missing_core.append("date")
    if regime_col is None:
        missing_core.append("regime label / market regime")
    if direct_stress is None and stress_state is None:
        missing_core.append("flat stress indicator: direct flat_stress or timing/risk state")
    if missing_core:
        raise KeyError(
            "Missing required core fields:\n"
            + "\n".join(f"- {item}" for item in missing_core)
            + "\nAdjust FIELD_MAP at the top of scripts/experiment_flat_regime_distributions.py."
        )

    variables = {}
    missing_vars = []
    for var in VARIABLES:
        col = find_first_existing(df.columns, FIELD_MAP[var])
        if col is None:
            missing_vars.append(var)
        else:
            variables[var] = col
    if missing_vars:
        raise KeyError(
            "Missing required macro variables:\n"
            + "\n".join(f"- {item}: candidates={FIELD_MAP[item]}" for item in missing_vars)
            + "\nAdjust FIELD_MAP at the top of scripts/experiment_flat_regime_distributions.py."
        )

    return ResolvedFields(
        date=date_col,
        regime=regime_col,
        stress_source=direct_stress if direct_stress is not None else stress_state,
        stress_is_direct_bool=direct_stress is not None,
        variables=variables,
    )


def normalize_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).ne(0)
    return series.astype(str).str.upper().isin(STRESS_VALUES)


def prepare_flat_sample(df: pd.DataFrame, fields: ResolvedFields) -> pd.DataFrame:
    out = df.copy()
    out[fields.date] = pd.to_datetime(out[fields.date])
    flat = out[out[fields.regime].astype(str).str.upper().eq("FLAT")].copy()
    if flat.empty:
        raise ValueError(f"No FLAT observations found in regime column: {fields.regime}")

    if fields.stress_is_direct_bool:
        flat["flat_stress"] = normalize_bool(flat[fields.stress_source])
    else:
        flat["flat_stress"] = flat[fields.stress_source].astype(str).str.upper().isin(STRESS_VALUES)

    if flat["flat_stress"].sum() == 0:
        raise ValueError(
            f"Flat stress indicator `{fields.stress_source}` produced zero stress observations. "
            "Please adjust FIELD_MAP or stress state values."
        )

    for var, col in fields.variables.items():
        flat[var] = pd.to_numeric(flat[col], errors="coerce")

    return flat[[fields.date, fields.regime, fields.stress_source, "flat_stress", *VARIABLES]].rename(
        columns={fields.date: "date", fields.regime: "regime", fields.stress_source: "stress_source_value"}
    )


def percentile_rank_against_all_flat(all_values: pd.Series, values: pd.Series) -> pd.Series:
    clean_all = all_values.dropna().sort_values()
    if clean_all.empty:
        return pd.Series(index=values.index, dtype=float)
    return values.apply(lambda x: np.nan if pd.isna(x) else clean_all.searchsorted(x, side="right") / len(clean_all))


def describe_sample(values: pd.Series) -> dict[str, float]:
    clean = values.dropna()
    q = clean.quantile([0.25, 0.75]) if not clean.empty else pd.Series([np.nan, np.nan], index=[0.25, 0.75])
    return {
        "count": int(clean.count()),
        "mean": clean.mean(),
        "median": clean.median(),
        "std": clean.std(),
        "min": clean.min(),
        "25%": q.loc[0.25],
        "75%": q.loc[0.75],
        "max": clean.max(),
    }


def compute_summary(flat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    stress_mask = flat["flat_stress"]
    for var in VARIABLES:
        all_values = flat[var]
        stress_percentiles = percentile_rank_against_all_flat(all_values, flat.loc[stress_mask, var])
        samples = {
            "all_flat": flat,
            "flat_stress": flat.loc[stress_mask],
            "flat_non_stress": flat.loc[~stress_mask],
        }
        for sample_name, sample_df in samples.items():
            row = {"variable": var, "sample": sample_name}
            row.update(describe_sample(sample_df[var]))
            if sample_name == "flat_stress":
                row["stress_percentile_mean"] = stress_percentiles.mean()
                row["stress_percentile_median"] = stress_percentiles.median()
            else:
                row["stress_percentile_mean"] = np.nan
                row["stress_percentile_median"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def plot_variable(flat: pd.DataFrame, var: str, output_dir: Path) -> None:
    total_n = int(flat[var].notna().sum())
    stress = flat.loc[flat["flat_stress"], var].dropna()
    non_stress_n = int((~flat["flat_stress"] & flat[var].notna()).sum())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(flat[var].dropna(), bins=40, alpha=0.45, color="#4C78A8", label="All FLAT")
    ax.hist(stress, bins=40, alpha=0.75, color="#F58518", label="FLAT stress")
    ax.set_title(f"{var}: FLAT distribution | flat total n={total_n}, stress n={len(stress)}, non-stress n={non_stress_n}")
    ax.set_xlabel(var)
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{var}_flat_distribution.png", dpi=160)
    plt.close(fig)


def plot_summary(flat: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(VARIABLES), figsize=(22, 4.5))
    total_flat = len(flat)
    stress_n = int(flat["flat_stress"].sum())
    non_stress_n = total_flat - stress_n
    for ax, var in zip(axes, VARIABLES):
        ax.hist(flat[var].dropna(), bins=35, alpha=0.45, color="#4C78A8", label="All FLAT")
        ax.hist(flat.loc[flat["flat_stress"], var].dropna(), bins=35, alpha=0.75, color="#F58518", label="FLAT stress")
        ax.set_title(var)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Count")
    axes[-1].legend(loc="best")
    fig.suptitle(f"FLAT macro distributions | flat total n={total_flat}, stress n={stress_n}, non-stress n={non_stress_n}")
    fig.tight_layout()
    fig.savefig(output_dir / "flat_macro_distribution_summary.png", dpi=180)
    plt.close(fig)


def concentration_comments(summary: pd.DataFrame) -> tuple[list[str], list[str]]:
    concentrated = []
    weak = []
    stress_rows = summary[summary["sample"].eq("flat_stress")].copy()
    for _, row in stress_rows.iterrows():
        var = row["variable"]
        mean_pct = row["stress_percentile_mean"]
        med_pct = row["stress_percentile_median"]
        if pd.isna(mean_pct):
            weak.append(f"{var}: insufficient valid stress observations.")
        elif mean_pct >= 0.65 and med_pct >= 0.60:
            concentrated.append(f"{var}: stress observations are concentrated in high percentiles (mean {mean_pct:.2f}, median {med_pct:.2f}).")
        elif mean_pct <= 0.35 and med_pct <= 0.40:
            concentrated.append(f"{var}: stress observations are concentrated in low percentiles (mean {mean_pct:.2f}, median {med_pct:.2f}).")
        else:
            weak.append(f"{var}: no strong percentile separation (mean {mean_pct:.2f}, median {med_pct:.2f}).")
    return concentrated, weak


def write_readme(
    output_dir: Path,
    input_path: Path,
    fields: ResolvedFields,
    flat: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    total_n = len(flat)
    stress_n = int(flat["flat_stress"].sum())
    non_stress_n = total_n - stress_n
    concentrated, weak = concentration_comments(summary)
    support = "Yes" if concentrated else "Not strongly from this single diagnostic"

    lines = [
        "# FLAT Regime Distribution Experiment",
        "",
        "## Purpose",
        "",
        "This independent experiment studies heterogeneity inside the FLAT regime. It compares macro variable distributions for all FLAT observations versus FLAT stress observations.",
        "",
        "No strategy logic was changed, and no mainline result was overwritten.",
        "",
        "## Input Data",
        "",
        f"- Source panel: `{input_path.relative_to(ROOT).as_posix()}`",
        f"- Date field: `{fields.date}`",
        f"- Regime field: `{fields.regime}`",
        f"- Flat stress source: `{fields.stress_source}`",
        f"- Stress interpretation: `{'direct boolean' if fields.stress_is_direct_bool else 'stress/risk state equals RISK-like value'}`",
        "",
        "## Fields Used",
        "",
    ]
    for var, col in fields.variables.items():
        lines.append(f"- {var}: `{col}`")
    lines.extend(
        [
            "",
            "## Sample Counts",
            "",
            f"- FLAT total n: {total_n}",
            f"- FLAT stress n: {stress_n}",
            f"- FLAT non-stress n: {non_stress_n}",
            "",
            "## Distribution Findings",
            "",
            "Variables with visible percentile concentration:",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in concentrated) if concentrated else lines.append("- None with the current heuristic thresholds.")
    lines.extend(["", "Variables without clear separation:", ""])
    lines.extend(f"- {item}" for item in weak) if weak else lines.append("- None.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"Support for further FLAT sub-regime testing: {support}.",
            "",
            "Potential next diagnostic splits, if supported by the percentile evidence:",
            "",
            "- flat high-rate",
            "- flat low-rate",
            "- flat credit-stress",
            "- flat inflation-stress",
            "",
            "This experiment is descriptive only. It does not add strategy triggers, allocation rules, or optimization.",
            "",
            "## Outputs",
            "",
            "- `flat_distribution_summary.csv`",
            "- `term_spread_flat_distribution.png`",
            "- `GS10_flat_distribution.png`",
            "- `credit_spread_flat_distribution.png`",
            "- `growth_flat_distribution.png`",
            "- `inflation_flat_distribution.png`",
            "- `flat_macro_distribution_summary.png`",
        ]
    )
    (output_dir / "README_flat_distribution_experiment.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel, input_path = load_panel()
    fields = resolve_fields(panel)
    flat = prepare_flat_sample(panel, fields)
    flat.to_csv(OUTPUT_DIR / "flat_regime_sample.csv", index=False)

    summary = compute_summary(flat)
    summary.to_csv(OUTPUT_DIR / "flat_distribution_summary.csv", index=False)

    plt.style.use("default")
    for var in VARIABLES:
        plot_variable(flat, var, OUTPUT_DIR)
    plot_summary(flat, OUTPUT_DIR)
    write_readme(OUTPUT_DIR, input_path, fields, flat, summary)

    print("FLAT regime distribution experiment completed.")
    print(f"input_panel: {input_path.relative_to(ROOT).as_posix()}")
    print(f"flat_total_n: {len(flat)}")
    print(f"flat_stress_n: {int(flat['flat_stress'].sum())}")
    print(f"flat_non_stress_n: {int((~flat['flat_stress']).sum())}")
    print(f"output_dir: {OUTPUT_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
