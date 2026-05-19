"""Independent diagnostic: FLAT regime asset behavior by absolute rate level.

This experiment does not change main strategy code or existing results.

It asks whether FLAT only captures yield-curve shape while missing the
absolute level of rates. Inside FLAT observations, it splits GS10 into high
and low rate groups using the full-sample GS10 median, then crosses that with
stress / normal periods.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "flat_rate_level_asset_behavior_experiment"
TRADING_DAYS = 252


FIELD_MAP = {
    "date": ["date"],
    "regime": ["macro_regime_confirmed", "market_regime", "regime", "regime_label"],
    "stress_state": [
        "timing_state",
        "risk_state",
        "BACKBONE_V2_UPGRADED_risk_state",
        "MATURE_BASELINE_REGIME_HEDGE_INV_VOL_risk_state",
    ],
    "flat_stress_bool": [
        "flat_stress",
        "FLAT_STRESS",
        "is_flat_stress",
        "FLAT_VIX_OR_CREDIT_STRESS",
    ],
    "GS10": ["GS10", "DGS10", "GS10_simple"],
    "SPY_return": ["SPY_return", "spy_daily_return", "SPY_daily_return"],
    "GOLD_return": ["GOLD_return", "GLD_return"],
    "CMDTY_FUT_return": ["CMDTY_FUT_return", "CMDTY_return", "commodities_return"],
    "IEF_return": ["IEF_return"],
    "CASH_return": ["CASH_return", "daily_rf", "rf_daily"],
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

ASSETS = ["SPY", "GOLD", "CMDTY_FUT", "IEF", "CASH"]
STRESS_VALUES = {"RISK", "FULL_RISK", "STRESS", "1", "TRUE", "YES"}
STATE_ORDER = [
    "FLAT_LOW_RATE_NORMAL",
    "FLAT_LOW_RATE_STRESS",
    "FLAT_HIGH_RATE_NORMAL",
    "FLAT_HIGH_RATE_STRESS",
]


@dataclass
class ResolvedFields:
    date: str
    regime: str
    stress_source: str
    stress_is_direct_bool: bool
    gs10: str
    returns: dict[str, str]


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
    raise FileNotFoundError("No usable candidate panel found.")


def resolve_fields(df: pd.DataFrame) -> ResolvedFields:
    date_col = find_first_existing(df.columns, FIELD_MAP["date"])
    regime_col = find_first_existing(df.columns, FIELD_MAP["regime"])
    direct_stress = find_first_existing(df.columns, FIELD_MAP["flat_stress_bool"])
    stress_state = find_first_existing(df.columns, FIELD_MAP["stress_state"])
    gs10_col = find_first_existing(df.columns, FIELD_MAP["GS10"])
    missing = []
    if date_col is None:
        missing.append("date")
    if regime_col is None:
        missing.append("regime")
    if direct_stress is None and stress_state is None:
        missing.append("flat stress indicator or timing/risk state")
    if gs10_col is None:
        missing.append("GS10")

    returns = {}
    for asset in ASSETS:
        key = f"{asset}_return"
        col = find_first_existing(df.columns, FIELD_MAP[key])
        if col is None:
            missing.append(key)
        else:
            returns[asset] = col

    if missing:
        raise KeyError(
            "Missing required fields:\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\nAdjust FIELD_MAP in scripts/experiment_flat_rate_level_asset_behavior.py."
        )

    return ResolvedFields(
        date=date_col,
        regime=regime_col,
        stress_source=direct_stress if direct_stress is not None else stress_state,
        stress_is_direct_bool=direct_stress is not None,
        gs10=gs10_col,
        returns=returns,
    )


def normalize_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).ne(0)
    return series.astype(str).str.upper().isin(STRESS_VALUES)


def prepare_sample(df: pd.DataFrame, fields: ResolvedFields) -> tuple[pd.DataFrame, float]:
    out = df.copy()
    out["date"] = pd.to_datetime(out[fields.date])
    out["regime"] = out[fields.regime].astype(str).str.upper()
    out["GS10"] = pd.to_numeric(out[fields.gs10], errors="coerce")
    gs10_median = float(out["GS10"].dropna().median())

    flat = out[out["regime"].eq("FLAT")].copy()
    if flat.empty:
        raise ValueError("No FLAT observations found.")

    if fields.stress_is_direct_bool:
        flat["flat_stress"] = normalize_bool(flat[fields.stress_source])
    else:
        flat["flat_stress"] = flat[fields.stress_source].astype(str).str.upper().isin(STRESS_VALUES)

    for asset, col in fields.returns.items():
        flat[f"{asset}_return"] = pd.to_numeric(flat[col], errors="coerce")

    flat["rate_bucket"] = np.where(flat["GS10"] >= gs10_median, "HIGH_RATE", "LOW_RATE")
    flat["stress_bucket"] = np.where(flat["flat_stress"], "STRESS", "NORMAL")
    flat["flat_rate_stress_state"] = "FLAT_" + flat["rate_bucket"] + "_" + flat["stress_bucket"]
    return flat, gs10_median


def max_drawdown(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    nav = (1.0 + clean).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def perf_metrics(returns: pd.Series, rf: pd.Series | None = None) -> dict[str, float]:
    clean = returns.dropna()
    if clean.empty:
        return {
            "n_obs": 0,
            "cumulative_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "max_drawdown": np.nan,
            "positive_day_ratio": np.nan,
            "worst_1d_return": np.nan,
            "best_1d_return": np.nan,
        }
    rf_aligned = rf.reindex(clean.index).fillna(0.0) if rf is not None else pd.Series(0.0, index=clean.index)
    excess = clean - rf_aligned
    ann_ret = (1.0 + clean).prod() ** (TRADING_DAYS / len(clean)) - 1.0
    ann_vol = clean.std() * np.sqrt(TRADING_DAYS)
    sharpe = excess.mean() / clean.std() * np.sqrt(TRADING_DAYS) if clean.std() and not np.isclose(clean.std(), 0.0) else np.nan
    return {
        "n_obs": int(clean.count()),
        "cumulative_return": float((1.0 + clean).prod() - 1.0),
        "annualized_return": float(ann_ret),
        "annualized_volatility": float(ann_vol),
        "sharpe_ratio": float(sharpe) if pd.notna(sharpe) else np.nan,
        "max_drawdown": max_drawdown(clean),
        "positive_day_ratio": float((clean > 0).mean()),
        "worst_1d_return": float(clean.min()),
        "best_1d_return": float(clean.max()),
    }


def compute_performance(flat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cash_rf = flat["CASH_return"] if "CASH_return" in flat.columns else None
    for state in STATE_ORDER:
        sub = flat[flat["flat_rate_stress_state"].eq(state)]
        for asset in ASSETS:
            ret_col = f"{asset}_return"
            row = {
                "state": state,
                "asset": asset,
                "rate_bucket": "HIGH_RATE" if "HIGH_RATE" in state else "LOW_RATE",
                "stress_bucket": "STRESS" if "STRESS" in state else "NORMAL",
                "GS10_mean": sub["GS10"].mean(),
                "GS10_median": sub["GS10"].median(),
            }
            row.update(perf_metrics(sub[ret_col], cash_rf.loc[sub.index] if cash_rf is not None else None))
            rows.append(row)
    return pd.DataFrame(rows)


def compute_state_counts(flat: pd.DataFrame, gs10_median: float) -> pd.DataFrame:
    counts = (
        flat.groupby("flat_rate_stress_state")
        .agg(
            n_obs=("date", "count"),
            start_date=("date", "min"),
            end_date=("date", "max"),
            GS10_mean=("GS10", "mean"),
            GS10_median=("GS10", "median"),
            stress_days=("flat_stress", "sum"),
        )
        .reset_index()
    )
    counts["full_sample_GS10_median_threshold"] = gs10_median
    return counts


def plot_metric_heatmaps(perf: pd.DataFrame, output_dir: Path) -> None:
    metrics = ["annualized_return", "sharpe_ratio", "max_drawdown"]
    titles = {
        "annualized_return": "Annualized Return",
        "sharpe_ratio": "Sharpe Ratio",
        "max_drawdown": "Max Drawdown",
    }
    for metric in metrics:
        pivot = perf.pivot(index="asset", columns="state", values=metric).reindex(index=ASSETS, columns=STATE_ORDER)
        fig, ax = plt.subplots(figsize=(12, 4.5))
        values = pivot.values.astype(float)
        if metric == "max_drawdown":
            cmap = "Reds_r"
        else:
            cmap = "RdYlGn"
        im = ax.imshow(values, aspect="auto", cmap=cmap)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                val = values[i, j]
                text = "" if np.isnan(val) else (f"{val:.2f}" if metric == "sharpe_ratio" else f"{val:.1%}")
                ax.text(j, i, text, ha="center", va="center", color="black", fontsize=9)
        ax.set_title(f"FLAT high/low rate x stress asset behavior: {titles[metric]}")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(output_dir / f"flat_rate_stress_{metric}_heatmap.png", dpi=170)
        plt.close(fig)


def plot_state_navs(flat: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    axes = axes.flatten()
    for ax, state in zip(axes, STATE_ORDER):
        sub = flat[flat["flat_rate_stress_state"].eq(state)].copy()
        for asset in ASSETS:
            ret = sub[f"{asset}_return"].dropna()
            if ret.empty:
                continue
            nav = (1.0 + ret).cumprod()
            ax.plot(sub.loc[ret.index, "date"], nav, label=asset, linewidth=1.2)
        ax.set_title(f"{state} | n={len(sub)}")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="best")
    fig.suptitle("Asset NAV within FLAT high/low rate x stress states")
    fig.tight_layout()
    fig.savefig(output_dir / "flat_rate_stress_asset_navs.png", dpi=170)
    plt.close(fig)


def write_readme(output_dir: Path, input_path: Path, fields: ResolvedFields, gs10_median: float, counts: pd.DataFrame, perf: pd.DataFrame) -> None:
    sharpe_pivot = perf.pivot(index="asset", columns="state", values="sharpe_ratio").reindex(index=ASSETS, columns=STATE_ORDER)
    annret_pivot = perf.pivot(index="asset", columns="state", values="annualized_return").reindex(index=ASSETS, columns=STATE_ORDER)
    best_by_state = []
    for state in STATE_ORDER:
        sub = perf[perf["state"].eq(state)].dropna(subset=["sharpe_ratio"])
        if not sub.empty:
            row = sub.loc[sub["sharpe_ratio"].idxmax()]
            best_by_state.append(f"- {state}: best Sharpe asset = {row['asset']} ({row['sharpe_ratio']:.2f})")

    lines = [
        "# FLAT Rate-Level Asset Behavior Experiment",
        "",
        "## Purpose",
        "",
        "This independent experiment tests whether FLAT regime hides important heterogeneity from the absolute level of rates. It splits FLAT observations into high-rate and low-rate groups using the full-sample GS10 median, then crosses each group with stress versus normal periods.",
        "",
        "No mainline strategy logic or existing results were modified.",
        "",
        "## Input and Fields",
        "",
        f"- Source panel: `{input_path.relative_to(ROOT).as_posix()}`",
        f"- Regime field: `{fields.regime}`",
        f"- Stress source: `{fields.stress_source}`",
        f"- GS10 field: `{fields.gs10}`",
        f"- Full-sample GS10 median threshold: {gs10_median:.4f}",
        "",
        "Return fields:",
        "",
    ]
    lines.extend(f"- {asset}: `{col}`" for asset, col in fields.returns.items())
    lines.extend(
        [
            "",
            "## State Counts",
            "",
            counts.to_markdown(index=False),
            "",
            "## Best Sharpe By State",
            "",
        ]
    )
    lines.extend(best_by_state)
    lines.extend(
        [
            "",
            "## Sharpe Summary",
            "",
            sharpe_pivot.round(3).to_markdown(),
            "",
            "## Annualized Return Summary",
            "",
            annret_pivot.applymap(lambda x: np.nan if pd.isna(x) else round(x, 4)).to_markdown(),
            "",
            "## Interpretation Guide",
            "",
            "- If asset rankings change meaningfully between high-rate and low-rate FLAT states, then FLAT curve shape alone may be insufficient.",
            "- If stress periods behave differently from normal periods within the same rate bucket, stress timing is doing distinct work from rate-level classification.",
            "- This experiment is descriptive only and does not introduce a new allocation rule.",
            "",
            "## Outputs",
            "",
            "- `flat_rate_state_counts.csv`",
            "- `flat_rate_asset_performance.csv`",
            "- `flat_rate_stress_annualized_return_heatmap.png`",
            "- `flat_rate_stress_sharpe_ratio_heatmap.png`",
            "- `flat_rate_stress_max_drawdown_heatmap.png`",
            "- `flat_rate_stress_asset_navs.png`",
        ]
    )
    (output_dir / "README_flat_rate_level_asset_behavior.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel, input_path = load_panel()
    fields = resolve_fields(panel)
    flat, gs10_median = prepare_sample(panel, fields)
    counts = compute_state_counts(flat, gs10_median)
    perf = compute_performance(flat)

    flat.to_csv(OUTPUT_DIR / "flat_rate_level_sample.csv", index=False)
    counts.to_csv(OUTPUT_DIR / "flat_rate_state_counts.csv", index=False)
    perf.to_csv(OUTPUT_DIR / "flat_rate_asset_performance.csv", index=False)
    plot_metric_heatmaps(perf, OUTPUT_DIR)
    plot_state_navs(flat, OUTPUT_DIR)
    write_readme(OUTPUT_DIR, input_path, fields, gs10_median, counts, perf)

    print("FLAT rate-level asset behavior experiment completed.")
    print(f"input_panel: {input_path.relative_to(ROOT).as_posix()}")
    print(f"full_sample_GS10_median: {gs10_median:.4f}")
    for _, row in counts.iterrows():
        print(f"{row['flat_rate_stress_state']}: n={int(row['n_obs'])}")
    print(f"output_dir: {OUTPUT_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
