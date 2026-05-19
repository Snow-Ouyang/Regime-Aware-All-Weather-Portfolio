"""Independent FLAT-regime GS10 threshold robustness scan.

This experiment does not modify mainline strategy code or existing outputs.

It scans GS10 percentile thresholds inside FLAT regime and evaluates whether
asset rankings are stable across threshold definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "flat_gs10_threshold_robustness"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"


DATA_FREQUENCY = "D"  # "D" daily or "M" monthly. Use "AUTO" to infer.
THRESHOLD_PERCENTILES = [0.30, 0.40, 0.50, 0.60, 0.70]
STRESS_VALUES = {"RISK", "FULL_RISK", "STRESS", "1", "TRUE", "YES"}
ASSET_ORDER = ["SPY", "GOLD", "CMDTY_FUT", "IEF", "CASH"]
REAL_ASSETS = ["GOLD", "CMDTY_FUT"]
DEFENSIVE_ASSETS = ["GOLD", "IEF", "CASH"]
STATE_ORDER = [
    "FLAT_LOW_RATE_NORMAL",
    "FLAT_LOW_RATE_STRESS",
    "FLAT_HIGH_RATE_NORMAL",
    "FLAT_HIGH_RATE_STRESS",
]


# Edit this map if future panels use different names.
FIELD_MAP = {
    "date": ["date"],
    "regime": ["macro_regime_confirmed", "regime", "market_regime", "regime_label"],
    "stress": [
        "flat_stress",
        "FLAT_STRESS",
        "is_flat_stress",
        "timing_state",
        "risk_state",
        "BACKBONE_V2_UPGRADED_risk_state",
        "MATURE_BASELINE_REGIME_HEDGE_INV_VOL_risk_state",
    ],
    "gs10": ["GS10", "DGS10", "GS10_simple"],
    "spy_ret": ["SPY_return", "spy_daily_return", "SPY_daily_return"],
    "gold_ret": ["GOLD_return", "GLD_return"],
    "cmdty_ret": ["CMDTY_FUT_return", "CMDTY_return", "commodities_return"],
    "ief_ret": ["IEF_return"],
    "cash_ret": ["CASH_return", "daily_rf", "rf_daily"],
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


@dataclass
class Fields:
    date: str
    regime: str
    stress: str
    gs10: str
    returns: dict[str, str]


def find_col(columns: Iterable[str], candidates: list[str]) -> str | None:
    cols = set(columns)
    lower = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in cols:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def load_panel() -> tuple[pd.DataFrame, Path]:
    for rel in INPUT_CANDIDATES:
        path = ROOT / rel
        if path.exists():
            df = pd.read_csv(path)
            if not df.empty:
                return df, path
    raise FileNotFoundError("No usable input panel found. Checked: " + ", ".join(INPUT_CANDIDATES))


def resolve_fields(df: pd.DataFrame) -> Fields:
    missing = []
    date = find_col(df.columns, FIELD_MAP["date"])
    regime = find_col(df.columns, FIELD_MAP["regime"])
    stress = find_col(df.columns, FIELD_MAP["stress"])
    gs10 = find_col(df.columns, FIELD_MAP["gs10"])
    if date is None:
        missing.append("date")
    if regime is None:
        missing.append("regime")
    if stress is None:
        missing.append("stress / flat stress / timing state")
    if gs10 is None:
        missing.append("GS10")

    returns = {}
    asset_key_map = {
        "SPY": "spy_ret",
        "GOLD": "gold_ret",
        "CMDTY_FUT": "cmdty_ret",
        "IEF": "ief_ret",
        "CASH": "cash_ret",
    }
    for asset, key in asset_key_map.items():
        col = find_col(df.columns, FIELD_MAP[key])
        if col is None:
            if asset in {"CMDTY_FUT", "IEF"}:
                warnings.warn(f"Optional asset missing and will be skipped: {asset} candidates={FIELD_MAP[key]}")
            else:
                missing.append(f"{asset} return")
        else:
            returns[asset] = col

    if missing:
        raise KeyError(
            "Missing required fields:\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\nAdjust FIELD_MAP at the top of scripts/experiment_flat_gs10_threshold_robustness.py."
        )
    return Fields(date=date, regime=regime, stress=stress, gs10=gs10, returns=returns)


def infer_frequency(dates: pd.Series) -> tuple[str, int]:
    if DATA_FREQUENCY.upper() == "D":
        return "D", 252
    if DATA_FREQUENCY.upper() == "M":
        return "M", 12
    sorted_dates = pd.to_datetime(dates).sort_values()
    median_gap = sorted_dates.diff().dt.days.median()
    if pd.notna(median_gap) and median_gap > 20:
        return "M", 12
    return "D", 252


def normalize_stress(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).ne(0)
    return series.astype(str).str.upper().isin(STRESS_VALUES)


def prepare_panel(df: pd.DataFrame, fields: Fields) -> tuple[pd.DataFrame, pd.DataFrame, int, str]:
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[fields.date])
    out["regime"] = df[fields.regime].astype(str).str.upper()
    out["flat_stress"] = normalize_stress(df[fields.stress])
    out["GS10"] = pd.to_numeric(df[fields.gs10], errors="coerce")
    for asset, col in fields.returns.items():
        out[f"{asset}_return"] = pd.to_numeric(df[col], errors="coerce")
    out = out.sort_values("date").reset_index(drop=True)
    freq, annualization = infer_frequency(out["date"])
    flat = out[out["regime"].eq("FLAT")].copy()
    if flat.empty:
        raise ValueError("No FLAT observations found.")
    if flat["flat_stress"].sum() == 0:
        raise ValueError(f"Stress field `{fields.stress}` produced zero FLAT stress observations.")
    return out, flat, annualization, freq


def max_drawdown(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    nav = (1.0 + clean).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def metrics(returns: pd.Series, annualization: int, cash_ret: pd.Series | None = None) -> dict[str, float]:
    clean = returns.dropna()
    if clean.empty:
        return {
            "count": 0,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "max_drawdown": np.nan,
            "win_rate": np.nan,
            "best_month_or_day": np.nan,
            "worst_month_or_day": np.nan,
            "cumulative_return": np.nan,
        }
    rf = cash_ret.reindex(clean.index).fillna(0.0) if cash_ret is not None else pd.Series(0.0, index=clean.index)
    vol = clean.std()
    ann_ret = (1.0 + clean).prod() ** (annualization / len(clean)) - 1.0
    ann_vol = vol * np.sqrt(annualization)
    sharpe = ((clean - rf).mean() / vol * np.sqrt(annualization)) if pd.notna(vol) and vol > 1e-10 else 0.0
    return {
        "count": int(clean.count()),
        "annualized_return": float(ann_ret),
        "annualized_volatility": float(ann_vol),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": max_drawdown(clean),
        "win_rate": float((clean > 0).mean()),
        "best_month_or_day": float(clean.max()),
        "worst_month_or_day": float(clean.min()),
        "cumulative_return": float((1.0 + clean).prod() - 1.0),
    }


def threshold_value(scope: str, percentile: float, full: pd.DataFrame, flat: pd.DataFrame) -> float:
    base = full if scope == "full_sample_gs10" else flat
    return float(base["GS10"].dropna().quantile(percentile))


def state_name(rate_bucket: str, stress: bool) -> str:
    return f"FLAT_{rate_bucket}_{'STRESS' if stress else 'NORMAL'}"


def scan_thresholds(full: pd.DataFrame, flat: pd.DataFrame, annualization: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    scopes = ["full_sample_gs10", "flat_sample_gs10"]
    rows = []
    count_rows = []
    assets = [a for a in ASSET_ORDER if f"{a}_return" in flat.columns]
    for scope in scopes:
        for pct in THRESHOLD_PERCENTILES:
            tv = threshold_value(scope, pct, full, flat)
            sample = flat.copy()
            sample["rate_bucket"] = np.where(sample["GS10"] <= tv, "LOW_RATE", "HIGH_RATE")
            sample["state"] = [state_name(r, s) for r, s in zip(sample["rate_bucket"], sample["flat_stress"])]
            counts = sample["state"].value_counts().reindex(STATE_ORDER, fill_value=0)
            count_row = {
                "threshold_scope": scope,
                "threshold_percentile": pct,
                "threshold_value": tv,
            }
            count_row.update({state: int(counts[state]) for state in STATE_ORDER})
            count_rows.append(count_row)
            cash = sample["CASH_return"] if "CASH_return" in sample.columns else None
            for state in STATE_ORDER:
                sub = sample[sample["state"].eq(state)]
                for asset in assets:
                    row = {
                        "threshold_scope": scope,
                        "threshold_percentile": pct,
                        "threshold_value": tv,
                        "state": state,
                        "asset": asset,
                    }
                    row.update(metrics(sub[f"{asset}_return"], annualization, cash.loc[sub.index] if cash is not None else None))
                    rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(count_rows)


def rank_assets(metrics_long: pd.DataFrame) -> pd.DataFrame:
    ranking_rows = []
    metric_specs = {
        "sharpe_ratio": False,
        "annualized_return": False,
        "max_drawdown": False,  # higher / less negative is better
    }
    for keys, grp in metrics_long.groupby(["threshold_scope", "threshold_percentile", "threshold_value", "state"]):
        for metric, ascending in metric_specs.items():
            ranked = grp.dropna(subset=[metric]).sort_values(metric, ascending=ascending)
            assets = ranked["asset"].tolist()
            row = {
                "threshold_scope": keys[0],
                "threshold_percentile": keys[1],
                "threshold_value": keys[2],
                "state": keys[3],
                "metric": metric,
            }
            for i in range(5):
                row[f"rank_{i+1}"] = assets[i] if i < len(assets) else ""
            ranking_rows.append(row)
    return pd.DataFrame(ranking_rows)


def value_for(metrics_long: pd.DataFrame, scope: str, pct: float, state: str, asset: str, metric: str) -> float:
    sub = metrics_long[
        metrics_long["threshold_scope"].eq(scope)
        & metrics_long["threshold_percentile"].eq(pct)
        & metrics_long["state"].eq(state)
        & metrics_long["asset"].eq(asset)
    ]
    return np.nan if sub.empty else float(sub.iloc[0][metric])


def rank_for(ranking: pd.DataFrame, scope: str, pct: float, state: str, metric: str, asset: str) -> float:
    sub = ranking[
        ranking["threshold_scope"].eq(scope)
        & ranking["threshold_percentile"].eq(pct)
        & ranking["state"].eq(state)
        & ranking["metric"].eq(metric)
    ]
    if sub.empty:
        return np.nan
    row = sub.iloc[0]
    for i in range(1, 6):
        if row.get(f"rank_{i}") == asset:
            return i
    return np.nan


def best_asset(metrics_long: pd.DataFrame, scope: str, pct: float, state: str, assets: list[str], metric: str) -> tuple[str, float]:
    sub = metrics_long[
        metrics_long["threshold_scope"].eq(scope)
        & metrics_long["threshold_percentile"].eq(pct)
        & metrics_long["state"].eq(state)
        & metrics_long["asset"].isin(assets)
    ].dropna(subset=[metric])
    if sub.empty:
        return "", np.nan
    row = sub.sort_values(metric, ascending=False).iloc[0]
    return str(row["asset"]), float(row[metric])


def diagnostic_summary(metrics_long: pd.DataFrame, counts: pd.DataFrame, ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, c in counts.iterrows():
        scope = c["threshold_scope"]
        pct = float(c["threshold_percentile"])
        tv = float(c["threshold_value"])
        state_counts = [int(c[state]) for state in STATE_ORDER]
        real_asset, real_sharpe = best_asset(metrics_long, scope, pct, "FLAT_HIGH_RATE_NORMAL", REAL_ASSETS, "sharpe_ratio")
        low_def, low_def_sharpe = best_asset(metrics_long, scope, pct, "FLAT_LOW_RATE_STRESS", DEFENSIVE_ASSETS, "sharpe_ratio")
        high_def, high_def_sharpe = best_asset(metrics_long, scope, pct, "FLAT_HIGH_RATE_STRESS", DEFENSIVE_ASSETS, "sharpe_ratio")
        spy_high_sharpe = value_for(metrics_long, scope, pct, "FLAT_HIGH_RATE_NORMAL", "SPY", "sharpe_ratio")
        row = {
            "threshold_scope": scope,
            "threshold_percentile": pct,
            "threshold_value": tv,
            "low_rate_normal_n": int(c["FLAT_LOW_RATE_NORMAL"]),
            "low_rate_stress_n": int(c["FLAT_LOW_RATE_STRESS"]),
            "high_rate_normal_n": int(c["FLAT_HIGH_RATE_NORMAL"]),
            "high_rate_stress_n": int(c["FLAT_HIGH_RATE_STRESS"]),
            "low_rate_normal_spy_sharpe": value_for(metrics_long, scope, pct, "FLAT_LOW_RATE_NORMAL", "SPY", "sharpe_ratio"),
            "low_rate_normal_spy_rank": rank_for(ranking, scope, pct, "FLAT_LOW_RATE_NORMAL", "sharpe_ratio", "SPY"),
            "high_rate_normal_best_real_asset": real_asset,
            "high_rate_normal_best_real_asset_sharpe": real_sharpe,
            "high_rate_normal_spy_sharpe": spy_high_sharpe,
            "high_rate_normal_real_asset_minus_spy_sharpe": real_sharpe - spy_high_sharpe if pd.notna(real_sharpe) and pd.notna(spy_high_sharpe) else np.nan,
            "low_rate_stress_best_defensive_asset": low_def,
            "low_rate_stress_best_defensive_sharpe": low_def_sharpe,
            "high_rate_stress_best_defensive_asset": high_def,
            "high_rate_stress_best_defensive_sharpe": high_def_sharpe,
            "sample_imbalance_ratio": min(state_counts) / max(state_counts) if max(state_counts) else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def recommended_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        spy_top_two = pd.notna(row["low_rate_normal_spy_rank"]) and row["low_rate_normal_spy_rank"] <= 2
        real_beats_spy = pd.notna(row["high_rate_normal_real_asset_minus_spy_sharpe"]) and row["high_rate_normal_real_asset_minus_spy_sharpe"] > 0
        enough_balance = pd.notna(row["sample_imbalance_ratio"]) and row["sample_imbalance_ratio"] >= 0.15
        defensive_ok = row["low_rate_stress_best_defensive_asset"] in DEFENSIVE_ASSETS and row["high_rate_stress_best_defensive_asset"] in DEFENSIVE_ASSETS
        score = int(spy_top_two) + int(real_beats_spy) + int(enough_balance) + int(defensive_ok)
        rec = dict(row)
        rec["low_rate_normal_spy_top_two"] = spy_top_two
        rec["high_rate_normal_real_asset_beats_spy"] = real_beats_spy
        rec["sample_balance_ok"] = enough_balance
        rec["defensive_assets_win_stress"] = defensive_ok
        rec["diagnostic_score"] = score
        rec["candidate_comment"] = "candidate" if score >= 3 else "diagnostic_only"
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["diagnostic_score", "sample_imbalance_ratio"], ascending=[False, False])


def plot_heatmap(metrics_long: pd.DataFrame, scope: str, metric: str, output_path: Path) -> None:
    sub = metrics_long[metrics_long["threshold_scope"].eq(scope)].copy()
    sub["col"] = sub["threshold_percentile"].map(lambda x: f"{int(x*100)}%") + "\n" + sub["state"].str.replace("FLAT_", "", regex=False)
    pivot = sub.pivot_table(index="asset", columns="col", values=metric, aggfunc="first").reindex(index=ASSET_ORDER)
    fig, ax = plt.subplots(figsize=(18, 5.5))
    values = pivot.values.astype(float)
    cmap = "RdYlGn" if metric != "max_drawdown" else "Reds_r"
    im = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if np.isnan(val):
                txt = ""
            elif metric == "sharpe_ratio":
                txt = f"{val:.2f}"
            else:
                txt = f"{val:.0%}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7)
    ax.set_title(f"{scope}: {metric} by GS10 threshold and FLAT state")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_line_charts(summary: pd.DataFrame, ranking: pd.DataFrame, scope: str) -> None:
    sub = summary[summary["threshold_scope"].eq(scope)].sort_values("threshold_percentile")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sub["threshold_percentile"] * 100, sub["high_rate_normal_real_asset_minus_spy_sharpe"], marker="o")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{scope}: high-rate normal real asset Sharpe minus SPY")
    ax.set_xlabel("GS10 threshold percentile")
    ax.set_ylabel("max(GOLD, CMDTY_FUT) Sharpe - SPY Sharpe")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{scope}_high_rate_normal_real_asset_minus_spy.png", dpi=170)
    plt.close(fig)

    rank_sub = ranking[
        ranking["threshold_scope"].eq(scope)
        & ranking["state"].eq("FLAT_LOW_RATE_NORMAL")
        & ranking["metric"].eq("sharpe_ratio")
    ].sort_values("threshold_percentile")
    ranks = []
    for _, row in rank_sub.iterrows():
        ranks.append(next((i for i in range(1, 6) if row.get(f"rank_{i}") == "SPY"), np.nan))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rank_sub["threshold_percentile"] * 100, ranks, marker="o")
    ax.invert_yaxis()
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_title(f"{scope}: low-rate normal SPY Sharpe rank")
    ax.set_xlabel("GS10 threshold percentile")
    ax.set_ylabel("SPY Sharpe rank")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{scope}_low_rate_normal_spy_rank.png", dpi=170)
    plt.close(fig)


def plot_state_counts(counts: pd.DataFrame, scope: str) -> None:
    sub = counts[counts["threshold_scope"].eq(scope)].sort_values("threshold_percentile")
    x = np.arange(len(sub))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, state in enumerate(STATE_ORDER):
        ax.bar(x + (idx - 1.5) * width, sub[state], width=width, label=state)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(p*100)}%" for p in sub["threshold_percentile"]])
    ax.set_title(f"{scope}: FLAT state counts by GS10 threshold")
    ax.set_xlabel("GS10 threshold percentile")
    ax.set_ylabel("Observation count")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{scope}_state_counts.png", dpi=170)
    plt.close(fig)


def plot_all(metrics_long: pd.DataFrame, counts: pd.DataFrame, ranking: pd.DataFrame, summary: pd.DataFrame) -> None:
    for scope in ["full_sample_gs10", "flat_sample_gs10"]:
        plot_heatmap(metrics_long, scope, "sharpe_ratio", FIG_DIR / f"{scope}_sharpe_heatmap.png")
        plot_heatmap(metrics_long, scope, "annualized_return", FIG_DIR / f"{scope}_annualized_return_heatmap.png")
        plot_heatmap(metrics_long, scope, "max_drawdown", FIG_DIR / f"{scope}_max_drawdown_heatmap.png")
        plot_line_charts(summary, ranking, scope)
        plot_state_counts(counts, scope)


def ranking_stability_comment(ranking: pd.DataFrame, scope: str, state: str, metric: str) -> str:
    mid = ranking[
        ranking["threshold_scope"].eq(scope)
        & ranking["state"].eq(state)
        & ranking["metric"].eq(metric)
        & ranking["threshold_percentile"].isin([0.40, 0.50, 0.60])
    ]
    if mid.empty:
        return "insufficient data"
    top_assets = mid["rank_1"].tolist()
    if len(set(top_assets)) == 1:
        return f"stable top asset across 40/50/60%: {top_assets[0]}"
    return "ranking changes across 40/50/60%, possible threshold sensitivity"


def write_readme(
    input_path: Path,
    fields: Fields,
    full: pd.DataFrame,
    flat: pd.DataFrame,
    counts: pd.DataFrame,
    summary: pd.DataFrame,
    candidates: pd.DataFrame,
    ranking: pd.DataFrame,
    freq: str,
    annualization: int,
) -> None:
    lines = [
        "# FLAT GS10 Threshold Robustness Scan",
        "",
        "## 1. Experiment Purpose",
        "",
        "FLAT regime describes yield-curve shape but not the absolute rate level. This experiment scans GS10 thresholds to test whether high-rate and low-rate FLAT states have different asset behavior.",
        "",
        "This is a diagnostic, not a strategy optimization. It evaluates ranking stability, state sample balance, and stress robustness rather than selecting the best backtest threshold.",
        "",
        "## 2. Data",
        "",
        f"- Input panel: `{input_path.relative_to(ROOT).as_posix()}`",
        f"- Frequency: `{freq}`",
        f"- Annualization: {annualization}",
        f"- Regime field: `{fields.regime}`",
        f"- Stress field: `{fields.stress}`",
        f"- GS10 field: `{fields.gs10}`",
        f"- Total sample n: {len(full)}",
        f"- FLAT sample n: {len(flat)}",
        f"- FLAT stress n: {int(flat['flat_stress'].sum())}",
        f"- FLAT normal n: {int((~flat['flat_stress']).sum())}",
        "",
        "Assets used:",
        "",
    ]
    lines.extend(f"- {asset}: `{col}`" for asset, col in fields.returns.items())
    lines.extend(
        [
            "",
            "## 3. Threshold Scopes",
            "",
            "- `full_sample_gs10`: threshold percentiles computed from all available GS10 dates.",
            "- `flat_sample_gs10`: threshold percentiles computed only from FLAT observations.",
            "",
            "Scanned percentiles: 30%, 40%, 50%, 60%, 70%.",
            "",
            "## 4. Sample Counts Summary",
            "",
            counts.to_markdown(index=False),
            "",
            "## 5. Normal Period Findings",
            "",
        ]
    )
    for scope in ["full_sample_gs10", "flat_sample_gs10"]:
        lines.append(f"- {scope} low-rate normal SPY rank stability: {ranking_stability_comment(ranking, scope, 'FLAT_LOW_RATE_NORMAL', 'sharpe_ratio')}.")
        scope_summary = summary[summary["threshold_scope"].eq(scope)]
        share_real = (scope_summary["high_rate_normal_real_asset_minus_spy_sharpe"] > 0).mean()
        lines.append(f"- {scope} high-rate normal real asset Sharpe > SPY in {share_real:.0%} of scanned thresholds.")
    lines.extend(
        [
            "",
            "## 6. Stress Period Findings",
            "",
        ]
    )
    for scope in ["full_sample_gs10", "flat_sample_gs10"]:
        scope_summary = summary[summary["threshold_scope"].eq(scope)]
        low_winners = scope_summary["low_rate_stress_best_defensive_asset"].value_counts().to_dict()
        high_winners = scope_summary["high_rate_stress_best_defensive_asset"].value_counts().to_dict()
        lines.append(f"- {scope} low-rate stress defensive winners: {low_winners}.")
        lines.append(f"- {scope} high-rate stress defensive winners: {high_winners}.")
    lines.extend(
        [
            "",
            "## 7. Threshold Stability",
            "",
        ]
    )
    for scope in ["full_sample_gs10", "flat_sample_gs10"]:
        mid = candidates[candidates["threshold_scope"].eq(scope) & candidates["threshold_percentile"].isin([0.40, 0.50, 0.60])]
        avg_score = mid["diagnostic_score"].mean() if not mid.empty else np.nan
        lines.append(f"- {scope}: average diagnostic score around 40/50/60% = {avg_score:.2f}.")
    lines.extend(
        [
            "",
            "## 8. Recommended Threshold Candidates",
            "",
            candidates.head(10)[
                [
                    "threshold_scope",
                    "threshold_percentile",
                    "threshold_value",
                    "sample_imbalance_ratio",
                    "diagnostic_score",
                    "candidate_comment",
                ]
            ].to_markdown(index=False),
            "",
            "## 9. Strategy Implications",
            "",
            "- If low-rate normal repeatedly ranks SPY near the top, FLAT_LOW_RATE_NORMAL may allow more equity exposure.",
            "- If high-rate normal repeatedly favors GOLD or CMDTY_FUT over SPY, FLAT_HIGH_RATE_NORMAL may require more real-asset exposure.",
            "- If low-rate stress favors GOLD / IEF / CASH, stress defense remains distinct from rate-level classification.",
            "- If high-rate stress favors CASH / GOLD while IEF weakens, rate/inflation shocks may require different hedge diagnostics.",
            "",
            "This experiment should not be promoted directly into final strategy rules without further out-of-sample and proxy robustness checks.",
        ]
    )
    (OUT / "README_flat_gs10_threshold_robustness.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    full_raw, input_path = load_panel()
    fields = resolve_fields(full_raw)
    full, flat, annualization, freq = prepare_panel(full_raw, fields)

    metrics_long, counts = scan_thresholds(full, flat, annualization)
    ranking = rank_assets(metrics_long)
    summary = diagnostic_summary(metrics_long, counts, ranking)
    candidates = recommended_candidates(summary)

    metrics_long.to_csv(TABLE_DIR / "flat_gs10_threshold_metrics_long.csv", index=False)
    ranking.to_csv(TABLE_DIR / "asset_ranking_by_threshold.csv", index=False)
    summary.to_csv(TABLE_DIR / "threshold_diagnostic_summary.csv", index=False)
    counts.to_csv(TABLE_DIR / "state_counts_by_threshold.csv", index=False)
    candidates.to_csv(TABLE_DIR / "recommended_threshold_candidates.csv", index=False)
    plot_all(metrics_long, counts, ranking, summary)
    write_readme(input_path, fields, full, flat, counts, summary, candidates, ranking, freq, annualization)

    print("FLAT GS10 threshold robustness scan completed.")
    print(f"input_file: {input_path.relative_to(ROOT).as_posix()}")
    print(f"total_sample_n: {len(full)}")
    print(f"flat_sample_n: {len(flat)}")
    print(f"flat_stress_n: {int(flat['flat_stress'].sum())}")
    print(f"flat_normal_n: {int((~flat['flat_stress']).sum())}")
    for scope in ["full_sample_gs10", "flat_sample_gs10"]:
        print(f"{scope} thresholds:")
        for _, row in counts[counts["threshold_scope"].eq(scope)].iterrows():
            counts_str = ", ".join(f"{state}={int(row[state])}" for state in STATE_ORDER)
            print(f"  p{int(row['threshold_percentile']*100)} value={row['threshold_value']:.4f}; {counts_str}")
    print("recommended_threshold_candidates:")
    for _, row in candidates.head(5).iterrows():
        print(
            f"  {row['threshold_scope']} p{int(row['threshold_percentile']*100)} "
            f"value={row['threshold_value']:.4f} score={row['diagnostic_score']} "
            f"imbalance={row['sample_imbalance_ratio']:.3f}"
        )
    print(f"output_dir: {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
