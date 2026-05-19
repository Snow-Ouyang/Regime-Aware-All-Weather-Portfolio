from __future__ import annotations

from pathlib import Path
import site
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "macro"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "regime"
FIGURES = ROOT / "figures" / "regime"
VENDOR = ROOT / ".vendor"

MODEL_FEATURES = [
    "growth_pc1",
    "inflation_pc1",
    "gs10",
    "term_spread_10y_1y",
    "credit_spread",
]
REGIME_ORDER = [
    "Late-Cycle / Inflationary Flat Curve",
    "Low-Rate / Steep Curve",
    "High-Rate / Inflation-Pressure",
    "Deflationary Macro-Financial Stress",
]
REGIME_COLORS = {
    "Late-Cycle / Inflationary Flat Curve": "#c44e52",
    "Low-Rate / Steep Curve": "#4c72b0",
    "High-Rate / Inflation-Pressure": "#dd8452",
    "Deflationary Macro-Financial Stress": "#8172b3",
}
DISPLAY_NAMES = {
    "growth_pc1": "Growth PC1",
    "inflation_pc1": "Inflation PC1",
    "gs10": "10Y Treasury Yield",
    "term_spread_10y_1y": "10Y-1Y Term Spread",
    "credit_spread": "BAA-AAA Credit Spread",
}
PENALTY = 1.0
N_STATES = 4
SEEDS = list(range(10))


def ensure_project_dirs() -> None:
    for path in [PROCESSED, RESULTS, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)


def ensure_jumpmodels_importable() -> None:
    if str(VENDOR) not in sys.path:
        sys.path.append(str(VENDOR))
    for path_str in site.getsitepackages() + [site.getusersitepackages()]:
        if path_str and path_str not in sys.path:
            sys.path.append(path_str)


def standardize_series(path: Path, date_col: str, value_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = df[[date_col, value_col]].copy()
    out.columns = ["date", "value"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date", "value"]).sort_values("date").drop_duplicates("date", keep="last")
    return out.reset_index(drop=True)


def infer_month_gap(series: pd.Series) -> int:
    diffs = series.sort_values().diff().dropna()
    if diffs.empty:
        return 1
    months = int(round(diffs.dt.days.median() / 30.4375))
    return max(months, 1)


def annualized_percent_change(df: pd.DataFrame, value_col: str = "value") -> pd.Series:
    gap = infer_month_gap(df["date"])
    growth = df[value_col] / df[value_col].shift(1)
    return (growth.pow(12 / gap) - 1) * 100


def zscore_frame(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    cols = list(cols)
    return pd.DataFrame(StandardScaler().fit_transform(df[cols]), columns=cols, index=df.index)


def extract_oriented_pc1(df: pd.DataFrame, cols: list[str], out_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_df = df.dropna(subset=cols).copy()
    X = zscore_frame(model_df, cols)
    pca = PCA(n_components=1)
    scores = pca.fit_transform(X).ravel()
    loadings = pd.Series(pca.components_[0], index=cols, dtype=float)
    sign_flip = float(loadings.mean()) < 0
    if sign_flip:
        scores = -scores
        loadings = -loadings
    out = model_df[["date"]].copy()
    out[out_name] = scores
    meta = pd.DataFrame(
        {
            "factor_name": [out_name] * len(cols),
            "source_variable": cols,
            "pc1_loading": loadings.values,
            "explained_variance_ratio": [float(pca.explained_variance_ratio_[0])] * len(cols),
            "sign_flip_applied": [bool(sign_flip)] * len(cols),
        }
    )
    return out, meta


def contiguous_run_lengths(series: pd.Series) -> pd.DataFrame:
    if series.empty:
        return pd.DataFrame(columns=["state_raw", "duration"])
    blocks = (series != series.shift()).cumsum()
    runs = series.groupby(blocks).agg(["first", "size"]).reset_index(drop=True)
    runs.columns = ["state_raw", "duration"]
    return runs


def percentile_rank(full_sample: pd.Series, value: float) -> float:
    full_sample = full_sample.dropna()
    if full_sample.empty or pd.isna(value):
        return np.nan
    return float((full_sample <= value).mean() * 100)


def assign_regime_names(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_means = panel.groupby("state_raw")[MODEL_FEATURES].mean()
    stress_score = (
        state_means["credit_spread"].rank(ascending=True)
        + state_means["growth_pc1"].rank(ascending=False)
        + state_means["inflation_pc1"].rank(ascending=False)
    )
    stress_state = int(stress_score.idxmax())
    remaining = [state for state in state_means.index.tolist() if state != stress_state]
    low_rate_state = int(state_means.loc[remaining, "term_spread_10y_1y"].idxmax())
    remaining = [state for state in remaining if state != low_rate_state]
    high_rate_state = int(state_means.loc[remaining, "gs10"].idxmax())
    late_cycle_state = int([state for state in remaining if state != high_rate_state][0])
    mapping = {
        late_cycle_state: "Late-Cycle / Inflationary Flat Curve",
        low_rate_state: "Low-Rate / Steep Curve",
        high_rate_state: "High-Rate / Inflation-Pressure",
        stress_state: "Deflationary Macro-Financial Stress",
    }
    out = panel.copy()
    out["regime_name"] = out["state_raw"].map(mapping)
    out["regime_order"] = out["regime_name"].map({name: idx for idx, name in enumerate(REGIME_ORDER)})
    mapping_df = pd.DataFrame(
        {
            "state_raw": list(mapping.keys()),
            "regime_name": list(mapping.values()),
        }
    ).sort_values("state_raw", ignore_index=True)
    return out, mapping_df


def build_profile_sentence(row: pd.Series) -> str:
    name = row["regime_name"]
    growth_pct = row["growth_pc1_mean_percentile"]
    infl_pct = row["inflation_pc1_mean_percentile"]
    rate_pct = row["gs10_mean_percentile"]
    slope_pct = row["term_spread_10y_1y_mean_percentile"]
    credit_pct = row["credit_spread_mean_percentile"]
    if name == "Deflationary Macro-Financial Stress":
        return (
            f"Extremely weak growth ({growth_pct:.1f} pct) and inflation ({infl_pct:.1f} pct), "
            f"very wide credit spreads ({credit_pct:.1f} pct), lower rates ({rate_pct:.1f} pct), "
            f"and a steeper curve ({slope_pct:.1f} pct)."
        )
    if name == "Low-Rate / Steep Curve":
        return (
            f"Lower-rate backdrop ({rate_pct:.1f} pct) with the steepest curve ({slope_pct:.1f} pct), "
            f"lower inflation pressure ({infl_pct:.1f} pct), and an easier recovery-like macro setting."
        )
    if name == "High-Rate / Inflation-Pressure":
        return (
            f"Very high rate levels ({rate_pct:.1f} pct), elevated inflation pressure ({infl_pct:.1f} pct), "
            f"and wider credit spreads ({credit_pct:.1f} pct) in a restrictive macro environment."
        )
    return (
        f"Flatter curve ({slope_pct:.1f} pct) with moderate positive inflation ({infl_pct:.1f} pct), "
        f"relatively tight credit ({credit_pct:.1f} pct), and a non-crisis late-cycle backdrop."
    )
