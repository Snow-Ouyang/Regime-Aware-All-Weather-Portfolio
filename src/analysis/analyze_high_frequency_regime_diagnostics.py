from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
RAW_MACRO_DIR = ROOT / "data" / "raw" / "macro"
RAW_RATE_DIR = RAW_MACRO_DIR / "rate"
RAW_VOL_DIR = RAW_MACRO_DIR / "volatility"
RAW_CREDIT_DIR = RAW_MACRO_DIR / "Credit"
DAILY_CLOSE_PATH = ROOT / "data" / "processed" / "assets" / "daily_adjusted_close.csv"
DAILY_RET_PATH = ROOT / "data" / "processed" / "assets" / "daily_returns.csv"
REGIME_PATH = ROOT / "data" / "processed" / "risk_factors" / "core_risk_factor_panel.csv"

RESULTS_DIR = ROOT / "results" / "high_frequency_regime_diagnostics"
FIGURES_DIR = ROOT / "figures" / "high_frequency_regime_diagnostics"
DISTRIBUTION_DIR = FIGURES_DIR / "distributions"

PANEL_PATH = RESULTS_DIR / "high_frequency_regime_feature_panel.csv"
COVERAGE_PATH = RESULTS_DIR / "high_frequency_feature_coverage.csv"
DISTRIBUTION_SUMMARY_PATH = RESULTS_DIR / "high_frequency_feature_distribution_by_regime.csv"
PERCENTILE_SUMMARY_PATH = RESULTS_DIR / "high_frequency_feature_regime_percentile_summary.csv"
REPORT_PATH = RESULTS_DIR / "HIGH_FREQUENCY_REGIME_DIAGNOSTICS.md"
PERCENTILE_BARS_PATH = FIGURES_DIR / "high_frequency_regime_percentile_bars.png"
PERCENTILE_HEATMAP_PATH = FIGURES_DIR / "high_frequency_regime_percentile_heatmap.png"

REGIME_ORDER = [
    "Late-Cycle / Inflationary Flat Curve",
    "Low-Rate / Steep Curve",
    "High-Rate / Inflation-Pressure",
    "Deflationary Macro-Financial Stress",
]

REGIME_COLORS = {
    "Late-Cycle / Inflationary Flat Curve": "#d95f02",
    "Low-Rate / Steep Curve": "#1b9e77",
    "High-Rate / Inflation-Pressure": "#7570b3",
    "Deflationary Macro-Financial Stress": "#e7298a",
}

FINAL_FEATURES = [
    "CREDIT_SPREAD_BAA_AAA",
    "DGS10",
    "DGS1",
    "TERM_SPREAD_10Y_1Y",
    "VIX_LEVEL",
    "SPY_DRAWDOWN_FROM_PREVIOUS_HIGH",
    "VIX_MAX_MONTH",
]

PRIMARY_FEATURES = [
    "CREDIT_SPREAD_BAA_AAA",
    "DGS10",
    "TERM_SPREAD_10Y_1Y",
    "VIX_LEVEL",
    "VIX_MAX_MONTH",
    "SPY_DRAWDOWN_FROM_PREVIOUS_HIGH",
]

REMOVED_FEATURES = [
    "WAAA",
    "WBAA",
    "D_CREDIT_SPREAD_BAA_AAA",
    "D_DGS10",
    "D_DGS1",
    "CREDIT_SPREAD_BAA_AAA_MAX_MONTH",
    "D_TERM_SPREAD_10Y_1Y",
    "SPY_DAILY_RETURN",
    "SPY_21D_RETURN",
    "SPY_63D_RETURN",
    "VIX_21D_MAX",
    "D_VIX",
    "SPY_DRAWDOWN_FROM_6M_HIGH",
    "SPY_DRAWDOWN_FROM_12M_HIGH",
    "3M change features",
    "redundant EOM suffix variables",
]


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DISTRIBUTION_DIR.mkdir(parents=True, exist_ok=True)


def to_month_end(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.to_period("M").dt.to_timestamp("M")


def detect_date_column(df: pd.DataFrame) -> str:
    for col in ["date", "DATE", "observation_date", "Date", "month", "Month"]:
        if col in df.columns:
            return col
    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().mean() > 0.8:
            return col
    raise ValueError("No date column detected.")


def detect_value_column(df: pd.DataFrame, exclude: str) -> str:
    candidates = [c for c in df.columns if c != exclude]
    ranked = []
    for col in candidates:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().mean() > 0.8:
            ranked.append((col, numeric.notna().sum()))
    if ranked:
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[0][0]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError("No value column detected.")


def load_series(path: Path, name: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    date_col = detect_date_column(raw)
    value_col = detect_value_column(raw, date_col)
    df = raw[[date_col, value_col]].copy()
    df.columns = ["date", name]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[name] = pd.to_numeric(df[name], errors="coerce")
    return df.dropna(subset=["date", name]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def load_regime_labels() -> pd.DataFrame:
    regime = pd.read_csv(REGIME_PATH, usecols=["date", "regime", "regime_name"])
    regime["date"] = to_month_end(pd.to_datetime(regime["date"]))
    return regime.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def attach_regime(df: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["regime_month"] = to_month_end(out["date"])
    out = out.merge(regime, left_on="regime_month", right_on="date", how="left", suffixes=("", "_regime"))
    return out.drop(columns=["date_regime"])


def make_long_feature(df: pd.DataFrame, feature: str, source_frequency: str) -> pd.DataFrame:
    if "regime_month" not in df.columns:
        df = df.copy()
        df["regime_month"] = df["date"]
    cols = ["date", "regime_month", "regime", "regime_name", feature]
    out = df[cols].dropna(subset=[feature]).copy()
    out = out.rename(columns={feature: "value"})
    out["feature"] = feature
    out["source_frequency"] = source_frequency
    return out[["date", "regime_month", "regime", "regime_name", "feature", "value", "source_frequency"]]


def monthly_last_mean_max(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    work = df.copy()
    work["date"] = to_month_end(work["date"])
    grouped = work.groupby("date")[value_col]
    return pd.DataFrame(
        {
            "date": grouped.last().index,
            value_col: grouped.last().values,
            f"{value_col}_MAX_MONTH": grouped.max().values,
            f"{value_col}_MEAN_MONTH": grouped.mean().values,
        }
    )


def build_vix_daily(vix_raw: pd.DataFrame) -> pd.DataFrame:
    return vix_raw[["date", "VIX_LEVEL"]].sort_values("date").reset_index(drop=True)


def build_vix_monthly(vix_raw: pd.DataFrame) -> pd.DataFrame:
    monthly = monthly_last_mean_max(vix_raw, "VIX_LEVEL").sort_values("date").reset_index(drop=True)
    monthly["VIX_MAX_MONTH"] = monthly["VIX_LEVEL_MAX_MONTH"]
    return monthly[["date", "VIX_MAX_MONTH"]]


def build_rates_monthly(dgs10_raw: pd.DataFrame, dgs1_raw: pd.DataFrame) -> pd.DataFrame:
    dgs10 = monthly_last_mean_max(dgs10_raw, "DGS10")[["date", "DGS10"]]
    dgs1 = monthly_last_mean_max(dgs1_raw, "DGS1")[["date", "DGS1"]]
    out = dgs10.merge(dgs1, on="date", how="inner").sort_values("date").reset_index(drop=True)
    out["TERM_SPREAD_10Y_1Y"] = out["DGS10"] - out["DGS1"]
    return out


def build_credit_monthly(waaa_raw: pd.DataFrame, wbaa_raw: pd.DataFrame) -> pd.DataFrame:
    merged = waaa_raw.merge(wbaa_raw, on="date", how="inner").sort_values("date").reset_index(drop=True)
    merged["CREDIT_SPREAD_BAA_AAA"] = merged["WBAA"] - merged["WAAA"]
    merged["date"] = to_month_end(merged["date"])
    grouped = merged.groupby("date")["CREDIT_SPREAD_BAA_AAA"]
    out = pd.DataFrame(
        {
            "date": grouped.last().index,
            "CREDIT_SPREAD_BAA_AAA": grouped.last().values,
        }
    ).sort_values("date").reset_index(drop=True)
    return out


def build_spy_monthly() -> pd.DataFrame:
    daily_close = pd.read_csv(DAILY_CLOSE_PATH)
    daily_ret = pd.read_csv(DAILY_RET_PATH)
    daily_close["date"] = pd.to_datetime(daily_close["date"])
    daily_ret["date"] = pd.to_datetime(daily_ret["date"])
    dclose = daily_close[["date", "SPY"]].dropna().rename(columns={"SPY": "SPY_CLOSE"})
    dret = daily_ret[["date", "SPY"]].dropna().rename(columns={"SPY": "SPY_DAILY_RETURN"})
    spy = dclose.merge(dret, on="date", how="inner").sort_values("date").reset_index(drop=True)
    spy["RUNNING_HIGH"] = spy["SPY_CLOSE"].cummax()
    spy["SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"] = spy["SPY_CLOSE"] / spy["RUNNING_HIGH"] - 1.0
    spy["ROLLING_6M_HIGH"] = spy["SPY_CLOSE"].rolling(126, min_periods=63).max()
    spy["ROLLING_12M_HIGH"] = spy["SPY_CLOSE"].rolling(252, min_periods=126).max()
    spy["SPY_DRAWDOWN_FROM_6M_HIGH"] = spy["SPY_CLOSE"] / spy["ROLLING_6M_HIGH"] - 1.0
    spy["SPY_DRAWDOWN_FROM_12M_HIGH"] = spy["SPY_CLOSE"] / spy["ROLLING_12M_HIGH"] - 1.0
    spy["SPY_REALIZED_VOL_1M"] = spy["SPY_DAILY_RETURN"].rolling(21, min_periods=15).std() * np.sqrt(252)
    spy["date"] = to_month_end(spy["date"])
    out = (
        spy.groupby("date")
        .agg(
            SPY_DRAWDOWN_FROM_PREVIOUS_HIGH=("SPY_DRAWDOWN_FROM_PREVIOUS_HIGH", "last"),
            SPY_REALIZED_VOL_1M=("SPY_REALIZED_VOL_1M", "last"),
        )
        .reset_index()
    )
    return out.sort_values("date").reset_index(drop=True)


def compute_coverage(long_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, grp in long_panel.groupby("feature"):
        row = {
            "factor_name": feature,
            "source_frequency": grp["source_frequency"].iloc[0],
            "first_valid_date": grp["date"].min().strftime("%Y-%m-%d"),
            "last_valid_date": grp["date"].max().strftime("%Y-%m-%d"),
            "valid_obs": int(len(grp)),
            "missing_obs": np.nan,
            "missing_ratio": np.nan,
        }
        for regime_name in REGIME_ORDER:
            subset = grp.loc[grp["regime_name"] == regime_name]
            row[f"{regime_name}__valid_obs"] = int(len(subset))
            row[f"{regime_name}__coverage_ratio"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["source_frequency", "factor_name"]).reset_index(drop=True)


def percentile_rank(full_sample: pd.Series, value: float) -> float:
    clean = full_sample.dropna()
    if clean.empty or pd.isna(value):
        return np.nan
    return float((clean <= value).mean() * 100.0)


def compute_distribution_summaries(long_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dist_rows = []
    pct_rows = []
    for feature, grp in long_panel.groupby("feature"):
        full = grp["value"].dropna()
        for regime_name in REGIME_ORDER:
            subset = grp.loc[grp["regime_name"] == regime_name, "value"].dropna()
            mean_value = subset.mean() if not subset.empty else np.nan
            row = {
                "feature": feature,
                "source_frequency": grp["source_frequency"].iloc[0],
                "regime_name": regime_name,
                "n_obs": int(len(subset)),
                "mean": float(mean_value) if pd.notna(mean_value) else np.nan,
                "median": float(subset.median()) if not subset.empty else np.nan,
                "std": float(subset.std()) if len(subset) > 1 else np.nan,
                "min": float(subset.min()) if not subset.empty else np.nan,
                "max": float(subset.max()) if not subset.empty else np.nan,
                "p05": float(subset.quantile(0.05)) if not subset.empty else np.nan,
                "p10": float(subset.quantile(0.10)) if not subset.empty else np.nan,
                "p25": float(subset.quantile(0.25)) if not subset.empty else np.nan,
                "p50": float(subset.quantile(0.50)) if not subset.empty else np.nan,
                "p75": float(subset.quantile(0.75)) if not subset.empty else np.nan,
                "p90": float(subset.quantile(0.90)) if not subset.empty else np.nan,
                "p95": float(subset.quantile(0.95)) if not subset.empty else np.nan,
                "percentile_rank_of_regime_mean": percentile_rank(full, mean_value),
            }
            dist_rows.append(row)
            pct_rows.append(
                {
                    "feature": feature,
                    "source_frequency": grp["source_frequency"].iloc[0],
                    "regime_name": regime_name,
                    "regime_mean": row["mean"],
                    "regime_median": row["median"],
                    "percentile_rank_of_regime_mean": row["percentile_rank_of_regime_mean"],
                    "percentile_rank_of_regime_median": percentile_rank(full, row["median"]),
                }
            )
    return pd.DataFrame(dist_rows), pd.DataFrame(pct_rows)


def plot_distribution_by_regime(long_panel: pd.DataFrame, feature: str) -> Path | None:
    feature_df = long_panel.loc[long_panel["feature"] == feature].copy()
    series = feature_df["value"].dropna()
    if series.empty:
        return None
    bins = np.histogram_bin_edges(series, bins="auto")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    plot_order = ["Full Sample", *REGIME_ORDER]
    y_max = 0
    hist_data = []
    for label in plot_order:
        values = series if label == "Full Sample" else feature_df.loc[feature_df["regime_name"] == label, "value"].dropna()
        counts, _ = np.histogram(values, bins=bins)
        hist_data.append((label, values, counts))
        y_max = max(y_max, counts.max() if len(counts) else 0)
    for ax, (label, values, _) in zip(axes, hist_data):
        ax.hist(series, bins=bins, color="#d9d9d9", alpha=0.65, edgecolor="white")
        if label != "Full Sample":
            ax.hist(values, bins=bins, color=REGIME_COLORS[label], alpha=0.85, edgecolor="white")
            if len(values) < 10:
                ax.text(0.98, 0.92, "Low n", transform=ax.transAxes, ha="right", va="top", fontsize=10, color="#7f0000")
        ax.set_title(label)
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
        ax.set_ylim(0, y_max * 1.10 if y_max > 0 else 1)
        ax.grid(alpha=0.15)
    axes[-1].axis("off")
    fig.suptitle(f"Distribution of {feature} by Regime", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = DISTRIBUTION_DIR / f"{feature}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_percentile_heatmap(percentiles: pd.DataFrame) -> None:
    subset = percentiles[percentiles["feature"].isin(PRIMARY_FEATURES)]
    pivot = subset.pivot(index="regime_name", columns="feature", values="percentile_rank_of_regime_mean").reindex(REGIME_ORDER)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlBu_r", vmin=0, vmax=100, ax=ax, cbar_kws={"label": "Percentile"})
    ax.set_title("High-Frequency Regime Mean Percentile Heatmap")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(PERCENTILE_HEATMAP_PATH, dpi=180)
    plt.close(fig)


def plot_percentile_bars(percentiles: pd.DataFrame) -> None:
    subset = percentiles[percentiles["feature"].isin(PRIMARY_FEATURES)]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharey=True)
    axes = axes.flatten()
    for ax, regime_name in zip(axes, REGIME_ORDER):
        regime_df = subset.loc[subset["regime_name"] == regime_name].set_index("feature").reindex(PRIMARY_FEATURES)
        ax.bar(range(len(regime_df)), regime_df["percentile_rank_of_regime_mean"], color=REGIME_COLORS[regime_name], alpha=0.85)
        ax.set_title(regime_name)
        ax.set_xticks(range(len(regime_df)))
        ax.set_xticklabels(regime_df.index, rotation=45, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentile")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("High-Frequency Regime Mean Percentile Profiles", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(PERCENTILE_BARS_PATH, dpi=180)
    plt.close(fig)


def write_report(loaded_files: list[str], long_panel: pd.DataFrame, coverage: pd.DataFrame, percentile_summary: pd.DataFrame) -> None:
    top_distinctive = (
        percentile_summary.groupby("feature")["percentile_rank_of_regime_mean"]
        .agg(lambda s: s.max() - s.min())
        .reset_index(name="dispersion")
        .sort_values("dispersion", ascending=False)
        .head(10)
    )
    lines = [
        "# High-Frequency Regime Diagnostics",
        "",
        "## Purpose",
        "",
        "This module keeps the fixed regime labels and checks whether a smaller set of higher-frequency market variables lines up with those historical regimes.",
        "",
        "## Data",
        "",
        "Loaded raw files:",
        *[f"- `{name}`" for name in loaded_files],
        "",
        "## Final feature set",
        "",
        *[f"- `{feature}`" for feature in FINAL_FEATURES],
        "",
        "## Why removed variables were dropped",
        "",
        "- 21D/63D returns and generic 3-month change features are window-based and can be lagging.",
        "- Daily returns are too noisy for regime diagnostics.",
        "- WAAA/WBAA standalone levels are less useful than the BAA-AAA spread.",
        "- EOM suffixes were removed from final names to reduce redundancy.",
        "",
        "## Why SPY_DRAWDOWN_FROM_PREVIOUS_HIGH was added",
        "",
        "- It measures current market stress relative to the prior peak.",
        "- It is more interpretable than fixed-window returns.",
        "- It can identify equity stress earlier than slow macro variables.",
        "",
        "## Early interpretation",
        "",
        "- Credit spread above roughly 1.5 appears consistent with abnormal credit stress.",
        "- If credit spread is high and DGS10 is also high, that is more consistent with High-Rate / Inflation-Pressure.",
        "- If credit spread is high but DGS10 is low, that is more consistent with Deflationary Macro-Financial Stress.",
        "- DGS10 above roughly 6.0-6.5 remains a candidate threshold for extreme high-rate environments.",
        "",
        "## Top distinctive features",
        "",
        *[f"- `{r.feature}`: regime mean percentile dispersion = {r.dispersion:.1f}" for r in top_distinctive.itertuples()],
        "",
        "## Caveat",
        "",
        "These are diagnostic observations, not final trading rules. Any future trigger must be tested with rolling or expanding thresholds to avoid look-ahead bias.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()

    loaded_files = [
        "VIXCLS.csv",
        "DGS1.csv",
        "DGS10.csv",
        "WAAA.csv",
        "WBAA.csv",
        "daily_adjusted_close.csv",
        "daily_returns.csv",
    ]

    regime = load_regime_labels()
    vix_raw = load_series(RAW_VOL_DIR / "VIXCLS.csv", "VIX_LEVEL")
    dgs10_raw = load_series(RAW_RATE_DIR / "DGS10.csv", "DGS10")
    dgs1_raw = load_series(RAW_RATE_DIR / "DGS1.csv", "DGS1")
    waaa_raw = load_series(RAW_CREDIT_DIR / "WAAA.csv", "WAAA")
    wbaa_raw = load_series(RAW_CREDIT_DIR / "WBAA.csv", "WBAA")

    panel = regime.copy()
    for block in [
        build_vix_monthly(vix_raw),
        build_rates_monthly(dgs10_raw, dgs1_raw),
        build_credit_monthly(waaa_raw, wbaa_raw),
        build_spy_monthly(),
    ]:
        panel = panel.merge(block, on="date", how="left")

    panel = panel.sort_values("date").reset_index(drop=True)
    vix_daily = attach_regime(build_vix_daily(vix_raw), regime)
    source_map = {feature: (panel, "monthly") for feature in FINAL_FEATURES if feature != "VIX_LEVEL"}
    source_map["VIX_LEVEL"] = (vix_daily, "daily")
    long_frames = [make_long_feature(df, feature, freq) for feature, (df, freq) in source_map.items() if feature in df.columns]
    long_panel = pd.concat(long_frames, ignore_index=True).sort_values(["feature", "date"]).reset_index(drop=True)
    long_panel.to_csv(PANEL_PATH, index=False)

    coverage = compute_coverage(long_panel)
    coverage.to_csv(COVERAGE_PATH, index=False)

    distribution_summary, percentile_summary = compute_distribution_summaries(long_panel)
    distribution_summary.to_csv(DISTRIBUTION_SUMMARY_PATH, index=False)
    percentile_summary.to_csv(PERCENTILE_SUMMARY_PATH, index=False)

    saved_distribution_paths = []
    for feature in FINAL_FEATURES:
        out = plot_distribution_by_regime(long_panel, feature)
        if out is not None:
            saved_distribution_paths.append(out)

    plot_percentile_bars(percentile_summary)
    plot_percentile_heatmap(percentile_summary)
    write_report(loaded_files, long_panel, coverage, percentile_summary)

    print("Final feature list:")
    for feature in FINAL_FEATURES:
        print(f"- {feature}")
    print("Removed feature list:")
    for feature in REMOVED_FEATURES:
        print(f"- {feature}")
    print("WAAA/WBAA standalone plots generated: False")
    print("EOM suffix variables retained in final outputs: False")
    print("VIX_LEVEL uses daily observations: True")
    print(f"Final panel date range: {long_panel['date'].min().strftime('%Y-%m-%d')} to {long_panel['date'].max().strftime('%Y-%m-%d')}")
    for path in [
        PANEL_PATH,
        COVERAGE_PATH,
        DISTRIBUTION_SUMMARY_PATH,
        PERCENTILE_SUMMARY_PATH,
        REPORT_PATH,
        PERCENTILE_BARS_PATH,
        PERCENTILE_HEATMAP_PATH,
        *saved_distribution_paths,
    ]:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
