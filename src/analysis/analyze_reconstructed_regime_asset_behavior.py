from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
RULE_STATE_INPUT = ROOT / "results" / "rule_diagnostics" / "rule_state_panel.csv"
DAILY_RETURNS_INPUT = ROOT / "data" / "processed" / "assets" / "daily_returns.csv"
DAILY_CLOSE_INPUT = ROOT / "data" / "processed" / "assets" / "daily_adjusted_close.csv"
BACKTEST_PANEL_INPUT = ROOT / "results" / "rule_based_backtest" / "rule_based_daily_backtest_panel.csv"
VIX_PATH = ROOT / "data" / "raw" / "macro" / "volatility" / "VIXCLS.csv"
DGS1_PATH = ROOT / "data" / "raw" / "macro" / "rate" / "DGS1.csv"
DGS10_PATH = ROOT / "data" / "raw" / "macro" / "rate" / "DGS10.csv"
WAAA_PATH = ROOT / "data" / "raw" / "macro" / "Credit" / "WAAA.csv"
WBAA_PATH = ROOT / "data" / "raw" / "macro" / "Credit" / "WBAA.csv"
DTB3_CANDIDATES = [
    ROOT / "data" / "raw" / "rate" / "DTB3.csv",
    ROOT / "data" / "raw" / "macro" / "rate" / "DTB3.csv",
]

RESULTS_DIR = ROOT / "results" / "reconstructed_regime_asset_behavior"
FIGURES_DIR = ROOT / "figures" / "reconstructed_regime_asset_behavior"

REGIME_PANEL_PATH = RESULTS_DIR / "reconstructed_regime_panel.csv"
REGIME_COUNTS_PATH = RESULTS_DIR / "reconstructed_regime_counts.csv"
FEATURE_SUMMARY_PATH = RESULTS_DIR / "reconstructed_regime_feature_summary.csv"
PERFORMANCE_PATH = RESULTS_DIR / "asset_performance_by_reconstructed_regime.csv"
ANN_RETURN_PATH = RESULTS_DIR / "asset_annualized_return_by_regime.csv"
SHARPE_PATH = RESULTS_DIR / "asset_sharpe_by_regime.csv"
MAX_DD_PATH = RESULTS_DIR / "asset_max_drawdown_by_regime.csv"
POSITIVE_PATH = RESULTS_DIR / "asset_positive_day_ratio_by_regime.csv"
DRAWDOWN_SUMMARY_PATH = RESULTS_DIR / "asset_drawdown_state_summary.csv"
SPY_DRAWDOWN_FREQ_PATH = RESULTS_DIR / "spy_drawdown_frequency_by_regime.csv"
CORRELATION_PATH = RESULTS_DIR / "asset_correlation_by_reconstructed_regime.csv"
OVERLAP_COUNTS_PATH = RESULTS_DIR / "reconstructed_vs_cluster_overlap_counts.csv"
OVERLAP_PCT_PATH = RESULTS_DIR / "reconstructed_vs_cluster_overlap_percentages.csv"
REPORT_PATH = RESULTS_DIR / "RECONSTRUCTED_REGIME_ASSET_BEHAVIOR.md"

REGIME_ORDER = ["VIX_STRESS", "HIGH_INFLATION", "INVERTED", "FLAT", "STEEP"]
FEATURES = ["VIX_LEVEL", "CREDIT_SPREAD_BAA_AAA", "DGS1", "DGS10", "TERM_SPREAD_10Y_1Y"]
REGIME_COLORS = {
    "VIX_STRESS": "#b2182b",
    "HIGH_INFLATION": "#ef8a62",
    "INVERTED": "#2166ac",
    "FLAT": "#67a9cf",
    "STEEP": "#1b7837",
}
ASSET_COLUMN_CANDIDATES = {
    "SPY": ["SPY"],
    "IEF": ["IEF"],
    "GOLD": ["GLD"],
    "CMDTY_FUT": ["GD=F"],
    "CASH": [],
}


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def read_fred_csv(path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
    value_col = next((c for c in df.columns if c != date_col), df.columns[-1])
    out = df[[date_col, value_col]].copy()
    out.columns = ["date", value_name]
    out["date"] = pd.to_datetime(out["date"])
    out[value_name] = pd.to_numeric(out[value_name].replace(".", np.nan), errors="coerce")
    return out.sort_values("date")


def load_rf_daily() -> pd.DataFrame:
    path = next((p for p in DTB3_CANDIDATES if p.exists()), None)
    if path is None:
        raise FileNotFoundError("DTB3.csv not found in expected paths.")
    rf = read_fred_csv(path, "DTB3")
    rf["DTB3_RATE"] = rf["DTB3"] / 100.0
    rf["RF_DAILY"] = (1.0 + rf["DTB3_RATE"].ffill()) ** (1.0 / 252.0) - 1.0
    return rf[["date", "RF_DAILY"]]


def load_asset_returns() -> tuple[pd.DataFrame, list[str]]:
    daily = pd.read_csv(DAILY_RETURNS_INPUT)
    daily["date"] = pd.to_datetime(daily["date"])
    out = daily[["date"]].copy()
    available_assets: list[str] = []
    for asset, candidates in ASSET_COLUMN_CANDIDATES.items():
        if asset == "CASH":
            continue
        source = next((col for col in candidates if col in daily.columns), None)
        if source is not None:
            out[f"{asset}_RETURN"] = pd.to_numeric(daily[source], errors="coerce")
            available_assets.append(asset)
    return out, available_assets


def load_backtest_panel() -> tuple[pd.DataFrame, list[str]]:
    panel = pd.read_csv(BACKTEST_PANEL_INPUT)
    panel["date"] = pd.to_datetime(panel["date"])
    out = panel[
        [
            "date",
            "RF_DAILY",
            "VIX_LEVEL",
            "DGS1",
            "DGS10",
            "CREDIT_SPREAD_BAA_AAA",
            "TERM_SPREAD_10Y_1Y",
        ]
    ].copy()
    asset_returns = None
    if DAILY_RETURNS_INPUT.exists():
        daily = pd.read_csv(DAILY_RETURNS_INPUT)
        daily["date"] = pd.to_datetime(daily["date"])
        cols = ["date"]
        mapping = {"SPY": "SPY_RETURN", "IEF": "IEF_RETURN", "GLD": "GOLD_RETURN", "GD=F": "CMDTY_FUT_RETURN"}
        for source in mapping:
            if source in daily.columns:
                cols.append(source)
        asset_returns = daily[cols].copy().rename(columns=mapping)
    else:
        asset_returns = panel[["date", "SPY_RET", "IEF_RET", "GOLD_RET"]].copy().rename(
            columns={"SPY_RET": "SPY_RETURN", "IEF_RET": "IEF_RETURN", "GOLD_RET": "GOLD_RETURN"}
        )
    out = out.merge(asset_returns, on="date", how="left")
    out["CASH_RETURN"] = out["RF_DAILY"]
    if DAILY_CLOSE_INPUT.exists():
        close_df = pd.read_csv(DAILY_CLOSE_INPUT, usecols=["date", "SPY"])
        close_df["date"] = pd.to_datetime(close_df["date"])
        close_df["SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"] = close_df["SPY"] / close_df["SPY"].cummax() - 1.0
        out = out.merge(close_df[["date", "SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"]], on="date", how="left")
    else:
        wealth = (1.0 + out["SPY_RETURN"].fillna(0.0)).cumprod()
        out["SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"] = wealth / wealth.cummax() - 1.0

    assets = ["SPY", "IEF", "GOLD", "CASH"]
    if "CMDTY_FUT_RETURN" in out.columns and out["CMDTY_FUT_RETURN"].notna().any():
        assets.insert(3, "CMDTY_FUT")
    return out, assets


def load_macro_features() -> pd.DataFrame:
    vix = read_fred_csv(VIX_PATH, "VIX_LEVEL")
    dgs1 = read_fred_csv(DGS1_PATH, "DGS1")
    dgs10 = read_fred_csv(DGS10_PATH, "DGS10")
    waaa = read_fred_csv(WAAA_PATH, "WAAA")
    wbaa = read_fred_csv(WBAA_PATH, "WBAA")
    credit = waaa.merge(wbaa, on="date", how="outer").sort_values("date")
    credit[["WAAA", "WBAA"]] = credit[["WAAA", "WBAA"]].ffill()
    credit["CREDIT_SPREAD_BAA_AAA"] = credit["WBAA"] - credit["WAAA"]

    panel = vix.merge(dgs1, on="date", how="inner").merge(dgs10, on="date", how="inner").merge(
        credit[["date", "CREDIT_SPREAD_BAA_AAA"]], on="date", how="inner"
    )
    panel["TERM_SPREAD_10Y_1Y"] = panel["DGS10"] - panel["DGS1"]
    return panel.sort_values("date")


def assign_reconstructed_regimes(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    conditions = [
        out["VIX_LEVEL"] >= 25,
        (out["CREDIT_SPREAD_BAA_AAA"] > 1.5) & (out["DGS1"] > 5),
        out["TERM_SPREAD_10Y_1Y"] < 0,
        out["TERM_SPREAD_10Y_1Y"] < 1,
    ]
    choices = ["VIX_STRESS", "HIGH_INFLATION", "INVERTED", "FLAT"]
    out["reconstructed_regime"] = np.select(conditions, choices, default="STEEP")
    out["reconstructed_regime"] = pd.Categorical(out["reconstructed_regime"], categories=REGIME_ORDER, ordered=True)
    return out


def load_cluster_labels_monthly() -> pd.DataFrame:
    if not RULE_STATE_INPUT.exists():
        return pd.DataFrame(columns=["date", "regime_name"])
    df = pd.read_csv(RULE_STATE_INPUT, usecols=["date", "regime_name"])
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp("M")
    return df.drop_duplicates("date", keep="last")


def build_output_panel() -> tuple[pd.DataFrame, list[str]]:
    if BACKTEST_PANEL_INPUT.exists():
        panel, available_assets = load_backtest_panel()
        panel = assign_reconstructed_regimes(panel)
    else:
        macro = assign_reconstructed_regimes(load_macro_features())
        asset_returns, available_assets = load_asset_returns()
        rf = load_rf_daily()
        panel = macro.merge(asset_returns, on="date", how="inner").merge(rf, on="date", how="inner")
        panel["CASH_RETURN"] = panel["RF_DAILY"]

        close_df = pd.read_csv(DAILY_CLOSE_INPUT, usecols=["date", "SPY"])
        close_df["date"] = pd.to_datetime(close_df["date"])
        close_df["SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"] = close_df["SPY"] / close_df["SPY"].cummax() - 1.0
        panel = panel.merge(close_df[["date", "SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"]], on="date", how="left")

    panel["date_month"] = panel["date"].dt.to_period("M").dt.to_timestamp("M")
    panel = panel.merge(load_cluster_labels_monthly(), left_on="date_month", right_on="date", how="left", suffixes=("", "_cluster"))
    panel = panel.drop(columns=["date_cluster"], errors="ignore")

    required = FEATURES + ["RF_DAILY", "CASH_RETURN"]
    required += [f"{asset}_RETURN" for asset in available_assets]
    panel = panel.dropna(subset=required).sort_values("date").reset_index(drop=True)
    panel = panel.drop(columns=["date_month"])
    assets = list(dict.fromkeys(available_assets + ["CASH"]))
    return panel, assets


def annualized_return_from_daily(s: pd.Series) -> float:
    s = s.dropna()
    if s.empty:
        return np.nan
    total = float((1.0 + s).prod())
    return total ** (252.0 / len(s)) - 1.0


def max_drawdown_from_returns(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    wealth = (1.0 + s).cumprod()
    dd = wealth / wealth.cummax() - 1.0
    return float(dd.min())


def performance_stats(series: pd.Series, rf_daily: pd.Series, asset_name: str) -> dict[str, float]:
    s = series.dropna()
    if s.empty:
        return {
            "n_obs": 0,
            "average_daily_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "Sharpe": np.nan,
            "max_drawdown": np.nan,
            "worst_day": np.nan,
            "best_day": np.nan,
            "positive_day_ratio": np.nan,
            "cumulative_return_within_regime": np.nan,
        }
    rf = rf_daily.loc[s.index]
    excess = s - rf
    ann_ret = annualized_return_from_daily(s)
    ann_vol = float(s.std(ddof=1) * np.sqrt(252)) if len(s) > 1 else np.nan
    if asset_name == "CASH":
        sharpe = 0.0
    else:
        excess_std = excess.std(ddof=1)
        sharpe = float(excess.mean() / excess_std * np.sqrt(252)) if pd.notna(excess_std) and excess_std != 0 else np.nan
    return {
        "n_obs": int(len(s)),
        "average_daily_return": float(s.mean()),
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "Sharpe": sharpe,
        "max_drawdown": max_drawdown_from_returns(s),
        "worst_day": float(s.min()),
        "best_day": float(s.max()),
        "positive_day_ratio": float((s > 0).mean()),
        "cumulative_return_within_regime": float((1.0 + s).prod() - 1.0),
    }


def feature_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime in REGIME_ORDER:
        grp = panel.loc[panel["reconstructed_regime"] == regime]
        for feature in FEATURES:
            s = grp[feature].dropna()
            rows.append(
                {
                    "reconstructed_regime": regime,
                    "feature": feature,
                    "n_obs": int(len(s)),
                    "mean": float(s.mean()) if not s.empty else np.nan,
                    "median": float(s.median()) if not s.empty else np.nan,
                    "std": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
                    "min": float(s.min()) if not s.empty else np.nan,
                    "p10": float(s.quantile(0.10)) if not s.empty else np.nan,
                    "p25": float(s.quantile(0.25)) if not s.empty else np.nan,
                    "p50": float(s.quantile(0.50)) if not s.empty else np.nan,
                    "p75": float(s.quantile(0.75)) if not s.empty else np.nan,
                    "p90": float(s.quantile(0.90)) if not s.empty else np.nan,
                    "max": float(s.max()) if not s.empty else np.nan,
                }
            )
    return pd.DataFrame(rows)


def asset_performance(panel: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    rows = []
    for regime in REGIME_ORDER:
        grp = panel.loc[panel["reconstructed_regime"] == regime].copy()
        grp = grp.set_index("date")
        for asset in assets:
            row = {"reconstructed_regime": regime, "asset": asset}
            row.update(performance_stats(grp[f"{asset}_RETURN"], grp["RF_DAILY"], asset))
            rows.append(row)
    return pd.DataFrame(rows)


def pivot_metric(perf: pd.DataFrame, metric: str, assets: list[str]) -> pd.DataFrame:
    return (
        perf.pivot_table(index="asset", columns="reconstructed_regime", values=metric, aggfunc="first")
        .reindex(index=assets, columns=REGIME_ORDER)
    )


def plot_heatmap(pivot: pd.DataFrame, title: str, path: Path, fmt: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap="RdYlBu_r", center=0 if "Sharpe" in title or "Return" in title else None, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_timeline(panel: pd.DataFrame) -> None:
    plot_df = panel[["date", "reconstructed_regime"]].dropna().copy()
    plot_df["regime_code"] = plot_df["reconstructed_regime"].cat.codes
    fig, ax = plt.subplots(figsize=(12, 4))
    for regime in REGIME_ORDER:
        subset = plot_df.loc[plot_df["reconstructed_regime"] == regime]
        ax.scatter(subset["date"], subset["regime_code"], s=6, label=regime, color=REGIME_COLORS[regime])
    ax.set_yticks(range(len(REGIME_ORDER)))
    ax.set_yticklabels(REGIME_ORDER)
    ax.set_title("Daily Reconstructed Rule-Based Regime Timeline")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "reconstructed_regime_timeline.png", dpi=180)
    plt.close(fig)


def plot_feature_heatmap(summary: pd.DataFrame) -> None:
    pivot = summary.pivot(index="reconstructed_regime", columns="feature", values="mean").reindex(REGIME_ORDER)
    normalized = pivot.copy()
    for col in normalized.columns:
        std = normalized[col].std(ddof=1)
        normalized[col] = (normalized[col] - normalized[col].mean()) / std if pd.notna(std) and std != 0 else np.nan
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.heatmap(normalized, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0, ax=ax)
    ax.set_title("Daily Reconstructed Regime Feature Heatmap")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "reconstructed_regime_feature_heatmap.png", dpi=180)
    plt.close(fig)


def plot_return_profiles(ann_returns: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    axes = axes.flatten()
    for ax, regime in zip(axes, REGIME_ORDER):
        values = ann_returns[regime]
        colors = ["#2b8cbe" if v >= 0 else "#d7301f" for v in values.fillna(0)]
        ax.bar(values.index, values.values, color=colors, alpha=0.85)
        ax.set_title(regime)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.2)
    axes[-1].axis("off")
    fig.suptitle("Daily Asset Annualized Return Profiles by Reconstructed Regime")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES_DIR / "asset_return_profiles_by_regime.png", dpi=180)
    plt.close(fig)


def drawdown_state_summary(panel: pd.DataFrame, assets: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for regime in REGIME_ORDER:
        grp = panel.loc[panel["reconstructed_regime"] == regime].copy().set_index("date")
        for asset in assets:
            s = grp[f"{asset}_RETURN"].dropna()
            if asset == "CASH":
                dd = pd.Series(0.0, index=s.index)
            else:
                wealth = (1.0 + s).cumprod()
                dd = wealth / wealth.cummax() - 1.0
            rows.append(
                {
                    "reconstructed_regime": regime,
                    "asset": asset,
                    "average_drawdown": float(dd.mean()) if not dd.empty else np.nan,
                    "median_drawdown": float(dd.median()) if not dd.empty else np.nan,
                    "worst_drawdown_observed": float(dd.min()) if not dd.empty else np.nan,
                    "p10_drawdown": float(dd.quantile(0.10)) if not dd.empty else np.nan,
                    "p05_drawdown": float(dd.quantile(0.05)) if not dd.empty else np.nan,
                }
            )
    freq_rows = []
    for regime in REGIME_ORDER:
        grp = panel.loc[panel["reconstructed_regime"] == regime]
        n = len(grp)
        dd = grp["SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"]
        freq_rows.append(
            {
                "reconstructed_regime": regime,
                "n_days": int(n),
                "spy_drawdown_lt_10_count": int((dd < -0.10).sum()),
                "spy_drawdown_lt_20_count": int((dd < -0.20).sum()),
                "spy_drawdown_lt_10_pct": float((dd < -0.10).mean()) if n else np.nan,
                "spy_drawdown_lt_20_pct": float((dd < -0.20).mean()) if n else np.nan,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(freq_rows)


def plot_spy_drawdown_frequency(freq: pd.DataFrame) -> None:
    plot_df = freq.melt(
        id_vars="reconstructed_regime",
        value_vars=["spy_drawdown_lt_10_pct", "spy_drawdown_lt_20_pct"],
        var_name="threshold",
        value_name="frequency",
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_df, x="reconstructed_regime", y="frequency", hue="threshold", ax=ax)
    ax.set_title("SPY Large Drawdown Frequency by Reconstructed Regime")
    ax.tick_params(axis="x", rotation=25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "spy_drawdown_frequency_by_regime.png", dpi=180)
    plt.close(fig)


def correlations_by_regime(panel: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    rows = []
    for regime in REGIME_ORDER:
        cols = [f"{asset}_RETURN" for asset in assets]
        grp = panel.loc[panel["reconstructed_regime"] == regime, cols].dropna(how="all")
        corr = grp.rename(columns={f"{asset}_RETURN": asset for asset in assets}).corr()
        for a in assets:
            for b in assets:
                rows.append(
                    {
                        "reconstructed_regime": regime,
                        "asset_1": a,
                        "asset_2": b,
                        "correlation": corr.loc[a, b] if a in corr.index and b in corr.columns else np.nan,
                    }
                )
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr.reindex(index=assets, columns=assets), annot=True, fmt=".2f", cmap="RdYlBu_r", vmin=-1, vmax=1, ax=ax)
        ax.set_title(f"Correlation: {regime}")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"correlation_heatmap_{regime.lower()}.png", dpi=180)
        plt.close(fig)
    return pd.DataFrame(rows)


def cluster_overlap(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "regime_name" not in panel.columns or panel["regime_name"].dropna().empty:
        empty = pd.DataFrame()
        return empty, empty
    counts = pd.crosstab(panel["reconstructed_regime"], panel["regime_name"]).reindex(index=REGIME_ORDER)
    pct = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pct, annot=True, fmt=".1%", cmap="Blues", ax=ax)
    ax.set_title("Reconstructed Regime vs Cluster Regime Overlap")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "reconstructed_vs_cluster_overlap_heatmap.png", dpi=180)
    plt.close(fig)
    return counts, pct


def write_report(panel: pd.DataFrame, perf: pd.DataFrame, assets: list[str]) -> None:
    lines = [
        "# Reconstructed Regime Asset Behavior",
        "",
        "## Why reconstructed regimes?",
        "",
        "This report studies asset behavior under observable daily rule-based regimes rather than using full-sample cluster labels as trading signals.",
        "",
        "## Regime definitions",
        "",
        "- `VIX_STRESS`: VIX >= 25",
        "- `HIGH_INFLATION`: credit spread > 1.5 and DGS1 > 5, after VIX stress is carved out",
        "- `INVERTED`: term spread < 0",
        "- `FLAT`: 0 <= term spread < 1",
        "- `STEEP`: term spread >= 1",
        "- There is no `NEUTRAL` bucket in this daily research panel because every retained daily observation maps to one of the rules above.",
        "",
        "## Measurement",
        "",
        "- Regime performance is computed on daily returns.",
        "- Annualized return uses compounded regime-specific daily returns.",
        "- Sharpe uses daily excess return relative to daily risk-free return from DTB3.",
        "- Cash uses the compounded daily DTB3 risk-free series and therefore has Sharpe set to 0 by construction.",
        "",
    ]
    for regime in REGIME_ORDER:
        regime_perf = perf.loc[perf["reconstructed_regime"] == regime]
        if regime_perf.empty:
            continue
        best_ret = regime_perf.sort_values("annualized_return", ascending=False).iloc[0]
        best_sharpe = regime_perf.sort_values("Sharpe", ascending=False).iloc[0]
        lowest_dd = regime_perf.sort_values("max_drawdown", ascending=False).iloc[0]
        lines.extend(
            [
                f"### {regime}",
                f"- Highest annualized return: `{best_ret['asset']}` ({best_ret['annualized_return']:.2%}, n={int(best_ret['n_obs'])}).",
                f"- Best Sharpe: `{best_sharpe['asset']}` ({best_sharpe['Sharpe']:.2f}).",
                f"- Lowest max drawdown: `{lowest_dd['asset']}` ({lowest_dd['max_drawdown']:.2%}).",
                "",
            ]
        )
    lines.append(f"Assets included: {', '.join(assets)}.")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    out_panel, assets = build_output_panel()
    out_panel.to_csv(REGIME_PANEL_PATH, index=False)

    counts = out_panel["reconstructed_regime"].value_counts().reindex(REGIME_ORDER, fill_value=0).reset_index()
    counts.columns = ["reconstructed_regime", "n_days"]
    counts["sample_share"] = counts["n_days"] / counts["n_days"].sum()
    counts.to_csv(REGIME_COUNTS_PATH, index=False)

    feat_summary = feature_summary(out_panel)
    feat_summary.to_csv(FEATURE_SUMMARY_PATH, index=False)
    plot_timeline(out_panel)
    plot_feature_heatmap(feat_summary)

    perf = asset_performance(out_panel, assets)
    perf.to_csv(PERFORMANCE_PATH, index=False)
    ann = pivot_metric(perf, "annualized_return", assets)
    sharpe = pivot_metric(perf, "Sharpe", assets)
    max_dd = pivot_metric(perf, "max_drawdown", assets)
    positive = pivot_metric(perf, "positive_day_ratio", assets)
    ann.to_csv(ANN_RETURN_PATH)
    sharpe.to_csv(SHARPE_PATH)
    max_dd.to_csv(MAX_DD_PATH)
    positive.to_csv(POSITIVE_PATH)
    plot_heatmap(ann, "Daily Asset Annualized Return by Reconstructed Regime", FIGURES_DIR / "asset_annualized_return_heatmap.png", ".2%")
    plot_heatmap(sharpe, "Daily Asset Sharpe by Reconstructed Regime", FIGURES_DIR / "asset_sharpe_heatmap.png", ".2f")
    plot_heatmap(max_dd, "Daily Asset Max Drawdown by Reconstructed Regime", FIGURES_DIR / "asset_max_drawdown_heatmap.png", ".2%")
    plot_heatmap(positive, "Daily Asset Positive Day Ratio by Reconstructed Regime", FIGURES_DIR / "asset_positive_day_ratio_heatmap.png", ".1%")
    plot_return_profiles(ann)

    dd_summary, dd_freq = drawdown_state_summary(out_panel, assets)
    dd_summary.to_csv(DRAWDOWN_SUMMARY_PATH, index=False)
    dd_freq.to_csv(SPY_DRAWDOWN_FREQ_PATH, index=False)
    plot_spy_drawdown_frequency(dd_freq)

    corr = correlations_by_regime(out_panel, assets)
    corr.to_csv(CORRELATION_PATH, index=False)

    overlap_counts, overlap_pct = cluster_overlap(out_panel)
    if not overlap_counts.empty:
        overlap_counts.to_csv(OVERLAP_COUNTS_PATH)
        overlap_pct.to_csv(OVERLAP_PCT_PATH)

    write_report(out_panel, perf, assets)

    best_return = perf.sort_values("annualized_return", ascending=False).groupby("reconstructed_regime").head(1)
    best_sharpe = perf.sort_values("Sharpe", ascending=False).groupby("reconstructed_regime").head(1)
    lowest_dd = perf.sort_values("max_drawdown", ascending=False).groupby("reconstructed_regime").head(1)
    print("Reconstructed regime counts:")
    print(counts.to_string(index=False))
    print("Assets included:")
    print(", ".join(assets))
    print("Asset with highest annualized return in each regime:")
    print(best_return[["reconstructed_regime", "asset", "annualized_return"]].to_string(index=False))
    print("Asset with best Sharpe in each regime:")
    print(best_sharpe[["reconstructed_regime", "asset", "Sharpe"]].to_string(index=False))
    print("Asset with lowest max drawdown in each regime:")
    print(lowest_dd[["reconstructed_regime", "asset", "max_drawdown"]].to_string(index=False))
    for path in [
        REGIME_PANEL_PATH,
        REGIME_COUNTS_PATH,
        FEATURE_SUMMARY_PATH,
        PERFORMANCE_PATH,
        ANN_RETURN_PATH,
        SHARPE_PATH,
        MAX_DD_PATH,
        POSITIVE_PATH,
        DRAWDOWN_SUMMARY_PATH,
        SPY_DRAWDOWN_FREQ_PATH,
        CORRELATION_PATH,
        REPORT_PATH,
        FIGURES_DIR / "reconstructed_regime_timeline.png",
        FIGURES_DIR / "reconstructed_regime_feature_heatmap.png",
        FIGURES_DIR / "asset_annualized_return_heatmap.png",
        FIGURES_DIR / "asset_sharpe_heatmap.png",
        FIGURES_DIR / "asset_max_drawdown_heatmap.png",
        FIGURES_DIR / "asset_positive_day_ratio_heatmap.png",
        FIGURES_DIR / "asset_return_profiles_by_regime.png",
        FIGURES_DIR / "spy_drawdown_frequency_by_regime.png",
    ]:
        if path.exists():
            print(f"Saved: {path}")


if __name__ == "__main__":
    main()
