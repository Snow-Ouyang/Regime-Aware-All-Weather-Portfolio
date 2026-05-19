from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from regime.utils import REGIME_COLORS, REGIME_ORDER


CORE_PANEL_PATH = ROOT / "data" / "processed" / "risk_factors" / "core_risk_factor_panel.csv"
RESULTS_DIR = ROOT / "results" / "risk_factor_regime_analysis"
FIGURES_DIR = ROOT / "figures" / "risk_factor_regime_analysis"

PERFORMANCE_PATH = RESULTS_DIR / "factor_performance_by_regime.csv"
ANNUAL_RETURN_PATH = RESULTS_DIR / "factor_annualized_return_by_regime.csv"
ANNUAL_VOL_PATH = RESULTS_DIR / "factor_annualized_vol_by_regime.csv"
SHARPE_PATH = RESULTS_DIR / "factor_sharpe_by_regime.csv"
MAX_DRAWDOWN_PATH = RESULTS_DIR / "factor_max_drawdown_by_regime.csv"
POSITIVE_RATIO_PATH = RESULTS_DIR / "factor_positive_month_ratio_by_regime.csv"
SHOCK_MEAN_PATH = RESULTS_DIR / "shock_factor_mean_by_regime.csv"
SHOCK_STD_PATH = RESULTS_DIR / "shock_factor_std_by_regime.csv"
KEY_PAIRWISE_PATH = RESULTS_DIR / "key_pairwise_correlations_by_regime.csv"
ROLE_SUMMARY_PATH = RESULTS_DIR / "regime_factor_role_summary.csv"
MARKDOWN_PATH = RESULTS_DIR / "FACTOR_BEHAVIOR_BY_REGIME.md"

ANNUAL_RETURN_HEATMAP_PATH = FIGURES_DIR / "factor_annualized_return_heatmap.png"
SHARPE_HEATMAP_PATH = FIGURES_DIR / "factor_sharpe_heatmap.png"
MAX_DRAWDOWN_HEATMAP_PATH = FIGURES_DIR / "factor_max_drawdown_heatmap.png"
POSITIVE_RATIO_HEATMAP_PATH = FIGURES_DIR / "factor_positive_month_ratio_heatmap.png"
RETURN_BOXPLOT_PATH = FIGURES_DIR / "return_factor_boxplots_by_regime.png"
AQR_BOXPLOT_PATH = FIGURES_DIR / "aqr_factor_boxplots_by_regime.png"
SHOCK_BOXPLOT_PATH = FIGURES_DIR / "shock_factor_boxplots_by_regime.png"
KEY_PAIRWISE_HEATMAP_PATH = FIGURES_DIR / "key_pairwise_correlations_heatmap.png"
REGIME_PROFILE_PATH = FIGURES_DIR / "regime_factor_return_profiles.png"

RETURN_LIKE_CANDIDATES = [
    "MKT_EXCESS",
    "AQR_FI_MARKET_EXCESS",
    "AQR_CMDTY_EW_EXCESS",
    "GOLD_EXCESS",
    "AQR_CMDTY_CARRY",
    "BAB",
    "AQR_TSMOM",
    "AQR_QMJ",
    "FF_SMB",
    "FF_HML",
    "FF_RMW",
    "FF_CMA",
    "FF_MOM",
]
SHOCK_CANDIDATES = [
    "D_GS10",
    "D_CREDIT_SPREAD",
    "D_VIX",
    "D_DOLLAR",
    "D_GROWTH_PC1",
    "D_INFLATION_PC1",
    "D_TERM_SPREAD_10Y_1Y",
    "D_TED_SPREAD",
]
CORRELATION_FACTOR_CANDIDATES = [
    "MKT_EXCESS",
    "AQR_FI_MARKET_EXCESS",
    "AQR_CMDTY_EW_EXCESS",
    "GOLD_EXCESS",
    "BAB",
    "AQR_TSMOM",
    "AQR_QMJ",
    "FF_SMB",
    "FF_HML",
    "FF_RMW",
    "FF_CMA",
    "FF_MOM",
]
PAIR_CANDIDATES = [
    ("MKT_EXCESS", "AQR_FI_MARKET_EXCESS"),
    ("MKT_EXCESS", "AQR_CMDTY_EW_EXCESS"),
    ("MKT_EXCESS", "GOLD_EXCESS"),
    ("MKT_EXCESS", "BAB"),
    ("MKT_EXCESS", "AQR_TSMOM"),
    ("MKT_EXCESS", "AQR_QMJ"),
    ("MKT_EXCESS", "FF_MOM"),
    ("AQR_FI_MARKET_EXCESS", "AQR_CMDTY_EW_EXCESS"),
    ("AQR_FI_MARKET_EXCESS", "GOLD_EXCESS"),
    ("AQR_CMDTY_EW_EXCESS", "GOLD_EXCESS"),
]
SMALL_SAMPLE_THRESHOLD = 24


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def load_core_panel() -> pd.DataFrame:
    panel = pd.read_csv(CORE_PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date", "regime_name"]).sort_values("date").reset_index(drop=True)
    return panel


def select_available_factors(panel: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    return_like = [col for col in RETURN_LIKE_CANDIDATES if col in panel.columns]
    shock = [col for col in SHOCK_CANDIDATES if col in panel.columns]
    missing = [col for col in RETURN_LIKE_CANDIDATES + SHOCK_CANDIDATES if col not in panel.columns]
    return return_like, shock, missing


def classify_factor_types(return_like: list[str], shock: list[str]) -> dict[str, str]:
    mapping = {col: "return_like" for col in return_like}
    mapping.update({col: "shock" for col in shock})
    return mapping


def compute_max_drawdown(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    cumulative = (1.0 + clean).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    return float(drawdown.min())


def compute_factor_performance_by_regime(
    panel: pd.DataFrame,
    return_like: list[str],
    shock: list[str],
) -> pd.DataFrame:
    factor_types = classify_factor_types(return_like, shock)
    rows: list[dict[str, object]] = []

    for regime_name in REGIME_ORDER:
        regime_panel = panel.loc[panel["regime_name"] == regime_name].copy()
        for factor in return_like + shock:
            series = pd.to_numeric(regime_panel[factor], errors="coerce").dropna()
            if series.empty:
                row = {
                    "regime_name": regime_name,
                    "factor_name": factor,
                    "factor_type": factor_types[factor],
                    "n_obs": 0,
                    "mean_monthly": np.nan,
                    "median_monthly": np.nan,
                    "std_monthly": np.nan,
                    "annualized_return": np.nan,
                    "annualized_volatility": np.nan,
                    "sharpe_ratio": np.nan,
                    "min_monthly": np.nan,
                    "max_monthly": np.nan,
                    "worst_month": "",
                    "best_month": "",
                    "positive_month_ratio": np.nan,
                    "max_drawdown": np.nan,
                }
                rows.append(row)
                continue

            mean_monthly = float(series.mean())
            median_monthly = float(series.median())
            std_monthly = float(series.std())
            min_idx = series.idxmin()
            max_idx = series.idxmax()
            positive_ratio = float((series > 0).mean())
            factor_type = factor_types[factor]
            annualized_vol = float(std_monthly * np.sqrt(12.0)) if not np.isnan(std_monthly) else np.nan
            if factor_type == "return_like":
                annualized_return = float((1.0 + mean_monthly) ** 12 - 1.0)
                sharpe = np.nan if annualized_vol in (0.0, np.nan) or pd.isna(annualized_vol) else annualized_return / annualized_vol
                max_drawdown = compute_max_drawdown(series)
            else:
                annualized_return = np.nan
                sharpe = np.nan
                max_drawdown = np.nan

            rows.append(
                {
                    "regime_name": regime_name,
                    "factor_name": factor,
                    "factor_type": factor_type,
                    "n_obs": int(series.count()),
                    "mean_monthly": mean_monthly,
                    "median_monthly": median_monthly,
                    "std_monthly": std_monthly,
                    "annualized_return": annualized_return,
                    "annualized_volatility": annualized_vol,
                    "sharpe_ratio": sharpe,
                    "min_monthly": float(series.min()),
                    "max_monthly": float(series.max()),
                    "worst_month": panel.loc[min_idx, "date"].strftime("%Y-%m-%d"),
                    "best_month": panel.loc[max_idx, "date"].strftime("%Y-%m-%d"),
                    "positive_month_ratio": positive_ratio,
                    "max_drawdown": max_drawdown,
                }
            )

    return pd.DataFrame(rows)


def create_metric_pivot(performance: pd.DataFrame, factors: list[str], metric: str) -> pd.DataFrame:
    subset = performance.loc[performance["factor_name"].isin(factors), ["factor_name", "regime_name", metric]].copy()
    pivot = subset.pivot(index="factor_name", columns="regime_name", values=metric)
    pivot = pivot.reindex(index=factors, columns=REGIME_ORDER)
    return pivot


def plot_heatmap(
    df: pd.DataFrame,
    title: str,
    path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
    fmt: str = "{:.2f}",
    cmap: str = "RdBu_r",
) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(1.8 + 1.6 * len(df.columns), 1.5 + 0.6 * len(df.index)))
    mat = df.to_numpy(dtype=float)
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            value = mat[i, j]
            text = "" if np.isnan(value) else fmt.format(value)
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="#111111")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Value")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_factor_boxplots(panel: pd.DataFrame, factors: list[str], title: str, path: Path) -> None:
    use = [factor for factor in factors if factor in panel.columns]
    if not use:
        return
    ncols = 3
    nrows = math.ceil(len(use) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, factor in zip(axes, use):
        groups = []
        labels = []
        colors = []
        for regime_name in REGIME_ORDER:
            vals = pd.to_numeric(panel.loc[panel["regime_name"] == regime_name, factor], errors="coerce").dropna()
            groups.append(vals.to_numpy() if not vals.empty else np.array([np.nan]))
            labels.append(regime_name)
            colors.append(REGIME_COLORS[regime_name])
        bp = ax.boxplot(groups, patch_artist=True, tick_labels=labels, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_title(factor)
        ax.grid(axis="y", alpha=0.18)
        ax.tick_params(axis="x", rotation=25)
    for ax in axes[len(use):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=16, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def compute_regime_correlation_matrices(panel: pd.DataFrame, factors: list[str]) -> dict[str, pd.DataFrame]:
    use = [factor for factor in factors if factor in panel.columns]
    matrices: dict[str, pd.DataFrame] = {}
    for regime_name in REGIME_ORDER:
        regime_panel = panel.loc[panel["regime_name"] == regime_name, use]
        matrices[regime_name] = regime_panel.corr().reindex(index=use, columns=use)
    return matrices


def plot_regime_correlation_heatmaps(matrices: dict[str, pd.DataFrame], panel: pd.DataFrame) -> list[Path]:
    saved_paths: list[Path] = []
    for regime_name in REGIME_ORDER:
        corr = matrices.get(regime_name)
        if corr is None or corr.empty:
            continue
        n_obs = int(panel.loc[panel["regime_name"] == regime_name, "date"].count())
        path = FIGURES_DIR / f"correlation_heatmap_{slugify(regime_name)}.png"
        plot_heatmap(
            corr,
            f"{regime_name} Correlation Heatmap (n={n_obs})",
            path,
            vmin=-1,
            vmax=1,
            fmt="{:.2f}",
        )
        saved_paths.append(path)
    return saved_paths


def compute_key_pairwise_correlations(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for regime_name in REGIME_ORDER:
        regime_panel = panel.loc[panel["regime_name"] == regime_name]
        for left, right in PAIR_CANDIDATES:
            if left not in regime_panel.columns or right not in regime_panel.columns:
                continue
            sample = regime_panel[[left, right]].dropna()
            corr = sample[left].corr(sample[right]) if len(sample) >= 2 else np.nan
            rows.append(
                {
                    "regime_name": regime_name,
                    "factor_pair": f"{left} vs {right}",
                    "correlation": corr,
                    "n_obs": int(len(sample)),
                }
            )
    return pd.DataFrame(rows)


def top_factors(metric_pivot: pd.DataFrame, regime_name: str, n: int = 3, ascending: bool = False) -> str:
    if regime_name not in metric_pivot.columns:
        return "None"
    series = metric_pivot[regime_name].dropna().sort_values(ascending=ascending)
    if series.empty:
        return "None"
    return ", ".join(series.head(n).index.tolist())


def compute_equity_hedge_candidates(pairwise: pd.DataFrame, regime_name: str) -> str:
    subset = pairwise.loc[pairwise["regime_name"] == regime_name].copy()
    candidates: list[str] = []
    for pair_label, candidate_name in [
        ("MKT_EXCESS vs AQR_FI_MARKET_EXCESS", "AQR_FI_MARKET_EXCESS"),
        ("MKT_EXCESS vs GOLD_EXCESS", "GOLD_EXCESS"),
        ("MKT_EXCESS vs BAB", "BAB"),
        ("MKT_EXCESS vs AQR_TSMOM", "AQR_TSMOM"),
    ]:
        row = subset.loc[subset["factor_pair"] == pair_label]
        if not row.empty and pd.notna(row["correlation"].iloc[0]) and row["correlation"].iloc[0] < 0:
            candidates.append(candidate_name)
    return ", ".join(candidates) if candidates else "None"


def create_regime_role_summary(
    performance: pd.DataFrame,
    pairwise: pd.DataFrame,
    return_like: list[str],
    shock: list[str],
) -> pd.DataFrame:
    ann_return = create_metric_pivot(performance, return_like, "annualized_return")
    sharpe = create_metric_pivot(performance, return_like, "sharpe_ratio")
    drawdown = create_metric_pivot(performance, return_like, "max_drawdown")
    shock_mean = create_metric_pivot(performance, shock, "mean_monthly")
    rows: list[dict[str, object]] = []

    for regime_name in REGIME_ORDER:
        regime_perf = performance.loc[(performance["regime_name"] == regime_name) & (performance["factor_type"] == "return_like")]
        n_obs = int(regime_perf["n_obs"].max()) if not regime_perf.empty else 0
        best_return = top_factors(ann_return, regime_name, ascending=False)
        worst_return = top_factors(ann_return, regime_name, ascending=True)
        best_sharpe = top_factors(sharpe, regime_name, ascending=False)
        largest_drawdown = top_factors(drawdown, regime_name, ascending=True)
        equity_hedges = compute_equity_hedge_candidates(pairwise, regime_name)

        inflation_candidates = []
        if regime_name in ann_return.columns:
            for candidate in ["AQR_CMDTY_EW_EXCESS", "GOLD_EXCESS", "AQR_CMDTY_CARRY"]:
                if candidate in ann_return.index and pd.notna(ann_return.loc[candidate, regime_name]) and ann_return.loc[candidate, regime_name] > 0:
                    inflation_candidates.append(candidate)
        defensive_candidates = []
        if regime_name in ann_return.columns:
            for candidate in ["AQR_FI_MARKET_EXCESS", "GOLD_EXCESS", "BAB", "AQR_TSMOM", "AQR_QMJ", "FF_RMW"]:
                if candidate in ann_return.index and pd.notna(ann_return.loc[candidate, regime_name]) and ann_return.loc[candidate, regime_name] > 0:
                    defensive_candidates.append(candidate)

        shock_desc = []
        if regime_name in shock_mean.columns:
            ranked = shock_mean[regime_name].dropna()
            if not ranked.empty:
                for factor_name, value in ranked.abs().sort_values(ascending=False).head(3).items():
                    signed_value = shock_mean.loc[factor_name, regime_name]
                    shock_desc.append(f"{factor_name} ({signed_value:+.3f})")

        short_interpretation = (
            f"{regime_name}: strongest return leaders [{best_return}], weak links [{worst_return}], "
            f"equity hedge candidates [{equity_hedges}], dominant shocks [{'; '.join(shock_desc) if shock_desc else 'None'}]."
        )
        rows.append(
            {
                "regime_name": regime_name,
                "n_obs": n_obs,
                "best_return_factors": best_return,
                "worst_return_factors": worst_return,
                "best_sharpe_factors": best_sharpe,
                "largest_drawdown_factors": largest_drawdown,
                "equity_hedge_candidates": equity_hedges,
                "inflation_hedge_candidates": ", ".join(inflation_candidates) if inflation_candidates else "None",
                "defensive_candidates": ", ".join(defensive_candidates) if defensive_candidates else "None",
                "key_macro_shocks": "; ".join(shock_desc) if shock_desc else "None",
                "short_interpretation": short_interpretation,
            }
        )
    return pd.DataFrame(rows)


def write_markdown_report(
    role_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    performance: pd.DataFrame,
    return_like: list[str],
    shock: list[str],
    small_sample_warnings: list[str],
) -> None:
    ann_return = create_metric_pivot(performance, return_like, "annualized_return")
    sharpe = create_metric_pivot(performance, return_like, "sharpe_ratio")
    drawdown = create_metric_pivot(performance, return_like, "max_drawdown")
    shock_mean = create_metric_pivot(performance, shock, "mean_monthly")
    lines = [
        "# Factor Behavior By Regime",
        "",
        "This report is descriptive. It summarizes how the constructed factors behaved inside the fixed simplified macro regimes. It does not make causal claims and it is not a prediction model.",
        "",
        "The current panel removes `AQR_value_all`, `AQR_momentum_all`, `AQR_carry_all`, and `AQR_defensive_all` because those cross-asset long-short research portfolios are harder to map directly into ETF implementation sleeves.",
        "The updated panel adds `AQR_TSMOM`, `AQR_QMJ`, and Fama-French factors so the analysis can focus on more interpretable academic factor roles before later ETF exposure mapping.",
        "`FF_MKT_RF` was checked as a validation series but removed because it was redundant with `MKT_EXCESS`.",
        "",
    ]
    if small_sample_warnings:
        lines.extend(["## Sample warnings", ""])
        for warning in small_sample_warnings:
            lines.append(f"- {warning}")
        lines.append("")

    for regime_name in REGIME_ORDER:
        lines.extend([f"## {regime_name}", ""])
        role_row = role_summary.loc[role_summary["regime_name"] == regime_name].iloc[0]
        eq_dur_corr = np.nan
        pair_row = pairwise.loc[
            (pairwise["regime_name"] == regime_name) & (pairwise["factor_pair"] == "MKT_EXCESS vs AQR_FI_MARKET_EXCESS")
        ]
        if not pair_row.empty:
            eq_dur_corr = pair_row["correlation"].iloc[0]

        macro_shocks = shock_mean[regime_name].dropna().abs().sort_values(ascending=False).head(3).index.tolist() if regime_name in shock_mean.columns else []
        best_return = role_row["best_return_factors"]
        worst_return = role_row["worst_return_factors"]
        best_sharpe = role_row["best_sharpe_factors"]
        worst_drawdown = role_row["largest_drawdown_factors"]
        hedges = role_row["equity_hedge_candidates"]
        inflation_hedges = role_row["inflation_hedge_candidates"]
        defensive = role_row["defensive_candidates"]
        lines.extend(
            [
                f"- Strongest positive performance: {best_return}.",
                f"- Weakest performance: {worst_return}.",
                f"- Best Sharpe: {best_sharpe}.",
                f"- Worst drawdown: {worst_drawdown}.",
                f"- Equity hedge candidates: {hedges}.",
                f"- Equity-fixed income correlation: {eq_dur_corr:.2f}." if pd.notna(eq_dur_corr) else "- Equity-fixed income correlation: unavailable.",
                f"- Commodity / gold / trend / defensive signals that look useful: {', '.join([x for x in [inflation_hedges, defensive] if x != 'None']) or 'None'}.",
                f"- Macro shocks with the largest absolute means: {', '.join(macro_shocks) if macro_shocks else 'None'}.",
                f"- Short interpretation: {role_row['short_interpretation']}",
                "",
            ]
        )
        if regime_name == "Deflationary Macro-Financial Stress" and int(role_row["n_obs"]) < SMALL_SAMPLE_THRESHOLD:
            lines.append("- Warning: the stress regime has a small sample, so ranking results are fragile.")
            lines.append("")

    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")


def plot_regime_factor_profiles(annualized_return: pd.DataFrame) -> None:
    if annualized_return.empty:
        return
    ncols = 2
    nrows = math.ceil(len(REGIME_ORDER) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 4.4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    y_min = np.nanmin(annualized_return.to_numpy(dtype=float))
    y_max = np.nanmax(annualized_return.to_numpy(dtype=float))
    y_pad = 0.05 * max(1e-6, y_max - y_min)

    for ax, regime_name in zip(axes, REGIME_ORDER):
        if regime_name not in annualized_return.columns:
            ax.axis("off")
            continue
        series = annualized_return[regime_name].dropna()
        colors = ["#2c7fb8" if value >= 0 else "#d95f0e" for value in series]
        ax.bar(series.index, series.values, color=colors, alpha=0.85)
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_title(regime_name)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.2)
    for ax in axes[len(REGIME_ORDER):]:
        ax.axis("off")
    fig.suptitle("Regime Factor Return Profiles", fontsize=16, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(REGIME_PROFILE_PATH, dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    panel = load_core_panel()
    return_like, shock, missing = select_available_factors(panel)
    performance = compute_factor_performance_by_regime(panel, return_like, shock)
    performance.to_csv(PERFORMANCE_PATH, index=False)

    annualized_return = create_metric_pivot(performance, return_like, "annualized_return")
    annualized_vol = create_metric_pivot(performance, return_like, "annualized_volatility")
    sharpe = create_metric_pivot(performance, return_like, "sharpe_ratio")
    drawdown = create_metric_pivot(performance, return_like, "max_drawdown")
    positive_ratio = create_metric_pivot(performance, return_like, "positive_month_ratio")
    shock_mean = create_metric_pivot(performance, shock, "mean_monthly")
    shock_std = create_metric_pivot(performance, shock, "std_monthly")

    annualized_return.to_csv(ANNUAL_RETURN_PATH)
    annualized_vol.to_csv(ANNUAL_VOL_PATH)
    sharpe.to_csv(SHARPE_PATH)
    drawdown.to_csv(MAX_DRAWDOWN_PATH)
    positive_ratio.to_csv(POSITIVE_RATIO_PATH)
    shock_mean.to_csv(SHOCK_MEAN_PATH)
    shock_std.to_csv(SHOCK_STD_PATH)

    plot_heatmap(annualized_return, "Annualized Return by Regime", ANNUAL_RETURN_HEATMAP_PATH, fmt="{:.2%}")
    plot_heatmap(sharpe, "Sharpe Ratio by Regime", SHARPE_HEATMAP_PATH, fmt="{:.2f}")
    plot_heatmap(drawdown, "Max Drawdown by Regime", MAX_DRAWDOWN_HEATMAP_PATH, fmt="{:.2%}")
    plot_heatmap(positive_ratio, "Positive Month Ratio by Regime", POSITIVE_RATIO_HEATMAP_PATH, vmin=0, vmax=1, fmt="{:.2%}")

    tradable_boxplot_factors = [col for col in ["MKT_EXCESS", "AQR_FI_MARKET_EXCESS", "AQR_CMDTY_EW_EXCESS", "GOLD_EXCESS", "AQR_CMDTY_CARRY", "BAB"] if col in panel.columns]
    aqr_boxplot_factors = [col for col in ["AQR_TSMOM", "AQR_QMJ", "FF_SMB", "FF_HML", "FF_RMW", "FF_CMA", "FF_MOM"] if col in panel.columns]
    shock_boxplot_factors = [col for col in SHOCK_CANDIDATES if col in panel.columns]

    plot_factor_boxplots(panel, tradable_boxplot_factors, "Return-Like Tradable Factors by Regime", RETURN_BOXPLOT_PATH)
    plot_factor_boxplots(panel, aqr_boxplot_factors, "AQR / Style Factors by Regime", AQR_BOXPLOT_PATH)
    plot_factor_boxplots(panel, shock_boxplot_factors, "Shock Factors by Regime", SHOCK_BOXPLOT_PATH)

    corr_factors = [factor for factor in CORRELATION_FACTOR_CANDIDATES if factor in panel.columns]
    corr_matrices = compute_regime_correlation_matrices(panel, corr_factors)
    corr_saved_paths: list[Path] = []
    for regime_name, corr in corr_matrices.items():
        path = RESULTS_DIR / f"correlation_matrix_{slugify(regime_name)}.csv"
        corr.to_csv(path)
        corr_saved_paths.append(path)
    corr_heatmap_paths = plot_regime_correlation_heatmaps(corr_matrices, panel)

    key_pairwise = compute_key_pairwise_correlations(panel)
    key_pairwise.to_csv(KEY_PAIRWISE_PATH, index=False)
    key_pairwise_pivot = key_pairwise.pivot(index="factor_pair", columns="regime_name", values="correlation").reindex(columns=REGIME_ORDER)
    plot_heatmap(key_pairwise_pivot, "Key Pairwise Correlations by Regime", KEY_PAIRWISE_HEATMAP_PATH, vmin=-1, vmax=1, fmt="{:.2f}")

    role_summary = create_regime_role_summary(performance, key_pairwise, return_like, shock)
    role_summary.to_csv(ROLE_SUMMARY_PATH, index=False)
    plot_regime_factor_profiles(annualized_return)

    small_sample_warnings = []
    regime_counts = panel.groupby("regime_name")["date"].count().to_dict()
    for regime_name in REGIME_ORDER:
        n_obs = int(regime_counts.get(regime_name, 0))
        if n_obs < SMALL_SAMPLE_THRESHOLD:
            small_sample_warnings.append(f"{regime_name} has only {n_obs} monthly observations.")

    write_markdown_report(role_summary, key_pairwise, performance, return_like, shock, small_sample_warnings)

    print(f"Missing requested factor columns: {', '.join(missing) if missing else 'None'}")
    print(f"Selected return-like factors: {len(return_like)}")
    print(f"Selected shock factors: {len(shock)}")
    print(f"Regimes analyzed: {', '.join(REGIME_ORDER)}")
    print(f"Small-sample regime warnings: {' | '.join(small_sample_warnings) if small_sample_warnings else 'None'}")

    saved_paths = [
        PERFORMANCE_PATH,
        ANNUAL_RETURN_PATH,
        ANNUAL_VOL_PATH,
        SHARPE_PATH,
        MAX_DRAWDOWN_PATH,
        POSITIVE_RATIO_PATH,
        SHOCK_MEAN_PATH,
        SHOCK_STD_PATH,
        KEY_PAIRWISE_PATH,
        ROLE_SUMMARY_PATH,
        MARKDOWN_PATH,
        ANNUAL_RETURN_HEATMAP_PATH,
        SHARPE_HEATMAP_PATH,
        MAX_DRAWDOWN_HEATMAP_PATH,
        POSITIVE_RATIO_HEATMAP_PATH,
        RETURN_BOXPLOT_PATH,
        AQR_BOXPLOT_PATH,
        SHOCK_BOXPLOT_PATH,
        KEY_PAIRWISE_HEATMAP_PATH,
        REGIME_PROFILE_PATH,
    ] + corr_saved_paths + corr_heatmap_paths

    for path in saved_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
