from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from regime.utils import REGIME_ORDER


FACTOR_PANEL_PATH = ROOT / "data" / "processed" / "risk_factors" / "core_risk_factor_panel.csv"
MONTHLY_RETURNS_PATH = ROOT / "data" / "processed" / "assets" / "monthly_returns.csv"

RESULTS_DIR = ROOT / "results" / "asset_selection" / "quality_sleeve"
FIGURES_DIR = ROOT / "figures" / "asset_selection" / "quality_sleeve"

COVERAGE_PATH = RESULTS_DIR / "quality_etf_coverage_report.csv"
PERFORMANCE_PATH = RESULTS_DIR / "quality_performance_by_regime.csv"
ANN_RETURN_PATH = RESULTS_DIR / "quality_annualized_return_by_regime.csv"
SHARPE_PATH = RESULTS_DIR / "quality_sharpe_by_regime.csv"
MAX_DD_PATH = RESULTS_DIR / "quality_max_drawdown_by_regime.csv"
REL_SPY_PATH = RESULTS_DIR / "quality_relative_to_spy_by_regime.csv"
CORR_PATH = RESULTS_DIR / "quality_correlation_by_regime.csv"
DOWNSIDE_PATH = RESULTS_DIR / "quality_downside_protection.csv"
REGRESSION_PATH = RESULTS_DIR / "quality_factor_exposure_regressions.csv"
QMJ_SUMMARY_PATH = RESULTS_DIR / "quality_qmj_exposure_summary.csv"
TSMOM_SUMMARY_PATH = RESULTS_DIR / "quality_tsmom_exposure_summary.csv"
ROLE_SCORE_PATH = RESULTS_DIR / "quality_role_score.csv"
REPORT_PATH = RESULTS_DIR / "QUALITY_SLEEVE_SELECTION.md"

ANN_RETURN_HEATMAP_PATH = FIGURES_DIR / "quality_annualized_return_heatmap.png"
SHARPE_HEATMAP_PATH = FIGURES_DIR / "quality_sharpe_heatmap.png"
MAX_DD_HEATMAP_PATH = FIGURES_DIR / "quality_max_drawdown_heatmap.png"
REL_SPY_HEATMAP_PATH = FIGURES_DIR / "quality_relative_to_spy_heatmap.png"
REGIME_PROFILE_PATH = FIGURES_DIR / "quality_regime_return_profiles.png"
VS_SPY_HEATMAP_PATH = FIGURES_DIR / "quality_vs_spy_correlation_by_regime.png"
VS_BOND_HEATMAP_PATH = FIGURES_DIR / "quality_vs_bond_correlation_by_regime.png"
VS_GOLD_HEATMAP_PATH = FIGURES_DIR / "quality_vs_gold_correlation_by_regime.png"
VS_QMJ_HEATMAP_PATH = FIGURES_DIR / "quality_vs_qmj_correlation_by_regime.png"
VS_TSMOM_HEATMAP_PATH = FIGURES_DIR / "quality_vs_tsmom_correlation_by_regime.png"
QMJ_BETA_BAR_PATH = FIGURES_DIR / "qmj_beta_bar.png"
TSMOM_BETA_BAR_PATH = FIGURES_DIR / "tsmom_beta_bar.png"
MARKET_BETA_BAR_PATH = FIGURES_DIR / "market_beta_bar.png"
R2_BAR_PATH = FIGURES_DIR / "regression_r2_bar.png"
QMJ_VS_MKT_SCATTER_PATH = FIGURES_DIR / "qmj_vs_market_beta_scatter.png"
ROLE_SCORE_BAR_PATH = FIGURES_DIR / "quality_role_score_bar.png"

PRIMARY_CANDIDATES = ["QUAL", "SPHQ", "SCHD", "DGRO", "VIG", "NOBL", "QGRO", "QGRW"]
DEFENSIVE_CANDIDATES = ["USMV", "SPLV", "ACWV"]
VALUE_QUALITY_CANDIDATES = ["VFQY", "AVUV", "DFAT", "DGRW"]
ALL_CANDIDATES = PRIMARY_CANDIDATES + DEFENSIVE_CANDIDATES + VALUE_QUALITY_CANDIDATES

REQUIRED_FACTORS = [
    "MKT_EXCESS",
    "AQR_QMJ",
    "AQR_TSMOM",
    "BAB",
    "FF_RMW",
    "FF_CMA",
    "FF_HML",
    "FF_MOM",
    "RF_MONTHLY",
    "GOLD_EXCESS",
    "AQR_FI_MARKET_EXCESS",
]

REGRESSIONS = {
    "qmj_focus": ["MKT_EXCESS", "AQR_QMJ"],
    "qmj_tsmom_focus": ["MKT_EXCESS", "AQR_QMJ", "AQR_TSMOM"],
    "academic_core": ["MKT_EXCESS", "AQR_QMJ", "AQR_TSMOM", "BAB", "FF_RMW", "FF_CMA"],
    "academic_extended": ["MKT_EXCESS", "AQR_QMJ", "AQR_TSMOM", "BAB", "FF_RMW", "FF_CMA", "FF_HML", "FF_MOM"],
}
MIN_CORRELATION_OBS = 6


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def to_month_end(series: pd.Series) -> pd.Series:
    parsed = series.map(lambda x: pd.to_datetime(x, errors="coerce"))
    return parsed.dt.to_period("M").dt.to_timestamp("M")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    factors = pd.read_csv(FACTOR_PANEL_PATH)
    monthly = pd.read_csv(MONTHLY_RETURNS_PATH)
    factors["date"] = to_month_end(factors["date"])
    monthly["date"] = to_month_end(monthly["date"])
    factors = factors.dropna(subset=["date", "regime_name"]).sort_values("date").drop_duplicates("date", keep="last")
    monthly = monthly.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    return factors.reset_index(drop=True), monthly.reset_index(drop=True)


def first_available(columns: list[str], df: pd.DataFrame) -> str | None:
    for col in columns:
        if col in df.columns:
            return col
    return None


def compute_max_drawdown(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    cumulative = (1.0 + clean).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    return float(drawdown.min())


def summarize_series(series: pd.Series, dates: pd.Series) -> dict[str, object]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {
            "n_obs": 0,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "worst_month": "",
            "best_month": "",
            "positive_month_ratio": np.nan,
        }
    mean_monthly = float(clean.mean())
    vol_monthly = float(clean.std())
    ann_return = float((1.0 + mean_monthly) ** 12 - 1.0)
    ann_vol = float(vol_monthly * np.sqrt(12.0)) if not np.isnan(vol_monthly) else np.nan
    sharpe = np.nan if pd.isna(ann_vol) or ann_vol == 0 else ann_return / ann_vol
    worst_idx = clean.idxmin()
    best_idx = clean.idxmax()
    return {
        "n_obs": int(clean.count()),
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": compute_max_drawdown(clean),
        "worst_month": dates.loc[worst_idx].strftime("%Y-%m-%d"),
        "best_month": dates.loc[best_idx].strftime("%Y-%m-%d"),
        "positive_month_ratio": float((clean > 0).mean()),
    }


def rank_score(series: pd.Series, ascending: bool) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)
    ranks = valid.rank(ascending=ascending, method="average")
    if len(valid) == 1:
        scaled = pd.Series(1.0, index=valid.index)
    else:
        scaled = 1.0 - (ranks - 1.0) / (len(valid) - 1.0)
    return scaled.reindex(series.index)


def create_coverage_report(panel: pd.DataFrame, etfs: list[str]) -> pd.DataFrame:
    rows = []
    for etf in etfs:
        series = pd.to_numeric(panel[etf], errors="coerce")
        valid_idx = series.dropna().index
        row = {
            "etf": etf,
            "first_valid_date": panel.loc[valid_idx[0], "date"].strftime("%Y-%m-%d") if len(valid_idx) else "",
            "last_valid_date": panel.loc[valid_idx[-1], "date"].strftime("%Y-%m-%d") if len(valid_idx) else "",
            "valid_months": int(series.count()),
            "missing_months": int(series.isna().sum()),
            "missing_ratio": float(series.isna().mean()),
        }
        for regime_name in REGIME_ORDER:
            mask = panel["regime_name"] == regime_name
            denom = int(mask.sum())
            numer = int(series.loc[mask].count())
            row[f"valid_obs_{regime_name}"] = numer
            row[f"coverage_ratio_{regime_name}"] = np.nan if denom == 0 else numer / denom
        rows.append(row)
    return pd.DataFrame(rows)


def compute_performance_by_regime(panel: pd.DataFrame, etfs: list[str], spy_col: str, common_dates: pd.Series) -> pd.DataFrame:
    rows = []
    for sample_type, sample_dates in [("own_sample", None), ("common_sample", common_dates)]:
        sample_panel = panel if sample_dates is None else panel.loc[panel["date"].isin(sample_dates)].copy()
        for etf in etfs:
            etf_excess_col = f"{etf}_EXCESS"
            for regime_name in REGIME_ORDER:
                subset = sample_panel.loc[sample_panel["regime_name"] == regime_name, ["date", etf_excess_col, "SPY_EXCESS", spy_col]].copy()
                etf_summary = summarize_series(subset[etf_excess_col], subset["date"])
                spy_summary = summarize_series(subset["SPY_EXCESS"], subset["date"])
                sample = subset.dropna(subset=[etf_excess_col, "SPY_EXCESS", spy_col])
                spy_negative = sample.loc[sample[spy_col] < 0]
                downside_capture = np.nan
                if not spy_negative.empty:
                    denom = abs(float(spy_negative[spy_col].mean()))
                    if denom > 0:
                        downside_capture = float(spy_negative[etf_excess_col].mean()) / denom
                rows.append(
                    {
                        "sample_type": sample_type,
                        "etf": etf,
                        "regime_name": regime_name,
                        **etf_summary,
                        "ETF_minus_SPY_annualized_return": etf_summary["annualized_return"] - spy_summary["annualized_return"] if pd.notna(etf_summary["annualized_return"]) and pd.notna(spy_summary["annualized_return"]) else np.nan,
                        "ETF_minus_SPY_sharpe_difference": etf_summary["sharpe"] - spy_summary["sharpe"] if pd.notna(etf_summary["sharpe"]) and pd.notna(spy_summary["sharpe"]) else np.nan,
                        "ETF_drawdown_minus_SPY_drawdown": etf_summary["max_drawdown"] - spy_summary["max_drawdown"] if pd.notna(etf_summary["max_drawdown"]) and pd.notna(spy_summary["max_drawdown"]) else np.nan,
                        "ETF_downside_capture_vs_SPY": downside_capture,
                    }
                )
    return pd.DataFrame(rows)


def create_metric_pivot(df: pd.DataFrame, metric: str, sample_type: str = "own_sample") -> pd.DataFrame:
    subset = df.loc[df["sample_type"] == sample_type, ["etf", "regime_name", metric]]
    return subset.pivot(index="etf", columns="regime_name", values=metric).reindex(columns=REGIME_ORDER)


def plot_heatmap(df: pd.DataFrame, title: str, path: Path, fmt: str = "{:.2f}", vmin: float | None = None, vmax: float | None = None) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(1.8 + 1.5 * len(df.columns), 1.6 + 0.55 * len(df.index)))
    mat = df.to_numpy(dtype=float)
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            value = mat[i, j]
            ax.text(j, i, "" if np.isnan(value) else fmt.format(value), ha="center", va="center", fontsize=8, color="#111111")
    fig.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_regime_profiles(df: pd.DataFrame) -> None:
    ncols = 2
    nrows = math.ceil(len(REGIME_ORDER) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.6 * nrows))
    axes = np.atleast_1d(axes).flatten()
    y_min = np.nanmin(df.to_numpy(dtype=float))
    y_max = np.nanmax(df.to_numpy(dtype=float))
    pad = 0.05 * max(1e-6, y_max - y_min)
    for ax, regime_name in zip(axes, REGIME_ORDER):
        series = df[regime_name].dropna()
        colors = ["#2c7fb8" if val >= 0 else "#d95f0e" for val in series]
        ax.bar(series.index, series.values, color=colors, alpha=0.85)
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_title(regime_name)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.2)
    for ax in axes[len(REGIME_ORDER):]:
        ax.axis("off")
    fig.suptitle("Quality ETF Regime Return Profiles", fontsize=16, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(REGIME_PROFILE_PATH, dpi=180)
    plt.close(fig)


def compute_correlations(panel: pd.DataFrame, etfs: list[str], refs: dict[str, str | None]) -> pd.DataFrame:
    rows = []
    for etf in etfs:
        for regime_name in REGIME_ORDER:
            regime_panel = panel.loc[panel["regime_name"] == regime_name]
            for role, ref_col in refs.items():
                if ref_col is None or ref_col not in regime_panel.columns:
                    rows.append({"etf": etf, "regime_name": regime_name, "reference_role": role, "reference_asset": ref_col or "", "correlation": np.nan, "n_obs": 0})
                    continue
                sample = regime_panel[[etf, ref_col]].dropna()
                corr = sample[etf].corr(sample[ref_col]) if len(sample) >= MIN_CORRELATION_OBS else np.nan
                rows.append({"etf": etf, "regime_name": regime_name, "reference_role": role, "reference_asset": ref_col, "correlation": corr, "n_obs": int(len(sample))})
    return pd.DataFrame(rows)


def compute_downside_protection(panel: pd.DataFrame, etfs: list[str], spy_col: str) -> pd.DataFrame:
    rows = []
    spy = pd.to_numeric(panel[spy_col], errors="coerce")
    bottom_cutoff = spy.quantile(0.1)
    neg = panel.loc[spy < 0]
    bottom = panel.loc[spy <= bottom_cutoff]
    for etf in etfs:
        neg_sample = neg[[etf, spy_col]].dropna()
        bottom_sample = bottom[[etf, spy_col]].dropna()
        spy_worst_episode = panel.loc[spy <= bottom_cutoff, etf]
        rows.append(
            {
                "etf": etf,
                "average_return_when_spy_negative": float(neg_sample[etf].mean()) if not neg_sample.empty else np.nan,
                "average_excess_return_over_spy_when_spy_negative": float((neg_sample[etf] - neg_sample[spy_col]).mean()) if not neg_sample.empty else np.nan,
                "hit_ratio_when_spy_negative": float((neg_sample[etf] > 0).mean()) if not neg_sample.empty else np.nan,
                "average_return_in_spy_worst_10pct": float(bottom_sample[etf].mean()) if not bottom_sample.empty else np.nan,
                "average_excess_return_over_spy_in_worst_10pct": float((bottom_sample[etf] - bottom_sample[spy_col]).mean()) if not bottom_sample.empty else np.nan,
                "hit_ratio_in_spy_worst_10pct": float((bottom_sample[etf] > 0).mean()) if not bottom_sample.empty else np.nan,
                "max_drawdown_during_spy_worst_episodes": compute_max_drawdown(spy_worst_episode) if not spy_worst_episode.dropna().empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def fit_regression(panel: pd.DataFrame, etf: str, factors: list[str]) -> dict[str, object]:
    target_col = f"{etf}_EXCESS"
    needed = [target_col] + factors
    sample = panel[["date"] + needed].dropna().copy()
    if len(sample) < max(24, len(factors) + 6):
        return {
            "regression_name": "",
            "etf": etf,
            "n_obs": int(len(sample)),
            "start_date": sample["date"].min().strftime("%Y-%m-%d") if not sample.empty else "",
            "end_date": sample["date"].max().strftime("%Y-%m-%d") if not sample.empty else "",
            "alpha": np.nan,
            "alpha_tstat": np.nan,
            "R2": np.nan,
            "adj_R2": np.nan,
        }
    y = sample[target_col]
    X = sm.add_constant(sample[factors])
    result = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    out = {
        "regression_name": "",
        "etf": etf,
        "n_obs": int(result.nobs),
        "start_date": sample["date"].min().strftime("%Y-%m-%d"),
        "end_date": sample["date"].max().strftime("%Y-%m-%d"),
        "alpha": result.params.get("const", np.nan),
        "alpha_tstat": result.tvalues.get("const", np.nan),
        "R2": result.rsquared,
        "adj_R2": result.rsquared_adj,
    }
    for factor in ["MKT_EXCESS", "AQR_QMJ", "AQR_TSMOM", "BAB", "FF_RMW", "FF_CMA", "FF_HML", "FF_MOM"]:
        out[f"beta_{factor}"] = result.params.get(factor, np.nan)
        out[f"beta_{factor}_tstat"] = result.tvalues.get(factor, np.nan)
    return out


def run_regressions(panel: pd.DataFrame, etfs: list[str]) -> pd.DataFrame:
    rows = []
    available = set(panel.columns)
    for etf in etfs:
        for name, factors in REGRESSIONS.items():
            use = [factor for factor in factors if factor in available]
            if len(use) != len(factors):
                continue
            row = fit_regression(panel, etf, use)
            row["regression_name"] = name
            rows.append(row)
    return pd.DataFrame(rows)


def create_exposure_summary(regressions: pd.DataFrame, beta_col: str, path: Path) -> pd.DataFrame:
    requested_cols = [
        "etf",
        "beta_MKT_EXCESS",
        "beta_MKT_EXCESS_tstat",
        beta_col,
        f"{beta_col}_tstat",
        "beta_AQR_QMJ",
        "beta_AQR_QMJ_tstat",
        "beta_AQR_TSMOM",
        "beta_AQR_TSMOM_tstat",
        "R2",
        "adj_R2",
        "n_obs",
        "start_date",
        "end_date",
    ]
    cols: list[str] = []
    for col in requested_cols:
        if col in regressions.columns and col not in cols:
            cols.append(col)
    summary = regressions.loc[regressions["regression_name"] == "qmj_tsmom_focus", cols].copy()
    summary.to_csv(path, index=False)
    return summary


def plot_bar_from_summary(df: pd.DataFrame, col: str, title: str, path: Path) -> None:
    if df.empty:
        return
    ordered = df.sort_values(col, ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(ordered["etf"], ordered[col], color="#4C78A8", alpha=0.85)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_qmj_vs_market(summary: pd.DataFrame, coverage: pd.DataFrame, downside: pd.DataFrame) -> None:
    if summary.empty:
        return
    merged = summary.merge(coverage[["etf", "valid_months"]], on="etf", how="left").merge(downside[["etf", "max_drawdown_during_spy_worst_episodes"]], on="etf", how="left")
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        merged["beta_MKT_EXCESS"],
        merged["beta_AQR_QMJ"],
        s=merged["valid_months"].fillna(0) * 2.5,
        c=merged["max_drawdown_during_spy_worst_episodes"],
        cmap="RdBu_r",
        alpha=0.8,
    )
    for _, row in merged.iterrows():
        ax.text(row["beta_MKT_EXCESS"], row["beta_AQR_QMJ"], row["etf"], fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("Market beta")
    ax.set_ylabel("QMJ beta")
    ax.set_title("QMJ vs Market Beta")
    fig.colorbar(scatter, ax=ax, label="Max drawdown during SPY worst episodes")
    fig.tight_layout()
    fig.savefig(QMJ_VS_MKT_SCATTER_PATH, dpi=180)
    plt.close(fig)


def compute_role_score(regressions: pd.DataFrame, performance: pd.DataFrame, downside: pd.DataFrame, corr: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    reg = regressions.loc[regressions["regression_name"] == "qmj_tsmom_focus"].set_index("etf")
    own = performance.loc[performance["sample_type"] == "own_sample"].copy()
    sharpe = own.pivot(index="etf", columns="regime_name", values="sharpe")
    max_dd = own.pivot(index="etf", columns="regime_name", values="max_drawdown")
    rel = own.pivot(index="etf", columns="regime_name", values="ETF_minus_SPY_sharpe_difference")
    corr_spy = corr.loc[corr["reference_role"] == "spy"].groupby("etf")["correlation"].mean()
    downside = downside.set_index("etf")
    coverage = coverage.set_index("etf")

    score = pd.DataFrame(index=coverage.index)
    qmj_beta = rank_score(reg.get("beta_AQR_QMJ", pd.Series(dtype=float)), ascending=False)
    qmj_t = rank_score(reg.get("beta_AQR_QMJ_tstat", pd.Series(dtype=float)), ascending=False)
    score["qmj_exposure_score"] = (qmj_beta + qmj_t) / 2.0
    tsmom_beta = rank_score(reg.get("beta_AQR_TSMOM", pd.Series(dtype=float)), ascending=False)
    tsmom_t = rank_score(reg.get("beta_AQR_TSMOM_tstat", pd.Series(dtype=float)), ascending=False)
    score["tsmom_exposure_score"] = (tsmom_beta + tsmom_t) / 2.0
    market_beta = reg.get("beta_MKT_EXCESS", pd.Series(dtype=float))
    market_penalty = 1.0 - (market_beta - 1.0).abs()
    score["market_beta_penalty"] = rank_score(market_penalty, ascending=False)
    downside_comp = (
        rank_score(downside["average_return_when_spy_negative"], ascending=False)
        + rank_score(downside["average_excess_return_over_spy_when_spy_negative"], ascending=False)
        + rank_score(downside["average_return_in_spy_worst_10pct"], ascending=False)
        + rank_score(downside["hit_ratio_in_spy_worst_10pct"], ascending=False)
    ) / 4.0
    score["downside_protection_score"] = downside_comp
    positive_sharpe_count = sharpe.gt(0).sum(axis=1)
    regime_dd = -max_dd.mean(axis=1)
    stress_edge = rel.get("Deflationary Macro-Financial Stress", pd.Series(dtype=float)).fillna(0) + rel.get("High-Rate / Inflation-Pressure", pd.Series(dtype=float)).fillna(0)
    score["regime_robustness_score"] = (
        rank_score(positive_sharpe_count, ascending=False)
        + rank_score(regime_dd, ascending=False)
        + rank_score(stress_edge, ascending=False)
    ) / 3.0
    score["diversification_score"] = rank_score(-corr_spy, ascending=False)
    score["coverage_score"] = rank_score(coverage["valid_months"], ascending=False)
    score["total_score"] = (
        0.25 * score["qmj_exposure_score"].fillna(0)
        + 0.10 * score["tsmom_exposure_score"].fillna(0)
        + 0.20 * score["downside_protection_score"].fillna(0)
        + 0.20 * score["regime_robustness_score"].fillna(0)
        + 0.10 * score["diversification_score"].fillna(0)
        + 0.10 * score["market_beta_penalty"].fillna(0)
        + 0.05 * score["coverage_score"].fillna(0)
    )
    return score.reset_index().rename(columns={"index": "etf"}).sort_values("total_score", ascending=False)


def write_report(etfs: list[str], missing: list[str], performance: pd.DataFrame, downside: pd.DataFrame, regressions: pd.DataFrame, corr: pd.DataFrame, role_score: pd.DataFrame) -> tuple[str, str, str, str, list[str]]:
    own = performance.loc[performance["sample_type"] == "own_sample"].copy()
    ann = own.pivot(index="etf", columns="regime_name", values="ETF_minus_SPY_annualized_return")
    downside = downside.set_index("etf")
    reg = regressions.loc[regressions["regression_name"] == "qmj_tsmom_focus"].set_index("etf")
    corr_spy = corr.loc[corr["reference_role"] == "spy"].groupby("etf")["correlation"].mean().sort_values()

    best_qmj_beta = reg["beta_AQR_QMJ"].sort_values(ascending=False).index[0] if "beta_AQR_QMJ" in reg and not reg["beta_AQR_QMJ"].dropna().empty else "None"
    best_qmj_t = reg["beta_AQR_QMJ_tstat"].sort_values(ascending=False).index[0] if "beta_AQR_QMJ_tstat" in reg and not reg["beta_AQR_QMJ_tstat"].dropna().empty else "None"
    downside_rank = role_score.set_index("etf")["downside_protection_score"].sort_values(ascending=False)
    best_downside = downside_rank.index[0] if not downside_rank.empty else "None"
    robustness_rank = role_score.set_index("etf")["regime_robustness_score"].sort_values(ascending=False)
    best_robustness = robustness_rank.index[0] if not robustness_rank.empty else "None"
    lowest_spy_corr = corr_spy.index[0] if not corr_spy.empty else "None"
    top3 = role_score["etf"].head(3).tolist()

    lines = [
        "# Quality Sleeve Selection",
        "",
        "## Purpose",
        "",
        "- AQR_QMJ is strong in the long-history factor analysis and looks like the most practical style sleeve to map into long-only equity ETFs.",
        "- AQR_TSMOM is useful as a secondary robustness filter, but it is harder to implement directly as a static ETF sleeve.",
        "",
        "## Candidate universe",
        "",
        f"- Evaluated ETFs: {', '.join(etfs)}",
        f"- Missing ETFs: {', '.join(missing) if missing else 'None'}",
        "",
        "## Performance by regime",
        "",
        f"- Late-Cycle / Inflationary Flat Curve relative leaders vs SPY: {', '.join(ann['Late-Cycle / Inflationary Flat Curve'].sort_values(ascending=False).head(3).index.tolist()) if 'Late-Cycle / Inflationary Flat Curve' in ann else 'None'}",
        f"- Low-Rate / Steep Curve relative leaders vs SPY: {', '.join(ann['Low-Rate / Steep Curve'].sort_values(ascending=False).head(3).index.tolist()) if 'Low-Rate / Steep Curve' in ann else 'None'}",
        f"- High-Rate / Inflation-Pressure relative leaders vs SPY: {', '.join(ann['High-Rate / Inflation-Pressure'].sort_values(ascending=False).head(3).index.tolist()) if 'High-Rate / Inflation-Pressure' in ann else 'None'}",
        f"- Deflationary Macro-Financial Stress relative leaders vs SPY: {', '.join(ann['Deflationary Macro-Financial Stress'].sort_values(ascending=False).head(3).index.tolist()) if 'Deflationary Macro-Financial Stress' in ann else 'None'}",
        "",
        "## Downside protection",
        "",
        f"- Best downside profile: {best_downside}",
        f"- Strongest average excess return over SPY when SPY is negative: {downside['average_excess_return_over_spy_when_spy_negative'].sort_values(ascending=False).index[0] if not downside.empty else 'None'}",
        "",
        "## Factor exposure",
        "",
        f"- Strongest QMJ beta: {best_qmj_beta}",
        f"- Strongest QMJ t-stat: {best_qmj_t}",
        f"- Lowest average SPY correlation: {lowest_spy_corr}",
        "- High market beta with weak QMJ loading should be treated as equity beta replication rather than genuine quality exposure.",
        "",
        "## Diversification",
        "",
        f"- Most robust diversified candidates by total score: {', '.join(top3) if top3 else 'None'}",
        "",
        "## Recommendation",
        "",
        f"- Primary quality candidate: {top3[0] if len(top3) > 0 else 'None'}",
        f"- Defensive quality candidate: {top3[1] if len(top3) > 1 else 'None'}",
        f"- Dividend-quality candidate: {top3[2] if len(top3) > 2 else 'None'}",
        "- Reject / not enough evidence: candidates with short history, weak QMJ exposure, or little downside advantage vs SPY.",
        "",
        "## Caveats",
        "",
        "- Quality ETFs are long-only and cannot fully replicate AQR_QMJ, which is a long-short quality-minus-junk factor.",
        "- TSMOM is a dynamic multi-asset strategy, so ETF exposure to AQR_TSMOM is only a secondary filter.",
        "- ETF samples are shorter than the academic factor samples.",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return best_qmj_beta, best_qmj_t, best_downside, best_robustness, top3


def main() -> None:
    ensure_dirs()
    factor_panel, monthly = load_data()
    available_etfs = [ticker for ticker in ALL_CANDIDATES if ticker in monthly.columns]
    missing_etfs = [ticker for ticker in ALL_CANDIDATES if ticker not in monthly.columns]
    refs = {
        "spy": first_available(["SPY"], monthly),
        "bond": first_available(["IEF", "SHY"], monthly),
        "gold": first_available(["GLD", "IAU"], monthly),
        "cash": first_available(["BIL", "SGOV", "SHV"], monthly),
        "qmj": "AQR_QMJ" if "AQR_QMJ" in factor_panel.columns else None,
        "tsmom": "AQR_TSMOM" if "AQR_TSMOM" in factor_panel.columns else None,
    }

    needed_factor_cols = ["date", "regime", "regime_name"] + [c for c in REQUIRED_FACTORS if c in factor_panel.columns]
    panel = factor_panel[needed_factor_cols].merge(monthly[["date"] + available_etfs + [c for c in [refs["spy"], refs["bond"], refs["gold"], refs["cash"]] if c is not None]], on="date", how="left")
    panel = panel.sort_values("date").reset_index(drop=True)

    if refs["spy"] is None:
        raise RuntimeError("SPY is required as the equity benchmark for quality ETF screening.")

    for etf in available_etfs + [refs["spy"]]:
        panel[f"{etf}_EXCESS"] = panel[etf] - panel["RF_MONTHLY"]

    common_cols = available_etfs + [refs["spy"]]
    common_dates = panel.loc[panel[common_cols].notna().all(axis=1), "date"] if common_cols else pd.Series(dtype="datetime64[ns]")

    coverage = create_coverage_report(panel, available_etfs)
    coverage.to_csv(COVERAGE_PATH, index=False)

    performance = compute_performance_by_regime(panel, available_etfs, refs["spy"], common_dates)
    performance.to_csv(PERFORMANCE_PATH, index=False)
    ann_return = create_metric_pivot(performance, "annualized_return")
    sharpe = create_metric_pivot(performance, "sharpe")
    max_dd = create_metric_pivot(performance, "max_drawdown")
    rel_spy = create_metric_pivot(performance, "ETF_minus_SPY_annualized_return")
    ann_return.to_csv(ANN_RETURN_PATH)
    sharpe.to_csv(SHARPE_PATH)
    max_dd.to_csv(MAX_DD_PATH)
    rel_spy.to_csv(REL_SPY_PATH)

    corr = compute_correlations(panel, available_etfs, refs)
    corr.to_csv(CORR_PATH, index=False)

    downside = compute_downside_protection(panel, available_etfs, refs["spy"])
    downside.to_csv(DOWNSIDE_PATH, index=False)

    regressions = run_regressions(panel, available_etfs)
    regressions.to_csv(REGRESSION_PATH, index=False)
    qmj_summary = create_exposure_summary(regressions, "beta_AQR_QMJ", QMJ_SUMMARY_PATH)
    tsmom_summary = create_exposure_summary(regressions, "beta_AQR_TSMOM", TSMOM_SUMMARY_PATH)

    role_score = compute_role_score(regressions, performance, downside, corr, coverage)
    role_score.to_csv(ROLE_SCORE_PATH, index=False)

    plot_heatmap(ann_return, "Quality ETF Annualized Return by Regime", ANN_RETURN_HEATMAP_PATH, fmt="{:.2%}")
    plot_heatmap(sharpe, "Quality ETF Sharpe by Regime", SHARPE_HEATMAP_PATH, fmt="{:.2f}")
    plot_heatmap(max_dd, "Quality ETF Max Drawdown by Regime", MAX_DD_HEATMAP_PATH, fmt="{:.2%}")
    plot_heatmap(rel_spy, "Quality ETF Return Relative to SPY by Regime", REL_SPY_HEATMAP_PATH, fmt="{:.2%}")
    plot_regime_profiles(ann_return)

    for role, path, title in [
        ("spy", VS_SPY_HEATMAP_PATH, "Quality vs SPY Correlation by Regime"),
        ("bond", VS_BOND_HEATMAP_PATH, "Quality vs Bond Correlation by Regime"),
        ("gold", VS_GOLD_HEATMAP_PATH, "Quality vs Gold Correlation by Regime"),
        ("qmj", VS_QMJ_HEATMAP_PATH, "Quality vs QMJ Correlation by Regime"),
        ("tsmom", VS_TSMOM_HEATMAP_PATH, "Quality vs TSMOM Correlation by Regime"),
    ]:
        ref_corr = corr.loc[corr["reference_role"] == role].pivot(index="etf", columns="regime_name", values="correlation").reindex(columns=REGIME_ORDER)
        plot_heatmap(ref_corr, title, path, fmt="{:.2f}", vmin=-1, vmax=1)

    plot_bar_from_summary(qmj_summary, "beta_AQR_QMJ", "QMJ Beta", QMJ_BETA_BAR_PATH)
    plot_bar_from_summary(tsmom_summary, "beta_AQR_TSMOM", "TSMOM Beta", TSMOM_BETA_BAR_PATH)
    plot_bar_from_summary(qmj_summary, "beta_MKT_EXCESS", "Market Beta", MARKET_BETA_BAR_PATH)
    plot_bar_from_summary(qmj_summary, "R2", "Regression R-squared", R2_BAR_PATH)
    plot_qmj_vs_market(qmj_summary, coverage, downside)
    plot_bar_from_summary(role_score, "total_score", "Quality Role Score", ROLE_SCORE_BAR_PATH)

    best_qmj_beta, best_qmj_t, best_downside, best_robustness, top3 = write_report(available_etfs, missing_etfs, performance, downside, regressions, corr, role_score)
    lowest_spy_corr = corr.loc[corr["reference_role"] == "spy"].groupby("etf")["correlation"].mean().sort_values().index[0] if not corr.empty else "None"

    print(f"Evaluated ETF list: {', '.join(available_etfs)}")
    print(f"Missing ETFs: {', '.join(missing_etfs) if missing_etfs else 'None'}")
    print(f"Best ETF by QMJ beta: {best_qmj_beta}")
    print(f"Best ETF by QMJ t-stat: {best_qmj_t}")
    print(f"Best ETF by downside protection: {best_downside}")
    print(f"Best ETF by regime robustness: {best_robustness}")
    print(f"ETF with lowest SPY correlation: {lowest_spy_corr}")
    print(f"Top 3 by total quality_role_score: {', '.join(top3) if top3 else 'None'}")

    for path in [
        COVERAGE_PATH,
        PERFORMANCE_PATH,
        ANN_RETURN_PATH,
        SHARPE_PATH,
        MAX_DD_PATH,
        REL_SPY_PATH,
        CORR_PATH,
        DOWNSIDE_PATH,
        REGRESSION_PATH,
        QMJ_SUMMARY_PATH,
        TSMOM_SUMMARY_PATH,
        ROLE_SCORE_PATH,
        REPORT_PATH,
        ANN_RETURN_HEATMAP_PATH,
        SHARPE_HEATMAP_PATH,
        MAX_DD_HEATMAP_PATH,
        REL_SPY_HEATMAP_PATH,
        REGIME_PROFILE_PATH,
        VS_SPY_HEATMAP_PATH,
        VS_BOND_HEATMAP_PATH,
        VS_GOLD_HEATMAP_PATH,
        VS_QMJ_HEATMAP_PATH,
        VS_TSMOM_HEATMAP_PATH,
        QMJ_BETA_BAR_PATH,
        TSMOM_BETA_BAR_PATH,
        MARKET_BETA_BAR_PATH,
        R2_BAR_PATH,
        QMJ_VS_MKT_SCATTER_PATH,
        ROLE_SCORE_BAR_PATH,
    ]:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
