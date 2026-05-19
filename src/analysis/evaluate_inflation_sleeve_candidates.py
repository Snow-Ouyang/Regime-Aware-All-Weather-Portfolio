from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from regime.utils import REGIME_ORDER


FACTOR_PANEL_PATH = ROOT / "data" / "processed" / "risk_factors" / "core_risk_factor_panel.csv"
MONTHLY_RETURNS_PATH = ROOT / "data" / "processed" / "assets" / "monthly_returns.csv"

RESULTS_DIR = ROOT / "results" / "asset_selection" / "inflation_sleeve"
FIGURES_DIR = ROOT / "figures" / "asset_selection" / "inflation_sleeve"

COVERAGE_PATH = RESULTS_DIR / "inflation_etf_coverage_report.csv"
PERFORMANCE_PATH = RESULTS_DIR / "inflation_performance_by_regime.csv"
ANN_RETURN_PATH = RESULTS_DIR / "inflation_annualized_return_by_regime.csv"
EXCESS_RETURN_PATH = RESULTS_DIR / "inflation_excess_return_by_regime.csv"
SHARPE_PATH = RESULTS_DIR / "inflation_sharpe_by_regime.csv"
MAX_DD_PATH = RESULTS_DIR / "inflation_max_drawdown_by_regime.csv"
POS_RATIO_PATH = RESULTS_DIR / "inflation_positive_month_ratio_by_regime.csv"
RELATIVE_PATH = RESULTS_DIR / "inflation_relative_to_core_assets_by_regime.csv"
CORR_PATH = RESULTS_DIR / "inflation_correlation_by_regime.csv"
HIGH_INFL_PATH = RESULTS_DIR / "inflation_high_inflation_months.csv"
RISING_RATE_PATH = RESULTS_DIR / "inflation_rising_rate_months.csv"
DOUBLE_NEG_PATH = RESULTS_DIR / "inflation_stock_bond_double_negative.csv"
TRIPLE_PRESSURE_PATH = RESULTS_DIR / "inflation_stock_bond_gold_pressure.csv"
SPECIAL_TESTS_PATH = RESULTS_DIR / "inflation_special_stress_tests.csv"
ROLE_SCORE_PATH = RESULTS_DIR / "inflation_role_score.csv"
CLASSIFICATION_PATH = RESULTS_DIR / "inflation_candidate_role_classification.csv"
REPORT_PATH = RESULTS_DIR / "INFLATION_SLEEVE_SELECTION.md"

ANN_RETURN_HEATMAP_PATH = FIGURES_DIR / "inflation_annualized_return_heatmap.png"
EXCESS_RETURN_HEATMAP_PATH = FIGURES_DIR / "inflation_excess_return_heatmap.png"
SHARPE_HEATMAP_PATH = FIGURES_DIR / "inflation_sharpe_heatmap.png"
MAX_DD_HEATMAP_PATH = FIGURES_DIR / "inflation_max_drawdown_heatmap.png"
POS_RATIO_HEATMAP_PATH = FIGURES_DIR / "inflation_positive_month_ratio_heatmap.png"
REGIME_PROFILE_PATH = FIGURES_DIR / "inflation_regime_return_profiles.png"
REL_SPY_HEATMAP_PATH = FIGURES_DIR / "inflation_relative_to_spy_heatmap.png"
REL_IEF_HEATMAP_PATH = FIGURES_DIR / "inflation_relative_to_ief_heatmap.png"
REL_GOLD_HEATMAP_PATH = FIGURES_DIR / "inflation_relative_to_gold_heatmap.png"
REL_CASH_HEATMAP_PATH = FIGURES_DIR / "inflation_relative_to_cash_heatmap.png"
VS_SPY_HEATMAP_PATH = FIGURES_DIR / "inflation_vs_spy_correlation_by_regime.png"
VS_IEF_HEATMAP_PATH = FIGURES_DIR / "inflation_vs_ief_correlation_by_regime.png"
VS_SHY_HEATMAP_PATH = FIGURES_DIR / "inflation_vs_shy_correlation_by_regime.png"
VS_GOLD_HEATMAP_PATH = FIGURES_DIR / "inflation_vs_gold_correlation_by_regime.png"
VS_CASH_HEATMAP_PATH = FIGURES_DIR / "inflation_vs_cash_correlation_by_regime.png"
HIGH_INFL_BAR_PATH = FIGURES_DIR / "inflation_high_inflation_months_return_bar.png"
RISING_RATE_BAR_PATH = FIGURES_DIR / "inflation_rising_rate_months_return_bar.png"
DOUBLE_NEG_BAR_PATH = FIGURES_DIR / "inflation_stock_bond_double_negative_return_bar.png"
TRIPLE_PRESSURE_BAR_PATH = FIGURES_DIR / "inflation_stock_bond_gold_pressure_return_bar.png"
ROLE_SCORE_BAR_PATH = FIGURES_DIR / "inflation_role_score_bar.png"

BROAD_COMMODITY = ["DBC", "PDBC", "GSG", "COMT", "BCI", "DJP"]
AGRI = ["DBA", "CORN", "WEAT", "SOYB"]
ENERGY = ["USO", "BNO", "UNG", "XLE"]
MATERIALS = ["XLB", "XME", "PICK", "COPX"]
TIPS = ["STIP", "VTIP", "TIP", "SCHP"]
REITS = ["VNQ", "REET"]
ALL_CANDIDATES = BROAD_COMMODITY + AGRI + ENERGY + MATERIALS + TIPS + REITS
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


def first_available(cols: list[str], df: pd.DataFrame) -> str | None:
    for col in cols:
        if col in df.columns:
            return col
    return None


def max_drawdown(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    cumulative = (1.0 + clean).cumprod()
    running_max = cumulative.cummax()
    dd = cumulative / running_max - 1.0
    return float(dd.min())


def summarize_return(series: pd.Series, dates: pd.Series, excess_series: pd.Series | None = None) -> dict[str, object]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {
            "n_obs": 0,
            "annualized_return": np.nan,
            "annualized_excess_return": np.nan,
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
    excess_ann = np.nan
    if excess_series is not None:
        excess_clean = pd.to_numeric(excess_series, errors="coerce").dropna()
        if not excess_clean.empty:
            excess_ann = float((1.0 + float(excess_clean.mean())) ** 12 - 1.0)
    worst_idx = clean.idxmin()
    best_idx = clean.idxmax()
    return {
        "n_obs": int(clean.count()),
        "annualized_return": ann_return,
        "annualized_excess_return": excess_ann,
        "annualized_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(clean),
        "worst_month": dates.loc[worst_idx].strftime("%Y-%m-%d"),
        "best_month": dates.loc[best_idx].strftime("%Y-%m-%d"),
        "positive_month_ratio": float((clean > 0).mean()),
    }


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


def compute_performance(panel: pd.DataFrame, etfs: list[str], common_dates: pd.Series) -> pd.DataFrame:
    rows = []
    for sample_type, sample_dates in [("own_sample", None), ("common_sample", common_dates)]:
        sample_panel = panel if sample_dates is None else panel.loc[panel["date"].isin(sample_dates)].copy()
        for etf in etfs:
            for regime_name in REGIME_ORDER:
                subset = sample_panel.loc[sample_panel["regime_name"] == regime_name, ["date", etf, f"{etf}_EXCESS"]].copy()
                summary = summarize_return(subset[etf], subset["date"], subset[f"{etf}_EXCESS"])
                rows.append({"sample_type": sample_type, "etf": etf, "regime_name": regime_name, **summary})
    return pd.DataFrame(rows)


def create_pivot(df: pd.DataFrame, metric: str, sample_type: str = "own_sample") -> pd.DataFrame:
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
    fig.suptitle("Inflation Sleeve Regime Return Profiles", fontsize=16, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(REGIME_PROFILE_PATH, dpi=180)
    plt.close(fig)


def compute_relative_performance(panel: pd.DataFrame, etfs: list[str], refs: dict[str, str | None]) -> pd.DataFrame:
    rows = []
    for etf in etfs:
        for regime_name in REGIME_ORDER:
            regime_panel = panel.loc[panel["regime_name"] == regime_name]
            etf_summary = summarize_return(regime_panel[etf], regime_panel["date"], regime_panel[f"{etf}_EXCESS"])
            row = {"etf": etf, "regime_name": regime_name}
            for key, ref in [("SPY", refs["spy"]), ("IEF", refs["ief"]), ("SHY", refs["shy"]), ("GOLD", refs["gold"]), ("CASH", refs["cash"])]:
                if ref is None or ref not in regime_panel.columns:
                    row[f"ETF_minus_{key}_annualized_return"] = np.nan
                    row[f"ETF_drawdown_minus_{key}_drawdown"] = np.nan
                    continue
                ref_summary = summarize_return(regime_panel[ref], regime_panel["date"], regime_panel[f"{ref}_EXCESS"] if f"{ref}_EXCESS" in regime_panel.columns else None)
                row[f"ETF_minus_{key}_annualized_return"] = etf_summary["annualized_return"] - ref_summary["annualized_return"] if pd.notna(etf_summary["annualized_return"]) and pd.notna(ref_summary["annualized_return"]) else np.nan
                row[f"ETF_drawdown_minus_{key}_drawdown"] = etf_summary["max_drawdown"] - ref_summary["max_drawdown"] if pd.notna(etf_summary["max_drawdown"]) and pd.notna(ref_summary["max_drawdown"]) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


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


def summarize_special_window(panel: pd.DataFrame, etfs: list[str], mask: pd.Series, refs: dict[str, str | None], label: str) -> pd.DataFrame:
    rows = []
    sample_panel = panel.loc[mask].copy()
    eq_gold_cols = [col for col in [refs["spy"], refs["ief"], refs["gold"]] if col is not None and col in sample_panel.columns]
    eq_gold_mean = sample_panel[eq_gold_cols].mean(axis=1) if eq_gold_cols else pd.Series(np.nan, index=sample_panel.index)
    for etf in etfs:
        series = pd.to_numeric(sample_panel[etf], errors="coerce").dropna()
        row = {
            "window_name": label,
            "etf": etf,
            "n_obs": int(series.count()),
            "average_return": float(series.mean()) if not series.empty else np.nan,
            "annualized_return": float((1.0 + float(series.mean())) ** 12 - 1.0) if not series.empty else np.nan,
            "hit_ratio": float((series > 0).mean()) if not series.empty else np.nan,
            "max_drawdown": max_drawdown(series) if not series.empty else np.nan,
            "excess_return_over_SPY": np.nan,
            "excess_return_over_IEF": np.nan,
            "excess_return_over_GLD": np.nan,
            "excess_return_over_CASH": np.nan,
            "relative_return_vs_equal_weight_SPY_IEF_GLD": np.nan,
        }
        for key, ref in [("SPY", refs["spy"]), ("IEF", refs["ief"]), ("GLD", refs["gold"]), ("CASH", refs["cash"])]:
            if ref is not None and ref in sample_panel.columns:
                aligned = sample_panel[[etf, ref]].dropna()
                if not aligned.empty:
                    row[f"excess_return_over_{key}"] = float((aligned[etf] - aligned[ref]).mean())
        if eq_gold_cols:
            aligned = pd.DataFrame({"etf": sample_panel[etf], "benchmark": eq_gold_mean}).dropna()
            if not aligned.empty:
                row["relative_return_vs_equal_weight_SPY_IEF_GLD"] = float((aligned["etf"] - aligned["benchmark"]).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def classify_role(etf: str) -> str:
    if etf in BROAD_COMMODITY:
        return "Broad commodity inflation beta"
    if etf in ENERGY:
        return "Energy-heavy inflation beta"
    if etf in MATERIALS:
        return "Materials / metals inflation equity"
    if etf in ["STIP", "VTIP"]:
        return "Short TIPS defensive inflation hedge"
    if etf in ["TIP", "SCHP"]:
        return "Broad TIPS inflation-linked bond"
    if etf in REITS:
        return "Real asset / REIT"
    if etf in AGRI:
        return "Broad commodity inflation beta"
    return "Not suitable"


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


def compute_role_score(perf: pd.DataFrame, rel: pd.DataFrame, corr: pd.DataFrame, coverage: pd.DataFrame, high_infl: pd.DataFrame, rising_rate: pd.DataFrame, double_neg: pd.DataFrame, triple_pressure: pd.DataFrame) -> pd.DataFrame:
    own = perf.loc[perf["sample_type"] == "own_sample"].copy()
    ann = own.pivot(index="etf", columns="regime_name", values="annualized_return")
    sharpe = own.pivot(index="etf", columns="regime_name", values="sharpe")
    dd = own.pivot(index="etf", columns="regime_name", values="max_drawdown")
    rel_own = rel.set_index(["etf", "regime_name"])
    corr_pivot = corr.pivot_table(index="etf", columns="reference_role", values="correlation", aggfunc="mean")
    coverage = coverage.set_index("etf")

    score = pd.DataFrame(index=coverage.index)
    infl_regime_raw = ann.get("Late-Cycle / Inflationary Flat Curve", pd.Series(dtype=float)) + 1.5 * ann.get("High-Rate / Inflation-Pressure", pd.Series(dtype=float)).fillna(0)
    score["inflation_regime_score"] = rank_score(infl_regime_raw, ascending=False)
    high_infl_avg = high_infl.groupby("etf")["annualized_return"].mean()
    score["high_inflation_month_score"] = rank_score(high_infl_avg, ascending=False)
    rising_rate_avg = rising_rate.groupby("etf")["annualized_return"].mean()
    score["rising_rate_score"] = rank_score(rising_rate_avg, ascending=False)
    double_neg_avg = double_neg.groupby("etf")["average_return"].mean()
    score["stock_bond_double_negative_score"] = rank_score(double_neg_avg, ascending=False)
    gold_fail_avg = triple_pressure.groupby("etf")["average_return"].mean()
    score["gold_failure_complement_score"] = rank_score(gold_fail_avg, ascending=False)
    score["drawdown_control_score"] = rank_score(-dd.mean(axis=1), ascending=False)
    divers_raw = -(corr_pivot.get("spy", pd.Series(dtype=float)).fillna(0) + corr_pivot.get("ief", pd.Series(dtype=float)).fillna(0) + corr_pivot.get("gold", pd.Series(dtype=float)).fillna(0)) / 3.0
    score["diversification_score"] = rank_score(divers_raw, ascending=False)
    score["coverage_score"] = rank_score(coverage["valid_months"], ascending=False)
    score["total_score"] = (
        0.20 * score["inflation_regime_score"].fillna(0.5)
        + 0.20 * score["high_inflation_month_score"].fillna(0.5)
        + 0.15 * score["rising_rate_score"].fillna(0.5)
        + 0.20 * score["stock_bond_double_negative_score"].fillna(0.5)
        + 0.10 * score["gold_failure_complement_score"].fillna(0.5)
        + 0.10 * score["drawdown_control_score"].fillna(0.5)
        + 0.05 * score["diversification_score"].fillna(0.5)
        + 0.05 * score["coverage_score"].fillna(0.5)
    )
    return score.reset_index().rename(columns={"index": "etf"}).sort_values("total_score", ascending=False)


def plot_bar(df: pd.DataFrame, xcol: str, ycol: str, title: str, path: Path) -> None:
    if df.empty:
        return
    ordered = df.sort_values(ycol, ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(ordered[xcol], ordered[ycol], color="#4C78A8", alpha=0.85)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(etfs: list[str], missing: list[str], perf: pd.DataFrame, special: dict[str, pd.DataFrame], corr: pd.DataFrame, role_score: pd.DataFrame) -> tuple[str, str, str, str, str, str, list[str]]:
    own = perf.loc[perf["sample_type"] == "own_sample"].copy()
    ann = own.pivot(index="etf", columns="regime_name", values="annualized_return")
    corr_spy = corr.loc[corr["reference_role"] == "spy"].groupby("etf")["correlation"].mean().sort_values()
    best_score = role_score["etf"].iloc[0] if not role_score.empty else "None"
    high_infl_best = special["high_inflation"].groupby("etf")["annualized_return"].mean().sort_values(ascending=False).index[0] if not special["high_inflation"].empty else "None"
    double_neg_best = special["double_negative"].groupby("etf")["average_return"].mean().sort_values(ascending=False).index[0] if not special["double_negative"].empty else "None"
    drawdown_best = own.groupby("etf")["max_drawdown"].mean().sort_values(ascending=False).index[0] if not own.empty else "None"
    broad_commodities = [etf for etf in role_score["etf"].tolist() if etf in BROAD_COMMODITY + AGRI][:1]
    broad_candidate = broad_commodities[0] if broad_commodities else "None"
    tips_candidates = [etf for etf in role_score["etf"].tolist() if etf in TIPS][:1]
    tips_candidate = tips_candidates[0] if tips_candidates else "None"
    rejected = []
    for _, row in role_score.iterrows():
        if row["total_score"] < role_score["total_score"].median():
            rejected.append(f"{row['etf']}: weaker inflation sleeve score")
    shortlist = role_score["etf"].head(3).tolist()

    lines = [
        "# Inflation Sleeve Selection",
        "",
        "## Purpose",
        "",
        "- Gold and bonds can both be pressured when real yields rise.",
        "- Cash is defensive but may not provide enough upside in inflationary shock environments.",
        "- The inflation sleeve is intended to complement SPY, IEF/SHY, Gold, and Cash when traditional stock-bond-gold mixes are under pressure.",
        "",
        "## Candidate universe",
        "",
        f"- Evaluated ETFs: {', '.join(etfs)}",
        f"- Missing ETFs: {', '.join(missing) if missing else 'None'}",
        "",
        "## Regime performance",
        "",
        f"- Best in Late-Cycle / Inflationary Flat Curve: {ann['Late-Cycle / Inflationary Flat Curve'].sort_values(ascending=False).index[0] if 'Late-Cycle / Inflationary Flat Curve' in ann and not ann['Late-Cycle / Inflationary Flat Curve'].dropna().empty else 'None'}",
        f"- Best in High-Rate / Inflation-Pressure: {ann['High-Rate / Inflation-Pressure'].sort_values(ascending=False).index[0] if 'High-Rate / Inflation-Pressure' in ann and not ann['High-Rate / Inflation-Pressure'].dropna().empty else 'insufficient ETF history for High-Rate / Inflation-Pressure regime'}",
        f"- Best in Deflationary Macro-Financial Stress: {ann['Deflationary Macro-Financial Stress'].sort_values(ascending=False).index[0] if 'Deflationary Macro-Financial Stress' in ann and not ann['Deflationary Macro-Financial Stress'].dropna().empty else 'None'}",
        "",
        "## Inflation diagnostics",
        "",
        f"- Best high inflation months candidate: {high_infl_best}",
        f"- Best stock-bond double-negative candidate: {double_neg_best}",
        f"- Best broad commodity candidate: {broad_candidate}",
        f"- Best TIPS candidate: {tips_candidate}",
        "",
        "## Diversification",
        "",
        f"- Lowest average SPY correlation: {corr_spy.index[0] if not corr_spy.empty else 'None'}",
        "",
        "## Drawdown",
        "",
        f"- Best drawdown control: {drawdown_best}",
        "",
        "## Recommendation",
        "",
        f"- Defensive inflation candidate: {tips_candidate}",
        f"- Strong inflation beta candidate: {broad_candidate}",
        f"- Optional aggressive inflation candidate: {shortlist[2] if len(shortlist) > 2 else 'None'}",
        f"- Rejected candidates: {'; '.join(rejected) if rejected else 'None'}",
        "",
        "## Caveats",
        "",
        "- Commodity ETFs can suffer severe losses in deflationary stress.",
        "- Energy/materials ETFs contain equity beta.",
        "- TIPS protect against CPI inflation but remain exposed to real yield risk.",
        "- ETF histories do not cover the historical 1970s-1980s high-inflation regime.",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return best_score, high_infl_best, double_neg_best, drawdown_best, broad_candidate, tips_candidate, rejected


def main() -> None:
    ensure_dirs()
    factor_panel, monthly = load_data()
    available = [ticker for ticker in ALL_CANDIDATES if ticker in monthly.columns]
    missing = [ticker for ticker in ALL_CANDIDATES if ticker not in monthly.columns]
    refs = {
        "spy": first_available(["SPY"], monthly),
        "ief": first_available(["IEF"], monthly),
        "shy": first_available(["SHY"], monthly),
        "gold": first_available(["GLD", "IAU"], monthly),
        "cash": first_available(["BIL", "SGOV", "SHV"], monthly),
        "rf": "RF_MONTHLY" if "RF_MONTHLY" in factor_panel.columns else None,
        "d_gs10": "D_GS10" if "D_GS10" in factor_panel.columns else None,
        "d_inflation_pc1": "D_INFLATION_PC1" if "D_INFLATION_PC1" in factor_panel.columns else None,
        "d_credit_spread": "D_CREDIT_SPREAD" if "D_CREDIT_SPREAD" in factor_panel.columns else None,
    }

    factor_cols = ["date", "regime", "regime_name"] + [c for c in ["RF_MONTHLY", "MKT_EXCESS", "GOLD_EXCESS", "AQR_FI_MARKET_EXCESS", "growth_pc1", "inflation_pc1", "gs10", "term_spread_10y_1y", "credit_spread", "D_GS10", "D_INFLATION_PC1", "D_CREDIT_SPREAD"] if c in factor_panel.columns]
    ref_asset_cols = [c for c in [refs["spy"], refs["ief"], refs["shy"], refs["gold"], refs["cash"]] if c is not None]
    panel = factor_panel[factor_cols].merge(monthly[["date"] + available + ref_asset_cols], on="date", how="left")
    panel = panel.sort_values("date").reset_index(drop=True)

    for col in available + [c for c in ref_asset_cols if c is not None]:
        if "RF_MONTHLY" in panel.columns:
            panel[f"{col}_EXCESS"] = panel[col] - panel["RF_MONTHLY"]

    common_dates = panel.loc[panel[available].notna().all(axis=1), "date"] if available else pd.Series(dtype="datetime64[ns]")

    coverage = create_coverage_report(panel, available)
    coverage.to_csv(COVERAGE_PATH, index=False)

    performance = compute_performance(panel, available, common_dates)
    performance.to_csv(PERFORMANCE_PATH, index=False)
    ann_return = create_pivot(performance, "annualized_return")
    excess_return = create_pivot(performance, "annualized_excess_return")
    sharpe = create_pivot(performance, "sharpe")
    max_dd = create_pivot(performance, "max_drawdown")
    pos_ratio = create_pivot(performance, "positive_month_ratio")
    ann_return.to_csv(ANN_RETURN_PATH)
    excess_return.to_csv(EXCESS_RETURN_PATH)
    sharpe.to_csv(SHARPE_PATH)
    max_dd.to_csv(MAX_DD_PATH)
    pos_ratio.to_csv(POS_RATIO_PATH)

    relative = compute_relative_performance(panel, available, refs)
    relative.to_csv(RELATIVE_PATH, index=False)

    corr_refs = {
        "spy": refs["spy"],
        "ief": refs["ief"],
        "shy": refs["shy"],
        "gold": refs["gold"],
        "cash": refs["cash"] if refs["cash"] is not None else refs["rf"],
        "d_gs10": refs["d_gs10"],
        "d_inflation_pc1": refs["d_inflation_pc1"],
        "d_credit_spread": refs["d_credit_spread"],
    }
    correlations = compute_correlations(panel, available, corr_refs)
    correlations.to_csv(CORR_PATH, index=False)

    infl_threshold = panel["inflation_pc1"].quantile(0.8) if "inflation_pc1" in panel.columns else np.nan
    d_infl_threshold = panel["D_INFLATION_PC1"].quantile(0.8) if "D_INFLATION_PC1" in panel.columns else np.nan
    high_infl_mask = pd.Series(False, index=panel.index)
    if "inflation_pc1" in panel.columns:
        high_infl_mask = high_infl_mask | (panel["inflation_pc1"] >= infl_threshold)
    if "D_INFLATION_PC1" in panel.columns:
        high_infl_mask = high_infl_mask | (panel["D_INFLATION_PC1"] >= d_infl_threshold)
    rising_rate_threshold = panel["D_GS10"].quantile(0.8) if "D_GS10" in panel.columns else np.nan
    rising_rate_mask = panel["D_GS10"] >= rising_rate_threshold if "D_GS10" in panel.columns else pd.Series(False, index=panel.index)
    double_negative_mask = (panel[refs["spy"]] < 0) & (panel[refs["ief"]] < 0) if refs["spy"] and refs["ief"] else pd.Series(False, index=panel.index)
    triple_pressure_mask = double_negative_mask & (panel[refs["gold"]] < 0) if refs["gold"] and refs["spy"] and refs["ief"] else pd.Series(False, index=panel.index)

    high_infl = summarize_special_window(panel, available, high_infl_mask.fillna(False), refs, "high_inflation_months")
    rising_rate = summarize_special_window(panel, available, rising_rate_mask.fillna(False), refs, "rising_rate_months")
    double_negative = summarize_special_window(panel, available, double_negative_mask.fillna(False), refs, "stock_bond_double_negative")
    triple_pressure = summarize_special_window(panel, available, triple_pressure_mask.fillna(False), refs, "stock_bond_gold_pressure")
    special = pd.concat([high_infl, rising_rate, double_negative, triple_pressure], ignore_index=True)

    high_infl.to_csv(HIGH_INFL_PATH, index=False)
    rising_rate.to_csv(RISING_RATE_PATH, index=False)
    double_negative.to_csv(DOUBLE_NEG_PATH, index=False)
    triple_pressure.to_csv(TRIPLE_PRESSURE_PATH, index=False)
    special.to_csv(SPECIAL_TESTS_PATH, index=False)

    role_score = compute_role_score(performance, relative, correlations, coverage, high_infl, rising_rate, double_negative, triple_pressure)
    role_score.to_csv(ROLE_SCORE_PATH, index=False)

    classification = pd.DataFrame({"etf": available, "role_classification": [classify_role(etf) for etf in available]})
    classification.to_csv(CLASSIFICATION_PATH, index=False)

    plot_heatmap(ann_return, "Inflation Sleeve Annualized Return by Regime", ANN_RETURN_HEATMAP_PATH, fmt="{:.2%}")
    plot_heatmap(excess_return, "Inflation Sleeve Annualized Excess Return by Regime", EXCESS_RETURN_HEATMAP_PATH, fmt="{:.2%}")
    plot_heatmap(sharpe, "Inflation Sleeve Sharpe by Regime", SHARPE_HEATMAP_PATH, fmt="{:.2f}")
    plot_heatmap(max_dd, "Inflation Sleeve Max Drawdown by Regime", MAX_DD_HEATMAP_PATH, fmt="{:.2%}")
    plot_heatmap(pos_ratio, "Inflation Sleeve Positive Month Ratio by Regime", POS_RATIO_HEATMAP_PATH, fmt="{:.2%}", vmin=0, vmax=1)
    plot_regime_profiles(ann_return)

    for metric, role, path, title in [
        ("ETF_minus_SPY_annualized_return", None, REL_SPY_HEATMAP_PATH, "Inflation Relative to SPY by Regime"),
        ("ETF_minus_IEF_annualized_return", None, REL_IEF_HEATMAP_PATH, "Inflation Relative to IEF by Regime"),
        ("ETF_minus_GOLD_annualized_return", None, REL_GOLD_HEATMAP_PATH, "Inflation Relative to Gold by Regime"),
        ("ETF_minus_CASH_annualized_return", None, REL_CASH_HEATMAP_PATH, "Inflation Relative to Cash by Regime"),
    ]:
        pivot = relative.pivot(index="etf", columns="regime_name", values=metric).reindex(columns=REGIME_ORDER)
        plot_heatmap(pivot, title, path, fmt="{:.2%}")

    for role, path, title in [
        ("spy", VS_SPY_HEATMAP_PATH, "Inflation vs SPY Correlation by Regime"),
        ("ief", VS_IEF_HEATMAP_PATH, "Inflation vs IEF Correlation by Regime"),
        ("shy", VS_SHY_HEATMAP_PATH, "Inflation vs SHY Correlation by Regime"),
        ("gold", VS_GOLD_HEATMAP_PATH, "Inflation vs Gold Correlation by Regime"),
        ("cash", VS_CASH_HEATMAP_PATH, "Inflation vs Cash Correlation by Regime"),
    ]:
        pivot = correlations.loc[correlations["reference_role"] == role].pivot(index="etf", columns="regime_name", values="correlation").reindex(columns=REGIME_ORDER)
        plot_heatmap(pivot, title, path, fmt="{:.2f}", vmin=-1, vmax=1)

    plot_bar(high_infl, "etf", "annualized_return", "High Inflation Months Annualized Return", HIGH_INFL_BAR_PATH)
    plot_bar(rising_rate, "etf", "annualized_return", "Rising Rate Months Annualized Return", RISING_RATE_BAR_PATH)
    plot_bar(double_negative, "etf", "average_return", "Stock-Bond Double Negative Average Return", DOUBLE_NEG_BAR_PATH)
    plot_bar(triple_pressure, "etf", "average_return", "Stock-Bond-Gold Pressure Average Return", TRIPLE_PRESSURE_BAR_PATH)
    plot_bar(role_score, "etf", "total_score", "Inflation Role Score", ROLE_SCORE_BAR_PATH)

    best_score, high_infl_best, double_neg_best, drawdown_best, broad_candidate, tips_candidate, rejected = write_report(available, missing, performance, {"high_inflation": high_infl, "double_negative": double_negative}, correlations, role_score)

    print(f"Evaluated ETF list: {', '.join(available)}")
    print(f"Missing ETFs: {', '.join(missing) if missing else 'None'}")
    print(f"Best ETF by inflation_role_score: {best_score}")
    print(f"Best ETF in high inflation months: {high_infl_best}")
    print(f"Best ETF in stock-bond double-negative months: {double_neg_best}")
    print(f"Best ETF by drawdown control: {drawdown_best}")
    print(f"Best broad commodity candidate: {broad_candidate}")
    print(f"Best TIPS candidate: {tips_candidate}")
    print(f"Rejected ETFs with reason: {'; '.join(rejected) if rejected else 'None'}")

    for path in [
        COVERAGE_PATH,
        PERFORMANCE_PATH,
        ANN_RETURN_PATH,
        EXCESS_RETURN_PATH,
        SHARPE_PATH,
        MAX_DD_PATH,
        POS_RATIO_PATH,
        RELATIVE_PATH,
        CORR_PATH,
        HIGH_INFL_PATH,
        RISING_RATE_PATH,
        DOUBLE_NEG_PATH,
        TRIPLE_PRESSURE_PATH,
        SPECIAL_TESTS_PATH,
        ROLE_SCORE_PATH,
        CLASSIFICATION_PATH,
        REPORT_PATH,
        ANN_RETURN_HEATMAP_PATH,
        EXCESS_RETURN_HEATMAP_PATH,
        SHARPE_HEATMAP_PATH,
        MAX_DD_HEATMAP_PATH,
        POS_RATIO_HEATMAP_PATH,
        REGIME_PROFILE_PATH,
        REL_SPY_HEATMAP_PATH,
        REL_IEF_HEATMAP_PATH,
        REL_GOLD_HEATMAP_PATH,
        REL_CASH_HEATMAP_PATH,
        VS_SPY_HEATMAP_PATH,
        VS_IEF_HEATMAP_PATH,
        VS_SHY_HEATMAP_PATH,
        VS_GOLD_HEATMAP_PATH,
        VS_CASH_HEATMAP_PATH,
        HIGH_INFL_BAR_PATH,
        RISING_RATE_BAR_PATH,
        DOUBLE_NEG_BAR_PATH,
        TRIPLE_PRESSURE_BAR_PATH,
        ROLE_SCORE_BAR_PATH,
    ]:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
