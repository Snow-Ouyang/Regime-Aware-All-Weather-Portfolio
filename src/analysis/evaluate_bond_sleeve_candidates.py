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

from regime.utils import REGIME_ORDER, REGIME_COLORS


REGIME_PANEL_PATH = ROOT / "data" / "processed" / "risk_factors" / "core_risk_factor_panel.csv"
MONTHLY_RETURNS_PATH = ROOT / "data" / "processed" / "assets" / "monthly_returns.csv"
DAILY_RETURNS_PATH = ROOT / "data" / "processed" / "assets" / "daily_returns.csv"

RESULTS_DIR = ROOT / "results" / "asset_selection" / "bond_sleeve"
FIGURES_DIR = ROOT / "figures" / "asset_selection" / "bond_sleeve"

PERF_PATH = RESULTS_DIR / "bond_performance_by_regime.csv"
ANN_RETURN_PATH = RESULTS_DIR / "bond_annualized_return_by_regime.csv"
SHARPE_PATH = RESULTS_DIR / "bond_sharpe_by_regime.csv"
MAX_DRAWDOWN_PATH = RESULTS_DIR / "bond_max_drawdown_by_regime.csv"
POSITIVE_RATIO_PATH = RESULTS_DIR / "bond_positive_month_ratio_by_regime.csv"
CORRELATION_PATH = RESULTS_DIR / "bond_correlation_by_regime.csv"
DOWNSIDE_PATH = RESULTS_DIR / "bond_downside_protection.csv"
COVERAGE_PATH = RESULTS_DIR / "bond_coverage_report.csv"
ROLE_SCORE_PATH = RESULTS_DIR / "bond_role_score.csv"
REPORT_PATH = RESULTS_DIR / "BOND_SLEEVE_SELECTION.md"

ANN_RETURN_HEATMAP_PATH = FIGURES_DIR / "bond_annualized_return_heatmap.png"
SHARPE_HEATMAP_PATH = FIGURES_DIR / "bond_sharpe_heatmap.png"
MAX_DRAWDOWN_HEATMAP_PATH = FIGURES_DIR / "bond_max_drawdown_heatmap.png"
ROLE_SCORE_BAR_PATH = FIGURES_DIR / "bond_role_score_bar.png"
REGIME_PROFILE_PATH = FIGURES_DIR / "bond_regime_return_profiles.png"
VS_SPY_HEATMAP_PATH = FIGURES_DIR / "bond_vs_spy_correlation_by_regime.png"
VS_GOLD_HEATMAP_PATH = FIGURES_DIR / "bond_vs_gold_correlation_by_regime.png"
VS_CASH_HEATMAP_PATH = FIGURES_DIR / "bond_vs_cash_correlation_by_regime.png"

BOND_CANDIDATES = ["SHY", "IEI", "IEF", "TLT", "GOVT", "EDV", "TIP", "STIP", "LQD", "VCIT", "VCSH"]
EQUITY_CHOICES = ["SPY"]
GOLD_CHOICES = ["GLD", "IAU"]
CASH_CHOICES = ["BIL", "SGOV", "SHV"]
COMMODITY_CHOICES = ["DBC", "GSG"]
SMALL_SAMPLE_THRESHOLD = 12


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def to_month_end(series: pd.Series) -> pd.Series:
    parsed = series.map(lambda x: pd.to_datetime(x, errors="coerce"))
    return parsed.dt.to_period("M").dt.to_timestamp("M")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    regimes = pd.read_csv(REGIME_PANEL_PATH, usecols=["date", "regime", "regime_name"])
    regimes["date"] = to_month_end(regimes["date"])
    monthly = pd.read_csv(MONTHLY_RETURNS_PATH)
    monthly["date"] = to_month_end(monthly["date"])
    monthly = monthly.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    regimes = regimes.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    return regimes, monthly


def pick_first_available(columns: list[str], df: pd.DataFrame) -> str | None:
    for col in columns:
        if col in df.columns:
            return col
    return None


def max_drawdown(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    cumulative = (1.0 + clean).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    return float(drawdown.min())


def summarize_return_series(series: pd.Series, dates: pd.Series) -> dict[str, object]:
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
        "max_drawdown": max_drawdown(clean),
        "worst_month": dates.loc[worst_idx].strftime("%Y-%m-%d"),
        "best_month": dates.loc[best_idx].strftime("%Y-%m-%d"),
        "positive_month_ratio": float((clean > 0).mean()),
    }


def compute_bond_performance(panel: pd.DataFrame, bonds: list[str], common_sample_dates: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_type, sample_dates in [("own_sample", None), ("common_sample", common_sample_dates)]:
        sample_panel = panel if sample_dates is None else panel.loc[panel["date"].isin(sample_dates)].copy()
        for etf in bonds:
            for regime_name in REGIME_ORDER:
                subset = sample_panel.loc[sample_panel["regime_name"] == regime_name, ["date", etf]].copy()
                summary = summarize_return_series(subset[etf], subset["date"])
                rows.append({"sample_type": sample_type, "etf": etf, "regime_name": regime_name, **summary})
    return pd.DataFrame(rows)


def create_pivot(df: pd.DataFrame, metric: str, sample_type: str = "own_sample") -> pd.DataFrame:
    subset = df.loc[df["sample_type"] == sample_type, ["etf", "regime_name", metric]]
    pivot = subset.pivot(index="etf", columns="regime_name", values=metric)
    return pivot.reindex(columns=REGIME_ORDER)


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


def compute_correlation_by_regime(panel: pd.DataFrame, bonds: list[str], refs: dict[str, str | None]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for etf in bonds:
        for regime_name in REGIME_ORDER:
            regime_panel = panel.loc[panel["regime_name"] == regime_name]
            for ref_role, ref_col in refs.items():
                if ref_col is None or ref_col not in regime_panel.columns:
                    corr = np.nan
                    n_obs = 0
                else:
                    sample = regime_panel[[etf, ref_col]].dropna()
                    corr = sample[etf].corr(sample[ref_col]) if len(sample) >= 2 else np.nan
                    n_obs = int(len(sample))
                rows.append(
                    {
                        "etf": etf,
                        "regime_name": regime_name,
                        "reference_role": ref_role,
                        "reference_asset": ref_col if ref_col is not None else "",
                        "correlation": corr,
                        "n_obs": n_obs,
                    }
                )
    return pd.DataFrame(rows)


def compute_downside_protection(panel: pd.DataFrame, bonds: list[str], spy_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    spy_series = pd.to_numeric(panel[spy_col], errors="coerce")
    bottom_cutoff = spy_series.quantile(0.1)
    spy_negative = panel.loc[spy_series < 0]
    spy_bottom = panel.loc[spy_series <= bottom_cutoff]
    stress_regime = "Deflationary Macro-Financial Stress"
    high_rate_regime = "High-Rate / Inflation-Pressure"

    for etf in bonds:
        neg_sample = spy_negative[[etf, spy_col]].dropna()
        bottom_sample = spy_bottom[[etf, spy_col]].dropna()
        stress_sample = panel.loc[panel["regime_name"] == stress_regime, etf].dropna()
        high_rate_sample = panel.loc[panel["regime_name"] == high_rate_regime, etf].dropna()
        rows.append(
            {
                "etf": etf,
                "avg_return_when_spy_negative": float(neg_sample[etf].mean()) if not neg_sample.empty else np.nan,
                "avg_return_in_spy_worst_10pct": float(bottom_sample[etf].mean()) if not bottom_sample.empty else np.nan,
                "hit_ratio_when_spy_negative": float((neg_sample[etf] > 0).mean()) if not neg_sample.empty else np.nan,
                "hit_ratio_in_spy_worst_10pct": float((bottom_sample[etf] > 0).mean()) if not bottom_sample.empty else np.nan,
                "avg_return_in_deflationary_stress": float(stress_sample.mean()) if not stress_sample.empty else np.nan,
                "max_drawdown_in_high_rate_regime": max_drawdown(high_rate_sample) if not high_rate_sample.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compute_coverage_report(panel: pd.DataFrame, bonds: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for etf in bonds:
        series = pd.to_numeric(panel[etf], errors="coerce")
        valid_idx = series.dropna().index
        regime_cover = {}
        for regime_name in REGIME_ORDER:
            mask = panel["regime_name"] == regime_name
            denom = int(mask.sum())
            numer = int(series.loc[mask].count())
            regime_cover[f"coverage_{regime_name}"] = np.nan if denom == 0 else numer / denom
        rows.append(
            {
                "etf": etf,
                "first_valid_date": panel.loc[valid_idx[0], "date"].strftime("%Y-%m-%d") if len(valid_idx) else "",
                "last_valid_date": panel.loc[valid_idx[-1], "date"].strftime("%Y-%m-%d") if len(valid_idx) else "",
                "valid_months": int(series.count()),
                "missing_ratio": float(series.isna().mean()),
                **regime_cover,
            }
        )
    return pd.DataFrame(rows)


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


def compute_role_score(perf: pd.DataFrame, corr: pd.DataFrame, downside: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    own = perf.loc[perf["sample_type"] == "own_sample"].copy()
    stress = own.loc[own["regime_name"] == "Deflationary Macro-Financial Stress"].set_index("etf")
    low_rate = own.loc[own["regime_name"] == "Low-Rate / Steep Curve"].set_index("etf")
    high_rate = own.loc[own["regime_name"] == "High-Rate / Inflation-Pressure"].set_index("etf")
    corr_spy = corr.loc[corr["reference_role"] == "equity_proxy"].groupby("etf")["correlation"].mean()
    downside = downside.set_index("etf")
    coverage = coverage.set_index("etf")

    score = pd.DataFrame(index=stress.index.union(low_rate.index).union(high_rate.index).union(downside.index).union(coverage.index))
    score["deflationary_stress_return_score"] = rank_score(stress["annualized_return"], ascending=False)
    score["deflationary_stress_sharpe_score"] = rank_score(stress["sharpe"], ascending=False)
    score["low_rate_steep_curve_score"] = rank_score(low_rate["sharpe"], ascending=False)
    score["high_rate_inflation_drawdown_penalty"] = rank_score(high_rate["max_drawdown"], ascending=True)
    score["correlation_with_spy_penalty"] = rank_score(corr_spy, ascending=True)
    downside_composite = (
        rank_score(downside["avg_return_when_spy_negative"], ascending=False)
        + rank_score(downside["avg_return_in_spy_worst_10pct"], ascending=False)
        + rank_score(downside["hit_ratio_when_spy_negative"], ascending=False)
        + rank_score(downside["hit_ratio_in_spy_worst_10pct"], ascending=False)
    ) / 4.0
    score["downside_protection_score"] = downside_composite
    score["data_coverage_score"] = rank_score(coverage["valid_months"], ascending=False)
    score["total_score"] = (
        score["deflationary_stress_return_score"].fillna(0)
        + score["deflationary_stress_sharpe_score"].fillna(0)
        + score["low_rate_steep_curve_score"].fillna(0)
        + score["high_rate_inflation_drawdown_penalty"].fillna(0)
        + score["correlation_with_spy_penalty"].fillna(0)
        + score["downside_protection_score"].fillna(0)
        + score["data_coverage_score"].fillna(0)
    ) / 7.0
    return score.reset_index().rename(columns={"index": "etf"}).sort_values("total_score", ascending=False)


def plot_role_scores(score: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(score["etf"], score["total_score"], color="#4C78A8", alpha=0.85)
    ax.set_title("Bond Role Score")
    ax.set_ylabel("Average normalized score")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(ROLE_SCORE_BAR_PATH, dpi=180)
    plt.close(fig)


def plot_regime_return_profiles(ann_return: pd.DataFrame) -> None:
    ncols = 2
    nrows = math.ceil(len(REGIME_ORDER) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.6 * nrows))
    axes = np.atleast_1d(axes).flatten()
    y_min = np.nanmin(ann_return.to_numpy(dtype=float))
    y_max = np.nanmax(ann_return.to_numpy(dtype=float))
    pad = 0.05 * max(1e-6, y_max - y_min)
    for ax, regime_name in zip(axes, REGIME_ORDER):
        series = ann_return[regime_name].dropna()
        colors = ["#2c7fb8" if val >= 0 else "#d95f0e" for val in series]
        ax.bar(series.index, series.values, color=colors, alpha=0.85)
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_title(regime_name)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.2)
    for ax in axes[len(REGIME_ORDER):]:
        ax.axis("off")
    fig.suptitle("Bond ETF Regime Return Profiles", fontsize=16, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(REGIME_PROFILE_PATH, dpi=180)
    plt.close(fig)


def write_report(
    role_score: pd.DataFrame,
    perf: pd.DataFrame,
    corr: pd.DataFrame,
    downside: pd.DataFrame,
    refs: dict[str, str | None],
) -> list[str]:
    own = perf.loc[perf["sample_type"] == "own_sample"].copy()
    stress = own.loc[own["regime_name"] == "Deflationary Macro-Financial Stress"].set_index("etf")
    high_rate = own.loc[own["regime_name"] == "High-Rate / Inflation-Pressure"].set_index("etf")
    low_rate = own.loc[own["regime_name"] == "Low-Rate / Steep Curve"].set_index("etf")
    corr_spy = corr.loc[corr["reference_role"] == "equity_proxy"].groupby("etf")["correlation"].mean().sort_values()
    shortlist = role_score["etf"].head(3).tolist()

    best_stress = stress["sharpe"].sort_values(ascending=False).index[0] if not stress["sharpe"].dropna().empty else "None"
    best_low_rate = low_rate["sharpe"].sort_values(ascending=False).index[0] if not low_rate["sharpe"].dropna().empty else "None"
    worst_high_rate = high_rate["max_drawdown"].sort_values().index[0] if not high_rate["max_drawdown"].dropna().empty else "None"
    lowest_spy_corr = corr_spy.index[0] if not corr_spy.empty else "None"

    lines = [
        "# Bond Sleeve Selection",
        "",
        "This note evaluates bond ETF candidates as implementable sleeve assets. It does not build a portfolio and it does not optimize weights.",
        "",
        f"- Equity reference: {refs.get('equity_proxy') or 'None'}",
        f"- Gold reference: {refs.get('gold_proxy') or 'None'}",
        f"- Cash reference: {refs.get('cash_proxy') or 'None'}",
        f"- Commodity reference: {refs.get('commodity_proxy') or 'None'}",
        "",
        f"## Which bond ETFs behave best in Deflationary Macro-Financial Stress?",
        "",
        f"- Best stress Sharpe: {best_stress}.",
        f"- Top stress annualized returns: {', '.join(stress['annualized_return'].sort_values(ascending=False).head(3).index.tolist()) if not stress.empty else 'None'}.",
        "",
        "## Which bond ETFs are most vulnerable in High-Rate / Inflation-Pressure?",
        "",
        f"- Worst high-rate drawdown: {worst_high_rate}.",
        f"- Weakest high-rate annualized returns: {', '.join(high_rate['annualized_return'].sort_values().head(3).index.tolist()) if not high_rate.empty else 'None'}.",
        "",
        "## Which bond ETFs have the best diversification against SPY?",
        "",
        f"- Lowest average correlation to SPY: {lowest_spy_corr}.",
        f"- Lowest-correlation group: {', '.join(corr_spy.head(3).index.tolist()) if not corr_spy.empty else 'None'}.",
        "",
        "## IEF/GOVT vs TLT",
        "",
        f"- IEF Sharpe in Low-Rate / Steep Curve: {low_rate.loc['IEF', 'sharpe']:.2f}." if "IEF" in low_rate.index and pd.notna(low_rate.loc["IEF", "sharpe"]) else "- IEF low-rate Sharpe unavailable.",
        f"- GOVT Sharpe in Low-Rate / Steep Curve: {low_rate.loc['GOVT', 'sharpe']:.2f}." if "GOVT" in low_rate.index and pd.notna(low_rate.loc["GOVT", "sharpe"]) else "- GOVT low-rate Sharpe unavailable.",
        f"- TLT Sharpe in Low-Rate / Steep Curve: {low_rate.loc['TLT', 'sharpe']:.2f}." if "TLT" in low_rate.index and pd.notna(low_rate.loc["TLT", "sharpe"]) else "- TLT low-rate Sharpe unavailable.",
        "- Use the role score and regime tables to judge whether IEF/GOVT offers a better balance than TLT between crisis convexity and inflation vulnerability.",
        "",
        "## Does TLT provide stronger crisis protection but worse inflation drawdown?",
        "",
        f"- TLT stress Sharpe: {stress.loc['TLT', 'sharpe']:.2f}." if "TLT" in stress.index and pd.notna(stress.loc["TLT", "sharpe"]) else "- TLT stress Sharpe unavailable.",
        f"- TLT high-rate drawdown: {high_rate.loc['TLT', 'max_drawdown']:.2%}." if "TLT" in high_rate.index and pd.notna(high_rate.loc["TLT", "max_drawdown"]) else "- TLT high-rate drawdown unavailable.",
        "",
        "## Should short-duration ETFs be treated as cash-like substitutes?",
        "",
        f"- Short-duration candidates to compare against cash sleeve: {', '.join([x for x in ['SHY', 'IEI', 'VCSH', 'STIP'] if x in role_score['etf'].tolist()])}.",
        "",
        "## Recommended shortlist",
        "",
        f"- Carry forward: {', '.join(shortlist) if shortlist else 'None'}.",
        "",
        "## Scoring formula",
        "",
        "- `deflationary_stress_return_score`: rank of annualized return in Deflationary Macro-Financial Stress.",
        "- `deflationary_stress_sharpe_score`: rank of Sharpe in Deflationary Macro-Financial Stress.",
        "- `low_rate_steep_curve_score`: rank of Sharpe in Low-Rate / Steep Curve.",
        "- `high_rate_inflation_drawdown_penalty`: better score for shallower drawdown in High-Rate / Inflation-Pressure.",
        "- `correlation_with_spy_penalty`: better score for lower average correlation with SPY across regimes.",
        "- `downside_protection_score`: average rank across SPY-down and SPY-worst-decile protection metrics.",
        "- `data_coverage_score`: rank of valid history length.",
        "- `total_score`: simple average of the normalized component scores.",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return [best_stress, best_low_rate, worst_high_rate, lowest_spy_corr, ", ".join(shortlist)]


def main() -> None:
    ensure_dirs()
    regimes, monthly = load_inputs()
    available_bonds = [ticker for ticker in BOND_CANDIDATES if ticker in monthly.columns]
    missing_bonds = [ticker for ticker in BOND_CANDIDATES if ticker not in monthly.columns]
    refs = {
        "equity_proxy": pick_first_available(EQUITY_CHOICES, monthly),
        "gold_proxy": pick_first_available(GOLD_CHOICES, monthly),
        "cash_proxy": pick_first_available(CASH_CHOICES, monthly),
        "commodity_proxy": pick_first_available(COMMODITY_CHOICES, monthly),
    }

    keep_cols = ["date"] + available_bonds + [col for col in refs.values() if col is not None]
    panel = regimes.merge(monthly[keep_cols], on="date", how="left")
    panel = panel.sort_values("date").reset_index(drop=True)

    common_mask = panel[available_bonds].notna().all(axis=1) if available_bonds else pd.Series(dtype=bool)
    common_dates = panel.loc[common_mask, "date"]

    performance = compute_bond_performance(panel, available_bonds, common_dates)
    performance.to_csv(PERF_PATH, index=False)

    ann_return = create_pivot(performance, "annualized_return")
    sharpe = create_pivot(performance, "sharpe")
    max_dd = create_pivot(performance, "max_drawdown")
    positive = create_pivot(performance, "positive_month_ratio")
    ann_return.to_csv(ANN_RETURN_PATH)
    sharpe.to_csv(SHARPE_PATH)
    max_dd.to_csv(MAX_DRAWDOWN_PATH)
    positive.to_csv(POSITIVE_RATIO_PATH)

    corr = compute_correlation_by_regime(panel, available_bonds, refs)
    corr.to_csv(CORRELATION_PATH, index=False)

    downside = compute_downside_protection(panel, available_bonds, refs["equity_proxy"]) if refs["equity_proxy"] else pd.DataFrame(columns=["etf"])
    downside.to_csv(DOWNSIDE_PATH, index=False)

    coverage = compute_coverage_report(panel, available_bonds)
    coverage.to_csv(COVERAGE_PATH, index=False)

    score = compute_role_score(performance, corr, downside, coverage)
    score.to_csv(ROLE_SCORE_PATH, index=False)

    plot_heatmap(ann_return, "Bond ETF Annualized Return by Regime", ANN_RETURN_HEATMAP_PATH, fmt="{:.2%}")
    plot_heatmap(sharpe, "Bond ETF Sharpe by Regime", SHARPE_HEATMAP_PATH, fmt="{:.2f}")
    plot_heatmap(max_dd, "Bond ETF Max Drawdown by Regime", MAX_DRAWDOWN_HEATMAP_PATH, fmt="{:.2%}")
    plot_role_scores(score)
    plot_regime_return_profiles(ann_return)

    for role, path in [("equity_proxy", VS_SPY_HEATMAP_PATH), ("gold_proxy", VS_GOLD_HEATMAP_PATH), ("cash_proxy", VS_CASH_HEATMAP_PATH)]:
        ref_corr = corr.loc[corr["reference_role"] == role].pivot(index="etf", columns="regime_name", values="correlation").reindex(columns=REGIME_ORDER)
        plot_heatmap(ref_corr, f"Bond vs {role.replace('_', ' ').title()} Correlation by Regime", path, fmt="{:.2f}", vmin=-1, vmax=1)

    summary = write_report(score, performance, corr, downside, refs)

    best_stress, best_low_rate, worst_high_rate, lowest_spy_corr, shortlist = summary
    print(f"Selected bond candidates: {', '.join(available_bonds)}")
    print(f"Missing bond candidates: {', '.join(missing_bonds) if missing_bonds else 'None'}")
    print(f"Best ETF by deflationary stress Sharpe: {best_stress}")
    print(f"Best ETF by low-rate Sharpe: {best_low_rate}")
    print(f"Worst ETF by high-rate drawdown: {worst_high_rate}")
    print(f"ETF with lowest average correlation to SPY: {lowest_spy_corr}")
    print(f"Recommended shortlist: {shortlist}")

    for path in [
        PERF_PATH,
        ANN_RETURN_PATH,
        SHARPE_PATH,
        MAX_DRAWDOWN_PATH,
        POSITIVE_RATIO_PATH,
        CORRELATION_PATH,
        DOWNSIDE_PATH,
        COVERAGE_PATH,
        ROLE_SCORE_PATH,
        REPORT_PATH,
        ANN_RETURN_HEATMAP_PATH,
        SHARPE_HEATMAP_PATH,
        MAX_DRAWDOWN_HEATMAP_PATH,
        ROLE_SCORE_BAR_PATH,
        REGIME_PROFILE_PATH,
        VS_SPY_HEATMAP_PATH,
        VS_GOLD_HEATMAP_PATH,
        VS_CASH_HEATMAP_PATH,
    ]:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
