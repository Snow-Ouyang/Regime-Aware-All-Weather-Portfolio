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

DAILY_CLOSE_PATH = ROOT / "data" / "processed" / "assets" / "daily_adjusted_close.csv"
DAILY_RETURNS_PATH = ROOT / "data" / "processed" / "assets" / "daily_returns.csv"
MONTHLY_CLOSE_PATH = ROOT / "data" / "processed" / "assets" / "monthly_adjusted_close.csv"
MONTHLY_RETURNS_PATH = ROOT / "data" / "processed" / "assets" / "monthly_returns.csv"
FACTOR_PANEL_PATH = ROOT / "data" / "processed" / "risk_factors" / "core_risk_factor_panel.csv"

RESULTS_DIR = ROOT / "results" / "asset_selection" / "equity_sleeve"
FIGURES_DIR = ROOT / "figures" / "asset_selection" / "equity_sleeve"

COVERAGE_PATH = RESULTS_DIR / "spy_qual_coverage_report.csv"
DAILY_STRAT_RET_PATH = RESULTS_DIR / "spy_qual_daily_strategy_returns.csv"
MONTHLY_STRAT_RET_PATH = RESULTS_DIR / "spy_qual_monthly_strategy_returns.csv"
DAILY_EQUITY_PATH = RESULTS_DIR / "spy_qual_daily_equity_curves.csv"
MONTHLY_EQUITY_PATH = RESULTS_DIR / "spy_qual_monthly_equity_curves.csv"
FULL_PERF_PATH = RESULTS_DIR / "spy_qual_full_sample_performance.csv"
REGIME_PERF_PATH = RESULTS_DIR / "spy_qual_performance_by_regime.csv"
REGIME_ANN_RETURN_PATH = RESULTS_DIR / "spy_qual_annualized_return_by_regime.csv"
REGIME_SHARPE_PATH = RESULTS_DIR / "spy_qual_sharpe_by_regime.csv"
REGIME_MAX_DD_PATH = RESULTS_DIR / "spy_qual_max_drawdown_by_regime.csv"
REGIME_POS_PATH = RESULTS_DIR / "spy_qual_positive_month_ratio_by_regime.csv"
REL_FULL_PATH = RESULTS_DIR / "spy_qual_relative_performance.csv"
REL_REGIME_PATH = RESULTS_DIR / "spy_qual_relative_performance_by_regime.csv"
DRAWDOWN_SERIES_PATH = RESULTS_DIR / "spy_qual_drawdown_series.csv"
TOP_DD_EPISODES_PATH = RESULTS_DIR / "spy_qual_top_drawdown_episodes.csv"
DOWNSIDE_PATH = RESULTS_DIR / "spy_qual_downside_protection.csv"
REGRESSION_PATH = RESULTS_DIR / "spy_qual_beta_alpha_regression.csv"
CAPTURE_PATH = RESULTS_DIR / "spy_qual_capture_ratios.csv"
REPORT_PATH = RESULTS_DIR / "SPY_QUAL_EQUITY_SLEEVE_COMPARISON.md"

WEALTH_CURVE_PATH = FIGURES_DIR / "spy_qual_same_start_wealth_curve.png"
WEALTH_CURVE_LOG_PATH = FIGURES_DIR / "spy_qual_same_start_wealth_curve_log.png"
ANN_RETURN_HEATMAP_PATH = FIGURES_DIR / "spy_qual_annualized_return_by_regime_heatmap.png"
SHARPE_HEATMAP_PATH = FIGURES_DIR / "spy_qual_sharpe_by_regime_heatmap.png"
MAX_DD_HEATMAP_PATH = FIGURES_DIR / "spy_qual_max_drawdown_by_regime_heatmap.png"
REGIME_PROFILE_PATH = FIGURES_DIR / "spy_qual_regime_return_profiles.png"
QUAL_MINUS_SPY_PATH = FIGURES_DIR / "qual_minus_spy_cumulative_relative_return.png"
MIX_MINUS_SPY_PATH = FIGURES_DIR / "mix_50_50_minus_spy_cumulative_relative_return.png"
REL_HEATMAP_PATH = FIGURES_DIR / "relative_return_by_regime_heatmap.png"
DRAWDOWN_CURVES_PATH = FIGURES_DIR / "spy_qual_drawdown_curves.png"

STRATEGIES = ["SPY_ONLY", "QUAL_ONLY", "SPY_QUAL_50_50"]


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def to_month_end(series: pd.Series) -> pd.Series:
    parsed = series.map(lambda x: pd.to_datetime(x, errors="coerce"))
    return parsed.dt.to_period("M").dt.to_timestamp("M")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_close = pd.read_csv(DAILY_CLOSE_PATH)
    daily_ret = pd.read_csv(DAILY_RETURNS_PATH)
    monthly_close = pd.read_csv(MONTHLY_CLOSE_PATH)
    monthly_ret = pd.read_csv(MONTHLY_RETURNS_PATH)
    factor_panel = pd.read_csv(FACTOR_PANEL_PATH, usecols=["date", "regime", "regime_name", "RF_MONTHLY"])
    for df in [daily_close, daily_ret]:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.dropna(subset=["date"], inplace=True)
        df.sort_values("date", inplace=True)
    for df in [monthly_close, monthly_ret, factor_panel]:
        df["date"] = to_month_end(df["date"])
        df.dropna(subset=["date"], inplace=True)
        df.sort_values("date", inplace=True)
        df.drop_duplicates("date", keep="last", inplace=True)
    return daily_close.reset_index(drop=True), daily_ret.reset_index(drop=True), monthly_close.reset_index(drop=True), monthly_ret.reset_index(drop=True), factor_panel.reset_index(drop=True)


def build_coverage_report(daily_close: pd.DataFrame, daily_ret: pd.DataFrame, monthly_close: pd.DataFrame, monthly_ret: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    rows = []
    starts = []
    ends = []
    for ticker in ["SPY", "QUAL"]:
        daily_valid = daily_close[["date", ticker]].dropna()
        monthly_valid = monthly_close[["date", ticker]].dropna()
        daily_ret_valid = daily_ret[["date", ticker]].dropna()
        monthly_ret_valid = monthly_ret[["date", ticker]].dropna()
        first_valid = max(daily_valid["date"].min(), daily_ret_valid["date"].min())
        last_valid = min(daily_valid["date"].max(), daily_ret_valid["date"].max())
        starts.append(first_valid)
        ends.append(last_valid)
        rows.append(
            {
                "ticker": ticker,
                "first_valid_date": first_valid.strftime("%Y-%m-%d"),
                "last_valid_date": last_valid.strftime("%Y-%m-%d"),
                "valid_daily_obs": int(len(daily_ret_valid.loc[(daily_ret_valid["date"] >= first_valid) & (daily_ret_valid["date"] <= last_valid)])),
                "valid_monthly_obs": int(len(monthly_ret_valid)),
                "missing_ratio": float(daily_ret[ticker].isna().mean()),
            }
        )
    common_start = max(starts)
    common_end = min(ends)
    return pd.DataFrame(rows), common_start, common_end


def construct_daily_strategies(daily_ret: pd.DataFrame, common_start: pd.Timestamp, common_end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = daily_ret.loc[(daily_ret["date"] >= common_start) & (daily_ret["date"] <= common_end), ["date", "SPY", "QUAL"]].dropna().copy()
    df["SPY_ONLY"] = df["SPY"]
    df["QUAL_ONLY"] = df["QUAL"]

    mix_returns = []
    w_spy = 0.5
    w_qual = 0.5
    current_month = None
    for _, row in df.iterrows():
        month = row["date"].to_period("M")
        if current_month is None or month != current_month:
            w_spy = 0.5
            w_qual = 0.5
            current_month = month
        port_ret = w_spy * row["SPY"] + w_qual * row["QUAL"]
        mix_returns.append(port_ret)
        post_spy = w_spy * (1.0 + row["SPY"])
        post_qual = w_qual * (1.0 + row["QUAL"])
        total = post_spy + post_qual
        w_spy = post_spy / total
        w_qual = post_qual / total
    df["SPY_QUAL_50_50"] = mix_returns

    equity = pd.DataFrame({"date": df["date"]})
    for col in STRATEGIES:
        equity[col] = (1.0 + df[col]).cumprod()
    return df[["date"] + STRATEGIES], equity


def derive_monthly_from_daily(daily_strategy_returns: pd.DataFrame, daily_equity: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_returns = daily_strategy_returns.copy()
    monthly_returns["month_end"] = monthly_returns["date"].dt.to_period("M").dt.to_timestamp("M")
    monthly_returns = monthly_returns.groupby("month_end")[STRATEGIES].apply(lambda g: (1.0 + g).prod() - 1.0).reset_index().rename(columns={"month_end": "date"})
    monthly_equity = pd.DataFrame({"date": monthly_returns["date"]})
    for col in STRATEGIES:
        monthly_equity[col] = (1.0 + monthly_returns[col]).cumprod()
    return monthly_returns, monthly_equity


def max_drawdown(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    running_max = clean.cummax()
    dd = clean / running_max - 1.0
    return float(dd.min())


def full_sample_performance(daily_equity: pd.DataFrame, daily_returns: pd.DataFrame, monthly_returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    start_date = daily_equity["date"].min()
    end_date = daily_equity["date"].max()
    n_years = max((end_date - start_date).days / 365.25, 1 / 365.25)
    for strategy in STRATEGIES:
        daily_ret = daily_returns[strategy].dropna()
        monthly_ret = monthly_returns[strategy].dropna()
        cumulative_return = float(daily_equity[strategy].iloc[-1] - 1.0)
        ann_return = float(daily_equity[strategy].iloc[-1] ** (1.0 / n_years) - 1.0)
        ann_vol = float(daily_ret.std() * np.sqrt(252.0))
        sharpe = np.nan if ann_vol == 0 or pd.isna(ann_vol) else ann_return / ann_vol
        dd = max_drawdown(daily_equity[strategy])
        calmar = np.nan if dd == 0 or pd.isna(dd) else ann_return / abs(dd)
        worst_month_idx = monthly_ret.idxmin()
        best_month_idx = monthly_ret.idxmax()
        rows.append(
            {
                "strategy": strategy,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "n_daily_obs": int(daily_ret.count()),
                "n_monthly_obs": int(monthly_ret.count()),
                "cumulative_return": cumulative_return,
                "annualized_return": ann_return,
                "annualized_volatility": ann_vol,
                "sharpe_ratio": sharpe,
                "max_drawdown": dd,
                "calmar_ratio": calmar,
                "worst_month": monthly_returns.loc[worst_month_idx, "date"].strftime("%Y-%m-%d"),
                "best_month": monthly_returns.loc[best_month_idx, "date"].strftime("%Y-%m-%d"),
                "positive_month_ratio": float((monthly_ret > 0).mean()),
                "skewness": float(monthly_ret.skew()),
                "kurtosis": float(monthly_ret.kurtosis()),
            }
        )
    return pd.DataFrame(rows)


def summarize_regime_series(monthly_returns: pd.DataFrame, strategy: str) -> dict[str, object]:
    clean = monthly_returns[strategy].dropna()
    if clean.empty:
        return {
            "n_obs": 0,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "max_drawdown": np.nan,
            "worst_month": "",
            "best_month": "",
            "positive_month_ratio": np.nan,
            "cumulative_return_within_regime": np.nan,
            "average_monthly_return": np.nan,
        }
    mean_monthly = float(clean.mean())
    vol_monthly = float(clean.std())
    ann_return = float((1.0 + mean_monthly) ** 12 - 1.0)
    ann_vol = float(vol_monthly * np.sqrt(12.0))
    sharpe = np.nan if ann_vol == 0 or pd.isna(ann_vol) else ann_return / ann_vol
    wealth = (1.0 + clean).cumprod()
    worst_idx = clean.idxmin()
    best_idx = clean.idxmax()
    return {
        "n_obs": int(clean.count()),
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown(wealth),
        "worst_month": monthly_returns.loc[worst_idx, "date"].strftime("%Y-%m-%d"),
        "best_month": monthly_returns.loc[best_idx, "date"].strftime("%Y-%m-%d"),
        "positive_month_ratio": float((clean > 0).mean()),
        "cumulative_return_within_regime": float(wealth.iloc[-1] - 1.0),
        "average_monthly_return": mean_monthly,
    }


def regime_performance(monthly_returns: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    merged = monthly_returns.merge(regimes[["date", "regime", "regime_name"]], on="date", how="left")
    rows = []
    for strategy in STRATEGIES:
        for regime_name in REGIME_ORDER:
            subset = merged.loc[merged["regime_name"] == regime_name, ["date", strategy]].copy()
            rows.append({"strategy": strategy, "regime_name": regime_name, **summarize_regime_series(subset, strategy)})
    return pd.DataFrame(rows), merged


def create_pivot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return df.pivot(index="strategy", columns="regime_name", values=metric).reindex(index=STRATEGIES, columns=REGIME_ORDER)


def plot_heatmap(df: pd.DataFrame, title: str, path: Path, fmt: str = "{:.2f}", vmin: float | None = None, vmax: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(1.8 + 1.6 * len(df.columns), 1.4 + 0.7 * len(df.index)))
    mat = df.to_numpy(dtype=float)
    im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", vmin=vmin, vmax=vmax)
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


def plot_wealth_curves(daily_equity: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> None:
    for path, logy in [(WEALTH_CURVE_PATH, False), (WEALTH_CURVE_LOG_PATH, True)]:
        fig, ax = plt.subplots(figsize=(11, 6))
        for strategy in STRATEGIES:
            ax.plot(daily_equity["date"], daily_equity[strategy], label=strategy)
        if logy:
            ax.set_yscale("log")
        ax.set_title("QUAL vs SPY Same-Start Buy-and-Hold Wealth Curve")
        ax.set_xlabel(f"Common sample: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
        ax.set_ylabel("Wealth Index")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)


def compute_relative_performance(monthly_returns: pd.DataFrame, merged_regimes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rel = pd.DataFrame({"date": monthly_returns["date"]})
    rel["QUAL_MINUS_SPY"] = monthly_returns["QUAL_ONLY"] - monthly_returns["SPY_ONLY"]
    rel["MIX_50_50_MINUS_SPY"] = monthly_returns["SPY_QUAL_50_50"] - monthly_returns["SPY_ONLY"]

    def summarize(series: pd.Series, benchmark: pd.Series) -> dict[str, object]:
        s = series.dropna()
        if s.empty:
            return {
                "average_monthly_relative_return": np.nan,
                "annualized_relative_return": np.nan,
                "relative_hit_ratio": np.nan,
                "tracking_error": np.nan,
                "information_ratio": np.nan,
                "max_relative_drawdown": np.nan,
                "number_of_outperforming_months": 0,
                "percent_outperforming_months": np.nan,
            }
        mean_monthly = float(s.mean())
        te = float(s.std() * np.sqrt(12.0))
        ir = np.nan if te == 0 or pd.isna(te) else ((1.0 + mean_monthly) ** 12 - 1.0) / te
        wealth = (1.0 + s).cumprod()
        return {
            "average_monthly_relative_return": mean_monthly,
            "annualized_relative_return": float((1.0 + mean_monthly) ** 12 - 1.0),
            "relative_hit_ratio": float((s > 0).mean()),
            "tracking_error": te,
            "information_ratio": ir,
            "max_relative_drawdown": max_drawdown(wealth),
            "number_of_outperforming_months": int((s > 0).sum()),
            "percent_outperforming_months": float((s > 0).mean()),
        }

    full_rows = []
    regime_rows = []
    merged = rel.merge(merged_regimes[["date", "regime_name"]], on="date", how="left")
    for name in ["QUAL_MINUS_SPY", "MIX_50_50_MINUS_SPY"]:
        full_rows.append({"relative_series": name, **summarize(rel[name], monthly_returns["SPY_ONLY"])})
        for regime_name in REGIME_ORDER:
            subset = merged.loc[merged["regime_name"] == regime_name, name]
            regime_rows.append({"relative_series": name, "regime_name": regime_name, **summarize(subset, subset)})
    return rel, pd.DataFrame(full_rows), pd.DataFrame(regime_rows)


def drawdown_series_and_episodes(daily_equity: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dd_df = pd.DataFrame({"date": daily_equity["date"]})
    episodes = []
    for strategy in STRATEGIES:
        wealth = daily_equity[strategy]
        running_max = wealth.cummax()
        dd = wealth / running_max - 1.0
        dd_df[strategy] = dd

        in_drawdown = False
        peak_idx = trough_idx = None
        for i in range(len(dd)):
            if not in_drawdown and dd.iloc[i] < 0:
                in_drawdown = True
                peak_idx = i - 1 if i > 0 else i
                trough_idx = i
            elif in_drawdown:
                if dd.iloc[i] < dd.iloc[trough_idx]:
                    trough_idx = i
                if dd.iloc[i] >= 0:
                    recovery_idx = i
                    episodes.append(
                        {
                            "strategy": strategy,
                            "peak_date": daily_equity.iloc[peak_idx]["date"].strftime("%Y-%m-%d"),
                            "trough_date": daily_equity.iloc[trough_idx]["date"].strftime("%Y-%m-%d"),
                            "recovery_date": daily_equity.iloc[recovery_idx]["date"].strftime("%Y-%m-%d"),
                            "drawdown_depth": float(dd.iloc[trough_idx]),
                            "duration_days": int((daily_equity.iloc[trough_idx]["date"] - daily_equity.iloc[peak_idx]["date"]).days),
                            "recovery_days": int((daily_equity.iloc[recovery_idx]["date"] - daily_equity.iloc[trough_idx]["date"]).days),
                        }
                    )
                    in_drawdown = False
                    peak_idx = trough_idx = None
        if in_drawdown and peak_idx is not None and trough_idx is not None:
            episodes.append(
                {
                    "strategy": strategy,
                    "peak_date": daily_equity.iloc[peak_idx]["date"].strftime("%Y-%m-%d"),
                    "trough_date": daily_equity.iloc[trough_idx]["date"].strftime("%Y-%m-%d"),
                    "recovery_date": "",
                    "drawdown_depth": float(dd.iloc[trough_idx]),
                    "duration_days": int((daily_equity.iloc[trough_idx]["date"] - daily_equity.iloc[peak_idx]["date"]).days),
                    "recovery_days": np.nan,
                }
            )
    episodes_df = pd.DataFrame(episodes).sort_values(["strategy", "drawdown_depth"]).groupby("strategy").head(10).reset_index(drop=True)
    return dd_df, episodes_df


def plot_drawdowns(dd_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for strategy in STRATEGIES:
        ax.plot(dd_df["date"], dd_df[strategy], label=strategy)
    ax.set_title("SPY vs QUAL Drawdown Curves")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(DRAWDOWN_CURVES_PATH, dpi=180)
    plt.close(fig)


def downside_protection(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    spy = monthly_returns["SPY_ONLY"]
    bottom_cutoff = spy.quantile(0.1)
    neg = monthly_returns.loc[spy < 0]
    bottom = monthly_returns.loc[spy <= bottom_cutoff]
    rows = []
    for strategy in STRATEGIES:
        neg_s = neg[strategy].dropna()
        bottom_s = bottom[strategy].dropna()
        rows.append(
            {
                "strategy": strategy,
                "average_return_when_spy_negative": float(neg_s.mean()) if not neg_s.empty else np.nan,
                "average_excess_return_over_spy_when_spy_negative": float((neg[strategy] - neg["SPY_ONLY"]).mean()) if not neg.empty else np.nan,
                "hit_ratio_when_spy_negative": float((neg_s > 0).mean()) if not neg_s.empty else np.nan,
                "average_return_during_spy_worst_10pct": float(bottom_s.mean()) if not bottom_s.empty else np.nan,
                "average_excess_return_over_spy_during_worst_10pct": float((bottom[strategy] - bottom["SPY_ONLY"]).mean()) if not bottom.empty else np.nan,
                "hit_ratio_during_spy_worst_10pct": float((bottom_s > 0).mean()) if not bottom_s.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def fit_beta_alpha(monthly_returns: pd.DataFrame, factor_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = monthly_returns.merge(factor_panel[["date", "RF_MONTHLY"]], on="date", how="left")
    rows = []
    capture_rows = []
    for strategy in ["QUAL_ONLY", "SPY_QUAL_50_50"]:
        sample = merged[["date", strategy, "SPY_ONLY", "RF_MONTHLY"]].dropna().copy()
        y = sample[strategy] - sample["RF_MONTHLY"]
        x = (sample["SPY_ONLY"] - sample["RF_MONTHLY"]).rename("SPY_EXCESS")
        X = sm.add_constant(x)
        result = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        rows.append(
            {
                "strategy": strategy,
                "alpha": result.params.get("const", np.nan),
                "alpha_tstat": result.tvalues.get("const", np.nan),
                "beta": result.params.get("SPY_EXCESS", np.nan),
                "beta_tstat": result.tvalues.get("SPY_EXCESS", np.nan),
                "R2": result.rsquared,
                "adj_R2": result.rsquared_adj,
                "n_obs": int(result.nobs),
                "start_date": sample["date"].min().strftime("%Y-%m-%d"),
                "end_date": sample["date"].max().strftime("%Y-%m-%d"),
            }
        )
        up = sample.loc[sample["SPY_ONLY"] > 0]
        down = sample.loc[sample["SPY_ONLY"] < 0]
        def beta_on(sub: pd.DataFrame) -> float:
            if len(sub) < 6:
                return np.nan
            x_sub = (sub["SPY_ONLY"] - sub["RF_MONTHLY"]).rename("SPY_EXCESS")
            r = sm.OLS(sub[strategy] - sub["RF_MONTHLY"], sm.add_constant(x_sub)).fit()
            return float(r.params.get("SPY_EXCESS", np.nan))
        upside_capture = np.nan if up["SPY_ONLY"].mean() == 0 or up.empty else float(up[strategy].mean() / up["SPY_ONLY"].mean())
        downside_capture = np.nan if down["SPY_ONLY"].mean() == 0 or down.empty else float(down[strategy].mean() / down["SPY_ONLY"].mean())
        capture_rows.append(
            {
                "strategy": strategy,
                "upside_beta": beta_on(up),
                "downside_beta": beta_on(down),
                "upside_capture_ratio": upside_capture,
                "downside_capture_ratio": downside_capture,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(capture_rows)


def plot_relative_curves(rel_df: pd.DataFrame) -> None:
    for col, path, title in [
        ("QUAL_MINUS_SPY", QUAL_MINUS_SPY_PATH, "QUAL minus SPY Cumulative Relative Return"),
        ("MIX_50_50_MINUS_SPY", MIX_MINUS_SPY_PATH, "50/50 SPY+QUAL minus SPY Cumulative Relative Return"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(rel_df["date"], (1.0 + rel_df[col]).cumprod(), label=col)
        ax.axhline(1.0, color="#333333", linewidth=0.8)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)


def plot_regime_profiles(pivot: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(REGIME_ORDER))
    width = 0.25
    for i, strategy in enumerate(STRATEGIES):
        ax.bar(x + (i - 1) * width, pivot.loc[strategy].values, width=width, label=strategy)
    ax.set_xticks(x)
    ax.set_xticklabels(REGIME_ORDER, rotation=25, ha="right")
    ax.set_title("SPY vs QUAL Regime Annualized Returns")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(REGIME_PROFILE_PATH, dpi=180)
    plt.close(fig)


def write_report(full_perf: pd.DataFrame, regime_perf: pd.DataFrame, rel_full: pd.DataFrame, regressions: pd.DataFrame, captures: pd.DataFrame, common_start: pd.Timestamp, common_end: pd.Timestamp) -> str:
    full_by_sharpe = full_perf.sort_values("sharpe_ratio", ascending=False)
    full_by_dd = full_perf.sort_values("max_drawdown", ascending=False)
    qual_reg = regressions.loc[regressions["strategy"] == "QUAL_ONLY"].iloc[0]
    qual_cap = captures.loc[captures["strategy"] == "QUAL_ONLY"].iloc[0]
    best_by_regime = regime_perf.sort_values(["regime_name", "sharpe_ratio"], ascending=[True, False]).groupby("regime_name").head(1)[["regime_name", "strategy"]]

    if pd.notna(qual_reg["alpha"]) and pd.notna(qual_reg["beta"]) and qual_reg["beta"] <= 1.05 and qual_reg["alpha"] > 0 and full_by_sharpe.iloc[0]["strategy"] == "QUAL_ONLY" and full_perf.loc[full_perf["strategy"] == "QUAL_ONLY", "max_drawdown"].iloc[0] >= full_perf.loc[full_perf["strategy"] == "SPY_ONLY", "max_drawdown"].iloc[0]:
        recommendation = "Replace SPY with QUAL"
    elif full_by_sharpe.iloc[0]["strategy"] == "SPY_QUAL_50_50" or qual_cap["downside_capture_ratio"] < 1:
        recommendation = "Use 50/50 SPY + QUAL"
    elif qual_reg["alpha"] <= 0 and qual_reg["beta"] > 1:
        recommendation = "Use SPY as equity sleeve"
    else:
        recommendation = "Keep QUAL as optional robustness check only"

    lines = [
        "# SPY vs QUAL Equity Sleeve Comparison",
        "",
        "## Why compare QUAL and SPY?",
        "",
        "- QUAL is not treated here as a separate hedge asset.",
        "- It is evaluated as a potential replacement or enhancement for the equity sleeve relative to SPY.",
        f"- All strategies use the same common sample: {common_start.strftime('%Y-%m-%d')} to {common_end.strftime('%Y-%m-%d')}, with the same initial capital of 1.0.",
        "- The 50/50 strategy is rebalanced on the first trading day of each month.",
        "",
        "## Full sample results",
        "",
        f"- Highest annualized return: {full_perf.sort_values('annualized_return', ascending=False).iloc[0]['strategy']}",
        f"- Highest Sharpe: {full_by_sharpe.iloc[0]['strategy']}",
        f"- Lowest max drawdown: {full_by_dd.iloc[0]['strategy']}",
        "",
        "## Regime results",
        "",
    ]
    for _, row in best_by_regime.iterrows():
        lines.append(f"- Best strategy in {row['regime_name']}: {row['strategy']}")
    lines.extend(
        [
            "- If High-Rate / Inflation-Pressure has no ETF coverage, the result should be treated as unavailable rather than interpreted.",
            "",
            "## Relative performance",
            "",
            f"- QUAL beta to SPY: {qual_reg['beta']:.3f}" if pd.notna(qual_reg["beta"]) else "- QUAL beta to SPY unavailable.",
            f"- QUAL alpha after controlling for SPY: {qual_reg['alpha']:.4f}" if pd.notna(qual_reg["alpha"]) else "- QUAL alpha unavailable.",
            f"- QUAL downside beta: {qual_cap['downside_beta']:.3f}" if pd.notna(qual_cap["downside_beta"]) else "- QUAL downside beta unavailable.",
            f"- QUAL downside capture ratio: {qual_cap['downside_capture_ratio']:.3f}" if pd.notna(qual_cap["downside_capture_ratio"]) else "- QUAL downside capture ratio unavailable.",
            "",
            "## Recommendation",
            "",
            f"- Recommendation: {recommendation}",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return recommendation


def main() -> None:
    ensure_dirs()
    daily_close, daily_ret, monthly_close, monthly_ret, factor_panel = load_inputs()
    coverage, common_start, common_end = build_coverage_report(daily_close, daily_ret, monthly_close, monthly_ret)
    coverage.to_csv(COVERAGE_PATH, index=False)

    daily_strategy_returns, daily_equity = construct_daily_strategies(daily_ret, common_start, common_end)
    monthly_strategy_returns, monthly_equity = derive_monthly_from_daily(daily_strategy_returns, daily_equity)

    daily_strategy_returns.to_csv(DAILY_STRAT_RET_PATH, index=False)
    monthly_strategy_returns.to_csv(MONTHLY_STRAT_RET_PATH, index=False)
    daily_equity.to_csv(DAILY_EQUITY_PATH, index=False)
    monthly_equity.to_csv(MONTHLY_EQUITY_PATH, index=False)

    plot_wealth_curves(daily_equity, common_start, common_end)

    full_perf = full_sample_performance(daily_equity, daily_strategy_returns, monthly_strategy_returns)
    full_perf.to_csv(FULL_PERF_PATH, index=False)

    regime_perf, merged_regimes = regime_performance(monthly_strategy_returns, factor_panel)
    regime_perf.to_csv(REGIME_PERF_PATH, index=False)
    ann_pivot = create_pivot(regime_perf, "annualized_return")
    sharpe_pivot = create_pivot(regime_perf, "sharpe_ratio")
    dd_pivot = create_pivot(regime_perf, "max_drawdown")
    pos_pivot = create_pivot(regime_perf, "positive_month_ratio")
    ann_pivot.to_csv(REGIME_ANN_RETURN_PATH)
    sharpe_pivot.to_csv(REGIME_SHARPE_PATH)
    dd_pivot.to_csv(REGIME_MAX_DD_PATH)
    pos_pivot.to_csv(REGIME_POS_PATH)

    plot_heatmap(ann_pivot, "SPY vs QUAL Annualized Return by Regime", ANN_RETURN_HEATMAP_PATH, fmt="{:.2%}")
    plot_heatmap(sharpe_pivot, "SPY vs QUAL Sharpe by Regime", SHARPE_HEATMAP_PATH, fmt="{:.2f}")
    plot_heatmap(dd_pivot, "SPY vs QUAL Max Drawdown by Regime", MAX_DD_HEATMAP_PATH, fmt="{:.2%}")
    plot_regime_profiles(ann_pivot)

    rel_df, rel_full, rel_regime = compute_relative_performance(monthly_strategy_returns, merged_regimes)
    rel_full.to_csv(REL_FULL_PATH, index=False)
    rel_regime.to_csv(REL_REGIME_PATH, index=False)
    rel_heat = rel_regime.pivot(index="relative_series", columns="regime_name", values="annualized_relative_return").reindex(columns=REGIME_ORDER)
    plot_heatmap(rel_heat, "Relative Return by Regime", REL_HEATMAP_PATH, fmt="{:.2%}")
    plot_relative_curves(rel_df)

    dd_series, dd_episodes = drawdown_series_and_episodes(daily_equity)
    dd_series.to_csv(DRAWDOWN_SERIES_PATH, index=False)
    dd_episodes.to_csv(TOP_DD_EPISODES_PATH, index=False)
    plot_drawdowns(dd_series)

    downside = downside_protection(monthly_strategy_returns)
    downside.to_csv(DOWNSIDE_PATH, index=False)

    regressions, captures = fit_beta_alpha(monthly_strategy_returns, factor_panel)
    regressions.to_csv(REGRESSION_PATH, index=False)
    captures.to_csv(CAPTURE_PATH, index=False)

    recommendation = write_report(full_perf, regime_perf, rel_full, regressions, captures, common_start, common_end)

    sharpe_rank = ", ".join(full_perf.sort_values("sharpe_ratio", ascending=False)["strategy"].tolist())
    dd_rank = ", ".join(full_perf.sort_values("max_drawdown", ascending=False)["strategy"].tolist())
    qual_reg = regressions.loc[regressions["strategy"] == "QUAL_ONLY"].iloc[0]
    qual_cap = captures.loc[captures["strategy"] == "QUAL_ONLY"].iloc[0]
    best_by_regime = regime_perf.sort_values(["regime_name", "sharpe_ratio"], ascending=[True, False]).groupby("regime_name").head(1)[["regime_name", "strategy"]]

    print(f"Common start date: {common_start.strftime('%Y-%m-%d')}")
    print(f"Common end date: {common_end.strftime('%Y-%m-%d')}")
    print(f"Full-sample Sharpe ranking: {sharpe_rank}")
    print(f"Full-sample max drawdown ranking: {dd_rank}")
    print(f"QUAL beta to SPY: {qual_reg['beta']:.4f}" if pd.notna(qual_reg["beta"]) else "QUAL beta to SPY: NaN")
    print(f"QUAL alpha: {qual_reg['alpha']:.6f}" if pd.notna(qual_reg["alpha"]) else "QUAL alpha: NaN")
    print(f"QUAL downside capture ratio: {qual_cap['downside_capture_ratio']:.4f}" if pd.notna(qual_cap["downside_capture_ratio"]) else "QUAL downside capture ratio: NaN")
    print("Best strategy by regime:")
    print(best_by_regime.to_string(index=False))
    print(f"Recommended equity sleeve choice: {recommendation}")

    for path in [
        COVERAGE_PATH,
        DAILY_STRAT_RET_PATH,
        MONTHLY_STRAT_RET_PATH,
        DAILY_EQUITY_PATH,
        MONTHLY_EQUITY_PATH,
        FULL_PERF_PATH,
        REGIME_PERF_PATH,
        REGIME_ANN_RETURN_PATH,
        REGIME_SHARPE_PATH,
        REGIME_MAX_DD_PATH,
        REGIME_POS_PATH,
        REL_FULL_PATH,
        REL_REGIME_PATH,
        DRAWDOWN_SERIES_PATH,
        TOP_DD_EPISODES_PATH,
        DOWNSIDE_PATH,
        REGRESSION_PATH,
        CAPTURE_PATH,
        REPORT_PATH,
        WEALTH_CURVE_PATH,
        WEALTH_CURVE_LOG_PATH,
        ANN_RETURN_HEATMAP_PATH,
        SHARPE_HEATMAP_PATH,
        MAX_DD_HEATMAP_PATH,
        REGIME_PROFILE_PATH,
        QUAL_MINUS_SPY_PATH,
        MIX_MINUS_SPY_PATH,
        REL_HEATMAP_PATH,
        DRAWDOWN_CURVES_PATH,
    ]:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
