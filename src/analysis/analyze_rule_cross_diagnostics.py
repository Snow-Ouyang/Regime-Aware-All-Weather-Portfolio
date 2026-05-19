from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
HF_PANEL_PATH = ROOT / "results" / "high_frequency_regime_diagnostics" / "high_frequency_regime_feature_panel.csv"
CORE_PANEL_PATH = ROOT / "data" / "processed" / "risk_factors" / "core_risk_factor_panel.csv"
MONTHLY_RET_PATH = ROOT / "data" / "processed" / "assets" / "monthly_returns.csv"
DAILY_CLOSE_PATH = ROOT / "data" / "processed" / "assets" / "daily_adjusted_close.csv"

RESULTS_DIR = ROOT / "results" / "rule_diagnostics"
FIGURES_DIR = ROOT / "figures" / "rule_diagnostics"

RULE_STATE_PANEL_PATH = RESULTS_DIR / "rule_state_panel.csv"
CURVE_VIX_LEVEL_PATH = RESULTS_DIR / "spy_performance_by_curve_and_vix_level.csv"
CURVE_VIX_MAX_PATH = RESULTS_DIR / "spy_performance_by_curve_and_vix_max.csv"
CURVE_CREDIT_PATH = RESULTS_DIR / "spy_performance_by_curve_and_credit.csv"
CURVE_ALT_CREDIT_PATH = RESULTS_DIR / "spy_performance_by_curve_alt_and_credit.csv"
INVERSION_SUMMARY_PATH = RESULTS_DIR / "inversion_performance_summary.csv"
INVERSION_CROSS_PATH = RESULTS_DIR / "inversion_cross_conditions.csv"
HIGH_RATE_RULE_PATH = RESULTS_DIR / "high_rate_inflation_rule_validation.csv"
FLAG_DIAG_PATH = RESULTS_DIR / "stress_recovery_flag_diagnostics.csv"
REPORT_PATH = RESULTS_DIR / "RULE_CROSS_DIAGNOSTICS.md"

REGIME_ORDER = [
    "Late-Cycle / Inflationary Flat Curve",
    "Low-Rate / Steep Curve",
    "High-Rate / Inflation-Pressure",
    "Deflationary Macro-Financial Stress",
]


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def to_month_end(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.to_period("M").dt.to_timestamp("M")


def load_monthly_rule_inputs() -> pd.DataFrame:
    core = pd.read_csv(CORE_PANEL_PATH)
    core["date"] = to_month_end(pd.to_datetime(core["date"]))
    core = core[[
        "date",
        "regime",
        "regime_name",
        "RF_MONTHLY",
    ]].drop_duplicates("date", keep="last")

    monthly_ret = pd.read_csv(MONTHLY_RET_PATH)
    monthly_ret["date"] = to_month_end(pd.to_datetime(monthly_ret["date"]))
    monthly_ret = monthly_ret.rename(columns={"SPY": "SPY_RETURN", "IEF": "IEF_RETURN", "GLD": "GLD_RETURN", "BIL": "BIL_RETURN", "SHY": "SHY_RETURN"})

    if HF_PANEL_PATH.exists():
        hf = pd.read_csv(HF_PANEL_PATH)
        hf["date"] = pd.to_datetime(hf["date"])
        if {"feature", "value"}.issubset(hf.columns):
            monthly_features = {}
            for feature in ["CREDIT_SPREAD_BAA_AAA", "DGS10", "DGS1", "TERM_SPREAD_10Y_1Y", "VIX_LEVEL", "VIX_MAX_MONTH", "SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"]:
                subset = hf.loc[hf["feature"] == feature, ["date", "value"]].copy()
                if subset.empty:
                    continue
                subset["month"] = to_month_end(subset["date"])
                if feature == "VIX_LEVEL":
                    subset = subset.sort_values("date").groupby("month")["value"].last().reset_index()
                else:
                    subset = subset.sort_values("date").groupby("month")["value"].last().reset_index()
                subset = subset.rename(columns={"month": "date", "value": feature})
                monthly_features[feature] = subset
            merged_hf = None
            for df in monthly_features.values():
                merged_hf = df if merged_hf is None else merged_hf.merge(df, on="date", how="outer")
        else:
            raise ValueError("High-frequency panel format not recognized.")
    else:
        raise FileNotFoundError("High-frequency regime feature panel not found.")

    panel = core.merge(merged_hf, on="date", how="left").merge(monthly_ret, on="date", how="left")
    panel["D_DGS10"] = panel["DGS10"].diff()
    panel["D_DGS1"] = panel["DGS1"].diff()
    panel["D_CREDIT_SPREAD_BAA_AAA"] = panel["CREDIT_SPREAD_BAA_AAA"].diff()
    panel["D_TERM_SPREAD_10Y_1Y"] = panel["TERM_SPREAD_10Y_1Y"].diff()
    return panel.sort_values("date").reset_index(drop=True)


def assign_rule_states(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["curve_state"] = np.select(
        [out["TERM_SPREAD_10Y_1Y"] < 0, out["TERM_SPREAD_10Y_1Y"] < 1],
        ["INVERTED", "FLAT"],
        default="STEEP",
    )
    out["curve_state_alt"] = np.select(
        [
            out["TERM_SPREAD_10Y_1Y"] < -0.5,
            out["TERM_SPREAD_10Y_1Y"] < 0,
            out["TERM_SPREAD_10Y_1Y"] < 1,
        ],
        ["DEEPLY_INVERTED", "MILDLY_INVERTED", "FLAT"],
        default="STEEP",
    )
    out["vix_level_bucket"] = pd.cut(
        out["VIX_LEVEL"],
        bins=[-np.inf, 20, 25, 30, np.inf],
        labels=["LOW", "WARNING", "PRE_STRESS", "STRESS"],
        right=False,
    )
    out["vix_max_bucket"] = pd.cut(
        out["VIX_MAX_MONTH"],
        bins=[-np.inf, 20, 25, 30, np.inf],
        labels=["LOW", "WARNING", "PRE_STRESS", "STRESS"],
        right=False,
    )
    out["credit_bucket"] = pd.cut(
        out["CREDIT_SPREAD_BAA_AAA"],
        bins=[-np.inf, 1.0, 1.5, np.inf],
        labels=["NORMAL", "ELEVATED", "STRESS"],
        right=False,
    )
    out["short_rate_bucket"] = pd.cut(
        out["DGS1"],
        bins=[-np.inf, 2, 5, np.inf],
        labels=["LOW_RATE", "NORMAL_RATE", "HIGH_RATE"],
        right=False,
    )
    out["high_rate_inflation_flag"] = (out["CREDIT_SPREAD_BAA_AAA"] > 1.5) & (out["DGS1"] > 5)
    out["high_rate_inflation_flag_broad"] = (out["CREDIT_SPREAD_BAA_AAA"] > 1.3) & (out["DGS1"] > 4.5)
    out["stress_flag"] = (
        (out["VIX_LEVEL"] >= 25).astype(int)
        + (out["CREDIT_SPREAD_BAA_AAA"] >= 1.5).astype(int)
        + (out["SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"] <= -0.10).astype(int)
    ) >= 2
    out["crisis_flag"] = (
        (out["VIX_LEVEL"] >= 30)
        | (out["CREDIT_SPREAD_BAA_AAA"] >= 1.8)
        | (out["SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"] <= -0.20)
    )
    out["recovery_watch_flag"] = (
        (out["SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"] <= -0.10)
        & ((out["VIX_LEVEL"] < 25) | (out["D_CREDIT_SPREAD_BAA_AAA"] < 0))
    )
    return out


def annualized_return(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    return float((1.0 + s.mean()) ** 12 - 1.0)


def annualized_vol(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 2:
        return np.nan
    return float(s.std() * np.sqrt(12))


def max_drawdown_from_returns(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    wealth = (1.0 + s).cumprod()
    dd = wealth / wealth.cummax() - 1.0
    return float(dd.min())


def performance_summary(series: pd.Series, drawdown_series: pd.Series | None = None) -> dict[str, float]:
    s = series.dropna()
    if s.empty:
        return {
            "n_obs": 0,
            "average_monthly_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "Sharpe": np.nan,
            "max_drawdown": np.nan,
            "worst_month": np.nan,
            "best_month": np.nan,
            "positive_month_ratio": np.nan,
        }
    ann_ret = annualized_return(s)
    ann_vol = annualized_vol(s)
    return {
        "n_obs": int(len(s)),
        "average_monthly_return": float(s.mean()),
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "Sharpe": float(ann_ret / ann_vol) if pd.notna(ann_ret) and pd.notna(ann_vol) and ann_vol != 0 else np.nan,
        "max_drawdown": max_drawdown_from_returns(s),
        "worst_month": float(s.min()),
        "best_month": float(s.max()),
        "positive_month_ratio": float((s > 0).mean()),
    }


def grouped_spy_performance(panel: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, grp in panel.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(performance_summary(grp["SPY_RETURN"]))
        row["average_SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"] = float(grp["SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"].mean()) if grp["SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"].notna().any() else np.nan
        row["median_SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"] = float(grp["SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"].median()) if grp["SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"].notna().any() else np.nan
        row["p10_SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"] = float(grp["SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"].quantile(0.10)) if grp["SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"].notna().any() else np.nan
        row["p05_SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"] = float(grp["SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"].quantile(0.05)) if grp["SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"].notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_heatmap(df: pd.DataFrame, index: str, columns: str, values: str, path: Path, title: str, fmt: str = ".2f") -> None:
    pivot = df.pivot(index=index, columns=columns, values=values)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap="RdYlBu_r", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def inversion_summary(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = {
        "NON_INVERTED": panel["TERM_SPREAD_10Y_1Y"] >= 0,
        "INVERTED": panel["TERM_SPREAD_10Y_1Y"] < 0,
        "DEEPLY_INVERTED": panel["TERM_SPREAD_10Y_1Y"] < -0.5,
    }
    assets = {
        "SPY": "SPY_RETURN",
        "IEF": "IEF_RETURN",
        "GOLD": "GLD_RETURN",
        "CASH": "RF_MONTHLY",
    }
    rows = []
    for group_name, mask in groups.items():
        grp = panel.loc[mask].copy()
        for asset_name, col in assets.items():
            row = {"group": group_name, "asset": asset_name}
            row.update(performance_summary(grp[col]))
            row["average_DGS1"] = float(grp["DGS1"].mean()) if grp["DGS1"].notna().any() else np.nan
            row["average_DGS10"] = float(grp["DGS10"].mean()) if grp["DGS10"].notna().any() else np.nan
            row["average_CREDIT_SPREAD_BAA_AAA"] = float(grp["CREDIT_SPREAD_BAA_AAA"].mean()) if grp["CREDIT_SPREAD_BAA_AAA"].notna().any() else np.nan
            row["average_VIX_LEVEL"] = float(grp["VIX_LEVEL"].mean()) if grp["VIX_LEVEL"].notna().any() else np.nan
            row["average_SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"] = float(grp["SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"].mean()) if grp["SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"].notna().any() else np.nan
            rows.append(row)
    condition_groups = {
        "Inverted + VIX low": (panel["TERM_SPREAD_10Y_1Y"] < 0) & (panel["VIX_LEVEL"] < 25),
        "Inverted + VIX high": (panel["TERM_SPREAD_10Y_1Y"] < 0) & (panel["VIX_LEVEL"] >= 25),
        "Inverted + credit normal": (panel["TERM_SPREAD_10Y_1Y"] < 0) & (panel["CREDIT_SPREAD_BAA_AAA"] < 1.5),
        "Inverted + credit stress": (panel["TERM_SPREAD_10Y_1Y"] < 0) & (panel["CREDIT_SPREAD_BAA_AAA"] >= 1.5),
        "Inverted + high short rate": (panel["TERM_SPREAD_10Y_1Y"] < 0) & (panel["DGS1"] >= 5),
    }
    cross_rows = []
    for name, mask in condition_groups.items():
        grp = panel.loc[mask]
        cross_rows.append({
            "condition": name,
            "n_obs": int(mask.sum()),
            "SPY_average_return": float(grp["SPY_RETURN"].mean()) if grp["SPY_RETURN"].notna().any() else np.nan,
            "IEF_average_return": float(grp["IEF_RETURN"].mean()) if grp["IEF_RETURN"].notna().any() else np.nan,
            "GOLD_average_return": float(grp["GLD_RETURN"].mean()) if grp["GLD_RETURN"].notna().any() else np.nan,
            "CASH_average_return": float(grp["RF_MONTHLY"].mean()) if grp["RF_MONTHLY"].notna().any() else np.nan,
            "SPY_drawdown_average": float(grp["SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"].mean()) if grp["SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"].notna().any() else np.nan,
        })
    return pd.DataFrame(rows), pd.DataFrame(cross_rows)


def high_rate_rule_validation(panel: pd.DataFrame) -> pd.DataFrame:
    assets = {"SPY": "SPY_RETURN", "IEF": "IEF_RETURN", "GOLD": "GLD_RETURN", "CASH": "RF_MONTHLY"}
    masks = {
        "high_rate_inflation_flag": panel["high_rate_inflation_flag"],
        "all_other_months": ~panel["high_rate_inflation_flag"],
        "flat_curve_months": panel["curve_state"] == "FLAT",
        "inverted_curve_months": panel["curve_state"] == "INVERTED",
        "cluster_high_rate_inflation": panel["regime_name"] == "High-Rate / Inflation-Pressure",
    }
    rows = []
    for label, mask in masks.items():
        grp = panel.loc[mask]
        for asset_name, col in assets.items():
            row = {"group": label, "asset": asset_name}
            row.update(performance_summary(grp[col]))
            rows.append(row)
    return pd.DataFrame(rows)


def overlap_table(panel: pd.DataFrame) -> pd.DataFrame:
    overlap = pd.crosstab(panel["high_rate_inflation_flag"], panel["regime_name"])
    return overlap.reset_index()


def stress_recovery_diagnostics(panel: pd.DataFrame) -> pd.DataFrame:
    flags = ["stress_flag", "crisis_flag", "recovery_watch_flag"]
    rows = []
    panel = panel.copy()
    panel["SPY_NEXT_1M"] = panel["SPY_RETURN"].shift(-1)
    panel["SPY_NEXT_3M"] = (
        (1 + panel["SPY_RETURN"].shift(-1))
        * (1 + panel["SPY_RETURN"].shift(-2))
        * (1 + panel["SPY_RETURN"].shift(-3))
        - 1
    )
    for flag in flags:
        grp = panel.loc[panel[flag]]
        rows.append({
            "flag": flag,
            "months": int(len(grp)),
            "average_SPY_same_month_return": float(grp["SPY_RETURN"].mean()) if grp["SPY_RETURN"].notna().any() else np.nan,
            "average_SPY_next_month_return": float(grp["SPY_NEXT_1M"].mean()) if grp["SPY_NEXT_1M"].notna().any() else np.nan,
            "average_SPY_next_3_month_return": float(grp["SPY_NEXT_3M"].mean()) if grp["SPY_NEXT_3M"].notna().any() else np.nan,
            "max_drawdown": max_drawdown_from_returns(grp["SPY_RETURN"]),
            "positive_next_month_ratio": float((grp["SPY_NEXT_1M"] > 0).mean()) if grp["SPY_NEXT_1M"].notna().any() else np.nan,
            "average_IEF_return": float(grp["IEF_RETURN"].mean()) if grp["IEF_RETURN"].notna().any() else np.nan,
            "average_GOLD_return": float(grp["GLD_RETURN"].mean()) if grp["GLD_RETURN"].notna().any() else np.nan,
            "average_CASH_return": float(grp["RF_MONTHLY"].mean()) if grp["RF_MONTHLY"].notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def write_report(panel: pd.DataFrame, curve_vix_level: pd.DataFrame, curve_credit: pd.DataFrame, inversion: pd.DataFrame, high_rate: pd.DataFrame, flags: pd.DataFrame) -> None:
    steep_low = curve_vix_level.loc[(curve_vix_level["curve_state"] == "STEEP") & (curve_vix_level["vix_level_bucket"] == "LOW")]
    steep_high = curve_vix_level.loc[(curve_vix_level["curve_state"] == "STEEP") & (curve_vix_level["vix_level_bucket"].isin(["PRE_STRESS", "STRESS"]))]
    inv_spy = inversion.loc[(inversion["group"] == "INVERTED") & (inversion["asset"] == "SPY")]
    high_rate_cash = high_rate.loc[(high_rate["group"] == "high_rate_inflation_flag") & (high_rate["asset"] == "CASH")]
    lines = [
        "# Rule Cross Diagnostics",
        "",
        "## Current rule hypotheses",
        "",
        "- High-rate inflation: credit spread > 1.5 and DGS1 > 5",
        "- Steep curve: term spread > 1",
        "- Flat curve: term spread between 0 and 1",
        "- Inverted curve: term spread < 0",
        "- VIX 25 as warning line",
        "- VIX 30 as stress line",
        "",
        "## Findings from curve x VIX",
        "",
        f"- Steep + low VIX average annualized SPY return: {steep_low['annualized_return'].iloc[0]:.2%}" if not steep_low.empty and pd.notna(steep_low['annualized_return'].iloc[0]) else "- Steep + low VIX sample unavailable.",
        f"- Steep + high VIX average annualized SPY return: {steep_high['annualized_return'].mean():.2%}" if not steep_high.empty and steep_high['annualized_return'].notna().any() else "- Steep + high VIX sample unavailable.",
        "- This helps check whether steep curve only behaves risk-on when volatility is calm.",
        "",
        "## Findings from curve x credit",
        "",
        "- Credit stress inside steep or flat curve is the main candidate explanation for hidden crisis/recovery months.",
        "",
        "## Inversion study",
        "",
        f"- Inverted SPY annualized return: {inv_spy['annualized_return'].iloc[0]:.2%}" if not inv_spy.empty and pd.notna(inv_spy['annualized_return'].iloc[0]) else "- Inverted SPY return unavailable.",
        "- Inversion should be tested separately from ordinary flat curve because short-rate carry and defensive asset behavior can differ materially.",
        "",
        "## High-rate inflation validation",
        "",
        f"- Cash annualized return under high-rate inflation flag: {high_rate_cash['annualized_return'].iloc[0]:.2%}" if not high_rate_cash.empty and pd.notna(high_rate_cash['annualized_return'].iloc[0]) else "- High-rate inflation cash result unavailable.",
        "- This validates whether the credit-spread-plus-short-rate rule overlaps with the clustered High-Rate / Inflation-Pressure state.",
        "",
        "## Preliminary allocation implications",
        "",
        "- Steep curve + low stress may support higher SPY.",
        "- Steep curve + high VIX/credit/drawdown should be treated as recovery/stress, not full risk-on.",
        "- Flat curve with normal VIX may not require aggressive de-risking.",
        "- Inverted curve may deserve a separate defensive/high-cash rule.",
        "- High-rate inflation should likely override ordinary curve rules and favor cash.",
        "",
        "## Caveats",
        "",
        "- These are diagnostic rules using full historical data.",
        "- Final strategy rules must use rolling/expanding thresholds or fixed ex-ante thresholds.",
        "- No final allocation decisions should be made from these diagnostics alone.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def plot_bar(df: pd.DataFrame, x: str, y: str, hue: str | None, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    if hue:
        sns.barplot(data=df, x=x, y=y, hue=hue, ax=ax)
    else:
        sns.barplot(data=df, x=x, y=y, ax=ax)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    panel = assign_rule_states(load_monthly_rule_inputs())
    panel.to_csv(RULE_STATE_PANEL_PATH, index=False)

    curve_vix_level = grouped_spy_performance(panel.dropna(subset=["curve_state", "vix_level_bucket"]), ["curve_state", "vix_level_bucket"])
    curve_vix_max = grouped_spy_performance(panel.dropna(subset=["curve_state", "vix_max_bucket"]), ["curve_state", "vix_max_bucket"])
    curve_vix_level.to_csv(CURVE_VIX_LEVEL_PATH, index=False)
    curve_vix_max.to_csv(CURVE_VIX_MAX_PATH, index=False)

    plot_heatmap(curve_vix_level, "curve_state", "vix_level_bucket", "annualized_return", FIGURES_DIR / "spy_return_by_curve_and_vix_level.png", "SPY Return by Curve and VIX Level", ".2%")
    plot_heatmap(curve_vix_level, "curve_state", "vix_level_bucket", "Sharpe", FIGURES_DIR / "spy_sharpe_by_curve_and_vix_level.png", "SPY Sharpe by Curve and VIX Level", ".2f")
    plot_heatmap(curve_vix_level, "curve_state", "vix_level_bucket", "max_drawdown", FIGURES_DIR / "spy_max_drawdown_by_curve_and_vix_level.png", "SPY Max Drawdown by Curve and VIX Level", ".2%")
    plot_heatmap(curve_vix_max, "curve_state", "vix_max_bucket", "annualized_return", FIGURES_DIR / "spy_return_by_curve_and_vix_max.png", "SPY Return by Curve and VIX Max", ".2%")
    plot_heatmap(curve_vix_max, "curve_state", "vix_max_bucket", "Sharpe", FIGURES_DIR / "spy_sharpe_by_curve_and_vix_max.png", "SPY Sharpe by Curve and VIX Max", ".2f")
    plot_heatmap(curve_vix_max, "curve_state", "vix_max_bucket", "max_drawdown", FIGURES_DIR / "spy_max_drawdown_by_curve_and_vix_max.png", "SPY Max Drawdown by Curve and VIX Max", ".2%")

    curve_credit = grouped_spy_performance(panel.dropna(subset=["curve_state", "credit_bucket"]), ["curve_state", "credit_bucket"])
    curve_alt_credit = grouped_spy_performance(panel.dropna(subset=["curve_state_alt", "credit_bucket"]), ["curve_state_alt", "credit_bucket"])
    curve_credit.to_csv(CURVE_CREDIT_PATH, index=False)
    curve_alt_credit.to_csv(CURVE_ALT_CREDIT_PATH, index=False)
    plot_heatmap(curve_credit, "curve_state", "credit_bucket", "annualized_return", FIGURES_DIR / "spy_return_by_curve_and_credit.png", "SPY Return by Curve and Credit", ".2%")
    plot_heatmap(curve_credit, "curve_state", "credit_bucket", "Sharpe", FIGURES_DIR / "spy_sharpe_by_curve_and_credit.png", "SPY Sharpe by Curve and Credit", ".2f")
    plot_heatmap(curve_credit, "curve_state", "credit_bucket", "max_drawdown", FIGURES_DIR / "spy_max_drawdown_by_curve_and_credit.png", "SPY Max Drawdown by Curve and Credit", ".2%")

    inversion_summary_df, inversion_cross = inversion_summary(panel)
    inversion_summary_df.to_csv(INVERSION_SUMMARY_PATH, index=False)
    inversion_cross.to_csv(INVERSION_CROSS_PATH, index=False)
    plot_heatmap(inversion_summary_df, "asset", "group", "annualized_return", FIGURES_DIR / "inversion_asset_return_heatmap.png", "Inversion Asset Return Heatmap", ".2%")
    plot_heatmap(inversion_summary_df, "asset", "group", "Sharpe", FIGURES_DIR / "inversion_asset_sharpe_heatmap.png", "Inversion Asset Sharpe Heatmap", ".2f")
    plot_heatmap(inversion_summary_df, "asset", "group", "max_drawdown", FIGURES_DIR / "inversion_asset_drawdown_heatmap.png", "Inversion Asset Drawdown Heatmap", ".2%")
    plot_bar(inversion_cross, "condition", "n_obs", None, "Inversion Condition Counts", FIGURES_DIR / "inversion_condition_counts.png")

    high_rate = high_rate_rule_validation(panel)
    high_rate.to_csv(HIGH_RATE_RULE_PATH, index=False)
    overlap = overlap_table(panel)
    overlap_long = overlap.melt(id_vars="high_rate_inflation_flag", var_name="regime_name", value_name="count")
    plot_bar(high_rate.loc[high_rate["group"] == "high_rate_inflation_flag"], "asset", "annualized_return", None, "High-Rate Inflation Asset Return", FIGURES_DIR / "high_rate_inflation_asset_return_bar.png")
    plot_bar(high_rate.loc[high_rate["group"] == "high_rate_inflation_flag"], "asset", "max_drawdown", None, "High-Rate Inflation Asset Drawdown", FIGURES_DIR / "high_rate_inflation_asset_drawdown_bar.png")
    plot_bar(overlap_long, "regime_name", "count", "high_rate_inflation_flag", "High-Rate Inflation Rule Overlap with Cluster", FIGURES_DIR / "high_rate_inflation_rule_overlap_with_cluster.png")

    flags = stress_recovery_diagnostics(panel)
    flags.to_csv(FLAG_DIAG_PATH, index=False)

    write_report(panel, curve_vix_level, curve_credit, inversion_summary_df, high_rate, flags)

    print("Rule state counts:")
    for col in ["curve_state", "curve_state_alt", "vix_level_bucket", "vix_max_bucket", "credit_bucket", "short_rate_bucket"]:
        print(f"{col}:")
        print(panel[col].value_counts(dropna=False).to_string())
    print("curve_state x vix_level_bucket count table:")
    print(pd.crosstab(panel["curve_state"], panel["vix_level_bucket"]).to_string())
    print("curve_state x credit_bucket count table:")
    print(pd.crosstab(panel["curve_state"], panel["credit_bucket"]).to_string())
    print(f"Inversion month count: {(panel['TERM_SPREAD_10Y_1Y'] < 0).sum()}")
    print(f"High-rate inflation flag count: {int(panel['high_rate_inflation_flag'].sum())}")
    print("Key findings summary:")
    print("- Steep curve needs to be read jointly with VIX and credit, not as unconditional risk-on.")
    print("- Inversion is now separated from ordinary flat curve for comparison.")
    print("- High-rate inflation rule can be checked directly against the clustered regime labels.")
    for path in [
        RULE_STATE_PANEL_PATH,
        CURVE_VIX_LEVEL_PATH,
        CURVE_VIX_MAX_PATH,
        CURVE_CREDIT_PATH,
        CURVE_ALT_CREDIT_PATH,
        INVERSION_SUMMARY_PATH,
        INVERSION_CROSS_PATH,
        HIGH_RATE_RULE_PATH,
        FLAG_DIAG_PATH,
        REPORT_PATH,
        FIGURES_DIR / "spy_return_by_curve_and_vix_level.png",
        FIGURES_DIR / "spy_sharpe_by_curve_and_vix_level.png",
        FIGURES_DIR / "spy_max_drawdown_by_curve_and_vix_level.png",
        FIGURES_DIR / "spy_return_by_curve_and_vix_max.png",
        FIGURES_DIR / "spy_sharpe_by_curve_and_vix_max.png",
        FIGURES_DIR / "spy_max_drawdown_by_curve_and_vix_max.png",
        FIGURES_DIR / "spy_return_by_curve_and_credit.png",
        FIGURES_DIR / "spy_sharpe_by_curve_and_credit.png",
        FIGURES_DIR / "spy_max_drawdown_by_curve_and_credit.png",
        FIGURES_DIR / "inversion_asset_return_heatmap.png",
        FIGURES_DIR / "inversion_asset_sharpe_heatmap.png",
        FIGURES_DIR / "inversion_asset_drawdown_heatmap.png",
        FIGURES_DIR / "inversion_condition_counts.png",
        FIGURES_DIR / "high_rate_inflation_asset_return_bar.png",
        FIGURES_DIR / "high_rate_inflation_asset_drawdown_bar.png",
        FIGURES_DIR / "high_rate_inflation_rule_overlap_with_cluster.png",
    ]:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
