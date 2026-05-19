from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]

STRATEGY_PANEL_PATH = ROOT / "results" / "regime_hedge_steep_sell_ief" / "daily_backtest_panel.csv"
RULE_PANEL_PATH = ROOT / "results" / "rule_diagnostics" / "rule_state_panel.csv"
RECON_PANEL_PATH = ROOT / "results" / "reconstructed_regime_asset_behavior" / "reconstructed_regime_panel.csv"
RISK_PANEL_PATHS = [
    ROOT / "data" / "processed" / "risk_factors" / "core_risk_factor_panel.csv",
    ROOT / "data" / "processed" / "risk_factors" / "extended_risk_factor_panel.csv",
]

RESULTS_DIR = ROOT / "results" / "covid_vs_flat_regime_diagnostic"
FIGURES_DIR = ROOT / "figures" / "covid_vs_flat_regime_diagnostic"

PANEL_OUT = RESULTS_DIR / "covid_flat_comparison_panel.csv"
ASSET_PERF_OUT = RESULTS_DIR / "asset_performance_covid_vs_flat.csv"
CUMRET_PIVOT_OUT = RESULTS_DIR / "asset_cumulative_return_covid_vs_flat.csv"
MDD_PIVOT_OUT = RESULTS_DIR / "asset_max_drawdown_covid_vs_flat.csv"
SHARPE_PIVOT_OUT = RESULTS_DIR / "asset_sharpe_covid_vs_flat.csv"
CORR_PIVOT_OUT = RESULTS_DIR / "asset_correlation_with_spy_covid_vs_flat.csv"
WEIGHT_CONTRIB_OUT = RESULTS_DIR / "strategy_weight_and_contribution_covid.csv"
CONTRIB_SUMMARY_OUT = RESULTS_DIR / "strategy_contribution_summary_covid_vs_flat.csv"
MACRO_DIST_OUT = RESULTS_DIR / "macro_variable_distribution_covid_vs_flat.csv"
ZSCORE_OUT = RESULTS_DIR / "macro_variable_zscore_percentile_covid.csv"
REPORT_OUT = RESULTS_DIR / "COVID_VS_FLAT_REGIME_DIAGNOSTIC.md"

FIG_ASSET = FIGURES_DIR / "covid_asset_performance_vs_flat.png"
FIG_NAV = FIGURES_DIR / "covid_asset_nav_paths.png"
FIG_FLAT_NAV = FIGURES_DIR / "flat_ex_covid_conditional_asset_nav.png"
FIG_STRESS_BOX = FIGURES_DIR / "market_stress_variables_boxplot.png"
FIG_MACRO_BOX = FIGURES_DIR / "macro_regime_variables_boxplot.png"
FIG_TIMELINE = FIGURES_DIR / "covid_macro_timeline.png"
FIG_CONTRIB = FIGURES_DIR / "covid_strategy_weight_contribution.png"
FIG_Z = FIGURES_DIR / "covid_vs_flat_zscore_heatmap.png"

CONFIG = {
    "covid_start": "2020-02-19",
    "covid_end": "2020-04-30",
    "assets": ["SPY", "IEF", "GOLD", "CMDTY_FUT", "CASH"],
}

ASSET_RETURN_COLS = {
    "SPY": "SPY_RETURN",
    "IEF": "IEF_RETURN",
    "GOLD": "GOLD_RETURN",
    "CMDTY_FUT": "CMDTY_FUT_RETURN",
    "CASH": "RF_DAILY",
}

REGIME_COLORS = {
    "HIGH_INFLATION": "#d95f02",
    "INVERTED": "#7570b3",
    "FLAT": "#1b9e77",
    "STEEP": "#66a61e",
    "NEUTRAL": "#999999",
}


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _first_existing(cols: list[str], candidates: list[str]) -> str | None:
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def load_strategy_panel() -> pd.DataFrame:
    if not STRATEGY_PANEL_PATH.exists():
        raise FileNotFoundError(f"Missing strategy panel: {STRATEGY_PANEL_PATH}")
    panel = pd.read_csv(STRATEGY_PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])

    rename = {}
    flexible = {
        "SPY_RETURN": ["SPY_RETURN", "SPY_ret", "spy_daily_return"],
        "IEF_RETURN": ["IEF_RETURN", "IEF_ret"],
        "GOLD_RETURN": ["GOLD_RETURN", "GOLD_ret"],
        "CMDTY_FUT_RETURN": ["CMDTY_FUT_RETURN", "CMDTY_ret"],
        "RF_DAILY": ["RF_DAILY", "daily_rf", "CASH_RETURN", "CASH_ret"],
    }
    for target, candidates in flexible.items():
        found = _first_existing(list(panel.columns), candidates)
        if found is not None and found != target:
            rename[found] = target
    panel = panel.rename(columns=rename)

    required = [
        "date",
        "macro_regime_confirmed",
        "monthly_either_state",
        "portfolio_return",
        "portfolio_nav",
        "SPY_RETURN",
        "IEF_RETURN",
        "GOLD_RETURN",
        "CMDTY_FUT_RETURN",
        "RF_DAILY",
    ]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"Missing required columns in strategy panel: {missing}")

    if "spy_price" not in panel.columns:
        panel["spy_price"] = (1.0 + panel["SPY_RETURN"]).cumprod()
        warnings.warn("spy_price not found; reconstructed proxy from SPY_RETURN.")
    return panel.sort_values("date").reset_index(drop=True)


def _load_extra_daily_panel(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        warnings.warn(f"Could not read {path}: {exc}")
        return None
    if "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date")


def load_macro_panel() -> tuple[pd.DataFrame, list[str]]:
    used_sources: list[str] = []
    frames: list[pd.DataFrame] = []

    for path in [RECON_PANEL_PATH, RULE_PANEL_PATH]:
        df = _load_extra_daily_panel(path)
        if df is None:
            continue
        keep = ["date"] + [
            c
            for c in df.columns
            if c
            in {
                "CREDIT_SPREAD_BAA_AAA",
                "DGS1",
                "DGS10",
                "GS1",
                "GS10",
                "TERM_SPREAD_10Y_1Y",
                "VIX_LEVEL",
                "growth",
                "inflation",
                "rate",
                "term_spread",
                "credit_spread",
                "growth_pc1",
                "inflation_pc1",
                "rate_pc",
                "PC1",
                "PC2",
                "D_GROWTH_PC1",
                "D_INFLATION_PC1",
                "D_CREDIT_SPREAD",
                "D_TERM_SPREAD_10Y_1Y",
            }
        ]
        frames.append(df[keep].copy())
        used_sources.append(str(path.relative_to(ROOT)))

    merged: pd.DataFrame | None = None
    for df in frames:
        merged = df if merged is None else merged.merge(df, on="date", how="outer", suffixes=("", "_dup"))
        dup_cols = [c for c in merged.columns if c.endswith("_dup")]
        for dup in dup_cols:
            base = dup[:-4]
            if base in merged.columns:
                merged[base] = merged[base].combine_first(merged[dup])
            else:
                merged = merged.rename(columns={dup: base})
        merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_dup")], errors="ignore")

    if merged is None:
        merged = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")})

    for path in RISK_PANEL_PATHS:
        rf = _load_extra_daily_panel(path)
        if rf is None:
            continue
        keep = ["date"] + [
            c
            for c in rf.columns
            if c
            in {
                "growth_pc1",
                "inflation_pc1",
                "gs10",
                "term_spread_10y_1y",
                "credit_spread",
                "D_GROWTH_PC1",
                "D_INFLATION_PC1",
                "D_GS10",
                "D_TERM_SPREAD_10Y_1Y",
                "D_CREDIT_SPREAD",
                "VIX_LEVEL",
            }
        ]
        rf = rf[keep].copy()
        rf = rf.rename(
            columns={
                "gs10": "GS10",
                "term_spread_10y_1y": "term_spread",
                "credit_spread": "credit_spread",
            }
        )
        # Monthly risk factor files are aligned to daily dates using last observation carried forward.
        merged = pd.merge_asof(
            merged.sort_values("date"),
            rf.sort_values("date"),
            on="date",
            direction="backward",
            suffixes=("", "_rf"),
        )
        for c in [x for x in merged.columns if x.endswith("_rf")]:
            base = c[:-3]
            if base in merged.columns:
                merged[base] = merged[base].combine_first(merged[c])
            else:
                merged = merged.rename(columns={c: base})
        merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_rf")], errors="ignore")
        used_sources.append(str(path.relative_to(ROOT)))
        break

    return merged.sort_values("date").drop_duplicates("date"), used_sources


def merge_covid_flat_panel(strategy: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    panel = strategy.merge(macro, on="date", how="left", suffixes=("", "_macro"))
    for c in [c for c in panel.columns if c.endswith("_macro")]:
        base = c[:-6]
        if base in panel.columns:
            panel[base] = panel[base].combine_first(panel[c])
        else:
            panel = panel.rename(columns={c: base})
    panel = panel.drop(columns=[c for c in panel.columns if c.endswith("_macro")], errors="ignore")

    if "CASH_RETURN" not in panel.columns:
        panel["CASH_RETURN"] = panel["RF_DAILY"]
    if "strategy_nav_rebased" not in panel.columns:
        panel["strategy_nav_rebased"] = (1.0 + panel["portfolio_return"]).cumprod()
    return panel.sort_values("date").reset_index(drop=True)


def define_sample_groups(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    out = panel.copy()
    covid_start = pd.Timestamp(CONFIG["covid_start"])
    covid_end = pd.Timestamp(CONFIG["covid_end"])
    covid = out["date"].between(covid_start, covid_end)
    flat = out["macro_regime_confirmed"].eq("FLAT")
    hold = out["monthly_either_state"].eq("HOLD")
    sell = out["monthly_either_state"].eq("SELL")
    groups = {
        "COVID_2020": covid,
        "FLAT_EX_COVID": flat & ~covid,
        "FLAT_HOLD_EX_COVID": flat & hold & ~covid,
        "FLAT_SELL_EX_COVID": flat & sell & ~covid,
        "ALL_EX_COVID": ~covid,
        "COVID_FLAT_ONLY": covid & flat,
    }
    out["is_covid_2020"] = covid
    out["is_flat_ex_covid"] = groups["FLAT_EX_COVID"]
    out["is_flat_hold_ex_covid"] = groups["FLAT_HOLD_EX_COVID"]
    out["is_flat_sell_ex_covid"] = groups["FLAT_SELL_EX_COVID"]
    out["is_covid_flat_only"] = groups["COVID_FLAT_ONLY"]
    out["sample_group"] = np.select(
        [
            groups["COVID_2020"],
            groups["FLAT_HOLD_EX_COVID"],
            groups["FLAT_SELL_EX_COVID"],
            groups["FLAT_EX_COVID"],
        ],
        ["COVID_2020", "FLAT_HOLD_EX_COVID", "FLAT_SELL_EX_COVID", "FLAT_EX_COVID"],
        default="ALL_EX_COVID",
    )
    return out, groups


def annualized_return(s: pd.Series) -> float:
    r = s.dropna()
    if r.empty:
        return np.nan
    return float((1.0 + r).prod() ** (252.0 / len(r)) - 1.0)


def max_drawdown(s: pd.Series) -> float:
    r = s.dropna()
    if r.empty:
        return np.nan
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def compute_asset_performance(panel: pd.DataFrame, groups: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for group_name, mask in groups.items():
        sub = panel.loc[mask].copy()
        for asset in CONFIG["assets"]:
            ret_col = ASSET_RETURN_COLS[asset]
            if ret_col not in sub.columns:
                continue
            s = sub[ret_col].dropna()
            if s.empty:
                continue
            rf = sub.loc[s.index, "RF_DAILY"]
            excess = s - rf
            spy = sub.loc[s.index, "SPY_RETURN"]
            spy_neg = spy < 0
            corr = np.nan if asset == "SPY" else float(s.corr(spy))
            spy_downside = spy.loc[spy < 0].sum()
            asset_downside = s.loc[spy < 0].sum()
            downside_capture = float(asset_downside / spy_downside) if spy_downside != 0 else np.nan
            ex_std = excess.std(ddof=1)
            sharpe = 0.0 if asset == "CASH" else (float(excess.mean() / ex_std * np.sqrt(252.0)) if pd.notna(ex_std) and ex_std != 0 else np.nan)
            rows.append(
                {
                    "sample_group": group_name,
                    "asset": asset,
                    "n_obs": int(len(s)),
                    "cumulative_return": float((1.0 + s).prod() - 1.0),
                    "annualized_return": annualized_return(s),
                    "annualized_volatility": float(s.std(ddof=1) * np.sqrt(252.0)) if len(s) > 1 else np.nan,
                    "Sharpe": sharpe,
                    "max_drawdown": max_drawdown(s),
                    "worst_day": float(s.min()),
                    "best_day": float(s.max()),
                    "positive_day_ratio": float((s > 0).mean()),
                    "correlation_with_SPY": corr,
                    "average_return_when_SPY_negative": float(s.loc[spy_neg].mean()) if spy_neg.any() else np.nan,
                    "downside_capture_vs_SPY": downside_capture,
                }
            )
    return pd.DataFrame(rows)


def compute_strategy_contributions(panel: pd.DataFrame, groups: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = panel.copy()
    for asset in CONFIG["assets"]:
        weight_col = f"actual_weight_{asset if asset != 'CMDTY_FUT' else 'CMDTY'}"
        ret_col = ASSET_RETURN_COLS[asset]
        contrib_col = f"contribution_{asset}"
        if weight_col in out.columns and ret_col in out.columns:
            out[contrib_col] = out[weight_col] * out[ret_col]
        else:
            out[contrib_col] = np.nan

    covid = out.loc[groups["COVID_2020"]].copy()
    rows = []
    for group_name in ["COVID_2020", "FLAT_EX_COVID"]:
        sub = out.loc[groups[group_name]].copy()
        if sub.empty:
            continue
        row = {"sample_group": group_name, "n_obs": int(len(sub))}
        for asset in CONFIG["assets"]:
            suffix = asset if asset != "CMDTY_FUT" else "CMDTY"
            wcol = f"actual_weight_{suffix}"
            ccol = f"contribution_{asset}"
            row[f"average_weight_{asset}"] = float(sub[wcol].mean()) if wcol in sub.columns else np.nan
            row[f"cumulative_contribution_{asset}"] = float(sub[ccol].sum()) if ccol in sub.columns else np.nan
            row[f"average_daily_contribution_{asset}"] = float(sub[ccol].mean()) if ccol in sub.columns else np.nan
            row[f"worst_daily_contribution_{asset}"] = float(sub[ccol].min()) if ccol in sub.columns else np.nan
        rows.append(row)
    return covid, pd.DataFrame(rows)


def macro_variable_candidates(panel: pd.DataFrame) -> list[str]:
    candidates = [
        "CREDIT_SPREAD_BAA_AAA",
        "DGS1",
        "GS1",
        "DGS10",
        "GS10",
        "TERM_SPREAD_10Y_1Y",
        "VIX_LEVEL",
        "growth",
        "inflation",
        "rate",
        "term_spread",
        "credit_spread",
        "growth_pc1",
        "inflation_pc1",
        "rate_pc",
        "PC1",
        "PC2",
        "D_GROWTH_PC1",
        "D_INFLATION_PC1",
        "D_CREDIT_SPREAD",
        "D_TERM_SPREAD_10Y_1Y",
    ]
    return [c for c in candidates if c in panel.columns and pd.api.types.is_numeric_dtype(panel[c])]


def compute_macro_distribution(panel: pd.DataFrame, groups: dict[str, pd.Series], variables: list[str]) -> pd.DataFrame:
    rows = []
    for group_name, mask in groups.items():
        sub = panel.loc[mask].copy()
        for var in variables:
            s = pd.to_numeric(sub[var], errors="coerce").dropna()
            if s.empty:
                continue
            rows.append(
                {
                    "sample_group": group_name,
                    "variable": var,
                    "n_obs": int(len(s)),
                    "mean": float(s.mean()),
                    "median": float(s.median()),
                    "std": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "p10": float(s.quantile(0.10)),
                    "p25": float(s.quantile(0.25)),
                    "p75": float(s.quantile(0.75)),
                    "p90": float(s.quantile(0.90)),
                    "start_value": float(s.iloc[0]),
                    "end_value": float(s.iloc[-1]),
                    "change_over_period": float(s.iloc[-1] - s.iloc[0]),
                    "max_20d_change": float(s.diff(20).max()) if len(s) > 20 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _percentile_score(value: float, reference: pd.Series) -> float:
    ref = reference.dropna()
    if ref.empty or pd.isna(value):
        return np.nan
    return float((ref <= value).mean())


def compute_zscore_percentile(panel: pd.DataFrame, groups: dict[str, pd.Series], variables: list[str]) -> pd.DataFrame:
    rows = []
    non_covid = panel.loc[~groups["COVID_2020"]]
    for group_name, mask in groups.items():
        sub = panel.loc[mask]
        for var in variables:
            ref = pd.to_numeric(non_covid[var], errors="coerce").dropna()
            s = pd.to_numeric(sub[var], errors="coerce").dropna()
            if ref.empty or s.empty:
                continue
            std = ref.std(ddof=1)
            z = (s - ref.mean()) / std if pd.notna(std) and std != 0 else pd.Series(np.nan, index=s.index)
            pct = s.apply(lambda x: _percentile_score(x, ref))
            rows.append(
                {
                    "sample_group": group_name,
                    "variable": var,
                    "n_obs": int(len(s)),
                    "average_zscore": float(z.mean()),
                    "max_zscore": float(z.max()),
                    "min_zscore": float(z.min()),
                    "average_percentile": float(pct.mean()),
                    "max_percentile": float(pct.max()),
                    "min_percentile": float(pct.min()),
                }
            )
    return pd.DataFrame(rows)


def _rebase_nav(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def plot_asset_comparison(asset_perf: pd.DataFrame, panel: pd.DataFrame, groups: dict[str, pd.Series]) -> None:
    comp = asset_perf.loc[asset_perf["sample_group"].isin(["COVID_2020", "FLAT_EX_COVID"])].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric, title in zip(
        axes,
        ["cumulative_return", "max_drawdown", "annualized_volatility"],
        ["Cumulative Return", "Max Drawdown", "Annualized Volatility"],
    ):
        sns.barplot(data=comp, x="asset", y=metric, hue="sample_group", ax=ax)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIG_ASSET, dpi=180)
    plt.close(fig)

    covid = panel.loc[groups["COVID_2020"]].copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    for asset in CONFIG["assets"]:
        ax.plot(covid["date"], _rebase_nav(covid[ASSET_RETURN_COLS[asset]]), label=asset)
    ax.plot(covid["date"], covid["portfolio_nav"] / covid["portfolio_nav"].iloc[0], label="REGIME_HEDGE_STEEP_SELL_IEF", linewidth=2.2, color="black")
    ax.set_title("COVID Asset NAV Paths")
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(FIG_NAV, dpi=180)
    plt.close(fig)

    flat = panel.loc[groups["FLAT_EX_COVID"]].copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    for asset in CONFIG["assets"]:
        ax.plot(flat["date"], _rebase_nav(flat[ASSET_RETURN_COLS[asset]]), label=asset)
    ax.set_title("FLAT_EX_COVID Conditional Asset NAV (stitched state days)")
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(FIG_FLAT_NAV, dpi=180)
    plt.close(fig)


def plot_macro_comparison(panel: pd.DataFrame, groups: dict[str, pd.Series], variables: list[str], zscores: pd.DataFrame) -> None:
    long_parts = []
    for group in ["COVID_2020", "FLAT_EX_COVID"]:
        sub = panel.loc[groups[group], ["date"] + variables].copy()
        sub["sample_group"] = group
        long_parts.append(sub)
    if not long_parts:
        return
    long = pd.concat(long_parts, ignore_index=True).melt(id_vars=["date", "sample_group"], var_name="variable", value_name="value")
    stress_vars = [v for v in ["VIX_LEVEL", "CREDIT_SPREAD_BAA_AAA", "credit_spread"] if v in variables]
    macro_vars = [v for v in variables if v not in stress_vars][:8]

    if stress_vars:
        fig, axes = plt.subplots(1, len(stress_vars), figsize=(5 * len(stress_vars), 5))
        axes = np.atleast_1d(axes)
        for ax, var in zip(axes, stress_vars):
            sns.boxplot(data=long.loc[long["variable"] == var], x="sample_group", y="value", ax=ax)
            ax.set_title(var)
            ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(FIG_STRESS_BOX, dpi=180)
        plt.close(fig)

    if macro_vars:
        n = len(macro_vars)
        fig, axes = plt.subplots(int(np.ceil(n / 3)), 3, figsize=(15, 4 * int(np.ceil(n / 3))))
        axes = np.atleast_1d(axes).flatten()
        for ax, var in zip(axes, macro_vars):
            sns.boxplot(data=long.loc[long["variable"] == var], x="sample_group", y="value", ax=ax)
            ax.set_title(var)
            ax.tick_params(axis="x", rotation=20)
        for ax in axes[n:]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(FIG_MACRO_BOX, dpi=180)
        plt.close(fig)

    heat = zscores.loc[zscores["sample_group"].isin(["COVID_2020", "FLAT_EX_COVID", "FLAT_HOLD_EX_COVID", "FLAT_SELL_EX_COVID"])]
    if not heat.empty:
        piv = heat.pivot(index="variable", columns="sample_group", values="average_zscore")
        fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * len(piv))))
        sns.heatmap(piv, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
        ax.set_title("Average Z-score vs Non-COVID Sample")
        fig.tight_layout()
        fig.savefig(FIG_Z, dpi=180)
        plt.close(fig)


def plot_covid_timeline(panel: pd.DataFrame, groups: dict[str, pd.Series]) -> None:
    covid = panel.loc[groups["COVID_2020"]].copy()
    if covid.empty:
        return
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True, gridspec_kw={"height_ratios": [2.5, 2, 2, 1.4]})
    ax1, ax2, ax3, ax4 = axes
    ax1.plot(covid["date"], covid["spy_price"] / covid["spy_price"].iloc[0], label="SPY", color="black")
    ax1.plot(covid["date"], covid["portfolio_nav"] / covid["portfolio_nav"].iloc[0], label="Strategy", color="tab:orange")
    ax1.set_title("COVID Macro Timeline")
    ax1.legend()
    if "VIX_LEVEL" in covid.columns:
        ax2.plot(covid["date"], covid["VIX_LEVEL"], label="VIX", color="tab:red")
    if "CREDIT_SPREAD_BAA_AAA" in covid.columns:
        ax2b = ax2.twinx()
        ax2b.plot(covid["date"], covid["CREDIT_SPREAD_BAA_AAA"], label="Credit spread", color="tab:purple")
    ax2.set_title("Market Stress Variables")
    for col, label in [("DGS1", "DGS1"), ("DGS10", "DGS10"), ("TERM_SPREAD_10Y_1Y", "Term spread")]:
        if col in covid.columns:
            ax3.plot(covid["date"], covid[col], label=label)
    ax3.legend()
    ax3.set_title("Rates / Curve")
    y = np.zeros(len(covid))
    for regime, color in REGIME_COLORS.items():
        mask = covid["macro_regime_confirmed"].eq(regime).to_numpy()
        ax4.fill_between(covid["date"], 0.55, 0.95, where=mask, color=color, alpha=0.7, label=regime)
    ax4.fill_between(covid["date"], 0.05, 0.45, where=covid["monthly_either_state"].eq("HOLD"), color="green", alpha=0.7, label="HOLD")
    ax4.fill_between(covid["date"], 0.05, 0.45, where=covid["monthly_either_state"].eq("SELL"), color="red", alpha=0.7, label="SELL")
    ax4.set_yticks([0.25, 0.75])
    ax4.set_yticklabels(["Timing", "Regime"])
    handles, labels = ax4.get_legend_handles_labels()
    ax4.legend(handles[:6], labels[:6], ncol=3, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_TIMELINE, dpi=180)
    plt.close(fig)


def plot_strategy_contribution(covid_contrib: pd.DataFrame) -> None:
    if covid_contrib.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].stackplot(
        covid_contrib["date"],
        covid_contrib["actual_weight_SPY"],
        covid_contrib["actual_weight_IEF"],
        covid_contrib["actual_weight_GOLD"],
        covid_contrib["actual_weight_CMDTY"],
        covid_contrib["actual_weight_CASH"],
        labels=CONFIG["assets"],
    )
    axes[0].legend(ncol=5, loc="upper left")
    axes[0].set_title("Strategy Weights During COVID")
    for asset in CONFIG["assets"]:
        ccol = f"contribution_{asset}"
        if ccol in covid_contrib.columns:
            axes[1].plot(covid_contrib["date"], covid_contrib[ccol].fillna(0).cumsum(), label=asset)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].legend(ncol=5, loc="lower left")
    axes[1].set_title("Cumulative Contribution During COVID")
    fig.tight_layout()
    fig.savefig(FIG_CONTRIB, dpi=180)
    plt.close(fig)


def write_markdown_report(
    asset_perf: pd.DataFrame,
    macro_dist: pd.DataFrame,
    zscores: pd.DataFrame,
    contrib_summary: pd.DataFrame,
    used_sources: list[str],
    missing_vars: list[str],
) -> None:
    covid_perf = asset_perf.loc[asset_perf["sample_group"] == "COVID_2020"].copy()
    flat_perf = asset_perf.loc[asset_perf["sample_group"] == "FLAT_EX_COVID"].copy()
    worst_covid = covid_perf.sort_values("cumulative_return").iloc[0]["asset"] if not covid_perf.empty else "N/A"
    best_def = (
        covid_perf.loc[covid_perf["asset"].isin(["IEF", "GOLD", "CASH"])]
        .sort_values("max_drawdown", ascending=False)
        .iloc[0]["asset"]
        if not covid_perf.empty
        else "N/A"
    )
    cmdty_line = ""
    if not covid_perf.empty and not flat_perf.empty:
        c_covid = covid_perf.loc[covid_perf["asset"] == "CMDTY_FUT", "cumulative_return"]
        c_flat = flat_perf.loc[flat_perf["asset"] == "CMDTY_FUT", "cumulative_return"]
        if not c_covid.empty and not c_flat.empty:
            cmdty_line = f"- CMDTY_FUT cumulative return: COVID {c_covid.iloc[0]:.2%}, FLAT_EX_COVID {c_flat.iloc[0]:.2%}."
    def _z_line(var: str) -> str:
        row = zscores.loc[(zscores["sample_group"] == "COVID_2020") & (zscores["variable"] == var)]
        if row.empty:
            return f"- {var}: unavailable."
        r = row.iloc[0]
        return f"- {var}: average z-score {r['average_zscore']:.2f}, average percentile {r['average_percentile']:.1%}."

    contrib_line = "- Strategy contribution unavailable."
    covid_contrib = contrib_summary.loc[contrib_summary["sample_group"] == "COVID_2020"]
    if not covid_contrib.empty:
        vals = {
            asset: float(covid_contrib[f"cumulative_contribution_{asset}"].iloc[0])
            for asset in CONFIG["assets"]
            if f"cumulative_contribution_{asset}" in covid_contrib.columns
        }
        if vals:
            contrib_line = "; ".join(f"{asset} {value:.2%}" for asset, value in vals.items())
            contrib_line = f"- COVID cumulative contribution by asset: {contrib_line}."

    missing_lines = [f"- {v}" for v in missing_vars] if missing_vars else ["- None."]

    lines = [
        "# COVID vs FLAT Regime Diagnostic",
        "",
        "## Purpose",
        "",
        "This diagnostic explains why `REGIME_HEDGE_STEEP_SELL_IEF` struggled during the COVID crash and tests whether COVID behaved like an ordinary FLAT regime.",
        "",
        "## Sample Definitions",
        "",
        f"- COVID_2020: {CONFIG['covid_start']} to {CONFIG['covid_end']}",
        "- FLAT_EX_COVID: macro_regime_confirmed == FLAT and outside COVID_2020.",
        "- FLAT_HOLD_EX_COVID / FLAT_SELL_EX_COVID split FLAT_EX_COVID by Monthly Either state.",
        "- COVID_FLAT_ONLY is reported separately when COVID days are also classified as FLAT.",
        "",
        "## Data Sources",
        "",
        *[f"- {src}" for src in used_sources],
        "",
        "## Asset Behavior Comparison",
        "",
        f"- Worst COVID asset by cumulative return: `{worst_covid}`.",
        f"- Most stable defensive asset in COVID by max drawdown: `{best_def}`.",
        cmdty_line,
        "- Short-window annualized returns are reported but should not be over-interpreted.",
        "",
        f"![COVID asset comparison](../../figures/covid_vs_flat_regime_diagnostic/{FIG_ASSET.name})",
        "",
        f"![COVID asset NAV](../../figures/covid_vs_flat_regime_diagnostic/{FIG_NAV.name})",
        "",
        "## Macro Variable Comparison",
        "",
        "- The key question is whether term-spread rules labeled COVID as FLAT while VIX and credit spread already showed an abnormal market stress shock.",
        "- `macro_variable_zscore_percentile_covid.csv` reports COVID variable z-scores relative to the non-COVID sample.",
        _z_line("VIX_LEVEL"),
        _z_line("CREDIT_SPREAD_BAA_AAA"),
        _z_line("TERM_SPREAD_10Y_1Y"),
        _z_line("growth_pc1"),
        _z_line("inflation_pc1"),
        "",
        f"![Market stress variables](../../figures/covid_vs_flat_regime_diagnostic/{FIG_STRESS_BOX.name})",
        "",
        f"![Z-score heatmap](../../figures/covid_vs_flat_regime_diagnostic/{FIG_Z.name})",
        "",
        f"![COVID macro timeline](../../figures/covid_vs_flat_regime_diagnostic/{FIG_TIMELINE.name})",
        "",
        "## Strategy Contribution",
        "",
        "- `strategy_weight_and_contribution_covid.csv` decomposes COVID daily contributions using actual start-of-day weights.",
        "- This helps separate losses from SPY exposure, CMDTY exposure, and regime/timing lag.",
        contrib_line,
        "",
        f"![COVID contribution](../../figures/covid_vs_flat_regime_diagnostic/{FIG_CONTRIB.name})",
        "",
        "## Interpretation",
        "",
        "- COVID-like crashes should be treated as an exception candidate to ordinary FLAT behavior if VIX/credit stress is extreme.",
        "- A fast crash overlay or VIX/credit defensive override is more directly targeted than changing the whole macro regime rule.",
        "- FLAT commodity exposure should be stress-tested at lower weights because commodities can add path risk in sudden liquidation events.",
        "- FLAT stress routing should test IEF/CASH instead of assuming ordinary FLAT real-asset behavior.",
        "",
        "## Caveats",
        "",
        "- COVID is a single event and should not be overfit.",
        "- Conditional FLAT_EX_COVID NAV is a stitched state diagnostic, not a directly investable continuous path.",
        "- Monthly macro variables such as growth/inflation can lag fast market shocks.",
        "",
        "## Missing Variables",
        "",
        *missing_lines,
        "",
        "## Next Step",
        "",
        "- Test FLAT CMDTY weight sensitivity.",
        "- Test VIX/credit spike override.",
        "- Test FLAT normal vs FLAT stress hedge routing into IEF/CASH.",
    ]
    REPORT_OUT.write_text("\n".join([x for x in lines if x is not None]), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    strategy = load_strategy_panel()
    macro, used_sources = load_macro_panel()
    panel = merge_covid_flat_panel(strategy, macro)
    panel, groups = define_sample_groups(panel)

    variables = macro_variable_candidates(panel)
    expected = ["CREDIT_SPREAD_BAA_AAA", "DGS1", "DGS10", "TERM_SPREAD_10Y_1Y", "VIX_LEVEL", "growth_pc1", "inflation_pc1"]
    missing_vars = [v for v in expected if v not in variables]

    panel.to_csv(PANEL_OUT, index=False)

    asset_perf = compute_asset_performance(panel, groups)
    asset_perf.to_csv(ASSET_PERF_OUT, index=False)
    asset_perf.pivot(index="sample_group", columns="asset", values="cumulative_return").to_csv(CUMRET_PIVOT_OUT)
    asset_perf.pivot(index="sample_group", columns="asset", values="max_drawdown").to_csv(MDD_PIVOT_OUT)
    asset_perf.pivot(index="sample_group", columns="asset", values="Sharpe").to_csv(SHARPE_PIVOT_OUT)
    asset_perf.pivot(index="sample_group", columns="asset", values="correlation_with_SPY").to_csv(CORR_PIVOT_OUT)

    covid_contrib, contrib_summary = compute_strategy_contributions(panel, groups)
    covid_contrib.to_csv(WEIGHT_CONTRIB_OUT, index=False)
    contrib_summary.to_csv(CONTRIB_SUMMARY_OUT, index=False)

    macro_dist = compute_macro_distribution(panel, groups, variables)
    macro_dist.to_csv(MACRO_DIST_OUT, index=False)
    zscores = compute_zscore_percentile(panel, groups, variables)
    zscores.to_csv(ZSCORE_OUT, index=False)

    plot_asset_comparison(asset_perf, panel, groups)
    plot_macro_comparison(panel, groups, variables, zscores)
    plot_covid_timeline(panel, groups)
    plot_strategy_contribution(covid_contrib)
    write_markdown_report(asset_perf, macro_dist, zscores, contrib_summary, used_sources, missing_vars)

    covid = panel.loc[groups["COVID_2020"]]
    regime_share = covid["macro_regime_confirmed"].value_counts(normalize=True).sort_values(ascending=False)
    covid_perf = asset_perf.loc[asset_perf["sample_group"] == "COVID_2020"].copy()
    flat_perf = asset_perf.loc[asset_perf["sample_group"] == "FLAT_EX_COVID"].copy()
    worst_asset = covid_perf.sort_values("cumulative_return").iloc[0]["asset"] if not covid_perf.empty else "N/A"
    cmdty_covid = covid_perf.loc[covid_perf["asset"] == "CMDTY_FUT", "cumulative_return"].iloc[0]
    cmdty_flat = flat_perf.loc[flat_perf["asset"] == "CMDTY_FUT", "cumulative_return"].iloc[0]
    defensive = covid_perf.loc[covid_perf["asset"].isin(["IEF", "GOLD", "CASH"])].sort_values("max_drawdown", ascending=False)
    best_def = defensive.iloc[0]["asset"] if not defensive.empty else "N/A"
    vix_z = zscores.loc[(zscores["sample_group"] == "COVID_2020") & (zscores["variable"] == "VIX_LEVEL"), "average_zscore"]
    credit_z = zscores.loc[(zscores["sample_group"] == "COVID_2020") & (zscores["variable"] == "CREDIT_SPREAD_BAA_AAA"), "average_zscore"]
    contrib_cols = [c for c in contrib_summary.columns if c.startswith("cumulative_contribution_")]
    covid_contrib_summary = contrib_summary.loc[contrib_summary["sample_group"] == "COVID_2020"]
    worst_contrib = "N/A"
    if not covid_contrib_summary.empty and contrib_cols:
        vals = covid_contrib_summary[contrib_cols].iloc[0].astype(float)
        worst_contrib = vals.idxmin().replace("cumulative_contribution_", "")

    print("1. COVID macro regime share:")
    print(regime_share.to_string())
    print(f"2. COVID mostly classified as FLAT: {bool(regime_share.index[0] == 'FLAT') if not regime_share.empty else False}")
    print(f"3. Worst asset in COVID: {worst_asset}")
    print(f"4. CMDTY_FUT COVID cumulative return vs FLAT_EX_COVID: {cmdty_covid:.2%} vs {cmdty_flat:.2%}")
    print(f"5. Best defensive asset in COVID by drawdown stability: {best_def}")
    print(f"6. COVID average VIX z-score / credit z-score: {vix_z.iloc[0] if not vix_z.empty else np.nan:.2f} / {credit_z.iloc[0] if not credit_z.empty else np.nan:.2f}")
    print(f"7. Missing growth/inflation variables: {', '.join([v for v in ['growth_pc1', 'inflation_pc1'] if v in missing_vars]) or 'None'}")
    print(f"8. Largest negative strategy contribution during COVID: {worst_contrib}")
    print("9. Diagnostic implication: test lower FLAT CMDTY, VIX/credit spike override, and FLAT stress routing to IEF/CASH.")
    print(f"Saved outputs: {RESULTS_DIR} and {FIGURES_DIR}")


if __name__ == "__main__":
    main()
