from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture

from final_strategy_source_only_core import ASSETS, FINAL_STRATEGY, REFINED_BASELINE, ROOT, SPY_BUY_HOLD, SPY_CASH_TIMING, build_final_source_only_panel


OUT = ROOT / "results" / "main_pipeline_final"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
FINAL = FINAL_STRATEGY
BASE = REFINED_BASELINE
SPY = SPY_BUY_HOLD
DISPLAY_STRATEGIES = [SPY_BUY_HOLD, SPY_CASH_TIMING, FINAL_STRATEGY]
RATE_STATE_ORDER = [
    "FLAT_LOW_RATE",
    "FLAT_MID_RATE",
    "FLAT_HIGH_RATE",
    "STEEP_LOW_RATE",
    "STEEP_MID_RATE",
    "STEEP_HIGH_RATE",
    "INVERTED",
]
OIL_LEVEL_ORDER = ["OIL_LEVEL_LOW", "OIL_LEVEL_MID", "OIL_LEVEL_HIGH"]

STATE_ORDER = [
    "FLAT_LOW_RATE_NORMAL",
    "FLAT_MID_RATE_NORMAL",
    "FLAT_LOWMID_RATE_STRESS",
    "FLAT_HIGH_RATE_NORMAL",
    "FLAT_HIGH_RATE_STRESS",
    "STEEP_LOW_RATE_NORMAL",
    "STEEP_LOW_RATE_STRESS",
    "STEEP_MID_RATE_NORMAL",
    "STEEP_MID_RATE_STRESS",
    "STEEP_HIGH_RATE_NORMAL",
    "STEEP_HIGH_RATE_STRESS",
    "INVERTED_NORMAL",
    "INVERTED_STRESS",
]
PURE_STATE_ORDER = STATE_ORDER


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def asset_behavior(panel: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group, sub in panel.groupby(group_col, dropna=False):
        for asset in ASSETS:
            r = sub[f"{asset}_return"].fillna(0.0)
            excess = r - sub["CASH_return"].fillna(0.0)
            nav = (1.0 + r).cumprod()
            ann_ret = nav.iloc[-1] ** (252.0 / len(r)) - 1.0
            vol = r.std() * (252.0 ** 0.5)
            excess_nav = (1.0 + excess).cumprod()
            ann_excess_ret = excess_nav.iloc[-1] ** (252.0 / len(excess)) - 1.0
            sharpe = 0.0 if asset == "CASH" else (ann_excess_ret / vol if vol > 0 else None)
            rows.append(
                {
                    group_col: group,
                    "asset": asset,
                    "n_obs": len(r),
                    "annualized_return": ann_ret,
                    "annualized_volatility": vol,
                    "annualized_excess_return_vs_cash": ann_excess_ret,
                    "Sharpe": sharpe,
                    "max_drawdown": (nav / nav.cummax() - 1.0).min(),
                    "cumulative_return": nav.iloc[-1] - 1.0,
                }
            )
    return pd.DataFrame(rows)


def heatmap_cross_state(row: pd.Series) -> str:
    regime = str(row["final_regime_confirmed"])
    alloc = str(row["final_allocation_state"])
    stress = str(row["final_state"]) == "FULL_RISK"
    if alloc == "FLAT_LOWMID_RATE_STRESS":
        return "FLAT_LOWMID_RATE_STRESS"
    if alloc == "STEEP_LOW_RATE_STRESS":
        return "STEEP_LOW_RATE_STRESS"
    if alloc in STATE_ORDER:
        return alloc
    if regime == "INVERTED":
        return "INVERTED_STRESS" if stress else "INVERTED_NORMAL"
    return alloc


def pure_cross_state(row: pd.Series) -> str:
    regime = str(row["final_regime_confirmed"])
    stress = str(row["final_state"]) == "FULL_RISK"
    if regime in {"FLAT_LOW_RATE", "FLAT_MID_RATE"} and stress:
        return "FLAT_LOWMID_RATE_STRESS"
    return f"{regime}_{'STRESS' if stress else 'NORMAL'}"


def short_oil_label(value: str) -> str:
    return value.replace("OIL_LEVEL_", "OIL_")


def oil_stress_cross_state(row: pd.Series) -> str:
    return f"{short_oil_label(str(row['oil_level_regime']))}__{str(row['final_state'])}"


def oil_nonrisk_rate_cross_state(row: pd.Series) -> str | None:
    if str(row["final_state"]) != "NON_RISK":
        return None
    return f"{str(row['final_regime_confirmed'])}__{short_oil_label(str(row['oil_level_regime']))}"


def plot_asset_behavior_heatmap(
    perf: pd.DataFrame,
    group_col: str,
    path_name: str,
    title: str,
    value_col: str = "annualized_return",
    value_label: str = "Annualized return",
    value_format: str = "percent",
    column_order: list[str] | None = None,
) -> None:
    heat = perf.pivot_table(index="asset", columns=group_col, values=value_col, aggfunc="first")
    preferred = column_order if column_order is not None else STATE_ORDER
    cols = [c for c in preferred if c in heat.columns] + [c for c in heat.columns if c not in preferred]
    heat = heat.reindex(index=ASSETS, columns=cols)
    if value_format == "percent":
        labels = heat.map(lambda x: "" if pd.isna(x) else f"{x:.1%}")
    else:
        labels = heat.map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    fig_w = max(12, 1.05 * len(heat.columns))
    fig, ax = plt.subplots(figsize=(fig_w, 5.8))
    sns.heatmap(
        heat,
        annot=labels,
        fmt="",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": value_label},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(FIG_DIR / path_name, dpi=170)
    plt.close(fig)


def plot_coverage_heatmap(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: str,
    path_name: str,
    title: str,
    row_order: list[str],
    col_order: list[str],
) -> None:
    heat = df.pivot(index=row_col, columns=col_col, values=value_col).reindex(index=row_order, columns=col_order)
    labels = heat.map(lambda x: "" if pd.isna(x) else f"{x:.1%}")
    fig_w = max(8, 1.05 * len(heat.columns))
    fig, ax = plt.subplots(figsize=(fig_w, 3.8))
    sns.heatmap(
        heat,
        annot=labels,
        fmt="",
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Coverage of all trading days"},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(FIG_DIR / path_name, dpi=170)
    plt.close(fig)


def crisis_performance(panel: pd.DataFrame) -> pd.DataFrame:
    windows = {
        "2008_GFC": ("2007-10-01", "2009-06-30"),
        "2011_EURO_DEBT": ("2011-06-01", "2011-12-31"),
        "2015_2016": ("2015-05-01", "2016-03-31"),
        "2018Q4": ("2018-10-01", "2019-01-31"),
        "COVID_2020": ("2020-02-01", "2020-06-30"),
        "2022_RATE_WAR": ("2021-11-01", "2023-03-31"),
        "2025_PULLBACK": ("2025-01-01", "2025-12-31"),
    }
    rows = []
    for name, (start, end) in windows.items():
        sub = panel.loc[panel["date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()
        if sub.empty:
            continue
        for strategy in DISPLAY_STRATEGIES:
            r = sub[f"{strategy}_return"].fillna(0.0)
            nav = (1.0 + r).cumprod()
            rows.append(
                {
                    "window": name,
                    "strategy": strategy,
                    "cumulative_return": nav.iloc[-1] - 1.0,
                    "max_drawdown": (nav / nav.cummax() - 1.0).min(),
                    "volatility": r.std() * (252.0 ** 0.5),
                }
            )
    return pd.DataFrame(rows)


CASE_WINDOWS = {
    "case_2008_GFC_final.png": ("2007-10-01", "2009-06-30"),
    "case_2011_euro_debt_final.png": ("2011-06-01", "2011-12-31"),
    "case_2015_2016_final.png": ("2015-05-01", "2016-03-31"),
    "case_2020_covid_final.png": ("2020-02-01", "2020-06-30"),
    "case_2022_rate_war_final.png": ("2021-11-01", "2023-03-31"),
    "case_2025_pullback_final.png": ("2025-01-01", "2025-12-31"),
}


def plot_oil_level_regime_background(panel: pd.DataFrame) -> None:
    sub = panel[["date", "oil_price", "oil_level_regime"]].dropna().copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(sub["date"], sub["oil_price"], color="#1f2937", linewidth=1.6, label="Oil price")

    episode_id = sub["oil_level_regime"].ne(sub["oil_level_regime"].shift()).cumsum()
    for _, grp in sub.groupby(episode_id):
        state = str(grp["oil_level_regime"].iloc[0])
        if state == "OIL_LEVEL_HIGH":
            ax.axvspan(grp["date"].iloc[0], grp["date"].iloc[-1], color="#ef4444", alpha=0.14, linewidth=0)
        elif state == "OIL_LEVEL_LOW":
            ax.axvspan(grp["date"].iloc[0], grp["date"].iloc[-1], color="#2563eb", alpha=0.14, linewidth=0)

    ax.plot([], [], color="#ef4444", linewidth=8, alpha=0.14, label="OIL_LEVEL_HIGH background")
    ax.plot([], [], color="#2563eb", linewidth=8, alpha=0.14, label="OIL_LEVEL_LOW background")
    ax.set_title("Oil Price with High/Low Oil Level Regime Background")
    ax.set_ylabel("Oil price")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "oil_level_regime_background.png", dpi=170)
    plt.close(fig)


def plot_outputs(panel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for strategy in DISPLAY_STRATEGIES:
        ax.plot(panel["date"], panel[f"{strategy}_nav"], label=strategy)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "final_equity_curve_comparison.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for strategy in DISPLAY_STRATEGIES:
        ax.plot(panel["date"], panel[f"{strategy}_drawdown"], label=strategy)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "final_drawdown_curve_comparison.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for strategy in DISPLAY_STRATEGIES:
        nav = panel[f"{strategy}_nav"]
        ax.plot(panel["date"], nav / nav.shift(252) - 1.0, label=strategy)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.legend()
    ax.grid(alpha=0.25)
    ax.set_title("Rolling 12M Return Comparison")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rolling_12m_return_comparison.png", dpi=170)
    plt.close(fig)

    weight_cols = [f"{FINAL}_weight_{asset}" for asset in ASSETS]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.stackplot(panel["date"], *[panel[col] for col in weight_cols], labels=ASSETS, alpha=0.9)
    ax.set_ylim(0, 1)
    ax.set_title(f"{FINAL} Weight Timeline")
    ax.legend(ncol=len(ASSETS), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "final_strategy_weights_timeline.png", dpi=170)
    plt.close(fig)

    plot_oil_level_regime_background(panel)


def plot_performance_bars(perf: pd.DataFrame) -> None:
    metrics = [
        ("CAGR", "CAGR"),
        ("Sharpe", "Sharpe"),
        ("Sortino", "Sortino"),
        ("MaxDD", "MaxDD"),
        ("Calmar", "Calmar"),
        ("final_equity", "Final Equity"),
        ("turnover", "Turnover"),
    ]
    plot_df = perf.loc[perf["strategy"].isin(DISPLAY_STRATEGIES)].copy()
    plot_df["strategy"] = pd.Categorical(plot_df["strategy"], categories=DISPLAY_STRATEGIES, ordered=True)
    plot_df = plot_df.sort_values("strategy")
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    axes = axes.flatten()
    for ax, (col, title) in zip(axes, metrics):
        vals = plot_df[col].astype(float)
        colors = ["#4C78A8", "#F58518", "#54A24B"]
        ax.bar(plot_df["strategy"].astype(str), vals, color=colors[: len(plot_df)])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=25)
        if col in {"CAGR", "MaxDD"}:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        for idx, value in enumerate(vals):
            label = f"{value:.1%}" if col in {"CAGR", "MaxDD"} else f"{value:.2f}"
            ax.text(idx, value, label, ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    axes[-1].axis("off")
    fig.suptitle("Final Strategy Performance Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "final_performance_bar_charts.png", dpi=170)
    plt.close(fig)


def plot_case_studies(panel: pd.DataFrame) -> None:
    for filename, (start, end) in CASE_WINDOWS.items():
        sub = panel.loc[panel["date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()
        if sub.empty:
            continue
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        for strategy in DISPLAY_STRATEGIES:
            local_nav = sub[f"{strategy}_nav"] / sub[f"{strategy}_nav"].iloc[0]
            axes[0].plot(sub["date"], local_nav, label=strategy)
        axes[0].set_title(f"{filename.replace('_', ' ').replace('.png', '')}: NAV, normalized to window start")
        axes[0].grid(alpha=0.25)
        axes[0].legend(loc="best")

        cash_nav = sub[f"{SPY_CASH_TIMING}_nav"] / sub[f"{SPY_CASH_TIMING}_nav"].iloc[0]
        for strategy in [SPY_BUY_HOLD, FINAL_STRATEGY]:
            local_nav = sub[f"{strategy}_nav"] / sub[f"{strategy}_nav"].iloc[0]
            axes[1].plot(sub["date"], local_nav / cash_nav - 1.0, label=f"{strategy} vs {SPY_CASH_TIMING}")
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].set_title(f"Relative performance versus {SPY_CASH_TIMING}")
        axes[1].grid(alpha=0.25)
        axes[1].legend(loc="best")

        for strategy in DISPLAY_STRATEGIES:
            local_nav = sub[f"{strategy}_nav"] / sub[f"{strategy}_nav"].iloc[0]
            dd = local_nav / local_nav.cummax() - 1.0
            axes[2].plot(sub["date"], dd, label=strategy)
        axes[2].set_title("Window drawdown")
        axes[2].grid(alpha=0.25)
        axes[2].legend(loc="best")

        fig.tight_layout()
        fig.savefig(FIG_DIR / filename, dpi=170)
        plt.close(fig)


def build_gs10_structure_outputs(
    panel: pd.DataFrame,
    family: str,
    state_prefix: str,
    band_lines: list[tuple[float, str, str]],
    fig_name: str,
    table_prefix: str,
) -> pd.DataFrame:
    sub = panel.loc[panel["macro_regime_confirmed"].eq(family), ["date", "GS10", "final_regime_confirmed"]].dropna().copy()
    sub = sub.loc[sub["final_regime_confirmed"].astype(str).str.startswith(state_prefix)].copy()
    x = sub[["GS10"]].to_numpy()

    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=0, n_init=20)
    gmm.fit(x)
    gmm_labels = gmm.predict(x)
    order = np.argsort(gmm.means_.ravel())
    remap = {old: new for new, old in enumerate(order)}
    ordered_means = gmm.means_[order].copy().reshape(-1, 1)
    ordered_covs = gmm.covariances_[order].copy().reshape(-1, 1, 1)
    ordered_weights = gmm.weights_[order].copy()

    sorted_labels = np.array([remap[i] for i in gmm_labels], dtype=int)
    startprob = np.bincount(sorted_labels, minlength=3).astype(float)
    startprob = startprob / startprob.sum()
    transmat = np.full((3, 3), 1e-3)
    for a, b in zip(sorted_labels[:-1], sorted_labels[1:]):
        transmat[a, b] += 1.0
    transmat = transmat / transmat.sum(axis=1, keepdims=True)

    hmm = GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=500,
        random_state=0,
        init_params="",
        params="st",
    )
    hmm.startprob_ = startprob
    hmm.transmat_ = transmat
    hmm.means_ = ordered_means
    hmm.covars_ = np.maximum(ordered_covs, 1e-4)
    hmm.fit(x)
    states = hmm.predict(x)

    name_map = {0: "LOW", 1: "MID", 2: "HIGH"}
    sub["hmm_state"] = states
    sub["hmm_state_name"] = sub["hmm_state"].map(name_map)

    summary = (
        sub.groupby("hmm_state_name")
        .agg(
            n_days=("GS10", "size"),
            mean_gs10=("GS10", "mean"),
            median_gs10=("GS10", "median"),
            min_gs10=("GS10", "min"),
            max_gs10=("GS10", "max"),
        )
        .reset_index()
    )
    summary["model_mean"] = summary["hmm_state_name"].map({name_map[i]: float(hmm.means_.ravel()[i]) for i in range(3)})
    summary["model_weight"] = summary["hmm_state_name"].map({name_map[i]: float((states == i).mean()) for i in range(3)})
    summary["family"] = family

    summary.to_csv(TABLE_DIR / f"{table_prefix}_hmm_summary.csv", index=False)
    sub[["date", "GS10", "final_regime_confirmed", "hmm_state_name"]].to_csv(TABLE_DIR / f"{table_prefix}_hmm_daily_states.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.histplot(data=sub, x="GS10", hue="final_regime_confirmed", stat="count", common_norm=False, bins=28, ax=axes[0], alpha=0.45)
    for xline, color, label in band_lines:
        axes[0].axvline(xline, color=color, linestyle="--", linewidth=1.8, label=label)
    axes[0].set_title(f"{family} GS10 full-sample distribution with hysteresis bands")
    axes[0].legend(frameon=False, fontsize=8)

    sns.kdeplot(data=sub, x="GS10", hue="final_regime_confirmed", common_norm=False, ax=axes[1], linewidth=2.0)
    for state, color in [("LOW", "#ef4444"), ("MID", "#f59e0b"), ("HIGH", "#2563eb")]:
        state_mean = float(summary.loc[summary["hmm_state_name"].eq(state), "model_mean"].iloc[0])
        axes[1].axvline(state_mean, color=color, linestyle=":", linewidth=2.0, label=f"HMM {state} mean={state_mean:.2f}")
    for xline, color, _label in band_lines:
        axes[1].axvline(xline, color=color, linestyle="--", linewidth=1.2, alpha=0.8)
    axes[1].set_title(f"{family} GS10 KDE with 3-state HMM means")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fig_name, dpi=170)
    plt.close(fig)
    return summary


def build_term_spread_structure_outputs(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    def fit_hmm_summary(
        sub: pd.DataFrame,
        value_col: str,
        n_components: int,
        state_names: dict[int, str],
        fig_name: str,
        table_prefix: str,
        threshold_lines: list[tuple[float, str, str]],
    ) -> pd.DataFrame:
        x = sub[[value_col]].to_numpy()
        gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=0, n_init=20)
        gmm.fit(x)
        gmm_labels = gmm.predict(x)
        order = np.argsort(gmm.means_.ravel())
        remap = {old: new for new, old in enumerate(order)}
        ordered_means = gmm.means_[order].copy().reshape(-1, 1)
        ordered_covs = gmm.covariances_[order].copy().reshape(-1, 1, 1)
        ordered_weights = gmm.weights_[order].copy()

        sorted_labels = np.array([remap[i] for i in gmm_labels], dtype=int)
        startprob = np.bincount(sorted_labels, minlength=n_components).astype(float)
        startprob = startprob / startprob.sum()
        transmat = np.full((n_components, n_components), 1e-3)
        for a, b in zip(sorted_labels[:-1], sorted_labels[1:]):
            transmat[a, b] += 1.0
        transmat = transmat / transmat.sum(axis=1, keepdims=True)

        hmm = GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            n_iter=500,
            random_state=0,
            init_params="",
            params="st",
        )
        hmm.startprob_ = startprob
        hmm.transmat_ = transmat
        hmm.means_ = ordered_means
        hmm.covars_ = np.maximum(ordered_covs, 1e-4)
        hmm.fit(x)
        states = hmm.predict(x)

        work = sub.copy()
        work["hmm_state"] = states
        work["hmm_state_name"] = work["hmm_state"].map(state_names)
        summary = (
            work.groupby("hmm_state_name")
            .agg(
                n_days=(value_col, "size"),
                mean_value=(value_col, "mean"),
                median_value=(value_col, "median"),
                min_value=(value_col, "min"),
                max_value=(value_col, "max"),
            )
            .reset_index()
        )
        summary["model_mean"] = summary["hmm_state_name"].map(
            {state_names[i]: float(ordered_means[i, 0]) for i in range(n_components)}
        )
        summary["model_weight"] = summary["hmm_state_name"].map(
            {state_names[i]: float(ordered_weights[i]) for i in range(n_components)}
        )
        summary = summary.sort_values("model_mean").reset_index(drop=True)

        work[["date", value_col, "hmm_state_name"]].to_csv(TABLE_DIR / f"{table_prefix}_hmm_daily_states.csv", index=False)
        summary.to_csv(TABLE_DIR / f"{table_prefix}_hmm_summary.csv", index=False)

        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
        sns.histplot(data=work, x=value_col, hue="hmm_state_name", stat="count", common_norm=False, bins=32, ax=axes[0], alpha=0.45)
        axes[0].set_title(f"{table_prefix} histogram")
        for xline, color, _label in threshold_lines:
            axes[0].axvline(xline, color=color, linestyle="--", linewidth=1.2, alpha=0.8)

        sns.kdeplot(data=work, x=value_col, hue="hmm_state_name", common_norm=False, ax=axes[1], linewidth=2.0)
        for state, color in zip(summary["hmm_state_name"], ["#ef4444", "#f59e0b", "#2563eb"][: len(summary)]):
            state_mean = float(summary.loc[summary["hmm_state_name"].eq(state), "model_mean"].iloc[0])
            axes[1].axvline(state_mean, color=color, linestyle=":", linewidth=2.0, label=f"{state} mean={state_mean:.2f}")
        for xline, color, _label in threshold_lines:
            axes[1].axvline(xline, color=color, linestyle="--", linewidth=1.2, alpha=0.8)
        axes[1].set_title(f"{table_prefix} KDE + HMM")
        axes[1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / fig_name, dpi=170)
        plt.close(fig)
        return summary

    full_sample = panel[["date", "TERM_SPREAD_10Y_1Y"]].dropna().copy()
    full_summary = fit_hmm_summary(
        full_sample,
        "TERM_SPREAD_10Y_1Y",
        n_components=3,
        state_names={0: "INVERTED", 1: "FLAT_ZONE", 2: "STEEP_ZONE"},
        fig_name="term_spread_full_sample_kde_hmm.png",
        table_prefix="term_spread_full_sample",
        threshold_lines=[
            (-0.10, "#ef4444", "flat->inverted -0.10"),
            (0.10, "#f97316", "inverted->flat 0.10"),
            (1.00, "#2563eb", "steep->flat 1.00"),
            (1.20, "#7c3aed", "flat->steep 1.20"),
        ],
    )

    non_inverted = full_sample.loc[full_sample["TERM_SPREAD_10Y_1Y"] >= 0].copy()
    non_inverted_summary = fit_hmm_summary(
        non_inverted,
        "TERM_SPREAD_10Y_1Y",
        n_components=2,
        state_names={0: "LOW_POSITIVE", 1: "HIGH_POSITIVE"},
        fig_name="term_spread_non_inverted_kde_hmm.png",
        table_prefix="term_spread_non_inverted",
        threshold_lines=[
            (-0.10, "#ef4444", "flat->inverted -0.10"),
            (0.10, "#f97316", "inverted->flat 0.10"),
            (1.00, "#2563eb", "steep->flat 1.00"),
            (1.20, "#7c3aed", "flat->steep 1.20"),
        ],
    )
    return full_summary, non_inverted_summary


def term_spread_readme_section(full_summary: pd.DataFrame, non_inverted_summary: pd.DataFrame) -> str:
    def state_line(df: pd.DataFrame, state: str) -> str:
        row = df.loc[df["hmm_state_name"].eq(state)].iloc[0]
        return f"- {state}: mean term spread `{float(row['model_mean']):.2f}`, sample weight `{float(row['model_weight']):.1%}`"

    return f"""

## Term Spread Structure and Outer Hysteresis

We also re-ran single-variable HMM + KDE diagnostics directly on `DGS10 - DGS1` to validate the outer `INVERTED / FLAT / STEEP` transition logic.

### Full-sample 3-state term spread HMM

{state_line(full_summary, "INVERTED")}
{state_line(full_summary, "FLAT_ZONE")}
{state_line(full_summary, "STEEP_ZONE")}

### Non-inverted 2-state term spread HMM

{state_line(non_inverted_summary, "LOW_POSITIVE")}
{state_line(non_inverted_summary, "HIGH_POSITIVE")}

Interpretation:

1. The full-sample HMM confirms that negative term spread is structurally distinct from the positive-rate states.
2. Inside the non-inverted sample, KDE/HMM still shows a low-positive zone and a high-positive zone rather than one single smooth cloud.
3. That internal structure is why the outer rule now uses hysteresis instead of a single cutoff:
   - `FLAT -> INVERTED = -0.10`
   - `INVERTED -> FLAT = 0.10`
   - `FLAT -> STEEP = 1.20`
   - `STEEP -> FLAT = 1.00`

The corresponding mainline figures are:

- `results/main_pipeline_final/figures/term_spread_full_sample_kde_hmm.png`
- `results/main_pipeline_final/figures/term_spread_non_inverted_kde_hmm.png`
"""


def stress_trigger_readme_section(flat_gs10_summary: pd.DataFrame, steep_gs10_summary: pd.DataFrame) -> str:
    def state_line(df: pd.DataFrame, state: str) -> str:
        row = df.loc[df["hmm_state_name"].eq(state)].iloc[0]
        return f"- {state}: mean GS10 `{float(row['model_mean']):.2f}`, sample weight `{float(row['model_weight']):.1%}`"

    return f"""

## GS10 Internal Structure and Regime Buffers

The final regime builder now diagnoses `FLAT` and `STEEP` separately with full-sample `GS10` KDE + HMM outputs. This is not a single global rate split. It is two separate internal-structure diagnostics that support low/mid/high classification within each family.

### FLAT GS10 structure

{state_line(flat_gs10_summary, "LOW")}
{state_line(flat_gs10_summary, "MID")}
{state_line(flat_gs10_summary, "HIGH")}

- Hysteresis bands:
  - `MID -> LOW = 1.1`
  - `LOW -> MID = 1.3`
  - `HIGH -> MID = 3.4`
  - `MID -> HIGH = 3.6`

### STEEP GS10 structure

{state_line(steep_gs10_summary, "LOW")}
{state_line(steep_gs10_summary, "MID")}
{state_line(steep_gs10_summary, "HIGH")}

- Hysteresis bands:
  - `MID -> LOW = 2.0`
  - `LOW -> MID = 2.3`
  - `HIGH -> MID = 3.0`
  - `MID -> HIGH = 3.2`

- All regime transitions use `2-day confirm`.

This does two things:

1. It reflects the internal structure visible in `GS10` inside `FLAT` and `STEEP`, rather than forcing both into one coarse threshold rule.
2. It reduces turnover by using hysteresis bands instead of single-point internal splits.

The corresponding mainline figures are:

- `results/main_pipeline_final/figures/flat_gs10_kde_hmm.png`
- `results/main_pipeline_final/figures/steep_gs10_kde_hmm.png`
"""


def main() -> None:
    ensure_dirs()
    panel, perf = build_final_source_only_panel()
    panel = panel.copy()
    panel["heatmap_cross_state"] = panel.apply(heatmap_cross_state, axis=1)
    panel["pure_cross_state"] = panel.apply(pure_cross_state, axis=1)
    panel["oil_stress_cross_state"] = panel.apply(oil_stress_cross_state, axis=1)
    panel["oil_nonrisk_rate_cross_state"] = panel.apply(oil_nonrisk_rate_cross_state, axis=1)
    panel.to_csv(OUT / "daily_backtest_panel.csv", index=False)
    panel.to_csv(TABLE_DIR / "daily_backtest_panel.csv", index=False)
    panel.to_csv(TABLE_DIR / "final_daily_panel.csv", index=False)

    display_perf = perf.loc[perf["strategy"].isin(DISPLAY_STRATEGIES)].copy()
    display_perf.to_csv(TABLE_DIR / "strategy_performance_comparison.csv", index=False)

    cross_behavior = asset_behavior(panel, "heatmap_cross_state")
    pure_behavior = asset_behavior(panel, "pure_cross_state")
    regime_behavior = asset_behavior(panel, "final_regime_confirmed")
    oil_stress_behavior = asset_behavior(panel, "oil_stress_cross_state")
    oil_nonrisk_panel = panel.loc[panel["oil_nonrisk_rate_cross_state"].notna()].copy()
    oil_nonrisk_behavior = asset_behavior(oil_nonrisk_panel, "oil_nonrisk_rate_cross_state")
    cross_behavior.to_csv(TABLE_DIR / "cross_state_asset_behavior.csv", index=False)
    pure_behavior.to_csv(TABLE_DIR / "pure_cross_state_asset_behavior.csv", index=False)
    regime_behavior.to_csv(TABLE_DIR / "regime_asset_behavior.csv", index=False)
    oil_stress_behavior.to_csv(TABLE_DIR / "oil_stress_asset_behavior.csv", index=False)
    oil_nonrisk_behavior.to_csv(TABLE_DIR / "oil_nonrisk_rate_asset_behavior.csv", index=False)
    crisis_performance(panel).to_csv(TABLE_DIR / "crisis_window_performance.csv", index=False)
    (
        panel.groupby("pure_cross_state")
        .agg(days=("date", "size"))
        .reset_index()
        .sort_values("days", ascending=False)
        .to_csv(TABLE_DIR / "pure_cross_state_day_counts.csv", index=False)
    )
    oil_stress_coverage = (
        panel.groupby(["oil_level_regime", "final_state"])
        .agg(days=("date", "size"))
        .reset_index()
        .assign(coverage=lambda df_: df_["days"] / len(panel))
    )
    oil_nonrisk_coverage = (
        oil_nonrisk_panel.groupby(["oil_level_regime", "final_regime_confirmed"])
        .agg(days=("date", "size"))
        .reset_index()
        .assign(coverage=lambda df_: df_["days"] / len(panel))
    )
    oil_stress_coverage.to_csv(TABLE_DIR / "oil_stress_coverage.csv", index=False)
    oil_nonrisk_coverage.to_csv(TABLE_DIR / "oil_nonrisk_rate_coverage.csv", index=False)

    panel[["date"] + [f"{FINAL}_weight_{asset}" for asset in ASSETS]].to_csv(TABLE_DIR / "final_daily_weights.csv", index=False)
    panel[
        [
            "date",
            f"{FINAL}_return",
            f"{FINAL}_nav",
            f"{FINAL}_drawdown",
            f"{FINAL}_turnover",
            f"{FINAL}_transaction_cost",
            "final_state",
            "final_allocation_state",
            "trigger_lock_active_locks",
            "final_regime_confirmed",
            "heatmap_cross_state",
            "pure_cross_state",
        ]
    ].to_csv(TABLE_DIR / "final_daily_returns.csv", index=False)

    plot_outputs(panel)
    plot_performance_bars(display_perf)
    plot_case_studies(panel)

    plot_asset_behavior_heatmap(
        cross_behavior,
        "heatmap_cross_state",
        "cross_state_asset_behavior_heatmap.png",
        "Asset Annualized Return by Final Cross-State Allocation",
    )
    plot_asset_behavior_heatmap(
        cross_behavior,
        "heatmap_cross_state",
        "cross_state_asset_sharpe_heatmap.png",
        "Asset Sharpe Ratio by Final Cross-State Allocation",
        value_col="Sharpe",
        value_label="Sharpe ratio",
        value_format="number",
    )
    plot_asset_behavior_heatmap(
        pure_behavior,
        "pure_cross_state",
        "pure_regime_stress_asset_behavior_heatmap.png",
        "Asset Annualized Return by Pure Regime x Stress Block",
    )
    plot_asset_behavior_heatmap(
        pure_behavior,
        "pure_cross_state",
        "pure_regime_stress_asset_sharpe_heatmap.png",
        "Asset Sharpe Ratio by Pure Regime x Stress Block",
        value_col="Sharpe",
        value_label="Sharpe ratio",
        value_format="number",
    )
    plot_asset_behavior_heatmap(
        regime_behavior,
        "final_regime_confirmed",
        "regime_asset_behavior_heatmap.png",
        "Asset Annualized Return by Final Regime",
    )
    plot_asset_behavior_heatmap(
        regime_behavior,
        "final_regime_confirmed",
        "regime_asset_sharpe_heatmap.png",
        "Asset Sharpe Ratio by Final Regime",
        value_col="Sharpe",
        value_label="Sharpe ratio",
        value_format="number",
        column_order=RATE_STATE_ORDER,
    )
    oil_stress_order = [f"{short_oil_label(oil)}__{stress}" for oil in OIL_LEVEL_ORDER for stress in ["NON_RISK", "FULL_RISK"]]
    plot_asset_behavior_heatmap(
        oil_stress_behavior,
        "oil_stress_cross_state",
        "oil_stress_asset_behavior_heatmap.png",
        "Asset Annualized Return by Oil Level x Stress",
        column_order=oil_stress_order,
    )
    plot_asset_behavior_heatmap(
        oil_stress_behavior,
        "oil_stress_cross_state",
        "oil_stress_asset_sharpe_heatmap.png",
        "Asset Sharpe Ratio by Oil Level x Stress",
        value_col="Sharpe",
        value_label="Sharpe ratio",
        value_format="number",
        column_order=oil_stress_order,
    )
    oil_nonrisk_rate_order = [
        f"{rate}__{short_oil_label(oil)}"
        for rate in RATE_STATE_ORDER
        for oil in OIL_LEVEL_ORDER
    ]
    plot_asset_behavior_heatmap(
        oil_nonrisk_behavior,
        "oil_nonrisk_rate_cross_state",
        "oil_nonrisk_rate_asset_behavior_heatmap.png",
        "Asset Annualized Return by Non-Stress Rate Regime x Oil Level",
        column_order=oil_nonrisk_rate_order,
    )
    plot_asset_behavior_heatmap(
        oil_nonrisk_behavior,
        "oil_nonrisk_rate_cross_state",
        "oil_nonrisk_rate_asset_sharpe_heatmap.png",
        "Asset Sharpe Ratio by Non-Stress Rate Regime x Oil Level",
        value_col="Sharpe",
        value_label="Sharpe ratio",
        value_format="number",
        column_order=oil_nonrisk_rate_order,
    )
    plot_coverage_heatmap(
        oil_stress_coverage,
        "oil_level_regime",
        "final_state",
        "coverage",
        "oil_stress_coverage_heatmap.png",
        "Coverage of Oil Level x Stress",
        OIL_LEVEL_ORDER,
        ["NON_RISK", "FULL_RISK"],
    )
    plot_coverage_heatmap(
        oil_nonrisk_coverage,
        "oil_level_regime",
        "final_regime_confirmed",
        "coverage",
        "oil_nonrisk_rate_coverage_heatmap.png",
        "Coverage of Non-Stress Rate Regime x Oil Level",
        OIL_LEVEL_ORDER,
        RATE_STATE_ORDER,
    )

    flat_gs10_summary = build_gs10_structure_outputs(
        panel,
        family="FLAT",
        state_prefix="FLAT_",
        band_lines=[
            (1.1, "#ef4444", "mid->low 1.1"),
            (1.3, "#f97316", "low->mid 1.3"),
            (3.4, "#2563eb", "high->mid 3.4"),
            (3.6, "#7c3aed", "mid->high 3.6"),
        ],
        fig_name="flat_gs10_kde_hmm.png",
        table_prefix="flat_gs10",
    )
    steep_gs10_summary = build_gs10_structure_outputs(
        panel,
        family="STEEP",
        state_prefix="STEEP_",
        band_lines=[
            (2.0, "#ef4444", "mid->low 2.0"),
            (2.3, "#f97316", "low->mid 2.3"),
            (3.0, "#2563eb", "high->mid 3.0"),
            (3.2, "#7c3aed", "mid->high 3.2"),
        ],
        fig_name="steep_gs10_kde_hmm.png",
        table_prefix="steep_gs10",
    )
    term_spread_full_summary, term_spread_non_inverted_summary = build_term_spread_structure_outputs(panel)

    sample_start = panel["date"].min().strftime("%Y-%m-%d")
    sample_end = panel["date"].max().strftime("%Y-%m-%d")
    final_perf = display_perf.loc[display_perf["strategy"].eq(FINAL)].iloc[0]

    readme = f"""# Final Source-Only Strategy Outputs

This folder is generated from `data/raw` and `data/processed` only using the canonical source-only settings.

Sample window:

- Start: `{sample_start}`
- End: `{sample_end}`
- Trading days: `{len(panel)}`

Final strategy headline metrics:

- CAGR: `{final_perf['CAGR']:.2%}`
- Annualized volatility: `{final_perf['annualized_volatility']:.2%}`
- Sharpe: `{final_perf['Sharpe']:.3f}`
- Max drawdown: `{final_perf['MaxDD']:.2%}`
- Final equity multiple: `{final_perf['final_equity']:.2f}x`

Final display strategies:

- `SPY_BUY_HOLD`: always 100% SPY.
- `SPY_CASH_TIMING`: SPY in non-risk, CASH in trigger-lock stress; uses the same VIX/CREDIT anchor state machine as the final hedge strategy.
- `FINAL_REGIME_HEDGE_TRIGGER_LOCK`: final hedge allocation with buffered term-spread transitions, oil-aware flat sleeves, expanded commodity ETFs, and the original mainline VIX/CREDIT stress trigger.

Key design choices:

- Credit spread is daily `DBAA - DAAA`, filled to the trading calendar before feature construction.
- Macro regime has no `NEUTRAL`: `INVERTED`, `FLAT`, `STEEP`, with outer term-spread hysteresis:
  - `FLAT -> INVERTED = -0.10`
  - `INVERTED -> FLAT = 0.10`
  - `FLAT -> STEEP = 1.20`
  - `STEEP -> FLAT = 1.00`
  - outer and inner regime transitions both use `2-day confirm`
- `FLAT` uses buffered `GS10` low/mid/high bands:
  - `MID -> LOW = 1.1`
  - `LOW -> MID = 1.3`
  - `HIGH -> MID = 3.4`
  - `MID -> HIGH = 3.6`
- `STEEP` uses buffered `GS10` low/mid/high bands:
  - `MID -> LOW = 2.0`
  - `LOW -> MID = 2.3`
  - `HIGH -> MID = 3.0`
  - `MID -> HIGH = 3.2`
- `STEEP_LOW_RATE` does not allow native credit entries.
- Carry-over stress is shown explicitly in the cross-state heatmap. `STEEP_LOW_RATE_STRESS` has no native trigger, but if an active stress period carries into `STEEP_LOW_RATE`, it remains a stress sleeve and is analyzed separately.
- `CASH_return` uses geometric daily DTB3.
- Oil level uses a `252-day` oil MA with three persistent states:
  - `HIGH`: entry at `+20%`, exit at `+5%`
  - `LOW`: entry at `-20%`, exit at `-10%`
  - confirmation is `10 trading days`
- When oil is `HIGH`, flat sleeves remove `GOLD` and `DBB` from the candidate set.
- Expanded asset set: `SPY`, `GOLD`, `IEF`, `DBC`, `DBB`, `DBA`, `CASH`.
- Inverse-vol window is 90 trading days.
- Transaction cost uses 10 bps one-way.

Final allocation settings:

- `FLAT_LOW_RATE_NORMAL`: SPY / DBC / DBB inverse-vol, but oil `HIGH` removes `DBB`.
- `FLAT_MID_RATE_NORMAL`: SPY / GOLD inverse-vol.
- `FLAT_LOWMID_RATE_STRESS`: 100% CASH.
- `FLAT_HIGH_RATE_NORMAL`: 40% IEF + 60% (GOLD / DBC inverse-vol), but oil `HIGH` removes `GOLD`.
- `FLAT_HIGH_RATE_STRESS`: 10% DBA + 90% GOLD, but oil `HIGH` collapses this sleeve to 100% DBA.
- `STEEP_LOW_RATE_NORMAL`: 100% SPY.
- `STEEP_LOW_RATE_STRESS`: 100% SPY.
- `STEEP_MID_RATE_NORMAL`: 100% SPY.
- `STEEP_MID_RATE_STRESS`: 100% IEF.
- `STEEP_HIGH_RATE_NORMAL`: SPY / GOLD inverse-vol.
- `STEEP_HIGH_RATE_STRESS`: 100% IEF.
- `INVERTED_NORMAL`: SPY / GOLD inverse-vol.
- `INVERTED_STRESS`: 90% CASH + 10% SPY.

Pure regime x stress outputs are also written into the mainline output set:

- `results/main_pipeline_final/figures/pure_regime_stress_asset_behavior_heatmap.png`
- `results/main_pipeline_final/figures/pure_regime_stress_asset_sharpe_heatmap.png`
- `results/main_pipeline_final/tables/pure_cross_state_asset_behavior.csv`
- `results/main_pipeline_final/figures/oil_level_regime_background.png`
- `results/main_pipeline_final/figures/oil_stress_asset_behavior_heatmap.png`
- `results/main_pipeline_final/figures/oil_nonrisk_rate_asset_behavior_heatmap.png`
- `results/main_pipeline_final/figures/oil_stress_coverage_heatmap.png`
- `results/main_pipeline_final/figures/oil_nonrisk_rate_coverage_heatmap.png`

Main run order:

1. `python scripts/run_final_strategy_source_only.py`
2. `python scripts/08_stress_trigger_diagnostics.py`
3. `python scripts/10_final_report_outputs.py`
4. `python scripts/hard_validate_main_pipeline_source_only.py`

Live dashboard:

- `python scripts/30_generate_live_regime_dashboard.py`
- Output HTML: `results/live_regime_dashboard/live_regime_dashboard.html`
- The dashboard uses the same canonical regime, stress, oil-level, and allocation logic as the main strategy and fetches the latest macro plus asset data with cache fallback.
"""
    readme += stress_trigger_readme_section(flat_gs10_summary, steep_gs10_summary)
    readme += term_spread_readme_section(term_spread_full_summary, term_spread_non_inverted_summary)
    readme += """

## Oil Level Layer

The oil level layer is part of the canonical mainline output. The chart below shows raw oil price with shaded persistent `HIGH` and `LOW` regime backgrounds:

![Oil level regime background](figures/oil_level_regime_background.png)

Cross-state outputs:

- `figures/oil_stress_asset_behavior_heatmap.png`
- `figures/oil_stress_coverage_heatmap.png`
- `figures/oil_nonrisk_rate_asset_behavior_heatmap.png`
- `figures/oil_nonrisk_rate_coverage_heatmap.png`
"""
    (OUT / "README_final_strategy.md").write_text(readme, encoding="utf-8")
    print("PASS source-only final report outputs")
    print(display_perf.to_string(index=False))


if __name__ == "__main__":
    main()
