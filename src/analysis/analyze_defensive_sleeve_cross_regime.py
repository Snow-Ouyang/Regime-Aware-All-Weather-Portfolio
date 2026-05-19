from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
INPUT_PANEL = ROOT / "results" / "reconstructed_regime_asset_behavior" / "reconstructed_regime_panel.csv"
RESULTS_DIR = ROOT / "results" / "defensive_sleeve_cross_regime"
FIGURES_DIR = ROOT / "figures" / "defensive_sleeve_cross_regime"

PANEL_PATH = RESULTS_DIR / "defensive_cross_state_panel.csv"
SAMPLE_SIZE_PATH = RESULTS_DIR / "cross_state_sample_size_table.csv"
PERFORMANCE_PATH = RESULTS_DIR / "asset_performance_by_macro_vix_cross_state.csv"
ANN_RETURN_PATH = RESULTS_DIR / "defensive_annualized_return_by_cross_state.csv"
SHARPE_PATH = RESULTS_DIR / "defensive_sharpe_by_cross_state.csv"
MAX_DD_PATH = RESULTS_DIR / "defensive_max_drawdown_by_cross_state.csv"
POSITIVE_PATH = RESULTS_DIR / "defensive_positive_day_ratio_by_cross_state.csv"
STRESS_PATH = RESULTS_DIR / "defensive_assets_when_vix_stress_by_macro_regime.csv"
NON_CRISIS_PATH = RESULTS_DIR / "assets_when_vix_non_crisis_by_macro_regime.csv"
NON_CRISIS_ANN_RETURN_PATH = RESULTS_DIR / "non_crisis_asset_annualized_return_by_macro_regime.csv"
NON_CRISIS_SHARPE_PATH = RESULTS_DIR / "non_crisis_asset_sharpe_by_macro_regime.csv"
NON_CRISIS_MAX_DD_PATH = RESULTS_DIR / "non_crisis_asset_max_drawdown_by_macro_regime.csv"
RANKING_PATH = RESULTS_DIR / "defensive_asset_ranking_by_cross_state.csv"
SCORE_PATH = RESULTS_DIR / "defensive_asset_composite_score_by_cross_state.csv"
BASKET_PATH = RESULTS_DIR / "defensive_basket_performance_by_cross_state.csv"
STRESS_BASKET_PATH = RESULTS_DIR / "defensive_basket_when_vix_stress_by_macro_regime.csv"
REPORT_PATH = RESULTS_DIR / "DEFENSIVE_SLEEVE_CROSS_REGIME_ANALYSIS.md"

MACRO_ORDER = ["HIGH_INFLATION", "INVERTED", "FLAT", "STEEP"]
VIX_ORDER = ["NORMAL", "WARNING", "STRESS"]
DEFENSIVE_ASSETS = ["CASH", "IEF", "GOLD", "CMDTY_FUT"]
ALL_ASSETS = ["CASH", "IEF", "GOLD", "CMDTY_FUT", "SPY"]
RETURN_COLS = {
    "CASH": "CASH_RETURN",
    "IEF": "IEF_RETURN",
    "GOLD": "GOLD_RETURN",
    "CMDTY_FUT": "CMDTY_FUT_RETURN",
    "SPY": "SPY_RETURN",
}
BASKETS = {
    "CASH_ONLY": {"CASH": 1.0},
    "IEF_ONLY": {"IEF": 1.0},
    "GOLD_ONLY": {"GOLD": 1.0},
    "50_50_IEF_GOLD": {"IEF": 0.5, "GOLD": 0.5},
    "50_50_CASH_IEF": {"CASH": 0.5, "IEF": 0.5},
    "50_50_CASH_GOLD": {"CASH": 0.5, "GOLD": 0.5},
    "EQUAL_CASH_IEF_GOLD": {"CASH": 1 / 3, "IEF": 1 / 3, "GOLD": 1 / 3},
}


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def annualized_return(daily_returns: pd.Series) -> float:
    s = daily_returns.dropna()
    if s.empty:
        return np.nan
    return float((1.0 + s).prod() ** (252.0 / len(s)) - 1.0)


def max_drawdown(daily_returns: pd.Series) -> float:
    s = daily_returns.dropna()
    if s.empty:
        return np.nan
    wealth = (1 + s).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def perf_stats(daily_returns: pd.Series, rf_daily: pd.Series, asset: str) -> dict[str, float]:
    s = daily_returns.dropna()
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
            "cumulative_return_within_state": np.nan,
        }
    rf = rf_daily.loc[s.index]
    excess = s - rf
    ann_ret = annualized_return(s)
    ann_vol = float(s.std(ddof=1) * np.sqrt(252)) if len(s) > 1 else np.nan
    if asset == "CASH":
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
        "max_drawdown": max_drawdown(s),
        "worst_day": float(s.min()),
        "best_day": float(s.max()),
        "positive_day_ratio": float((s > 0).mean()),
        "cumulative_return_within_state": float((1 + s).prod() - 1),
    }


def assign_spell_ids(panel: pd.DataFrame, group_cols: list[str], active_mask: pd.Series | None = None) -> pd.DataFrame:
    out = panel.sort_values("date").reset_index(drop=True).copy()
    if active_mask is None:
        active = pd.Series(True, index=out.index)
    else:
        active = active_mask.reindex(out.index).fillna(False)
    spell_ids: list[float] = []
    current_spell = 0
    prev_active = False
    prev_key = None
    for i, row in out.iterrows():
        is_active = bool(active.iloc[i])
        if not is_active:
            spell_ids.append(np.nan)
            prev_active = False
            prev_key = None
            continue
        key = tuple(row[col] for col in group_cols)
        if (not prev_active) or (key != prev_key):
            current_spell += 1
        spell_ids.append(float(current_spell))
        prev_active = True
        prev_key = key
    out["spell_id"] = spell_ids
    return out


def spell_level_asset_metrics(panel: pd.DataFrame, group_cols: list[str], assets: list[str], active_mask: pd.Series | None = None) -> pd.DataFrame:
    spell_panel = assign_spell_ids(panel, group_cols=group_cols, active_mask=active_mask)
    spell_panel = spell_panel.dropna(subset=["spell_id"]).copy()
    rows = []
    for spell_id, grp in spell_panel.groupby("spell_id", observed=False):
        key_dict = {col: grp.iloc[0][col] for col in group_cols}
        for asset in assets:
            ret_col = RETURN_COLS[asset]
            stats = perf_stats(grp[ret_col], grp["RF_DAILY"], asset)
            rows.append(
                {
                    "spell_id": int(spell_id),
                    **key_dict,
                    "asset": asset,
                    "asset_role": "defensive" if asset in DEFENSIVE_ASSETS else "risk asset reference",
                    "spell_n_obs": stats["n_obs"],
                    **stats,
                    "correlation_with_SPY": np.nan
                    if asset == "SPY"
                    else float(grp[[ret_col, "SPY_RETURN"]].dropna().corr().iloc[0, 1])
                    if len(grp[[ret_col, "SPY_RETURN"]].dropna()) >= 3
                    else np.nan,
                    "average_daily_return_when_SPY_return_negative": float(grp.loc[grp["SPY_RETURN"] < 0, ret_col].dropna().mean())
                    if not grp.loc[grp["SPY_RETURN"] < 0, ret_col].dropna().empty
                    else np.nan,
                    "average_daily_return_when_SPY_drawdown_below_10pct": float(grp.loc[grp["SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"] < -0.10, ret_col].dropna().mean())
                    if "SPY_DRAWDOWN_FROM_PREVIOUS_HIGH" in grp.columns and not grp.loc[grp["SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"] < -0.10, ret_col].dropna().empty
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def aggregate_spell_metrics(spell_metrics: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if spell_metrics.empty:
        return pd.DataFrame()
    metric_cols = [
        "average_daily_return",
        "annualized_return",
        "annualized_volatility",
        "Sharpe",
        "max_drawdown",
        "worst_day",
        "best_day",
        "positive_day_ratio",
        "cumulative_return_within_state",
        "correlation_with_SPY",
        "average_daily_return_when_SPY_return_negative",
        "average_daily_return_when_SPY_drawdown_below_10pct",
    ]
    rows = []
    for keys, grp in spell_metrics.groupby(group_cols + ["asset", "asset_role"], observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols + ["asset", "asset_role"], keys))
        weights = grp["spell_n_obs"].astype(float)
        row["spell_count"] = int(grp["spell_id"].nunique())
        row["n_obs"] = int(grp["spell_n_obs"].sum())
        row["avg_spell_duration"] = float(grp["spell_n_obs"].mean())
        row["LOW_SAMPLE"] = row["n_obs"] < 10
        for col in metric_cols:
            valid = grp[[col, "spell_n_obs"]].dropna()
            if valid.empty:
                row[col] = np.nan
            else:
                row[col] = float(np.average(valid[col], weights=valid["spell_n_obs"]))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_spell_drawdowns(panel: pd.DataFrame, group_cols: list[str], assets: list[str], active_mask: pd.Series | None = None) -> pd.DataFrame:
    spell_metrics = spell_level_asset_metrics(panel, group_cols=group_cols, assets=assets, active_mask=active_mask)
    if spell_metrics.empty:
        return pd.DataFrame(columns=group_cols + ["asset", "spell_count", "avg_spell_duration", "max_drawdown"])
    rows = []
    for keys, grp in spell_metrics.groupby(group_cols + ["asset"], observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols + ["asset"], keys))
        weights = grp["spell_n_obs"].astype(float)
        row["spell_count"] = int(grp["spell_id"].nunique())
        row["avg_spell_duration"] = float(grp["spell_n_obs"].mean())
        valid = grp[["max_drawdown", "spell_n_obs"]].dropna()
        row["max_drawdown"] = float(np.average(valid["max_drawdown"], weights=valid["spell_n_obs"])) if not valid.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def load_panel() -> pd.DataFrame:
    if not INPUT_PANEL.exists():
        raise FileNotFoundError(f"Missing input panel: {INPUT_PANEL}")
    panel = pd.read_csv(INPUT_PANEL)
    panel["date"] = pd.to_datetime(panel["date"])
    required = [
        "date",
        "VIX_LEVEL",
        "CREDIT_SPREAD_BAA_AAA",
        "DGS1",
        "DGS10",
        "TERM_SPREAD_10Y_1Y",
        "RF_DAILY",
        "SPY_RETURN",
        "IEF_RETURN",
        "GOLD_RETURN",
        "CASH_RETURN",
    ]
    missing = [col for col in required if col not in panel.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return panel.sort_values("date").reset_index(drop=True)


def assign_cross_states(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    macro_conditions = [
        (out["CREDIT_SPREAD_BAA_AAA"] > 1.5) & (out["DGS1"] > 5),
        out["TERM_SPREAD_10Y_1Y"] < 0,
        (out["TERM_SPREAD_10Y_1Y"] >= 0) & (out["TERM_SPREAD_10Y_1Y"] < 1),
        out["TERM_SPREAD_10Y_1Y"] >= 1,
    ]
    out["macro_regime"] = np.select(macro_conditions, ["HIGH_INFLATION", "INVERTED", "FLAT", "STEEP"], default="STEEP")
    out["vix_state"] = np.select(
        [
            out["VIX_LEVEL"] < 20,
            (out["VIX_LEVEL"] >= 20) & (out["VIX_LEVEL"] < 25),
            out["VIX_LEVEL"] >= 25,
        ],
        ["NORMAL", "WARNING", "STRESS"],
        default="NORMAL",
    )
    out["vix_stress_flag"] = out["VIX_LEVEL"] >= 25
    out["cross_state"] = out["macro_regime"] + "_" + out["vix_state"]
    return out


def sample_size_table(panel: pd.DataFrame) -> pd.DataFrame:
    table = (
        panel.groupby(["macro_regime", "vix_state"], observed=False)
        .size()
        .reset_index(name="n_obs")
        .sort_values(["macro_regime", "vix_state"])
    )
    table["LOW_SAMPLE"] = table["n_obs"] < 10
    return table


def compute_asset_performance(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dd = aggregate_spell_drawdowns(panel, group_cols=["macro_regime", "vix_state", "cross_state"], assets=ALL_ASSETS)
    for (macro, vix, cross), grp in panel.groupby(["macro_regime", "vix_state", "cross_state"], observed=False):
        for asset in ALL_ASSETS:
            stats = perf_stats(grp[RETURN_COLS[asset]], grp["RF_DAILY"], asset)
            dd_row = dd.loc[(dd["macro_regime"] == macro) & (dd["vix_state"] == vix) & (dd["cross_state"] == cross) & (dd["asset"] == asset)]
            if not dd_row.empty:
                stats["max_drawdown"] = float(dd_row["max_drawdown"].iloc[0])
                stats["spell_count"] = int(dd_row["spell_count"].iloc[0])
                stats["avg_spell_duration"] = float(dd_row["avg_spell_duration"].iloc[0])
            else:
                stats["spell_count"] = np.nan
                stats["avg_spell_duration"] = np.nan
            stats.update(
                {
                    "macro_regime": macro,
                    "vix_state": vix,
                    "cross_state": cross,
                    "asset": asset,
                    "asset_role": "defensive" if asset in DEFENSIVE_ASSETS else "risk asset reference",
                    "LOW_SAMPLE": stats["n_obs"] < 10,
                    "correlation_with_SPY": np.nan
                    if asset == "SPY"
                    else float(grp[[RETURN_COLS[asset], "SPY_RETURN"]].dropna().corr().iloc[0, 1])
                    if len(grp[[RETURN_COLS[asset], "SPY_RETURN"]].dropna()) >= 3
                    else np.nan,
                }
            )
            rows.append(stats)
    return pd.DataFrame(rows)


def pivot_metric(perf: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        perf.loc[perf["asset"].isin(ALL_ASSETS)]
        .pivot_table(index="asset", columns="cross_state", values=metric, aggfunc="first")
        .reindex(index=ALL_ASSETS)
    )


def plot_heatmap(data: pd.DataFrame, title: str, path: Path, fmt: str = ".2f", center: float | None = None) -> None:
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(max(10, 0.7 * data.shape[1]), 4.8))
    sns.heatmap(data, annot=True, fmt=fmt, cmap="RdBu_r", center=center, linewidths=0.5, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_sample_size_heatmap(samples: pd.DataFrame) -> None:
    pivot = samples.pivot(index="macro_regime", columns="vix_state", values="n_obs").reindex(index=MACRO_ORDER, columns=VIX_ORDER)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues", linewidths=0.5, ax=ax)
    ax.set_title("Cross-State Sample Size")
    ax.set_xlabel("VIX state")
    ax.set_ylabel("Macro regime")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cross_state_sample_size_heatmap.png", dpi=180)
    plt.close(fig)


def focus_vix_stress(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    stress = panel.loc[panel["vix_stress_flag"]].copy()
    dd = aggregate_spell_drawdowns(panel, group_cols=["macro_regime"], assets=ALL_ASSETS, active_mask=panel["vix_stress_flag"])
    for macro, grp in stress.groupby("macro_regime", observed=False):
        for asset in ALL_ASSETS:
            ret_col = RETURN_COLS[asset]
            stats = perf_stats(grp[ret_col], grp["RF_DAILY"], asset)
            dd_row = dd.loc[(dd["macro_regime"] == macro) & (dd["asset"] == asset)]
            if not dd_row.empty:
                stats["max_drawdown"] = float(dd_row["max_drawdown"].iloc[0])
                stats["spell_count"] = int(dd_row["spell_count"].iloc[0])
                stats["avg_spell_duration"] = float(dd_row["avg_spell_duration"].iloc[0])
            else:
                stats["spell_count"] = np.nan
                stats["avg_spell_duration"] = np.nan
            spy_down = grp.loc[grp["SPY_RETURN"] < 0, ret_col].dropna()
            row = {
                "macro_regime": macro,
                "asset": asset,
                **stats,
                "average_daily_return_when_SPY_return_negative": float(spy_down.mean()) if not spy_down.empty else np.nan,
                "correlation_with_SPY": float(grp[[ret_col, "SPY_RETURN"]].dropna().corr().iloc[0, 1])
                if len(grp[[ret_col, "SPY_RETURN"]].dropna()) >= 3
                else np.nan,
                "LOW_SAMPLE": stats["n_obs"] < 10,
            }
            if "SPY_DRAWDOWN_FROM_PREVIOUS_HIGH" in grp.columns:
                dd_ret = grp.loc[grp["SPY_DRAWDOWN_FROM_PREVIOUS_HIGH"] < -0.10, ret_col].dropna()
                row["average_daily_return_when_SPY_drawdown_below_10pct"] = float(dd_ret.mean()) if not dd_ret.empty else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def focus_vix_non_crisis(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    non_crisis = panel.loc[panel["VIX_LEVEL"] < 25].copy()
    dd = aggregate_spell_drawdowns(panel, group_cols=["macro_regime"], assets=ALL_ASSETS, active_mask=panel["VIX_LEVEL"] < 25)
    for macro, grp in non_crisis.groupby("macro_regime", observed=False):
        for asset in ALL_ASSETS:
            ret_col = RETURN_COLS[asset]
            stats = perf_stats(grp[ret_col], grp["RF_DAILY"], asset)
            dd_row = dd.loc[(dd["macro_regime"] == macro) & (dd["asset"] == asset)]
            if not dd_row.empty:
                stats["max_drawdown"] = float(dd_row["max_drawdown"].iloc[0])
                stats["spell_count"] = int(dd_row["spell_count"].iloc[0])
                stats["avg_spell_duration"] = float(dd_row["avg_spell_duration"].iloc[0])
            else:
                stats["spell_count"] = np.nan
                stats["avg_spell_duration"] = np.nan
            rows.append(
                {
                    "macro_regime": macro,
                    "asset": asset,
                    "asset_role": "defensive" if asset in DEFENSIVE_ASSETS else "risk asset reference",
                    **stats,
                    "correlation_with_SPY": np.nan
                    if asset == "SPY"
                    else float(grp[[ret_col, "SPY_RETURN"]].dropna().corr().iloc[0, 1])
                    if len(grp[[ret_col, "SPY_RETURN"]].dropna()) >= 3
                    else np.nan,
                    "LOW_SAMPLE": stats["n_obs"] < 10,
                }
            )
    return pd.DataFrame(rows)


def pivot_macro_metric(perf: pd.DataFrame, metric: str) -> pd.DataFrame:
    return perf.pivot_table(index="asset", columns="macro_regime", values=metric, aggfunc="first").reindex(index=ALL_ASSETS, columns=MACRO_ORDER)


def rank_to_score(series: pd.Series, ascending: bool) -> pd.Series:
    valid = series.dropna()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    if valid.empty:
        return out
    ranks = valid.rank(method="average", ascending=ascending)
    if len(valid) == 1:
        out.loc[valid.index] = 1.0
    else:
        out.loc[valid.index] = 1 - (ranks - 1) / (len(valid) - 1)
    return out


def defensive_ranking(perf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    defensive = perf.loc[perf["asset"].isin(ALL_ASSETS)].copy()
    ranking_rows = []
    score_rows = []
    for cross, grp in defensive.groupby("cross_state", observed=False):
        g = grp.set_index("asset").copy()
        metrics = {
            "annualized_return": False,
            "Sharpe": False,
            "max_drawdown": False,
            "worst_day": False,
            "correlation_with_SPY": True,
        }
        score_parts = []
        for metric, ascending in metrics.items():
            rank = g[metric].rank(method="average", ascending=ascending)
            score = rank_to_score(g[metric], ascending=ascending)
            score_parts.append(score.rename(f"{metric}_score"))
            for asset in g.index:
                ranking_rows.append(
                    {
                        "cross_state": cross,
                        "asset": asset,
                        "metric": metric,
                        "rank": rank.loc[asset] if asset in rank.index else np.nan,
                        "score": score.loc[asset] if asset in score.index else np.nan,
                    }
                )
        scores = pd.concat(score_parts, axis=1)
        scores["composite_score"] = scores.mean(axis=1, skipna=True)
        scores["cross_state"] = cross
        score_rows.append(scores.reset_index())
    return pd.DataFrame(ranking_rows), pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()


def basket_returns(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel[["date", "macro_regime", "vix_state", "cross_state", "SPY_RETURN", "RF_DAILY", "vix_stress_flag"]].copy()
    for basket, weights in BASKETS.items():
        ret = pd.Series(0.0, index=panel.index)
        valid = pd.Series(True, index=panel.index)
        for asset, weight in weights.items():
            col = RETURN_COLS[asset]
            ret = ret + panel[col] * weight
            valid = valid & panel[col].notna()
        out[basket] = ret.where(valid)
    return out


def basket_spell_metrics(panel: pd.DataFrame, stress_only: bool = False) -> pd.DataFrame:
    data = basket_returns(panel)
    if stress_only:
        group_cols = ["macro_regime"]
        active_mask = data["vix_stress_flag"]
    else:
        group_cols = ["macro_regime", "vix_state", "cross_state"]
        active_mask = None
    spell_panel = assign_spell_ids(data, group_cols=group_cols, active_mask=active_mask)
    spell_panel = spell_panel.dropna(subset=["spell_id"]).copy()
    rows = []
    for spell_id, grp in spell_panel.groupby("spell_id", observed=False):
        key_dict = {col: grp.iloc[0][col] for col in group_cols}
        for basket in BASKETS:
            stats = perf_stats(grp[basket], grp["RF_DAILY"], basket)
            rows.append(
                {
                    "spell_id": int(spell_id),
                    **key_dict,
                    "basket": basket,
                    "spell_n_obs": stats["n_obs"],
                    **stats,
                    "correlation_with_SPY": float(grp[[basket, "SPY_RETURN"]].dropna().corr().iloc[0, 1]) if len(grp[[basket, "SPY_RETURN"]].dropna()) >= 3 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def aggregate_basket_spell_metrics(spell_metrics: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if spell_metrics.empty:
        return pd.DataFrame()
    metric_cols = [
        "average_daily_return",
        "annualized_return",
        "annualized_volatility",
        "Sharpe",
        "max_drawdown",
        "worst_day",
        "best_day",
        "positive_day_ratio",
        "cumulative_return_within_state",
        "correlation_with_SPY",
    ]
    rows = []
    for keys, grp in spell_metrics.groupby(group_cols + ["basket"], observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols + ["basket"], keys))
        row["spell_count"] = int(grp["spell_id"].nunique())
        row["n_obs"] = int(grp["spell_n_obs"].sum())
        row["avg_spell_duration"] = float(grp["spell_n_obs"].mean())
        row["LOW_SAMPLE"] = row["n_obs"] < 10
        for col in metric_cols:
            valid = grp[[col, "spell_n_obs"]].dropna()
            row[col] = float(np.average(valid[col], weights=valid["spell_n_obs"])) if not valid.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def compute_basket_performance(panel: pd.DataFrame, stress_only: bool = False) -> pd.DataFrame:
    data = basket_returns(panel)
    if stress_only:
        data = data.loc[data["vix_stress_flag"]].copy()
        group_cols = ["macro_regime"]
        active_mask = panel["vix_stress_flag"]
    else:
        group_cols = ["macro_regime", "vix_state", "cross_state"]
        active_mask = None
    spell_metrics = basket_spell_metrics(panel, stress_only=stress_only)
    dd = aggregate_basket_spell_metrics(spell_metrics, group_cols=group_cols)[group_cols + ["basket", "spell_count", "avg_spell_duration", "max_drawdown"]]
    rows = []
    for keys, grp in data.groupby(group_cols, observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_dict = dict(zip(group_cols, keys))
        for basket in BASKETS:
            stats = perf_stats(grp[basket], grp["RF_DAILY"], basket)
            dd_row = dd.loc[np.logical_and.reduce([dd[col] == key_dict[col] for col in group_cols]) & (dd["basket"] == basket)] if not dd.empty else pd.DataFrame()
            if not dd_row.empty:
                stats["max_drawdown"] = float(dd_row["max_drawdown"].iloc[0])
                stats["spell_count"] = int(dd_row["spell_count"].iloc[0])
                stats["avg_spell_duration"] = float(dd_row["avg_spell_duration"].iloc[0])
            else:
                stats["spell_count"] = np.nan
                stats["avg_spell_duration"] = np.nan
            stats.update(key_dict)
            stats["basket"] = basket
            stats["LOW_SAMPLE"] = stats["n_obs"] < 10
            stats["correlation_with_SPY"] = float(grp[[basket, "SPY_RETURN"]].dropna().corr().iloc[0, 1]) if len(grp[[basket, "SPY_RETURN"]].dropna()) >= 3 else np.nan
            rows.append(stats)
    return pd.DataFrame(rows)


def plot_stress_heatmaps(stress: pd.DataFrame, scores: pd.DataFrame) -> None:
    for metric, name, fmt, center in [
        ("annualized_return", "stress_only_defensive_return_heatmap.png", ".2%", 0),
        ("max_drawdown", "stress_only_defensive_drawdown_heatmap.png", ".2%", None),
    ]:
        pivot = stress.pivot(index="macro_regime", columns="asset", values=metric).reindex(index=MACRO_ORDER, columns=ALL_ASSETS)
        plot_heatmap(pivot, metric.replace("_", " ").title() + " in VIX Stress", FIGURES_DIR / name, fmt=fmt, center=center)
    stress_states = [f"{macro}_STRESS" for macro in MACRO_ORDER]
    score_pivot = scores.loc[scores["cross_state"].isin(stress_states)].copy()
    if not score_pivot.empty:
        score_pivot["macro_regime"] = score_pivot["cross_state"].str.replace("_STRESS", "", regex=False)
        pivot = score_pivot.pivot(index="macro_regime", columns="asset", values="composite_score").reindex(index=MACRO_ORDER, columns=ALL_ASSETS)
        plot_heatmap(pivot, "Defensive Composite Score in VIX Stress", FIGURES_DIR / "stress_only_defensive_composite_score_heatmap.png", fmt=".2f")


def plot_non_crisis_heatmaps(non_crisis: pd.DataFrame) -> None:
    if non_crisis.empty:
        return
    for metric, path, fmt, center in [
        ("annualized_return", "non_crisis_asset_return_heatmap_by_macro.png", ".2%", 0),
        ("Sharpe", "non_crisis_asset_sharpe_heatmap_by_macro.png", ".2f", 0),
        ("max_drawdown", "non_crisis_asset_drawdown_heatmap_by_macro.png", ".2%", None),
    ]:
        pivot = pivot_macro_metric(non_crisis, metric)
        plot_heatmap(pivot, f"Asset {metric.replace('_', ' ').title()} by Macro Regime, VIX < 25", FIGURES_DIR / path, fmt=fmt, center=center)


def plot_basket_stress_bars(stress_baskets: pd.DataFrame) -> None:
    if stress_baskets.empty:
        return
    for metric, path, title in [
        ("Sharpe", "stress_defensive_basket_sharpe_by_macro.png", "Defensive Basket Sharpe in VIX Stress"),
        ("max_drawdown", "stress_defensive_basket_drawdown_by_macro.png", "Defensive Basket Max Drawdown in VIX Stress"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(data=stress_baskets, x="macro_regime", y=metric, hue="basket", ax=ax)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / path, dpi=180)
        plt.close(fig)


def write_report(
    samples: pd.DataFrame,
    stress: pd.DataFrame,
    non_crisis: pd.DataFrame,
    scores: pd.DataFrame,
    stress_baskets: pd.DataFrame,
) -> None:
    lines = [
        "# Defensive Sleeve Cross-Regime Analysis",
        "",
        "## Purpose",
        "",
        "This diagnostic does not build allocation weights. It studies where defensive capital historically performed better when VIX stress appeared under different observable macro states.",
        "",
        "## Method",
        "",
        "- Macro regime is reconstructed without allowing VIX to override the macro state.",
        "- VIX state is crossed with the macro regime: `NORMAL`, `WARNING`, `STRESS`, or `UNKNOWN` when VIX history is unavailable.",
        "- The cross-state comparison includes Cash, IEF, GLD-based Gold, GD=F-based commodity futures proxy, and SPY. SPY remains the risk-asset reference, but it is now included in the main output tables and heatmaps.",
        "- Performance metrics are now computed spell-by-spell on contiguous regime/state windows, then aggregated within each state. This avoids stitched-state drawdown distortion.",
        "- Cells with fewer than 10 valid asset observations are flagged as low sample.",
        "",
        "## VIX Stress Findings",
        "",
    ]
    for macro in MACRO_ORDER:
        sub = stress.loc[stress["macro_regime"] == macro].copy()
        if sub.empty:
            continue
        best_ret = sub.sort_values("annualized_return", ascending=False).iloc[0]
        best_sharpe = sub.sort_values("Sharpe", ascending=False).iloc[0]
        best_dd = sub.sort_values("max_drawdown", ascending=False).iloc[0]
        score_sub = scores.loc[scores["cross_state"] == f"{macro}_STRESS"].sort_values("composite_score", ascending=False)
        best_score = score_sub.iloc[0] if not score_sub.empty else None
        lines.extend(
            [
                f"### {macro} + VIX_STRESS",
                "",
                f"- Highest annualized return: `{best_ret['asset']}` ({best_ret['annualized_return']:.2%}, n={int(best_ret['n_obs'])}).",
                f"- Best Sharpe: `{best_sharpe['asset']}` ({best_sharpe['Sharpe']:.2f}).",
                f"- Lowest drawdown: `{best_dd['asset']}` ({best_dd['max_drawdown']:.2%}).",
                f"- Composite diagnostic leader: `{best_score['asset']}` ({best_score['composite_score']:.2f})." if best_score is not None else "- Composite diagnostic leader: unavailable.",
                "",
            ]
        )
    lines.extend(
        [
            "## Non-Crisis Findings",
            "",
            "The non-crisis sample is defined as `VIX_LEVEL < 25`. It keeps the macro regime dimension intact and compares SPY, Cash, IEF, GLD-based Gold, and GD=F-based commodity futures proxy without treating VIX stress as a separate override state.",
            "",
        ]
    )
    for macro in MACRO_ORDER:
        sub = non_crisis.loc[non_crisis["macro_regime"] == macro].copy()
        if sub.empty:
            continue
        best_ret = sub.sort_values("annualized_return", ascending=False).iloc[0]
        best_sharpe = sub.sort_values("Sharpe", ascending=False).iloc[0]
        best_dd = sub.sort_values("max_drawdown", ascending=False).iloc[0]
        lines.extend(
            [
                f"- `{macro}`: highest return `{best_ret['asset']}` ({best_ret['annualized_return']:.2%}), best Sharpe `{best_sharpe['asset']}` ({best_sharpe['Sharpe']:.2f}), lowest drawdown `{best_dd['asset']}` ({best_dd['max_drawdown']:.2%}).",
            ]
        )
    ief_top = scores.loc[(scores["cross_state"].str.endswith("_STRESS"))].sort_values(["cross_state", "composite_score"], ascending=[True, False]).groupby("cross_state").head(1)
    ief_states = ief_top.loc[ief_top["asset"] == "IEF", "cross_state"].tolist()
    lines.extend(
        [
            "## IEF Interpretation",
            "",
            f"IEF is top-ranked in these VIX stress cross-states: {', '.join(ief_states) if ief_states else 'none in the current composite ranking'}. This does not mean IEF is useless; the basket table should be used to check whether it lowers volatility or SPY correlation when combined with Cash or Gold.",
            "",
            "## Gold Interpretation",
            "",
            "Gold can dominate return in some stress states, but its standalone drawdown and day-to-day volatility should be checked before treating it as the only defensive destination.",
            "",
            "## Cash Interpretation",
            "",
            "Cash is the cleanest diversifier and tends to become more attractive when short rates are high. It can dominate Sharpe or drawdown rankings even when another asset has higher return.",
            "",
            "## Basket Diagnostics",
            "",
        ]
    )
    if not stress_baskets.empty:
        for macro, grp in stress_baskets.groupby("macro_regime", observed=False):
            best = grp.sort_values("Sharpe", ascending=False).iloc[0]
            lines.append(f"- `{macro}`: best stress basket by Sharpe is `{best['basket']}` ({best['Sharpe']:.2f}, n={int(best['n_obs'])}).")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- These are diagnostics, not final allocation rules.",
            "- Some cross states have small samples, especially after requiring ETF return history.",
            "- A final strategy still needs no-look-ahead implementation, confirmation/hysteresis, transaction costs, and robustness checks.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = assign_cross_states(load_panel())
    panel.to_csv(PANEL_PATH, index=False)

    samples = sample_size_table(panel)
    samples.to_csv(SAMPLE_SIZE_PATH, index=False)
    plot_sample_size_heatmap(samples)

    perf = compute_asset_performance(panel)
    perf.to_csv(PERFORMANCE_PATH, index=False)

    ann = pivot_metric(perf, "annualized_return")
    sharpe = pivot_metric(perf, "Sharpe")
    max_dd = pivot_metric(perf, "max_drawdown")
    positive = pivot_metric(perf, "positive_day_ratio")
    ann.to_csv(ANN_RETURN_PATH)
    sharpe.to_csv(SHARPE_PATH)
    max_dd.to_csv(MAX_DD_PATH)
    positive.to_csv(POSITIVE_PATH)
    plot_heatmap(ann, "Defensive Asset Annualized Return by Cross-State", FIGURES_DIR / "defensive_asset_return_heatmap_cross_state.png", fmt=".2%", center=0)
    plot_heatmap(sharpe, "Defensive Asset Sharpe by Cross-State", FIGURES_DIR / "defensive_asset_sharpe_heatmap_cross_state.png", fmt=".2f", center=0)
    plot_heatmap(max_dd, "Defensive Asset Max Drawdown by Cross-State", FIGURES_DIR / "defensive_asset_drawdown_heatmap_cross_state.png", fmt=".2%")

    stress = focus_vix_stress(panel)
    stress.to_csv(STRESS_PATH, index=False)
    non_crisis = focus_vix_non_crisis(panel)
    non_crisis.to_csv(NON_CRISIS_PATH, index=False)
    non_crisis_ann = pivot_macro_metric(non_crisis, "annualized_return")
    non_crisis_sharpe = pivot_macro_metric(non_crisis, "Sharpe")
    non_crisis_dd = pivot_macro_metric(non_crisis, "max_drawdown")
    non_crisis_ann.to_csv(NON_CRISIS_ANN_RETURN_PATH)
    non_crisis_sharpe.to_csv(NON_CRISIS_SHARPE_PATH)
    non_crisis_dd.to_csv(NON_CRISIS_MAX_DD_PATH)
    plot_non_crisis_heatmaps(non_crisis)
    ranking, scores = defensive_ranking(perf)
    ranking.to_csv(RANKING_PATH, index=False)
    scores.to_csv(SCORE_PATH, index=False)
    plot_stress_heatmaps(stress, scores)

    baskets = compute_basket_performance(panel, stress_only=False)
    stress_baskets = compute_basket_performance(panel, stress_only=True)
    baskets.to_csv(BASKET_PATH, index=False)
    stress_baskets.to_csv(STRESS_BASKET_PATH, index=False)
    plot_basket_stress_bars(stress_baskets)

    write_report(samples, stress, non_crisis, scores, stress_baskets)

    print("Macro regime x VIX state sample counts:")
    print(samples.pivot(index="macro_regime", columns="vix_state", values="n_obs").fillna(0).astype(int).to_string())

    stress_best_return = stress.sort_values("annualized_return", ascending=False).groupby("macro_regime").head(1)
    stress_best_sharpe = stress.sort_values("Sharpe", ascending=False).groupby("macro_regime").head(1)
    stress_best_dd = stress.sort_values("max_drawdown", ascending=False).groupby("macro_regime").head(1)
    print("Best defensive asset by annualized return for each VIX_STRESS macro regime:")
    print(stress_best_return[["macro_regime", "asset", "annualized_return", "n_obs", "LOW_SAMPLE"]].to_string(index=False))
    print("Best defensive asset by Sharpe for each VIX_STRESS macro regime:")
    print(stress_best_sharpe[["macro_regime", "asset", "Sharpe", "n_obs", "LOW_SAMPLE"]].to_string(index=False))
    print("Lowest drawdown defensive asset for each VIX_STRESS macro regime:")
    print(stress_best_dd[["macro_regime", "asset", "max_drawdown", "n_obs", "LOW_SAMPLE"]].to_string(index=False))
    stress_top = scores.loc[scores["cross_state"].str.endswith("_STRESS")].sort_values(["cross_state", "composite_score"], ascending=[True, False]).groupby("cross_state").head(1)
    ief_states = stress_top.loc[stress_top["asset"] == "IEF", "cross_state"].tolist()
    print(f"IEF top-ranked in stress cross-states: {', '.join(ief_states) if ief_states else 'none'}")
    non_crisis_best_return = non_crisis.sort_values("annualized_return", ascending=False).groupby("macro_regime").head(1)
    print("Best asset by annualized return for each non-crisis macro regime (VIX < 25):")
    print(non_crisis_best_return[["macro_regime", "asset", "annualized_return", "n_obs", "LOW_SAMPLE"]].to_string(index=False))
    for path in [
        PANEL_PATH,
        SAMPLE_SIZE_PATH,
        PERFORMANCE_PATH,
        ANN_RETURN_PATH,
        SHARPE_PATH,
        MAX_DD_PATH,
        POSITIVE_PATH,
        STRESS_PATH,
        NON_CRISIS_PATH,
        NON_CRISIS_ANN_RETURN_PATH,
        NON_CRISIS_SHARPE_PATH,
        NON_CRISIS_MAX_DD_PATH,
        RANKING_PATH,
        SCORE_PATH,
        BASKET_PATH,
        STRESS_BASKET_PATH,
        REPORT_PATH,
        FIGURES_DIR / "defensive_asset_return_heatmap_cross_state.png",
        FIGURES_DIR / "defensive_asset_sharpe_heatmap_cross_state.png",
        FIGURES_DIR / "defensive_asset_drawdown_heatmap_cross_state.png",
        FIGURES_DIR / "stress_only_defensive_return_heatmap.png",
        FIGURES_DIR / "stress_only_defensive_drawdown_heatmap.png",
        FIGURES_DIR / "stress_only_defensive_composite_score_heatmap.png",
        FIGURES_DIR / "stress_defensive_basket_sharpe_by_macro.png",
        FIGURES_DIR / "stress_defensive_basket_drawdown_by_macro.png",
        FIGURES_DIR / "non_crisis_asset_return_heatmap_by_macro.png",
        FIGURES_DIR / "non_crisis_asset_sharpe_heatmap_by_macro.png",
        FIGURES_DIR / "non_crisis_asset_drawdown_heatmap_by_macro.png",
        FIGURES_DIR / "cross_state_sample_size_heatmap.png",
    ]:
        if path.exists():
            print(f"Saved: {path}")


if __name__ == "__main__":
    main()
