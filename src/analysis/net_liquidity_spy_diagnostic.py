from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "macro" / "dollar"
OUTPUT_DIR = ROOT / "results" / "net_liquidity_spy_diagnostic"
FIGURE_DIR = ROOT / "figures" / "net_liquidity_spy_diagnostic"

SPY_CANDIDATES = [
    ROOT / "results" / "spy_cash_stress_recovery_timing" / "daily_backtest_panel.csv",
    ROOT / "results" / "spy_cash_stress_recovery_with_credit" / "daily_backtest_panel.csv",
]

CONFIG = {
    "raw_dir": str(RAW_DIR),
    "output_dir": str(OUTPUT_DIR),
    "figure_dir": str(FIGURE_DIR),
    "trading_days_per_year": 252,
    "weeks_to_days": {"4W": 20, "13W": 65, "26W": 130},
    "zscore_window": 252,
    "case_study_windows": {
        "2018Q4": ["2018-10-01", "2019-01-31"],
        "2019": ["2019-01-01", "2019-12-31"],
        "COVID_2020": ["2020-02-01", "2020-06-30"],
        "2022": ["2021-11-01", "2023-03-31"],
        "2023": ["2023-01-01", "2023-12-31"],
        "2024_2026": ["2024-01-01", "2026-12-31"],
    },
}

PANEL_OUT = OUTPUT_DIR / "net_liquidity_daily_panel.csv"
UNIT_OUT = OUTPUT_DIR / "unit_check_summary.csv"
CORR_OUT = OUTPUT_DIR / "correlation_summary.csv"
LEAD_LAG_OUT = OUTPUT_DIR / "lead_lag_correlation_summary.csv"
ROLLING_CORR_OUT = OUTPUT_DIR / "rolling_correlation_panel.csv"
STATE_COUNTS_OUT = OUTPUT_DIR / "liquidity_state_counts.csv"
STATE_PERF_OUT = OUTPUT_DIR / "spy_performance_by_liquidity_state.csv"
MACRO_STATE_PERF_OUT = OUTPUT_DIR / "spy_performance_by_macro_regime_and_liquidity_state.csv"
REPORT_OUT = OUTPUT_DIR / "NET_LIQUIDITY_SPY_DIAGNOSTIC.md"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_csv_series(path: Path, preferred_name: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=[".", "NA", "", " "])
    date_candidates = [c for c in df.columns if "date" in c.lower()]
    date_col = date_candidates[0] if date_candidates else df.columns[0]
    value_col = None
    if preferred_name:
        for col in df.columns:
            if col.lower() == preferred_name.lower():
                value_col = col
                break
    if value_col is None:
        non_date = [c for c in df.columns if c != date_col]
        numeric_counts = {}
        for col in non_date:
            numeric_counts[col] = pd.to_numeric(df[col], errors="coerce").notna().sum()
        value_col = max(numeric_counts, key=numeric_counts.get)
    out = df[[date_col, value_col]].rename(columns={date_col: "date", value_col: preferred_name or value_col})
    out["date"] = pd.to_datetime(out["date"])
    out[out.columns[1]] = pd.to_numeric(out[out.columns[1]], errors="coerce")
    out = out.sort_values("date").drop_duplicates("date")
    out[out.columns[1]] = out[out.columns[1]].ffill()
    return out


def load_spy_panel() -> pd.DataFrame:
    for path in SPY_CANDIDATES:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "date" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"])
        if "spy_price" not in df.columns and "SPY_BUY_HOLD_nav" in df.columns:
            df["spy_price"] = pd.to_numeric(df["SPY_BUY_HOLD_nav"], errors="coerce")
        if "spy_price" not in df.columns:
            continue
        df["spy_price"] = pd.to_numeric(df["spy_price"], errors="coerce")
        if "spy_daily_return" not in df.columns:
            df["spy_daily_return"] = df["spy_price"].pct_change()
        else:
            df["spy_daily_return"] = pd.to_numeric(df["spy_daily_return"], errors="coerce")
        keep = [
            "date",
            "spy_price",
            "spy_daily_return",
            "macro_regime_confirmed",
            "stress_entry_signal",
        ]
        return df[[c for c in keep if c in df.columns]].sort_values("date").drop_duplicates("date").reset_index(drop=True)
    raise FileNotFoundError("Could not locate a usable SPY daily panel.")


def merge_liquidity_components(spy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = {
        "WALCL": RAW_DIR / "WALCL.csv",
        "WDTGAL": RAW_DIR / "WDTGAL.csv",
        "RRPONTSYD": RAW_DIR / "RRPONTSYD.csv",
        "WRESBAL": RAW_DIR / "WRESBAL.csv",
        "DTWEXBGS": RAW_DIR / "dollar index.csv",
    }
    loaded = {}
    unit_rows = []
    for name, path in files.items():
        if not path.exists():
            warnings.warn(f"Missing liquidity file: {path}")
            continue
        series = load_csv_series(path, name)
        loaded[name] = series
        val = pd.to_numeric(series[name], errors="coerce")
        unit_rows.append(
            {
                "series": name,
                "source_file": str(path.relative_to(ROOT)),
                "raw_min": float(val.min()),
                "raw_median": float(val.median()),
                "raw_max": float(val.max()),
                "raw_start": series["date"].min().date().isoformat(),
                "raw_end": series["date"].max().date().isoformat(),
                "unit_assumption": "millions USD" if name in ["WALCL", "WDTGAL", "WRESBAL"] else "billions USD" if name == "RRPONTSYD" else "index",
            }
        )
    out = spy.sort_values("date").copy()
    for name, series in loaded.items():
        out = pd.merge_asof(out.sort_values("date"), series.sort_values("date"), on="date", direction="backward")
    return out, pd.DataFrame(unit_rows)


def standardize_units(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["WALCL_BN"] = pd.to_numeric(out.get("WALCL"), errors="coerce") / 1000.0
    out["TGA_BN"] = pd.to_numeric(out.get("WDTGAL"), errors="coerce") / 1000.0
    out["RRP_BN"] = pd.to_numeric(out.get("RRPONTSYD"), errors="coerce")
    out["RESERVES_BN"] = pd.to_numeric(out.get("WRESBAL"), errors="coerce") / 1000.0
    out["DOLLAR_INDEX"] = pd.to_numeric(out.get("DTWEXBGS"), errors="coerce")
    out["NET_LIQ_BN"] = out["WALCL_BN"] - out["TGA_BN"] - out["RRP_BN"]
    out["NET_LIQ_EX_RRP_BN"] = out["WALCL_BN"] - out["TGA_BN"]
    return out


def _zscore(s: pd.Series, window: int) -> pd.Series:
    roll = s.rolling(window, min_periods=window)
    return (s - roll.mean()) / roll.std(ddof=1).replace(0, np.nan)


def build_net_liquidity_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    components = {
        "WALCL": "WALCL_BN",
        "TGA": "TGA_BN",
        "RRP": "RRP_BN",
        "RESERVES": "RESERVES_BN",
        "NET_LIQ": "NET_LIQ_BN",
    }
    for label, col in components.items():
        for tenor, days in CONFIG["weeks_to_days"].items():
            out[f"{label}_{tenor}_CHANGE"] = out[col] - out[col].shift(days)
    out["NET_LIQ_EX_RRP_13W_CHANGE"] = out["NET_LIQ_EX_RRP_BN"] - out["NET_LIQ_EX_RRP_BN"].shift(CONFIG["weeks_to_days"]["13W"])
    out["NET_LIQ_13W_PCT_CHANGE"] = out["NET_LIQ_BN"].pct_change(CONFIG["weeks_to_days"]["13W"], fill_method=None)
    out["WALCL_13W_PCT_CHANGE"] = out["WALCL_BN"].pct_change(CONFIG["weeks_to_days"]["13W"], fill_method=None)
    out["RESERVES_13W_PCT_CHANGE"] = out["RESERVES_BN"].pct_change(CONFIG["weeks_to_days"]["13W"], fill_method=None)
    for label in ["NET_LIQ", "WALCL", "RESERVES", "TGA", "RRP"]:
        out[f"{label}_13W_CHANGE_Z_252D"] = _zscore(out[f"{label}_13W_CHANGE"], CONFIG["zscore_window"])
    out["SPY_NAV"] = out["spy_price"] / out["spy_price"].iloc[0]
    out["SPY_13W_RETURN"] = out["spy_price"] / out["spy_price"].shift(CONFIG["weeks_to_days"]["13W"]) - 1.0
    out["SPY_26W_RETURN"] = out["spy_price"] / out["spy_price"].shift(CONFIG["weeks_to_days"]["26W"]) - 1.0
    out["SPY_DRAWDOWN_FROM_HIGH"] = out["spy_price"] / out["spy_price"].cummax() - 1.0
    for h in [21, 42, 63, 126]:
        out[f"forward_spy_return_{h}d"] = out["spy_price"].shift(-h) / out["spy_price"] - 1.0
    return out


def _corr(df: pd.DataFrame, x: str, y: str) -> float:
    sub = df[[x, y]].dropna()
    return float(sub[x].corr(sub[y])) if len(sub) > 2 else np.nan


def compute_correlation_analysis(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    same_pairs = [
        ("NET_LIQ_13W_CHANGE", "SPY_13W_RETURN"),
        ("NET_LIQ_26W_CHANGE", "SPY_26W_RETURN"),
        ("WALCL_13W_CHANGE", "SPY_13W_RETURN"),
        ("RESERVES_13W_CHANGE", "SPY_13W_RETURN"),
        ("TGA_13W_CHANGE", "SPY_13W_RETURN"),
        ("RRP_13W_CHANGE", "SPY_13W_RETURN"),
    ]
    corr = pd.DataFrame([{"x": x, "y": y, "correlation": _corr(panel, x, y)} for x, y in same_pairs])
    lead_rows = []
    for x in ["NET_LIQ_13W_CHANGE", "WALCL_13W_CHANGE", "RESERVES_13W_CHANGE", "TGA_13W_CHANGE", "RRP_13W_CHANGE"]:
        for h in [21, 42, 63, 126]:
            lead_rows.append({"liquidity_feature": x, "forward_horizon_days": h, "correlation": _corr(panel, x, f"forward_spy_return_{h}d")})
    lead = pd.DataFrame(lead_rows)
    roll = panel[["date", "NET_LIQ_13W_CHANGE", "SPY_13W_RETURN"]].copy()
    roll["rolling_corr_252d"] = roll["NET_LIQ_13W_CHANGE"].rolling(252, min_periods=126).corr(roll["SPY_13W_RETURN"])
    roll["rolling_corr_504d"] = roll["NET_LIQ_13W_CHANGE"].rolling(504, min_periods=252).corr(roll["SPY_13W_RETURN"])
    return corr, lead, roll


def build_exploratory_liquidity_states(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["liquidity_state_sign"] = np.where(out["NET_LIQ_13W_CHANGE"] > 0, "EASING", "TIGHTENING")
    out["liquidity_state_z"] = np.select(
        [out["NET_LIQ_13W_CHANGE_Z_252D"] > 0.5, out["NET_LIQ_13W_CHANGE_Z_252D"] < -0.5],
        ["EASING", "TIGHTENING"],
        default="NEUTRAL",
    )
    out.loc[out["NET_LIQ_13W_CHANGE_Z_252D"].isna(), "liquidity_state_z"] = np.nan
    out["fed_balance_sheet_state"] = np.where(out["WALCL_13W_CHANGE"] > 0, "EXPANDING", "CONTRACTING")
    out.loc[out["WALCL_13W_CHANGE"].isna(), "fed_balance_sheet_state"] = np.nan
    return out


def _perf_stats(sub: pd.DataFrame) -> dict:
    s = sub["spy_daily_return"].dropna()
    if s.empty:
        return {"n_obs": 0, "annualized_return": np.nan, "annualized_volatility": np.nan, "Sharpe": np.nan, "max_drawdown": np.nan, "positive_day_ratio": np.nan, "avg_13W_forward_return": np.nan, "avg_26W_forward_return": np.nan}
    ann = (1 + s).prod() ** (252 / len(s)) - 1
    vol = s.std(ddof=1) * np.sqrt(252)
    sharpe = s.mean() / s.std(ddof=1) * np.sqrt(252) if s.std(ddof=1) != 0 else np.nan
    wealth = (1 + s).cumprod()
    mdd = (wealth / wealth.cummax() - 1).min()
    return {
        "n_obs": len(s),
        "annualized_return": ann,
        "annualized_volatility": vol,
        "Sharpe": sharpe,
        "max_drawdown": mdd,
        "positive_day_ratio": (s > 0).mean(),
        "avg_13W_forward_return": sub["forward_spy_return_63d"].mean(),
        "avg_26W_forward_return": sub["forward_spy_return_126d"].mean(),
    }


def compute_state_performance(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counts = []
    for col in ["liquidity_state_sign", "liquidity_state_z", "fed_balance_sheet_state"]:
        vc = panel[col].value_counts(dropna=False).rename_axis("state").reset_index(name="n_obs")
        vc.insert(0, "state_type", col)
        counts.append(vc)
    perf_rows = []
    for col in ["liquidity_state_sign", "liquidity_state_z", "fed_balance_sheet_state"]:
        for state, sub in panel.dropna(subset=[col]).groupby(col):
            perf_rows.append({"state_type": col, "state": state, **_perf_stats(sub)})
    macro_rows = []
    if "macro_regime_confirmed" in panel.columns:
        for (regime, state), sub in panel.dropna(subset=["macro_regime_confirmed", "liquidity_state_z"]).groupby(["macro_regime_confirmed", "liquidity_state_z"]):
            stats = _perf_stats(sub)
            stats["avg_forward_return_63d"] = sub["forward_spy_return_63d"].mean()
            stats["avg_forward_return_126d"] = sub["forward_spy_return_126d"].mean()
            stats["stress_entry_frequency"] = sub["stress_entry_signal"].mean() if "stress_entry_signal" in sub.columns else np.nan
            macro_rows.append({"macro_regime_confirmed": regime, "liquidity_state_z": state, **stats})
    return pd.concat(counts, ignore_index=True), pd.DataFrame(perf_rows), pd.DataFrame(macro_rows)


def write_case_study_panels(panel: pd.DataFrame) -> None:
    cols = [
        "date",
        "SPY_NAV",
        "SPY_DRAWDOWN_FROM_HIGH",
        "WALCL_BN",
        "TGA_BN",
        "RRP_BN",
        "RESERVES_BN",
        "NET_LIQ_BN",
        "NET_LIQ_13W_CHANGE",
        "NET_LIQ_13W_CHANGE_Z_252D",
        "liquidity_state_z",
        "fed_balance_sheet_state",
        "macro_regime_confirmed",
    ]
    for name, (start, end) in CONFIG["case_study_windows"].items():
        sub = panel[panel["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        sub[[c for c in cols if c in sub.columns]].to_csv(OUTPUT_DIR / f"case_study_{name}.csv", index=False)


def plot_net_liquidity_vs_spy(panel: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax1.plot(panel["date"], panel["SPY_NAV"], color="black", label="SPY NAV")
    ax1.set_yscale("log")
    ax1.set_ylabel("SPY NAV, log")
    ax2 = ax1.twinx()
    ax2.plot(panel["date"], panel["NET_LIQ_BN"], color="#1f77b4", label="Net liquidity")
    ax2.plot(panel["date"], panel["NET_LIQ_EX_RRP_BN"], color="#1f77b4", ls="--", alpha=0.7, label="Net liq ex RRP")
    ax2.set_ylabel("Billion USD")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "net_liquidity_vs_spy_level.png", dpi=160)
    plt.close(fig)


def plot_components(panel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    for col in ["WALCL_BN", "TGA_BN", "RRP_BN", "RESERVES_BN", "NET_LIQ_BN"]:
        ax.plot(panel["date"], panel[col], label=col)
    ax.set_ylabel("Billion USD")
    ax.legend()
    ax.set_title("Dollar liquidity components")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "net_liquidity_components.png", dpi=160)
    plt.close(fig)


def plot_correlations(panel: pd.DataFrame, corr: pd.DataFrame, lead: pd.DataFrame, rolling: pd.DataFrame) -> None:
    sub = panel[["NET_LIQ_13W_CHANGE", "SPY_13W_RETURN"]].dropna()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.regplot(data=sub, x="NET_LIQ_13W_CHANGE", y="SPY_13W_RETURN", scatter_kws={"s": 8, "alpha": 0.35}, line_kws={"color": "red"}, ax=ax)
    ax.set_title(f"Net liquidity 13W change vs SPY 13W return, corr={sub.corr().iloc[0,1]:.2f}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "net_liquidity_change_vs_spy_return.png", dpi=160)
    plt.close(fig)

    nl = lead[lead["liquidity_feature"].eq("NET_LIQ_13W_CHANGE")]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=nl, x="forward_horizon_days", y="correlation", ax=ax, color="#4c78a8")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Lead-lag correlation: net liquidity vs forward SPY")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "lead_lag_correlation_bar.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rolling["date"], rolling["rolling_corr_252d"], label="252d")
    ax.plot(rolling["date"], rolling["rolling_corr_504d"], label="504d")
    ax.axhline(0, color="black", lw=0.8)
    ax.legend()
    ax.set_title("Rolling correlation: net liquidity 13W change vs SPY 13W return")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "rolling_correlation_netliq_spy.png", dpi=160)
    plt.close(fig)


def plot_state_performance(state_perf: pd.DataFrame, macro_perf: pd.DataFrame) -> None:
    sub = state_perf[state_perf["state_type"].eq("liquidity_state_z")]
    if not sub.empty:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, metric in zip(axes, ["annualized_return", "Sharpe", "max_drawdown"]):
            sns.barplot(data=sub, x="state", y=metric, ax=ax)
            ax.set_title(metric)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "spy_performance_by_liquidity_state.png", dpi=160)
        plt.close(fig)
    if not macro_perf.empty:
        heat = macro_perf.pivot_table(index="macro_regime_confirmed", columns="liquidity_state_z", values="annualized_return")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(heat, annot=True, fmt=".2%", cmap="RdYlGn", center=0, ax=ax)
        ax.set_title("SPY annualized return by macro x liquidity state")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "macro_regime_liquidity_heatmap.png", dpi=160)
        plt.close(fig)


def plot_case_studies(panel: pd.DataFrame) -> None:
    for name, (start, end) in CONFIG["case_study_windows"].items():
        if name not in ["2018Q4", "COVID_2020", "2022", "2023"]:
            continue
        sub = panel[panel["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        if sub.empty:
            continue
        fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
        axes[0].plot(sub["date"], sub["SPY_NAV"] / sub["SPY_NAV"].iloc[0], color="black", label="SPY")
        axes[0].plot(sub["date"], sub["SPY_DRAWDOWN_FROM_HIGH"], color="red", alpha=0.7, label="SPY DD")
        axes[0].legend()
        axes[1].plot(sub["date"], sub["NET_LIQ_BN"], label="NET_LIQ_BN")
        axes[1].plot(sub["date"], sub["NET_LIQ_13W_CHANGE"], label="13W change")
        axes[1].legend()
        for col in ["WALCL_BN", "TGA_BN", "RRP_BN"]:
            axes[2].plot(sub["date"], sub[col], label=col)
        axes[2].legend()
        states = {"EASING": 2, "NEUTRAL": 1, "TIGHTENING": 0}
        axes[3].step(sub["date"], sub["liquidity_state_z"].map(states), where="post", label="liquidity state")
        axes[3].set_yticks([0, 1, 2])
        axes[3].set_yticklabels(["TIGHTENING", "NEUTRAL", "EASING"])
        axes[3].legend()
        fig.suptitle(name)
        fig.tight_layout()
        out_name = "case_study_2020_liquidity.png" if name == "COVID_2020" else f"case_study_{name}_liquidity.png"
        fig.savefig(FIGURE_DIR / out_name, dpi=160)
        plt.close(fig)


def write_markdown_report(corr: pd.DataFrame, lead: pd.DataFrame, state_perf: pd.DataFrame, macro_perf: pd.DataFrame, unit_summary: pd.DataFrame) -> None:
    latest = pd.read_csv(PANEL_OUT).iloc[-1]
    lines = [
        "# Net Liquidity / SPY Diagnostic",
        "",
        "## Purpose",
        "",
        "This module constructs dollar net liquidity and compares it with SPY. It does not define final QE/QT/Neutral states and does not implement a strategy.",
        "",
        "## Data and Units",
        "",
        "- WALCL, WDTGAL, and WRESBAL are treated as millions of USD and converted to billion USD by dividing by 1000.",
        "- RRPONTSYD is treated as billion USD and is not divided by 1000.",
        "- `NET_LIQ_BN = WALCL_BN - TGA_BN - RRP_BN`.",
        "- `NET_LIQ_EX_RRP_BN = WALCL_BN - TGA_BN`.",
        "",
        unit_summary.to_markdown(index=False),
        "",
        "## Latest Values",
        "",
        f"- Latest NET_LIQ_BN: {latest['NET_LIQ_BN']:.1f} billion USD.",
        f"- Latest NET_LIQ_EX_RRP_BN: {latest['NET_LIQ_EX_RRP_BN']:.1f} billion USD.",
        "",
        "## Correlation Analysis",
        "",
        corr.to_markdown(index=False),
        "",
        "## Lead-Lag Correlation",
        "",
        lead.to_markdown(index=False),
        "",
        "## Exploratory Liquidity States",
        "",
        "The sign and z-score states are exploratory only. They are not final QE/QT/Neutral definitions.",
        "",
        state_perf.to_markdown(index=False),
        "",
        "## Macro Regime x Liquidity State",
        "",
        macro_perf.to_markdown(index=False) if not macro_perf.empty else "_Macro regime not available._",
        "",
        "## Interpretation",
        "",
        "- Net liquidity can share broad phases with SPY, but the rolling relationship is not assumed stable.",
        "- TGA and RRP matter because they can offset Fed balance sheet expansion.",
        "- The next step should test whether liquidity states improve regime interpretation before using them as trading signals.",
    ]
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    spy = load_spy_panel()
    merged, unit_summary = merge_liquidity_components(spy)
    panel = build_exploratory_liquidity_states(build_net_liquidity_features(standardize_units(merged)))
    corr, lead, rolling = compute_correlation_analysis(panel)
    counts, state_perf, macro_perf = compute_state_performance(panel)

    panel.to_csv(PANEL_OUT, index=False)
    unit_summary.to_csv(UNIT_OUT, index=False)
    corr.to_csv(CORR_OUT, index=False)
    lead.to_csv(LEAD_LAG_OUT, index=False)
    rolling.to_csv(ROLLING_CORR_OUT, index=False)
    counts.to_csv(STATE_COUNTS_OUT, index=False)
    state_perf.to_csv(STATE_PERF_OUT, index=False)
    macro_perf.to_csv(MACRO_STATE_PERF_OUT, index=False)
    write_case_study_panels(panel)
    plot_net_liquidity_vs_spy(panel)
    plot_components(panel)
    plot_correlations(panel, corr, lead, rolling)
    plot_state_performance(state_perf, macro_perf)
    plot_case_studies(panel)
    write_markdown_report(corr, lead, state_perf, macro_perf, unit_summary)

    latest = panel.dropna(subset=["NET_LIQ_BN"]).iloc[-1]
    target_corr = corr.loc[corr["x"].eq("NET_LIQ_13W_CHANGE") & corr["y"].eq("SPY_13W_RETURN"), "correlation"].iloc[0]
    nl_lead = lead[lead["liquidity_feature"].eq("NET_LIQ_13W_CHANGE")].sort_values("correlation", ascending=False).iloc[0]
    z_perf = state_perf[state_perf["state_type"].eq("liquidity_state_z")]
    best_macro = macro_perf.sort_values("annualized_return", ascending=False).iloc[0] if not macro_perf.empty else None
    worst_macro = macro_perf.sort_values("annualized_return", ascending=True).iloc[0] if not macro_perf.empty else None
    print(f"1. Sample range: {panel['date'].iloc[0].date()} to {panel['date'].iloc[-1].date()}")
    for _, row in unit_summary.iterrows():
        print(f"2. {row['series']} raw median {row['raw_median']:.2f}, unit assumption: {row['unit_assumption']}")
    print(f"3. Latest NET_LIQ_BN: {latest['NET_LIQ_BN']:.1f}bn")
    print(f"4. corr(NET_LIQ_13W_CHANGE, SPY_13W_RETURN): {target_corr:.3f}")
    print(f"5. Highest lead-lag horizon: {int(nl_lead['forward_horizon_days'])}d corr {nl_lead['correlation']:.3f}")
    print("6. Liquidity state SPY performance:")
    for _, row in z_perf.iterrows():
        print(f"   {row['state']}: ann {row['annualized_return']:.2%}, Sharpe {row['Sharpe']:.2f}")
    if best_macro is not None:
        print(f"7. Best macro x liquidity: {best_macro['macro_regime_confirmed']} + {best_macro['liquidity_state_z']} ann {best_macro['annualized_return']:.2%}")
        print(f"   Worst macro x liquidity: {worst_macro['macro_regime_confirmed']} + {worst_macro['liquidity_state_z']} ann {worst_macro['annualized_return']:.2%}")
    print(f"8. Obvious positive same-period correlation: {bool(target_corr > 0.15)}")
    print(f"9. Saved outputs: {OUTPUT_DIR} and {FIGURE_DIR}")


if __name__ == "__main__":
    main()
