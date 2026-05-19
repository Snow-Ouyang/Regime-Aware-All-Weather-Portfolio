"""Run independent flat-rate-refined baseline strategy experiment.

This script does not modify the main strategy implementation or overwrite
existing baseline outputs. It reads the validated final panel, keeps the
current mature baseline unchanged outside FLAT regime, and only refines FLAT
allocation by GS10 rate level and stress state.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "flat_rate_refined_baseline"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"

CONFIG = {
    "strategy_name": "flat_rate_refined_baseline",
    "baseline_name": "MATURE_REGIME_HEDGE_FINAL",
    "gs10_threshold": 2.9,
    "confirmation_days": 3,
    "invvol_window": 120,
    "one_way_cost_bps": 5,
    "trading_days_per_year": 252,
}

INPUT_CANDIDATES = [
    "results/09_final_strategy/mature_regime_hedge_final/daily_backtest_panel.csv",
    "results/mature_regime_hedge_final/daily_backtest_panel.csv",
]
GS10_CANDIDATES = [
    "results/drawdown_2015_2016_forensic_diagnostic/forensic_daily_panel.csv",
    "results/06_2015_2016_repair/drawdown_2015_2016_forensic_diagnostic/forensic_daily_panel.csv",
    "results/08_allocation/regime_aware_risk_parity_allocation/daily_backtest_panel.csv",
    "results/regime_aware_risk_parity_allocation/daily_backtest_panel.csv",
]

ASSETS = ["SPY", "GOLD", "CMDTY_FUT", "IEF", "CASH"]
WEIGHT_COLS = {asset: f"{CONFIG['baseline_name']}_weight_{asset}" for asset in ASSETS}
RETURN_COLS = {asset: f"{asset}_return" for asset in ASSETS}


def load_first_existing(candidates: list[str]) -> tuple[pd.DataFrame, Path]:
    for rel in candidates:
        path = ROOT / rel
        if path.exists():
            df = pd.read_csv(path)
            if not df.empty:
                return df, path
    raise FileNotFoundError("No candidate input file found:\n" + "\n".join(candidates))


def load_panel() -> tuple[pd.DataFrame, Path, Path | None]:
    panel, panel_path = load_first_existing(INPUT_CANDIDATES)
    panel["date"] = pd.to_datetime(panel["date"])
    if "GS10" in panel.columns:
        return panel, panel_path, None

    gs10_path_used = None
    for rel in GS10_CANDIDATES:
        path = ROOT / rel
        if not path.exists():
            continue
        gs = pd.read_csv(path, usecols=lambda c: c in {"date", "GS10", "DGS10", "GS10_simple"})
        if "date" not in gs.columns:
            continue
        gs["date"] = pd.to_datetime(gs["date"])
        for col in ["GS10", "DGS10", "GS10_simple"]:
            if col in gs.columns:
                panel = panel.merge(gs[["date", col]].rename(columns={col: "GS10"}), on="date", how="left")
                gs10_path_used = path
                return panel, panel_path, gs10_path_used
    raise KeyError("Missing GS10 in final panel and no GS10 candidate panel could be merged.")


def validate_panel(panel: pd.DataFrame) -> None:
    required = ["date", "macro_regime_confirmed", "timing_state", "GS10", f"{CONFIG['baseline_name']}_return", f"{CONFIG['baseline_name']}_nav"]
    required += list(RETURN_COLS.values())
    required += list(WEIGHT_COLS.values())
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise KeyError("Missing required fields:\n" + "\n".join(f"- {c}" for c in missing))
    if panel["GS10"].isna().all():
        raise ValueError("GS10 is entirely missing after merge.")


def confirm_binary_state(raw_state: pd.Series, consecutive_days: int) -> pd.Series:
    values = raw_state.astype(str).tolist()
    if not values:
        return pd.Series(dtype=str)
    confirmed = values[0]
    candidate = values[0]
    count = 0
    out = []
    for value in values:
        if value == confirmed:
            candidate = confirmed
            count = 0
        else:
            if value == candidate:
                count += 1
            else:
                candidate = value
                count = 1
            if count >= consecutive_days:
                confirmed = candidate
                candidate = confirmed
                count = 0
        out.append(confirmed)
    return pd.Series(out, index=raw_state.index)


def compute_rolling_invvol(panel: pd.DataFrame, pool: list[str]) -> pd.DataFrame:
    returns = panel[[RETURN_COLS[a] for a in pool]].copy()
    vols = returns.rolling(CONFIG["invvol_window"], min_periods=max(20, CONFIG["invvol_window"] // 3)).std().shift(1)
    weights = pd.DataFrame(0.0, index=panel.index, columns=ASSETS)
    for idx in panel.index:
        v = vols.loc[idx].rename({RETURN_COLS[a]: a for a in pool})
        valid = v.replace([np.inf, -np.inf], np.nan).dropna()
        valid = valid[valid > 1e-10]
        if valid.empty:
            alloc = pd.Series(1.0 / len(pool), index=pool)
        else:
            raw = 1.0 / valid
            alloc = raw / raw.sum()
        for asset, weight in alloc.items():
            weights.loc[idx, asset] = float(weight)
    return weights


def build_flat_states(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    raw_rate = np.where(out["GS10"] > CONFIG["gs10_threshold"], "HIGH_RATE_FLAT", "LOW_RATE_FLAT")
    out["flat_rate_raw_state"] = raw_rate
    out["flat_rate_confirmed_state"] = confirm_binary_state(out["flat_rate_raw_state"], CONFIG["confirmation_days"])
    out["flat_stress_state"] = np.where(out["timing_state"].astype(str).str.upper().eq("RISK"), "STRESS", "NORMAL")
    out["flat_refined_state"] = np.where(
        out["macro_regime_confirmed"].astype(str).str.upper().eq("FLAT"),
        "FLAT_" + out["flat_rate_confirmed_state"].str.replace("_FLAT", "", regex=False) + "_" + out["flat_stress_state"],
        "NON_FLAT_BASELINE",
    )
    return out


def target_weights(panel: pd.DataFrame) -> pd.DataFrame:
    baseline_weights = pd.DataFrame({asset: panel[WEIGHT_COLS[asset]].astype(float) for asset in ASSETS}, index=panel.index)
    low_normal_inv = compute_rolling_invvol(panel, ["SPY", "CMDTY_FUT", "GOLD"])
    high_normal_inv = compute_rolling_invvol(panel, ["CMDTY_FUT", "GOLD"])
    targets = baseline_weights.copy()

    low_normal = panel["flat_refined_state"].eq("FLAT_LOW_RATE_NORMAL")
    high_normal = panel["flat_refined_state"].eq("FLAT_HIGH_RATE_NORMAL")
    low_stress = panel["flat_refined_state"].eq("FLAT_LOW_RATE_STRESS")
    high_stress = panel["flat_refined_state"].eq("FLAT_HIGH_RATE_STRESS")

    targets.loc[low_normal, ASSETS] = low_normal_inv.loc[low_normal, ASSETS]
    targets.loc[high_normal, ASSETS] = high_normal_inv.loc[high_normal, ASSETS]
    targets.loc[low_stress, ASSETS] = 0.0
    targets.loc[low_stress, ["GOLD", "IEF"]] = [0.40, 0.60]
    targets.loc[high_stress, ASSETS] = 0.0
    targets.loc[high_stress, ["GOLD", "CASH"]] = [0.20, 0.80]

    # Monthly rebalance / state-change update: carry targets inside unchanged FLAT normal states.
    effective_targets = targets.copy()
    month_start = panel["date"].dt.to_period("M").ne(panel["date"].shift().dt.to_period("M"))
    state_change = panel["flat_refined_state"].ne(panel["flat_refined_state"].shift())
    for i in range(1, len(panel)):
        state = panel.loc[i, "flat_refined_state"]
        normal_state = state in {"FLAT_LOW_RATE_NORMAL", "FLAT_HIGH_RATE_NORMAL"}
        if normal_state and not (month_start.iloc[i] or state_change.iloc[i]):
            effective_targets.loc[i, ASSETS] = effective_targets.loc[i - 1, ASSETS]
    return effective_targets


def run_refined_backtest(panel: pd.DataFrame) -> pd.DataFrame:
    targets = target_weights(panel)
    weights = targets.shift(1)
    weights.iloc[0] = targets.iloc[0]
    # Non-FLAT must follow the baseline exactly.
    non_flat = ~panel["macro_regime_confirmed"].astype(str).str.upper().eq("FLAT")
    for asset in ASSETS:
        weights.loc[non_flat, asset] = panel.loc[non_flat, WEIGHT_COLS[asset]].astype(float)

    returns = panel[[RETURN_COLS[a] for a in ASSETS]].fillna(0.0)
    gross = sum(weights[a] * returns[RETURN_COLS[a]] for a in ASSETS)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    cost = 0.5 * turnover * CONFIG["one_way_cost_bps"] / 10000.0
    net = gross - cost
    nav = (1.0 + net.fillna(0.0)).cumprod()
    dd = nav / nav.cummax() - 1.0

    out = panel[["date", "macro_regime_confirmed", "timing_state", "GS10", "flat_rate_raw_state", "flat_rate_confirmed_state", "flat_stress_state", "flat_refined_state"]].copy()
    for asset in ASSETS:
        out[f"{CONFIG['strategy_name']}_weight_{asset}"] = weights[asset]
    out[f"{CONFIG['strategy_name']}_gross_return"] = gross
    out[f"{CONFIG['strategy_name']}_return"] = net
    out[f"{CONFIG['strategy_name']}_nav"] = nav
    out[f"{CONFIG['strategy_name']}_drawdown"] = dd
    out[f"{CONFIG['strategy_name']}_turnover"] = turnover
    out[f"{CONFIG['strategy_name']}_transaction_cost"] = cost
    out[f"{CONFIG['baseline_name']}_return"] = panel[f"{CONFIG['baseline_name']}_return"]
    out[f"{CONFIG['baseline_name']}_nav"] = panel[f"{CONFIG['baseline_name']}_nav"]
    out[f"{CONFIG['baseline_name']}_drawdown"] = panel[f"{CONFIG['baseline_name']}_drawdown"]
    out[f"{CONFIG['baseline_name']}_turnover"] = panel[f"{CONFIG['baseline_name']}_turnover"]
    out[f"{CONFIG['baseline_name']}_transaction_cost"] = panel[f"{CONFIG['baseline_name']}_transaction_cost"]
    for asset in ASSETS:
        out[f"{CONFIG['baseline_name']}_weight_{asset}"] = panel[WEIGHT_COLS[asset]]
        out[RETURN_COLS[asset]] = panel[RETURN_COLS[asset]]
    return out


def max_drawdown(returns: pd.Series) -> float:
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def worst_rolling_return(returns: pd.Series, window: int) -> float:
    if len(returns.dropna()) < window:
        return np.nan
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    return float((nav / nav.shift(window) - 1.0).min())


def performance_metrics(df: pd.DataFrame, strategy: str, ret_col: str, turnover_col: str, cost_col: str) -> dict[str, float | str]:
    r = df[ret_col].fillna(0.0)
    nav = (1.0 + r).cumprod()
    ann = CONFIG["trading_days_per_year"]
    ann_ret = nav.iloc[-1] ** (ann / len(r)) - 1.0
    ann_vol = r.std() * np.sqrt(ann)
    downside = r[r < 0].std() * np.sqrt(ann)
    sharpe = r.mean() / r.std() * np.sqrt(ann) if r.std() > 1e-12 else np.nan
    sortino = r.mean() / downside * ann if downside and downside > 1e-12 else np.nan
    mdd = float((nav / nav.cummax() - 1.0).min())
    monthly = r.groupby(df["date"].dt.to_period("M")).apply(lambda x: (1.0 + x).prod() - 1.0)
    return {
        "strategy": strategy,
        "start_date": df["date"].iloc[0].date().isoformat(),
        "end_date": df["date"].iloc[-1].date().isoformat(),
        "CAGR": ann_ret,
        "annualized_volatility": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "max_drawdown": mdd,
        "Calmar": ann_ret / abs(mdd) if mdd < 0 else np.nan,
        "win_rate": float((r > 0).mean()),
        "worst_day": float(r.min()),
        "worst_month": float(monthly.min()) if not monthly.empty else np.nan,
        "worst_12m_return": worst_rolling_return(r, 252),
        "turnover": float(df[turnover_col].sum()) if turnover_col in df.columns else np.nan,
        "total_transaction_cost": float(df[cost_col].sum()) if cost_col in df.columns else np.nan,
        "final_equity": float(nav.iloc[-1]),
    }


def conditional_metrics(panel: pd.DataFrame, strategy: str, ret_col: str, group_col: str) -> pd.DataFrame:
    rows = []
    for group, sub in panel.groupby(group_col, dropna=False):
        if len(sub) == 0:
            continue
        m = performance_metrics(sub, strategy, ret_col, f"{strategy}_turnover", f"{strategy}_transaction_cost")
        m[group_col] = group
        m["n_obs"] = len(sub)
        rows.append(m)
    return pd.DataFrame(rows)


def write_outputs(panel: pd.DataFrame, source_path: Path, gs10_source: Path | None) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    name = CONFIG["strategy_name"]
    baseline = CONFIG["baseline_name"]

    panel[["date", "macro_regime_confirmed", "timing_state", "GS10", "flat_rate_raw_state", "flat_rate_confirmed_state", "flat_stress_state", "flat_refined_state", *[f"{name}_weight_{a}" for a in ASSETS], *[f"{baseline}_weight_{a}" for a in ASSETS]]].to_csv(TABLE_DIR / "daily_weights.csv", index=False)
    panel[["date", "macro_regime_confirmed", "timing_state", "flat_refined_state", f"{name}_return", f"{name}_nav", f"{name}_drawdown", f"{name}_turnover", f"{name}_transaction_cost", f"{baseline}_return", f"{baseline}_nav", f"{baseline}_drawdown"]].to_csv(TABLE_DIR / "daily_returns.csv", index=False)

    refined_metrics = performance_metrics(panel, name, f"{name}_return", f"{name}_turnover", f"{name}_transaction_cost")
    baseline_metrics = performance_metrics(panel, baseline, f"{baseline}_return", f"{baseline}_turnover", f"{baseline}_transaction_cost")
    perf = pd.DataFrame([baseline_metrics, refined_metrics])
    perf.to_csv(TABLE_DIR / "performance_summary.csv", index=False)
    comparison = perf.set_index("strategy").T
    comparison["delta_refined_minus_baseline"] = np.nan
    numeric_mask = pd.to_numeric(comparison[name], errors="coerce").notna() & pd.to_numeric(comparison[baseline], errors="coerce").notna()
    comparison.loc[numeric_mask, "delta_refined_minus_baseline"] = (
        pd.to_numeric(comparison.loc[numeric_mask, name], errors="coerce")
        - pd.to_numeric(comparison.loc[numeric_mask, baseline], errors="coerce")
    )
    comparison.reset_index(names="metric").to_csv(TABLE_DIR / "comparison_vs_baseline.csv", index=False)

    regime_rows = []
    for strategy in [baseline, name]:
        ret_col = f"{strategy}_return"
        weight_prefix = strategy
        for regime, sub in panel.groupby("macro_regime_confirmed"):
            row = performance_metrics(sub, strategy, ret_col, f"{strategy}_turnover", f"{strategy}_transaction_cost")
            row["macro_regime_confirmed"] = regime
            row["n_obs"] = len(sub)
            for asset in ASSETS:
                col = f"{weight_prefix}_weight_{asset}"
                row[f"avg_weight_{asset}"] = sub[col].mean() if col in sub.columns else np.nan
            regime_rows.append(row)
    pd.DataFrame(regime_rows).to_csv(TABLE_DIR / "regime_allocation_summary.csv", index=False)

    flat_rows = []
    flat = panel[panel["macro_regime_confirmed"].astype(str).str.upper().eq("FLAT")]
    for state, sub in flat.groupby("flat_refined_state"):
        row = performance_metrics(sub, name, f"{name}_return", f"{name}_turnover", f"{name}_transaction_cost")
        row["flat_refined_state"] = state
        row["n_obs"] = len(sub)
        for asset in ASSETS:
            row[f"avg_weight_{asset}"] = sub[f"{name}_weight_{asset}"].mean()
        flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(TABLE_DIR / "flat_state_allocation_summary.csv", index=False)

    plot_results(panel)
    write_readme(panel, perf, source_path, gs10_source)


def plot_results(panel: pd.DataFrame) -> None:
    name = CONFIG["strategy_name"]
    baseline = CONFIG["baseline_name"]
    date = panel["date"]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(date, panel[f"{baseline}_nav"], label=baseline)
    ax.plot(date, panel[f"{name}_nav"], label=name)
    ax.set_yscale("log")
    ax.set_title("Equity curve vs baseline")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "equity_curve_vs_baseline.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(date, panel[f"{baseline}_drawdown"], label=baseline)
    ax.plot(date, panel[f"{name}_drawdown"], label=name)
    ax.set_title("Drawdown curve vs baseline")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "drawdown_curve_vs_baseline.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for strategy in [baseline, name]:
        rolling = (1.0 + panel[f"{strategy}_return"].fillna(0.0)).rolling(252).apply(np.prod, raw=True) - 1.0
        ax.plot(date, rolling, label=strategy)
    ax.set_title("Rolling 12M return vs baseline")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rolling_12m_return_vs_baseline.png", dpi=170)
    plt.close(fig)

    flat = panel[panel["macro_regime_confirmed"].astype(str).str.upper().eq("FLAT")]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(flat["date"], *[flat[f"{name}_weight_{a}"] for a in ASSETS], labels=ASSETS, alpha=0.85)
    ax.set_title("Flat-state refined weights timeline")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", ncol=5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "flat_state_weights_timeline.png", dpi=170)
    plt.close(fig)

    exposure = panel.groupby("flat_refined_state")[[f"{name}_weight_{a}" for a in ASSETS]].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    exposure.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Regime exposure summary")
    ax.set_ylabel("Average weight")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "regime_exposure_summary.png", dpi=170)
    plt.close(fig)


def write_readme(panel: pd.DataFrame, perf: pd.DataFrame, source_path: Path, gs10_source: Path | None) -> None:
    name = CONFIG["strategy_name"]
    baseline = CONFIG["baseline_name"]
    flat = panel[panel["macro_regime_confirmed"].astype(str).str.upper().eq("FLAT")]
    flat_counts = flat["flat_refined_state"].value_counts().to_dict()
    perf_table = perf[["strategy", "CAGR", "annualized_volatility", "Sharpe", "Sortino", "max_drawdown", "Calmar", "final_equity", "turnover", "total_transaction_cost"]].copy()
    lines = [
        "# Flat-Rate-Refined Baseline Experiment",
        "",
        "## Purpose",
        "",
        "This independent strategy experiment starts from the current mature baseline and changes only FLAT regime allocation. Non-FLAT regime logic follows the baseline weights directly.",
        "",
        "## Motivation",
        "",
        "FLAT describes yield-curve shape but not the absolute rate level. Earlier diagnostics showed different asset behavior across low-rate normal, high-rate normal, low-rate stress, and high-rate stress FLAT states.",
        "",
        "## Inputs",
        "",
        f"- Baseline panel: `{source_path.relative_to(ROOT).as_posix()}`",
        f"- GS10 source: `{gs10_source.relative_to(ROOT).as_posix() if gs10_source else 'baseline panel'}`",
        f"- Baseline strategy: `{baseline}`",
        "",
        "## Rules",
        "",
        f"- GS10 <= {CONFIG['gs10_threshold']}: low-rate FLAT.",
        f"- GS10 > {CONFIG['gs10_threshold']}: high-rate FLAT.",
        f"- GS10 high/low state confirmation: {CONFIG['confirmation_days']} consecutive trading days.",
        "- FLAT_LOW_RATE_NORMAL: SPY / CMDTY_FUT / GOLD inverse volatility.",
        "- FLAT_LOW_RATE_STRESS: 40% GOLD + 60% IEF.",
        "- FLAT_HIGH_RATE_NORMAL: CMDTY_FUT / GOLD inverse volatility.",
        "- FLAT_HIGH_RATE_STRESS: 20% GOLD + 80% CASH.",
        "- Non-FLAT regimes: baseline weights are used directly.",
        "",
        "## Results",
        "",
        perf_table.to_markdown(index=False),
        "",
        "## FLAT State Counts",
        "",
    ]
    lines.extend(f"- {k}: {v}" for k, v in flat_counts.items())
    lines.extend(
        [
            "",
            "## Risk Notes",
            "",
            "This is a refinement experiment, not a final strategy change. The GS10 threshold and FLAT sub-state allocations come from descriptive diagnostics and should not be interpreted as optimized parameters.",
        ]
    )
    (OUT / "README_flat_rate_refined_baseline.md").write_text("\n".join(lines), encoding="utf-8")


def print_diagnostics(panel: pd.DataFrame) -> None:
    name = CONFIG["strategy_name"]
    baseline = CONFIG["baseline_name"]
    bm = performance_metrics(panel, baseline, f"{baseline}_return", f"{baseline}_turnover", f"{baseline}_transaction_cost")
    rm = performance_metrics(panel, name, f"{name}_return", f"{name}_turnover", f"{name}_transaction_cost")
    flat = panel[panel["macro_regime_confirmed"].astype(str).str.upper().eq("FLAT")]
    print("flat_rate_refined_baseline completed.")
    print(f"baseline final equity: {bm['final_equity']:.4f}")
    print(f"flat_rate_refined_baseline final equity: {rm['final_equity']:.4f}")
    print(f"baseline max drawdown: {bm['max_drawdown']:.2%}")
    print(f"flat_rate_refined_baseline max drawdown: {rm['max_drawdown']:.2%}")
    print(f"baseline Sharpe: {bm['Sharpe']:.3f}")
    print(f"flat_rate_refined_baseline Sharpe: {rm['Sharpe']:.3f}")
    print(f"FLAT sample count: {len(flat)}")
    print(f"LOW_RATE_FLAT sample count: {int(flat['flat_rate_confirmed_state'].eq('LOW_RATE_FLAT').sum())}")
    print(f"HIGH_RATE_FLAT sample count: {int(flat['flat_rate_confirmed_state'].eq('HIGH_RATE_FLAT').sum())}")
    print(f"FLAT_NORMAL sample count: {int(flat['flat_stress_state'].eq('NORMAL').sum())}")
    print(f"FLAT_STRESS sample count: {int(flat['flat_stress_state'].eq('STRESS').sum())}")
    for state, sub in flat.groupby("flat_refined_state"):
        weights = ", ".join(f"{a}={sub[f'{name}_weight_{a}'].mean():.2%}" for a in ASSETS)
        print(f"{state}: n={len(sub)}, avg weights: {weights}")
    print(f"output_dir: {OUT.relative_to(ROOT).as_posix()}")


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    panel, source_path, gs10_source = load_panel()
    validate_panel(panel)
    panel = panel.sort_values("date").reset_index(drop=True)
    panel = build_flat_states(panel)
    out = run_refined_backtest(panel)
    write_outputs(out, source_path, gs10_source)
    print_diagnostics(out)


if __name__ == "__main__":
    main()
