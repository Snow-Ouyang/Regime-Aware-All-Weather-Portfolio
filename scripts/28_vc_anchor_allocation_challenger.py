from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_strategy_source_only_core import (
    ASSETS,
    FINAL_STRATEGY,
    INV_VOL_WINDOW,
    ROOT,
    compute_strategy,
    monthly_hold_weights,
    normalize_weight_dict,
    performance_metrics,
)


OUT = ROOT / "results" / "vc_anchor_allocation_challenger"
FIG = OUT / "figures"
MAIN = ROOT / "results" / "main_pipeline_final" / "tables" / "daily_backtest_panel.csv"

WINDOWS = {
    "2008_GFC": ("2007-10-01", "2009-06-30"),
    "2011_EURO_DEBT": ("2011-06-01", "2011-12-31"),
    "2015_2016": ("2015-05-01", "2016-03-31"),
    "2018Q4": ("2018-10-01", "2019-01-31"),
    "COVID_2020": ("2020-02-01", "2020-06-30"),
    "2022_RATE_WAR": ("2021-11-01", "2023-03-31"),
    "2025_PULLBACK": ("2025-01-01", None),
}


@dataclass(frozen=True)
class MixVariant:
    name: str
    cash_weight: float
    ief_weight: float


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def load_redesign_module():
    path = ROOT / "scripts" / "19_daily_credit_trigger_redesign.py"
    spec = importlib.util.spec_from_file_location("daily_credit_redesign_28", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_state_machine_module():
    path = ROOT / "scripts" / "26_vix_credit_anchor_exit_grid.py"
    spec = importlib.util.spec_from_file_location("vc_anchor_exit_grid_26", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_panel(mod19) -> pd.DataFrame:
    if not MAIN.exists():
        raise FileNotFoundError(f"Missing main pipeline panel: {MAIN}")
    raw = pd.read_csv(MAIN, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    panel = mod19.add_credit_features(mod19.prepare_panel(raw))
    panel["SPY_MA30"] = panel["spy_price"].rolling(30, min_periods=30).mean()
    panel["SPY_MA40"] = panel["spy_price"].rolling(40, min_periods=40).mean()
    panel["SPY_above_MA30"] = panel["spy_price"] > panel["SPY_MA30"]
    panel["SPY_above_MA40"] = panel["spy_price"] > panel["SPY_MA40"]
    return panel


def baseline_frame(panel: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"date": panel["date"]})
    for asset in ASSETS:
        out[f"weight_{asset}"] = panel[f"{FINAL_STRATEGY}_weight_{asset}"]
    for suffix in ["return", "nav", "drawdown", "turnover", "transaction_cost"]:
        out[f"{FINAL_STRATEGY}_{suffix}"] = panel[f"{FINAL_STRATEGY}_{suffix}"]
    return out


def build_mix_variants() -> list[MixVariant]:
    variants = []
    for cash_pct in range(10, 100, 10):
        cash_w = cash_pct / 100.0
        variants.append(MixVariant(name=f"VC_ANCHOR_ALLOC_STEEP_CASH{cash_pct:02d}_IEF{100-cash_pct:02d}", cash_weight=cash_w, ief_weight=1.0 - cash_w))
    return variants


def build_state(panel: pd.DataFrame, mod26) -> pd.DataFrame:
    variant = mod26.Variant(name="EXIT_Z0.9_MA50", unlock_z_threshold=0.9, unlock_trend_rule="MA50")
    return mod26.simulate_anchor(panel, variant)


def make_weights(panel: pd.DataFrame, state: pd.DataFrame, mix: MixVariant) -> pd.DataFrame:
    flat_low_normal = monthly_hold_weights(panel, ["SPY", "CMDTY_FUT"], window=INV_VOL_WINDOW)
    flat_high_normal = monthly_hold_weights(panel, ["GOLD", "CMDTY_FUT"], window=INV_VOL_WINDOW)
    steep_high_normal = monthly_hold_weights(panel, ["SPY", "GOLD", "CMDTY_FUT"], window=INV_VOL_WINDOW)
    inverted_normal = monthly_hold_weights(panel, ["SPY", "GOLD"], window=INV_VOL_WINDOW)

    weights = pd.DataFrame(0.0, index=panel.index, columns=ASSETS)
    allocation_states: list[str] = []

    for i, row in panel.iterrows():
        final_regime = row["final_regime_confirmed"]
        stress = bool(state.loc[i, "stress_active"])
        if final_regime == "FLAT_LOW_RATE":
            if stress:
                w = {"CASH": 1.0}
                st = "FLAT_LOW_RATE_STRESS"
            else:
                w = flat_low_normal.loc[i].to_dict()
                st = "FLAT_LOW_RATE_NORMAL"
        elif final_regime == "FLAT_HIGH_RATE":
            if stress:
                w = {"IEF": 1.0}
                st = "FLAT_HIGH_RATE_STRESS"
            else:
                w = flat_high_normal.loc[i].to_dict()
                st = "FLAT_HIGH_RATE_NORMAL"
        elif final_regime == "STEEP_LOW_RATE":
            w = {"SPY": 1.0}
            st = "STEEP_LOW_RATE_NORMAL"
        elif final_regime == "STEEP_HIGH_RATE":
            if stress:
                w = {"CASH": mix.cash_weight, "IEF": mix.ief_weight}
                st = "STEEP_HIGH_RATE_STRESS"
            else:
                w = steep_high_normal.loc[i].to_dict()
                st = "STEEP_HIGH_RATE_NORMAL"
        elif final_regime == "INVERTED":
            w = inverted_normal.loc[i].to_dict()
            st = "INVERTED"
        else:
            raise ValueError(f"Unexpected final regime: {final_regime}")
        weights.loc[i, ASSETS] = pd.Series(normalize_weight_dict(w))
        allocation_states.append(st)

    state = state.copy()
    state["allocation_state"] = allocation_states
    return weights, state


def make_frame(panel: pd.DataFrame, state: pd.DataFrame, weights: pd.DataFrame, strategy: str) -> pd.DataFrame:
    strat = compute_strategy(panel, weights, strategy)
    return pd.concat([panel[["date"]], weights.add_prefix("weight_"), strat, state.drop(columns=["date"])], axis=1)


def perf_row(frame: pd.DataFrame, strategy: str, mix: MixVariant | None = None) -> dict[str, object]:
    p = performance_metrics(frame, strategy)
    row = {
        "strategy": strategy,
        "CAGR": p["CAGR"],
        "Sharpe": p["Sharpe"],
        "Sortino": p["Sortino"],
        "MaxDD": p["MaxDD"],
        "Calmar": p["Calmar"],
        "Final Equity": p["final_equity"],
        "annualized_vol": p["annualized_volatility"],
        "turnover": p["turnover"],
        "transaction_cost_drag": p["transaction_cost"],
        "avg_weight_SPY": float(frame["weight_SPY"].mean()),
        "avg_weight_GOLD": float(frame["weight_GOLD"].mean()),
        "avg_weight_CMDTY_FUT": float(frame["weight_CMDTY_FUT"].mean()),
        "avg_weight_IEF": float(frame["weight_IEF"].mean()),
        "avg_weight_CASH": float(frame["weight_CASH"].mean()),
        "stress_days": int(frame["stress_active"].sum()) if "stress_active" in frame else np.nan,
    }
    if mix is not None:
        row["steep_stress_cash_weight"] = mix.cash_weight
        row["steep_stress_ief_weight"] = mix.ief_weight
    return row


def crisis_row(frame: pd.DataFrame, strategy: str, window: str, start: str, end: str | None) -> dict[str, object]:
    mask = frame["date"] >= pd.Timestamp(start)
    if end is not None:
        mask &= frame["date"] <= pd.Timestamp(end)
    sub = frame.loc[mask].copy()
    if len(sub) == 0:
        return {"strategy": strategy, "window": window}
    ret = sub[f"{strategy}_return"].fillna(0.0)
    nav = (1.0 + ret).cumprod()
    ann_vol = float(ret.std(ddof=1) * np.sqrt(252.0))
    ann_ret = float(nav.iloc[-1] ** (252.0 / len(sub)) - 1.0)
    return {
        "strategy": strategy,
        "window": window,
        "cumulative_return": float(nav.iloc[-1] - 1.0),
        "max_drawdown": float((nav / nav.cummax() - 1.0).min()),
        "Sharpe": float(ann_ret / ann_vol) if ann_vol > 0 else np.nan,
        "avg_weight_SPY": float(sub["weight_SPY"].mean()),
        "avg_weight_GOLD": float(sub["weight_GOLD"].mean()),
        "avg_weight_CMDTY_FUT": float(sub["weight_CMDTY_FUT"].mean()),
        "avg_weight_IEF": float(sub["weight_IEF"].mean()),
        "avg_weight_CASH": float(sub["weight_CASH"].mean()),
        "stress_days": int(sub["stress_active"].sum()) if "stress_active" in sub else np.nan,
    }


def plot_curves(frames: dict[str, pd.DataFrame], names: list[str], kind: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    for name in names:
        ax.plot(frames[name]["date"], frames[name][f"{name}_{kind}"], label=name)
    if kind == "nav":
        ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    ax.set_title(path.stem.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    mod19 = load_redesign_module()
    mod26 = load_state_machine_module()
    panel = load_panel(mod19)
    state = build_state(panel, mod26)

    frames = {}
    baseline = baseline_frame(panel)
    frames[FINAL_STRATEGY] = baseline
    perf_rows = [perf_row(baseline, FINAL_STRATEGY)]
    crisis_rows = []
    for window, (start, end) in WINDOWS.items():
        crisis_rows.append(crisis_row(baseline, FINAL_STRATEGY, window, start, end))

    for mix in build_mix_variants():
        weights, state_with_alloc = make_weights(panel, state, mix)
        frame = make_frame(panel, state_with_alloc, weights, mix.name)
        frames[mix.name] = frame
        perf_rows.append(perf_row(frame, mix.name, mix))
        for window, (start, end) in WINDOWS.items():
            row = crisis_row(frame, mix.name, window, start, end)
            row["steep_stress_cash_weight"] = mix.cash_weight
            row["steep_stress_ief_weight"] = mix.ief_weight
            crisis_rows.append(row)

    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(OUT / "vc_anchor_allocation_challenger_performance.csv", index=False)
    crisis_df = pd.DataFrame(crisis_rows)
    crisis_df.to_csv(OUT / "vc_anchor_allocation_challenger_crisis.csv", index=False)

    challengers = perf_df.loc[perf_df["strategy"] != FINAL_STRATEGY].copy()
    challengers["Sharpe_rank"] = challengers["Sharpe"].rank(ascending=False, method="min")
    challengers["MaxDD_rank"] = challengers["MaxDD"].rank(ascending=False, method="min")
    challengers["FinalEquity_rank"] = challengers["Final Equity"].rank(ascending=False, method="min")
    challengers["composite_score"] = 0.4 * challengers["Sharpe_rank"] + 0.3 * challengers["MaxDD_rank"] + 0.3 * challengers["FinalEquity_rank"]
    challengers = challengers.sort_values(["composite_score", "Sharpe"], ascending=[True, False]).reset_index(drop=True)
    challengers.to_csv(OUT / "vc_anchor_allocation_challenger_ranked.csv", index=False)

    top_names = [FINAL_STRATEGY] + challengers.head(4)["strategy"].tolist()
    plot_curves(frames, top_names, "nav", FIG / "vc_anchor_allocation_challenger_equity_curve.png")
    plot_curves(frames, top_names, "drawdown", FIG / "vc_anchor_allocation_challenger_drawdown_curve.png")

    print("baseline final strategy")
    print(perf_df.loc[perf_df["strategy"] == FINAL_STRATEGY, ["CAGR", "Sharpe", "MaxDD", "Final Equity"]].to_string(index=False))
    print("top allocation challengers")
    print(
        challengers[
            [
                "strategy",
                "steep_stress_cash_weight",
                "steep_stress_ief_weight",
                "CAGR",
                "Sharpe",
                "MaxDD",
                "Final Equity",
                "avg_weight_GOLD",
                "avg_weight_IEF",
                "avg_weight_CASH",
                "composite_score",
            ]
        ].head(10).to_string(index=False)
    )
    print("output path")
    print(str(OUT))


if __name__ == "__main__":
    main()
