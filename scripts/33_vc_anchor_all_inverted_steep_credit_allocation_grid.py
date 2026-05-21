from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
import sys

import matplotlib.pyplot as plt
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


OUT = ROOT / "results" / "vc_anchor_all_inverted_steep_credit_allocation_grid"
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
class Variant:
    name: str
    steep_low_spy_weight: float
    steep_low_ief_weight: float
    inverted_stress_cash_weight: float


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def load_module(script_name: str, module_name: str):
    path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, path)
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
    return mod19.add_credit_features(mod19.prepare_panel(raw))


def baseline_frame(panel: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"date": panel["date"]})
    for asset in ASSETS:
        out[f"weight_{asset}"] = panel[f"{FINAL_STRATEGY}_weight_{asset}"]
    for suffix in ["return", "nav", "drawdown", "turnover", "transaction_cost"]:
        out[f"{FINAL_STRATEGY}_{suffix}"] = panel[f"{FINAL_STRATEGY}_{suffix}"]
    if "trigger_lock_full_risk_state" in panel.columns:
        out["stress_active"] = panel["trigger_lock_full_risk_state"].eq("FULL_RISK")
    return out


def build_state(panel: pd.DataFrame, mod31) -> pd.DataFrame:
    variant = mod31.Variant("VC_ANCHOR_ALL_INVERTED_STEEP_CREDIT", steep_vix_enabled=False, steep_credit_enabled=True)
    return mod31.simulate_anchor(panel, variant)


def build_variants() -> list[Variant]:
    variants: list[Variant] = []
    for spy_w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        ief_w = 1.0 - spy_w
        for cash_w in [0.0, 0.1, 0.2, 0.3]:
            name = f"VCAIC_SLS_SPY{int(spy_w*100):02d}_IEF{int(ief_w*100):02d}_INVSTR_CASH{int(cash_w*100):02d}"
            variants.append(
                Variant(
                    name=name,
                    steep_low_spy_weight=spy_w,
                    steep_low_ief_weight=ief_w,
                    inverted_stress_cash_weight=cash_w,
                )
            )
    return variants


def allocation_matrices(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "FLAT_LOW_RATE_NORMAL": monthly_hold_weights(panel, ["SPY", "CMDTY_FUT"], window=INV_VOL_WINDOW),
        "FLAT_HIGH_RATE_NORMAL": monthly_hold_weights(panel, ["GOLD", "CMDTY_FUT"], window=INV_VOL_WINDOW),
        "STEEP_HIGH_RATE_NORMAL": monthly_hold_weights(panel, ["SPY", "GOLD", "CMDTY_FUT"], window=INV_VOL_WINDOW),
        "INVERTED_NORMAL": monthly_hold_weights(panel, ["SPY", "GOLD"], window=INV_VOL_WINDOW),
        "INVERTED_STRESS_BASE": monthly_hold_weights(panel, ["SPY", "GOLD"], window=INV_VOL_WINDOW),
    }


def mixed_inv_vol_row(row_weights: dict[str, float], cash_weight: float) -> dict[str, float]:
    scaled = {k: (1.0 - cash_weight) * v for k, v in row_weights.items()}
    scaled["CASH"] = cash_weight
    return normalize_weight_dict(scaled)


def make_weights(panel: pd.DataFrame, state: pd.DataFrame, variant: Variant, matrices: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
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
                w = matrices["FLAT_LOW_RATE_NORMAL"].loc[i].to_dict()
                st = "FLAT_LOW_RATE_NORMAL"
        elif final_regime == "FLAT_HIGH_RATE":
            if stress:
                w = {"IEF": 1.0}
                st = "FLAT_HIGH_RATE_STRESS"
            else:
                w = matrices["FLAT_HIGH_RATE_NORMAL"].loc[i].to_dict()
                st = "FLAT_HIGH_RATE_NORMAL"
        elif final_regime == "STEEP_LOW_RATE":
            if stress:
                w = {"SPY": variant.steep_low_spy_weight, "IEF": variant.steep_low_ief_weight}
                st = "STEEP_LOW_RATE_STRESS"
            else:
                w = {"SPY": 1.0}
                st = "STEEP_LOW_RATE_NORMAL"
        elif final_regime == "STEEP_HIGH_RATE":
            if stress:
                w = {"CASH": 0.10, "IEF": 0.90}
                st = "STEEP_HIGH_RATE_STRESS"
            else:
                w = matrices["STEEP_HIGH_RATE_NORMAL"].loc[i].to_dict()
                st = "STEEP_HIGH_RATE_NORMAL"
        elif final_regime == "INVERTED":
            if stress:
                base = matrices["INVERTED_STRESS_BASE"].loc[i].to_dict()
                w = mixed_inv_vol_row(base, variant.inverted_stress_cash_weight)
                st = "INVERTED_STRESS"
            else:
                w = matrices["INVERTED_NORMAL"].loc[i].to_dict()
                st = "INVERTED_NORMAL"
        else:
            raise ValueError(f"Unexpected final regime: {final_regime}")

        weights.loc[i, ASSETS] = pd.Series(normalize_weight_dict(w))
        allocation_states.append(st)

    out_state = state.copy()
    out_state["allocation_state"] = allocation_states
    return weights, out_state


def make_frame(panel: pd.DataFrame, state: pd.DataFrame, weights: pd.DataFrame, strategy: str) -> pd.DataFrame:
    strat = compute_strategy(panel, weights, strategy)
    return pd.concat([panel[["date"]], weights.add_prefix("weight_"), strat, state.drop(columns=["date"])], axis=1)


def perf_row(frame: pd.DataFrame, strategy: str, variant: Variant | None = None) -> dict[str, object]:
    p = performance_metrics(frame, strategy)
    stress_days = int(frame["stress_active"].sum()) if "stress_active" in frame.columns else None
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
        "stress_days": stress_days,
    }
    if variant is not None:
        row["steep_low_spy_weight"] = variant.steep_low_spy_weight
        row["steep_low_ief_weight"] = variant.steep_low_ief_weight
        row["inverted_stress_cash_weight"] = variant.inverted_stress_cash_weight
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
    ann_vol = float(ret.std(ddof=1) * (252.0 ** 0.5))
    ann_ret = float(nav.iloc[-1] ** (252.0 / len(sub)) - 1.0)
    return {
        "strategy": strategy,
        "window": window,
        "cumulative_return": float(nav.iloc[-1] - 1.0),
        "max_drawdown": float((nav / nav.cummax() - 1.0).min()),
        "Sharpe": float(ann_ret / ann_vol) if ann_vol > 0 else None,
        "avg_weight_SPY": float(sub["weight_SPY"].mean()),
        "avg_weight_GOLD": float(sub["weight_GOLD"].mean()),
        "avg_weight_CMDTY_FUT": float(sub["weight_CMDTY_FUT"].mean()),
        "avg_weight_IEF": float(sub["weight_IEF"].mean()),
        "avg_weight_CASH": float(sub["weight_CASH"].mean()),
        "stress_days": int(sub["stress_active"].sum()) if "stress_active" in sub.columns else None,
    }


def plot_curves(frames: dict[str, pd.DataFrame], names: list[str], kind: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    for name in names:
        ax.plot(frames[name]["date"], frames[name][f"{name}_{kind}"], label=name)
    if kind == "nav":
        ax.set_yscale("log")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.2)
    ax.set_title(path.stem.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    mod19 = load_module("19_daily_credit_trigger_redesign.py", "daily_credit_redesign_33")
    mod31 = load_module("31_vc_anchor_steep_enable_test.py", "vc_anchor_steep_enable_33")
    panel = load_panel(mod19)
    state = build_state(panel, mod31)
    matrices = allocation_matrices(panel)

    frames: dict[str, pd.DataFrame] = {}
    baseline = baseline_frame(panel)
    frames[FINAL_STRATEGY] = baseline
    perf_rows = [perf_row(baseline, FINAL_STRATEGY)]
    crisis_rows = []
    for window, (start, end) in WINDOWS.items():
        crisis_rows.append(crisis_row(baseline, FINAL_STRATEGY, window, start, end))

    for variant in build_variants():
        strategy = variant.name
        weights, state_with_alloc = make_weights(panel, state, variant, matrices)
        frame = make_frame(panel, state_with_alloc, weights, strategy)
        frames[strategy] = frame
        perf_rows.append(perf_row(frame, strategy, variant))
        for window, (start, end) in WINDOWS.items():
            row = crisis_row(frame, strategy, window, start, end)
            row["steep_low_spy_weight"] = variant.steep_low_spy_weight
            row["steep_low_ief_weight"] = variant.steep_low_ief_weight
            row["inverted_stress_cash_weight"] = variant.inverted_stress_cash_weight
            crisis_rows.append(row)

    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(OUT / "allocation_grid_performance.csv", index=False)
    crisis_df = pd.DataFrame(crisis_rows)
    crisis_df.to_csv(OUT / "allocation_grid_crisis.csv", index=False)

    challengers = perf_df.loc[perf_df["strategy"] != FINAL_STRATEGY].copy()
    challengers["Sharpe_rank"] = challengers["Sharpe"].rank(ascending=False, method="min")
    challengers["MaxDD_rank"] = challengers["MaxDD"].rank(ascending=False, method="min")
    challengers["Final_Equity_rank"] = challengers["Final Equity"].rank(ascending=False, method="min")
    challengers["composite_score"] = (
        0.35 * challengers["Sharpe_rank"]
        + 0.30 * challengers["MaxDD_rank"]
        + 0.35 * challengers["Final_Equity_rank"]
    )
    challengers = challengers.sort_values(["composite_score", "Sharpe"], ascending=[True, False])
    challengers.to_csv(OUT / "allocation_grid_ranked.csv", index=False)

    top_names = [FINAL_STRATEGY] + challengers.head(4)["strategy"].tolist()
    plot_curves(frames, top_names, "nav", FIG / "allocation_grid_equity_curve.png")
    plot_curves(frames, top_names, "drawdown", FIG / "allocation_grid_drawdown_curve.png")

    print("Baseline final:")
    print(perf_df.loc[perf_df["strategy"] == FINAL_STRATEGY, ["CAGR", "Sharpe", "MaxDD", "Final Equity"]].to_string(index=False))
    print("\nTop allocation challengers:")
    print(
        challengers.head(8)[
            [
                "strategy",
                "steep_low_spy_weight",
                "steep_low_ief_weight",
                "inverted_stress_cash_weight",
                "CAGR",
                "Sharpe",
                "MaxDD",
                "Final Equity",
                "composite_score",
            ]
        ].to_string(index=False)
    )
    print(f"\nOutputs written to: {OUT}")


if __name__ == "__main__":
    main()
