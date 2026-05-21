from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_strategy_source_only_core import ASSETS, ROOT, SPY_CASH_TIMING, compute_strategy, performance_metrics


OUT = ROOT / "results" / "vc_anchor_inverted_enable_test"
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
    inverted_vix_enabled: bool
    inverted_credit_enabled: bool


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def load_redesign_module():
    path = ROOT / "scripts" / "19_daily_credit_trigger_redesign.py"
    spec = importlib.util.spec_from_file_location("daily_credit_redesign_30", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_panel(mod) -> pd.DataFrame:
    if not MAIN.exists():
        raise FileNotFoundError(f"Missing main pipeline panel: {MAIN}")
    raw = pd.read_csv(MAIN, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return mod.add_credit_features(mod.prepare_panel(raw))


def baseline_spy_timing_frame(panel: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"date": panel["date"], "regime": panel["refined_regime_confirmed"]})
    for asset in ASSETS:
        out[f"weight_{asset}"] = panel[f"{SPY_CASH_TIMING}_weight_{asset}"]
    for suffix in ["return", "nav", "drawdown", "turnover", "transaction_cost"]:
        out[f"{SPY_CASH_TIMING}_{suffix}"] = panel[f"{SPY_CASH_TIMING}_{suffix}"]
    locks = panel["trigger_lock_active_locks"].fillna("").astype(str)
    out["stress_active"] = panel["trigger_lock_full_risk_state"].eq("FULL_RISK")
    out["vix_lock_active"] = locks.str.contains("VIX")
    out["credit_lock_active"] = locks.str.contains("CREDIT")
    out["active_locks"] = locks
    out["locks_added_today"] = panel["trigger_lock_locks_added_today"].fillna("").astype(str)
    out["locks_unlocked_today"] = panel["trigger_lock_locks_unlocked_today"].fillna("").astype(str)
    return out


def build_variants() -> list[Variant]:
    return [
        Variant("VC_ANCHOR_BASELINE", inverted_vix_enabled=False, inverted_credit_enabled=False),
        Variant("VC_ANCHOR_VIX_INVERTED", inverted_vix_enabled=True, inverted_credit_enabled=False),
        Variant("VC_ANCHOR_CREDIT_INVERTED", inverted_vix_enabled=False, inverted_credit_enabled=True),
        Variant("VC_ANCHOR_ALL_INVERTED", inverted_vix_enabled=True, inverted_credit_enabled=True),
    ]


def vix_entry_cond(row: pd.Series, variant: Variant) -> bool:
    if row["VIX_ZSCORE_120D"] < 3.0:
        return False
    regime = row["refined_regime_confirmed"]
    if regime in {"FLAT_LOW_RATE", "FLAT_HIGH_RATE"}:
        return True
    if regime == "INVERTED":
        return variant.inverted_vix_enabled
    return False  # STEEP VIX OFF


def vix_unlock_cond(row: pd.Series) -> bool:
    return bool((row["VIX_ZSCORE_120D"] < 1.5) and row["SPY_above_MA20"])


def credit_entry_cond(row: pd.Series, variant: Variant) -> bool:
    regime = row["refined_regime_confirmed"]
    enabled = regime in {"FLAT_LOW_RATE", "FLAT_HIGH_RATE"} or (regime == "INVERTED" and variant.inverted_credit_enabled)
    if not enabled:
        return False
    return bool(
        (row["SPY_DD"] <= -0.05)
        and pd.notna(row["D_CREDIT_15D"])
        and (row["D_CREDIT_15D"] > 0.10)
        and (not bool(row["SPY_above_MA20"]))
    )


def credit_unlock_cond(row: pd.Series) -> bool:
    return bool(
        row["SPY_above_MA50"]
        and pd.notna(row["D_CREDIT_15D"])
        and (row["D_CREDIT_15D"] < 0)
        and pd.notna(row["CREDIT_LEVEL_Z_252D"])
        and (row["CREDIT_LEVEL_Z_252D"] < 0.9)
    )


def simulate_anchor(panel: pd.DataFrame, variant: Variant) -> pd.DataFrame:
    pending_vix = False
    pending_credit = False
    pending_anchor = ""
    rows = []
    for _, row in panel.iterrows():
        current_vix = pending_vix
        current_credit = pending_credit
        current_anchor = pending_anchor
        stress_active = current_vix or current_credit or bool(current_anchor)

        next_vix = current_vix
        next_credit = current_credit
        next_anchor = current_anchor

        vix_ent = vix_entry_cond(row, variant)
        credit_ent = credit_entry_cond(row, variant)
        vix_unl = vix_unlock_cond(row)
        credit_unl = credit_unlock_cond(row)

        added_today: list[str] = []
        unlocked_today: list[str] = []

        if not current_anchor:
            if vix_ent and credit_ent:
                next_anchor = "BOTH"
                next_vix = True
                next_credit = True
                added_today.extend(["VIX", "CREDIT"])
            elif vix_ent:
                next_anchor = "VIX"
                next_vix = True
                added_today.append("VIX")
            elif credit_ent:
                next_anchor = "CREDIT"
                next_credit = True
                added_today.append("CREDIT")
        else:
            if vix_ent and not current_vix:
                next_vix = True
                added_today.append("VIX")
            if credit_ent and not current_credit:
                next_credit = True
                added_today.append("CREDIT")

            if current_anchor == "VIX":
                if vix_unl:
                    next_anchor = ""
                    next_vix = False
                    next_credit = False
                    unlocked_today.extend(["VIX", "CREDIT"] if current_credit else ["VIX"])
            elif current_anchor == "CREDIT":
                if credit_unl:
                    next_anchor = ""
                    next_vix = False
                    next_credit = False
                    unlocked_today.extend(["CREDIT", "VIX"] if current_vix else ["CREDIT"])
            else:
                if current_vix and vix_unl:
                    next_vix = False
                    unlocked_today.append("VIX")
                if current_credit and credit_unl:
                    next_credit = False
                    unlocked_today.append("CREDIT")
                if not next_vix and not next_credit:
                    next_anchor = ""

        pending_vix = next_vix
        pending_credit = next_credit
        pending_anchor = next_anchor
        rows.append(
            {
                "date": row["date"],
                "stress_active": stress_active,
                "vix_lock_active": current_vix,
                "credit_lock_active": current_credit,
                "anchor": current_anchor,
                "active_locks": "+".join([x for x, flag in [("VIX", current_vix), ("CREDIT", current_credit)] if flag]),
                "locks_added_today": "+".join(added_today),
                "locks_unlocked_today": "+".join(unlocked_today),
                "regime": row["refined_regime_confirmed"],
            }
        )
    return pd.DataFrame(rows)


def make_frame(panel: pd.DataFrame, state: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=panel.index, columns=ASSETS)
    weights.loc[~state["stress_active"].astype(bool), "SPY"] = 1.0
    weights.loc[state["stress_active"].astype(bool), "CASH"] = 1.0
    strat = compute_strategy(panel, weights, strategy_name)
    return pd.concat([panel[["date"]], weights.add_prefix("weight_"), strat, state.drop(columns=["date"])], axis=1)


def perf_row(frame: pd.DataFrame, strategy: str, variant: Variant | None = None) -> dict[str, object]:
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
        "time_in_stress": int(frame["stress_active"].sum()),
        "time_in_vix_lock": int(frame["vix_lock_active"].sum()),
        "time_in_credit_lock": int(frame["credit_lock_active"].sum()),
        "inverted_stress_days": int((frame["stress_active"] & frame["regime"].eq("INVERTED")).sum()),
        "entries": int(frame["locks_added_today"].astype(str).ne("").sum()),
        "unlocks": int(frame["locks_unlocked_today"].astype(str).ne("").sum()),
    }
    if variant is not None:
        row["inverted_vix_enabled"] = variant.inverted_vix_enabled
        row["inverted_credit_enabled"] = variant.inverted_credit_enabled
    return row


def crisis_row(frame: pd.DataFrame, strategy: str, window: str, start: str, end: str | None) -> dict[str, object]:
    mask = frame["date"] >= pd.Timestamp(start)
    if end is not None:
        mask &= frame["date"] <= pd.Timestamp(end)
    sub = frame.loc[mask]
    ret = sub[f"{strategy}_return"].fillna(0.0)
    if len(sub) == 0:
        return {"strategy": strategy, "window": window}
    nav = (1.0 + ret).cumprod()
    ann_vol = float(ret.std(ddof=1) * np.sqrt(252.0))
    ann_ret = float(nav.iloc[-1] ** (252.0 / len(sub)) - 1.0)
    return {
        "strategy": strategy,
        "window": window,
        "cumulative_return": float(nav.iloc[-1] - 1.0),
        "max_drawdown": float((nav / nav.cummax() - 1.0).min()),
        "Sharpe": float(ann_ret / ann_vol) if ann_vol > 0 else np.nan,
        "time_in_stress": int(sub["stress_active"].sum()),
        "time_in_vix_lock": int(sub["vix_lock_active"].sum()),
        "time_in_credit_lock": int(sub["credit_lock_active"].sum()),
        "inverted_stress_days": int((sub["stress_active"] & sub["regime"].eq("INVERTED")).sum()),
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
    panel = load_panel(mod19)

    frames: dict[str, pd.DataFrame] = {}
    baseline = baseline_spy_timing_frame(panel)
    frames[SPY_CASH_TIMING] = baseline
    perf_rows = [perf_row(baseline, SPY_CASH_TIMING)]
    crisis_rows = []
    for window, (start, end) in WINDOWS.items():
        crisis_rows.append(crisis_row(baseline, SPY_CASH_TIMING, window, start, end))

    for variant in build_variants():
        strategy = variant.name
        state = simulate_anchor(panel, variant)
        frame = make_frame(panel, state, strategy)
        frames[strategy] = frame
        perf_rows.append(perf_row(frame, strategy, variant))
        for window, (start, end) in WINDOWS.items():
            crisis_rows.append(crisis_row(frame, strategy, window, start, end))

    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(OUT / "vc_anchor_inverted_enable_performance.csv", index=False)
    crisis_df = pd.DataFrame(crisis_rows)
    crisis_df.to_csv(OUT / "vc_anchor_inverted_enable_crisis.csv", index=False)

    challengers = perf_df.loc[perf_df["strategy"] != SPY_CASH_TIMING].copy()
    challengers["Sharpe_rank"] = challengers["Sharpe"].rank(ascending=False, method="min")
    challengers["MaxDD_rank"] = challengers["MaxDD"].rank(ascending=False, method="min")
    challengers["Final_Equity_rank"] = challengers["Final Equity"].rank(ascending=False, method="min")
    challengers["composite_score"] = (
        0.4 * challengers["Sharpe_rank"] + 0.3 * challengers["MaxDD_rank"] + 0.3 * challengers["Final_Equity_rank"]
    )
    challengers = challengers.sort_values(["composite_score", "Sharpe"], ascending=[True, False])
    challengers.to_csv(OUT / "vc_anchor_inverted_enable_ranked.csv", index=False)

    names = [SPY_CASH_TIMING] + challengers["strategy"].tolist()
    plot_curves(frames, names, "nav", FIG / "vc_anchor_inverted_enable_equity_curve.png")
    plot_curves(frames, names, "drawdown", FIG / "vc_anchor_inverted_enable_drawdown_curve.png")

    print("Baseline SPY cash timing:")
    print(perf_df.loc[perf_df["strategy"] == SPY_CASH_TIMING, ["CAGR", "Sharpe", "MaxDD", "Final Equity"]].to_string(index=False))
    print("\nInverted enable challengers:")
    print(challengers[["strategy", "CAGR", "Sharpe", "MaxDD", "Final Equity", "inverted_stress_days", "composite_score"]].to_string(index=False))
    print(f"\nOutputs written to: {OUT}")


if __name__ == "__main__":
    main()
