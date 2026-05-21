from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from final_strategy_source_only_core import ASSETS, ROOT


OUT = ROOT / "results" / "vc_anchor_steep_enable_test"
FIG = OUT / "figures"
TABLE = OUT
MAIN = ROOT / "results" / "main_pipeline_final" / "tables" / "daily_backtest_panel.csv"

STATE_ORDER = [
    "FLAT_LOW_RATE_NORMAL",
    "FLAT_LOW_RATE_STRESS",
    "FLAT_HIGH_RATE_NORMAL",
    "FLAT_HIGH_RATE_STRESS",
    "STEEP_LOW_RATE_NORMAL",
    "STEEP_LOW_RATE_STRESS",
    "STEEP_HIGH_RATE_NORMAL",
    "STEEP_HIGH_RATE_STRESS",
    "INVERTED_NORMAL",
    "INVERTED_STRESS",
]


def load_module(script_name: str, module_name: str):
    path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


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
            sharpe = 0.0 if asset == "CASH" else (ann_excess_ret / vol if vol and vol > 0 else None)
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


def plot_heatmap(perf: pd.DataFrame, group_col: str, value_col: str, path: Path, title: str, percent: bool) -> None:
    heat = perf.pivot_table(index="asset", columns=group_col, values=value_col, aggfunc="first")
    cols = [c for c in STATE_ORDER if c in heat.columns] + [c for c in heat.columns if c not in STATE_ORDER]
    heat = heat.reindex(index=ASSETS, columns=cols)
    labels = heat.map(lambda x: "" if pd.isna(x) else (f"{x:.1%}" if percent else f"{x:.2f}"))
    fig_w = max(12, 1.0 * len(heat.columns))
    fig, ax = plt.subplots(figsize=(fig_w, 5.8))
    sns.heatmap(
        heat,
        annot=labels,
        fmt="",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": value_col},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    mod19 = load_module("19_daily_credit_trigger_redesign.py", "daily_credit_redesign_32")
    mod31 = load_module("31_vc_anchor_steep_enable_test.py", "vc_anchor_steep_enable_32")

    raw = pd.read_csv(MAIN, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    panel = mod19.add_credit_features(mod19.prepare_panel(raw))
    variant = mod31.Variant("VC_ANCHOR_ALL_INVERTED_STEEP_CREDIT", steep_vix_enabled=False, steep_credit_enabled=True)
    state = mod31.simulate_anchor(panel, variant)

    panel["vc_anchor_all_inverted_steep_credit_stress"] = state["stress_active"].astype(bool)
    panel["vc_anchor_all_inverted_steep_credit_cross_state"] = (
        panel["final_regime_confirmed"].astype(str)
        + "_"
        + panel["vc_anchor_all_inverted_steep_credit_stress"].map({False: "NORMAL", True: "STRESS"})
    )

    behavior = asset_behavior(panel, "vc_anchor_all_inverted_steep_credit_cross_state")
    behavior.to_csv(TABLE / "vc_anchor_all_inverted_steep_credit_cross_state_asset_behavior.csv", index=False)
    counts = panel.groupby("vc_anchor_all_inverted_steep_credit_cross_state").size().rename("n_days").reset_index()
    counts.to_csv(TABLE / "vc_anchor_all_inverted_steep_credit_cross_state_day_counts.csv", index=False)

    plot_heatmap(
        behavior,
        "vc_anchor_all_inverted_steep_credit_cross_state",
        "annualized_return",
        FIG / "vc_anchor_all_inverted_steep_credit_asset_behavior_heatmap.png",
        "Asset Annualized Return by VC_ANCHOR_ALL_INVERTED_STEEP_CREDIT Regime x Stress State",
        percent=True,
    )
    plot_heatmap(
        behavior,
        "vc_anchor_all_inverted_steep_credit_cross_state",
        "Sharpe",
        FIG / "vc_anchor_all_inverted_steep_credit_asset_sharpe_heatmap.png",
        "Asset Sharpe Ratio by VC_ANCHOR_ALL_INVERTED_STEEP_CREDIT Regime x Stress State",
        percent=False,
    )
    print("generated VC_ANCHOR_ALL_INVERTED_STEEP_CREDIT cross-state asset behavior heatmaps")
    print(str(TABLE / "vc_anchor_all_inverted_steep_credit_cross_state_asset_behavior.csv"))
    print(str(FIG / "vc_anchor_all_inverted_steep_credit_asset_behavior_heatmap.png"))
    print(str(FIG / "vc_anchor_all_inverted_steep_credit_asset_sharpe_heatmap.png"))


if __name__ == "__main__":
    main()
