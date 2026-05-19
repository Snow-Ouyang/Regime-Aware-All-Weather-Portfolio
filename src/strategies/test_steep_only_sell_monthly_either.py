from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TIMING_PANEL_PATH = ROOT / "results" / "spy_cash_timing_frequency_test" / "daily_backtest_panel.csv"
REGIME_PANEL_PATH = ROOT / "results" / "reconstructed_regime_asset_behavior" / "reconstructed_regime_panel.csv"

RESULTS_DIR = ROOT / "results" / "steep_only_sell_test"
FIGURES_DIR = ROOT / "figures" / "steep_only_sell_test"

DAILY_OUT = RESULTS_DIR / "steep_only_sell_daily_panel.csv"
PERF_OUT = RESULTS_DIR / "steep_only_sell_performance_summary.csv"
FIG_EQUITY = FIGURES_DIR / "steep_only_sell_equity_curve.png"
FIG_DD = FIGURES_DIR / "steep_only_sell_drawdown.png"

TRANSACTION_COST_BPS = 20.0


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    if not TIMING_PANEL_PATH.exists():
        raise FileNotFoundError(f"Missing timing panel: {TIMING_PANEL_PATH}")
    if not REGIME_PANEL_PATH.exists():
        raise FileNotFoundError(f"Missing regime panel: {REGIME_PANEL_PATH}")

    timing = pd.read_csv(TIMING_PANEL_PATH)
    regime = pd.read_csv(REGIME_PANEL_PATH, usecols=["date", "CREDIT_SPREAD_BAA_AAA", "DGS1", "TERM_SPREAD_10Y_1Y"])
    timing["date"] = pd.to_datetime(timing["date"])
    regime["date"] = pd.to_datetime(regime["date"])

    regime["macro_regime"] = np.select(
        [
            (pd.to_numeric(regime["CREDIT_SPREAD_BAA_AAA"], errors="coerce") > 1.5)
            & (pd.to_numeric(regime["DGS1"], errors="coerce") > 5.0),
            pd.to_numeric(regime["TERM_SPREAD_10Y_1Y"], errors="coerce") < 0.0,
            (pd.to_numeric(regime["TERM_SPREAD_10Y_1Y"], errors="coerce") >= 0.0)
            & (pd.to_numeric(regime["TERM_SPREAD_10Y_1Y"], errors="coerce") < 1.0),
            pd.to_numeric(regime["TERM_SPREAD_10Y_1Y"], errors="coerce") >= 1.0,
        ],
        ["HIGH_INFLATION", "INVERTED", "FLAT", "STEEP"],
        default="NEUTRAL",
    )

    panel = timing.merge(regime[["date", "macro_regime"]], on="date", how="left")
    panel["macro_regime"] = panel["macro_regime"].ffill().fillna("NEUTRAL")
    return panel.sort_values("date").reset_index(drop=True)


def build_test_strategy(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["signal_weight_spy"] = np.where(
        (out["monthly_either_weight_spy"] < 0.5) & (out["macro_regime"] == "STEEP"),
        0.0,
        1.0,
    )
    out["TEST_STEEP_ONLY_SELL_weight_spy"] = out["signal_weight_spy"].shift(1)
    out.loc[out.index[0], "TEST_STEEP_ONLY_SELL_weight_spy"] = out["signal_weight_spy"].iloc[0]
    out["TEST_STEEP_ONLY_SELL_weight_spy"] = out["TEST_STEEP_ONLY_SELL_weight_spy"].ffill().fillna(1.0)
    out["TEST_STEEP_ONLY_SELL_weight_cash"] = 1.0 - out["TEST_STEEP_ONLY_SELL_weight_spy"]
    out["TEST_STEEP_ONLY_SELL_turnover_flag"] = (
        out["TEST_STEEP_ONLY_SELL_weight_spy"] != out["TEST_STEEP_ONLY_SELL_weight_spy"].shift(1)
    ).fillna(False)
    out.loc[out.index[0], "TEST_STEEP_ONLY_SELL_turnover_flag"] = False
    out["TEST_STEEP_ONLY_SELL_transaction_cost"] = np.where(
        out["TEST_STEEP_ONLY_SELL_turnover_flag"], TRANSACTION_COST_BPS / 10000.0, 0.0
    )
    out["TEST_STEEP_ONLY_SELL_return"] = (
        out["TEST_STEEP_ONLY_SELL_weight_spy"] * out["spy_daily_return"].fillna(0.0)
        + out["TEST_STEEP_ONLY_SELL_weight_cash"] * out["daily_rf"]
        - out["TEST_STEEP_ONLY_SELL_transaction_cost"]
    )
    out["TEST_STEEP_ONLY_SELL_nav"] = (1.0 + out["TEST_STEEP_ONLY_SELL_return"]).cumprod()
    return out


def perf_row(panel: pd.DataFrame, strategy: str, ret_col: str, nav_col: str, weight_col: str | None) -> dict[str, object]:
    s = panel[ret_col].dropna()
    rf = panel.loc[s.index, "daily_rf"]
    ann_ret = float((1.0 + s).prod() ** (252.0 / len(s)) - 1.0)
    ann_vol = float(s.std(ddof=1) * np.sqrt(252.0)) if len(s) > 1 else np.nan
    if strategy == "CASH_ONLY":
        sharpe = 0.0
    else:
        ex = s - rf
        ex_std = ex.std(ddof=1)
        sharpe = float(ex.mean() / ex_std * np.sqrt(252.0)) if pd.notna(ex_std) and ex_std != 0 else np.nan
    wealth = (1.0 + s).cumprod()
    mdd = float((wealth / wealth.cummax() - 1.0).min())
    calmar = float(ann_ret / abs(mdd)) if mdd < 0 else np.nan
    turns = int(panel[f"{strategy}_turnover_flag"].sum()) if f"{strategy}_turnover_flag" in panel.columns else 0
    avg_spy = float(panel[weight_col].mean()) if weight_col else (1.0 if strategy == "SPY_BUY_HOLD" else 0.0)
    return {
        "strategy": strategy,
        "start_date": panel["date"].iloc[0].date().isoformat(),
        "end_date": panel["date"].iloc[-1].date().isoformat(),
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
        "calmar_ratio": calmar,
        "final_nav": float(panel[nav_col].iloc[-1]),
        "number_of_switches": turns,
        "time_in_spy": avg_spy,
        "time_in_cash": 1.0 - avg_spy,
    }


def compute_performance(panel: pd.DataFrame) -> pd.DataFrame:
    rows = [
        perf_row(panel, "TEST_STEEP_ONLY_SELL", "TEST_STEEP_ONLY_SELL_return", "TEST_STEEP_ONLY_SELL_nav", "TEST_STEEP_ONLY_SELL_weight_spy"),
        perf_row(panel, "MONTHLY_EITHER_CONFIRM", "MONTHLY_EITHER_CONFIRM_return", "MONTHLY_EITHER_CONFIRM_nav", "MONTHLY_EITHER_CONFIRM_weight_spy"),
        perf_row(panel, "SPY_BUY_HOLD", "SPY_BUY_HOLD_return", "SPY_BUY_HOLD_nav", None),
        perf_row(panel, "CASH_ONLY", "CASH_ONLY_return", "CASH_ONLY_nav", None),
    ]
    return pd.DataFrame(rows)


def plot_results(panel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, col in [
        ("TEST_STEEP_ONLY_SELL", "TEST_STEEP_ONLY_SELL_nav"),
        ("MONTHLY_EITHER_CONFIRM", "MONTHLY_EITHER_CONFIRM_nav"),
        ("SPY_BUY_HOLD", "SPY_BUY_HOLD_nav"),
        ("CASH_ONLY", "CASH_ONLY_nav"),
    ]:
        ax.plot(panel["date"], panel[col], label=name)
    ax.set_yscale("log")
    ax.set_title("Steep-Only SELL Test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_EQUITY, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for name, col in [
        ("TEST_STEEP_ONLY_SELL", "TEST_STEEP_ONLY_SELL_return"),
        ("MONTHLY_EITHER_CONFIRM", "MONTHLY_EITHER_CONFIRM_return"),
        ("SPY_BUY_HOLD", "SPY_BUY_HOLD_return"),
    ]:
        wealth = (1.0 + panel[col]).cumprod()
        dd = wealth / wealth.cummax() - 1.0
        ax.plot(panel["date"], dd, label=name)
    ax.set_title("Drawdown Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DD, dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    panel = build_test_strategy(load_data())
    perf = compute_performance(panel)
    panel.to_csv(DAILY_OUT, index=False)
    perf.to_csv(PERF_OUT, index=False)
    plot_results(panel)

    test_row = perf.loc[perf["strategy"] == "TEST_STEEP_ONLY_SELL"].iloc[0]
    base_row = perf.loc[perf["strategy"] == "MONTHLY_EITHER_CONFIRM"].iloc[0]
    print(f"Test strategy annualized return: {test_row['annualized_return']:.2%}")
    print(f"Test strategy Sharpe: {test_row['sharpe_ratio']:.2f}")
    print(f"Test strategy max drawdown: {test_row['max_drawdown']:.2%}")
    print(f"Baseline annualized return: {base_row['annualized_return']:.2%}")
    print(f"Baseline Sharpe: {base_row['sharpe_ratio']:.2f}")
    print(f"Baseline max drawdown: {base_row['max_drawdown']:.2%}")
    print(f"Saved: {DAILY_OUT}")
    print(f"Saved: {PERF_OUT}")
    print(f"Saved: {FIG_EQUITY}")
    print(f"Saved: {FIG_DD}")


if __name__ == "__main__":
    main()
