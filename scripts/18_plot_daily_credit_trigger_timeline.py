from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_strategy_source_only_core import ROOT


OUT = ROOT / "results" / "credit_daily_trigger_visualization"
FIG = OUT / "figures"
MAIN = ROOT / "results" / "main_pipeline_final" / "tables" / "daily_backtest_panel.csv"


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def load_daily_credit() -> pd.DataFrame:
    aaa = pd.read_csv(ROOT / "data" / "raw" / "macro" / "Credit" / "DAAA.csv", parse_dates=["observation_date"])
    baa = pd.read_csv(ROOT / "data" / "raw" / "macro" / "Credit" / "DBAA.csv", parse_dates=["observation_date"])
    aaa = aaa.rename(columns={"observation_date": "date", "DAAA": "DAAA"})[["date", "DAAA"]]
    baa = baa.rename(columns={"observation_date": "date", "DBAA": "DBAA"})[["date", "DBAA"]]
    df = aaa.merge(baa, on="date", how="outer").sort_values("date").drop_duplicates("date")
    df["DAAA"] = pd.to_numeric(df["DAAA"], errors="coerce")
    df["DBAA"] = pd.to_numeric(df["DBAA"], errors="coerce")
    df["CREDIT_SPREAD"] = df["DBAA"] - df["DAAA"]
    df["D_CREDIT_15D"] = df["CREDIT_SPREAD"] - df["CREDIT_SPREAD"].shift(15)
    return df


def load_panel() -> pd.DataFrame:
    if not MAIN.exists():
        raise FileNotFoundError(f"Missing main pipeline panel: {MAIN}")
    panel = pd.read_csv(MAIN, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    panel["SPY_MA20"] = panel["SPY_MA20"] if "SPY_MA20" in panel.columns else panel["spy_price"].rolling(20, min_periods=20).mean()
    panel["SPY_above_MA20"] = panel["spy_price"] > panel["SPY_MA20"]
    return panel


def build_signal_panel() -> pd.DataFrame:
    panel = load_panel()
    credit = load_daily_credit()
    out = panel[
        [
            "date",
            "spy_price",
            "SPY_MA20",
            "SPY_above_MA20",
            "spy_drawdown_from_previous_high",
            "refined_regime_confirmed",
        ]
    ].merge(credit, on="date", how="left")
    out = out.dropna(subset=["CREDIT_SPREAD"]).reset_index(drop=True)
    out["credit_entry_signal"] = (
        out["refined_regime_confirmed"].isin(["FLAT_LOW_RATE", "FLAT_HIGH_RATE"])
        & (out["spy_drawdown_from_previous_high"] <= -0.05)
        & (out["D_CREDIT_15D"] > 0.10)
    )
    out["credit_unlock_signal"] = (out["D_CREDIT_15D"] < 0) & out["SPY_above_MA20"]
    return out


def build_credit_lock_events(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    pending_lock = False
    lock_active_today = []
    entry_dates = []
    unlock_dates = []
    entry_flags = []
    unlock_flags = []
    for _, row in df.iterrows():
        active = pending_lock
        lock_active_today.append(active)
        entry = False
        unlock = False
        if (not active) and bool(row["credit_entry_signal"]):
            pending_lock = True
            entry = True
            entry_dates.append(row["date"])
        elif active and bool(row["credit_unlock_signal"]):
            pending_lock = False
            unlock = True
            unlock_dates.append(row["date"])
        else:
            pending_lock = active
        entry_flags.append(entry)
        unlock_flags.append(unlock)
    df = df.copy()
    df["credit_lock_active"] = pd.Series(lock_active_today, index=df.index)
    df["entry_signal_flag"] = pd.Series(entry_flags, index=df.index)
    df["unlock_signal_flag"] = pd.Series(unlock_flags, index=df.index)
    return df, pd.DataFrame({"entry_date": entry_dates}), pd.DataFrame({"unlock_date": unlock_dates})


def plot_timeline(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    axes[0].plot(df["date"], df["CREDIT_SPREAD"], color="firebrick", linewidth=1.0, label="DBAA - DAAA")
    axes[0].scatter(
        df.loc[df["entry_signal_flag"], "date"],
        df.loc[df["entry_signal_flag"], "CREDIT_SPREAD"],
        marker="^",
        s=28,
        color="black",
        label="Credit entry signal",
        zorder=3,
    )
    axes[0].scatter(
        df.loc[df["unlock_signal_flag"], "date"],
        df.loc[df["unlock_signal_flag"], "CREDIT_SPREAD"],
        marker="v",
        s=28,
        color="royalblue",
        label="Credit unlock signal",
        zorder=3,
    )
    axes[0].set_title("Daily Credit Spread with Credit Trigger / Unlock Signals")
    axes[0].set_ylabel("Credit spread")
    axes[0].legend(frameon=False, ncol=3)
    axes[0].grid(alpha=0.2)

    axes[1].plot(df["date"], df["spy_price"], color="dimgray", linewidth=1.0, label="SPY price")
    axes[1].plot(df["date"], df["SPY_MA20"], color="orange", linewidth=0.9, label="SPY MA20")
    active = df["credit_lock_active"].fillna(False).astype(bool)
    if active.any():
        start = active & ~active.shift(1, fill_value=False)
        end = active & ~active.shift(-1, fill_value=False)
        starts = start[start].index.tolist()
        ends = end[end].index.tolist()
        for s, e in zip(starts, ends):
            axes[1].axvspan(df.loc[s, "date"], df.loc[e, "date"], color="lightsteelblue", alpha=0.25)
    axes[1].scatter(
        df.loc[df["entry_signal_flag"], "date"],
        df.loc[df["entry_signal_flag"], "spy_price"],
        marker="^",
        s=20,
        color="black",
        zorder=3,
    )
    axes[1].scatter(
        df.loc[df["unlock_signal_flag"], "date"],
        df.loc[df["unlock_signal_flag"], "spy_price"],
        marker="v",
        s=20,
        color="royalblue",
        zorder=3,
    )
    axes[1].set_title("SPY Price, MA20, and Credit Lock Active Windows")
    axes[1].set_ylabel("SPY")
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False, ncol=2)
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIG / "daily_credit_spread_trigger_unlock_timeline.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    df = build_signal_panel()
    df, entries, unlocks = build_credit_lock_events(df)
    df.to_csv(OUT / "daily_credit_signal_panel.csv", index=False)
    entries.to_csv(OUT / "credit_entry_dates.csv", index=False)
    unlocks.to_csv(OUT / "credit_unlock_dates.csv", index=False)
    plot_timeline(df)
    print("daily credit spread timeline saved to")
    print(str(FIG / "daily_credit_spread_trigger_unlock_timeline.png"))
    print("entry signals")
    print(len(entries))
    print("unlock signals")
    print(len(unlocks))


if __name__ == "__main__":
    main()
