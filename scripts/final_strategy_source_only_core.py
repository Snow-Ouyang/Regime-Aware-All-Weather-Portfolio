"""Source-only canonical builder for the final regime-hedge strategy.

This module intentionally reads only source data under data/raw and
data/processed. Existing results are allowed only by callers for comparison,
never as required inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_INPUTS = {
    "asset_returns": ROOT / "data" / "processed" / "assets" / "daily_returns.csv",
    "asset_prices": ROOT / "data" / "processed" / "assets" / "daily_adjusted_close.csv",
    "dgs10": ROOT / "data" / "raw" / "macro" / "rate" / "DGS10.csv",
    "dgs1": ROOT / "data" / "raw" / "macro" / "rate" / "DGS1.csv",
    "dtb3": ROOT / "data" / "raw" / "macro" / "rate" / "DTB3.csv",
    "vix": ROOT / "data" / "raw" / "macro" / "volatility" / "VIXCLS.csv",
    "waaa": ROOT / "data" / "raw" / "macro" / "Credit" / "DAAA.csv",
    "wbaa": ROOT / "data" / "raw" / "macro" / "Credit" / "DBAA.csv",
}

ASSETS = ["SPY", "GOLD", "CMDTY_FUT", "IEF", "CASH"]
SPY_BUY_HOLD = "SPY_BUY_HOLD"
SPY_CASH_TIMING = "SPY_CASH_TIMING"
FINAL_STRATEGY = "FINAL_REGIME_HEDGE_TRIGGER_LOCK"
REFINED_BASELINE = "FLAT_RATE_REFINED_L50_H30"
ASSET_RETURN_MAP = {
    "SPY_return": "SPY",
    "GOLD_return": "GLD",
    "CMDTY_FUT_return": "GD=F",
    "IEF_return": "IEF",
}
CONFIRMATION_DAYS = 3
GS10_THRESHOLD = 3.0
STEEP_GS1_THRESHOLD = 0.3
INV_VOL_WINDOW = 90
ONE_WAY_COST_BPS = 10.0
RECOVERY_WINDOW = 20
TRIGGER_LOCK_CREDIT_WINDOW = 15
TRIGGER_LOCK_CREDIT_ENTRY_THRESHOLD = 0.10
TRIGGER_LOCK_CREDIT_EXIT_THRESHOLD = 0.0
TRIGGER_LOCK_CREDIT_LEVEL_Z_EXIT_THRESHOLD = 0.9


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def read_source_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing source input: {rel(path)}")
    return pd.read_csv(path, **kwargs)


def read_fred_series(path: Path, value_name: str) -> pd.DataFrame:
    df = read_source_csv(path, parse_dates=["observation_date"])
    if value_name not in df.columns:
        raise ValueError(f"{rel(path)} missing {value_name}")
    out = df.rename(columns={"observation_date": "date"})[["date", value_name]].copy()
    out[value_name] = pd.to_numeric(out[value_name].replace(".", np.nan), errors="coerce")
    return out.sort_values("date").drop_duplicates("date")


def confirm_state(raw: Iterable[str], confirmation_days: int = CONFIRMATION_DAYS, initial: str | None = None) -> list[str]:
    values = [str(v) for v in raw]
    if not values:
        return []
    current = str(initial if initial is not None else values[0])
    candidate = current
    count = 0
    confirmed: list[str] = []
    for value in values:
        if value == current:
            candidate = current
            count = 0
        elif value == candidate:
            count += 1
        else:
            candidate = value
            count = 1
        if candidate != current and count >= confirmation_days:
            current = candidate
            candidate = current
            count = 0
        confirmed.append(current)
    return confirmed


def confirm_steep_rate_split(df: pd.DataFrame, threshold: float = STEEP_GS1_THRESHOLD) -> pd.Series:
    """Confirm STEEP low/high short-rate labels within confirmed STEEP blocks."""
    steep = df["refined_regime_confirmed"].eq("STEEP")
    raw = pd.Series(index=df.index, dtype="object")
    raw.loc[steep] = df.loc[steep, "GS1"].le(threshold).map({True: "STEEP_LOW_RATE", False: "STEEP_HIGH_RATE"})
    confirmed = pd.Series(index=df.index, dtype="object")
    block_id = (steep.ne(steep.shift(1)) & steep).cumsum()
    for _, idx in df.index[steep].to_series().groupby(block_id[steep]):
        labels = raw.loc[idx].astype(str).tolist()
        confirmed.loc[idx] = confirm_state(labels, confirmation_days=CONFIRMATION_DAYS, initial=labels[0])
    return confirmed


def build_monthly_either_state(df: pd.DataFrame) -> pd.Series:
    monthly = df[["date", "spy_price"]].dropna().copy()
    monthly = monthly.set_index("date").resample("ME").last().dropna().reset_index()
    monthly["spy_12m_return"] = monthly["spy_price"] / monthly["spy_price"].shift(12) - 1.0
    monthly["spy_10m_sma"] = monthly["spy_price"].rolling(10, min_periods=10).mean()
    monthly["antonacci_state"] = np.where(monthly["spy_12m_return"] > 0, "HOLD", "SELL")
    monthly["faber_state"] = np.where(monthly["spy_price"] > monthly["spy_10m_sma"], "HOLD", "SELL")
    monthly.loc[monthly["spy_12m_return"].isna(), "antonacci_state"] = "HOLD"
    monthly.loc[monthly["spy_10m_sma"].isna(), "faber_state"] = "HOLD"
    monthly["monthly_either_state"] = np.where(
        monthly["antonacci_state"].eq("SELL") & monthly["faber_state"].eq("SELL"),
        "SELL",
        "HOLD",
    )
    merged = pd.merge_asof(
        df[["date"]].sort_values("date"),
        monthly[["date", "monthly_either_state"]].sort_values("date"),
        on="date",
        direction="backward",
    )
    return merged["monthly_either_state"].fillna("HOLD")


def build_source_panel() -> pd.DataFrame:
    returns = read_source_csv(SOURCE_INPUTS["asset_returns"], parse_dates=["date"])
    prices = read_source_csv(SOURCE_INPUTS["asset_prices"], parse_dates=["date"])
    dgs10 = read_fred_series(SOURCE_INPUTS["dgs10"], "DGS10")
    dgs1 = read_fred_series(SOURCE_INPUTS["dgs1"], "DGS1")
    dtb3 = read_fred_series(SOURCE_INPUTS["dtb3"], "DTB3")
    vix = read_fred_series(SOURCE_INPUTS["vix"], "VIXCLS").rename(columns={"VIXCLS": "VIX_LEVEL"})
    waaa = read_fred_series(SOURCE_INPUTS["waaa"], "DAAA").rename(columns={"DAAA": "WAAA"})
    wbaa = read_fred_series(SOURCE_INPUTS["wbaa"], "DBAA").rename(columns={"DBAA": "WBAA"})

    panel = prices[["date", "SPY"]].rename(columns={"SPY": "spy_price"}).copy()
    for out_col, src_col in ASSET_RETURN_MAP.items():
        if src_col not in returns.columns:
            raise ValueError(f"Missing source return column {src_col} for {out_col}")
        panel = panel.merge(returns[["date", src_col]].rename(columns={src_col: out_col}), on="date", how="left")
    for frame in [dgs10, dgs1, dtb3, vix, waaa, wbaa]:
        panel = panel.merge(frame, on="date", how="left")
    panel = panel.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    for col in ["DGS10", "DGS1", "DTB3", "VIX_LEVEL", "WAAA", "WBAA"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce").ffill().bfill()

    panel["GS10"] = panel["DGS10"]
    panel["GS1"] = panel["DGS1"]
    panel["TERM_SPREAD_10Y_1Y"] = panel["DGS10"] - panel["DGS1"]
    panel["CREDIT_SPREAD_BAA_AAA"] = panel["WBAA"] - panel["WAAA"]
    panel["D_CREDIT_SPREAD_20D"] = panel["CREDIT_SPREAD_BAA_AAA"] - panel["CREDIT_SPREAD_BAA_AAA"].shift(20)
    panel["D_CREDIT_SPREAD_15D"] = panel["CREDIT_SPREAD_BAA_AAA"] - panel["CREDIT_SPREAD_BAA_AAA"].shift(
        TRIGGER_LOCK_CREDIT_WINDOW
    )
    panel["CASH_return"] = (1.0 + panel["DTB3"].ffill() / 100.0) ** (1.0 / 252.0) - 1.0
    panel["daily_rf"] = panel["CASH_return"]

    panel["spy_drawdown_from_previous_high"] = panel["spy_price"] / panel["spy_price"].cummax() - 1.0
    panel["SPY_MA20"] = panel["spy_price"].rolling(20, min_periods=20).mean()
    panel["SPY_MA50"] = panel["spy_price"].rolling(50, min_periods=50).mean()
    panel["SPY_CROSS_ABOVE_MA20"] = (panel["spy_price"] > panel["SPY_MA20"]) & (
        panel["spy_price"].shift(1) <= panel["SPY_MA20"].shift(1)
    )
    panel["SPY_above_MA20"] = panel["spy_price"] > panel["SPY_MA20"]
    panel["SPY_above_MA50"] = panel["spy_price"] > panel["SPY_MA50"]
    vix_roll = panel["VIX_LEVEL"].rolling(120, min_periods=120)
    panel["VIX_ZSCORE_120D"] = (panel["VIX_LEVEL"] - vix_roll.mean()) / vix_roll.std(ddof=1).replace(0, np.nan)
    credit_roll = panel["CREDIT_SPREAD_BAA_AAA"].rolling(252, min_periods=126)
    panel["CREDIT_LEVEL_Z_252D"] = (
        (panel["CREDIT_SPREAD_BAA_AAA"] - credit_roll.mean()) / credit_roll.std(ddof=1).replace(0, np.nan)
    )

    cmdty_price = (1.0 + panel["CMDTY_FUT_return"].fillna(0.0)).cumprod()
    panel["CMDTY_FUT_price"] = cmdty_price
    panel["CMDTY_RET60"] = cmdty_price / cmdty_price.shift(60) - 1.0
    panel["CMDTY_RET20"] = cmdty_price / cmdty_price.shift(20) - 1.0

    raw_regime = np.select(
        [
            panel["TERM_SPREAD_10Y_1Y"] < 0,
            panel["TERM_SPREAD_10Y_1Y"] <= 1,
            panel["TERM_SPREAD_10Y_1Y"] > 1,
        ],
        ["INVERTED", "FLAT", "STEEP"],
        default="FLAT",
    )
    panel["macro_regime_raw"] = raw_regime
    panel["macro_regime_confirmed"] = confirm_state(raw_regime, confirmation_days=CONFIRMATION_DAYS, initial=str(raw_regime[0]))
    if panel["macro_regime_confirmed"].eq("NEUTRAL").any():
        raise RuntimeError("Canonical source-only regime must not produce NEUTRAL.")

    refined_raw = np.where(
        panel["macro_regime_raw"].eq("FLAT") if isinstance(panel["macro_regime_raw"], pd.Series) else raw_regime == "FLAT",
        np.where(panel["GS10"] <= GS10_THRESHOLD, "FLAT_LOW_RATE", "FLAT_HIGH_RATE"),
        raw_regime,
    )
    panel["refined_regime_raw"] = refined_raw
    panel["refined_regime_confirmed"] = confirm_state(refined_raw, confirmation_days=CONFIRMATION_DAYS, initial=str(refined_raw[0]))
    panel["steep_rate_regime_confirmed"] = confirm_steep_rate_split(panel)
    panel["final_regime_confirmed"] = panel["refined_regime_confirmed"]
    panel.loc[panel["refined_regime_confirmed"].eq("STEEP"), "final_regime_confirmed"] = panel.loc[
        panel["refined_regime_confirmed"].eq("STEEP"), "steep_rate_regime_confirmed"
    ]
    panel["monthly_either_state"] = build_monthly_either_state(panel)

    panel = panel.loc[panel["date"] >= pd.Timestamp("2006-03-16")].reset_index(drop=True)
    for col in ["SPY_return", "GOLD_return", "CMDTY_FUT_return", "IEF_return", "CASH_return", "daily_rf"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce").fillna(0.0)
    return panel


def inverse_vol_weights(df: pd.DataFrame, pool: list[str], window: int = INV_VOL_WINDOW) -> pd.DataFrame:
    ret = df[[f"{asset}_return" for asset in pool]].rename(columns={f"{asset}_return": asset for asset in pool})
    vol = ret.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(252.0)
    inv = 1.0 / vol.replace(0, np.nan)
    weights = inv.div(inv.sum(axis=1), axis=0)
    # Before a full window is available, use equal weights only to make the
    # source-only pipeline runnable; affected rows are flagged in output.
    eq = pd.Series({asset: 1.0 / len(pool) for asset in pool})
    weights = weights.apply(lambda row: eq if row.isna().all() else row.fillna(0.0) / row.fillna(0.0).sum(), axis=1)
    return weights.reindex(columns=pool).fillna(0.0)


def first_trading_day_of_month(dates: pd.Series) -> pd.Series:
    month = dates.dt.to_period("M")
    return month.ne(month.shift(1))


def monthly_hold_weights(df: pd.DataFrame, pool: list[str], window: int = INV_VOL_WINDOW) -> pd.DataFrame:
    daily = inverse_vol_weights(df, pool, window=window)
    rebalance = first_trading_day_of_month(df["date"])
    held = pd.DataFrame(index=df.index, columns=pool, dtype=float)
    current = pd.Series({asset: 1.0 / len(pool) for asset in pool})
    for i in df.index:
        if bool(rebalance.iloc[i]):
            current = daily.loc[i, pool]
            if not np.isfinite(current.sum()) or current.sum() <= 0:
                current = pd.Series({asset: 1.0 / len(pool) for asset in pool})
            else:
                current = current / current.sum()
        held.loc[i, pool] = current
    return held.fillna(0.0)


def normalize_weight_dict(weights: dict[str, float]) -> dict[str, float]:
    clean = {asset: max(0.0, float(weights.get(asset, 0.0))) for asset in ASSETS}
    total = sum(clean.values())
    if total <= 0:
        raise ValueError("Weight sum must be positive.")
    return {asset: clean[asset] / total for asset in ASSETS}


def build_backbone_and_states(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["FLAT_VIX_STRESS"] = out["macro_regime_confirmed"].eq("FLAT") & (out["VIX_ZSCORE_120D"] >= 3.0)
    out["FLAT_CREDIT_DD5_STRESS"] = out["macro_regime_confirmed"].eq("FLAT") & (
        (out["spy_drawdown_from_previous_high"] <= -0.05) & (out["D_CREDIT_SPREAD_20D"] > 0.10)
    )
    out["STEEP_EITHER_SELL_STRESS"] = out["macro_regime_confirmed"].eq("STEEP") & out["monthly_either_state"].eq("SELL")
    out["STEEP_CREDIT_DD5_STRESS"] = out["macro_regime_confirmed"].eq("STEEP") & (
        (out["spy_drawdown_from_previous_high"] <= -0.05) & (out["D_CREDIT_SPREAD_20D"] > 0.10)
    )
    out["STEEP_CMDTY_RET60_NEG10"] = out["macro_regime_confirmed"].eq("STEEP") & (out["CMDTY_RET60"] < -0.10)
    out["BACKBONE_V2_ENTRY_SIGNAL"] = (
        out["FLAT_VIX_STRESS"]
        | out["FLAT_CREDIT_DD5_STRESS"]
        | out["STEEP_EITHER_SELL_STRESS"]
        | out["STEEP_CREDIT_DD5_STRESS"]
    )
    out["R3_RECOVERY"] = out["SPY_CROSS_ABOVE_MA20"]

    full_risk = []
    state = "NON_RISK"
    pending = "NON_RISK"
    for _, row in out.iterrows():
        state = pending
        full_risk.append(state == "FULL_RISK")
        pending = state
        if state != "FULL_RISK" and bool(row["BACKBONE_V2_ENTRY_SIGNAL"]):
            pending = "FULL_RISK"
        elif state == "FULL_RISK" and bool(row["R3_RECOVERY"]):
            pending = "NON_RISK"
    out["full_risk_state"] = np.where(full_risk, "FULL_RISK", "NON_RISK")

    slow_overlay = []
    pending_overlay = False
    overlay = False
    for _, row in out.iterrows():
        overlay = pending_overlay
        if row["full_risk_state"] == "FULL_RISK":
            overlay = False
        slow_overlay.append(overlay)
        pending_overlay = overlay
        if row["full_risk_state"] != "FULL_RISK" and not overlay and bool(row["STEEP_CMDTY_RET60_NEG10"]):
            pending_overlay = True
        elif overlay and bool(row["R3_RECOVERY"]):
            pending_overlay = False
    out["steep_slow_growth_overlay_state"] = slow_overlay
    out["is_stress_state"] = out["full_risk_state"].eq("FULL_RISK") | out["refined_regime_confirmed"].isin(
        ["FLAT_LOW_RATE", "FLAT_HIGH_RATE"]
    ) & out["full_risk_state"].eq("FULL_RISK")
    return out


def base_refined_weights(df: pd.DataFrame, inv_vol_window: int = INV_VOL_WINDOW) -> tuple[pd.DataFrame, pd.Series]:
    out = df.copy()
    flat_low_normal = monthly_hold_weights(out, ["SPY", "CMDTY_FUT"], window=inv_vol_window)
    flat_high_normal = monthly_hold_weights(out, ["GOLD", "CMDTY_FUT"], window=inv_vol_window)
    steep_high_normal = monthly_hold_weights(out, ["SPY", "CMDTY_FUT"], window=inv_vol_window)
    inverted_normal = monthly_hold_weights(out, ["SPY", "GOLD"], window=inv_vol_window)
    weights = pd.DataFrame(0.0, index=out.index, columns=ASSETS)
    states = []
    for i, row in out.iterrows():
        refined = row["refined_regime_confirmed"]
        full_risk = row["full_risk_state"] == "FULL_RISK"
        slow = bool(row["steep_slow_growth_overlay_state"])
        if refined == "FLAT_LOW_RATE":
            if full_risk:
                w = {"GOLD": 1.0}
                state = "FLAT_LOW_RATE_STRESS"
            else:
                w = flat_low_normal.loc[i].to_dict()
                state = "FLAT_LOW_RATE_NORMAL"
        elif refined == "FLAT_HIGH_RATE":
            if full_risk:
                w = {"IEF": 0.90, "CASH": 0.10}
                state = "FLAT_HIGH_RATE_STRESS"
            else:
                w = flat_high_normal.loc[i].to_dict()
                state = "FLAT_HIGH_RATE_NORMAL"
        elif refined == "STEEP":
            if full_risk:
                w = {"GOLD": 0.30, "IEF": 0.70}
                state = "STEEP_FULL_RISK"
            elif slow:
                w = {"SPY": 0.50, "IEF": 0.50}
                state = "STEEP_SLOW_GROWTH_OVERLAY"
            elif row["steep_rate_regime_confirmed"] == "STEEP_HIGH_RATE":
                w = steep_high_normal.loc[i].to_dict()
                state = "STEEP_HIGH_RATE_NORMAL"
            else:
                w = {"SPY": 1.0}
                state = "STEEP_LOW_RATE_NORMAL"
        elif refined == "INVERTED":
            w = inverted_normal.loc[i].to_dict()
            state = "INVERTED"
        else:
            raise ValueError(f"Unexpected refined regime: {refined}")
        weights.loc[i, ASSETS] = pd.Series(normalize_weight_dict(w))
        states.append(state)
    return weights, pd.Series(states, index=out.index, name="flat_refined_state")


def apply_flat_low_recovery(df: pd.DataFrame, base_weights: pd.DataFrame, base_states: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    weights = base_weights.copy()
    active: list[bool] = []
    remaining = 0
    was_stress = False
    for i, row in df.iterrows():
        is_stress = bool(base_states.iloc[i].endswith("_STRESS") or row["full_risk_state"] == "FULL_RISK")
        is_flat_low_normal = base_states.iloc[i] == "FLAT_LOW_RATE_NORMAL"
        if is_stress:
            remaining = 0
            flag = False
        else:
            if was_stress and is_flat_low_normal:
                remaining = RECOVERY_WINDOW
            if remaining > 0 and is_flat_low_normal:
                selected = [asset for asset in ["SPY", "CMDTY_FUT", "GOLD"] if pd.notna(row.get(f"{asset}_return", np.nan))]
                if selected:
                    weights.loc[i, ASSETS] = 0.0
                    for asset in selected:
                        weights.loc[i, asset] = 1.0 / len(selected)
                    flag = True
                else:
                    flag = False
                remaining -= 1
            else:
                remaining = 0
                flag = False
        active.append(flag)
        was_stress = is_stress
    return weights, pd.Series(active, index=df.index, name="recovery_flat_low_active")


def normal_allocation_by_regime(
    i: int,
    final_regime: str,
    flat_low_normal: pd.DataFrame,
    flat_high_normal: pd.DataFrame,
    steep_high_normal: pd.DataFrame,
    inverted_normal: pd.DataFrame,
) -> tuple[dict[str, float], str]:
    if final_regime == "FLAT_LOW_RATE":
        return flat_low_normal.loc[i].to_dict(), "FLAT_LOW_RATE_NORMAL"
    if final_regime == "FLAT_HIGH_RATE":
        return flat_high_normal.loc[i].to_dict(), "FLAT_HIGH_RATE_NORMAL"
    if final_regime == "STEEP_HIGH_RATE":
        return steep_high_normal.loc[i].to_dict(), "STEEP_HIGH_RATE_NORMAL"
    if final_regime == "STEEP_LOW_RATE":
        return {"SPY": 1.0}, "STEEP_LOW_RATE_NORMAL"
    if final_regime == "INVERTED":
        return inverted_normal.loc[i].to_dict(), "INVERTED_NORMAL"
    raise ValueError(f"Unexpected final regime: {final_regime}")


def stress_allocation_by_regime(
    i: int,
    final_regime: str,
    inverted_normal: pd.DataFrame,
) -> tuple[dict[str, float], str]:
    if final_regime == "FLAT_LOW_RATE":
        return {"CASH": 1.0}, "FLAT_LOW_RATE_STRESS"
    if final_regime == "FLAT_HIGH_RATE":
        return {"IEF": 1.0}, "FLAT_HIGH_RATE_STRESS"
    if final_regime == "STEEP_LOW_RATE":
        return {"SPY": 0.60, "IEF": 0.40}, "STEEP_LOW_RATE_STRESS"
    if final_regime == "STEEP_HIGH_RATE":
        return {"CASH": 0.10, "IEF": 0.90}, "STEEP_HIGH_RATE_STRESS"
    if final_regime == "INVERTED":
        base = inverted_normal.loc[i].to_dict()
        scaled = {asset: 0.90 * float(weight) for asset, weight in base.items()}
        scaled["CASH"] = scaled.get("CASH", 0.0) + 0.10
        return scaled, "INVERTED_STRESS"
    raise ValueError(f"Unexpected final regime: {final_regime}")


def allowed_trigger_locks(row: pd.Series) -> set[str]:
    regime = row["final_regime_confirmed"]
    locks: set[str] = set()
    vix_entry = bool(row["VIX_ZSCORE_120D"] >= 3.0)
    credit_entry = bool(
        (row["D_CREDIT_SPREAD_15D"] > TRIGGER_LOCK_CREDIT_ENTRY_THRESHOLD)
        and (not bool(row["SPY_above_MA20"]))
    )
    if regime in {"FLAT_LOW_RATE", "FLAT_HIGH_RATE", "INVERTED"}:
        if vix_entry:
            locks.add("VIX")
    if regime in {"FLAT_LOW_RATE", "FLAT_HIGH_RATE", "STEEP_LOW_RATE", "STEEP_HIGH_RATE", "INVERTED"}:
        if credit_entry:
            locks.add("CREDIT")
    return locks


def unlock_trigger_locks(row: pd.Series, active_locks: set[str]) -> set[str]:
    unlocked: set[str] = set()
    spy_above_ma20 = bool(row["SPY_above_MA20"])
    vix_unlock = bool((row["VIX_ZSCORE_120D"] < 1.5) and spy_above_ma20)
    credit_unlock = bool(
        bool(row["SPY_above_MA50"])
        and pd.notna(row["CREDIT_LEVEL_Z_252D"])
        and (row["CREDIT_LEVEL_Z_252D"] < TRIGGER_LOCK_CREDIT_LEVEL_Z_EXIT_THRESHOLD)
    )
    if "VIX" in active_locks and vix_unlock:
        unlocked.add("VIX")
    if "CREDIT" in active_locks and credit_unlock:
        unlocked.add("CREDIT")
    return unlocked


def build_trigger_lock_final_weights(
    df: pd.DataFrame, inv_vol_window: int = INV_VOL_WINDOW
) -> tuple[pd.DataFrame, pd.DataFrame]:
    flat_low_normal = monthly_hold_weights(df, ["SPY", "CMDTY_FUT"], window=inv_vol_window)
    flat_high_normal = monthly_hold_weights(df, ["GOLD", "CMDTY_FUT"], window=inv_vol_window)
    steep_high_normal = monthly_hold_weights(df, ["SPY", "GOLD", "CMDTY_FUT"], window=inv_vol_window)
    inverted_normal = monthly_hold_weights(df, ["SPY", "GOLD"], window=inv_vol_window)
    weights = pd.DataFrame(0.0, index=df.index, columns=ASSETS)

    pending_vix = False
    pending_credit = False
    pending_anchor = ""
    rows = []

    for i, row in df.iterrows():
        current_vix = pending_vix
        current_credit = pending_credit
        current_anchor = pending_anchor
        current_full_risk = current_vix or current_credit or bool(current_anchor)
        current_locks = {name for name, flag in [("VIX", current_vix), ("CREDIT", current_credit)] if flag}
        final_regime = row["final_regime_confirmed"]

        lock_added_today: set[str] = set()
        lock_unlocked_today: set[str] = set()
        entry_signal = False
        exit_signal = False

        vix_ent = "VIX" in allowed_trigger_locks(row)
        credit_ent = "CREDIT" in allowed_trigger_locks(row)
        unlocked = unlock_trigger_locks(row, current_locks)
        vix_unl = "VIX" in unlocked
        credit_unl = "CREDIT" in unlocked

        next_vix = current_vix
        next_credit = current_credit
        next_anchor = current_anchor

        if current_full_risk:
            w, allocation_state = stress_allocation_by_regime(i, final_regime, inverted_normal)

            if not current_anchor:
                current_anchor = "BOTH" if current_vix and current_credit else "VIX" if current_vix else "CREDIT" if current_credit else ""

            if vix_ent and not current_vix:
                next_vix = True
                lock_added_today.add("VIX")
            if credit_ent and not current_credit:
                next_credit = True
                lock_added_today.add("CREDIT")

            if current_anchor == "VIX":
                if vix_unl:
                    next_anchor = ""
                    next_vix = False
                    next_credit = False
                    lock_unlocked_today.update({"VIX"} | ({"CREDIT"} if current_credit else set()))
            elif current_anchor == "CREDIT":
                if credit_unl:
                    next_anchor = ""
                    next_vix = False
                    next_credit = False
                    lock_unlocked_today.update({"CREDIT"} | ({"VIX"} if current_vix else set()))
            else:
                if current_vix and vix_unl:
                    next_vix = False
                    lock_unlocked_today.add("VIX")
                if current_credit and credit_unl:
                    next_credit = False
                    lock_unlocked_today.add("CREDIT")
                if not next_vix and not next_credit:
                    next_anchor = ""

            pending_vix = next_vix
            pending_credit = next_credit
            pending_anchor = next_anchor
            if not pending_vix and not pending_credit and not pending_anchor:
                exit_signal = True
        else:
            w, allocation_state = normal_allocation_by_regime(
                i,
                final_regime,
                flat_low_normal,
                flat_high_normal,
                steep_high_normal,
                inverted_normal,
            )
            if vix_ent and credit_ent:
                entry_signal = True
                lock_added_today.update({"VIX", "CREDIT"})
                pending_vix = True
                pending_credit = True
                pending_anchor = "BOTH"
            elif vix_ent:
                entry_signal = True
                lock_added_today.add("VIX")
                pending_vix = True
                pending_credit = False
                pending_anchor = "VIX"
            elif credit_ent:
                entry_signal = True
                lock_added_today.add("CREDIT")
                pending_vix = False
                pending_credit = True
                pending_anchor = "CREDIT"
            else:
                pending_vix = False
                pending_credit = False
                pending_anchor = ""

        weights.loc[i, ASSETS] = pd.Series(normalize_weight_dict(w))
        rows.append(
            {
                "trigger_lock_full_risk_state": "FULL_RISK" if current_full_risk else "NON_RISK",
                "trigger_lock_active_locks": "+".join(sorted(current_locks)),
                "trigger_lock_anchor_state": current_anchor,
                "trigger_lock_locks_added_today": "+".join(sorted(lock_added_today)),
                "trigger_lock_locks_unlocked_today": "+".join(sorted(lock_unlocked_today)),
                "trigger_lock_entry_signal": entry_signal,
                "trigger_lock_exit_signal": exit_signal,
                "final_allocation_state": allocation_state,
                "trigger_lock_vix_entry_condition": vix_ent,
                "trigger_lock_credit_entry_condition": credit_ent,
                "trigger_lock_cmdty_entry_condition": False,
            }
        )

    return weights, pd.DataFrame(rows, index=df.index)


def compute_strategy(df: pd.DataFrame, weights: pd.DataFrame, name: str) -> pd.DataFrame:
    returns = df[[f"{asset}_return" for asset in ASSETS]].rename(columns={f"{asset}_return": asset for asset in ASSETS}).fillna(0.0)
    gross = (weights[ASSETS] * returns[ASSETS]).sum(axis=1)
    turnover = weights[ASSETS].diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    cost = 0.5 * turnover * ONE_WAY_COST_BPS / 10000.0
    ret = gross - cost
    nav = (1.0 + ret).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    out = pd.DataFrame(
        {
            f"{name}_return": ret,
            f"{name}_nav": nav,
            f"{name}_drawdown": drawdown,
            f"{name}_turnover": turnover,
            f"{name}_transaction_cost": cost,
        }
    )
    for asset in ASSETS:
        out[f"{name}_weight_{asset}"] = weights[asset]
    return out


def performance_metrics(df: pd.DataFrame, name: str) -> dict:
    ret = df[f"{name}_return"].fillna(0.0)
    nav = df[f"{name}_nav"]
    n = len(ret)
    ann_ret = float(nav.iloc[-1] ** (252.0 / n) - 1.0)
    ann_vol = float(ret.std(ddof=1) * np.sqrt(252.0))
    downside = ret[ret < 0].std(ddof=1) * np.sqrt(252.0)
    maxdd = float(df[f"{name}_drawdown"].min())
    return {
        "strategy": name,
        "CAGR": ann_ret,
        "annualized_volatility": ann_vol,
        "Sharpe": float(ann_ret / ann_vol) if ann_vol > 0 else np.nan,
        "Sortino": float(ann_ret / downside) if pd.notna(downside) and downside > 0 else np.nan,
        "MaxDD": maxdd,
        "Calmar": float(ann_ret / abs(maxdd)) if maxdd < 0 else np.nan,
        "final_equity": float(nav.iloc[-1]),
        "win_rate": float((ret > 0).mean()),
        "worst_day": float(ret.min()),
        "worst_12m_return": float((nav / nav.shift(252) - 1.0).min()),
        "turnover": float(df.get(f"{name}_turnover", pd.Series(0.0, index=df.index)).sum()),
        "transaction_cost": float(df.get(f"{name}_transaction_cost", pd.Series(0.0, index=df.index)).sum()),
        "number_of_trades": int((df.get(f"{name}_turnover", pd.Series(0.0, index=df.index)) > 1e-8).sum()),
    }


def build_final_source_only_panel(inv_vol_window: int = INV_VOL_WINDOW) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = build_source_panel()
    panel = build_backbone_and_states(panel)
    base_weights, base_states = base_refined_weights(panel, inv_vol_window=inv_vol_window)
    final_weights, trigger_lock_state = build_trigger_lock_final_weights(panel, inv_vol_window=inv_vol_window)

    base = compute_strategy(panel, base_weights, REFINED_BASELINE)
    final = compute_strategy(panel, final_weights, FINAL_STRATEGY)
    spy_weights = pd.DataFrame(0.0, index=panel.index, columns=ASSETS)
    spy_weights["SPY"] = 1.0
    spy = compute_strategy(panel, spy_weights, SPY_BUY_HOLD)
    spy_cash_weights = pd.DataFrame(0.0, index=panel.index, columns=ASSETS)
    spy_cash_weights["SPY"] = np.where(trigger_lock_state["trigger_lock_full_risk_state"].eq("FULL_RISK"), 0.0, 1.0)
    spy_cash_weights["CASH"] = 1.0 - spy_cash_weights["SPY"]
    spy_cash = compute_strategy(panel, spy_cash_weights, SPY_CASH_TIMING)

    out = pd.concat([panel, trigger_lock_state, base, final, spy, spy_cash], axis=1)
    out["flat_refined_state"] = base_states
    out["recovery_flat_low_active"] = False
    out["final_state"] = np.where(
        out["trigger_lock_full_risk_state"].eq("FULL_RISK"),
        "FULL_RISK",
        "NON_RISK",
    )
    perf = pd.DataFrame(
        [
            performance_metrics(out, SPY_BUY_HOLD),
            performance_metrics(out, SPY_CASH_TIMING),
            performance_metrics(out, FINAL_STRATEGY),
            performance_metrics(out, REFINED_BASELINE),
        ]
    )
    return out, perf


def write_source_only_outputs(
    output_dir: Path | None = None, inv_vol_window: int = INV_VOL_WINDOW
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = output_dir or ROOT / "results" / "final_strategy_source_only"
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    panel, perf = build_final_source_only_panel(inv_vol_window=inv_vol_window)
    panel.to_csv(out_dir / "daily_backtest_panel.csv", index=False)
    perf.to_csv(table_dir / "performance_summary.csv", index=False)
    panel[
        [
            "date",
            "macro_regime_confirmed",
            "refined_regime_confirmed",
            "final_regime_confirmed",
            "steep_rate_regime_confirmed",
            "flat_refined_state",
            "final_allocation_state",
            "final_state",
            "trigger_lock_active_locks",
            "trigger_lock_entry_signal",
            "trigger_lock_exit_signal",
        ]
    ].to_csv(table_dir / "state_panel.csv", index=False)
    return panel, perf
