from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIG = {
    "output_dir": Path("results/drawdown_2015_2016_forensic_diagnostic"),
    "figure_dir": Path("figures/drawdown_2015_2016_forensic_diagnostic"),
    "case_start": "2015-05-01",
    "case_end": "2016-03-31",
    "trading_days_per_year": 252,
    "technical_windows": [10, 20, 50, 100, 200],
    "momentum_windows": [5, 10, 20, 60, 120, 252],
    "macro_variables": [
        "growth_pc1",
        "inflation_pc1",
        "VIX_LEVEL",
        "VIX_ZSCORE_120D",
        "CREDIT_SPREAD_BAA_AAA",
        "D_CREDIT_SPREAD_20D",
        "term_spread",
        "GS10",
        "GS1",
        "GS10_minus_GS1",
    ],
    "asset_list": ["SPY", "CMDTY_FUT", "GOLD", "IEF", "CASH"],
}

PANEL_CANDIDATES = [
    Path("results/regime_aware_risk_parity_allocation/daily_backtest_panel.csv"),
    Path("results/regime_aware_hedge_allocation_steep_mix/daily_backtest_panel.csv"),
    Path("results/spy_cash_backbone_upgrade_ablation/daily_backtest_panel.csv"),
    Path("results/flat_vix_credit_trigger_diagnostic/full_backtest_panel.csv"),
]

RISK_PANEL_CANDIDATES = [
    Path("results/regime_aware_risk_parity_allocation/daily_backtest_panel.csv"),
    Path("results/regime_aware_hedge_allocation_steep_mix/daily_backtest_panel.csv"),
    Path("results/spy_cash_backbone_upgrade_ablation/daily_backtest_panel.csv"),
]

MONTHLY_SIGNAL_PATH = Path("results/spy_cash_timing_frequency_test/monthly_signal_panel.csv")
PROCESSED_MACRO_PATH = Path("data/processed/risk_factors/extended_risk_factor_panel.csv")
PROCESSED_SIMPLE_MACRO_PATH = Path("data/processed/regime_inputs_simplified.csv")
PROCESSED_ASSET_RETURNS = Path("data/processed/assets/daily_returns.csv")
RAW_RATE_PATHS = {
    "GS10": Path("data/raw/macro/rate/GS10.csv"),
    "GS1": Path("data/raw/macro/rate/GS1.csv"),
}


def ensure_dirs() -> None:
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["figure_dir"].mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        for alt in ["DATE", "Date", "observation_date"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "date"})
                break
    if "date" not in df.columns:
        raise ValueError(f"No date column in {path}")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def _merge_missing(base: pd.DataFrame, other: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    add = [c for c in cols if c not in base.columns and c in other.columns]
    if not add:
        return base
    return base.merge(other[["date"] + add], on="date", how="left")


def _max_drawdown(ret: pd.Series) -> float:
    nav = (1 + ret.fillna(0.0)).cumprod()
    return float((nav / nav.cummax() - 1).min()) if not nav.empty else np.nan


def _annualized_return(ret: pd.Series) -> float:
    ret = ret.dropna()
    if ret.empty:
        return np.nan
    years = len(ret) / CONFIG["trading_days_per_year"]
    nav = (1 + ret).prod()
    return float(nav ** (1 / years) - 1) if years > 0 and nav > 0 else np.nan


def _annualized_vol(ret: pd.Series) -> float:
    ret = ret.dropna()
    return float(ret.std(ddof=0) * math.sqrt(CONFIG["trading_days_per_year"])) if len(ret) > 1 else np.nan


def _sharpe(ret: pd.Series, rf: pd.Series) -> float:
    x = ret.dropna()
    rf = rf.reindex(x.index).fillna(0.0)
    excess = x - rf
    std = excess.std(ddof=0)
    return float(excess.mean() / std * math.sqrt(CONFIG["trading_days_per_year"])) if std and std > 0 else np.nan


def _beta(asset: pd.Series, spy: pd.Series) -> float:
    aligned = pd.concat([asset, spy], axis=1).dropna()
    if len(aligned) < 2:
        return np.nan
    var = aligned.iloc[:, 1].var(ddof=0)
    if var <= 0:
        return np.nan
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1], ddof=0)[0, 1]
    return float(cov / var)


def _correlation(a: pd.Series, b: pd.Series) -> float:
    aligned = pd.concat([a, b], axis=1).dropna()
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1])) if len(aligned) > 1 else np.nan


def _percentile(full: pd.Series, value: float) -> float:
    full = full.dropna()
    if full.empty or pd.isna(value):
        return np.nan
    return float((full <= value).mean())


def _first_true_date(mask: pd.Series, dates: pd.Series) -> Optional[pd.Timestamp]:
    idx = mask.fillna(False)
    if not idx.any():
        return None
    return pd.Timestamp(dates.loc[idx.idxmax()])


def load_base_panel() -> pd.DataFrame:
    frames: List[Tuple[Path, pd.DataFrame]] = []
    for path in PANEL_CANDIDATES:
        if path.exists():
            frames.append((path, _read_csv(path)))
    if not frames:
        raise FileNotFoundError("No base panel found.")
    panel = frames[0][1].copy()
    print(f"Loaded base panel: {frames[0][0]}")
    needed = [
        "spy_price",
        "spy_daily_return",
        "daily_rf",
        "macro_regime_confirmed",
        "monthly_either_state",
        "VIX_LEVEL",
        "VIX_ZSCORE_120D",
        "CREDIT_SPREAD_BAA_AAA",
        "D_CREDIT_SPREAD_20D",
        "spy_drawdown_from_previous_high",
        "SPY_MA20",
        "SPY_CROSS_ABOVE_MA20",
        "SPY_return",
        "GOLD_return",
        "CMDTY_FUT_return",
        "IEF_return",
        "CASH_return",
        "timing_state",
        "cross_state",
        "entry_reason",
        "BACKBONE_V2_UPGRADED_risk_state",
        "BACKBONE_V2_UPGRADED_weight_spy",
        "BACKBONE_V2_UPGRADED_weight_cash",
    ]
    for _, df in frames[1:]:
        panel = _merge_missing(panel, df, needed)
    panel["spy_price"] = pd.to_numeric(panel["spy_price"], errors="coerce")
    panel["spy_daily_return"] = pd.to_numeric(panel.get("spy_daily_return", panel.get("SPY_return")), errors="coerce")
    panel["daily_rf"] = pd.to_numeric(panel["daily_rf"], errors="coerce").fillna(0.0)
    if "spy_drawdown_from_previous_high" not in panel.columns:
        panel["spy_drawdown_from_previous_high"] = panel["spy_price"] / panel["spy_price"].cummax() - 1
    if "SPY_MA20" not in panel.columns:
        panel["SPY_MA20"] = panel["spy_price"].rolling(20, min_periods=20).mean()
    if "SPY_CROSS_ABOVE_MA20" not in panel.columns:
        panel["SPY_CROSS_ABOVE_MA20"] = (panel["spy_price"] > panel["SPY_MA20"]) & (
            panel["spy_price"].shift(1) <= panel["SPY_MA20"].shift(1)
        )
    if "VIX_ZSCORE_120D" not in panel.columns:
        roll = panel["VIX_LEVEL"].rolling(120, min_periods=120)
        panel["VIX_ZSCORE_120D"] = (panel["VIX_LEVEL"] - roll.mean()) / roll.std(ddof=0)
    if "D_CREDIT_SPREAD_20D" not in panel.columns:
        panel["D_CREDIT_SPREAD_20D"] = panel["CREDIT_SPREAD_BAA_AAA"] - panel["CREDIT_SPREAD_BAA_AAA"].shift(20)
    for w in CONFIG["technical_windows"]:
        col = f"SPY_MA{w}"
        if col not in panel.columns:
            panel[col] = panel["spy_price"].rolling(w, min_periods=w).mean()
    panel["SPY_CROSS_BELOW_MA20"] = (panel["spy_price"] < panel["SPY_MA20"]) & (panel["spy_price"].shift(1) >= panel["SPY_MA20"].shift(1))
    panel["SPY_CROSS_BELOW_MA50"] = (panel["spy_price"] < panel["SPY_MA50"]) & (panel["spy_price"].shift(1) >= panel["SPY_MA50"].shift(1))
    panel["SPY_CROSS_BELOW_MA200"] = (panel["spy_price"] < panel["SPY_MA200"]) & (panel["spy_price"].shift(1) >= panel["SPY_MA200"].shift(1))
    if "timing_state" not in panel.columns:
        if "BACKBONE_V2_SPY_CASH_weight_SPY" in panel.columns:
            panel["timing_state"] = np.where(pd.to_numeric(panel["BACKBONE_V2_SPY_CASH_weight_SPY"], errors="coerce").fillna(1.0) >= 0.5, "NON_RISK", "RISK")
        elif "BACKBONE_V2_UPGRADED_weight_spy" in panel.columns:
            panel["timing_state"] = np.where(pd.to_numeric(panel["BACKBONE_V2_UPGRADED_weight_spy"], errors="coerce").fillna(1.0) >= 0.5, "NON_RISK", "RISK")
        else:
            panel["timing_state"] = "NON_RISK"
    panel["cross_state"] = panel["macro_regime_confirmed"].fillna("NEUTRAL").astype(str) + "_" + panel["timing_state"].fillna("NON_RISK").astype(str)
    panel["entry_reason"] = panel.get("entry_reason", pd.Series(index=panel.index, dtype=object))
    return panel


def load_macro_variables(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel[["date"]].copy()
    if PROCESSED_MACRO_PATH.exists():
        macro = _read_csv(PROCESSED_MACRO_PATH).rename(
            columns={
                "gs10": "GS10",
                "credit_spread": "credit_spread_ext",
                "term_spread_10y_1y": "term_spread",
            }
        )
        keep = ["date", "growth_pc1", "inflation_pc1", "GS10", "term_spread", "credit_spread_ext", "VIX_LEVEL"]
        out = out.merge(macro[[c for c in keep if c in macro.columns]], on="date", how="left")
    if PROCESSED_SIMPLE_MACRO_PATH.exists():
        simple = _read_csv(PROCESSED_SIMPLE_MACRO_PATH).rename(
            columns={"gs10": "GS10_simple", "credit_spread": "credit_spread_simple", "term_spread_10y_1y": "term_spread_simple"}
        )
        out = out.merge(simple, on="date", how="left")
    for fld in ["growth_pc1", "inflation_pc1"]:
        if fld in panel.columns and fld not in out.columns:
            out[fld] = panel[fld]
    out["GS10"] = out.get("GS10", pd.Series(index=out.index, dtype=float)).combine_first(out.get("GS10_simple"))
    out["term_spread"] = out.get("term_spread", pd.Series(index=out.index, dtype=float)).combine_first(out.get("term_spread_simple"))
    credit_panel = pd.to_numeric(panel.get("CREDIT_SPREAD_BAA_AAA"), errors="coerce")
    out["CREDIT_SPREAD_BAA_AAA"] = credit_panel.combine_first(out.get("credit_spread_ext")).combine_first(out.get("credit_spread_simple"))
    out["VIX_LEVEL"] = pd.to_numeric(panel.get("VIX_LEVEL"), errors="coerce").combine_first(pd.to_numeric(out.get("VIX_LEVEL"), errors="coerce"))
    # raw GS1/GS10 for levels if available
    for name, path in RAW_RATE_PATHS.items():
        if path.exists():
            raw = pd.read_csv(path)
            date_col = "DATE" if "DATE" in raw.columns else raw.columns[0]
            val_col = [c for c in raw.columns if c != date_col][0]
            raw = raw.rename(columns={date_col: "date", val_col: name})
            raw["date"] = pd.to_datetime(raw["date"])
            raw[name] = pd.to_numeric(raw[name], errors="coerce")
            out = out.merge(raw[["date", name]], on="date", how="left", suffixes=("", "_raw"))
            if f"{name}_raw" in out.columns:
                out[name] = pd.to_numeric(out[name], errors="coerce").combine_first(pd.to_numeric(out[f"{name}_raw"], errors="coerce"))
                out = out.drop(columns=[f"{name}_raw"])
    if "GS1" not in out.columns:
        out["GS1"] = np.where(out["GS10"].notna() & out["term_spread"].notna(), out["GS10"] - out["term_spread"], np.nan)
    out["GS10_minus_GS1"] = out["GS10"] - out["GS1"]
    out["D_GS10_20D"] = out["GS10"] - out["GS10"].shift(20)
    out["D_GS1_20D"] = out["GS1"] - out["GS1"].shift(20)
    out["D_term_spread_20D"] = out["term_spread"] - out["term_spread"].shift(20)
    out["D_CREDIT_SPREAD_20D"] = pd.to_numeric(panel["D_CREDIT_SPREAD_20D"], errors="coerce").combine_first(
        out["CREDIT_SPREAD_BAA_AAA"] - out["CREDIT_SPREAD_BAA_AAA"].shift(20)
    )
    out["VIX_ZSCORE_120D"] = pd.to_numeric(panel["VIX_ZSCORE_120D"], errors="coerce")
    out["macro_regime_confirmed"] = panel["macro_regime_confirmed"].to_numpy()
    out = out.ffill()
    merged = panel.merge(out, on=["date", "macro_regime_confirmed"], how="left", suffixes=("", "_macro"))
    for field in ["growth_pc1", "inflation_pc1", "GS10", "GS1", "GS10_minus_GS1", "term_spread", "CREDIT_SPREAD_BAA_AAA", "VIX_LEVEL", "VIX_ZSCORE_120D", "D_CREDIT_SPREAD_20D"]:
        if field not in merged.columns:
            x = f"{field}_x"
            y = f"{field}_y"
            if x in merged.columns or y in merged.columns:
                merged[field] = pd.to_numeric(merged.get(x), errors="coerce").combine_first(pd.to_numeric(merged.get(y), errors="coerce"))
        else:
            y = f"{field}_macro"
            if y in merged.columns:
                merged[field] = pd.to_numeric(merged[field], errors="coerce").combine_first(pd.to_numeric(merged[y], errors="coerce"))
    drop_cols = [c for c in merged.columns if c.endswith("_x") or c.endswith("_y") or c.endswith("_macro")]
    if drop_cols:
        merged = merged.drop(columns=drop_cols, errors="ignore")
    for c in ["growth_pc1", "inflation_pc1", "GS10", "GS1", "GS10_minus_GS1", "term_spread", "D_GS10_20D", "D_GS1_20D", "D_term_spread_20D"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")
    merged.to_csv(CONFIG["output_dir"] / "forensic_daily_panel.csv", index=False)
    return merged


def load_asset_returns(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    if PROCESSED_ASSET_RETURNS.exists():
        asset = _read_csv(PROCESSED_ASSET_RETURNS).rename(columns={"GLD": "GLD_return", "DBC": "DBC_return", "IWM": "IWM_return", "IJH": "IJH_return", "TLT": "TLT_return", "EDV": "EDV_return"})
        out = out.merge(asset, on="date", how="left")
    mapping = {
        "SPY_return": ["SPY_return", "spy_daily_return", "spy_return"],
        "CMDTY_FUT_return": ["CMDTY_FUT_return", "DBC_return", "GSG_return", "GD=F"],
        "GOLD_return": ["GOLD_return", "GLD_return"],
        "IEF_return": ["IEF_return"],
        "CASH_return": ["CASH_return", "daily_rf"],
        "TLT_return": ["TLT_return"],
        "EDV_return": ["EDV_return"],
        "IJH_return": ["IJH_return"],
        "IWM_return": ["IWM_return"],
    }
    for target, sources in mapping.items():
        if target in out.columns:
            out[target] = pd.to_numeric(out[target], errors="coerce")
            continue
        for src in sources:
            if src in out.columns:
                out[target] = pd.to_numeric(out[src], errors="coerce")
                break
        if target not in out.columns:
            out[target] = np.nan
    return out


def identify_spy_drawdown_episode(panel: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Timestamp]]:
    start = pd.Timestamp(CONFIG["case_start"])
    end = pd.Timestamp(CONFIG["case_end"])
    case = panel[(panel["date"] >= start) & (panel["date"] <= end)].copy().reset_index(drop=True)
    case["local_prev_high"] = case["spy_price"].cummax()
    case["local_dd"] = case["spy_price"] / case["local_prev_high"] - 1
    trough_idx = int(case["local_dd"].idxmin())
    trough_row = case.loc[trough_idx]
    peak_price = float(case.loc[:trough_idx, "spy_price"].cummax().iloc[-1])
    peak_idx = int(case.loc[:trough_idx, "spy_price"].eq(peak_price).iloc[::-1].idxmax())
    peak_row = case.loc[peak_idx]
    recovery_date = pd.NaT
    after = case.loc[trough_idx + 1 :]
    rec = after.loc[after["spy_price"] >= peak_price]
    if not rec.empty:
        recovery_date = pd.Timestamp(rec.iloc[0]["date"])
    episode = pd.DataFrame(
        [
            {
                "window_start": start,
                "window_end": end,
                "peak_date": pd.Timestamp(peak_row["date"]),
                "trough_date": pd.Timestamp(trough_row["date"]),
                "peak_price": float(peak_row["spy_price"]),
                "trough_price": float(trough_row["spy_price"]),
                "max_drawdown": float(trough_row["local_dd"]),
                "recovery_date": recovery_date,
                "drawdown_duration_days": int((recovery_date - pd.Timestamp(peak_row["date"])).days) if pd.notna(recovery_date) else np.nan,
                "days_peak_to_trough": int((pd.Timestamp(trough_row["date"]) - pd.Timestamp(peak_row["date"])).days),
                "days_trough_to_recovery": int((recovery_date - pd.Timestamp(trough_row["date"])).days) if pd.notna(recovery_date) else np.nan,
            }
        ]
    )
    episode.to_csv(CONFIG["output_dir"] / "spy_drawdown_episode_2015_2016.csv", index=False)
    return episode, {"peak_date": pd.Timestamp(peak_row["date"]), "trough_date": pd.Timestamp(trough_row["date"]), "recovery_date": recovery_date}


def _window_slices(panel: pd.DataFrame, episode_dates: Dict[str, pd.Timestamp]) -> Dict[str, pd.DataFrame]:
    start = pd.Timestamp(CONFIG["case_start"])
    end = pd.Timestamp(CONFIG["case_end"])
    peak = episode_dates["peak_date"]
    trough = episode_dates["trough_date"]
    pre_idx = panel.index[panel["date"].eq(peak)]
    pre_start = panel.loc[max(0, int(pre_idx[0]) - 60), "date"] if len(pre_idx) else peak
    windows = {
        "full_case": panel[(panel["date"] >= start) & (panel["date"] <= end)].copy(),
        "peak_to_trough": panel[(panel["date"] >= peak) & (panel["date"] <= trough)].copy(),
        "pre_drawdown_60d": panel[(panel["date"] >= pre_start) & (panel["date"] <= peak)].copy(),
        "first_leg": panel[(panel["date"] >= pd.Timestamp("2015-08-01")) & (panel["date"] <= pd.Timestamp("2015-09-30"))].copy(),
        "second_leg": panel[(panel["date"] >= pd.Timestamp("2015-12-01")) & (panel["date"] <= pd.Timestamp("2016-02-29"))].copy(),
    }
    return windows


def compute_macro_percentiles(panel: pd.DataFrame, episode_dates: Dict[str, pd.Timestamp]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    case = panel[(panel["date"] >= pd.Timestamp(CONFIG["case_start"])) & (panel["date"] <= pd.Timestamp(CONFIG["case_end"]))].copy()
    peak = episode_dates["peak_date"]
    trough = episode_dates["trough_date"]
    rows = []
    daily_rows = []
    peak_row = panel.loc[panel["date"].eq(peak)].iloc[0]
    trough_row = panel.loc[panel["date"].eq(trough)].iloc[0]
    for var in CONFIG["macro_variables"]:
        if var not in panel.columns:
            print(f"Warning: macro variable missing: {var}")
            continue
        full = pd.to_numeric(panel[var], errors="coerce").dropna()
        sub = pd.to_numeric(case[var], errors="coerce").dropna()
        if full.empty or sub.empty:
            continue
        level_type = "risk_high_bad"
        if var in ["growth_pc1"]:
            level_type = "growth_low_bad"
        elif var in ["term_spread", "GS10_minus_GS1"]:
            level_type = "curve_low_bad"
        rows.append(
            {
                "variable": var,
                "full_sample_start": panel.loc[full.index.min(), "date"],
                "full_sample_end": panel.loc[full.index.max(), "date"],
                "case_mean": sub.mean(),
                "case_median": sub.median(),
                "case_min": sub.min(),
                "case_max": sub.max(),
                "full_sample_mean": full.mean(),
                "full_sample_median": full.median(),
                "full_sample_std": full.std(ddof=0),
                "case_mean_percentile": _percentile(full, sub.mean()),
                "case_median_percentile": _percentile(full, sub.median()),
                "case_min_percentile": _percentile(full, sub.min()),
                "case_max_percentile": _percentile(full, sub.max()),
                "percentile_at_peak": _percentile(full, peak_row.get(var)),
                "percentile_at_trough": _percentile(full, trough_row.get(var)),
                "zscore_mean_in_case": (sub.mean() - full.mean()) / full.std(ddof=0) if full.std(ddof=0) > 0 else np.nan,
                "zscore_at_peak": (peak_row.get(var) - full.mean()) / full.std(ddof=0) if full.std(ddof=0) > 0 else np.nan,
                "zscore_at_trough": (trough_row.get(var) - full.mean()) / full.std(ddof=0) if full.std(ddof=0) > 0 else np.nan,
                "direction_interpretation": level_type,
            }
        )
        for _, r in case[["date", "macro_regime_confirmed", var]].dropna().iterrows():
            daily_rows.append(
                {
                    "date": r["date"],
                    "variable": var,
                    "variable_value": r[var],
                    "full_sample_percentile": _percentile(full, r[var]),
                    "zscore": (r[var] - full.mean()) / full.std(ddof=0) if full.std(ddof=0) > 0 else np.nan,
                    "macro_regime_confirmed": r["macro_regime_confirmed"],
                }
            )
    summary = pd.DataFrame(rows)
    daily = pd.DataFrame(daily_rows)
    summary.to_csv(CONFIG["output_dir"] / "macro_percentile_summary_2015_2016.csv", index=False)
    daily.to_csv(CONFIG["output_dir"] / "macro_percentile_daily_2015_2016.csv", index=False)
    return summary, daily


def analyze_regime_distribution(panel: pd.DataFrame, windows: Dict[str, pd.DataFrame], episode_dates: Dict[str, pd.Timestamp]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    case = windows["full_case"].copy()
    rows = []
    for regime, sub in case.groupby("macro_regime_confirmed"):
        rows.append(
            {
                "regime": regime,
                "n_days": len(sub),
                "percentage": len(sub) / len(case),
                "cumulative_SPY_return_in_regime": (1 + sub["SPY_return"].fillna(0.0)).prod() - 1,
                "SPY_max_drawdown_in_regime": _max_drawdown(sub["SPY_return"]),
                "CMDTY_FUT_return": (1 + sub["CMDTY_FUT_return"].fillna(0.0)).prod() - 1,
                "GOLD_return": (1 + sub["GOLD_return"].fillna(0.0)).prod() - 1,
                "IEF_return": (1 + sub["IEF_return"].fillna(0.0)).prod() - 1,
                "CASH_return": (1 + sub["CASH_return"].fillna(0.0)).prod() - 1,
            }
        )
    dist = pd.DataFrame(rows)
    dist.to_csv(CONFIG["output_dir"] / "regime_distribution_2015_2016.csv", index=False)

    peak = episode_dates["peak_date"]
    trough = episode_dates["trough_date"]
    peak_to_trough = panel[(panel["date"] >= peak) & (panel["date"] <= trough)].copy()
    regime_change_dates = peak_to_trough.loc[peak_to_trough["macro_regime_confirmed"] != peak_to_trough["macro_regime_confirmed"].shift(1), "date"]
    first_change_after_peak = regime_change_dates.iloc[1] if len(regime_change_dates) > 1 else pd.NaT
    timing = pd.DataFrame(
        [
            {
                "regime_at_peak": peak_to_trough.iloc[0]["macro_regime_confirmed"],
                "regime_at_trough": peak_to_trough.iloc[-1]["macro_regime_confirmed"],
                "dominant_regime_peak_to_trough": peak_to_trough["macro_regime_confirmed"].mode().iloc[0],
                "whether_regime_changed_before_trough": peak_to_trough["macro_regime_confirmed"].nunique() > 1,
                "first_regime_change_date": first_change_after_peak,
                "days_from_regime_change_to_trough": (trough - first_change_after_peak).days if pd.notna(first_change_after_peak) else np.nan,
            }
        ]
    )
    timing.to_csv(CONFIG["output_dir"] / "regime_timing_2015_2016.csv", index=False)
    return dist, timing


def compute_asset_performance_windows(panel: pd.DataFrame, windows: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    assets = CONFIG["asset_list"] + [a for a in ["TLT", "EDV", "IJH", "IWM"] if f"{a}_return" in panel.columns]
    rows = []
    for window_name, sub in windows.items():
        spy = sub["SPY_return"]
        rf = sub["CASH_return"].fillna(0.0)
        for asset in assets:
            col = f"{asset}_return"
            if col not in sub.columns:
                continue
            ret = pd.to_numeric(sub[col], errors="coerce")
            valid = ret.dropna()
            if valid.empty:
                continue
            rows.append(
                {
                    "window": window_name,
                    "asset": asset,
                    "cumulative_return": (1 + valid).prod() - 1,
                    "annualized_return": _annualized_return(valid),
                    "annualized_volatility": _annualized_vol(valid),
                    "Sharpe": _sharpe(valid, rf.loc[valid.index]),
                    "max_drawdown": _max_drawdown(valid),
                    "correlation_with_SPY": _correlation(valid, spy.loc[valid.index]),
                    "beta_to_SPY": _beta(valid, spy.loc[valid.index]),
                    "worst_1d_return": valid.min(),
                    "positive_day_ratio": (valid > 0).mean(),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(CONFIG["output_dir"] / "asset_performance_2015_2016.csv", index=False)
    return out


def build_spy_technical_indicators(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    for w in CONFIG["technical_windows"]:
        out[f"SPY_MA{w}"] = out["spy_price"].rolling(w, min_periods=w).mean()
        out[f"SPY_below_MA{w}"] = out["spy_price"] < out[f"SPY_MA{w}"]
        out[f"SPY_cross_below_MA{w}"] = (out["spy_price"] < out[f"SPY_MA{w}"]) & (out["spy_price"].shift(1) >= out[f"SPY_MA{w}"].shift(1))
    for w in CONFIG["momentum_windows"]:
        out[f"SPY_RET_{w}D"] = out["spy_price"] / out["spy_price"].shift(w) - 1.0
    out["SPY_DD_FROM_HIGH"] = out["spy_price"] / out["spy_price"].cummax() - 1.0
    for dd in [3, 5, 8, 10, 15]:
        out[f"SPY_DD_{dd}"] = out["SPY_DD_FROM_HIGH"] <= (-dd / 100.0)
    out["SPY_REALIZED_VOL_20D"] = out["spy_daily_return"].rolling(20, min_periods=20).std(ddof=0) * math.sqrt(CONFIG["trading_days_per_year"])
    out["SPY_REALIZED_VOL_60D"] = out["spy_daily_return"].rolling(60, min_periods=60).std(ddof=0) * math.sqrt(CONFIG["trading_days_per_year"])
    rv252 = out["SPY_REALIZED_VOL_20D"].rolling(252, min_periods=126)
    out["SPY_VOL_Z_252D"] = (out["SPY_REALIZED_VOL_20D"] - rv252.mean()) / rv252.std(ddof=0)
    if "CMDTY_FUT_return" in out.columns:
        price = (1 + out["CMDTY_FUT_return"].fillna(0.0)).cumprod()
        out["CMDTY_RET_20D"] = price / price.shift(20) - 1.0
        out["CMDTY_RET_60D"] = price / price.shift(60) - 1.0
        out["CMDTY_RET_120D"] = price / price.shift(120) - 1.0
        out["CMDTY_MA60"] = price.rolling(60, min_periods=60).mean()
        out["CMDTY_below_MA60"] = price < out["CMDTY_MA60"]
        out["CMDTY_DD_FROM_HIGH"] = price / price.cummax() - 1.0
        out["CMDTY_SPY_relative_return_60D"] = out["CMDTY_RET_60D"] - out["SPY_RET_60D"]
    for pair in [("IJH", "SPY"), ("IWM", "SPY")]:
        if f"{pair[0]}_return" in out.columns:
            asset_price = (1 + out[f"{pair[0]}_return"].fillna(0.0)).cumprod()
            spy_price = (1 + out["SPY_return"].fillna(0.0)).cumprod()
            out[f"{pair[0]}_SPY_relative_return_60D"] = (asset_price / spy_price) / (asset_price / spy_price).shift(60) - 1.0
    case = out[(out["date"] >= pd.Timestamp(CONFIG["case_start"])) & (out["date"] <= pd.Timestamp(CONFIG["case_end"]))].copy()
    case.to_csv(CONFIG["output_dir"] / "spy_technical_panel_2015_2016.csv", index=False)
    return out


def compute_signal_first_trigger_table(panel: pd.DataFrame, episode_dates: Dict[str, pd.Timestamp]) -> pd.DataFrame:
    peak = episode_dates["peak_date"]
    trough = episode_dates["trough_date"]
    sub = panel[(panel["date"] >= peak) & (panel["date"] <= trough)].copy().reset_index(drop=True)
    backbone_entry = _first_true_date(sub.get("BACKBONE_V2_ENTRY_SIGNAL", pd.Series(False, index=sub.index)), sub["date"])
    signal_map = {
        "SPY_DD_LE_3": sub["SPY_DD_FROM_HIGH"] <= -0.03,
        "SPY_DD_LE_5": sub["SPY_DD_FROM_HIGH"] <= -0.05,
        "SPY_DD_LE_8": sub["SPY_DD_FROM_HIGH"] <= -0.08,
        "SPY_DD_LE_10": sub["SPY_DD_FROM_HIGH"] <= -0.10,
        "SPY_CROSS_BELOW_MA20": sub["SPY_cross_below_MA20"],
        "SPY_CROSS_BELOW_MA50": sub["SPY_cross_below_MA50"],
        "SPY_CROSS_BELOW_MA100": sub["SPY_cross_below_MA100"],
        "SPY_CROSS_BELOW_MA200": sub["SPY_cross_below_MA200"],
        "SPY_BELOW_MA50_5D": sub["SPY_below_MA50"].rolling(5, min_periods=5).sum() >= 5,
        "SPY_BELOW_MA200_5D": sub["SPY_below_MA200"].rolling(5, min_periods=5).sum() >= 5,
        "SPY_RET_20D_LT_NEG5": sub["SPY_RET_20D"] < -0.05,
        "SPY_RET_60D_LT_NEG8": sub["SPY_RET_60D"] < -0.08,
        "SPY_RET_120D_LT_NEG10": sub["SPY_RET_120D"] < -0.10,
        "SPY_RET_252D_LT_0": sub["SPY_RET_252D"] < 0,
        "CMDTY_RET_60D_LT_NEG10": sub.get("CMDTY_RET_60D", pd.Series(np.nan, index=sub.index)) < -0.10,
        "CMDTY_RET_120D_LT_NEG15": sub.get("CMDTY_RET_120D", pd.Series(np.nan, index=sub.index)) < -0.15,
        "CMDTY_BELOW_MA60": sub.get("CMDTY_below_MA60", pd.Series(False, index=sub.index)),
        "CMDTY_DD_LE_NEG10": sub.get("CMDTY_DD_FROM_HIGH", pd.Series(np.nan, index=sub.index)) <= -0.10,
        "SPY_DD5_AND_CMDTY_RET60_NEG10": (sub["SPY_DD_FROM_HIGH"] <= -0.05) & (sub.get("CMDTY_RET_60D", pd.Series(np.nan, index=sub.index)) < -0.10),
        "SPY_DD5_CMDTY_RET60_NEG10_CREDIT_POS": (sub["SPY_DD_FROM_HIGH"] <= -0.05) & (sub.get("CMDTY_RET_60D", pd.Series(np.nan, index=sub.index)) < -0.10) & (sub["D_CREDIT_SPREAD_20D"] > 0),
        "CREDIT_CHG20_GT_0": sub["D_CREDIT_SPREAD_20D"] > 0,
        "CREDIT_CHG20_GT_0_05": sub["D_CREDIT_SPREAD_20D"] > 0.05,
        "CREDIT_CHG20_GT_0_10": sub["D_CREDIT_SPREAD_20D"] > 0.10,
        "CREDIT_LVL_PCT_GT_70": sub["CREDIT_SPREAD_BAA_AAA"].rank(pct=True) > 0.70,
        "CREDIT_LVL_PCT_GT_80": sub["CREDIT_SPREAD_BAA_AAA"].rank(pct=True) > 0.80,
        "CREDIT_LVL_PCT_GT_90": sub["CREDIT_SPREAD_BAA_AAA"].rank(pct=True) > 0.90,
        "VIX_Z_GT_2_0": sub["VIX_ZSCORE_120D"] > 2.0,
        "VIX_Z_GT_2_5": sub["VIX_ZSCORE_120D"] > 2.5,
        "VIX_Z_GT_3_0": sub["VIX_ZSCORE_120D"] > 3.0,
        "VIX_LEVEL_GT_20": sub["VIX_LEVEL"] > 20,
        "VIX_LEVEL_GT_25": sub["VIX_LEVEL"] > 25,
        "VIX_LEVEL_GT_30": sub["VIX_LEVEL"] > 30,
        "MONTHLY_EITHER_SELL": sub["monthly_either_state"].eq("SELL"),
    }
    rows = []
    for name, signal in signal_map.items():
        trigger = _first_true_date(signal.fillna(False), sub["date"])
        if trigger is None:
            rows.append({"signal_name": name, "first_trigger_date": pd.NaT, "whether_triggered_before_trough": False, "whether_triggered_before_current_backbone": False})
            continue
        trig_row = sub.loc[sub["date"].eq(trigger)].iloc[0]
        rows.append(
            {
                "signal_name": name,
                "first_trigger_date": trigger,
                "days_after_peak": int((trigger - peak).days),
                "days_before_trough": int((trough - trigger).days),
                "SPY_drawdown_at_trigger": trig_row["SPY_DD_FROM_HIGH"],
                "VIX_z_at_trigger": trig_row["VIX_ZSCORE_120D"],
                "credit_change_at_trigger": trig_row["D_CREDIT_SPREAD_20D"],
                "macro_regime_at_trigger": trig_row["macro_regime_confirmed"],
                "whether_triggered_before_trough": trigger <= trough,
                "whether_triggered_before_current_backbone": trigger <= backbone_entry if backbone_entry is not None else np.nan,
                "comments": "",
            }
        )
    out = pd.DataFrame(rows).sort_values(["first_trigger_date", "signal_name"], na_position="last")
    out.to_csv(CONFIG["output_dir"] / "signal_first_trigger_table_2015_2016.csv", index=False)
    return out


def analyze_monthly_either_failure(panel: pd.DataFrame, episode_dates: Dict[str, pd.Timestamp]) -> pd.DataFrame:
    monthly = panel.set_index("date")[["spy_price", "daily_rf", "monthly_either_state", "macro_regime_confirmed", "VIX_ZSCORE_120D", "CREDIT_SPREAD_BAA_AAA"]].copy()
    monthly["cash_nav"] = (1 + monthly["daily_rf"].fillna(0.0)).cumprod()
    month_end = monthly.resample("ME").last().dropna(subset=["spy_price"]).reset_index()
    month_end["spy_12m_return"] = month_end["spy_price"].pct_change(12)
    month_end["cash_12m_return"] = month_end["cash_nav"].pct_change(12)
    month_end["antonacci_abs_mom_state"] = np.where((month_end["spy_12m_return"] - month_end["cash_12m_return"]) > 0, "HOLD", "SELL")
    month_end["spy_10m_sma"] = month_end["spy_price"].rolling(10, min_periods=10).mean()
    month_end["spy_above_10m_sma"] = month_end["spy_price"] > month_end["spy_10m_sma"]
    month_end["faber_sma_state"] = np.where(month_end["spy_above_10m_sma"], "HOLD", "SELL")
    month_end["reconstructed_monthly_either_state"] = np.where(
        (month_end["antonacci_abs_mom_state"] == "HOLD") | (month_end["faber_sma_state"] == "HOLD"), "HOLD", "SELL"
    )
    month_end["SPY_drawdown_from_high_at_month_end"] = month_end["spy_price"] / month_end["spy_price"].cummax() - 1
    month_end["days_until_next_state_change"] = np.nan
    for i in range(len(month_end)):
        future = month_end.loc[i + 1 :]
        cur = month_end.loc[i, "monthly_either_state"]
        change = future.loc[future["monthly_either_state"] != cur]
        if not change.empty:
            month_end.loc[i, "days_until_next_state_change"] = (change.iloc[0]["date"] - month_end.loc[i, "date"]).days
    dates = panel["date"]
    rets = panel["SPY_return"].fillna(0.0)
    out_rows = []
    for _, r in month_end.iterrows():
        after = panel.loc[panel["date"] > r["date"]]
        nxt21 = (1 + after["SPY_return"].head(21).fillna(0.0)).prod() - 1 if not after.empty else np.nan
        nxt63 = (1 + after["SPY_return"].head(63).fillna(0.0)).prod() - 1 if not after.empty else np.nan
        out_rows.append(
            {
                "month_end_date": r["date"],
                "monthly_either_state": r["monthly_either_state"],
                "antonacci_abs_mom_state": r["antonacci_abs_mom_state"],
                "faber_sma_state": r["faber_sma_state"],
                "spy_price_month_end": r["spy_price"],
                "spy_12m_return": r["spy_12m_return"],
                "spy_10m_sma": r["spy_10m_sma"],
                "spy_above_10m_sma": r["spy_above_10m_sma"],
                "cash_return_proxy": r["cash_12m_return"],
                "days_until_next_state_change": r["days_until_next_state_change"],
                "SPY_drawdown_from_high_at_month_end": r["SPY_drawdown_from_high_at_month_end"],
                "SPY_return_next_21d": nxt21,
                "SPY_return_next_63d": nxt63,
                "macro_regime_confirmed_at_month_end": r["macro_regime_confirmed"],
                "VIX_ZSCORE_120D_at_month_end": r["VIX_ZSCORE_120D"],
                "CREDIT_SPREAD_at_month_end": r["CREDIT_SPREAD_BAA_AAA"],
                "CMDTY_RET60_at_month_end": panel.loc[panel["date"].eq(r["date"]), "CMDTY_RET_60D"].iloc[0] if "CMDTY_RET_60D" in panel.columns and any(panel["date"].eq(r["date"])) else np.nan,
            }
        )
    out = pd.DataFrame(out_rows)
    out = out[(out["month_end_date"] >= pd.Timestamp("2015-01-01")) & (out["month_end_date"] <= pd.Timestamp(CONFIG["case_end"]))]
    out.to_csv(CONFIG["output_dir"] / "monthly_either_failure_analysis.csv", index=False)
    return out


def rebuild_backbone_signal_timeline(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["FLAT_VIX_STRESS"] = out["macro_regime_confirmed"].eq("FLAT") & (out["VIX_ZSCORE_120D"] >= 3.0)
    out["FLAT_CREDIT_DD5_STRESS"] = out["macro_regime_confirmed"].eq("FLAT") & (out["spy_drawdown_from_previous_high"] <= -0.05) & (out["D_CREDIT_SPREAD_20D"] > 0.10)
    out["STEEP_EITHER_SELL_STRESS"] = out["macro_regime_confirmed"].eq("STEEP") & out["monthly_either_state"].eq("SELL")
    out["STEEP_CREDIT_DD5_STRESS"] = out["macro_regime_confirmed"].eq("STEEP") & (out["spy_drawdown_from_previous_high"] <= -0.05) & (out["D_CREDIT_SPREAD_20D"] > 0.10)
    out["BACKBONE_V2_ENTRY_SIGNAL"] = out["FLAT_VIX_STRESS"] | out["FLAT_CREDIT_DD5_STRESS"] | out["STEEP_EITHER_SELL_STRESS"] | out["STEEP_CREDIT_DD5_STRESS"]
    if "timing_state" in out.columns:
        out["BACKBONE_V2_RISK_STATE"] = out["timing_state"].eq("RISK")
    elif "BACKBONE_V2_UPGRADED_weight_spy" in out.columns:
        out["BACKBONE_V2_RISK_STATE"] = pd.to_numeric(out["BACKBONE_V2_UPGRADED_weight_spy"], errors="coerce").fillna(1.0) < 0.5
    else:
        state = []
        cur = "NON_RISK"
        for _, r in out.iterrows():
            state.append(cur == "RISK")
            if cur == "NON_RISK" and bool(r["BACKBONE_V2_ENTRY_SIGNAL"]):
                cur = "RISK"
            elif cur == "RISK" and bool(r["SPY_CROSS_ABOVE_MA20"]):
                cur = "NON_RISK"
        out["BACKBONE_V2_RISK_STATE"] = state
    out["R3_RECOVERY"] = out["SPY_CROSS_ABOVE_MA20"]
    case = out[(out["date"] >= pd.Timestamp(CONFIG["case_start"])) & (out["date"] <= pd.Timestamp(CONFIG["case_end"]))].copy()
    cols = [
        "date", "macro_regime_confirmed", "monthly_either_state", "FLAT_VIX_STRESS", "FLAT_CREDIT_DD5_STRESS",
        "STEEP_EITHER_SELL_STRESS", "STEEP_CREDIT_DD5_STRESS", "BACKBONE_V2_ENTRY_SIGNAL", "BACKBONE_V2_RISK_STATE",
        "R3_RECOVERY", "spy_drawdown_from_previous_high", "SPY_return", "CMDTY_RET_60D", "VIX_ZSCORE_120D", "D_CREDIT_SPREAD_20D"
    ]
    case[[c for c in cols if c in case.columns]].to_csv(CONFIG["output_dir"] / "backbone_signal_timeline_2015_2016.csv", index=False)
    return out


def evaluate_repair_signal_candidates(panel: pd.DataFrame, episode_dates: Dict[str, pd.Timestamp], backbone_case: pd.DataFrame) -> pd.DataFrame:
    peak = episode_dates["peak_date"]
    trough = episode_dates["trough_date"]
    signals = {
        "PURE_TECHNICAL_DD8": panel["SPY_DD_FROM_HIGH"] <= -0.08,
        "PURE_TECHNICAL_BELOW_MA200_5D": panel["SPY_below_MA200"].rolling(5, min_periods=5).sum() >= 5,
        "PURE_TECHNICAL_RET60_NEG8": panel["SPY_RET_60D"] < -0.08,
        "COMMODITY_GROWTH_DD5_CMDTY60_NEG10": (panel["SPY_DD_FROM_HIGH"] <= -0.05) & (panel["CMDTY_RET_60D"] < -0.10),
        "COMMODITY_GROWTH_DD5_CMDTY60_NEG10_CREDIT_POS": (panel["SPY_DD_FROM_HIGH"] <= -0.05) & (panel["CMDTY_RET_60D"] < -0.10) & (panel["D_CREDIT_SPREAD_20D"] > 0),
        "COMMODITY_GROWTH_CMDTY60_NEG10_VIX_LT3": (panel["CMDTY_RET_60D"] < -0.10) & (panel["VIX_ZSCORE_120D"] < 3.0),
        "MILD_CREDIT_DD5_CREDIT_0_05": (panel["SPY_DD_FROM_HIGH"] <= -0.05) & (panel["D_CREDIT_SPREAD_20D"] > 0.05),
        "MILD_CREDIT_DD5_CREDIT_0": (panel["SPY_DD_FROM_HIGH"] <= -0.05) & (panel["D_CREDIT_SPREAD_20D"] > 0),
        "TECH_CMDTY_MA200_AND_CMDTY60_NEG10": panel["SPY_below_MA200"] & (panel["CMDTY_RET_60D"] < -0.10),
        "TECH_CMDTY_RET60_NEG8_AND_CMDTY60_NEG10": (panel["SPY_RET_60D"] < -0.08) & (panel["CMDTY_RET_60D"] < -0.10),
    }
    future_21_mdd = []
    for i in range(len(panel)):
        fwd = panel["SPY_return"].iloc[i + 1 : i + 22].fillna(0.0)
        if fwd.empty:
            future_21_mdd.append(np.nan)
            continue
        nav = (1 + fwd).cumprod()
        future_21_mdd.append(float((nav / nav.cummax() - 1).min()))
    panel = panel.copy()
    panel["future_21d_mdd"] = future_21_mdd
    rows = []
    for name, sig in signals.items():
        event = sig.fillna(False) & ~sig.fillna(False).shift(1).fillna(False)
        events = panel.loc[event].copy()
        event_count = len(events)
        false_alarm_rate = float((events["future_21d_mdd"] > -0.05).mean()) if event_count else np.nan
        p_21 = float((events["future_21d_mdd"] <= -0.05).mean()) if event_count else np.nan
        case_trigger = panel.loc[(panel["date"] >= peak) & (panel["date"] <= trough) & sig.fillna(False)]
        first = case_trigger.iloc[0] if not case_trigger.empty else None
        rows.append(
            {
                "signal_name": name,
                "first_trigger_date_in_case": first["date"] if first is not None else pd.NaT,
                "days_after_peak": (pd.Timestamp(first["date"]) - peak).days if first is not None else np.nan,
                "days_before_trough": (trough - pd.Timestamp(first["date"])).days if first is not None else np.nan,
                "SPY_dd_at_trigger": first["SPY_DD_FROM_HIGH"] if first is not None else np.nan,
                "historical_event_count_full_sample": event_count,
                "false_alarm_rate_21d_full_sample": false_alarm_rate,
                "P_21D_MDD_LT_NEG5_full_sample": p_21,
                "overlaps_COVID_undesirably": bool(events["date"].between(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-06-30")).any()),
                "overlaps_2022_undesirably": bool(events["date"].between(pd.Timestamp("2021-11-01"), pd.Timestamp("2023-03-31")).any()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["days_after_peak", "false_alarm_rate_21d_full_sample"], na_position="last")
    out.to_csv(CONFIG["output_dir"] / "repair_signal_candidates_summary.csv", index=False)
    return out


def plot_macro_percentiles(panel: pd.DataFrame, daily_pct: pd.DataFrame) -> None:
    vars_to_plot = ["growth_pc1", "inflation_pc1", "VIX_ZSCORE_120D", "CREDIT_SPREAD_BAA_AAA", "D_CREDIT_SPREAD_20D", "term_spread", "GS10", "GS1"]
    case = panel[(panel["date"] >= pd.Timestamp(CONFIG["case_start"])) & (panel["date"] <= pd.Timestamp(CONFIG["case_end"]))].copy()
    fig, axes = plt.subplots(len(vars_to_plot) + 2, 1, figsize=(14, 2.3 * (len(vars_to_plot) + 2)), sharex=True)
    axes[0].plot(case["date"], case["spy_drawdown_from_previous_high"], color="black")
    axes[0].set_title("SPY Drawdown")
    axes[1].plot(case["date"], case["macro_regime_confirmed"].astype("category").cat.codes, color="tab:gray")
    axes[1].set_title("Regime Strip")
    for ax, var in zip(axes[2:], vars_to_plot):
        sub = daily_pct[daily_pct["variable"].eq(var)]
        if sub.empty:
            ax.text(0.5, 0.5, f"{var} missing", ha="center", va="center")
        else:
            ax.plot(pd.to_datetime(sub["date"]), sub["full_sample_percentile"], label=f"{var} pct")
            if var in case.columns:
                ax2 = ax.twinx()
                ax2.plot(case["date"], case[var], color="tab:red", alpha=0.35)
            ax.set_title(var)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "case_2015_2016_macro_percentiles.png", dpi=150)
    plt.close(fig)


def plot_asset_navs(windows: Dict[str, pd.DataFrame]) -> None:
    case = windows["full_case"].copy()
    fig, ax = plt.subplots(figsize=(14, 7))
    for asset in CONFIG["asset_list"]:
        col = f"{asset}_return"
        if col not in case.columns:
            continue
        nav = (1 + case[col].fillna(0.0)).cumprod()
        ax.plot(case["date"], nav / nav.iloc[0], label=asset)
    ax.legend()
    ax.set_title("2015-2016 Asset NAVs")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "case_2015_2016_asset_navs.png", dpi=150)
    plt.close(fig)


def plot_spy_technicals(panel: pd.DataFrame) -> None:
    case = panel[(panel["date"] >= pd.Timestamp(CONFIG["case_start"])) & (panel["date"] <= pd.Timestamp(CONFIG["case_end"]))].copy()
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(case["date"], case["spy_price"], label="SPY")
    for w in [20, 50, 100, 200]:
        axes[0].plot(case["date"], case[f"SPY_MA{w}"], label=f"MA{w}")
    axes[0].legend(ncol=5, fontsize=8)
    axes[0].set_title("SPY Price and Moving Averages")
    axes[1].plot(case["date"], case["SPY_DD_FROM_HIGH"], color="black")
    axes[1].set_title("SPY Drawdown")
    axes[2].plot(case["date"], case["SPY_RET_20D"], label="RET20D")
    axes[2].plot(case["date"], case["SPY_RET_60D"], label="RET60D")
    axes[2].plot(case["date"], case["SPY_RET_120D"], label="RET120D")
    axes[2].legend()
    axes[2].set_title("SPY Momentum")
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "case_2015_2016_spy_technical.png", dpi=150)
    plt.close(fig)


def plot_backbone_timeline(panel: pd.DataFrame) -> None:
    case = panel[(panel["date"] >= pd.Timestamp(CONFIG["case_start"])) & (panel["date"] <= pd.Timestamp(CONFIG["case_end"]))].copy()
    fig, axes = plt.subplots(7, 1, figsize=(14, 12), sharex=True)
    axes[0].plot(case["date"], case["spy_drawdown_from_previous_high"], color="black")
    axes[0].set_title("SPY Drawdown")
    axes[1].plot(case["date"], case["macro_regime_confirmed"].astype("category").cat.codes)
    axes[1].set_title("Regime")
    axes[2].plot(case["date"], case["monthly_either_state"].eq("SELL").astype(int))
    axes[2].set_title("Monthly Either SELL")
    for ax, col in zip(axes[3:], ["FLAT_VIX_STRESS", "FLAT_CREDIT_DD5_STRESS", "STEEP_EITHER_SELL_STRESS", "BACKBONE_V2_RISK_STATE"]):
        ax.plot(case["date"], case[col].astype(int))
        ax.set_title(col)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "case_2015_2016_backbone_timeline.png", dpi=150)
    plt.close(fig)


def plot_monthly_either_components(monthly: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(monthly["month_end_date"], monthly["spy_price_month_end"], label="SPY")
    axes[0].plot(monthly["month_end_date"], monthly["spy_10m_sma"], label="10M SMA")
    axes[0].legend()
    axes[0].set_title("Month-End SPY and 10M SMA")
    axes[1].plot(monthly["month_end_date"], monthly["spy_12m_return"], label="SPY 12M return")
    axes[1].plot(monthly["month_end_date"], monthly["cash_return_proxy"], label="Cash 12M return")
    axes[1].legend()
    axes[1].set_title("Antonacci Inputs")
    axes[2].plot(monthly["month_end_date"], monthly["monthly_either_state"].eq("SELL").astype(int), label="Monthly Either SELL")
    axes[2].plot(monthly["month_end_date"], monthly["SPY_drawdown_from_high_at_month_end"], label="SPY DD")
    axes[2].legend()
    axes[2].set_title("Monthly Either vs Drawdown")
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "monthly_either_component_2015_2016.png", dpi=150)
    plt.close(fig)


def write_markdown_report(
    episode: pd.DataFrame,
    macro_summary: pd.DataFrame,
    regime_dist: pd.DataFrame,
    regime_timing: pd.DataFrame,
    asset_perf: pd.DataFrame,
    trigger_table: pd.DataFrame,
    monthly_failure: pd.DataFrame,
    repair_signals: pd.DataFrame,
    backbone: pd.DataFrame,
) -> None:
    out = CONFIG["output_dir"] / "2015_2016_DRAWDOWN_FORENSIC_REPORT.md"
    peak = pd.to_datetime(episode.loc[0, "peak_date"])
    trough = pd.to_datetime(episode.loc[0, "trough_date"])
    backbone_case = backbone[(backbone["date"] >= pd.Timestamp(CONFIG["case_start"])) & (backbone["date"] <= pd.Timestamp(CONFIG["case_end"]))]
    entry = backbone_case.loc[backbone_case["BACKBONE_V2_ENTRY_SIGNAL"]]
    earliest_backbone = entry["date"].min() if not entry.empty else pd.NaT
    monthly_sell = monthly_failure.loc[monthly_failure["monthly_either_state"].eq("SELL"), "month_end_date"]
    first_monthly_sell = monthly_sell.min() if not monthly_sell.empty else pd.NaT
    top_signals = repair_signals.head(3)
    content = f"""# 2015_2016_DRAWDOWN_FORENSIC_REPORT

## Purpose

This report isolates the 2015-05-01 to 2016-03-31 missed drawdown window and diagnoses why the current framework did not avoid it cleanly.

## Drawdown Episode Definition

{episode.to_markdown(index=False)}

## Macro Percentile Analysis

{macro_summary.to_markdown(index=False)}

## Regime Context

{regime_dist.to_markdown(index=False)}

{regime_timing.to_markdown(index=False)}

## Asset Behavior

{asset_perf.to_markdown(index=False)}

## SPY Technical Trigger Timing

{trigger_table.to_markdown(index=False)}

## Monthly Either Failure

{monthly_failure.to_markdown(index=False)}

## Repair Signal Candidates

{repair_signals.to_markdown(index=False)}

## Summary

- Peak date: `{peak.date()}`
- Trough date: `{trough.date()}`
- First Monthly Either SELL in case: `{first_monthly_sell.date() if pd.notna(first_monthly_sell) else "not triggered"}`
- First backbone entry signal in case: `{earliest_backbone.date() if pd.notna(earliest_backbone) else "not triggered"}`
- Top repair candidates by early trigger / event quality: `{", ".join(top_signals["signal_name"].tolist())}`
"""
    out.write_text(content, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = load_base_panel()
    panel = load_macro_variables(panel)
    panel = load_asset_returns(panel)
    episode, episode_dates = identify_spy_drawdown_episode(panel)
    windows = _window_slices(panel, episode_dates)
    macro_summary, macro_daily = compute_macro_percentiles(panel, episode_dates)
    regime_dist, regime_timing = analyze_regime_distribution(panel, windows, episode_dates)
    asset_perf = compute_asset_performance_windows(panel, windows)
    panel = build_spy_technical_indicators(panel)
    backbone = rebuild_backbone_signal_timeline(panel)
    trigger_table = compute_signal_first_trigger_table(backbone, episode_dates)
    monthly_failure = analyze_monthly_either_failure(panel, episode_dates)
    repair_signals = evaluate_repair_signal_candidates(backbone, episode_dates, backbone)

    plot_macro_percentiles(panel, macro_daily)
    plot_asset_navs(windows)
    plot_spy_technicals(panel)
    plot_backbone_timeline(backbone)
    plot_monthly_either_components(monthly_failure)

    write_markdown_report(
        episode,
        macro_summary,
        regime_dist,
        regime_timing,
        asset_perf,
        trigger_table,
        monthly_failure,
        repair_signals,
        backbone,
    )

    peak = episode.loc[0, "peak_date"]
    trough = episode.loc[0, "trough_date"]
    print(f"1. 2015-2016 SPY peak / trough / maxDD: {peak} / {trough} / {episode.loc[0, 'max_drawdown']:.2%}")
    print("2. macro_regime distribution in window:")
    print(regime_dist[["regime", "n_days", "percentage"]].to_string(index=False))
    print("3. macro percentile summary:")
    print(macro_summary[["variable", "percentile_at_peak", "percentile_at_trough"]].to_string(index=False))
    core = asset_perf[asset_perf["window"].isin(["full_case", "peak_to_trough"]) & asset_perf["asset"].isin(CONFIG["asset_list"])]
    print("4. asset performance full-case / peak-to-trough:")
    print(core[["window", "asset", "cumulative_return", "max_drawdown"]].to_string(index=False))
    monthly_sell = monthly_failure.loc[monthly_failure["monthly_either_state"].eq("SELL"), "month_end_date"]
    print(f"5. Monthly Either trigger: {monthly_sell.min() if not monthly_sell.empty else 'not triggered'}")
    entry = backbone.loc[(backbone["date"] >= pd.Timestamp(CONFIG["case_start"])) & (backbone["date"] <= pd.Timestamp(CONFIG["case_end"])) & backbone["BACKBONE_V2_ENTRY_SIGNAL"], "date"]
    print(f"6. Backbone V2 trigger: {entry.min() if not entry.empty else 'not triggered'}")
    print("7. Earliest trigger signals:")
    print(trigger_table.head(5)[["signal_name", "first_trigger_date", "days_after_peak"]].to_string(index=False))
    print("8. Top repair candidates:")
    print(repair_signals.head(3)[["signal_name", "first_trigger_date_in_case", "false_alarm_rate_21d_full_sample", "P_21D_MDD_LT_NEG5_full_sample"]].to_string(index=False))
    print(f"9. Output path: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
