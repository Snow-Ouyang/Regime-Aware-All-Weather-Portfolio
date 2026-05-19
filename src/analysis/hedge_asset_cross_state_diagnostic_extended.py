"""Extended hedge asset cross-state diagnostic.

This module extends the original hedge asset cross-state diagnostic with:
- Equity satellites: IJH, IWM
- Duration hedges: TLT, EDV

It reuses the current BACKBONE_V2_UPGRADED timing definition and the project's
Yahoo Finance download framework for missing local tickers.
"""

from __future__ import annotations

import os
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


try:
    from src.data.download_yahoo_assets import download_yahoo_adjusted_close
except Exception:  # pragma: no cover - local import fallback
    download_yahoo_adjusted_close = None


CONFIG = {
    "output_dir": Path("results/hedge_asset_cross_state_diagnostic_extended"),
    "figure_dir": Path("figures/hedge_asset_cross_state_diagnostic_extended"),
    "new_tickers": ["IJH", "IWM", "TLT", "EDV"],
    "assets": ["SPY", "GOLD", "IEF", "TLT", "EDV", "CASH", "CMDTY_FUT", "IJH", "IWM"],
    "download_start": "2005-01-01",
    "timing_backbone": "BACKBONE_V2_UPGRADED",
    "trading_days_per_year": 252,
    "case_windows": {
        "2008_GFC": ["2007-10-01", "2009-06-30"],
        "2015_2016": ["2015-05-01", "2016-03-31"],
        "RUSSIA_UKRAINE_WAR": ["2022-02-24", "2022-06-30"],
        "2018Q4": ["2018-10-01", "2019-01-31"],
        "COVID_2020": ["2020-02-01", "2020-06-30"],
        "2022": ["2021-11-01", "2023-03-31"],
        "2023": ["2023-01-01", "2023-12-31"],
        "2025_PULLBACK": ["2025-01-01", "2025-12-31"],
        "2024_2026": ["2024-01-01", "latest"],
    },
}


def _set_proxy() -> None:
    proxy = "http://127.0.0.1:7890"
    os.environ["HTTP_PROXY"] = proxy
    os.environ["HTTPS_PROXY"] = proxy


BASE_PANEL_CANDIDATES = [
    Path("results/spy_cash_backbone_upgrade_ablation/daily_backtest_panel.csv"),
    Path("results/regime_aware_risk_parity_allocation/daily_backtest_panel.csv"),
    Path("results/regime_aware_hedge_allocation_v1/daily_backtest_panel.csv"),
]

EVENT_LOG_CANDIDATES = [
    Path("results/spy_cash_backbone_upgrade_ablation/risk_state_event_log.csv"),
    Path("results/spy_cash_stress_recovery_with_credit/risk_state_event_log.csv"),
    Path("results/spy_cash_stress_recovery_with_commodity/risk_state_event_log.csv"),
]

RETURN_PANEL_CANDIDATES = [
    Path("results/regime_aware_risk_parity_allocation/daily_backtest_panel.csv"),
    Path("results/regime_aware_hedge_allocation_steep_mix/daily_backtest_panel.csv"),
    Path("results/regime_aware_hedge_allocation_steep_test/daily_backtest_panel.csv"),
    Path("results/regime_aware_hedge_allocation_v1/daily_backtest_panel.csv"),
    Path("results/regime_hedge_steep_sell_ief/daily_backtest_panel.csv"),
    Path("results/reconstructed_regime_asset_behavior/reconstructed_regime_panel.csv"),
]


def ensure_dirs() -> None:
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["figure_dir"].mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "raw" / "asset").mkdir(parents=True, exist_ok=True)


def _display_path(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT.resolve()))
    except Exception:
        try:
            return str(p.relative_to(ROOT))
        except Exception:
            return str(p)


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = _first_existing(df.columns, ["date", "DATE", "Date", "observation_date", "datetime"])
    if not date_col:
        raise ValueError(f"No date column found in {path}")
    if date_col != "date":
        df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date")


def _first_existing(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols = list(cols)
    col_set = set(cols)
    for candidate in candidates:
        if candidate in col_set:
            return candidate
    lower_map = {str(c).lower(): c for c in cols}
    for candidate in candidates:
        key = str(candidate).lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _to_bool_state(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    s = series.astype(str).str.upper()
    return s.isin(["RISK", "TRUE", "1", "SELL", "CASH"])


def load_base_panel() -> pd.DataFrame:
    for path in BASE_PANEL_CANDIDATES:
        if path.exists():
            df = _read_csv(path)
            print(f"Loaded base panel: {path}")
            return df
    raise FileNotFoundError("No base panel found for extended cross-state diagnostic.")


def load_event_log() -> Optional[pd.DataFrame]:
    for path in EVENT_LOG_CANDIDATES:
        if not path.exists():
            continue
        log = pd.read_csv(path)
        if "event_date" not in log.columns:
            continue
        log["event_date"] = pd.to_datetime(log["event_date"])
        if "strategy" in log.columns:
            strategy_col = log["strategy"].astype(str)
            if CONFIG["timing_backbone"] in set(strategy_col):
                log = log[strategy_col.eq(CONFIG["timing_backbone"])]
            elif "BACKBONE_V2_UPGRADED" in set(strategy_col):
                log = log[strategy_col.eq("BACKBONE_V2_UPGRADED")]
        if "event_type" in log.columns:
            log = log[log["event_type"].astype(str).eq("ENTER_RISK")]
        if not log.empty:
            print(f"Loaded risk event log: {path}")
            return log
    warnings.warn("No risk event log found; entry reasons will be inferred.")
    return None


def _ensure_core_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "spy_daily_return" not in out.columns and "SPY_return" in out.columns:
        out["spy_daily_return"] = out["SPY_return"]
    if "spy_price" not in out.columns and "SPY_price" in out.columns:
        out["spy_price"] = out["SPY_price"]
    if "spy_drawdown_from_previous_high" not in out.columns and "spy_price" in out.columns:
        out["spy_drawdown_from_previous_high"] = out["spy_price"] / out["spy_price"].cummax() - 1
    if "SPY_MA20" not in out.columns and "spy_price" in out.columns:
        out["SPY_MA20"] = out["spy_price"].rolling(20, min_periods=20).mean()
    if "SPY_CROSS_ABOVE_MA20" not in out.columns and {"spy_price", "SPY_MA20"}.issubset(out.columns):
        out["SPY_CROSS_ABOVE_MA20"] = (
            out["spy_price"].gt(out["SPY_MA20"])
            & out["spy_price"].shift(1).le(out["SPY_MA20"].shift(1))
        )
    if "VIX_ZSCORE_120D" not in out.columns and "VIX_LEVEL" in out.columns:
        mean_ = out["VIX_LEVEL"].rolling(120, min_periods=120).mean()
        std_ = out["VIX_LEVEL"].rolling(120, min_periods=120).std(ddof=0)
        out["VIX_ZSCORE_120D"] = (out["VIX_LEVEL"] - mean_) / std_
    if "D_CREDIT_SPREAD_20D" not in out.columns and "CREDIT_SPREAD_BAA_AAA" in out.columns:
        out["D_CREDIT_SPREAD_20D"] = out["CREDIT_SPREAD_BAA_AAA"] - out["CREDIT_SPREAD_BAA_AAA"].shift(20)
    return out


def _load_raw_asset_series(path: Path, asset: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    raw = pd.read_csv(path, na_values=[".", "NA", ""])
    date_col = _first_existing(raw.columns, ["date", "DATE", "Date", "observation_date"])
    if not date_col:
        return None
    preferred = [
        asset,
        f"{asset}_price",
        "Adj Close",
        "adj_close",
        "adjusted_close",
        "Close",
        "close",
        "VALUE",
        "value",
    ]
    value_col = _first_existing(raw.columns, preferred)
    if not value_col:
        numeric_cols = [c for c in raw.columns if c != date_col and pd.to_numeric(raw[c], errors="coerce").notna().any()]
        if not numeric_cols:
            return None
        value_col = numeric_cols[0]
    out = raw[[date_col, value_col]].rename(columns={date_col: "date", value_col: f"{asset}_price"})
    out["date"] = pd.to_datetime(out["date"])
    out[f"{asset}_price"] = pd.to_numeric(out[f"{asset}_price"], errors="coerce")
    out = out.sort_values("date").drop_duplicates("date")
    out[f"{asset}_return"] = out[f"{asset}_price"].pct_change(fill_method=None)
    return out


def _asset_file_candidates(asset: str) -> List[Path]:
    return [
        ROOT / "data" / "raw" / "asset" / f"{asset}.csv",
        ROOT / "data" / "raw" / "etf" / f"{asset}.csv",
        ROOT / "data" / "raw" / f"{asset}.csv",
        ROOT / "data" / "raw" / "macro" / "commodity" / f"{asset}.csv",
        ROOT / "data" / "raw" / "macro" / "dollar" / f"{asset}.csv",
    ]


def download_or_load_new_assets() -> Tuple[Dict[str, pd.DataFrame], Dict[str, str], List[str]]:
    loaded: Dict[str, pd.DataFrame] = {}
    origins: Dict[str, str] = {}
    notes: List[str] = []

    for ticker in CONFIG["new_tickers"]:
        for path in _asset_file_candidates(ticker):
            asset_df = _load_raw_asset_series(path, ticker)
            if asset_df is not None:
                loaded[ticker] = asset_df
                origins[ticker] = _display_path(path)
                break

    missing = [ticker for ticker in CONFIG["new_tickers"] if ticker not in loaded]
    if missing and download_yahoo_adjusted_close is not None:
        _set_proxy()
        prices, log_df = download_yahoo_adjusted_close(missing, start_date=CONFIG["download_start"])
        if not log_df.empty:
            log_df.to_csv(CONFIG["output_dir"] / "new_asset_download_log.csv", index=False)
            failed = log_df[log_df["download_status"].astype(str).ne("success")]
            for _, row in failed.iterrows():
                notes.append(f"download failed for {row['ticker']}: {row.get('error_message', '')}")
        for ticker in missing:
            if ticker not in prices.columns:
                notes.append(f"download failed or empty for {ticker}")
                continue
            asset_df = prices[[ticker]].reset_index().rename(columns={"date": "date", ticker: f"{ticker}_price"})
            if "date" not in asset_df.columns:
                asset_df = asset_df.rename(columns={asset_df.columns[0]: "date"})
            asset_df["date"] = pd.to_datetime(asset_df["date"])
            asset_df[f"{ticker}_return"] = asset_df[f"{ticker}_price"].pct_change(fill_method=None)
            save_path = ROOT / "data" / "raw" / "asset" / f"{ticker}.csv"
            asset_df.rename(columns={f"{ticker}_price": "Adj Close"}).to_csv(save_path, index=False)
            asset_df = asset_df.rename(columns={"Adj Close": f"{ticker}_price"})
            loaded[ticker] = asset_df
            origins[ticker] = _display_path(save_path)
            notes.append(f"downloaded {ticker} with project Yahoo Finance framework")

    return loaded, origins, notes


def load_existing_asset_returns(base: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    panel = base.copy()
    panel = _ensure_core_fields(panel)
    notes: List[str] = []
    sources: Dict[str, str] = {}

    if "SPY_return" not in panel.columns:
        panel["SPY_return"] = panel.get("spy_daily_return")
    if "SPY_nav" not in panel.columns and "SPY_return" in panel.columns:
        panel["SPY_nav"] = (1 + pd.to_numeric(panel["SPY_return"], errors="coerce").fillna(0.0)).cumprod()
    sources["SPY"] = "base_panel"

    panel["CASH_return"] = pd.to_numeric(panel.get("CASH_return", panel.get("daily_rf")), errors="coerce").fillna(0.0)
    panel["CASH_nav"] = (1 + panel["CASH_return"]).cumprod()
    sources["CASH"] = "base_panel.daily_rf"

    alias_map = {
        "GOLD": ["GOLD_return", "GLD_return", "gold_return"],
        "IEF": ["IEF_return", "ief_return"],
        "CMDTY_FUT": ["CMDTY_FUT_return", "CMDTY_return", "commodity_return"],
        "IJH": ["IJH_return"],
        "IWM": ["IWM_return"],
        "TLT": ["TLT_return"],
        "EDV": ["EDV_return"],
    }

    for path in RETURN_PANEL_CANDIDATES:
        if not path.exists():
            continue
        src = _read_csv(path)
        rename_map = {}
        for asset, candidates in alias_map.items():
            if f"{asset}_return" in panel.columns:
                continue
            col = _first_existing(src.columns, candidates)
            if col:
                rename_map[col] = f"{asset}_return"
        if not rename_map:
            continue
        add = src[["date"] + list(rename_map.keys())].rename(columns=rename_map)
        for col in rename_map.values():
            if col in panel.columns:
                panel = panel.drop(columns=[col])
            panel = panel.merge(add, on="date", how="left")
        for asset, col in rename_map.items():
            pass
        for old, new in rename_map.items():
            sources[new.replace("_return", "")] = _display_path(path)
        notes.append(f"merged returns from {_display_path(path)}")

    raw_aliases = {
        "GOLD": ["GOLD", "GLD"],
        "IEF": ["IEF"],
        "CMDTY_FUT": ["CMDTY_FUT", "CMDTY"],
        "IJH": ["IJH"],
        "IWM": ["IWM"],
        "TLT": ["TLT"],
        "EDV": ["EDV"],
    }
    downloaded_assets, downloaded_sources, download_notes = download_or_load_new_assets()
    notes.extend(download_notes)

    for asset, aliases in raw_aliases.items():
        if f"{asset}_return" in panel.columns:
            continue
        raw = None
        origin = None
        if asset in downloaded_assets:
            raw = downloaded_assets[asset]
            origin = downloaded_sources.get(asset)
        else:
            for alias in aliases:
                for path in _asset_file_candidates(alias):
                        raw = _load_raw_asset_series(path, asset)
                        if raw is not None:
                            origin = _display_path(path)
                            break
                if raw is not None:
                    break
        if raw is None:
            notes.append(f"missing asset data: {asset}")
            continue
        panel = panel.merge(raw[["date", f"{asset}_return"]], on="date", how="left")
        if f"{asset}_price" in raw.columns:
            panel = panel.merge(raw[["date", f"{asset}_price"]], on="date", how="left")
        sources[asset] = origin or "downloaded"

    for asset in CONFIG["assets"]:
        ret_col = f"{asset}_return"
        if ret_col not in panel.columns:
            continue
        panel[ret_col] = pd.to_numeric(panel[ret_col], errors="coerce")
        if asset == "CASH":
            panel[ret_col] = panel[ret_col].fillna(0.0)
        panel[f"{asset}_nav"] = (1 + panel[ret_col].fillna(0.0)).cumprod()

    return panel.sort_values("date").reset_index(drop=True), sources, notes


def build_backbone_v2_state(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    weight_col = _first_existing(
        df.columns,
        [
            "BACKBONE_V2_UPGRADED_weight_spy",
            "best_weight_spy",
            "BACKBONE_V2_SPY_CASH_weight_SPY",
        ],
    )
    risk_col = _first_existing(
        df.columns,
        [
            "BACKBONE_V2_UPGRADED_risk_state",
            "BACKBONE_V2_SPY_CASH_risk_state",
        ],
    )
    if weight_col:
        is_risk = pd.to_numeric(df[weight_col], errors="coerce").fillna(1.0) < 0.5
    elif risk_col:
        is_risk = _to_bool_state(df[risk_col])
    else:
        flat_signal = df["macro_regime_confirmed"].eq("FLAT") & (
            df["VIX_ZSCORE_120D"].ge(3.0)
            | (
                df["spy_drawdown_from_previous_high"].le(-0.05)
                & df["D_CREDIT_SPREAD_20D"].gt(0.10)
            )
        )
        steep_signal = df["macro_regime_confirmed"].eq("STEEP") & (
            df["monthly_either_state"].astype(str).eq("SELL")
            | (
                df["spy_drawdown_from_previous_high"].le(-0.05)
                & df["D_CREDIT_SPREAD_20D"].gt(0.10)
            )
        )
        entry_signal = flat_signal | steep_signal
        is_risk = pd.Series(False, index=df.index)
        state = False
        for i in range(len(df)):
            is_risk.iloc[i] = state
            if state:
                if bool(df.iloc[i].get("SPY_CROSS_ABOVE_MA20", False)):
                    state = False
            else:
                if bool(entry_signal.iloc[i]):
                    state = True
        is_risk = is_risk.astype(bool)
    df["timing_state"] = np.where(is_risk, "RISK", "NON_RISK")
    df["cross_state"] = df["macro_regime_confirmed"].fillna("NEUTRAL").astype(str) + "_" + df["timing_state"]
    return df


def _infer_entry_reason(row: pd.Series) -> str:
    if row.get("macro_regime_confirmed") == "FLAT" and pd.notna(row.get("VIX_ZSCORE_120D")) and row["VIX_ZSCORE_120D"] >= 3.0:
        return "FLAT_VIX_STRESS"
    if row.get("macro_regime_confirmed") == "FLAT" and row.get("spy_drawdown_from_previous_high", 0) <= -0.05 and row.get("D_CREDIT_SPREAD_20D", 0) > 0.10:
        return "FLAT_CREDIT_DD5_STRESS"
    if row.get("macro_regime_confirmed") == "STEEP" and str(row.get("monthly_either_state", "")).upper() == "SELL":
        return "STEEP_EITHER_SELL_STRESS"
    if row.get("macro_regime_confirmed") == "STEEP" and row.get("spy_drawdown_from_previous_high", 0) <= -0.05 and row.get("D_CREDIT_SPREAD_20D", 0) > 0.10:
        return "STEEP_CREDIT_DD5_STRESS"
    return "OTHER"


def build_cross_state_labels(panel: pd.DataFrame, event_log: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = panel.copy()
    reason_map: Dict[pd.Timestamp, str] = {}
    if event_log is not None and {"event_date", "reason"}.issubset(event_log.columns):
        reason_map = (
            event_log.drop_duplicates("event_date")
            .assign(reason=lambda x: x["reason"].astype(str))
            .set_index("event_date")["reason"]
            .to_dict()
        )
    df["entry_reason"] = ""
    starts = df.index[df["timing_state"].eq("RISK") & ~df["timing_state"].shift(1, fill_value="NON_RISK").eq("RISK")]
    for start_idx in starts:
        reason = reason_map.get(pd.Timestamp(df.loc[start_idx, "date"]), _infer_entry_reason(df.loc[start_idx]))
        end_idx = start_idx
        while end_idx + 1 < len(df) and df.loc[end_idx + 1, "timing_state"] == "RISK":
            end_idx += 1
        df.loc[start_idx:end_idx, "entry_reason"] = reason
    return df


def extract_risk_episodes(panel: pd.DataFrame) -> pd.DataFrame:
    records = []
    is_risk = panel["timing_state"].eq("RISK")
    starts = panel.index[is_risk & ~is_risk.shift(1, fill_value=False)]
    for episode_id, start_idx in enumerate(starts, 1):
        end_idx = start_idx
        while end_idx + 1 < len(panel) and panel.loc[end_idx + 1, "timing_state"] == "RISK":
            end_idx += 1
        sub = panel.loc[start_idx:end_idx].copy()
        spy_nav = (1 + sub["SPY_return"].fillna(0.0)).cumprod()
        records.append(
            {
                "episode_id": episode_id,
                "risk_start_date": sub["date"].iloc[0],
                "risk_end_date": sub["date"].iloc[-1],
                "duration_days": len(sub),
                "entry_reason": sub["entry_reason"].replace("", np.nan).dropna().iloc[0] if sub["entry_reason"].replace("", np.nan).notna().any() else "OTHER",
                "macro_regime_at_entry": sub["macro_regime_confirmed"].iloc[0],
                "dominant_macro_regime": sub["macro_regime_confirmed"].mode().iloc[0],
                "SPY_drawdown_at_entry": sub["spy_drawdown_from_previous_high"].iloc[0],
                "VIX_ZSCORE_at_entry": sub.get("VIX_ZSCORE_120D", pd.Series([np.nan])).iloc[0],
                "CREDIT_SPREAD_at_entry": sub.get("CREDIT_SPREAD_BAA_AAA", pd.Series([np.nan])).iloc[0],
                "D_CREDIT_SPREAD_20D_at_entry": sub.get("D_CREDIT_SPREAD_20D", pd.Series([np.nan])).iloc[0],
                "SPY_return_during_episode": spy_nav.iloc[-1] - 1,
                "SPY_max_drawdown_during_episode": (spy_nav / spy_nav.cummax() - 1).min(),
                "SPY_max_runup_during_episode": spy_nav.max() - 1,
            }
        )
    return pd.DataFrame(records)


def _max_drawdown(ret: pd.Series) -> float:
    ret = pd.to_numeric(ret, errors="coerce").dropna()
    if ret.empty:
        return np.nan
    nav = (1 + ret.fillna(0.0)).cumprod()
    return float((nav / nav.cummax() - 1).min())


def _perf_from_returns(ret: pd.Series, rf: Optional[pd.Series] = None, min_obs: int = 2) -> Dict[str, float]:
    ret = pd.to_numeric(ret, errors="coerce").dropna()
    if len(ret) < min_obs:
        return {
            "n_obs": len(ret),
            "start_date": pd.NaT,
            "end_date": pd.NaT,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "Sharpe": np.nan,
            "max_drawdown": np.nan,
            "average_daily_return": np.nan,
            "positive_day_ratio": np.nan,
            "worst_1d_return": np.nan,
            "best_1d_return": np.nan,
            "cumulative_return_within_group": np.nan,
        }
    nav = (1 + ret).cumprod()
    years = len(ret) / CONFIG["trading_days_per_year"]
    ann = nav.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = ret.std(ddof=0) * math.sqrt(CONFIG["trading_days_per_year"])
    if rf is not None:
        rf = pd.to_numeric(rf.reindex(ret.index), errors="coerce").fillna(0.0)
        excess = ret - rf
    else:
        excess = ret
    sharpe = excess.mean() / excess.std(ddof=0) * math.sqrt(CONFIG["trading_days_per_year"]) if excess.std(ddof=0) > 0 else np.nan
    return {
        "n_obs": len(ret),
        "start_date": ret.index.min(),
        "end_date": ret.index.max(),
        "annualized_return": ann,
        "annualized_volatility": vol,
        "Sharpe": sharpe,
        "max_drawdown": _max_drawdown(ret),
        "average_daily_return": ret.mean(),
        "positive_day_ratio": (ret > 0).mean(),
        "worst_1d_return": ret.min(),
        "best_1d_return": ret.max(),
        "cumulative_return_within_group": nav.iloc[-1] - 1,
    }


def _corr_beta(asset_ret: pd.Series, spy_ret: pd.Series) -> Tuple[float, float]:
    aligned = pd.concat([asset_ret, spy_ret], axis=1).dropna()
    if len(aligned) < 10:
        return np.nan, np.nan
    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    spy_var = aligned.iloc[:, 1].var(ddof=0)
    beta = aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / spy_var if spy_var > 0 else np.nan
    return corr, beta


def _capture_ratios(asset_ret: pd.Series, spy_ret: pd.Series) -> Tuple[float, float]:
    aligned = pd.concat([asset_ret, spy_ret], axis=1).dropna()
    if len(aligned) < 10:
        return np.nan, np.nan
    asset = aligned.iloc[:, 0]
    spy = aligned.iloc[:, 1]
    up_mask = spy > 0
    down_mask = spy < 0
    upside = asset[up_mask].mean() / spy[up_mask].mean() if up_mask.any() and spy[up_mask].mean() != 0 else np.nan
    downside = asset[down_mask].mean() / spy[down_mask].mean() if down_mask.any() and spy[down_mask].mean() != 0 else np.nan
    return downside, upside


def compute_asset_performance_by_state(panel: pd.DataFrame, assets: List[str], group_col: str) -> pd.DataFrame:
    rows = []
    for group_name, sub in panel.groupby(group_col, dropna=False):
        rf = sub["CASH_return"] if "CASH_return" in sub.columns else None
        for asset in assets:
            ret_col = f"{asset}_return"
            if ret_col not in sub.columns:
                continue
            asset_series = pd.to_numeric(sub[ret_col], errors="coerce")
            valid = asset_series.dropna()
            perf = _perf_from_returns(valid, rf=rf.loc[valid.index] if rf is not None else None)
            rows.append(
                {
                    "asset": asset,
                    "group_name": group_name,
                    **perf,
                }
            )
    return pd.DataFrame(rows)


def _subset_summary(panel: pd.DataFrame, group_col: str, name: str, asset: str) -> Dict[str, object]:
    sub = panel[panel[group_col].eq(name)].copy()
    ret_col = f"{asset}_return"
    if sub.empty or ret_col not in sub.columns:
        return {}
    ret = pd.to_numeric(sub[ret_col], errors="coerce")
    rf = sub["CASH_return"] if "CASH_return" in sub.columns else None
    perf = _perf_from_returns(ret, rf=rf)
    corr, beta = _corr_beta(ret, sub["SPY_return"])
    downside, upside = _capture_ratios(ret, sub["SPY_return"])
    return {
        "group_name": name,
        "asset": asset,
        "correlation_with_SPY": corr,
        "beta_to_SPY": beta,
        "downside_capture_vs_SPY": downside,
        "upside_capture_vs_SPY": upside,
        **perf,
    }


def compute_risk_episode_asset_performance(panel: pd.DataFrame, episodes: pd.DataFrame, assets: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = []
    for _, ep in episodes.iterrows():
        sub = panel[(panel["date"] >= ep["risk_start_date"]) & (panel["date"] <= ep["risk_end_date"])].copy()
        rf = sub["CASH_return"] if "CASH_return" in sub.columns else None
        for asset in assets:
            ret_col = f"{asset}_return"
            if ret_col not in sub.columns:
                continue
            perf = _perf_from_returns(sub[ret_col], rf=rf)
            records.append({**ep.to_dict(), "asset": asset, **perf})
    detail = pd.DataFrame(records)
    if detail.empty:
        return detail, pd.DataFrame(), pd.DataFrame()
    detail["rank_by_return"] = detail.groupby("episode_id")["cumulative_return_within_group"].rank(ascending=False, method="min")
    detail["rank_by_maxdd"] = detail.groupby("episode_id")["max_drawdown"].rank(ascending=False, method="min")
    detail["rank_by_sharpe"] = detail.groupby("episode_id")["Sharpe"].rank(ascending=False, method="min")
    summary = (
        detail.groupby(["entry_reason", "asset"], dropna=False)
        .agg(
            episode_count=("episode_id", "nunique"),
            cumulative_return=("cumulative_return_within_group", "mean"),
            max_drawdown=("max_drawdown", "mean"),
            Sharpe=("Sharpe", "mean"),
            annualized_return=("annualized_return", "mean"),
            annualized_volatility=("annualized_volatility", "mean"),
            avg_rank_by_return=("rank_by_return", "mean"),
            avg_rank_by_maxdd=("rank_by_maxdd", "mean"),
            avg_rank_by_sharpe=("rank_by_sharpe", "mean"),
        )
        .reset_index()
    )
    steep_summary = (
        detail[detail["dominant_macro_regime"].eq("STEEP")]
        .groupby("asset", dropna=False)
        .agg(
            episode_count=("episode_id", "nunique"),
            cumulative_return=("cumulative_return_within_group", "mean"),
            max_drawdown=("max_drawdown", "mean"),
            Sharpe=("Sharpe", "mean"),
            annualized_return=("annualized_return", "mean"),
            annualized_volatility=("annualized_volatility", "mean"),
        )
        .reset_index()
    )
    return detail, summary, steep_summary


def compute_case_studies(panel: pd.DataFrame, assets: List[str]) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    daily_panels: Dict[str, pd.DataFrame] = {}
    rows = []
    max_date = panel["date"].max()
    for case_name, (start, end) in CONFIG["case_windows"].items():
        start_dt = pd.Timestamp(start)
        end_dt = max_date if end == "latest" else min(pd.Timestamp(end), max_date)
        sub = panel[(panel["date"] >= start_dt) & (panel["date"] <= end_dt)].copy()
        if sub.empty:
            continue
        for asset in assets:
            ret_col = f"{asset}_return"
            if ret_col in sub.columns:
                sub[f"{asset}_nav"] = (1 + sub[ret_col].fillna(0.0)).cumprod()
        daily_panels[case_name] = sub
        for asset in assets:
            ret_col = f"{asset}_return"
            if ret_col not in sub.columns:
                continue
            perf = _perf_from_returns(sub[ret_col], rf=sub["CASH_return"])
            corr, beta = _corr_beta(sub[ret_col], sub["SPY_return"])
            rows.append(
                {
                    "case_name": case_name,
                    "asset": asset,
                    "correlation_with_SPY": corr,
                    "beta_to_SPY": beta,
                    "volatility": perf["annualized_volatility"],
                    "Sharpe": perf["Sharpe"],
                    "cumulative_return": perf["cumulative_return_within_group"],
                    "max_drawdown": perf["max_drawdown"],
                }
            )
    return daily_panels, pd.DataFrame(rows)


def compute_correlations_and_betas(panel: pd.DataFrame, assets: List[str], daily_panels: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cross_corr_rows = []
    cross_beta_rows = []
    case_corr_rows = []
    case_beta_rows = []

    for cross_state, sub in panel.groupby("cross_state"):
        asset_cols = [f"{asset}_return" for asset in assets if f"{asset}_return" in sub.columns]
        if len(asset_cols) < 2:
            continue
        corr = sub[asset_cols].corr()
        for a1 in corr.index:
            for a2 in corr.columns:
                cross_corr_rows.append(
                    {
                        "cross_state": cross_state,
                        "asset_1": a1.replace("_return", ""),
                        "asset_2": a2.replace("_return", ""),
                        "correlation": corr.loc[a1, a2],
                    }
                )
        for asset in assets:
            ret_col = f"{asset}_return"
            if asset == "SPY" or ret_col not in sub.columns:
                continue
            corr_val, beta_val = _corr_beta(sub[ret_col], sub["SPY_return"])
            cross_beta_rows.append(
                {
                    "cross_state": cross_state,
                    "asset": asset,
                    "correlation_with_SPY": corr_val,
                    "beta_to_SPY": beta_val,
                }
            )

    for case_name, sub in daily_panels.items():
        asset_cols = [f"{asset}_return" for asset in assets if f"{asset}_return" in sub.columns]
        if len(asset_cols) < 2:
            continue
        corr = sub[asset_cols].corr()
        for a1 in corr.index:
            for a2 in corr.columns:
                case_corr_rows.append(
                    {
                        "case_name": case_name,
                        "asset_1": a1.replace("_return", ""),
                        "asset_2": a2.replace("_return", ""),
                        "correlation": corr.loc[a1, a2],
                    }
                )
        for asset in assets:
            ret_col = f"{asset}_return"
            if asset == "SPY" or ret_col not in sub.columns:
                continue
            corr_val, beta_val = _corr_beta(sub[ret_col], sub["SPY_return"])
            case_beta_rows.append(
                {
                    "case_name": case_name,
                    "asset": asset,
                    "correlation_with_SPY": corr_val,
                    "beta_to_SPY": beta_val,
                }
            )

    return (
        pd.DataFrame(cross_corr_rows),
        pd.DataFrame(cross_beta_rows),
        pd.DataFrame(case_corr_rows),
        pd.DataFrame(case_beta_rows),
    )


def _normalize(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if not np.isfinite(x.max() - x.min()) or (x.max() - x.min()) == 0:
        return pd.Series(0.5, index=series.index)
    return (x - x.min()) / (x.max() - x.min())


def rank_steep_nonrisk_equity_satellites(steep_df: pd.DataFrame) -> pd.DataFrame:
    sub = steep_df[(steep_df["group_name"] == "STEEP_NON_RISK") & (steep_df["asset"].isin(["SPY", "IJH", "IWM"]))].copy()
    if sub.empty:
        return sub
    beta_excess = (sub["beta_to_SPY"] - 1.0).abs()
    sub["score"] = (
        0.35 * _normalize(sub["Sharpe"])
        + 0.25 * _normalize(sub["annualized_return"])
        + 0.20 * _normalize(-sub["max_drawdown"].abs())
        - 0.20 * _normalize(beta_excess)
    )
    sub["rank"] = sub["score"].rank(ascending=False, method="min")
    return sub.sort_values("rank")


def rank_steep_risk_duration_hedges(steep_df: pd.DataFrame) -> pd.DataFrame:
    sub = steep_df[(steep_df["group_name"] == "STEEP_RISK") & (steep_df["asset"].isin(["IEF", "TLT", "EDV", "CASH", "GOLD"]))].copy()
    if sub.empty:
        return sub
    cumulative_col = "cumulative_return"
    if cumulative_col not in sub.columns:
        if "cumulative_return_within_group" in sub.columns:
            cumulative_col = "cumulative_return_within_group"
        else:
            sub[cumulative_col] = np.nan
    sub["score"] = (
        0.30 * _normalize(sub["Sharpe"])
        + 0.25 * _normalize(sub[cumulative_col])
        + 0.25 * _normalize(-sub["max_drawdown"].abs())
        + 0.20 * _normalize(-sub["correlation_with_SPY"])
    )
    sub["rank"] = sub["score"].rank(ascending=False, method="min")
    return sub.sort_values("rank")


def rank_overall_by_cross_state(cross_perf: pd.DataFrame, beta_df: pd.DataFrame) -> pd.DataFrame:
    if cross_perf.empty:
        return cross_perf
    merged = cross_perf.merge(beta_df, on=["cross_state", "asset"], how="left")
    rows = []
    for cross_state, sub in merged.groupby("cross_state"):
        tmp = sub.copy()
        tmp["score"] = (
            0.35 * _normalize(tmp["Sharpe"])
            + 0.25 * _normalize(tmp["annualized_return"])
            + 0.20 * _normalize(-tmp["max_drawdown"].abs())
            + 0.20 * _normalize(-tmp["correlation_with_SPY"].fillna(0.0))
        )
        tmp["rank"] = tmp["score"].rank(ascending=False, method="min")
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _save_heatmap(pivot: pd.DataFrame, title: str, path: Path, fmt: str = ".2f") -> None:
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 0.8), max(5, len(pivot.index) * 0.4)))
    data = pivot.astype(float)
    im = ax.imshow(data.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data.iloc[i, j]
            if np.isfinite(value):
                ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_heatmaps(cross_perf: pd.DataFrame, cross_corr: pd.DataFrame) -> None:
    fig_dir = CONFIG["figure_dir"]
    for metric, filename, fmt in [
        ("Sharpe", "asset_sharpe_by_cross_state_heatmap_extended.png", ".2f"),
        ("annualized_return", "asset_return_by_cross_state_heatmap_extended.png", ".1%"),
        ("max_drawdown", "asset_maxdd_by_cross_state_heatmap_extended.png", ".1%"),
    ]:
        pivot = cross_perf.pivot_table(index="asset", columns="cross_state", values=metric)
        _save_heatmap(pivot, metric, fig_dir / filename, fmt=fmt)
    for state, filename in [("STEEP_RISK", "correlation_heatmap_STEEP_RISK.png"), ("STEEP_NON_RISK", "correlation_heatmap_STEEP_NON_RISK.png")]:
        sub = cross_corr[cross_corr["cross_state"].eq(state)]
        if sub.empty:
            continue
        pivot = sub.pivot(index="asset_1", columns="asset_2", values="correlation")
        _save_heatmap(pivot, f"Correlation {state}", fig_dir / filename, fmt=".2f")


def plot_steep_diagnostics(steep_df: pd.DataFrame, steep_risk_detail: pd.DataFrame) -> None:
    fig_dir = CONFIG["figure_dir"]
    equity = steep_df[(steep_df["group_name"] == "STEEP_NON_RISK") & (steep_df["asset"].isin(["SPY", "IJH", "IWM"]))].copy()
    if not equity.empty:
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        metrics = ["annualized_return", "Sharpe", "max_drawdown", "beta_to_SPY"]
        for ax, metric in zip(axes, metrics):
            ax.bar(equity["asset"], equity[metric])
            ax.set_title(metric)
        fig.tight_layout()
        fig.savefig(fig_dir / "steep_nonrisk_equity_satellite_bar.png", dpi=150)
        plt.close(fig)

    duration = steep_df[(steep_df["group_name"] == "STEEP_RISK") & (steep_df["asset"].isin(["IEF", "TLT", "EDV", "GOLD", "CASH"]))].copy()
    if not duration.empty:
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        cumulative_col = "cumulative_return"
        if cumulative_col not in duration.columns:
            if "cumulative_return_within_group" in duration.columns:
                cumulative_col = "cumulative_return_within_group"
            else:
                duration[cumulative_col] = np.nan
        metric_map = {
            "cumulative_return": cumulative_col,
            "Sharpe": "Sharpe",
            "max_drawdown": "max_drawdown",
            "correlation_with_SPY": "correlation_with_SPY",
        }
        for ax, (title, metric) in zip(axes, metric_map.items()):
            ax.bar(duration["asset"], duration[metric])
            ax.set_title(title)
        fig.tight_layout()
        fig.savefig(fig_dir / "steep_risk_duration_hedge_bar.png", dpi=150)
        plt.close(fig)

    if not steep_risk_detail.empty:
        selected = steep_risk_detail[steep_risk_detail["asset"].isin(["IEF", "TLT", "EDV", "GOLD", "CASH"])].copy()
        if not selected.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            groups = list(selected.groupby("asset"))
            axes[0].boxplot([g["cumulative_return_within_group"].dropna().values for _, g in groups], tick_labels=[k for k, _ in groups], showfliers=False)
            axes[0].axhline(0, color="black", linewidth=0.8)
            axes[0].set_title("STEEP_RISK episode return")
            axes[1].boxplot([g["max_drawdown"].dropna().values for _, g in groups], tick_labels=[k for k, _ in groups], showfliers=False)
            axes[1].set_title("STEEP_RISK episode max drawdown")
            fig.tight_layout()
            fig.savefig(fig_dir / "steep_risk_duration_episode_boxplot.png", dpi=150)
            plt.close(fig)


def _plot_state_strip(ax, dates: pd.Series, values: pd.Series, title: str) -> None:
    codes = pd.Categorical(values).codes
    ax.imshow([codes], aspect="auto", extent=[dates.iloc[0], dates.iloc[-1], 0, 1], cmap="tab20")
    ax.set_yticks([])
    ax.set_title(title, loc="left", fontsize=9)


def plot_case_studies(daily_panels: Dict[str, pd.DataFrame]) -> None:
    fig_dir = CONFIG["figure_dir"]
    mapping = {
        "2008_GFC": "case_study_2008_duration_hedges.png",
        "2022": "case_study_2022_duration_hedges.png",
        "2025_PULLBACK": "case_study_2025_equity_satellite.png",
    }
    for case_name, filename in mapping.items():
        if case_name not in daily_panels:
            continue
        sub = daily_panels[case_name].copy()
        fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [2, 2, 0.5, 1.25]})
        axes[0].plot(sub["date"], sub["SPY_nav"] / sub["SPY_nav"].iloc[0], label="SPY")
        axes[0].plot(sub["date"], 1 + sub["spy_drawdown_from_previous_high"], label="SPY dd index", alpha=0.7)
        axes[0].legend(loc="best", fontsize=8)
        if case_name in {"2008_GFC", "2022"}:
            for asset in ["IEF", "TLT", "EDV", "GOLD", "CASH"]:
                col = f"{asset}_nav"
                if col in sub.columns:
                    axes[1].plot(sub["date"], sub[col] / sub[col].iloc[0], label=asset)
        else:
            for asset in ["SPY", "IJH", "IWM", "GOLD"]:
                col = f"{asset}_nav"
                if col in sub.columns:
                    axes[1].plot(sub["date"], sub[col] / sub[col].iloc[0], label=asset)
        axes[1].legend(loc="best", ncol=4, fontsize=8)
        _plot_state_strip(axes[2], sub["date"], sub["cross_state"], "cross state")
        axes[3].plot(sub["date"], sub["VIX_ZSCORE_120D"], label="VIX z120")
        axes[3].plot(sub["date"], sub["D_CREDIT_SPREAD_20D"], label="credit chg20")
        axes[3].axhline(0, color="black", linewidth=0.8)
        axes[3].legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=150)
        plt.close(fig)

    if "2024_2026" in daily_panels:
        sub = daily_panels["2024_2026"].copy()
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        for asset in ["SPY", "IJH", "IWM"]:
            col = f"{asset}_nav"
            if col in sub.columns:
                axes[0].plot(sub["date"], sub[col] / sub[col].iloc[0], label=asset)
        axes[0].legend(loc="best", fontsize=8)
        axes[0].set_title("2024-2026 equity satellites")
        _plot_state_strip(axes[1], sub["date"], sub["cross_state"], "cross state")
        fig.tight_layout()
        fig.savefig(fig_dir / "case_study_steep_nonrisk_equity_2009_2013_or_best_available.png", dpi=150)
        plt.close(fig)


def write_markdown_report(
    assets: List[str],
    asset_sources: Dict[str, str],
    asset_start_dates: Dict[str, pd.Timestamp],
    cross_perf: pd.DataFrame,
    steep_df: pd.DataFrame,
    steep_nonrisk_rank: pd.DataFrame,
    steep_risk_rank: pd.DataFrame,
    case_summary: pd.DataFrame,
) -> None:
    out = CONFIG["output_dir"] / "HEDGE_ASSET_CROSS_STATE_DIAGNOSTIC_EXTENDED.md"

    def top_table(df: pd.DataFrame, cols: List[str], n: int = 10) -> str:
        if df.empty:
            return "_No data._"
        keep = [c for c in cols if c in df.columns]
        if not keep:
            return "_No matching columns._"
        return df[keep].head(n).to_markdown(index=False)

    steep_risk_return_col = "cumulative_return" if "cumulative_return" in steep_risk_rank.columns else "cumulative_return_within_group"

    content = f"""# Hedge Asset Cross-State Diagnostic Extended

## Purpose

This diagnostic extends the original cross-state asset study with IJH, IWM, TLT, and EDV. It does not change the timing backbone and does not run an allocation backtest.

## Added Assets

- IJH: mid-cap equity satellite
- IWM: small-cap equity satellite
- TLT: long-duration Treasury
- EDV: extended-duration Treasury

## Method

- Timing state uses BACKBONE_V2_UPGRADED.
- cross_state = macro_regime_confirmed x timing_state.
- Each asset is evaluated on its own available sample, with n_obs shown in all summary tables.

## Asset Availability

{pd.DataFrame([{"asset": a, "source": asset_sources.get(a, "missing"), "start_date": asset_start_dates.get(a)} for a in assets]).to_markdown(index=False)}

## Cross-State Overview

Top rows from cross-state performance:

{top_table(cross_perf.sort_values(["cross_state", "Sharpe"], ascending=[True, False]), ["cross_state", "asset", "n_obs", "annualized_return", "Sharpe", "max_drawdown"], 24)}

## STEEP_NON_RISK Equity Satellite Findings

{top_table(steep_nonrisk_rank.sort_values("rank"), ["asset", "annualized_return", "Sharpe", "max_drawdown", "beta_to_SPY", "score", "rank"], 10)}

## STEEP_RISK Duration Hedge Findings

{top_table(steep_risk_rank.sort_values("rank"), ["asset", "annualized_return", "Sharpe", "max_drawdown", steep_risk_return_col, "correlation_with_SPY", "score", "rank"], 10)}

## Case Studies

{top_table(case_summary.sort_values(["case_name", "Sharpe"], ascending=[True, False]), ["case_name", "asset", "cumulative_return", "max_drawdown", "Sharpe", "beta_to_SPY"], 30)}

## Interpretation

- STEEP_NON_RISK should only include IJH or IWM in the next round if they improve Sharpe without simply adding beta.
- STEEP_RISK should only replace IEF if TLT or EDV provide better convexity without creating a 2022-style inflation-shock failure.
- Long-duration candidates need to be judged on both 2008 and 2022, not on one crisis alone.

## Recommended Next Step

- Duration hedge test candidates: 100% TLT, 100% EDV, 80% IEF / 20% TLT, 80% IEF / 20% EDV
- Equity satellite test candidates: 90% SPY / 10% IJH, 90% SPY / 10% IWM, 80% SPY / 20% IJH, 80% SPY / 20% IWM

## Caveats

- EDV starts later than the core ETF sample.
- Small-cap and mid-cap satellites have higher beta than SPY.
- Long duration performs poorly in inflation-rate shocks.
- This is still diagnostic evidence, not a strategy backtest.
"""
    out.write_text(content, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    base = load_base_panel()
    event_log = load_event_log()
    panel, asset_sources, notes = load_existing_asset_returns(base)
    panel = build_backbone_v2_state(panel)
    panel = build_cross_state_labels(panel, event_log)

    assets = [asset for asset in CONFIG["assets"] if f"{asset}_return" in panel.columns]
    asset_start_dates = {
        asset: panel.loc[pd.to_numeric(panel[f"{asset}_return"], errors="coerce").notna(), "date"].min()
        for asset in assets
    }

    macro_perf = compute_asset_performance_by_state(panel, assets, "macro_regime_confirmed")
    timing_perf = compute_asset_performance_by_state(panel, assets, "timing_state")
    cross_perf = compute_asset_performance_by_state(panel, assets, "cross_state")
    entry_perf = compute_asset_performance_by_state(panel[panel["entry_reason"].ne("")], assets, "entry_reason")

    macro_perf.to_csv(CONFIG["output_dir"] / "asset_performance_by_macro_regime.csv", index=False)
    timing_perf.to_csv(CONFIG["output_dir"] / "asset_performance_by_timing_state.csv", index=False)
    cross_perf.to_csv(CONFIG["output_dir"] / "asset_performance_by_cross_state.csv", index=False)
    entry_perf.to_csv(CONFIG["output_dir"] / "asset_performance_by_entry_reason.csv", index=False)

    steep_rows = []
    for group_name in ["STEEP_NON_RISK", "STEEP_RISK"]:
        for asset in ["SPY", "IJH", "IWM", "IEF", "TLT", "EDV", "GOLD", "CASH", "CMDTY_FUT"]:
            row = _subset_summary(panel, "cross_state", group_name, asset)
            if row:
                steep_rows.append(row)
    steep_df = pd.DataFrame(steep_rows)
    steep_df.to_csv(CONFIG["output_dir"] / "steep_asset_diagnostic.csv", index=False)

    episodes = extract_risk_episodes(panel)
    episodes.to_csv(CONFIG["output_dir"] / "risk_episodes.csv", index=False)
    risk_detail, risk_summary, steep_risk_summary = compute_risk_episode_asset_performance(panel, episodes, assets)
    risk_detail.to_csv(CONFIG["output_dir"] / "risk_episode_asset_performance.csv", index=False)
    risk_summary.to_csv(CONFIG["output_dir"] / "risk_episode_asset_summary_by_entry_reason.csv", index=False)
    steep_risk_summary.to_csv(CONFIG["output_dir"] / "steep_risk_episode_asset_summary.csv", index=False)

    daily_panels, case_asset_summary = compute_case_studies(panel, assets)
    for case_name, case_df in daily_panels.items():
        case_df.to_csv(CONFIG["output_dir"] / f"case_study_{case_name}.csv", index=False)
    case_asset_summary.to_csv(CONFIG["output_dir"] / "case_study_asset_summary.csv", index=False)

    cross_corr, cross_beta, case_corr, case_beta = compute_correlations_and_betas(panel, assets, daily_panels)
    cross_corr.to_csv(CONFIG["output_dir"] / "asset_correlation_by_cross_state.csv", index=False)
    cross_beta.to_csv(CONFIG["output_dir"] / "asset_beta_to_spy_by_cross_state.csv", index=False)
    case_corr.to_csv(CONFIG["output_dir"] / "asset_correlation_by_case.csv", index=False)
    case_beta.to_csv(CONFIG["output_dir"] / "asset_beta_to_spy_by_case.csv", index=False)

    steep_nonrisk_rank = rank_steep_nonrisk_equity_satellites(steep_df)
    steep_risk_rank = rank_steep_risk_duration_hedges(steep_df)
    cross_rank = rank_overall_by_cross_state(cross_perf.rename(columns={"group_name": "cross_state"}), cross_beta)
    steep_nonrisk_rank.to_csv(CONFIG["output_dir"] / "steep_nonrisk_equity_satellite_ranking.csv", index=False)
    steep_risk_rank.to_csv(CONFIG["output_dir"] / "steep_risk_duration_hedge_ranking.csv", index=False)
    cross_rank.to_csv(CONFIG["output_dir"] / "hedge_asset_ranking_by_cross_state.csv", index=False)

    panel.to_csv(CONFIG["output_dir"] / "hedge_asset_cross_state_daily_panel.csv", index=False)
    pd.DataFrame(
        [{"asset": asset, "source": asset_sources.get(asset, ""), "effective_start_date": asset_start_dates.get(asset)} for asset in assets]
    ).to_csv(CONFIG["output_dir"] / "asset_availability_summary.csv", index=False)
    pd.DataFrame({"note": notes}).to_csv(CONFIG["output_dir"] / "data_notes.csv", index=False)

    plot_heatmaps(cross_perf.rename(columns={"group_name": "cross_state"}), cross_corr)
    plot_steep_diagnostics(steep_df, risk_detail[risk_detail["dominant_macro_regime"].eq("STEEP")].copy())
    plot_case_studies(daily_panels)
    write_markdown_report(assets, asset_sources, asset_start_dates, cross_perf.rename(columns={"group_name": "cross_state"}), steep_df, steep_nonrisk_rank, steep_risk_rank, case_asset_summary)


if __name__ == "__main__":
    main()
