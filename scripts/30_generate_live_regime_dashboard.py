"""Generate a static live regime dashboard without touching final strategy outputs."""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from io import BytesIO
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

OUTPUT_DIR = ROOT / "results" / "live_regime_dashboard"
FIGURES_DIR = OUTPUT_DIR / "figures"
CACHE_DIR = ROOT / "data" / "live_cache"
FRED_CACHE = CACHE_DIR / "fred_macro_daily.csv"
ASSET_CACHE = CACHE_DIR / "asset_prices_daily.csv"
NETWORK_TIMEOUT_SECONDS = 15

FRED_SERIES = ["DTB3", "DBAA", "DAAA", "DGS10", "DGS1", "VIXCLS"]
FRED_FALLBACK_SERIES = {"DTB3": "TB3MS"}
YFINANCE_TICKERS = {
    "SPY": "SPY",
    "GOLD": "GLD",
    "IEF": "IEF",
    "DBC": "DBC",
    "DBB": "DBB",
    "DBA": "DBA",
    "OIL_SIGNAL": "CL=F",
}
YFINANCE_PROXY = os.environ.get("YFINANCE_PROXY", "").strip()

FINAL_STRATEGY = "FINAL_REGIME_HEDGE_TRIGGER_LOCK"
SPY_BUY_HOLD = "SPY_BUY_HOLD"

REGIME_INTERPRETATION = {
    "FLAT_LOW_RATE": {
        "economic_meaning": "A flat curve with low long rates usually points to subdued nominal growth expectations and limited rate pressure.",
        "typical_risk": "The main risk is that defensive rate-sensitive exposure can lag if growth risk fades quickly.",
        "preferred_assets": "The final strategy uses SPY, DBC, and DBB inverse-vol in the normal state; if oil is HIGH, DBB is removed.",
        "stress_behavior": "Stress locks move the sleeve to cash for FLAT_LOW/MID stress states.",
        "monitoring_focus": "Watch VIX, credit widening, oil level, and whether the curve shifts out of the flat regime.",
    },
    "FLAT_MID_RATE": {
        "economic_meaning": "A flat curve with mid-level rates is a balanced but late-cycle-looking state.",
        "typical_risk": "Risk assets can still work, but the flat curve leaves less cushion if credit or volatility deteriorates.",
        "preferred_assets": "The final strategy uses SPY and gold inverse-vol in the normal state; if oil is HIGH, the sleeve collapses to SPY.",
        "stress_behavior": "Stress locks move the sleeve to cash for FLAT_LOW/MID stress states.",
        "monitoring_focus": "Watch VIX, credit widening, SPY trend confirmation, and whether oil HIGH removes gold from the flat sleeve.",
    },
    "FLAT_HIGH_RATE": {
        "economic_meaning": "A flat curve with high long rates reflects elevated rate pressure while the curve is not steep enough to signal a broad pro-growth steep regime.",
        "typical_risk": "Equity beta can be vulnerable if high rates tighten financial conditions or credit spreads widen.",
        "preferred_assets": "The final strategy uses 40% IEF plus 60% gold/DBC inverse-vol in the normal state; if oil is HIGH, gold is removed from the floating sleeve.",
        "stress_behavior": "Stress shifts to 10% DBA plus 90% gold; if oil is HIGH, the sleeve collapses to 100% DBA.",
        "monitoring_focus": "Watch the term spread boundary near STEEP, credit spread changes, VIX shocks, and oil level changes.",
    },
    "STEEP_LOW_RATE": {
        "economic_meaning": "A steep curve with low long rates often reflects recovery expectations while policy remains easy.",
        "typical_risk": "The main risk is a failed recovery or a commodity reversal.",
        "preferred_assets": "The final strategy holds 100% SPY in the normal state.",
        "stress_behavior": "If an existing stress lock carries into this state, the strategy holds SPY until unlock.",
        "monitoring_focus": "Watch regime persistence, SPY trend, and whether stress carries over from the previous state.",
    },
    "STEEP_MID_RATE": {
        "economic_meaning": "A steep curve with mid-level long rates is typically the cleanest pro-growth state.",
        "typical_risk": "The main risk is a fast volatility or credit shock that breaks the growth trend.",
        "preferred_assets": "The final strategy holds SPY in the normal state.",
        "stress_behavior": "Stress shifts to IEF.",
        "monitoring_focus": "Watch credit spread changes, SPY trend, and whether rates move into the high bucket.",
    },
    "STEEP_HIGH_RATE": {
        "economic_meaning": "A steep curve with high long rates can combine growth participation with inflation or rate pressure.",
        "typical_risk": "Equities can still participate, but rate-sensitive valuation pressure and commodity shocks matter.",
        "preferred_assets": "The final strategy uses SPY and gold inverse-vol in the normal state.",
        "stress_behavior": "Stress shifts to IEF.",
        "monitoring_focus": "Watch credit widening, VIX, and whether the term spread compresses back into FLAT.",
    },
    "INVERTED": {
        "economic_meaning": "An inverted curve usually reflects restrictive policy and elevated recession risk.",
        "typical_risk": "Equity drawdowns and credit stress can appear with a lag after inversion.",
        "preferred_assets": "The final strategy uses SPY and gold in the normal state.",
        "stress_behavior": "Stress keeps 90% cash and 10% SPY.",
        "monitoring_focus": "Watch credit level z-score, VIX unlock conditions, and SPY trend.",
    },
}


try:
    import final_strategy_source_only_core as final_core

    FINAL_IMPORT_WARNING = ""
except Exception as exc:  # pragma: no cover - exercised only when local module breaks
    final_core = None
    FINAL_IMPORT_WARNING = (
        "live dashboard uses replicated final logic; please verify consistency with main pipeline. "
        f"Import error: {exc}"
    )

ASSETS = list(getattr(final_core, "ASSETS", ["SPY", "GOLD", "IEF", "DBC", "DBB", "DBA", "CASH"]))
RISK_ASSETS = [asset for asset in ASSETS if asset != "CASH"]
LIVE_PRICE_COLUMNS = RISK_ASSETS + ["OIL_SIGNAL"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the live regime dashboard.")
    parser.add_argument("--lookback-years", type=int, default=2)
    parser.add_argument("--max-lookback-years", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-fetch-cache", nargs="?", const=True, default=False, type=str_to_bool)
    parser.add_argument("--refresh", action="store_true", default=False)
    return parser.parse_args()


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def ensure_dirs(output_dir: Path) -> tuple[Path, Path]:
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return output_dir, figures_dir


def cache_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path, parse_dates=["date"])
    except Exception:
        return False
    if df.empty or "date" not in df.columns:
        return False
    latest = pd.to_datetime(df["date"]).max().date()
    return (date.today() - latest).days < 1


def fred_graph_url(series_id: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def fred_api_url(series_id: str, api_key: str) -> str:
    params = urlencode(
        {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": "1980-01-01",
        }
    )
    return f"https://api.stlouisfed.org/fred/series/observations?{params}"


def fetch_one_fred_series(series_id: str, warnings: list[str]) -> tuple[pd.DataFrame, str]:
    api_key = os.environ.get("FRED_API_KEY", "").strip()
    source = series_id
    try:
        if api_key:
            with urlopen(fred_api_url(series_id, api_key), timeout=NETWORK_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
            obs = pd.DataFrame(payload["observations"])
            df = obs[["date", "value"]].rename(columns={"value": series_id})
        else:
            with urlopen(fred_graph_url(series_id), timeout=NETWORK_TIMEOUT_SECONDS) as response:
                df = pd.read_csv(BytesIO(response.read())).rename(columns={"observation_date": "date"})
            if series_id not in df.columns:
                value_cols = [c for c in df.columns if c != "date"]
                df = df.rename(columns={value_cols[0]: series_id})
        df["date"] = pd.to_datetime(df["date"])
        df[series_id] = pd.to_numeric(df[series_id].replace(".", np.nan), errors="coerce")
        return df[["date", series_id]].sort_values("date"), source
    except Exception as exc:
        fallback = FRED_FALLBACK_SERIES.get(series_id)
        if not fallback:
            raise
        warnings.append(f"{series_id} unavailable from FRED ({exc}); using {fallback} fallback with different frequency.")
        with urlopen(fred_graph_url(fallback), timeout=NETWORK_TIMEOUT_SECONDS) as response:
            df = pd.read_csv(BytesIO(response.read())).rename(columns={"observation_date": "date", fallback: series_id})
        df["date"] = pd.to_datetime(df["date"])
        df[series_id] = pd.to_numeric(df[series_id].replace(".", np.nan), errors="coerce")
        return df[["date", series_id]].sort_values("date"), fallback


def fetch_fred_macro(refresh: bool, use_cache: bool, warnings: list[str]) -> pd.DataFrame:
    if use_cache and not refresh and cache_is_fresh(FRED_CACHE):
        return pd.read_csv(FRED_CACHE, parse_dates=["date"])

    frames = []
    sources: dict[str, str] = {}
    for series_id in FRED_SERIES:
        try:
            frame, source = fetch_one_fred_series(series_id, warnings)
        except Exception as exc:
            if use_cache and FRED_CACHE.exists():
                warnings.append(f"FRED refresh failed for {series_id}; using existing macro cache. Error: {exc}")
                return pd.read_csv(FRED_CACHE, parse_dates=["date"])
            raise
        frames.append(frame)
        sources[series_id] = source

    macro = frames[0]
    for frame in frames[1:]:
        macro = macro.merge(frame, on="date", how="outer")
    macro = macro.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    macro.to_csv(FRED_CACHE, index=False)
    if sources.get("DTB3") == "TB3MS":
        warnings.append("3-month T-bill uses TB3MS fallback; it is monthly and forward-filled for daily cash accrual.")
    return macro


def import_yfinance() -> Any:
    try:
        import yfinance as yf

        return yf
    except Exception:
        sys.path.insert(0, str(ROOT / "src" / "data"))
        from download_yahoo_assets import import_yfinance as local_import_yfinance

        return local_import_yfinance()


def set_yfinance_proxy() -> None:
    if not YFINANCE_PROXY:
        return
    os.environ["HTTP_PROXY"] = YFINANCE_PROXY
    os.environ["HTTPS_PROXY"] = YFINANCE_PROXY
    os.environ["http_proxy"] = YFINANCE_PROXY
    os.environ["https_proxy"] = YFINANCE_PROXY


def configure_yfinance_proxy(yf: Any) -> None:
    if not YFINANCE_PROXY:
        return
    proxy_config = {"http": YFINANCE_PROXY, "https": YFINANCE_PROXY}
    if hasattr(yf, "config") and hasattr(yf.config, "network"):
        yf.config.network.proxy = proxy_config
        return
    if hasattr(yf, "set_config"):
        try:
            yf.set_config(proxy=proxy_config)
        except TypeError:
            yf.set_config({"proxy": proxy_config})


def extract_yfinance_close(downloaded: pd.DataFrame, ticker: str) -> pd.Series:
    if downloaded.empty:
        raise ValueError("empty yfinance response")
    if isinstance(downloaded.columns, pd.MultiIndex):
        if ticker in downloaded.columns.get_level_values(0):
            sub = downloaded[ticker]
        elif ticker in downloaded.columns.get_level_values(-1):
            sub = downloaded.xs(ticker, axis=1, level=-1)
        else:
            raise KeyError(f"{ticker} not in yfinance columns")
    else:
        sub = downloaded
    close_col = "Adj Close" if "Adj Close" in sub.columns else "Close"
    if close_col not in sub.columns:
        raise KeyError(f"{ticker} has no Close/Adj Close column")
    s = pd.to_numeric(sub[close_col], errors="coerce")
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = ticker
    return s


def fetch_asset_prices(refresh: bool, use_cache: bool, warnings: list[str]) -> pd.DataFrame:
    cached = read_asset_cache() if use_cache else None
    if cached is not None and not refresh and cache_is_fresh(ASSET_CACHE):
        cached = merge_missing_local_assets(cached, warnings)
        cached = enforce_common_asset_latest(cached, warnings)
        return cached

    download_start = "1990-01-01"
    if cached is not None and not cached.empty:
        latest_cached = pd.to_datetime(cached["date"]).max()
        # Re-download a small overlap so adjusted data revisions or late rows can be corrected.
        download_start = (latest_cached - pd.Timedelta(days=7)).strftime("%Y-%m-%d")

    set_yfinance_proxy()
    try:
        yf = import_yfinance()
    except Exception as exc:
        if cached is not None and not cached.empty:
            warnings.append(f"Unable to import yfinance; using existing asset price cache. Error: {exc}")
            return cached
        raise
    configure_yfinance_proxy(yf)
    price_series: list[pd.Series] = []
    for asset, ticker in YFINANCE_TICKERS.items():
        try:
            downloaded = yf.download(
                ticker,
                start=download_start,
                end=(pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=False,
            )
            s = extract_yfinance_close(downloaded, ticker)
            if s.dropna().empty:
                raise ValueError("no non-null prices")
            s.name = asset
            price_series.append(s)
        except Exception as exc:
            warnings.append(f"Asset price download failed for {asset} ({ticker}): {exc}")

    if not price_series:
        if cached is not None and not cached.empty:
            warnings.append("Using existing asset price cache because incremental yfinance download did not return usable data.")
            cached = merge_missing_local_assets(cached, warnings)
            cached = enforce_common_asset_latest(cached, warnings)
            return cached
        local = load_local_asset_price_fallback(warnings)
        local.to_csv(ASSET_CACHE, index=False)
        return local

    prices = pd.concat(price_series, axis=1).sort_index()
    prices.index.name = "date"
    out = prices.reset_index()
    if cached is not None and not cached.empty:
        out = merge_price_cache(cached, out)
    out = merge_missing_local_assets(out, warnings)
    out = enforce_common_asset_latest(out, warnings)
    out.to_csv(ASSET_CACHE, index=False)
    return out


def read_asset_cache() -> pd.DataFrame | None:
    if not ASSET_CACHE.exists():
        return None
    try:
        cached = pd.read_csv(ASSET_CACHE, parse_dates=["date"])
    except Exception:
        return None
    if cached.empty or "date" not in cached.columns:
        return None
    return cached.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def merge_price_cache(cached: pd.DataFrame, incremental: pd.DataFrame) -> pd.DataFrame:
    cols = ["date"] + [asset for asset in LIVE_PRICE_COLUMNS if asset in set(cached.columns) | set(incremental.columns)]
    cached_work = cached.reindex(columns=cols)
    incremental_work = incremental.reindex(columns=cols)
    merged = pd.concat([cached_work, incremental_work], ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"]).dt.tz_localize(None)
    merged = merged.sort_values("date").drop_duplicates("date", keep="last")
    return merged.reset_index(drop=True)


def enforce_common_asset_latest(prices: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    needed = LIVE_PRICE_COLUMNS
    latest_by_asset = {}
    for asset in needed:
        if asset not in prices.columns:
            continue
        valid = prices.loc[pd.to_numeric(prices[asset], errors="coerce").notna(), "date"]
        if not valid.empty:
            latest_by_asset[asset] = pd.Timestamp(valid.max())
    if not latest_by_asset:
        return prices
    common_latest = min(latest_by_asset.values())
    newest = max(latest_by_asset.values())
    if common_latest < newest:
        detail = ", ".join(f"{asset}={dt:%Y-%m-%d}" for asset, dt in latest_by_asset.items())
        warnings.append(f"Asset panel truncated to common latest date {common_latest:%Y-%m-%d}; per-asset latest dates: {detail}.")
    out = prices.loc[pd.to_datetime(prices["date"]) <= common_latest].copy()
    return out.sort_values("date").reset_index(drop=True)


def load_local_asset_price_fallback(warnings: list[str]) -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "assets" / "daily_adjusted_close.csv"
    oil_path = ROOT / "data" / "raw" / "market" / "oil_level_raw_prices.csv"
    if not path.exists():
        raise RuntimeError("No asset price series could be downloaded and local processed asset prices are unavailable.")
    local = pd.read_csv(path, parse_dates=["date"])
    rename = {"GLD": "GOLD"}
    keep = ["date", "SPY", "GLD", "IEF", "DBC", "DBB", "DBA"]
    missing = [c for c in keep if c not in local.columns]
    if missing:
        raise RuntimeError(f"Local processed asset prices are missing columns: {missing}")
    out = local[keep].rename(columns=rename)
    if oil_path.exists():
        oil = pd.read_csv(oil_path, parse_dates=["date"])
        oil_col = "CL=F" if "CL=F" in oil.columns else "MCL=F" if "MCL=F" in oil.columns else None
        if oil_col is not None:
            out = out.merge(oil[["date", oil_col]].rename(columns={oil_col: "OIL_SIGNAL"}), on="date", how="left")
        else:
            warnings.append("Local oil raw file exists but has neither CL=F nor MCL=F.")
    else:
        warnings.append("Local oil raw fallback file is missing.")
    warnings.append("Using local processed asset price fallback because yfinance download did not return usable data.")
    return out


def merge_missing_local_assets(prices: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    local = load_local_asset_price_fallback(warnings)
    needed = [asset for asset in LIVE_PRICE_COLUMNS if asset in local.columns]
    merged = prices.merge(local[["date"] + needed], on="date", how="outer", suffixes=("", "_local"))
    updated_assets: list[str] = []
    for asset in needed:
        local_col = f"{asset}_local"
        if local_col in merged.columns:
            merged[asset] = merged.get(asset, pd.Series(np.nan, index=merged.index)).combine_first(merged[local_col])
            merged = merged.drop(columns=[local_col])
            updated_assets.append(asset)
    if updated_assets:
        warnings.append(f"Extended or backfilled live asset series from local data: {', '.join(updated_assets)}.")
    return merged.sort_values("date").reset_index(drop=True)


def load_data(refresh: bool, use_cache: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    warnings: list[str] = []
    macro = fetch_fred_macro(refresh=refresh, use_cache=use_cache, warnings=warnings)
    prices = fetch_asset_prices(refresh=refresh, use_cache=use_cache, warnings=warnings)
    return macro, prices, warnings


def confirmation_days() -> int:
    return int(getattr(final_core, "RATE_CONFIRMATION_DAYS", 2)) if final_core else 2


def confirm_state(raw: list[str], initial: str | None = None) -> list[str]:
    if final_core:
        return final_core.confirm_state(raw, confirmation_days=confirmation_days(), initial=initial)
    values = [str(v) for v in raw]
    current = str(initial or values[0])
    candidate = current
    count = 0
    confirmed = []
    for value in values:
        if value == current:
            candidate = current
            count = 0
        elif value == candidate:
            count += 1
        else:
            candidate = value
            count = 1
        if candidate != current and count >= confirmation_days():
            current = candidate
            candidate = current
            count = 0
        confirmed.append(current)
    return confirmed


def build_monthly_either_state(df: pd.DataFrame) -> pd.Series:
    if final_core:
        return final_core.build_monthly_either_state(df)
    monthly = df[["date", "spy_price"]].dropna().set_index("date").resample("ME").last().dropna().reset_index()
    monthly["spy_12m_return"] = monthly["spy_price"] / monthly["spy_price"].shift(12) - 1.0
    monthly["spy_10m_sma"] = monthly["spy_price"].rolling(10, min_periods=10).mean()
    monthly["monthly_either_state"] = np.where(
        (monthly["spy_12m_return"] <= 0) & (monthly["spy_price"] <= monthly["spy_10m_sma"]), "SELL", "HOLD"
    )
    return pd.merge_asof(df[["date"]], monthly[["date", "monthly_either_state"]], on="date", direction="backward")[
        "monthly_either_state"
    ].fillna("HOLD")


def build_live_oil_state(prices: pd.DataFrame) -> pd.DataFrame:
    oil = pd.to_numeric(prices["OIL_SIGNAL"], errors="coerce").dropna()
    if oil.empty:
        return pd.DataFrame(columns=["date", "oil_price", "oil_ma_252d", "oil_price_to_ma", "oil_level_regime", "oil_level_reason"])
    oil.index = pd.to_datetime(prices.loc[oil.index, "date"]).dt.tz_localize(None)
    oil = pd.Series(oil.to_numpy(), index=oil.index, name="oil_price").sort_index()

    ma_window = int(getattr(final_core, "OIL_MA_WINDOW", 252)) if final_core else 252
    confirm_days = int(getattr(final_core, "OIL_CONFIRM_DAYS", 10)) if final_core else 10
    high_entry = float(getattr(final_core, "OIL_HIGH_ENTRY", 0.20)) if final_core else 0.20
    high_exit = float(getattr(final_core, "OIL_HIGH_EXIT", 0.05)) if final_core else 0.05
    low_entry = float(getattr(final_core, "OIL_LOW_ENTRY", -0.20)) if final_core else -0.20
    low_exit = float(getattr(final_core, "OIL_LOW_EXIT", -0.10)) if final_core else -0.10
    high_label = str(getattr(final_core, "OIL_LEVEL_HIGH", "OIL_LEVEL_HIGH")) if final_core else "OIL_LEVEL_HIGH"
    mid_label = str(getattr(final_core, "OIL_LEVEL_MID", "OIL_LEVEL_MID")) if final_core else "OIL_LEVEL_MID"
    low_label = str(getattr(final_core, "OIL_LEVEL_LOW", "OIL_LEVEL_LOW")) if final_core else "OIL_LEVEL_LOW"

    ma = oil.rolling(ma_window, min_periods=ma_window).mean()
    dev = oil / ma - 1.0
    if final_core and hasattr(final_core, "consecutive_true"):
        enter_high = final_core.consecutive_true(dev >= high_entry, confirm_days)
        exit_high = final_core.consecutive_true(dev <= high_exit, confirm_days)
        enter_low = final_core.consecutive_true(dev <= low_entry, confirm_days)
        exit_low = final_core.consecutive_true(dev >= low_exit, confirm_days)
    else:
        enter_high = (dev >= high_entry).rolling(confirm_days, min_periods=confirm_days).sum().eq(confirm_days)
        exit_high = (dev <= high_exit).rolling(confirm_days, min_periods=confirm_days).sum().eq(confirm_days)
        enter_low = (dev <= low_entry).rolling(confirm_days, min_periods=confirm_days).sum().eq(confirm_days)
        exit_low = (dev >= low_exit).rolling(confirm_days, min_periods=confirm_days).sum().eq(confirm_days)

    current: str | float = np.nan
    states: list[str | float] = []
    reasons: list[str] = []
    for dt in oil.index:
        if pd.isna(dev.loc[dt]):
            states.append(np.nan)
            reasons.append("WARMUP")
            continue
        if pd.isna(current):
            current = mid_label
            reasons.append("INITIAL_MID")
        elif current == mid_label:
            if bool(enter_high.loc[dt]):
                current = high_label
                reasons.append("ENTER_HIGH")
            elif bool(enter_low.loc[dt]):
                current = low_label
                reasons.append("ENTER_LOW")
            else:
                reasons.append("STAY_MID")
        elif current == high_label:
            if bool(exit_high.loc[dt]):
                current = mid_label
                reasons.append("EXIT_HIGH_TO_MID")
            else:
                reasons.append("STAY_HIGH")
        else:
            if bool(exit_low.loc[dt]):
                current = mid_label
                reasons.append("EXIT_LOW_TO_MID")
            else:
                reasons.append("STAY_LOW")
        states.append(current)

    out = pd.DataFrame(
        {
            "date": oil.index,
            "oil_price": oil.values,
            "oil_ma_252d": ma.values,
            "oil_price_to_ma": dev.values,
            "oil_level_regime": states,
            "oil_level_reason": reasons,
        }
    )
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out


def build_live_source_panel(macro: pd.DataFrame, prices: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
    macro = macro.copy()
    macro["date"] = pd.to_datetime(macro["date"]).dt.tz_localize(None)

    for asset in LIVE_PRICE_COLUMNS:
        if asset not in prices.columns:
            prices[asset] = np.nan
            warnings.append(f"Missing asset price column {asset}; related returns and allocations may be degraded.")

    panel = prices[["date", "SPY"]].rename(columns={"SPY": "spy_price"})
    panel["date"] = pd.to_datetime(panel["date"]).dt.tz_localize(None)
    returns = prices[["date"] + RISK_ASSETS].copy()
    for asset in RISK_ASSETS:
        returns[f"{asset}_return"] = pd.to_numeric(returns[asset], errors="coerce").pct_change(fill_method=None)
    panel = panel.merge(returns[["date"] + [f"{asset}_return" for asset in RISK_ASSETS]], on="date", how="left")
    panel = panel.merge(macro[["date"] + [c for c in FRED_SERIES if c in macro.columns]], on="date", how="left")
    oil_state = build_live_oil_state(prices)
    if oil_state.empty:
        panel["oil_price"] = np.nan
        panel["oil_ma_252d"] = np.nan
        panel["oil_price_to_ma"] = np.nan
        panel["oil_level_regime"] = np.nan
        panel["oil_level_reason"] = np.nan
        warnings.append("Oil signal series is unavailable; live dashboard fell back to an empty oil state.")
    else:
        panel = pd.merge_asof(panel.sort_values("date"), oil_state.sort_values("date"), on="date", direction="backward")
    panel = panel.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    for col in ["DGS10", "DGS1", "DTB3", "VIXCLS", "DAAA", "DBAA"]:
        if col not in panel.columns:
            panel[col] = np.nan
            warnings.append(f"Missing FRED series {col}.")
        panel[col] = pd.to_numeric(panel[col], errors="coerce").ffill().bfill()

    panel = panel.rename(columns={"VIXCLS": "VIX_LEVEL", "DAAA": "WAAA", "DBAA": "WBAA"})
    panel["GS10"] = panel["DGS10"]
    panel["GS1"] = panel["DGS1"]
    panel["TERM_SPREAD_10Y_1Y"] = panel["DGS10"] - panel["DGS1"]
    panel["CREDIT_SPREAD_BAA_AAA"] = panel["WBAA"] - panel["WAAA"]
    panel["D_CREDIT_SPREAD_20D"] = panel["CREDIT_SPREAD_BAA_AAA"] - panel["CREDIT_SPREAD_BAA_AAA"].shift(20)
    credit_window = int(getattr(final_core, "TRIGGER_LOCK_CREDIT_WINDOW", 15)) if final_core else 15
    panel["D_CREDIT_SPREAD_15D"] = panel["CREDIT_SPREAD_BAA_AAA"] - panel["CREDIT_SPREAD_BAA_AAA"].shift(credit_window)
    panel["CASH_return"] = (1.0 + panel["DTB3"] / 100.0) ** (1.0 / 252.0) - 1.0
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
        panel["CREDIT_SPREAD_BAA_AAA"] - credit_roll.mean()
    ) / credit_roll.std(ddof=1).replace(0, np.nan)
    panel["DBC_price"] = pd.to_numeric(prices["DBC"], errors="coerce").reindex(prices.index).ffill().to_numpy()
    panel["DBC_RET60"] = panel["DBC_price"] / panel["DBC_price"].shift(60) - 1.0
    panel["DBC_RET20"] = panel["DBC_price"] / panel["DBC_price"].shift(20) - 1.0

    if final_core and hasattr(final_core, "classify_macro_hysteresis"):
        raw_regime = final_core.classify_macro_hysteresis(panel["TERM_SPREAD_10Y_1Y"])
    else:
        raw_regime = np.select(
            [panel["TERM_SPREAD_10Y_1Y"] < 0, panel["TERM_SPREAD_10Y_1Y"] <= 1, panel["TERM_SPREAD_10Y_1Y"] > 1],
            ["INVERTED", "FLAT", "STEEP"],
            default="FLAT",
        )
    panel["macro_regime_raw"] = raw_regime
    panel["macro_regime_confirmed"] = confirm_state(list(raw_regime), initial=str(raw_regime[0]))
    panel["monthly_either_state"] = build_monthly_either_state(panel)

    for col in [f"{asset}_return" for asset in RISK_ASSETS] + ["CASH_return", "daily_rf"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")

    required = [f"{asset}_return" for asset in RISK_ASSETS] + ["CASH_return", "daily_rf", "oil_level_regime"]
    panel = panel.dropna(subset=required).reset_index(drop=True)

    if final_core:
        flat_regime = final_core.classify_flat_three_state(panel)
        final_regime = final_core.classify_steep_three_state(panel, flat_regime)
    else:
        flat_regime = panel["macro_regime_confirmed"].astype(str)
        final_regime = flat_regime
    panel["refined_regime_confirmed"] = final_regime
    panel["final_regime_confirmed"] = final_regime
    panel["flat_refined_state"] = final_regime
    panel["steep_rate_regime_confirmed"] = np.where(final_regime.astype(str).str.startswith("STEEP_"), final_regime, pd.NA)
    return panel.dropna(subset=["spy_price"]).reset_index(drop=True)


def compute_live_strategy(panel: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    if not final_core:
        raise RuntimeError(FINAL_IMPORT_WARNING)
    inv_vol_window = int(getattr(final_core, "INV_VOL_WINDOW", 90))
    final_weights, trigger_state = final_core.build_trigger_lock_final_weights(panel, inv_vol_window=inv_vol_window)
    final = final_core.compute_strategy(panel, final_weights, FINAL_STRATEGY)
    spy_weights = pd.DataFrame(0.0, index=panel.index, columns=ASSETS)
    spy_weights["SPY"] = 1.0
    spy = final_core.compute_strategy(panel, spy_weights, SPY_BUY_HOLD)
    out = pd.concat([panel, trigger_state, final, spy], axis=1)
    stress_active = trigger_state["trigger_lock_full_risk_state"].eq("FULL_RISK")
    out["full_risk_state"] = trigger_state["trigger_lock_full_risk_state"]
    out["final_state"] = np.where(stress_active, "FULL_RISK", "NON_RISK")
    return out


def latest_series_dates(macro: pd.DataFrame, prices: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    fred_dates = {}
    for col in FRED_SERIES:
        if col in macro.columns:
            latest = macro.loc[pd.to_numeric(macro[col], errors="coerce").notna(), "date"].max()
            fred_dates[col] = "" if pd.isna(latest) else pd.Timestamp(latest).strftime("%Y-%m-%d")
    asset_dates = {}
    for col in LIVE_PRICE_COLUMNS:
        if col in prices.columns:
            latest = prices.loc[pd.to_numeric(prices[col], errors="coerce").notna(), "date"].max()
            asset_dates[col] = "" if pd.isna(latest) else pd.Timestamp(latest).strftime("%Y-%m-%d")
    asset_dates["CASH"] = fred_dates.get("DTB3", "")
    return fred_dates, asset_dates


def add_freshness_warnings(panel: pd.DataFrame, fred_dates: dict[str, str], asset_dates: dict[str, str], warnings: list[str]) -> pd.Timestamp:
    latest_candidates = [pd.Timestamp(v) for v in list(fred_dates.values()) + list(asset_dates.values()) if v]
    latest_common = panel["date"].max()
    if latest_candidates:
        newest = max(latest_candidates)
        for name, value in {**fred_dates, **asset_dates}.items():
            if value and (newest - pd.Timestamp(value)).days > 3:
                warnings.append(f"{name} latest date {value} lags newest available data date {newest:%Y-%m-%d}.")
    if "VIXCLS" in fred_dates and "SPY" in asset_dates and fred_dates["VIXCLS"] and asset_dates["SPY"]:
        vix_date = pd.Timestamp(fred_dates["VIXCLS"])
        spy_date = pd.Timestamp(asset_dates["SPY"])
        if vix_date < spy_date:
            warnings.append(f"VIXCLS latest date {vix_date:%Y-%m-%d} is earlier than SPY {spy_date:%Y-%m-%d}; using common strategy panel date {latest_common:%Y-%m-%d}.")
    return latest_common


def contiguous_start(df: pd.DataFrame, col: str, latest_idx: int) -> pd.Timestamp:
    value = df.loc[latest_idx, col]
    start_idx = latest_idx
    while start_idx > 0 and df.loc[start_idx - 1, col] == value:
        start_idx -= 1
    return pd.Timestamp(df.loc[start_idx, "date"])


def current_stress_start(df: pd.DataFrame, latest_idx: int) -> pd.Timestamp | None:
    if str(df.loc[latest_idx, "trigger_lock_full_risk_state"]) != "FULL_RISK":
        return None
    start_idx = latest_idx
    while start_idx > 0 and str(df.loc[start_idx - 1, "trigger_lock_full_risk_state"]) == "FULL_RISK":
        start_idx -= 1
    return pd.Timestamp(df.loc[start_idx, "date"])


def regime_boundary_signal(row: pd.Series) -> dict[str, Any]:
    regime = str(row["final_regime_confirmed"])
    gs10 = float(row["GS10"])
    term = float(row["TERM_SPREAD_10Y_1Y"])
    candidates = [
        ("flat->inverted threshold", float(getattr(final_core, "OUTER_FLAT_TO_INV", -0.10)) if final_core else -0.10, term),
        ("inverted->flat threshold", float(getattr(final_core, "OUTER_INV_TO_FLAT", 0.10)) if final_core else 0.10, term),
        ("flat->steep threshold", float(getattr(final_core, "OUTER_FLAT_TO_STEEP", 1.20)) if final_core else 1.20, term),
        ("steep->flat threshold", float(getattr(final_core, "OUTER_STEEP_TO_FLAT", 1.00)) if final_core else 1.00, term),
    ]
    if regime.startswith("FLAT_"):
        candidates += [
            ("flat low/mid lower band", getattr(final_core, "FLAT_MID_TO_LOW", 1.1), gs10),
            ("flat low/mid upper band", getattr(final_core, "FLAT_LOW_TO_MID", 1.3), gs10),
            ("flat mid/high lower band", getattr(final_core, "FLAT_HIGH_TO_MID", 3.4), gs10),
            ("flat mid/high upper band", getattr(final_core, "FLAT_MID_TO_HIGH", 3.6), gs10),
        ]
    if regime.startswith("STEEP_"):
        candidates += [
            ("steep low/mid lower band", getattr(final_core, "STEEP_MID_TO_LOW", 2.0), gs10),
            ("steep low/mid upper band", getattr(final_core, "STEEP_LOW_TO_MID", 2.3), gs10),
            ("steep mid/high lower band", getattr(final_core, "STEEP_HIGH_TO_MID", 3.0), gs10),
            ("steep mid/high upper band", getattr(final_core, "STEEP_MID_TO_HIGH", 3.2), gs10),
        ]
    name, threshold, current = min(candidates, key=lambda x: abs(x[2] - x[1]))
    return {
        "category": "Regime variables",
        "signal": "Nearest regime boundary",
        "current_value": current,
        "threshold_name": name,
        "threshold_value": threshold,
        "distance": current - threshold,
        "condition_met": abs(current - threshold) < 0.10,
        "status": "near boundary" if abs(current - threshold) < 0.10 else "away from boundary",
        "interpretation": f"Current regime is {regime}; nearest boundary is {name}.",
    }


def build_signal_distance(row: pd.Series) -> pd.DataFrame:
    credit_entry = float(getattr(final_core, "TRIGGER_LOCK_CREDIT_ENTRY_THRESHOLD", 0.10)) if final_core else 0.10
    credit_unlock = float(getattr(final_core, "TRIGGER_LOCK_CREDIT_LEVEL_Z_EXIT_THRESHOLD", 0.9)) if final_core else 0.9
    rows = [
        {
            "category": "VIX",
            "signal": "VIX Z-score",
            "current_value": row["VIX_ZSCORE_120D"],
            "threshold_name": "entry",
            "threshold_value": 3.0,
            "distance": row["VIX_ZSCORE_120D"] - 3.0,
            "condition_met": bool(row["VIX_ZSCORE_120D"] >= 3.0),
            "status": "entry met" if bool(row["VIX_ZSCORE_120D"] >= 3.0) else "below entry",
            "interpretation": "VIX stress entry is active at or above 3.0 z-score.",
        },
        {
            "category": "VIX",
            "signal": "VIX Z-score",
            "current_value": row["VIX_ZSCORE_120D"],
            "threshold_name": "unlock",
            "threshold_value": 1.5,
            "distance": row["VIX_ZSCORE_120D"] - 1.5,
            "condition_met": bool((row["VIX_ZSCORE_120D"] < 1.5) and row["SPY_above_MA20"]),
            "status": "unlock condition met" if bool((row["VIX_ZSCORE_120D"] < 1.5) and row["SPY_above_MA20"]) else "unlock not met",
            "interpretation": "VIX lock unlock also requires SPY above MA20.",
        },
        {
            "category": "Credit",
            "signal": "Credit 15D change",
            "current_value": row["D_CREDIT_SPREAD_15D"],
            "threshold_name": "entry",
            "threshold_value": credit_entry,
            "distance": row["D_CREDIT_SPREAD_15D"] - credit_entry,
            "condition_met": bool((row["D_CREDIT_SPREAD_15D"] > credit_entry) and (not row["SPY_above_MA20"])),
            "status": "entry met" if bool((row["D_CREDIT_SPREAD_15D"] > credit_entry) and (not row["SPY_above_MA20"])) else "below entry",
            "interpretation": "Credit lock entry also requires SPY below MA20.",
        },
        {
            "category": "Credit",
            "signal": "Credit level Z",
            "current_value": row["CREDIT_LEVEL_Z_252D"],
            "threshold_name": "unlock",
            "threshold_value": credit_unlock,
            "distance": row["CREDIT_LEVEL_Z_252D"] - credit_unlock,
            "condition_met": bool(row["SPY_above_MA50"] and row["CREDIT_LEVEL_Z_252D"] < credit_unlock),
            "status": "unlock condition met" if bool(row["SPY_above_MA50"] and row["CREDIT_LEVEL_Z_252D"] < credit_unlock) else "unlock not met",
            "interpretation": "Credit unlock also requires SPY above MA50.",
        },
        {
            "category": "SPY trend",
            "signal": "SPY vs MA20",
            "current_value": row["spy_price"],
            "threshold_name": "MA20",
            "threshold_value": row["SPY_MA20"],
            "distance": row["spy_price"] - row["SPY_MA20"],
            "condition_met": bool(row["SPY_above_MA20"]),
            "status": "above MA20" if bool(row["SPY_above_MA20"]) else "below MA20",
            "interpretation": "Positive distance means SPY is above MA20.",
        },
        {
            "category": "SPY trend",
            "signal": "SPY vs MA50",
            "current_value": row["spy_price"],
            "threshold_name": "MA50",
            "threshold_value": row["SPY_MA50"],
            "distance": row["spy_price"] - row["SPY_MA50"],
            "condition_met": bool(row["SPY_above_MA50"]),
            "status": "above MA50" if bool(row["SPY_above_MA50"]) else "below MA50",
            "interpretation": "Positive distance means SPY is above MA50.",
        },
        {
            "category": "SPY trend",
            "signal": "SPY drawdown from previous high",
            "current_value": row["spy_drawdown_from_previous_high"],
            "threshold_name": "drawdown monitor",
            "threshold_value": -0.05,
            "distance": row["spy_drawdown_from_previous_high"] - (-0.05),
            "condition_met": bool(row["spy_drawdown_from_previous_high"] <= -0.05),
            "status": "drawdown trigger zone" if bool(row["spy_drawdown_from_previous_high"] <= -0.05) else "above drawdown monitor",
            "interpretation": "Drawdown is monitored as context for risk stress.",
        },
        {
            "category": "Regime variables",
            "signal": "Term spread boundary",
            "current_value": row["TERM_SPREAD_10Y_1Y"],
            "threshold_name": "flat->steep threshold",
            "threshold_value": float(getattr(final_core, "OUTER_FLAT_TO_STEEP", 1.20)) if final_core else 1.20,
            "distance": row["TERM_SPREAD_10Y_1Y"] - (float(getattr(final_core, "OUTER_FLAT_TO_STEEP", 1.20)) if final_core else 1.20),
            "condition_met": bool(row["TERM_SPREAD_10Y_1Y"] > (float(getattr(final_core, "OUTER_FLAT_TO_STEEP", 1.20)) if final_core else 1.20)),
            "status": "beyond steep entry" if bool(row["TERM_SPREAD_10Y_1Y"] > (float(getattr(final_core, "OUTER_FLAT_TO_STEEP", 1.20)) if final_core else 1.20)) else "below steep entry",
            "interpretation": "Outer macro regime uses hysteresis, not a single 0/1 threshold.",
        },
    ]
    if pd.notna(row.get("oil_price_to_ma")):
        rows.extend(
            [
                {
                    "category": "Oil level",
                    "signal": "Oil price vs 252d MA",
                    "current_value": row["oil_price_to_ma"],
                    "threshold_name": "HIGH entry",
                    "threshold_value": float(getattr(final_core, "OIL_HIGH_ENTRY", 0.20)) if final_core else 0.20,
                    "distance": row["oil_price_to_ma"] - (float(getattr(final_core, "OIL_HIGH_ENTRY", 0.20)) if final_core else 0.20),
                    "condition_met": bool(row["oil_price_to_ma"] >= (float(getattr(final_core, "OIL_HIGH_ENTRY", 0.20)) if final_core else 0.20)),
                    "status": "above HIGH entry" if bool(row["oil_price_to_ma"] >= (float(getattr(final_core, "OIL_HIGH_ENTRY", 0.20)) if final_core else 0.20)) else "below HIGH entry",
                    "interpretation": "Oil HIGH removes GOLD and DBB from flat sleeves.",
                },
                {
                    "category": "Oil level",
                    "signal": "Oil price vs 252d MA",
                    "current_value": row["oil_price_to_ma"],
                    "threshold_name": "LOW entry",
                    "threshold_value": float(getattr(final_core, "OIL_LOW_ENTRY", -0.20)) if final_core else -0.20,
                    "distance": row["oil_price_to_ma"] - (float(getattr(final_core, "OIL_LOW_ENTRY", -0.20)) if final_core else -0.20),
                    "condition_met": bool(row["oil_price_to_ma"] <= (float(getattr(final_core, "OIL_LOW_ENTRY", -0.20)) if final_core else -0.20)),
                    "status": "below LOW entry" if bool(row["oil_price_to_ma"] <= (float(getattr(final_core, "OIL_LOW_ENTRY", -0.20)) if final_core else -0.20)) else "above LOW entry",
                    "interpretation": "Oil LOW keeps the flat sleeves fully open.",
                },
            ]
        )
    rows.append(regime_boundary_signal(row))
    out = pd.DataFrame(rows)
    fallback_status = pd.Series(
        np.where(out["condition_met"], "condition met", "condition not met"),
        index=out.index,
    )
    out["status"] = out["status"].fillna(fallback_status)
    return out


def build_current_allocation(panel: pd.DataFrame, latest_idx: int) -> pd.DataFrame:
    row = panel.loc[latest_idx]
    state = str(row["final_allocation_state"])
    stress = str(row["trigger_lock_full_risk_state"]) == "FULL_RISK"
    returns = panel[[f"{asset}_return" for asset in ASSETS]].rename(columns={f"{asset}_return": asset for asset in ASSETS})
    inv_window = int(getattr(final_core, "INV_VOL_WINDOW", 90)) if final_core else 90
    vol = returns.rolling(inv_window, min_periods=1).std(ddof=1).mul(np.sqrt(252.0)).iloc[latest_idx]
    active_pool = [asset for asset in ASSETS if float(row.get(f"{FINAL_STRATEGY}_weight_{asset}", 0.0)) > 1e-8]
    inv = 1.0 / vol.reindex(active_pool).replace(0, np.nan)
    raw = inv / inv.sum(skipna=True)
    rows = []
    for asset in ASSETS:
        weight = float(row.get(f"{FINAL_STRATEGY}_weight_{asset}", 0.0))
        method = "fixed_hedge_sleeve" if stress else ("fixed_single_asset" if weight >= 0.999 else "inverse_vol_monthly_hold")
        if asset == "CASH":
            reason = "cash sleeve uses FRED 3-month T-bill daily accrual (DTB3, or TB3MS fallback)"
        elif asset in {"DBC", "DBB", "DBA"}:
            reason = (
                f"{asset} is active under {row['final_allocation_state']}; oil state is {row['oil_level_regime']}."
                if stress or str(row["final_regime_confirmed"]).startswith("FLAT_")
                else f"{asset} is inactive in the current regime template."
            )
        else:
            reason = "stress allocation rationale: trigger-lock is active" if stress else f"normal allocation for {row['final_regime_confirmed']}"
        rows.append(
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "state": state,
                "asset": asset,
                "weight": weight,
                "allocation_method": method,
                "realized_vol": float(vol.get(asset, np.nan)) if pd.notna(vol.get(asset, np.nan)) else np.nan,
                "inverse_vol_raw_weight": float(raw.get(asset, np.nan)) if pd.notna(raw.get(asset, np.nan)) else np.nan,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def weights_summary(row: pd.Series) -> str:
    parts = []
    for asset in ASSETS:
        value = float(row.get(f"{FINAL_STRATEGY}_weight_{asset}", 0.0))
        if abs(value) > 1e-8:
            parts.append(f"{asset} {value:.2%}")
    return ", ".join(parts) if parts else "No active weight"


def allocation_method_from_row(row: pd.Series) -> str:
    if str(row.get("trigger_lock_full_risk_state", "")) == "FULL_RISK":
        return "fixed_hedge_sleeve"
    weights = [float(row.get(f"{FINAL_STRATEGY}_weight_{asset}", 0.0)) for asset in ASSETS]
    if max(weights) >= 0.999:
        return "fixed_single_asset"
    return "inverse_vol_monthly_hold"


def rebalance_reason(panel: pd.DataFrame, i: int, weight_changed: bool) -> str:
    if i == 0:
        return "first_day"
    row = panel.iloc[i]
    prev = panel.iloc[i - 1]
    if row["final_regime_confirmed"] != prev["final_regime_confirmed"]:
        return "regime_change"
    if row["trigger_lock_full_risk_state"] == "FULL_RISK" and prev["trigger_lock_full_risk_state"] != "FULL_RISK":
        return "stress_entry"
    if row["trigger_lock_full_risk_state"] != "FULL_RISK" and prev["trigger_lock_full_risk_state"] == "FULL_RISK":
        return "stress_unlock"
    if bool(row.get("trigger_lock_entry_signal", False)):
        return "stress_entry"
    if bool(row.get("trigger_lock_exit_signal", False)):
        return "stress_unlock"
    if row["final_allocation_state"] != prev["final_allocation_state"]:
        return "allocation_universe_change"
    if bool(row["date"].to_period("M") != prev["date"].to_period("M")):
        return "scheduled_rebalance"
    if weight_changed:
        return "inverse_vol_update"
    return "other"


def build_dynamic_weights(panel: pd.DataFrame, regime_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = panel.loc[panel["date"] >= regime_start].copy().reset_index(drop=True)
    weight_cols = [f"{FINAL_STRATEGY}_weight_{asset}" for asset in ASSETS]
    weight_diff = sub[weight_cols].diff().abs().sum(axis=1).fillna(0.0)
    rebalance_flags = weight_diff > 1e-10
    if not sub.empty:
        rebalance_flags.iloc[0] = True

    rows = []
    rebalance_rows = []
    for i, row in sub.iterrows():
        reason = rebalance_reason(sub, i, bool(rebalance_flags.iloc[i])) if bool(rebalance_flags.iloc[i]) else ""
        active_locks = str(row.get("trigger_lock_active_locks", ""))
        rows.append(
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "macro_regime": row["final_regime_confirmed"],
                "stress_state": row["trigger_lock_full_risk_state"],
                "active_locks": "" if active_locks == "nan" else active_locks,
                "rebalance_flag": bool(rebalance_flags.iloc[i]),
                "rebalance_reason": reason,
                "allocation_method": allocation_method_from_row(row),
            }
        )
        for asset in ASSETS:
            rows[-1][f"{asset}_weight"] = float(row.get(f"{FINAL_STRATEGY}_weight_{asset}", 0.0))
        if bool(rebalance_flags.iloc[i]):
            prev = sub.iloc[i - 1] if i > 0 else None
            rebalance_rows.append(
                {
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "rebalance_reason": reason,
                    "previous_state": "" if prev is None else str(prev["final_allocation_state"]),
                    "new_state": str(row["final_allocation_state"]),
                    "previous_weights_summary": "" if prev is None else weights_summary(prev),
                    "new_weights_summary": weights_summary(row),
                    "active_locks": "" if active_locks == "nan" else active_locks,
                }
            )

    dynamic = pd.DataFrame(rows)
    rebalances = pd.DataFrame(
        rebalance_rows,
        columns=[
            "date",
            "rebalance_reason",
            "previous_state",
            "new_state",
            "previous_weights_summary",
            "new_weights_summary",
            "active_locks",
        ],
    )
    return dynamic, rebalances


def slice_regime_performance(panel: pd.DataFrame, regime_start: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, float]]:
    sub = panel.loc[panel["date"] >= regime_start].copy()
    perf = pd.DataFrame(
        {
            "date": sub["date"].dt.strftime("%Y-%m-%d"),
            "strategy_return": sub[f"{FINAL_STRATEGY}_return"],
            "spy_return": sub[f"{SPY_BUY_HOLD}_return"],
        }
    )
    perf["strategy_nav"] = (1.0 + perf["strategy_return"]).cumprod()
    perf["spy_nav"] = (1.0 + perf["spy_return"]).cumprod()
    perf["strategy_drawdown"] = perf["strategy_nav"] / perf["strategy_nav"].cummax() - 1.0
    perf["spy_drawdown"] = perf["spy_nav"] / perf["spy_nav"].cummax() - 1.0
    summary = {
        "strategy_return": float(perf["strategy_nav"].iloc[-1] - 1.0),
        "spy_return": float(perf["spy_nav"].iloc[-1] - 1.0),
        "excess_return": float(perf["strategy_nav"].iloc[-1] - perf["spy_nav"].iloc[-1]),
        "strategy_max_drawdown": float(perf["strategy_drawdown"].min()),
        "spy_max_drawdown": float(perf["spy_drawdown"].min()),
        "strategy_current_drawdown": float(perf["strategy_drawdown"].iloc[-1]),
        "spy_current_drawdown": float(perf["spy_drawdown"].iloc[-1]),
    }
    return perf, summary


def max_drawdown_from_returns(ret: pd.Series) -> float:
    nav = (1.0 + ret.fillna(0.0)).cumprod()
    return float((nav / nav.cummax() - 1.0).min()) if len(nav) else np.nan


def stress_episodes(panel: pd.DataFrame, regime_start: pd.Timestamp) -> pd.DataFrame:
    sub = panel.loc[panel["date"] >= regime_start].copy().reset_index(drop=True)
    active = sub["trigger_lock_full_risk_state"].eq("FULL_RISK")
    if not active.any():
        return pd.DataFrame(
            columns=[
                "episode_id",
                "start_date",
                "end_date",
                "active_trigger",
                "duration_days",
                "SPY_return",
                "strategy_return",
                "CASH_return",
                "SPY_max_drawdown",
                "strategy_max_drawdown",
                "avoided_drawdown_estimate",
            ]
        )
    group_id = active.ne(active.shift()).cumsum()
    rows = []
    episode_id = 1
    for _, grp in sub.loc[active].groupby(group_id[active]):
        rows.append(
            {
                "episode_id": episode_id,
                "start_date": grp["date"].iloc[0].strftime("%Y-%m-%d"),
                "end_date": grp["date"].iloc[-1].strftime("%Y-%m-%d"),
                "active_trigger": "+".join(sorted(set("+".join(grp["trigger_lock_active_locks"].fillna("")).split("+")) - {""})),
                "duration_days": int(len(grp)),
                "SPY_return": float((1.0 + grp[f"{SPY_BUY_HOLD}_return"]).prod() - 1.0),
                "strategy_return": float((1.0 + grp[f"{FINAL_STRATEGY}_return"]).prod() - 1.0),
                "CASH_return": float((1.0 + grp["CASH_return"]).prod() - 1.0),
                "SPY_max_drawdown": max_drawdown_from_returns(grp[f"{SPY_BUY_HOLD}_return"]),
                "strategy_max_drawdown": max_drawdown_from_returns(grp[f"{FINAL_STRATEGY}_return"]),
                "avoided_drawdown_estimate": max_drawdown_from_returns(grp[f"{SPY_BUY_HOLD}_return"])
                - max_drawdown_from_returns(grp[f"{FINAL_STRATEGY}_return"]),
            }
        )
        episode_id += 1
    return pd.DataFrame(rows)


def pct(x: float | None) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.2%}"


def num(x: float | None) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.3f}"


def write_csv_outputs(
    output_dir: Path,
    status: dict[str, Any],
    signal_distance: pd.DataFrame,
    allocation: pd.DataFrame,
    regime_perf: pd.DataFrame,
    episodes: pd.DataFrame,
    dynamic_weights: pd.DataFrame,
    rebalances: pd.DataFrame,
) -> None:
    (output_dir / "current_status.json").write_text(json.dumps(json_safe(status), indent=2), encoding="utf-8")
    signal_distance.to_csv(output_dir / "current_signal_distance.csv", index=False)
    allocation.to_csv(output_dir / "current_allocation.csv", index=False)
    regime_perf.to_csv(output_dir / "current_regime_performance.csv", index=False)
    episodes.to_csv(output_dir / "current_stress_episodes.csv", index=False)
    dynamic_weights.to_csv(output_dir / "current_dynamic_weights.csv", index=False)
    rebalances.to_csv(output_dir / "current_rebalance_dates.csv", index=False)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else None
    if pd.isna(value):
        return None
    return value


def plot_allocation(allocation: pd.DataFrame, path: Path, state: str) -> None:
    data = allocation.sort_values("weight")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(data["asset"], data["weight"], color="#2f6f73")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_xlim(0, max(1.0, data["weight"].max() * 1.15))
    ax.set_title(f"Current Allocation - {state}")
    ax.set_xlabel("Weight")
    for i, v in enumerate(data["weight"]):
        ax.text(v + 0.01, i, f"{v:.1%}", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_signal_distance(signal: pd.DataFrame, path: Path) -> None:
    data = signal.loc[signal["distance"].notna()].copy()
    data["label"] = data["signal"] + " (" + data["threshold_name"] + ")"
    data = data.sort_values("distance")
    colors = np.where(data["condition_met"], "#b4493f", "#3d6f9f")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(data["label"], data["distance"], color=colors)
    ax.axvline(0, color="#222222", linewidth=1)
    ax.set_title("Current Signal Distance to Threshold")
    ax.set_xlabel("Current value minus threshold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


ASSET_COLORS = {
    "SPY": "#3d4f8f",
    "GOLD": "#c9a65a",
    "IEF": "#5e8cb8",
    "DBC": "#2f6f73",
    "DBB": "#8d5a2b",
    "DBA": "#7b8f46",
    "CASH": "#8b96a3",
}


def shade_stress_periods(ax: plt.Axes, data: pd.DataFrame) -> None:
    stress = data["trigger_lock_full_risk_state"].eq("FULL_RISK") if "trigger_lock_full_risk_state" in data.columns else pd.Series(False, index=data.index)
    if stress.any():
        starts = stress.ne(stress.shift()).cumsum()
        for _, grp in data.loc[stress].groupby(starts[stress]):
            ax.axvspan(grp["date"].iloc[0], grp["date"].iloc[-1], color="#b4493f", alpha=0.13, linewidth=0)


def annotate_latest_distance(ax: plt.Axes, x: pd.Timestamp, y: float, threshold: float, label: str) -> None:
    if pd.isna(y) or pd.isna(threshold):
        return
    ax.plot([x], [y], marker="o", color="#111111", markersize=4)
    ax.vlines(x, min(y, threshold), max(y, threshold), color="#111111", linewidth=1.2, linestyle=":")
    ax.annotate(
        f"{label}: {y - threshold:+.2f}",
        xy=(x, (y + threshold) / 2),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color="#111111",
    )


def plot_dynamic_weights_rebalance_bar(dynamic_weights: pd.DataFrame, rebalances: pd.DataFrame, path: Path) -> None:
    data = dynamic_weights.loc[dynamic_weights["rebalance_flag"]].copy()
    if data.empty:
        data = dynamic_weights.tail(1).copy()
    data["date_label"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")
    weight_cols = {asset: f"{asset}_weight" for asset in ASSETS}
    fig_width = max(11, min(22, 0.42 * len(data) + 5))
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    x = np.arange(len(data))
    bottoms = np.zeros(len(data))
    legend_seen: set[str] = set()
    for i, (_, row) in enumerate(data.iterrows()):
        ordered = sorted(weight_cols.items(), key=lambda kv: float(row[kv[1]]), reverse=True)
        bottom = 0.0
        for asset, col in ordered:
            weight = float(row[col])
            if weight <= 1e-8:
                continue
            label = asset if asset not in legend_seen else "_nolegend_"
            legend_seen.add(asset)
            ax.bar(i, weight, bottom=bottom, color=ASSET_COLORS.get(asset, "#64748b"), width=0.72, label=label)
            if weight >= 0.08:
                ax.text(i, bottom + weight / 2, f"{asset}\n{weight:.0%}", ha="center", va="center", fontsize=8, color="white")
            bottom += weight
        bottoms[i] = bottom
    reason_lookup = rebalances.set_index("date")["rebalance_reason"].to_dict() if not rebalances.empty else {}
    tick_labels = [f"{d}\n{reason_lookup.get(d, '')}" for d in data["date_label"]]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=50, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    ax.set_ylim(0, max(1.0, bottoms.max() if len(bottoms) else 1.0))
    ax.set_title("Target Weights at Rebalance Dates")
    ax.set_ylabel("Target weight")
    ax.legend(loc="upper left", ncol=min(len(ASSETS), 7), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_vix_signal(panel: pd.DataFrame, regime_start: pd.Timestamp, path: Path) -> None:
    data = panel.loc[panel["date"] >= regime_start].copy()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    shade_stress_periods(ax, data)
    ax.plot(data["date"], data["VIX_ZSCORE_120D"], color="#6f4a8e", linewidth=1.8, label="VIX Z-score")
    ax.axhline(3.0, color="#b4493f", linestyle="--", linewidth=1.2, label="entry 3.0")
    ax.axhline(1.5, color="#2f6f73", linestyle="--", linewidth=1.2, label="unlock 1.5")
    latest = data.iloc[-1]
    annotate_latest_distance(ax, latest["date"], float(latest["VIX_ZSCORE_120D"]), 3.0, "to entry")
    annotate_latest_distance(ax, latest["date"], float(latest["VIX_ZSCORE_120D"]), 1.5, "to unlock")
    ax.set_title("VIX Z-score Monitor")
    ax.set_ylabel("Z-score")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_credit_15d_signal(panel: pd.DataFrame, regime_start: pd.Timestamp, path: Path) -> None:
    threshold = float(getattr(final_core, "TRIGGER_LOCK_CREDIT_ENTRY_THRESHOLD", 0.10)) if final_core else 0.10
    data = panel.loc[panel["date"] >= regime_start].copy()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    shade_stress_periods(ax, data)
    ax.plot(data["date"], data["D_CREDIT_SPREAD_15D"], color="#8d5a2b", linewidth=1.8, label="Credit 15D change")
    ax.axhline(threshold, color="#b4493f", linestyle="--", linewidth=1.2, label=f"entry {threshold:.2f}")
    latest = data.iloc[-1]
    annotate_latest_distance(ax, latest["date"], float(latest["D_CREDIT_SPREAD_15D"]), threshold, "to entry")
    ax.set_title("Credit 15D Change Monitor")
    ax.set_ylabel("Spread change")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_credit_level_signal(panel: pd.DataFrame, regime_start: pd.Timestamp, path: Path) -> None:
    threshold = float(getattr(final_core, "TRIGGER_LOCK_CREDIT_LEVEL_Z_EXIT_THRESHOLD", 0.9)) if final_core else 0.9
    data = panel.loc[panel["date"] >= regime_start].copy()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    shade_stress_periods(ax, data)
    ax.plot(data["date"], data["CREDIT_LEVEL_Z_252D"], color="#3d6f9f", linewidth=1.8, label="Credit level Z")
    ax.axhline(threshold, color="#2f6f73", linestyle="--", linewidth=1.2, label=f"unlock {threshold:.2f}")
    latest = data.iloc[-1]
    annotate_latest_distance(ax, latest["date"], float(latest["CREDIT_LEVEL_Z_252D"]), threshold, "to unlock")
    ax.set_title("Credit Level Z Monitor")
    ax.set_ylabel("Z-score")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_oil_signal(panel: pd.DataFrame, regime_start: pd.Timestamp, path: Path) -> None:
    data = panel.loc[panel["date"] >= regime_start].copy()
    if data.empty:
        return
    high_entry = float(getattr(final_core, "OIL_HIGH_ENTRY", 0.20)) if final_core else 0.20
    low_entry = float(getattr(final_core, "OIL_LOW_ENTRY", -0.20)) if final_core else -0.20
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(data["date"], data["oil_price_to_ma"], color="#8d5a2b", linewidth=1.8, label="Oil price / 252d MA - 1")
    ax.axhline(high_entry, color="#b4493f", linestyle="--", linewidth=1.2, label=f"HIGH entry {high_entry:.0%}")
    ax.axhline(low_entry, color="#2f6f73", linestyle="--", linewidth=1.2, label=f"LOW entry {low_entry:.0%}")
    latest = data.iloc[-1]
    annotate_latest_distance(ax, latest["date"], float(latest["oil_price_to_ma"]), high_entry, "to HIGH")
    annotate_latest_distance(ax, latest["date"], float(latest["oil_price_to_ma"]), low_entry, "to LOW")
    ax.set_title(f"Oil Level Monitor ({latest['oil_level_regime']})")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    ax.set_ylabel("Deviation from 252d MA")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_term_spread_signal(panel: pd.DataFrame, regime_start: pd.Timestamp, path: Path) -> None:
    data = panel.loc[panel["date"] >= regime_start].copy()
    if data.empty:
        return
    f2i = float(getattr(final_core, "OUTER_FLAT_TO_INV", -0.10)) if final_core else -0.10
    i2f = float(getattr(final_core, "OUTER_INV_TO_FLAT", 0.10)) if final_core else 0.10
    f2s = float(getattr(final_core, "OUTER_FLAT_TO_STEEP", 1.20)) if final_core else 1.20
    s2f = float(getattr(final_core, "OUTER_STEEP_TO_FLAT", 1.00)) if final_core else 1.00
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(data["date"], data["TERM_SPREAD_10Y_1Y"], color="#3d6f9f", linewidth=1.8, label="10Y - 1Y term spread")
    for threshold, color, label in [
        (f2i, "#b4493f", "flat->inverted"),
        (i2f, "#f97316", "inverted->flat"),
        (s2f, "#2f6f73", "steep->flat"),
        (f2s, "#7c3aed", "flat->steep"),
    ]:
        ax.axhline(threshold, color=color, linestyle="--", linewidth=1.2, label=f"{label} {threshold:.2f}")
    latest = data.iloc[-1]
    ax.plot([latest["date"]], [latest["TERM_SPREAD_10Y_1Y"]], marker="o", color="#111111", markersize=4)
    ax.set_title(f"Term Spread Monitor ({latest['final_regime_confirmed']})")
    ax.set_ylabel("10Y - 1Y spread")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_spy_trend_signal(panel: pd.DataFrame, regime_start: pd.Timestamp, path: Path) -> None:
    data = panel.loc[panel["date"] >= regime_start].copy()
    fig, ax = plt.subplots(figsize=(11, 5.0))
    shade_stress_periods(ax, data)
    ax.plot(data["date"], data["spy_price"], color="#263238", linewidth=1.9, label="SPY")
    ax.plot(data["date"], data["SPY_MA20"], color="#3d4f8f", linewidth=1.4, label="MA20")
    ax.plot(data["date"], data["SPY_MA50"], color="#2f6f73", linewidth=1.4, label="MA50")
    latest = data.iloc[-1]
    annotate_latest_distance(ax, latest["date"], float(latest["spy_price"]), float(latest["SPY_MA20"]), "to MA20")
    annotate_latest_distance(ax, latest["date"], float(latest["spy_price"]), float(latest["SPY_MA50"]), "to MA50")
    ax.set_title("SPY Trend Monitor")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_nav(regime_perf: pd.DataFrame, path: Path) -> None:
    data = regime_perf.copy()
    data["date"] = pd.to_datetime(data["date"])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(data["date"], data["strategy_nav"], label="Final strategy", color="#2f6f73", linewidth=2)
    ax.plot(data["date"], data["spy_nav"], label="SPY", color="#3d4f8f", linewidth=2)
    ax.set_title("Regime-to-Date NAV vs SPY")
    ax.set_ylabel("NAV, normalized to 1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_drawdown(regime_perf: pd.DataFrame, path: Path) -> None:
    data = regime_perf.copy()
    data["date"] = pd.to_datetime(data["date"])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(data["date"], data["strategy_drawdown"], label="Final strategy", color="#2f6f73", linewidth=2)
    ax.plot(data["date"], data["spy_drawdown"], label="SPY", color="#3d4f8f", linewidth=2)
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_title("Regime-to-Date Drawdown vs SPY")
    ax.set_ylabel("Drawdown")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def shade_regimes(ax: plt.Axes, panel: pd.DataFrame) -> None:
    colors = {
        "INVERTED": "#9d77b9",
        "FLAT_LOW_RATE": "#7fa9a0",
        "FLAT_MID_RATE": "#8dbd8a",
        "FLAT_HIGH_RATE": "#c9a65a",
        "STEEP_LOW_RATE": "#86a8c9",
        "STEEP_MID_RATE": "#5e8cb8",
        "STEEP_HIGH_RATE": "#bf7d62",
    }
    data = panel[["date", "final_regime_confirmed"]].reset_index(drop=True)
    breaks = data["final_regime_confirmed"].ne(data["final_regime_confirmed"].shift()).cumsum()
    for _, grp in data.groupby(breaks):
        regime = str(grp["final_regime_confirmed"].iloc[0])
        ax.axvspan(grp["date"].iloc[0], grp["date"].iloc[-1], color=colors.get(regime, "#dddddd"), alpha=0.14)


def plot_timeline(panel: pd.DataFrame, regime_start: pd.Timestamp, path: Path) -> None:
    data = panel.loc[panel["date"] >= regime_start - pd.Timedelta(days=180)].copy()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    shade_regimes(ax, data)
    ax.plot(data["date"], data["spy_price"] / data["spy_price"].iloc[0], color="#263238", linewidth=1.8, label="SPY price index")
    stress = data["trigger_lock_full_risk_state"].eq("FULL_RISK")
    if stress.any():
        starts = stress.ne(stress.shift()).cumsum()
        for _, grp in data.loc[stress].groupby(starts[stress]):
            ax.axvspan(grp["date"].iloc[0], grp["date"].iloc[-1], color="#b4493f", alpha=0.22)
    ax.axvline(data["date"].iloc[-1], color="#111111", linestyle="--", linewidth=1)
    ax.set_title("Current Regime Timeline")
    ax.set_ylabel("SPY price index")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_summary(status: dict[str, Any], path: Path) -> None:
    alloc = status["current_allocation"]
    metrics = status["regime_to_date"]
    lines = [
        ("Latest date", status["latest_date"]),
        ("Regime", status["current_regime"]),
        ("Stress", status["stress_state"]),
        ("Active locks", ", ".join(status["active_locks"]) or "None"),
        ("Strategy RTD", pct(metrics["strategy_return"])),
        ("SPY RTD", pct(metrics["spy_return"])),
        ("Excess", pct(metrics["excess_return"])),
        ("Largest weight", max(alloc, key=alloc.get)),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axis("off")
    y = 0.9
    ax.text(0.02, y, "Live Regime Dashboard Summary", fontsize=18, weight="bold")
    y -= 0.12
    for label, value in lines:
        ax.text(0.04, y, label, fontsize=11, color="#555555")
        ax.text(0.36, y, str(value), fontsize=12, color="#111111", weight="bold")
        y -= 0.09
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_figures(
    figures_dir: Path,
    panel: pd.DataFrame,
    regime_start: pd.Timestamp,
    allocation: pd.DataFrame,
    signal: pd.DataFrame,
    regime_perf: pd.DataFrame,
    status: dict[str, Any],
    dynamic_weights: pd.DataFrame,
    rebalances: pd.DataFrame,
) -> None:
    plot_allocation(allocation, figures_dir / "current_allocation_bar.png", status["stress_state"])
    plot_dynamic_weights_rebalance_bar(dynamic_weights, rebalances, figures_dir / "current_dynamic_weights_rebalance_bar.png")
    plot_oil_signal(panel, regime_start, figures_dir / "current_signal_oil_level.png")
    plot_term_spread_signal(panel, regime_start, figures_dir / "current_signal_term_spread.png")
    plot_vix_signal(panel, regime_start, figures_dir / "current_signal_vix_zscore.png")
    plot_credit_15d_signal(panel, regime_start, figures_dir / "current_signal_credit_15d_change.png")
    plot_credit_level_signal(panel, regime_start, figures_dir / "current_signal_credit_level_z.png")
    plot_spy_trend_signal(panel, regime_start, figures_dir / "current_signal_spy_trend.png")
    plot_nav(regime_perf, figures_dir / "current_regime_nav_vs_spy.png")
    plot_drawdown(regime_perf, figures_dir / "current_regime_drawdown_vs_spy.png")
    plot_timeline(panel, regime_start, figures_dir / "current_regime_timeline.png")
    plot_summary(status, figures_dir / "current_dashboard_summary.png")


def html_format_value(col: str, value: Any) -> str:
    if pd.isna(value):
        return ""
    col_l = col.lower()
    if isinstance(value, (np.bool_, bool)):
        return "True" if bool(value) else "False"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if any(token in col_l for token in ["weight", "return", "drawdown", "maxdd", "cash_return", "spy_return"]):
            sign = "+" if "return" in col_l and v > 0 else ""
            return f"{sign}{v:.2%}"
        if "sharpe" in col_l or "z" in col_l:
            return f"{v:.2f}"
        if "rate" in col_l or "spread" in col_l or "threshold" in col_l or "distance" in col_l or "value" in col_l:
            return f"{v:.2f}"
        return f"{v:.4f}"
    return html.escape(str(value))


def table_html(df: pd.DataFrame, max_rows: int = 30, columns: list[str] | None = None) -> str:
    if df.empty:
        return "<p class='muted'>No stress lock has been triggered in the current regime.</p>"
    show = df.head(max_rows).copy()
    if columns is not None:
        show = show[[c for c in columns if c in show.columns]]
    for col in show.columns:
        show[col] = show[col].map(lambda x, c=col: html_format_value(c, x))
    return show.to_html(index=False, classes="data-table", border=0, escape=False)


def interpretation_bullets(status: dict[str, Any], signal: pd.DataFrame) -> list[str]:
    regime = status["current_regime"]
    stress = status["stress_state"]
    metrics = status["regime_to_date"]
    alloc = status["current_allocation"]
    largest_asset = max(alloc, key=alloc.get)
    interp = REGIME_INTERPRETATION.get(regime, {})
    signal_work = signal.loc[signal["distance"].notna()].copy()
    signal_work["abs_distance"] = signal_work["distance"].abs()
    nearest = signal_work.sort_values("abs_distance").iloc[0] if not signal_work.empty else None
    active_locks = ", ".join(status["active_locks"]) or "none"
    beat = metrics["strategy_return"] > metrics["spy_return"]
    ts = status["term_spread_thresholds"]
    ts_text = (
        f"flat->inverted {ts['flat_to_inverted']:.2f}, inverted->flat {ts['inverted_to_flat']:.2f}, "
        f"flat->steep {ts['flat_to_steep']:.2f}, steep->flat {ts['steep_to_flat']:.2f}"
    )
    return [
        f"{regime}: {interp.get('economic_meaning', 'No predefined interpretation is available for this regime.')}",
        f"Stress state is {stress}; active locks are {active_locks}.",
        f"Oil state is {status['oil_level_regime']}; oil price is {status['oil_price_to_ma']:+.1%} versus its 252-day average.",
        f"Outer macro regime uses term-spread hysteresis with thresholds {ts_text}.",
        f"The largest sleeve is {largest_asset}; the displayed target weights come directly from the final allocation rule.",
        "The closest monitored signal is "
        + (f"{nearest['signal']} ({nearest['threshold_name']}), currently {nearest['status']}." if nearest is not None else "not available."),
        f"Regime-to-date, the strategy has {'outperformed' if beat else 'lagged'} SPY.",
    ]


def regime_interpretation_html(regime: str) -> str:
    info = REGIME_INTERPRETATION.get(regime)
    if not info:
        return "<p class='muted'>No predefined interpretation is available for this regime.</p>"
    labels = [
        ("Economic Meaning", "economic_meaning"),
        ("Typical Risk", "typical_risk"),
        ("Preferred Assets", "preferred_assets"),
        ("Stress Behavior", "stress_behavior"),
        ("Monitoring Focus", "monitoring_focus"),
    ]
    return "<div class='interpretation-grid'>" + "".join(
        f"<div class='info-row'><div class='info-label'>{label}</div><div>{html.escape(info[key])}</div></div>"
        for label, key in labels
    ) + "</div>"


def all_regime_interpretations_html() -> str:
    blocks = []
    for regime, info in REGIME_INTERPRETATION.items():
        items = "".join(
            f"<li><strong>{html.escape(key.replace('_', ' ').title())}:</strong> {html.escape(value)}</li>"
            for key, value in info.items()
        )
        blocks.append(f"<details><summary>{html.escape(regime)}</summary><ul>{items}</ul></details>")
    return "\n".join(blocks)


def allocation_explanation(status: dict[str, Any], allocation: pd.DataFrame) -> str:
    state = allocation["state"].iloc[0] if not allocation.empty else status["stress_state"]
    active = allocation.loc[allocation["weight"].abs() > 1e-8, "asset"].tolist()
    active_text = " / ".join(active) if active else "no active asset"
    method = allocation["allocation_method"].mode().iloc[0] if not allocation.empty else ""
    parts = [
        f"The current state is {state}. Under this state, the final strategy target allocation is {active_text}.",
        "The current weights are generated by the final allocation rule and held until the next scheduled or state-driven rebalance.",
    ]
    if "inverse_vol" in method:
        parts.append("Weights are based on realized volatility over the final strategy lookback window.")
    if status["stress_state"] == "FULL_RISK":
        parts.append("The portfolio is currently defensive because a stress lock is active.")
    if status["oil_level_regime"] == "OIL_LEVEL_HIGH":
        parts.append("Because oil is HIGH, the flat sleeve removes GOLD and DBB where applicable.")
    return " ".join(parts)


def write_html(
    output_dir: Path,
    status: dict[str, Any],
    allocation: pd.DataFrame,
    dynamic_weights: pd.DataFrame,
    rebalances: pd.DataFrame,
    signal: pd.DataFrame,
    regime_perf: pd.DataFrame,
    episodes: pd.DataFrame,
    fred_dates: dict[str, str],
    asset_dates: dict[str, str],
) -> Path:
    metrics = status["regime_to_date"]
    allocation_summary = ", ".join(f"{k} {pct(v)}" for k, v in status["current_allocation"].items() if abs(v) > 1e-8) or "No active weight"
    cards = [
        ("Latest data date", status["latest_date"]),
        ("Macro regime", status["current_regime"]),
        ("Stress state", status["stress_state"]),
        ("Oil level", status["oil_level_regime"]),
        ("Active locks", ", ".join(status["active_locks"]) or "None"),
        ("Regime start", status["regime_start_date"]),
        ("Days in regime", str(status["days_in_regime"])),
        ("Allocation", allocation_summary),
        ("Strategy vs SPY RTD", f"{pct(metrics['strategy_return'])} vs {pct(metrics['spy_return'])}"),
        ("Current DD vs SPY", f"{pct(metrics['strategy_current_drawdown'])} vs {pct(metrics['spy_current_drawdown'])}"),
    ]
    bullets = interpretation_bullets(status, signal)
    warnings_section = (
        "<section><h2>Warnings</h2><ul>"
        + "".join(f"<li>{html.escape(w)}</li>" for w in status["warnings"])
        + "</ul></section>"
        if status["warnings"]
        else ""
    )
    card_html = "".join(f"<div class='card'><div class='card-label'>{html.escape(k)}</div><div class='card-value'>{html.escape(v)}</div></div>" for k, v in cards)
    allocation_note = allocation_explanation(status, allocation)
    signal_cols = ["category", "signal", "current_value", "threshold_value", "distance", "status", "interpretation"]
    rebalance_recent = rebalances.tail(10).iloc[::-1].reset_index(drop=True)
    content = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Live Regime Dashboard</title>
<style>
body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; color: #1f2933; background: #f7f8fa; }}
header {{ background: #1f2933; color: white; padding: 28px 36px; }}
header h1 {{ margin: 0 0 6px 0; font-size: 30px; letter-spacing: 0; }}
header p {{ margin: 0; color: #d6dde5; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 28px 24px 42px; }}
section {{ margin: 0 0 32px; }}
h2 {{ font-size: 20px; margin: 0 0 14px; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }}
.card {{ background: white; border: 1px solid #d9dee5; border-radius: 8px; padding: 14px; min-height: 72px; }}
.card-label {{ color: #64717f; font-size: 12px; text-transform: uppercase; }}
.card-value {{ font-size: 18px; font-weight: 700; margin-top: 8px; overflow-wrap: anywhere; }}
.plot {{ width: 100%; max-width: 100%; background: white; border: 1px solid #d9dee5; border-radius: 8px; margin: 8px 0 16px; }}
.grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }}
.signal-grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
.signal-grid .plot {{ margin-bottom: 0; }}
.data-table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d9dee5; font-size: 13px; }}
.data-table th, .data-table td {{ border-bottom: 1px solid #e5e9ef; padding: 8px 10px; text-align: left; vertical-align: top; }}
.data-table th {{ background: #eef2f5; color: #374151; }}
.muted {{ color: #64717f; }}
.note {{ background: white; border: 1px solid #d9dee5; border-radius: 8px; padding: 12px 14px; color: #374151; }}
.interpretation-grid {{ background: white; border: 1px solid #d9dee5; border-radius: 8px; padding: 14px; }}
.info-row {{ display: grid; grid-template-columns: 180px 1fr; gap: 12px; border-bottom: 1px solid #e5e9ef; padding: 10px 0; }}
.info-row:last-child {{ border-bottom: 0; }}
.info-label {{ font-weight: 700; color: #374151; }}
details {{ background: white; border: 1px solid #d9dee5; border-radius: 8px; padding: 12px 14px; margin: 8px 0; }}
summary {{ cursor: pointer; font-weight: 700; }}
footer {{ color: #64717f; border-top: 1px solid #d9dee5; padding-top: 18px; font-size: 13px; }}
ul {{ background: white; border: 1px solid #d9dee5; border-radius: 8px; padding: 14px 18px 14px 34px; }}
</style>
</head>
<body>
<header>
<h1>Live Regime Dashboard</h1>
<p>Current macro regime, stress monitor, and regime-to-date strategy performance.</p>
</header>
<main>
<section>
<h2>Current State</h2>
<div class="cards">{card_html}</div>
</section>
<section>
<h2>Current Regime Interpretation</h2>
{regime_interpretation_html(status["current_regime"])}
<ul>{"".join(f"<li>{html.escape(item)}</li>" for item in bullets)}</ul>
</section>
<section>
<h2>Current Allocation</h2>
<img class="plot" src="figures/current_allocation_bar.png" alt="Current allocation bar chart">
{table_html(allocation)}
<p class="note">{html.escape(allocation_note)}</p>
</section>
<section>
<h2>Dynamic Weights Since Current Regime Start</h2>
<p class="muted">This chart shows how target weights evolved during the current regime. Vertical markers indicate rebalance dates.</p>
<img class="plot" src="figures/current_dynamic_weights_rebalance_bar.png" alt="Target weights at rebalance dates">
{table_html(rebalance_recent, max_rows=10)}
</section>
<section>
<h2>Signal Distance</h2>
<div class="signal-grid">
<img class="plot" src="figures/current_signal_term_spread.png" alt="Term spread monitor">
<img class="plot" src="figures/current_signal_oil_level.png" alt="Oil level monitor">
<img class="plot" src="figures/current_signal_vix_zscore.png" alt="VIX Z-score monitor">
<img class="plot" src="figures/current_signal_credit_15d_change.png" alt="Credit 15D change monitor">
<img class="plot" src="figures/current_signal_credit_level_z.png" alt="Credit level Z monitor">
<img class="plot" src="figures/current_signal_spy_trend.png" alt="SPY trend monitor">
</div>
{table_html(signal, columns=signal_cols)}
</section>
<section>
<h2>Regime-to-Date Performance</h2>
<div class="grid-2">
<img class="plot" src="figures/current_regime_nav_vs_spy.png" alt="Regime-to-date NAV vs SPY">
<img class="plot" src="figures/current_regime_drawdown_vs_spy.png" alt="Regime-to-date drawdown vs SPY">
</div>
{table_html(regime_perf.tail(12), max_rows=12)}
</section>
<section>
<h2>Regime Timeline</h2>
<img class="plot" src="figures/current_regime_timeline.png" alt="Current regime timeline">
</section>
<section>
<h2>Stress Episodes</h2>
{table_html(episodes)}
</section>
{warnings_section}
<section>
<h2>All Regime Interpretations</h2>
{all_regime_interpretations_html()}
</section>
<footer>This dashboard is generated from historical and latest available data. It is research output, not investment advice.</footer>
</main>
</body>
</html>
"""
    path = output_dir / "live_regime_dashboard.html"
    path.write_text(content, encoding="utf-8")
    return path


def build_status(
    panel: pd.DataFrame,
    latest_idx: int,
    regime_start: pd.Timestamp,
    stress_start: pd.Timestamp | None,
    allocation: pd.DataFrame,
    signal: pd.DataFrame,
    perf_summary: dict[str, float],
    warnings: list[str],
    fred_dates: dict[str, str],
    asset_dates: dict[str, str],
) -> dict[str, Any]:
    row = panel.loc[latest_idx]
    active_locks = [x for x in str(row["trigger_lock_active_locks"]).split("+") if x and x != "nan"]
    status = {
        "latest_date": row["date"].strftime("%Y-%m-%d"),
        "current_regime": str(row["final_regime_confirmed"]),
        "stress_state": str(row["trigger_lock_full_risk_state"]),
        "oil_level_regime": str(row["oil_level_regime"]),
        "oil_price_to_ma": float(row["oil_price_to_ma"]) if pd.notna(row["oil_price_to_ma"]) else np.nan,
        "active_locks": active_locks,
        "active_trigger_reason": "+".join(active_locks) if active_locks else "",
        "regime_start_date": regime_start.strftime("%Y-%m-%d"),
        "days_in_regime": int((row["date"] - regime_start).days),
        "current_stress_start_date": None if stress_start is None else stress_start.strftime("%Y-%m-%d"),
        "days_in_current_stress": None if stress_start is None else int((row["date"] - stress_start).days),
        "current_allocation": {r["asset"]: float(r["weight"]) for _, r in allocation.iterrows()},
        "regime_to_date": perf_summary,
        "term_spread_thresholds": {
            "flat_to_inverted": float(getattr(final_core, "OUTER_FLAT_TO_INV", -0.10)) if final_core else -0.10,
            "inverted_to_flat": float(getattr(final_core, "OUTER_INV_TO_FLAT", 0.10)) if final_core else 0.10,
            "flat_to_steep": float(getattr(final_core, "OUTER_FLAT_TO_STEEP", 1.20)) if final_core else 1.20,
            "steep_to_flat": float(getattr(final_core, "OUTER_STEEP_TO_FLAT", 1.00)) if final_core else 1.00,
        },
        "signals": {
            "vix_zscore": signal.loc[signal["category"].eq("VIX")].to_dict(orient="records"),
            "credit": signal.loc[signal["category"].eq("Credit")].to_dict(orient="records"),
            "spy_trend": signal.loc[signal["category"].eq("SPY trend")].to_dict(orient="records"),
            "regime_variables": signal.loc[signal["category"].eq("Regime variables")].to_dict(orient="records"),
        },
        "data_freshness": {
            "fred": fred_dates,
            "assets": asset_dates,
        },
        "warnings": warnings,
    }
    return status


def terminal_summary(status: dict[str, Any], rebalances: pd.DataFrame, output_path: Path) -> None:
    latest_rebalance = "n/a" if rebalances.empty else str(rebalances["date"].iloc[-1])
    print(f"latest date: {status['latest_date']}")
    print(f"current regime: {status['current_regime']}")
    print(f"stress state: {status['stress_state']}")
    print(f"current allocation: {status['current_allocation']}")
    print(f"latest rebalance date: {latest_rebalance}")
    print(f"number of rebalances in current regime: {len(rebalances)}")
    print(f"strategy regime-to-date return: {status['regime_to_date']['strategy_return']:.2%}")
    print(f"SPY regime-to-date return: {status['regime_to_date']['spy_return']:.2%}")
    print(f"output path: {output_path.relative_to(ROOT).as_posix()}")
    print(f"warnings count: {len(status['warnings'])}")


def main() -> None:
    args = parse_args()
    output_dir, figures_dir = ensure_dirs(args.output_dir)
    warnings: list[str] = []
    if FINAL_IMPORT_WARNING:
        warnings.append(FINAL_IMPORT_WARNING)

    macro, prices, fetch_warnings = load_data(refresh=args.refresh, use_cache=not args.no_fetch_cache)
    warnings.extend(fetch_warnings)
    fred_dates, asset_dates = latest_series_dates(macro, prices)

    panel = build_live_source_panel(macro, prices, warnings)
    panel = compute_live_strategy(panel, warnings)
    latest_common = add_freshness_warnings(panel, fred_dates, asset_dates, warnings)
    panel = panel.loc[panel["date"] <= latest_common].reset_index(drop=True)
    latest_idx = int(panel.index[-1])

    regime_start = contiguous_start(panel, "final_regime_confirmed", latest_idx)
    if regime_start == panel["date"].iloc[0]:
        warnings.append("current regime start may be truncated by available data.")
    stress_start = current_stress_start(panel, latest_idx)

    allocation = build_current_allocation(panel, latest_idx)
    signal = build_signal_distance(panel.loc[latest_idx])
    regime_perf, perf_summary = slice_regime_performance(panel, regime_start)
    episodes = stress_episodes(panel, regime_start)
    dynamic_weights, rebalances = build_dynamic_weights(panel, regime_start)
    status = build_status(
        panel,
        latest_idx,
        regime_start,
        stress_start,
        allocation,
        signal,
        perf_summary,
        warnings,
        fred_dates,
        asset_dates,
    )

    write_csv_outputs(output_dir, status, signal, allocation, regime_perf, episodes, dynamic_weights, rebalances)
    write_figures(figures_dir, panel, regime_start, allocation, signal, regime_perf, status, dynamic_weights, rebalances)
    html_path = write_html(output_dir, status, allocation, dynamic_weights, rebalances, signal, regime_perf, episodes, fred_dates, asset_dates)
    terminal_summary(status, rebalances, html_path)


if __name__ == "__main__":
    main()
