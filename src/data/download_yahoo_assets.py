from __future__ import annotations

from datetime import date
from pathlib import Path
import os
import sys
import site
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / ".vendor"
YF_VENDOR = ROOT / ".vendor_yf"
if str(VENDOR) not in sys.path:
    sys.path.append(str(VENDOR))


CONFIG_PATH = ROOT / "config" / "asset_universe_all_weather.csv"
RAW_DIR = ROOT / "data" / "raw" / "assets"
PROCESSED_DIR = ROOT / "data" / "processed" / "assets"
RESULTS_DIR = ROOT / "results" / "asset_universe"

RAW_PRICES_PATH = RAW_DIR / "yahoo_daily_adjusted_close.csv"
DAILY_PRICES_PATH = PROCESSED_DIR / "daily_adjusted_close.csv"
DAILY_RETURNS_PATH = PROCESSED_DIR / "daily_returns.csv"
MONTHLY_PRICES_PATH = PROCESSED_DIR / "monthly_adjusted_close.csv"
MONTHLY_RETURNS_PATH = PROCESSED_DIR / "monthly_returns.csv"
AVAILABILITY_PATH = RESULTS_DIR / "asset_data_availability.csv"
DOWNLOAD_LOG_PATH = RESULTS_DIR / "yahoo_download_log.csv"
MARKDOWN_PATH = RESULTS_DIR / "ASSET_UNIVERSE_DATA.md"

START_DATE = "1990-01-01"
END_DATE = date.today().isoformat()
FINAL_REQUIRED_TICKERS = ["SPY", "GLD", "GD=F", "IEF"]


def ensure_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def set_proxy() -> None:
    proxy = "http://127.0.0.1:7890"
    os.environ["HTTP_PROXY"] = proxy
    os.environ["HTTPS_PROXY"] = proxy


def import_yfinance():
    user_site = site.getusersitepackages()
    candidate_paths = [str(YF_VENDOR), user_site, str(VENDOR)]
    import importlib
    original_sys_path = list(sys.path)
    attempted: list[str] = []

    for path in candidate_paths:
        if not path or not os.path.exists(path):
            continue
        attempted.append(path)
        try:
            sys.modules.pop("yfinance", None)
            cleaned = [p for p in original_sys_path if p not in candidate_paths]
            sys.path = [path] + cleaned
            yf = importlib.import_module("yfinance")
            if hasattr(yf, "download") and hasattr(yf, "Ticker"):
                return yf
        except Exception:
            continue
        finally:
            sys.path = list(original_sys_path)

    raise RuntimeError(
        "Unable to import a valid yfinance package with download() and Ticker() interfaces. "
        f"Tried paths: {attempted}"
    )


def load_asset_universe(path: Path = CONFIG_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["ticker", "asset_class", "role", "description", "priority", "notes"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Asset universe file is missing columns: {missing}")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    return df


def extract_adjusted_close(downloaded: pd.DataFrame, ticker: str) -> pd.Series:
    if downloaded.empty:
        raise ValueError("Downloaded frame is empty.")

    if isinstance(downloaded.columns, pd.MultiIndex):
        if ticker not in downloaded.columns.get_level_values(0):
            raise KeyError(f"{ticker} not found in downloaded columns.")
        ticker_frame = downloaded[ticker].copy()
    else:
        ticker_frame = downloaded.copy()

    candidate_cols = [col for col in ["Adj Close", "Close"] if col in ticker_frame.columns]
    if not candidate_cols:
        raise KeyError(f"No adjusted-close-like column found for {ticker}.")

    series = ticker_frame[candidate_cols[0]].copy()
    series.name = ticker
    series.index = pd.to_datetime(series.index)
    series = pd.to_numeric(series, errors="coerce")
    return series


def download_yahoo_adjusted_close(
    tickers: Iterable[str],
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    yf = import_yfinance()

    price_series: list[pd.Series] = []
    log_rows: list[dict[str, object]] = []

    for ticker in tickers:
        try:
            downloaded = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=False,
            )
            series = extract_adjusted_close(downloaded, ticker)
            if series.dropna().empty:
                raise ValueError("No non-null price history returned.")
            price_series.append(series)
            first_valid = series.first_valid_index()
            last_valid = series.last_valid_index()
            log_rows.append(
                {
                    "ticker": ticker,
                    "download_status": "success",
                    "error_message": "",
                    "first_valid_date": first_valid.strftime("%Y-%m-%d") if first_valid is not None else "",
                    "last_valid_date": last_valid.strftime("%Y-%m-%d") if last_valid is not None else "",
                    "daily_obs": int(series.count()),
                }
            )
        except Exception as exc:
            log_rows.append(
                {
                    "ticker": ticker,
                    "download_status": "failed",
                    "error_message": str(exc),
                    "first_valid_date": "",
                    "last_valid_date": "",
                    "daily_obs": 0,
                }
            )

    if price_series:
        daily_prices = pd.concat(price_series, axis=1).sort_index()
        daily_prices.index.name = "date"
    else:
        daily_prices = pd.DataFrame()
        daily_prices.index.name = "date"

    download_log = pd.DataFrame(log_rows)
    return daily_prices, download_log


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change(fill_method=None)
    returns.index.name = prices.index.name
    return returns


def resample_monthly_last(prices: pd.DataFrame) -> pd.DataFrame:
    monthly = prices.resample("M").last()
    monthly.index.name = "date"
    return monthly


def ratio_missing_after_first_valid(series: pd.Series) -> float:
    first_valid = series.first_valid_index()
    if first_valid is None:
        return np.nan
    active = series.loc[first_valid:]
    if active.empty:
        return np.nan
    return float(active.isna().mean())


def create_availability_report(
    universe: pd.DataFrame,
    daily_prices: pd.DataFrame,
    monthly_prices: pd.DataFrame,
    download_log: pd.DataFrame,
) -> pd.DataFrame:
    log_lookup = download_log.set_index("ticker")
    rows: list[dict[str, object]] = []

    for _, asset in universe.iterrows():
        ticker = asset["ticker"]
        daily_series = daily_prices[ticker] if ticker in daily_prices.columns else pd.Series(dtype=float)
        monthly_series = monthly_prices[ticker] if ticker in monthly_prices.columns else pd.Series(dtype=float)
        log_row = log_lookup.loc[ticker] if ticker in log_lookup.index else None
        first_valid = daily_series.first_valid_index() if not daily_series.empty else None
        last_valid = daily_series.last_valid_index() if not daily_series.empty else None

        rows.append(
            {
                "ticker": ticker,
                "asset_class": asset["asset_class"],
                "role": asset["role"],
                "first_valid_date": first_valid.strftime("%Y-%m-%d") if first_valid is not None else "",
                "last_valid_date": last_valid.strftime("%Y-%m-%d") if last_valid is not None else "",
                "daily_obs": int(daily_series.count()) if not daily_series.empty else 0,
                "monthly_obs": int(monthly_series.count()) if not monthly_series.empty else 0,
                "missing_daily_ratio": ratio_missing_after_first_valid(daily_series) if not daily_series.empty else np.nan,
                "missing_monthly_ratio": ratio_missing_after_first_valid(monthly_series) if not monthly_series.empty else np.nan,
                "download_status": str(log_row["download_status"]) if log_row is not None else "failed",
                "error_message": str(log_row["error_message"]) if log_row is not None else "Missing download log entry",
            }
        )

    return pd.DataFrame(rows)


def write_markdown_note() -> None:
    lines = [
        "# Asset Universe Data",
        "",
        "## Why this universe is grouped by asset class",
        "",
        "- The universe is organized by broad economic role so later selection logic can compare substitutes within equity, duration, inflation hedges, credit, commodities, precious metals, real assets, currencies, and alternatives.",
        "- This keeps the future regime-aware all-weather module focused on cross-regime behavior instead of treating every ETF as an unstructured ticker list.",
        "",
        "## Why daily adjusted close is used",
        "",
        "- Daily adjusted close is enough for the current research stage because it captures split and distribution-adjusted total-return price history while preserving maximum flexibility for later resampling.",
        "- We will aggregate to monthly when building regime-conditioned allocation and exposure studies, but the daily panel is the cleanest canonical raw market dataset.",
        "",
        "## Why ETFs have different start dates",
        "",
        "- Many ETFs launched well after 1990, so the combined panel is intentionally sparse in early history.",
        "- Missing values are kept as `NaN` rather than filled forward because unavailable pre-launch history should not be fabricated.",
        "",
        "## How this data will be used later",
        "",
        "- The downloaded universe will support all-weather factor exposure analysis, asset substitution within sleeves, and regime-conditioned asset selection.",
        "- Availability diagnostics also help identify which assets are robust enough for long-history backtests and which are better suited to recent-period extensions.",
        "",
        "## How to rerun",
        "",
        "```powershell",
        "python src\\data\\download_yahoo_assets.py",
        "```",
    ]
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")


def print_validation(availability: pd.DataFrame, daily_prices: pd.DataFrame) -> None:
    success_count = int((availability["download_status"] == "success").sum())
    failed_count = int((availability["download_status"] != "success").sum())
    print(f"Successful tickers: {success_count}")
    print(f"Failed tickers: {failed_count}")

    if not daily_prices.empty:
        print(f"Combined panel first date: {daily_prices.index.min().strftime('%Y-%m-%d')}")
        print(f"Combined panel last date: {daily_prices.index.max().strftime('%Y-%m-%d')}")
    else:
        print("Combined panel first date: n/a")
        print("Combined panel last date: n/a")

    low_history = availability.loc[availability["monthly_obs"] < 60, "ticker"].tolist()
    print("Tickers with fewer than 60 monthly observations:", ", ".join(low_history) if low_history else "None")

    post_launch_missing = availability.loc[
        availability["missing_daily_ratio"].fillna(0) > 0,
        ["ticker", "missing_daily_ratio"],
    ]
    if post_launch_missing.empty:
        print("Tickers with missing data after first valid date: None")
    else:
        joined = ", ".join(
            f"{row.ticker} ({row.missing_daily_ratio:.3f})" for row in post_launch_missing.itertuples(index=False)
        )
        print(f"Tickers with missing data after first valid date: {joined}")


def save_panel(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    out.to_csv(path)


def resolve_requested_tickers(universe: pd.DataFrame) -> list[str]:
    requested = os.environ.get("YAHOO_TICKERS", "").strip()
    if not requested:
        allowed = set(universe["ticker"].tolist())
        missing = [ticker for ticker in FINAL_REQUIRED_TICKERS if ticker not in allowed]
        if missing:
            raise ValueError(f"Final required tickers are missing from asset universe config: {missing}")
        return FINAL_REQUIRED_TICKERS
    requested_list = [ticker.strip().upper() for ticker in requested.split(",") if ticker.strip()]
    allowed = set(universe["ticker"].tolist())
    unknown = [ticker for ticker in requested_list if ticker not in allowed]
    if unknown:
        raise ValueError(f"Requested tickers are not present in asset universe: {unknown}")
    return requested_list


def main() -> None:
    ensure_dirs()
    set_proxy()
    universe = load_asset_universe()
    tickers = resolve_requested_tickers(universe)
    daily_prices, download_log = download_yahoo_adjusted_close(tickers)
    daily_returns = compute_returns(daily_prices) if not daily_prices.empty else pd.DataFrame()
    monthly_prices = resample_monthly_last(daily_prices) if not daily_prices.empty else pd.DataFrame()
    monthly_returns = compute_returns(monthly_prices) if not monthly_prices.empty else pd.DataFrame()

    filtered_universe = universe.loc[universe["ticker"].isin(tickers)].copy()
    availability = create_availability_report(filtered_universe, daily_prices, monthly_prices, download_log)

    success_count = int((download_log["download_status"] == "success").sum())
    if success_count == 0:
        download_log.to_csv(DOWNLOAD_LOG_PATH, index=False)
        availability.to_csv(AVAILABILITY_PATH, index=False)
        raise RuntimeError(
            "Yahoo download returned zero successful tickers. Existing processed asset files were left unchanged."
        )

    save_panel(daily_prices, RAW_PRICES_PATH)
    save_panel(daily_prices, DAILY_PRICES_PATH)
    save_panel(daily_returns, DAILY_RETURNS_PATH)
    save_panel(monthly_prices, MONTHLY_PRICES_PATH)
    save_panel(monthly_returns, MONTHLY_RETURNS_PATH)
    availability.to_csv(AVAILABILITY_PATH, index=False)
    download_log.to_csv(DOWNLOAD_LOG_PATH, index=False)
    write_markdown_note()

    print_validation(availability, daily_prices)
    for path in [
        CONFIG_PATH,
        RAW_PRICES_PATH,
        DAILY_PRICES_PATH,
        DAILY_RETURNS_PATH,
        MONTHLY_PRICES_PATH,
        MONTHLY_RETURNS_PATH,
        AVAILABILITY_PATH,
        DOWNLOAD_LOG_PATH,
        MARKDOWN_PATH,
    ]:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
