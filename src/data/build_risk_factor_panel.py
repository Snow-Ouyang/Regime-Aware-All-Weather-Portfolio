from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from regime.utils import REGIME_ORDER


REGIME_INPUT_PATH = ROOT / "data" / "processed" / "regime_inputs_simplified.csv"
REGIME_LABELS_PATH = ROOT / "results" / "regime" / "simplified_regime_labels.csv"

AQR_DIR = ROOT / "data" / "processed" / "AQR"
AQR_MKT_PATH = AQR_DIR / "mkt.csv"
AQR_RF_PATH = AQR_DIR / "rf.csv"
AQR_DURATION_PATH = AQR_DIR / "duration.csv"
AQR_COMMODITIES_PATH = AQR_DIR / "commodities.csv"
AQR_BAB_PATH = AQR_DIR / "BAB.csv"
AQR_TSMOM_PATH = AQR_DIR / "TSMOM.csv"
AQR_QMJ_PATH = AQR_DIR / "QMJ.csv"
FAMA_FRENCH_PATH = ROOT / "data" / "raw" / "fama french factor.csv"

GOLD_PATH = ROOT / "data" / "raw" / "gold.csv"
VIX_PATH = ROOT / "data" / "raw" / "macro" / "volatility" / "VIXCLS.csv"
TED_PATH = ROOT / "data" / "raw" / "macro" / "volatility" / "TEDRATE.csv"
DOLLAR_DIR = ROOT / "data" / "raw" / "macro" / "dollar"

PROCESSED_DIR = ROOT / "data" / "processed" / "risk_factors"
RESULTS_DIR = ROOT / "results" / "risk_factors"
FIGURES_DIR = ROOT / "figures" / "risk_factors"

CORE_PANEL_PATH = PROCESSED_DIR / "core_risk_factor_panel.csv"
LONG_HISTORY_PANEL_PATH = PROCESSED_DIR / "long_history_risk_factor_panel.csv"
AVAILABILITY_PATH = RESULTS_DIR / "risk_factor_availability.csv"
SUMMARY_PATH = RESULTS_DIR / "risk_factor_summary.csv"
REGIME_COVERAGE_PATH = RESULTS_DIR / "factor_regime_coverage.csv"
CORE_CORR_PATH = RESULTS_DIR / "core_risk_factor_correlation.csv"
STATE_CORR_PATH = RESULTS_DIR / "regime_state_variable_correlation.csv"
MARKDOWN_PATH = RESULTS_DIR / "RISK_FACTOR_PANEL.md"
CORR_HEATMAP_PATH = FIGURES_DIR / "core_risk_factor_correlation_heatmap.png"
STATE_CORR_HEATMAP_PATH = FIGURES_DIR / "regime_state_variable_correlation_heatmap.png"

RETURN_LIKE_COLUMNS = [
    "RF_MONTHLY",
    "MKT_EXCESS",
    "AQR_FI_MARKET_EXCESS",
    "AQR_CMDTY_EW_EXCESS",
    "AQR_CMDTY_CARRY",
    "GOLD_LEVEL",
    "GOLD_RET",
    "GOLD_EXCESS",
    "BAB",
    "AQR_TSMOM",
    "AQR_QMJ",
    "FF_SMB",
    "FF_HML",
    "FF_RMW",
    "FF_CMA",
    "FF_MOM",
    "FF_RF",
]
SHOCK_COLUMNS = [
    "D_GROWTH_PC1",
    "D_INFLATION_PC1",
    "D_GS10",
    "D_TERM_SPREAD_10Y_1Y",
    "D_CREDIT_SPREAD",
    "D_VIX",
    "D_TED_SPREAD",
    "D_DOLLAR",
]
CORE_COLUMNS = [
    "date",
    "regime",
    "regime_name",
    "growth_pc1",
    "inflation_pc1",
    "gs10",
    "term_spread_10y_1y",
    "credit_spread",
    "D_GROWTH_PC1",
    "D_INFLATION_PC1",
    "D_GS10",
    "D_TERM_SPREAD_10Y_1Y",
    "D_CREDIT_SPREAD",
    "D_VIX",
    "D_TED_SPREAD",
    "D_DOLLAR",
    "VIX_LEVEL",
    "TED_LEVEL",
    "VIX_MONTHLY_AVG",
    "VIX_MONTHLY_MAX",
    "TED_MONTHLY_AVG",
    "TED_MONTHLY_MAX",
    "DOLLAR_LEVEL",
    "RF_MONTHLY",
    "MKT_EXCESS",
    "AQR_FI_MARKET_EXCESS",
    "AQR_CMDTY_EW_EXCESS",
    "AQR_CMDTY_CARRY",
    "GOLD_LEVEL",
    "GOLD_RET",
    "GOLD_EXCESS",
    "BAB",
    "AQR_TSMOM",
    "AQR_QMJ",
    "FF_SMB",
    "FF_HML",
    "FF_RMW",
    "FF_CMA",
    "FF_MOM",
    "FF_RF",
]


def ensure_dirs() -> None:
    for path in [PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def to_month_end(series: pd.Series) -> pd.Series:
    parsed = series.map(lambda x: pd.to_datetime(x, errors="coerce"))
    return parsed.dt.to_period("M").dt.to_timestamp("M")


def parse_scale_series(series: pd.Series) -> tuple[pd.Series, str]:
    raw = series.astype(str).str.strip()
    has_percent = raw.str.contains("%", regex=False).any()
    cleaned = raw.str.replace("%", "", regex=False).str.replace(",", "", regex=False)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    non_null = numeric.dropna()

    if has_percent:
        return numeric / 100.0, "Converted percent strings to decimals by dividing by 100."
    if not non_null.empty and float(non_null.abs().median()) > 0.2:
        return numeric / 100.0, "Converted numeric values to decimals because median absolute value exceeded 0.2."
    return numeric, "Left values unchanged because they already appear to be decimals."


def build_metadata_entry(source_file: str, scale_note: str = "") -> dict[str, str]:
    return {"source_file": source_file, "scale_conversion_note": scale_note}


def load_regime_block() -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    regime_inputs = pd.read_csv(REGIME_INPUT_PATH)
    regime_labels = pd.read_csv(REGIME_LABELS_PATH)
    regime_inputs["date"] = to_month_end(regime_inputs["date"])
    regime_labels["date"] = to_month_end(regime_labels["date"])

    out = regime_inputs[["date", "growth_pc1", "inflation_pc1", "gs10", "term_spread_10y_1y", "credit_spread"]].merge(
        regime_labels[["date", "state_raw", "regime_name"]].rename(columns={"state_raw": "regime"}),
        on="date",
        how="left",
    )
    out = out.sort_values("date").reset_index(drop=True)
    out["D_GROWTH_PC1"] = out["growth_pc1"].diff()
    out["D_INFLATION_PC1"] = out["inflation_pc1"].diff()
    out["D_GS10"] = out["gs10"].diff()
    out["D_TERM_SPREAD_10Y_1Y"] = out["term_spread_10y_1y"].diff()
    out["D_CREDIT_SPREAD"] = out["credit_spread"].diff()

    metadata = {
        "regime": build_metadata_entry(REGIME_LABELS_PATH.name),
        "regime_name": build_metadata_entry(REGIME_LABELS_PATH.name),
    }
    for col in ["growth_pc1", "inflation_pc1", "gs10", "term_spread_10y_1y", "credit_spread"]:
        metadata[col] = build_metadata_entry(REGIME_INPUT_PATH.name)
    for col in ["D_GROWTH_PC1", "D_INFLATION_PC1", "D_GS10", "D_TERM_SPREAD_10Y_1Y", "D_CREDIT_SPREAD"]:
        metadata[col] = build_metadata_entry(REGIME_INPUT_PATH.name)
    return out, metadata


def load_daily_level_monthly_stats(path: Path, raw_value_col: str, base_name: str) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    df[raw_value_col] = pd.to_numeric(df[raw_value_col], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df["month_end"] = df["date"].dt.to_period("M").dt.to_timestamp("M")
    monthly = df.groupby("month_end")[raw_value_col].agg(["last", "mean", "max"]).reset_index()
    monthly.columns = ["date", f"{base_name}_LEVEL", f"{base_name}_MONTHLY_AVG", f"{base_name}_MONTHLY_MAX"]
    monthly[f"D_{base_name}"] = monthly[f"{base_name}_LEVEL"].diff()
    metadata = {
        f"{base_name}_LEVEL": build_metadata_entry(path.name),
        f"{base_name}_MONTHLY_AVG": build_metadata_entry(path.name),
        f"{base_name}_MONTHLY_MAX": build_metadata_entry(path.name),
        f"D_{base_name}": build_metadata_entry(path.name),
    }
    return monthly, metadata


def load_dollar_block() -> tuple[pd.DataFrame | None, dict[str, dict[str, str]], str, str | None]:
    csv_files = sorted(DOLLAR_DIR.glob("*.csv"))
    if not csv_files:
        return None, {}, "Dollar data unavailable: no CSV found.", None
    candidate_files = [path.name for path in csv_files]
    for path in csv_files:
        df = pd.read_csv(path)
        date_col = next((col for col in df.columns if "date" in col.lower()), None)
        if date_col is None:
            continue
        value_candidates = [col for col in df.columns if col != date_col]
        ranked = sorted(
            value_candidates,
            key=lambda col: (0 if any(token in col.lower() for token in ["dtwex", "dxy", "dollar", "close", "value"]) else 1, col),
        )
        for value_col in ranked:
            tmp = df.copy()
            tmp["date"] = pd.to_datetime(tmp[date_col], errors="coerce")
            tmp["DOLLAR_LEVEL"] = pd.to_numeric(tmp[value_col], errors="coerce")
            tmp = tmp.dropna(subset=["date", "DOLLAR_LEVEL"]).sort_values("date")
            if tmp.empty:
                continue
            tmp["date"] = tmp["date"].dt.to_period("M").dt.to_timestamp("M")
            tmp = tmp.drop_duplicates("date", keep="last")
            tmp["D_DOLLAR"] = np.log(tmp["DOLLAR_LEVEL"]).diff()
            note = f"Loaded dollar data from {path.name} using column {value_col}. Candidate files: {', '.join(candidate_files)}."
            if len(candidate_files) > 1:
                note = "Warning: multiple dollar CSV candidates found. " + note
            metadata = {
                "DOLLAR_LEVEL": build_metadata_entry(path.name),
                "D_DOLLAR": build_metadata_entry(path.name),
            }
            return tmp[["date", "DOLLAR_LEVEL", "D_DOLLAR"]], metadata, note, path.name
    return None, {}, f"Dollar data unavailable: no valid date/value columns found across {', '.join(candidate_files)}.", None


def load_gold_block() -> tuple[pd.DataFrame | None, dict[str, dict[str, str]], str]:
    if not GOLD_PATH.exists():
        return None, {}, "Gold data unavailable: data/raw/gold.csv not found."
    df = pd.read_csv(GOLD_PATH)
    date_col = next((col for col in df.columns if "date" in col.lower()), None)
    value_col = next((col for col in df.columns if col.lower() in {"usd", "value", "close", "gold"}), None)
    if date_col is None or value_col is None:
        return None, {}, "Gold data unavailable: date/value columns not found."
    df["date"] = to_month_end(df[date_col])
    df["GOLD_LEVEL"] = pd.to_numeric(df[value_col].astype(str).str.replace(",", "", regex=False), errors="coerce")
    df = df.dropna(subset=["date", "GOLD_LEVEL"]).sort_values("date").drop_duplicates("date", keep="last")
    df["GOLD_RET"] = df["GOLD_LEVEL"].pct_change()
    metadata = {
        "GOLD_LEVEL": build_metadata_entry(GOLD_PATH.name),
        "GOLD_RET": build_metadata_entry(GOLD_PATH.name),
    }
    return df[["date", "GOLD_LEVEL", "GOLD_RET"]], metadata, "Loaded gold spot/price history from raw gold.csv."


def load_aqr_single_factor(
    path: Path,
    rename_map: dict[str, str],
) -> tuple[pd.DataFrame | None, dict[str, dict[str, str]], list[str]]:
    if not path.exists():
        return None, {}, [f"{path.name} not found."]
    df = pd.read_csv(path)
    date_col = next((col for col in df.columns if "date" in col.lower()), None)
    if date_col is None:
        return None, {}, [f"{path.name} missing date column."]
    out = pd.DataFrame({"date": to_month_end(df[date_col])})
    metadata: dict[str, dict[str, str]] = {}
    notes: list[str] = []
    for raw_col, target_col in rename_map.items():
        if raw_col not in df.columns:
            notes.append(f"{path.name} missing column {raw_col}.")
            continue
        parsed, scale_note = parse_scale_series(df[raw_col])
        out[target_col] = parsed
        metadata[target_col] = build_metadata_entry(path.name, scale_note)
        notes.append(f"{target_col}: {scale_note}")
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    return out, metadata, notes


def pick_best_column(columns: list[str], preferences: list[str]) -> str | None:
    lowered = {col.lower(): col for col in columns}
    for pref in preferences:
        for col in columns:
            if col.lower() == pref.lower():
                return col
    for pref in preferences:
        for col in columns:
            if pref.lower() in col.lower():
                return col
    return columns[0] if columns else None


def load_fama_french_block() -> tuple[pd.DataFrame | None, dict[str, dict[str, str]], list[str], str]:
    if not FAMA_FRENCH_PATH.exists():
        return None, {}, [f"{FAMA_FRENCH_PATH.name} not found."], ""
    df = pd.read_csv(FAMA_FRENCH_PATH)
    date_col = next((col for col in df.columns if "date" in col.lower()), None)
    if date_col is None:
        return None, {}, [f"{FAMA_FRENCH_PATH.name} missing date column."], ""
    out = pd.DataFrame({"date": to_month_end(df[date_col])})
    metadata: dict[str, dict[str, str]] = {}
    notes: list[str] = []
    selected: list[str] = []

    column_map = {
        "FF_MKT_RF": ["mkt-rf", "mkt_rf", "mktrf", "market-rf", "market_rf"],
        "FF_SMB": ["smb"],
        "FF_HML": ["hml"],
        "FF_RMW": ["rmw"],
        "FF_CMA": ["cma"],
        "FF_MOM": ["mom", "umd"],
        "FF_RF": ["rf"],
    }
    non_date_cols = [col for col in df.columns if col != date_col]
    for target, prefs in column_map.items():
        source_col = pick_best_column(non_date_cols, prefs)
        if source_col is None:
            continue
        parsed, note = parse_scale_series(df[source_col])
        out[target] = parsed
        metadata[target] = build_metadata_entry(FAMA_FRENCH_PATH.name, note)
        notes.append(f"{target} <- {source_col}: {note}")
        selected.append(f"{target}<-{source_col}")
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    return out, metadata, notes, ", ".join(selected)


def load_long_history_aqr_blocks() -> tuple[list[pd.DataFrame], dict[str, dict[str, str]], list[str], dict[str, str]]:
    blocks: list[pd.DataFrame] = []
    metadata: dict[str, dict[str, str]] = {}
    notes: list[str] = []
    source_map: dict[str, str] = {}

    block_specs = [
        (AQR_RF_PATH, {"rf": "RF_MONTHLY"}),
        (AQR_MKT_PATH, {"mkt": "MKT_EXCESS"}),
        (AQR_DURATION_PATH, {"AQR_FI_MARKET": "AQR_FI_MARKET"}),
        (AQR_COMMODITIES_PATH, {"AQR_CMDTY_EW_EXCESS": "AQR_CMDTY_EW_EXCESS", "AQR_CMDTY_CARRY": "AQR_CMDTY_CARRY"}),
        (AQR_BAB_PATH, {"BAB": "BAB"}),
        (AQR_TSMOM_PATH, {"TSMOM": "AQR_TSMOM"}),
        (AQR_QMJ_PATH, {"QMJ": "AQR_QMJ"}),
    ]

    for path, rename_map in block_specs:
        block, block_meta, block_notes = load_aqr_single_factor(path, rename_map)
        notes.extend(block_notes)
        if block is not None:
            blocks.append(block)
            metadata.update(block_meta)
            for target_col in block_meta:
                source_map[target_col] = path.name
    return blocks, metadata, notes, source_map


def merge_blocks(base: pd.DataFrame, blocks: list[pd.DataFrame]) -> pd.DataFrame:
    out = base.copy()
    for block in blocks:
        out = out.merge(block, on="date", how="left")
    return out


def create_availability_report(panel: pd.DataFrame, metadata: dict[str, dict[str, str]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for col in [c for c in panel.columns if c != "date"]:
        series = pd.to_numeric(panel[col], errors="coerce") if col not in {"regime_name"} else panel[col]
        numeric = pd.to_numeric(panel[col], errors="coerce")
        first_valid = numeric.first_valid_index() if col != "regime_name" else panel[col].first_valid_index()
        last_valid = numeric.last_valid_index() if col != "regime_name" else panel[col].last_valid_index()
        meta = metadata.get(col, build_metadata_entry(""))
        rows.append(
            {
                "factor_name": col,
                "source_file": meta.get("source_file", ""),
                "first_valid_date": panel.loc[first_valid, "date"].strftime("%Y-%m-%d") if first_valid is not None else "",
                "last_valid_date": panel.loc[last_valid, "date"].strftime("%Y-%m-%d") if last_valid is not None else "",
                "valid_obs": int(panel[col].count()),
                "missing_obs": int(panel[col].isna().sum()),
                "missing_ratio": float(panel[col].isna().mean()),
                "mean": float(numeric.mean()) if not numeric.dropna().empty else np.nan,
                "std": float(numeric.std()) if not numeric.dropna().empty else np.nan,
                "min": float(numeric.min()) if not numeric.dropna().empty else np.nan,
                "max": float(numeric.max()) if not numeric.dropna().empty else np.nan,
                "scale_conversion_note": meta.get("scale_conversion_note", ""),
            }
        )
    return pd.DataFrame(rows)


def create_regime_coverage_report(panel: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for factor in factors:
        if factor not in panel.columns:
            continue
        for regime_name in REGIME_ORDER:
            subset = panel.loc[panel["regime_name"] == regime_name, factor]
            total = len(subset)
            valid = int(subset.count())
            missing = int(subset.isna().sum())
            rows.append(
                {
                    "factor_name": factor,
                    "regime_name": regime_name,
                    "valid_obs": valid,
                    "missing_obs": missing,
                    "coverage_ratio": np.nan if total == 0 else valid / total,
                }
            )
    return pd.DataFrame(rows)


def compute_correlation(panel: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    use = [factor for factor in factors if factor in panel.columns]
    return panel[use].apply(pd.to_numeric, errors="coerce").corr()


def plot_heatmap(corr: pd.DataFrame, title: str, path: Path, vmin: float = -1.0, vmax: float = 1.0) -> None:
    if corr.empty:
        return
    fig, ax = plt.subplots(figsize=(1.8 + 1.5 * len(corr.columns), 1.6 + 0.6 * len(corr.index)))
    mat = corr.to_numpy(dtype=float)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            value = mat[i, j]
            ax.text(j, i, "" if np.isnan(value) else f"{value:.2f}", ha="center", va="center", fontsize=8, color="#111111")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Correlation")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_markdown(context: dict[str, object]) -> None:
    lines = [
        "# Risk Factor Panel",
        "",
        "## Why long-history factors replace ETF-based factors",
        "",
        "- The regime analysis is intended to cover historical environments such as 1974-1991, where ETF histories are missing or too short.",
        "- The core panel therefore uses long-history factor and index series instead of modern ETF implementation proxies.",
        "",
        "## Core design choices",
        "",
        "- `RF_MONTHLY` comes from AQR `rf.csv`, not GS3M.",
        "- `MKT_EXCESS` comes from AQR `mkt.csv`, not SPY.",
        "- `AQR_FI_MARKET_EXCESS` is built from AQR `duration.csv` and used as a long-history fixed income / duration-related proxy rather than a pure U.S. Treasury ETF proxy.",
        "- `AQR_CMDTY_EW_EXCESS` replaces DBC-based commodity returns for historical coverage.",
        "- `GOLD_EXCESS` is built from long-history gold spot/price data in `gold.csv`, not GLD/IAU.",
        "- Credit is represented by `credit_spread` and `D_CREDIT_SPREAD`, not HYG.",
        "- ETFs remain reserved for the future implementation layer, not the historical research layer.",
        "",
        "## Why some AQR style factors were removed",
        "",
        "- `AQR_value_all`, `AQR_momentum_all`, `AQR_carry_all`, and `AQR_defensive_all` are cross-asset long-short research portfolios.",
        "- They remain useful as research benchmarks, but they are harder to map directly into ETF implementation sleeves.",
        "- The current round focuses on more interpretable academic factors with clearer implementation analogs.",
        "",
        "## Why AQR_TSMOM, AQR_QMJ, and Fama-French factors were added",
        "",
        "- `AQR_TSMOM` captures long-history trend-following / time-series momentum exposure and can help identify crisis or macro-transition diversification.",
        "- `AQR_QMJ` captures quality-minus-junk exposure and is more directly related to quality / profitability implementation candidates.",
        "- Fama-French factors add interpretable academic style dimensions:",
        "  - `FF_HML`: equity value",
        "  - `FF_RMW`: profitability / quality",
        "  - `FF_CMA`: conservative investment",
        "  - `FF_SMB`: size",
        "  - `FF_MOM`: equity momentum",
        "- `FF_MKT_RF` was evaluated for validation, but removed from the final panel because it was redundant with `MKT_EXCESS`.",
        "- These academic factors are still not directly investable assets. They are included to identify promising factor roles before later ETF exposure mapping.",
        "",
        "## Scale handling",
        "",
        "- AQR and Fama-French files are converted from percent to decimal when the median absolute value exceeds 0.2 or when percent strings are detected.",
        "",
        "## Source assumptions",
        "",
        "- `mkt.csv` is treated as an excess market return series and used directly as `MKT_EXCESS`.",
        "- `duration.csv` is treated as a raw fixed income market return series and converted to `AQR_FI_MARKET_EXCESS` by subtracting `RF_MONTHLY`.",
        "- `commodities.csv` uses `AQR_CMDTY_EW_EXCESS` directly because the file already labels it as excess.",
        "",
        "## Coverage notes",
        "",
        f"- Dollar note: {context['dollar_note']}",
        f"- Gold note: {context['gold_note']}",
        f"- AQR scale notes: {' || '.join(context['aqr_scale_notes']) if context['aqr_scale_notes'] else 'None'}",
        f"- Fama-French column mapping: {context['ff_mapping_note'] or 'None'}",
        f"- Missing optional factor files: {', '.join(context['missing_optional_factors']) if context['missing_optional_factors'] else 'None'}",
    ]
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")


def create_summary_report(panel: pd.DataFrame, availability: pd.DataFrame, context: dict[str, object]) -> pd.DataFrame:
    rows = [
        {"metric": "sample_start", "value": panel["date"].min().strftime("%Y-%m-%d")},
        {"metric": "sample_end", "value": panel["date"].max().strftime("%Y-%m-%d")},
        {"metric": "number_total_columns", "value": str(len(panel.columns))},
        {"metric": "aqr_scale_conversion_notes", "value": " || ".join(context["aqr_scale_notes"]) if context["aqr_scale_notes"] else "None"},
        {"metric": "missing_optional_factors", "value": ", ".join(context["missing_optional_factors"]) if context["missing_optional_factors"] else "None"},
        {"metric": "dollar_note", "value": context["dollar_note"]},
        {"metric": "gold_note", "value": context["gold_note"]},
        {"metric": "ff_mapping_note", "value": context["ff_mapping_note"] or "None"},
        {"metric": "factors_high_missing_ratio", "value": ", ".join(availability.loc[availability["missing_ratio"] > 0.25, "factor_name"]) or "None"},
    ]
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    regime_block, regime_meta = load_regime_block()
    vix_block, vix_meta = load_daily_level_monthly_stats(VIX_PATH, "VIXCLS", "VIX")
    ted_block, ted_meta = load_daily_level_monthly_stats(TED_PATH, "TEDRATE", "TED")
    dollar_block, dollar_meta, dollar_note, dollar_source = load_dollar_block()
    gold_block, gold_meta, gold_note = load_gold_block()
    aqr_blocks, aqr_meta, aqr_scale_notes, aqr_source_map = load_long_history_aqr_blocks()
    ff_block, ff_meta, ff_notes, ff_mapping_note = load_fama_french_block()

    panel_blocks = [vix_block, ted_block] + ([dollar_block] if dollar_block is not None else []) + ([gold_block] if gold_block is not None else []) + aqr_blocks + ([ff_block] if ff_block is not None else [])
    panel = merge_blocks(regime_block, panel_blocks)
    panel = panel.sort_values("date").reset_index(drop=True)

    if "AQR_FI_MARKET" in panel.columns:
        panel["AQR_FI_MARKET_EXCESS"] = panel["AQR_FI_MARKET"] - panel["RF_MONTHLY"]
        panel = panel.drop(columns=["AQR_FI_MARKET"])
        aqr_meta["AQR_FI_MARKET_EXCESS"] = build_metadata_entry(AQR_DURATION_PATH.name, "Converted duration raw return to excess return by subtracting RF_MONTHLY.")
        if "AQR_FI_MARKET" in aqr_source_map:
            del aqr_source_map["AQR_FI_MARKET"]
        aqr_source_map["AQR_FI_MARKET_EXCESS"] = AQR_DURATION_PATH.name
    if "GOLD_RET" in panel.columns and "RF_MONTHLY" in panel.columns:
        panel["GOLD_EXCESS"] = panel["GOLD_RET"] - panel["RF_MONTHLY"]
        gold_meta["GOLD_EXCESS"] = build_metadata_entry(GOLD_PATH.name, "Computed GOLD_EXCESS as GOLD_RET minus RF_MONTHLY.")

    metadata = {}
    metadata.update(regime_meta)
    metadata.update(vix_meta)
    metadata.update(ted_meta)
    metadata.update(dollar_meta)
    metadata.update(gold_meta)
    metadata.update(aqr_meta)
    metadata.update(ff_meta)

    for col in CORE_COLUMNS:
        if col not in panel.columns:
            panel[col] = np.nan
            metadata.setdefault(col, build_metadata_entry(""))
    core_panel = panel[CORE_COLUMNS].copy()
    long_history_panel = core_panel.copy()

    availability = create_availability_report(core_panel, metadata)
    coverage_factors = [col for col in RETURN_LIKE_COLUMNS + SHOCK_COLUMNS if col in core_panel.columns]
    regime_coverage = create_regime_coverage_report(core_panel, coverage_factors)
    core_corr_factors = [
        factor
        for factor in [
            "MKT_EXCESS",
            "AQR_FI_MARKET_EXCESS",
            "AQR_CMDTY_EW_EXCESS",
            "GOLD_EXCESS",
            "AQR_CMDTY_CARRY",
            "BAB",
            "AQR_TSMOM",
            "AQR_QMJ",
            "FF_SMB",
            "FF_HML",
            "FF_RMW",
            "FF_CMA",
            "FF_MOM",
            "D_CREDIT_SPREAD",
            "D_GS10",
            "D_VIX",
            "D_DOLLAR",
        ]
        if factor in core_panel.columns
    ]
    state_corr_factors = [
        factor
        for factor in [
            "growth_pc1",
            "inflation_pc1",
            "gs10",
            "term_spread_10y_1y",
            "credit_spread",
            "DOLLAR_LEVEL",
            "VIX_LEVEL",
            "TED_LEVEL",
        ]
        if factor in core_panel.columns
    ]
    core_corr = compute_correlation(core_panel, core_corr_factors)
    state_corr = compute_correlation(core_panel, state_corr_factors)

    context = {
        "aqr_scale_notes": aqr_scale_notes + ff_notes,
        "missing_optional_factors": [
            path.name
            for path in [AQR_TSMOM_PATH, AQR_QMJ_PATH, FAMA_FRENCH_PATH]
            if not path.exists()
        ],
        "dollar_note": dollar_note,
        "gold_note": gold_note,
        "ff_mapping_note": ff_mapping_note,
        "source_map": {
            "MKT_EXCESS": aqr_source_map.get("MKT_EXCESS", ""),
            "RF_MONTHLY": aqr_source_map.get("RF_MONTHLY", ""),
            "AQR_FI_MARKET_EXCESS": aqr_source_map.get("AQR_FI_MARKET_EXCESS", ""),
            "AQR_CMDTY_EW_EXCESS": aqr_source_map.get("AQR_CMDTY_EW_EXCESS", ""),
            "GOLD_EXCESS": GOLD_PATH.name if "GOLD_EXCESS" in core_panel.columns else "",
            "AQR_TSMOM": aqr_source_map.get("AQR_TSMOM", ""),
            "AQR_QMJ": aqr_source_map.get("AQR_QMJ", ""),
            "FAMA_FRENCH": FAMA_FRENCH_PATH.name if ff_block is not None else "",
        },
    }
    summary = create_summary_report(core_panel, availability, context)
    write_markdown(context)

    core_panel.to_csv(CORE_PANEL_PATH, index=False)
    long_history_panel.to_csv(LONG_HISTORY_PANEL_PATH, index=False)
    availability.to_csv(AVAILABILITY_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    regime_coverage.to_csv(REGIME_COVERAGE_PATH, index=False)
    core_corr.to_csv(CORE_CORR_PATH)
    state_corr.to_csv(STATE_CORR_PATH)
    plot_heatmap(core_corr, "Core Risk Factor Correlation Heatmap", CORR_HEATMAP_PATH)
    plot_heatmap(state_corr, "Regime State Variable Correlation Heatmap", STATE_CORR_HEATMAP_PATH)

    panel_start = core_panel["date"].min().strftime("%Y-%m-%d")
    panel_end = core_panel["date"].max().strftime("%Y-%m-%d")
    print(f"Panel sample: {panel_start} to {panel_end}")
    print(f"Factors included in final panel: {', '.join([col for col in CORE_COLUMNS if col in core_panel.columns])}")
    removed_factors = ["AQR_value_all", "AQR_momentum_all", "AQR_carry_all", "AQR_defensive_all"]
    removed_absent = {factor: factor not in core_panel.columns for factor in removed_factors}
    print(f"Removed factors absent: {removed_absent}")
    key_factors = ["MKT_EXCESS", "RF_MONTHLY", "AQR_FI_MARKET_EXCESS", "AQR_CMDTY_EW_EXCESS", "GOLD_EXCESS", "BAB", "AQR_TSMOM", "AQR_QMJ", "FF_SMB", "FF_HML", "FF_RMW", "FF_CMA", "FF_MOM"]
    print("Valid observations for key/new factors:")
    for factor in key_factors:
        if factor in core_panel.columns:
            print(f"- {factor}: {int(core_panel[factor].count())}")
    print("Coverage of key/new factors in each regime:")
    coverage_view = regime_coverage.loc[regime_coverage["factor_name"].isin(["MKT_EXCESS", "AQR_FI_MARKET_EXCESS", "AQR_CMDTY_EW_EXCESS", "GOLD_EXCESS", "AQR_TSMOM", "AQR_QMJ", "FF_SMB", "FF_HML", "FF_RMW", "FF_CMA", "FF_MOM"])]
    print(coverage_view.to_string(index=False))
    high_rate_coverage = coverage_view.loc[coverage_view["regime_name"] == "High-Rate / Inflation-Pressure", ["factor_name", "coverage_ratio"]]
    print("High-Rate / Inflation-Pressure coverage:")
    print(high_rate_coverage.to_string(index=False))
    etf_tokens = ["SPY", "IEF", "HYG", "DBC", "GLD", "IAU", "BIL", "SGOV", "SHV", "DUR_EXCESS", "CMDTY_EXCESS", "HY_CREDIT_RESIDUAL"]
    etf_columns_remaining = [col for col in core_panel.columns if col in etf_tokens]
    print(f"Any ETF-based factor remains in long-history core panel: {bool(etf_columns_remaining)}")
    print(f"ETF-like columns still present: {', '.join(etf_columns_remaining) if etf_columns_remaining else 'None'}")
    print(f"Source file used for MKT_EXCESS: {context['source_map']['MKT_EXCESS']}")
    print(f"Source file used for RF_MONTHLY: {context['source_map']['RF_MONTHLY']}")
    print(f"Source file used for AQR_FI_MARKET_EXCESS: {context['source_map']['AQR_FI_MARKET_EXCESS']}")
    print(f"Source file used for AQR_CMDTY_EW_EXCESS: {context['source_map']['AQR_CMDTY_EW_EXCESS']}")
    print(f"Source file used for GOLD_EXCESS: {context['source_map']['GOLD_EXCESS']}")
    print(f"Source file used for AQR_TSMOM: {context['source_map']['AQR_TSMOM']}")
    print(f"Source file used for AQR_QMJ: {context['source_map']['AQR_QMJ']}")
    print(f"Source file used for Fama-French factors: {context['source_map']['FAMA_FRENCH']}")

    for path in [
        CORE_PANEL_PATH,
        LONG_HISTORY_PANEL_PATH,
        AVAILABILITY_PATH,
        SUMMARY_PATH,
        REGIME_COVERAGE_PATH,
        CORE_CORR_PATH,
        STATE_CORR_PATH,
        CORR_HEATMAP_PATH,
        STATE_CORR_HEATMAP_PATH,
        MARKDOWN_PATH,
    ]:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
