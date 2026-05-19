from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from regime.utils import RAW, PROCESSED, ensure_project_dirs, standardize_series, annualized_percent_change, extract_oriented_pc1


OUTPUT_PATH = PROCESSED / "regime_inputs_simplified.csv"
FACTOR_SUMMARY_PATH = ROOT / "results" / "regime" / "factor_construction_summary.csv"


def build_growth_factor() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfnai = standardize_series(RAW / "Growth" / "CFNAI revised.csv", "observation_date", "CFNAI").rename(columns={"value": "cfnai"})
    gdp = standardize_series(RAW / "Growth" / "GDP revised.csv", "observation_date", "GDPC1")
    gdp["gdp_amom"] = annualized_percent_change(gdp)
    gdp = gdp[["date", "gdp_amom"]]
    ipgr = standardize_series(RAW / "Growth" / "IPGR.csv", "period_start_date", "INDPRO")
    ipgr["ipgr_amom"] = annualized_percent_change(ipgr)
    ipgr = ipgr[["date", "ipgr_amom"]]

    monthly = pd.date_range(
        start=min(frame["date"].min() for frame in [cfnai, gdp, ipgr]),
        end=max(frame["date"].max() for frame in [cfnai, gdp, ipgr]),
        freq="MS",
    )
    panel = pd.DataFrame({"date": monthly})
    for frame in [cfnai, gdp, ipgr]:
        panel = panel.merge(frame, on="date", how="left")
    panel["gdp_amom"] = panel["gdp_amom"].ffill(limit=2)
    growth_pc, meta = extract_oriented_pc1(panel, ["cfnai", "gdp_amom", "ipgr_amom"], "growth_pc1")
    meta["factor_group"] = "growth"
    meta["transformation"] = meta["source_variable"].map(
        {
            "cfnai": "level",
            "gdp_amom": "annualized percent change",
            "ipgr_amom": "annualized percent change",
        }
    )
    return growth_pc, meta


def build_inflation_factor() -> tuple[pd.DataFrame, pd.DataFrame]:
    cpi = standardize_series(RAW / "inflation" / "CPI.csv", "period_start_date", "CPIAUCSL")
    cpi["cpi_amom"] = annualized_percent_change(cpi)
    cpi = cpi[["date", "cpi_amom"]]
    ppi = standardize_series(RAW / "inflation" / "PPI revised.csv", "observation_date", "PPIACO")
    ppi["ppi_amom"] = annualized_percent_change(ppi)
    ppi = ppi[["date", "ppi_amom"]]
    core_cpi = standardize_series(RAW / "inflation" / "core cpi.csv", "observation_date", "CORESTICKM158SFRBATL")
    core_cpi = core_cpi.rename(columns={"value": "core_cpi_ar"})

    monthly = pd.date_range(
        start=min(frame["date"].min() for frame in [cpi, ppi, core_cpi]),
        end=max(frame["date"].max() for frame in [cpi, ppi, core_cpi]),
        freq="MS",
    )
    panel = pd.DataFrame({"date": monthly})
    for frame in [cpi, ppi, core_cpi]:
        panel = panel.merge(frame, on="date", how="left")
    inflation_pc, meta = extract_oriented_pc1(panel, ["cpi_amom", "ppi_amom", "core_cpi_ar"], "inflation_pc1")
    meta["factor_group"] = "inflation"
    meta["transformation"] = meta["source_variable"].map(
        {
            "cpi_amom": "annualized percent change",
            "ppi_amom": "annualized percent change",
            "core_cpi_ar": "provided monthly percent change at annual rate",
        }
    )
    return inflation_pc, meta


def build_rate_credit_panel() -> pd.DataFrame:
    gs1 = standardize_series(RAW / "rate" / "GS1.csv", "observation_date", "GS1").rename(columns={"value": "gs1"})
    gs10 = standardize_series(RAW / "rate" / "GS10.csv", "observation_date", "GS10").rename(columns={"value": "gs10"})
    aaa = standardize_series(RAW / "Credit" / "AAA.csv", "observation_date", "AAA").rename(columns={"value": "aaa"})
    baa = standardize_series(RAW / "Credit" / "BAA.csv", "observation_date", "BAA").rename(columns={"value": "baa"})

    monthly = pd.date_range(
        start=min(frame["date"].min() for frame in [gs1, gs10, aaa, baa]),
        end=max(frame["date"].max() for frame in [gs1, gs10, aaa, baa]),
        freq="MS",
    )
    panel = pd.DataFrame({"date": monthly})
    for frame in [gs1, gs10, aaa, baa]:
        panel = panel.merge(frame, on="date", how="left")
    panel["term_spread_10y_1y"] = panel["gs10"] - panel["gs1"]
    panel["credit_spread"] = panel["baa"] - panel["aaa"]
    return panel[["date", "gs10", "term_spread_10y_1y", "credit_spread"]]


def main() -> None:
    ensure_project_dirs()
    growth_pc, growth_meta = build_growth_factor()
    inflation_pc, inflation_meta = build_inflation_factor()
    rate_credit = build_rate_credit_panel()

    regime_inputs = (
        growth_pc
        .merge(inflation_pc, on="date", how="inner")
        .merge(rate_credit, on="date", how="inner")
        .dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )
    regime_inputs.to_csv(OUTPUT_PATH, index=False)

    factor_summary = pd.concat([growth_meta, inflation_meta], ignore_index=True)
    factor_summary.insert(0, "retained_variables", factor_summary.groupby("factor_name")["source_variable"].transform(lambda s: ", ".join(s)))
    factor_summary["baseline_specification"] = "4 states, penalty=1.0, no ISM, no sentiment"
    factor_summary.to_csv(FACTOR_SUMMARY_PATH, index=False)

    print(f"Saved regime inputs to {OUTPUT_PATH}")
    print(f"Saved factor construction summary to {FACTOR_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
