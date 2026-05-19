"""Field-level lineage and mismatch audit for source-only rebuild failures.

This diagnostic is intentionally read-only with respect to existing mainline
outputs. It writes only to results/source_only_audit/.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "source_only_audit"
AUDITED_FIELDS = [
    "macro_regime_confirmed",
    "CASH_return",
    "VIX_ZSCORE_120D",
    "CREDIT_SPREAD_BAA_AAA",
    "D_CREDIT_SPREAD_20D",
    "CMDTY_RET60",
]

REFERENCE_PANEL = ROOT / "results" / "09_final_strategy" / "mature_regime_hedge_final" / "daily_backtest_panel.csv"
REFERENCE_FINAL_RETURNS = ROOT / "results" / "main_pipeline_final" / "tables" / "final_daily_returns.csv"
REFERENCE_FINAL_WEIGHTS = ROOT / "results" / "main_pipeline_final" / "tables" / "final_daily_weights.csv"
SOURCE_KEY_PANEL = ROOT / "results" / "main_pipeline_final" / "hard_dependency_validation" / "source_rebuilt_key_panel.csv"


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def iter_code_files() -> Iterable[Path]:
    for base in [ROOT / "scripts", ROOT / "src"]:
        if base.exists():
            yield from base.rglob("*.py")


def classify_role(line: str) -> str:
    text = line.strip()
    lower = text.lower()
    if ".to_csv" in lower or "write_" in lower:
        return "saved"
    if "read_csv" in lower or "pd.read" in lower:
        return "read"
    if any(k in lower for k in ["strategy", "weight_", "entry_signal", "risk_state", "final_state", "timing_state"]):
        return "used_in_strategy"
    if "=" in text and not text.startswith("#"):
        if any(k in lower for k in ["rolling", "shift", "fillna", "ffill", "bfill", "diff", "pct_change", "merge", "resample", "rename"]):
            return "transformed"
        return "generated"
    return "read"


def suspected_parameters(line: str, context: str) -> str:
    joined = f"{context}\n{line}"
    flags = []
    patterns = [
        ("rolling", r"rolling\(([^)]*)\)"),
        ("shift", r"shift\(([^)]*)\)"),
        ("fillna", r"fillna\(([^)]*)\)"),
        ("ffill", r"\.ffill\("),
        ("bfill", r"\.bfill\("),
        ("diff", r"diff\(([^)]*)\)"),
        ("pct_change", r"pct_change\(([^)]*)\)"),
        ("resample", r"resample\(([^)]*)\)"),
        ("merge", r"\.merge\("),
    ]
    for name, pattern in patterns:
        matches = re.findall(pattern, joined)
        if matches:
            if matches == [""]:
                flags.append(name)
            else:
                flags.append(f"{name}: {matches[:3]}")
    numeric_thresholds = re.findall(r"[<>]=?\s*-?\d+(?:\.\d+)?", joined)
    if numeric_thresholds:
        flags.append("thresholds: " + ", ".join(sorted(set(numeric_thresholds))[:8]))
    return "; ".join(flags)


def build_field_lineage_table() -> pd.DataFrame:
    rows = []
    for path in iter_code_files():
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        full_text = "\n".join(lines)
        touches_results = "results" in full_text
        for i, line in enumerate(lines, start=1):
            for field in AUDITED_FIELDS:
                if field not in line:
                    continue
                context = "\n".join(lines[max(0, i - 4) : min(len(lines), i + 3)])
                rows.append(
                    {
                        "field": field,
                        "file_path": rel(path),
                        "line": i,
                        "role": classify_role(line),
                        "relevant_function_or_code_snippet": line.strip()[:500],
                        "suspected_parameters": suspected_parameters(line, context),
                        "whether_it_touches_results_intermediate_outputs": bool(touches_results),
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["field", "file_path", "line"]).reset_index(drop=True)
    return out


def load_or_build_source_panel() -> pd.DataFrame:
    if SOURCE_KEY_PANEL.exists():
        return pd.read_csv(SOURCE_KEY_PANEL, parse_dates=["date"])
    sys.path.insert(0, str(ROOT / "scripts"))
    import hard_validate_main_pipeline_source_only as hard

    return hard.build_source_key_panel()


def load_reference_panel() -> pd.DataFrame:
    if not REFERENCE_PANEL.exists():
        raise FileNotFoundError(f"Missing reference panel: {rel(REFERENCE_PANEL)}")
    return pd.read_csv(REFERENCE_PANEL, parse_dates=["date"])


def field_pair(field: str) -> tuple[str, str]:
    if field == "macro_regime_confirmed":
        return "macro_regime_confirmed", "macro_regime_confirmed_source_rebuild"
    return field, field


def numeric_sign_changes(diff: pd.Series) -> int:
    x = diff.dropna()
    if x.empty:
        return 0
    signs = np.sign(x.to_numpy())
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


def build_field_diff_summary(ref: pd.DataFrame, source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = ref.merge(source, on="date", how="inner", suffixes=("_reference", "_source"))
    rows = []
    detail = merged[["date"]].copy()
    for field in AUDITED_FIELDS:
        ref_col, src_col = field_pair(field)
        if field != "macro_regime_confirmed":
            ref_col = f"{field}_reference" if f"{field}_reference" in merged.columns else field
            src_col = f"{field}_source" if f"{field}_source" in merged.columns else field
        if ref_col not in merged.columns or src_col not in merged.columns:
            rows.append(
                {
                    "field": field,
                    "exact_match": False,
                    "max_abs_diff": np.nan,
                    "mean_abs_diff": np.nan,
                    "nonzero_diff_count": np.nan,
                    "first_diff_date": "",
                    "last_diff_date": "",
                    "number_of_sign_changes_if_numeric": np.nan,
                    "notes": f"missing columns: {ref_col}, {src_col}",
                }
            )
            continue
        detail[f"{field}_reference"] = merged[ref_col]
        detail[f"{field}_source_rebuild"] = merged[src_col]
        if field == "macro_regime_confirmed":
            mismatch = merged[ref_col].astype(str) != merged[src_col].astype(str)
            rows.append(
                {
                    "field": field,
                    "exact_match": bool(not mismatch.any()),
                    "max_abs_diff": np.nan,
                    "mean_abs_diff": np.nan,
                    "nonzero_diff_count": int(mismatch.sum()),
                    "first_diff_date": merged.loc[mismatch, "date"].min().date().isoformat() if mismatch.any() else "",
                    "last_diff_date": merged.loc[mismatch, "date"].max().date().isoformat() if mismatch.any() else "",
                    "number_of_sign_changes_if_numeric": np.nan,
                    "notes": "categorical regime mismatch count",
                }
            )
        else:
            diff = pd.to_numeric(merged[ref_col], errors="coerce") - pd.to_numeric(merged[src_col], errors="coerce")
            mismatch = diff.abs() > 1e-10
            rows.append(
                {
                    "field": field,
                    "exact_match": bool(not mismatch.any()),
                    "max_abs_diff": float(diff.abs().max(skipna=True)),
                    "mean_abs_diff": float(diff.abs().mean(skipna=True)),
                    "nonzero_diff_count": int(mismatch.sum()),
                    "first_diff_date": merged.loc[mismatch, "date"].min().date().isoformat() if mismatch.any() else "",
                    "last_diff_date": merged.loc[mismatch, "date"].max().date().isoformat() if mismatch.any() else "",
                    "number_of_sign_changes_if_numeric": numeric_sign_changes(diff),
                    "notes": "",
                }
            )
    return pd.DataFrame(rows), detail


def compress_boolean_intervals(df: pd.DataFrame, flag_col: str, label_cols: list[str]) -> pd.DataFrame:
    sub = df[df[flag_col]].copy()
    if sub.empty:
        return pd.DataFrame()
    intervals = []
    start_idx = sub.index[0]
    prev_idx = sub.index[0]
    for idx in sub.index[1:]:
        contiguous = idx == prev_idx + 1
        same_labels = all(df.loc[idx, c] == df.loc[prev_idx, c] for c in label_cols)
        if not contiguous or not same_labels:
            block = df.loc[start_idx:prev_idx]
            row = df.loc[start_idx]
            intervals.append(
                {
                    "start_date": block["date"].iloc[0],
                    "end_date": block["date"].iloc[-1],
                    "length": len(block),
                    **{c: row[c] for c in label_cols},
                }
            )
            start_idx = idx
        prev_idx = idx
    block = df.loc[start_idx:prev_idx]
    row = df.loc[start_idx]
    intervals.append(
        {
            "start_date": block["date"].iloc[0],
            "end_date": block["date"].iloc[-1],
            "length": len(block),
            **{c: row[c] for c in label_cols},
        }
    )
    return pd.DataFrame(intervals)


def build_macro_mismatch_outputs(ref: pd.DataFrame, source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = ref.merge(source[["date", "macro_regime_confirmed_source_rebuild"]], on="date", how="inner")
    merged["macro_regime_mismatch"] = merged["macro_regime_confirmed"].astype(str) != merged["macro_regime_confirmed_source_rebuild"].astype(str)
    details = merged.loc[
        merged["macro_regime_mismatch"],
        [
            "date",
            "macro_regime_confirmed",
            "macro_regime_confirmed_source_rebuild",
            "timing_state",
            "final_state",
            "overlay_state",
            "monthly_either_state",
            "CMDTY_RET60",
            "VIX_ZSCORE_120D",
            "D_CREDIT_SPREAD_20D",
        ],
    ].copy()
    intervals = compress_boolean_intervals(
        merged,
        "macro_regime_mismatch",
        ["macro_regime_confirmed", "macro_regime_confirmed_source_rebuild"],
    )
    if not intervals.empty:
        intervals = intervals.rename(
            columns={
                "macro_regime_confirmed": "reference_regime",
                "macro_regime_confirmed_source_rebuild": "rebuild_regime",
            }
        )
        intervals["is_few_day_shift_pattern"] = intervals["length"] <= 3
        intervals["near_stress_or_recovery"] = intervals.apply(
            lambda row: bool(
                merged.loc[
                    (merged["date"] >= pd.to_datetime(row["start_date"]) - pd.Timedelta(days=10))
                    & (merged["date"] <= pd.to_datetime(row["end_date"]) + pd.Timedelta(days=10)),
                    ["timing_state", "final_state", "overlay_state"],
                ]
                .astype(str)
                .isin(["RISK", "FULL_RISK", "True"])
                .any()
                .any()
            ),
            axis=1,
        )
    confusion = pd.crosstab(merged["macro_regime_confirmed"], merged["macro_regime_confirmed_source_rebuild"])
    return details, intervals, confusion.reset_index()


def compute_signal_panel(panel: pd.DataFrame, macro_col: str) -> pd.DataFrame:
    out = pd.DataFrame({"date": panel["date"]})
    macro = panel[macro_col].astype(str)
    out["FLAT_VIX_STRESS"] = macro.eq("FLAT") & (panel["VIX_ZSCORE_120D"] >= 3.0)
    out["FLAT_CREDIT_DD5_STRESS"] = macro.eq("FLAT") & (
        (panel["spy_drawdown_from_previous_high"] <= -0.05) & (panel["D_CREDIT_SPREAD_20D"] > 0.10)
    )
    out["STEEP_EITHER_SELL_STRESS"] = macro.eq("STEEP") & panel["monthly_either_state"].astype(str).eq("SELL")
    out["STEEP_CREDIT_DD5_STRESS"] = macro.eq("STEEP") & (
        (panel["spy_drawdown_from_previous_high"] <= -0.05) & (panel["D_CREDIT_SPREAD_20D"] > 0.10)
    )
    out["STEEP_CMDTY_RET60_NEG10"] = macro.eq("STEEP") & (panel["CMDTY_RET60"] < -0.10)
    out["BACKBONE_V2_ENTRY_SIGNAL"] = out[
        ["FLAT_VIX_STRESS", "FLAT_CREDIT_DD5_STRESS", "STEEP_EITHER_SELL_STRESS", "STEEP_CREDIT_DD5_STRESS"]
    ].any(axis=1)
    out["FLAT_LOW_RATE"] = macro.eq("FLAT") & (pd.to_numeric(panel.get("GS10", np.nan), errors="coerce") <= 2.9)
    out["FLAT_HIGH_RATE"] = macro.eq("FLAT") & (pd.to_numeric(panel.get("GS10", np.nan), errors="coerce") > 2.9)
    return out


def build_strategy_impact_summary(ref: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    src_cols = [
        "date",
        "macro_regime_confirmed_source_rebuild",
        "CASH_return",
        "VIX_ZSCORE_120D",
        "CREDIT_SPREAD_BAA_AAA",
        "D_CREDIT_SPREAD_20D",
        "CMDTY_RET60",
        "GS10",
    ]
    merged = ref.merge(source[[c for c in src_cols if c in source.columns]], on="date", how="inner", suffixes=("_reference", "_source"))
    ref_variant = merged.copy()
    ref_variant["macro_reference"] = ref_variant["macro_regime_confirmed"]
    source_variant = merged.copy()
    source_variant["macro_source"] = source_variant["macro_regime_confirmed_source_rebuild"]
    for f in ["CASH_return", "VIX_ZSCORE_120D", "CREDIT_SPREAD_BAA_AAA", "D_CREDIT_SPREAD_20D", "CMDTY_RET60", "GS10"]:
        ref_name = f"{f}_reference" if f"{f}_reference" in ref_variant.columns else f
        if ref_name in ref_variant.columns:
            ref_variant[f] = ref_variant[ref_name]
    for f in ["CASH_return", "VIX_ZSCORE_120D", "CREDIT_SPREAD_BAA_AAA", "D_CREDIT_SPREAD_20D", "CMDTY_RET60", "GS10"]:
        src_name = f"{f}_source" if f"{f}_source" in source_variant.columns else f
        ref_name = f"{f}_reference" if f"{f}_reference" in source_variant.columns else f
        if src_name in source_variant.columns:
            source_variant[f] = source_variant[src_name]
        elif ref_name in source_variant.columns:
            source_variant[f] = source_variant[ref_name]
    ref_sig = compute_signal_panel(ref_variant, "macro_reference")
    src_sig = compute_signal_panel(source_variant, "macro_source")
    rows = []
    signal_cols = [c for c in ref_sig.columns if c != "date"]
    final_weight_cols = [c for c in ref.columns if c.startswith("MATURE_REGIME_HEDGE_FINAL_weight_")]
    final_weight_changed_context = pd.Series(False, index=merged.index)
    if final_weight_cols:
        final_weight_changed_context = ref[final_weight_cols].abs().sum(axis=1).reindex(merged.index).fillna(0) > 0
    field_to_signals = {
        "macro_regime_confirmed": signal_cols,
        "CASH_return": ["portfolio_return_only"],
        "VIX_ZSCORE_120D": ["FLAT_VIX_STRESS", "BACKBONE_V2_ENTRY_SIGNAL"],
        "CREDIT_SPREAD_BAA_AAA": ["D_CREDIT_SPREAD_20D", "FLAT_CREDIT_DD5_STRESS", "STEEP_CREDIT_DD5_STRESS", "BACKBONE_V2_ENTRY_SIGNAL"],
        "D_CREDIT_SPREAD_20D": ["FLAT_CREDIT_DD5_STRESS", "STEEP_CREDIT_DD5_STRESS", "BACKBONE_V2_ENTRY_SIGNAL"],
        "CMDTY_RET60": ["STEEP_CMDTY_RET60_NEG10"],
    }
    diff_summary, _ = build_field_diff_summary(ref, source)
    diff_lookup = diff_summary.set_index("field").to_dict("index")
    for field in AUDITED_FIELDS:
        signals = [s for s in field_to_signals[field] if s in ref_sig.columns]
        changed_days = 0
        if signals:
            changed = pd.Series(False, index=ref_sig.index)
            for sig in signals:
                changed |= ref_sig[sig].astype(bool) != src_sig[sig].astype(bool)
            changed_days = int(changed.sum())
        elif field == "CASH_return":
            ref_col = "CASH_return_reference" if "CASH_return_reference" in merged.columns else "CASH_return"
            src_col = "CASH_return_source" if "CASH_return_source" in merged.columns else "CASH_return"
            changed = (pd.to_numeric(merged[ref_col], errors="coerce") - pd.to_numeric(merged[src_col], errors="coerce")).abs() > 1e-10
            changed_days = int(changed.sum())
        mismatch_count = int(diff_lookup.get(field, {}).get("nonzero_diff_count", 0) or 0)
        impact = "low"
        rationale = "field mismatch does not alter audited primitive signals"
        if field == "CASH_return" and mismatch_count:
            impact = "medium"
            rationale = "CASH return changes portfolio return on days with CASH weight, but does not change entry signals"
        if changed_days > 0 and field != "CASH_return":
            impact = "high"
            rationale = "field mismatch changes at least one primitive/final-state input signal"
        rows.append(
            {
                "field": field,
                "source_mismatch_days": mismatch_count,
                "affected_signal_days": changed_days,
                "strategy_position_diff_proxy_days": changed_days,
                "impact_level": impact,
                "rationale": rationale,
                "signals_checked": "|".join(signals) if signals else "portfolio_return_only",
            }
        )
    return pd.DataFrame(rows)


def write_readme(lineage: pd.DataFrame, diff: pd.DataFrame, intervals: pd.DataFrame, impact: pd.DataFrame) -> None:
    high = impact[impact["impact_level"].eq("high")]
    medium = impact[impact["impact_level"].eq("medium")]
    exact = diff[diff["exact_match"]]
    mismatch = diff[~diff["exact_match"]]
    lines = [
        "# Source-Only Field Lineage Audit",
        "",
        "## Purpose",
        "",
        "This audit explains why the source-only rebuild used by `hard_validate_main_pipeline_source_only.py` does not yet match the current final reference panel.",
        "",
        "## Summary",
        "",
        f"- Fields audited: {len(AUDITED_FIELDS)}",
        f"- Exact-match fields: {len(exact)}",
        f"- Mismatched fields: {len(mismatch)}",
        f"- Macro mismatch intervals: {0 if intervals.empty else len(intervals)}",
        f"- High-impact fields: {', '.join(high['field'].tolist()) if not high.empty else 'none'}",
        f"- Medium-impact fields: {', '.join(medium['field'].tolist()) if not medium.empty else 'none'}",
        "",
        "## Field Difference Findings",
        "",
        diff.to_markdown(index=False),
        "",
        "## Strategy Impact",
        "",
        impact.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- `SPY_return`, `GOLD_return`, `CMDTY_FUT_return`, and `IEF_return` already matched in the hard validation, so asset ETF/futures return lineage is not the blocker.",
        "- `macro_regime_confirmed` mismatch is strategy-relevant because it changes whether FLAT/STEEP/INVERTED signals are allowed.",
        "- `VIX_ZSCORE_120D`, `D_CREDIT_SPREAD_20D`, and `CMDTY_RET60` are strategy-relevant because they directly feed stress or slow-growth triggers.",
        "- `CASH_return` is mostly a return-accounting mismatch rather than an entry-signal mismatch, but it affects final NAV whenever CASH has weight.",
        "- Credit-spread mismatches are likely data-frequency/series lineage issues: several scripts use monthly `AAA/BAA`, others weekly `WAAA/WBAA`, and some intermediate panels carry already-forward-filled values.",
        "- VIX and commodity mismatches are likely rolling-window alignment/shift/source-price issues and should be standardized before old validated panels are removed.",
        "",
        "## Repair Priority",
        "",
        "1. Canonicalize source constructors for credit spread, VIX z-score, CASH return, and CMDTY_RET60 under `src/utils` or a source-only mainline data module.",
        "2. Rebuild `macro_regime_confirmed` from that canonical source with the exact existing confirmation logic and compare transition dates.",
        "3. Move final strategy generation to source-only panels, using validated outputs only as comparison references.",
        "4. Re-run `scripts/hard_validate_main_pipeline_source_only.py`; only after all rows pass should validated/intermediate outputs be considered safe to archive.",
    ]
    (OUT / "README_source_only_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    lineage = build_field_lineage_table()
    lineage.to_csv(OUT / "field_lineage_table.csv", index=False)

    ref = load_reference_panel()
    source = load_or_build_source_panel()
    diff, detail = build_field_diff_summary(ref, source)
    diff.to_csv(OUT / "field_diff_summary.csv", index=False)
    detail.to_csv(OUT / "field_diff_detail.csv", index=False)

    macro_details, intervals, confusion = build_macro_mismatch_outputs(ref, source)
    macro_details.to_csv(OUT / "macro_regime_mismatch_dates.csv", index=False)
    intervals.to_csv(OUT / "macro_regime_mismatch_intervals.csv", index=False)
    confusion.to_csv(OUT / "macro_regime_confusion_matrix.csv", index=False)

    impact = build_strategy_impact_summary(ref, source)
    impact.to_csv(OUT / "strategy_impact_summary.csv", index=False)

    write_readme(lineage, diff, intervals, impact)
    print("Field lineage audit complete.")
    print(f"Lineage rows: {len(lineage)}")
    print(f"Mismatched fields: {int((~diff['exact_match']).sum())}")
    print(f"Macro mismatch intervals: {0 if intervals.empty else len(intervals)}")
    print(f"High-impact fields: {', '.join(impact.loc[impact['impact_level'].eq('high'), 'field']) or 'none'}")
    print(f"Output: {rel(OUT)}")


if __name__ == "__main__":
    main()
