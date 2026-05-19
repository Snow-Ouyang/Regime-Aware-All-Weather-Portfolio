"""Smoke test and dependency audit for the cleaned research pipeline.

This script does not delete or move files. It checks the main research
pipeline, compiles core scripts, maps final-report dependencies, and writes
KEEP / ARCHIVE / DO_NOT_DELETE dry-run reports.
"""

from __future__ import annotations

import csv
import os
import py_compile
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
ARCHIVE_ROOT = ROOT / "archive" / "exploratory_unused"


@dataclass
class PipelineModule:
    module_name: str
    required_scripts: list[str]
    required_inputs: list[str]
    required_outputs: list[str]
    notes: str = ""


PIPELINE_MODULES = [
    PipelineModule(
        "01_regime_discovery_clustering",
        ["src/regime/jump_model_clustering.py", "src/regime/fit_simplified_regime_model.py"],
        ["data"],
        ["results/01_regime_discovery", "results/regime"],
        "Jump model / clustering summary and regime variable distributions.",
    ),
    PipelineModule(
        "02_rule_based_regime_construction",
        [
            "src/regime/build_rule_based_regime.py",
            "src/regime/plot_regime_diagnostics.py",
            "src/regime/build_regime_dataset.py",
            "src/regime/plot_regime_outputs.py",
        ],
        ["data"],
        ["results/02_rule_based_regime", "results/rule_diagnostics"],
        "FLAT / STEEP / INVERTED regime panel, timeline, and interpretation diagnostics.",
    ),
    PipelineModule(
        "03_monthly_timing_backbone",
        [
            "src/timing/reproduce_monthly_timing.py",
            "src/strategies/absolute_momentum_spy_cash.py",
            "src/strategies/faber_spy_cash_timing.py",
            "src/analysis/monthly_either_crash_brake_diagnostics.py",
        ],
        ["data"],
        [
            "results/03_monthly_timing",
            "results/absolute_momentum_spy_cash",
            "results/faber_spy_cash_timing",
            "results/monthly_either_crash_brake_diagnostics",
        ],
        "Antonacci, Faber, and Monthly Either timing outputs.",
    ),
    PipelineModule(
        "04_high_frequency_stress_triggers",
        [
            "src/timing/build_backbone_v2.py",
            "src/strategies/spy_cash_backbone_upgrade_ablation.py",
            "src/analysis/flat_vix_credit_trigger_diagnostic.py",
            "src/analysis/credit_trigger_by_regime_diagnostic.py",
        ],
        ["results/spy_cash_backbone_upgrade_ablation/daily_backtest_panel.csv"],
        [
            "results/04_stress_triggers",
            "results/spy_cash_backbone_upgrade_ablation/daily_backtest_panel.csv",
            "results/flat_vix_credit_trigger_diagnostic",
            "results/credit_trigger_by_regime_diagnostic",
        ],
        "VIX z-score, credit plus drawdown, and regime-dependent stress trigger diagnostics.",
    ),
    PipelineModule(
        "05_recovery_rule",
        ["src/timing/recovery_rule_diagnostic.py", "src/analysis/stress_recovery_grid_search.py"],
        ["results/spy_cash_backbone_upgrade_ablation/daily_backtest_panel.csv"],
        ["results/05_recovery", "results/stress_recovery_grid_search"],
        "Recovery comparison summary and final R3 evidence.",
    ),
    PipelineModule(
        "06_2015_2016_slow_growth_repair",
        [
            "src/diagnostics/drawdown_2015_2016_forensic.py",
            "src/diagnostics/commodity_trigger_by_regime.py",
            "src/strategies/mature_strategy_steep_cmdty_overlay_50spy50ief.py",
        ],
        ["results/spy_cash_backbone_upgrade_ablation/daily_backtest_panel.csv"],
        [
            "results/06_2015_2016_repair",
            "results/drawdown_2015_2016_forensic_diagnostic",
            "results/commodity_crash_transmission_by_regime",
            "results/mature_steep_cmdty_overlay_50spy50ief",
        ],
        "2015-2016 forensic, commodity trigger by regime, and STEEP overlay evidence.",
    ),
    PipelineModule(
        "07_cross_state_asset_behavior",
        [
            "src/diagnostics/cross_state_asset_behavior.py",
            "src/diagnostics/bond_sleeve_diagnostic.py",
            "src/analysis/hedge_asset_cross_state_diagnostic_extended.py",
            "src/analysis/evaluate_bond_sleeve_candidates.py",
        ],
        ["results/spy_cash_backbone_upgrade_ablation/daily_backtest_panel.csv"],
        ["results/07_cross_state_asset_behavior", "results/hedge_asset_cross_state_diagnostic_extended"],
        "Macro regime x stress state asset performance and heatmaps.",
    ),
    PipelineModule(
        "08_allocation",
        [
            "src/allocation/inverse_vol_allocation.py",
            "src/allocation/risk_parity_comparison.py",
            "src/analysis/invvol_window_grid_search.py",
            "src/strategies/regime_aware_risk_parity_allocation.py",
        ],
        ["results/spy_cash_backbone_upgrade_ablation/daily_backtest_panel.csv"],
        ["results/08_allocation", "results/regime_aware_risk_parity_allocation", "results/invvol_window_grid_search"],
        "Inverse-vol allocation and fixed vs inverse-vol vs ERC comparison.",
    ),
    PipelineModule(
        "09_final_strategy",
        ["src/allocation/final_strategy_backtest.py", "src/strategies/mature_regime_hedge_final.py"],
        ["results/regime_aware_risk_parity_allocation/daily_backtest_panel.csv"],
        [
            "results/09_final_strategy/mature_regime_hedge_final/performance_summary.csv",
            "results/09_final_strategy/mature_regime_hedge_final/crisis_performance.csv",
            "results/09_final_strategy/mature_regime_hedge_final/daily_backtest_panel.csv",
            "figures/09_final_strategy/mature_regime_hedge_final/final_equity_curve_log.png",
            "figures/09_final_strategy/mature_regime_hedge_final/final_drawdown_comparison.png",
            "figures/09_final_strategy/mature_regime_hedge_final/final_weight_stack.png",
        ],
        "MATURE_REGIME_HEDGE_FINAL backtest and final figures.",
    ),
]


FINAL_CORE_FILES = [
    "README.md",
    "reports/FINAL_REPORT.md",
    "reports/OUTPUT_INDEX.md",
    "config/project_config.yaml",
    "requirements.txt",
    "data/README.md",
    "src/strategies/mature_regime_hedge_final.py",
    "src/allocation/final_strategy_backtest.py",
    "results/09_final_strategy/mature_regime_hedge_final/performance_summary.csv",
    "results/09_final_strategy/mature_regime_hedge_final/crisis_performance.csv",
    "results/09_final_strategy/mature_regime_hedge_final/daily_backtest_panel.csv",
    "results/09_final_strategy/mature_regime_hedge_final/final_strategy_component_attribution.csv",
    "results/09_final_strategy/mature_regime_hedge_final/final_strategy_decision_summary.csv",
    "figures/09_final_strategy/mature_regime_hedge_final/final_equity_curve_log.png",
    "figures/09_final_strategy/mature_regime_hedge_final/final_drawdown_comparison.png",
    "figures/09_final_strategy/mature_regime_hedge_final/final_weight_stack.png",
    "figures/09_final_strategy/mature_regime_hedge_final/case_2015_2016_final.png",
    "figures/09_final_strategy/mature_regime_hedge_final/case_2022_rate_war_final.png",
    "figures/09_final_strategy/mature_regime_hedge_final/case_2025_pullback_final.png",
]


KEEP_DIRS = [
    "config",
    "data",
    "reports",
    "src/regime",
    "src/timing",
    "src/diagnostics",
    "src/allocation",
    "src/utils",
    "results/01_regime_discovery",
    "results/02_rule_based_regime",
    "results/03_monthly_timing",
    "results/04_stress_triggers",
    "results/05_recovery",
    "results/06_2015_2016_repair",
    "results/07_cross_state_asset_behavior",
    "results/08_allocation",
    "results/09_final_strategy",
    "figures/01_regime_discovery",
    "figures/02_rule_based_regime",
    "figures/03_monthly_timing",
    "figures/04_stress_triggers",
    "figures/05_recovery",
    "figures/06_2015_2016_repair",
    "figures/07_cross_state_asset_behavior",
    "figures/08_allocation",
    "figures/09_final_strategy",
]


DO_NOT_DELETE_DIRS = [
    "data/raw",
    "data/processed",
    "results/09_final_strategy",
    "figures/09_final_strategy",
    "results/regime_aware_risk_parity_allocation",
    "results/spy_cash_backbone_upgrade_ablation",
    "results/mature_regime_hedge_final",
    "figures/mature_regime_hedge_final",
    "src/strategies/mature_regime_hedge_final.py",
]


ARCHIVE_NAME_PATTERNS = [
    "factor",
    "aqr",
    "fama",
    "two_fund",
    "steep_test",
    "steep_mix",
    "regime_aware_hedge_allocation_v1",
    "commodity_stress_trigger_diagnostic",
    "growth_factor_stress_trigger_diagnostic",
    "stress_recovery_price_trigger_diagnostic",
    "spy_cash_stress_recovery_with_commodity",
    "spy_cash_stress_recovery_with_credit",
    "spy_cash_stress_recovery_timing",
    "steep_risk_duration_and_ijh_experiment",
    "diagnose_2022_strategy_divergence",
    "net_liquidity",
    "covid_vs_flat",
    "defensive_sleeve",
]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def exists(path: str) -> bool:
    return (ROOT / path).exists()


def compile_script(path: str) -> tuple[bool, str]:
    full = ROOT / path
    if not full.exists() or not full.is_file():
        return False, "missing"
    try:
        py_compile.compile(str(full), doraise=True)
        return True, "compiled"
    except Exception as exc:  # pragma: no cover - smoke script diagnostics
        return False, f"compile_error: {exc}"


def evaluate_module(module: PipelineModule) -> dict[str, str]:
    missing_scripts = [p for p in module.required_scripts if not exists(p)]
    missing_inputs = [p for p in module.required_inputs if not exists(p)]
    missing_outputs = [p for p in module.required_outputs if not exists(p)]

    compile_notes = []
    compile_failures = []
    for script in module.required_scripts:
        ok, note = compile_script(script)
        compile_notes.append(f"{script}: {note}")
        if not ok:
            compile_failures.append(script)

    if missing_scripts or compile_failures:
        status = "FAIL"
    elif missing_outputs or missing_inputs:
        status = "WARNING"
    else:
        status = "PASS"

    return {
        "module_name": module.module_name,
        "status": status,
        "required_scripts": "; ".join(module.required_scripts),
        "required_inputs": "; ".join(module.required_inputs),
        "required_outputs": "; ".join(module.required_outputs),
        "missing_files": "; ".join(missing_scripts + missing_inputs + missing_outputs),
        "notes": module.notes + " | " + " | ".join(compile_notes),
    }


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(c, "")).replace("\n", " ") for c in columns) + " |")
    return "\n".join(out)


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_markdown_dependencies(markdown_path: Path) -> list[str]:
    if not markdown_path.exists():
        return []
    text = markdown_path.read_text(encoding="utf-8", errors="ignore")
    refs = []
    for match in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", text):
        ref = match.group(1).strip()
        if ref.startswith(("http://", "https://")):
            continue
        resolved = (markdown_path.parent / ref).resolve()
        try:
            refs.append(rel(resolved))
        except ValueError:
            refs.append(ref)
    for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", text):
        ref = match.group(1).strip()
        if ref.startswith(("http://", "https://", "#")):
            continue
        resolved = (markdown_path.parent / ref).resolve()
        try:
            refs.append(rel(resolved))
        except ValueError:
            refs.append(ref)
    return sorted(set(refs))


def find_read_csv_dependencies(script_path: Path) -> list[str]:
    if not script_path.exists():
        return []
    text = script_path.read_text(encoding="utf-8", errors="ignore")
    candidates = set()
    for match in re.finditer(r"read_csv\((?:r)?[\"']([^\"']+)[\"']", text):
        value = match.group(1)
        if value.startswith(("http://", "https://")):
            continue
        candidates.add(value.replace("\\", "/"))
    return sorted(candidates)


def path_is_referenced(path: str, references: set[str]) -> bool:
    p = path.replace("\\", "/").strip("/")
    return any(ref == p or ref.startswith(p + "/") or p.startswith(ref + "/") for ref in references)


def collect_keep_paths(final_refs: set[str]) -> list[dict[str, str]]:
    keep = set(FINAL_CORE_FILES) | final_refs
    for module in PIPELINE_MODULES:
        keep.update(module.required_scripts)
        keep.update(module.required_outputs)
    keep.update(KEEP_DIRS)
    rows = []
    for path in sorted(keep):
        rows.append(
            {
                "path": path,
                "reason": "final report dependency" if path in final_refs else "main pipeline / final strategy core",
                "exists": "yes" if exists(path) else "no",
            }
        )
    return rows


def collect_do_not_delete(final_refs: set[str]) -> list[dict[str, str]]:
    dnd = set(FINAL_CORE_FILES) | final_refs | set(DO_NOT_DELETE_DIRS)
    for path in ["README.md", "reports/FINAL_REPORT.md", "reports/OUTPUT_INDEX.md"]:
        dnd.add(path)
    return [
        {
            "path": path,
            "reason": "raw/final/validated dependency or final report reference",
            "exists": "yes" if exists(path) else "no",
        }
        for path in sorted(dnd)
    ]


def scan_archive_candidates(final_refs: set[str]) -> list[dict[str, str]]:
    candidates = []
    protected_prefixes = set(KEEP_DIRS) | set(DO_NOT_DELETE_DIRS)
    for parent in ["results", "figures"]:
        base = ROOT / parent
        if not base.exists():
            continue
        for child in sorted([p for p in base.iterdir() if p.is_dir()]):
            child_rel = rel(child)
            name = child.name.lower()
            protected = any(child_rel == p or child_rel.startswith(p.rstrip("/") + "/") for p in protected_prefixes)
            referenced = path_is_referenced(child_rel, final_refs)
            pattern_hit = any(pattern in name for pattern in ARCHIVE_NAME_PATTERNS)
            numbered_mainline = bool(re.match(r"^\d\d_", child.name))
            if protected or referenced or numbered_mainline:
                continue
            if pattern_hit or parent in {"results", "figures"}:
                reason = "exploratory/duplicate output not referenced by final report"
                risk = "low" if pattern_hit else "medium"
                candidates.append(
                    {
                        "source_path": child_rel,
                        "target_path": f"archive/exploratory_unused/{child_rel}",
                        "reason": reason,
                        "risk_level": risk,
                        "referenced_by_final_report": "yes" if referenced else "no",
                        "safe_to_move": "yes" if not referenced and risk in {"low", "medium"} else "no",
                    }
                )

    existing_archive = ARCHIVE_ROOT
    if existing_archive.exists():
        for child in sorted(existing_archive.rglob("*")):
            if child.is_dir():
                continue
            child_rel = rel(child)
            candidates.append(
                {
                    "source_path": child_rel,
                    "target_path": child_rel,
                    "reason": "already archived exploratory material",
                    "risk_level": "low",
                    "referenced_by_final_report": "no",
                    "safe_to_move": "already_archived",
                }
            )
    return candidates


def write_smoke_reports(module_rows: list[dict[str, str]]) -> None:
    cols = ["module_name", "status", "required_scripts", "required_inputs", "required_outputs", "missing_files", "notes"]
    write_csv(REPORT_DIR / "MAIN_PIPELINE_SMOKE_TEST.csv", module_rows, cols)
    summary = {
        "PASS": sum(1 for r in module_rows if r["status"] == "PASS"),
        "WARNING": sum(1 for r in module_rows if r["status"] == "WARNING"),
        "FAIL": sum(1 for r in module_rows if r["status"] == "FAIL"),
    }
    md = [
        "# Main Pipeline Smoke Test",
        "",
        "This report is a dry-run check. No files were moved or deleted.",
        "",
        f"- PASS modules: {summary['PASS']}",
        f"- WARNING modules: {summary['WARNING']}",
        f"- FAIL modules: {summary['FAIL']}",
        "",
        markdown_table(module_rows, cols),
        "",
    ]
    (REPORT_DIR / "MAIN_PIPELINE_SMOKE_TEST.md").write_text("\n".join(md), encoding="utf-8")


def write_dependency_map(final_refs: set[str]) -> None:
    final_strategy_script = ROOT / "src" / "strategies" / "mature_regime_hedge_final.py"
    final_read_csv = find_read_csv_dependencies(final_strategy_script)
    md = [
        "# Dependency Map",
        "",
        "This map identifies files needed by the final README, final report, and final strategy implementation.",
        "",
        "## 1. Final Report Dependencies",
        "",
        "### Markdown image/link dependencies",
        "",
    ]
    md.extend(f"- `{p}` ({'exists' if exists(p) else 'missing'})" for p in sorted(final_refs))
    md.extend(
        [
            "",
            "### Final result tables",
            "",
            "- `results/09_final_strategy/mature_regime_hedge_final/performance_summary.csv`",
            "- `results/09_final_strategy/mature_regime_hedge_final/crisis_performance.csv`",
            "- `results/09_final_strategy/mature_regime_hedge_final/final_strategy_component_attribution.csv`",
            "- `results/09_final_strategy/mature_regime_hedge_final/final_strategy_decision_summary.csv`",
            "",
            "### Final strategy scripts",
            "",
            "- `src/allocation/final_strategy_backtest.py`",
            "- `src/strategies/mature_regime_hedge_final.py`",
            "",
            "## 2. Final Strategy Dependencies",
            "",
            "The final strategy script uses validated project panels and reconstructs missing fields only when required.",
            "",
            "### CSV paths detected in final strategy script",
            "",
        ]
    )
    md.extend(f"- `{p}`" for p in final_read_csv) if final_read_csv else md.append("- No direct literal `read_csv` path detected.")
    md.extend(
        [
            "",
            "### Required return and signal fields",
            "",
            "- Returns: `SPY_return`, `GOLD_return`, `CMDTY_FUT_return`, `IEF_return`, `CASH_return` or equivalents.",
            "- Stress signals: `VIX_ZSCORE_120D`, `D_CREDIT_SPREAD_20D`, `spy_drawdown_from_previous_high`, `monthly_either_state`, `SPY_CROSS_ABOVE_MA20`.",
            "- Commodity overlay: `CMDTY_RET60` or fields needed to reconstruct it.",
            "- Regime: `macro_regime_confirmed` with only `FLAT`, `STEEP`, `INVERTED` for the final strategy.",
            "",
            "## 3. Upstream Dependencies",
            "",
        ]
    )
    for module in PIPELINE_MODULES:
        md.append(f"### {module.module_name}")
        md.append("")
        md.append("Scripts:")
        md.extend(f"- `{p}`" for p in module.required_scripts)
        md.append("")
        md.append("Outputs:")
        md.extend(f"- `{p}`" for p in module.required_outputs)
        md.append("")
    md.extend(
        [
            "## 4. Exploratory Outputs",
            "",
            "Exploratory outputs are candidates for archive only if they are not referenced by README, FINAL_REPORT, or final strategy outputs.",
        ]
    )
    (REPORT_DIR / "DEPENDENCY_MAP.md").write_text("\n".join(md), encoding="utf-8")


def write_classification_reports(keep_rows: list[dict[str, str]], archive_rows: list[dict[str, str]], dnd_rows: list[dict[str, str]]) -> None:
    keep_cols = ["path", "reason", "exists"]
    move_cols = ["source_path", "target_path", "reason", "risk_level", "referenced_by_final_report", "safe_to_move"]
    (REPORT_DIR / "KEEP_LIST.md").write_text(
        "# Keep List\n\n" + markdown_table(keep_rows, keep_cols) + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "DO_NOT_DELETE_LIST.md").write_text(
        "# Do Not Delete List\n\n" + markdown_table(dnd_rows, keep_cols) + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "ARCHIVE_CANDIDATE_LIST.md").write_text(
        "# Archive Candidate List\n\nThis is a dry-run list. No files were moved by this script.\n\n"
        + markdown_table(archive_rows, move_cols)
        + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "ARCHIVE_MOVE_PLAN.md").write_text(
        "# Archive Move Plan\n\nThis is a dry-run plan. Actual archive moves require explicit approval.\n\n"
        + markdown_table(archive_rows, move_cols)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    module_rows = [evaluate_module(module) for module in PIPELINE_MODULES]
    write_smoke_reports(module_rows)

    readme_refs = set(parse_markdown_dependencies(ROOT / "README.md"))
    final_report_refs = set(parse_markdown_dependencies(ROOT / "reports" / "FINAL_REPORT.md"))
    final_refs = readme_refs | final_report_refs | set(FINAL_CORE_FILES)

    write_dependency_map(final_refs)
    keep_rows = collect_keep_paths(final_refs)
    dnd_rows = collect_do_not_delete(final_refs)
    archive_rows = scan_archive_candidates(final_refs)
    write_classification_reports(keep_rows, archive_rows, dnd_rows)

    fail_count = sum(1 for r in module_rows if r["status"] == "FAIL")
    warning_count = sum(1 for r in module_rows if r["status"] == "WARNING")
    high_risk_count = sum(1 for r in archive_rows if r["risk_level"] == "high")
    print("MAIN PIPELINE SMOKE TEST")
    print(f"status: {'PASS' if fail_count == 0 else 'FAIL'}")
    print(f"fail_modules: {fail_count}")
    print(f"warning_modules: {warning_count}")
    print(f"final_report_dependency_count: {len(final_refs)}")
    print(f"keep_count: {len(keep_rows)}")
    print(f"archive_candidate_count: {len(archive_rows)}")
    print(f"high_risk_archive_candidate_count: {high_risk_count}")
    print(f"recommend_actual_archive_move: {'yes' if fail_count == 0 else 'no'}")
    print("reports:")
    for path in [
        "reports/MAIN_PIPELINE_SMOKE_TEST.md",
        "reports/MAIN_PIPELINE_SMOKE_TEST.csv",
        "reports/DEPENDENCY_MAP.md",
        "reports/KEEP_LIST.md",
        "reports/ARCHIVE_CANDIDATE_LIST.md",
        "reports/DO_NOT_DELETE_LIST.md",
        "reports/ARCHIVE_MOVE_PLAN.md",
    ]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
