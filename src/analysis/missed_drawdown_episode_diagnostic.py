from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "missed_drawdown_episode_diagnostic"
FIGURE_DIR = ROOT / "figures" / "missed_drawdown_episode_diagnostic"

BASE_PANEL = ROOT / "results" / "spy_cash_stress_recovery_timing" / "daily_backtest_panel.csv"
FEATURE_SOURCES = [
    ROOT / "results" / "regime_hedge_steep_sell_ief" / "daily_backtest_panel.csv",
    ROOT / "results" / "reconstructed_regime_asset_behavior" / "reconstructed_regime_panel.csv",
    ROOT / "results" / "rule_diagnostics" / "rule_state_panel.csv",
    ROOT / "results" / "high_frequency_regime_diagnostics" / "high_frequency_regime_feature_panel.csv",
]

CONFIG = {
    "output_dir": str(OUTPUT_DIR),
    "figure_dir": str(FIGURE_DIR),
    "drawdown_thresholds": [-0.05, -0.08],
    "recovery_drawdown_level": -0.02,
    "max_episode_days": 126,
    "missed_grace_days": 10,
    "case_study_windows": {
        "2015_2016": ["2015-05-01", "2016-03-31"],
        "2019": ["2018-10-01", "2019-12-31"],
        "2022": ["2021-11-01", "2023-03-31"],
    },
}

EPISODE_TABLE = OUTPUT_DIR / "drawdown_episode_table.csv"
SUMMARY_TABLE = OUTPUT_DIR / "drawdown_episode_summary.csv"
SUMMARY_BY_REGIME = OUTPUT_DIR / "drawdown_episode_summary_by_regime.csv"
FEATURE_COMPARISON = OUTPUT_DIR / "missed_vs_captured_feature_comparison.csv"
TOP_MISSED = OUTPUT_DIR / "top_missed_drawdown_episodes.csv"
SIGNAL_COVERAGE = OUTPUT_DIR / "candidate_signal_coverage_on_missed_episodes.csv"
SIGNAL_SUMMARY = OUTPUT_DIR / "candidate_signal_summary.csv"
REPORT = OUTPUT_DIR / "MISSED_DRAWDOWN_EPISODE_DIAGNOSTIC.md"

FIG_TIMELINE = FIGURE_DIR / "missed_drawdown_timeline.png"
FIG_SCATTER = FIGURE_DIR / "missed_episode_scatter.png"
FIG_REGIME = FIGURE_DIR / "missed_by_regime_bar.png"
FIG_HEATMAP = FIGURE_DIR / "missed_vs_captured_features_heatmap.png"
FIG_SIGNAL = FIGURE_DIR / "candidate_signal_coverage_bar.png"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        warnings.warn(f"Could not read {path}: {exc}")
        return None
    if "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date")


def _first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def load_base_panel() -> pd.DataFrame:
    base = _read_csv(BASE_PANEL)
    if base is None:
        raise FileNotFoundError(f"Missing required base panel: {BASE_PANEL}")

    required = ["date", "macro_regime_confirmed", "VIX_LEVEL", "VIX_ZSCORE_120D"]
    missing = [c for c in required if c not in base.columns]
    if missing:
        raise ValueError(f"Base panel is missing required columns: {missing}")

    price_col = _first_existing(base, ["spy_price", "SPY_PRICE"])
    ret_col = _first_existing(base, ["spy_daily_return", "SPY_RETURN", "SPY_ret"])
    if price_col is None and ret_col is None:
        raise ValueError("Base panel must contain SPY price or SPY return.")
    if price_col is None:
        base["spy_daily_return"] = pd.to_numeric(base[ret_col], errors="coerce")
        base["spy_price"] = (1.0 + base["spy_daily_return"].fillna(0.0)).cumprod()
    else:
        base["spy_price"] = pd.to_numeric(base[price_col], errors="coerce")
        if ret_col is None:
            base["spy_daily_return"] = base["spy_price"].pct_change()
        else:
            base["spy_daily_return"] = pd.to_numeric(base[ret_col], errors="coerce").combine_first(base["spy_price"].pct_change())

    if "R3_weight_spy" not in base.columns and "R3_risk_state" not in base.columns:
        raise ValueError("Base panel must contain R3_risk_state or R3_weight_spy.")
    if "R8_weight_spy" not in base.columns and "R8_risk_state" not in base.columns:
        warnings.warn("R8 state/weight missing; R8 comparison will be limited.")
    return base


def merge_macro_market_features(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    wanted = [
        "CREDIT_SPREAD_BAA_AAA",
        "D_CREDIT_SPREAD_BAA_AAA",
        "D_CREDIT_SPREAD",
        "DGS1",
        "DGS10",
        "TERM_SPREAD_10Y_1Y",
        "IEF_RETURN",
        "GOLD_RETURN",
        "CMDTY_FUT_RETURN",
        "CASH_RETURN",
        "RF_DAILY",
        "growth_pc1",
        "inflation_pc1",
        "rate_pc1",
        "PC1",
        "PC2",
    ]
    for path in FEATURE_SOURCES:
        src = _read_csv(path)
        if src is None:
            continue
        # Long feature panel is not useful for direct daily joins here.
        if {"feature", "value"}.issubset(src.columns):
            continue
        keep = ["date"] + [c for c in wanted if c in src.columns and c not in out.columns]
        # Normalize common return aliases from the regime hedge panel.
        alias_map = {
            "IEF_ret": "IEF_RETURN",
            "GOLD_ret": "GOLD_RETURN",
            "CMDTY_ret": "CMDTY_FUT_RETURN",
            "CASH_ret": "CASH_RETURN",
        }
        for old, new in alias_map.items():
            if old in src.columns and new not in out.columns and new not in src.columns:
                src[new] = src[old]
                keep.append(new)
        keep = list(dict.fromkeys([c for c in keep if c in src.columns]))
        if len(keep) > 1:
            out = out.merge(src[keep], on="date", how="left")

    if "daily_rf" in out.columns and "CASH_RETURN" not in out.columns:
        out["CASH_RETURN"] = out["daily_rf"]
    if "RF_DAILY" in out.columns and "CASH_RETURN" not in out.columns:
        out["CASH_RETURN"] = out["RF_DAILY"]

    if "CREDIT_SPREAD_BAA_AAA" in out.columns:
        d_credit = pd.to_numeric(out["D_CREDIT_SPREAD"], errors="coerce") if "D_CREDIT_SPREAD" in out.columns else pd.Series(np.nan, index=out.index)
        d_credit_baa = pd.to_numeric(out["D_CREDIT_SPREAD_BAA_AAA"], errors="coerce") if "D_CREDIT_SPREAD_BAA_AAA" in out.columns else pd.Series(np.nan, index=out.index)
        out["D_CREDIT_SPREAD"] = d_credit.combine_first(d_credit_baa)
        out["D_CREDIT_SPREAD_20D"] = pd.to_numeric(out["CREDIT_SPREAD_BAA_AAA"], errors="coerce").diff(20)
        cs_roll = pd.to_numeric(out["CREDIT_SPREAD_BAA_AAA"], errors="coerce").rolling(120, min_periods=60)
        out["CREDIT_SPREAD_ZSCORE"] = (
            pd.to_numeric(out["CREDIT_SPREAD_BAA_AAA"], errors="coerce") - cs_roll.mean()
        ) / cs_roll.std(ddof=1).replace(0.0, np.nan)
    if "DGS10" in out.columns:
        out["DGS10_20D_CHANGE"] = pd.to_numeric(out["DGS10"], errors="coerce").diff(20)
    if "DGS1" in out.columns:
        out["DGS1_20D_CHANGE"] = pd.to_numeric(out["DGS1"], errors="coerce").diff(20)
    if "TERM_SPREAD_10Y_1Y" in out.columns:
        out["TERM_SPREAD_20D_CHANGE"] = pd.to_numeric(out["TERM_SPREAD_10Y_1Y"], errors="coerce").diff(20)
    out["VIX_5D_CHANGE"] = pd.to_numeric(out["VIX_LEVEL"], errors="coerce").pct_change(5)
    out["VIX_20D_CHANGE"] = pd.to_numeric(out["VIX_LEVEL"], errors="coerce").pct_change(20)

    if "CMDTY_FUT_RETURN" in out.columns:
        out["CMDTY_FUT_PRICE"] = (1.0 + pd.to_numeric(out["CMDTY_FUT_RETURN"], errors="coerce").fillna(0.0)).cumprod()
        out["CMDTY_FUT_MA60"] = out["CMDTY_FUT_PRICE"].rolling(60, min_periods=30).mean()
        out["CMDTY_FUT_DRAWDOWN"] = out["CMDTY_FUT_PRICE"] / out["CMDTY_FUT_PRICE"].cummax() - 1.0
    return out


def build_drawdown_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["previous_high"] = out["spy_price"].cummax()
    out["spy_drawdown_from_previous_high"] = out["spy_price"] / out["previous_high"] - 1.0
    out["R3_NORMAL"] = out["R3_weight_spy"].ge(0.5) if "R3_weight_spy" in out.columns else out["R3_risk_state"].astype(str).str.upper().eq("NORMAL")
    if "R8_weight_spy" in out.columns:
        out["R8_NORMAL"] = out["R8_weight_spy"].ge(0.5)
    elif "R8_risk_state" in out.columns:
        out["R8_NORMAL"] = out["R8_risk_state"].astype(str).str.upper().eq("NORMAL")
    else:
        out["R8_NORMAL"] = np.nan
    return out


def _forward_return(panel: pd.DataFrame, idx: int, days: int) -> float:
    end = min(idx + days, len(panel) - 1)
    if end <= idx:
        return np.nan
    return float(panel["spy_price"].iloc[end] / panel["spy_price"].iloc[idx] - 1.0)


def _forward_mdd(panel: pd.DataFrame, idx: int, days: int) -> float:
    end = min(idx + days, len(panel) - 1)
    path = panel["spy_price"].iloc[idx : end + 1]
    if len(path) < 2:
        return np.nan
    wealth = path / path.iloc[0]
    return float((wealth / wealth.cummax() - 1.0).min())


def extract_drawdown_episodes(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dd = panel["spy_drawdown_from_previous_high"].to_numpy()
    r3_risk = ~panel["R3_NORMAL"].fillna(False).to_numpy()
    r8_risk = ~panel["R8_NORMAL"].fillna(False).to_numpy()
    max_len = int(CONFIG["max_episode_days"])
    recovery_level = float(CONFIG["recovery_drawdown_level"])

    for threshold in CONFIG["drawdown_thresholds"]:
        i = 1
        eid = 0
        while i < len(panel):
            crossed = dd[i] <= threshold and dd[i - 1] > threshold and bool(panel["R3_NORMAL"].iloc[i])
            if not crossed:
                i += 1
                continue
            start = i
            max_end = min(start + max_len, len(panel) - 1)
            end = max_end
            for j in range(start + 1, max_end + 1):
                if dd[j] >= recovery_level or r3_risk[j]:
                    end = j
                    break
            sub = panel.iloc[start : end + 1]
            trough_local = int(np.argmin(sub["spy_drawdown_from_previous_high"].to_numpy()))
            trough = start + trough_local
            eid += 1
            r3_dates = panel.loc[start:end, "date"][r3_risk[start : end + 1]]
            r8_dates = panel.loc[start:end, "date"][r8_risk[start : end + 1]]
            r3_within5 = bool(r3_risk[start : min(start + 5, len(panel) - 1) + 1].any())
            r3_within10 = bool(r3_risk[start : min(start + 10, len(panel) - 1) + 1].any())
            r8_within5 = bool(r8_risk[start : min(start + 5, len(panel) - 1) + 1].any())
            r8_within10 = bool(r8_risk[start : min(start + 10, len(panel) - 1) + 1].any())
            rows.append(
                {
                    "threshold": threshold,
                    "episode_id": eid,
                    "start_idx": start,
                    "end_idx": end,
                    "trough_idx": trough,
                    "start_date": panel["date"].iloc[start],
                    "end_date": panel["date"].iloc[end],
                    "trough_date": panel["date"].iloc[trough],
                    "duration_days": int(end - start + 1),
                    "days_to_trough": int(trough - start),
                    "recovery_days_after_trough": int(end - trough),
                    "macro_regime_at_start": panel["macro_regime_confirmed"].iloc[start],
                    "dominant_macro_regime": sub["macro_regime_confirmed"].mode().iloc[0] if not sub["macro_regime_confirmed"].mode().empty else np.nan,
                    "monthly_either_state_at_start": panel.get("monthly_either_state", pd.Series(index=panel.index)).iloc[start],
                    "R3_state_at_start": panel.get("R3_risk_state", pd.Series(index=panel.index)).iloc[start],
                    "R8_state_at_start": panel.get("R8_risk_state", pd.Series(index=panel.index)).iloc[start],
                    "spy_drawdown_at_start": float(dd[start]),
                    "spy_max_drawdown_episode": float(sub["spy_drawdown_from_previous_high"].min()),
                    "spy_return_start_to_trough": float(panel["spy_price"].iloc[trough] / panel["spy_price"].iloc[start] - 1.0),
                    "spy_return_start_to_end": float(panel["spy_price"].iloc[end] / panel["spy_price"].iloc[start] - 1.0),
                    "spy_forward_return_21d_from_start": _forward_return(panel, start, 21),
                    "spy_forward_return_63d_from_start": _forward_return(panel, start, 63),
                    "spy_forward_mdd_21d_from_start": _forward_mdd(panel, start, 21),
                    "spy_forward_mdd_63d_from_start": _forward_mdd(panel, start, 63),
                    "R3_entered_risk_within_5d": r3_within5,
                    "R3_entered_risk_within_10d": r3_within10,
                    "R3_entered_risk_anytime": bool(r3_risk[start : end + 1].any()),
                    "R3_risk_entry_date": r3_dates.iloc[0] if len(r3_dates) else pd.NaT,
                    "missed_by_R3_10d": not r3_within10,
                    "missed_by_R3_full": not bool(r3_risk[start : end + 1].any()),
                    "R8_entered_risk_within_5d": r8_within5,
                    "R8_entered_risk_within_10d": r8_within10,
                    "R8_entered_risk_anytime": bool(r8_risk[start : end + 1].any()),
                    "R8_risk_entry_date": r8_dates.iloc[0] if len(r8_dates) else pd.NaT,
                    "missed_by_R8_10d": not r8_within10,
                    "missed_by_R8_full": not bool(r8_risk[start : end + 1].any()),
                }
            )
            i = end + 1
    return pd.DataFrame(rows)


def _val(panel: pd.DataFrame, col: str, idx: int) -> float:
    if col not in panel.columns:
        return np.nan
    return float(pd.to_numeric(panel[col], errors="coerce").iloc[idx])


def _cumret(panel: pd.DataFrame, col: str, start: int, end: int) -> float:
    if col not in panel.columns:
        return np.nan
    s = pd.to_numeric(panel[col].iloc[start : end + 1], errors="coerce").dropna()
    if s.empty:
        return np.nan
    return float((1.0 + s).prod() - 1.0)


def compute_episode_features(panel: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    out = episodes.copy()
    for n, ep in out.iterrows():
        start, trough, end = int(ep["start_idx"]), int(ep["trough_idx"]), int(ep["end_idx"])
        sub = panel.iloc[start : end + 1]
        for col in ["VIX_LEVEL", "VIX_ZSCORE_120D", "CREDIT_SPREAD_BAA_AAA", "DGS1", "DGS10", "TERM_SPREAD_10Y_1Y", "growth_pc1", "inflation_pc1", "rate_pc1"]:
            out.loc[n, f"{col}_start"] = _val(panel, col, start)
            out.loc[n, f"{col}_trough"] = _val(panel, col, trough)
        out.loc[n, "max_VIX_LEVEL_episode"] = pd.to_numeric(sub.get("VIX_LEVEL"), errors="coerce").max() if "VIX_LEVEL" in sub else np.nan
        out.loc[n, "max_VIX_ZSCORE_120D_episode"] = pd.to_numeric(sub.get("VIX_ZSCORE_120D"), errors="coerce").max() if "VIX_ZSCORE_120D" in sub else np.nan
        out.loc[n, "max_CREDIT_SPREAD_BAA_AAA_episode"] = pd.to_numeric(sub.get("CREDIT_SPREAD_BAA_AAA"), errors="coerce").max() if "CREDIT_SPREAD_BAA_AAA" in sub else np.nan
        out.loc[n, "VIX_20D_change_start"] = _val(panel, "VIX_20D_CHANGE", start)
        out.loc[n, "VIX_5D_change_start"] = _val(panel, "VIX_5D_CHANGE", start)
        out.loc[n, "D_CREDIT_SPREAD_start"] = _val(panel, "D_CREDIT_SPREAD", start)
        out.loc[n, "D_CREDIT_SPREAD_20D_start"] = _val(panel, "D_CREDIT_SPREAD_20D", start)
        out.loc[n, "max_D_CREDIT_SPREAD_20D_episode"] = pd.to_numeric(sub.get("D_CREDIT_SPREAD_20D"), errors="coerce").max() if "D_CREDIT_SPREAD_20D" in sub else np.nan
        out.loc[n, "DGS10_20D_CHANGE_start"] = _val(panel, "DGS10_20D_CHANGE", start)
        out.loc[n, "DGS1_20D_CHANGE_start"] = _val(panel, "DGS1_20D_CHANGE", start)
        for asset, ret_col in [("IEF", "IEF_RETURN"), ("GOLD", "GOLD_RETURN"), ("CMDTY_FUT", "CMDTY_FUT_RETURN"), ("CASH", "CASH_RETURN")]:
            out.loc[n, f"{asset}_return_start_to_trough"] = _cumret(panel, ret_col, start, trough)
            out.loc[n, f"{asset}_return_episode"] = _cumret(panel, ret_col, start, end)
    rename = {
        "VIX_LEVEL_start": "VIX_LEVEL_start",
        "VIX_ZSCORE_120D_start": "VIX_ZSCORE_120D_start",
    }
    return out.rename(columns=rename)


def compute_summary_tables(episodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for threshold, sub in episodes.groupby("threshold"):
        missed = sub[sub["missed_by_R3_10d"]]
        captured = sub[~sub["missed_by_R3_10d"]]
        rows.append(
            {
                "threshold": threshold,
                "total_episodes": len(sub),
                "missed_by_R3_10d_count": int(sub["missed_by_R3_10d"].sum()),
                "missed_by_R3_10d_rate": float(sub["missed_by_R3_10d"].mean()),
                "missed_by_R3_full_count": int(sub["missed_by_R3_full"].sum()),
                "missed_by_R3_full_rate": float(sub["missed_by_R3_full"].mean()),
                "missed_by_R8_10d_rate": float(sub["missed_by_R8_10d"].mean()),
                "missed_by_R8_full_rate": float(sub["missed_by_R8_full"].mean()),
                "avg_max_drawdown_all": float(sub["spy_max_drawdown_episode"].mean()),
                "avg_max_drawdown_missed_R3": float(missed["spy_max_drawdown_episode"].mean()) if not missed.empty else np.nan,
                "avg_max_drawdown_captured_R3": float(captured["spy_max_drawdown_episode"].mean()) if not captured.empty else np.nan,
                "median_max_drawdown_missed_R3": float(missed["spy_max_drawdown_episode"].median()) if not missed.empty else np.nan,
                "avg_duration_missed_R3": float(missed["duration_days"].mean()) if not missed.empty else np.nan,
                "avg_days_to_trough_missed_R3": float(missed["days_to_trough"].mean()) if not missed.empty else np.nan,
            }
        )
    overall = pd.DataFrame(rows)

    by_regime = (
        episodes.groupby(["threshold", "macro_regime_at_start"], dropna=False)
        .agg(
            episode_count=("episode_id", "count"),
            missed_by_R3_10d_rate=("missed_by_R3_10d", "mean"),
            missed_by_R3_full_rate=("missed_by_R3_full", "mean"),
            missed_by_R8_10d_rate=("missed_by_R8_10d", "mean"),
            avg_spy_max_drawdown_episode=("spy_max_drawdown_episode", "mean"),
            avg_VIX_ZSCORE_start=("VIX_ZSCORE_120D_start", "mean"),
            avg_max_VIX_ZSCORE_episode=("max_VIX_ZSCORE_120D_episode", "mean"),
            avg_D_CREDIT_SPREAD_20D_start=("D_CREDIT_SPREAD_20D_start", "mean"),
            avg_CREDIT_SPREAD_start=("CREDIT_SPREAD_BAA_AAA_start", "mean"),
            avg_TERM_SPREAD_start=("TERM_SPREAD_10Y_1Y_start", "mean"),
            avg_CMDTY_return_start_to_trough=("CMDTY_FUT_return_start_to_trough", "mean"),
            avg_IEF_return_start_to_trough=("IEF_return_start_to_trough", "mean"),
            avg_GOLD_return_start_to_trough=("GOLD_return_start_to_trough", "mean"),
        )
        .reset_index()
    )

    features = [
        "VIX_ZSCORE_120D_start",
        "VIX_LEVEL_start",
        "CREDIT_SPREAD_BAA_AAA_start",
        "D_CREDIT_SPREAD_20D_start",
        "TERM_SPREAD_10Y_1Y_start",
        "DGS1_start",
        "DGS10_start",
        "CMDTY_FUT_return_start_to_trough",
        "GOLD_return_start_to_trough",
        "IEF_return_start_to_trough",
        "growth_pc1_start",
        "inflation_pc1_start",
    ]
    comp_rows = []
    primary = episodes[episodes["threshold"].eq(-0.05)]
    for feature in features:
        if feature not in primary.columns:
            continue
        missed = pd.to_numeric(primary.loc[primary["missed_by_R3_10d"], feature], errors="coerce").dropna()
        captured = pd.to_numeric(primary.loc[~primary["missed_by_R3_10d"], feature], errors="coerce").dropna()
        pooled = np.sqrt((missed.var(ddof=1) / len(missed) if len(missed) > 1 else 0) + (captured.var(ddof=1) / len(captured) if len(captured) > 1 else 0))
        comp_rows.append(
            {
                "feature": feature,
                "mean_missed": missed.mean() if len(missed) else np.nan,
                "mean_captured": captured.mean() if len(captured) else np.nan,
                "median_missed": missed.median() if len(missed) else np.nan,
                "median_captured": captured.median() if len(captured) else np.nan,
                "difference": (missed.mean() - captured.mean()) if len(missed) and len(captured) else np.nan,
                "simple_t_stat": (missed.mean() - captured.mean()) / pooled if pooled and pooled > 0 else np.nan,
            }
        )
    comparison = pd.DataFrame(comp_rows)

    top = (
        primary[primary["missed_by_R3_10d"]]
        .sort_values("spy_max_drawdown_episode")
        .head(20)
        .drop(columns=["start_idx", "end_idx", "trough_idx"], errors="ignore")
    )
    return overall, by_regime, comparison, top


def build_candidate_signals(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["S1_PRICE_DD_5"] = out["spy_drawdown_from_previous_high"] <= -0.05
    out["S2_PRICE_DD_8"] = out["spy_drawdown_from_previous_high"] <= -0.08
    out["S3_CREDIT_WIDEN"] = out.get("D_CREDIT_SPREAD_20D", pd.Series(index=out.index, dtype=float)) > 0
    out["S4_DD5_AND_CREDIT_WIDEN"] = out["S1_PRICE_DD_5"] & out["S3_CREDIT_WIDEN"]
    if "CMDTY_FUT_PRICE" in out.columns and "CMDTY_FUT_MA60" in out.columns:
        out["S5_CMDTY_BELOW_MA60"] = out["CMDTY_FUT_PRICE"] < out["CMDTY_FUT_MA60"]
        out["S6_DD5_AND_CMDTY_WEAK"] = out["S1_PRICE_DD_5"] & out["S5_CMDTY_BELOW_MA60"]
    else:
        out["S5_CMDTY_BELOW_MA60"] = False
        out["S6_DD5_AND_CMDTY_WEAK"] = False
    out["S7_TERM_SPREAD_FALLING"] = out.get("TERM_SPREAD_20D_CHANGE", pd.Series(index=out.index, dtype=float)) < 0
    out["S8_DD5_AND_VIX_ABOVE_1_5Z"] = out["S1_PRICE_DD_5"] & (out["VIX_ZSCORE_120D"] >= 1.5)
    return out


def _signal_events(mask: pd.Series) -> pd.Series:
    return mask.fillna(False) & ~mask.fillna(False).shift(1, fill_value=False)


def analyze_candidate_signal_coverage(panel: pd.DataFrame, episodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    signals = [c for c in panel.columns if any(c.startswith(f"S{i}_") for i in range(1, 9))]
    missed = episodes[(episodes["threshold"].eq(-0.05)) & (episodes["missed_by_R3_10d"])]
    rows = []
    for _, ep in missed.iterrows():
        start = int(ep["start_idx"])
        lo = max(0, start - 5)
        hi = min(len(panel) - 1, start + 5)
        for sig in signals:
            window = panel.iloc[lo : hi + 1]
            fired = window[sig].fillna(False).astype(bool)
            if fired.any():
                first_idx = int(fired[fired].index[0])
                first_date = panel.loc[first_idx, "date"]
                days = first_idx - start
            else:
                first_date = pd.NaT
                days = np.nan
            rows.append(
                {
                    "threshold": ep["threshold"],
                    "episode_id": ep["episode_id"],
                    "start_date": ep["start_date"],
                    "signal_name": sig,
                    "signal_triggered_near_start": bool(fired.any()),
                    "signal_first_trigger_date": first_date,
                    "days_from_episode_start_to_signal": days,
                }
            )
    coverage = pd.DataFrame(rows)

    n_years = (panel["date"].iloc[-1] - panel["date"].iloc[0]).days / 365.25
    summary_rows = []
    for sig in signals:
        sub = coverage[coverage["signal_name"].eq(sig)]
        all_events = int(_signal_events(panel[sig]).sum())
        summary_rows.append(
            {
                "signal_name": sig,
                "missed_episode_coverage_rate": float(sub["signal_triggered_near_start"].mean()) if not sub.empty else np.nan,
                "avg_days_from_start_to_signal": float(sub.loc[sub["signal_triggered_near_start"], "days_from_episode_start_to_signal"].mean()) if not sub.empty else np.nan,
                "median_days_from_start_to_signal": float(sub.loc[sub["signal_triggered_near_start"], "days_from_episode_start_to_signal"].median()) if not sub.empty else np.nan,
                "false_positive_count_all_sample": max(0, all_events - int(sub["signal_triggered_near_start"].sum())) if not sub.empty else all_events,
                "events_per_year_all_sample": all_events / n_years if n_years > 0 else np.nan,
                "notes": "diagnostic only, not a trading rule",
            }
        )
    return coverage, pd.DataFrame(summary_rows).sort_values(["missed_episode_coverage_rate", "events_per_year_all_sample"], ascending=[False, True])


def write_case_studies(panel: pd.DataFrame) -> None:
    for name, (start, end) in CONFIG["case_study_windows"].items():
        sub = panel[panel["date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()
        cols = [
            "date",
            "spy_price",
            "spy_drawdown_from_previous_high",
            "R3_risk_state",
            "R8_risk_state",
            "stress_entry_signal",
            "VIX_LEVEL",
            "VIX_ZSCORE_120D",
            "macro_regime_confirmed",
            "monthly_either_state",
            "CREDIT_SPREAD_BAA_AAA",
            "D_CREDIT_SPREAD_20D",
            "TERM_SPREAD_10Y_1Y",
            "DGS1",
            "DGS10",
            "CMDTY_FUT_RETURN",
            "CMDTY_FUT_DRAWDOWN",
            "IEF_RETURN",
            "GOLD_RETURN",
        ]
        sub[[c for c in cols if c in sub.columns]].to_csv(OUTPUT_DIR / f"case_study_{name}.csv", index=False)


def _shade_episodes(ax: plt.Axes, episodes: pd.DataFrame) -> None:
    primary = episodes[episodes["threshold"].eq(-0.05)]
    for _, ep in primary.iterrows():
        color = "#d62728" if ep["missed_by_R3_10d"] else "#2ca02c"
        ax.axvspan(pd.Timestamp(ep["start_date"]), pd.Timestamp(ep["end_date"]), color=color, alpha=0.13, lw=0)


def plot_timeline(panel: pd.DataFrame, episodes: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1.2, 0.8]})
    nav = panel["spy_price"] / panel["spy_price"].iloc[0]
    axes[0].plot(panel["date"], nav, color="black", lw=1.2, label="SPY normalized")
    _shade_episodes(axes[0], episodes)
    axes[0].legend(loc="upper left")
    axes[0].set_title("Missed vs captured drawdown episodes")

    axes[1].plot(panel["date"], panel["spy_drawdown_from_previous_high"], color="#444444")
    for level in [-0.05, -0.08, -0.10]:
        axes[1].axhline(level, color="red", ls="--", lw=0.8)
    _shade_episodes(axes[1], episodes)
    axes[1].set_ylabel("SPY drawdown")

    axes[2].fill_between(panel["date"], 0, 1, where=~panel["R3_NORMAL"].fillna(False), color="#d62728", alpha=0.35, label="R3 risk")
    axes[2].fill_between(panel["date"], 1, 2, where=~panel["R8_NORMAL"].fillna(False), color="#ff7f0e", alpha=0.35, label="R8 risk")
    axes[2].set_yticks([0.5, 1.5])
    axes[2].set_yticklabels(["R3", "R8"])
    axes[2].legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_TIMELINE, dpi=160)
    plt.close(fig)


def plot_feature_comparison(comparison: pd.DataFrame) -> None:
    if comparison.empty:
        return
    data = comparison.set_index("feature")[["mean_missed", "mean_captured"]].copy()
    data = data.sub(data.mean(axis=1), axis=0).div(data.std(axis=1).replace(0, np.nan), axis=0)
    fig, ax = plt.subplots(figsize=(8, max(4, len(data) * 0.35)))
    sns.heatmap(data, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Missed vs captured standardized feature means")
    fig.tight_layout()
    fig.savefig(FIG_HEATMAP, dpi=160)
    plt.close(fig)


def plot_case_studies(panel: pd.DataFrame, episodes: pd.DataFrame) -> None:
    for name, (start, end) in CONFIG["case_study_windows"].items():
        sub = panel[panel["date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()
        eps = episodes[(episodes["threshold"].eq(-0.05)) & (episodes["start_date"].between(pd.Timestamp(start), pd.Timestamp(end)))]
        if sub.empty:
            continue
        fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
        axes[0].plot(sub["date"], sub["spy_price"] / sub["spy_price"].iloc[0], label="SPY", color="black")
        axes[0].plot(sub["date"], sub["spy_drawdown_from_previous_high"], label="SPY drawdown", color="red", alpha=0.8)
        axes[0].legend(loc="best")
        axes[1].plot(sub["date"], sub["VIX_ZSCORE_120D"], color="#9467bd")
        axes[1].axhline(3.0, color="red", ls="--")
        axes[1].set_ylabel("VIX z")
        if "CREDIT_SPREAD_BAA_AAA" in sub:
            axes[2].plot(sub["date"], sub["CREDIT_SPREAD_BAA_AAA"], label="credit spread", color="#1f77b4")
        if "D_CREDIT_SPREAD_20D" in sub:
            axes[2].plot(sub["date"], sub["D_CREDIT_SPREAD_20D"], label="20d credit chg", color="#ff7f0e")
        axes[2].legend(loc="best")
        if "CMDTY_FUT_PRICE" in sub:
            axes[3].plot(sub["date"], sub["CMDTY_FUT_PRICE"] / sub["CMDTY_FUT_PRICE"].iloc[0], label="CMDTY", color="#8c564b")
        if "CMDTY_FUT_DRAWDOWN" in sub:
            axes[3].plot(sub["date"], sub["CMDTY_FUT_DRAWDOWN"], label="CMDTY DD", color="red", alpha=0.7)
        axes[3].legend(loc="best")
        axes[4].fill_between(sub["date"], 0, 1, where=~sub["R3_NORMAL"].fillna(False), color="#d62728", alpha=0.35, label="R3 risk")
        axes[4].fill_between(sub["date"], 1, 2, where=~sub["R8_NORMAL"].fillna(False), color="#ff7f0e", alpha=0.35, label="R8 risk")
        axes[4].set_yticks([0.5, 1.5])
        axes[4].set_yticklabels(["R3", "R8"])
        axes[4].legend(loc="best")
        for ax in axes:
            for _, ep in eps.iterrows():
                ax.axvline(pd.Timestamp(ep["start_date"]), color="red", lw=1.0, alpha=0.8)
                ax.axvline(pd.Timestamp(ep["trough_date"]), color="black", lw=1.0, ls="--", alpha=0.7)
                ax.axvline(pd.Timestamp(ep["end_date"]), color="green", lw=1.0, alpha=0.6)
        fig.suptitle(f"Case study {name}")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"case_study_{name}.png", dpi=160)
        plt.close(fig)


def plot_results(panel: pd.DataFrame, episodes: pd.DataFrame, by_regime: pd.DataFrame, comparison: pd.DataFrame, signal_summary: pd.DataFrame) -> None:
    plot_timeline(panel, episodes)
    primary = episodes[episodes["threshold"].eq(-0.05)].copy()
    if not primary.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        x = primary.get("VIX_ZSCORE_120D_start", pd.Series(index=primary.index))
        y = primary.get("D_CREDIT_SPREAD_20D_start", primary.get("CREDIT_SPREAD_BAA_AAA_start", pd.Series(index=primary.index)))
        sizes = 1000 * primary["spy_max_drawdown_episode"].abs().clip(0.03, 0.2)
        sns.scatterplot(x=x, y=y, hue=primary["missed_by_R3_10d"], size=sizes, sizes=(40, 250), ax=ax)
        for _, ep in primary.sort_values("spy_max_drawdown_episode").head(8).iterrows():
            ax.text(ep.get("VIX_ZSCORE_120D_start", np.nan), ep.get("D_CREDIT_SPREAD_20D_start", np.nan), str(pd.Timestamp(ep["start_date"]).date())[:7], fontsize=8)
        ax.set_xlabel("VIX z-score at start")
        ax.set_ylabel("20d credit spread change at start")
        ax.set_title("Missed drawdown episode feature scatter")
        fig.tight_layout()
        fig.savefig(FIG_SCATTER, dpi=160)
        plt.close(fig)

    if not by_regime.empty:
        sub = by_regime[by_regime["threshold"].eq(-0.05)]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        sns.barplot(data=sub, x="macro_regime_at_start", y="episode_count", ax=axes[0], color="#4c78a8")
        sns.barplot(data=sub, x="macro_regime_at_start", y="missed_by_R3_10d_rate", ax=axes[1], color="#e45756")
        sns.barplot(data=sub, x="macro_regime_at_start", y="avg_spy_max_drawdown_episode", ax=axes[2], color="#72b7b2")
        for ax in axes:
            ax.tick_params(axis="x", rotation=35)
        axes[0].set_title("Episode count")
        axes[1].set_title("R3 missed rate")
        axes[2].set_title("Avg max drawdown")
        fig.tight_layout()
        fig.savefig(FIG_REGIME, dpi=160)
        plt.close(fig)

    plot_feature_comparison(comparison)
    if not signal_summary.empty:
        fig, ax1 = plt.subplots(figsize=(11, 5))
        sns.barplot(data=signal_summary, x="signal_name", y="missed_episode_coverage_rate", ax=ax1, color="#4c78a8")
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis="x", rotation=45)
        ax2 = ax1.twinx()
        ax2.plot(range(len(signal_summary)), signal_summary["events_per_year_all_sample"], color="#e45756", marker="o")
        ax2.set_ylabel("Events per year")
        ax1.set_title("Candidate signal coverage on missed episodes")
        fig.tight_layout()
        fig.savefig(FIG_SIGNAL, dpi=160)
        plt.close(fig)
    plot_case_studies(panel, episodes)


def write_markdown_report(summary: pd.DataFrame, by_regime: pd.DataFrame, comparison: pd.DataFrame, signal_summary: pd.DataFrame, top: pd.DataFrame, missing_cols: list[str]) -> None:
    primary = summary[summary["threshold"].eq(-0.05)].iloc[0] if not summary[summary["threshold"].eq(-0.05)].empty else pd.Series(dtype=float)
    top_signals = signal_summary.head(3)[["signal_name", "missed_episode_coverage_rate", "events_per_year_all_sample"]].to_markdown(index=False) if not signal_summary.empty else "_No signal summary available._"
    top_eps = top[["start_date", "trough_date", "macro_regime_at_start", "spy_max_drawdown_episode", "VIX_ZSCORE_120D_start", "D_CREDIT_SPREAD_20D_start"]].head(10).to_markdown(index=False) if not top.empty else "_No missed episodes._"
    lines = [
        "# Missed Drawdown Episode Diagnostic",
        "",
        "## Purpose",
        "",
        "This diagnostic identifies SPY drawdown episodes that occurred while the current stress-recovery strategy was still in NORMAL / 100% SPY. It does not add a new trigger or change allocation.",
        "",
        "## Method",
        "",
        "- SPY drawdown is measured from cumulative previous high.",
        "- Episodes start when R3 is NORMAL and SPY drawdown first crosses -5% or -8%.",
        "- Episodes end when drawdown recovers to -2%, R3 enters RISK, 126 trading days pass, or the sample ends.",
        "- `missed_by_R3_10d` means R3 did not enter RISK within 10 trading days after episode start.",
        "- `missed_by_R3_full` means R3 never entered RISK during the episode.",
        "",
        "## Overall Missed Drawdown Results",
        "",
        summary.to_markdown(index=False),
        "",
        f"For the -5% threshold, total episodes = {int(primary.get('total_episodes', 0))}, R3 missed-by-10d rate = {primary.get('missed_by_R3_10d_rate', np.nan):.1%}, R8 missed-by-10d rate = {primary.get('missed_by_R8_10d_rate', np.nan):.1%}.",
        "",
        "## Regime Analysis",
        "",
        by_regime.to_markdown(index=False),
        "",
        "## Feature Comparison",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Top Missed Episodes",
        "",
        top_eps,
        "",
        "## Candidate Signal Coverage",
        "",
        top_signals,
        "",
        "These candidate signals are diagnostic only. High coverage must be weighed against all-sample event frequency / false positive risk before any strategy test.",
        "",
        "## Case Studies",
        "",
        "- 2015-2016: inspect credit, commodity weakness, and whether VIX z-score remained below the current crisis threshold.",
        "- 2019: inspect whether the drawdown was a short policy/trade scare that may not justify a full risk exit.",
        "- 2022: inspect whether inflation/rate shock features are weakly captured by VIX-only fast stress.",
        "",
        "## Interpretation",
        "",
        "- VIX z-score >= 3.0 is a high-confidence crisis detector, not a medium correction detector.",
        "- Missed drawdowns should be reviewed for credit widening, commodity weakness, and rate/term-spread deterioration before adding any overlay.",
        "- The next diagnostic should test false alarm and forward drawdown behavior for 1-2 candidate signals, not immediately add them to the allocation model.",
        "",
        "## Missing Columns",
        "",
        ", ".join(missing_cols) if missing_cols else "None material.",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = build_candidate_signals(build_drawdown_features(merge_macro_market_features(load_base_panel())))
    important_optional = ["CREDIT_SPREAD_BAA_AAA", "D_CREDIT_SPREAD_20D", "DGS1", "DGS10", "TERM_SPREAD_10Y_1Y", "CMDTY_FUT_RETURN", "IEF_RETURN", "GOLD_RETURN"]
    missing_cols = [c for c in important_optional if c not in panel.columns]
    if missing_cols:
        warnings.warn(f"Missing optional diagnostic columns: {missing_cols}")

    episodes = compute_episode_features(panel, extract_drawdown_episodes(panel))
    summary, by_regime, comparison, top = compute_summary_tables(episodes)
    coverage, signal_summary = analyze_candidate_signal_coverage(panel, episodes)
    write_case_studies(panel)

    episodes.drop(columns=["start_idx", "end_idx", "trough_idx"], errors="ignore").to_csv(EPISODE_TABLE, index=False)
    summary.to_csv(SUMMARY_TABLE, index=False)
    by_regime.to_csv(SUMMARY_BY_REGIME, index=False)
    comparison.to_csv(FEATURE_COMPARISON, index=False)
    top.to_csv(TOP_MISSED, index=False)
    coverage.to_csv(SIGNAL_COVERAGE, index=False)
    signal_summary.to_csv(SIGNAL_SUMMARY, index=False)
    plot_results(panel, episodes, by_regime, comparison, signal_summary)
    write_markdown_report(summary, by_regime, comparison, signal_summary, top, missing_cols)

    primary = episodes[episodes["threshold"].eq(-0.05)]
    missed = primary[primary["missed_by_R3_10d"]]
    r3_rate = primary["missed_by_R3_10d"].mean() if not primary.empty else np.nan
    r8_rate = primary["missed_by_R8_10d"].mean() if not primary.empty else np.nan
    top5 = missed.sort_values("spy_max_drawdown_episode").head(5)
    def case_hit(start: str, end: str) -> bool:
        return bool(missed["start_date"].between(pd.Timestamp(start), pd.Timestamp(end)).any())
    common_regime = missed["macro_regime_at_start"].mode().iloc[0] if not missed.empty and not missed["macro_regime_at_start"].mode().empty else "n/a"
    print(f"1. -5% drawdown episodes: {len(primary)}")
    print(f"2. R3 missed_by_10d: {int(primary['missed_by_R3_10d'].sum()) if not primary.empty else 0} ({r3_rate:.1%})")
    print(f"3. R8 missed_by_10d: {int(primary['missed_by_R8_10d'].sum()) if not primary.empty else 0} ({r8_rate:.1%})")
    print("4. Top 5 missed episodes by max drawdown:")
    for _, row in top5.iterrows():
        print(f"   {pd.Timestamp(row['start_date']).date()} to {pd.Timestamp(row['trough_date']).date()} | {row['macro_regime_at_start']} | maxDD {row['spy_max_drawdown_episode']:.2%}")
    print(f"5. 2015-2016 missed: {case_hit('2015-05-01', '2016-03-31')}")
    print(f"6. 2019 missed: {case_hit('2018-10-01', '2019-12-31')}")
    print(f"7. 2022 missed: {case_hit('2021-11-01', '2023-03-31')}")
    print(f"8. Most common missed regime: {common_regime}")
    print("9. Top candidate signals by coverage:")
    for _, row in signal_summary.head(3).iterrows():
        print(f"   {row['signal_name']}: coverage {row['missed_episode_coverage_rate']:.1%}, events/year {row['events_per_year_all_sample']:.2f}")
    print("10. Recommendation: run a separate false-alarm / forward-drawdown diagnostic for the top 1-2 candidate signals before strategy changes.")
    print(f"Saved outputs: {OUTPUT_DIR} and {FIGURE_DIR}")


if __name__ == "__main__":
    main()
