"""Regime-aware risk parity / risk budget allocation backtest."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


CONFIG = {
    "output_dir": Path("results/regime_aware_risk_parity_allocation"),
    "figure_dir": Path("figures/regime_aware_risk_parity_allocation"),
    "one_way_cost_bps": 5,
    "monthly_rebalance": True,
    "risk_window": 120,
    "timing_backbone": "BACKBONE_V2_UPGRADED",
    "flat_asset_pool": ["SPY", "GOLD", "CMDTY_FUT"],
    "inverted_asset_pool": ["SPY", "GOLD"],
    "flat_target_budget": {"SPY": 0.50, "GOLD": 0.35, "CMDTY_FUT": 0.15},
    "inverted_target_budget": {"SPY": 0.70, "GOLD": 0.30},
    "flat_bounds": {"SPY": [0.30, 0.80], "GOLD": [0.10, 0.60], "CMDTY_FUT": [0.00, 0.25]},
    "inverted_bounds": {"SPY": [0.50, 0.85], "GOLD": [0.15, 0.50]},
    "fixed_rules": {
        "FLAT_RISK": {"SPY": 0.00, "GOLD": 1.00, "CMDTY_FUT": 0.00, "IEF": 0.00, "CASH": 0.00},
        "STEEP_NON_RISK": {"SPY": 1.00, "GOLD": 0.00, "CMDTY_FUT": 0.00, "IEF": 0.00, "CASH": 0.00},
        "STEEP_RISK": {"SPY": 0.00, "GOLD": 0.00, "CMDTY_FUT": 0.00, "IEF": 1.00, "CASH": 0.00},
        "FALLBACK_NON_RISK": {"SPY": 0.80, "GOLD": 0.00, "CMDTY_FUT": 0.00, "IEF": 0.00, "CASH": 0.20},
        "FALLBACK_RISK": {"SPY": 0.00, "GOLD": 0.00, "CMDTY_FUT": 0.00, "IEF": 0.00, "CASH": 1.00},
    },
}

PANEL_CANDIDATES = [
    Path("results/regime_aware_hedge_allocation_steep_mix/daily_backtest_panel.csv"),
    Path("results/regime_aware_hedge_allocation_steep_test/daily_backtest_panel.csv"),
    Path("results/regime_aware_hedge_allocation_v1/daily_backtest_panel.csv"),
    Path("results/spy_cash_backbone_upgrade_ablation/daily_backtest_panel.csv"),
]

ASSET_PANEL_CANDIDATES = [
    Path("results/regime_aware_hedge_allocation_steep_mix/daily_backtest_panel.csv"),
    Path("results/regime_aware_hedge_allocation_steep_test/daily_backtest_panel.csv"),
    Path("results/regime_aware_hedge_allocation_v1/daily_backtest_panel.csv"),
    Path("results/regime_hedge_steep_sell_ief/daily_backtest_panel.csv"),
    Path("results/reconstructed_regime_asset_behavior/reconstructed_regime_panel.csv"),
]

CRISIS_WINDOWS = {
    "2008_GFC": ("2007-10-01", "2009-06-30"),
    "2015_2016": ("2015-05-01", "2016-03-31"),
    "2018Q4": ("2018-10-01", "2019-01-31"),
    "COVID_2020": ("2020-02-01", "2020-06-30"),
    "2022": ("2021-11-01", "2023-03-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2025_PULLBACK": ("2025-01-01", "2025-12-31"),
    "2024_2026": ("2024-01-01", "2026-12-31"),
}

ASSETS = ["SPY", "GOLD", "CMDTY_FUT", "IEF", "CASH"]
STRATEGIES = [
    "SPY_BUY_HOLD",
    "BACKBONE_V2_SPY_CASH",
    "REGIME_HEDGE_FIXED_BASELINE",
    "REGIME_HEDGE_INV_VOL",
    "REGIME_HEDGE_TARGET_RB_INV_VOL",
    "REGIME_HEDGE_ERC",
    "REGIME_HEDGE_V1_ORIGINAL",
    "STATIC_60_30_10",
    "STATIC_70_20_10",
    "STATIC_60_30_10_CMDTY",
]


def ensure_dirs() -> None:
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["figure_dir"].mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        for col in ["DATE", "Date", "observation_date"]:
            if col in df.columns:
                df = df.rename(columns={col: "date"})
                break
    if "date" not in df.columns:
        raise ValueError(f"No date column found in {path}")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def _merge_missing(base: pd.DataFrame, other: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in base.columns and c in other.columns]
    if not missing:
        return base
    return base.merge(other[["date"] + missing], on="date", how="left")


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    full = {asset: float(weights.get(asset, 0.0)) for asset in ASSETS}
    total = sum(full.values())
    if total <= 0:
        raise ValueError("Zero-sum weight set.")
    return {k: v / total for k, v in full.items()}


def _max_drawdown(ret: pd.Series) -> float:
    nav = (1 + ret.fillna(0.0)).cumprod()
    if nav.empty:
        return np.nan
    return float((nav / nav.cummax() - 1).min())


def _is_first_trading_day_of_month(dates: pd.Series) -> pd.Series:
    period = dates.dt.to_period("M")
    return period.ne(period.shift(1, fill_value=period.iloc[0]))


def load_base_panel() -> pd.DataFrame:
    frames: List[Tuple[Path, pd.DataFrame]] = []
    for path in PANEL_CANDIDATES:
        if path.exists():
            frames.append((path, _read_csv(path)))
    if not frames:
        raise FileNotFoundError("No base panel found.")
    panel = frames[0][1].copy()
    print(f"Loaded primary panel: {frames[0][0]}")
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
        "timing_state",
        "cross_state",
        "entry_reason",
        "BACKBONE_V2_SPY_CASH_weight_SPY",
        "BACKBONE_V2_SPY_CASH_weight_CASH",
        "BACKBONE_V2_SPY_CASH_return",
        "BACKBONE_V2_SPY_CASH_nav",
        "SPY_return",
        "GOLD_return",
        "IEF_return",
        "CASH_return",
        "CMDTY_FUT_return",
        "CMDTY_RETURN",
        "CMDTY_FUT_RETURN",
        "CMDTY_return",
        "CMDTY_ret",
    ]
    for _, df in frames[1:]:
        panel = _merge_missing(panel, df, needed)

    if "spy_drawdown_from_previous_high" not in panel.columns:
        panel["spy_drawdown_from_previous_high"] = panel["spy_price"] / panel["spy_price"].cummax() - 1
    if "VIX_ZSCORE_120D" not in panel.columns:
        roll = panel["VIX_LEVEL"].rolling(CONFIG["risk_window"])
        panel["VIX_ZSCORE_120D"] = (panel["VIX_LEVEL"] - roll.mean()) / roll.std(ddof=0)
    if "D_CREDIT_SPREAD_20D" not in panel.columns:
        panel["D_CREDIT_SPREAD_20D"] = panel["CREDIT_SPREAD_BAA_AAA"] - panel["CREDIT_SPREAD_BAA_AAA"].shift(20)
    if "SPY_MA20" not in panel.columns:
        panel["SPY_MA20"] = panel["spy_price"].rolling(20).mean()
    if "SPY_CROSS_ABOVE_MA20" not in panel.columns:
        panel["SPY_CROSS_ABOVE_MA20"] = (panel["spy_price"] > panel["SPY_MA20"]) & (
            panel["spy_price"].shift(1) <= panel["SPY_MA20"].shift(1)
        )
    panel["spy_daily_return"] = pd.to_numeric(panel["spy_daily_return"], errors="coerce")
    panel["daily_rf"] = pd.to_numeric(panel["daily_rf"], errors="coerce").fillna(0.0)
    panel["macro_regime_confirmed"] = panel["macro_regime_confirmed"].fillna("NEUTRAL").astype(str)
    panel["monthly_either_state"] = panel["monthly_either_state"].fillna("UNKNOWN").astype(str)
    return panel


def load_asset_returns(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    if "SPY_return" not in out.columns:
        out["SPY_return"] = pd.to_numeric(out["spy_daily_return"], errors="coerce")
    if "CASH_return" not in out.columns:
        out["CASH_return"] = pd.to_numeric(out["daily_rf"], errors="coerce").fillna(0.0)

    for path in ASSET_PANEL_CANDIDATES:
        if not path.exists():
            continue
        src = _read_csv(path)
        merge_cols = ["date"]
        rename_map: Dict[str, str] = {}
        mapping = [
            (["GOLD_return", "GOLD_RETURN", "GLD_return", "GLD_RETURN"], "GOLD_return"),
            (["IEF_return", "IEF_RETURN"], "IEF_return"),
            (["CMDTY_FUT_return", "CMDTY_FUT_RETURN", "CMDTY_return", "CMDTY_RETURN", "CMDTY_ret"], "CMDTY_FUT_return"),
        ]
        for candidates, final in mapping:
            col = next((c for c in candidates if c in src.columns), None)
            if col and final not in out.columns:
                merge_cols.append(col)
                rename_map[col] = final
        if len(merge_cols) > 1:
            out = out.merge(src[merge_cols].rename(columns=rename_map), on="date", how="left")

    required = ["SPY_return", "GOLD_return", "CMDTY_FUT_return", "IEF_return", "CASH_return"]
    for col in required:
        if col not in out.columns:
            raise ValueError(f"Missing required asset return column: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce")
    valid = out[required].notna().all(axis=1)
    if not valid.any():
        raise ValueError("No overlapping SPY/GOLD/CMDTY_FUT/IEF/CASH sample.")
    start_idx = valid.idxmax()
    out = out.loc[start_idx:].reset_index(drop=True)
    for asset in ASSETS:
        out[f"{asset}_NAV"] = (1 + out[f"{asset}_return"].fillna(0.0)).cumprod()
    return out


def build_backbone_v2_state(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    if "BACKBONE_V2_SPY_CASH_weight_SPY" in df.columns:
        inferred = pd.Series(
            np.where(pd.to_numeric(df["BACKBONE_V2_SPY_CASH_weight_SPY"], errors="coerce") >= 0.5, "NON_RISK", "RISK"),
            index=df.index,
        )
        if "timing_state" not in df.columns:
            df["timing_state"] = inferred
        else:
            df["timing_state"] = df["timing_state"].fillna(inferred)
    elif "BACKBONE_V2_UPGRADED_weight_spy" in df.columns:
        df["BACKBONE_V2_SPY_CASH_weight_SPY"] = pd.to_numeric(df["BACKBONE_V2_UPGRADED_weight_spy"], errors="coerce").fillna(1.0)
        df["BACKBONE_V2_SPY_CASH_weight_CASH"] = pd.to_numeric(
            df.get("BACKBONE_V2_UPGRADED_weight_cash", 1 - df["BACKBONE_V2_SPY_CASH_weight_SPY"]),
            errors="coerce",
        ).fillna(1 - df["BACKBONE_V2_SPY_CASH_weight_SPY"])
        df["BACKBONE_V2_SPY_CASH_return"] = pd.to_numeric(df["BACKBONE_V2_UPGRADED_return"], errors="coerce").fillna(0.0)
        df["BACKBONE_V2_SPY_CASH_nav"] = (1 + df["BACKBONE_V2_SPY_CASH_return"]).cumprod()
        df["timing_state"] = inferred
    else:
        raise ValueError("Missing BACKBONE_V2 timing weights.")

    if "cross_state" not in df.columns:
        df["cross_state"] = df["macro_regime_confirmed"] + "_" + df["timing_state"]
    if "entry_reason" not in df.columns:
        df["entry_reason"] = ""
    return df


def compute_rolling_volatility(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    for asset in ["SPY", "GOLD", "CMDTY_FUT", "IEF"]:
        out[f"{asset}_vol_120d"] = out[f"{asset}_return"].rolling(CONFIG["risk_window"]).std(ddof=0) * math.sqrt(252)
    return out


def compute_inverse_vol_weights(vols: pd.Series, assets: List[str]) -> Optional[Dict[str, float]]:
    vals = pd.to_numeric(vols.reindex(assets), errors="coerce")
    if vals.isna().any() or (vals <= 0).any():
        return None
    raw = 1.0 / vals
    raw = raw / raw.sum()
    return {asset: float(raw.loc[asset]) for asset in assets}


def compute_target_rb_inverse_vol_weights(vols: pd.Series, budgets: Dict[str, float], assets: List[str]) -> Optional[Dict[str, float]]:
    vals = pd.to_numeric(vols.reindex(assets), errors="coerce")
    if vals.isna().any() or (vals <= 0).any():
        return None
    raw = pd.Series({asset: budgets[asset] / vals.loc[asset] for asset in assets})
    raw = raw / raw.sum()
    return {asset: float(raw.loc[asset]) for asset in assets}


def _erc_objective(weights: np.ndarray, cov: np.ndarray, target: np.ndarray) -> float:
    sigma = math.sqrt(max(weights @ cov @ weights, 1e-16))
    mrc = cov @ weights / sigma
    rc = weights * mrc
    rc_pct = rc / sigma
    return float(np.sum((rc_pct - target) ** 2))


def _risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    sigma = math.sqrt(max(weights @ cov @ weights, 1e-16))
    mrc = cov @ weights / sigma
    rc = weights * mrc
    return rc / sigma


def compute_erc_weights(
    cov: pd.DataFrame,
    assets: List[str],
    bounds_map: Dict[str, List[float]],
    target_budget: Optional[Dict[str, float]] = None,
) -> Tuple[Optional[Dict[str, float]], str]:
    cov = cov.reindex(index=assets, columns=assets)
    if cov.isna().any().any():
        return None, "covariance_nan"
    cov_vals = cov.to_numpy(dtype=float)
    if np.any(np.diag(cov_vals) <= 0):
        return None, "non_positive_variance"
    n = len(assets)
    target = np.array([target_budget[a] for a in assets], dtype=float) if target_budget is not None else np.repeat(1.0 / n, n)
    x0 = target.copy()
    bounds = [tuple(bounds_map[a]) for a in assets]
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    try:
        res = minimize(
            _erc_objective,
            x0=x0,
            args=(cov_vals, target),
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 500, "ftol": 1e-12},
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"optimizer_exception:{type(exc).__name__}"
    if not res.success:
        return None, f"optimizer_failed:{res.message}"
    w = np.clip(res.x, 0, None)
    if w.sum() <= 0:
        return None, "zero_solution"
    w = w / w.sum()
    return {asset: float(val) for asset, val in zip(assets, w)}, ""


def _full_weight_template(weights: Dict[str, float]) -> Dict[str, float]:
    return {asset: float(weights.get(asset, 0.0)) for asset in ASSETS}


def _apply_pool_weights(pool_weights: Dict[str, float]) -> Dict[str, float]:
    out = {asset: 0.0 for asset in ASSETS}
    for asset, weight in pool_weights.items():
        out[asset] = float(weight)
    total = sum(out.values())
    if total <= 0:
        raise ValueError("Pool weight sum must be positive.")
    return {asset: weight / total for asset, weight in out.items()}


def _strategy_fixed_weights(name: str, regime: str, timing_state: str) -> Dict[str, float]:
    if name == "REGIME_HEDGE_V1_ORIGINAL":
        rules = {
            "INVERTED": {"SPY": 0.70, "GOLD": 0.20, "CMDTY_FUT": 0.0, "IEF": 0.0, "CASH": 0.10},
            "FLAT_NON_RISK": {"SPY": 0.60, "GOLD": 0.40, "CMDTY_FUT": 0.0, "IEF": 0.0, "CASH": 0.0},
            "FLAT_RISK": {"SPY": 0.0, "GOLD": 1.0, "CMDTY_FUT": 0.0, "IEF": 0.0, "CASH": 0.0},
            "STEEP_NON_RISK": {"SPY": 0.90, "GOLD": 0.0, "CMDTY_FUT": 0.0, "IEF": 0.10, "CASH": 0.0},
            "STEEP_RISK": {"SPY": 0.0, "GOLD": 0.0, "CMDTY_FUT": 0.0, "IEF": 1.0, "CASH": 0.0},
            "FALLBACK_NON_RISK": {"SPY": 0.80, "GOLD": 0.0, "CMDTY_FUT": 0.0, "IEF": 0.0, "CASH": 0.20},
            "FALLBACK_RISK": {"SPY": 0.0, "GOLD": 0.0, "CMDTY_FUT": 0.0, "IEF": 0.0, "CASH": 1.0},
        }
    elif name == "REGIME_HEDGE_FIXED_BASELINE":
        rules = {
            "INVERTED": {"SPY": 0.75, "GOLD": 0.25, "CMDTY_FUT": 0.0, "IEF": 0.0, "CASH": 0.0},
            "FLAT_NON_RISK": {"SPY": 0.60, "GOLD": 0.30, "CMDTY_FUT": 0.10, "IEF": 0.0, "CASH": 0.0},
            "FLAT_RISK": CONFIG["fixed_rules"]["FLAT_RISK"],
            "STEEP_NON_RISK": CONFIG["fixed_rules"]["STEEP_NON_RISK"],
            "STEEP_RISK": CONFIG["fixed_rules"]["STEEP_RISK"],
            "FALLBACK_NON_RISK": CONFIG["fixed_rules"]["FALLBACK_NON_RISK"],
            "FALLBACK_RISK": CONFIG["fixed_rules"]["FALLBACK_RISK"],
        }
    else:
        raise ValueError(f"Unsupported fixed strategy: {name}")

    if regime == "INVERTED":
        key = "INVERTED"
    elif regime == "FLAT":
        key = f"FLAT_{timing_state}"
    elif regime == "STEEP":
        key = f"STEEP_{timing_state}"
    else:
        key = "FALLBACK_RISK" if timing_state == "RISK" else "FALLBACK_NON_RISK"
    return _normalize_weights(rules[key])


def build_allocation_rules(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = panel.copy()
    first_month = _is_first_trading_day_of_month(df["date"])

    fl_inv_cols = {asset: [] for asset in CONFIG["flat_asset_pool"]}
    fl_rb_cols = {asset: [] for asset in CONFIG["flat_asset_pool"]}
    fl_erc_cols = {asset: [] for asset in CONFIG["flat_asset_pool"]}
    inv_inv_cols = {asset: [] for asset in CONFIG["inverted_asset_pool"]}
    inv_rb_cols = {asset: [] for asset in CONFIG["inverted_asset_pool"]}
    inv_erc_cols = {asset: [] for asset in CONFIG["inverted_asset_pool"]}
    erc_flags: List[bool] = []
    erc_reasons: List[str] = []
    risk_contrib_rows: List[Dict[str, object]] = []

    last_flat_inv = _apply_pool_weights({"SPY": 0.60, "GOLD": 0.30, "CMDTY_FUT": 0.10})
    last_flat_rb = _apply_pool_weights({"SPY": 0.60, "GOLD": 0.30, "CMDTY_FUT": 0.10})
    last_flat_erc = _apply_pool_weights({"SPY": 0.60, "GOLD": 0.30, "CMDTY_FUT": 0.10})
    last_inv_inv = _apply_pool_weights({"SPY": 0.75, "GOLD": 0.25})
    last_inv_rb = _apply_pool_weights({"SPY": 0.75, "GOLD": 0.25})
    last_inv_erc = _apply_pool_weights({"SPY": 0.75, "GOLD": 0.25})
    prev_cross_state = None

    for i, row in df.iterrows():
        regime = str(row["macro_regime_confirmed"])
        timing_state = str(row["timing_state"])
        cross_state = "INVERTED" if regime == "INVERTED" else (f"{regime}_{timing_state}" if regime in {"FLAT", "STEEP"} else f"{regime}_{timing_state}")
        recalc = bool(first_month.iloc[i]) or (i > 0 and cross_state != prev_cross_state)
        prev_cross_state = cross_state

        fallback_flag = False
        fallback_reason = ""
        if recalc and cross_state == "FLAT_NON_RISK" and i >= CONFIG["risk_window"] - 1:
            window = df.loc[i - CONFIG["risk_window"] + 1 : i, [f"{a}_return" for a in CONFIG["flat_asset_pool"]]].copy()
            window.columns = CONFIG["flat_asset_pool"]
            vols = window.std(ddof=0) * math.sqrt(252)
            inv_w = compute_inverse_vol_weights(vols, CONFIG["flat_asset_pool"])
            rb_w = compute_target_rb_inverse_vol_weights(vols, CONFIG["flat_target_budget"], CONFIG["flat_asset_pool"])
            cov = window.cov() * 252
            erc_w, erc_reason = compute_erc_weights(cov, CONFIG["flat_asset_pool"], CONFIG["flat_bounds"])
            if inv_w is not None:
                last_flat_inv = _apply_pool_weights(inv_w)
            if rb_w is not None:
                last_flat_rb = _apply_pool_weights(rb_w)
            if erc_w is not None:
                last_flat_erc = _apply_pool_weights(erc_w)
            else:
                fallback_flag = True
                fallback_reason = erc_reason or "erc_failed"
                last_flat_erc = last_flat_rb if rb_w is not None else last_flat_inv

            for name, weights, target in [
                ("FLAT_INV_VOL", last_flat_inv, None),
                ("FLAT_TARGET_RB", last_flat_rb, CONFIG["flat_target_budget"]),
                ("FLAT_ERC", last_flat_erc, {a: 1 / len(CONFIG["flat_asset_pool"]) for a in CONFIG["flat_asset_pool"]} if not fallback_flag else None),
            ]:
                w = np.array([weights[a] for a in CONFIG["flat_asset_pool"]], dtype=float)
                sigma = math.sqrt(max(float(w @ cov.to_numpy() @ w), 1e-16))
                rc_pct = _risk_contributions(w, cov.to_numpy())
                row_rc = {"regime_state": "FLAT_NON_RISK", "method": name}
                for asset, rc in zip(CONFIG["flat_asset_pool"], rc_pct):
                    row_rc[f"avg_RC_{asset}"] = rc
                    row_rc[f"target_RC_{asset}"] = target[asset] if target is not None and asset in target else np.nan
                row_rc["tracking_error_to_target_RC"] = float(
                    np.sqrt(
                        np.nansum(
                            [
                                (row_rc[f"avg_RC_{asset}"] - row_rc[f"target_RC_{asset}"]) ** 2
                                for asset in CONFIG["flat_asset_pool"]
                                if not np.isnan(row_rc[f"target_RC_{asset}"])
                            ]
                        )
                    )
                ) if target is not None else np.nan
                row_rc["avg_realized_vol"] = sigma
                risk_contrib_rows.append(row_rc)
        elif recalc and cross_state == "INVERTED" and i >= CONFIG["risk_window"] - 1:
            window = df.loc[i - CONFIG["risk_window"] + 1 : i, [f"{a}_return" for a in CONFIG["inverted_asset_pool"]]].copy()
            window.columns = CONFIG["inverted_asset_pool"]
            vols = window.std(ddof=0) * math.sqrt(252)
            inv_w = compute_inverse_vol_weights(vols, CONFIG["inverted_asset_pool"])
            rb_w = compute_target_rb_inverse_vol_weights(vols, CONFIG["inverted_target_budget"], CONFIG["inverted_asset_pool"])
            cov = window.cov() * 252
            erc_w, erc_reason = compute_erc_weights(cov, CONFIG["inverted_asset_pool"], CONFIG["inverted_bounds"])
            if inv_w is not None:
                last_inv_inv = _apply_pool_weights(inv_w)
            if rb_w is not None:
                last_inv_rb = _apply_pool_weights(rb_w)
            if erc_w is not None:
                last_inv_erc = _apply_pool_weights(erc_w)
            else:
                fallback_flag = True
                fallback_reason = erc_reason or "erc_failed"
                last_inv_erc = last_inv_rb if rb_w is not None else last_inv_inv

            for name, weights, target in [
                ("INVERTED_INV_VOL", last_inv_inv, None),
                ("INVERTED_TARGET_RB", last_inv_rb, CONFIG["inverted_target_budget"]),
                ("INVERTED_ERC", last_inv_erc, {a: 1 / len(CONFIG["inverted_asset_pool"]) for a in CONFIG["inverted_asset_pool"]} if not fallback_flag else None),
            ]:
                w = np.array([weights[a] for a in CONFIG["inverted_asset_pool"]], dtype=float)
                sigma = math.sqrt(max(float(w @ cov.to_numpy() @ w), 1e-16))
                rc_pct = _risk_contributions(w, cov.to_numpy())
                row_rc = {"regime_state": "INVERTED", "method": name}
                for asset, rc in zip(CONFIG["inverted_asset_pool"], rc_pct):
                    row_rc[f"avg_RC_{asset}"] = rc
                    row_rc[f"target_RC_{asset}"] = target[asset] if target is not None and asset in target else np.nan
                row_rc["tracking_error_to_target_RC"] = float(
                    np.sqrt(
                        np.nansum(
                            [
                                (row_rc[f"avg_RC_{asset}"] - row_rc[f"target_RC_{asset}"]) ** 2
                                for asset in CONFIG["inverted_asset_pool"]
                                if not np.isnan(row_rc[f"target_RC_{asset}"])
                            ]
                        )
                    )
                ) if target is not None else np.nan
                row_rc["avg_realized_vol"] = sigma
                risk_contrib_rows.append(row_rc)

        for asset in CONFIG["flat_asset_pool"]:
            fl_inv_cols[asset].append(last_flat_inv.get(asset, 0.0) if cross_state == "FLAT_NON_RISK" else np.nan)
            fl_rb_cols[asset].append(last_flat_rb.get(asset, 0.0) if cross_state == "FLAT_NON_RISK" else np.nan)
            fl_erc_cols[asset].append(last_flat_erc.get(asset, 0.0) if cross_state == "FLAT_NON_RISK" else np.nan)
        for asset in CONFIG["inverted_asset_pool"]:
            inv_inv_cols[asset].append(last_inv_inv.get(asset, 0.0) if cross_state == "INVERTED" else np.nan)
            inv_rb_cols[asset].append(last_inv_rb.get(asset, 0.0) if cross_state == "INVERTED" else np.nan)
            inv_erc_cols[asset].append(last_inv_erc.get(asset, 0.0) if cross_state == "INVERTED" else np.nan)
        erc_flags.append(fallback_flag)
        erc_reasons.append(fallback_reason)

    for asset in CONFIG["flat_asset_pool"]:
        df[f"FLAT_INV_VOL_weight_{asset}"] = fl_inv_cols[asset]
        df[f"FLAT_TARGET_RB_weight_{asset}"] = fl_rb_cols[asset]
        df[f"FLAT_ERC_weight_{asset}"] = fl_erc_cols[asset]
    for asset in CONFIG["inverted_asset_pool"]:
        df[f"INVERTED_INV_VOL_weight_{asset}"] = inv_inv_cols[asset]
        df[f"INVERTED_TARGET_RB_weight_{asset}"] = inv_rb_cols[asset]
        df[f"INVERTED_ERC_weight_{asset}"] = inv_erc_cols[asset]
    df["ERC_fallback_flag"] = erc_flags
    df["ERC_fallback_reason"] = erc_reasons

    strategy_target_frames: Dict[str, pd.DataFrame] = {}
    for strategy in STRATEGIES:
        weights_rows = []
        for _, row in df.iterrows():
            regime = str(row["macro_regime_confirmed"])
            timing_state = str(row["timing_state"])
            if strategy == "SPY_BUY_HOLD":
                weights_rows.append(_normalize_weights({"SPY": 1.0, "GOLD": 0.0, "CMDTY_FUT": 0.0, "IEF": 0.0, "CASH": 0.0}))
                continue
            if strategy == "BACKBONE_V2_SPY_CASH":
                weights_rows.append(_normalize_weights({"SPY": 1.0 if timing_state == "NON_RISK" else 0.0, "CASH": 0.0 if timing_state == "NON_RISK" else 1.0, "GOLD": 0.0, "CMDTY_FUT": 0.0, "IEF": 0.0}))
                continue
            if strategy == "REGIME_HEDGE_V1_ORIGINAL":
                weights_rows.append(_strategy_fixed_weights("REGIME_HEDGE_V1_ORIGINAL", regime, timing_state))
                continue
            if strategy == "STATIC_60_30_10":
                weights_rows.append(_normalize_weights({"SPY": 0.60, "GOLD": 0.30, "IEF": 0.10, "CMDTY_FUT": 0.0, "CASH": 0.0}))
                continue
            if strategy == "STATIC_70_20_10":
                weights_rows.append(_normalize_weights({"SPY": 0.70, "GOLD": 0.20, "IEF": 0.0, "CMDTY_FUT": 0.0, "CASH": 0.10}))
                continue
            if strategy == "STATIC_60_30_10_CMDTY":
                weights_rows.append(_normalize_weights({"SPY": 0.60, "GOLD": 0.30, "IEF": 0.0, "CMDTY_FUT": 0.10, "CASH": 0.0}))
                continue

            if regime == "FLAT":
                if timing_state == "RISK":
                    weights_rows.append(_normalize_weights(CONFIG["fixed_rules"]["FLAT_RISK"]))
                else:
                    if strategy == "REGIME_HEDGE_FIXED_BASELINE":
                        weights_rows.append(_normalize_weights({"SPY": 0.60, "GOLD": 0.30, "CMDTY_FUT": 0.10, "IEF": 0.0, "CASH": 0.0}))
                    elif strategy == "REGIME_HEDGE_INV_VOL":
                        weights_rows.append(_apply_pool_weights({a: row.get(f"FLAT_INV_VOL_weight_{a}", 0.0) for a in CONFIG["flat_asset_pool"]}))
                    elif strategy == "REGIME_HEDGE_TARGET_RB_INV_VOL":
                        weights_rows.append(_apply_pool_weights({a: row.get(f"FLAT_TARGET_RB_weight_{a}", 0.0) for a in CONFIG["flat_asset_pool"]}))
                    elif strategy == "REGIME_HEDGE_ERC":
                        weights_rows.append(_apply_pool_weights({a: row.get(f"FLAT_ERC_weight_{a}", 0.0) for a in CONFIG["flat_asset_pool"]}))
                    else:
                        raise ValueError(f"Unknown strategy {strategy}")
            elif regime == "INVERTED":
                if strategy == "REGIME_HEDGE_FIXED_BASELINE":
                    weights_rows.append(_normalize_weights({"SPY": 0.75, "GOLD": 0.25, "CMDTY_FUT": 0.0, "IEF": 0.0, "CASH": 0.0}))
                elif strategy == "REGIME_HEDGE_INV_VOL":
                    weights_rows.append(_apply_pool_weights({a: row.get(f"INVERTED_INV_VOL_weight_{a}", 0.0) for a in CONFIG["inverted_asset_pool"]}))
                elif strategy == "REGIME_HEDGE_TARGET_RB_INV_VOL":
                    weights_rows.append(_apply_pool_weights({a: row.get(f"INVERTED_TARGET_RB_weight_{a}", 0.0) for a in CONFIG["inverted_asset_pool"]}))
                elif strategy == "REGIME_HEDGE_ERC":
                    weights_rows.append(_apply_pool_weights({a: row.get(f"INVERTED_ERC_weight_{a}", 0.0) for a in CONFIG["inverted_asset_pool"]}))
                else:
                    raise ValueError(f"Unknown strategy {strategy}")
            elif regime == "STEEP":
                key = "STEEP_RISK" if timing_state == "RISK" else "STEEP_NON_RISK"
                weights_rows.append(_normalize_weights(CONFIG["fixed_rules"][key]))
            else:
                key = "FALLBACK_RISK" if timing_state == "RISK" else "FALLBACK_NON_RISK"
                weights_rows.append(_normalize_weights(CONFIG["fixed_rules"][key]))
        target_df = pd.DataFrame(weights_rows)
        strategy_target_frames[strategy] = target_df

    risk_contrib = pd.DataFrame(risk_contrib_rows)
    return df, strategy_target_frames, risk_contrib


def run_multi_asset_backtest(panel: pd.DataFrame, strategy_name: str, target_weights: pd.DataFrame, monthly_rebalance: bool) -> pd.DataFrame:
    df = panel.copy()
    tw = target_weights[ASSETS].copy().div(target_weights[ASSETS].sum(axis=1), axis=0).fillna(0.0)
    first_month = _is_first_trading_day_of_month(df["date"])
    for asset in ASSETS:
        df[f"{strategy_name}_weight_{asset}"] = np.nan
    df[f"{strategy_name}_return"] = np.nan
    df[f"{strategy_name}_nav"] = np.nan
    df[f"turnover_{strategy_name}"] = np.nan
    df[f"transaction_cost_{strategy_name}"] = np.nan
    current = tw.iloc[0].to_dict()
    prev_target = tw.iloc[0].to_dict()
    nav = 1.0
    for i in range(len(df)):
        desired = tw.iloc[i].to_dict()
        target_changed = any(abs(desired[a] - prev_target[a]) > 1e-12 for a in ASSETS)
        turnover = 0.0
        cost = 0.0
        if i > 0 and (target_changed or (monthly_rebalance and bool(first_month.iloc[i]))):
            turnover = float(sum(abs(desired[a] - current[a]) for a in ASSETS))
            cost = 0.5 * turnover * CONFIG["one_way_cost_bps"] / 10000
            current = desired.copy()
        gross = float(sum(current[a] * df.iloc[i][f"{a}_return"] for a in ASSETS))
        net = gross - cost
        nav *= 1 + net
        for asset in ASSETS:
            df.loc[i, f"{strategy_name}_weight_{asset}"] = current[asset]
        df.loc[i, f"{strategy_name}_return"] = net
        df.loc[i, f"{strategy_name}_nav"] = nav
        df.loc[i, f"turnover_{strategy_name}"] = turnover
        df.loc[i, f"transaction_cost_{strategy_name}"] = cost
        denom = 1 + gross
        if denom != 0:
            current = {a: current[a] * (1 + df.iloc[i][f"{a}_return"]) / denom for a in ASSETS}
        prev_target = desired.copy()
    return df


def compute_performance_metrics(panel: pd.DataFrame, strategy_name: str, erc_fallback_count: Optional[int] = None) -> Dict[str, object]:
    ret = panel[f"{strategy_name}_return"].fillna(0.0)
    rf = panel["CASH_return"].fillna(0.0)
    nav = (1 + ret).cumprod()
    years = len(ret) / 252
    ann = nav.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = ret.std(ddof=0) * math.sqrt(252)
    excess = ret - rf
    sharpe = excess.mean() / excess.std(ddof=0) * math.sqrt(252) if excess.std(ddof=0) > 0 else np.nan
    maxdd = _max_drawdown(ret)
    calmar = ann / abs(maxdd) if pd.notna(maxdd) and maxdd < 0 else np.nan
    return {
        "strategy": strategy_name,
        "start_date": panel["date"].iloc[0],
        "end_date": panel["date"].iloc[-1],
        "annualized_return": ann,
        "annualized_volatility": vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": maxdd,
        "calmar_ratio": calmar,
        "final_nav": nav.iloc[-1],
        "number_of_rebalances": int((panel[f"turnover_{strategy_name}"] > 0).sum()),
        "total_turnover": panel[f"turnover_{strategy_name}"].fillna(0.0).sum(),
        "transaction_cost_drag": panel[f"transaction_cost_{strategy_name}"].fillna(0.0).sum(),
        "avg_weight_SPY": panel[f"{strategy_name}_weight_SPY"].mean(),
        "avg_weight_GOLD": panel[f"{strategy_name}_weight_GOLD"].mean(),
        "avg_weight_CMDTY_FUT": panel[f"{strategy_name}_weight_CMDTY_FUT"].mean(),
        "avg_weight_IEF": panel[f"{strategy_name}_weight_IEF"].mean(),
        "avg_weight_CASH": panel[f"{strategy_name}_weight_CASH"].mean(),
        "time_in_risk": panel["timing_state"].eq("RISK").mean(),
        "erc_fallback_count": erc_fallback_count if erc_fallback_count is not None else np.nan,
    }


def compute_crisis_performance(panel: pd.DataFrame, strategies: List[str]) -> pd.DataFrame:
    rows = []
    for period, (start, end) in CRISIS_WINDOWS.items():
        sub = panel[(panel["date"] >= pd.Timestamp(start)) & (panel["date"] <= pd.Timestamp(end))]
        if sub.empty:
            continue
        for strategy in strategies:
            ret = sub[f"{strategy}_return"].fillna(0.0)
            rf = sub["CASH_return"].fillna(0.0)
            nav = (1 + ret).cumprod()
            years = len(ret) / 252
            ann = nav.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
            vol = ret.std(ddof=0) * math.sqrt(252)
            sharpe = (ret - rf).mean() / (ret - rf).std(ddof=0) * math.sqrt(252) if (ret - rf).std(ddof=0) > 0 else np.nan
            rows.append(
                {
                    "period": period,
                    "strategy": strategy,
                    "cumulative_return": nav.iloc[-1] - 1,
                    "max_drawdown": _max_drawdown(ret),
                    "volatility": vol,
                    "Sharpe": sharpe,
                    "avg_weight_SPY": sub[f"{strategy}_weight_SPY"].mean(),
                    "avg_weight_GOLD": sub[f"{strategy}_weight_GOLD"].mean(),
                    "avg_weight_CMDTY_FUT": sub[f"{strategy}_weight_CMDTY_FUT"].mean(),
                    "avg_weight_IEF": sub[f"{strategy}_weight_IEF"].mean(),
                    "avg_weight_CASH": sub[f"{strategy}_weight_CASH"].mean(),
                    "turnover": sub[f"turnover_{strategy}"].fillna(0.0).sum(),
                    "cost_drag": sub[f"transaction_cost_{strategy}"].fillna(0.0).sum(),
                    "annualized_return": ann,
                }
            )
    return pd.DataFrame(rows)


def compute_cross_state_performance(panel: pd.DataFrame, strategies: List[str]) -> pd.DataFrame:
    rows = []
    for cross_state, sub in panel.groupby("cross_state", dropna=False):
        for strategy in strategies:
            ret = sub[f"{strategy}_return"].fillna(0.0)
            rf = sub["CASH_return"].fillna(0.0)
            nav = (1 + ret).cumprod()
            years = len(ret) / 252
            ann = nav.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
            vol = ret.std(ddof=0) * math.sqrt(252)
            sharpe = (ret - rf).mean() / (ret - rf).std(ddof=0) * math.sqrt(252) if (ret - rf).std(ddof=0) > 0 else np.nan
            rows.append(
                {
                    "cross_state": cross_state,
                    "strategy": strategy,
                    "n_obs": len(sub),
                    "annualized_return": ann,
                    "volatility": vol,
                    "Sharpe": sharpe,
                    "max_drawdown": _max_drawdown(ret),
                    "avg_weight_SPY": sub[f"{strategy}_weight_SPY"].mean(),
                    "avg_weight_GOLD": sub[f"{strategy}_weight_GOLD"].mean(),
                    "avg_weight_CMDTY_FUT": sub[f"{strategy}_weight_CMDTY_FUT"].mean(),
                    "avg_weight_IEF": sub[f"{strategy}_weight_IEF"].mean(),
                    "avg_weight_CASH": sub[f"{strategy}_weight_CASH"].mean(),
                }
            )
    return pd.DataFrame(rows)


def compute_risk_contribution_summary(panel: pd.DataFrame, risk_contrib: pd.DataFrame) -> pd.DataFrame:
    if risk_contrib.empty:
        return risk_contrib
    return (
        risk_contrib.groupby(["regime_state", "method"], dropna=False)
        .mean(numeric_only=True)
        .reset_index()
    )


def plot_equity_curves(panel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    show = ["SPY_BUY_HOLD", "BACKBONE_V2_SPY_CASH", "REGIME_HEDGE_V1_ORIGINAL", "REGIME_HEDGE_FIXED_BASELINE", "REGIME_HEDGE_INV_VOL", "REGIME_HEDGE_TARGET_RB_INV_VOL", "REGIME_HEDGE_ERC"]
    for strategy in show:
        ax.plot(panel["date"], panel[f"{strategy}_nav"], label=strategy)
    ax.set_yscale("log")
    ax.set_title("Equity Curves (Log Scale)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "equity_curve_log.png", dpi=150)
    plt.close(fig)


def plot_drawdowns(panel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    show = ["SPY_BUY_HOLD", "BACKBONE_V2_SPY_CASH", "REGIME_HEDGE_V1_ORIGINAL", "REGIME_HEDGE_FIXED_BASELINE", "REGIME_HEDGE_INV_VOL", "REGIME_HEDGE_TARGET_RB_INV_VOL", "REGIME_HEDGE_ERC"]
    for strategy in show:
        nav = panel[f"{strategy}_nav"]
        ax.plot(panel["date"], nav / nav.cummax() - 1, label=strategy)
    ax.legend(ncol=2, fontsize=8)
    ax.set_title("Drawdown Comparison")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "drawdown_comparison.png", dpi=150)
    plt.close(fig)


def plot_weight_stacks(panel: pd.DataFrame) -> None:
    mapping = {
        "REGIME_HEDGE_INV_VOL": "weight_stack_INV_VOL.png",
        "REGIME_HEDGE_TARGET_RB_INV_VOL": "weight_stack_TARGET_RB_INV_VOL.png",
        "REGIME_HEDGE_ERC": "weight_stack_ERC.png",
    }
    for strategy, filename in mapping.items():
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.stackplot(
            panel["date"],
            panel[f"{strategy}_weight_SPY"],
            panel[f"{strategy}_weight_GOLD"],
            panel[f"{strategy}_weight_CMDTY_FUT"],
            panel[f"{strategy}_weight_IEF"],
            panel[f"{strategy}_weight_CASH"],
            labels=ASSETS,
        )
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", ncol=5, fontsize=8)
        ax.set_title(strategy)
        fig.tight_layout()
        fig.savefig(CONFIG["figure_dir"] / filename, dpi=150)
        plt.close(fig)


def plot_flat_inverted_weight_comparison(panel: pd.DataFrame) -> None:
    flat = panel[panel["cross_state"].eq("FLAT_NON_RISK")]
    inv = panel[panel["cross_state"].eq("INVERTED")]
    if not flat.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = ["Fixed", "InvVol", "TargetRB", "ERC"]
        data = {
            "SPY": [
                0.60,
                flat["FLAT_INV_VOL_weight_SPY"].mean(),
                flat["FLAT_TARGET_RB_weight_SPY"].mean(),
                flat["FLAT_ERC_weight_SPY"].mean(),
            ],
            "GOLD": [
                0.30,
                flat["FLAT_INV_VOL_weight_GOLD"].mean(),
                flat["FLAT_TARGET_RB_weight_GOLD"].mean(),
                flat["FLAT_ERC_weight_GOLD"].mean(),
            ],
            "CMDTY_FUT": [
                0.10,
                flat["FLAT_INV_VOL_weight_CMDTY_FUT"].mean(),
                flat["FLAT_TARGET_RB_weight_CMDTY_FUT"].mean(),
                flat["FLAT_ERC_weight_CMDTY_FUT"].mean(),
            ],
        }
        x = np.arange(len(labels))
        bottom = np.zeros(len(labels))
        for asset, vals in data.items():
            ax.bar(x, vals, bottom=bottom, label=asset)
            bottom += np.array(vals)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title("FLAT_NON_RISK Weight Comparison")
        fig.tight_layout()
        fig.savefig(CONFIG["figure_dir"] / "flat_weight_comparison.png", dpi=150)
        plt.close(fig)

    if not inv.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = ["Fixed", "InvVol", "TargetRB", "ERC"]
        data = {
            "SPY": [0.75, inv["INVERTED_INV_VOL_weight_SPY"].mean(), inv["INVERTED_TARGET_RB_weight_SPY"].mean(), inv["INVERTED_ERC_weight_SPY"].mean()],
            "GOLD": [0.25, inv["INVERTED_INV_VOL_weight_GOLD"].mean(), inv["INVERTED_TARGET_RB_weight_GOLD"].mean(), inv["INVERTED_ERC_weight_GOLD"].mean()],
        }
        x = np.arange(len(labels))
        bottom = np.zeros(len(labels))
        for asset, vals in data.items():
            ax.bar(x, vals, bottom=bottom, label=asset)
            bottom += np.array(vals)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title("INVERTED Weight Comparison")
        fig.tight_layout()
        fig.savefig(CONFIG["figure_dir"] / "inverted_weight_comparison.png", dpi=150)
        plt.close(fig)


def plot_risk_contribution(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    metrics = [c for c in summary.columns if c.startswith("avg_RC_")]
    pivot = summary.set_index(["regime_state", "method"])[metrics]
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pivot.fillna(np.nan), aspect="auto", cmap="RdYlGn")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{a}|{b}" for a, b in pivot.index])
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_title("Risk Contribution Heatmap")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "risk_contribution_heatmap.png", dpi=150)
    plt.close(fig)


def plot_case_studies(panel: pd.DataFrame) -> None:
    cases = {
        "2015_2016": ("2015-05-01", "2016-03-31"),
        "2022": ("2021-11-01", "2023-03-31"),
        "2025_PULLBACK": ("2025-01-01", "2025-12-31"),
        "2008_GFC": ("2007-10-01", "2009-06-30"),
    }
    show_strats = ["REGIME_HEDGE_FIXED_BASELINE", "REGIME_HEDGE_INV_VOL", "REGIME_HEDGE_TARGET_RB_INV_VOL", "REGIME_HEDGE_ERC"]
    for name, (start, end) in cases.items():
        sub = panel[(panel["date"] >= pd.Timestamp(start)) & (panel["date"] <= pd.Timestamp(end))].copy()
        if sub.empty:
            continue
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1.5, 1.2, 0.8]})
        for strategy in show_strats:
            axes[0].plot(sub["date"], sub[f"{strategy}_nav"] / sub[f"{strategy}_nav"].iloc[0], label=strategy)
        axes[0].legend(fontsize=8, ncol=2)
        for asset in ASSETS:
            axes[1].plot(sub["date"], sub[f"{asset}_NAV"] / sub[f"{asset}_NAV"].iloc[0], label=asset)
        axes[1].legend(fontsize=8, ncol=5)
        axes[2].stackplot(
            sub["date"],
            sub["REGIME_HEDGE_ERC_weight_SPY"],
            sub["REGIME_HEDGE_ERC_weight_GOLD"],
            sub["REGIME_HEDGE_ERC_weight_CMDTY_FUT"],
            sub["REGIME_HEDGE_ERC_weight_IEF"],
            sub["REGIME_HEDGE_ERC_weight_CASH"],
            labels=ASSETS,
        )
        axes[2].legend(fontsize=7, ncol=5)
        axes[3].plot(sub["date"], sub["timing_state"].map({"NON_RISK": 0, "RISK": 1}), label="timing_state")
        axes[3].legend(fontsize=8)
        axes[3].set_yticks([0, 1], ["NON_RISK", "RISK"])
        fig.tight_layout()
        fig.savefig(CONFIG["figure_dir"] / f"crisis_case_study_{name}.png", dpi=150)
        plt.close(fig)


def write_markdown_report(perf: pd.DataFrame, rc_summary: pd.DataFrame, erc_fallback_count: int) -> None:
    p = perf.set_index("strategy")
    lines = [
        "# RISK_PARITY_ALLOCATION_REPORT",
        "",
        "## 1. Purpose",
        "This round adds risk parity / risk budget allocation to the regime-aware allocation framework. Timing backbone still decides RISK vs NON_RISK; risk parity only controls normal sleeves.",
        "",
        "## 2. Framework",
        "- Timing backbone = BACKBONE_V2_UPGRADED",
        "- NON_RISK uses fixed / inverse-vol / target risk budget inverse-vol / ERC",
        "- RISK uses full hedge sleeves",
        "- CASH is excluded from risk parity because near-zero volatility would force mechanical over-allocation",
        "",
        "## 3. Strategy definitions",
        "- REGIME_HEDGE_FIXED_BASELINE",
        "- REGIME_HEDGE_INV_VOL",
        "- REGIME_HEDGE_TARGET_RB_INV_VOL",
        "- REGIME_HEDGE_ERC",
        "- REGIME_HEDGE_V1_ORIGINAL",
        "- BACKBONE_V2_SPY_CASH",
        "",
        "## 4. Risk parity methodology",
        "- Inverse-vol uses rolling 120d annualized vol",
        "- Target RB inverse-vol uses target_budget / vol",
        "- ERC uses rolling 120d covariance with long-only bounds",
        f"- ERC fallback count: {erc_fallback_count}",
        "",
        "## 5. Main performance comparison",
        f"- Fixed baseline Sharpe: {p.loc['REGIME_HEDGE_FIXED_BASELINE', 'sharpe_ratio']:.2f}",
        f"- Inverse-vol Sharpe: {p.loc['REGIME_HEDGE_INV_VOL', 'sharpe_ratio']:.2f}",
        f"- Target RB inverse-vol Sharpe: {p.loc['REGIME_HEDGE_TARGET_RB_INV_VOL', 'sharpe_ratio']:.2f}",
        f"- ERC Sharpe: {p.loc['REGIME_HEDGE_ERC', 'sharpe_ratio']:.2f}",
        "",
        "## 6. Cross-state findings",
        "See `performance_by_cross_state.csv` and `risk_parity_weight_summary.csv` for weight behavior and realized outcomes by cross-state.",
        "",
        "## 7. Crisis period analysis",
        "Focus on 2008, 2015-2016, 2022, 2025 pullback, and COVID.",
        "",
        "## 8. Risk contribution analysis",
        "Use `risk_contribution_summary.csv` and the heatmap to compare achieved vs target risk contribution in FLAT_NON_RISK and INVERTED.",
        "",
        "## 9. Interpretation",
        "The practical question is whether dynamic risk budgeting improves Sharpe without adding too much turnover and without making the framework dependent on noisy covariance estimates.",
        "",
        "## 10. Recommendation",
        "Use the best balance across Sharpe, MaxDD, turnover, and crisis stability as the next allocation candidate.",
        "",
        "## 11. Caveats",
        "- This remains in-sample.",
        "- Covariance is noisy in crisis periods, though this framework uses it only in NON_RISK sleeves.",
        "- Commodity futures data quality and tradability matter.",
        "- ERC depends on optimizer stability and fallback handling.",
    ]
    (CONFIG["output_dir"] / "RISK_PARITY_ALLOCATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel = load_base_panel()
    panel = load_asset_returns(panel)
    panel = build_backbone_v2_state(panel)
    panel = compute_rolling_volatility(panel)

    if "entry_reason" not in panel.columns:
        panel["entry_reason"] = ""
    if "cross_state" not in panel.columns:
        panel["cross_state"] = panel["macro_regime_confirmed"] + "_" + panel["timing_state"]

    panel, strategy_targets, risk_contrib = build_allocation_rules(panel)
    erc_fallback_count = int(panel["ERC_fallback_flag"].sum())

    for strategy in STRATEGIES:
        panel = run_multi_asset_backtest(panel, strategy, strategy_targets[strategy], monthly_rebalance=CONFIG["monthly_rebalance"])

    daily_cols = [
        "date",
        "macro_regime_confirmed",
        "timing_state",
        "cross_state",
        "entry_reason",
        "VIX_LEVEL",
        "VIX_ZSCORE_120D",
        "CREDIT_SPREAD_BAA_AAA",
        "D_CREDIT_SPREAD_20D",
        "spy_drawdown_from_previous_high",
        "SPY_CROSS_ABOVE_MA20",
        "SPY_return",
        "GOLD_return",
        "CMDTY_FUT_return",
        "IEF_return",
        "CASH_return",
    ]
    for prefix, pool in [
        ("FLAT_INV_VOL", CONFIG["flat_asset_pool"]),
        ("FLAT_TARGET_RB", CONFIG["flat_asset_pool"]),
        ("FLAT_ERC", CONFIG["flat_asset_pool"]),
        ("INVERTED_INV_VOL", CONFIG["inverted_asset_pool"]),
        ("INVERTED_TARGET_RB", CONFIG["inverted_asset_pool"]),
        ("INVERTED_ERC", CONFIG["inverted_asset_pool"]),
    ]:
        for asset in pool:
            daily_cols.append(f"{prefix}_weight_{asset}")
    daily_cols += ["ERC_fallback_flag", "ERC_fallback_reason"]

    for strategy in STRATEGIES:
        daily_cols += [
            f"{strategy}_weight_SPY",
            f"{strategy}_weight_GOLD",
            f"{strategy}_weight_CMDTY_FUT",
            f"{strategy}_weight_IEF",
            f"{strategy}_weight_CASH",
            f"{strategy}_return",
            f"{strategy}_nav",
            f"turnover_{strategy}",
            f"transaction_cost_{strategy}",
        ]
    panel[daily_cols].to_csv(CONFIG["output_dir"] / "daily_backtest_panel.csv", index=False)

    perf_rows = []
    for strategy in STRATEGIES:
        fallback = erc_fallback_count if strategy == "REGIME_HEDGE_ERC" else None
        perf_rows.append(compute_performance_metrics(panel, strategy, fallback))
    perf = pd.DataFrame(perf_rows)
    perf.to_csv(CONFIG["output_dir"] / "performance_summary.csv", index=False)

    crisis = compute_crisis_performance(panel, STRATEGIES)
    crisis.to_csv(CONFIG["output_dir"] / "crisis_performance.csv", index=False)

    cross_perf = compute_cross_state_performance(panel, STRATEGIES)
    cross_perf.to_csv(CONFIG["output_dir"] / "performance_by_cross_state.csv", index=False)

    weight_summary_rows = []
    for strategy in ["REGIME_HEDGE_FIXED_BASELINE", "REGIME_HEDGE_INV_VOL", "REGIME_HEDGE_TARGET_RB_INV_VOL", "REGIME_HEDGE_ERC"]:
        for cross_state, sub in panel.groupby("cross_state", dropna=False):
            ws = sub[[f"{strategy}_weight_SPY", f"{strategy}_weight_GOLD", f"{strategy}_weight_CMDTY_FUT", f"{strategy}_weight_IEF", f"{strategy}_weight_CASH"]]
            if ws.empty:
                continue
            row = {"strategy": strategy, "regime_state": cross_state}
            for asset in ASSETS:
                col = f"{strategy}_weight_{asset}"
                row[f"avg_weight_{asset}"] = sub[col].mean()
            row["min_weight"] = ws.min(axis=1).mean()
            row["max_weight"] = ws.max(axis=1).mean()
            row["weight_volatility"] = ws.std(ddof=0).mean()
            row["avg_realized_vol"] = sub[f"{strategy}_return"].std(ddof=0) * math.sqrt(252)
            row["avg_turnover"] = sub[f"turnover_{strategy}"].mean()
            weight_summary_rows.append(row)
    weight_summary = pd.DataFrame(weight_summary_rows)
    weight_summary.to_csv(CONFIG["output_dir"] / "risk_parity_weight_summary.csv", index=False)

    rc_summary = compute_risk_contribution_summary(panel, risk_contrib)
    rc_summary.to_csv(CONFIG["output_dir"] / "risk_contribution_summary.csv", index=False)

    plot_equity_curves(panel)
    plot_drawdowns(panel)
    plot_weight_stacks(panel)
    plot_flat_inverted_weight_comparison(panel)
    plot_risk_contribution(rc_summary)
    plot_case_studies(panel)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    show = perf[perf["strategy"].isin(["BACKBONE_V2_SPY_CASH", "REGIME_HEDGE_V1_ORIGINAL", "REGIME_HEDGE_FIXED_BASELINE", "REGIME_HEDGE_INV_VOL", "REGIME_HEDGE_TARGET_RB_INV_VOL", "REGIME_HEDGE_ERC"])]
    for ax, metric in zip(axes.ravel(), ["annualized_return", "sharpe_ratio", "max_drawdown", "calmar_ratio", "final_nav", "total_turnover"]):
        ax.bar(show["strategy"], show[metric])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(CONFIG["figure_dir"] / "performance_bar_charts.png", dpi=150)
    plt.close(fig)

    write_markdown_report(perf, rc_summary, erc_fallback_count)

    p = perf.set_index("strategy")
    print(f"1. Sample range: {panel['date'].iloc[0].date()} to {panel['date'].iloc[-1].date()}")
    for strategy in ["SPY_BUY_HOLD", "BACKBONE_V2_SPY_CASH", "REGIME_HEDGE_FIXED_BASELINE", "REGIME_HEDGE_INV_VOL", "REGIME_HEDGE_TARGET_RB_INV_VOL", "REGIME_HEDGE_ERC", "REGIME_HEDGE_V1_ORIGINAL"]:
        row = p.loc[strategy]
        print(f"2. {strategy}: AnnRet {row['annualized_return']:.2%}, Sharpe {row['sharpe_ratio']:.2f}, MaxDD {row['max_drawdown']:.2%}, Final NAV {row['final_nav']:.2f}")
    best_sharpe = perf.sort_values("sharpe_ratio", ascending=False).iloc[0]
    best_maxdd = perf.sort_values("max_drawdown", ascending=False).iloc[0]
    print(f"3. Best Sharpe strategy: {best_sharpe['strategy']} ({best_sharpe['sharpe_ratio']:.2f})")
    print(f"4. Lowest MaxDD strategy: {best_maxdd['strategy']} ({best_maxdd['max_drawdown']:.2%})")
    print(
        "5. Fixed vs InvVol vs ERC Sharpe: "
        f"{p.loc['REGIME_HEDGE_FIXED_BASELINE','sharpe_ratio']:.2f} / "
        f"{p.loc['REGIME_HEDGE_INV_VOL','sharpe_ratio']:.2f} / "
        f"{p.loc['REGIME_HEDGE_ERC','sharpe_ratio']:.2f}"
    )
    crisis_pivot = crisis.pivot(index="period", columns="strategy", values="cumulative_return") if not crisis.empty else pd.DataFrame()
    for period, label in [("2015_2016", "6. 2015-2016"), ("2022", "7. 2022"), ("2025_PULLBACK", "8. 2025 pullback")]:
        if period in crisis_pivot.index:
            vals = crisis_pivot.loc[period, ["REGIME_HEDGE_FIXED_BASELINE", "REGIME_HEDGE_INV_VOL", "REGIME_HEDGE_TARGET_RB_INV_VOL", "REGIME_HEDGE_ERC"]].to_dict()
            print(f"{label}: {vals}")
    print(f"9. ERC fallback count: {erc_fallback_count}")
    recommendation = best_sharpe["strategy"]
    if best_sharpe["strategy"] == "REGIME_HEDGE_ERC" and p.loc["REGIME_HEDGE_ERC", "total_turnover"] > p.loc["REGIME_HEDGE_TARGET_RB_INV_VOL", "total_turnover"] * 1.5:
        recommendation = "REGIME_HEDGE_TARGET_RB_INV_VOL"
    print(f"10. Recommended next allocation method: {recommendation}")


if __name__ == "__main__":
    main()
