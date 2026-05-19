"""Validation helpers for final strategy scripts."""

import pandas as pd


VALID_REGIMES = {"FLAT", "STEEP", "INVERTED"}


def unexpected_regime_dates(panel: pd.DataFrame, regime_col: str = "macro_regime_confirmed") -> pd.DataFrame:
    if regime_col not in panel.columns:
        raise KeyError(f"Missing required regime column: {regime_col}")
    mask = ~panel[regime_col].isin(VALID_REGIMES)
    return panel.loc[mask, ["date", regime_col]].copy() if "date" in panel.columns else panel.loc[mask, [regime_col]].copy()
