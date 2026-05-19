"""Small performance helpers used by project notebooks and ad hoc checks."""

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def max_drawdown(returns: pd.Series) -> float:
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def annualized_return(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    return float((1.0 + clean).prod() ** (TRADING_DAYS_PER_YEAR / len(clean)) - 1.0)


def annualized_volatility(returns: pd.Series) -> float:
    return float(returns.dropna().std() * np.sqrt(TRADING_DAYS_PER_YEAR))
