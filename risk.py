"""Risk model: EWMA volatility and shrunk covariance, both strictly backward-looking."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def ewma_vol(returns: pd.DataFrame, halflife: int = 60, min_periods: int = 60) -> pd.DataFrame:
    """Annualised EWMA volatility using data up to and including each date."""
    var = (returns ** 2).ewm(halflife=halflife, min_periods=min_periods).mean()
    return np.sqrt(var * 252)


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last available trading day of each calendar month in `index`."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).max().to_numpy())


def shrunk_cov(returns: pd.DataFrame, end: pd.Timestamp, window: int = 252,
               min_obs: int = 120) -> pd.DataFrame:
    """Ledoit-Wolf covariance (annualised) over the `window` days ending at `end`.

    Assets with too little history are given a diagonal entry from their own sample variance
    and zero correlation, so a late-starting asset never blocks the whole book.
    """
    win = returns.loc[:end].tail(window)
    ok = win.columns[win.notna().sum() >= min_obs]
    cov = pd.DataFrame(np.nan, index=returns.columns, columns=returns.columns)
    if len(ok) >= 2:
        x = win[ok].fillna(0.0).to_numpy()
        lw = LedoitWolf().fit(x).covariance_ * 252
        cov.loc[ok, ok] = lw
    var = win.var() * 252
    for a in returns.columns:
        if a not in ok and pd.notna(var.get(a)):
            cov.loc[a, :] = 0.0
            cov.loc[:, a] = 0.0
            cov.loc[a, a] = var[a]
    return cov
