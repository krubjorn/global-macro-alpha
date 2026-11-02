"""Combine sleeve returns. Every weight uses only past sleeve returns (expanding or rolling windows)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def _vol_target(r: pd.Series, target: float, window: int) -> pd.Series:
    """Scale by target / trailing realised vol (lagged one month), leverage capped to [0.25, 3]."""
    realised = r.rolling(window, min_periods=12).std().shift(1) * np.sqrt(12)
    k = (target / realised).clip(0.25, 3.0).fillna(1.0)
    return r * k


def equal_weight(sleeves: pd.DataFrame) -> pd.DataFrame:
    live = sleeves.notna() & sleeves.ne(0)
    w = live.div(live.sum(axis=1), axis=0).fillna(0.0)
    return w


def inverse_vol(sleeves: pd.DataFrame, window: int) -> pd.DataFrame:
    vol = sleeves.rolling(window, min_periods=12).std().shift(1)
    inv = (1 / vol).where(sleeves.ne(0))
    return inv.div(inv.sum(axis=1), axis=0).fillna(0.0)


def shrink_mv(sleeves: pd.DataFrame, window: int, intensity: float) -> pd.DataFrame:
    """Long-only mean-variance weights: means shrunk to their cross-sleeve average,
    Ledoit-Wolf covariance, both estimated on the `window` months before each date."""
    w = pd.DataFrame(0.0, index=sleeves.index, columns=sleeves.columns)
    x = sleeves.fillna(0.0)
    for i in range(len(x)):
        hist = x.iloc[max(0, i - window):i]
        live = hist.columns[(hist != 0).sum() >= 24]
        if len(live) < 2:
            continue
        mu = hist[live].mean()
        mu = (1 - intensity) * mu + intensity * mu.mean()
        cov = LedoitWolf().fit(hist[live].to_numpy()).covariance_
        raw = np.linalg.solve(cov, mu.to_numpy())
        raw = np.clip(raw, 0, None)
        if raw.sum() > 0:
            w.iloc[i, [x.columns.get_loc(c) for c in live]] = raw / raw.sum()
    return w


def combine(sleeves: pd.DataFrame, params: dict) -> dict[str, pd.Series]:
    """sleeves: monthly net returns, one column per signal. Returns one series per method."""
    cfg = params["combine"]
    target = params["portfolio"]["target_vol"]
    win = cfg["vol_window_m"]
    weights = {
        "equal_weight": equal_weight(sleeves),
        "inverse_vol": inverse_vol(sleeves, win),
        "shrink_mv": shrink_mv(sleeves, cfg["shrink_mv_window_m"], cfg["shrink_intensity"]),
    }
    out = {}
    for name in cfg["methods"]:
        r = (weights[name] * sleeves.fillna(0.0)).sum(axis=1)
        out[name] = _vol_target(r, target, win)
    return out
