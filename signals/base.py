"""Shared helpers for signal construction."""
from __future__ import annotations

import numpy as np
import pandas as pd


def monthly_returns(daily: pd.DataFrame) -> pd.DataFrame:
    """Compounded calendar-month returns, indexed at calendar month-end. NaN if no data."""
    m = (1 + daily.fillna(0)).resample("ME").prod() - 1
    has = daily.notna().resample("ME").sum() > 0
    return m.where(has)


def xs_rank(raw: pd.DataFrame, meta: pd.DataFrame, clip: float = 2.0, min_assets: int = 3) -> pd.DataFrame:
    """Rank within asset class each date, mapped to a unit-variance score in [-sqrt(3), sqrt(3)].

    Classes with fewer than `min_assets` live signals that date get zero.
    """
    out = pd.DataFrame(np.nan, index=raw.index, columns=raw.columns)
    for cls, members in meta.groupby("cls").groups.items():
        cols = [c for c in members if c in raw.columns]
        sub = raw[cols]
        n = sub.notna().sum(axis=1)
        pct = sub.rank(axis=1, pct=True)
        # centre on the mid-rank so the score is mean-zero within class
        centred = pct.sub((1 + 1 / n) / 2, axis=0)
        score = centred.mul(np.sqrt(12), axis=0)
        out[cols] = score.where(n >= min_assets, 0.0)
    return out.clip(-clip, clip)


def ts_scale(raw: pd.DataFrame, clip: float = 2.0, min_periods: int = 36) -> pd.DataFrame:
    """Divide by the expanding (past-only) RMS of the signal itself, then clip."""
    rms = np.sqrt((raw ** 2).expanding(min_periods=min_periods).mean().shift(1))
    return (raw / rms).clip(-clip, clip)


def align_month_end(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.DatetimeIndex(df.index) + pd.offsets.MonthEnd(0)
    return df.groupby(level=0).last()
