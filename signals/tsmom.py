"""Time-series momentum (Moskowitz, Ooi & Pedersen 2012), tanh-scaled by EWMA volatility.

    s_{i,t} = tanh( r_{i, t-L-skip+1 : t-skip} / (sigma_{i,t} * sqrt(L/12)) )
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..risk import ewma_vol
from .base import monthly_returns


def compute(panel, params: dict, lookback_m: int | None = None, skip_m: int | None = None) -> pd.DataFrame:
    cfg = params["signals"]["tsmom"]
    L = lookback_m or cfg["lookback_m"]
    skip = cfg["skip_m"] if skip_m is None else skip_m
    mret = monthly_returns(panel.returns)
    logr = np.log1p(mret)
    cum = np.expm1(logr.shift(skip).rolling(L, min_periods=int(L * 0.8)).sum())
    vol = ewma_vol(panel.returns, params["risk"]["vol_halflife_days"]).resample("ME").last()
    vol = vol.clip(lower=params["portfolio"]["vol_floor"])
    s = np.tanh(cum / (vol * np.sqrt(L / 12)))
    return s.clip(-params["signals"]["clip_z"], params["signals"]["clip_z"])
