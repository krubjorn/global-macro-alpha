"""Value: higher = cheaper. Ranked within asset class.

  FX         : minus 5y change in log real exchange rate (clean.fx_value)
  bonds      : real yield (clean.bond_value)
  equities   : earnings yield vs own 10y history (clean.equity_value)
  commodities: minus 5y log return of the excess-return index (computed here from prices)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import monthly_returns, xs_rank


def compute(panel, params: dict) -> pd.DataFrame:
    raw = panel.value.copy()
    m = params["signals"]["value"]["commodity_reversal_m"]
    comm = panel.meta.index[panel.meta["cls"] == "commodity"]
    logp = np.log1p(monthly_returns(panel.returns[comm])).cumsum()
    raw = raw.reindex(index=raw.index.union(logp.index))
    raw[comm] = -(logp - logp.shift(m))
    return xs_rank(raw, panel.meta, params["signals"]["clip_z"])
