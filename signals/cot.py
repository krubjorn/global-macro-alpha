"""Positioning (contrarian): minus the 3-year z-score of net speculative positions / open interest.

Point-in-time: Tuesday positions become usable `publication_lag_days` later (Friday release).
"""
from __future__ import annotations

import pandas as pd


def compute(panel, params: dict) -> pd.DataFrame:
    cfg = params["signals"]["cot"]
    cot = panel.cot.copy()
    if cot.empty:
        return pd.DataFrame(columns=panel.meta.index, dtype=float)
    cot["available"] = pd.to_datetime(cot["report_date"]) + pd.Timedelta(days=cfg["publication_lag_days"])
    wide = cot.pivot_table(index="available", columns="asset", values="net_spec_oi").sort_index()
    w = cfg["z_window_weeks"]
    z = (wide - wide.rolling(w, min_periods=w // 2).mean()) / wide.rolling(w, min_periods=w // 2).std()
    monthly = z.resample("ME").last()                      # last value available by month-end
    clip = params["signals"]["clip_z"]
    return (-monthly).clip(-clip, clip).reindex(columns=panel.meta.index)
